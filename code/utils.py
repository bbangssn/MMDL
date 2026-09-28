"""Shared model registry, lazy loading, inference, and scoring helpers."""

import ast
import copy
import json
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
from vllm_compat import FLASHINFER_SAMPLER_ENABLED, FLASHINFER_SAMPLER_REASON
from vllm import LLM, SamplingParams


ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = ROOT / "models"
DATASET_CACHE = ROOT / "datasets"
RESULTS_ROOT = ROOT / "results"


MMMU_SUBJECTS = (
    "Accounting", "Agriculture", "Architecture_and_Engineering", "Art",
    "Art_Theory", "Basic_Medical_Science", "Biology", "Chemistry",
    "Clinical_Medicine", "Computer_Science", "Design",
    "Diagnostics_and_Laboratory_Medicine", "Economics", "Electronics",
    "Energy_and_Power", "Finance", "Geography", "History", "Literature",
    "Manage", "Marketing", "Materials", "Math", "Mechanical_Engineering",
    "Music", "Pharmacy", "Physics", "Psychology", "Public_Health", "Sociology",
)


MMMU_CATEGORIES = {
    "Art and Design": {"Art", "Art_Theory", "Design", "Music"},
    "Business": {"Accounting", "Economics", "Finance", "Manage", "Marketing"},
    "Science": {"Biology", "Chemistry", "Geography", "Math", "Physics"},
    "Health and Medicine": {
        "Basic_Medical_Science", "Clinical_Medicine",
        "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health",
    },
    "Humanities and Social Science": {
        "History", "Literature", "Psychology", "Sociology",
    },
    "Tech and Engineering": {
        "Agriculture", "Architecture_and_Engineering", "Computer_Science",
        "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering",
    },
}


def category_for_subject(subject: str) -> str:
    normalized = subject.replace(" ", "_")
    for category, subjects in MMMU_CATEGORIES.items():
        if normalized in subjects:
            return category
    return "Unknown"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: str
    load_kwargs: dict[str, Any] = field(default_factory=dict)


# Add base or fine-tuned checkpoints here. These are descriptions, not loaded models.
MODELS = [
    ModelSpec(
        name="qwen3-vl-4b-instruct",
        path="Qwen/Qwen3-VL-4B-Instruct",
    ),
]


def resolve_model(value: str) -> ModelSpec:
    if value.isdigit():
        index = int(value)
        if 0 <= index < len(MODELS):
            return MODELS[index]
    else:
        for spec in MODELS:
            if spec.name == value:
                return spec

    available = ", ".join(f"{i}:{spec.name}" for i, spec in enumerate(MODELS))
    raise ValueError(f"Unknown model '{value}'. Available models: {available}")


def print_models() -> None:
    for index, spec in enumerate(MODELS):
        print(f"{index}: {spec.name} ({spec.path})")


class VLLM:
    """One lazily selected Qwen vision-language model served by vLLM."""

    def __init__(self, spec: ModelSpec, max_model_length: int = 9048,
                 gpu_memory_utilization: float = 0.8,
                 max_images_per_prompt: int = 7, seed: int = 42):
        self.spec = spec
        self.processor = AutoProcessor.from_pretrained(
            spec.path, cache_dir=MODEL_CACHE, use_fast=True
        )
        self.model = LLM(
            model=spec.path, download_dir=str(MODEL_CACHE),
            tensor_parallel_size=1, gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True, max_model_len=max_model_length,
            limit_mm_per_prompt={"image": max_images_per_prompt}, seed=seed,
            **spec.load_kwargs,
        )

    def _prepare_input(self, prompt, images, min_pixels, max_pixels):
        content = []
        for image in images:
            item = {"type": "image", "image": image}
            if min_pixels is not None:
                item["min_pixels"] = min_pixels
            if max_pixels is not None:
                item["max_pixels"] = max_pixels
            content.append(item)
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages, image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True, return_video_metadata=True,
        )
        multimodal = {}
        if image_inputs is not None:
            multimodal["image"] = image_inputs
        if video_inputs is not None:
            multimodal["video"] = video_inputs
        return {
            "prompt": text, "multi_modal_data": multimodal,
            "mm_processor_kwargs": video_kwargs,
        }

    @staticmethod
    def _sampling(max_new_tokens, generation_kwargs):
        options = dict(generation_kwargs or {})
        do_sample = bool(options.pop("do_sample", False))
        return SamplingParams(
            temperature=float(options.pop("temperature", 0.7)) if do_sample else 0.0,
            top_p=float(options.pop("top_p", 0.8)),
            top_k=int(options.pop("top_k", 20)),
            repetition_penalty=float(options.pop("repetition_penalty", 1.0)),
            presence_penalty=float(options.pop("presence_penalty", 1.5)),
            seed=options.pop("seed", None), max_tokens=max_new_tokens,
            stop_token_ids=[], **options,
        )

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[str]:
        inputs, sampling = [], []
        for request in requests:
            inputs.append(self._prepare_input(
                request["prompt"], request.get("images", []),
                request.get("min_pixels"), request.get("max_pixels"),
            ))
            sampling.append(self._sampling(
                request["max_new_tokens"], request.get("generation_kwargs")
            ))
        outputs = self.model.generate(inputs, sampling_params=sampling, use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def generate(self, prompt: str, images: list[Any], max_new_tokens: int,
                 max_model_length: int | None = None,
                 generation_kwargs: dict[str, Any] | None = None,
                 min_pixels: int | None = None,
                 max_pixels: int | None = None) -> str:
        del max_model_length
        return self.generate_batch([{
            "prompt": prompt, "images": images,
            "max_new_tokens": max_new_tokens,
            "generation_kwargs": generation_kwargs,
            "min_pixels": min_pixels, "max_pixels": max_pixels,
        }])[0]


class TextJudge:
    """A local text-only instruction model used after VLM inference."""

    def __init__(self, model_path: str, cache_dir: Path):
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir=cache_dir, local_files_only=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(
        self, prompt: str, max_new_tokens: int, max_model_length: int
    ) -> str:
        # Qwen's official MMMU wrapper sends the extraction prompt as one user message.
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(next(self.model.parameters()).device)
        input_length = inputs["input_ids"].shape[1]
        available_tokens = max_model_length - input_length
        if available_tokens < 1:
            raise ValueError(
                f"Judge input length ({input_length}) reaches MAX_MODEL_LENGTH "
                f"({max_model_length})."
            )
        output_ids = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=min(max_new_tokens, available_tokens),
            pad_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = output_ids[:, input_length:]
        return self.tokenizer.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

def parse_options(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(option) for option in value]
    if not value:
        return []
    parsed = ast.literal_eval(value)
    return [str(option) for option in parsed]


def get_images(sample: dict[str, Any]) -> list[Any]:
    if sample.get("image") is not None:
        return [sample["image"]]
    return [sample[f"image_{i}"] for i in range(1, 8) if sample.get(f"image_{i}") is not None]


def get_images_in_token_order(sample: dict[str, Any], text: str) -> list[Any]:
    """Follow MMMU-Pro's official <image i> occurrence order."""
    order = [int(value) for value in re.findall(r"<image\s+(\d+)>", text)]
    return [sample[f"image_{index}"] for index in order]


def multiple_choice_prompt(question: str, options: list[str]) -> str:
    choices = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        f"{question}\n\n{choices}\n\n"
        "Solve the problem and end with exactly 'Answer: X', where X is the option letter."
    )


def qwen_mmmu_prompt(question: str, options: list[str]) -> str:
    """Prompt used by QwenLM/Qwen3-VL's public MMMU evaluation."""
    prompt = f"Question: {question}\n"
    if options:
        prompt += "Options:\n"
        prompt += "".join(
            f"{chr(65 + index)}. {option}\n"
            for index, option in enumerate(options)
        )
        prompt += "Please select the correct answer from the options above. \n"
    return prompt.rstrip()


def qwen_rule_extract(answer: str, options: list[str]) -> str:
    """Qwen's official rule-based option extractor; empty means unresolved."""
    choices = {chr(65 + index): str(option) for index, option in enumerate(options)}
    rejects = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    if any(message in answer for message in rejects):
        return "Z"

    answer_mod = copy.copy(str(answer))
    for char in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(char, " ")
    splits = [part.strip() for part in answer_mod.split()]
    candidates = [choice for choice in choices if choice in splits]
    if len(candidates) == 1:
        if "A" in splits and len(splits) > 3:
            return ""
        return candidates[0]

    lowered = str(answer).lower()
    text_candidates = [
        choice for choice, option in choices.items() if option.lower() in lowered
    ]
    return text_candidates[0] if len(text_candidates) == 1 else ""


def official_mmmu_choice_extract(
    response: str,
    options: list[str],
    rng: random.Random,
) -> tuple[str, bool]:
    """MMMU/MMMU-Pro official multiple-choice response parser."""
    all_choices = [chr(65 + index) for index in range(len(options))]
    index_to_answer = dict(zip(all_choices, options))

    last_answer_pos = response.rfind("Answer:")
    if last_answer_pos != -1:
        answer_str = response[last_answer_pos + len("Answer:") :].strip()
        matching = [choice for choice in all_choices if choice in answer_str]
        if len(matching) == 1:
            return matching[0], False

    response = str(response)
    for char in [",", ".", "!", "?", ";", ":", "'"]:
        response = response.strip(char)
    response = f" {response} "

    index_answer = True
    answer_with_brackets = False
    candidates = [choice for choice in all_choices if f"({choice})" in response]
    if candidates:
        answer_with_brackets = True
    if not candidates:
        candidates = [choice for choice in all_choices if f"{choice} " in response]
    if not candidates:
        candidates = [choice for choice in all_choices if f"{choice}." in response]
    if not candidates and len(response.split()) > 5:
        candidates = [
            choice
            for choice, answer in index_to_answer.items()
            if answer.lower() in response.lower()
        ]
        index_answer = False

    if not candidates:
        return rng.choice(all_choices), True
    if len(candidates) == 1:
        return candidates[0], False

    if index_answer and answer_with_brackets:
        positions = [response.rfind(f"({choice})") for choice in candidates]
    elif index_answer:
        positions = [response.rfind(f" {choice} ") for choice in candidates]
    else:
        positions = [
            response.lower().rfind(index_to_answer[choice].lower())
            for choice in candidates
        ]
    return candidates[max(range(len(positions)), key=positions.__getitem__)], False


def open_answer_prompt(question: str) -> str:
    return (
        f"{question}\n\nSolve the problem and end with exactly "
        "'Answer: <your short answer>'."
    )


def extract_choice(response: str, choices: list[str]) -> str:
    letters = "".join(choices)
    answer_matches = re.findall(
        rf"(?i)answer\s*:\s*\(?([{letters}])\)?\b", response
    )
    if answer_matches:
        return answer_matches[-1].upper()

    bracket_matches = re.findall(rf"\(([{letters}])\)", response.upper())
    if bracket_matches:
        return bracket_matches[-1]

    bare_matches = re.findall(rf"\b([{letters}])(?:\.|\b)", response.upper())
    return bare_matches[-1] if bare_matches else ""


def extract_open_answer(response: str) -> str:
    matches = re.findall(r"(?is)answer\s*:\s*(.+)", response)
    return matches[-1].strip() if matches else response.strip()


def normalize_answer(value: Any) -> str:
    text = str(value).strip().lower().replace(",", "")
    text = re.sub(r"\s+", " ", text)
    try:
        return f"{float(text):.6g}"
    except ValueError:
        return text.rstrip(". ")


def score_open(prediction: str, answer: Any) -> bool:
    gold_answers = answer if isinstance(answer, list) else [answer]
    pred = normalize_answer(prediction)
    return any(normalize_answer(gold) in pred for gold in gold_answers)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read completed records, tolerating only a truncated final line."""
    if not path.exists():
        return []

    rows = []
    lines = path.read_bytes().splitlines(keepends=True)
    for index, raw_line in enumerate(lines):
        if not raw_line.strip():
            continue
        try:
            rows.append(json.loads(raw_line.decode("utf-8")))
        except (UnicodeDecodeError, json.JSONDecodeError):
            if index == len(lines) - 1:
                valid_bytes = sum(len(value) for value in lines[:index])
                with path.open("r+b") as file:
                    file.truncate(valid_bytes)
                print(f"Warning: removed truncated final line in {path}", flush=True)
                break
            raise
    return rows


def load_completed_predictions(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Load prior predictions and reject ambiguous duplicate sample IDs."""
    rows = read_jsonl(path)
    completed_ids: set[str] = set()
    for row in rows:
        sample_id = str(row["id"])
        if sample_id in completed_ids:
            raise ValueError(f"Duplicate sample id {sample_id!r} in {path}")
        completed_ids.add(sample_id)
    return rows, completed_ids


def ensure_run_config(
    config_path: Path,
    predictions_path: Path,
    config: dict[str, Any],
) -> None:
    """Prevent appending predictions produced with incompatible settings."""
    if config_path.exists():
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != config:
            raise ValueError(
                f"Run configuration differs from {config_path}. "
                "Back up/remove the result directory before starting a new experiment."
            )
        return
    if predictions_path.exists():
        raise ValueError(
            f"{predictions_path} exists without config.json; refusing to mix runs. "
            "Back it up or move it before running."
        )
    write_json(config_path, config)


def import_single_legacy_run(
    output_dir: Path,
    predictions_path: Path,
    config_path: Path,
    config: dict[str, Any],
) -> None:
    """One-time import from the former timestamp-directory layout."""
    if predictions_path.exists() or config_path.exists():
        return
    legacy_paths = sorted(output_dir.glob("*/predictions.jsonl"))
    if not legacy_paths:
        return
    if len(legacy_paths) > 1:
        raise ValueError(
            f"Found multiple legacy runs under {output_dir}; choose one and copy it "
            f"to {predictions_path} before resuming."
        )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(legacy_paths[0], predictions_path)
    write_json(config_path, config)
    print(f"Imported legacy predictions from {legacy_paths[0]}", flush=True)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary.replace(path)
