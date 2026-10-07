"""Experiment 09: repeated option-order shuffles on MMMU Validation."""

import os

# Set before torch/transformers imports. Physical GPU 0 becomes cuda:0.
VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import random
import time

from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment, runtime_summary
from utils import (
    DATASET_CACHE,
    MMMU_SUBJECTS,
    RESULTS_ROOT,
    VLLM,
    append_jsonl,
    category_for_subject,
    ensure_run_config,
    get_images,
    load_completed_predictions,
    parse_options,
    qwen_mmmu_prompt,
    qwen_rule_extract,
    resolve_model,
    write_json,
)


DATA_PATH = os.environ.get("MMMU_DATA_PATH", "MMMU/MMMU")
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"

# EDIT THESE SETTINGS
MODEL = 0
PERMUTATION_REPETITIONS = 2
RUN_TAG = f"all-validation-random-shuffle-{PERMUTATION_REPETITIONS}x-official-penalty-1p5"
PERMUTATION_CONDITIONS = tuple(
    f"option_permuted_{index:02d}"
    for index in range(1, PERMUTATION_REPETITIONS + 1)
)
EXPECTED_MCQ_COUNT = 847
BATCH_SIZE = 32
LOG_EVERY_BATCHES = 1
MAX_NEW_TOKENS = 2048
MAX_MODEL_LENGTH = 9048
GPU_MEMORY_UTILIZATION = 0.5
MAX_IMAGES_PER_PROMPT = 7
REPETITION_PENALTY = 1.0
PRESENCE_PENALTY = 1.5
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
SEED = 42
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
CONDITIONS = PERMUTATION_CONDITIONS


def load_validation_mcqs() -> list[dict]:
    """Load every multiple-choice item from all MMMU validation subjects."""
    samples = []
    for subject in MMMU_SUBJECTS:
        dataset = load_dataset(
            DATA_PATH,
            subject,
            split="validation",
            cache_dir=DATASET_CACHE,
            revision=(DATASET_REVISION if DATA_PATH == "MMMU/MMMU" else None),
        )
        for raw in dataset:
            if str(raw["question_type"]) == "multiple-choice":
                sample = dict(raw)
                sample["_subject"] = subject
                samples.append(sample)
    if len(samples) != EXPECTED_MCQ_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_MCQ_COUNT} MMMU validation MCQs, got {len(samples)}. "
            "Check the pinned dataset revision before running the experiment."
        )
    return samples


def shuffle_options(
    options: list[str], sample_id: str, repetition: int
) -> tuple[list[str], list[int]]:
    """Create a deterministic non-identity shuffle for one sample/repetition."""
    mapping = list(range(len(options)))
    rng = random.Random(f"{SEED}:option-shuffle:{repetition}:{sample_id}")
    while mapping == list(range(len(options))):
        rng.shuffle(mapping)
    return [options[index] for index in mapping], mapping


def canonical_letter(displayed_letter: str, mapping: list[int]) -> str:
    if len(displayed_letter) != 1 or not displayed_letter.isalpha():
        return displayed_letter
    displayed_index = ord(displayed_letter.upper()) - ord("A")
    if not 0 <= displayed_index < len(mapping):
        return displayed_letter
    return chr(ord("A") + mapping[displayed_index])


def per_sample_seed(sample_id: str) -> int:
    """Use common random numbers across shuffle repetitions."""
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def group_summary(group: list[dict]) -> dict:
    correct = sum(bool(row["correct"]) for row in group)
    nonfallback = [row for row in group if not bool(row["fallback_used"])]
    return {
        "total": len(group),
        "correct": correct,
        "accuracy": correct / len(group) if group else None,
        "fallbacks": sum(bool(row["fallback_used"]) for row in group),
        "nonfallback_total": len(nonfallback),
        "accuracy_excluding_fallback": (
            sum(bool(row["correct"]) for row in nonfallback) / len(nonfallback)
            if nonfallback else None
        ),
        "runtime": runtime_summary(group),
        "displayed_prediction_counts": dict(sorted(Counter(
            str(row["prediction"]) if str(row["prediction"]) else "unresolved"
            for row in group
        ).items())),
    }


def summarize(rows: list[dict], sample_count: int) -> dict:
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    conditions = {
        condition: group_summary(by_condition.get(condition, []))
        for condition in CONDITIONS
    }
    categories = {}
    subjects = {}
    for condition in CONDITIONS:
        category_groups = defaultdict(list)
        subject_groups = defaultdict(list)
        for row in by_condition.get(condition, []):
            category_groups[row["category"]].append(row)
            subject_groups[row["subject"]].append(row)
        categories[condition] = {
            key: group_summary(value)
            for key, value in sorted(category_groups.items())
        }
        subjects[condition] = {
            key: group_summary(value)
            for key, value in sorted(subject_groups.items())
        }
    permutation_accuracies = [
        conditions[condition]["accuracy"]
        for condition in PERMUTATION_CONDITIONS
        if conditions[condition]["accuracy"] is not None
    ]
    return {
        "sample_count": sample_count,
        "permutation_repetitions": PERMUTATION_REPETITIONS,
        "conditions": conditions,
        "categories": categories,
        "subjects": subjects,
        "mean_permuted_accuracy": (
            sum(permutation_accuracies) / len(permutation_accuracies)
            if permutation_accuracies else None
        ),
    }


def main() -> None:
    session_started = time.perf_counter()
    spec = resolve_model(str(MODEL))
    samples = load_validation_mcqs()
    sample_count = len(samples)

    output_dir = (
        RESULTS_ROOT / "09_mmmu_option_permutation" / RUN_TAG / spec.name
    )
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU validation option-order permutation",
        "dataset": DATA_PATH,
        "dataset_revision": DATASET_REVISION if DATA_PATH == "MMMU/MMMU" else None,
        "split": "validation",
        "model": spec.name,
        "model_path": spec.path,
        "sample_count": sample_count,
        "selection": "all multiple-choice items in MMMU validation",
        "conditions": list(CONDITIONS),
        "permutation_repetitions": PERMUTATION_REPETITIONS,
        "permutation": "deterministic per-sample non-identity random shuffle",
        "backend": "vllm",
        "parser": "qwen_rule_extract; no random fallback",
        "batch_size": BATCH_SIZE,
        "max_images_per_prompt": MAX_IMAGES_PER_PROMPT,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "repetition_penalty": REPETITION_PENALTY,
        "presence_penalty": PRESENCE_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
        "visible_gpu": VISIBLE_GPU,
    }
    ensure_run_config(output_dir / "config.json", predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    rows, completed_ids = load_completed_predictions(predictions_path)
    expected_ids = {
        f"{condition}:{sample['id']}"
        for condition in CONDITIONS
        for sample in samples
    }
    unknown_ids = completed_ids - expected_ids
    if unknown_ids:
        raise ValueError(f"Unknown completed IDs: {sorted(unknown_ids)[:5]}")

    target_total = sample_count * len(CONDITIONS)
    bar = tqdm(
        total=target_total,
        initial=min(len(completed_ids), target_total),
        desc="MMMU option permutation",
        unit="condition",
        dynamic_ncols=True,
    )
    runner = None
    generated_this_session = 0
    batches_done = 0
    for condition in CONDITIONS:
        pending = [
            sample for sample in samples
            if f"{condition}:{sample['id']}" not in completed_ids
        ]
        if pending and runner is None:
            runner = VLLM(
                spec,
                MAX_MODEL_LENGTH,
                GPU_MEMORY_UTILIZATION,
                MAX_IMAGES_PER_PROMPT,
                SEED,
            )
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            requests = []
            metadata = []
            for sample in batch:
                sample_id = str(sample["id"])
                original_options = parse_options(sample["options"])
                if len(original_options) < 2:
                    raise ValueError(f"Expected MCQ options for {sample_id}")
                repetition = int(condition.rsplit("_", 1)[1])
                displayed_options, mapping = shuffle_options(
                    original_options, sample_id, repetition
                )
                prompt = qwen_mmmu_prompt(
                    str(sample["question"]), displayed_options
                )
                requests.append({
                    "prompt": prompt,
                    "images": get_images(sample),
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "generation_kwargs": {
                        "do_sample": True,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "repetition_penalty": REPETITION_PENALTY,
                        "presence_penalty": PRESENCE_PENALTY,
                        "seed": per_sample_seed(sample_id),
                    },
                    "min_pixels": MIN_PIXELS,
                    "max_pixels": MAX_PIXELS,
                })
                metadata.append((sample, original_options, displayed_options, mapping))

            batch_started = time.perf_counter()
            responses = runner.generate_batch(requests)
            batch_elapsed = time.perf_counter() - batch_started
            for response, metadata_item in zip(responses, metadata):
                sample, original_options, displayed_options, mapping = metadata_item
                sample_id = str(sample["id"])
                prediction = qwen_rule_extract(response, displayed_options)
                fallback = prediction in ("", "Z")
                canonical_prediction = canonical_letter(prediction, mapping)
                gold = str(sample["answer"]).strip().upper()
                row = {
                    "id": f"{condition}:{sample_id}",
                    "sample_id": sample_id,
                    "condition": condition,
                    "category": category_for_subject(str(sample["_subject"])),
                    "subject": str(sample["_subject"]),
                    "question_type": "multiple-choice",
                    "question": str(sample["question"]),
                    "ground_truth": gold,
                    "gold": gold,
                    "original_options": original_options,
                    "displayed_options": displayed_options,
                    "new_to_original_index": mapping,
                    "prediction": prediction,
                    "canonical_prediction": canonical_prediction,
                    "response": response,
                    "correct": canonical_prediction == gold,
                    "fallback_used": fallback,
                    "image_count": len(get_images(sample)),
                    "elapsed_seconds": round(batch_elapsed / len(batch), 3),
                    "runtime_measurement": "batch_wall_time_divided_by_batch_size",
                    "batch_elapsed_seconds": round(batch_elapsed, 3),
                }
                append_jsonl(predictions_path, row)
                rows.append(row)
                completed_ids.add(str(row["id"]))
                generated_this_session += 1
                bar.update(1)

            batches_done += 1
            current = summarize(rows, sample_count)
            accuracy = current["conditions"][condition]["accuracy"]
            bar.set_postfix(
                condition=condition,
                accuracy=f"{accuracy:.3f}",
                batch=f"{len(batch)} in {batch_elapsed:.1f}s",
            )
            if batches_done % LOG_EVERY_BATCHES == 0:
                write_json(progress_path, {
                    "status": "running",
                    "completed_conditions": len(completed_ids),
                    "target_conditions": target_total,
                    "summary": current,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })
    bar.close()

    summary = summarize(rows, sample_count)
    summary.update({
        "benchmark": "MMMU validation option-order permutation",
        "model": spec.name,
        "status": "completed" if len(completed_ids) == target_total else "partial",
        "generated_this_session": generated_this_session,
        "session_elapsed_seconds": round(time.perf_counter() - session_started, 3),
    })
    write_json(output_dir / "summary.json", summary)
    write_json(progress_path, {
        **summary,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
