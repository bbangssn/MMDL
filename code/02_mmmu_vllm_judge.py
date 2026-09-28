"""Experiment 02-vLLM: judge MMMU parser fallbacks from the vLLM run."""

import os

# Set this before importing torch/transformers. Physical GPU 0 becomes cuda:0.
VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU

from datetime import datetime
import re
import time

from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import (
    aggregate_rows,
    collect_environment,
    normalize_prediction_rows,
    runtime_summary,
)
from utils import (
    DATASET_CACHE,
    MODEL_CACHE,
    MMMU_SUBJECTS,
    RESULTS_ROOT,
    TextJudge,
    append_jsonl,
    ensure_run_config,
    load_completed_predictions,
    parse_options,
    qwen_rule_extract,
    resolve_model,
    write_json,
)


DATA_PATH = os.environ.get("MMMU_DATA_PATH", "MMMU/MMMU")
EXPECTED_SOURCE_TOTAL = 900

# EDIT THESE SETTINGS
SOURCE_MODEL = 0  # Index in utils.MODELS
SOURCE_RUN_TAG = "official-penalty-1p5"
JUDGE_MODEL_NAME = "llama-3.1-8b-instruct"
JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
LIMIT = None  # Number of fallback samples to judge; None judges all of them
LOG_EVERY = 10
MAX_NEW_TOKENS = 4096
MAX_MODEL_LENGTH = 9048
PROMPT_VERSION = "qwen3-vl-official-build_prompt"


def build_judge_prompt(
    question: str,
    options: list[str],
    source_response: str,
    open_question_reformatted: bool,
) -> str:
    """Qwen3-VL official MMMU eval_utils.build_prompt, reproduced verbatim."""
    del open_question_reformatted  # The official prompt handles every item identically.
    option_text = "There are several options: \n"
    for index, option in enumerate(options):
        option_text += f"{chr(65 + index)}. {option}\n"
    template = (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z. \n"
        "Example 1: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2: \n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: Z\n"
        "Example 3: \n"
        "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
    )
    return template.format(question, option_text, source_response)


def parse_judge_answer(response: str, options: list[str]) -> str:
    """Apply the official rule/text inference logic to the judge response."""
    stripped = response.strip().upper()
    if stripped == "Z":
        return ""
    return qwen_rule_extract(response, options)


def main() -> None:
    session_started = time.perf_counter()
    source_spec = resolve_model(str(SOURCE_MODEL))
    source_dir = (
        RESULTS_ROOT / "00_mmmu_vllm" / SOURCE_RUN_TAG / source_spec.name
    )
    source_path = source_dir / "predictions.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing vLLM predictions: {source_path}. "
            "Run code/00_mmmu_vllm.py first."
        )

    source_rows, _ = load_completed_predictions(source_path)
    normalize_prediction_rows(source_rows)
    if len(source_rows) != EXPECTED_SOURCE_TOTAL:
        raise RuntimeError(
            f"Judge requires the complete MMMU Validation run: "
            f"{len(source_rows)}/{EXPECTED_SOURCE_TOTAL} samples found."
        )

    fallback_rows = [
        row
        for row in source_rows
        if bool(row.get("fallback_used", row.get("prediction") in ("", "Z", None)))
    ]
    fallback_by_id = {str(row["id"]): row for row in fallback_rows}

    output_dir = (
        RESULTS_ROOT / "02_mmmu_vllm_judge" / SOURCE_RUN_TAG /
        source_spec.name / JUDGE_MODEL_NAME
    )
    predictions_path = output_dir / "judge_predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU validation vLLM LLM-as-a-Judge",
        "dataset": DATA_PATH,
        "split": "validation",
        "source_model": source_spec.name,
        "source_backend": "vllm",
        "source_run_tag": SOURCE_RUN_TAG,
        "source_predictions": str(source_path.relative_to(RESULTS_ROOT)),
        "judge_model": JUDGE_MODEL_NAME,
        "judge_model_path": JUDGE_MODEL_ID,
        "judge_backend": "transformers",
        "judge_scope": "rule-parser fallbacks only",
        "prompt_version": PROMPT_VERSION,
        "official_prompt": True,
        "official_default_judge": "gpt-3.5-turbo-0125",
        "do_sample": False,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH,
    }
    config_path = output_dir / "config.json"
    ensure_run_config(config_path, predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    judge_rows, completed_ids = load_completed_predictions(predictions_path)
    unknown_ids = completed_ids - fallback_by_id.keys()
    if unknown_ids:
        raise ValueError(
            f"Judge output contains IDs absent from current fallback set: "
            f"{sorted(unknown_ids)[:5]}"
        )
    if judge_rows:
        print(f"Resuming from {len(judge_rows)} judged samples.", flush=True)

    target_total = min(len(fallback_rows), LIMIT) if LIMIT is not None else len(fallback_rows)
    progress_bar = tqdm(
        total=target_total,
        initial=min(len(judge_rows), target_total),
        desc="MMMU Llama judge",
        unit="sample",
        dynamic_ncols=True,
    )
    judge = None

    for subject in MMMU_SUBJECTS:
        dataset = load_dataset(
            DATA_PATH,
            subject,
            split="validation",
            cache_dir=DATASET_CACHE,
        )
        for sample in dataset:
            if LIMIT is not None and len(judge_rows) >= LIMIT:
                break

            sample_id = str(sample["id"])
            if sample_id not in fallback_by_id or sample_id in completed_ids:
                continue

            source = fallback_by_id[sample_id]
            options = parse_options(sample["options"])
            open_question = not options
            if open_question:
                options = [str(sample["answer"]).strip(), "Other Answers"]
                gold = "A"
            else:
                gold = str(sample["answer"]).strip().upper()

            if judge is None:
                judge = TextJudge(JUDGE_MODEL_ID, MODEL_CACHE / "hub")

            prompt = build_judge_prompt(
                str(sample["question"]),
                options,
                str(source["response"]),
                open_question,
            )
            started = time.perf_counter()
            judge_response = judge.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                max_model_length=MAX_MODEL_LENGTH,
            )
            elapsed_seconds = time.perf_counter() - started
            prediction = parse_judge_answer(judge_response, options)
            resolved = bool(prediction)
            correct = resolved and prediction == gold

            row = {
                "id": sample["id"],
                "category": source["category"],
                "subject": subject,
                "source_prediction": source.get("prediction", ""),
                "source_response": source["response"],
                "judge_prediction": prediction,
                "judge_raw_output": judge_response,
                "judge_resolved": resolved,
                "final_prediction": prediction if resolved else source.get("prediction", ""),
                "gold": gold,
                "correct": correct,
                "open_question_reformatted": open_question,
                "elapsed_seconds": round(elapsed_seconds, 3),
            }
            append_jsonl(predictions_path, row)
            judge_rows.append(row)
            completed_ids.add(sample_id)

            resolved_count = sum(bool(value["judge_resolved"]) for value in judge_rows)
            recovered_correct = sum(bool(value["correct"]) for value in judge_rows)
            progress_bar.update(1)
            progress_bar.set_postfix(
                resolved=resolved_count,
                recovered=recovered_correct,
                subject=subject,
            )

            if len(judge_rows) % LOG_EVERY == 0:
                write_json(progress_path, {
                    "status": "running",
                    "judged": len(judge_rows),
                    "target": len(fallback_rows),
                    "resolved": resolved_count,
                    "recovered_correct": recovered_correct,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

        if LIMIT is not None and len(judge_rows) >= LIMIT:
            break

    progress_bar.close()
    judged_by_id = {str(row["id"]): row for row in judge_rows}
    final_rows = []
    for source in source_rows:
        final = dict(source)
        judged = judged_by_id.get(str(source["id"]))
        if judged and judged["judge_resolved"]:
            final["prediction"] = judged["judge_prediction"]
            final["correct"] = bool(judged["correct"])
            final["judge_used"] = True
        else:
            final["judge_used"] = False
        final_rows.append(final)

    baseline_correct = sum(bool(row["correct"]) for row in source_rows)
    final_correct = sum(bool(row["correct"]) for row in final_rows)
    resolved_count = sum(bool(row["judge_resolved"]) for row in judge_rows)
    recovered_correct = sum(bool(row["correct"]) for row in judge_rows)
    status = "completed" if len(judge_rows) >= len(fallback_rows) else "limit_reached"
    summary = {
        "benchmark": "MMMU validation vLLM LLM-as-a-Judge",
        "status": status,
        "source_model": source_spec.name,
        "source_backend": "vllm",
        "source_run_tag": SOURCE_RUN_TAG,
        "judge_model": JUDGE_MODEL_NAME,
        "total_samples": len(source_rows),
        "fallback_samples": len(fallback_rows),
        "judged_samples": len(judge_rows),
        "judge_resolved": resolved_count,
        "judge_unresolved": len(judge_rows) - resolved_count,
        "recovered_correct": recovered_correct,
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / len(source_rows),
        "final_correct": final_correct,
        "final_accuracy": final_correct / len(source_rows),
        "accuracy_delta": (final_correct - baseline_correct) / len(source_rows),
        "judge_protocol": {
            "official_prompt": True,
            "official_judge_model": False,
            "official_default_judge": "gpt-3.5-turbo-0125",
            "local_judge_substitution": JUDGE_MODEL_ID,
            "scope": "rule-parser fallbacks only",
            "prompt_version": PROMPT_VERSION,
            "retry_count": 1,
            "random_fallback": False,
            "known_differences": [
                "Llama 3.1 8B replaces the official default GPT-3.5 judge.",
                "Deterministic local inference is not retried 25 times.",
                "Failed extraction remains unresolved instead of random A/B/C/D/Z fallback.",
            ],
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_model_length": MAX_MODEL_LENGTH,
        },
        "runtime": {
            **runtime_summary(judge_rows),
            "current_session_elapsed_seconds": round(
                time.perf_counter() - session_started, 3
            ),
        },
        "categories": aggregate_rows(final_rows, "category"),
        "subjects": aggregate_rows(final_rows, "subject"),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(progress_path, {
        **summary,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(summary)


if __name__ == "__main__":
    main()
