"""Experiment 06: judge MMMU parser fallbacks with the OpenAI API."""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)
from tqdm.auto import tqdm

from reporting import (
    aggregate_rows,
    collect_environment,
    normalize_prediction_rows,
    runtime_summary,
)
from utils import (
    DATASET_CACHE,
    MMMU_SUBJECTS,
    RESULTS_ROOT,
    ROOT,
    append_jsonl,
    ensure_run_config,
    load_completed_predictions,
    parse_options,
    qwen_rule_extract,
    resolve_model,
    write_json,
)


DATA_PATH = "MMMU/MMMU"
EXPECTED_SOURCE_TOTAL = 900

# EDIT THESE SETTINGS
SOURCE_MODEL = 0  # Index in utils.MODELS
SOURCE_RUN_TAG = "official-penalty-1p5"
JUDGE_MODEL = "gpt-3.5-turbo-0125"
JUDGE_RUN_TAG = "official-prompt-25-retries"
KEY_FILE = ROOT / "openai_key.json"

# Qwen's public MMMU evaluator re-queries an unresolved judge response up to 25 times.
JUDGE_RETRY_COUNT = 5
# Qwen's API wrapper separately retries transient request failures five times.
API_REQUEST_ATTEMPTS = 5
RETRY_WAIT_SECONDS = 5.0
API_TIMEOUT_SECONDS = 60.0
MAX_COMPLETION_TOKENS = 4096

RANDOM_SEED = 42
LIMIT = None  # Number of fallback samples to judge; None judges all of them
LOG_EVERY = 10
PROMPT_VERSION = "qwen3-vl-official-build_prompt"
API_FAILURE_MESSAGE = "Failed to obtain answer via API."


def load_api_config(path: Path) -> tuple[str, str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing API key file: {path}. Fill in openai_key.json first."
        )
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    api_key = str(config.get("OPENAI_API_KEY", "")).strip()
    base_url = str(
        config.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    ).strip()
    if not api_key:
        raise ValueError(f"OPENAI_API_KEY is empty in {path}.")
    return api_key, base_url


def build_judge_prompt(
    question: str,
    options: list[str],
    source_response: str,
) -> str:
    """Qwen3-VL public MMMU eval_utils.build_prompt, reproduced verbatim."""
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
    """Return a valid option, or an empty string for Z/ambiguous output."""
    if response.strip().upper() == "Z":
        return ""
    prediction = qwen_rule_extract(response, options)
    return "" if prediction == "Z" else prediction


def request_judge(client: OpenAI, prompt: str, rng: random.Random) -> str:
    """Make one judge query, retrying transient API failures separately."""
    last_error: Exception | None = None
    for attempt in range(1, API_REQUEST_ATTEMPTS + 1):
        try:
            completion = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=MAX_COMPLETION_TOKENS,
            )
            return (completion.choices[0].message.content or "").strip()
        except (
            APIConnectionError,
            APITimeoutError,
            InternalServerError,
            RateLimitError,
        ) as error:
            last_error = error
            if attempt < API_REQUEST_ATTEMPTS:
                time.sleep(rng.random() * RETRY_WAIT_SECONDS * 2)
    if last_error is not None:
        print(
            f"OpenAI API failed after {API_REQUEST_ATTEMPTS} attempts: "
            f"{type(last_error).__name__}",
            flush=True,
        )
    return API_FAILURE_MESSAGE


def judge_until_resolved(
    client: OpenAI,
    prompt: str,
    options: list[str],
    rng: random.Random,
) -> tuple[str, str, int, list[str]]:
    """Repeat official-style judge queries until a non-Z option is extracted."""
    responses: list[str] = []
    for attempt in range(1, JUDGE_RETRY_COUNT + 1):
        response = request_judge(client, prompt, rng)
        responses.append(response)
        prediction = parse_judge_answer(response, options)
        if prediction:
            return prediction, response, attempt, responses
        if attempt < JUDGE_RETRY_COUNT:
            time.sleep(rng.random() * RETRY_WAIT_SECONDS * 2)
    return "", responses[-1] if responses else "", JUDGE_RETRY_COUNT, responses


def main() -> None:
    if JUDGE_RETRY_COUNT < 1 or API_REQUEST_ATTEMPTS < 1:
        raise ValueError("Retry counts must be positive integers.")

    session_started = time.perf_counter()
    source_spec = resolve_model(str(SOURCE_MODEL))
    source_dir = RESULTS_ROOT / "00_mmmu_vllm" / SOURCE_RUN_TAG / source_spec.name
    source_path = source_dir / "predictions.jsonl"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing vLLM predictions: {source_path}. Run code/00_mmmu_vllm.py first."
        )

    source_rows, _ = load_completed_predictions(source_path)
    normalize_prediction_rows(source_rows)
    if len(source_rows) != EXPECTED_SOURCE_TOTAL:
        raise RuntimeError(
            "Judge requires the complete MMMU Validation run: "
            f"{len(source_rows)}/{EXPECTED_SOURCE_TOTAL} samples found."
        )

    fallback_rows = [
        row for row in source_rows
        if bool(row.get("fallback_used", row.get("prediction") in ("", "Z", None)))
    ]
    fallback_by_id = {str(row["id"]): row for row in fallback_rows}

    safe_judge_name = JUDGE_MODEL.replace("/", "--")
    output_dir = (
        RESULTS_ROOT / "06_mmmu_openai_judge" / SOURCE_RUN_TAG /
        source_spec.name / safe_judge_name / JUDGE_RUN_TAG
    )
    predictions_path = output_dir / "judge_predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU validation OpenAI LLM-as-a-Judge",
        "dataset": DATA_PATH,
        "split": "validation",
        "source_model": source_spec.name,
        "source_backend": "vllm",
        "source_run_tag": SOURCE_RUN_TAG,
        "source_predictions": str(source_path.relative_to(RESULTS_ROOT)),
        "judge_model": JUDGE_MODEL,
        "judge_backend": "openai-api",
        "judge_run_tag": JUDGE_RUN_TAG,
        "judge_scope": "rule-parser fallbacks only",
        "prompt_version": PROMPT_VERSION,
        "official_prompt": True,
        "temperature": 0,
        "judge_retry_count": JUDGE_RETRY_COUNT,
        "api_request_attempts": API_REQUEST_ATTEMPTS,
        "api_timeout_seconds": API_TIMEOUT_SECONDS,
        "retry_wait_seconds": RETRY_WAIT_SECONDS,
        "max_completion_tokens": MAX_COMPLETION_TOKENS,
        "random_fallback": True,
        "random_seed": RANDOM_SEED,
    }
    ensure_run_config(output_dir / "config.json", predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    judge_rows, completed_ids = load_completed_predictions(predictions_path)
    unknown_ids = completed_ids - fallback_by_id.keys()
    if unknown_ids:
        raise ValueError(
            "Judge output contains IDs absent from the current fallback set: "
            f"{sorted(unknown_ids)[:5]}"
        )
    if judge_rows:
        print(f"Resuming from {len(judge_rows)} judged samples.", flush=True)

    api_key, base_url = load_api_config(KEY_FILE)
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=API_TIMEOUT_SECONDS,
        max_retries=0,
    )
    target_total = min(len(fallback_rows), LIMIT) if LIMIT is not None else len(fallback_rows)
    progress_bar = tqdm(
        total=target_total,
        initial=min(len(judge_rows), target_total),
        desc=f"MMMU {JUDGE_MODEL} judge",
        unit="sample",
        dynamic_ncols=True,
    )

    stop = False
    for subject in MMMU_SUBJECTS:
        dataset = load_dataset(
            DATA_PATH,
            subject,
            split="validation",
            cache_dir=DATASET_CACHE,
        )
        for sample in dataset:
            if LIMIT is not None and len(judge_rows) >= LIMIT:
                stop = True
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

            prompt = build_judge_prompt(
                str(sample["question"]), options, str(source["response"])
            )
            sample_rng = random.Random(f"{RANDOM_SEED}:{sample_id}")
            started = time.perf_counter()
            prediction, last_response, attempts, responses = judge_until_resolved(
                client, prompt, options, sample_rng
            )
            elapsed_seconds = time.perf_counter() - started

            resolved = bool(prediction)
            final_prediction = prediction
            if not resolved:
                final_prediction = sample_rng.choice(
                    [chr(65 + index) for index in range(len(options))] + ["Z"]
                )

            row = {
                "id": sample["id"],
                "category": source["category"],
                "subject": subject,
                "source_prediction": source.get("prediction", ""),
                "source_response": source["response"],
                "judge_prediction": prediction,
                "judge_raw_output": last_response,
                "judge_all_outputs": responses,
                "judge_attempts": attempts,
                "judge_resolved": resolved,
                "random_fallback_used": not resolved,
                "final_prediction": final_prediction,
                "gold": gold,
                "correct": final_prediction == gold,
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
        if stop:
            break

    progress_bar.close()
    judged_by_id = {str(row["id"]): row for row in judge_rows}
    final_rows = []
    for source in source_rows:
        final = dict(source)
        judged = judged_by_id.get(str(source["id"]))
        if judged:
            final["prediction"] = judged["final_prediction"]
            final["correct"] = bool(judged["correct"])
            final["judge_used"] = True
            final["random_fallback_used"] = bool(judged["random_fallback_used"])
        else:
            final["judge_used"] = False
        final_rows.append(final)

    baseline_correct = sum(bool(row["correct"]) for row in source_rows)
    final_correct = sum(bool(row["correct"]) for row in final_rows)
    resolved_count = sum(bool(row["judge_resolved"]) for row in judge_rows)
    status = "completed" if len(judge_rows) >= len(fallback_rows) else "limit_reached"
    summary = {
        "benchmark": "MMMU validation OpenAI LLM-as-a-Judge",
        "status": status,
        "source_model": source_spec.name,
        "source_backend": "vllm",
        "source_run_tag": SOURCE_RUN_TAG,
        "judge_model": JUDGE_MODEL,
        "total_samples": len(source_rows),
        "fallback_samples": len(fallback_rows),
        "judged_samples": len(judge_rows),
        "judge_resolved": resolved_count,
        "judge_unresolved": len(judge_rows) - resolved_count,
        "random_fallback_samples": sum(
            bool(row["random_fallback_used"]) for row in judge_rows
        ),
        "total_api_queries": sum(int(row["judge_attempts"]) for row in judge_rows),
        "recovered_correct": sum(bool(row["correct"]) for row in judge_rows),
        "baseline_correct": baseline_correct,
        "baseline_accuracy": baseline_correct / len(source_rows),
        "final_correct": final_correct,
        "final_accuracy": final_correct / len(source_rows),
        "accuracy_delta": (final_correct - baseline_correct) / len(source_rows),
        "judge_protocol": config,
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
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
