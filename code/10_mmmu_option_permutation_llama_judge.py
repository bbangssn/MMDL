"""Experiment 10: local-Llama judging for Experiment 09 fallbacks."""

import os

# Set before torch/transformers imports. Physical GPU 0 becomes cuda:0.
VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU

from collections import Counter, defaultdict
from datetime import datetime
import json
import random
import re
import time

from tqdm.auto import tqdm

from reporting import collect_environment, runtime_summary
from utils import (
    MODEL_CACHE,
    RESULTS_ROOT,
    TextJudge,
    append_jsonl,
    ensure_run_config,
    load_completed_predictions,
    qwen_rule_extract,
    read_jsonl,
    resolve_model,
    strict_judge_extract,
    strict_judge_prompt,
    write_json,
)


# EDIT THESE SETTINGS
SOURCE_MODEL = 0
PERMUTATION_REPETITIONS = 2
SOURCE_RUN_TAG = f"all-validation-random-shuffle-{PERMUTATION_REPETITIONS}x-official-penalty-1p5"
PERMUTATION_CONDITIONS = tuple(
    f"option_permuted_{index:02d}"
    for index in range(1, PERMUTATION_REPETITIONS + 1)
)
CONDITIONS = PERMUTATION_CONDITIONS
JUDGE_MODEL_NAME = "llama-3.1-8b-instruct"
JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_RUN_TAG = "strict-output-v1-random-fallback"
RANDOM_SEED = 42
LIMIT = None  # Number of fallback rows to judge; None judges every fallback.
LOG_EVERY = 10
MAX_NEW_TOKENS = 8
MAX_MODEL_LENGTH = 9048
PROMPT_VERSION = "llama-strict-output-v1"


def build_judge_prompt(question: str, options: list[str], source_response: str) -> str:
    return strict_judge_prompt(question, options, source_response)


def parse_judge_answer(response: str, options: list[str]) -> tuple[str, str]:
    return strict_judge_extract(response, options)


def canonical_letter(displayed_letter: str, mapping: list[int]) -> str:
    if len(displayed_letter) != 1 or not displayed_letter.isalpha():
        return displayed_letter
    displayed_index = ord(displayed_letter.upper()) - ord("A")
    if not 0 <= displayed_index < len(mapping):
        return displayed_letter
    return chr(ord("A") + mapping[displayed_index])


def group_summary(source: list[dict], final: list[dict]) -> dict:
    judged = [row for row in final if bool(row.get("judge_used"))]
    nonrandom = [row for row in final if not bool(row.get("random_fallback_used"))]
    return {
        "total": len(final),
        "rule_correct": sum(bool(row["correct"]) for row in source),
        "rule_accuracy": (
            sum(bool(row["correct"]) for row in source) / len(source)
            if source else None
        ),
        "rule_fallbacks": sum(bool(row["fallback_used"]) for row in source),
        "judge_resolved": sum(bool(row.get("judge_resolved")) for row in judged),
        "random_fallbacks": sum(
            bool(row.get("random_fallback_used")) for row in judged
        ),
        "final_correct": sum(bool(row["correct"]) for row in final),
        "final_accuracy": (
            sum(bool(row["correct"]) for row in final) / len(final)
            if final else None
        ),
        "nonrandom_total": len(nonrandom),
        "accuracy_excluding_random_fallback": (
            sum(bool(row["correct"]) for row in nonrandom) / len(nonrandom)
            if nonrandom else None
        ),
        "displayed_prediction_counts": dict(sorted(Counter(
            str(row["prediction"]) if str(row["prediction"]) else "unresolved"
            for row in final
        ).items())),
    }


def summarize(source_rows: list[dict], final_rows: list[dict], judge_rows: list[dict]) -> dict:
    conditions = {}
    categories = {}
    subjects = {}
    for condition in CONDITIONS:
        source = [row for row in source_rows if row["condition"] == condition]
        final = [row for row in final_rows if row["condition"] == condition]
        conditions[condition] = group_summary(source, final)
        source_by_category = defaultdict(list)
        final_by_category = defaultdict(list)
        source_by_subject = defaultdict(list)
        final_by_subject = defaultdict(list)
        for row in source:
            source_by_category[row["category"]].append(row)
            source_by_subject[row["subject"]].append(row)
        for row in final:
            final_by_category[row["category"]].append(row)
            final_by_subject[row["subject"]].append(row)
        categories[condition] = {
            key: group_summary(source_by_category[key], value)
            for key, value in sorted(final_by_category.items())
        }
        subjects[condition] = {
            key: group_summary(source_by_subject[key], value)
            for key, value in sorted(final_by_subject.items())
        }
    final_accuracies = [
        conditions[condition]["final_accuracy"]
        for condition in PERMUTATION_CONDITIONS
        if conditions[condition]["final_accuracy"] is not None
    ]
    return {
        "conditions": conditions,
        "categories": categories,
        "subjects": subjects,
        "permutation_repetitions": PERMUTATION_REPETITIONS,
        "mean_final_accuracy": (
            sum(final_accuracies) / len(final_accuracies)
            if final_accuracies else None
        ),
        "judge_parser_methods": dict(sorted(Counter(
            str(row.get("judge_parser_method", "unknown")) for row in judge_rows
        ).items())),
    }


def write_jsonl(path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def main() -> None:
    session_started = time.perf_counter()
    spec = resolve_model(str(SOURCE_MODEL))
    source_dir = (
        RESULTS_ROOT / "09_mmmu_option_permutation" / SOURCE_RUN_TAG / spec.name
    )
    source_path = source_dir / "predictions.jsonl"
    source_summary_path = source_dir / "summary.json"
    source_rows = read_jsonl(source_path)
    if not source_summary_path.exists():
        raise FileNotFoundError(
            f"Missing {source_summary_path}. Complete Experiment 09 first."
        )
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_counts = Counter(str(row["condition"]) for row in source_rows)
    expected_per_condition = int(source_summary["sample_count"])
    expected_counts = {condition: expected_per_condition for condition in CONDITIONS}
    if source_summary.get("status") != "completed" or dict(source_counts) != expected_counts:
        raise RuntimeError(
            f"Experiment 09 must be complete: expected {expected_counts}, "
            f"got {dict(source_counts)} with status={source_summary.get('status')!r}"
        )
    fallback_rows = [row for row in source_rows if bool(row["fallback_used"])]
    fallback_by_id = {str(row["id"]): row for row in fallback_rows}

    output_dir = (
        RESULTS_ROOT / "10_mmmu_option_permutation_llama_judge" /
        SOURCE_RUN_TAG / spec.name / JUDGE_MODEL_NAME / JUDGE_RUN_TAG
    )
    judge_path = output_dir / "judge_predictions.jsonl"
    final_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU validation option-order permutation local judge",
        "source_experiment": "09_mmmu_option_permutation",
        "source_predictions": str(source_path.relative_to(RESULTS_ROOT)),
        "source_model": spec.name,
        "source_run_tag": SOURCE_RUN_TAG,
        "conditions": list(CONDITIONS),
        "judge_model": JUDGE_MODEL_ID,
        "judge_backend": "transformers",
        "judge_run_tag": JUDGE_RUN_TAG,
        "prompt_version": PROMPT_VERSION,
        "judge_retry_count": 1,
        "judge_parser": "exact, last explicit Output:X, last single-letter line, qwen rule",
        "random_fallback": True,
        "random_seed": RANDOM_SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH,
        "visible_gpu": VISIBLE_GPU,
    }
    ensure_run_config(output_dir / "config.json", judge_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    judge_rows, completed_ids = load_completed_predictions(judge_path)
    unknown_ids = completed_ids - fallback_by_id.keys()
    if unknown_ids:
        raise ValueError(f"Unknown completed IDs: {sorted(unknown_ids)[:5]}")

    target_total = min(len(fallback_rows), LIMIT) if LIMIT is not None else len(fallback_rows)
    bar = tqdm(
        total=target_total,
        initial=min(len(judge_rows), target_total),
        desc="MMMU option permutation Llama judge",
        unit="fallback",
        dynamic_ncols=True,
    )
    judge = None
    for source in fallback_rows:
        if LIMIT is not None and len(judge_rows) >= LIMIT:
            break
        record_id = str(source["id"])
        if record_id in completed_ids:
            continue
        options = [str(value) for value in source["displayed_options"]]
        prompt = build_judge_prompt(
            str(source["question"]), options, str(source["response"])
        )
        if judge is None:
            judge = TextJudge(JUDGE_MODEL_ID, MODEL_CACHE / "hub")
        started = time.perf_counter()
        judge_response = judge.generate(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            max_model_length=MAX_MODEL_LENGTH,
        )
        elapsed_seconds = time.perf_counter() - started

        prediction, parser_method = parse_judge_answer(judge_response, options)
        resolved = bool(prediction)
        final_prediction = prediction
        if not resolved:
            fallback_rng = random.Random(f"{RANDOM_SEED}:{record_id}")
            final_prediction = fallback_rng.choice(
                [chr(65 + index) for index in range(len(options))] + ["Z"]
            )
        canonical_prediction = canonical_letter(
            final_prediction, list(source["new_to_original_index"])
        )
        row = {
            "id": record_id,
            "sample_id": str(source["sample_id"]),
            "condition": source["condition"],
            "category": source["category"],
            "subject": source["subject"],
            "source_prediction": source["prediction"],
            "source_response": source["response"],
            "judge_prediction": prediction,
            "judge_raw_output": judge_response,
            "judge_parser_method": parser_method,
            "judge_resolved": resolved,
            "random_fallback_used": not resolved,
            "final_prediction": final_prediction,
            "canonical_prediction": canonical_prediction,
            "gold": source["gold"],
            "correct": canonical_prediction == source["gold"],
            "elapsed_seconds": round(elapsed_seconds, 3),
        }
        append_jsonl(judge_path, row)
        judge_rows.append(row)
        completed_ids.add(record_id)
        bar.update(1)
        bar.set_postfix(
            condition=source["condition"],
            resolved=sum(bool(value["judge_resolved"]) for value in judge_rows),
        )
        if len(judge_rows) % LOG_EVERY == 0:
            write_json(progress_path, {
                "status": "running",
                "judged": len(judge_rows),
                "target": len(fallback_rows),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
    bar.close()

    judged_by_id = {str(row["id"]): row for row in judge_rows}
    final_rows = []
    for source in source_rows:
        final = dict(source)
        judged = judged_by_id.get(str(source["id"]))
        if judged:
            final["prediction"] = judged["final_prediction"]
            final["canonical_prediction"] = judged["canonical_prediction"]
            final["correct"] = bool(judged["correct"])
            final["judge_used"] = True
            final["judge_resolved"] = bool(judged["judge_resolved"])
            final["judge_parser_method"] = judged["judge_parser_method"]
            final["random_fallback_used"] = bool(judged["random_fallback_used"])
        else:
            final["judge_used"] = False
            final["random_fallback_used"] = False
        final_rows.append(final)
    write_jsonl(final_path, final_rows)

    summary = summarize(source_rows, final_rows, judge_rows)
    summary.update({
        "benchmark": "MMMU validation option-order permutation local judge",
        "status": "completed" if len(judge_rows) >= len(fallback_rows) else "limit_reached",
        "source_model": spec.name,
        "judge_model": JUDGE_MODEL_ID,
        "source_samples": len(source_rows),
        "fallback_samples": len(fallback_rows),
        "judged_samples": len(judge_rows),
        "judge_resolved": sum(bool(row["judge_resolved"]) for row in judge_rows),
        "random_fallback_samples": sum(
            bool(row["random_fallback_used"]) for row in judge_rows
        ),
        "runtime": {
            **runtime_summary(judge_rows),
            "current_session_elapsed_seconds": round(
                time.perf_counter() - session_started, 3
            ),
        },
    })
    write_json(output_dir / "summary.json", summary)
    write_json(progress_path, {
        **summary,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
