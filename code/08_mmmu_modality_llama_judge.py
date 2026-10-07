"""Experiment 08: local-Llama judging for Experiment 07 parser fallbacks."""

import os

# Set before torch/transformers imports. Physical GPU 0 becomes cuda:0.
VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU

from collections import defaultdict
from datetime import datetime
import json
import random
import time

from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment, runtime_summary
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
    read_jsonl,
    resolve_model,
    strict_judge_extract,
    strict_judge_prompt,
    write_json,
)


DATA_PATH = os.environ.get("MMMU_DATA_PATH", "MMMU/MMMU")
EXPECTED_SOURCE_TOTAL = 800
CONDITIONS = ("full", "text_only", "shuffled_image", "blank_image")

# EDIT THESE SETTINGS
SOURCE_MODEL = 0
SOURCE_RUN_TAG = "official-penalty-1p5"
JUDGE_MODEL_NAME = "llama-3.1-8b-instruct"
JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_RUN_TAG = "strict-output-v1-random-fallback"
REUSE_BASELINE_JUDGE_OUTPUTS = False
RANDOM_SEED = 42
LIMIT = None  # Number of fallback rows to judge; None judges all conditions.
LOG_EVERY = 10
MAX_NEW_TOKENS = 8
MAX_MODEL_LENGTH = 9048
BOOTSTRAP_REPETITIONS = 5000
PROMPT_VERSION = "llama-strict-output-v1"


def build_judge_prompt(question: str, options: list[str], source_response: str) -> str:
    return strict_judge_prompt(question, options, source_response)


def parse_judge_answer(response: str, options: list[str]) -> str:
    return strict_judge_extract(response, options)[0]


def load_sample_map() -> dict[str, dict]:
    samples = {}
    for subject in MMMU_SUBJECTS:
        dataset = load_dataset(
            DATA_PATH, subject, split="validation", cache_dir=DATASET_CACHE
        )
        for raw in dataset:
            sample = dict(raw)
            sample["_subject"] = subject
            samples[str(sample["id"])] = sample
    return samples


def evaluation_target(sample: dict) -> tuple[list[str], str, bool]:
    options = parse_options(sample["options"])
    answer = str(sample["answer"]).strip()
    if not options:
        return [answer, "Other Answers"], "A", True
    return options, answer.upper(), False


def bootstrap_delta_ci(final_rows: list[dict], condition: str) -> list[float] | None:
    by_condition = defaultdict(dict)
    for row in final_rows:
        by_condition[row["condition"]][str(row["sample_id"])] = bool(row["correct"])
    full = by_condition.get("full", {})
    other = by_condition.get(condition, {})
    sample_ids = sorted(full.keys() & other.keys())
    if len(sample_ids) != 200:
        return None
    differences = [int(other[i]) - int(full[i]) for i in sample_ids]
    rng = random.Random(RANDOM_SEED + 800)
    draws = [
        sum(rng.choice(differences) for _ in sample_ids) / len(sample_ids)
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def group_summary(source: list[dict], final: list[dict]) -> dict:
    judge_rows = [row for row in final if row.get("judge_used")]
    return {
        "total": len(final),
        "rule_correct": sum(bool(row["correct"]) for row in source),
        "rule_accuracy": (
            sum(bool(row["correct"]) for row in source) / len(source) if source else None
        ),
        "rule_fallbacks": sum(bool(row["fallback_used"]) for row in source),
        "judge_resolved": sum(bool(row.get("judge_resolved")) for row in judge_rows),
        "random_fallbacks": sum(
            bool(row.get("random_fallback_used")) for row in judge_rows
        ),
        "final_correct": sum(bool(row["correct"]) for row in final),
        "final_accuracy": (
            sum(bool(row["correct"]) for row in final) / len(final) if final else None
        ),
    }


def summarize(source_rows: list[dict], final_rows: list[dict]) -> dict:
    conditions = {}
    categories = {}
    for condition in CONDITIONS:
        source_group = [row for row in source_rows if row["condition"] == condition]
        final_group = [row for row in final_rows if row["condition"] == condition]
        conditions[condition] = group_summary(source_group, final_group)
        if condition != "full":
            conditions[condition]["final_delta_vs_full"] = (
                conditions[condition]["final_accuracy"]
                - conditions["full"]["final_accuracy"]
            )
            conditions[condition]["final_delta_95pct_bootstrap_ci"] = (
                bootstrap_delta_ci(final_rows, condition)
            )

        category_source = defaultdict(list)
        category_final = defaultdict(list)
        for row in source_group:
            category_source[row["category"]].append(row)
        for row in final_group:
            category_final[row["category"]].append(row)
        categories[condition] = {
            category: group_summary(category_source[category], group)
            for category, group in sorted(category_final.items())
        }
    return {"conditions": conditions, "categories": categories}


def write_final_predictions(path, rows: list[dict]) -> None:
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
        RESULTS_ROOT / "07_mmmu_modality_ablation" / SOURCE_RUN_TAG / spec.name
    )
    source_path = source_dir / "predictions.jsonl"
    source_rows = read_jsonl(source_path)
    if len(source_rows) != EXPECTED_SOURCE_TOTAL:
        raise RuntimeError(
            f"Experiment 07 must be complete: {len(source_rows)}/{EXPECTED_SOURCE_TOTAL}"
        )
    fallback_rows = [row for row in source_rows if bool(row["fallback_used"])]
    fallback_by_id = {str(row["id"]): row for row in fallback_rows}

    output_dir = (
        RESULTS_ROOT / "08_mmmu_modality_llama_judge" / SOURCE_RUN_TAG /
        spec.name / JUDGE_MODEL_NAME / JUDGE_RUN_TAG
    )
    judge_path = output_dir / "judge_predictions.jsonl"
    final_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU validation modality ablation local judge",
        "source_experiment": "07_mmmu_modality_ablation",
        "source_predictions": str(source_path.relative_to(RESULTS_ROOT)),
        "source_model": spec.name,
        "source_run_tag": SOURCE_RUN_TAG,
        "conditions": list(CONDITIONS),
        "judge_model": JUDGE_MODEL_ID,
        "judge_backend": "transformers",
        "judge_run_tag": JUDGE_RUN_TAG,
        "prompt_version": PROMPT_VERSION,
        "judge_retry_count": 1,
        "random_fallback": True,
        "random_seed": RANDOM_SEED,
        "reuse_baseline_judge_outputs": REUSE_BASELINE_JUDGE_OUTPUTS,
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

    baseline_by_id = {}
    baseline_path = (
        RESULTS_ROOT / "02_mmmu_vllm_judge" / SOURCE_RUN_TAG / spec.name /
        JUDGE_MODEL_NAME / JUDGE_RUN_TAG / "judge_predictions.jsonl"
    )
    if REUSE_BASELINE_JUDGE_OUTPUTS and baseline_path.exists():
        baseline_by_id = {
            str(row["id"]): row for row in read_jsonl(baseline_path)
        }
        print(f"Reusable Experiment 02 judge rows: {len(baseline_by_id)}", flush=True)

    samples = load_sample_map()
    target_total = min(len(fallback_rows), LIMIT) if LIMIT is not None else len(fallback_rows)
    bar = tqdm(
        total=target_total,
        initial=min(len(judge_rows), target_total),
        desc="MMMU modality Llama judge",
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
        sample_id = str(source["sample_id"])
        sample = samples[sample_id]
        options, gold, open_question = evaluation_target(sample)
        prompt = build_judge_prompt(
            str(sample["question"]), options, str(source["response"])
        )

        legacy = baseline_by_id.get(sample_id) if source["condition"] == "full" else None
        reused = legacy is not None
        if reused:
            judge_response = str(legacy.get("judge_raw_output", ""))
            elapsed_seconds = float(legacy.get("elapsed_seconds", 0.0))
        else:
            if judge is None:
                judge = TextJudge(JUDGE_MODEL_ID, MODEL_CACHE / "hub")
            started = time.perf_counter()
            judge_response = judge.generate(
                prompt,
                max_new_tokens=MAX_NEW_TOKENS,
                max_model_length=MAX_MODEL_LENGTH,
            )
            elapsed_seconds = time.perf_counter() - started

        prediction = parse_judge_answer(judge_response, options)
        resolved = bool(prediction)
        final_prediction = prediction
        if not resolved:
            fallback_rng = random.Random(f"{RANDOM_SEED}:{record_id}")
            final_prediction = fallback_rng.choice(
                [chr(65 + index) for index in range(len(options))] + ["Z"]
            )
        row = {
            "id": record_id,
            "sample_id": sample_id,
            "condition": source["condition"],
            "category": source["category"],
            "subject": source["subject"],
            "source_prediction": source["prediction"],
            "source_response": source["response"],
            "judge_prediction": prediction,
            "judge_raw_output": judge_response,
            "judge_resolved": resolved,
            "random_fallback_used": not resolved,
            "reused_baseline_judge_output": reused,
            "final_prediction": final_prediction,
            "gold": gold,
            "correct": final_prediction == gold,
            "open_question_reformatted": open_question,
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
            final["correct"] = bool(judged["correct"])
            final["judge_used"] = True
            final["judge_resolved"] = bool(judged["judge_resolved"])
            final["random_fallback_used"] = bool(judged["random_fallback_used"])
        else:
            final["judge_used"] = False
        final_rows.append(final)
    write_final_predictions(final_path, final_rows)

    summary = summarize(source_rows, final_rows)
    summary.update({
        "benchmark": "MMMU validation modality ablation local judge",
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
        "reused_baseline_judge_outputs": sum(
            bool(row["reused_baseline_judge_output"]) for row in judge_rows
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
