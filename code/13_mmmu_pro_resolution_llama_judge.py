"""Experiment 13: strict Llama judge and paired analysis for Experiment 12."""

import os

VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU

from collections import defaultdict
from datetime import datetime
import json
import random
import time

from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment
from utils import (
    DATASET_CACHE, MODEL_CACHE, RESULTS_ROOT, TextJudge, append_jsonl,
    ensure_run_config, load_completed_predictions, parse_options,
    strict_judge_extract, strict_judge_prompt, write_json,
)


# EDIT THESE SETTINGS
SOURCE_DIR = RESULTS_ROOT / "12_mmmu_pro_resolution_scale" / "qwen3-vl-4b-instruct"
DATA_PATH = os.environ.get("MMMU_PRO_DATA_PATH", "MMMU/MMMU_Pro")
DATASET_CONFIG = "standard (10 options)"
JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
JUDGE_RUN_TAG = "strict-output-v1-random-fallback"
MAX_NEW_TOKENS = 8
MAX_MODEL_LENGTH = 9048
RANDOM_SEED = 42
BOOTSTRAP_REPETITIONS = 10_000
LOG_EVERY = 5


def paired_summary(rows: list[dict]) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        groups[row["condition"]][str(row["sample_id"])] = row
    direct = groups["direct"]
    result = {}
    for condition in ("direct", "low_resolution", "high_resolution"):
        group = groups[condition]
        ids = sorted(set(direct) & set(group))
        accuracy = sum(bool(group[i]["correct"]) for i in ids) / len(ids)
        delta = [int(group[i]["correct"]) - int(direct[i]["correct"]) for i in ids]
        rng = random.Random(f"{RANDOM_SEED}:{condition}:bootstrap")
        draws = []
        if condition != "direct":
            for _ in range(BOOTSTRAP_REPETITIONS):
                draws.append(sum(delta[rng.randrange(len(delta))] for _ in ids) / len(ids))
            draws.sort()
        result[condition] = {
            "total": len(ids),
            "correct": sum(bool(group[i]["correct"]) for i in ids),
            "accuracy": accuracy,
            "delta_vs_direct": sum(delta) / len(delta),
            "paired_bootstrap_95_ci": (
                [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]
                if draws else [0.0, 0.0]
            ),
            "direct_wrong_to_condition_correct": sum(
                not direct[i]["correct"] and group[i]["correct"] for i in ids
            ),
            "direct_correct_to_condition_wrong": sum(
                direct[i]["correct"] and not group[i]["correct"] for i in ids
            ),
            "agreement_with_direct": sum(
                direct[i]["prediction"] == group[i]["prediction"] for i in ids
            ) / len(ids),
            "random_fallbacks": sum(
                bool(group[i].get("random_fallback_used")) for i in ids
            ),
            "categories": {
                category: {
                    "total": len(category_rows),
                    "correct": sum(bool(row["correct"]) for row in category_rows),
                    "accuracy": sum(bool(row["correct"]) for row in category_rows)
                    / len(category_rows),
                }
                for category, category_rows in sorted(
                    (
                        category,
                        [row for row in group.values() if row["category"] == category],
                    )
                    for category in {row["category"] for row in group.values()}
                )
            },
        }
    return result


def compare_conditions(rows: list[dict], left: str, right: str) -> dict:
    groups = defaultdict(dict)
    for row in rows:
        groups[row["condition"]][str(row["sample_id"])] = row
    ids = sorted(set(groups[left]) & set(groups[right]))
    deltas = [
        int(groups[right][i]["correct"]) - int(groups[left][i]["correct"])
        for i in ids
    ]
    rng = random.Random(f"{RANDOM_SEED}:{left}:{right}:bootstrap")
    draws = sorted(
        sum(deltas[rng.randrange(len(deltas))] for _ in ids) / len(ids)
        for _ in range(BOOTSTRAP_REPETITIONS)
    )
    return {
        "left": left,
        "right": right,
        "n": len(ids),
        "right_minus_left_accuracy": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [
            draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]
        ],
        "left_wrong_to_right_correct": sum(
            not groups[left][i]["correct"] and groups[right][i]["correct"]
            for i in ids
        ),
        "left_correct_to_right_wrong": sum(
            groups[left][i]["correct"] and not groups[right][i]["correct"]
            for i in ids
        ),
    }


def main() -> None:
    started = time.perf_counter()
    source_path = SOURCE_DIR / "predictions.jsonl"
    source_rows, _ = load_completed_predictions(source_path)
    if len(source_rows) != 360:
        raise RuntimeError(f"Experiment 12 must be complete: {len(source_rows)}/360")
    fallbacks = [row for row in source_rows if bool(row.get("fallback_used"))]
    fallback_by_id = {str(row["id"]): row for row in fallbacks}

    output_dir = RESULTS_ROOT / "13_mmmu_pro_resolution_llama_judge" / JUDGE_RUN_TAG
    judge_path = output_dir / "judge_predictions.jsonl"
    final_path = output_dir / "predictions.jsonl"
    config = {
        "benchmark": "MMMU-Pro scaled resolution strict Llama judge",
        "source_predictions": str(source_path.relative_to(RESULTS_ROOT)),
        "fallback_samples": len(fallbacks),
        "judge_model": JUDGE_MODEL_ID,
        "prompt_version": "llama-strict-output-v1",
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH,
        "random_seed": RANDOM_SEED,
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
    }
    ensure_run_config(output_dir / "config.json", judge_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    dataset = load_dataset(
        DATA_PATH, DATASET_CONFIG, split="test", cache_dir=DATASET_CACHE
    )
    samples = {str(sample["id"]): sample for sample in dataset}
    judge_rows, completed = load_completed_predictions(judge_path)
    judge = None
    bar = tqdm(
        total=len(fallbacks), initial=len(judge_rows),
        desc="Resolution strict Llama judge", unit="fallback", dynamic_ncols=True,
    )
    for source in fallbacks:
        record_id = str(source["id"])
        if record_id in completed:
            continue
        sample = samples[str(source["sample_id"])]
        options = parse_options(sample["options"])
        prompt = strict_judge_prompt(
            str(sample["question"]), options, str(source["response"])
        )
        if judge is None:
            judge = TextJudge(JUDGE_MODEL_ID, MODEL_CACHE / "hub")
        item_started = time.perf_counter()
        raw = judge.generate(
            prompt, max_new_tokens=MAX_NEW_TOKENS,
            max_model_length=MAX_MODEL_LENGTH,
        )
        elapsed = time.perf_counter() - item_started
        prediction = strict_judge_extract(raw, options)[0]
        resolved = bool(prediction)
        if not resolved:
            prediction = random.Random(f"{RANDOM_SEED}:{record_id}").choice(
                [chr(65 + i) for i in range(len(options))] + ["Z"]
            )
        row = {
            "id": record_id,
            "sample_id": source["sample_id"],
            "condition": source["condition"],
            "judge_prediction": prediction,
            "judge_raw_output": raw,
            "judge_resolved": resolved,
            "random_fallback_used": not resolved,
            "gold": source["ground_truth"],
            "correct": prediction == source["ground_truth"],
            "elapsed_seconds": round(elapsed, 3),
        }
        append_jsonl(judge_path, row)
        judge_rows.append(row)
        completed.add(record_id)
        bar.update(1)
        bar.set_postfix(resolved=sum(x["judge_resolved"] for x in judge_rows))
        if len(judge_rows) % LOG_EVERY == 0:
            write_json(output_dir / "progress.json", {
                "status": "running", "judged": len(judge_rows),
                "target": len(fallbacks),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
    bar.close()

    judged = {str(row["id"]): row for row in judge_rows}
    final_rows = []
    for source in source_rows:
        final = dict(source)
        item = judged.get(str(source["id"]))
        if item:
            final["prediction"] = item["judge_prediction"]
            final["canonical_prediction"] = item["judge_prediction"]
            final["correct"] = item["correct"]
            final["judge_used"] = True
            final["random_fallback_used"] = item["random_fallback_used"]
        else:
            final["judge_used"] = False
            final["random_fallback_used"] = False
        final_rows.append(final)

    # This derived file is rebuilt after every judge item is complete.
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in final_rows),
        encoding="utf-8",
    )
    summary = {
        "status": "completed",
        "source_samples": len(source_rows),
        "fallback_samples": len(fallbacks),
        "judge_resolved": sum(row["judge_resolved"] for row in judge_rows),
        "random_fallbacks": sum(row["random_fallback_used"] for row in judge_rows),
        "conditions": paired_summary(final_rows),
        "low_vs_high": compare_conditions(
            final_rows, "low_resolution", "high_resolution"
        ),
        "session_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "progress.json", {
        **summary, "updated_at": datetime.now().isoformat(timespec="seconds")
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
