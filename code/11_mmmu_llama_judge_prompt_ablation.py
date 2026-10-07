"""Experiment 11: prompt/length ablation for the local MMMU Llama judge."""

import os

VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU

from collections import defaultdict
from datetime import datetime
import json
import random
import re
import time

from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment
from utils import (
    DATASET_CACHE,
    MMMU_SUBJECTS,
    MODEL_CACHE,
    RESULTS_ROOT,
    TextJudge,
    append_jsonl,
    ensure_run_config,
    load_completed_predictions,
    parse_options,
    qwen_rule_extract,
    read_jsonl,
    resolve_model,
    write_json,
)


DATA_PATH = os.environ.get("MMMU_DATA_PATH", "MMMU/MMMU")

# EDIT THESE SETTINGS
SOURCE_MODEL = 0
SOURCE_RUN_TAG = "official-penalty-1p5"
JUDGE_MODEL_NAME = "llama-3.1-8b-instruct"
JUDGE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
RUN_TAG = "strict-output-v1"
VARIANTS = {
    "strict_128": 128,
    "strict_8": 8,
}
MAX_MODEL_LENGTH = 9048
RANDOM_SEED = 42
LIMIT = None  # None evaluates all 596 source fallbacks.
LOG_EVERY = 10


def strict_prompt(question: str, options: list[str], answer: str) -> str:
    option_text = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    valid = ", ".join(
        [chr(65 + index) for index in range(len(options))] + ["Z"]
    )
    return (
        "You are a classification function. Match the proposed answer to the "
        "closest option. Return Z only if none of the options match.\n\n"
        "OUTPUT RULES:\n"
        "- Return exactly one uppercase letter.\n"
        "- Do not explain your reasoning.\n"
        "- Do not repeat the question, options, or answer.\n"
        "- Do not add punctuation, formatting, or any other words.\n"
        f"- The only valid outputs are: {valid}\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_text}\n"
        f"Proposed answer: {answer}\n\n"
        "Output:"
    )


def robust_parse(response: str, options: list[str]) -> tuple[str, str]:
    valid = {chr(65 + index) for index in range(len(options))}
    valid_with_z = valid | {"Z"}
    stripped = response.strip().upper()
    if stripped in valid_with_z:
        return ("" if stripped == "Z" else stripped), "exact"

    matches = re.findall(
        r"(?:\bMY\s+|\bYOUR\s+)?\bOUTPUT\s*:\s*[*_`]*\(?([A-Z])\)?",
        response.upper(),
    )
    if matches and matches[-1] in valid_with_z:
        return ("" if matches[-1] == "Z" else matches[-1]), "explicit_output"

    single_lines = [
        line.strip().strip("*_`()[]{}.: ").upper()
        for line in response.splitlines()
    ]
    single_lines = [value for value in single_lines if value in valid_with_z]
    if single_lines:
        value = single_lines[-1]
        return ("" if value == "Z" else value), "last_single_letter_line"

    prediction = qwen_rule_extract(response, options)
    return ("" if prediction == "Z" else prediction), "qwen_rule_extract"


def fallback_label(options: list[str], sample_id: str) -> str:
    rng = random.Random(f"{RANDOM_SEED}:{sample_id}")
    return rng.choice(
        [chr(65 + index) for index in range(len(options))] + ["Z"]
    )


def summarize_condition(rows: list[dict], gpt_by_id: dict[str, dict]) -> dict:
    resolved = [row for row in rows if row["judge_resolved"]]
    both_resolved = [
        row for row in resolved
        if gpt_by_id.get(str(row["sample_id"]), {}).get("judge_resolved")
    ]
    return {
        "total": len(rows),
        "judge_resolved": len(resolved),
        "resolution_rate": len(resolved) / len(rows) if rows else None,
        "random_fallbacks": sum(row["random_fallback_used"] for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None,
        "gpt_final_agreement": (
            sum(
                row["final_prediction"]
                == gpt_by_id[str(row["sample_id"])]["final_prediction"]
                for row in rows
            ) / len(rows) if rows else None
        ),
        "both_resolved_count": len(both_resolved),
        "gpt_agreement_when_both_resolved": (
            sum(
                row["judge_prediction"]
                == gpt_by_id[str(row["sample_id"])]["judge_prediction"]
                for row in both_resolved
            ) / len(both_resolved) if both_resolved else None
        ),
        "mean_output_characters": (
            sum(len(row["judge_raw_output"]) for row in rows) / len(rows)
            if rows else None
        ),
        "mean_output_words": (
            sum(len(row["judge_raw_output"].split()) for row in rows) / len(rows)
            if rows else None
        ),
        "mean_elapsed_seconds": (
            sum(float(row.get("elapsed_seconds", 0.0)) for row in rows) / len(rows)
            if rows else None
        ),
    }


def reference_rows(
    source_judge_rows: list[dict],
    cases: dict[str, dict],
    robust: bool,
) -> list[dict]:
    rows = []
    for source in source_judge_rows:
        sample_id = str(source["id"])
        case = cases[sample_id]
        if robust:
            prediction, parser_method = robust_parse(
                str(source["judge_raw_output"]), case["options"]
            )
        else:
            prediction = str(source.get("judge_prediction", ""))
            parser_method = "legacy_official_style"
        resolved = bool(prediction)
        final_prediction = prediction or fallback_label(case["options"], sample_id)
        rows.append({
            "sample_id": sample_id,
            "judge_prediction": prediction,
            "judge_raw_output": str(source["judge_raw_output"]),
            "judge_resolved": resolved,
            "random_fallback_used": not resolved,
            "final_prediction": final_prediction,
            "gold": case["gold"],
            "correct": final_prediction == case["gold"],
            "parser_method": parser_method,
            "elapsed_seconds": float(source.get("elapsed_seconds", 0.0)),
        })
    return rows


def main() -> None:
    started = time.perf_counter()
    source_spec = resolve_model(str(SOURCE_MODEL))
    source_dir = (
        RESULTS_ROOT / "02_mmmu_vllm_judge" / SOURCE_RUN_TAG /
        source_spec.name / JUDGE_MODEL_NAME / "greedy-random-fallback"
    )
    source_judge_path = source_dir / "judge_predictions.jsonl"
    source_judge_rows = read_jsonl(source_judge_path)
    if len(source_judge_rows) != 596:
        raise RuntimeError(f"Expected 596 source fallbacks, got {len(source_judge_rows)}")
    source_by_id = {str(row["id"]): row for row in source_judge_rows}

    gpt_path = (
        RESULTS_ROOT / "06_mmmu_openai_judge" / SOURCE_RUN_TAG /
        source_spec.name / "gpt-3.5-turbo-0125" /
        "official-prompt-25-retries" / "judge_predictions.jsonl"
    )
    gpt_by_id = {str(row["id"]): row for row in read_jsonl(gpt_path)}

    cases = {}
    for subject in MMMU_SUBJECTS:
        dataset = load_dataset(
            DATA_PATH, subject, split="validation", cache_dir=DATASET_CACHE
        )
        for sample in dataset:
            sample_id = str(sample["id"])
            if sample_id not in source_by_id:
                continue
            options = parse_options(sample["options"])
            if options:
                gold = str(sample["answer"]).strip().upper()
            else:
                options = [str(sample["answer"]).strip(), "Other Answers"]
                gold = "A"
            cases[sample_id] = {
                "sample_id": sample_id,
                "question": str(sample["question"]),
                "options": options,
                "gold": gold,
                "source_response": str(source_by_id[sample_id]["source_response"]),
            }
    if len(cases) != len(source_judge_rows):
        raise RuntimeError(f"Loaded {len(cases)}/{len(source_judge_rows)} cases")

    selected_ids = list(cases)[:LIMIT] if LIMIT is not None else list(cases)
    output_dir = (
        RESULTS_ROOT / "11_mmmu_llama_judge_prompt_ablation" /
        SOURCE_RUN_TAG / source_spec.name / JUDGE_MODEL_NAME / RUN_TAG
    )
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU Llama judge prompt ablation",
        "source_run_tag": SOURCE_RUN_TAG,
        "source_model": source_spec.name,
        "source_fallbacks": len(source_judge_rows),
        "selected_samples": len(selected_ids),
        "judge_model": JUDGE_MODEL_ID,
        "variants": VARIANTS,
        "max_model_length": MAX_MODEL_LENGTH,
        "random_seed": RANDOM_SEED,
        "limit": LIMIT,
        "visible_gpu": VISIBLE_GPU,
    }
    ensure_run_config(output_dir / "config.json", predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    rows, completed_ids = load_completed_predictions(predictions_path)
    expected_ids = {
        f"{variant}:{sample_id}"
        for variant in VARIANTS for sample_id in selected_ids
    }
    unknown = completed_ids - expected_ids
    if unknown:
        raise ValueError(f"Unknown completed IDs: {sorted(unknown)[:5]}")

    total = len(selected_ids) * len(VARIANTS)
    bar = tqdm(
        total=total, initial=len(completed_ids),
        desc="Llama judge prompt ablation", unit="judgment", dynamic_ncols=True,
    )
    judge = None
    for variant, max_new_tokens in VARIANTS.items():
        for sample_id in selected_ids:
            record_id = f"{variant}:{sample_id}"
            if record_id in completed_ids:
                continue
            case = cases[sample_id]
            if judge is None:
                judge = TextJudge(JUDGE_MODEL_ID, MODEL_CACHE / "hub")
            prompt = strict_prompt(
                case["question"], case["options"], case["source_response"]
            )
            sample_started = time.perf_counter()
            response = judge.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                max_model_length=MAX_MODEL_LENGTH,
            )
            elapsed = time.perf_counter() - sample_started
            prediction, parser_method = robust_parse(response, case["options"])
            resolved = bool(prediction)
            final_prediction = prediction or fallback_label(case["options"], sample_id)
            row = {
                "id": record_id,
                "sample_id": sample_id,
                "variant": variant,
                "max_new_tokens": max_new_tokens,
                "judge_prediction": prediction,
                "judge_raw_output": response,
                "parser_method": parser_method,
                "judge_resolved": resolved,
                "random_fallback_used": not resolved,
                "final_prediction": final_prediction,
                "gold": case["gold"],
                "correct": final_prediction == case["gold"],
                "gpt_final_prediction": gpt_by_id[sample_id]["final_prediction"],
                "gpt_final_agreement": (
                    final_prediction == gpt_by_id[sample_id]["final_prediction"]
                ),
                "elapsed_seconds": round(elapsed, 3),
            }
            append_jsonl(predictions_path, row)
            rows.append(row)
            completed_ids.add(record_id)
            bar.update(1)
            bar.set_postfix(variant=variant, resolved=resolved)
            if len(rows) % LOG_EVERY == 0:
                write_json(progress_path, {
                    "status": "running",
                    "completed": len(rows),
                    "target": total,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })
    bar.close()

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    selected_source_rows = [
        row for row in source_judge_rows if str(row["id"]) in selected_ids
    ]
    references = {
        "official_style_existing": reference_rows(
            selected_source_rows, cases, robust=False
        ),
        "existing_output_robust_parser": reference_rows(
            selected_source_rows, cases, robust=True
        ),
    }
    summaries = {
        name: summarize_condition(values, gpt_by_id)
        for name, values in {**references, **grouped}.items()
    }
    summary = {
        "benchmark": "MMMU Llama judge prompt ablation",
        "status": "completed" if len(completed_ids) == total else "partial",
        "sample_count": len(selected_ids),
        "conditions": summaries,
        "decision_metrics": [
            "resolution_rate", "accuracy", "gpt_final_agreement",
            "gpt_agreement_when_both_resolved", "mean_output_characters",
            "mean_elapsed_seconds",
        ],
        "session_elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(progress_path, {
        **summary, "updated_at": datetime.now().isoformat(timespec="seconds")
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
