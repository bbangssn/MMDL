"""Reproducible paired analysis for experiments 04 and 05."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from utils import RESULTS_ROOT, resolve_model, write_json


# EDIT THESE SETTINGS
MODEL = 0
BOOTSTRAP_REPETITIONS = 20_000
BOOTSTRAP_SEED = 43


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * q)]


def paired_stats(base: list[dict], other: list[dict]) -> dict:
    base_map = {str(row["sample_id"]): row for row in base}
    pairs = [(base_map[str(row["sample_id"])], row) for row in other]
    deltas = [int(b["correct"]) - int(a["correct"]) for a, b in pairs]
    rng = random.Random(BOOTSTRAP_SEED)
    boot = [
        sum(deltas[rng.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]
    usable = [(a, b) for a, b in pairs if not a["fallback_used"] and not b["fallback_used"]]
    return {
        "n": len(pairs),
        "delta_accuracy": sum(deltas) / len(deltas),
        "paired_bootstrap_95_ci": [quantile(boot, 0.025), quantile(boot, 0.975)],
        "direct_wrong_to_condition_correct": sum(not a["correct"] and b["correct"] for a, b in pairs),
        "direct_correct_to_condition_wrong": sum(a["correct"] and not b["correct"] for a, b in pairs),
        "both_nonfallback_n": len(usable),
        "both_nonfallback_delta_accuracy": (
            sum(int(b["correct"]) - int(a["correct"]) for a, b in usable) / len(usable)
            if usable else None
        ),
    }


def condition_summary(rows: list[dict]) -> dict:
    usable = [row for row in rows if not row["fallback_used"]]
    return {
        "n": len(rows), "correct": sum(row["correct"] for row in rows),
        "accuracy": sum(row["correct"] for row in rows) / len(rows),
        "fallbacks": sum(row["fallback_used"] for row in rows),
        "nonfallback_n": len(usable),
        "accuracy_excluding_fallback": (
            sum(row["correct"] for row in usable) / len(usable) if usable else None
        ),
        "mean_elapsed_seconds": sum(row["elapsed_seconds"] for row in rows) / len(rows),
    }


def main() -> None:
    model = resolve_model(str(MODEL)).name
    root = RESULTS_ROOT / "04_bottleneck_probes" / model
    rows = read_jsonl(root / "predictions.jsonl")
    control = read_jsonl(
        RESULTS_ROOT / "05_conflict_control" / model / "predictions.jsonl"
    )
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)
    grouped["shuffled_no_instruction"] = control
    direct = grouped["direct"]

    summaries = {name: condition_summary(group) for name, group in grouped.items()}
    paired = {
        name: paired_stats(direct, group)
        for name, group in grouped.items() if name != "direct"
    }

    direct_by_id = {str(row["sample_id"]): row for row in direct}
    permuted = grouped["option_permuted"]
    perm_usable = [
        row for row in permuted
        if not row["fallback_used"] and not direct_by_id[str(row["sample_id"])]["fallback_used"]
    ]
    option_order = {
        "both_nonfallback_n": len(perm_usable),
        "content_consistent_with_direct": sum(
            row["canonical_prediction"] == direct_by_id[str(row["sample_id"])]["canonical_prediction"]
            for row in perm_usable
        ),
        "raw_letter_sticky": sum(
            row["prediction"] == direct_by_id[str(row["sample_id"])]["prediction"]
            for row in perm_usable
        ),
    }

    seed_groups = [grouped[f"sample_seed_{index}"] for index in range(3)]
    seeds_by_id = [
        {str(row["sample_id"]): row for row in group} for group in seed_groups
    ]
    stable = []
    majority_correct = []
    for sample_id in direct_by_id:
        candidates = [group[sample_id]["canonical_prediction"] for group in seeds_by_id]
        stable.append(len(set(candidates)) == 1)
        majority = Counter(candidates).most_common(1)[0][0]
        majority_correct.append(majority == direct_by_id[sample_id]["ground_truth"])
    sampling = {
        "all_three_predictions_same": sum(stable) / len(stable),
        "items_with_seed_variation": len(stable) - sum(stable),
        "majority_vote_accuracy": sum(majority_correct) / len(majority_correct),
        "per_seed_accuracy": [summaries[f"sample_seed_{i}"]["accuracy"] for i in range(3)],
    }

    leakage = {}
    for condition in ("describe_then_solve", "knowledge_notes_then_solve"):
        group = grouped[condition]
        hits = [
            row for row in group
            if re.search(r"(?i)(?:answer|final answer)\s*(?:is|:)?\s*[A-J]\b", row.get("intermediate") or "")
        ]
        leakage[condition] = {"n": len(group), "explicit_answer_like_intermediates": len(hits)}

    category = {}
    for name, group in grouped.items():
        bins = defaultdict(list)
        for row in group:
            bins[row["category"]].append(row)
        category[name] = {
            key: sum(row["correct"] for row in values) / len(values)
            for key, values in sorted(bins.items())
        }

    result = {
        "model": model, "sample_count": len(direct),
        "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
        "conditions": summaries, "paired_vs_direct": paired,
        "option_order": option_order, "sampling": sampling,
        "intermediate_answer_leakage": leakage,
        "accuracy_by_category": category,
        "interpretation_guardrails": [
            "This is a directional 24-item probe, not a benchmark score.",
            "Items were selected for one image and an official explanation.",
            "Fallback-excluded accuracy is reported because random fallback can obscure prompting effects.",
            "Oracle rationale can reveal answer semantics and is only an upper-bound diagnostic.",
        ],
    }
    write_json(root / "analysis.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
