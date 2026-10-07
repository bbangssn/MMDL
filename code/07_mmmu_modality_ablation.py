"""Experiment 07: paired modality ablation on 200 MMMU Validation items."""

import os

# Set before torch/transformers imports. Physical GPU 0 becomes cuda:0.
VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from collections import defaultdict
from datetime import datetime
import hashlib
import json
import random
import re
import time

from PIL import Image
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
    read_jsonl,
    resolve_model,
    write_json,
)


DATA_PATH = os.environ.get("MMMU_DATA_PATH", "MMMU/MMMU")
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"

# EDIT THESE SETTINGS
MODEL = 0
SOURCE_RUN_TAG = "official-penalty-1p5"
SAMPLE_COUNT = 200
CONDITIONS = ("full", "text_only", "shuffled_image", "blank_image")
RUN_CONDITIONS = ("text_only", "shuffled_image", "blank_image")
LOG_EVERY_BATCHES = 1
BATCH_SIZE = 32
MAX_NEW_TOKENS = 2048
MAX_MODEL_LENGTH = 9048
GPU_MEMORY_UTILIZATION = 0.80
MAX_IMAGES_PER_PROMPT = 7
REPETITION_PENALTY = 1.0
PRESENCE_PENALTY = 1.5
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
SEED = 42
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
BOOTSTRAP_REPETITIONS = 5000


def load_all_samples() -> list[dict]:
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
            sample = dict(raw)
            sample["_subject"] = subject
            samples.append(sample)
    return samples


def image_count(sample: dict) -> int:
    return len(get_images(sample))


def select_balanced(samples: list[dict]) -> list[dict]:
    """Round-robin subjects so each contributes six or seven image items."""
    rng = random.Random(SEED)
    image_count_frequency = defaultdict(int)
    for sample in samples:
        image_count_frequency[image_count(sample)] += 1

    groups: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        count = image_count(sample)
        # All selected items need images and a different equal-count donor.
        if count > 0 and image_count_frequency[count] >= 2:
            groups[str(sample["_subject"])].append(sample)
    for group in groups.values():
        rng.shuffle(group)

    selected = []
    subjects = sorted(groups)
    round_index = 0
    while len(selected) < SAMPLE_COUNT:
        added = False
        for subject in subjects:
            if round_index < len(groups[subject]) and len(selected) < SAMPLE_COUNT:
                selected.append(groups[subject][round_index])
                added = True
        if not added:
            break
        round_index += 1
    if len(selected) != SAMPLE_COUNT:
        raise RuntimeError(f"Could select only {len(selected)}/{SAMPLE_COUNT} items")
    return selected


def build_donor_map(selected: list[dict], all_samples: list[dict]) -> dict[str, str]:
    """Assign a deterministic different sample with the same image count."""
    selected_by_count: dict[int, list[str]] = defaultdict(list)
    all_by_count: dict[int, list[str]] = defaultdict(list)
    for sample in selected:
        selected_by_count[image_count(sample)].append(str(sample["id"]))
    for sample in all_samples:
        if image_count(sample) > 0:
            all_by_count[image_count(sample)].append(str(sample["id"]))

    donors = {}
    for count, sample_ids in selected_by_count.items():
        if len(sample_ids) >= 2:
            for index, sample_id in enumerate(sample_ids):
                donors[sample_id] = sample_ids[(index + 1) % len(sample_ids)]
            continue
        sample_id = sample_ids[0]
        candidates = [value for value in all_by_count[count] if value != sample_id]
        if not candidates:
            raise RuntimeError(f"No shuffled donor has {count} image(s)")
        donors[sample_id] = candidates[0]
    return donors


def per_sample_seed(sample_id: str) -> int:
    """Use common random numbers across every altered modality condition."""
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def evaluation_target(sample: dict) -> tuple[list[str], str, bool]:
    options = parse_options(sample["options"])
    answer = str(sample["answer"]).strip()
    open_question = not options
    if open_question:
        return [answer, "Other Answers"], "A", True
    return options, answer.upper(), False


def bootstrap_delta_ci(rows: list[dict], condition: str) -> list[float] | None:
    by_condition = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][str(row["sample_id"])] = bool(row["correct"])
    full = by_condition.get("full", {})
    other = by_condition.get(condition, {})
    sample_ids = sorted(full.keys() & other.keys())
    if len(sample_ids) != SAMPLE_COUNT:
        return None
    differences = [int(other[i]) - int(full[i]) for i in sample_ids]
    rng = random.Random(SEED + 300)
    draws = [
        sum(rng.choice(differences) for _ in sample_ids) / len(sample_ids)
        for _ in range(BOOTSTRAP_REPETITIONS)
    ]
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def group_summary(rows: list[dict]) -> dict:
    correct = sum(bool(row["correct"]) for row in rows)
    nonfallback = [row for row in rows if not bool(row["fallback_used"])]
    nonfallback_correct = sum(bool(row["correct"]) for row in nonfallback)
    return {
        "total": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows) if rows else None,
        "fallbacks": sum(bool(row["fallback_used"]) for row in rows),
        "nonfallback_total": len(nonfallback),
        "accuracy_excluding_fallback": (
            nonfallback_correct / len(nonfallback) if nonfallback else None
        ),
    }


def summarize(rows: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    full_by_id = {
        str(row["sample_id"]): row for row in by_condition.get("full", [])
    }

    conditions = {}
    categories = {}
    subjects = {}
    for condition in CONDITIONS:
        group = by_condition.get(condition, [])
        condition_summary = group_summary(group)
        paired = [row for row in group if str(row["sample_id"]) in full_by_id]
        agreement = sum(
            row["prediction"] == full_by_id[str(row["sample_id"])]["prediction"]
            for row in paired
        )
        full_accuracy = (
            sum(bool(row["correct"]) for row in full_by_id.values()) / len(full_by_id)
            if full_by_id else None
        )
        condition_summary.update({
            "prediction_agreement_with_full": agreement / len(paired) if paired else None,
            "accuracy_delta_vs_full": (
                condition_summary["accuracy"] - full_accuracy
                if condition_summary["accuracy"] is not None and full_accuracy is not None
                else None
            ),
            "accuracy_delta_95pct_bootstrap_ci": (
                bootstrap_delta_ci(rows, condition) if condition != "full" else None
            ),
            "runtime": runtime_summary(group),
        })
        conditions[condition] = condition_summary

        category_groups = defaultdict(list)
        subject_groups = defaultdict(list)
        for row in group:
            category_groups[row["category"]].append(row)
            subject_groups[row["subject"]].append(row)
        categories[condition] = {
            key: group_summary(value) for key, value in sorted(category_groups.items())
        }
        subjects[condition] = {
            key: group_summary(value) for key, value in sorted(subject_groups.items())
        }

    return {
        "sample_count": SAMPLE_COUNT,
        "conditions": conditions,
        "categories": categories,
        "subjects": subjects,
    }


def main() -> None:
    session_started = time.perf_counter()
    if "full" not in CONDITIONS or any(x not in CONDITIONS for x in RUN_CONDITIONS):
        raise ValueError("CONDITIONS/RUN_CONDITIONS are inconsistent")

    spec = resolve_model(str(MODEL))
    output_dir = (
        RESULTS_ROOT / "07_mmmu_modality_ablation" / SOURCE_RUN_TAG / spec.name
    )
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "sample_manifest.json"
    config = {
        "benchmark": "MMMU validation modality ablation",
        "dataset": DATA_PATH,
        "dataset_revision": DATASET_REVISION if DATA_PATH == "MMMU/MMMU" else None,
        "split": "validation",
        "model": spec.name,
        "model_path": spec.path,
        "sample_count": SAMPLE_COUNT,
        "selection": "subject-balanced deterministic round-robin",
        "conditions": list(CONDITIONS),
        "run_conditions": list(RUN_CONDITIONS),
        "source_run_tag": SOURCE_RUN_TAG,
        "full_source": (
            f"00_mmmu_vllm/{SOURCE_RUN_TAG}/{spec.name}/predictions.jsonl"
        ),
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

    all_samples = load_all_samples()
    by_id = {str(sample["id"]): sample for sample in all_samples}
    selected = select_balanced(all_samples)
    donor_map = build_donor_map(selected, all_samples)
    manifest = {
        "selection": config["selection"],
        "seed": SEED,
        "samples": [
            {
                "id": str(sample["id"]),
                "subject": str(sample["_subject"]),
                "category": category_for_subject(str(sample["_subject"])),
                "question_type": "open" if not parse_options(sample["options"]) else "multiple-choice",
                "image_count": image_count(sample),
                "shuffled_donor_id": donor_map[str(sample["id"])],
            }
            for sample in selected
        ],
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Existing sample manifest differs from current selection")
    else:
        write_json(manifest_path, manifest)

    rows, completed_ids = load_completed_predictions(predictions_path)
    full_source = (
        RESULTS_ROOT / "00_mmmu_vllm" / SOURCE_RUN_TAG /
        spec.name / "predictions.jsonl"
    )
    source_rows = {str(row["id"]): row for row in read_jsonl(full_source)}
    if len(source_rows) != 900:
        raise RuntimeError(f"Full MMMU baseline is incomplete: {len(source_rows)}/900")

    # Reuse the exact full-image outputs from Experiment 00.
    for sample in selected:
        sample_id = str(sample["id"])
        record_id = f"full:{sample_id}"
        if record_id in completed_ids:
            continue
        source = source_rows[sample_id]
        _, gold, open_question = evaluation_target(sample)
        row = {
            "id": record_id,
            "sample_id": sample_id,
            "condition": "full",
            "category": category_for_subject(str(sample["_subject"])),
            "subject": str(sample["_subject"]),
            "question_type": "open" if open_question else "multiple-choice",
            "ground_truth": str(sample["answer"]),
            "gold": gold,
            "prediction": source.get("prediction", source.get("parsed_answer", "")),
            "response": source["response"],
            "correct": bool(source["correct"]),
            "fallback_used": bool(source.get("fallback_used", False)),
            "image_count": image_count(sample),
            "image_source_id": sample_id,
            "elapsed_seconds": None,
            "reused_from_full_baseline": True,
        }
        append_jsonl(predictions_path, row)
        rows.append(row)
        completed_ids.add(record_id)

    target_total = SAMPLE_COUNT * len(CONDITIONS)
    bar = tqdm(
        total=target_total,
        initial=len(completed_ids),
        desc="MMMU validation modality ablation",
        unit="condition",
        dynamic_ncols=True,
    )
    runner = None
    generated_this_session = 0
    batches_done = 0
    for condition in RUN_CONDITIONS:
        pending = [
            sample for sample in selected
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
                prompt = qwen_mmmu_prompt(
                    str(sample["question"]), parse_options(sample["options"])
                )
                original_images = get_images(sample)
                image_source_id = sample_id
                if condition == "text_only":
                    prompt = re.sub(r"<image\s*\d*>", "[Image omitted]", prompt)
                    images = []
                    image_source_id = None
                elif condition == "shuffled_image":
                    image_source_id = donor_map[sample_id]
                    images = get_images(by_id[image_source_id])
                elif condition == "blank_image":
                    images = [
                        Image.new("RGB", image.size, "white")
                        for image in original_images
                    ]
                    image_source_id = "blank"
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                requests.append({
                    "prompt": prompt,
                    "images": images,
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
                metadata.append((sample, image_source_id, len(original_images)))

            batch_started = time.perf_counter()
            responses = runner.generate_batch(requests)
            batch_elapsed = time.perf_counter() - batch_started
            for response, (sample, image_source_id, original_image_count) in zip(
                responses, metadata
            ):
                sample_id = str(sample["id"])
                record_id = f"{condition}:{sample_id}"
                options, gold, open_question = evaluation_target(sample)
                prediction = qwen_rule_extract(response, options)
                fallback = prediction in ("", "Z")
                row = {
                    "id": record_id,
                    "sample_id": sample_id,
                    "condition": condition,
                    "category": category_for_subject(str(sample["_subject"])),
                    "subject": str(sample["_subject"]),
                    "question_type": "open" if open_question else "multiple-choice",
                    "ground_truth": str(sample["answer"]),
                    "gold": gold,
                    "prediction": prediction,
                    "response": response,
                    "correct": prediction == gold,
                    "fallback_used": fallback,
                    "image_count": original_image_count,
                    "image_source_id": image_source_id,
                    "elapsed_seconds": round(batch_elapsed / len(batch), 3),
                    "runtime_measurement": "batch_wall_time_divided_by_batch_size",
                    "batch_elapsed_seconds": round(batch_elapsed, 3),
                    "reused_from_full_baseline": False,
                }
                append_jsonl(predictions_path, row)
                rows.append(row)
                completed_ids.add(record_id)
                generated_this_session += 1
                bar.update(1)

            batches_done += 1
            current = summarize(rows)
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
    summary = summarize(rows)
    summary.update({
        "benchmark": "MMMU validation modality ablation",
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
