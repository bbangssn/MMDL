"""Experiment 03: paired modality ablation on 200 MMMU-Pro standard-10 items."""

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
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment, runtime_summary
from utils import (
    DATASET_CACHE,
    RESULTS_ROOT,
    VLLM,
    append_jsonl,
    category_for_subject,
    ensure_run_config,
    get_images_in_token_order,
    load_completed_predictions,
    official_mmmu_choice_extract,
    parse_options,
    read_jsonl,
    resolve_model,
    write_json,
)


DATA_PATH = os.environ.get("MMMU_PRO_DATA_PATH", "MMMU/MMMU_Pro")
DATASET_CONFIG = "standard (10 options)"

# EDIT THESE SETTINGS
MODEL = 0
SAMPLE_COUNT = 200
CONDITIONS = ("full", "text_only", "shuffled_image", "blank_image")
RUN_CONDITIONS = ("text_only", "shuffled_image", "blank_image")
LOG_EVERY = 10
BATCH_SIZE = 32
MAX_NEW_TOKENS = 2048
MAX_MODEL_LENGTH = 9048
GPU_MEMORY_UTILIZATION = 0.80
MAX_IMAGES_PER_PROMPT = 40
MULTI_IMAGE_RESIZE_THRESHOLD = 7
MULTI_IMAGE_MAX_PIXELS = 160 * 28 * 28
PRESENCE_PENALTY = 1.5
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
SEED = 42
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28


def make_prompt(sample: dict) -> str:
    options = parse_options(sample["options"])
    choices = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        f"{sample['question']}\n{choices}\n"
        "Answer with the option letter from the given choices directly."
    )


def select_balanced(samples: list[dict]) -> list[dict]:
    """Round-robin subjects so each contributes six or seven items."""
    rng = random.Random(SEED)
    groups: dict[str, list[dict]] = defaultdict(list)
    image_count_frequency = defaultdict(int)
    for sample in samples:
        image_count_frequency[image_count(sample)] += 1
    for sample in samples:
        count = image_count(sample)
        # A shuffled control needs another item with the same image count.
        if count and image_count_frequency[count] >= 2:
            groups[str(sample["subject"])].append(sample)
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


def image_count(sample: dict) -> int:
    return len(get_images_in_token_order(sample, make_prompt(sample)))


def build_donor_map(selected: list[dict], all_samples: list[dict]) -> dict[str, str]:
    """Deterministic derangement within equal-image-count groups."""
    groups: dict[int, list[str]] = defaultdict(list)
    for sample in selected:
        groups[image_count(sample)].append(str(sample["id"]))
    all_by_count: dict[int, list[str]] = defaultdict(list)
    for sample in all_samples:
        all_by_count[image_count(sample)].append(str(sample["id"]))
    donors = {}
    for count, ids in groups.items():
        if len(ids) >= 2:
            for index, sample_id in enumerate(ids):
                donors[sample_id] = ids[(index + 1) % len(ids)]
            continue
        sample_id = ids[0]
        candidates = [value for value in all_by_count[count] if value != sample_id]
        if not candidates:
            raise RuntimeError(f"No shuffled donor has {count} image(s)")
        donors[sample_id] = candidates[0]
    return donors


def per_sample_seed(sample_id: str) -> int:
    digest = hashlib.sha256(f"{SEED}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def bootstrap_delta_ci(rows: list[dict], condition: str) -> list[float] | None:
    by_condition = defaultdict(dict)
    for row in rows:
        by_condition[row["condition"]][str(row["sample_id"])] = bool(row["correct"])
    full = by_condition.get("full", {})
    other = by_condition.get(condition, {})
    ids = sorted(full.keys() & other.keys())
    if len(ids) != SAMPLE_COUNT:
        return None
    differences = [int(other[i]) - int(full[i]) for i in ids]
    rng = random.Random(SEED + 300)
    draws = []
    for _ in range(5000):
        draws.append(sum(rng.choice(differences) for _ in ids) / len(ids))
    draws.sort()
    return [draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws))]]


def summarize(rows: list[dict]) -> dict:
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    full_by_id = {
        str(row["sample_id"]): row for row in by_condition.get("full", [])
    }
    conditions = {}
    for condition in CONDITIONS:
        group = by_condition.get(condition, [])
        correct = sum(bool(row["correct"]) for row in group)
        paired = [
            row for row in group if str(row["sample_id"]) in full_by_id
        ]
        agreement = sum(
            row["prediction"] == full_by_id[str(row["sample_id"])]["prediction"]
            for row in paired
        )
        conditions[condition] = {
            "total": len(group),
            "correct": correct,
            "accuracy": correct / len(group) if group else None,
            "fallbacks": sum(bool(row["fallback_used"]) for row in group),
            "prediction_agreement_with_full": (
                agreement / len(paired) if paired else None
            ),
            "accuracy_delta_vs_full": (
                correct / len(group)
                - sum(bool(row["correct"]) for row in full_by_id.values())
                / len(full_by_id)
                if group and full_by_id else None
            ),
            "accuracy_delta_95pct_bootstrap_ci": (
                bootstrap_delta_ci(rows, condition) if condition != "full" else None
            ),
            "runtime": runtime_summary(group),
        }
    return {"sample_count": SAMPLE_COUNT, "conditions": conditions}


def main() -> None:
    started_session = time.perf_counter()
    if "full" not in CONDITIONS or any(x not in CONDITIONS for x in RUN_CONDITIONS):
        raise ValueError("CONDITIONS/RUN_CONDITIONS are inconsistent")

    spec = resolve_model(str(MODEL))
    output_dir = RESULTS_ROOT / "03_modality_ablation" / spec.name
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    manifest_path = output_dir / "sample_manifest.json"
    config = {
        "benchmark": "MMMU-Pro standard10 modality ablation",
        "dataset": DATA_PATH,
        "dataset_config": DATASET_CONFIG,
        "split": "test",
        "model": spec.name,
        "model_path": spec.path,
        "sample_count": SAMPLE_COUNT,
        "conditions": list(CONDITIONS),
        "run_conditions": list(RUN_CONDITIONS),
        "full_source": "01_mmmu_pro/standard10/<model>/predictions.jsonl",
        "backend": "vllm",
        "batch_size": BATCH_SIZE,
        "max_images_per_prompt": MAX_IMAGES_PER_PROMPT,
        "multi_image_resize_threshold": MULTI_IMAGE_RESIZE_THRESHOLD,
        "multi_image_max_pixels": MULTI_IMAGE_MAX_PIXELS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
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

    dataset = load_dataset(
        DATA_PATH, DATASET_CONFIG, split="test", cache_dir=DATASET_CACHE
    )
    all_samples = [dict(sample) for sample in dataset]
    by_id = {str(sample["id"]): sample for sample in all_samples}
    selected = select_balanced(all_samples)
    donor_map = build_donor_map(selected, all_samples)
    manifest = {
        "selection": "subject-balanced deterministic round-robin",
        "seed": SEED,
        "samples": [
            {
                "id": str(sample["id"]),
                "subject": str(sample["subject"]),
                "category": category_for_subject(str(sample["subject"])),
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
        RESULTS_ROOT / "01_mmmu_pro" / "standard10" / spec.name
        / "predictions.jsonl"
    )
    source_rows = {str(row["id"]): row for row in read_jsonl(full_source)}
    if len(source_rows) < len(dataset):
        raise RuntimeError(f"Full baseline is incomplete: {len(source_rows)}/{len(dataset)}")

    # Import the already-computed full condition once.
    for sample in selected:
        sample_id = str(sample["id"])
        record_id = f"full:{sample_id}"
        if record_id in completed_ids:
            continue
        source = source_rows[sample_id]
        row = {
            "id": record_id,
            "sample_id": sample_id,
            "condition": "full",
            "category": category_for_subject(str(sample["subject"])),
            "subject": str(sample["subject"]),
            "ground_truth": str(sample["answer"]).strip().upper(),
            "prediction": source["prediction"],
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
        desc="MMMU-Pro modality ablation",
        unit="condition",
        dynamic_ncols=True,
    )
    runner = None
    generated_this_session = 0
    for condition in RUN_CONDITIONS:
        parser_rng = random.Random(SEED + CONDITIONS.index(condition))
        pending = [
            sample for sample in selected
            if f"{condition}:{sample['id']}" not in completed_ids
        ]
        if pending and runner is None:
            runner = VLLM(
                spec, MAX_MODEL_LENGTH, GPU_MEMORY_UTILIZATION,
                MAX_IMAGES_PER_PROMPT, SEED,
            )
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            requests = []
            metadata = []
            for sample in batch:
                sample_id = str(sample["id"])
                record_id = f"{condition}:{sample_id}"
                prompt = make_prompt(sample)
                original_images = get_images_in_token_order(sample, prompt)
                image_source_id = sample_id
                if condition == "text_only":
                    prompt = re.sub(r"<image\s+\d+>", "[Image omitted]", prompt)
                    images = []
                    image_source_id = None
                elif condition == "shuffled_image":
                    image_source_id = donor_map[sample_id]
                    donor = by_id[image_source_id]
                    images = get_images_in_token_order(donor, make_prompt(donor))
                elif condition == "blank_image":
                    images = [
                        Image.new("RGB", image.size, "white")
                        for image in original_images
                    ]
                    image_source_id = "blank"
                else:
                    raise ValueError(f"Unknown condition: {condition}")

                sample_seed = per_sample_seed(sample_id)
                adaptive_resize = len(images) > MULTI_IMAGE_RESIZE_THRESHOLD
                effective_min_pixels = (
                    MULTI_IMAGE_MAX_PIXELS if adaptive_resize else MIN_PIXELS
                )
                effective_max_pixels = (
                    MULTI_IMAGE_MAX_PIXELS if adaptive_resize else MAX_PIXELS
                )
                requests.append({
                    "prompt": prompt,
                    "images": images,
                    "max_new_tokens": MAX_NEW_TOKENS,
                    "generation_kwargs": {
                        "do_sample": True,
                        "temperature": TEMPERATURE,
                        "top_p": TOP_P,
                        "top_k": TOP_K,
                        "repetition_penalty": 1.0,
                        "presence_penalty": PRESENCE_PENALTY,
                        "seed": sample_seed,
                    },
                    "min_pixels": effective_min_pixels,
                    "max_pixels": effective_max_pixels,
                })
                metadata.append((
                    sample, record_id, original_images, image_source_id,
                    adaptive_resize, effective_min_pixels, effective_max_pixels,
                ))

            batch_started = time.perf_counter()
            responses = runner.generate_batch(requests)
            batch_elapsed = time.perf_counter() - batch_started
            for response, metadata_item in zip(responses, metadata):
                (sample, record_id, original_images, image_source_id,
                 adaptive_resize, effective_min_pixels,
                 effective_max_pixels) = metadata_item
                sample_id = str(sample["id"])
                options = parse_options(sample["options"])
                prediction, fallback = official_mmmu_choice_extract(
                    response, options, parser_rng
                )
                gold = str(sample["answer"]).strip().upper()
                row = {
                    "id": record_id,
                    "sample_id": sample_id,
                    "condition": condition,
                    "category": category_for_subject(str(sample["subject"])),
                    "subject": str(sample["subject"]),
                    "ground_truth": gold,
                    "prediction": prediction,
                    "response": response,
                    "correct": prediction == gold,
                    "fallback_used": fallback,
                    "image_count": len(original_images),
                    "image_source_id": image_source_id,
                    "elapsed_seconds": round(batch_elapsed / len(batch), 3),
                    "runtime_measurement": (
                        "batch_wall_time_divided_by_batch_size"
                    ),
                    "batch_elapsed_seconds": round(batch_elapsed, 3),
                    "adaptive_multi_image_resize": adaptive_resize,
                    "effective_min_pixels": effective_min_pixels,
                    "effective_max_pixels": effective_max_pixels,
                    "reused_from_full_baseline": False,
                }
                append_jsonl(predictions_path, row)
                rows.append(row)
                completed_ids.add(record_id)
                generated_this_session += 1
                bar.update(1)
            current = summarize(rows)
            accuracy = current["conditions"][condition]["accuracy"]
            bar.set_postfix(
                condition=condition,
                accuracy=f"{accuracy:.3f}",
                batch=f"{len(batch)} in {batch_elapsed:.1f}s",
            )
            if generated_this_session % LOG_EVERY < len(batch):
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
        "benchmark": "MMMU-Pro standard10 modality ablation",
        "model": spec.name,
        "status": "completed" if len(completed_ids) == target_total else "partial",
        "generated_this_session": generated_this_session,
        "session_elapsed_seconds": round(time.perf_counter() - started_session, 3),
    })
    write_json(output_dir / "summary.json", summary)
    write_json(progress_path, {
        **summary,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
