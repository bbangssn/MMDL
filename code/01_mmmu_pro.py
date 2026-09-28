"""Run the MMMU-Pro vision or standard 10-option benchmark."""

import os

# Set this before importing torch/transformers. Physical GPU 0 becomes cuda:0.
VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from collections import defaultdict
from datetime import datetime
import time
import random

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from utils import (
    DATASET_CACHE,
    RESULTS_ROOT,
    VLLM,
    append_jsonl,
    category_for_subject,
    ensure_run_config,
    get_images,
    get_images_in_token_order,
    import_single_legacy_run,
    load_completed_predictions,
    official_mmmu_choice_extract,
    parse_options,
    resolve_model,
    write_json,
)
from reporting import (
    aggregate_rows,
    collect_environment,
    normalize_prediction_rows,
    runtime_summary,
)


DATA_PATH = os.environ.get("MMMU_PRO_DATA_PATH", "MMMU/MMMU_Pro")
SETTINGS = {"vision": "vision", "standard10": "standard (10 options)"}

# EDIT THESE SETTINGS
MODEL = 0  # Index in benchmarks.common.MODELS, or a model name such as "qwen3-vl-4b-instruct"
SETTING = ["vision", "standard10"][1] # "vision" or "standard10"
MODE = "direct"  # "direct" matches the official MMMU-Pro prompt mode
LIMIT = None  # None runs the full benchmark; use a small integer for a smoke test
LOG_EVERY = 500  # Rewrite progress.json every N completed samples
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


def main() -> None:
    session_started = time.perf_counter()
    if SETTING not in SETTINGS:
        raise ValueError(f"SETTING must be one of {tuple(SETTINGS)}, got {SETTING!r}")

    if MODE != "direct":
        raise ValueError("Only the official direct prompt is currently supported.")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    parser_rng = random.Random(SEED)

    spec = resolve_model(str(MODEL))
    output_dir = RESULTS_ROOT / "01_mmmu_pro" / SETTING / spec.name
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": f"MMMU-Pro {SETTING}", "dataset": DATA_PATH,
        "dataset_config": SETTINGS[SETTING], "split": "test",
        "setting": SETTING, "mode": MODE, "model": spec.name,
        "model_path": spec.path, "backend": "vllm", "batch_size": BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_images_per_prompt": MAX_IMAGES_PER_PROMPT,
        "multi_image_resize_threshold": MULTI_IMAGE_RESIZE_THRESHOLD,
        "multi_image_max_pixels": MULTI_IMAGE_MAX_PIXELS,
        "presence_penalty": PRESENCE_PENALTY,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH, "temperature": TEMPERATURE,
        "top_p": TOP_P, "top_k": TOP_K, "seed": SEED,
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
    }
    config_path = output_dir / "config.json"
    import_single_legacy_run(output_dir, predictions_path, config_path, config)
    ensure_run_config(config_path, predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())
    prior_rows, completed_ids = load_completed_predictions(predictions_path)
    normalize_prediction_rows(prior_rows)
    prior_by_id = {str(row["id"]): row for row in prior_rows}
    print(f"Live results: {output_dir}", flush=True)

    dataset = load_dataset(
        DATA_PATH,
        SETTINGS[SETTING],
        split="test",
        cache_dir=DATASET_CACHE,
    )
    total = len(prior_rows)
    correct = sum(bool(row["correct"]) for row in prior_rows)
    fallbacks = sum(bool(row.get("fallback_used", False)) for row in prior_rows)
    subject_scores: dict[str, list[int]] = defaultdict(list)
    for row in prior_rows:
        subject_scores[str(row["subject"])].append(int(bool(row["correct"])))
    if total:
        print(f"Resuming from {total} completed samples.", flush=True)
    runner = None  # Load only if an unfinished sample exists.
    target_total = min(len(dataset), LIMIT) if LIMIT is not None else len(dataset)
    progress_bar = tqdm(
        total=target_total,
        initial=min(total, target_total),
        desc=f"MMMU-Pro {SETTING}",
        unit="sample",
        dynamic_ncols=True,
    )

    pending = []
    for dataset_index, sample in enumerate(dataset):
        if dataset_index >= target_total:
            break
        options = parse_options(sample.get("options", []))
        sample_id = str(sample["id"])
        if sample_id in completed_ids:
            official_mmmu_choice_extract(
                prior_by_id[sample_id]["response"], options, parser_rng
            )
            continue
        pending.append((dataset_index, sample, options))

    if pending:
        runner = VLLM(
            spec, MAX_MODEL_LENGTH, GPU_MEMORY_UTILIZATION,
            MAX_IMAGES_PER_PROMPT, SEED,
        )
    for batch_start in range(0, len(pending), BATCH_SIZE):
        batch = pending[batch_start:batch_start + BATCH_SIZE]
        requests, metadata = [], []
        for dataset_index, sample, options in batch:
            subject = str(sample["subject"])
            category = category_for_subject(subject)
            if SETTING == "vision":
                prompt = (
                    "Answer with the option letter from the given choices directly. "
                    "The last line of your response should be of the following format: "
                    "'Answer: $LETTER' (without quotes) where LETTER is one of options."
                )
                images = get_images(sample)
            else:
                choices_text = "\n".join(
                    f"{chr(65 + index)}. {option}"
                    for index, option in enumerate(options)
                )
                prompt = (
                    f"{sample['question']}\n{choices_text}\n"
                    "Answer with the option letter from the given choices directly."
                )
                images = get_images_in_token_order(sample, prompt)
            effective_min_pixels = MIN_PIXELS
            effective_max_pixels = MAX_PIXELS
            adaptive_resize = len(images) > MULTI_IMAGE_RESIZE_THRESHOLD
            if adaptive_resize:
                effective_min_pixels = MULTI_IMAGE_MAX_PIXELS
                effective_max_pixels = MULTI_IMAGE_MAX_PIXELS
            requests.append({
                "prompt": prompt, "images": images,
                "max_new_tokens": MAX_NEW_TOKENS,
                "generation_kwargs": {
                    "do_sample": True, "temperature": TEMPERATURE,
                    "top_p": TOP_P, "top_k": TOP_K,
                    "repetition_penalty": 1.0,
                    "presence_penalty": PRESENCE_PENALTY,
                    "seed": SEED + dataset_index,
                },
                "min_pixels": effective_min_pixels,
                "max_pixels": effective_max_pixels,
            })
            metadata.append((
                sample, options, subject, category, len(images),
                adaptive_resize, effective_min_pixels, effective_max_pixels,
            ))

        batch_started = time.perf_counter()
        responses = runner.generate_batch(requests)
        batch_elapsed = time.perf_counter() - batch_started
        for response, metadata_item in zip(responses, metadata):
            (sample, options, subject, category, image_count, adaptive_resize,
             effective_min_pixels, effective_max_pixels) = metadata_item
            prediction, fallback_used = official_mmmu_choice_extract(
                response, options, parser_rng
            )
            fallbacks += int(fallback_used)
            is_correct = prediction == str(sample["answer"]).strip().upper()
            total += 1
            correct += int(is_correct)
            subject_scores[subject].append(int(is_correct))
            row = {
                "id": sample["id"], "category": category, "subject": subject,
                "question_type": "multiple-choice",
                "ground_truth": sample["answer"],
                "parsed_answer": prediction, "prediction": prediction,
                "model_raw_output": response, "response": response,
                "correct": is_correct, "fallback_used": fallback_used,
                "elapsed_seconds": round(batch_elapsed / len(batch), 3),
                "runtime_measurement": "batch_wall_time_divided_by_batch_size",
                "batch_elapsed_seconds": round(batch_elapsed, 3),
                "image_count": image_count,
                "adaptive_multi_image_resize": adaptive_resize,
                "effective_min_pixels": effective_min_pixels,
                "effective_max_pixels": effective_max_pixels,
            }
            append_jsonl(predictions_path, row)
            prior_rows.append(row)
            completed_ids.add(str(sample["id"]))
            progress_bar.update(1)
        progress_bar.set_postfix(
            accuracy=f"{correct / total:.3f}", fallbacks=fallbacks,
            batch=f"{len(batch)} in {batch_elapsed:.1f}s",
        )
        if total % LOG_EVERY < len(batch):
            write_json(progress_path, {
                "status": "running", "benchmark": f"MMMU-Pro {SETTING}",
                "model": spec.name, "last_sample_id": batch[-1][1]["id"],
                "total": total, "correct": correct, "fallbacks": fallbacks,
                "accuracy": correct / total,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })

    progress_bar.close()
    summary = {
        "benchmark": f"MMMU-Pro {SETTING}",
        "model": spec.name,
        "total": total,
        "correct": correct,
        "fallbacks": fallbacks,
        "accuracy": correct / total if total else 0.0,
        "status": "completed" if total >= len(dataset) else "partial",
        "protocol": {
            "reference": "MMMU-Benchmark/MMMU mmmu-pro",
            "setting": SETTING,
            "mode": MODE,
            "backend": "vllm",
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_model_length": MAX_MODEL_LENGTH,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repetition_penalty": 1.0,
            "presence_penalty": PRESENCE_PENALTY,
            "seed": SEED,
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "max_images_per_prompt": MAX_IMAGES_PER_PROMPT,
            "multi_image_resize_threshold": MULTI_IMAGE_RESIZE_THRESHOLD,
            "multi_image_max_pixels": MULTI_IMAGE_MAX_PIXELS,
            "adaptive_resize_samples": sum(
                bool(row.get("adaptive_multi_image_resize", False))
                for row in prior_rows
            ),
            "parser": "official parse_multi_choice_response",
        },
        "runtime": {
            **runtime_summary(prior_rows),
            "current_session_elapsed_seconds": round(time.perf_counter() - session_started, 3),
        },
        "categories": aggregate_rows(prior_rows, "category"),
        "subjects": aggregate_rows(prior_rows, "subject"),
    }
    write_json(output_dir / "summary.json", summary)
    final_status = (
        "completed" if total >= len(dataset) else "limit_reached"
    )
    write_json(progress_path, {
        "status": final_status,
        **summary,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    })
    print(summary)


if __name__ == "__main__":
    main()
