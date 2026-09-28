"""Experiment 05: shuffled-image control without a conflict warning."""

import os

VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from datetime import datetime
import importlib.util
import json
import random
import time
from pathlib import Path

from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment, runtime_summary
from utils import (
    DATASET_CACHE, RESULTS_ROOT, VLLM, append_jsonl, ensure_run_config,
    load_completed_predictions, official_mmmu_choice_extract, parse_options,
    resolve_model, write_json,
)


# EDIT THESE SETTINGS
MODEL = 0
BATCH_SIZE = 24
MAX_NEW_TOKENS = 512
MAX_MODEL_LENGTH = 9048
GPU_MEMORY_UTILIZATION = 0.80
MAX_IMAGES_PER_PROMPT = 7
PRESENCE_PENALTY = 1.5
SEED = 43

DATA_PATH = os.environ.get("MMMU_PRO_DATA_PATH", "MMMU/MMMU_Pro")
DATASET_CONFIG = "standard (10 options)"


def load_probe_helpers():
    path = Path(__file__).with_name("04_bottleneck_probes.py")
    spec = importlib.util.spec_from_file_location("bottleneck_probes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> None:
    probes = load_probe_helpers()
    model_spec = resolve_model(str(MODEL))
    source_dir = RESULTS_ROOT / "04_bottleneck_probes" / model_spec.name
    manifest = json.loads((source_dir / "sample_manifest.json").read_text())
    output_dir = RESULTS_ROOT / "05_conflict_control" / model_spec.name
    predictions_path = output_dir / "predictions.jsonl"
    config = {
        "benchmark": "MMMU-Pro shuffled-image control",
        "dataset": DATA_PATH, "dataset_config": DATASET_CONFIG, "split": "test",
        "model": model_spec.name, "model_path": model_spec.path,
        "sample_ids": [entry["id"] for entry in manifest["samples"]],
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH, "seed": SEED,
        "visible_gpu": VISIBLE_GPU, "backend": "vllm",
        "batch_size": BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "presence_penalty": PRESENCE_PENALTY,
        "condition": "shuffled_no_instruction",
    }
    ensure_run_config(output_dir / "config.json", predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    dataset = load_dataset(DATA_PATH, DATASET_CONFIG, split="test", cache_dir=DATASET_CACHE)
    by_id = {str(sample["id"]): dict(sample) for sample in dataset}
    rows, completed = load_completed_predictions(predictions_path)
    runner = None
    parser_rng = random.Random(SEED)
    started_all = time.perf_counter()
    entries = [
        entry for entry in manifest["samples"]
        if f"shuffled_no_instruction:{entry['id']}" not in completed
    ]
    bar = tqdm(
        total=len(manifest["samples"]), initial=len(rows),
        desc="Conflict control", unit="sample", dynamic_ncols=True,
    )
    if entries:
        runner = VLLM(
            model_spec, MAX_MODEL_LENGTH, GPU_MEMORY_UTILIZATION,
            MAX_IMAGES_PER_PROMPT, SEED,
        )
    for batch_start in range(0, len(entries), BATCH_SIZE):
        batch = entries[batch_start:batch_start + BATCH_SIZE]
        requests = []
        metadata = []
        for entry in batch:
            sample_id = entry["id"]
            sample, donor = by_id[sample_id], by_id[entry["donor_id"]]
            prompt = probes.prompt_for(sample)
            images = probes.images_for(donor, probes.prompt_for(donor))
            request_seed = probes.seed_sample(sample_id)
            requests.append(probes.generation_request(
                prompt, images, MAX_NEW_TOKENS, seed=request_seed
            ))
            metadata.append((entry, sample))
        started = time.perf_counter()
        responses = runner.generate_batch(requests)
        batch_elapsed = time.perf_counter() - started
        for response, (entry, sample) in zip(responses, metadata):
            sample_id = entry["id"]
            record_id = f"shuffled_no_instruction:{sample_id}"
            options = parse_options(sample["options"])
            prediction, fallback = official_mmmu_choice_extract(
                response, options, parser_rng
            )
            gold = str(sample["answer"]).strip().upper()
            row = {
                "id": record_id, "sample_id": sample_id,
                "condition": "shuffled_no_instruction",
                "category": entry["category"], "subject": entry["subject"],
                "ground_truth": gold, "prediction": prediction,
                "canonical_prediction": prediction,
                "correct": prediction == gold,
                "fallback_used": fallback, "response": response,
                "intermediate": None,
                "image_source_id": entry["donor_id"],
                "elapsed_seconds": round(batch_elapsed / len(batch), 3),
                "runtime_measurement": "batch_wall_time_divided_by_batch_size",
                "batch_elapsed_seconds": round(batch_elapsed, 3),
            }
            append_jsonl(predictions_path, row)
            rows.append(row)
            completed.add(record_id)
            bar.update(1)
        bar.set_postfix(
            accuracy=f"{sum(r['correct'] for r in rows)/len(rows):.3f}",
            batch=f"{len(batch)} in {batch_elapsed:.1f}s",
        )
        write_json(output_dir / "progress.json", {
            "status": "running", "completed": len(rows),
            "target": len(manifest["samples"]),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })

    bar.close()
    summary = {
        "benchmark": "MMMU-Pro shuffled-image control", "model": model_spec.name,
        "status": "completed" if len(rows) == len(manifest["samples"]) else "partial",
        "total": len(rows), "correct": sum(r["correct"] for r in rows),
        "accuracy": sum(r["correct"] for r in rows) / len(rows),
        "fallbacks": sum(r["fallback_used"] for r in rows),
        "runtime": runtime_summary(rows),
        "session_elapsed_seconds": round(time.perf_counter() - started_all, 3),
    }
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "progress.json", {
        **summary, "updated_at": datetime.now().isoformat(timespec="seconds")
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
