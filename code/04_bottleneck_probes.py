"""Experiment 04: small diagnostic probes for major MMMU-Pro bottlenecks."""

import os

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

import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from reporting import collect_environment, runtime_summary
from utils import (
    DATASET_CACHE, RESULTS_ROOT, VLLM, append_jsonl, category_for_subject,
    ensure_run_config, get_images_in_token_order, load_completed_predictions,
    official_mmmu_choice_extract, parse_options, resolve_model, write_json,
)


DATA_PATH = os.environ.get("MMMU_PRO_DATA_PATH", "MMMU/MMMU_Pro")
DATASET_CONFIG = "standard (10 options)"

# EDIT THESE SETTINGS
MODEL = 0
SAMPLES_PER_CATEGORY = 4
CONDITIONS = (
    "direct", "brief_cot", "structured_reasoning", "grounded_answer",
    "low_resolution", "high_resolution", "option_permuted",
    "shuffled_with_conflict_instruction", "describe_then_solve",
    "knowledge_notes_then_solve", "oracle_rationale",
    "sample_seed_0", "sample_seed_1", "sample_seed_2",
)
LOG_EVERY = 8
BATCH_SIZE = 24
MAX_NEW_TOKENS = 512
MAX_MODEL_LENGTH = 9048
GPU_MEMORY_UTILIZATION = 0.80
MAX_IMAGES_PER_PROMPT = 7
PRESENCE_PENALTY = 1.5
SEED = 43
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
DEFAULT_MIN_PIXELS = 1280 * 28 * 28
DEFAULT_MAX_PIXELS = 5120 * 28 * 28
LOW_MIN_PIXELS = 256 * 28 * 28
LOW_MAX_PIXELS = 1280 * 28 * 28
HIGH_MIN_PIXELS = 1280 * 28 * 28
HIGH_MAX_PIXELS = 7168 * 28 * 28


def prompt_for(sample: dict, options: list[str] | None = None) -> str:
    options = options or parse_options(sample["options"])
    choices = "\n".join(
        f"{chr(65 + index)}. {option}" for index, option in enumerate(options)
    )
    return (
        f"{sample['question']}\n{choices}\n"
        "End with exactly 'Answer: X', where X is one option letter."
    )


def images_for(sample: dict, prompt: str):
    return get_images_in_token_order(sample, prompt)


def valid_explanation(sample: dict) -> bool:
    return str(sample.get("explanation", "")).strip() not in ("", "?", "None", "null")


def select_samples(samples: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    groups = defaultdict(list)
    for sample in samples:
        prompt = prompt_for(sample)
        count = len(images_for(sample, prompt))
        # One-image items isolate resolution without exceeding the 9,048-token cap.
        if valid_explanation(sample) and count == 1:
            groups[category_for_subject(str(sample["subject"]))].append(sample)
    selected = []
    for category in sorted(groups):
        rng.shuffle(groups[category])
        chosen = groups[category][:SAMPLES_PER_CATEGORY]
        if len(chosen) != SAMPLES_PER_CATEGORY:
            raise RuntimeError(f"Only {len(chosen)} valid samples for {category}")
        selected.extend(chosen)
    if len(selected) != 6 * SAMPLES_PER_CATEGORY:
        raise RuntimeError(f"Expected {6*SAMPLES_PER_CATEGORY}, got {len(selected)}")
    return selected


def donor_map(selected: list[dict], all_samples: list[dict]) -> dict[str, str]:
    by_count = defaultdict(list)
    for sample in all_samples:
        count = len(images_for(sample, prompt_for(sample)))
        by_count[count].append(str(sample["id"]))
    result = {}
    for sample in selected:
        sample_id = str(sample["id"])
        count = len(images_for(sample, prompt_for(sample)))
        candidates = [value for value in by_count[count] if value != sample_id]
        if not candidates:
            raise RuntimeError(f"No donor with {count} images for {sample_id}")
        index = int.from_bytes(hashlib.sha256(sample_id.encode()).digest()[:4], "big")
        result[sample_id] = candidates[index % len(candidates)]
    return result


def seed_sample(sample_id: str, offset: int = 0) -> int:
    digest = hashlib.sha256(f"{SEED}:{sample_id}:{offset}".encode()).digest()
    value = int.from_bytes(digest[:4], "big")
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    return value


def permute_options(options: list[str]) -> tuple[list[str], list[int]]:
    # Deterministic one-position rotation. mapping[new index] = original index.
    mapping = list(range(1, len(options))) + [0]
    return [options[index] for index in mapping], mapping


def generate(runner, prompt, images, max_tokens, do_sample=False,
             min_pixels=DEFAULT_MIN_PIXELS, max_pixels=DEFAULT_MAX_PIXELS,
             seed=None):
    kwargs = {
        "do_sample": do_sample, "repetition_penalty": 1.0,
        "presence_penalty": PRESENCE_PENALTY, "seed": seed,
    }
    if do_sample:
        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
    return runner.generate(
        prompt, images, max_tokens, max_model_length=MAX_MODEL_LENGTH,
        generation_kwargs=kwargs, min_pixels=min_pixels, max_pixels=max_pixels,
    )



def generation_request(prompt, images, max_tokens, do_sample=False,
                       min_pixels=DEFAULT_MIN_PIXELS,
                       max_pixels=DEFAULT_MAX_PIXELS, seed=None):
    kwargs = {
        "do_sample": do_sample,
        "repetition_penalty": 1.0,
        "presence_penalty": PRESENCE_PENALTY,
        "seed": seed,
    }
    if do_sample:
        kwargs.update(temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K)
    return {
        "prompt": prompt,
        "images": images,
        "max_new_tokens": max_tokens,
        "generation_kwargs": kwargs,
        "min_pixels": min_pixels,
        "max_pixels": max_pixels,
    }

def summarize(rows: list[dict], sample_count: int) -> dict:
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)
    direct = {str(row["sample_id"]): row for row in by_condition.get("direct", [])}
    result = {}
    for condition in CONDITIONS:
        group = by_condition.get(condition, [])
        paired = [row for row in group if str(row["sample_id"]) in direct]
        correct = sum(bool(row["correct"]) for row in group)
        result[condition] = {
            "total": len(group), "correct": correct,
            "accuracy": correct / len(group) if group else None,
            "fallbacks": sum(bool(row["fallback_used"]) for row in group),
            "delta_vs_direct": (
                correct / len(group)
                - sum(bool(row["correct"]) for row in direct.values()) / len(direct)
                if group and direct else None
            ),
            "canonical_agreement_with_direct": (
                sum(row["canonical_prediction"] == direct[str(row["sample_id"])]["canonical_prediction"] for row in paired)
                / len(paired) if paired else None
            ),
            "runtime": runtime_summary(group),
        }
    # Per-item sampled answer stability across three seeds.
    sampled = []
    for sample_id in direct:
        predictions = [
            next((r["canonical_prediction"] for r in by_condition[f"sample_seed_{i}"]
                  if str(r["sample_id"]) == sample_id), None)
            for i in range(3)
        ]
        if all(value is not None for value in predictions):
            sampled.append(len(set(predictions)) == 1)
    return {
        "sample_count": sample_count,
        "conditions": result,
        "three_seed_prediction_stability": (
            sum(sampled) / len(sampled) if sampled else None
        ),
    }


def main() -> None:
    session_started = time.perf_counter()
    spec = resolve_model(str(MODEL))
    output_dir = RESULTS_ROOT / "04_bottleneck_probes" / spec.name
    predictions_path = output_dir / "predictions.jsonl"
    config = {
        "benchmark": "MMMU-Pro small bottleneck probes",
        "dataset": DATA_PATH, "dataset_config": DATASET_CONFIG, "split": "test",
        "model": spec.name, "model_path": spec.path,
        "samples_per_category": SAMPLES_PER_CATEGORY,
        "conditions": list(CONDITIONS), "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH, "seed": SEED,
        "visible_gpu": VISIBLE_GPU, "backend": "vllm",
        "batch_size": BATCH_SIZE,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "presence_penalty": PRESENCE_PENALTY,
        "default_min_pixels": DEFAULT_MIN_PIXELS,
        "default_max_pixels": DEFAULT_MAX_PIXELS,
        "low_min_pixels": LOW_MIN_PIXELS, "low_max_pixels": LOW_MAX_PIXELS,
        "high_min_pixels": HIGH_MIN_PIXELS, "high_max_pixels": HIGH_MAX_PIXELS,
    }
    ensure_run_config(output_dir / "config.json", predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())

    dataset = load_dataset(DATA_PATH, DATASET_CONFIG, split="test", cache_dir=DATASET_CACHE)
    samples = [dict(value) for value in dataset]
    by_id = {str(value["id"]): value for value in samples}
    selected = select_samples(samples)
    donors = donor_map(selected, samples)
    manifest = {
        "selection": "4 explanation-bearing one-image items per category",
        "samples": [{
            "id": str(x["id"]), "category": category_for_subject(str(x["subject"])),
            "subject": str(x["subject"]), "donor_id": donors[str(x["id"])],
        } for x in selected],
    }
    manifest_path = output_dir / "sample_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Existing manifest differs")
    if not manifest_path.exists():
        write_json(manifest_path, manifest)

    rows, completed = load_completed_predictions(predictions_path)
    total_target = len(selected) * len(CONDITIONS)
    bar = tqdm(total=total_target, initial=len(rows), desc="Bottleneck probes",
               unit="condition", dynamic_ncols=True)
    runner = None
    generated = 0

    prior_by_id = {str(row["id"]): row for row in rows}
    for condition in CONDITIONS:
        parser_rng = random.Random(SEED + CONDITIONS.index(condition))
        pending = [
            sample for sample in selected
            if f"{condition}:{sample['id']}" not in completed
        ]
        if pending and runner is None:
            runner = VLLM(
                spec, MAX_MODEL_LENGTH, GPU_MEMORY_UTILIZATION,
                MAX_IMAGES_PER_PROMPT, SEED,
            )
        for batch_start in range(0, len(pending), BATCH_SIZE):
            batch = pending[batch_start:batch_start + BATCH_SIZE]
            prepared = []
            intermediate_requests = []
            for sample in batch:
                sample_id = str(sample["id"])
                options = parse_options(sample["options"])
                used_options = options
                mapping = list(range(len(options)))
                prompt = prompt_for(sample)
                images = images_for(sample, prompt)
                max_tokens = MAX_NEW_TOKENS
                do_sample = False
                min_pixels = DEFAULT_MIN_PIXELS
                max_pixels = DEFAULT_MAX_PIXELS
                image_source_id = sample_id
                intermediate_kind = None

                if condition == "brief_cot":
                    prompt += "\nGive at most two short reasoning sentences before the final answer."
                    max_tokens = 160
                elif condition == "structured_reasoning":
                    prompt += ("\nUse this checklist briefly: (1) visual evidence, "
                               "(2) relevant principle, (3) compare choices, (4) verify.")
                elif condition == "grounded_answer":
                    prompt += "\nState one decisive visual observation, then give the final answer."
                    max_tokens = 192
                elif condition == "low_resolution":
                    min_pixels, max_pixels = LOW_MIN_PIXELS, LOW_MAX_PIXELS
                elif condition == "high_resolution":
                    min_pixels, max_pixels = HIGH_MIN_PIXELS, HIGH_MAX_PIXELS
                elif condition == "option_permuted":
                    used_options, mapping = permute_options(options)
                    prompt = prompt_for(sample, used_options)
                    images = images_for(sample, prompt)
                elif condition == "shuffled_with_conflict_instruction":
                    image_source_id = donors[sample_id]
                    donor = by_id[image_source_id]
                    images = images_for(donor, prompt_for(donor))
                    prompt += ("\nUse an image only when it is relevant and consistent "
                               "with the question; ignore irrelevant visual content.")
                elif condition == "describe_then_solve":
                    intermediate_kind = condition
                    description_prompt = (
                        f"Question context: {sample['question']}\n"
                        "Describe only the visual facts needed to solve the question. "
                        "Do not select an option or give the answer."
                    )
                    intermediate_requests.append(generation_request(
                        description_prompt, images, 384,
                        seed=seed_sample(sample_id),
                    ))
                elif condition == "knowledge_notes_then_solve":
                    intermediate_kind = condition
                    notes_prompt = (
                        prompt_for(sample)
                        + "\nList only the relevant domain principles or formulas. "
                          "Do not select an option and do not state the final answer."
                    )
                    intermediate_requests.append(generation_request(
                        notes_prompt, images, 384,
                        seed=seed_sample(sample_id),
                    ))
                elif condition == "oracle_rationale":
                    prompt += f"\nReference rationale:\n{str(sample['explanation']).strip()}"
                    max_tokens = 192
                elif condition.startswith("sample_seed_"):
                    do_sample = True
                    max_tokens = 256
                elif condition != "direct":
                    raise ValueError(condition)

                offset = int(condition.rsplit("_", 1)[1]) if condition.startswith("sample_seed_") else 0
                prepared.append({
                    "sample": sample, "options": options,
                    "used_options": used_options, "mapping": mapping,
                    "prompt": prompt, "images": images,
                    "max_tokens": max_tokens, "do_sample": do_sample,
                    "min_pixels": min_pixels, "max_pixels": max_pixels,
                    "request_seed": seed_sample(sample_id, offset),
                    "image_source_id": image_source_id,
                    "intermediate_kind": intermediate_kind,
                })

            batch_total_started = time.perf_counter()
            intermediates = [None] * len(prepared)
            if intermediate_requests:
                intermediates = runner.generate_batch(intermediate_requests)
                for item, intermediate in zip(prepared, intermediates):
                    if item["intermediate_kind"] == "describe_then_solve":
                        item["prompt"] = (
                            prompt_for(item["sample"])
                            + f"\nVisual description:\n{intermediate}"
                        )
                        item["prompt"] = re.sub(
                            r"<image\s+\d+>", "[see description]", item["prompt"]
                        )
                        item["images"] = []
                        item["max_tokens"] = 256
                    elif item["intermediate_kind"] == "knowledge_notes_then_solve":
                        item["prompt"] += f"\nPotential knowledge notes:\n{intermediate}"
                        item["max_tokens"] = 256

            requests = [generation_request(
                item["prompt"], item["images"], item["max_tokens"],
                item["do_sample"], item["min_pixels"], item["max_pixels"],
                item["request_seed"],
            ) for item in prepared]
            responses = runner.generate_batch(requests)
            batch_elapsed = time.perf_counter() - batch_total_started
            response_by_id = {
                str(item["sample"]["id"]): (response, item, intermediate)
                for item, response, intermediate in zip(
                    prepared, responses, intermediates
                )
            }

            # Replay completed rows so randomized official fallback parsing has
            # exactly the same RNG state after an interrupted/resumed run.
            new_ids = set(response_by_id)
            for sample in selected:
                sample_id = str(sample["id"])
                record_id = f"{condition}:{sample_id}"
                if sample_id not in new_ids:
                    if record_id in prior_by_id:
                        prior = prior_by_id[record_id]
                        official_mmmu_choice_extract(
                            prior["response"], parse_options(sample["options"]),
                            parser_rng,
                        )
                    continue
                response, item, intermediate = response_by_id[sample_id]
                prediction, fallback = official_mmmu_choice_extract(
                    response, item["used_options"], parser_rng
                )
                predicted_new_index = ord(prediction) - 65 if prediction else -1
                mapping = item["mapping"]
                canonical_prediction = (
                    chr(65 + mapping[predicted_new_index])
                    if 0 <= predicted_new_index < len(mapping) else prediction
                )
                gold = str(sample["answer"]).strip().upper()
                row = {
                    "id": record_id, "sample_id": sample_id,
                    "condition": condition,
                    "category": category_for_subject(str(sample["subject"])),
                    "subject": str(sample["subject"]), "ground_truth": gold,
                    "prediction": prediction,
                    "canonical_prediction": canonical_prediction,
                    "correct": canonical_prediction == gold,
                    "fallback_used": fallback, "response": response,
                    "intermediate": intermediate,
                    "image_source_id": item["image_source_id"],
                    "elapsed_seconds": round(batch_elapsed / len(batch), 3),
                    "runtime_measurement": "batch_wall_time_divided_by_batch_size",
                    "batch_elapsed_seconds": round(batch_elapsed, 3),
                }
                append_jsonl(predictions_path, row)
                rows.append(row)
                completed.add(record_id)
                prior_by_id[record_id] = row
                generated += 1
                bar.update(1)
            current = summarize(rows, len(selected))
            accuracy = current["conditions"][condition]["accuracy"]
            bar.set_postfix(
                condition=condition, accuracy=f"{accuracy:.3f}",
                batch=f"{len(batch)} in {batch_elapsed:.1f}s",
            )
            if generated % LOG_EVERY < len(batch):
                write_json(output_dir / "progress.json", {
                    "status": "running", "completed": len(rows),
                    "target": total_target, "summary": current,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                })

    bar.close()
    summary = summarize(rows, len(selected))
    summary.update({
        "benchmark": "MMMU-Pro small bottleneck probes", "model": spec.name,
        "status": "completed" if len(rows) == total_target else "partial",
        "generated_this_session": generated,
        "session_elapsed_seconds": round(time.perf_counter()-session_started, 3),
    })
    write_json(output_dir / "summary.json", summary)
    write_json(output_dir / "progress.json", {
        **summary, "updated_at": datetime.now().isoformat(timespec="seconds")
    })
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
