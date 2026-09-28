"""Experiment 00-vLLM: MMMU Validation with the official-style vLLM path."""

import os

VISIBLE_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = VISIBLE_GPU
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import atexit
from dataclasses import replace
from datetime import datetime
import json
import subprocess
import threading
import time



from vllm_compat import (
    FLASHINFER_SAMPLER_ENABLED, FLASHINFER_SAMPLER_REASON,
)

from datasets import load_dataset
from qwen_vl_utils import process_vision_info
from tqdm.auto import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from reporting import aggregate_rows, collect_environment, runtime_summary
from utils import (
    DATASET_CACHE, MMMU_SUBJECTS, MODEL_CACHE, RESULTS_ROOT,
    append_jsonl, category_for_subject, ensure_run_config, get_images,
    load_completed_predictions, parse_options, qwen_mmmu_prompt,
    qwen_rule_extract, resolve_model, write_json,
)


DATA_PATH = os.environ.get("MMMU_DATA_PATH", "MMMU/MMMU")
MODEL_PATH = os.environ.get("MMDL_MODEL_PATH")
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_REVISION = "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68"
EXPECTED_TOTAL = 900

# EDIT THESE SETTINGS
MODEL = 0
RUN_TAG = "official-penalty-1p5"  # Change when comparing a different recipe.
LIMIT = None
BATCH_SIZE = 32
LOG_EVERY_BATCHES = 1
MAX_NEW_TOKENS = 2048
MAX_MODEL_LENGTH = 9048
TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20
REPETITION_PENALTY = 1.0
PRESENCE_PENALTY = 1.5  # Official Qwen vLLM recipe.
SEED = 42
MIN_PIXELS = 1280 * 28 * 28
MAX_PIXELS = 5120 * 28 * 28
GPU_MEMORY_UTILIZATION = 0.80
MAX_IMAGES_PER_PROMPT = 7
TENSOR_PARALLEL_SIZE = 1
VRAM_POLL_INTERVAL_SECONDS = 0.5


class GPUMemoryMonitor:
    """Poll whole-device VRAM so vLLM worker processes are included."""

    def __init__(self, physical_gpu: str, interval: float):
        self.physical_gpu = physical_gpu
        self.interval = interval
        self.baseline_mib = None
        self.peak_mib = 0
        self.total_mib = None
        self.samples = 0
        self.errors = 0
        self._stop = threading.Event()
        self._thread = None

    def _query(self):
        output = subprocess.check_output(
            [
                "nvidia-smi", f"--id={self.physical_gpu}",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().splitlines()[0]
        used, total = (int(value.strip()) for value in output.split(","))
        return used, total

    def _run(self):
        while not self._stop.is_set():
            try:
                used, total = self._query()
                if self.baseline_mib is None:
                    self.baseline_mib = used
                self.peak_mib = max(self.peak_mib, used)
                self.total_mib = total
                self.samples += 1
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                self.errors += 1
            self._stop.wait(self.interval)

    def start(self):
        used, total = self._query()
        self.baseline_mib = used
        self.peak_mib = used
        self.total_mib = total
        self.samples = 1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def snapshot(self) -> dict:
        baseline = self.baseline_mib or 0
        return {
            "physical_gpu": self.physical_gpu,
            "source": "nvidia-smi whole-device polling",
            "poll_interval_seconds": self.interval,
            "baseline_used_mib": baseline,
            "peak_used_mib": self.peak_mib,
            "peak_increase_mib": self.peak_mib - baseline,
            "total_mib": self.total_mib,
            "peak_used_gib": round(self.peak_mib / 1024, 3),
            "peak_increase_gib": round((self.peak_mib - baseline) / 1024, 3),
            "utilization_at_peak_percent": (
                round(100 * self.peak_mib / self.total_mib, 2)
                if self.total_mib else None
            ),
            "samples": self.samples,
            "poll_errors": self.errors,
            "scope_note": (
                "Whole physical GPU usage, including vLLM worker processes and any "
                "other process on that GPU. Baseline is recorded before model loading."
            ),
        }

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval * 3))
        try:
            used, total = self._query()
            self.peak_mib = max(self.peak_mib, used)
            self.total_mib = total
            self.samples += 1
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            self.errors += 1
        return self.snapshot()


def prepare_vllm_input(sample: dict, prompt: str, processor) -> dict:
    content = []
    for image in get_images(sample):
        content.append({
            "type": "image", "image": image,
            "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        })
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    multimodal = {}
    if image_inputs is not None:
        multimodal["image"] = image_inputs
    if video_inputs is not None:
        multimodal["video"] = video_inputs
    return {
        "prompt": text,
        "multi_modal_data": multimodal,
        "mm_processor_kwargs": video_kwargs,
    }


def output_latency(output, batch_elapsed: float, batch_size: int) -> tuple[float, str]:
    metrics = getattr(output, "metrics", None)
    arrival = getattr(metrics, "arrival_time", None)
    finished = getattr(metrics, "finished_time", None)
    if arrival is not None and finished is not None and finished >= arrival:
        return float(finished - arrival), "vllm_request_latency"
    return batch_elapsed / batch_size, "batch_wall_time_divided_by_batch_size"


def load_pending(completed_ids: set[str], target_total: int) -> list[dict]:
    pending = []
    selected = 0
    for subject in MMMU_SUBJECTS:
        dataset = load_dataset(
            DATA_PATH, subject, split="validation", cache_dir=DATASET_CACHE,
            revision=(DATASET_REVISION if DATA_PATH == "MMMU/MMMU" else None),
        )
        for raw in dataset:
            if selected >= target_total:
                return pending
            selected += 1
            sample_id = str(raw["id"])
            if sample_id in completed_ids:
                continue
            sample = dict(raw)
            sample["_subject"] = subject
            pending.append(sample)
    return pending


def main() -> None:
    session_started = time.perf_counter()
    print(
        f"FlashInfer sampler: {'enabled' if FLASHINFER_SAMPLER_ENABLED else 'disabled'} "
        f"({FLASHINFER_SAMPLER_REASON})",
        flush=True,
    )
    spec = resolve_model(str(MODEL))
    if MODEL_PATH:
        spec = replace(spec, path=MODEL_PATH)
    model_revision = (
        MODEL_REVISION if spec.path == "Qwen/Qwen3-VL-4B-Instruct" else None
    )
    output_dir = RESULTS_ROOT / "00_mmmu_vllm" / RUN_TAG / spec.name
    predictions_path = output_dir / "predictions.jsonl"
    progress_path = output_dir / "progress.json"
    config = {
        "benchmark": "MMMU validation", "dataset": DATA_PATH,
        "dataset_revision": (
            DATASET_REVISION if DATA_PATH == "MMMU/MMMU" else None
        ),
        "split": "validation", "model": spec.name, "model_path": spec.path,
        "model_revision": model_revision,
        "backend": "vllm", "run_tag": RUN_TAG, "batch_size": BATCH_SIZE,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_model_length": MAX_MODEL_LENGTH,
        "temperature": TEMPERATURE, "top_p": TOP_P, "top_k": TOP_K,
        "repetition_penalty": REPETITION_PENALTY,
        "presence_penalty": PRESENCE_PENALTY, "seed": SEED,
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "max_images_per_prompt": MAX_IMAGES_PER_PROMPT,
        "tensor_parallel_size": TENSOR_PARALLEL_SIZE,
        "visible_gpu": VISIBLE_GPU,
        "vllm_use_flashinfer_sampler": FLASHINFER_SAMPLER_ENABLED,
    }
    ensure_run_config(output_dir / "config.json", predictions_path, config)
    write_json(output_dir / "environment.json", collect_environment())
    rows, completed_ids = load_completed_predictions(predictions_path)
    target_total = min(EXPECTED_TOTAL, LIMIT) if LIMIT is not None else EXPECTED_TOTAL
    pending = load_pending(completed_ids, target_total)
    print(f"Live results: {output_dir}", flush=True)
    if rows:
        print(f"Resuming from {len(rows)} completed samples.", flush=True)
    if not pending:
        print("No pending samples; rebuilding summary only.", flush=True)

    monitor = None
    peak_vram_path = output_dir / "peak_vram.json"
    peak_state = {"saved": False}

    def save_peak_vram():
        if monitor is None or peak_state["saved"]:
            return None
        value = monitor.stop()
        value.update({
            "benchmark": "MMMU validation", "backend": "vllm",
            "model": spec.name, "run_tag": RUN_TAG,
            "gpu_memory_utilization_setting": GPU_MEMORY_UTILIZATION,
            "vllm_use_flashinfer_sampler": FLASHINFER_SAMPLER_ENABLED,
            "measured_at": datetime.now().isoformat(timespec="seconds"),
        })
        write_json(peak_vram_path, value)
        peak_state["saved"] = True
        return value

    if pending:
        monitor = GPUMemoryMonitor(VISIBLE_GPU, VRAM_POLL_INTERVAL_SECONDS)
        monitor.start()
        atexit.register(save_peak_vram)
    processor = llm = None
    if pending:
        processor = AutoProcessor.from_pretrained(
            spec.path, cache_dir=MODEL_CACHE, use_fast=True,
            revision=model_revision,
        )
        llm = LLM(
            model=spec.path,
            revision=model_revision,
            download_dir=str(MODEL_CACHE),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            trust_remote_code=True,
            max_model_len=MAX_MODEL_LENGTH,
            limit_mm_per_prompt={"image": MAX_IMAGES_PER_PROMPT},
            seed=SEED,
        )
    sampling = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=REPETITION_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
        stop_token_ids=[],
    )
    bar = tqdm(
        total=target_total, initial=min(len(rows), target_total),
        desc="MMMU validation (vLLM)", unit="sample", dynamic_ncols=True,
    )
    batches_done = 0
    for start in range(0, len(pending), BATCH_SIZE):
        samples = pending[start:start + BATCH_SIZE]
        inputs = []
        metadata = []
        for sample in samples:
            options = parse_options(sample["options"])
            answer = str(sample["answer"]).strip()
            open_question = not options
            evaluation_options = [answer, "Other Answers"] if open_question else options
            gold = "A" if open_question else answer.upper()
            prompt = qwen_mmmu_prompt(str(sample["question"]), options)
            inputs.append(prepare_vllm_input(sample, prompt, processor))
            metadata.append((sample, evaluation_options, gold, open_question))

        batch_started = time.perf_counter()
        outputs = llm.generate(inputs, sampling_params=sampling, use_tqdm=False)
        batch_elapsed = time.perf_counter() - batch_started
        for output, (sample, options, gold, open_question) in zip(outputs, metadata):
            response = output.outputs[0].text.strip()
            prediction = qwen_rule_extract(response, options)
            fallback = prediction in ("", "Z")
            elapsed, runtime_method = output_latency(output, batch_elapsed, len(samples))
            row = {
                "id": sample["id"],
                "category": category_for_subject(sample["_subject"]),
                "subject": sample["_subject"],
                "question_type": "open" if open_question else "multiple-choice",
                "ground_truth": sample["answer"], "gold": gold,
                "parsed_answer": prediction, "prediction": prediction,
                "model_raw_output": response, "response": response,
                "correct": prediction == gold, "fallback_used": fallback,
                "open_question_reformatted": open_question,
                "elapsed_seconds": round(elapsed, 3),
                "runtime_measurement": runtime_method,
                "batch_elapsed_seconds": round(batch_elapsed, 3),
            }
            append_jsonl(predictions_path, row)
            rows.append(row); completed_ids.add(str(sample["id"])); bar.update(1)
        batches_done += 1
        correct = sum(bool(row["correct"]) for row in rows)
        unresolved = sum(bool(row["fallback_used"]) for row in rows)
        bar.set_postfix(
            accuracy=f"{correct/len(rows):.3f}", unresolved=unresolved,
            batch=f"{len(samples)} in {batch_elapsed:.1f}s",
            peak_vram=f"{monitor.snapshot()['peak_used_gib']:.1f}GiB",
        )
        if batches_done % LOG_EVERY_BATCHES == 0:
            write_json(progress_path, {
                "status": "running", "completed": len(rows), "target": target_total,
                "correct": correct, "unresolved": unresolved,
                "accuracy": correct / len(rows),
                "last_sample_id": samples[-1]["id"],
                "peak_vram": monitor.snapshot(),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            })
    bar.close()

    if monitor is not None:
        peak_vram = save_peak_vram()
    elif peak_vram_path.exists():
        peak_vram = json.loads(peak_vram_path.read_text())
    else:
        peak_vram = None

    correct = sum(bool(row["correct"]) for row in rows)
    unresolved = sum(bool(row["fallback_used"]) for row in rows)
    summary = {
        "benchmark": "MMMU validation", "model": spec.name,
        "backend": "vllm", "total": len(rows), "correct": correct,
        "unresolved": unresolved,
        "accuracy": correct / len(rows) if rows else 0.0,
        "status": "completed" if len(rows) == target_total else "partial",
        "officially_comparable": False,
        "protocol": config,
        "runtime": {
            **runtime_summary(rows),
            "session_wall_seconds": round(time.perf_counter() - session_started, 3),
            "note": "Per-request vLLM latency when available; otherwise batch wall time / batch size.",
        },
        "peak_vram": peak_vram,
        "categories": aggregate_rows(rows, "category") if rows else {},
        "subjects": aggregate_rows(rows, "subject") if rows else {},
    }
    write_json(output_dir / "summary.json", summary)
    write_json(progress_path, {
        **summary, "updated_at": datetime.now().isoformat(timespec="seconds")
    })
    print(summary)


if __name__ == "__main__":
    main()
