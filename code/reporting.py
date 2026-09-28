"""Runtime metadata and benchmark aggregation helpers."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from collections import defaultdict
from typing import Any

import torch

from utils import category_for_subject


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def collect_environment() -> dict[str, Any]:
    memory_bytes = None
    try:
        memory_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        pass

    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {
            "name": properties.name,
            "memory_gib": round(properties.total_memory / 1024**3, 2),
        }

    return {
        "platform": platform.platform(),
        "os": {"system": platform.system(), "release": platform.release()},
        "python": sys.version.split()[0],
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "ram_gib": round(memory_bytes / 1024**3, 2) if memory_bytes else None,
        "gpu": gpu,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda": torch.version.cuda,
        "packages": {
            name: _version(name)
            for name in (
                "torch", "transformers", "datasets", "accelerate",
                "qwen-vl-utils", "vllm", "tqdm",
            )
        },
    }


def normalize_prediction_rows(rows: list[dict[str, Any]]) -> None:
    """Backfill report fields absent from results produced by older scripts."""
    for row in rows:
        row.setdefault("category", category_for_subject(str(row["subject"])))


def aggregate_rows(
    rows: list[dict[str, Any]], key: str
) -> dict[str, dict[str, int | float | None]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)

    result = {}
    for name, group in sorted(groups.items()):
        correct = sum(bool(row["correct"]) for row in group)
        elapsed = [
            float(row["elapsed_seconds"])
            for row in group
            if row.get("elapsed_seconds") is not None
        ]
        result[name] = {
            "count": len(group),
            "correct": correct,
            "accuracy": correct / len(group),
            "elapsed_seconds": round(sum(elapsed), 3) if elapsed else None,
            "timed_samples": len(elapsed),
        }
    return result


def runtime_summary(rows: list[dict[str, Any]]) -> dict[str, int | float | None]:
    elapsed = [
        float(row["elapsed_seconds"])
        for row in rows
        if row.get("elapsed_seconds") is not None
    ]
    total = sum(elapsed)
    return {
        "total_elapsed_seconds": round(total, 3) if elapsed else None,
        "mean_elapsed_seconds": round(total / len(elapsed), 3) if elapsed else None,
        "timed_samples": len(elapsed),
        "untimed_legacy_samples": len(rows) - len(elapsed),
    }
