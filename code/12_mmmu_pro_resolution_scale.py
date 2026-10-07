"""Experiment 12: scaled paired image-resolution probe on MMMU-Pro."""

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).with_name("04_bottleneck_probes.py")
SPEC = importlib.util.spec_from_file_location("bottleneck_probes", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not load {SOURCE}")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


# EDIT THESE SETTINGS
probe.EXPERIMENT_DIR = "12_mmmu_pro_resolution_scale"
probe.BENCHMARK_NAME = "MMMU-Pro scaled paired resolution probe"
probe.SAMPLES_PER_CATEGORY = 20  # 6 categories x 20 = 120 paired samples
probe.REQUIRE_EXPLANATION = False  # Resolution comparison needs no rationale.
probe.CONDITIONS = ("direct", "low_resolution", "high_resolution")
probe.BATCH_SIZE = 20
probe.LOG_EVERY = 20


if __name__ == "__main__":
    probe.main()
