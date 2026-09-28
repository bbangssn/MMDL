"""Run the complete MMMU validation baseline and local-judge pipeline."""

from pathlib import Path
import subprocess
import sys


CODE_DIR = Path(__file__).resolve().parent
STAGES = ("00_mmmu_vllm.py", "02_mmmu_vllm_judge.py")


def main() -> None:
    for stage in STAGES:
        print(f"\n=== Running {stage} ===", flush=True)
        subprocess.run([sys.executable, str(CODE_DIR / stage)], check=True)


if __name__ == "__main__":
    main()
