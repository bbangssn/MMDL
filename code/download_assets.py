"""Prepare Qwen3-VL, MMMU, and MMMU-Pro with their runtime libraries."""

from pathlib import Path

from datasets import get_dataset_config_names, load_dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


DATASET_ROOT = Path(__file__).resolve().parent.parent / "datasets"
MODEL_ROOT = Path(__file__).resolve().parent.parent / "models"
MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_REVISION = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATASET_REVISIONS = {
    "MMMU/MMMU": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68",
    "MMMU/MMMU_Pro": None,
}


def prepare_model() -> None:
    print(f"\nLoading model and processor: {MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        MODEL_ID, cache_dir=MODEL_ROOT, revision=MODEL_REVISION
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        cache_dir=MODEL_ROOT,
        revision=MODEL_REVISION,
        dtype="auto",
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"Ready: {type(model).__name__}, processor={type(processor).__name__}")


def prepare_dataset(dataset_id: str) -> None:
    revision = DATASET_REVISIONS[dataset_id]
    configs = get_dataset_config_names(dataset_id, revision=revision)
    print(f"\nPreparing {dataset_id} ({len(configs)} configs)")
    for index, config in enumerate(configs, start=1):
        dataset = load_dataset(
            dataset_id, config, cache_dir=DATASET_ROOT, revision=revision
        )
        sizes = {split: len(rows) for split, rows in dataset.items()}
        print(f"[{index}/{len(configs)}] {config}: {sizes}")


def main() -> None:
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)

    prepare_model()
    for dataset_id in DATASET_REVISIONS:
        prepare_dataset(dataset_id)

    print("\nAll assets are cached and runtime-ready.")


if __name__ == "__main__":
    main()
