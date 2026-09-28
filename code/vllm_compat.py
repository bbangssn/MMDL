"""Configure vLLM sampler compatibility before importing vLLM."""

import os
import re
import subprocess

import torch


def _cuda_toolkit_version() -> tuple[int, int] | None:
    try:
        output = subprocess.check_output(
            ["nvcc", "--version"], text=True, stderr=subprocess.DEVNULL
        )
        match = re.search(r"release (\d+)\.(\d+)", output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (OSError, subprocess.SubprocessError):
        pass
    match = re.match(r"(\d+)\.(\d+)", torch.version.cuda or "")
    return (int(match.group(1)), int(match.group(2))) if match else None


def configure_flashinfer_sampler() -> tuple[bool, str]:
    explicit = os.environ.get("VLLM_USE_FLASHINFER_SAMPLER")
    if explicit is not None:
        enabled = explicit.strip().lower() not in {"0", "false", "no", "off"}
        return enabled, "explicit environment override"

    capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    toolkit = _cuda_toolkit_version()
    minimum = (11, 8) if capability == (8, 9) else None
    if capability is not None and capability[0] == 12:
        minimum = (12, 9)
    if minimum is not None and (toolkit is None or toolkit < minimum):
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        return False, (
            f"auto-disabled for SM {capability[0]}.{capability[1]} with CUDA "
            f"toolkit {toolkit}; requires >= {minimum}"
        )
    return True, "vLLM default (compatible or not applicable)"


FLASHINFER_SAMPLER_ENABLED, FLASHINFER_SAMPLER_REASON = configure_flashinfer_sampler()
