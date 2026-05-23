"""
Backend генерации изображений — StoryDiffusion на локальном GPU.
"""
from __future__ import annotations

from app.services.storydiffusion_service import (
    generation_status_hint,
    run_storydiffusion_generation,
)


def run_sdxl_generation(
    job_id: str,
    nsm_result: dict,
    cim_result: dict,
    cfg_scale: float,
    ddim_steps: int,
) -> list[dict]:
    """Alias для совместимости API — использует StoryDiffusion."""
    return run_storydiffusion_generation(
        job_id=job_id,
        nsm_result=nsm_result,
        cim_result=cim_result,
        cfg_scale=cfg_scale,
        ddim_steps=ddim_steps,
    )


def gpu_status() -> dict:
    try:
        import torch
    except ImportError:
        return {
            "gpu_available": False,
            "progress_hint": "Установите requirements-gpu.txt",
            "backend": "storydiffusion",
        }

    hint = generation_status_hint()
    available = torch.cuda.is_available() and "CUDA не" not in hint and "PyTorch" not in hint
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None

    return {
        "gpu_available": available,
        "progress_hint": hint,
        "device_name": name,
        "backend": "storydiffusion",
    }
