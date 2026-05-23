"""
Генерация сцен через StoryDiffusion (Consistent Self-Attention).
https://github.com/HVision-NKU/StoryDiffusion
"""
from __future__ import annotations

import base64
import gc
import random
from pathlib import Path

from app.core.config import settings
from app.services import storydiffusion_engine as sd

_pipe = None
_device = None


def _storydiffusion_root() -> Path:
    root = Path(settings.STORYDIFFUSION_DIR)
    if not root.is_dir():
        raise RuntimeError(
            f"StoryDiffusion не найден: {root}. "
            "Выполните: bash scripts/setup_storydiffusion.sh"
        )
    return root


def _get_device():
    from app.services.cuda_compat import resolve_torch_device

    device = resolve_torch_device("cuda")
    if device == "cpu":
        raise RuntimeError(
            "StoryDiffusion требует CUDA. Переустановите torch cu124 для Tesla V100."
        )
    return device


def _get_pipeline():
    global _pipe, _device
    if _pipe is not None:
        return _pipe, _device

    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline

    _storydiffusion_root()
    _device = _get_device()
    dtype = torch.float16

    model_id = settings.STORYDIFFUSION_MODEL_ID
    print(f"[StoryDiffusion] Loading {model_id} on {_device}...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(_device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
    except Exception:
        pass
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _pipe = pipe
    return _pipe, _device


def _char_lookup(nsm: dict) -> dict[str, str]:
    return {
        c.get("name", ""): (c.get("canonical_appearance") or c.get("appearance") or "")
        for c in nsm.get("characters", [])
        if c.get("name")
    }


def _global_character_description(nsm: dict) -> str:
    parts = [v for v in _char_lookup(nsm).values() if v]
    return ", ".join(parts) if parts else "person in a dream"


def _build_prompts(nsm: dict) -> list[tuple[int, str]]:
    """(scene_id, text prompt) в порядке сцен."""
    char_desc = _global_character_description(nsm)
    lookup = _char_lookup(nsm)
    ordered = sorted(nsm.get("scenes", []), key=lambda s: int(s.get("scene_id", 0)))
    out: list[tuple[int, str]] = []

    for scene in ordered:
        scene_id = int(scene["scene_id"])
        body = scene.get("sdxl_prompt") or scene.get("description") or scene.get("location", "")
        scene_chars = scene.get("characters") or []
        char_bits = [lookup[n] for n in scene_chars if n in lookup and lookup[n]]
        if char_bits:
            prefix = ", ".join(char_bits)
            prompt = f"{prefix}, {body}" if body else prefix
        else:
            prompt = f"{char_desc}, {body}" if body else char_desc
        out.append((scene_id, prompt.strip().strip(",")))

    return out


def run_storydiffusion_generation(
    job_id: str,
    nsm_result: dict,
    cim_result: dict,
    cfg_scale: float,
    ddim_steps: int,
) -> list[dict]:
    """Генерирует scene_*.png с consistent self-attention StoryDiffusion."""
    import torch

    _storydiffusion_root()
    pipe, device = _get_pipeline()

    scene_prompts = _build_prompts(nsm_result)
    if not scene_prompts:
        raise ValueError("NSM не содержит сцен для генерации")
    if len(scene_prompts) < 3:
        raise ValueError(
            "StoryDiffusion требует минимум 3 сцены (consistent self-attention). "
            f"Сейчас: {len(scene_prompts)}."
        )

    id_length = min(settings.STORYDIFFUSION_ID_LENGTH, len(scene_prompts))
    id_length = max(3, id_length)

    sd.height = settings.IMAGE_SIZE
    sd.width = settings.IMAGE_SIZE
    sd.sa32 = settings.STORYDIFFUSION_SA32
    sd.sa64 = settings.STORYDIFFUSION_SA64

    style = settings.STORYDIFFUSION_STYLE
    neg_base = "blurry, low quality, distorted, text, watermark, bad anatomy, bad hands"
    raw_texts = [p for _, p in scene_prompts]
    id_texts = raw_texts[:id_length]
    real_texts = raw_texts[id_length:]

    id_texts, negative_prompt = sd.apply_style(style, id_texts, neg_base)

    seed = settings.DDIM_SEED_BASE
    sd.setup_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    sd.write = True
    sd.cur_step = 0
    sd.attn_count = 0
    sd.set_attention_processor(pipe.unet, id_length, is_ipadapter=False)

    print(
        f"[StoryDiffusion] job={job_id} scenes={len(scene_prompts)} "
        f"id_length={id_length} style={style}"
    )

    with torch.inference_mode():
        id_images = pipe(
            id_texts,
            num_inference_steps=ddim_steps,
            guidance_scale=cfg_scale,
            height=sd.height,
            width=sd.width,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images

    indexed_images: list[tuple[int, object]] = list(
        zip([sid for sid, _ in scene_prompts[:id_length]], id_images)
    )

    sd.write = False
    for scene_id, prompt_text in scene_prompts[id_length:]:
        sd.cur_step = 0
        styled = sd.apply_style_positive(style, prompt_text)
        img = pipe(
            styled,
            num_inference_steps=ddim_steps,
            guidance_scale=cfg_scale,
            height=sd.height,
            width=sd.width,
            negative_prompt=negative_prompt,
            generator=generator,
        ).images[0]
        indexed_images.append((scene_id, img))

    output_dir = Path(settings.OUTPUT_DIR) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_by_id = dict(scene_prompts)
    emotion_by_id = {
        int(s["scene_id"]): s.get("emotion", "")
        for s in nsm_result.get("scenes", [])
    }

    scenes_out = []
    for scene_id, image in sorted(indexed_images, key=lambda x: x[0]):
        img_path = output_dir / f"scene_{scene_id}.png"
        image.save(img_path)
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        scenes_out.append(
            {
                "scene_id": scene_id,
                "prompt": prompt_by_id.get(scene_id, ""),
                "emotion": emotion_by_id.get(scene_id, ""),
                "image_base64": b64,
                "image_url": f"/api/generate/image/{job_id}/{scene_id}",
            }
        )
        print(f"[StoryDiffusion] saved scene_{scene_id}.png")

    torch.cuda.empty_cache()
    gc.collect()
    print(f"[StoryDiffusion] Job {job_id} done — {len(scenes_out)} scenes")
    return scenes_out


def generation_status_hint() -> str:
    try:
        import torch

        from app.services.cuda_compat import torch_cuda_kernel_usable

        if not torch.cuda.is_available():
            return "CUDA не найдена — StoryDiffusion требует GPU"
        if not torch_cuda_kernel_usable():
            return "PyTorch без ядер для GPU — установите torch cu124"
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        model = settings.STORYDIFFUSION_MODEL_ID.split("/")[-1]
        return f"StoryDiffusion ({model}) — {name} ({vram:.1f} GB)"
    except ImportError:
        return "Установите requirements-gpu.txt + bash scripts/setup_storydiffusion.sh"
