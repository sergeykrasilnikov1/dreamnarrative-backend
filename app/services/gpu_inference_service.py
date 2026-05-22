"""
SDXL + LAF inference на локальном GPU (Student GPU Container / свой сервер).
"""
from __future__ import annotations

import base64
import gc
from pathlib import Path

from app.core.config import settings

# Глобальный кэш pipeline (дорогая загрузка — один раз на процесс)
_pipe = None
_device = None


def _get_device():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError(
            "CUDA недоступна. Запускайте backend на Student GPU Container "
            "(JupyterLab / VSCode с GPU)."
        )
    return device, torch.float16


def _build_char_embeddings(nsm: dict, dtype, device):
    import torch
    import torch.nn.functional as F

    chars = nsm.get("characters", [])
    if not chars:
        return None

    torch.manual_seed(42)
    emb = torch.randn(len(chars), 1024, device=device, dtype=dtype)
    return F.normalize(emb, dim=-1)


def _inject_laf_processors(pipe, char_embeddings, lam_csa: float, mu_cca: float):
    import torch
    from diffusers.models.attention_processor import AttnProcessor2_0

    csa_store: dict = {}

    class CSAProcessor:
        def __init__(self, layer_key: str, lam: float):
            self.layer_key = layer_key
            self.lam = lam
            self.attn = AttnProcessor2_0()

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, **kwargs):
            h_text = self.attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            if encoder_hidden_states is None and self.layer_key in csa_store:
                store = csa_store[self.layer_key]
                all_k = store.get("keys", [])
                all_v = store.get("values", [])
                if all_k:
                    k_concat = torch.cat(all_k + [hidden_states], dim=1)
                    v_concat = torch.cat(all_v + [hidden_states], dim=1)
                    scale = hidden_states.shape[-1] ** -0.5
                    attn_w = torch.softmax(hidden_states @ k_concat.transpose(-2, -1) * scale, dim=-1)
                    h_text = h_text + self.lam * (attn_w @ v_concat)
                store.setdefault("keys", []).append(hidden_states.detach())
                store.setdefault("values", []).append(hidden_states.detach())
            return h_text

    class CCAProcessor:
        def __init__(self, char_emb, mu: float):
            self.char_emb = char_emb
            self.mu = mu
            self.attn = AttnProcessor2_0()
            self.proj = None

        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, **kwargs):
            import torch

            h_text = self.attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)
            if self.char_emb is not None and len(self.char_emb) > 0:
                bsz, _, dim = hidden_states.shape
                char_emb = self.char_emb.to(hidden_states.device, hidden_states.dtype)
                if self.proj is None or self.proj.in_features != char_emb.shape[-1]:
                    self.proj = torch.nn.Linear(char_emb.shape[-1], dim, bias=False).to(
                        hidden_states.device, hidden_states.dtype
                    )
                    torch.nn.init.xavier_uniform_(self.proj.weight)
                char_proj = self.proj(char_emb).unsqueeze(0).expand(bsz, -1, -1)
                scale = dim ** -0.5
                attn_w = torch.softmax(hidden_states @ char_proj.transpose(-2, -1) * scale, dim=-1)
                h_text = h_text + self.mu * (attn_w @ char_proj)
            return h_text

    attn_procs = {}
    for name in pipe.unet.attn_processors.keys():
        if "attn1" in name and ("down_blocks.0" in name or "up_blocks.3" in name or
                                "down_blocks.1" in name or "up_blocks.2" in name):
            attn_procs[name] = CSAProcessor(name, lam_csa)
        elif "attn2" in name and any(
            x in name for x in ("down_blocks.1", "down_blocks.2", "up_blocks.0", "up_blocks.1")
        ):
            attn_procs[name] = CCAProcessor(char_embeddings, mu_cca)
        else:
            attn_procs[name] = AttnProcessor2_0()
    pipe.unet.set_attn_processor(attn_procs)


def _get_pipeline():
    global _pipe, _device
    if _pipe is not None:
        return _pipe, _device

    import torch
    from diffusers import DDIMScheduler, StableDiffusionXLPipeline

    _device, dtype = _get_device()
    print(f"[GPU] Loading SDXL on {_device}...")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        settings.SDXL_MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(_device)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    _pipe = pipe
    return _pipe, _device


def run_sdxl_generation(
    job_id: str,
    nsm_result: dict,
    cim_result: dict,
    cfg_scale: float,
    ddim_steps: int,
) -> list[dict]:
    """Генерирует scene_*.png в OUTPUT_DIR/{job_id}/, возвращает метаданные сцен."""
    import torch

    pipe, device = _get_pipeline()
    dtype = torch.float16

    output_dir = Path(settings.OUTPUT_DIR) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    nsm = nsm_result
    cfg = {
        "cfg_scale": cfg_scale,
        "ddim_steps": ddim_steps,
        "seed_base": settings.DDIM_SEED_BASE,
        "image_size": settings.IMAGE_SIZE,
        "lambda_csa": settings.LAF_LAMBDA_CSA,
        "mu_cca": settings.LAF_MU_CCA,
    }

    char_emb = _build_char_embeddings(nsm, dtype, device)
    if char_emb is not None:
        _inject_laf_processors(pipe, char_emb, cfg["lambda_csa"], cfg["mu_cca"])

    scenes_out = []
    scenes = nsm.get("scenes", [])
    total = len(scenes)

    for i, scene in enumerate(scenes, start=1):
        scene_id = scene["scene_id"]
        seed = cfg["seed_base"] + scene_id - 1
        prompt = scene.get("sdxl_prompt", scene.get("description", ""))
        neg = scene.get("negative_prompt", "blurry, low quality, distorted, text, watermark")

        print(f"[GPU] Scene {i}/{total} id={scene_id} seed={seed}")

        generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=neg,
                num_inference_steps=cfg["ddim_steps"],
                guidance_scale=cfg["cfg_scale"],
                width=cfg["image_size"],
                height=cfg["image_size"],
                generator=generator,
            )

        img_path = output_dir / f"scene_{scene_id}.png"
        result.images[0].save(img_path)

        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        scenes_out.append({
            "scene_id": scene_id,
            "prompt": prompt,
            "emotion": scene.get("emotion", ""),
            "image_base64": b64,
            "image_url": f"/api/generate/image/{job_id}/{scene_id}",
        })

        del result
        torch.cuda.empty_cache()
        gc.collect()

    print(f"[GPU] Job {job_id} done — {len(scenes_out)} scenes")
    return scenes_out


def gpu_status() -> dict:
    """Статус GPU для polling API."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {
                "gpu_available": False,
                "progress_hint": "CUDA не найдена — запустите на GPU-контейнере",
            }
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "gpu_available": True,
            "progress_hint": f"GPU: {name} ({vram:.1f} GB VRAM)",
            "device_name": name,
        }
    except ImportError:
        return {
            "gpu_available": False,
            "progress_hint": "Установите requirements-gpu.txt (torch, diffusers)",
        }
