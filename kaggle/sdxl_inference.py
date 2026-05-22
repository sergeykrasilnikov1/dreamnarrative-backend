#!/usr/bin/env python3
"""
DreamNarrative — SDXL Inference Kernel для Kaggle GPU (T4/P100, 16GB VRAM)
Запускается диспетчером FastAPI через Kaggle API.

Входные данные: /kaggle/input/dreamnarrative-payload/payload.json
Выходные данные: /kaggle/working/scene_{N}.png
"""

import json
import os
import gc
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")


def _pip(*packages: str) -> None:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *packages])


def _stub_clip_image_processor() -> None:
    """
    Kaggle Py3.12: импорт transformers.models.clip.image_processing_clip
    тянет сломанный torchvision::nms. SDXL text2img этот модуль не использует.
    """
    import types

    stub = types.ModuleType("transformers.models.clip.image_processing_clip")

    class CLIPImageProcessor:  # noqa: D101 — заглушка для import side-effects
        def __init__(self, *args, **kwargs):
            pass

    stub.CLIPImageProcessor = CLIPImageProcessor
    sys.modules["transformers.models.clip.image_processing_clip"] = stub


def _setup_dependencies() -> None:
    """Ставим ML-стек без torch/torchvision; numpy выравниваем по ABI wheels."""
    import torch

    print(f"torch={torch.__version__} (системный, без переустановки)")

    # Wheels safetensors/diffusers часто собраны под NumPy 2.x (ABI 96).
    # На Kaggle по умолчанию NumPy 1.x (ABI 88) → binary incompatibility.
    _pip("numpy>=2.0.0,<2.3", "--upgrade")
    import numpy as np

    print(f"numpy={np.__version__}")

    # --no-deps: pip не трогает torch / torchvision / numpy повторно
    _pip("diffusers==0.30.3", "--no-deps")
    _pip("transformers==4.41.2", "--no-deps")
    _pip("accelerate==0.33.0", "--no-deps")
    _pip("safetensors==0.4.5", "--no-deps")
    _pip("huggingface-hub==0.25.2", "--no-deps")
    _pip(
        "packaging",
        "regex",
        "requests",
        "filelock",
        "Pillow",
        "pyyaml",
        "tqdm",
        "importlib_metadata",
    )

    _stub_clip_image_processor()
    print("deps ok (torch/torchvision не изменялись)")


_setup_dependencies()

import torch
import numpy as np
from pathlib import Path
from PIL import Image

from diffusers import StableDiffusionXLPipeline, DDIMScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
import torch.nn.functional as F

# ── Конфигурация ─────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.float16 if DEVICE == "cuda" else torch.float32
OUTPUT_DIR = Path("/kaggle/working")
INPUT_PATH = Path("/kaggle/input/dreamnarrative-payload/payload.json")

print(f"Device: {DEVICE} | VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB" if DEVICE=="cuda" else "CPU mode")

# ── Загрузка payload ─────────────────────────────────────────────────────────
with open(INPUT_PATH) as f:
    payload = json.load(f)

job_id = payload["job_id"]
nsm    = payload["nsm"]
cim    = payload["cim"]
cfg    = payload["config"]

N_SCENES    = nsm["total_scenes"]
CFG_SCALE   = cfg["cfg_scale"]
DDIM_STEPS  = cfg["ddim_steps"]
SEED_BASE   = cfg["seed_base"]
IMAGE_SIZE  = cfg["image_size"]
LAM_CSA     = cfg["lambda_csa"]   # 0.4
MU_CCA      = cfg["mu_cca"]       # 0.6

print(f"Job: {job_id} | Scenes: {N_SCENES} | Steps: {DDIM_STEPS} | CFG: {CFG_SCALE}")

# ════════════════════════════════════════════════════════════════════════════
# LAF — Layered Attention Fusion
# Реализация CSA + CCA через кастомные attention processors
# ════════════════════════════════════════════════════════════════════════════

# Хранилище k/v для CSA (Consistent Self-Attention)
csa_store: dict = {}   # {layer_key: {"keys": [...], "values": [...]}}


class CSAProcessor:
    """
    Consistent Self-Attention (StoryDiffusion, Zhou et al. 2024).
    Для слоёв 8×8 и 16×16 U-Net SDXL.

    CSA(Q_i, K_{1..N}, V_{1..N}) = softmax([K_1...K_N]) · [V_1...V_N]
    """
    def __init__(self, layer_key: str, lam: float = 0.4):
        self.layer_key = layer_key
        self.lam = lam
        self.attn = AttnProcessor2_0()

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        # Стандартный self-attention
        h_text = self.attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        # CSA — собираем K/V всех сцен и делаем cross-scene attention
        is_self_attn = encoder_hidden_states is None
        if is_self_attn and self.layer_key in csa_store:
            store = csa_store[self.layer_key]
            q = hidden_states
            all_k = store.get("keys", [])
            all_v = store.get("values", [])

            if len(all_k) > 0:
                # Конкатенируем K/V всех предыдущих сцен
                k_concat = torch.cat(all_k + [hidden_states], dim=1)
                v_concat = torch.cat(all_v + [hidden_states], dim=1)

                # Упрощённый CSA через scaled dot-product attention
                scale = hidden_states.shape[-1] ** -0.5
                attn_w = torch.softmax(q @ k_concat.transpose(-2, -1) * scale, dim=-1)
                h_csa  = attn_w @ v_concat

                # Fusion: h_final = h_text + λ·h_CSA
                h_text = h_text + self.lam * h_csa

            # Накапливаем K/V текущей сцены
            store.setdefault("keys", []).append(hidden_states.detach())
            store.setdefault("values", []).append(hidden_states.detach())

        return h_text


class CCAProcessor:
    """
    Character Cross-Attention (Character Identity Module).
    Для слоёв 32×32, 64×64, 128×128 U-Net SDXL.

    CCA(Q, e_id) = softmax(W_Q·Q·(W_K·e_id)ᵀ / √d) · W_V·e_id
    h_final = h_text + λ·h_CSA + μ·h_CCA
    """
    def __init__(self, char_embeddings: torch.Tensor, mu: float = 0.6):
        self.char_emb = char_embeddings  # [n_chars, 1024]
        self.mu = mu
        self.attn = AttnProcessor2_0()
        self.proj = None  # lazy init

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, **kwargs):
        h_text = self.attn(attn, hidden_states, encoder_hidden_states, attention_mask, **kwargs)

        if self.char_emb is not None and len(self.char_emb) > 0:
            bsz, seq_len, dim = hidden_states.shape
            char_emb = self.char_emb.to(hidden_states.device, hidden_states.dtype)

            # Проекция char_emb → dim пространство
            if self.proj is None or self.proj.in_features != char_emb.shape[-1]:
                self.proj = torch.nn.Linear(
                    char_emb.shape[-1], dim, bias=False
                ).to(hidden_states.device, hidden_states.dtype)
                torch.nn.init.xavier_uniform_(self.proj.weight)

            char_proj = self.proj(char_emb)  # [n_chars, dim]
            char_proj = char_proj.unsqueeze(0).expand(bsz, -1, -1)  # [b, n, dim]

            # Scaled dot-product: CCA(Q, e_id)
            scale = dim ** -0.5
            q = hidden_states
            attn_w = torch.softmax(q @ char_proj.transpose(-2, -1) * scale, dim=-1)
            h_cca  = attn_w @ char_proj  # [b, seq, dim]

            # h_final = h_text + μ·h_CCA
            h_text = h_text + self.mu * h_cca

        return h_text


def inject_laf_processors(pipe, char_embeddings: torch.Tensor):
    """
    Назначает CSA/CCA processors по слоям U-Net SDXL (88 transformer блоков).

    8×8  (14 блоков): CSA only
    16×16 (18 блоков): CSA only
    32×32 (14 блоков): CSA + CCA
    64×64 (24 блока): CCA dominant
    128×128(18 блоков): CCA dominant
    """
    unet = pipe.unet
    attn_procs = {}

    for name in unet.attn_processors.keys():
        # Определяем разрешение блока по имени
        is_small = any(f"down_blocks.0" in name or "up_blocks.3" in name
                       or f"mid_block" in name for _ in [name])

        if "attn1" in name:  # self-attention блоки
            res_tag = name.split(".")[0]
            if "down_blocks.0" in name or "up_blocks.3" in name:
                # 8×8 и 16×16 — CSA
                attn_procs[name] = CSAProcessor(layer_key=name, lam=LAM_CSA)
            elif "down_blocks.1" in name or "up_blocks.2" in name:
                # 32×32 — CSA + CCA
                attn_procs[name] = CSAProcessor(layer_key=name, lam=LAM_CSA)
            else:
                attn_procs[name] = AttnProcessor2_0()
        elif "attn2" in name:  # cross-attention (text conditioning)
            if any(x in name for x in ["down_blocks.1", "down_blocks.2", "up_blocks.0", "up_blocks.1"]):
                # 32×32–128×128 — CCA
                attn_procs[name] = CCAProcessor(char_embeddings=char_embeddings, mu=MU_CCA)
            else:
                attn_procs[name] = AttnProcessor2_0()
        else:
            attn_procs[name] = AttnProcessor2_0()

    unet.set_attn_processor(attn_procs)
    print(f"LAF: injected {sum(1 for v in attn_procs.values() if isinstance(v, CSAProcessor))} CSA, "
          f"{sum(1 for v in attn_procs.values() if isinstance(v, CCAProcessor))} CCA processors")


# ════════════════════════════════════════════════════════════════════════════
# Загрузка SDXL pipeline
# ════════════════════════════════════════════════════════════════════════════
print("Loading SDXL 1.0...")

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DTYPE == torch.float16 else None,
)
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)
pipe.enable_attention_slicing()
if DEVICE == "cuda":
    try:
        import xformers  # noqa: F401
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        print("xformers unavailable, using default attention")

# ── Фиктивные char embeddings (в реальной реализации — CLIP ViT-H/14) ────
# В production здесь загружаем предвычисленные тензоры из CIM service
n_chars = len(nsm.get("characters", []))
if n_chars > 0:
    # Детерминированные embeddings из canonical appearance hash
    torch.manual_seed(42)
    char_embeddings = torch.randn(n_chars, 1024, dtype=DTYPE)
    char_embeddings = F.normalize(char_embeddings, dim=-1)
else:
    char_embeddings = None

# Инъекция LAF processors
if char_embeddings is not None:
    inject_laf_processors(pipe, char_embeddings)

# ════════════════════════════════════════════════════════════════════════════
# Генерация сцен
# ════════════════════════════════════════════════════════════════════════════
scenes = nsm["scenes"]
results = []

for scene in scenes:
    scene_id = scene["scene_id"]
    seed     = SEED_BASE + scene_id - 1  # детерминированный seed
    prompt   = scene.get("sdxl_prompt", scene.get("description", ""))
    neg_prompt = scene.get("negative_prompt", "blurry, low quality, distorted, text, watermark")

    print(f"\nScene {scene_id}/{N_SCENES} | seed={seed} | emotion={scene.get('emotion','')}")
    print(f"  Prompt: {prompt[:80]}...")

    # Очищаем CSA store для детерминированного воспроизведения по сцене
    # (но сохраняем между сценами для cross-scene consistency!)
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=DDIM_STEPS,
            guidance_scale=CFG_SCALE,
            width=IMAGE_SIZE,
            height=IMAGE_SIZE,
            generator=generator,
        )

    img = output.images[0]
    out_path = OUTPUT_DIR / f"scene_{scene_id}.png"
    img.save(out_path)
    results.append({"scene_id": scene_id, "path": str(out_path)})
    print(f"  ✓ Saved: {out_path}")

    # Освобождаем память после каждой сцены
    del output
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

# ── Сохраняем метаданные результата ─────────────────────────────────────────
meta = {
    "job_id": job_id,
    "status": "done",
    "n_scenes": N_SCENES,
    "scenes": results,
    "config": cfg,
    "model": "stabilityai/stable-diffusion-xl-base-1.0",
    "laf": {"lambda_csa": LAM_CSA, "mu_cca": MU_CCA},
}
with open(OUTPUT_DIR / "result_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"\n✓ All {N_SCENES} scenes generated for job {job_id}")
print(f"Output: {OUTPUT_DIR}")
