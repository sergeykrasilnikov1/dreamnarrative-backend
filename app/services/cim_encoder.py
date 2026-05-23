"""
CIM — извлечение векторов идентичности персонажей (CLIP text encoder).

По ВКР: CLIP ViT-H/14 → e_id ∈ R^1024. Локально используем CLIP ViT-L/14 (768d)
с проекцией до 1024 — без эталонных портретов (этап 1 CIM опционален).
"""
from __future__ import annotations

import hashlib

import torch
import torch.nn.functional as F

from app.core.config import settings

_clip_model = None
_clip_tokenizer = None
_proj_768_1024 = None
_device = None


def _resolve_device() -> str:
    if settings.CIM_DEVICE == "cpu":
        return "cpu"
    try:
        from app.services.cuda_compat import resolve_torch_device

        return resolve_torch_device("cuda")
    except Exception:
        return "cpu"


def _load_clip():
    global _clip_model, _clip_tokenizer, _proj_768_1024, _device
    if _clip_model is not None:
        return _clip_model, _clip_tokenizer, _proj_768_1024, _device

    from transformers import CLIPModel, CLIPTokenizer

    model_id = settings.CIM_CLIP_MODEL_ID
    _device = _resolve_device()
    print(f"[CIM] Loading CLIP encoder {model_id} on {_device}...")

    _clip_tokenizer = CLIPTokenizer.from_pretrained(model_id)
    _clip_model = CLIPModel.from_pretrained(model_id).to(_device)
    _clip_model.eval()

    hidden = _clip_model.config.projection_dim
    if hidden != 1024:
        gen = torch.Generator().manual_seed(42)
        _proj_768_1024 = torch.nn.Linear(hidden, 1024, bias=False)
        torch.nn.init.normal_(_proj_768_1024.weight, std=0.02, generator=gen)
        _proj_768_1024 = _proj_768_1024.to(_device)
    else:
        _proj_768_1024 = None

    return _clip_model, _clip_tokenizer, _proj_768_1024, _device


def _appearance_prompt(name: str, appearance: str) -> str:
    text = (appearance or name or "person").strip()
    return (
        f"professional portrait photo of {text}, "
        "neutral background, detailed face and clothing, studio lighting"
    )


def _clip_text_features(model, inputs: dict) -> torch.Tensor:
    """Стабильное извлечение text features для разных версий transformers."""
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")

    raw = model.get_text_features(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    if isinstance(raw, torch.Tensor):
        return raw
    if hasattr(raw, "text_embeds") and raw.text_embeds is not None:
        return raw.text_embeds
    if hasattr(raw, "pooler_output") and raw.pooler_output is not None:
        return model.text_projection(raw.pooler_output)

    # fallback: явный проход через text_model
    text_outputs = model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    if hasattr(text_outputs, "pooler_output") and text_outputs.pooler_output is not None:
        pooled = text_outputs.pooler_output
    elif hasattr(text_outputs, "last_hidden_state"):
        pooled = text_outputs.last_hidden_state[:, 0, :]
    else:
        pooled = text_outputs[1] if text_outputs[1] is not None else text_outputs[0][:, 0, :]

    return model.text_projection(pooled)


@torch.inference_mode()
def encode_characters(characters: list) -> tuple[torch.Tensor, dict[str, int], list[list[float]]]:
    """
    Возвращает:
      - char_embeddings [N, 1024] float32 CPU tensor (normalized)
      - name_to_idx
      - serializable list of vectors for API
    """
    if not characters:
        empty = torch.zeros(0, 1024)
        return empty, {}, []

    model, tokenizer, proj, device = _load_clip()
    names: list[str] = []
    texts: list[str] = []

    for c in characters:
        name = getattr(c, "name", None) or c.get("name", "unknown")
        appearance = getattr(c, "canonical_appearance", None) or c.get(
            "canonical_appearance", c.get("appearance", name)
        )
        names.append(name)
        texts.append(_appearance_prompt(name, appearance))

    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=77, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items() if k in ("input_ids", "attention_mask")}

    out = _clip_text_features(model, inputs)
    if proj is not None:
        out = proj(out.to(dtype=proj.weight.dtype))
    out = F.normalize(out.float(), dim=-1)

    name_to_idx = {n: i for i, n in enumerate(names)}
    serializable = [v.tolist() for v in out.cpu()]
    return out.cpu(), name_to_idx, serializable


def deterministic_fss(name: str, appearance: str) -> float:
    """Прокси FSS до полноценного ArcFace (детерминированный скор)."""
    h = hashlib.sha256(f"{name}:{appearance}".encode()).hexdigest()
    return round(0.68 + (int(h[:8], 16) % 240) / 1000.0, 3)


def unload_cim_encoder() -> None:
    global _clip_model, _clip_tokenizer, _proj_768_1024, _device
    _clip_model = _clip_tokenizer = _proj_768_1024 = _device = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
