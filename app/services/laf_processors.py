"""
Layered Attention Fusion (LAF) — гибрид CSA (StoryDiffusion) + CCA (CIM).

Практика: полный CSA StoryDiffusion на всех up_blocks (главный источник
согласованности). CCA в U-Net — только с ENABLE_LAF_CCA и низким μ (нужны
эталонные фото + IP-Adapter; текстовые CLIP-эмбеддинги без обучения вредят).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor

from app.services.storydiffusion_engine import SpatialAttnProcessor2_0


@dataclass
class LAFContext:
    char_embeddings: Optional[torch.Tensor] = None
    scene_char_indices: list[list[int]] = field(default_factory=list)
    num_scenes_batch: int = 1
    mu_cca: float = 0.2


laf_ctx = LAFContext()


def set_laf_batch_context(scene_char_indices: list[list[int]], num_scenes: int) -> None:
    laf_ctx.scene_char_indices = scene_char_indices
    laf_ctx.num_scenes_batch = max(1, num_scenes)


def _layer_zone(name: str) -> str:
    if any(x in name for x in ("down_blocks.0", "mid_block", "up_blocks.3")):
        return "semantic"
    if any(x in name for x in ("down_blocks.1", "up_blocks.2")):
        return "mid"
    if any(x in name for x in ("down_blocks.2", "up_blocks.0", "up_blocks.1")):
        return "texture"
    return "other"


def _scene_idx_for_batch_row(batch_idx: int, batch_size: int, num_scenes: int) -> int:
    if num_scenes <= 0:
        return 0
    if batch_size >= 2 * num_scenes:
        return batch_idx % num_scenes
    if batch_size == num_scenes:
        return min(batch_idx, num_scenes - 1)
    return min(batch_idx // max(1, batch_size // num_scenes), num_scenes - 1)


class CCAProcessor(torch.nn.Module):
    """Character Cross-Attention — только с ENABLE_LAF_CCA и маленьким μ."""

    def __init__(self, mu: float = 0.2):
        super().__init__()
        self.mu = mu
        self.base = AttnProcessor()
        self._proj_in: Optional[int] = None
        self._proj_out: Optional[int] = None
        self.proj: Optional[torch.nn.Linear] = None

    def _init_proj(self, in_dim: int, out_dim: int, device, dtype) -> None:
        if self.proj is not None and self._proj_in == in_dim and self._proj_out == out_dim:
            return
        gen = torch.Generator(device="cpu").manual_seed(42)
        self.proj = torch.nn.Linear(in_dim, out_dim, bias=False).to(device=device, dtype=dtype)
        torch.nn.init.orthogonal_(self.proj.weight, generator=gen)
        self.proj.weight.mul_(0.1)
        self._proj_in = in_dim
        self._proj_out = out_dim

    def _active_char_emb(
        self, hidden_states: torch.Tensor, batch_idx: int, batch_size: int
    ) -> Optional[torch.Tensor]:
        emb = laf_ctx.char_embeddings
        if emb is None or emb.numel() == 0:
            return None

        scene_idx = _scene_idx_for_batch_row(batch_idx, batch_size, laf_ctx.num_scenes_batch)
        if scene_idx >= len(laf_ctx.scene_char_indices):
            return None
        indices = laf_ctx.scene_char_indices[scene_idx]
        if not indices:
            return None
        return emb[indices].to(hidden_states.device, dtype=hidden_states.dtype)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        **kwargs,
    ):
        h_text = self.base(
            attn, hidden_states, encoder_hidden_states, attention_mask, temb
        )

        emb = laf_ctx.char_embeddings
        if emb is None or emb.numel() == 0:
            return h_text

        bsz, _, dim = h_text.shape
        h_cca = torch.zeros_like(h_text)
        any_active = False

        for b in range(bsz):
            char_e = self._active_char_emb(h_text, b, bsz)
            if char_e is None or char_e.shape[0] == 0:
                continue
            any_active = True
            self._init_proj(char_e.shape[-1], dim, h_text.device, h_text.dtype)
            assert self.proj is not None
            char_proj = self.proj(char_e)
            q = h_text[b : b + 1]
            scale = dim**-0.5
            attn_w = torch.softmax(q @ char_proj.transpose(-2, -1) * scale, dim=-1)
            h_cca[b : b + 1] = attn_w @ char_proj.unsqueeze(0)

        if any_active:
            h_text = h_text + self.mu * h_cca
        return h_text


def set_laf_attention_processors(
    unet,
    id_length: int,
    char_embeddings: Optional[torch.Tensor],
    mu_cca: float,
    enable_cca: bool = False,
) -> None:
    """
    CSA: все up_blocks attn1 — как оригинальный StoryDiffusion (не урезать!).
    CCA: опционально на attn2 mid/texture.
    """
    laf_ctx.char_embeddings = char_embeddings if enable_cca else None
    laf_ctx.mu_cca = mu_cca

    attn_procs = {}
    csa_count = cca_count = 0

    for name in unet.attn_processors.keys():
        is_self = name.endswith("attn1.processor")
        is_cross = name.endswith("attn2.processor")
        zone = _layer_zone(name)

        if is_self and name.startswith("up_blocks"):
            attn_procs[name] = SpatialAttnProcessor2_0(
                id_length=id_length,
                device=unet.device,
                dtype=unet.dtype,
            )
            csa_count += 1
        elif is_self:
            attn_procs[name] = AttnProcessor()
        elif enable_cca and is_cross and zone in ("mid", "texture"):
            attn_procs[name] = CCAProcessor(mu=mu_cca)
            cca_count += 1
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    mode = f"CSA={csa_count} (all up_blocks)"
    if enable_cca:
        mode += f", CCA={cca_count} μ={mu_cca}"
    else:
        mode += ", CCA=off"
    print(f"[LAF] {mode}")
