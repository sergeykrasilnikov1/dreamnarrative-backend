"""
Layered Attention Fusion (LAF) — гибрид CSA (StoryDiffusion) + CCA (CIM).

По ВКР (раздел 5.4):
  8×8, 16×16  — CSA only  (семантика)
  32×32       — CSA + CCA (λ, μ)
  64×64, 128×128 — CCA only (текстуры, детали лица)

h_final = h_text + λ·h_CSA + μ·h_CCA  (на промежуточных слоях)
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor

from app.services.storydiffusion_engine import SpatialAttnProcessor2_0


@dataclass
class LAFContext:
    char_embeddings: Optional[torch.Tensor] = None  # [N, D] on CPU
    scene_char_indices: list[list[int]] = field(default_factory=list)
    num_scenes_batch: int = 1
    lam_csa: float = 0.4
    mu_cca: float = 0.6


laf_ctx = LAFContext()


def set_laf_batch_context(scene_char_indices: list[list[int]], num_scenes: int) -> None:
    laf_ctx.scene_char_indices = scene_char_indices
    laf_ctx.num_scenes_batch = max(1, num_scenes)


def _layer_zone(name: str) -> str:
    """semantic | mid | texture | other"""
    if any(x in name for x in ("down_blocks.0", "mid_block", "up_blocks.3")):
        return "semantic"
    if any(x in name for x in ("down_blocks.1", "up_blocks.2")):
        return "mid"
    if any(x in name for x in ("down_blocks.2", "up_blocks.0", "up_blocks.1")):
        return "texture"
    return "other"


def _scene_idx_for_batch_row(batch_idx: int, batch_size: int, num_scenes: int) -> int:
    """SDXL + CFG: batch часто 2×num_scenes (cond/uncond)."""
    if num_scenes <= 0:
        return 0
    if batch_size >= 2 * num_scenes and batch_size % num_scenes == 0:
        return batch_idx % num_scenes
    if batch_size == num_scenes:
        return min(batch_idx, num_scenes - 1)
    return min(batch_idx // max(1, batch_size // num_scenes), num_scenes - 1)


class CCAProcessor(torch.nn.Module):
    """
    Character Cross-Attention (ВКР §3.3, §5.4).
    CCA(Q, e_id) = softmax(Q·K_id^T / √d) · V_id
    """

    def __init__(self, mu: float = 0.6):
        super().__init__()
        self.mu = mu
        self.base = AttnProcessor()
        self.proj: Optional[torch.nn.Linear] = None

    def _active_char_emb(
        self, hidden_states: torch.Tensor, batch_idx: int, batch_size: int
    ) -> Optional[torch.Tensor]:
        emb = laf_ctx.char_embeddings
        if emb is None or emb.numel() == 0:
            return None

        num_scenes = laf_ctx.num_scenes_batch
        scene_idx = _scene_idx_for_batch_row(batch_idx, batch_size, num_scenes)
        indices = (
            laf_ctx.scene_char_indices[scene_idx]
            if scene_idx < len(laf_ctx.scene_char_indices)
            else list(range(emb.shape[0]))
        )
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

        bsz, seq_len, dim = h_text.shape
        h_cca = torch.zeros_like(h_text)
        any_active = False

        for b in range(bsz):
            char_e = self._active_char_emb(h_text, b, bsz)
            if char_e is None or char_e.shape[0] == 0:
                continue
            any_active = True
            if self.proj is None or self.proj.in_features != char_e.shape[-1]:
                self.proj = torch.nn.Linear(
                    char_e.shape[-1], dim, bias=False
                ).to(h_text.device, dtype=h_text.dtype)
                torch.nn.init.xavier_uniform_(self.proj.weight)

            char_proj = self.proj(char_e)  # [n_chars, dim]
            q = h_text[b : b + 1]
            scale = dim**-0.5
            attn_w = torch.softmax(
                q @ char_proj.transpose(-2, -1) * scale, dim=-1
            )
            h_cca[b : b + 1] = attn_w @ char_proj.unsqueeze(0)

        if any_active:
            h_text = h_text + self.mu * h_cca
        return h_text


def set_laf_attention_processors(
    unet,
    id_length: int,
    char_embeddings: Optional[torch.Tensor],
    lam_csa: float,
    mu_cca: float,
) -> None:
    """
    Регистрирует LAF-процессоры на U-Net SDXL.
    CSA: StoryDiffusion SpatialAttnProcessor2_0 на up_blocks.2/3 (семантика + mid).
    CCA: CCAProcessor на attn2 mid + texture блоков.
    """
    laf_ctx.char_embeddings = char_embeddings
    laf_ctx.lam_csa = lam_csa
    laf_ctx.mu_cca = mu_cca

    attn_procs = {}
    csa_count = cca_count = 0

    for name in unet.attn_processors.keys():
        is_self = name.endswith("attn1.processor")
        is_cross = name.endswith("attn2.processor")
        zone = _layer_zone(name)

        if is_self and "up_blocks" in name:
            try:
                block_id = int(name.split("up_blocks.")[1].split(".")[0])
            except (IndexError, ValueError):
                block_id = 0
            # up_blocks.3, .2 — CSA (StoryDiffusion, семантика + mid)
            if block_id >= 2:
                attn_procs[name] = SpatialAttnProcessor2_0(
                    id_length=id_length,
                    device=unet.device,
                    dtype=unet.dtype,
                )
                csa_count += 1
            else:
                attn_procs[name] = AttnProcessor()
        elif is_self:
            attn_procs[name] = AttnProcessor()
        elif is_cross and zone in ("mid", "texture"):
            attn_procs[name] = CCAProcessor(mu=mu_cca)
            cca_count += 1
        else:
            attn_procs[name] = AttnProcessor()

    unet.set_attn_processor(copy.deepcopy(attn_procs))
    print(
        f"[LAF] processors: CSA={csa_count} (StoryDiffusion up_blocks≥2), "
        f"CCA={cca_count} (attn2 mid/texture), λ={lam_csa}, μ={mu_cca}"
    )
