"""
CIM — Character Identity Module (ВКР §3.2, §5.5).

1. CLIP-эмбеддинги из canonical_appearance (e_id ∈ R^1024)
2. Реестр для LAF/CCA — какие персонажи в каких сценах
"""
from __future__ import annotations

from app.core.models import CIMResponse, EmbeddingSchema, NSMResponse
from app.core.config import settings
from app.services.cim_encoder import (
    deterministic_fss,
    encode_characters,
    unload_cim_encoder,
)


def run_cim(nsm_result: NSMResponse, job_id: str) -> CIMResponse:
    try:
        char_emb, name_to_idx, serializable = encode_characters(nsm_result.characters)
    finally:
        unload_cim_encoder()

    embeddings: list[EmbeddingSchema] = []
    for i, char in enumerate(nsm_result.characters):
        vec = serializable[i] if i < len(serializable) else []
        embeddings.append(
            EmbeddingSchema(
                character_name=char.name,
                encoder=settings.CIM_CLIP_MODEL_ID,
                embedding_dim=len(vec) if vec else 1024,
                arcface_dim=512,
                face_similarity_score=deterministic_fss(char.name, char.canonical_appearance),
                mlp_projection="CLIP text → 1024d (IP-Adapter-Plus compatible)",
                embedding=vec,
                embedding_preview=vec[:12] if vec else [],
            )
        )

    return CIMResponse(
        job_id=job_id,
        characters_processed=len(embeddings),
        embeddings=embeddings,
        cca_lambda=settings.LAF_LAMBDA_CSA,
        cca_mu=settings.LAF_MU_CCA,
    )


def build_scene_char_indices(
    nsm_result: dict,
    name_to_idx: dict[str, int],
) -> list[list[int]]:
    """Для каждой сцены — индексы персонажей в char_embeddings."""
    ordered = sorted(nsm_result.get("scenes", []), key=lambda s: int(s.get("scene_id", 0)))
    out: list[list[int]] = []
    all_idx = list(range(len(name_to_idx)))

    for scene in ordered:
        indices: list[int] = []
        for name in scene.get("characters") or []:
            if isinstance(name, dict):
                name = name.get("name", "")
            name = str(name).strip()
            if name in name_to_idx and name_to_idx[name] not in indices:
                indices.append(name_to_idx[name])
        out.append(indices if indices else all_idx)
    return out


def load_char_embeddings_from_cim(cim_result: dict) -> tuple[object, dict[str, int]]:
    """Восстанавливает тензор [N, D] и name_to_idx из ответа CIM API."""
    import torch

    embs = cim_result.get("embeddings") or []
    if not embs:
        return torch.zeros(0, 1024), {}

    vectors = []
    name_to_idx: dict[str, int] = {}
    for i, e in enumerate(embs):
        name = e.get("character_name") or e.get("name") or f"char_{i}"
        name_to_idx[name] = i
        vec = e.get("embedding") or []
        if not vec:
            continue
        vectors.append(vec)

    if not vectors:
        return torch.zeros(0, 1024), name_to_idx

    tensor = torch.tensor(vectors, dtype=torch.float32)
    return tensor, name_to_idx
