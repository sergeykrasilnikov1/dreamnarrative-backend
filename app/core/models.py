from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class PipelineRequest(BaseModel):
    dream_text: str = Field(..., min_length=20, max_length=5000, description="Текст описания сна")
    n_scenes: int = Field(default=4, ge=3, le=8)
    cfg_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    ddim_steps: int = Field(default=30, ge=10, le=50)
    seed_base: int = Field(default=1)


class CharacterSchema(BaseModel):
    name: str
    appearance: str
    canonical_appearance: str
    scenes_present: list[int]


class SceneSchema(BaseModel):
    scene_id: int
    description: str
    characters: list[str]
    objects: list[str]
    location: str
    emotion: str
    sdxl_prompt: str
    negative_prompt: str


class NSMResponse(BaseModel):
    job_id: str
    total_scenes: int
    characters: list[CharacterSchema]
    scenes: list[SceneSchema]
    clip_threshold: float
    arcface_threshold: float
    llm_model: str
    processing_time_ms: int


class EmbeddingSchema(BaseModel):
    character_name: str
    encoder: str = "CLIP ViT-H/14"
    embedding_dim: int = 1024
    arcface_dim: int = 512
    face_similarity_score: float
    mlp_projection: str = "1024 → 768"


class CIMResponse(BaseModel):
    job_id: str
    characters_processed: int
    embeddings: list[EmbeddingSchema]
    cca_lambda: float
    cca_mu: float


class GenerationStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    ERROR = "error"


class SceneResult(BaseModel):
    scene_id: int
    image_url: Optional[str] = None
    image_base64: Optional[str] = None
    prompt: str
    emotion: str
    seed: int
    status: GenerationStatus


class GenerateResponse(BaseModel):
    job_id: str
    status: GenerationStatus
    scenes: list[SceneResult]
    metrics: Optional[dict] = None
    gpu_progress_hint: Optional[str] = None
    error_message: Optional[str] = None


class PipelineStatus(BaseModel):
    job_id: str
    stage: str
    progress: int
    message: str
    result: Optional[dict] = None
