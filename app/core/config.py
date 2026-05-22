from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # Groq API (NSM — LLaMA 3.1 8B)
    GROQ_API_KEY: str = "gsk_YOUR_GROQ_KEY_HERE"
    GROQ_MODEL: str = "llama-3.1-8b-instant"
    GROQ_TEMPERATURE: float = 0.3
    GROQ_MAX_TOKENS: int = 4096

    # SDXL на локальном GPU (Student GPU Container)
    SDXL_MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # Pipeline defaults
    DEFAULT_SCENES: int = 4
    DEFAULT_CFG: float = 7.5
    DEFAULT_STEPS: int = 30
    DDIM_SEED_BASE: int = 1
    CLIP_THRESHOLD: float = 0.80
    ARCFACE_THRESHOLD: float = 0.85
    LAF_LAMBDA_CSA: float = 0.4
    LAF_MU_CCA: float = 0.6
    IMAGE_SIZE: int = 1024

    # Output
    OUTPUT_DIR: str = "outputs"
    MAX_CONCURRENT_JOBS: int = 2

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()

settings = get_settings()
