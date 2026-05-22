from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # LLM для NSM (local = без VPN на GPU-сервере)
    LLM_PROVIDER: str = "local"  # local | openai | groq
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "http://127.0.0.1:11434/v1"  # Ollama
    LLM_MODEL: str = "llama3.2"  # для openai/groq
    LLM_MODEL_ID: str = "Qwen/Qwen3.5-4B"  # для local (Qwen 3.5, text+vision)
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: float = 120.0
    LLM_JSON_MODE: bool = True  # openai-compatible JSON mode

    # Legacy Groq (если LLM_PROVIDER=groq)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # SDXL на локальном GPU
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
    IMAGE_SIZE: int = 512

    OUTPUT_DIR: str = "outputs"
    MAX_CONCURRENT_JOBS: int = 2

    class Config:
        env_file = ".env"

    def llm_configured(self) -> bool:
        if self.LLM_PROVIDER.lower() == "local":
            return True
        key = self.LLM_API_KEY or self.GROQ_API_KEY
        return bool(key) and not key.startswith("gsk_YOUR") and key != "your_groq_api_key"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
