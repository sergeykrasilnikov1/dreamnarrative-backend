from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache

_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    # LLM для NSM (local = без VPN на GPU-сервере)
    LLM_PROVIDER: str = "local"  # local | openrouter | openai | groq
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = "http://127.0.0.1:11434/v1"  # Ollama / generic OpenAI API
    LLM_MODEL: str = "llama3.2"  # для openai/groq
    LLM_MODEL_ID: str = "Qwen/Qwen3.5-4B"  # для local (Qwen 3.5)
    LLM_DEVICE: str = "auto"  # auto | cuda | cpu (только local)

    # OpenRouter — DeepSeek V4 Flash и др.
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    OPENROUTER_MODEL: str = "deepseek/deepseek-v4-flash"
    OPENROUTER_HTTP_REFERER: str = "https://dreamnarrative.local"
    OPENROUTER_APP_TITLE: str = "DreamNarrative"

    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 4096
    LLM_TIMEOUT: float = 120.0
    LLM_JSON_MODE: bool = True  # openai-compatible JSON mode

    # Legacy Groq (если LLM_PROVIDER=groq)
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.1-8b-instant"

    # StoryDiffusion (https://github.com/HVision-NKU/StoryDiffusion)
    STORYDIFFUSION_DIR: str = str(_ENV_FILE.parent / "third_party" / "StoryDiffusion")
    STORYDIFFUSION_MODEL_ID: str = "SG161222/RealVisXL_V4.0"
    STORYDIFFUSION_STYLE: str = "Photographic"
    STORYDIFFUSION_ID_LENGTH: int = 3  # legacy min; батч = все сцены (см. run_storydiffusion_generation)
    STORYDIFFUSION_SA32: float = 0.5
    STORYDIFFUSION_SA64: float = 0.5
    STORYDIFFUSION_STYLES: list[str] = [
        "(No style)",
        "Japanese Anime",
        "Digital/Oil Painting",
        "Pixar/Disney Character",
        "Photographic",
        "Comic book",
        "Line art",
        "Black and White Film Noir",
        "Isometric Rooms",
    ]
    STORYDIFFUSION_IMAGE_SIZES: list[int] = [512, 768, 1024]

    # Legacy alias (не используется напрямую)
    SDXL_MODEL_ID: str = "SG161222/RealVisXL_V4.0"

    # Pipeline defaults
    NSM_MIN_SCENES: int = 3
    NSM_MAX_SCENES: int = 8
    DEFAULT_SCENES: int = 4  # legacy, LLM выбирает сам
    DEFAULT_CFG: float = 7.5
    DEFAULT_STEPS: int = 30
    DDIM_SEED_BASE: int = 1
    CLIP_THRESHOLD: float = 0.80
    ARCFACE_THRESHOLD: float = 0.85
    LAF_LAMBDA_CSA: float = 0.4
    LAF_MU_CCA: float = 0.2  # низкий μ: CCA без IP-Adapter иначе ломает лица
    ENABLE_LAF: bool = True  # обогащение промптов canonical_appearance из CIM
    ENABLE_LAF_CCA: bool = False  # UNet CCA — только с эталонными фото / IP-Adapter

    # CIM — CLIP identity encoder
    CIM_CLIP_MODEL_ID: str = "openai/clip-vit-large-patch14"
    CIM_DEVICE: str = "auto"  # auto | cuda | cpu
    IMAGE_SIZE: int = 512

    OUTPUT_DIR: str = "outputs"
    MAX_CONCURRENT_JOBS: int = 2

    class Config:
        env_file = ".env"

    def llm_model_label(self) -> str:
        p = self.LLM_PROVIDER.lower()
        if p == "local":
            return self.LLM_MODEL_ID
        if p == "openrouter":
            return self.OPENROUTER_MODEL
        return self.LLM_MODEL

    def llm_configured(self) -> bool:
        p = self.LLM_PROVIDER.lower()
        if p == "local":
            return True
        if p == "openrouter":
            key = self.OPENROUTER_API_KEY or self.LLM_API_KEY
            return bool(key) and key not in ("", "your_openrouter_api_key", "sk-or-YOUR")
        key = self.LLM_API_KEY or self.GROQ_API_KEY
        return bool(key) and not key.startswith("gsk_YOUR") and key != "your_groq_api_key"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
