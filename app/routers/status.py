from fastapi import APIRouter
from app.core.config import settings
from app.core.store import job_store
from app.services.gpu_inference_service import gpu_status

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "llm_provider": settings.LLM_PROVIDER,
        "llm_model": settings.LLM_MODEL_ID if settings.LLM_PROVIDER == "local" else settings.LLM_MODEL,
        "gpu": gpu_status(),
        "active_jobs": len(job_store),
    }


@router.get("/jobs", summary="Список всех задач")
async def list_jobs():
    return [
        {
            "job_id": k,
            "status": v.get("gen_status", "unknown"),
            "scenes": len(v.get("scenes", [])),
        }
        for k, v in job_store.items()
    ]
