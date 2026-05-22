from fastapi import APIRouter
from app.core.config import settings
from app.core.store import job_store
from app.services.gpu_inference_service import gpu_status

router = APIRouter()


@router.get("/health", summary="Health check")
async def health():
    return {
        "status": "ok",
        "groq_model": settings.GROQ_MODEL,
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
