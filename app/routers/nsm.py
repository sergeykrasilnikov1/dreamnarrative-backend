from fastapi import APIRouter, HTTPException, BackgroundTasks
import uuid
from app.core.models import PipelineRequest, NSMResponse
from app.services.nsm_service import run_nsm
from app.core.store import job_store

router = APIRouter()


@router.post("/run", response_model=NSMResponse, summary="Запуск NSM сегментации через Groq LLaMA 3.1 8B")
async def nsm_run(req: PipelineRequest):
    job_id = str(uuid.uuid4())[:8]
    try:
        result = run_nsm(req.dream_text, req.n_scenes, job_id)
        job_store[job_id] = {"nsm": result.model_dump(), "status": "nsm_done"}
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NSM error: {str(e)}")
