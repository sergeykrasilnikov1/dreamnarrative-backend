from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.core.models import NSMResponse, CIMResponse
from app.services.cim_service import run_cim
from app.core.store import job_store

router = APIRouter()


class CIMRequest(BaseModel):
    job_id: str
    nsm_result: NSMResponse


@router.post("/run", response_model=CIMResponse, summary="Запуск CIM — Character Identity Module")
async def cim_run(req: CIMRequest):
    try:
        result = run_cim(req.nsm_result, req.job_id)
        if req.job_id in job_store:
            job_store[req.job_id]["cim"] = result.model_dump()
            job_store[req.job_id]["status"] = "cim_done"
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CIM error: {str(e)}")
