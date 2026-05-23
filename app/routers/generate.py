from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
import asyncio

from app.core.models import (
    NSMResponse, CIMResponse, GenerateResponse, GenerationStatus, SceneResult
)
from app.core.config import settings
from app.core.store import job_store
from app.services.gpu_inference_service import run_sdxl_generation, gpu_status

router = APIRouter()


class GenerateRequest(BaseModel):
    job_id: str
    nsm_result: NSMResponse
    cim_result: CIMResponse
    cfg_scale: float = 7.5
    ddim_steps: int = 30
    image_size: int = settings.IMAGE_SIZE
    style: str = settings.STORYDIFFUSION_STYLE


async def _run_generation_background(job_id: str, req: GenerateRequest):
    """Фоновая задача: StoryDiffusion на локальном GPU."""
    store = job_store.get(job_id, {})
    store["gen_status"] = GenerationStatus.RUNNING
    job_store[job_id] = store

    try:
        scenes_data = await asyncio.to_thread(
            run_sdxl_generation,
            job_id=job_id,
            nsm_result=req.nsm_result.model_dump(),
            cim_result=req.cim_result.model_dump(),
            cfg_scale=req.cfg_scale,
            ddim_steps=req.ddim_steps,
            image_size=req.image_size,
            style=req.style,
        )
        store["gen_status"] = GenerationStatus.DONE
        store["scenes"] = scenes_data
    except Exception as e:
        store["gen_status"] = GenerationStatus.ERROR
        store["gen_error"] = str(e)


@router.post("/start", response_model=GenerateResponse, summary="Запуск StoryDiffusion генерации на GPU")
async def generate_start(req: GenerateRequest, background_tasks: BackgroundTasks):
    job_id = req.job_id

    pending_scenes = [
        SceneResult(
            scene_id=s.scene_id,
            prompt=s.sdxl_prompt,
            emotion=s.emotion,
            seed=settings.DDIM_SEED_BASE + s.scene_id - 1,
            status=GenerationStatus.PENDING,
        )
        for s in req.nsm_result.scenes
    ]

    job_store[job_id] = job_store.get(job_id, {})
    job_store[job_id]["gen_status"] = GenerationStatus.PENDING
    job_store[job_id]["pending_scenes"] = [s.model_dump() for s in pending_scenes]
    job_store[job_id]["nsm"] = req.nsm_result.model_dump()

    background_tasks.add_task(_run_generation_background, job_id, req)

    hint = gpu_status().get("progress_hint", "")

    return GenerateResponse(
        job_id=job_id,
        status=GenerationStatus.PENDING,
        scenes=pending_scenes,
        gpu_progress_hint=hint,
    )


@router.get("/status/{job_id}", response_model=GenerateResponse, summary="Статус генерации")
async def generate_status(job_id: str):
    store = job_store.get(job_id)
    if not store:
        raise HTTPException(status_code=404, detail="Job not found")

    gen_status = store.get("gen_status", GenerationStatus.PENDING)
    scenes_data = store.get("scenes", [])
    nsm_data = store.get("nsm", {})

    scenes = []
    for sd in scenes_data:
        scenes.append(SceneResult(
            scene_id=sd["scene_id"],
            image_url=sd.get("image_url"),
            image_base64=sd.get("image_base64"),
            prompt=sd.get("prompt", ""),
            emotion=sd.get("emotion", ""),
            seed=settings.DDIM_SEED_BASE + sd["scene_id"] - 1,
            status=GenerationStatus.DONE if sd.get("image_url") else gen_status,
        ))

    if not scenes and nsm_data.get("scenes"):
        for s in nsm_data["scenes"]:
            scenes.append(SceneResult(
                scene_id=s["scene_id"],
                prompt=s["sdxl_prompt"],
                emotion=s["emotion"],
                seed=settings.DDIM_SEED_BASE + s["scene_id"] - 1,
                status=gen_status,
            ))

    gpu_hint = None
    if gen_status in (GenerationStatus.PENDING, GenerationStatus.RUNNING):
        gpu_hint = gpu_status().get("progress_hint")

    return GenerateResponse(
        job_id=job_id,
        status=gen_status,
        scenes=scenes,
        metrics=store.get("metrics"),
        gpu_progress_hint=gpu_hint or store.get("gen_progress"),
        error_message=store.get("gen_error"),
    )


@router.get("/options", summary="Параметры StoryDiffusion для UI")
async def generation_options():
    return {
        "image_size": settings.IMAGE_SIZE,
        "style": settings.STORYDIFFUSION_STYLE,
        "styles": settings.STORYDIFFUSION_STYLES,
        "image_sizes": settings.STORYDIFFUSION_IMAGE_SIZES,
        "cfg_scale": settings.DEFAULT_CFG,
        "ddim_steps": settings.DEFAULT_STEPS,
    }


@router.get("/image/{job_id}/{scene_id}", summary="Получить изображение сцены")
async def get_image(job_id: str, scene_id: int):
    img_path = Path(settings.OUTPUT_DIR) / job_id / f"scene_{scene_id}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(img_path, media_type="image/png")
