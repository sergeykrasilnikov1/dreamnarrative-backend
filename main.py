from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn

from app.routers import nsm, cim, generate, status
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.services.gpu_inference_service import gpu_status

    gpu = gpu_status()
    print(f"DreamNarrative API — Groq: {settings.GROQ_API_KEY[:8]}...")
    print(f"SDXL backend: local GPU — {gpu.get('progress_hint', 'n/a')}")
    yield
    print("Shutdown.")


app = FastAPI(
    title="DreamNarrative API",
    description="NSM · CIM · LAF Pipeline — ВКР Красильников 2026",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(nsm.router,      prefix="/api/nsm",      tags=["NSM"])
app.include_router(cim.router,      prefix="/api/cim",      tags=["CIM"])
app.include_router(generate.router, prefix="/api/generate", tags=["Generate"])
app.include_router(status.router,   prefix="/api",          tags=["Status"])

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
