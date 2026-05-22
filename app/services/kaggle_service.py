"""
Kaggle Kernel dispatcher — запускает SDXL inference на GPU T4/P100.
Передаёт NSM/CIM результаты через Kaggle Dataset, получает изображения обратно.
"""
import json
import time
import base64
import zipfile
import tempfile
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException

from app.core.config import settings

KERNEL_SCRIPT = Path(__file__).resolve().parents[2] / "kaggle" / "sdxl_inference.py"
RUNNING_STATES = {"", "running", "queued", "pending", "starting"}
TERMINAL_ERROR = {"error", "cancelled"}


def _get_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api


def _kernel_ref(kernel_slug: str) -> str:
    return f"{settings.KAGGLE_USERNAME}/{kernel_slug}"


def _dataset_ref() -> str:
    return f"{settings.KAGGLE_USERNAME}/{settings.KAGGLE_DATASET_SLUG}"


def _write_dataset_metadata(folder: Path) -> None:
    meta = {
        "title": "DreamNarrative Payload",
        "id": _dataset_ref(),
        "licenses": [{"name": "CC0-1.0"}],
    }
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))


def _write_kernel_metadata(folder: Path) -> None:
    meta = {
        "id": _kernel_ref(settings.KAGGLE_KERNEL_SLUG),
        "title": "DreamNarrative SDXL Inference",
        "code_file": "sdxl_inference.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [_dataset_ref()],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (folder / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))


def _upload_payload_dataset(api: KaggleApi, job_id: str, payload: dict) -> None:
    """Загружает payload.json в Kaggle Dataset (создаёт dataset при первом запуске)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        (folder / "payload.json").write_text(json.dumps(payload))
        _write_dataset_metadata(folder)

        try:
            api.dataset_create_version(
                folder=str(folder),
                version_notes=f"job_{job_id}",
                dir_mode="zip",
                quiet=True,
            )
        except (ApiException, ValueError, OSError):
            api.dataset_create_new(
                folder=str(folder),
                public=False,
                dir_mode="zip",
                quiet=True,
            )


def _push_kernel_run(api: KaggleApi) -> None:
    """Пушит script kernel и запускает новый run на GPU."""
    if not KERNEL_SCRIPT.is_file():
        raise FileNotFoundError(f"Kernel script not found: {KERNEL_SCRIPT}")

    with tempfile.TemporaryDirectory() as tmpdir:
        folder = Path(tmpdir)
        shutil.copy(KERNEL_SCRIPT, folder / "sdxl_inference.py")
        _write_kernel_metadata(folder)
        api.kernels_push(str(folder))


def dispatch_to_kaggle(job_id: str, nsm_result: dict, cim_result: dict, cfg: float, steps: int) -> str:
    """
    1. Загружает payload в Kaggle Dataset (input для kernel)
    2. Запускает kernel dreamnarrative-sdxl-inference
    3. Возвращает kernel slug для polling
    """
    api = _get_kaggle_api()

    payload = {
        "job_id": job_id,
        "nsm": nsm_result,
        "cim": cim_result,
        "config": {
            "cfg_scale": cfg,
            "ddim_steps": steps,
            "seed_base": settings.DDIM_SEED_BASE,
            "image_size": settings.IMAGE_SIZE,
            "lambda_csa": settings.LAF_LAMBDA_CSA,
            "mu_cca": settings.LAF_MU_CCA,
        },
    }

    _upload_payload_dataset(api, job_id, payload)
    _push_kernel_run(api)

    return settings.KAGGLE_KERNEL_SLUG


def kaggle_kernel_url() -> str:
    return f"https://www.kaggle.com/code/{settings.KAGGLE_USERNAME}/{settings.KAGGLE_KERNEL_SLUG}"


def _parse_log_entries(raw: str) -> list[str]:
    if not raw.strip().startswith("["):
        return [raw] if raw.strip() else []
    lines = []
    for entry in json.loads(raw):
        data = (entry.get("data") or "").rstrip("\n")
        if data:
            lines.append(data)
    return lines


def _fetch_kernel_log_tail(api: KaggleApi, kernel_slug: str) -> str:
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            api.kernels_output(_kernel_ref(kernel_slug), tmpdir, quiet=True)
            log_path = Path(tmpdir) / f"{kernel_slug}.log"
            if not log_path.is_file():
                return ""
            lines = _parse_log_entries(log_path.read_text(errors="replace"))
            return "\n".join(lines[-20:])[:1500]
    except Exception:
        return ""


def get_kernel_live_status(kernel_slug: str | None = None) -> dict:
    """Текущий статус run на Kaggle + подсказка из лога (если доступен)."""
    slug = kernel_slug or settings.KAGGLE_KERNEL_SLUG
    api = _get_kaggle_api()
    ref = _kernel_ref(slug)

    try:
        status = api.kernels_status(ref)
    except ApiException as e:
        return {
            "kaggle_run_status": "unknown",
            "kaggle_progress_hint": f"Kaggle API: {e.body or e.reason}",
            "kaggle_kernel_url": kaggle_kernel_url(),
        }

    state = (status.get("status") or "unknown").lower()
    hint = ""

    if state == "running":
        tail = _fetch_kernel_log_tail(api, slug)
        for line in reversed(tail.splitlines()):
            stripped = line.strip()
            if any(
                key in stripped
                for key in ("Scene ", "Loading SDXL", "Saved:", "Job:", "LAF:", "Device:", "✓")
            ):
                hint = stripped[:200]
                break
        if not hint:
            hint = "GPU run активен — откройте Kaggle, вкладка Logs/Output"
    elif state == "complete":
        hint = "Run завершён, скачивание результатов..."
    elif state in TERMINAL_ERROR:
        hint = (status.get("failureMessage") or "").strip()
        if not hint:
            hint = _fetch_kernel_log_error(api, slug)[:300]

    return {
        "kaggle_run_status": state,
        "kaggle_progress_hint": hint or None,
        "kaggle_kernel_url": kaggle_kernel_url(),
    }


def _fetch_kernel_log_error(api: KaggleApi, kernel_slug: str) -> str:
    """Извлекает текст ошибки из .log, если Kaggle не заполнил failureMessage."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            api.kernels_output(_kernel_ref(kernel_slug), tmpdir, quiet=True)
            log_path = Path(tmpdir) / f"{kernel_slug}.log"
            if not log_path.is_file():
                return ""

            raw = log_path.read_text(errors="replace")
            if raw.strip().startswith("["):
                lines = []
                for entry in json.loads(raw):
                    data = entry.get("data", "")
                    if entry.get("stream_name") == "stderr" or "Error" in data or "Traceback" in data:
                        lines.append(data.rstrip("\n"))
                if lines:
                    return "\n".join(lines[-12:])[:1200]
            return raw[-1200:]
    except Exception:
        return ""


def poll_kaggle_status(kernel_slug: str, max_wait: int | None = None) -> dict:
    """Поллинг статуса Kaggle Kernel до завершения."""
    api = _get_kaggle_api()
    max_wait = max_wait or settings.KAGGLE_POLL_MAX_WAIT
    start = time.time()
    ref = _kernel_ref(kernel_slug)

    while time.time() - start < max_wait:
        try:
            status = api.kernels_status(ref)
        except ApiException as e:
            if e.status == 403:
                return {
                    "status": "error",
                    "message": (
                        f"Нет доступа к kernel {ref}. Создайте его на Kaggle "
                        f"или проверьте KAGGLE_USERNAME / KAGGLE_KERNEL_SLUG."
                    ),
                }
            raise

        state = (status.get("status") or "").lower()

        if state == "complete":
            return {"status": "done", "kernel_ref": kernel_slug}
        if state in TERMINAL_ERROR:
            # Сразу после push API может ещё отдавать error от прошлого run
            if time.time() - start < 60:
                time.sleep(10)
                continue
            msg = (status.get("failureMessage") or "").strip()
            if not msg:
                msg = _fetch_kernel_log_error(api, kernel_slug).strip()
            if not msg:
                msg = "Kernel failed (см. лог на Kaggle)"
            return {"status": "error", "message": msg}

        if state not in RUNNING_STATES:
            # Неизвестный статус — продолжаем ждать
            pass

        time.sleep(10)

    return {"status": "timeout", "message": f"Kaggle kernel не завершился за {max_wait}s"}


def fetch_kaggle_outputs(kernel_slug: str, job_id: str) -> list[dict]:
    """Скачивает output изображения из Kaggle Kernel output."""
    api = _get_kaggle_api()
    output_dir = Path(settings.OUTPUT_DIR) / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    api.kernels_output(_kernel_ref(kernel_slug), str(output_dir))

    for zf in output_dir.glob("*.zip"):
        with zipfile.ZipFile(zf) as z:
            z.extractall(output_dir)

    scenes = []
    for img_path in sorted(output_dir.glob("scene_*.png")):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        scene_id = int(img_path.stem.split("_")[1])
        scenes.append({
            "scene_id": scene_id,
            "image_base64": b64,
            "image_url": f"/api/generate/image/{job_id}/{scene_id}",
        })

    return scenes
