# DreamNarrative — Backend

**ВКР Красильников 2026 | NSM · CIM · LAF Pipeline**

Генерация SDXL выполняется на **локальном GPU** (Student GPU Container), без Kaggle.

## Быстрый старт на Student GPU Container

### 1. Скопируйте проект в постоянное хранилище

```bash
# В JupyterLab или VSCode терминале
cd /home/student/work
# загрузите/склонируйте dreamnarrative-backend сюда
cd dreamnarrative-backend
```

### 2. Настройте `.env`

```bash
cp .env.example .env
# GROQ_API_KEY=gsk_...   ← https://console.groq.com
```

### 3. Установите зависимости и запустите

```bash
chmod +x scripts/run_student_gpu.sh
./scripts/run_student_gpu.sh
```

Или вручную:

```bash
pip install -r requirements.txt -r requirements-gpu.txt
python -c "import torch; print(torch.cuda.is_available())"
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 4. Откройте в браузере

По инструкции платформы:

`https://<домен-сервера>/<ваш-username>/proxy/8000/`

Логин контейнера (sudo): `student` / `password`

---

## Структура

```
dreamnarrative-backend/
├── main.py
├── requirements.txt          # API (FastAPI, Groq)
├── requirements-gpu.txt      # torch, diffusers (только на GPU-сервере)
├── scripts/run_student_gpu.sh
├── app/services/gpu_inference_service.py  # SDXL + LAF
└── static/index.html         # UI
```

## API

| Method | Path | Описание |
|--------|------|----------|
| POST | `/api/nsm/run` | NSM (Groq) |
| POST | `/api/cim/run` | CIM |
| POST | `/api/generate/start` | SDXL на GPU |
| GET | `/api/generate/status/{job_id}` | Статус |
| GET | `/api/generate/image/{job_id}/{scene_id}` | PNG |

## Прогресс генерации

- В UI: polling `status=running` — это нормально (SDXL долго считает).
- На сервере: в терминале uvicorn строки `[GPU] Scene 1/4...`
- `IMAGE_SIZE=512` в `.env` — меньше VRAM (можно 1024 на мощной GPU).

## Примечание

Папка `kaggle/` оставлена для справки; пайплайн её больше не использует.
