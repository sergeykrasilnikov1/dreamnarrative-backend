# DreamNarrative — Backend

**ВКР Красильников 2026 | NSM · CIM · LAF Pipeline**

Генерация изображений — **[StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion)** (Consistent Self-Attention) на локальном GPU.

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
# По умолчанию LLM на GPU без VPN:
# LLM_PROVIDER=local
# LLM_MODEL_ID=Qwen/Qwen3.5-4B
```

### 3. Установите зависимости и запустите

```bash
chmod +x scripts/run_student_gpu.sh
./scripts/run_student_gpu.sh
```

Или вручную:

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
bash scripts/setup_storydiffusion.sh
uvicorn main:app --host 0.0.0.0 --port 8000
```

## StoryDiffusion

Минимум **3 сцены** в NSM (требование consistent self-attention).

| Переменная | По умолчанию | Описание |
|------------|--------------|----------|
| `STORYDIFFUSION_MODEL_ID` | `SG161222/RealVisXL_V4.0` | SDXL backbone |
| `STORYDIFFUSION_STYLE` | `Photographic` | Стиль (см. `utils/style_template.py` в репо) |
| `STORYDIFFUSION_ID_LENGTH` | `3` | Сколько первых сцен в identity-батче |
| `STORYDIFFUSION_SA32/SA64` | `0.5` | Сила paired attention |

### 4. Проверка интернета (Jupyter)

```bash
# В JupyterLab откройте и выполните все ячейки:
notebooks/check_internet.ipynb
```

### 5. Откройте в браузере

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
├── scripts/setup_storydiffusion.sh
├── third_party/StoryDiffusion/   # git clone (setup script)
├── app/services/storydiffusion_service.py
└── static/index.html
```

## LLM для NSM (без Groq / VPN)

| `LLM_PROVIDER` | Описание |
|----------------|----------|
| `local` | **По умолчанию.** Qwen3.5-4B на GPU, затем выгрузка → StoryDiffusion |
| `openrouter` | [OpenRouter](https://openrouter.ai) — `deepseek/deepseek-v4-flash`, нужен `OPENROUTER_API_KEY` |
| `openai` | Ollama / vLLM: `LLM_BASE_URL=http://127.0.0.1:11434/v1` |
| `groq` | Groq API (legacy) |

**OpenRouter (DeepSeek V4 Flash):**
```env
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=deepseek/deepseek-v4-flash
```
Бесплатный tier: `deepseek/deepseek-v4-flash:free`

Первый запуск `local` скачает модель с HuggingFace (~9 GB). Нужен `transformers>=4.52`.

## API

| Method | Path | Описание |
|--------|------|----------|
| POST | `/api/nsm/run` | NSM (LLM) |
| POST | `/api/cim/run` | CIM |
| POST | `/api/generate/start` | StoryDiffusion на GPU |
| GET | `/api/generate/status/{job_id}` | Статус |
| GET | `/api/generate/image/{job_id}/{scene_id}` | PNG |

## Прогресс генерации

- В UI: polling `status=running` — это нормально (SDXL долго считает).
- На сервере: в терминале uvicorn строки `[GPU] Scene 1/4...`
- `IMAGE_SIZE=512` в `.env` — меньше VRAM (можно 1024 на мощной GPU).

## Примечание

Папка `kaggle/` оставлена для справки; пайплайн её больше не использует.
