#!/usr/bin/env bash
# Запуск DreamNarrative на Student GPU Container
# Рабочая папка: /home/student/work  (постоянное хранилище)

set -euo pipefail

WORKDIR="${WORKDIR:-/home/student/work/dreamnarrative-backend}"
PORT="${PORT:-8000}"

cd "$WORKDIR"

echo "==> Python: $(python3 --version)"
echo "==> Workdir: $WORKDIR"

python3 -m pip install -q -r requirements.txt
python3 -m pip install -q -r requirements-gpu.txt

python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "==> Starting API on 0.0.0.0:$PORT"
echo "    Внешний URL: https://<ваш-домен>/<username>/proxy/$PORT/"
exec python3 -m uvicorn main:app --host 0.0.0.0 --port "$PORT"
