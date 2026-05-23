#!/usr/bin/env bash
# Клонирование StoryDiffusion для consistent self-attention генерации
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TARGET="${ROOT}/third_party/StoryDiffusion"

mkdir -p "${ROOT}/third_party"

if [[ -d "${TARGET}/.git" ]]; then
  echo "==> StoryDiffusion уже установлен: ${TARGET}"
  git -C "${TARGET}" pull --ff-only || true
else
  echo "==> Cloning StoryDiffusion..."
  git clone --depth 1 https://github.com/HVision-NKU/StoryDiffusion.git "${TARGET}"
fi

echo "==> OK: ${TARGET}"
echo "    Модель по умолчанию: SG161222/RealVisXL_V4.0 (скачается при первом запуске)"
