"""
Универсальный LLM-клиент для NSM (без Groq).

Провайдеры:
  - local      — HuggingFace Qwen на GPU/CPU (Student GPU Container)
  - openrouter — OpenRouter API (DeepSeek V4 Flash и др.)
  - openai     — любой OpenAI-compatible API (Ollama)
  - groq       — legacy
"""
from __future__ import annotations

import json
import re
from typing import Optional

import httpx

from app.core.config import settings

_local_model = None
_local_tokenizer = None


def _parse_json_response(raw: str) -> dict:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def chat_json(system: str, user: str) -> tuple[str, str]:
    """
    Возвращает (raw_text, model_name).
    """
    provider = settings.LLM_PROVIDER.lower()
    if provider == "local":
        return _chat_local(system, user), settings.LLM_MODEL_ID
    if provider == "openrouter":
        return _chat_openrouter(system, user), settings.OPENROUTER_MODEL
    if provider == "openai":
        return _chat_openai_compatible(system, user), settings.LLM_MODEL
    if provider == "groq":
        return _chat_groq(system, user), settings.LLM_MODEL
    raise ValueError(
        f"Неизвестный LLM_PROVIDER={settings.LLM_PROVIDER!r}. "
        "Используйте: local, openrouter, openai, groq"
    )


def _chat_openrouter(system: str, user: str) -> str:
    api_key = settings.OPENROUTER_API_KEY or settings.LLM_API_KEY
    if not api_key or api_key in ("your_openrouter_api_key", "sk-or-YOUR"):
        raise ValueError(
            "OPENROUTER_API_KEY не задан в .env (https://openrouter.ai/keys)"
        )

    base = settings.OPENROUTER_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if settings.OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = settings.OPENROUTER_HTTP_REFERER
    if settings.OPENROUTER_APP_TITLE:
        headers["X-Title"] = settings.OPENROUTER_APP_TITLE

    body = {
        "model": settings.OPENROUTER_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if settings.LLM_JSON_MODE:
        body["response_format"] = {"type": "json_object"}

    return _post_chat_completions(url, headers, body, provider_label="OpenRouter")


def _chat_openai_compatible(system: str, user: str) -> str:
    base = settings.LLM_BASE_URL.rstrip("/")
    url = f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if settings.LLM_API_KEY:
        headers["Authorization"] = f"Bearer {settings.LLM_API_KEY}"

    body = {
        "model": settings.LLM_MODEL,
        "temperature": settings.LLM_TEMPERATURE,
        "max_tokens": settings.LLM_MAX_TOKENS,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    if settings.LLM_JSON_MODE:
        body["response_format"] = {"type": "json_object"}

    return _post_chat_completions(url, headers, body, provider_label="LLM API")


def _post_chat_completions(
    url: str, headers: dict, body: dict, provider_label: str = "LLM API"
) -> str:
    try:
        with httpx.Client(timeout=settings.LLM_TIMEOUT) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            data = resp.json()
    except httpx.ConnectError as e:
        raise ConnectionError(
            f"Не удалось подключиться к {provider_label} ({url}). "
            "Проверьте сеть или LLM_PROVIDER=local. "
            f"Оригинал: {e}"
        ) from e
    except httpx.HTTPStatusError as e:
        raise ValueError(
            f"{provider_label} HTTP {e.response.status_code}: {e.response.text[:300]}"
        ) from e

    message = data["choices"][0]["message"]
    content = message.get("content") or ""
    if not content.strip() and message.get("reasoning"):
        content = message["reasoning"]
    return content


def _chat_groq(system: str, user: str) -> str:
    from groq import Groq

    api_key = settings.LLM_API_KEY or settings.GROQ_API_KEY
    if not api_key or api_key.startswith("gsk_YOUR"):
        raise ValueError("LLM_API_KEY / GROQ_API_KEY не задан в .env")

    client = Groq(api_key=api_key)
    completion = client.chat.completions.create(
        model=settings.LLM_MODEL or settings.GROQ_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    return completion.choices[0].message.content


def _is_qwen35(model_id: str) -> bool:
    mid = model_id.lower()
    return "qwen3.5" in mid or "qwen3_5" in mid


def _local_dtype():
    import torch

    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _load_local_model(model_id: str):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    try:
        from transformers import AutoModelForImageTextToText
    except ImportError:
        AutoModelForImageTextToText = None

    dtype = _local_dtype()
    load_kwargs = {"torch_dtype": dtype, "trust_remote_code": True}
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    model_type = getattr(config, "model_type", "")

    if model_type == "qwen3_5" and AutoModelForImageTextToText is not None:
        return AutoModelForImageTextToText.from_pretrained(model_id, **load_kwargs)

    return AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)


def _strip_thinking(text: str) -> str:
    """Qwen3.5 может вернуть блок reasoning перед ответом."""
    close = "</think>"
    if close in text:
        return text.split(close, 1)[-1].strip()
    return text.strip()


def _get_local_llm():
    global _local_model, _local_tokenizer
    if _local_model is not None:
        return _local_model, _local_tokenizer

    import torch
    from transformers import AutoTokenizer

    model_id = settings.LLM_MODEL_ID
    print(f"[LLM] Loading {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = _load_local_model(model_id)
    model.eval()

    _local_model = model
    _local_tokenizer = tokenizer
    print(f"[LLM] Ready on {'cuda' if torch.cuda.is_available() else 'cpu'}")
    return _local_model, _local_tokenizer


def _chat_local(system: str, user: str) -> str:
    import torch

    model, tokenizer = _get_local_llm()
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    model_id = settings.LLM_MODEL_ID
    if hasattr(tokenizer, "apply_chat_template"):
        template_kwargs = {}
        if _is_qwen35(model_id):
            template_kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **template_kwargs,
        )
    else:
        prompt = f"{system}\n\n{user}\n\nОтвет (только JSON):"

    inputs = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": settings.LLM_MAX_TOKENS,
        "temperature": settings.LLM_TEMPERATURE,
        "do_sample": settings.LLM_TEMPERATURE > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if _is_qwen35(model_id):
        gen_kwargs["top_p"] = 0.8
        gen_kwargs["top_k"] = 20

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)

    new_tokens = out[0][inputs["input_ids"].shape[1] :]
    raw = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return _strip_thinking(raw)


def unload_local_llm() -> None:
    """Освобождает VRAM перед загрузкой SDXL."""
    global _local_model, _local_tokenizer
    if _local_model is None:
        return

    import gc
    import torch

    del _local_model
    del _local_tokenizer
    _local_model = None
    _local_tokenizer = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[LLM] Model unloaded (VRAM freed for SDXL)")
