import json
import time

from app.core.config import settings
from app.core.models import NSMResponse, CharacterSchema, SceneSchema
from app.services.llm_client import chat_json, unload_local_llm, _parse_json_response


NSM_SYSTEM_PROMPT = """You are a dream narrative segmentation system.
Given a dream description, you MUST output ONLY valid JSON — no explanation, no markdown.

Output format:
{
  "characters": [
    {
      "name": "unique_identifier_name",
      "appearance": "detailed visual description",
      "canonical_appearance": "concise stable appearance for SDXL prompt"
    }
  ],
  "scenes": [
    {
      "scene_id": 1,
      "description": "what happens in this scene",
      "characters": ["character names present"],
      "objects": ["key objects"],
      "location": "detailed location description for image generation",
      "emotion": "wonder",
      "sdxl_prompt": "optimized SDXL prompt: location, characters, atmosphere, style",
      "negative_prompt": "blurry, low quality, distorted, text, watermark"
    }
  ]
}"""


def build_user_prompt(dream_text: str, n_scenes: int) -> str:
    return f"""Segment this dream into exactly {n_scenes} sequential scenes.
Extract all characters with their canonical appearance for consistent image generation.
Build detailed SDXL-optimized prompts (include: location, characters, lighting, mood, art style).

Dream text:
\"\"\"{dream_text}\"\"\"

Output {n_scenes} scenes in the JSON format. ONLY JSON, no other text."""


def run_nsm(dream_text: str, n_scenes: int, job_id: str) -> NSMResponse:
    t0 = time.time()

    raw = ""
    model_name = settings.LLM_MODEL
    try:
        raw, model_name = chat_json(NSM_SYSTEM_PROMPT, build_user_prompt(dream_text, n_scenes))
        data = _parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM вернул невалидный JSON: {e}. Preview: {raw[:200]!r}") from e
    except ConnectionError:
        raise
    except Exception as e:
        raise ValueError(f"LLM error ({settings.LLM_PROVIDER}): {e}") from e
    finally:
        if settings.LLM_PROVIDER.lower() == "local":
            unload_local_llm()

    elapsed_ms = int((time.time() - t0) * 1000)

    characters = []
    for c in data.get("characters", []):
        if isinstance(c, str):
            c = {"name": c, "appearance": c, "canonical_appearance": c}
        name = c.get("name") or "unknown"
        characters.append(
            CharacterSchema(
                name=name,
                appearance=c.get("appearance", ""),
                canonical_appearance=c.get("canonical_appearance", c.get("appearance", "")),
                scenes_present=[
                    int(s["scene_id"])
                    for s in data.get("scenes", [])
                    if name in (s.get("characters") or [])
                ],
            )
        )

    scenes = []
    for s in data.get("scenes", []):
        scenes.append(
            SceneSchema(
                scene_id=int(s.get("scene_id", len(scenes) + 1)),
                description=s.get("description", ""),
                characters=[str(x) for x in s.get("characters", [])],
                objects=[str(x) for x in s.get("objects", [])],
                location=s.get("location", ""),
                emotion=s.get("emotion", "wonder"),
                sdxl_prompt=s.get("sdxl_prompt", s.get("description", "")),
                negative_prompt=s.get(
                    "negative_prompt",
                    "blurry, low quality, distorted, text, watermark, nsfw",
                ),
            )
        )

    if not scenes:
        raise ValueError("LLM не вернул ни одной сцены")

    return NSMResponse(
        job_id=job_id,
        total_scenes=len(scenes),
        characters=characters,
        scenes=scenes,
        clip_threshold=settings.CLIP_THRESHOLD,
        arcface_threshold=settings.ARCFACE_THRESHOLD,
        llm_model=f"{settings.LLM_PROVIDER}:{model_name}",
        processing_time_ms=elapsed_ms,
    )
