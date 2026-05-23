import json
import time

from app.core.config import settings
from app.core.models import NSMResponse, CharacterSchema, SceneSchema
from app.services.llm_client import chat_json, unload_local_llm, _parse_json_response


NSM_SYSTEM_PROMPT = """You are a dream narrative segmentation system.
Given a dream description, you MUST output ONLY valid JSON — no explanation, no markdown.

Decide yourself how many sequential scenes (between 3 and 8) best represent the dream.
Simple dreams → fewer scenes; complex dreams with many events → more scenes.

Output format:
{
  "characters": [
    {
      "name": "unique_identifier_name",
      "appearance": "detailed visual description",
      "canonical_appearance": "concise stable appearance for image generation"
    }
  ],
  "scenes": [
    {
      "scene_id": 1,
      "description": "what happens in this scene (narrative)",
      "characters": ["character_name"],
      "objects": ["key objects"],
      "location": "detailed location for image generation",
      "emotion": "fear",
      "sdxl_prompt": "FULL English prompt for image model: location, all visible characters with appearance, action, lighting, mood, cinematic style — one complete paragraph, ready to generate",
      "negative_prompt": "blurry, low quality, distorted, text, watermark"
    }
  ]
}

CRITICAL:
- Every scene MUST have a non-empty sdxl_prompt (complete generative prompt, not just location).
- scenes[].characters: ONLY string names from characters[].name — never null, never objects.
- scene_id: 1, 2, 3, ... sequential."""


def build_user_prompt(dream_text: str) -> str:
    return f"""Analyze this dream and split it into the right number of scenes (3–8).
Extract characters with canonical_appearance for consistent image generation.
For each scene write sdxl_prompt: a FULL prompt for Stable Diffusion / StoryDiffusion (English).

Dream text:
\"\"\"{dream_text}\"\"\"

Output JSON only. Choose scene count based on dream complexity."""


def _extract_scene_character_names(raw_chars, known_names: set[str]) -> list[str]:
    """Парсит characters сцены: строки, объекты {{name:...}}, отбрасывает null."""
    names: list[str] = []
    for item in raw_chars or []:
        if item is None:
            continue
        if isinstance(item, str):
            n = item.strip()
            if n and n.lower() not in ("null", "none", "unknown"):
                names.append(n)
        elif isinstance(item, dict):
            n = (item.get("name") or item.get("character") or item.get("id") or "").strip()
            if n:
                names.append(n)
    # только известные имена; остальное — как есть (LLM мог слегка изменить id)
    out: list[str] = []
    for n in names:
        if n in known_names and n not in out:
            out.append(n)
        elif n not in out:
            out.append(n)
    return out


def _infer_scene_characters(
    scene: dict, scene_id: int, char_defs: list[dict]
) -> list[str]:
    """Если LLM вернул null — берём из characters[].scenes_present или текста сцены."""
    from_scenes_present = [
        c["name"]
        for c in char_defs
        if c.get("name") and scene_id in (c.get("scenes_present") or [])
    ]
    if from_scenes_present:
        return from_scenes_present

    text = " ".join(
        str(scene.get(k, ""))
        for k in ("description", "location", "sdxl_prompt")
    ).lower()
    found: list[str] = []
    for c in char_defs:
        name = (c.get("name") or "").strip()
        if not name:
            continue
        key = name.replace("_", " ")
        if name in text or key in text:
            found.append(name)
            continue
        for part in name.split("_"):
            if len(part) > 3 and part in text:
                found.append(name)
                break
    return found


def _compose_sdxl_prompt(scene: dict, char_defs: list[dict]) -> str:
    """Полный промпт для генерации, если LLM не вернул sdxl_prompt."""
    for key in ("sdxl_prompt", "image_prompt", "generation_prompt"):
        val = (scene.get(key) or "").strip()
        if val:
            return val

    lookup = {
        c["name"]: (c.get("canonical_appearance") or c.get("appearance") or "")
        for c in char_defs
    }
    parts: list[str] = []
    for name in scene.get("characters") or []:
        if name in lookup and lookup[name]:
            parts.append(lookup[name])
    if scene.get("location"):
        parts.append(str(scene["location"]))
    if scene.get("description"):
        parts.append(str(scene["description"]))
    if scene.get("emotion"):
        parts.append(f"{scene['emotion']} mood, emotional atmosphere")
    objs = [str(o) for o in (scene.get("objects") or []) if o]
    if objs:
        parts.append(", ".join(objs))
    parts.append("cinematic dream scene, highly detailed, atmospheric lighting")
    return ", ".join(p for p in parts if p)


def run_nsm(dream_text: str, job_id: str) -> NSMResponse:
    t0 = time.time()

    raw = ""
    model_name = settings.LLM_MODEL
    try:
        raw, model_name = chat_json(NSM_SYSTEM_PROMPT, build_user_prompt(dream_text))
        data = _parse_json_response(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM вернул невалидный JSON: {e}. Preview: {raw[:200]!r}") from e
    except ConnectionError:
        raise
    except Exception as e:
        raise ValueError(f"LLM error ({settings.LLM_PROVIDER}): {e}") from e
    finally:
        unload_local_llm()

    elapsed_ms = int((time.time() - t0) * 1000)

    char_defs: list[dict] = []
    for c in data.get("characters", []):
        if isinstance(c, str):
            c = {"name": c, "appearance": c, "canonical_appearance": c}
        if not isinstance(c, dict):
            continue
        name = (c.get("name") or "").strip() or "unknown"
        char_defs.append({**c, "name": name})

    known_names = {c["name"] for c in char_defs}

    scenes: list[SceneSchema] = []
    for s in data.get("scenes", []):
        scene_id = int(s.get("scene_id", len(scenes) + 1))
        scene_chars = _extract_scene_character_names(s.get("characters"), known_names)
        if not scene_chars:
            scene_chars = _infer_scene_characters(s, scene_id, char_defs)

        objects = [
            str(x).strip()
            for x in (s.get("objects") or [])
            if x is not None and str(x).strip()
        ]

        scenes.append(
            SceneSchema(
                scene_id=scene_id,
                description=s.get("description", ""),
                characters=scene_chars,
                objects=objects,
                location=s.get("location", ""),
                emotion=s.get("emotion", "wonder"),
                sdxl_prompt=_compose_sdxl_prompt({**s, "characters": scene_chars}, char_defs),
                negative_prompt=s.get(
                    "negative_prompt",
                    "blurry, low quality, distorted, text, watermark, nsfw",
                ),
            )
        )

    if not scenes:
        raise ValueError("LLM не вернул ни одной сцены")
    if len(scenes) < settings.NSM_MIN_SCENES:
        raise ValueError(
            f"LLM вернул {len(scenes)} сцен, нужно минимум {settings.NSM_MIN_SCENES} "
            "(StoryDiffusion). Попробуйте ещё раз или опишите сон подробнее."
        )
    if len(scenes) > settings.NSM_MAX_SCENES:
        scenes = scenes[: settings.NSM_MAX_SCENES]

    characters = [
        CharacterSchema(
            name=c["name"],
            appearance=c.get("appearance", ""),
            canonical_appearance=c.get("canonical_appearance", c.get("appearance", "")),
            scenes_present=[sc.scene_id for sc in scenes if c["name"] in sc.characters],
        )
        for c in char_defs
    ]

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
