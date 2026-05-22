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

Output {n_scenes} scenes in the JSON format. ONLY JSON, no other text.

RULES for scenes[].characters:
- Use ONLY string identifiers from the top-level characters[].name list.
- Example: "characters": ["flying_person", "silver_hair_woman"] — NOT null, NOT {{}} objects."""


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
                sdxl_prompt=s.get("sdxl_prompt", s.get("description", "")),
                negative_prompt=s.get(
                    "negative_prompt",
                    "blurry, low quality, distorted, text, watermark, nsfw",
                ),
            )
        )

    if not scenes:
        raise ValueError("LLM не вернул ни одной сцены")

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
