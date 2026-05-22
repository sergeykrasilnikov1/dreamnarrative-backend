import json
import time
import uuid
from groq import Groq

from app.core.config import settings
from app.core.models import NSMResponse, CharacterSchema, SceneSchema


NSM_SYSTEM_PROMPT = """You are a dream narrative segmentation system.
Given a dream description, you MUST output ONLY valid JSON — no explanation, no markdown.

Output format:
{
  "characters": [
    {
      "name": "unique identifier name",
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
      "emotion": "primary emotion (wonder|fear|confusion|awe|anxiety|peace|curiosity|dread)",
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
    client = Groq(api_key=settings.GROQ_API_KEY)
    t0 = time.time()

    completion = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        temperature=settings.GROQ_TEMPERATURE,
        max_tokens=settings.GROQ_MAX_TOKENS,
        messages=[
            {"role": "system", "content": NSM_SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(dream_text, n_scenes)},
        ],
        response_format={"type": "json_object"},
    )

    raw = completion.choices[0].message.content
    data = json.loads(raw)

    elapsed_ms = int((time.time() - t0) * 1000)

    # Validate & enrich scenes
    chars_map = {c["name"]: c for c in data.get("characters", [])}
    characters = [
        CharacterSchema(
            name=c["name"],
            appearance=c.get("appearance", ""),
            canonical_appearance=c.get("canonical_appearance", c.get("appearance", "")),
            scenes_present=[
                s["scene_id"]
                for s in data.get("scenes", [])
                if c["name"] in s.get("characters", [])
            ],
        )
        for c in data.get("characters", [])
    ]

    scenes = [
        SceneSchema(
            scene_id=s["scene_id"],
            description=s.get("description", ""),
            characters=s.get("characters", []),
            objects=s.get("objects", []),
            location=s.get("location", ""),
            emotion=s.get("emotion", "wonder"),
            sdxl_prompt=s.get("sdxl_prompt", s.get("description", "")),
            negative_prompt=s.get(
                "negative_prompt",
                "blurry, low quality, distorted, text, watermark, nsfw",
            ),
        )
        for s in data.get("scenes", [])
    ]

    return NSMResponse(
        job_id=job_id,
        total_scenes=len(scenes),
        characters=characters,
        scenes=scenes,
        clip_threshold=settings.CLIP_THRESHOLD,
        arcface_threshold=settings.ARCFACE_THRESHOLD,
        llm_model=settings.GROQ_MODEL,
        processing_time_ms=elapsed_ms,
    )
