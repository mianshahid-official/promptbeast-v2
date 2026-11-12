"""
Story Bible Extractor and Manager for PromptBeast-V2 (Step 2).
Chunks scenes into <=10 scenes or ~60s narration, extracts entities via Groq,
merges into Master Story Bible with deduplication and conflict logging, then locks it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.groq_engine import GroqEngine

ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "characters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "appearance": {"type": "string"},
                    "clothing": {"type": "string"},
                    "colors": {"type": "string"},
                    "features": {"type": "string"}
                },
                "required": ["name", "appearance", "clothing", "colors", "features"],
                "additionalProperties": False
            }
        },
        "locations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "visual_traits": {"type": "string"},
                    "time_of_day": {"type": "string"},
                    "atmosphere": {"type": "string"}
                },
                "required": ["name", "visual_traits", "time_of_day", "atmosphere"],
                "additionalProperties": False
            }
        },
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"}
                },
                "required": ["name", "description"],
                "additionalProperties": False
            }
        }
    },
    "required": ["characters", "locations", "objects"],
    "additionalProperties": False
}

EXTRACTION_SYSTEM_PROMPT = """You are an expert story continuity supervisor and visual entity extractor.
Analyze the provided narration scenes and extract visual entities (characters, locations, key objects/props).
Focus on concrete physical visual traits, colors, clothing, and atmosphere to maintain visual continuity.
Do not invent entities that are not mentioned or implied by the narration.
Return strictly valid JSON conforming to the schema."""


def chunk_scenes_for_bible(scenes: List[Dict[str, Any]], max_scenes: int = 10, max_duration: float = 60.0) -> List[List[Dict[str, Any]]]:
    """Groups scenes into chunks of up to 10 scenes or ~60s of narration."""
    chunks: List[List[Dict[str, Any]]] = []
    current_chunk: List[Dict[str, Any]] = []
    current_dur = 0.0

    for sc in scenes:
        dur = float(sc.get("duration", 0.0))
        if current_chunk and (len(current_chunk) >= max_scenes or (current_dur + dur > max_duration)):
            chunks.append(current_chunk)
            current_chunk = [sc]
            current_dur = dur
        else:
            current_chunk.append(sc)
            current_dur += dur

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def extract_entities_for_chunk(chunk: List[Dict[str, Any]], engine: GroqEngine, model: Optional[str] = None) -> Dict[str, Any]:
    """Calls Groq to extract entities for a single chunk of scenes."""
    narration_lines = []
    for sc in chunk:
        s_id = sc.get("scene_id", "")
        text = sc.get("text", "")
        narration_lines.append(f"Scene {s_id}: {text}")

    prompt = "Narration Scenes:\n" + "\n".join(narration_lines)
    return engine.generate_structured(
        prompt=prompt,
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        schema=ENTITY_EXTRACTION_SCHEMA,
        schema_name="entity_extraction",
        model=model
    )


def _entity_desc(entity: Dict[str, Any], entity_type: str) -> str:
    if entity_type == "characters":
        return f"appearance: {entity.get('appearance')}, clothing: {entity.get('clothing')}, colors: {entity.get('colors')}, features: {entity.get('features')}"
    elif entity_type == "locations":
        return f"traits: {entity.get('visual_traits')}, time: {entity.get('time_of_day')}, atmosphere: {entity.get('atmosphere')}"
    elif entity_type == "objects":
        return str(entity.get("description", ""))
    return str(entity)


def merge_chunk_entities(raw_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges all chunk extractions into one Master Story Bible:
    - Deduplicates by normalized name.
    - If two chunks describe the same entity differently, keep the FIRST description
      and log the conflict (entity name, both descriptions).
    - Locks the bible immediately (locked: True).
    """
    master_characters: Dict[str, Dict[str, Any]] = {}
    master_locations: Dict[str, Dict[str, Any]] = {}
    master_objects: Dict[str, Dict[str, Any]] = {}
    conflicts: List[Dict[str, Any]] = []

    for chunk_idx, chunk_data in enumerate(raw_chunks):
        # Characters
        for char in chunk_data.get("characters", []):
            name = char.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in master_characters:
                master_characters[key] = char
            else:
                first_desc = _entity_desc(master_characters[key], "characters")
                second_desc = _entity_desc(char, "characters")
                if first_desc != second_desc:
                    conflicts.append({
                        "entity_type": "character",
                        "name": name,
                        "first_description": first_desc,
                        "second_description": second_desc,
                        "chunk_index": chunk_idx
                    })

        # Locations
        for loc in chunk_data.get("locations", []):
            name = loc.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in master_locations:
                master_locations[key] = loc
            else:
                first_desc = _entity_desc(master_locations[key], "locations")
                second_desc = _entity_desc(loc, "locations")
                if first_desc != second_desc:
                    conflicts.append({
                        "entity_type": "location",
                        "name": name,
                        "first_description": first_desc,
                        "second_description": second_desc,
                        "chunk_index": chunk_idx
                    })

        # Objects
        for obj in chunk_data.get("objects", []):
            name = obj.get("name", "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in master_objects:
                master_objects[key] = obj
            else:
                first_desc = _entity_desc(master_objects[key], "objects")
                second_desc = _entity_desc(obj, "objects")
                if first_desc != second_desc:
                    conflicts.append({
                        "entity_type": "object",
                        "name": name,
                        "first_description": first_desc,
                        "second_description": second_desc,
                        "chunk_index": chunk_idx
                    })

    return {
        "locked": True,
        "characters": list(master_characters.values()),
        "locations": list(master_locations.values()),
        "objects": list(master_objects.values()),
        "conflicts": conflicts
    }


def generate_story_bible(
    scenes: List[Dict[str, Any]],
    engine: GroqEngine,
    model: Optional[str] = None,
    raw_entities_path: Optional[Union[str, Path]] = None,
    master_bible_path: Optional[Union[str, Path]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Step 2 Pipeline:
    Extracts raw entities per chunk, merges into Master Story Bible,
    and saves checkpoints 02_raw_entities.json and 03_master_bible.json.
    """
    chunks = chunk_scenes_for_bible(scenes)
    raw_chunks: List[Dict[str, Any]] = []

    for i, chunk in enumerate(chunks):
        res = extract_entities_for_chunk(chunk, engine, model=model)
        raw_chunks.append({
            "chunk_index": i,
            "scenes": [s.get("scene_id") for s in chunk],
            **res
        })

    master_bible = merge_chunk_entities(raw_chunks)

    if raw_entities_path:
        p_raw = Path(raw_entities_path).resolve()
        p_raw.parent.mkdir(parents=True, exist_ok=True)
        with open(p_raw, "w", encoding="utf-8") as f:
            json.dump(raw_chunks, f, indent=2, ensure_ascii=False)

    if master_bible_path:
        p_mb = Path(master_bible_path).resolve()
        p_mb.parent.mkdir(parents=True, exist_ok=True)
        with open(p_mb, "w", encoding="utf-8") as f:
            json.dump(master_bible, f, indent=2, ensure_ascii=False)

    return raw_chunks, master_bible

