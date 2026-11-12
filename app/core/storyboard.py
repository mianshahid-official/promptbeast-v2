"""
Storyboard Beat Planner for PromptBeast-V2 (Step 3).
Plans one visual beat per scene using ONLY locked bible entries.
Copies exact narration words, sets composition, and logs warnings for missing entities.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.groq_engine import GroqEngine

BEATS_PLANNING_SCHEMA = {
    "type": "object",
    "properties": {
        "beats": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "visual_action": {"type": "string"},
                    "composition": {"type": "string"},
                    "bible_refs": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "missing_elements": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["scene_id", "visual_action", "composition", "bible_refs", "missing_elements"],
                "additionalProperties": False
            }
        }
    },
    "required": ["beats"],
    "additionalProperties": False
}

STORYBOARD_SYSTEM_PROMPT = """You are a master storyboard artist and film director.
For each narration scene, plan exactly ONE visual beat:
1. visual_action: What happens visually on screen to illustrate the exact narration words.
2. composition: Camera framing and motion (e.g. wide establishing shot, medium shot, close up, slow zoom-in, pan left).
3. bible_refs: Names of characters, locations, or props from the LOCKED BIBLE that appear in this beat.
4. missing_elements: If a scene needs an entity not present in the locked bible, describe it generically in visual_action and list its generic name here. DO NOT invent new named entities.
Strictly adhere to the provided JSON schema."""


def build_bible_context(master_bible: Dict[str, Any]) -> str:
    """Formats locked master bible into compact text context for storyboard planning."""
    lines = ["=== LOCKED STORY BIBLE ==="]
    chars = master_bible.get("characters", [])
    if chars:
        lines.append("Characters:")
        for c in chars:
            lines.append(f"- {c.get('name')}: {c.get('appearance')}, clothing: {c.get('clothing')}, colors: {c.get('colors')}, features: {c.get('features')}")

    locs = master_bible.get("locations", [])
    if locs:
        lines.append("Locations:")
        for l in locs:
            lines.append(f"- {l.get('name')}: {l.get('visual_traits')}, time: {l.get('time_of_day')}, atmosphere: {l.get('atmosphere')}")

    objs = master_bible.get("objects", [])
    if objs:
        lines.append("Objects/Props:")
        for o in objs:
            lines.append(f"- {o.get('name')}: {o.get('description')}")

    return "\n".join(lines)


def plan_storyboard_beats(
    scenes: List[Dict[str, Any]],
    master_bible: Dict[str, Any],
    style_anchor: str,
    engine: GroqEngine,
    model: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    batch_size: int = 4
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Step 3 Pipeline:
    Plans one beat per scene, referencing locked bible and style anchor.
    Saves to 04_storyboard_beats.json.
    """
    bible_ctx = build_bible_context(master_bible)
    known_names = set()
    for c in master_bible.get("characters", []):
        known_names.add(c.get("name", "").strip().lower())
    for l in master_bible.get("locations", []):
        known_names.add(l.get("name", "").strip().lower())
    for o in master_bible.get("objects", []):
        known_names.add(o.get("name", "").strip().lower())

    all_beats: List[Dict[str, Any]] = []
    warnings: List[str] = []

    # Map scenes by scene_id
    scene_lookup = {sc["scene_id"]: sc for sc in scenes}

    # Batch scenes for LLM beat planning
    for i in range(0, len(scenes), batch_size):
        batch = scenes[i:i + batch_size]
        scenes_prompt = ["=== SCENES TO PLAN ==="]
        for sc in batch:
            scenes_prompt.append(
                f"Scene {sc['scene_id']} [{sc['start']:.2f}s - {sc['end']:.2f}s]: \"{sc['text']}\""
            )

        full_prompt = f"{bible_ctx}\n\nStyle Anchor:\n{style_anchor}\n\n" + "\n".join(scenes_prompt)
        res = engine.generate_structured(
            prompt=full_prompt,
            system_prompt=STORYBOARD_SYSTEM_PROMPT,
            schema=BEATS_PLANNING_SCHEMA,
            schema_name="storyboard_beats",
            model=model
        )

        planned_beats = {b["scene_id"]: b for b in res.get("beats", [])}

        for sc in batch:
            s_id = sc["scene_id"]
            p_beat = planned_beats.get(s_id, {})
            visual_action = p_beat.get("visual_action") or f"Visual depiction of {sc['text'][:60]}..."
            composition = p_beat.get("composition") or "Medium shot, eye-level angle, slow zoom-in."
            raw_refs = p_beat.get("bible_refs", [])
            missing = p_beat.get("missing_elements", [])

            # Filter valid bible refs
            valid_refs = []
            for r in raw_refs:
                if str(r).strip().lower() in known_names:
                    valid_refs.append(r)
                else:
                    msg = f"Scene {s_id}: referenced unlisted entity '{r}' not in locked bible; treating generically."
                    warnings.append(msg)

            for m in missing:
                msg = f"Scene {s_id}: missing bible entity '{m}'; described generically."
                warnings.append(msg)

            beat_entry = {
                "scene_id": s_id,
                "start": sc["start"],
                "end": sc["end"],
                "duration": sc["duration"],
                "narration_words": sc["text"],  # exact, copied from scene
                "visual_action": visual_action,
                "composition": composition,
                "bible_refs": valid_refs,
                "style_anchor": style_anchor.strip()
            }
            all_beats.append(beat_entry)

    out_data = {
        "beats": all_beats,
        "warnings": warnings
    }

    if checkpoint_path:
        cp = Path(checkpoint_path).resolve()
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)

    return all_beats, warnings
