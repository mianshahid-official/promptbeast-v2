"""
Image Prompt Generator for PromptBeast-V2 (Step 4).
Generates strictly 180–220 word prompts using:
(1) locked style block verbatim
(2) bible entries verbatim
(3) visual action & composition from storyboard beat
(4) exact narration words as context
(5) negative instruction (no text)

Rules:
- No headings, simple prose, no bullet-point lists
- Batching: 2 prompts per request (fits 950-token output cap)
- Strict JSON Schema output
- 180-220 word count gate: regenerate once if violated; if still failed, mark scene failed
- Skip-and-retry: failed scenes skipped, retried once at end, still-failed goes red with per-scene Retry
- Saves to 05_final_prompts.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.groq_engine import GroqEngine
from app.core.validator import ValidationError, lint_prompt_drift

PROMPT_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "scene_id": {"type": "string"},
                    "prompt": {"type": "string"},
                    "word_count": {"type": "integer"},
                    "bible_refs": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["scene_id", "prompt", "word_count", "bible_refs"],
                "additionalProperties": False
            }
        }
    },
    "required": ["scenes"],
    "additionalProperties": False
}


def load_style_master_prompt(style_id: str, styles_root: Union[str, Path] = "styles") -> str:
    """
    Loads master prompt for active style. Fails loudly if file is missing.
    """
    p = Path(styles_root) / style_id / "master_prompt.txt"
    if not p.exists():
        raise ValidationError(
            f"Active style master prompt missing: '{p.resolve()}'. "
            f"Every style must have its own master_prompt.txt."
        )
    with open(p, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValidationError(f"Active style master prompt is empty: '{p.resolve()}'.")
    return content


def format_bible_entries_for_scene(beat: Dict[str, Any], master_bible: Dict[str, Any]) -> str:
    """Extracts verbatim bible entries for entities referenced in this scene beat."""
    refs = set(str(r).strip().lower() for r in beat.get("bible_refs", []))
    entries = []

    for c in master_bible.get("characters", []):
        if c.get("name", "").strip().lower() in refs:
            entries.append(
                f"{c.get('name')}: appearance {c.get('appearance')}, clothing {c.get('clothing')}, colors {c.get('colors')}, features {c.get('features')}."
            )

    for l in master_bible.get("locations", []):
        if l.get("name", "").strip().lower() in refs:
            entries.append(
                f"{l.get('name')}: traits {l.get('visual_traits')}, time of day {l.get('time_of_day')}, atmosphere {l.get('atmosphere')}."
            )

    for o in master_bible.get("objects", []):
        if o.get("name", "").strip().lower() in refs:
            entries.append(f"{o.get('name')}: {o.get('description')}.")

    return " ".join(entries)


def compose_prompt_request(
    beats: List[Dict[str, Any]],
    master_bible: Dict[str, Any],
    style_anchor: str
) -> str:
    """Builds user prompt requesting exactly 2 scene prompts in strict order."""
    lines = [
        "Generate image prompts for the following scene beats.",
        "MANDATORY REQUIREMENTS FOR EACH SCENE PROMPT:",
        "- Word count must be strictly between 180 and 220 words. Count your words carefully.",
        "- Order of elements inside the prompt MUST be:",
        "  1. The style anchor block verbatim",
        "  2. Verbatim bible entries for characters, locations, and objects in this scene",
        "  3. Visual action and composition from the beat",
        "  4. Exact narration words as context",
        "  5. Direct negative instruction: 'Image must contain no text.'",
        "- Style: simple flowing direct prose only. NO headings (no '#', '##', or 'Scene:'), and NO bullet points or lists.",
        "- Strict character presence: only depict characters listed in the scene's bible entries. If the scene depicts background crowds, passengers, or zombies, explicitly specify that the protagonist is not present in the scene, and never depict the protagonist as an infected zombie.",
        ""
    ]

    for b in beats:
        s_id = b["scene_id"]
        narration = b["narration_words"]
        action = b["visual_action"]
        comp = b["composition"]
        b_entries = format_bible_entries_for_scene(b, master_bible)

        lines.append(f"--- SCENE {s_id} ---")
        lines.append(f"Style Anchor: {style_anchor.strip()}")
        lines.append(f"Bible Entries: {b_entries if b_entries else 'None'}")
        lines.append(f"Visual Action: {action}")
        lines.append(f"Composition: {comp}")
        lines.append(f"Narration: \"{narration}\"")
        lines.append("")

    return "\n".join(lines)


def generate_single_scene_prompt(
    beat: Dict[str, Any],
    master_bible: Dict[str, Any],
    style_id: str,
    engine: GroqEngine,
    model: Optional[str] = None,
    styles_root: Union[str, Path] = "styles"
) -> Dict[str, Any]:
    """Generates prompt for a single scene (used for retry or individual generation)."""
    master_prompt = load_style_master_prompt(style_id, styles_root)
    style_anchor = beat.get("style_anchor", "")
    user_prompt = compose_prompt_request([beat], master_bible, style_anchor)

    res = engine.generate_structured(
        prompt=user_prompt,
        system_prompt=master_prompt,
        schema=PROMPT_BATCH_SCHEMA,
        schema_name="scene_prompts",
        model=model
    )
    scenes = res.get("scenes", [])
    if not scenes:
        raise RuntimeError(f"No prompt returned for scene {beat.get('scene_id')}")

    item = scenes[0]
    prompt_str = item.get("prompt", "").strip()
    is_valid, w_count, violations = lint_prompt_drift(prompt_str, beat["scene_id"])

    # If invalid, regenerate once
    if not is_valid:
        regen_prompt = (
            f"{user_prompt}\n\nATTENTION: Previous attempt failed with word count {w_count}. "
            f"You MUST produce exactly 180 to 220 words. No headings, no bullets."
        )
        res_retry = engine.generate_structured(
            prompt=regen_prompt,
            system_prompt=master_prompt,
            schema=PROMPT_BATCH_SCHEMA,
            schema_name="scene_prompts",
            model=model
        )
        retry_scenes = res_retry.get("scenes", [])
        if retry_scenes:
            item = retry_scenes[0]
            prompt_str = item.get("prompt", "").strip()
            is_valid, w_count, violations = lint_prompt_drift(prompt_str, beat["scene_id"])

    status = "SUCCESS" if is_valid else "FAILED"
    return {
        "scene_id": beat["scene_id"],
        "start": beat["start"],
        "end": beat["end"],
        "duration": beat["duration"],
        "narration_words": beat["narration_words"],
        "prompt": prompt_str,
        "word_count": w_count,
        "status": status,
        "violations": violations,
        "bible_refs": item.get("bible_refs", beat.get("bible_refs", []))
    }


def generate_image_prompts(
    beats: List[Dict[str, Any]],
    master_bible: Dict[str, Any],
    style_id: str,
    engine: GroqEngine,
    model: Optional[str] = None,
    checkpoint_path: Optional[Union[str, Path]] = None,
    styles_root: Union[str, Path] = "styles"
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Step 4 Pipeline:
    Batches 2 prompts per request to fit 950-token output cap.
    Enforces 180-220 word count gate with single regeneration.
    Implements skip-and-retry at end.
    Saves to 05_final_prompts.json.
    """
    master_prompt = load_style_master_prompt(style_id, styles_root)
    beat_lookup = {b["scene_id"]: b for b in beats}
    final_prompts: List[Dict[str, Any]] = []
    failed_scenes: List[Dict[str, Any]] = []

    # Process in batches of 2
    for i in range(0, len(beats), 2):
        batch = beats[i:i + 2]
        style_anchor = batch[0].get("style_anchor", "")
        user_prompt = compose_prompt_request(batch, master_bible, style_anchor)

        try:
            res = engine.generate_structured(
                prompt=user_prompt,
                system_prompt=master_prompt,
                schema=PROMPT_BATCH_SCHEMA,
                schema_name="scene_prompts",
                model=model
            )
            ret_scenes = {s["scene_id"]: s for s in res.get("scenes", [])}
        except Exception as e:
            ret_scenes = {}

        for b in batch:
            s_id = b["scene_id"]
            item = ret_scenes.get(s_id)
            if not item:
                # Scene skipped in batch; mark failed for final retry
                failed_scenes.append(b)
                continue

            prompt_str = item.get("prompt", "").strip()
            is_valid, w_count, violations = lint_prompt_drift(prompt_str, s_id)

            if not is_valid:
                # Regenerate once
                try:
                    retry_res = generate_single_scene_prompt(
                        b, master_bible, style_id, engine, model=model, styles_root=styles_root
                    )
                    final_prompts.append(retry_res)
                    if retry_res["status"] == "FAILED":
                        failed_scenes.append(b)
                except Exception:
                    final_prompts.append({
                        "scene_id": s_id,
                        "start": b["start"],
                        "end": b["end"],
                        "duration": b["duration"],
                        "narration_words": b["narration_words"],
                        "prompt": prompt_str,
                        "word_count": w_count,
                        "status": "FAILED",
                        "violations": violations,
                        "bible_refs": b.get("bible_refs", [])
                    })
                    failed_scenes.append(b)
            else:
                final_prompts.append({
                    "scene_id": s_id,
                    "start": b["start"],
                    "end": b["end"],
                    "duration": b["duration"],
                    "narration_words": b["narration_words"],
                    "prompt": prompt_str,
                    "word_count": w_count,
                    "status": "SUCCESS",
                    "violations": [],
                    "bible_refs": item.get("bible_refs", b.get("bible_refs", []))
                })

    # Skip-and-retry: Retry all failed scenes once at the end
    still_failed: List[Dict[str, Any]] = []
    if failed_scenes:
        for b in failed_scenes:
            try:
                retried = generate_single_scene_prompt(
                    b, master_bible, style_id, engine, model=model, styles_root=styles_root
                )
                # Replace in final_prompts or append
                idx = next((k for k, p in enumerate(final_prompts) if p["scene_id"] == b["scene_id"]), None)
                if idx is not None:
                    final_prompts[idx] = retried
                else:
                    final_prompts.append(retried)

                if retried["status"] == "FAILED":
                    still_failed.append(retried)
            except Exception as e:
                still_failed.append(b)

    # Sort final prompts by scene_id
    final_prompts.sort(key=lambda x: str(x.get("scene_id", "000")))

    # Save checkpoint 05_final_prompts.json
    if checkpoint_path:
        cp = Path(checkpoint_path).resolve()
        cp.parent.mkdir(parents=True, exist_ok=True)
        with open(cp, "w", encoding="utf-8") as f:
            json.dump(final_prompts, f, indent=2, ensure_ascii=False)

    return final_prompts, still_failed

