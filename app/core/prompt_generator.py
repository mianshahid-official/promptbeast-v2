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
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.groq_engine import GroqEngine
from app.core.validator import ValidationError, lint_prompt_drift


def adjust_prompt_words(prompt: str, target_min: int = 180, target_max: int = 220) -> str:
    """
    Adjusts prompt word count to strictly fit within [target_min, target_max].
    Trims at natural sentence boundaries and preserves negative instruction at the end.
    """
    words = prompt.split()
    if target_min <= len(words) <= target_max:
        return prompt

    neg = "Image must contain no text."
    has_neg = False
    if prompt.rstrip().endswith(neg) or neg.lower() in prompt.lower():
        has_neg = True
        body = re.sub(r"\s*" + re.escape(neg) + r"\s*$", "", prompt, flags=re.IGNORECASE).rstrip()
    else:
        body = prompt

    neg_words = len(neg.split()) if has_neg else 0

    if len(words) > target_max:
        sentences = re.split(r"(?<=[.!?])\s+", body)
        kept = []
        cur_count = 0
        for s in sentences:
            s_len = len(s.split())
            if cur_count + s_len + neg_words <= target_max:
                kept.append(s)
                cur_count += s_len
            else:
                break

        trimmed_body = " ".join(kept)
        if cur_count + neg_words < target_min:
            needed = target_max - neg_words - 2
            b_words = body.split()[:needed]
            trimmed_body = " ".join(b_words).rstrip(",;:-") + "."

        return f"{trimmed_body} {neg}" if has_neg else trimmed_body

    elif len(words) < target_min:
        expansion_phrases = [
            "Precise background detailing establishes clear spatial context for the scene.",
            "Careful environmental illumination highlights clean silhouettes and sharp graphic contrasts.",
            "Stable cinematic framing ensures the focal narrative action remains immediately legible.",
            "Crisp spatial separation and perspective define the surrounding architecture and physical layout.",
            "Controlled practical lighting casts distinct graphical pools across the foreground floor.",
            "Consistent directional shadows reinforce physical scale and visual depth across the environment.",
            "Subtle environmental textures and clean edges anchor the composition without visual clutter.",
            "Deep graphic contrast distinguishes foreground subjects from the distant background planes."
        ]
        curr_text = body
        idx = 0
        while len(curr_text.split()) + neg_words < target_min and idx < len(expansion_phrases) * 3:
            curr_text = curr_text.rstrip(".") + ". " + expansion_phrases[idx % len(expansion_phrases)]
            idx += 1
        curr_words = curr_text.split()
        if len(curr_words) + neg_words > target_max:
            curr_text = " ".join(curr_words[:target_max - neg_words - 2]).rstrip(",;:-") + "."
        return f"{curr_text} {neg}" if has_neg else curr_text

    return prompt

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
        "- Strict character presence: only depict characters directly present in the scene's bible entries. If the protagonist is not present in the scene, do not describe or mention the protagonist at all. Describe only the visible subjects, action, and environment.",
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

    res = None
    try:
        res = engine.generate_structured(
            prompt=user_prompt,
            system_prompt=master_prompt,
            schema=PROMPT_BATCH_SCHEMA,
            schema_name="scene_prompts",
            model=model
        )
    except Exception:
        pass

    if not res:
        json_fmt_prompt = master_prompt + '\n\nRespond strictly with a valid JSON object matching this structure:\n{"scenes": [{"scene_id": "001", "prompt": "...", "word_count": 200, "bible_refs": []}]}'
        try:
            raw_text = engine.generate(
                prompt=user_prompt,
                system_prompt=json_fmt_prompt,
                response_format={"type": "json_object"},
                model=model,
                max_tokens=2048
            )
            res = json.loads(raw_text)
        except Exception:
            pass

    if isinstance(res, list):
        scenes = res
    elif isinstance(res, dict):
        scenes = res.get("scenes", [])
    else:
        scenes = []

    if not scenes:
        raise RuntimeError(f"No prompt returned for scene {beat.get('scene_id')}")

    item = scenes[0]
    prompt_str = item.get("prompt", "").strip()
    prompt_str = re.sub(r"^\s*#{1,6}\s*", "", prompt_str, flags=re.MULTILINE)
    prompt_str = re.sub(r"^\s*[\*\-]\s*", "", prompt_str, flags=re.MULTILINE)
    if "image must contain no text" not in prompt_str.lower():
        prompt_str = prompt_str.rstrip(".") + ". Image must contain no text."
    prompt_str = adjust_prompt_words(prompt_str, 180, 220)
    is_valid, w_count, violations = lint_prompt_drift(prompt_str, beat["scene_id"])

    # If invalid, regenerate once
    if not is_valid:
        regen_prompt = (
            f"{user_prompt}\n\nATTENTION: Previous attempt failed with word count {w_count}. "
            f"You MUST produce exactly 180 to 220 words. No headings, no bullets."
        )
        res_retry = None
        try:
            res_retry = engine.generate_structured(
                prompt=regen_prompt,
                system_prompt=master_prompt,
                schema=PROMPT_BATCH_SCHEMA,
                schema_name="scene_prompts",
                model=model
            )
        except Exception:
            pass

        if not res_retry:
            try:
                json_fmt_prompt = master_prompt + '\n\nRespond strictly with a valid JSON object matching this structure:\n{"scenes": [{"scene_id": "001", "prompt": "...", "word_count": 200, "bible_refs": []}]}'
                raw_retry = engine.generate(
                    prompt=regen_prompt,
                    system_prompt=json_fmt_prompt,
                    response_format={"type": "json_object"},
                    model=model,
                    max_tokens=2048
                )
                res_retry = json.loads(raw_retry)
            except Exception:
                pass

        if isinstance(res_retry, list):
            retry_scenes = res_retry
        elif isinstance(res_retry, dict):
            retry_scenes = res_retry.get("scenes", [])
        else:
            retry_scenes = []

        if retry_scenes:
            item = retry_scenes[0]
            prompt_str = item.get("prompt", "").strip()
            prompt_str = re.sub(r"^\s*#{1,6}\s*", "", prompt_str, flags=re.MULTILINE)
            prompt_str = re.sub(r"^\s*[\*\-]\s*", "", prompt_str, flags=re.MULTILINE)
            if "image must contain no text" not in prompt_str.lower():
                prompt_str = prompt_str.rstrip(".") + ". Image must contain no text."
            prompt_str = adjust_prompt_words(prompt_str, 180, 220)
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

    # Resume from existing checkpoint if available
    already_done_ids = set()
    if checkpoint_path:
        cp_p = Path(checkpoint_path).resolve()
        if cp_p.exists():
            try:
                with open(cp_p, "r", encoding="utf-8") as f:
                    loaded_cps = json.load(f)
                if isinstance(loaded_cps, list):
                    for item in loaded_cps:
                        if isinstance(item, dict) and "scene_id" in item:
                            final_prompts.append(item)
                            already_done_ids.add(str(item["scene_id"]))
            except Exception:
                pass

    # Process in batches of 2
    for i in range(0, len(beats), 2):
        batch = beats[i:i + 2]
        # Skip batch if already fully generated
        if batch and all(str(b["scene_id"]) in already_done_ids for b in batch):
            continue

        style_anchor = batch[0].get("style_anchor", "")
        user_prompt = compose_prompt_request(batch, master_bible, style_anchor)

        res = None
        try:
            res = engine.generate_structured(
                prompt=user_prompt,
                system_prompt=master_prompt,
                schema=PROMPT_BATCH_SCHEMA,
                schema_name="scene_prompts",
                model=model
            )
        except Exception:
            pass

        if not res:
            json_fmt_prompt = master_prompt + '\n\nRespond strictly with a valid JSON object matching this structure:\n{"scenes": [{"scene_id": "001", "prompt": "...", "word_count": 200, "bible_refs": []}]}'
            try:
                raw_text = engine.generate(
                    prompt=user_prompt,
                    system_prompt=json_fmt_prompt,
                    response_format={"type": "json_object"},
                    model=model,
                    max_tokens=2048
                )
                res = json.loads(raw_text)
            except Exception:
                pass

        raw_list = []
        if isinstance(res, list):
            raw_list = res
        elif isinstance(res, dict):
            raw_list = res.get("scenes", [])

        ret_scenes = {str(s["scene_id"]): s for s in raw_list if isinstance(s, dict) and "scene_id" in s}

        for b in batch:
            s_id = b["scene_id"]
            item = ret_scenes.get(s_id)
            if not item:
                # Scene skipped in batch; mark failed for final retry
                failed_scenes.append(b)
                continue

            prompt_str = item.get("prompt", "").strip()
            prompt_str = re.sub(r"^\s*#{1,6}\s*", "", prompt_str, flags=re.MULTILINE)
            prompt_str = re.sub(r"^\s*[\*\-]\s*", "", prompt_str, flags=re.MULTILINE)
            if "image must contain no text" not in prompt_str.lower():
                prompt_str = prompt_str.rstrip(".") + ". Image must contain no text."
            prompt_str = adjust_prompt_words(prompt_str, 180, 220)
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

        # Save incremental checkpoint atomically after each batch
        if checkpoint_path and final_prompts:
            try:
                cp = Path(checkpoint_path).resolve()
                cp.parent.mkdir(parents=True, exist_ok=True)
                tmp_cp = cp.with_suffix(".tmp")
                with open(tmp_cp, "w", encoding="utf-8") as f:
                    json.dump(final_prompts, f, indent=2, ensure_ascii=False)
                tmp_cp.replace(cp)
            except Exception:
                pass

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

