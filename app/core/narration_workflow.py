"""
Narration-Aware Workflow Engine for PromptBeast.
Handles JSON script import, visual ideas alignment, dynamic story bible extraction,
and batch image prompt generation via Groq strictly adhering to user-selected visual styles.
"""
import json
import re
import difflib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from app.models.scene import VisualScene
from app.models.timeline import NarrationData, WordTimestamp
from app.presets.loader import PresetManager
from app.ai.groq_engine import GroqEngine, GroqMultiKeyManager


# =============================================================================
# STEP 1: IMPORT SCRIPT JSON & VALIDATION
# =============================================================================

class JSONImportResult:
    def __init__(
        self,
        success: bool,
        message: str,
        story_text: str = "",
        narration: Optional[NarrationData] = None,
        scenes: Optional[List[VisualScene]] = None,
        raw_data: Optional[Dict[str, Any]] = None
    ):
        self.success = success
        self.message = message
        self.story_text = story_text
        self.narration = narration
        self.scenes = scenes or []
        self.raw_data = raw_data or {}


def import_script_json(file_path_or_str: str) -> JSONImportResult:
    """
    Parses and validates a script JSON file or string.
    Supports:
    1. Whisper/Speech JSON: {"text": "...", "words": [{"word": "...", "start": 0.0, "end": 0.5}, ...]}
    2. Storyboard JSON: {"scenes": [{"scene_id": "001", "start": 0.0, "end": 4.0, "narration": "...", "prompt": "..."}]}
    3. Segments/Subtitle JSON: {"segments": [{"start": 0.0, "end": 3.0, "text": "..."}]}
    4. Plain dictionary with text/words.
    """
    p = Path(file_path_or_str)
    raw_content = ""
    if p.exists() and p.is_file():
        try:
            raw_content = p.read_text(encoding="utf-8")
        except Exception as e:
            return JSONImportResult(False, f"Could not read file: {e}")
    else:
        raw_content = file_path_or_str

    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as jde:
        return JSONImportResult(False, f"Malformed JSON: {jde.msg} (line {jde.lineno}, col {jde.colno})")

    if not isinstance(data, dict):
        return JSONImportResult(False, "JSON root must be an object (dictionary).")

    # Format 1: Whisper / Speech-to-Text format with 'text' and 'words'
    if "words" in data and isinstance(data["words"], list):
        text = data.get("text", "")
        word_list: List[WordTimestamp] = []
        for w in data["words"]:
            if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                word_list.append(WordTimestamp(
                    word=str(w["word"]),
                    start=float(w["start"]),
                    end=float(w["end"])
                ))

        if not word_list:
            return JSONImportResult(False, "JSON 'words' array was found but contained no valid word timestamps.")

        # Reconstruct text if empty
        if not text:
            text = " ".join(w.word for w in word_list).strip()

        duration = float(data.get("duration", word_list[-1].end))
        narration = NarrationData(
            text=text,
            words=word_list,
            duration_seconds=duration,
            is_estimated=False
        )
        return JSONImportResult(
            success=True,
            message=f"Successfully imported audio script JSON: {len(word_list)} words, duration {duration:.1f}s.",
            story_text=text,
            narration=narration,
            raw_data=data
        )

    # Format 2: Storyboard scenes array
    if "scenes" in data and isinstance(data["scenes"], list):
        scenes: List[VisualScene] = []
        full_text_parts = []
        for i, s in enumerate(data["scenes"]):
            if not isinstance(s, dict):
                continue
            s_id = str(s.get("scene_id") or f"{i+1:03d}")
            start = float(s.get("start", 0.0))
            end = float(s.get("end", start + 3.0))
            dur = float(s.get("duration", max(0.5, end - start)))
            narr = str(s.get("narration") or s.get("narration_excerpt") or "")
            visual_idea = str(s.get("visual_idea") or s.get("visual_subject") or "")
            prompt = str(s.get("prompt") or "")
            if narr:
                full_text_parts.append(narr)

            scenes.append(VisualScene(
                scene_id=s_id,
                index=i + 1,
                start=start,
                end=end,
                duration=dur,
                narration_excerpt=narr,
                visual_subject=visual_idea or narr[:60],
                visual_idea=visual_idea,
                prompt=prompt,
                generation_status="generated" if prompt else "pending"
            ))

        story_text = "\n\n".join(full_text_parts)
        return JSONImportResult(
            success=True,
            message=f"Successfully imported storyboard JSON with {len(scenes)} predefined scenes.",
            story_text=story_text,
            scenes=scenes,
            raw_data=data
        )

    # Format 3: Segments array (like Whisper segments)
    if "segments" in data and isinstance(data["segments"], list):
        word_list = []
        full_text_parts = []
        for seg in data["segments"]:
            if not isinstance(seg, dict):
                continue
            seg_text = str(seg.get("text", "")).strip()
            if seg_text:
                full_text_parts.append(seg_text)
            seg_words = seg.get("words", [])
            for w in seg_words:
                if isinstance(w, dict) and "word" in w and "start" in w and "end" in w:
                    word_list.append(WordTimestamp(
                        word=str(w["word"]),
                        start=float(w["start"]),
                        end=float(w["end"])
                    ))

        story_text = data.get("text", " ".join(full_text_parts))
        if word_list:
            duration = word_list[-1].end
            narration = NarrationData(
                text=story_text,
                words=word_list,
                duration_seconds=duration,
                is_estimated=False
            )
            return JSONImportResult(
                success=True,
                message=f"Successfully imported segmented script JSON with {len(word_list)} word timestamps.",
                story_text=story_text,
                narration=narration,
                raw_data=data
            )
        else:
            return JSONImportResult(
                success=True,
                message=f"Imported script segments without word-level timestamps ({len(data['segments'])} segments).",
                story_text=story_text,
                raw_data=data
            )

    # Format 4: Fallback generic text object
    if "text" in data and isinstance(data["text"], str):
        story_text = data["text"].strip()
        if story_text:
            return JSONImportResult(
                success=True,
                message=f"Imported plain text script from JSON ({len(story_text)} chars).",
                story_text=story_text,
                raw_data=data
            )

    return JSONImportResult(
        False,
        "JSON structure not recognized. Expected either a speech transcription with 'words', a storyboard with 'scenes', or 'segments'."
    )


# =============================================================================
# STEP 2: NARRATION & VISUAL IDEAS PARSING AND ALIGNMENT
# =============================================================================

def normalize_text_token(t: str) -> str:
    """Normalizes number words and removes punctuation for robust speech-to-script alignment."""
    t = t.lower()
    t = re.sub(r"\b(sixty|60)\b", "60", t)
    t = re.sub(r"\b(forty|40)\b", "40", t)
    t = re.sub(r"\b(twenty|20)\b", "20", t)
    t = re.sub(r"\b(nineteen|19)\b", "19", t)
    t = re.sub(r"\b(fifteen|15)\b", "15", t)
    t = re.sub(r"\b(eleven|11)\b", "11", t)
    t = re.sub(r"\b(ten|10)\b", "10", t)
    t = re.sub(r"\b(four|4)\b", "4", t)
    t = re.sub(r"\b(three|3)\b", "3", t)
    t = re.sub(r"\b(two|2)\b", "2", t)
    t = re.sub(r"\b(one|1)\b", "1", t)
    t = re.sub(r"[^\w\s]", "", t)
    return " ".join(t.split())


def parse_visual_breakdown_text(raw_text: str) -> List[Dict[str, str]]:
    """
    Parses user-supplied narration and visual ideas.
    Supports:
    1. 'Visual 1 Words: <narration> Visual: <visual idea>'
    2. '1. Narration: ... Visual: ...'
    3. 'Words: ... \n Visual: ...'
    4. Plain paragraphs if no visual tags are found.
    """
    items = []
    # Pattern 1: Visual X Words: ... Visual: ...
    pattern1 = re.compile(
        r"Visual\s+(\d+)\s+Words:\s*(.*?)\s*Visual:\s*(.*?)(?=Visual\s+\d+\s+Words:|$)",
        re.DOTALL | re.IGNORECASE
    )
    matches1 = pattern1.findall(raw_text)
    if matches1:
        for num, words, vis in matches1:
            items.append({
                "id": str(num).strip(),
                "words": words.strip(),
                "visual_idea": vis.strip()
            })
        return items

    # Pattern 2: Lines with "Words:" and "Visual:"
    pattern2 = re.compile(
        r"(?:Scene|Shot|Visual)?\s*(\d+)?[.:]?\s*Words?:\s*(.*?)\s*Visuals?:\s*(.*?)(?=(?:Scene|Shot|Visual|\d+[.:])\s*Words?:|$)",
        re.DOTALL | re.IGNORECASE
    )
    matches2 = pattern2.findall(raw_text)
    if matches2 and len(matches2) > 1:
        for idx, (num, words, vis) in enumerate(matches2):
            s_id = num.strip() if num.strip() else str(idx + 1)
            items.append({
                "id": s_id,
                "words": words.strip(),
                "visual_idea": vis.strip()
            })
        return items

    # Pattern 3: Paragraph fallback
    paras = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    for i, p in enumerate(paras):
        items.append({
            "id": str(i + 1),
            "words": p,
            "visual_idea": f"Visual scene illustrating: {p[:80]}"
        })
    return items


def align_visuals_with_timestamps(
    visual_items: List[Dict[str, str]],
    narration: Optional[NarrationData] = None,
    total_audio_duration: Optional[float] = None
) -> List[VisualScene]:
    """
    Combines user-supplied visual breakdown with exact word-level timestamps from narration.
    Uses monotonic sequence alignment with sequence matching to achieve 100% synchronization.
    If no word timestamps exist, computes duration dynamically from pacing rules.
    """
    scenes: List[VisualScene] = []
    if not visual_items:
        return scenes

    if narration and narration.words:
        words = narration.words
        speech_norm = [normalize_text_token(w.word) for w in words]
        curr_w_idx = 0

        for idx, item in enumerate(visual_items):
            s_num = item.get("id") or str(idx + 1)
            try:
                s_id = f"{int(s_num):03d}"
            except ValueError:
                s_id = f"{idx + 1:03d}"

            v_words = item["words"]
            v_vis = item["visual_idea"]
            v_norm = normalize_text_token(v_words).split()

            if not v_norm:
                continue

            best_start = curr_w_idx
            best_len = max(1, len(v_norm))
            best_score = 0.0

            search_limit = min(len(words), curr_w_idx + 80)
            target_str = " ".join(v_norm)

            for s_idx in range(curr_w_idx, search_limit):
                for length in range(max(1, len(v_norm) - 2), min(len(v_norm) + 4, len(words) - s_idx + 1)):
                    sub = " ".join(speech_norm[s_idx:s_idx + length])
                    score = difflib.SequenceMatcher(None, target_str, sub).ratio()
                    if score > best_score:
                        best_score = score
                        best_start = s_idx
                        best_len = length
                    if score >= 0.98:
                        break
                if best_score >= 0.98:
                    break

            end_idx = min(len(words) - 1, best_start + best_len - 1)
            start_time = words[best_start].start
            end_time = words[end_idx].end

            # Avoid zero duration
            if end_time <= start_time:
                end_time = start_time + 1.2

            curr_w_idx = end_idx + 1

            scenes.append(VisualScene(
                scene_id=s_id,
                index=idx + 1,
                start=round(start_time, 2),
                end=round(end_time, 2),
                duration=round(end_time - start_time, 2),
                narration_excerpt=v_words,
                visual_subject=v_vis[:80],
                visual_idea=v_vis,
                prompt="",
                generation_status="pending",
                start_word_index=best_start,
                end_word_index=end_idx
            ))

    else:
        # Fallback: estimate timings proportionally
        total_dur = total_audio_duration or float(len(visual_items) * 3.5)
        avg_dur = max(1.5, total_dur / len(visual_items))
        curr_time = 0.0
        for idx, item in enumerate(visual_items):
            s_num = item.get("id") or str(idx + 1)
            try:
                s_id = f"{int(s_num):03d}"
            except ValueError:
                s_id = f"{idx + 1:03d}"

            start_t = curr_time
            end_t = curr_time + avg_dur
            curr_time = end_t
            scenes.append(VisualScene(
                scene_id=s_id,
                index=idx + 1,
                start=round(start_t, 2),
                end=round(end_t, 2),
                duration=round(avg_dur, 2),
                narration_excerpt=item["words"],
                visual_subject=item["visual_idea"][:80],
                visual_idea=item["visual_idea"],
                prompt="",
                generation_status="pending"
            ))

    return scenes


# =============================================================================
# STEP 3: DYNAMIC STORY BIBLE & INTELLIGENCE EXTRACTION
# =============================================================================

def load_bible_extraction_system_prompt() -> str:
    p = Path("config/system_prompts/story_bible_extraction.txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "You are an expert story analyst. Extract comprehensive story, character, environment, and style bibles in JSON."


def extract_story_bibles_groq(
    story_text: str,
    visual_ideas_summary: str = "",
    model: str = "qwen/qwen3.8-27b",
    engine: Optional[GroqEngine] = None
) -> Dict[str, Any]:
    """
    Extracts dynamic story bibles (Story, Character, Environment, Props, Visual Style)
    via Groq LLM inference without any hardcoded story assumptions.
    """
    groq_eng = engine or GroqEngine(default_model=model)
    system_prompt = load_bible_extraction_system_prompt()

    user_prompt = f"""ANALYZE THE FOLLOWING STORY SCRIPT AND EXTRACT FULL PRODUCTION BIBLES:

### SCRIPT NARRATION:
{story_text}

### USER-SUPPLIED VISUAL IDEAS / BEATS:
{visual_ideas_summary[:3500] if visual_ideas_summary else 'No additional visual notes provided.'}

Extract comprehensive, highly detailed production bibles strictly matching the required JSON schema.
Ensure character appearances, environments, props, and visual continuity rules are thoroughly defined.
Return ONLY valid JSON.
"""

    response_text = groq_eng.generate_text(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.2,
        max_tokens=2500,
        model=model,
        stage_name="BIBLE_EXTRACTION"
    )

    clean_json = response_text.strip()
    if clean_json.startswith("```"):
        clean_json = re.sub(r"^```(?:json)?\s*", "", clean_json)
        clean_json = re.sub(r"\s*```$", "", clean_json)

    try:
        bibles_data = json.loads(clean_json)
        return bibles_data
    except json.JSONDecodeError as jde:
        m = re.search(r"(\{.*\})", clean_json, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                pass
        raise RuntimeError(f"Failed to parse Groq story bibles response as JSON: {jde}. Raw snippet: {clean_json[:200]}")


def parse_custom_bible_text(bible_text: str) -> Dict[str, Any]:
    """
    Parses human-readable/markdown production bibles into structured dictionaries.
    Directly supports sections:
    1. CHARACTER BIBLE (CHARACTER X: NAME, Role, Appearance, Clothing, Important Props, Visual Continuity)
    2. ENVIRONMENT BIBLE (LOCATION X: NAME, Architecture, Interior, Progression, Details)
    3. OBJECT AND PROP BIBLE (Equipment, Food & Water, Personal Items, Structural)
    4. CINEMATIC STYLE BIBLE (Genre, Lighting, Color, Composition)
    5. CHRONOLOGY AND CONTINUITY BIBLE (Timeline events)
    6. IMPORTANT VISUAL CONTINUITY RULES
    7. FINAL STORY VISUAL IDENTITY
    """
    bibles: Dict[str, Any] = {
        "story_bible": {
            "premise": "Realistic survival horror documentary.",
            "summary": "Survival struggle in an abandoned shopping mall.",
            "genre_and_tone": "Grounded survival documentary horror",
            "setting_period": "Modern Day",
            "visual_identity": ""
        },
        "characters": [],
        "environments": [],
        "key_objects": [],
        "visual_style_bible": {
            "color_palette": "Cold grey, dark blue, black, concrete grey, muted green",
            "lighting_rules": "Flashlight beams, emergency lights, lightning, and deep shadows",
            "camera_guidelines": "Wide establishing, medium character, close-up focal objects, mezzanine views"
        },
        "visual_continuity_rules": [],
        "chronological_events": []
    }

    # 1. CHARACTER BIBLE
    char_match = re.search(r"(?:1\.?\s*)?CHARACTER BIBLE(.*?)(?=\d+\.\s*ENVIRONMENT|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if char_match:
        c_text = char_match.group(1)
        c_blocks = re.split(r"CHARACTER\s+\d+:\s*", c_text, flags=re.IGNORECASE)[1:]
        for block in c_blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if not lines:
                continue
            name = lines[0]
            role = ""
            app_lines = []
            cloth_lines = []
            prop_lines = []
            rule_lines = []
            curr_mode = "app"

            for l in lines[1:]:
                llow = l.lower()
                if llow.startswith("role:"):
                    role = l.split(":", 1)[1].strip()
                elif llow.startswith(("appearance:", "build:", "age:", "gender:")):
                    curr_mode = "app"
                    app_lines.append(l)
                elif llow.startswith("clothing:"):
                    curr_mode = "cloth"
                elif llow.startswith("important props:"):
                    curr_mode = "props"
                elif llow.startswith(("visual continuity:", "important rules:", "continuity:")):
                    curr_mode = "rules"
                    rule_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith("personality through visual expression:"):
                    curr_mode = "app"
                    app_lines.append(l)
                else:
                    if curr_mode == "cloth":
                        cloth_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "props":
                        prop_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "rules":
                        rule_lines.append(l.lstrip("-*• "))
                    else:
                        app_lines.append(l)

            bibles["characters"].append({
                "name": name,
                "role": role or "Key Figure",
                "physical_description": " ".join(app_lines).strip() or "Realistic physical features.",
                "clothing_and_gear": ", ".join(cloth_lines).strip() or "Practical attire.",
                "important_props": ", ".join(prop_lines).strip(),
                "consistency_rules": " ".join(rule_lines).strip() or f"Keep {name} appearance consistent."
            })

    # 2. ENVIRONMENT BIBLE
    env_match = re.search(r"(?:2\.?\s*)?ENVIRONMENT BIBLE(.*?)(?=\d+\.\s*OBJECT|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if env_match:
        e_text = env_match.group(1)
        e_blocks = re.split(r"LOCATION\s+\d+:\s*", e_text, flags=re.IGNORECASE)[1:]
        for block in e_blocks:
            lines = [l.strip() for l in block.strip().split("\n") if l.strip()]
            if not lines:
                continue
            name = lines[0]
            arch_lines = []
            features_lines = []
            lighting_lines = []
            details_lines = []
            curr_mode = "arch"

            for l in lines[1:]:
                llow = l.lower()
                if llow.startswith(("architecture:", "location:", "primary setting:")):
                    curr_mode = "arch"
                    arch_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith(("interior design:", "interior arrangement:")):
                    curr_mode = "features"
                    features_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith(("visual progression:", "atmosphere:")):
                    curr_mode = "lighting"
                    lighting_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                elif llow.startswith(("important details:", "important items:", "important props:", "important continuity:")):
                    curr_mode = "details"
                    details_lines.append(l.split(":", 1)[1].strip() if ":" in l else l)
                else:
                    if curr_mode == "features":
                        features_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "lighting":
                        lighting_lines.append(l.lstrip("-*• "))
                    elif curr_mode == "details":
                        details_lines.append(l.lstrip("-*• "))
                    else:
                        arch_lines.append(l.lstrip("-*• "))

            bibles["environments"].append({
                "name": name,
                "architectural_style": " ".join(arch_lines).strip() or "Story location architecture.",
                "environmental_features": " ".join(features_lines).strip() or "Interior features and elements.",
                "lighting_and_atmosphere": " ".join(lighting_lines).strip() or "Atmospheric lighting.",
                "setting_constraints": " ".join(details_lines).strip()
            })

    # 3. OBJECT AND PROP BIBLE
    obj_match = re.search(r"(?:3\.?\s*)?OBJECT AND PROP BIBLE(.*?)(?=\d+\.\s*CINEMATIC|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if obj_match:
        o_text = obj_match.group(1)
        items = re.findall(r"(?:^[•\-\*]\s*|^)([A-Za-z0-9\s\',]{3,40})(?::\s*([^\n]+)|$)", o_text, re.MULTILINE)
        for name, desc in items:
            name_clean = name.strip().lstrip("•-* ")
            if not name_clean or name_clean.lower().startswith(("maintain", "survival equipment", "food and water", "important personal", "structural elements")):
                continue
            bibles["key_objects"].append({
                "name": name_clean,
                "visual_description": desc.strip() if desc else f"Key survival item: {name_clean}",
                "significance": desc.strip() if desc else "Essential survival prop"
            })

    # 4. CINEMATIC STYLE BIBLE
    style_match = re.search(r"(?:4\.?\s*)?CINEMATIC STYLE BIBLE(.*?)(?=\d+\.\s*CHRONOLOGY|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if style_match:
        s_text = style_match.group(1)
        for l in s_text.split("\n"):
            llow = l.lower()
            if "lighting progression" in llow or "lighting" in llow:
                bibles["visual_style_bible"]["lighting_rules"] = l.strip()
            elif "color progression" in llow or "color" in llow:
                bibles["visual_style_bible"]["color_palette"] = l.strip()
            elif "composition variety" in llow or "composition" in llow:
                bibles["visual_style_bible"]["camera_guidelines"] = l.strip()

    # 5. CHRONOLOGY AND CONTINUITY BIBLE
    chron_match = re.search(r"(?:5\.?\s*)?CHRONOLOGY AND CONTINUITY BIBLE(.*?)(?=\d+\.\s*IMPORTANT|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if chron_match:
        c_lines = chron_match.group(1).strip().split("\n")
        events = []
        for l in c_lines:
            l_clean = l.strip().lstrip("•-*| ")
            if l_clean and not l_clean.lower().startswith(("timeline", "major event", "---")):
                events.append(l_clean)
        bibles["chronological_events"] = events[:25]

    # 6. RULES & IDENTITY
    rules_match = re.search(r"IMPORTANT VISUAL CONTINUITY RULES(.*?)(?=\d+\.\s*FINAL|\Z)", bible_text, re.DOTALL | re.IGNORECASE)
    if rules_match:
        r_lines = [rl.strip().lstrip("•-* ") for rl in rules_match.group(1).strip().split("\n") if rl.strip()]
        bibles["visual_continuity_rules"] = r_lines

    identity_match = re.search(r"FINAL STORY VISUAL IDENTITY(.*?)\Z", bible_text, re.DOTALL | re.IGNORECASE)
    if identity_match:
        bibles["story_bible"]["visual_identity"] = identity_match.group(1).strip()

    return bibles


# =============================================================================
# STEP 4 & 5: DEDICATED GROQ IMAGE PROMPT GENERATION (BATCHED & EXPANDED)
# =============================================================================

def load_image_prompt_system_prompt() -> str:
    p = Path("config/system_prompts/image_prompt_generation.txt")
    if p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "You are an expert image prompt architect. Generate detailed, style-compliant image prompts in JSON."


def robust_extract_scenes_from_llm_output(clean_json: str) -> List[Dict[str, Any]]:
    """
    Parses LLM output into a list of scene dictionaries.
    Handles:
    1. Perfect JSON: {"scenes": [...]} or [...]
    2. Markdown wrapped JSON
    3. Trailing commas or subtle formatting glitches
    4. Truncated JSON streams (extracts all completed scene objects even if the last one was cut off)
    5. Regex recovery fallback
    """
    if not clean_json or not clean_json.strip():
        return []

    text = clean_json.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    # Attempt 1: Direct JSON parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "scenes" in parsed and isinstance(parsed["scenes"], list):
            return parsed["scenes"]
        elif isinstance(parsed, list):
            return parsed
    except Exception:
        pass

    # Attempt 2: Repair braces and trailing commas
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        sub = text[first_brace:last_brace + 1]
        try:
            p = json.loads(sub)
            if isinstance(p, dict) and "scenes" in p:
                return p["scenes"]
        except Exception:
            fixed = re.sub(r",\s*([}\]])", r"\1", sub)
            try:
                p = json.loads(fixed)
                if isinstance(p, dict) and "scenes" in p:
                    return p["scenes"]
            except Exception:
                pass

    # Attempt 3: Truncated stream recovery
    if first_brace != -1:
        matches = list(re.finditer(r"\}\s*,?", text))
        for m in reversed(matches):
            candidate = text[first_brace:m.end()].rstrip(", \t\r\n")
            for closing in ["]}", "}", "]}}", "}\n}"]:
                try:
                    p = json.loads(candidate + closing)
                    if isinstance(p, dict) and "scenes" in p and p["scenes"]:
                        return p["scenes"]
                    elif isinstance(p, list) and p:
                        return p
                except Exception:
                    pass

    # Attempt 4: Regex extraction of individual completed scene objects
    scenes = []
    obj_pattern = re.compile(
        r'\{\s*"scene_id"\s*:\s*"(?P<id>[^"]+)"\s*,\s*"visual_subject"\s*:\s*"(?P<subj>(?:[^"\\]|\\.)*)"\s*,\s*"prompt"\s*:\s*"(?P<prompt>(?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    for m in obj_pattern.finditer(text):
        try:
            s_id = m.group("id").strip()
            prompt_str = m.group("prompt").encode('utf-8').decode('unicode_escape', errors='ignore')
            subj_str = m.group("subj").encode('utf-8').decode('unicode_escape', errors='ignore')
            scenes.append({
                "scene_id": s_id,
                "visual_subject": subj_str,
                "prompt": prompt_str
            })
        except Exception:
            pass

    if scenes:
        return scenes

    # Loose fallback regex for scene_id and prompt
    loose_pattern = re.compile(
        r'"scene_id"\s*:\s*"([^"]+)"[\s\S]*?"prompt"\s*:\s*"((?:[^"\\]|\\.)*)"',
        re.DOTALL
    )
    for m in loose_pattern.finditer(text):
        try:
            s_id = m.group(1).strip()
            prompt_str = m.group(2).encode('utf-8').decode('unicode_escape', errors='ignore')
            scenes.append({
                "scene_id": s_id,
                "prompt": prompt_str
            })
        except Exception:
            pass

    return scenes


def generate_prompts_batch(
    scenes_batch: List[VisualScene],
    bibles: Dict[str, Any],
    story_context: str,
    visual_style_id: str = "zombie_survival",
    custom_style: Optional[str] = None,
    aspect_ratio: str = "16:9",
    model: str = "qwen/qwen3.8-27b",
    engine: Optional[GroqEngine] = None
) -> List[Dict[str, Any]]:
    """
    Sends a batch of scenes to Groq along with extracted bibles, narration, timestamps,
    and user visual ideas to generate detailed, concrete image prompts STRICTLY obeying
    the user-selected visual style directives.
    """
    groq_eng = engine or GroqEngine(default_model=model)
    system_prompt = load_image_prompt_system_prompt()

    # Retrieve Authoritative Visual Style
    style_preset = PresetManager.get_style_by_id(visual_style_id)
    style_name = style_preset.name if style_preset else visual_style_id
    style_instructions = style_preset.system_instructions if style_preset else (custom_style or "Cinematic realism")

    # Build concise representation of bibles
    char_summary = "\n".join(
        f"- {c.get('name')}: {c.get('physical_description', '')}. Wardrobe: {c.get('clothing_and_gear', '')}. Rules: {c.get('consistency_rules', '')}"
        for c in bibles.get("characters", [])
    )
    env_summary = "\n".join(
        f"- {e.get('name')}: {e.get('architectural_style', '')}. Atmosphere: {e.get('lighting_and_atmosphere', '')}"
        for e in bibles.get("environments", [])
    )
    style_info = bibles.get("visual_style_bible", {})
    style_summary = f"Color Palette: {style_info.get('color_palette', 'Consistent with selected style')}. Lighting: {style_info.get('lighting_rules', 'Atmospheric')}. Camera: {style_info.get('camera_guidelines', 'Eye-level cinematic framing')}."

    scenes_input = []
    for s in scenes_batch:
        scenes_input.append({
            "scene_id": s.scene_id,
            "start": s.start,
            "end": s.end,
            "duration": s.duration,
            "narration": s.narration_excerpt,
            "visual_idea": getattr(s, "visual_idea", "") or s.visual_subject
        })

    user_prompt = f"""AUTHORITATIVE USER-SELECTED VISUAL STYLE:
Style Name: {style_name}
Aspect Ratio: {aspect_ratio}

STYLE DIRECTIVES (MANDATORY & OVERRIDING):
{style_instructions}

RELEVANT STORY BIBLES:
### Characters:
{char_summary or 'No specific character entries.'}

### Environments:
{env_summary or 'No specific environment entries.'}

### Story Style Elements:
{style_summary}

STORY CONTEXT (OVERALL ARC):
{story_context[:800]}

SCENES TO GENERATE IN THIS BATCH:
{json.dumps(scenes_input, indent=2)}

INSTRUCTIONS FOR EACH SCENE:
1. Translate the narration and visual idea into a detailed, concrete image-generation prompt describing what the viewer actually SEES.
2. STRICT STYLE ENFORCEMENT: Every single prompt MUST strictly follow the 'STYLE DIRECTIVES' above for '{style_name}'.
   - If '{style_name}' mandates a pure flat 2D cartoon with a locked hero character (e.g. backwards red cap, charcoal hoodie) and uniform pale olive-green zombie horde, apply those exact design elements!
   - If '{style_name}' mandates photographic or historical styles, apply those exact camera, lens, and lighting directives!
3. Do not invent contradictory characters or ignore the supplied visual ideas.
4. Append model parameter flags: --ar {aspect_ratio} --style raw
5. Return JSON matching the schema with the exact scene_ids provided.
"""

    resp_text = groq_eng.generate_text(
        prompt=user_prompt,
        system_prompt=system_prompt,
        temperature=0.3,
        max_tokens=2500,
        model=model,
        stage_name=f"PROMPT_EXPANSION_{style_name.upper().replace(' ', '_')}"
    )

    items = robust_extract_scenes_from_llm_output(resp_text)
    if not items:
        clean_snip = resp_text.strip()[:200] if resp_text else "EMPTY_RESPONSE"
        raise RuntimeError(f"Batch prompt parsing failed: No valid scenes recovered from model output. Snippet: {clean_snip}")

    return items
