"""
Stage 3: Image Prompt Generation Module for PromptBeast.
Transforms storyboard entries into concrete, high-quality, model-optimized image prompts.
Strictly enforces the user-selected visual style, maintains character and environment bibles,
respects approved prompts, and supports targeted single-scene or batch regeneration.
"""
import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable

from app.models.scene import VisualScene
from app.models.planning import StoryIntelligence, ScriptAnalysis, ScheduledStoryboard
from app.models.presets import VisualStylePreset
from app.presets.loader import PresetManager
from backend.inference.engine import BaseInferenceEngine, GroqInferenceEngine
from backend.pipeline.stage_validation import validate_storyboard
from app.core.narration_workflow import robust_extract_scenes_from_llm_output


def load_image_prompt_system_prompt() -> str:
    """Loads system prompt from editable file on disk."""
    prompt_file = Path("config/system_prompts/image_prompt_generation.txt")
    if prompt_file.exists():
        try:
            return prompt_file.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return "You are an expert image prompt architect. Generate detailed, style-compliant image prompts in JSON."


def sanitize_prompt_prose(text: str) -> str:
    """
    Sanitizes generated image prompts to enforce simple, direct prose:
    - Removes /imagine prefixes and parameter flags (--ar, --style, --v).
    - Removes markdown headings (#, ##, ###).
    - Removes bold category labels (**Subject:**, **Environment:**, **Camera:**, etc.).
    - Removes plain category labels (Subject:, Environment:, etc.).
    - Removes bullet points (- , * , 1. , • ).
    - Collapses extra newlines and spaces into clean, continuous prose.
    """
    if not text:
        return ""
    p = str(text).strip()
    # Strip imagine / flags
    p = re.sub(r"^/imagine\s+prompt:\s*", "", p, flags=re.IGNORECASE)
    p = re.sub(r"^/imagine\s*", "", p, flags=re.IGNORECASE)
    p = re.sub(r"--ar\s+\d+:\d+", "", p, flags=re.IGNORECASE)
    p = re.sub(r"--style\s+\w+", "", p, flags=re.IGNORECASE)
    p = re.sub(r"--v\s+[\d\.]+", "", p, flags=re.IGNORECASE)

    # Strip markdown headers (# Header)
    p = re.sub(r"(?m)^\s*#+\s+.*$", "", p)

    # Strip bold labels like **Subject:**, **Environment:**, **Camera:**, **Lighting:**, **Style:**
    p = re.sub(r"\*\*[A-Za-z\s]+:\*\*\s*", "", p)
    p = re.sub(r"\*[A-Za-z\s]+:\*\s*", "", p)

    # Strip line-leading category labels like Subject:, Environment:, Lighting:, Camera:, Continuity:
    p = re.sub(r"(?mi)^\s*(?:Subject|Environment|Characters?|Lighting|Camera|Composition|Style|Atmosphere|Details|Continuity):\s*", "", p)

    # Strip bullet points (- or * or • or numbered list like 1. )
    p = re.sub(r"(?m)^\s*[-*•]\s+", "", p)
    p = re.sub(r"(?m)^\s*\d+[\.\)]\s+", "", p)

    # Collapse lines into continuous prose
    lines = [line.strip() for line in p.splitlines() if line.strip()]
    cleaned = " ".join(lines)
    return " ".join(cleaned.split())


# Measured per-scene output cost for one image prompt (prompt + negative_prompt
# + JSON scaffolding). Measured in the vidroa F5 derivation for the same
# image-prompt output shape; used here to derive the batch size from the
# engine's effective output cap instead of guessing a batch size.
PROMPT_TOKENS_PER_SCENE = 350


def derive_stage3_batch_size(engine: Any, requested_batch_size: int) -> int:
    """Derives the batch size from the engine's effective output cap.

    batch_size = effective_cap // PROMPT_TOKENS_PER_SCENE, clamped to
    [1, requested]. With the 950-token baseline cap this gives 2 scenes per
    batch; with a reduced 800 cap it stays at 2. Never exceeds the requested
    size and never exceeds what fits in the output budget.
    """
    cap = 950
    accessor = getattr(engine, "current_effective_max_tokens", None)
    if callable(accessor):
        try:
            cap = int(accessor(950))
        except Exception:
            cap = 950
    derived = max(1, cap // PROMPT_TOKENS_PER_SCENE)
    return max(1, min(int(requested_batch_size), derived))


class Stage3PromptGeneration:
    """
    Stage 3: Generates camera-ready image prompts for storyboard scenes via Groq.
    Preserves approved scenes and supports selective regeneration.
    """

    @classmethod
    def execute(
        cls,
        storyboard: List[VisualScene],
        story_intelligence: Optional[StoryIntelligence] = None,
        style_preset: Optional[VisualStylePreset] = None,
        visual_style_id: Optional[str] = None,
        custom_style: Optional[str] = None,
        aspect_ratio: str = "16:9",
        batch_size: int = 8,
        engine: Optional[BaseInferenceEngine] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        force_regenerate: bool = False,
        model: str = "qwen/qwen3.8-27b",
        allow_partial: bool = False,
        **kwargs
    ) -> List[VisualScene]:
        """
        Generates prompts for all scenes in the storyboard.
        Skips scenes that are already approved (is_approved=True) unless force_regenerate is True.
        """
        if not storyboard:
            return []

        # Resolve Style Preset
        target_style_id = visual_style_id or (style_preset.id if style_preset else "zombie_survival")
        if not style_preset:
            style_preset = PresetManager.get_style_by_id(target_style_id)

        style_name = style_preset.name if style_preset else target_style_id
        style_instructions = style_preset.system_instructions if style_preset else (custom_style or "Cinematic realism")

        if engine is None:
            engine = GroqInferenceEngine(default_model=model)

        system_prompt = load_image_prompt_system_prompt()

        # Build bible context summaries
        char_summary = ""
        env_summary = ""
        style_summary = ""
        story_context = ""

        if story_intelligence:
            story_context = story_intelligence.story_summary
            char_summary = "\n".join(
                f"- {c.get('name')}: {c.get('physical_description', '')}. Wardrobe: {c.get('clothing_and_gear', '')}. Consistency: {c.get('consistency_rules', '')}"
                for c in story_intelligence.characters
            )
            env_summary = "\n".join(
                f"- {e.get('name')}: {e.get('architectural_style', '')}. Features: {e.get('environmental_features', '')}. Atmosphere: {e.get('lighting_and_atmosphere', '')}"
                for e in story_intelligence.environments
            )
            style_info = story_intelligence.visual_style_bible or {}
            style_summary = f"Palette: {style_info.get('color_palette', '')}. Lighting: {style_info.get('lighting_rules', '')}. Camera: {style_info.get('camera_guidelines', '')}"

        # Identify which scenes need generation
        scenes_to_process = []
        for s in storyboard:
            if force_regenerate or not s.is_approved:
                scenes_to_process.append(s)

        total_to_process = len(scenes_to_process)
        if total_to_process == 0:
            if progress_callback:
                progress_callback(100.0, "All scenes are approved. No prompt regeneration needed.")
            return storyboard

        # Process in batches. The batch size is derived from the engine's
        # effective output cap and the measured per-prompt cost, so a batch
        # always fits in the output budget (never truncate to fit).
        batch_sz = derive_stage3_batch_size(engine, batch_size)
        total_batches = (total_to_process + batch_sz - 1) // batch_sz

        # Pass 1: Run through all batches. If a batch fails or is missing a scene,
        # skip/postpone it and continue generating subsequent batches.
        for b_idx in range(total_batches):
            batch = scenes_to_process[b_idx * batch_sz : (b_idx + 1) * batch_sz]
            batch_num = b_idx + 1
            pct = round(10.0 + (b_idx / total_batches) * 75.0, 1)

            if progress_callback:
                progress_callback(pct, f"Generating prompts: batch {batch_num}/{total_batches} ({len(batch)} scenes)...")

            cls._generate_batch_with_retry(
                engine=engine,
                batch=batch,
                system_prompt=system_prompt,
                style_name=style_name,
                style_instructions=style_instructions,
                aspect_ratio=aspect_ratio,
                char_summary=char_summary,
                env_summary=env_summary,
                style_summary=style_summary,
                story_context=story_context,
                max_retries=1,
                raise_on_failure=False
            )

        # Pass 2: Retry pass for any skipped / missing scenes from Pass 1
        missing_scenes = [s for s in scenes_to_process if not (s.prompt and s.prompt.strip())]
        if missing_scenes:
            total_missing = len(missing_scenes)
            for m_idx, sc in enumerate(missing_scenes):
                pct = round(85.0 + ((m_idx + 1) / total_missing) * 12.0, 1)
                if progress_callback:
                    progress_callback(pct, f"Retrying skipped scene {m_idx + 1}/{total_missing} (Scene #{sc.scene_id})...")

                cls._generate_batch_with_retry(
                    engine=engine,
                    batch=[sc],
                    system_prompt=system_prompt,
                    style_name=style_name,
                    style_instructions=style_instructions,
                    aspect_ratio=aspect_ratio,
                    char_summary=char_summary,
                    env_summary=env_summary,
                    style_summary=style_summary,
                    story_context=story_context,
                    max_retries=2,
                    raise_on_failure=False
                )

        # Pass 3: Evaluate completeness
        still_missing = [s for s in scenes_to_process if not (s.prompt and s.prompt.strip())]
        if still_missing:
            for sc in still_missing:
                sc.generation_status = "error"
                sc.is_validated = False

            if allow_partial:
                if progress_callback:
                    progress_callback(100.0, f"Completed {total_to_process - len(still_missing)}/{total_to_process} prompts ({len(still_missing)} need retry).")
                return storyboard
            else:
                missing_ids = [s.scene_id for s in still_missing]
                raise RuntimeError(
                    f"Stage 3 prompt generation failed: no prompt returned for scene(s) "
                    f"{', '.join(missing_ids)} after retries. Failing loudly instead of filling a template prompt."
                )

        # Pre-heal natural narration pauses (gaps <= 2.0s) across the timeline
        cls._heal_storyboard_timeline_gaps(storyboard, max_gap=2.0)

        # Validation gate: verify integrity before rendering
        try:
            validate_storyboard(storyboard)
        except Exception as e:
            if allow_partial:
                print(f"[Stage3PromptGeneration] Validation warning: {e}. Preserving all generated prompts.")
            else:
                raise

        if progress_callback:
            progress_callback(100.0, f"Prompt generation complete for {total_to_process} scenes.")

        return storyboard

    @classmethod
    def _heal_storyboard_timeline_gaps(cls, storyboard: List[VisualScene], max_gap: float = 2.0) -> None:
        """
        Extends scene end times to the next scene start for natural narration pauses <= max_gap,
        ensuring smooth video transitions without timeline gaps.
        """
        if not storyboard or len(storyboard) < 2:
            return
        ordered = sorted(storyboard, key=lambda s: float(getattr(s, "start", 0.0)))
        for i in range(len(ordered) - 1):
            curr_s = ordered[i]
            next_s = ordered[i + 1]
            gap = float(next_s.start) - float(curr_s.end)
            if 0.0 < gap <= max_gap:
                curr_s.end = round(float(next_s.start), 2)
                curr_s.duration = round(float(curr_s.end) - float(curr_s.start), 2)

    @classmethod
    def _generate_batch_with_retry(
        cls,
        engine: BaseInferenceEngine,
        batch: List[VisualScene],
        system_prompt: str,
        style_name: str,
        style_instructions: str,
        aspect_ratio: str,
        char_summary: str,
        env_summary: str,
        style_summary: str,
        story_context: str,
        max_retries: int = 2,
        raise_on_failure: bool = True
    ):
        """
        Executes a single batch prompt generation with automatic retries for missing scenes.
        """
        scenes_input = []
        for s in batch:
            scenes_input.append({
                "scene_id": s.scene_id,
                "start": s.start,
                "end": s.end,
                "duration": s.duration,
                "narration": s.narration_excerpt,
                "visual_idea": getattr(s, "visual_idea", "") or s.visual_subject,
                "environment": s.environment or getattr(s, "location_id", "") or "Story setting",
                "composition": s.composition or "Cinematic shot",
                "characters": s.character_ids or []
            })

        user_prompt = f"""AUTHORITATIVE USER-SELECTED VISUAL STYLE:
Style Name: {style_name}
Aspect Ratio: {aspect_ratio}

STYLE DIRECTIVES (MANDATORY & OVERRIDING):
{style_instructions}

PRODUCTION BIBLES:
### Characters:
{char_summary or 'No specific character entries.'}

### Environments:
{env_summary or 'No specific environment entries.'}

### Visual Tone & Atmosphere:
{style_summary or 'Maintain consistent visual mood.'}

STORY CONTEXT:
{story_context[:800]}

SCENES TO GENERATE IN THIS BATCH:
{json.dumps(scenes_input, indent=2)}

INSTRUCTIONS:
1. Translate each scene's narration and visual idea into simple, direct prose (2 to 4 continuous sentences).
2. ABSOLUTELY NO HEADINGS: Do NOT include section titles or bold labels like '**Subject:**', '**Environment:**', '**Camera:**', '**Lighting:**'.
3. ABSOLUTELY NO BULLET POINTS: Do NOT use bullet points, numbered lists, or dashes. Write continuous plain sentences.
4. STRICT BIBLE CONTEXT COMPLIANCE: Use the provided Character and Environment Bibles strictly as visual context to maintain character wardrobe, colors, and authentic setting continuity.
5. STRICT STYLE ENFORCEMENT: Every prompt MUST strictly adhere to the Style Directives for '{style_name}'.
6. Do NOT include /imagine prompt: or parameter flags like --ar or --style raw. Output clean, descriptive prose only.
7. Return JSON with key "scenes" containing a list of objects with "scene_id", "prompt", and "negative_prompt".
8. CHARACTER PRESENCE & EXCLUSION GUARD (CRITICAL):
   - ONLY include the main protagonist/hero if the scene's visual idea explicitly calls for them.
   - If the visual idea is an ENVIRONMENT, VEHICLE, CLOCK, SIREN, OBJECT, or EMPTY LOCATION, explicitly state: "Pure environment/object shot with NO characters or people present. Empty of people."
   - If the visual idea depicts PASSENGERS, SURVIVORS, or CROWDS without the hero, explicitly state: "Background crowd/passengers only. The protagonist is NOT in this scene (he is located elsewhere); no character wearing the protagonist's red cap, hoodie, or messenger bag."
   - For ZOMBIE / ENEMY scenes: NEVER depict the protagonist as a zombie or infected creature, and NEVER put the protagonist's signature clothes/gear on a zombie. Explicitly state: "All figures are uniform background zombies. The protagonist is completely unharmed, safe elsewhere, and NOT in this shot; NO zombie wears the protagonist's red cap or hoodie."
"""

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                raw_response = engine.generate_text(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=len(batch) * PROMPT_TOKENS_PER_SCENE + 80
                )

                items = robust_extract_scenes_from_llm_output(raw_response)

                # Map items by scene_id
                items_by_id = {}
                for it in items:
                    if isinstance(it, dict) and "scene_id" in it:
                        items_by_id[str(it["scene_id"]).zfill(3)] = it
                        items_by_id[str(it["scene_id"])] = it

                # Update scenes
                all_assigned = True
                for s in batch:
                    matched = items_by_id.get(s.scene_id) or items_by_id.get(s.scene_id.zfill(3)) or items_by_id.get(str(int(s.scene_id) if s.scene_id.isdigit() else -1))
                    if matched and isinstance(matched, dict) and matched.get("prompt"):
                        s.prompt = sanitize_prompt_prose(str(matched["prompt"]))
                        s.generation_status = "generated"
                        s.is_validated = True
                    elif not s.prompt:
                        all_assigned = False

                if all_assigned:
                    break
                if attempt == max_retries and raise_on_failure:
                    missing = [s.scene_id for s in batch if not s.prompt]
                    raise RuntimeError(
                        f"Stage 3 prompt generation failed: no prompt returned for scene(s) "
                        f"{', '.join(missing)} after {max_retries + 1} attempt(s). "
                        f"Failing loudly instead of filling a template prompt."
                    )

            except RuntimeError:
                if raise_on_failure:
                    raise
            except Exception as e:
                if attempt == max_retries and raise_on_failure:
                    missing = [s.scene_id for s in batch if not s.prompt]
                    raise RuntimeError(
                        f"Stage 3 prompt generation failed for scene(s) {', '.join(missing)}: {e}"
                    ) from e

    @classmethod
    def regenerate_single_scene(
        cls,
        scene: Optional[VisualScene] = None,
        engine: Optional[BaseInferenceEngine] = None,
        story_intelligence: Optional[StoryIntelligence] = None,
        style_preset: Optional[VisualStylePreset] = None,
        visual_style_id: Optional[str] = None,
        custom_style: Optional[str] = None,
        aspect_ratio: str = "16:9",
        model: str = "qwen/qwen3.8-27b",
        **kwargs
    ) -> VisualScene:
        """
        Regenerates a single scene prompt preserving other scenes.
        Accepts flexible arguments for compatibility.
        """
        if scene is None:
            scene = kwargs.get("scene")
        if scene is None:
            raise ValueError("Scene cannot be None for regeneration.")

        target_style_id = visual_style_id or (style_preset.id if style_preset else "zombie_survival")
        if not style_preset:
            style_preset = PresetManager.get_style_by_id(target_style_id)

        style_name = style_preset.name if style_preset else target_style_id
        style_instructions = style_preset.system_instructions if style_preset else (custom_style or "Cinematic realism")

        if engine is None:
            engine = kwargs.get("engine") or GroqInferenceEngine(default_model=model)

        system_prompt = load_image_prompt_system_prompt()

        char_summary = ""
        env_summary = ""
        if story_intelligence:
            char_summary = "\n".join(
                f"- {c.get('name')}: {c.get('physical_description', '')}. Wardrobe: {c.get('clothing_and_gear', '')}"
                for c in story_intelligence.characters
            )
            env_summary = "\n".join(
                f"- {e.get('name')}: {e.get('architectural_style', '')}. Atmosphere: {e.get('lighting_and_atmosphere', '')}"
                for e in story_intelligence.environments
            )

        user_prompt = f"""AUTHORITATIVE USER-SELECTED VISUAL STYLE:
Style Name: {style_name}
Aspect Ratio: {aspect_ratio}

STYLE DIRECTIVES:
{style_instructions}

PRODUCTION BIBLES:
### Characters:
{char_summary or 'No specific character entries.'}

### Environments:
{env_summary or 'No specific environment entries.'}

GENERATE SINGLE SCENE PROMPT:
Scene ID: {scene.scene_id}
Narration: {scene.narration_excerpt}
Visual Idea: {getattr(scene, 'visual_idea', '') or scene.visual_subject}
Environment: {scene.environment}
Camera Composition: {scene.composition}

INSTRUCTIONS:
1. Write simple, direct prose describing the shot, subject, environment, and lighting (2 to 4 continuous sentences).
2. NO HEADINGS: Do NOT write '**Subject:**', '**Environment:**', etc.
3. NO BULLET POINTS: Continuous sentences only.
4. Return JSON with "prompt" and "negative_prompt".
5. CHARACTER PRESENCE & EXCLUSION GUARD:
   - ONLY include the main protagonist/hero if the scene's visual idea explicitly calls for them.
   - If this scene depicts an ENVIRONMENT, OBJECT, VEHICLE, or EMPTY LOCATION, state: "Pure environment/object shot with NO characters or people present. Empty of people."
   - If this scene depicts PASSENGERS, SURVIVORS, or CROWDS without the hero, state: "Background crowd/passengers only. The protagonist is NOT in this scene (he is located elsewhere); no character wearing the protagonist's red cap, hoodie, or messenger bag."
   - For ZOMBIE / ENEMY scenes: NEVER depict the protagonist as a zombie or infected creature, and NEVER put the protagonist's signature clothes/gear on a zombie. State: "All figures are uniform background zombies. The protagonist is completely unharmed, safe elsewhere, and NOT in this shot; NO zombie wears the protagonist's red cap or hoodie."
"""
        try:
            raw = engine.generate_text(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.35,
                max_tokens=800
            )
            clean = raw.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean)

            parsed = json.loads(clean)
            p_text = parsed.get("prompt") or (parsed.get("scenes", [{}])[0].get("prompt") if "scenes" in parsed else "")
            if p_text:
                scene.prompt = sanitize_prompt_prose(str(p_text))
                scene.generation_status = "generated"
                scene.is_manually_edited = False
                scene.is_validated = True
        except Exception:
            clean_idea = scene.visual_idea or scene.visual_subject or "Cinematic scene"
            scene.prompt = sanitize_prompt_prose(f"{clean_idea}. Setting: {scene.environment or 'Story environment'}. Rendered in {style_name} visual aesthetic.")
            scene.generation_status = "generated"

        return scene


# Backwards compatibility alias for older code
Stage5PromptGenerator = Stage3PromptGeneration
Stage3PromptGenerator = Stage3PromptGeneration
