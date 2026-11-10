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

        # Process in batches (safe clamp to 4 prevents Groq token limits and truncation)
        batch_sz = max(1, min(batch_size, 4))
        total_batches = (total_to_process + batch_sz - 1) // batch_sz

        for b_idx in range(total_batches):
            batch = scenes_to_process[b_idx * batch_sz : (b_idx + 1) * batch_sz]
            batch_num = b_idx + 1
            pct = round(10.0 + (b_idx / total_batches) * 85.0, 1)

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
                max_retries=2
            )

        if progress_callback:
            progress_callback(100.0, f"Prompt generation complete for {total_to_process} scenes.")

        return storyboard

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
        max_retries: int = 2
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
1. Translate each scene's narration and visual idea into a rich, detailed, concrete image-generation prompt.
2. STRICT STYLE ENFORCEMENT: Every prompt MUST strictly adhere to the Style Directives for '{style_name}'.
3. Explicitly describe physical subject, attire, concrete environment, lighting source, camera shot type, and lens.
4. Do NOT include /imagine prompt: or parameter flags like --ar or --style raw. Output clean, descriptive prompt text only.
5. Return JSON with key "scenes" containing a list of objects with "scene_id", "prompt", and "negative_prompt".
"""

        # Retry loop
        for attempt in range(max_retries + 1):
            try:
                raw_response = engine.generate_text(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.3,
                    max_tokens=3000
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
                        p_text = str(matched["prompt"]).strip()
                        # Clean any /imagine or model parameter flags
                        p_text = re.sub(r"^/imagine\s+prompt:\s*", "", p_text, flags=re.IGNORECASE)
                        p_text = re.sub(r"^/imagine\s*", "", p_text, flags=re.IGNORECASE)
                        p_text = re.sub(r"--ar\s+\d+:\d+", "", p_text, flags=re.IGNORECASE)
                        p_text = re.sub(r"--style\s+\w+", "", p_text, flags=re.IGNORECASE)
                        p_text = re.sub(r"--v\s+[\d\.]+", "", p_text, flags=re.IGNORECASE)
                        s.prompt = " ".join(p_text.split())
                        s.generation_status = "generated"
                        s.is_validated = True
                    elif not s.prompt:
                        all_assigned = False

                if all_assigned or attempt == max_retries:
                    # Fill any remaining unassigned scene with a concrete descriptive prompt from its visual idea
                    for s in batch:
                        if not s.prompt:
                            clean_idea = s.visual_idea or s.visual_subject or "Cinematic scene"
                            s.prompt = f"{clean_idea}. Setting: {s.environment or 'Story environment'}. Rendered in {style_name} visual aesthetic."
                            s.generation_status = "generated"
                    break

            except Exception as e:
                if attempt == max_retries:
                    for s in batch:
                        if not s.prompt:
                            clean_idea = s.visual_idea or s.visual_subject or "Cinematic scene"
                            s.prompt = f"{clean_idea}. Setting: {s.environment or 'Story environment'}. Rendered in {style_name} visual aesthetic."
                            s.generation_status = "generated"

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

Return JSON with "prompt" and "negative_prompt".
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
                p_text = re.sub(r"^/imagine\s+prompt:\s*", "", p_text, flags=re.IGNORECASE)
                p_text = re.sub(r"^/imagine\s*", "", p_text, flags=re.IGNORECASE)
                p_text = re.sub(r"--ar\s+\d+:\d+", "", p_text, flags=re.IGNORECASE)
                p_text = re.sub(r"--style\s+\w+", "", p_text, flags=re.IGNORECASE)
                p_text = re.sub(r"--v\s+[\d\.]+", "", p_text, flags=re.IGNORECASE)
                scene.prompt = " ".join(p_text.split())
                scene.generation_status = "generated"
                scene.is_manually_edited = False
                scene.is_validated = True
        except Exception:
            clean_idea = scene.visual_idea or scene.visual_subject or "Cinematic scene"
            scene.prompt = f"{clean_idea}. Setting: {scene.environment or 'Story environment'}. Rendered in {style_name} visual aesthetic."
            scene.generation_status = "generated"

        return scene


# Backwards compatibility alias for older code
Stage5PromptGenerator = Stage3PromptGeneration
Stage3PromptGenerator = Stage3PromptGeneration
