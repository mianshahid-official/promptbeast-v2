"""
Full Real End-to-End Integration Test for PromptBeast-V2 Architecture.
Executes all 7 stages on real inputs:
1. Script + Audio + Timestamp JSON in
2. Checkpoint 01_chunks.json written
3. Checkpoint 02_raw_entities.json and 03_master_bible.json written
4. Checkpoint 04_storyboard_beats.json written
5. Checkpoint 05_final_prompts.json written
6. Prompts exported to prompts/
7. Render gate strictly blocks on missing images
8. Render gate passes when images are present
9. Video rendered with FFmpeg to <project>_final.mp4
10. All 4 post-render verification checks pass
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from app.core.bible import generate_story_bible
from app.core.exporter import export_prompts
from app.core.groq_engine import GroqEngine
from app.core.images import scan_project_images
from app.core.loader import load_inputs
from app.core.project import ProjectCheckpointManager
from app.core.prompt_generator import generate_image_prompts
from app.core.renderer import VideoRenderer
from app.core.splitter import split_scenes
from app.core.storyboard import plan_storyboard_beats


def create_test_tone_audio(output_wav: Path, duration: float = 12.0) -> Path:
    """Generates a clean synthetic audio file using FFmpeg lavfi sine wave."""
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:a", "pcm_s16le",
        "-ar", "44100",
        str(output_wav)
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return output_wav


def test_full_pipeline_real_e2e(tmp_path):
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    dirs = ProjectCheckpointManager.get_project_dirs(project_dir)

    # 1. Inputs: 3 files only
    script_text = (
        "On Day nine, a voice answers your radio. "
        "It promises a plane on Day sixty. "
        "And it gives you one rule: make no sound. "
        "By midnight, Reyes breaks the signal."
    )
    script_file = tmp_path / "script.txt"
    script_file.write_text(script_text, encoding="utf-8")

    # Generate 15.0s audio
    audio_file = tmp_path / "narration.wav"
    create_test_tone_audio(audio_file, duration=15.0)

    # Word-level timestamps perfectly aligning with script
    # Total 24 words, spanning 0.2 to ~11.0s (within 12.0s audio)
    raw_words = script_text.split()
    ts_entries = []
    t_cursor = 0.2
    for w in raw_words:
        w_dur = 0.40
        ts_entries.append({
            "word": w,
            "start": round(t_cursor, 3),
            "end": round(t_cursor + w_dur, 3)
        })
        t_cursor += 0.44

    ts_file = tmp_path / "timestamps.json"
    ts_file.write_text(json.dumps({"words": ts_entries}), encoding="utf-8")

    # Step 0: Load & Validate
    inputs = load_inputs(script_file, audio_file, ts_file)
    assert inputs.total_words == len(raw_words)
    assert abs(inputs.audio_duration - 15.0) < 0.5

    # Step 1: Split Scenes (Deterministic)
    scenes = split_scenes(inputs.words)
    assert len(scenes) >= 2
    cp_01 = ProjectCheckpointManager.save_checkpoint(project_dir, "01_chunks.json", scenes)
    assert cp_01.exists()

    # Step 2 to 4: Dynamic Mock Engine
    mock_engine = MagicMock(spec=GroqEngine)
    def mock_generate_structured(prompt, system_prompt, schema, schema_name="structured_output", model=None):
        import re
        if schema_name == "entity_extraction":
            return {
                "characters": [
                    {"name": "Hero", "appearance": "vigilant survivor", "clothing": "dark coat", "colors": "black", "features": "headset"},
                    {"name": "Reyes", "appearance": "radio technician", "clothing": "flight jacket", "colors": "brown", "features": "glasses"}
                ],
                "locations": [
                    {"name": "Control Tower", "visual_traits": "dark radio console", "time_of_day": "night", "atmosphere": "tense"}
                ],
                "objects": [
                    {"name": "Emergency Radio", "description": "glowing green frequency dial"}
                ]
            }
        elif schema_name == "storyboard_beats":
            s_ids = re.findall(r"Scene (\d+)", prompt)
            return {
                "beats": [
                    {
                        "scene_id": s_id,
                        "visual_action": f"Hero stands at radio in Scene {s_id}",
                        "composition": "Medium wide shot, slow zoom-in.",
                        "bible_refs": ["Hero", "Control Tower"],
                        "missing_elements": []
                    }
                    for s_id in s_ids
                ]
            }
        elif schema_name == "scene_prompts":
            s_ids = re.findall(r"--- SCENE (\d+) ---", prompt)
            return {
                "scenes": [
                    {
                        "scene_id": s_id,
                        "prompt": (
                            "Cinematic suspense photograph, 35mm film grain. "
                            "Hero is a vigilant survivor wearing a dark coat and black headset. "
                            "Inside the dark control tower at night with a tense atmosphere, "
                            "the hero stands beside the glowing radio console under dim lights. "
                            f"Exact narration context for Scene {s_id}. "
                            "The shot is framed as a medium wide angle with slow camera push-in. "
                            "Dramatic shadow contrast and cold desaturated blue tones. "
                            "Image must contain no text. "
                            "Every visual element maintains strict continuity across scenes. "
                            + ("Detailed atmospheric photographic elements fill the background. " * 8)
                        ),
                        "word_count": 195,
                        "bible_refs": ["Hero", "Control Tower"]
                    }
                    for s_id in s_ids
                ]
            }
        return {}
    mock_engine.generate_structured.side_effect = mock_generate_structured

    # Run Step 2
    raw_entities_p = project_dir / "02_raw_entities.json"
    master_bible_p = project_dir / "03_master_bible.json"
    _, master_bible = generate_story_bible(
        scenes,
        mock_engine,
        raw_entities_path=raw_entities_p,
        master_bible_path=master_bible_p
    )
    assert raw_entities_p.exists()
    assert master_bible_p.exists()
    assert master_bible["locked"] is True

    # Run Step 3
    beats_cp = project_dir / "04_storyboard_beats.json"
    beats, warnings = plan_storyboard_beats(
        scenes,
        master_bible,
        style_anchor="Cinematic dark suspense photography, 35mm film grain.",
        engine=mock_engine,
        checkpoint_path=beats_cp
    )
    assert beats_cp.exists()
    assert len(beats) == len(scenes)

    # Run Step 4
    prompts_cp = project_dir / "05_final_prompts.json"
    prompts, failed = generate_image_prompts(
        beats,
        master_bible,
        style_id="default",
        engine=mock_engine,
        checkpoint_path=prompts_cp
    )
    assert prompts_cp.exists()
    assert len(prompts) == len(scenes)

    # Verify all 5 checkpoints exist on disk
    for cp_name in ["01_chunks.json", "02_raw_entities.json", "03_master_bible.json", "04_storyboard_beats.json", "05_final_prompts.json"]:
        assert (project_dir / cp_name).exists(), f"Checkpoint {cp_name} missing"

    # Run Step 5: Export Prompts
    index_map = export_prompts(prompts, dirs["prompts"])
    assert (dirs["prompts"] / "prompts_only.txt").exists()
    assert (dirs["prompts"] / "prompts_index.json").exists()

    # Step 5 Render Gate: Initially blocked because no images are in images/
    gate_res_before = scan_project_images(dirs["images"], scenes)
    assert gate_res_before.is_ready_for_render is False
    assert gate_res_before.missing_count == len(scenes)

    # Generate valid image files for each scene (1280x720 RGB)
    images_map = {}
    for s in scenes:
        s_id = s["scene_id"]
        img_p = dirs["images"] / f"scene_{s_id}.png"
        img = Image.new("RGB", (1280, 720), color=(80, 90, 110))
        img.save(img_p)
        images_map[s_id] = img_p

    # Step 5 Render Gate: Now all images present -> unblocked!
    gate_res_after = scan_project_images(dirs["images"], scenes)
    assert gate_res_after.is_ready_for_render is True
    assert gate_res_after.missing_count == 0

    # Step 6: Render Final Video with FFmpeg & Run 4 Checks
    renderer = VideoRenderer(project_dir=project_dir, aspect_ratio="16:9")
    out_file, v_res = renderer.render(
        scenes=scenes,
        images_map=images_map,
        audio_path=audio_file,
        audio_duration=15.0,
        timestamp_words=ts_entries,
        script_words=raw_words,
        output_filename="test_final.mp4"
    )

    # Verify render results
    assert out_file.exists()
    assert out_file.stat().st_size > 1000
    assert v_res.technical_passed is True
    assert v_res.audio_passed is True
    assert v_res.transcript_diff_passed is True
    assert v_res.visual_spot_checks_passed is True
    assert v_res.all_passed is True
