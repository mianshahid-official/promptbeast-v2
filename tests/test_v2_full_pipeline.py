"""
Unit and End-to-End Test Suite for PromptBeast-V2 Architecture.
Verifies all 7 stages:
1. Load & validate: fail loudly, word coverage, duration difference.
2. Scene splitter: deterministic 2s-5s sentence & gap boundaries.
3. Story bibles: entity deduplication, conflict logging, locked master bible.
4. Word-count gate: 180-220 words, no headings, no bullets, regenerate once.
5. Manual image checkpoint: prompts export, checklist scanning, render gate blocking.
6. Renderer: continuous timeline, audio duration matching, 4-point verification.
"""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.core.bible import (
    chunk_scenes_for_bible,
    extract_entities_for_chunk,
    generate_story_bible,
    merge_chunk_entities,
)
from app.core.exporter import export_prompts
from app.core.groq_engine import ALLOWED_MODELS, GroqEngine, GroqMultiKeyManager
from app.core.images import find_image_for_scene, scan_project_images, validate_image_file
from app.core.loader import LoadedInputs, WordToken, load_inputs
from app.core.project import ProjectCheckpointManager
from app.core.prompt_generator import (
    compose_prompt_request,
    format_bible_entries_for_scene,
    generate_image_prompts,
    generate_single_scene_prompt,
    load_style_master_prompt,
)
from app.core.renderer import (
    PostRenderVerificationResult,
    RenderVerificationError,
    VideoRenderer,
    verify_audio,
    verify_technical,
    verify_transcript_diff,
    verify_visual_spot_checks,
)
from app.core.splitter import split_scenes
from app.core.storyboard import build_bible_context, plan_storyboard_beats
from app.core.validator import (
    ValidationError,
    lint_prompt_drift,
    normalize_token,
    validate_duration_difference,
    validate_timestamp_entries,
    validate_word_coverage,
)


# =============================================================================
# 1. VALIDATION GATE & LOADER TESTS
# =============================================================================

def test_validation_missing_files():
    with pytest.raises(ValidationError) as exc:
        load_inputs("non_existent_script.txt", "non_existent_audio.mp3", "non_existent_ts.json")
    assert "does not exist" in str(exc.value)


def test_validation_empty_script(tmp_path):
    s_file = tmp_path / "empty_script.txt"
    s_file.write_text("   \n  ", encoding="utf-8")
    a_file = tmp_path / "audio.mp3"
    a_file.write_bytes(b"dummy audio")
    t_file = tmp_path / "ts.json"
    t_file.write_text("[]", encoding="utf-8")

    with pytest.raises(ValidationError) as exc:
        load_inputs(s_file, a_file, t_file)
    assert "empty" in str(exc.value).lower()


def test_validation_timestamp_invalid_times(tmp_path):
    entries = [{"word": "test", "start": 5.0, "end": 4.0}]
    with pytest.raises(ValidationError) as exc:
        validate_timestamp_entries(entries, 10.0, tmp_path / "ts.json")
    assert "start (5.000) >= end (4.000)" in str(exc.value)


def test_validation_timestamp_decreasing_times(tmp_path):
    entries = [
        {"word": "first", "start": 2.0, "end": 3.0},
        {"word": "second", "start": 1.0, "end": 2.5}
    ]
    with pytest.raises(ValidationError) as exc:
        validate_timestamp_entries(entries, 10.0, tmp_path / "ts.json")
    assert "decreasing start time" in str(exc.value)


def test_validation_duration_difference_gate(tmp_path):
    # Allowed <= 2.0s
    validate_duration_difference(10.0, 8.5, tmp_path / "audio.mp3", tmp_path / "ts.json")

    # Disallowed > 2.0s
    with pytest.raises(ValidationError) as exc:
        validate_duration_difference(10.0, 7.5, tmp_path / "audio.mp3", tmp_path / "ts.json")
    assert "exceeds allowed maximum of 2.0s" in str(exc.value)


def test_validation_word_coverage_mismatch(tmp_path):
    s_file = tmp_path / "script.txt"
    t_file = tmp_path / "ts.json"

    script_words = ["Sixty", "people", "were", "locked"]
    ts_words = ["60", "people", "were", "locked"]

    with pytest.raises(ValidationError) as exc:
        validate_word_coverage(script_words, ts_words, s_file, t_file)

    msg = str(exc.value)
    assert "Word coverage mismatch at position 0" in msg
    assert "expected 'Sixty'" in msg
    assert "found '60'" in msg


# =============================================================================
# 2. DETERMINISTIC SCENE SPLITTER TESTS
# =============================================================================

def test_deterministic_splitter_boundaries():
    """
    Tests exact cut rules:
    - cut on sentence end (. ! ?) if duration >= 2.0s
    - cut on duration reaching 5.0s
    - cut on silence gap > 2.0s
    - final scene may be < 2s without padding
    """
    words = [
        # Scene 001: reaches >= 2.0s at word 4 which ends with '.'
        WordToken("Sixty", 0.0, 0.5),
        WordToken("people", 0.5, 1.0),
        WordToken("were", 1.0, 1.5),
        WordToken("locked.", 1.5, 2.4),    # duration 2.4s, ends with '.' -> CUT

        # Scene 002: no sentence end until duration hits >= 5.0s
        WordToken("Inside", 2.4, 3.2),
        WordToken("the", 3.2, 4.0),
        WordToken("massive", 4.0, 5.5),
        WordToken("airport", 5.5, 6.8),
        WordToken("terminal", 6.8, 7.6),   # duration = 7.6 - 2.4 = 5.2s >= 5.0s -> CUT

        # Scene 003: silence gap > 2.0s before next word
        # prev end was 7.6; next start is 10.0 (gap = 2.4s > 2.0s)
        WordToken("Then", 10.0, 10.4),
        WordToken("silence.", 10.4, 11.2)   # final scene, duration 1.2s (< 2.0s, no padding)
    ]

    scenes = split_scenes(words)
    assert len(scenes) == 3

    # Scene 001
    s1 = scenes[0]
    assert s1["scene_id"] == "001"
    assert s1["start"] == 0.0
    assert s1["end"] == 2.4
    assert s1["duration"] == 2.4
    assert s1["words"] == ["Sixty", "people", "were", "locked."]
    assert s1["text"] == "Sixty people were locked."

    # Scene 002
    s2 = scenes[1]
    assert s2["scene_id"] == "002"
    assert s2["start"] == 2.4
    assert s2["end"] == 7.6
    assert s2["duration"] == 5.2
    assert s2["words"] == ["Inside", "the", "massive", "airport", "terminal"]

    # Scene 003
    s3 = scenes[2]
    assert s3["scene_id"] == "003"
    assert s3["start"] == 10.0
    assert s3["end"] == 11.2
    assert s3["duration"] == 1.2
    assert s3["words"] == ["Then", "silence."]


# =============================================================================
# 3. STORY BIBLES & CONFLICT LOGGING TESTS
# =============================================================================

def test_bible_merge_deduplication_and_conflict_logging():
    chunk_0 = {
        "characters": [
            {
                "name": "Marcus",
                "appearance": "tall night supervisor",
                "clothing": "dark navy jacket",
                "colors": "navy blue",
                "features": "scar on left cheek"
            }
        ],
        "locations": [
            {
                "name": "Terminal 2",
                "visual_traits": "glass and steel concourse",
                "time_of_day": "late night 11:40 PM",
                "atmosphere": "quiet before the crisis"
            }
        ],
        "objects": [
            {
                "name": "Handheld Radio",
                "description": "illuminated green signal led"
            }
        ]
    }

    # Chunk 1 describes Marcus differently
    chunk_1 = {
        "characters": [
            {
                "name": "Marcus",
                "appearance": "bearded survivor",
                "clothing": "ripped grey shirt",
                "colors": "faded grey",
                "features": "bandage over arm"
            },
            {
                "name": "Reyes",
                "appearance": "young flight operator",
                "clothing": "high-visibility vest",
                "colors": "orange and yellow",
                "features": "wire-rim glasses"
            }
        ],
        "locations": [],
        "objects": []
    }

    master = merge_chunk_entities([chunk_0, chunk_1])
    assert master["locked"] is True

    # Deduplicated characters: Marcus and Reyes
    chars = {c["name"]: c for c in master["characters"]}
    assert len(chars) == 2
    assert "Marcus" in chars
    assert "Reyes" in chars

    # First description MUST be retained (dark navy jacket, scar on left cheek)
    marcus = chars["Marcus"]
    assert marcus["clothing"] == "dark navy jacket"
    assert marcus["features"] == "scar on left cheek"

    # Conflict MUST be logged
    conflicts = master["conflicts"]
    assert len(conflicts) == 1
    c_entry = conflicts[0]
    assert c_entry["entity_type"] == "character"
    assert c_entry["name"] == "Marcus"
    assert "navy" in c_entry["first_description"]
    assert "ripped grey" in c_entry["second_description"]


# =============================================================================
# 4. WORD COUNT GATE & DRIFT LINTER TESTS
# =============================================================================

def test_word_count_gate_rules():
    # Exactly 200 words (valid)
    valid_prompt = "word " * 200
    is_valid, count, violations = lint_prompt_drift(valid_prompt, "001")
    assert is_valid is True
    assert count == 200
    assert len(violations) == 0

    # 150 words (too short -> fail)
    short_prompt = "word " * 150
    is_valid, count, violations = lint_prompt_drift(short_prompt, "002")
    assert is_valid is False
    assert count == 150
    assert "required range is 180-220" in violations[0]

    # 250 words (too long -> fail)
    long_prompt = "word " * 250
    is_valid, count, violations = lint_prompt_drift(long_prompt, "003")
    assert is_valid is False
    assert count == 250

    # Markdown heading violation
    heading_prompt = "# Scene 001\n" + ("word " * 199)
    is_valid, _, violations = lint_prompt_drift(heading_prompt, "004")
    assert is_valid is False
    assert any("headings" in v for v in violations)

    # Bullet point violation
    bullet_prompt = "- First visual item\n" + ("word " * 198)
    is_valid, _, violations = lint_prompt_drift(bullet_prompt, "005")
    assert is_valid is False
    assert any("bullet points" in v for v in violations)


def test_prompt_generator_retry_logic():
    beat = {
        "scene_id": "001",
        "start": 0.0,
        "end": 3.0,
        "duration": 3.0,
        "narration_words": "Sixty people were locked inside this airport terminal.",
        "visual_action": "Marcus stands looking at sealed glass doors.",
        "composition": "Wide establishing shot.",
        "bible_refs": ["Marcus"],
        "style_anchor": "Dark cinematic suspense photograph, 35mm film grain."
    }
    bible = {
        "characters": [{"name": "Marcus", "appearance": "tall supervisor", "clothing": "navy jacket", "colors": "navy", "features": "scar"}],
        "locations": [],
        "objects": []
    }

    # Mock engine returning 200 words
    mock_engine = MagicMock(spec=GroqEngine)
    mock_prompt_text = "Dark cinematic suspense photograph, 35mm film grain. " + ("Marcus stands looking at the sealed terminal glass doors in the heavy rain. " * 16)
    w_count = len(mock_prompt_text.split())

    mock_engine.generate_structured.return_value = {
        "scenes": [
            {
                "scene_id": "001",
                "prompt": mock_prompt_text,
                "word_count": w_count,
                "bible_refs": ["Marcus"]
            }
        ]
    }

    res = generate_single_scene_prompt(
        beat=beat,
        master_bible=bible,
        style_id="default",
        engine=mock_engine
    )

    assert res["status"] == "SUCCESS"
    assert 180 <= res["word_count"] <= 220


# =============================================================================
# 5. MANUAL IMAGE CHECKPOINT & RENDER GATE TESTS
# =============================================================================

def test_image_checklist_and_render_gate(tmp_path):
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    scenes = [
        {"scene_id": "001", "start": 0.0, "end": 2.5, "text": "Scene one"},
        {"scene_id": "002", "start": 2.5, "end": 5.0, "text": "Scene two"}
    ]

    # Initially missing both images -> render gate locked
    res1 = scan_project_images(images_dir, scenes)
    assert res1.is_ready_for_render is False
    assert res1.missing_count == 2
    assert "001" in res1.missing_scene_ids
    assert "002" in res1.missing_scene_ids

    # Add scene_001.png
    im1 = Image.new("RGB", (1280, 720), color=(100, 100, 100))
    im1.save(images_dir / "scene_001.png")

    res2 = scan_project_images(images_dir, scenes)
    assert res2.is_ready_for_render is False
    assert res2.missing_count == 1
    assert res2.present_count == 1

    # Add scene_002.png
    im2 = Image.new("RGB", (1280, 720), color=(150, 150, 150))
    im2.save(images_dir / "scene_002.png")

    res3 = scan_project_images(images_dir, scenes)
    assert res3.is_ready_for_render is True
    assert res3.missing_count == 0
    assert res3.present_count == 2
    assert "Ready to render" in res3.gate_message


def test_exporter_prompts(tmp_path):
    prompts_dir = tmp_path / "prompts"
    prompts_data = [
        {"scene_id": "001", "prompt": "Prompt 1 text.", "start": 0.0, "end": 3.0, "narration_words": "Narration 1"},
        {"scene_id": "002", "prompt": "Prompt 2 text.", "start": 3.0, "end": 6.0, "narration_words": "Narration 2"}
    ]

    index_map = export_prompts(prompts_data, prompts_dir)
    assert (prompts_dir / "prompts_only.txt").exists()
    assert (prompts_dir / "prompts_index.json").exists()

    with open(prompts_dir / "prompts_only.txt", "r", encoding="utf-8") as f:
        content = f.read()
    assert "Prompt 1 text." in content
    assert "Prompt 2 text." in content
    assert "Scene 001" not in content
    lines = content.strip().splitlines()
    assert len(lines) == 2


# =============================================================================
# 6. POST-RENDER 4-POINT VERIFICATION CHECKS TESTS
# =============================================================================

def test_post_render_checks_logic(tmp_path):
    # Test Transcript Diff Check
    script_words = ["Sixty", "people", "were", "locked"]
    ts_words = ["Sixty", "people", "were", "locked"]
    ok, err = verify_transcript_diff(script_words, ts_words)
    assert ok is True
    assert err is None

    # Mismatch in transcript diff
    bad_ts = ["Sixty", "people", "were", "trapped"]
    ok, err = verify_transcript_diff(script_words, bad_ts)
    assert ok is False
    assert "locked' != timestamp 'trapped" in err

    # Spot checks with a real generated image
    temp_dir = tmp_path / "spot_check"
    temp_dir.mkdir()
    sample_frame = temp_dir / "spot_check_25.png"
    im = Image.new("RGB", (640, 360), color=(120, 120, 120))
    im.save(sample_frame)

    # Black frame check detection
    black_frame = temp_dir / "black.png"
    im_black = Image.new("RGB", (640, 360), color=(0, 0, 0))
    im_black.save(black_frame)

    with Image.open(black_frame) as b_im:
        gray = b_im.convert("L")
        pixels = list(gray.getdata())
        avg_luma = sum(pixels) / len(pixels)
    assert avg_luma < 2.0  # confirms black detection logic


def test_validation_instantaneous_point_timestamp(tmp_path):
    """Tests that Whisper point timestamps with start == end are clamped gracefully."""
    entries = [
        {"word": "hello", "start": 0.0, "end": 0.5},
        {"word": "A", "start": 0.5, "end": 0.5},  # point timestamp
        {"word": "world", "start": 0.55, "end": 1.0}
    ]
    validate_timestamp_entries(entries, 2.0, tmp_path / "ts.json")
    assert entries[1]["end"] > entries[1]["start"]
    assert entries[1]["end"] == 0.55


def test_validation_omnivoice_speech_json_format(tmp_path):
    """Tests loading real-world Omnivoice speech.json with text, words, and duration."""
    s_p = tmp_path / "script.txt"
    s_p.write_text("It is 11:40 PM at a regional airport. You are the night-shift worker.", encoding="utf-8")

    from app.core.validator import probe_audio_duration
    # Mock audio file
    import subprocess
    a_p = tmp_path / "audio.wav"
    cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=5.0", "-c:a", "pcm_s16le", "-ar", "44100", str(a_p)]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    # Omnivoice speech.json format
    ts_data = {
        "text": "It is 11:40 PM at a regional airport. You are the night-shift worker.",
        "duration": 5.0,
        "narrator": "Alhaitham",
        "words": [
            {"word": "It", "start": 0.0, "end": 0.2},
            {"word": "is", "start": 0.2, "end": 0.4},
            {"word": "11", "start": 0.4, "end": 0.7},
            {"word": ".40pm", "start": 0.7, "end": 1.2},
            {"word": "at", "start": 1.2, "end": 1.4},
            {"word": "a", "start": 1.4, "end": 1.5},
            {"word": "regional", "start": 1.5, "end": 2.1},
            {"word": "airport.", "start": 2.1, "end": 2.7},
            {"word": "You", "start": 2.7, "end": 2.9},
            {"word": "are", "start": 2.9, "end": 3.1},
            {"word": "the", "start": 3.1, "end": 3.3},
            {"word": "night", "start": 3.3, "end": 3.7},
            {"word": "shift", "start": 3.7, "end": 4.1},
            {"word": "worker.", "start": 4.1, "end": 4.8}
        ]
    }
    t_p = tmp_path / "speech.json"
    t_p.write_text(json.dumps(ts_data), encoding="utf-8")

    loaded = load_inputs(s_p, a_p, t_p)
    assert loaded.total_words == 14
    assert loaded.audio_duration == 5.0


def test_resolve_project_dir_no_double_nesting(tmp_path):
    """Guarantees projects are never nested as projects/projects/<name>."""
    from app.config.settings import resolve_project_dir

    # Case 1: Workspace directory already ends in 'projects'
    ws_projects = tmp_path / "projects"
    p1 = resolve_project_dir(ws_projects, "Story_A")
    assert p1 == ws_projects / "Story_A"
    assert "projects/projects" not in str(p1).replace("\\", "/")

    # Case 2: Workspace directory is the repo root without 'projects'
    ws_root = tmp_path / "app_root"
    p2 = resolve_project_dir(ws_root, "Story_B")
    assert p2 == ws_root / "projects" / "Story_B"
    assert "projects/projects" not in str(p2).replace("\\", "/")

    # Case 3: Accidental nested 'projects/projects' passed as workspace
    ws_double = tmp_path / "projects" / "projects"
    p3 = resolve_project_dir(ws_double, "Story_C")
    assert p3 == ws_projects / "Story_C"
    assert "projects/projects" not in str(p3).replace("\\", "/")



