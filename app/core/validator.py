"""
Input validation and auditing engine for PromptBeast-V2.
Enforces strict validation rules: fails loudly, never guesses, never auto-fixes.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class ValidationError(Exception):
    """Raised when any pipeline input or integrity check fails."""
    pass


def normalize_token(text: str) -> str:
    """Strips all punctuation and lowercases for word coverage comparison."""
    if not text:
        return ""
    return re.sub(r"[^\w]", "", text).lower()


def probe_audio_duration(audio_path: str | Path) -> float:
    """
    Probes audio file duration using ffprobe.
    Fails loudly if ffprobe fails or duration is missing.
    """
    path = Path(audio_path).resolve()
    if not path.exists():
        raise ValidationError(f"Audio file does not exist: {path}")

    # Verify ffprobe is available
    ffprobe_bin = shutil.which("ffprobe")
    if not ffprobe_bin:
        # Check standard local paths on Windows
        candidate_paths = [
            Path("C:/ffmpeg/bin/ffprobe.exe"),
            Path("C:/Program Files/ffmpeg/bin/ffprobe.exe"),
            Path("ffmpeg/bin/ffprobe.exe")
        ]
        for c in candidate_paths:
            if c.exists():
                ffprobe_bin = str(c)
                break

    cmd = [
        ffprobe_bin or "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        dur_str = res.stdout.strip()
        dur = float(dur_str)
        if dur <= 0:
            raise ValidationError(f"Audio file '{path}' has invalid duration: {dur}s")
        return dur
    except Exception as e:
        raise ValidationError(f"Failed to probe audio duration for '{path}' via ffprobe: {e}")


def validate_timestamp_entries(entries: List[Dict[str, Any]], audio_duration: float, file_path: Path) -> None:
    """
    Validates word-level timestamp entries:
    - Every entry has word (string), start (number), end (number)
    - start < end
    - times non-decreasing
    - every end <= audio duration
    """
    if not entries:
        raise ValidationError(f"Timestamp file '{file_path}' contains an empty words array.")

    prev_start = -1.0
    for idx, item in enumerate(entries):
        if not isinstance(item, dict):
            raise ValidationError(f"Timestamp file '{file_path}' entry at index {idx} is not an object: {item}")

        word = item.get("word")
        if word is None or not isinstance(word, str):
            raise ValidationError(f"Timestamp file '{file_path}' entry at index {idx} missing 'word' string: {item}")

        start = item.get("start")
        end = item.get("end")
        if start is None or not isinstance(start, (int, float)):
            raise ValidationError(f"Timestamp file '{file_path}' entry at index {idx} invalid 'start': {start}")
        if end is None or not isinstance(end, (int, float)):
            raise ValidationError(f"Timestamp file '{file_path}' entry at index {idx} invalid 'end': {end}")

        start = float(start)
        end = float(end)

        if start >= end:
            raise ValidationError(
                f"Timestamp file '{file_path}' entry at index {idx} ('{word}') has start ({start:.3f}) >= end ({end:.3f})"
            )

        if start < prev_start:
            raise ValidationError(
                f"Timestamp file '{file_path}' entry at index {idx} ('{word}') has decreasing start time "
                f"({start:.3f} < previous {prev_start:.3f})"
            )
        prev_start = start

        # Epsilon tolerance of 0.15s for container/codec padding
        if end > (audio_duration + 0.15):
            raise ValidationError(
                f"Timestamp file '{file_path}' entry at index {idx} ('{word}') end time ({end:.3f}s) "
                f"exceeds audio duration ({audio_duration:.3f}s)"
            )


def validate_word_coverage(
    script_words: List[str],
    ts_words: List[str],
    script_path: Path,
    ts_path: Path
) -> None:
    """
    Validates word coverage:
    Timestamp words joined in order must equal script words exactly.
    Compares after stripping punctuation and lowercasing.
    Fails on first mismatch naming position, expected word, and found word.
    """
    clean_script = [normalize_token(w) for w in script_words]
    clean_ts = [normalize_token(w) for w in ts_words]

    # Filter out empty tokens if any
    filtered_script = [(idx, orig, clean) for idx, (orig, clean) in enumerate(zip(script_words, clean_script)) if clean]
    filtered_ts = [(idx, orig, clean) for idx, (orig, clean) in enumerate(zip(ts_words, clean_ts)) if clean]

    min_len = min(len(filtered_script), len(filtered_ts))
    for i in range(min_len):
        _, s_orig, s_clean = filtered_script[i]
        _, t_orig, t_clean = filtered_ts[i]
        if s_clean != t_clean:
            raise ValidationError(
                f"Word coverage mismatch at position {i}: "
                f"expected '{s_orig}' (from script '{script_path}'), "
                f"found '{t_orig}' (from timestamp JSON '{ts_path}'). "
                f"Normalized comparison: '{s_clean}' != '{t_clean}'."
            )

    if len(filtered_script) != len(filtered_ts):
        raise ValidationError(
            f"Word count mismatch between files: "
            f"script '{script_path}' has {len(filtered_script)} words, "
            f"timestamp JSON '{ts_path}' has {len(filtered_ts)} words."
        )


def validate_duration_difference(audio_duration: float, last_ts_end: float, audio_path: Path, ts_path: Path) -> None:
    """
    Audio duration vs last timestamp end: difference must be <= 2.0s, else fail and show both numbers.
    """
    diff = abs(audio_duration - last_ts_end)
    if diff > 2.0:
        raise ValidationError(
            f"Duration mismatch: audio duration is {audio_duration:.2f}s ('{audio_path}'), "
            f"last timestamp end is {last_ts_end:.2f}s ('{ts_path}'). "
            f"Difference of {diff:.2f}s exceeds allowed maximum of 2.0s."
        )


def lint_prompt_drift(
    prompt: str,
    scene_id: str,
    target_min: int = 180,
    target_max: int = 220
) -> Tuple[bool, int, List[str]]:
    """
    Lints a generated prompt for rules:
    - Word count must be in [target_min, target_max] (180-220)
    - No markdown headings (# or ##)
    - No bullet-point lists (* or - at start of line)
    Returns (is_valid, word_count, list_of_violations).
    """
    words = prompt.split()
    count = len(words)
    violations = []

    if count < target_min or count > target_max:
        violations.append(f"Scene {scene_id} prompt has {count} words; required range is {target_min}-{target_max}.")

    # Heading check
    if re.search(r"^\s*#{1,6}\s+", prompt, re.MULTILINE):
        violations.append(f"Scene {scene_id} prompt contains markdown headings, which are strictly forbidden.")

    # Bullet point check
    if re.search(r"^\s*[\*\-]\s+", prompt, re.MULTILINE) or re.search(r"^\s*\d+\.\s+", prompt, re.MULTILINE):
        violations.append(f"Scene {scene_id} prompt contains bullet points or numbered lists, which are strictly forbidden.")

    return len(violations) == 0, count, violations
