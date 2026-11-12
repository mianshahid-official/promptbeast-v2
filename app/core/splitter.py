"""
Deterministic Scene Splitter for PromptBeast-V2 (Step 1).
Pure local deterministic algorithm without LLM.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

from app.core.loader import WordToken


def _ends_sentence(word: str) -> bool:
    """Checks whether the word ends with sentence punctuation (. ! ?)."""
    clean = word.rstrip(" \t\n\r\"'”’)]}")
    return clean.endswith((".", "!", "?"))


def _build_scene(scene_index: int, tokens: List[WordToken]) -> Dict[str, Any]:
    start_t = round(tokens[0].start, 3)
    end_t = round(tokens[-1].end, 3)
    dur = round(end_t - start_t, 3)
    word_strings = [w.word for w in tokens]
    word_timestamps = [w.to_dict() for w in tokens]

    return {
        "scene_id": f"{scene_index:03d}",
        "start": start_t,
        "end": end_t,
        "duration": dur,
        "words": word_strings,
        "word_timestamps": word_timestamps,
        "text": " ".join(word_strings)
    }


def split_scenes(words: List[WordToken]) -> List[Dict[str, Any]]:
    """
    Splits narration words into deterministic visual scenes based on:
    1. Silence gap > 2.0s: end current scene before gap, start next scene after gap.
    2. Cut condition:
       (a) duration (last word end - first word start) reaches 5.0s, OR
       (b) duration >= 2.0s AND last added word ends a sentence (. ! ?).
    3. Final scene may be shorter than 2s without padding.
    """
    if not words:
        return []

    scenes: List[Dict[str, Any]] = []
    current_tokens: List[WordToken] = [words[0]]

    for i in range(1, len(words)):
        curr_word = words[i]

        if not current_tokens:
            current_tokens = [curr_word]
            continue

        prev_word = current_tokens[-1]

        # Check silence gap > 2.0s between consecutive words
        gap = curr_word.start - prev_word.end
        if gap > 2.0:
            scenes.append(_build_scene(len(scenes) + 1, current_tokens))
            current_tokens = [curr_word]
            continue

        current_tokens.append(curr_word)
        scene_start = current_tokens[0].start
        current_dur = curr_word.end - scene_start

        # Check cut conditions:
        is_sentence_end = _ends_sentence(curr_word.word)
        if current_dur >= 5.0 or (current_dur >= 2.0 and is_sentence_end):
            scenes.append(_build_scene(len(scenes) + 1, current_tokens))
            current_tokens = []
            continue

    # Flush final tokens
    if current_tokens:
        scenes.append(_build_scene(len(scenes) + 1, current_tokens))

    return scenes


def save_chunks_checkpoint(scenes: List[Dict[str, Any]], checkpoint_path: Union[str, Path]) -> Path:
    """Saves Step 1 scenes to 01_chunks.json checkpoint."""
    path = Path(checkpoint_path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scenes, f, indent=2, ensure_ascii=False)
    return path


def load_chunks_checkpoint(checkpoint_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads Step 1 scenes from 01_chunks.json checkpoint."""
    path = Path(checkpoint_path).resolve()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
