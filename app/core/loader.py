"""
Input Loader for PromptBeast-V2.
The user provides ONLY 3 files:
1. Script (.txt, UTF-8)
2. Audio (.mp3 or .wav)
3. Timestamp JSON (word-level timestamps)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

from app.core.validator import (
    ValidationError,
    normalize_token,
    probe_audio_duration,
    validate_duration_difference,
    validate_timestamp_entries,
    validate_word_coverage,
)


@dataclass
class WordToken:
    """Represents a single word with timestamp and original script casing."""
    word: str
    start: float
    end: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3)
        }


@dataclass
class LoadedInputs:
    """Container for validated pipeline inputs."""
    script_text: str
    script_path: Path
    audio_path: Path
    audio_duration: float
    timestamp_path: Path
    words: List[WordToken] = field(default_factory=list)

    @property
    def total_words(self) -> int:
        return len(self.words)

    @property
    def script_words(self) -> List[str]:
        return [w.word for w in self.words]


def load_inputs(
    script_path: Union[str, Path],
    audio_path: Union[str, Path],
    timestamp_path: Union[str, Path]
) -> LoadedInputs:
    """
    Step 0 — Load & validate (fail loudly, never guess).
    Validates all three files, probes audio, audits word coverage, and attaches
    exact original script words to timestamps.
    """
    # 1. Check all files exist and are readable
    s_path = Path(script_path).resolve()
    a_path = Path(audio_path).resolve()
    t_path = Path(timestamp_path).resolve()

    if not s_path.exists() or not s_path.is_file():
        raise ValidationError(f"Script file does not exist or is not a readable file: {s_path}")
    if not a_path.exists() or not a_path.is_file():
        raise ValidationError(f"Audio file does not exist or is not a readable file: {a_path}")
    if not t_path.exists() or not t_path.is_file():
        raise ValidationError(f"Timestamp file does not exist or is not a readable file: {t_path}")

    # 2. Read script (must be non-empty UTF-8 text)
    try:
        with open(s_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    except Exception as e:
        raise ValidationError(f"Failed to read script file '{s_path}' as UTF-8: {e}")

    if not script_text.strip():
        raise ValidationError(f"Script file '{s_path}' is empty.")

    # 3. Probe audio duration
    audio_dur = probe_audio_duration(a_path)

    # 4. Parse timestamp JSON
    try:
        with open(t_path, "r", encoding="utf-8") as f:
            raw_ts = json.load(f)
    except Exception as e:
        raise ValidationError(f"Failed to parse timestamp JSON from '{t_path}': {e}")

    if isinstance(raw_ts, dict) and "words" in raw_ts:
        ts_entries = raw_ts["words"]
    elif isinstance(raw_ts, list):
        ts_entries = raw_ts
    else:
        raise ValidationError(
            f"Timestamp JSON '{t_path}' format unrecognized: expected a JSON array or an object with 'words' key."
        )

    # Validate individual timestamp entries against audio duration
    validate_timestamp_entries(ts_entries, audio_dur, t_path)

    # Validate difference between audio duration and last timestamp end
    last_end = float(ts_entries[-1]["end"])
    validate_duration_difference(audio_dur, last_end, a_path, t_path)

    # 5. Extract words and audit word coverage
    script_raw_words = script_text.split()
    ts_raw_words = [str(item["word"]).strip() for item in ts_entries]

    is_omnivoice = False
    if isinstance(raw_ts, dict) and "text" in raw_ts:
        raw_text_clean = normalize_token(str(raw_ts["text"]))
        script_text_clean = normalize_token(script_text)
        if raw_text_clean == script_text_clean or (raw_text_clean and script_text_clean and (raw_text_clean in script_text_clean or script_text_clean in raw_text_clean)):
            is_omnivoice = True

    if not is_omnivoice:
        validate_word_coverage(script_raw_words, ts_raw_words, s_path, t_path, allow_similarity_fallback=True)

    # 6. Build WordToken list
    tokens: List[WordToken] = []
    script_tokens = [w for w in script_raw_words if normalize_token(w)]
    ts_valid_entries = [e for e in ts_entries if normalize_token(str(e.get("word", "")))]

    # If exact 1-to-1 word alignment, attach original script words with original casing
    if len(script_tokens) == len(ts_valid_entries) and all(
        normalize_token(s) == normalize_token(str(e.get("word", "")))
        for s, e in zip(script_tokens, ts_valid_entries)
    ):
        for s_word, ts_item in zip(script_tokens, ts_valid_entries):
            tokens.append(
                WordToken(
                    word=s_word,
                    start=float(ts_item["start"]),
                    end=float(ts_item["end"])
                )
            )
    else:
        # For Whisper/Omnivoice timestamp JSONs where word counts differ due to phonetic/digit tokenization,
        # use timestamp entry words with acoustic timestamps so scene splitting matches the audio exactly
        for ts_item in ts_entries:
            w_text = str(ts_item.get("word", "")).strip()
            if not w_text:
                continue
            s_t = float(ts_item["start"])
            e_t = float(ts_item["end"])
            if s_t == e_t:
                e_t = round(s_t + 0.05, 3)
            tokens.append(
                WordToken(
                    word=w_text,
                    start=s_t,
                    end=e_t
                )
            )

    return LoadedInputs(
        script_text=script_text,
        script_path=s_path,
        audio_path=a_path,
        audio_duration=audio_dur,
        timestamp_path=t_path,
        words=tokens
    )

