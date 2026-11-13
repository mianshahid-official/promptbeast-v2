"""
Prompt and Asset Exporter for PromptBeast-V2 (Step 5).
Exports per-scene text prompts, combined prompts_only.txt, and prompts_index.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union


def export_prompts(
    prompts_data: List[Dict[str, Any]],
    prompts_dir: Union[str, Path]
) -> Dict[str, Any]:
    """
    Exports prompts to the project's prompts/ directory:
    - prompt_001.txt, prompt_002.txt, ...
    - prompts_only.txt (headed by scene number and time range)
    - prompts_index.json (mapping scene_id -> prompt file -> expected image file name)
    """
    p_dir = Path(prompts_dir).resolve()
    p_dir.mkdir(parents=True, exist_ok=True)

    index_map: Dict[str, Any] = {}
    combined_lines: List[str] = []

    for item in prompts_data:
        s_id = str(item.get("scene_id", "001")).zfill(3)
        raw_prompt = str(item.get("prompt", "")).strip()
        clean_prompt = " ".join(raw_prompt.split())
        start = item.get("start", 0.0)
        end = item.get("end", 0.0)
        word_count = len(clean_prompt.split())

        expected_image = f"scene_{s_id}.png"

        # Append to combined prompts list (pure prompt only, 1 line per scene, no headers)
        if clean_prompt:
            combined_lines.append(clean_prompt)

        # Add to index
        index_map[s_id] = {
            "scene_id": s_id,
            "expected_image": expected_image,
            "start": start,
            "end": end,
            "word_count": word_count,
            "narration": item.get("narration_words") or item.get("text", "")
        }

    # Write prompts_only.txt
    combined_file = p_dir / "prompts_only.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write("\n".join(combined_lines).strip() + "\n")

    # Write prompts_index.json
    index_file = p_dir / "prompts_index.json"
    with open(index_file, "w", encoding="utf-8") as f:
        json.dump(index_map, f, indent=2, ensure_ascii=False)

    return index_map

