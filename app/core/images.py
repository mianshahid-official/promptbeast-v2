"""
Manual Image Checkpoint and Render Gate for PromptBeast-V2 (Step 5).
Scans project images/ folder, validates image files, warns on low resolution,
and enforces the strict render gate (Render disabled until 100% of scenes have valid images).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from PIL import Image


SUPPORTED_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp"]


@dataclass
class ImageSceneStatus:
    scene_id: str
    start: float
    end: float
    narration: str
    word_count: int
    status: str  # "present", "missing", "invalid"
    image_filename: str
    image_path: Optional[Path] = None
    width: Optional[int] = None
    height: Optional[int] = None
    warning: Optional[str] = None
    error: Optional[str] = None


@dataclass
class ImageChecklistResult:
    items: List[ImageSceneStatus] = field(default_factory=list)
    total_scenes: int = 0
    present_count: int = 0
    missing_count: int = 0
    invalid_count: int = 0
    missing_scene_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    is_ready_for_render: bool = False
    gate_message: str = ""


def find_image_for_scene(images_dir: Path, scene_id: str) -> Optional[Path]:
    """
    Finds image file matching scene_id.
    Accepts: scene_001.png, scene_001.jpg, scene_001.webp, 001.png, scene1.png, etc.
    """
    clean_id = str(scene_id).zfill(3)
    num_str = str(int(clean_id)) if clean_id.isdigit() else clean_id

    candidate_stems = [
        f"scene_{clean_id}",
        f"scene_{num_str}",
        clean_id,
        num_str,
        f"prompt_{clean_id}",
    ]

    for stem in candidate_stems:
        for ext in SUPPORTED_EXTENSIONS:
            p = images_dir / f"{stem}{ext}"
            if p.exists() and p.is_file():
                return p

    # Fallback: scan directory for any file containing the scene_id
    for f in images_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            name_lower = f.stem.lower()
            if f"scene_{clean_id}" in name_lower or f"scene_{num_str}" in name_lower:
                return f

    return None


def validate_image_file(image_path: Path, target_resolution: Tuple[int, int] = (1280, 720)) -> Tuple[bool, Optional[Tuple[int, int]], Optional[str], Optional[str]]:
    """
    Validates that file is a readable image.
    Returns (is_valid, (width, height), warning, error).
    """
    try:
        with Image.open(image_path) as img:
            img.verify()

        # Reopen to read dimensions (verify closes the file handle)
        with Image.open(image_path) as img:
            w, h = img.size

        warning = None
        target_w, target_h = target_resolution
        if w < target_w or h < target_h:
            warning = f"Image resolution ({w}x{h}) is smaller than project resolution ({target_w}x{target_h})."

        return True, (w, h), warning, None
    except Exception as e:
        return False, None, None, f"Image file '{image_path.name}' is unreadable or corrupted: {e}"


def scan_project_images(
    images_dir: Union[str, Path],
    scenes_data: List[Dict[str, Any]],
    target_resolution: Tuple[int, int] = (1280, 720)
) -> ImageChecklistResult:
    """
    Audits the images/ folder against all scenes.
    Enforces Render Gate: Render button stays disabled until 100% of scenes have valid images.
    """
    img_dir = Path(images_dir).resolve()
    img_dir.mkdir(parents=True, exist_ok=True)

    items: List[ImageSceneStatus] = []
    missing_ids: List[str] = []
    all_warnings: List[str] = []
    present_cnt = 0
    missing_cnt = 0
    invalid_cnt = 0

    for sc in scenes_data:
        s_id = str(sc.get("scene_id", "001")).zfill(3)
        start = float(sc.get("start", 0.0))
        end = float(sc.get("end", 0.0))
        narration = str(sc.get("narration_words") or sc.get("text", "")).strip()
        prompt_text = str(sc.get("prompt", "")).strip()
        w_count = len(prompt_text.split()) if prompt_text else 0

        found_path = find_image_for_scene(img_dir, s_id)
        expected_name = f"scene_{s_id}.png"

        if not found_path:
            items.append(
                ImageSceneStatus(
                    scene_id=s_id,
                    start=start,
                    end=end,
                    narration=narration[:50] + ("..." if len(narration) > 50 else ""),
                    word_count=w_count,
                    status="missing",
                    image_filename=expected_name,
                    image_path=None
                )
            )
            missing_ids.append(s_id)
            missing_cnt += 1
            continue

        # File found; validate readable image
        is_valid, dims, warn, err = validate_image_file(found_path, target_resolution)
        if not is_valid:
            items.append(
                ImageSceneStatus(
                    scene_id=s_id,
                    start=start,
                    end=end,
                    narration=narration[:50] + ("..." if len(narration) > 50 else ""),
                    word_count=w_count,
                    status="invalid",
                    image_filename=found_path.name,
                    image_path=found_path,
                    error=err
                )
            )
            invalid_cnt += 1
            missing_ids.append(s_id)
        else:
            w, h = dims or (None, None)
            if warn:
                all_warnings.append(f"Scene {s_id}: {warn}")
            items.append(
                ImageSceneStatus(
                    scene_id=s_id,
                    start=start,
                    end=end,
                    narration=narration[:50] + ("..." if len(narration) > 50 else ""),
                    word_count=w_count,
                    status="present",
                    image_filename=found_path.name,
                    image_path=found_path,
                    width=w,
                    height=h,
                    warning=warn
                )
            )
            present_cnt += 1

    total = len(scenes_data)
    is_ready = (total > 0 and missing_cnt == 0 and invalid_cnt == 0)

    if is_ready:
        gate_msg = f"All {total} scenes have valid images. Ready to render."
    else:
        reasons = []
        if missing_cnt > 0:
            reasons.append(f"{missing_cnt} missing ({', '.join(missing_ids[:10])}{'...' if len(missing_ids) > 10 else ''})")
        if invalid_cnt > 0:
            reasons.append(f"{invalid_cnt} invalid/corrupted")
        gate_msg = f"Render gate locked: {', '.join(reasons)}. Drop images into {img_dir}."

    return ImageChecklistResult(
        items=items,
        total_scenes=total,
        present_count=present_cnt,
        missing_count=missing_cnt,
        invalid_count=invalid_cnt,
        missing_scene_ids=missing_ids,
        warnings=all_warnings,
        is_ready_for_render=is_ready,
        gate_message=gate_msg
    )
