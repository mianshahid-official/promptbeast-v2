"""
Project state and checkpoint lifecycle manager for PromptBeast-V2.
Handles 7-stage resume-on-crash checkpoints:
01_chunks.json -> 02_raw_entities.json -> 03_master_bible.json -> 04_storyboard_beats.json -> 05_final_prompts.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from app.core.project_manager import ProjectManager
from app.models.project import ProjectConfig, ProjectState, StoryProject


STAGE_CHECKPOINTS = [
    "01_chunks.json",
    "02_raw_entities.json",
    "03_master_bible.json",
    "04_storyboard_beats.json",
    "05_final_prompts.json"
]


class ProjectCheckpointManager:
    """Manages stage checkpoints for resume-on-crash functionality."""

    @staticmethod
    def get_project_dirs(project_dir: Union[str, Path]) -> Dict[str, Path]:
        root = Path(project_dir).resolve()
        prompts = root / "prompts"
        images = root / "images"
        output = root / "output"
        prompts.mkdir(parents=True, exist_ok=True)
        images.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        return {
            "root": root,
            "prompts": prompts,
            "images": images,
            "output": output
        }

    @staticmethod
    def save_checkpoint(project_dir: Union[str, Path], filename: str, data: Any) -> Path:
        root = Path(project_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        cp_path = root / filename
        with open(cp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return cp_path

    @staticmethod
    def load_checkpoint(project_dir: Union[str, Path], filename: str) -> Optional[Any]:
        root = Path(project_dir).resolve()
        cp_path = root / filename
        if not cp_path.exists():
            return None
        with open(cp_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_latest_checkpoint(project_dir: Union[str, Path]) -> Tuple[Optional[str], Optional[Any]]:
        """
        Returns (latest_checkpoint_name, loaded_data) or (None, None).
        Checks in reverse order so the latest passed stage is returned.
        """
        root = Path(project_dir).resolve()
        for cp_name in reversed(STAGE_CHECKPOINTS):
            cp_path = root / cp_name
            if cp_path.exists() and cp_path.stat().st_size > 0:
                try:
                    data = ProjectCheckpointManager.load_checkpoint(root, cp_name)
                    if data:
                        return cp_name, data
                except Exception:
                    continue
        return None, None


__all__ = [
    "ProjectManager",
    "StoryProject",
    "ProjectConfig",
    "ProjectState",
    "STAGE_CHECKPOINTS",
    "ProjectCheckpointManager"
]

