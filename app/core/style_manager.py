"""
Style Profile and Master Prompt Manager for PromptBeast.
Discovers and loads self-contained styles from the styles/ directory.
Each style contains:
- styles/<style_id>/profile.json (rules, locks, format, subtitles)
- styles/<style_id>/master_prompt.txt (style-specific instructions for Groq prompt expansion)
"""
import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class StyleProfile:
    id: str
    name: str
    category: str = "Visual Style"
    description: str = ""
    output_format: str = "16:9"
    subtitle_style: str = "standard_clean"
    hero_lock: str = ""
    style_lock: str = ""
    master_prompt: str = ""
    is_fallback: bool = False

    @property
    def style_prompt(self) -> str:
        return self.style_lock or self.description or ""


class StyleManager:
    """Manages active visual styles, profiles, and per-style master prompts."""

    _active_style: Optional[StyleProfile] = None
    _profiles_cache: Dict[str, StyleProfile] = {}
    _styles_root: Optional[Path] = None

    @classmethod
    def get_style_profile(cls, style_id: str) -> StyleProfile:
        return cls.get_style(style_id)

    @classmethod
    def get_styles_root(cls) -> Path:
        if cls._styles_root is not None and cls._styles_root.exists():
            return cls._styles_root

        # Check candidate locations for styles/ directory
        candidates = [
            Path("styles").resolve(),
            Path(__file__).resolve().parent.parent.parent / "styles",
            Path(__file__).resolve().parent.parent / "styles",
        ]
        for c in candidates:
            if c.exists() and c.is_dir():
                cls._styles_root = c
                return c

        default_path = (Path(__file__).resolve().parent.parent.parent / "styles").resolve()
        cls._styles_root = default_path
        return default_path

    @classmethod
    def normalize_style_id(cls, raw_id: str) -> str:
        if not raw_id:
            return "default"
        s = raw_id.strip().lower().replace("-", "_")

        # Canonical aliases
        if s in ("zombie", "zombies", "zombie_survival"):
            return "zombie_survival"
        elif s in ("whatif", "what_if", "whatif_flat_vector"):
            return "what_if"
        elif s in ("ink", "ink_wash", "graphic_sketch"):
            return "ink"
        elif s in ("default", "fallback", "standard"):
            return "default"
        elif s in ("century", "century_old_photograph", "vintage_photo", "vintage_photograph"):
            return "century_old_photograph"
        elif s in ("2d_stick", "2d_stickman", "2d_stick_figure"):
            return "2d_stickman"
        elif s in ("3d_stick", "3d_stickman", "3d_stick_figure"):
            return "3d_stickman"
        elif s in ("3d_red_blue_stickman", "red_blue_stickman"):
            return "3d_red_blue_stickman"
        return s

    @classmethod
    def load_default_master_prompt(cls) -> str:
        root = cls.get_styles_root()
        default_file = root / "default" / "master_prompt.txt"
        if default_file.exists():
            try:
                content = default_file.read_text(encoding="utf-8").strip()
                if content:
                    return content
            except Exception:
                pass

        # Secondary fallback: config/system_prompts/default.txt or image_prompt_generation.txt
        sys_candidates = [
            Path("config/system_prompts/default.txt"),
            Path("config/system_prompts/image_prompt_generation.txt"),
            Path(__file__).resolve().parent.parent.parent / "config" / "system_prompts" / "default.txt",
            Path(__file__).resolve().parent.parent.parent / "config" / "system_prompts" / "image_prompt_generation.txt",
        ]
        for sc in sys_candidates:
            if sc.exists():
                try:
                    return sc.read_text(encoding="utf-8").strip()
                except Exception:
                    pass

        return (
            "You are the master AI Image Prompt Architect. "
            "Generate concise, direct prose image prompts in JSON with no headings or bullet points."
        )

    @classmethod
    def get_style(cls, style_id: str) -> StyleProfile:
        norm_id = cls.normalize_style_id(style_id)
        if norm_id in cls._profiles_cache:
            return cls._profiles_cache[norm_id]

        root = cls.get_styles_root()
        style_dir = root / norm_id

        # If not found directly, check if what_if points to whatif_flat_vector or vice versa
        if not style_dir.exists():
            if norm_id == "what_if" and (root / "whatif_flat_vector").exists():
                style_dir = root / "whatif_flat_vector"
            else:
                style_dir = root / "default"

        profile_path = style_dir / "profile.json"
        master_prompt_path = style_dir / "master_prompt.txt"

        data: Dict = {}
        if profile_path.exists():
            try:
                data = json.loads(profile_path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"[StyleManager] Warning loading {profile_path}: {e}")

        master_prompt = ""
        if master_prompt_path.exists():
            try:
                master_prompt = master_prompt_path.read_text(encoding="utf-8").strip()
            except Exception as e:
                print(f"[StyleManager] Warning loading {master_prompt_path}: {e}")

        # Fallback to default master prompt if missing
        if not master_prompt:
            master_prompt = cls.load_default_master_prompt()

        s_id = data.get("id", norm_id)
        name = data.get("name", s_id.replace("_", " ").title())
        profile = StyleProfile(
            id=s_id,
            name=name,
            category=data.get("category", "Visual Style"),
            description=data.get("description", ""),
            output_format=data.get("output_format", "16:9"),
            subtitle_style=data.get("subtitle_style", "standard_clean"),
            hero_lock=data.get("hero_lock", ""),
            style_lock=data.get("style_lock", ""),
            master_prompt=master_prompt,
            is_fallback=(norm_id not in ("zombie_survival", "what_if", "ink") and not profile_path.exists())
        )

        cls._profiles_cache[norm_id] = profile
        return profile

    @classmethod
    def set_active_style(cls, style_id: str) -> StyleProfile:
        profile = cls.get_style(style_id)
        cls._active_style = profile
        return profile

    @classmethod
    def get_active_style(cls) -> StyleProfile:
        if cls._active_style is None:
            cls._active_style = cls.get_style("default")
        return cls._active_style

    @classmethod
    def get_active_master_prompt(cls) -> str:
        return cls.get_active_style().master_prompt

    @classmethod
    def get_all_styles(cls) -> List[StyleProfile]:
        root = cls.get_styles_root()
        results = []
        if root.exists():
            for d in sorted(root.iterdir()):
                if d.is_dir() and (d / "profile.json").exists():
                    results.append(cls.get_style(d.name))
        return results

