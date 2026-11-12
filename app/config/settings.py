"""
User settings persistence using local JSON config.
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from app.config.constants import (
    DEFAULT_MODAL_API_URL,
    DEFAULT_MODEL,
    DEFAULT_MIN_DURATION,
    DEFAULT_PREFERRED_DURATION,
    DEFAULT_MAX_DURATION,
    DEFAULT_BATCH_SIZE,
)


def get_default_workspace_dir() -> Path:
    base_dir = Path(__file__).resolve().parent.parent.parent
    proj_dir = base_dir / "projects"
    proj_dir.mkdir(parents=True, exist_ok=True)
    return proj_dir


def resolve_project_dir(workspace_dir: Union[str, Path], project_name: str) -> Path:
    """
    Resolves canonical project directory without duplicate 'projects/projects' nesting.
    Guarantees project is placed directly in <workspace_projects>/<project_name>.
    """
    ws = Path(workspace_dir).resolve()
    while ws.name.lower() == "projects" and ws.parent.name.lower() == "projects":
        ws = ws.parent

    if ws.name.lower() != "projects":
        proj_root = ws / "projects"
    else:
        proj_root = ws

    proj_root.mkdir(parents=True, exist_ok=True)
    clean_name = "".join(c for c in project_name if c.isalnum() or c in (" ", "_", "-")).strip()
    if not clean_name:
        clean_name = "Untitled_Project"
    return proj_root / clean_name


class AppSettings(BaseModel):
    modal_api_url: str = Field(default="")
    api_key: Optional[str] = Field(default="")
    auth_token: Optional[str] = Field(default="")
    default_model: str = Field(default=DEFAULT_MODEL)
    default_subject_id: str = Field(default="history")
    default_visual_style_id: str = Field(default="zombie_survival")
    default_min_duration: float = Field(default=DEFAULT_MIN_DURATION)
    default_preferred_duration: float = Field(default=4.5)
    default_max_duration: float = Field(default=6.0)
    default_batch_size: int = Field(default=DEFAULT_BATCH_SIZE)
    transcription_engine: str = Field(default="local")
    workspace_directory: str = Field(default_factory=lambda: str(get_default_workspace_dir()))
    recent_projects: List[str] = Field(default_factory=list)
    auto_save_interval_seconds: int = Field(default=60)
    window_geometry: Optional[Dict[str, int]] = None


class SettingsManager:
    """Manages reading and writing application settings."""

    _SETTINGS_FILE = Path.home() / ".promptbeast" / "settings.json"

    @classmethod
    def get_settings_file_path(cls) -> Path:
        cls._SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        return cls._SETTINGS_FILE

    @classmethod
    def load_settings(cls) -> AppSettings:
        path = cls.get_settings_file_path()
        if not path.exists():
            # Check environment variables
            env_url = os.environ.get("PROMPTBEAST_API_URL")
            env_key = os.environ.get("GROQ_API_KEY") or os.environ.get("PROMPTBEAST_API_KEY")
            env_token = os.environ.get("AUTH_TOKEN")
            settings = AppSettings()
            if env_url:
                settings.modal_api_url = env_url
            if env_key:
                settings.api_key = env_key
            if env_token:
                settings.auth_token = env_token
            cls.save_settings(settings)
            return settings

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            settings = AppSettings(**data)
            default_ws = str(get_default_workspace_dir())
            ws_path = Path(settings.workspace_directory) if settings.workspace_directory else None
            needs_save = False

            # Ensure workspace directory points to active project repository without double projects
            if (
                not settings.workspace_directory
                or "PromptBeast_Projects" in settings.workspace_directory
                or not ws_path
                or not ws_path.exists()
                or (ws_path.name.lower() == "projects" and ws_path.parent.name.lower() == "projects")
                or (Path(__file__).resolve().parent.parent.parent.name.lower() == "v2promptbeast" and "promptbeast-v2" not in str(ws_path).lower() and "v2promptbeast" not in str(ws_path).lower())
            ):
                settings.workspace_directory = default_ws
                needs_save = True

            # Normalize recent projects list to remove stale or double-nested paths
            cleaned_recents: List[str] = []
            for rp in settings.recent_projects:
                norm_rp = rp.replace("/projects/projects", "/projects").replace("\\projects\\projects", "\\projects")
                # Also check if it exists in current workspace
                candidate_curr = str(Path(default_ws) / Path(norm_rp).name)
                if Path(candidate_curr).exists() and candidate_curr not in cleaned_recents:
                    cleaned_recents.append(candidate_curr)
                elif Path(norm_rp).exists() and norm_rp not in cleaned_recents:
                    cleaned_recents.append(norm_rp)

            if cleaned_recents != settings.recent_projects:
                settings.recent_projects = cleaned_recents
                needs_save = True

            if needs_save:
                cls.save_settings(settings)
            return settings
        except Exception:
            return AppSettings()

    @classmethod
    def save_settings(cls, settings: AppSettings) -> None:
        path = cls.get_settings_file_path()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(settings.model_dump(), f, indent=2)

    @classmethod
    def add_recent_project(cls, project_dir: str) -> None:
        settings = cls.load_settings()
        resolved = str(Path(project_dir).resolve())
        if resolved in settings.recent_projects:
            settings.recent_projects.remove(resolved)
        settings.recent_projects.insert(0, resolved)
        settings.recent_projects = settings.recent_projects[:10]
        cls.save_settings(settings)
