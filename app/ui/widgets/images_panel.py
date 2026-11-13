"""
Manual Image Checkpoint and Render Panel for PromptBeast-V2 (Steps 5 & 6).
- Project Selector dropdown: effortlessly switch between projects or auto-load active project.
- Chrome Browser Profile Generator: launches Google Chrome profiles (acc1..acc6) with Gemini Bulk Canvas,
  copies prompts to Windows clipboard, and auto-configures image download directory.
- Live Auto-Watch: detects new image drops into images/ folder every 3 seconds.
- Table checklist: scene number, time range, narration excerpt, prompt word count, image status, image filename.
- Render Gate: Render button strictly disabled until 100% of scenes have valid images.
- Step 6 FFmpeg rendering with multi-threaded encoding and 4-point verification check.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.browser_profiles_manager import BrowserProfilesManager
from app.core.images import ImageChecklistResult, scan_project_images
from app.core.renderer import PostRenderVerificationResult, VideoRenderer


def natural_sort_key(p: Union[Path, str]):
    """Natural sorting key: puts numbered filenames in numerical order (001, 002, 010)."""
    name = Path(p).name
    nums = re.findall(r"\d+", name)
    return (0, int(nums[0]), name.lower()) if nums else (1, 0, name.lower())


class VideoRenderWorker(QThread):
    """Background worker executing Step 6 FFmpeg video render and verification."""
    render_progress = Signal(int, str)
    render_finished = Signal(str, object)
    render_failed = Signal(str)

    def __init__(
        self,
        project_dir: Path,
        scenes: List[Dict[str, Any]],
        images_map: Dict[str, Path],
        audio_path: Path,
        audio_duration: float,
        timestamp_words: List[Dict[str, Any]],
        script_words: List[str],
        aspect_ratio: str = "16:9",
        parent=None
    ):
        super().__init__(parent)
        self.project_dir = project_dir
        self.scenes = scenes
        self.images_map = images_map
        self.audio_path = audio_path
        self.audio_duration = audio_duration
        self.timestamp_words = timestamp_words
        self.script_words = script_words
        self.aspect_ratio = aspect_ratio

    def run(self):
        try:
            renderer = VideoRenderer(
                project_dir=self.project_dir,
                aspect_ratio=self.aspect_ratio,
                progress_callback=lambda p, m: self.render_progress.emit(p, m)
            )
            out_file, v_res = renderer.render(
                scenes=self.scenes,
                images_map=self.images_map,
                audio_path=self.audio_path,
                audio_duration=self.audio_duration,
                timestamp_words=self.timestamp_words,
                script_words=self.script_words
            )
            self.render_finished.emit(str(out_file), v_res)
        except Exception as e:
            self.render_failed.emit(str(e))


class ImagesPanel(QWidget):
    """Step 5 & 6 UI: Image Checklist Table, Chrome Bulk Generator, and Render Gate."""
    render_completed = Signal(str)
    sync_completed = Signal(list, str, object)

    def __init__(self, parent=None):
        super().__init__(parent)
        from app.config.settings import SettingsManager, resolve_project_dir
        self.settings = SettingsManager.load_settings()

        self.project_dir: Path = resolve_project_dir(self.settings.workspace_directory, "Active_Project")
        self.images_dir: Path = self.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)

        self.scenes_data: List[Dict[str, Any]] = []
        self.audio_path: Optional[Path] = None
        self.audio_duration: float = 0.0
        self.timestamp_words: List[Dict[str, Any]] = []
        self.script_words: List[str] = []
        self.checklist_result: Optional[ImageChecklistResult] = None
        self.render_worker: Optional[VideoRenderWorker] = None
        self._last_image_count: int = -1

        self._init_ui()
        self.discover_projects()

        # Auto-watch timer for image drop-ins
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(3000)
        self._watch_timer.timeout.connect(self._on_watch_tick)
        self._watch_timer.start()

    # =========================================================================
    # UI INITIALIZATION
    # =========================================================================

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # ── Header Banner ───────────────────────────────────────────────────
        header = QFrame()
        header.setStyleSheet("background: #18181b; border-radius: 8px; padding: 12px; border: 1px solid #27272a;")
        h_layout = QVBoxLayout(header)
        h_layout.setSpacing(8)

        # Title row
        h_top_row = QHBoxLayout()
        h_title = QLabel("🖼️ Step 5: Manual Image Checkpoint & Render Gate")
        h_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f4f4f5;")
        h_top_row.addWidget(h_title)
        h_top_row.addStretch()

        # Active Project Selector Row
        proj_sel_row = QHBoxLayout()
        proj_sel_row.setSpacing(8)
        lbl_proj = QLabel("Active Project:")
        lbl_proj.setStyleSheet("color: #a1a1aa; font-weight: bold; font-size: 12px;")
        proj_sel_row.addWidget(lbl_proj)

        self.project_combo = QComboBox()
        self.project_combo.setMinimumWidth(260)
        self.project_combo.setStyleSheet("""
            QComboBox {
                background: #09090b; color: #f4f4f5; border: 1px solid #3f3f46;
                border-radius: 4px; padding: 4px 8px; font-size: 12px; font-weight: 500;
            }
            QComboBox::drop-down { border: none; }
            QComboBox QAbstractItemView {
                background: #18181b; color: #f4f4f5; selection-background-color: #27272a;
            }
        """)
        self.project_combo.currentIndexChanged.connect(self._on_project_combo_changed)
        proj_sel_row.addWidget(self.project_combo)

        self.btn_refresh_projects = QPushButton("🔄 Refresh Projects")
        self.btn_refresh_projects.setStyleSheet("""
            QPushButton {
                background: #27272a; color: #e4e4e7; border: 1px solid #3f3f46;
                border-radius: 4px; padding: 4px 10px; font-size: 11px;
            }
            QPushButton:hover { background: #3f3f46; }
        """)
        self.btn_refresh_projects.clicked.connect(self.discover_projects)
        proj_sel_row.addWidget(self.btn_refresh_projects)
        proj_sel_row.addStretch()

        h_top_row.addLayout(proj_sel_row)
        h_layout.addLayout(h_top_row)

        self.lbl_images_dir = QLabel("📁 Images Directory: (No project loaded)")
        self.lbl_images_dir.setStyleSheet("color: #a1a1aa; font-family: monospace; font-size: 11px;")
        h_layout.addWidget(self.lbl_images_dir)
        layout.addWidget(header)

        # ── Chrome Bulk Generator & Action Controls Bar ─────────────────────
        action_bar = QFrame()
        action_bar.setStyleSheet("""
            QFrame {
                background: #0d1117; border-radius: 8px; padding: 10px; border: 1px solid #30363d;
            }
        """)
        act_layout = QHBoxLayout(action_bar)
        act_layout.setSpacing(10)
        act_layout.setContentsMargins(10, 8, 10, 8)

        # Chrome Profile Selection
        lbl_profile = QLabel("👤 Profile:")
        lbl_profile.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        act_layout.addWidget(lbl_profile)

        self.profile_combo = QComboBox()
        self.profile_combo.setMinimumWidth(180)
        self.profile_combo.setStyleSheet("""
            QComboBox {
                background: #161b22; color: #e6edf3; border: 1px solid #30363d;
                border-radius: 4px; padding: 4px 8px; font-size: 12px;
            }
            QComboBox QAbstractItemView {
                background: #161b22; color: #e6edf3; selection-background-color: #1f6feb;
            }
        """)
        self._refresh_profiles_combo()
        act_layout.addWidget(self.profile_combo)

        # Launch Chrome & Generate Button
        self.btn_open_chrome = QPushButton("🚀 Open Chrome & Generate Images")
        self.btn_open_chrome.setToolTip(
            "Opens Chrome with selected profile, configures image download directory directly to this project's images/ folder, and copies all scene prompts to clipboard."
        )
        self.btn_open_chrome.setStyleSheet("""
            QPushButton {
                background: #0284c7; color: white; font-weight: bold; font-size: 12px;
                padding: 6px 14px; border-radius: 6px; border: 1px solid #0369a1;
            }
            QPushButton:hover { background: #0369a1; }
        """)
        self.btn_open_chrome.clicked.connect(self._on_open_chrome_generator)
        act_layout.addWidget(self.btn_open_chrome)

        # Copy All Prompts Button
        self.btn_copy_prompts = QPushButton("📋 Copy All Prompts")
        self.btn_copy_prompts.setStyleSheet("""
            QPushButton {
                background: #21262d; color: #c9d1d9; font-weight: bold; font-size: 12px;
                padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d;
            }
            QPushButton:hover { background: #30363d; color: white; }
        """)
        self.btn_copy_prompts.clicked.connect(self._on_copy_all_prompts)
        act_layout.addWidget(self.btn_copy_prompts)

        act_layout.addStretch(1)

        # Open Images Folder
        self.btn_open_folder = QPushButton("📂 Open Images Folder")
        self.btn_open_folder.setStyleSheet("""
            QPushButton {
                background: #21262d; color: #c9d1d9; font-size: 12px;
                padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d;
            }
            QPushButton:hover { background: #30363d; color: white; }
        """)
        self.btn_open_folder.clicked.connect(self._open_images_folder)
        act_layout.addWidget(self.btn_open_folder)

        # Check Images Button
        self.btn_check_images = QPushButton("🔄 Check Images")
        self.btn_check_images.setStyleSheet("""
            QPushButton {
                background: #21262d; color: #c9d1d9; font-size: 12px;
                padding: 6px 12px; border-radius: 6px; border: 1px solid #30363d;
            }
            QPushButton:hover { background: #30363d; color: white; }
        """)
        self.btn_check_images.clicked.connect(self.refresh_checklist)
        act_layout.addWidget(self.btn_check_images)

        # Render Final Video Button
        self.btn_render = QPushButton("🎥 Render Final Video (Step 6)")
        self.btn_render.setStyleSheet("""
            QPushButton {
                background: #16a34a;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 7px 18px;
                border-radius: 6px;
                border: 1px solid #15803d;
            }
            QPushButton:hover { background: #15803d; }
            QPushButton:disabled { background: #27272a; color: #52525b; border: 1px solid #3f3f46; }
        """)
        self.btn_render.setEnabled(False)
        self.btn_render.clicked.connect(self._on_start_render)
        act_layout.addWidget(self.btn_render)

        layout.addWidget(action_bar)

        # ── Render Gate Banner ──────────────────────────────────────────────
        self.gate_banner = QLabel("Render Gate: Waiting for scenes...")
        self.gate_banner.setStyleSheet("""
            background: #27272a;
            color: #f59e0b;
            font-weight: bold;
            font-size: 13px;
            padding: 10px;
            border-radius: 6px;
            border: 1px solid #3f3f46;
        """)
        layout.addWidget(self.gate_banner)

        # ── Checklist Table ─────────────────────────────────────────────────
        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels([
            "Scene #",
            "Time Range",
            "Narration Excerpt",
            "Prompt Words",
            "Status",
            "Image Filename"
        ])
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(2, QHeaderView.Stretch)
        header_view.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        self.table.setStyleSheet("""
            QTableWidget {
                background-color: #09090b;
                gridline-color: #27272a;
                color: #e4e4e7;
                font-size: 12px;
                border: 1px solid #27272a;
            }
            QHeaderView::section {
                background-color: #18181b;
                color: #a1a1aa;
                padding: 6px;
                font-weight: bold;
                border: 1px solid #27272a;
            }
        """)
        layout.addWidget(self.table, 1)

        # ── Render Progress ─────────────────────────────────────────────────
        self.render_progress_bar = QProgressBar()
        self.render_progress_bar.setRange(0, 100)
        self.render_progress_bar.setValue(0)
        self.render_progress_bar.setVisible(False)
        self.render_progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                text-align: center;
                color: #f4f4f5;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #16a34a;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.render_progress_bar)

    # =========================================================================
    # PROJECT DISCOVERY & LOADING
    # =========================================================================

    def _refresh_profiles_combo(self):
        """Populates profile dropdown with acc1..acc6 and their login states."""
        self.profile_combo.clear()
        profiles = BrowserProfilesManager.get_all_profiles()
        for p in profiles:
            name = p["name"]
            status = p["status_label"]
            label = f"{name} ({status})"
            self.profile_combo.addItem(label, name)

    def discover_projects(self):
        """Discovers existing projects from workspace and populates the project selector."""
        from app.config.settings import SettingsManager, resolve_project_dir
        self.settings = SettingsManager.load_settings()
        ws = Path(self.settings.workspace_directory)
        projects_root = ws / "projects" if (ws / "projects").is_dir() else ws

        candidates: List[Path] = []
        if projects_root.is_dir():
            for item in projects_root.iterdir():
                if item.is_dir() and not item.name.startswith("."):
                    candidates.append(item)

        # Sort projects by modification time (most recent first)
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        self.project_combo.blockSignals(True)
        self.project_combo.clear()

        # Prioritize selecting "test run" or the first project with actual scenes
        best_idx = 0
        found_test_run = False
        found_real_project = False

        for idx, p in enumerate(candidates):
            self.project_combo.addItem(p.name, str(p.resolve()))
            has_scenes = (p / "05_final_prompts.json").is_file() or (p / "01_chunks.json").is_file()
            if p.name.lower() == "test run":
                best_idx = idx
                found_test_run = True
            elif not found_test_run and has_scenes and not found_real_project:
                best_idx = idx
                found_real_project = True

        if candidates:
            self.project_combo.setCurrentIndex(best_idx)
            chosen_path = Path(self.project_combo.itemData(best_idx))
            self.load_project_from_dir(chosen_path)

        self.project_combo.blockSignals(False)

    def _on_project_combo_changed(self, idx: int):
        path_str = self.project_combo.itemData(idx)
        if path_str:
            self.load_project_from_dir(Path(path_str))

    def load_project_from_dir(self, p_dir: Path):
        """Loads a project from directory, reading chunks, prompts, audio, and timestamps."""
        self.project_dir = Path(p_dir).resolve()
        self.images_dir = self.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_images_dir.setText(f"📁 Images Directory: {self.images_dir}")

        # Sync project_combo
        idx = self.project_combo.findText(self.project_dir.name)
        if idx >= 0 and self.project_combo.currentIndex() != idx:
            self.project_combo.blockSignals(True)
            self.project_combo.setCurrentIndex(idx)
            self.project_combo.blockSignals(False)

        # 1. Load scenes
        scenes = []
        p_final = self.project_dir / "05_final_prompts.json"
        p_chunks = self.project_dir / "01_chunks.json"
        p_idx = self.project_dir / "prompts" / "prompts_index.json"
        p_state = self.project_dir / "pipeline_state.json"

        if p_final.is_file():
            try:
                with open(p_final, "r", encoding="utf-8") as f:
                    scenes = json.load(f)
            except Exception:
                pass
        elif p_chunks.is_file():
            try:
                with open(p_chunks, "r", encoding="utf-8") as f:
                    raw_chunks = json.load(f)
                scenes = [
                    {
                        "scene_id": str(c.get("scene_id", f"{i+1:03d}")).zfill(3),
                        "start": c.get("start", 0.0),
                        "end": c.get("end", 0.0),
                        "duration": c.get("duration", 0.0),
                        "narration_words": c.get("text", ""),
                        "prompt": c.get("prompt", "")
                    }
                    for i, c in enumerate(raw_chunks)
                ]
            except Exception:
                pass
        elif p_idx.is_file():
            try:
                with open(p_idx, "r", encoding="utf-8") as f:
                    idx_data = json.load(f)
                scenes = [
                    {
                        "scene_id": s_id,
                        "start": data.get("start", 0.0),
                        "end": data.get("end", 0.0),
                        "duration": data.get("duration", 0.0),
                        "narration_words": data.get("narration", ""),
                        "prompt": data.get("prompt", "")
                    }
                    for s_id, data in idx_data.items()
                ]
            except Exception:
                pass

        self.scenes_data = scenes

        # 2. Look for audio file in project
        audio_candidates = ["speech.mp3", "narration.mp3", "audio.mp3", "speech.wav", "narration.wav", "audio.wav"]
        self.audio_path = None
        for ac in audio_candidates:
            ap = self.project_dir / ac
            if ap.is_file():
                self.audio_path = ap
                break
        if not self.audio_path:
            for f in self.project_dir.iterdir():
                if f.is_file() and f.suffix.lower() in (".mp3", ".wav", ".aac", ".m4a"):
                    self.audio_path = f
                    break

        # 3. Determine audio duration
        self.audio_duration = 0.0
        if self.audio_path and self.audio_path.is_file():
            try:
                import soundfile as sf
                info = sf.info(str(self.audio_path))
                self.audio_duration = round(info.duration, 2)
            except Exception:
                if self.scenes_data:
                    self.audio_duration = round(max((s.get("end", 0.0) for s in self.scenes_data), default=0.0), 2)

        # 4. Load timestamp words
        self.timestamp_words = []
        p_speech_json = self.project_dir / "speech.json"
        if p_speech_json.is_file():
            try:
                with open(p_speech_json, "r", encoding="utf-8") as f:
                    ts_data = json.load(f)
                    if isinstance(ts_data, dict) and "words" in ts_data:
                        self.timestamp_words = ts_data["words"]
                    elif isinstance(ts_data, list):
                        self.timestamp_words = ts_data
            except Exception:
                pass

        if not self.timestamp_words and p_chunks.is_file():
            try:
                with open(p_chunks, "r", encoding="utf-8") as f:
                    raw_chunks = json.load(f)
                words = []
                for c in raw_chunks:
                    words.extend(c.get("word_timestamps", []))
                self.timestamp_words = words
            except Exception:
                pass

        # 5. Load script words
        self.script_words = []
        for txt_file in self.project_dir.glob("*.txt"):
            if "prompt" not in txt_file.name.lower() and "bible" not in txt_file.name.lower():
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        text = f.read()
                        self.script_words = [w for w in text.split() if w.strip()]
                        break
                except Exception:
                    pass

        self.refresh_checklist()

    # =========================================================================
    # CHROME BULK GENERATION & CLIPBOARD
    # =========================================================================

    def _get_clean_prompts_payload(self) -> str:
        """Retrieves clean 1-line-per-scene prompts for bulk generator."""
        prompts_file = self.project_dir / "prompts" / "prompts_only.txt"
        if prompts_file.is_file():
            try:
                with open(prompts_file, "r", encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]
                    if lines:
                        return "\n".join(lines)
            except Exception:
                pass

        if self.scenes_data:
            lines = []
            for sc in self.scenes_data:
                prompt_text = sc.get("prompt", "").strip()
                if prompt_text:
                    clean = " ".join(prompt_text.split())
                    lines.append(clean)
            if lines:
                return "\n".join(lines)

        return ""

    def _on_copy_all_prompts(self):
        """Copies clean prompts_only to Windows clipboard."""
        payload = self._get_clean_prompts_payload()
        if not payload:
            QMessageBox.warning(self, "No Prompts", "No prompts found for this project.")
            return

        clipboard = QApplication.clipboard()
        clipboard.setText(payload)
        count = len(payload.splitlines())

        orig_txt = self.btn_copy_prompts.text()
        self.btn_copy_prompts.setText(f"✓ Copied {count} Prompts!")
        QTimer.singleShot(2500, lambda: self.btn_copy_prompts.setText(orig_txt))

    def _on_open_chrome_generator(self):
        """
        Launches Google Chrome with selected profile, configures direct image downloads
        to project's images/ folder, and copies all scene prompts to clipboard.
        """
        profile_name = self.profile_combo.currentData() or "acc1"
        payload = self._get_clean_prompts_payload()
        prompt_count = len(payload.splitlines()) if payload else 0

        # Copy prompts to clipboard
        if payload:
            clipboard = QApplication.clipboard()
            clipboard.setText(payload)

        # Configure Chrome download directory directly to this project's images/
        BrowserProfilesManager.configure_download_dir(profile_name, self.images_dir)

        # Launch Chrome
        ok, msg = BrowserProfilesManager.open_browser_profile(
            profile_name,
            download_dir=self.images_dir,
            prompts_text=payload
        )

        if not ok:
            QMessageBox.critical(self, "Chrome Launch Failed", f"Could not launch Chrome:\n\n{msg}")
            return

        QMessageBox.information(
            self,
            "Chrome Bulk Generator Launched",
            f"🚀 Chrome opened successfully with profile '{profile_name}'!\n\n"
            f"• {prompt_count} scene prompts copied to your clipboard.\n"
            f"• Chrome automatic download directory configured to:\n  {self.images_dir}\n\n"
            "Workflow Instructions:\n"
            "1. In Chrome, paste your prompts into the bulk image generator (Ctrl+V).\n"
            "2. Generate your images and download them.\n"
            "3. The images will automatically appear in the checklist below.\n"
            "4. Once 100% of scenes have valid images, click 'Render Final Video'!"
        )

    # =========================================================================
    # CHECKLIST & RENDER GATE
    # =========================================================================

    def _on_watch_tick(self):
        """Periodic auto-watch: updates checklist if image files were added/removed."""
        if not self.isVisible() or not self.images_dir.exists():
            return

        try:
            curr_count = len([f for f in self.images_dir.iterdir() if f.is_file()])
            if curr_count != self._last_image_count:
                self._last_image_count = curr_count
                self.refresh_checklist()
        except OSError:
            pass

    def refresh_checklist(self):
        """Scans images/ folder and updates checklist table and render gate."""
        if not self.scenes_data:
            self.gate_banner.setText("Render Gate: No scenes loaded. Select an active project above or run pipeline in Production.")
            self.gate_banner.setStyleSheet("""
                background: #27272a; color: #f59e0b; font-weight: bold;
                font-size: 13px; padding: 10px; border-radius: 6px; border: 1px solid #3f3f46;
            """)
            self.btn_render.setEnabled(False)
            self.table.setRowCount(0)
            return

        self.checklist_result = scan_project_images(self.images_dir, self.scenes_data)
        res = self.checklist_result

        # Update table
        self.table.setRowCount(len(res.items))
        for row_idx, item in enumerate(res.items):
            # Scene #
            item_num = QTableWidgetItem(f"Scene {item.scene_id}")
            item_num.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 0, item_num)

            # Time Range
            item_time = QTableWidgetItem(f"{item.start:.2f}s - {item.end:.2f}s")
            item_time.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 1, item_time)

            # Narration
            item_narr = QTableWidgetItem(item.narration)
            self.table.setItem(row_idx, 2, item_narr)

            # Prompt Words
            item_words = QTableWidgetItem(str(item.word_count) if item.word_count else "-")
            item_words.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row_idx, 3, item_words)

            # Status
            item_status = QTableWidgetItem(item.status.upper())
            item_status.setTextAlignment(Qt.AlignCenter)
            if item.status == "present":
                item_status.setForeground(QColor("#4ade80"))
            elif item.status == "missing":
                item_status.setForeground(QColor("#f87171"))
            else:
                item_status.setForeground(QColor("#fbbf24"))
            self.table.setItem(row_idx, 4, item_status)

            # Filename
            item_fn = QTableWidgetItem(item.image_filename)
            self.table.setItem(row_idx, 5, item_fn)

        # Update Render Gate Banner
        if res.is_ready_for_render:
            self.gate_banner.setText(f"✅ {res.gate_message}")
            self.gate_banner.setStyleSheet("""
                background: #14532d; color: #4ade80; font-weight: bold;
                font-size: 13px; padding: 10px; border-radius: 6px; border: 1px solid #16a34a;
            """)
            self.btn_render.setEnabled(True)
        else:
            self.gate_banner.setText(f"⛔ {res.gate_message}")
            self.gate_banner.setStyleSheet("""
                background: #450a0a; color: #f87171; font-weight: bold;
                font-size: 13px; padding: 10px; border-radius: 6px; border: 1px solid #dc2626;
            """)
            self.btn_render.setEnabled(False)

    def _open_images_folder(self):
        self.images_dir.mkdir(parents=True, exist_ok=True)
        os.startfile(str(self.images_dir))

    def _on_start_render(self):
        if not self.checklist_result or not self.checklist_result.is_ready_for_render:
            QMessageBox.critical(self, "Render Gate Locked", "Cannot render: some scene images are missing or invalid.")
            return

        # Prepare images map: scene_id -> image path
        images_map: Dict[str, Path] = {}
        for item in self.checklist_result.items:
            if item.image_path:
                images_map[item.scene_id] = item.image_path

        self.btn_render.setEnabled(False)
        self.render_progress_bar.setVisible(True)
        self.render_progress_bar.setValue(0)

        # Use audio path if available or fallback to narration.mp3
        audio_p = self.audio_path or (self.project_dir / "speech.mp3")
        if not audio_p.exists():
            audio_p = self.project_dir / "narration.mp3"

        self.render_worker = VideoRenderWorker(
            project_dir=self.project_dir,
            scenes=self.scenes_data,
            images_map=images_map,
            audio_path=audio_p,
            audio_duration=self.audio_duration,
            timestamp_words=self.timestamp_words,
            script_words=self.script_words,
            parent=self
        )
        self.render_worker.render_progress.connect(self._on_render_progress)
        self.render_worker.render_finished.connect(self._on_render_finished)
        self.render_worker.render_failed.connect(self._on_render_failed)
        self.render_worker.start()

    def _on_render_progress(self, val: int, msg: str):
        self.render_progress_bar.setValue(val)
        self.gate_banner.setText(f"🎬 {msg}")

    def _on_render_finished(self, out_path_str: str, v_res: PostRenderVerificationResult):
        self.btn_render.setEnabled(True)
        self.render_progress_bar.setValue(100)
        self.gate_banner.setText(f"🎉 Render Complete & Verified: {Path(out_path_str).name}")
        self.gate_banner.setStyleSheet("""
            background: #14532d; color: #4ade80; font-weight: bold;
            font-size: 13px; padding: 10px; border-radius: 6px; border: 1px solid #16a34a;
        """)
        QMessageBox.information(
            self,
            "Video Ready",
            f"Video rendered successfully and passed all 4 post-render checks!\n\nOutput: {out_path_str}"
        )
        self.render_completed.emit(out_path_str)

    def _on_render_failed(self, error_msg: str):
        self.btn_render.setEnabled(True)
        self.gate_banner.setText(f"❌ Render Error: {error_msg}")
        self.gate_banner.setStyleSheet("""
            background: #450a0a; color: #f87171; font-weight: bold;
            font-size: 13px; padding: 10px; border-radius: 6px; border: 1px solid #dc2626;
        """)
        QMessageBox.critical(self, "Render Failed", f"Render failed:\n\n{error_msg}")

    # =========================================================================
    # BACKWARD COMPATIBILITY INTERFACES
    # =========================================================================

    def set_project_context(
        self,
        project_dir: Path,
        scenes_data: List[Dict[str, Any]],
        audio_path: Optional[Path] = None,
        audio_duration: float = 0.0,
        timestamp_words: Optional[List[Dict[str, Any]]] = None,
        script_words: Optional[List[str]] = None
    ):
        self.project_dir = Path(project_dir).resolve()
        self.images_dir = self.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.lbl_images_dir.setText(f"📁 Images Directory: {self.images_dir}")

        if scenes_data:
            self.scenes_data = scenes_data
            self.audio_path = audio_path
            self.audio_duration = audio_duration
            self.timestamp_words = timestamp_words or []
            self.script_words = script_words or []
            self.refresh_checklist()
        else:
            # If empty scenes passed, check if project_dir has scenes on disk
            has_disk_scenes = (self.project_dir / "05_final_prompts.json").is_file() or (self.project_dir / "01_chunks.json").is_file()
            if has_disk_scenes:
                self.load_project_from_dir(self.project_dir)
            elif not self.scenes_data:
                self.load_project_from_dir(self.project_dir)

    def set_project(self, project_dir: Union[Path, str]):
        self.load_project_from_dir(Path(project_dir))

    def set_audio_path(self, audio_path: Union[Path, str]):
        if audio_path:
            self.audio_path = Path(audio_path)

    @property
    def scenes(self) -> List[Any]:
        return self.scenes_data

    @scenes.setter
    def scenes(self, val: List[Any]):
        if val:
            if hasattr(val[0], "start_time"):
                # VisualScene objects
                self.scenes_data = [
                    {
                        "scene_id": getattr(s, "scene_id", f"{getattr(s, 'index', i+1):03d}"),
                        "start": getattr(s, "start_time", 0.0),
                        "end": getattr(s, "end_time", 0.0),
                        "duration": getattr(s, "duration", 0.0),
                        "narration_words": getattr(s, "narration_text", ""),
                        "prompt": getattr(s, "image_prompt", "")
                    }
                    for i, s in enumerate(val)
                ]
            elif isinstance(val[0], dict):
                self.scenes_data = val
        else:
            self.scenes_data = []
        self.refresh_checklist()

    @property
    def narration(self) -> Any:
        return None

    @narration.setter
    def narration(self, val: Any):
        if val and hasattr(val, "duration_seconds"):
            self.audio_duration = val.duration_seconds

    def _populate_table(self):
        self.refresh_checklist()

    def _refresh_generate_btn_label(self):
        pass

    def _scan_images(self):
        self.refresh_checklist()
