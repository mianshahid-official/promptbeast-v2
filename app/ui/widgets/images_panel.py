"""
Manual Image Checkpoint and Render Panel for PromptBeast-V2 (Steps 5 & 6).
- Table checklist: scene number, time range, narration excerpt, prompt word count, image status, image filename.
- User drops images into images/ folder (scene_001.png, scene_001.jpg, scene_001.webp).
- "Check Images" button re-scans folder.
- Render Gate: Render button strictly disabled until 100% of scenes have valid images.
- Executes Step 6 FFmpeg rendering and 4-point verification check.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QApplication,
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

from app.core.images import ImageChecklistResult, scan_project_images
from app.core.renderer import PostRenderVerificationResult, VideoRenderer


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
    """Step 5 & 6 UI: Image Checklist Table and Render Gate."""
    render_completed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        from app.config.settings import SettingsManager, resolve_project_dir
        settings = SettingsManager.load_settings()
        self.project_dir: Path = resolve_project_dir(settings.workspace_directory, "Active_Project")
        self.images_dir: Path = self.project_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_data: List[Dict[str, Any]] = []
        self.audio_path: Optional[Path] = None
        self.audio_duration: float = 0.0
        self.timestamp_words: List[Dict[str, Any]] = []
        self.script_words: List[str] = []
        self.checklist_result: Optional[ImageChecklistResult] = None
        self.render_worker: Optional[VideoRenderWorker] = None

        self._init_ui()

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
        self.scenes_data = scenes_data
        self.audio_path = audio_path
        self.audio_duration = audio_duration
        self.timestamp_words = timestamp_words or []
        self.script_words = script_words or []
        self.lbl_images_dir.setText(f"📁 Images Directory: {self.images_dir}")
        self.refresh_checklist()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(16, 16, 16, 16)

        # Header Banner
        header = QFrame()
        header.setStyleSheet("background: #18181b; border-radius: 8px; padding: 12px; border: 1px solid #27272a;")
        h_layout = QVBoxLayout(header)
        h_title = QLabel("🖼️ Step 5: Manual Image Checkpoint & Render Gate")
        h_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f4f4f5;")
        self.lbl_images_dir = QLabel("📁 Images Directory: (No project loaded)")
        self.lbl_images_dir.setStyleSheet("color: #a1a1aa; font-family: monospace; font-size: 11px;")
        h_layout.addWidget(h_title)
        h_layout.addWidget(self.lbl_images_dir)
        layout.addWidget(header)

        # Status / Gate Banner
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

        # Action Buttons Row
        btn_row = QHBoxLayout()
        self.btn_open_folder = QPushButton("📂 Open Images Folder")
        self.btn_open_folder.clicked.connect(self._open_images_folder)
        self.btn_check_images = QPushButton("🔄 Check Images")
        self.btn_check_images.clicked.connect(self.refresh_checklist)
        self.btn_render = QPushButton("🎥 Render Final Video (Step 6)")
        self.btn_render.setStyleSheet("""
            QPushButton {
                background: #16a34a;
                color: white;
                font-weight: bold;
                font-size: 13px;
                padding: 8px 18px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #15803d; }
            QPushButton:disabled { background: #27272a; color: #52525b; border: 1px solid #3f3f46; }
        """)
        self.btn_render.setEnabled(False)
        self.btn_render.clicked.connect(self._on_start_render)

        btn_row.addWidget(self.btn_open_folder)
        btn_row.addWidget(self.btn_check_images)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_render)
        layout.addLayout(btn_row)

        # Checklist Table
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

        # Render Progress
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

    def refresh_checklist(self):
        """Scans images/ folder and updates checklist table and render gate."""
        if not self.scenes_data:
            self.gate_banner.setText("Render Gate: No scenes loaded. Run pipeline in Studio first.")
            self.btn_render.setEnabled(False)
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

        self.render_worker = VideoRenderWorker(
            project_dir=self.project_dir,
            scenes=self.scenes_data,
            images_map=images_map,
            audio_path=self.audio_path or (self.project_dir / "narration.mp3"),
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
