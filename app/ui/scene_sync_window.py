"""
Standalone Window for Scene Sync — PromptBeast Pro.
Launched directly via run-scene-sync.bat (python -m app.main --mode scene_sync).
Provides the dedicated one-pager assembly studio:
Audio + Word Timestamps JSON + Storyboard + Images Folder -> Final Synchronized Video.
"""
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QApplication,
)

from app.ui.theme import DARK_THEME_QSS
from app.ui.widgets.scene_sync_panel import SceneSyncPanel


class SceneSyncWindow(QMainWindow):
    """Standalone window hosting only the Scene Sync one-pager assembly tool."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("PromptBeast Pro — Scene Sync Studio (Standalone Video Assembler)")
        self.setMinimumSize(1080, 680)
        self.resize(1380, 860)

        # Apply dark theme
        self.setStyleSheet(DARK_THEME_QSS)

        # Central container
        central_widget = QWidget(self)
        central_layout = QVBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        # Top Bar Banner
        top_bar = QFrame()
        top_bar.setFixedHeight(54)
        top_bar.setStyleSheet("""
            QFrame {
                background-color: #0B0F17;
                border-bottom: 1px solid rgba(163, 255, 18, 0.25);
                padding: 4px 16px;
            }
        """)
        tb_layout = QHBoxLayout(top_bar)
        tb_layout.setContentsMargins(16, 6, 16, 6)
        tb_layout.setSpacing(12)

        lbl_title = QLabel("⚡ PROMPTBEAST PRO · Scene Sync")
        lbl_title.setStyleSheet("""
            color: #A3FF12;
            font-size: 14px;
            font-weight: 800;
            letter-spacing: 0.5px;
        """)
        tb_layout.addWidget(lbl_title)

        lbl_desc = QLabel("One-Pager Assembly: Audio + Speech JSON + Storyboard + Images ➔ Final Video")
        lbl_desc.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 500;")
        tb_layout.addWidget(lbl_desc)

        tb_layout.addStretch()

        badge_standalone = QLabel("Standalone Tool Mode")
        badge_standalone.setStyleSheet("""
            color: #38BDF8;
            font-size: 11px;
            font-weight: 700;
            background-color: rgba(56, 189, 248, 0.12);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 6px;
            padding: 4px 12px;
        """)
        tb_layout.addWidget(badge_standalone)

        central_layout.addWidget(top_bar)

        # Scene Sync All-in-One Panel
        self.scene_sync_panel = SceneSyncPanel(self)
        central_layout.addWidget(self.scene_sync_panel, 1)

        self.setCentralWidget(central_widget)

