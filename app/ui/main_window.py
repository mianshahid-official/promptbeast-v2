"""
Master Application Window for PromptBeast Production Studio.
Unified 4-Step Pipeline:
1. 🎙️ Story Narration + Local Neural TTS Voiceover (Kokoro & Piper)
2. 🎨 AI Storyboard + 3-Pane Bulk Prompts Studio
3. 🖼️ Image Assets Ingestion + Scene Auto-Binding
4. 🎬 Subtitle Styling, Ken Burns Motion + Native FFmpeg Video Render
"""
import os
import json
from pathlib import Path
from typing import Optional, List
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, Signal, QUrl
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QCheckBox,
    QFileDialog,
    QMessageBox,
    QScrollArea,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QStatusBar,
    QPlainTextEdit,
    QTabWidget,
    QApplication,
    QListView,
    QGridLayout,
)

from app.ui.widgets.custom_combo import PBComboBox
from app.ui.widgets.sidebar_nav import SidebarNav
from app.ui.widgets.live_board_panel import LiveBoardPanel
from app.ui.widgets.profiles_panel import ProfilesManagerPanel
from app.ui.widgets.style_thumbnail_matrix import StyleThumbnailMatrix
from app.core.browser_profiles_manager import BrowserProfilesManager
from PySide6.QtWidgets import QStackedWidget
from app.config.constants import (
    APP_NAME,
    APP_DISPLAY_NAME,
    APP_VERSION,
    DEFAULT_MIN_DURATION,
    DEFAULT_PREFERRED_DURATION,
    DEFAULT_MAX_DURATION,
)
from app.config.settings import SettingsManager
from app.models.project import StoryProject, ProjectConfig, ProjectState, ImageCountMode, TimingMode
from app.models.scene import VisualScene
from app.models.timeline import NarrationData
from app.models.job import JobStage, JobStatus, ProgressInfo
from app.presets.loader import PresetManager
from app.core.story_loader import StoryLoader
from app.core.timestamp_parser import TimestampParser
from app.core.timeline_engine import TimelineEngine
from app.core.validation import TimelineValidator
from app.core.project_manager import ProjectManager
from app.core.export import StoryboardExporter
from app.core.video_renderer import VideoRenderer
from app.ui.connection_worker import ConnectionCheckWorker
from app.ai.job_worker import CloudJobWorker, SceneRegenerateWorker
from app.ai.schemas import JobCreateRequest, SceneRegenerateRequest

from app.ui.theme import (
    DARK_THEME_QSS,
    COLOR_BRAND_PRIMARY,
    COLOR_BRAND_EMERALD,
    COLOR_BRAND_MINT,
    COLOR_BRAND_DEEP,
    COLOR_BG_MAIN,
    COLOR_BG_SECONDARY,
    COLOR_BG_WORKSPACE,
    COLOR_GLASS_SURFACE,
    COLOR_GLASS_SURFACE_2,
    COLOR_HOVER_SURFACE,
    COLOR_BORDER_SUBTLE,
    COLOR_BORDER_GREEN,
    COLOR_BORDER_ACTIVE,
    COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY,
    COLOR_TEXT_MUTED,
    COLOR_STATUS_SUCCESS,
    COLOR_STATUS_WARNING,
    COLOR_STATUS_ERROR,
    COLOR_STATUS_DISABLED,
)
from app.ui.widgets.scene_editor import SceneEditorWidget
from app.ui.widgets.prompt_grid import PromptGridWidget
from app.ui.widgets.progress_panel import ProgressPanelWidget
from app.ui.widgets.status_banner import StatusBannerWidget
from app.ui.widgets.audio_transcription_panel import AudioTranscriptionPanel
from app.ui.widgets.image_asset_panel import ImageAssetPanel
from app.ui.widgets.images_panel import ImagesPanel
from app.ui.widgets.render_panel import RenderPanel
from app.ui.dialogs.settings_dialog import SettingsDialog
from app.ui.dialogs.preset_editor_dialog import PresetEditorDialog
from app.ui.dialogs.export_dialog import ExportDialog
from app.ui.dialogs.recovery_dialog import RecoveryDialog
from app.ui.dialogs.project_history_dialog import ProjectHistoryDialog
from app.ui.dialogs.style_preview_dialog import StylePreviewDialog
from app.ui.widgets.narration_workflow_panel import NarrationWorkflowPanel



class SinglePageNavHelper:
    """Provides backward-compatibility for legacy tab index calls by smooth-scrolling to section and switching tabs."""
    def __init__(self, scroll_area, section_map, studio_tabs=None, stacked_pages=None, sidebar=None):
        self.scroll_area = scroll_area
        self.section_map = section_map
        self.studio_tabs = studio_tabs
        self.stacked_pages = stacked_pages
        self.sidebar = sidebar
        self._current_index = 0

    def setCurrentIndex(self, idx: int):
        self._current_index = idx
        widget = self.section_map.get(idx)
        if widget and self.stacked_pages:
            self.stacked_pages.setCurrentWidget(widget)
        elif widget and self.scroll_area:
            self.scroll_area.ensureWidgetVisible(widget)
        if self.sidebar:
            idx_to_key = {0: "production", 1: "studio", 2: "images", 3: "deliverables", 4: "profiles"}
            key = idx_to_key.get(idx)
            if key and hasattr(self.sidebar, "set_current_page"):
                self.sidebar.set_current_page(key)
        if self.studio_tabs:
            if idx == 2:  # Tab 2: Images
                self.studio_tabs.setCurrentIndex(1)
            elif idx == 3:  # Tab 3: Render Studio
                self.studio_tabs.setCurrentIndex(0)

    def currentIndex(self) -> int:
        return self._current_index


class MainWindow(QMainWindow):
    """Main application window for PromptBeast Production Studio."""

    def __init__(self, style_id: Optional[str] = None):
        super().__init__()

        # Resolve active style profile from launcher or default
        from app.core.style_manager import StyleManager
        if style_id:
            self.active_style_profile = StyleManager.set_active_style(style_id)
        else:
            self.active_style_profile = StyleManager.get_active_style()

        self.setWindowTitle(f"{APP_DISPLAY_NAME} — Style: {self.active_style_profile.name}")
        self.setMinimumSize(1024, 600)
        self.resize(1440, 840)

        # Initialize presets & settings
        PresetManager.initialize()
        self.settings = SettingsManager.load_settings()

        # Active Project
        self.project = StoryProject(
            name="Untitled_Project",
            config=ProjectConfig(),
            state=ProjectState()
        )
        self.project.config.visual_style_id = self.active_style_profile.id
        self.project.state.style_preset_id = self.active_style_profile.id

        # Workers
        self.active_job_worker: Optional[CloudJobWorker] = None
        self.active_regen_worker: Optional[SceneRegenerateWorker] = None
        self._cloud_worker: Optional[ConnectionCheckWorker] = None

        # Voice sample audio preview player (Qt Multimedia with FFmpeg 7.1.5 backend)
        self._voice_audio_output = QAudioOutput(self)
        self._voice_audio_output.setVolume(1.0)
        self._voice_player = QMediaPlayer(self)
        self._voice_player.setAudioOutput(self._voice_audio_output)

        # Build UI
        self._setup_menus()
        self._setup_top_cloud_bar()
        self._setup_main_tabs()
        self._setup_status_bar()

        # Populate presets
        self._populate_subject_combo()
        self._populate_voice_combo()
        self._populate_project_combo()

        # Apply dark theme
        self.setStyleSheet(DARK_THEME_QSS)

        # Initial check for connection in background
        QTimer.singleShot(400, self._check_cloud_connection)

        # Check for crash recovery / unfinished projects
        QTimer.singleShot(800, self._check_project_recovery)

    # =========================================================================
    # MENUS & TOP CLOUD BAR
    # =========================================================================

    def _setup_menus(self):
        menubar = self.menuBar()

        # File Menu
        file_menu = menubar.addMenu("&File")

        new_act = QAction("&New Project", self)
        new_act.setShortcut(QKeySequence.StandardKey.New)
        new_act.triggered.connect(self._on_new_project)
        file_menu.addAction(new_act)

        open_act = QAction("&Open Project...", self)
        open_act.setShortcut(QKeySequence.StandardKey.Open)
        open_act.triggered.connect(self._on_open_project)
        file_menu.addAction(open_act)

        history_act = QAction("📁 &Projects History & Workspace...", self)
        history_act.setShortcut(QKeySequence("Ctrl+H"))
        history_act.triggered.connect(self._on_open_project_history)
        file_menu.addAction(history_act)

        save_act = QAction("&Save Project", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._on_save_project)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save Project &As...", self)
        save_as_act.triggered.connect(self._on_save_project_as)
        file_menu.addAction(save_as_act)

        file_menu.addSeparator()

        export_act = QAction("&Export Prompts and JSON...", self)
        export_act.setShortcut(QKeySequence("Ctrl+E"))
        export_act.triggered.connect(self._on_export_dialog)
        file_menu.addAction(export_act)

        open_prompts_act = QAction("📄 &Open Prompts File (prompts_only.txt)...", self)
        open_prompts_act.setShortcut(QKeySequence("Ctrl+Shift+P"))
        open_prompts_act.triggered.connect(self._on_open_prompts_file)
        file_menu.addAction(open_prompts_act)

        file_menu.addSeparator()

        exit_act = QAction("E&xit", self)
        exit_act.setShortcut(QKeySequence.StandardKey.Quit)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        # Edit Menu
        edit_menu = menubar.addMenu("&Edit")
        split_act = QAction("&Split Selected Scene", self)
        split_act.setShortcut(QKeySequence("Ctrl+K"))
        split_act.triggered.connect(lambda: self.scene_editor._on_split_clicked())
        edit_menu.addAction(split_act)

        merge_act = QAction("&Merge With Next", self)
        merge_act.setShortcut(QKeySequence("Ctrl+M"))
        merge_act.triggered.connect(lambda: self.scene_editor._on_merge_clicked())
        edit_menu.addAction(merge_act)

        regen_act = QAction("&Regenerate Scene Prompt", self)
        regen_act.setShortcut(QKeySequence("Ctrl+R"))
        regen_act.triggered.connect(lambda: self.scene_editor._on_regenerate_clicked())
        edit_menu.addAction(regen_act)

        # Presets Menu
        presets_menu = menubar.addMenu("&Presets")
        preset_mgr_act = QAction("&Preset and Visual Style Manager...", self)
        preset_mgr_act.triggered.connect(self._on_open_preset_editor)
        presets_menu.addAction(preset_mgr_act)

        # Cloud Menu
        cloud_menu = menubar.addMenu("&AI Engine")
        test_conn_act = QAction("&Test Groq AI Connection", self)
        test_conn_act.triggered.connect(self._check_cloud_connection)
        cloud_menu.addAction(test_conn_act)

        settings_act = QAction("&Settings...", self)
        settings_act.setShortcut(QKeySequence.StandardKey.Preferences)
        settings_act.triggered.connect(self._on_open_settings)
        cloud_menu.addAction(settings_act)

    def _setup_top_cloud_bar(self):
        """Top banner showing project actions and engine status."""
        self.top_bar_widget = QFrame()
        self.top_bar_widget.setFixedHeight(56)
        self.top_bar_widget.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_GLASS_SURFACE};
                border-bottom: 1px solid {COLOR_BORDER_GREEN};
                padding: 4px 14px;
            }}
        """)
        tb_layout = QHBoxLayout(self.top_bar_widget)
        tb_layout.setContentsMargins(14, 6, 14, 6)
        tb_layout.setSpacing(12)

        # Hidden project selector for compatibility
        self.project_select_combo = PBComboBox()
        self.project_select_combo.setVisible(False)
        self.project_select_combo.addItem("Untitled_Project")
        self.project_select_combo.addItem("+ New Project...")
        self.project_select_combo.currentIndexChanged.connect(self._on_top_project_changed)

        # Active Project Indicator (Auto-named from audio script)
        self.active_project_lbl = QLabel("📁 Active Project: Auto-created on Audio Load")
        self.active_project_lbl.setStyleSheet(f"""
            color: {COLOR_BRAND_PRIMARY};
            font-size: 11px;
            font-weight: 700;
            background-color: rgba(163, 255, 18, 0.08);
            border: 1px solid rgba(163, 255, 18, 0.25);
            border-radius: 6px;
            padding: 4px 12px;
        """)
        tb_layout.addWidget(self.active_project_lbl)

        # Active Visual Style Indicator Badge (Read-only, launched via .bat)
        self.active_style_badge_lbl = QLabel(f"🎨 Style: {self.active_style_profile.name}")
        self.active_style_badge_lbl.setStyleSheet(f"""
            color: #38BDF8;
            font-size: 11px;
            font-weight: 700;
            background-color: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-radius: 6px;
            padding: 4px 12px;
        """)
        tb_layout.addWidget(self.active_style_badge_lbl)

        tb_layout.addStretch()

        # Engine / Server Status
        self.cloud_status_lbl = QLabel("● Ready / Idle")
        self.cloud_status_lbl.setStyleSheet(f"""
            color: {COLOR_STATUS_SUCCESS};
            font-size: 11px;
            font-weight: 700;
            background-color: rgba(16, 185, 129, 0.12);
            padding: 4px 12px;
            border-radius: 12px;
            border: 1px solid rgba(16, 185, 129, 0.25);
        """)
        tb_layout.addWidget(self.cloud_status_lbl)

        # Retain reference handles for tests and headless automation
        self.top_run_btn = QPushButton("▶ Run Pipeline")
        self.top_run_btn.clicked.connect(self._on_start_generation)
        self.top_stop_btn = QPushButton("■ Stop")
        self.top_stop_btn.clicked.connect(self._on_cancel_generation)


    # =========================================================================
    # SINGLE-PAGE STUDIO DASHBOARD (NO TABS)
    # =========================================================================

    def _setup_main_tabs(self):
        """Constructs the modern Single-Page Studio Dashboard (No horizontal scrolling, clean 2-row settings)."""
        central_container = QWidget()
        cc_layout = QVBoxLayout(central_container)
        cc_layout.setContentsMargins(0, 0, 0, 0)
        cc_layout.setSpacing(0)

        # 1. Top Bar
        cc_layout.addWidget(self.top_bar_widget)

        # 2. Main Scroll Area for complete single-page dashboard
        self.main_scroll = QScrollArea()
        self.main_scroll.setWidgetResizable(True)
        self.main_scroll.setFrameShape(QFrame.Shape.NoFrame)
        # CRITICAL: Prevent horizontal scroll completely
        self.main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.main_scroll.setStyleSheet("QScrollArea { background-color: #0e121a; border: none; }")
        main_scroll = self.main_scroll

        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setContentsMargins(16, 12, 16, 18)
        scroll_layout.setSpacing(12)

        # =========================================================================
        # UNIFIED PRODUCTION STUDIO DASHBOARD (Single Professional Container)
        # =========================================================================
        sec1_group = QFrame()
        sec1_group.setObjectName("ProductionCard")
        sec1_group.setStyleSheet(f"""
            QFrame#ProductionCard {{
                background-color: {COLOR_GLASS_SURFACE};
                border: 1px solid {COLOR_BORDER_GREEN};
                border-radius: 12px;
                padding: 16px;
            }}
        """)
        sec1_layout = QVBoxLayout(sec1_group)
        sec1_layout.setContentsMargins(14, 14, 14, 14)
        sec1_layout.setSpacing(12)

        # Header bar with mode switch buttons & Dynamic Project Name
        s1_hdr = QHBoxLayout()
        s1_hdr.setSpacing(8)

        self.btn_mode_script = QPushButton("📄 Text Script")
        # Header bar
        s1_title = QLabel("🎙️ Step 1: Voiceover Audio & Word Timestamps")
        s1_title.setStyleSheet(f"font-size: 13px; font-weight: 700; color: {COLOR_TEXT_PRIMARY};")
        s1_hdr.addWidget(s1_title)
        s1_hdr.addStretch()

        # Hidden project name input & script text edit for full backward compatibility
        from datetime import datetime
        init_name = f"Project_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.proj_name_input = QLineEdit(init_name)
        self.proj_name_input.setVisible(False)
        self.proj_name_input.textChanged.connect(self._on_project_name_changed)

        self.story_text_edit = QPlainTextEdit()
        self.story_text_edit.setVisible(False)
        self.story_text_edit.textChanged.connect(self._on_story_text_edited)

        self.btn_mode_script = QPushButton()
        self.btn_mode_script.setVisible(False)
        self.btn_mode_audio = QPushButton()
        self.btn_mode_audio.setVisible(False)
        self.browse_story_btn = QPushButton()
        self.browse_story_btn.setVisible(False)
        self.history_proj_btn = QPushButton()
        self.history_proj_btn.setVisible(False)
        self.story_stats_lbl = QLabel()
        self.story_stats_lbl.setVisible(False)
        self.script_mode_container = QWidget()
        self.script_mode_container.setVisible(False)

        sec1_layout.addLayout(s1_hdr)

        # Narration-Aware Studio Workflow Panel (Steps 0 to 5)
        self.workflow_panel = NarrationWorkflowPanel(self, active_style_id=self.active_style_profile.id)
        self.workflow_panel.pipeline_succeeded.connect(self._on_pipeline_succeeded)
        self.workflow_panel.navigate_to_images.connect(lambda: self.sidebar.set_current_page("images"))
        sec1_layout.addWidget(self.workflow_panel)

        # Audio Ingestion Panel (Headless compatibility instance)
        self.audio_panel = AudioTranscriptionPanel()
        self.audio_panel.story_text_provider = lambda: self.story_text_edit.toPlainText().strip()
        self.audio_panel.audio_ready.connect(self._on_audio_ready)
        self.audio_panel.setVisible(False)
        sec1_layout.addWidget(self.audio_panel)

        # Hidden Word timestamps indicator for compatibility
        self.ts_row_container = QWidget()
        self.ts_row_container.setVisible(False)
        self.ts_path_lbl = QLabel("Word Timestamps: Auto-generated on Voiceover")
        self.browse_ts_btn = QPushButton("Import Timestamps JSON...")
        self.browse_ts_btn.clicked.connect(self._on_browse_timestamps)

        # Hidden compatibility handles for Voice & Self-TTS
        self.voice_row_container = QWidget()
        self.voice_row_container.setVisible(False)
        self.voice_combo = PBComboBox()
        self.voice_combo.addItem("Default Audio Narration", "audio")
        self.preview_voice_btn = QPushButton("▶ Audition Voice")
        self.preview_voice_btn.clicked.connect(self._on_preview_voice)
        self.voice_speed_spin = QDoubleSpinBox()
        self.voice_speed_spin.setValue(1.0)
        self.balance_guard_badge = QLabel("🛡️ Audio First Engine")
        self.chk_modal_tts = QCheckBox()
        self.chk_modal_tts.setChecked(False)

        # =========================================================================
        # UNIFIED SETTINGS BAR (CLEAN, PROFESSIONAL & COMPACT)
        # =========================================================================
        img_settings_box = QWidget()
        img_settings_box.setStyleSheet("background: transparent; border: none;")
        is_layout = QVBoxLayout(img_settings_box)
        is_layout.setContentsMargins(0, 4, 0, 0)
        is_layout.setSpacing(10)

        # Visual Art Style (Locked via .bat launcher, read-only display)
        self.style_matrix = StyleThumbnailMatrix(self)
        self.style_matrix.setVisible(False)

        style_read_only_row = QHBoxLayout()
        style_read_only_row.setSpacing(10)
        self.lbl_main_active_style = QLabel(f"🎨 Active Style: {self.active_style_profile.name}")
        self.lbl_main_active_style.setStyleSheet("""
            color: #38BDF8;
            font-size: 12px;
            font-weight: 700;
            background-color: rgba(56, 189, 248, 0.10);
            border: 1px solid rgba(56, 189, 248, 0.35);
            border-left: 3px solid #10B981;
            border-radius: 6px;
            padding: 6px 14px;
        """)
        style_read_only_row.addWidget(self.lbl_main_active_style)
        style_read_only_row.addStretch()
        is_layout.addLayout(style_read_only_row)

        # Row 2: Unified Format & Profile Settings Row
        params_row = QHBoxLayout()
        params_row.setSpacing(10)

        ar_lbl = QLabel("📐 Aspect Ratio:")
        ar_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 600;")
        params_row.addWidget(ar_lbl)

        self.top_aspect_combo = PBComboBox()
        self.top_aspect_combo.setFixedHeight(28)
        self.top_aspect_combo.setFixedWidth(160)
        self.top_aspect_combo.addItem("16:9 (Landscape)", "16:9")
        self.top_aspect_combo.addItem("9:16 (Portrait / Shorts)", "9:16")
        self.top_aspect_combo.addItem("1:1 (Square)", "1:1")
        if getattr(self, "active_style_profile", None) and self.active_style_profile.output_format == "9:16":
            self.top_aspect_combo.setCurrentIndex(1)
        else:
            self.top_aspect_combo.setCurrentIndex(0)
        self.top_aspect_combo.currentIndexChanged.connect(self._on_top_aspect_changed)
        params_row.addWidget(self.top_aspect_combo)

        prof_lbl = QLabel("👤 Profile:")
        prof_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; font-weight: 600; margin-left: 6px;")
        params_row.addWidget(prof_lbl)

        self.browser_profile_combo = PBComboBox()
        self.browser_profile_combo.setFixedHeight(28)
        self.browser_profile_combo.setFixedWidth(140)
        for i in range(1, 7):
            self.browser_profile_combo.addItem(f"acc{i}" + (" (Primary)" if i == 1 else ""), f"acc{i}")
        params_row.addWidget(self.browser_profile_combo)

        self.open_login_btn = QPushButton("Open & Login Profile")
        self.open_login_btn.setFixedHeight(28)
        self.open_login_btn.setCursor(Qt.PointingHandCursor)
        self.open_login_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_GLASS_SURFACE_2};
                color: {COLOR_BRAND_EMERALD};
                font-size: 11px;
                font-weight: 600;
                border: 1px solid rgba(16, 185, 129, 0.35);
                border-radius: 6px;
                padding: 0 12px;
            }}
            QPushButton:hover {{
                background-color: rgba(16, 185, 129, 0.15);
            }}
        """)
        self.open_login_btn.clicked.connect(self._on_quick_open_profile)
        params_row.addWidget(self.open_login_btn)

        self.profile_status_lbl = QLabel("• Google / Gemini Account Signed In (acc1)")
        self.profile_status_lbl.setStyleSheet(f"color: {COLOR_BRAND_EMERALD}; font-size: 10px; font-weight: 600;")
        params_row.addWidget(self.profile_status_lbl)

        params_row.addStretch()

        self.pacing_badge = QLabel("⚡ AI Pacing (Semantic Cuts)")
        self.pacing_badge.setStyleSheet(f"""
            color: {COLOR_BRAND_PRIMARY};
            font-weight: 700;
            font-size: 10px;
            background-color: rgba(163, 255, 18, 0.12);
            padding: 4px 10px;
            border-radius: 6px;
            border: 1px solid rgba(163, 255, 18, 0.25);
        """)
        params_row.addWidget(self.pacing_badge)

        is_layout.addLayout(params_row)

        # Compatibility hidden instances for hook clips and subject
        self.hook_clips_spin = QSpinBox()
        self.hook_clips_spin.setValue(0)
        self.hook_clips_spin.setVisible(False)
        self.subject_combo = PBComboBox()
        self.subject_combo.setVisible(False)
        self.subject_combo.currentIndexChanged.connect(self._on_subject_changed)


        # Headless compatibility instances
        self.max_dur_spin = QDoubleSpinBox()
        self.max_dur_spin.setValue(120.0)
        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setValue(0.5)
        self.pref_dur_spin = QDoubleSpinBox()
        self.pref_dur_spin.setValue(4.5)
        self.count_auto_radio = QRadioButton()
        self.count_auto_radio.setChecked(True)
        self.count_exact_radio = QRadioButton()
        self.count_group = QButtonGroup()
        self.count_group.addButton(self.count_auto_radio)
        self.count_group.addButton(self.count_exact_radio)
        self.target_count_spin = QSpinBox()
        self.target_count_spin.setValue(30)

        # Hidden timing mode combo helper
        self.timing_combo = PBComboBox()
        self.timing_combo.addItems(["AI Decides (Semantic Boundaries)", "Target Duration", "Fixed Duration Intervals"])
        self.timing_combo.setVisible(False)
        is_layout.addWidget(self.timing_combo)

        self.custom_subject_edit = QPlainTextEdit()
        self.custom_subject_edit.setMaximumHeight(35)
        self.custom_subject_edit.setVisible(False)
        is_layout.addWidget(self.custom_subject_edit)

        self.custom_style_edit = QPlainTextEdit()
        self.custom_style_edit.setMaximumHeight(35)
        self.custom_style_edit.setVisible(False)
        is_layout.addWidget(self.custom_style_edit)

        # Master Generation Action Row (50/50 Next & Stop CTAs)
        gen_action_row = QHBoxLayout()
        gen_action_row.setSpacing(12)

        self.start_gen_btn = QPushButton("➔ Next: Proceed to Studio (Review & Expand Prompts)")
        self.start_gen_btn.setFixedHeight(46)
        self.start_gen_btn.setCursor(Qt.PointingHandCursor)
        self.start_gen_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:0.5 #34D399, stop:1 #A3FF12);
                color: #080D0B;
                font-weight: 800;
                font-size: 13px;
                letter-spacing: 0.5px;
                border-radius: 8px;
                border: none;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34D399, stop:0.5 #A3FF12, stop:1 #B8FF24);
            }}
            QPushButton:disabled {{
                background-color: {COLOR_GLASS_SURFACE_2};
                color: {COLOR_STATUS_DISABLED};
                border: 1px solid {COLOR_BORDER_SUBTLE};
            }}
        """)
        self.start_gen_btn.clicked.connect(self._on_start_generation)
        gen_action_row.addWidget(self.start_gen_btn, 1)

        self.btn_stop_generation = QPushButton("■ Stop")
        self.btn_stop_generation.setFixedHeight(46)
        self.btn_stop_generation.setCursor(Qt.PointingHandCursor)
        self.btn_stop_generation.setStyleSheet(f"""
            QPushButton {{
                background: rgba(255, 100, 124, 0.16);
                color: {COLOR_STATUS_ERROR};
                font-weight: 800;
                font-size: 13px;
                letter-spacing: 0.5px;
                border-radius: 8px;
                border: 1px solid rgba(255, 100, 124, 0.45);
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background: rgba(255, 100, 124, 0.32);
                border-color: {COLOR_STATUS_ERROR};
                color: #FFFFFF;
            }}
            QPushButton:disabled {{
                background-color: {COLOR_GLASS_SURFACE_2};
                color: {COLOR_STATUS_DISABLED};
                border: 1px solid {COLOR_BORDER_SUBTLE};
            }}
        """)
        self.btn_stop_generation.clicked.connect(self._on_cancel_generation)
        gen_action_row.addWidget(self.btn_stop_generation, 1)

        is_layout.addLayout(gen_action_row)

        self.progress_panel = ProgressPanelWidget()
        self.progress_panel.cancel_requested.connect(self._on_cancel_generation)
        is_layout.addWidget(self.progress_panel)

        # AI PLANNING APPROVAL BANNER (Appears after Stage 1 & 2 before Prompts Generation)
        self.plan_approval_banner = QFrame()
        self.plan_approval_banner.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_GLASS_SURFACE};
                border: 1px solid {COLOR_BORDER_GREEN};
                border-radius: 8px;
                padding: 10px 14px;
                margin-top: 6px;
            }}
        """)
        pab_lay = QHBoxLayout(self.plan_approval_banner)
        pab_lay.setContentsMargins(10, 8, 10, 8)
        pab_lay.setSpacing(12)

        self.plan_summary_lbl = QLabel("AI Planned 0 Visual Scenes...")
        self.plan_summary_lbl.setStyleSheet(f"color: {COLOR_TEXT_PRIMARY}; font-size: 12px; font-weight: 600;")
        pab_lay.addWidget(self.plan_summary_lbl, 1)

        self.approve_plan_btn = QPushButton("✓ Approve Plan and Expand Prompts")
        self.approve_plan_btn.setFixedHeight(34)
        self.approve_plan_btn.setCursor(Qt.PointingHandCursor)
        self.approve_plan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BRAND_PRIMARY};
                color: #080D0B;
                font-weight: 800;
                font-size: 12px;
                border-radius: 6px;
                border: none;
                padding: 0 16px;
            }}
            QPushButton:hover {{
                background-color: #B8FF24;
            }}
        """)
        self.approve_plan_btn.clicked.connect(self._on_approve_plan)
        pab_lay.addWidget(self.approve_plan_btn)

        self.replan_btn = QPushButton("↺ Re-Plan")
        self.replan_btn.setFixedHeight(34)
        self.replan_btn.setCursor(Qt.PointingHandCursor)
        self.replan_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_GLASS_SURFACE_2};
                color: {COLOR_TEXT_SECONDARY};
                font-weight: 600;
                font-size: 11px;
                border-radius: 6px;
                border: 1px solid {COLOR_BORDER_SUBTLE};
                padding: 0 12px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_HOVER_SURFACE};
                color: {COLOR_TEXT_PRIMARY};
                border-color: {COLOR_BORDER_GREEN};
            }}
        """)
        self.replan_btn.clicked.connect(self._on_replan)
        pab_lay.addWidget(self.replan_btn)

        self.plan_approval_banner.setVisible(False)
        # Placed prominently right above the Prompt Grid Matrix in s4_layout

        img_settings_box.setVisible(False)
        sec1_layout.addWidget(img_settings_box)
        scroll_layout.addWidget(sec1_group)

        # =========================================================================
        # SECTION 2: COLLAPSIBLE STORY ANALYSIS & STORY BIBLE
        # =========================================================================
        sec2_box = QFrame()
        sec2_box.setStyleSheet("background-color: #131824; border: 1px solid #202a3e; border-radius: 8px;")
        s2_layout = QVBoxLayout(sec2_box)
        s2_layout.setContentsMargins(10, 6, 10, 8)
        s2_layout.setSpacing(6)

        s2_hdr = QHBoxLayout()
        s2_title = QLabel("📖 <b>STORY ANALYSIS & STORY BIBLE</b>")
        s2_title.setStyleSheet("color: #c084fc; font-size: 11px;")
        self.bible_badge = QLabel("READY FOR STORY ANALYSIS")
        self.bible_badge.setStyleSheet("background-color: #1e1b4b; color: #a5b4fc; font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 3px; border: 1px solid #4338ca;")
        s2_hdr.addWidget(s2_title)
        s2_hdr.addWidget(self.bible_badge)
        s2_hdr.addStretch()

        self.toggle_bible_btn = QPushButton("Fold ▲")
        self.toggle_bible_btn.setFixedHeight(22)
        self.toggle_bible_btn.setStyleSheet("background-color: #1a2234; color: #cbd5e1; font-size: 10px; border: 1px solid #28354d; border-radius: 3px; padding: 0 8px;")
        s2_hdr.addWidget(self.toggle_bible_btn)
        s2_layout.addLayout(s2_hdr)

        self.bible_cards_container = QWidget()
        self.bible_cards_layout = QHBoxLayout(self.bible_cards_container)
        self.bible_cards_layout.setContentsMargins(0, 2, 0, 0)
        self.bible_cards_layout.setSpacing(8)

        s2_layout.addWidget(self.bible_cards_container)
        self.toggle_bible_btn.clicked.connect(lambda: self._toggle_container(self.bible_cards_container, self.toggle_bible_btn))
        # Add to scroll layout and hide so C++ keeps lifecycle without taking screen space
        scroll_layout.addWidget(sec2_box)
        sec2_box.setVisible(False)

        # Initialize Story Bible UI with current project analysis or prompt guidance
        self._update_story_bible_ui(self.project.state.analysis)

        # =========================================================================
        # 02 — STUDIO MATRIX (UNIFIED PROMPT GRID CONTAINER)
        # =========================================================================
        sec4_box = QFrame()
        sec4_box.setObjectName("StudioContainer")
        sec4_box.setStyleSheet(f"""
            QFrame#StudioContainer {{
                background-color: {COLOR_BG_MAIN};
                border: none;
            }}
        """)
        s4_layout = QVBoxLayout(sec4_box)
        s4_layout.setContentsMargins(16, 12, 16, 14)
        s4_layout.setSpacing(10)

        # Prominent AI Planning Approval Banner right inside the Matrix Section
        s4_layout.addWidget(self.plan_approval_banner)

        # Strict Prompt Card Grid Matrix
        self.prompt_grid = PromptGridWidget()
        self.prompt_grid.scene_updated.connect(self._on_scene_edited)
        self.prompt_grid.delete_requested.connect(self._on_delete_scene)
        self.prompt_grid.add_scene_requested.connect(self._on_add_manual_scene)
        self.prompt_grid.export_requested.connect(self._on_export_dialog)
        self.prompt_grid.open_file_requested.connect(self._on_open_prompts_file)
        self.prompt_grid.approve_requested.connect(self._on_approve_plan)
        self.prompt_grid.retry_scene_requested.connect(self._on_retry_single_scene)
        s4_layout.addWidget(self.prompt_grid, 1)

        # Step Navigation Button from Studio -> Images (Step 3 Image Generation Hub)
        self.btn_goto_images = QPushButton("➔ Next Step: Generate Images & Sync (Go to Images)")
        self.btn_goto_images.setFixedHeight(38)
        self.btn_goto_images.setCursor(Qt.PointingHandCursor)
        self.btn_goto_images.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_GLASS_SURFACE};
                color: {COLOR_BRAND_PRIMARY};
                font-size: 13px;
                font-weight: 700;
                border: 1px solid {COLOR_BORDER_GREEN};
                border-radius: 8px;
                padding: 0 16px;
                margin-top: 6px;
            }}
            QPushButton:hover {{
                background-color: rgba(163, 255, 18, 0.15);
                border-color: {COLOR_BRAND_PRIMARY};
            }}
        """)
        self.btn_goto_images.clicked.connect(lambda: self.sidebar.set_current_page("images"))
        s4_layout.addWidget(self.btn_goto_images)

        self.studio_page = sec4_box

        # Headless compatibility instances
        self.scenes_list = QListWidget()
        self.scene_editor = SceneEditorWidget()
        self.scene_search_input = QLineEdit()
        self.scene_count_badge = QLabel()

        # Step Navigation Button from Production -> Studio
        self.btn_goto_studio = QPushButton("➔ Review Storyboard & Prompts (Go to Studio)")
        self.btn_goto_studio.setFixedHeight(44)
        self.btn_goto_studio.setCursor(Qt.PointingHandCursor)
        self.btn_goto_studio.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(16, 185, 129, 0.28), stop:1 rgba(163, 255, 18, 0.22));
                color: #A3FF12;
                font-size: 13px;
                font-weight: 700;
                border: 1px solid rgba(163, 255, 18, 0.55);
                border-radius: 8px;
                padding: 0 16px;
                margin-top: 10px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(16, 185, 129, 0.45), stop:1 rgba(163, 255, 18, 0.40));
                border-color: #A3FF12;
                color: #FFFFFF;
            }}
        """)
        self.btn_goto_studio.clicked.connect(lambda: self.sidebar.set_current_page("studio"))
        scroll_layout.addWidget(self.btn_goto_studio)

        # 03 — Images Panel (Step 5 Manual Image Checkpoint & Step 6 Render)
        self.images_panel = ImagesPanel(self)
        self.images_panel.render_completed.connect(self._on_render_completed)

        # RenderPanel (Deliverables) and ImageAssetPanel for headless compatibility
        self.tab3_images = ImageAssetPanel()
        self.tab3_images.images_updated.connect(self._on_images_updated)
        self.tab3_images.audio_selected.connect(self._on_asset_audio_selected)
        self.tab3_images.narration_selected.connect(self._on_asset_narration_selected)
        self.tab3_images.scenes_realigned.connect(self._on_asset_scenes_realigned)

        # 04 — RenderPanel (Deliverables)
        self.tab4_render = RenderPanel()
        self.tab4_render.render_completed.connect(self._on_render_completed)

        # 05 — ProfilesManagerPanel (Chrome browser profiles acc1..acc6)
        self.profiles_panel = ProfilesManagerPanel(self)

        # 06 — SYSTEM: Console
        self.console_page = self._create_console_page()

        main_scroll.setWidget(scroll_widget)

        # Body: Sidebar Navigation on Left + Stacked Pages on Right
        body_container = QWidget()
        body_layout = QHBoxLayout(body_container)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self.sidebar = SidebarNav(self)
        self.sidebar.page_changed.connect(self._on_sidebar_nav_changed)
        body_layout.addWidget(self.sidebar)

        self.stacked_pages = QStackedWidget(self)

        # 01 — Production (Main Studio Dashboard)
        self.stacked_pages.addWidget(main_scroll)

        # 02 — Studio (Prompt Matrix & Scene Editor)
        self.stacked_pages.addWidget(self.studio_page)

        # 03 — Images (Gemini Image Generation Hub & Scene Auto-Pacing)
        self.stacked_pages.addWidget(self.images_panel)

        # 04 — Deliverables (Video Motion & Export)
        self.stacked_pages.addWidget(self.tab4_render)

        # 05 — Profiles (Chrome Browser Manager)
        self.stacked_pages.addWidget(self.profiles_panel)

        # SYSTEM: Console
        self.stacked_pages.addWidget(self.console_page)

        body_layout.addWidget(self.stacked_pages, 1)
        cc_layout.addWidget(body_container, 1)
        self.setCentralWidget(central_container)

        # Default active page: Production
        self.stacked_pages.setCurrentWidget(main_scroll)

        # Tab navigation helper with backward compatibility for legacy tab calls
        self.tabs = SinglePageNavHelper(
            main_scroll,
            {
                0: sec1_group,
                1: self.studio_page,
                2: self.images_panel,
                3: self.tab4_render,
                4: self.profiles_panel,
            },
            stacked_pages=self.stacked_pages,
            sidebar=self.sidebar,
        )

    def _create_queue_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(24, 20, 24, 20)
        lbl = QLabel("📋 Active Pipeline Queue & Background Tasks")
        lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #f8fafc;")
        l.addWidget(lbl)
        sub = QLabel("Queued storyboard generation, image rendering, and speech synthesis jobs.")
        sub.setStyleSheet("font-size: 12px; color: #64748b;")
        l.addWidget(sub)

        box = QFrame()
        box.setStyleSheet("background-color: #0b111a; border: 1px solid #1a2536; border-radius: 10px; padding: 20px;")
        bl = QVBoxLayout(box)
        self.queue_status_lbl = QLabel("Queue is currently idle. Press 'Run Pipeline' in Production to enqueue a job.")
        self.queue_status_lbl.setStyleSheet("color: #94a3b8; font-size: 13px;")
        bl.addWidget(self.queue_status_lbl)
        l.addWidget(box)
        l.addStretch()
        return w

    def _create_voiceover_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(24, 20, 24, 20)
        lbl = QLabel("🎙️ Voiceover Studio & Audio Narration")
        lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #f8fafc;")
        l.addWidget(lbl)
        sub = QLabel("Synthesize studio voiceovers using Kokoro Local Engine or transcribe existing audio files via Whisper.")
        sub.setStyleSheet("font-size: 12px; color: #64748b;")
        l.addWidget(sub)

        # Host Audio Transcription Panel inside Voiceover page
        if hasattr(self, "audio_panel"):
            l.addWidget(self.audio_panel)
        l.addStretch()
        return w

    def _create_timeline_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(24, 20, 24, 20)
        lbl = QLabel("⏱️ Interactive Visual Storyboard Timeline")
        lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #f8fafc;")
        l.addWidget(lbl)
        sub = QLabel("View and fine-tune scene cut points aligned with word-level speech timestamps.")
        sub.setStyleSheet("font-size: 12px; color: #64748b;")
        l.addWidget(sub)

        from app.ui.widgets.timeline_view import TimelineCanvas
        self.timeline_view = TimelineCanvas()
        l.addWidget(self.timeline_view, 1)
        return w

    def _create_console_page(self) -> QWidget:
        w = QWidget()
        l = QVBoxLayout(w)
        l.setContentsMargins(24, 20, 24, 20)
        l.setSpacing(10)
        lbl = QLabel("💻 Live System & Pipeline Console")
        lbl.setStyleSheet("font-size: 18px; font-weight: 700; color: #f8fafc;")
        l.addWidget(lbl)

        self.console_text = QPlainTextEdit()
        self.console_text.setReadOnly(True)
        self.console_text.setStyleSheet("""
            QPlainTextEdit {
                background-color: #05080f;
                color: #34d399;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #1a2536;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        self.console_text.appendPlainText("[SYSTEM] PromptBeast Desktop Studio initialized.")
        self.console_text.appendPlainText("[SYSTEM] Chrome Browser Profiles Manager loaded (acc1..acc6).")
        self.console_text.appendPlainText("[SYSTEM] Visual Art Style Matrix ready (26 styles loaded).")
        l.addWidget(self.console_text, 1)
        return w

    def _prepare_images_panel(self):
        """Prepares images_panel project directory, audio, and scenes."""
        if hasattr(self, "project") and self.project:
            if not self.project.project_dir:
                p_name = (
                    self.proj_name_input.text().strip()
                    if hasattr(self, "proj_name_input") and self.proj_name_input.text().strip()
                    else self.project.name
                )
                if not p_name or p_name == "Untitled_Project":
                    raw_s = self.story_text_edit.toPlainText().strip() if hasattr(self, "story_text_edit") else ""
                    p_name = self.generate_project_name_from_script(raw_s) if raw_s else f"Project_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    if hasattr(self, "proj_name_input"):
                        self.proj_name_input.setText(p_name)
                from app.config.settings import resolve_project_dir
                canonical_p = resolve_project_dir(self.settings.workspace_directory, p_name)
                canonical_p.mkdir(parents=True, exist_ok=True)
                self.project.project_dir = str(canonical_p.resolve())

            p_dir = Path(self.project.project_dir)
            scenes_data = []
            if hasattr(self, "workflow_pipeline_data") and self.workflow_pipeline_data:
                scenes_data = self.workflow_pipeline_data.get("prompts") or self.workflow_pipeline_data.get("scenes") or []
            elif self.project.state.scenes:
                scenes_data = [
                    {
                        "scene_id": f"{s.index:03d}",
                        "start": s.start_time,
                        "end": s.end_time,
                        "duration": s.duration,
                        "narration_words": s.narration_text,
                        "text": s.narration_text,
                        "prompt": getattr(s, "image_prompt", "")
                    }
                    for s in self.project.state.scenes
                ]

            audio_p = Path(self.project.state.audio_file_path) if self.project.state.audio_file_path else None
            audio_dur = self.project.state.narration.duration if self.project.state.narration else 0.0

            t_words = []
            s_words = []
            if hasattr(self, "workflow_pipeline_data") and self.workflow_pipeline_data:
                inp = self.workflow_pipeline_data.get("inputs")
                if inp:
                    t_words = [w.to_dict() for w in inp.words]
                    s_words = inp.script_words

            self.images_panel.set_project_context(
                project_dir=p_dir,
                scenes_data=scenes_data,
                audio_path=audio_p,
                audio_duration=audio_dur,
                timestamp_words=t_words,
                script_words=s_words
            )

    def _on_pipeline_succeeded(self, result: dict):
        """Called when workflow pipeline completes Steps 0 to 5."""
        self.workflow_pipeline_data = result
        proj_dir_str = result.get("project_dir")
        if proj_dir_str:
            self.project.project_dir = proj_dir_str
            self.project.name = result.get("project_name", Path(proj_dir_str).name)
            if hasattr(self, "proj_name_input"):
                self.proj_name_input.setText(self.project.name)
            if hasattr(self, "active_project_lbl"):
                self.active_project_lbl.setText(f"📁 {self.project.name}")
            from app.config.settings import SettingsManager
            SettingsManager.add_recent_project(proj_dir_str)

        inputs = result.get("inputs")
        if inputs:
            self.project.state.audio_file_path = str(inputs.audio_path)
            wt_list = [WordTimestamp(word=w.word, start=w.start, end=w.end) for w in inputs.words]
            self.project.state.narration = NarrationData(words=wt_list, duration=inputs.audio_duration)
        scenes = result.get("scenes", [])
        self.project.state.scenes = [
            VisualScene(
                index=int(s.get("scene_id", idx + 1)),
                start_time=s.get("start", 0.0),
                end_time=s.get("end", 0.0),
                duration=s.get("duration", 0.0),
                narration_text=s.get("text", ""),
                narration_words=s.get("words", [])
            )
            for idx, s in enumerate(scenes)
        ]
        self._prepare_images_panel()

    def _load_project_from_dir(self, dir_path: str):
        """Loads a project directly from disk directory path."""
        try:
            proj = ProjectManager.load_project(dir_path)
            self._apply_project_to_ui(proj, auto_switch_tab=True)
            self.sidebar.set_current_page("production")
            QMessageBox.information(
                self,
                "Project Loaded",
                f"Loaded project '{proj.name}' successfully!\n"
                f"• Story text: {len(proj.state.story_raw_text or '')} chars\n"
                f"• Scenes: {len(proj.state.scenes)}\n"
                f"• Voiceover: {'Ready' if proj.state.audio_file_path else 'Not generated yet'}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", f"Could not load project: {str(e)}")

    def _on_sidebar_nav_changed(self, key: str):
        k = key.lower()
        if hasattr(self, "sidebar") and k in self.sidebar._buttons:
            self.sidebar._buttons[k].setChecked(True)
        if k in ("production", "01"):
            self.stacked_pages.setCurrentWidget(self.main_scroll)
        elif k in ("studio", "02", "scene_studio"):
            self.stacked_pages.setCurrentWidget(self.studio_page)
            if hasattr(self, "project") and self.project and self.project.state.scenes:
                self.prompt_grid.load_scenes(self.project.state.scenes)
                if hasattr(self, "plan_approval_banner"):
                    num = len(self.project.state.scenes)
                    self.plan_summary_lbl.setText(f"📋 {num} visual scenes planned. Review your storyboard below, then click Approve to expand prompts via Groq.")
                    self.approve_plan_btn.setText("✓ Approve Plan and Expand Prompts")
                    self.approve_plan_btn.setEnabled(True)
                    self.plan_approval_banner.setVisible(True)
        elif k in ("images", "generate_images", "03", "image_assets"):
            self.stacked_pages.setCurrentWidget(self.images_panel)
            self._prepare_images_panel()
        elif k in ("deliverables", "04", "final_render"):
            self.stacked_pages.setCurrentWidget(self.tab4_render)
            if hasattr(self, "project") and self.project:
                self.tab4_render.load_project_data(
                    self.project.state.scenes,
                    self.project.state.audio_file_path,
                    self.project.state.narration
                )
        elif k in ("profiles", "05", "profiles_accounts"):
            self.stacked_pages.setCurrentWidget(self.profiles_panel)
            self.profiles_panel.refresh_profiles()
        elif k == "console":
            self.stacked_pages.setCurrentWidget(self.console_page)
        elif k == "settings":
            self._on_open_settings()
            # Restore active page highlight in sidebar
            curr = self.stacked_pages.currentWidget()
            if curr == self.main_scroll:
                self.sidebar.set_current_page("production")
            elif curr == self.studio_page:
                self.sidebar.set_current_page("studio")
            elif curr == self.images_panel:
                self.sidebar.set_current_page("images")
            elif curr == self.tab4_render:
                self.sidebar.set_current_page("deliverables")
            elif curr == self.profiles_panel:
                self.sidebar.set_current_page("profiles")
            elif curr == self.console_page:
                self.sidebar.set_current_page("console")

    def _on_scene_sync_completed(self, scenes, audio_path, narration):
        if hasattr(self, "project") and self.project:
            self.project.state.scenes = scenes
            if audio_path:
                self.project.state.audio_file_path = audio_path
            if narration:
                self.project.state.narration = narration
        if hasattr(self, "tab4_render"):
            self.tab4_render.load_project_data(scenes, audio_path, narration)

    def _on_matrix_style_selected(self, style_id: str):
        from app.core.style_manager import StyleManager
        self.active_style_profile = StyleManager.set_active_style(style_id)
        if hasattr(self, "lbl_main_active_style"):
            self.lbl_main_active_style.setText(f"🎨 Active Style: {self.active_style_profile.name}")
        if hasattr(self, "project") and self.project:
            self.project.config.visual_style_id = style_id
            if hasattr(self.project, "state") and self.project.state:
                self.project.state.style_preset_id = style_id
                th_h = getattr(self.project.state, "thumbnail_headline", "") or ""
                if th_h:
                    from app.core.thumbnail_prompt_builder import build_viral_thumbnail_prompt
                    new_th = build_viral_thumbnail_prompt(headline=th_h, style_id=style_id)
                    self.project.state.thumbnail_prompt = new_th
                    if hasattr(self, "tab4_render"):
                        self.tab4_render.set_thumbnail_prompt(new_th)

    def _on_top_aspect_changed(self, idx: int):
        val = self.top_aspect_combo.currentData() or "16:9"
        if hasattr(self, "tab4_render") and hasattr(self.tab4_render, "settings"):
            self.tab4_render.settings.aspect_ratio = val
            if hasattr(self.tab4_render, "_update_summary"):
                self.tab4_render._update_summary()

    def _on_quick_open_profile(self):
        prof_name = self.browser_profile_combo.currentData() or "acc1"
        download_dir = None
        prompts_payload = None
        if hasattr(self, "project") and self.project and self.project.project_dir:
            download_dir = Path(self.project.project_dir) / "images"
            download_dir.mkdir(parents=True, exist_ok=True)
            if self.project.state.scenes:
                th_p = getattr(self.project.state, "thumbnail_prompt", "") or ""
                th_h = getattr(self.project.state, "thumbnail_headline", "") or ""
                if (not th_p or th_h not in th_p) and th_h:
                    from app.core.thumbnail_prompt_builder import build_viral_thumbnail_prompt
                    sid = getattr(self.project.state, "style_preset_id", "3d_red_blue_stickman") or "3d_red_blue_stickman"
                    th_p = build_viral_thumbnail_prompt(headline=th_h, style_id=sid)
                    self.project.state.thumbnail_prompt = th_p
                prompts_payload = StoryboardExporter.export_to_prompts_only(
                    self.project.state.scenes,
                    thumbnail_prompt=th_p
                )

        ok, msg = BrowserProfilesManager.open_browser_profile(
            prof_name,
            download_dir=download_dir,
            prompts_text=prompts_payload
        )
        if ok:
            scene_cnt = len(self.project.state.scenes) if (hasattr(self, "project") and self.project and self.project.state.scenes) else 0
            extra_msg = f" ({scene_cnt} prompts copied to clipboard!)" if scene_cnt > 0 else ""
            self.statusBar().showMessage(f"🚀 {msg}{extra_msg}", 6000)
        else:
            QMessageBox.warning(self, "Profile Launch Error", msg)

    def _on_top_project_changed(self, idx: int):
        val = self.project_select_combo.currentText()
        if val == "+ New Project...":
            self._on_new_project()
        else:
            dir_path = self.project_select_combo.itemData(idx)
            if dir_path and Path(dir_path).exists():
                try:
                    proj = ProjectManager.load_project(str(dir_path))
                    self._apply_project_to_ui(proj, auto_switch_tab=False)
                except Exception as e:
                    print(f"[TopProjectCombo] Could not load '{dir_path}': {e}")
            elif val and val != "Untitled_Project":
                if hasattr(self, "proj_name_input"):
                    self.proj_name_input.setText(val)

    def _on_delete_active_project(self):
        name = self.proj_name_input.text().strip()
        reply = QMessageBox.question(
            self,
            "Delete Project",
            f"Are you sure you want to delete project '{name}'?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.statusBar().showMessage(f"Project '{name}' cleared.", 3000)

    def _set_source_mode(self, mode: str):
        active_qss = f"""
            background-color: {COLOR_BRAND_PRIMARY};
            color: #080D0B;
            font-weight: 700;
            border-radius: 6px;
            border: none;
            padding: 0 14px;
        """
        inactive_qss = f"""
            background-color: {COLOR_GLASS_SURFACE_2};
            color: {COLOR_TEXT_SECONDARY};
            font-weight: 500;
            border-radius: 6px;
            border: 1px solid {COLOR_BORDER_SUBTLE};
            padding: 0 14px;
        """
        if mode == "script":
            self.btn_mode_script.setStyleSheet(active_qss)
            self.btn_mode_audio.setStyleSheet(inactive_qss)
            self.script_mode_container.setVisible(True)
            self.audio_panel.setVisible(False)
            if hasattr(self, "voice_row_container"):
                self.voice_row_container.setVisible(True)
            if hasattr(self, "ts_row_container"):
                self.ts_row_container.setVisible(True)
        else:
            self.btn_mode_audio.setStyleSheet(active_qss)
            self.btn_mode_script.setStyleSheet(inactive_qss)
            self.script_mode_container.setVisible(False)
            self.audio_panel.setVisible(True)
            if hasattr(self, "voice_row_container"):
                self.voice_row_container.setVisible(False)
            if hasattr(self, "ts_row_container"):
                self.ts_row_container.setVisible(False)

    def _toggle_container(self, widget: QWidget, btn: QPushButton):
        if widget.isVisible():
            widget.setVisible(False)
            btn.setText("Unfold ▼")
        else:
            widget.setVisible(True)
            btn.setText("Fold ▲")

    def _setup_status_bar(self):
        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self.status_banner = StatusBannerWidget()
        self.status_banner.setVisible(False)

    # =========================================================================
    # EVENT HANDLERS & SYNC
    # =========================================================================

    @staticmethod
    def generate_project_name_from_script(script_text: str) -> str:
        """Extracts first 3-4 clean words from script + YYYYMMDD_HHMM timestamp."""
        import re
        from datetime import datetime
        clean = re.sub(r'[^\w\s]', '', script_text)
        words = [w for w in clean.split() if w.strip()]
        if words:
            base = "_".join(words[:4])
        else:
            base = "Project"
        base = base[:35].strip("_")
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{base}_{ts}"

    def _on_story_text_edited(self):
        text = self.story_text_edit.toPlainText()
        self.project.state.story_raw_text = text
        if hasattr(self, "audio_panel"):
            self.audio_panel.set_story_text(text)
        w_count = len(text.split())
        c_count = len(text)
        self.story_stats_lbl.setText(f"{w_count} words | {c_count} chars")

        # Automatically update project name from script if user hasn't explicitly customized it
        if text.strip() and not getattr(self, "_user_custom_proj_name", False):
            dynamic_name = self.generate_project_name_from_script(text)
            self._auto_naming = True
            if hasattr(self, "proj_name_input"):
                self.proj_name_input.setText(dynamic_name)
            self.project.name = dynamic_name
            self._auto_naming = False

    def _on_audio_ready(self, audio_path: str, narration: Optional[NarrationData] = None, story_text: str = ""):
        if audio_path:
            self.project.state.audio_file_path = audio_path
        if narration:
            self.project.state.narration = narration
            self.ts_path_lbl.setText(f"✅ Narration Audio ({narration.duration_seconds:.1f}s, {len(narration.words)} words)")
            self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        else:
            self.project.state.narration = None
            self.ts_path_lbl.setText("⚠️ Audio file selected — Timestamps missing. Run Whisper or Import JSON.")
            self.ts_path_lbl.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")

        # Extract text source from transcript
        text_source = story_text.strip()
        if not text_source and narration and narration.words:
            text_source = " ".join(w.word for w in narration.words).strip()

        if text_source:
            self.story_text_edit.setPlainText(text_source)
            self.project.state.story_raw_text = text_source

            # Auto-create project folder with the first few words of the script + timestamp
            auto_name = self.generate_project_name_from_script(text_source)
            self.project.name = auto_name
            self.proj_name_input.setText(auto_name)

            from app.config.settings import resolve_project_dir
            proj_dir = resolve_project_dir(self.settings.workspace_directory, auto_name)
            self.project.project_dir = str(proj_dir.resolve())

            # Copy narration audio into projects/<auto_name>/audio/
            if audio_path and Path(audio_path).exists():
                aud_dir = proj_dir / "audio"
                aud_dir.mkdir(parents=True, exist_ok=True)
                dest_audio = aud_dir / Path(audio_path).name
                if str(dest_audio.resolve()) != str(Path(audio_path).resolve()):
                    try:
                        import shutil
                        shutil.copy2(audio_path, dest_audio)
                        self.project.state.audio_file_path = str(dest_audio)
                    except Exception:
                        pass

            if hasattr(self, "active_project_lbl"):
                self.active_project_lbl.setText(f"📁 {auto_name}")

        elif not self.project.project_dir:
            proj_name = self.proj_name_input.text().strip() or "Untitled_Project"
            proj = ProjectManager.create_project(proj_name, self.settings.workspace_directory, add_unique_id=True)
            self.project.project_dir = proj.project_dir
            self.project.name = proj.name
            self.proj_name_input.setText(proj.name)

        try:
            self._sync_project_config()
            ProjectManager.save_project(self.project)
            SettingsManager.add_recent_project(self.project.project_dir)
        except Exception as e:
            print(f"Notice saving project on audio ready: {e}")

        self.tab4_render.load_project_data(
            scenes=self.project.state.scenes,
            audio_path=self.project.state.audio_file_path,
            narration=self.project.state.narration
        )
        if narration:
            self.statusBar().showMessage(f"Narration loaded ({narration.duration_seconds:.1f}s). Ready for storyboard generation.", 5000)
        else:
            self.statusBar().showMessage("Audio loaded without timestamps. Import timestamps or run Whisper before planning.", 5000)

    def _on_workflow_story_imported(self, story_text: str, narration: Optional[NarrationData]):
        if hasattr(self, "story_text_edit"):
            self.story_text_edit.setPlainText(story_text)
        if self.project:
            self.project.state.story_raw_text = story_text
            self.project.state.narration = narration
            if hasattr(self.workflow_panel, "json_path") and self.workflow_panel.json_path:
                self.project.state.story_file_path = self.workflow_panel.json_path
            if hasattr(self.workflow_panel, "audio_path") and self.workflow_panel.audio_path:
                self.project.state.audio_file_path = self.workflow_panel.audio_path
            self.active_project_lbl.setText(f"📁 {self.project.name} (Script Loaded)")

    def _on_workflow_scenes_aligned(self, scenes: List[VisualScene]):
        if self.project:
            self.project.state.scenes = scenes
            if hasattr(self.workflow_panel, "audio_path") and self.workflow_panel.audio_path:
                self.project.state.audio_file_path = self.workflow_panel.audio_path
            if hasattr(self.workflow_panel, "narration") and self.workflow_panel.narration:
                self.project.state.narration = self.workflow_panel.narration
        self.prompt_grid.load_scenes(scenes)
        self.active_project_lbl.setText(f"📁 {self.project.name} ({len(scenes)} visual scenes synchronized)")
        self.statusBar().showMessage(f"Synchronized {len(scenes)} visual scenes with audio timestamps.", 4000)

    def _on_workflow_bibles_updated(self, bibles: dict):
        if self.project:
            self.project.state.extracted_bibles = bibles
        self.statusBar().showMessage("Story bibles updated.", 3000)

    def _on_workflow_prompts_generated(self, scenes: List[VisualScene]):
        if self.project:
            self.project.state.scenes = scenes
            if hasattr(self.workflow_panel, "audio_path") and self.workflow_panel.audio_path:
                self.project.state.audio_file_path = self.workflow_panel.audio_path
            if hasattr(self.workflow_panel, "narration") and self.workflow_panel.narration:
                self.project.state.narration = self.workflow_panel.narration
            self.project.state.last_generation_status = "COMPLETED"
        self.prompt_grid.load_scenes(scenes)
        if hasattr(self, "images_panel"):
            if hasattr(self.workflow_panel, "audio_path") and self.workflow_panel.audio_path:
                self.images_panel.set_audio_path(self.workflow_panel.audio_path)
            self.images_panel.scenes = scenes
            self.images_panel._populate_table()
            self.images_panel._refresh_generate_btn_label()
        if hasattr(self, "tab4_render"):
            self.tab4_render.load_project_data(
                scenes=scenes,
                audio_path=getattr(self.workflow_panel, "audio_path", None),
                narration=getattr(self.workflow_panel, "narration", None)
            )
        try:
            self._on_save_project()
        except Exception:
            pass
        self.statusBar().showMessage(f"Generated {len(scenes)} image prompts successfully! Ready for Studio review.", 5000)

    def _on_voiceover_generated(self, audio_path: str, narration: Optional[NarrationData] = None):
        self.project.state.story_raw_text = self.story_text_edit.toPlainText().strip()
        self.project.state.audio_file_path = audio_path
        self.project.state.narration = narration
        if narration:
            self.ts_path_lbl.setText(f"✅ Voiceover Narration ({narration.duration_seconds:.1f}s, {len(narration.words)} words)")
            self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        else:
            self.ts_path_lbl.setText("⚠️ Voiceover audio ready without word timestamps.")
            self.ts_path_lbl.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")

        # Save project to disk immediately
        if self.project.project_dir:
            try:
                self._sync_project_config()
                ProjectManager.save_project(self.project)
            except Exception as e:
                print(f"Notice auto-saving project: {e}")

        # Update render panel
        self.tab4_render.load_project_data(
            scenes=self.project.state.scenes,
            audio_path=audio_path,
            narration=narration
        )

        # Automatically advance to Tab 2: Prompts + Storyboard
        self.tabs.setCurrentIndex(1)
        self.statusBar().showMessage(f"Voiceover generated & saved to {self.project.name}. Ready for prompts generation.", 5000)

    def _on_images_updated(self, updated_scenes: List[VisualScene]):
        self.project.state.scenes = updated_scenes
        self.tab4_render.load_project_data(
            scenes=updated_scenes,
            audio_path=self.project.state.audio_file_path,
            narration=self.project.state.narration
        )

    def _on_asset_audio_selected(self, audio_path: str):
        self.project.state.audio_file_path = audio_path
        self.tab4_render.set_audio_file(audio_path)
        scenes = self.project.state.scenes or (self.tab3_images.scenes if hasattr(self, "tab3_images") else [])
        if scenes:
            self.tab4_render.load_project_data(
                scenes=scenes,
                audio_path=audio_path,
                narration=self.project.state.narration
            )
        self.ts_path_lbl.setText(f"✅ Voiceover Audio: {Path(audio_path).name}")
        self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")

    def _on_asset_narration_selected(self, narration: NarrationData):
        self.project.state.narration = narration
        scenes = self.project.state.scenes or (self.tab3_images.scenes if hasattr(self, "tab3_images") else [])
        self.tab4_render.load_project_data(
            scenes=scenes,
            audio_path=self.project.state.audio_file_path,
            narration=narration
        )
        self.ts_path_lbl.setText(f"✅ Timings ({narration.duration_seconds:.1f}s, {len(narration.words)} words)")
        self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")

    def _on_asset_scenes_realigned(self, scenes: List[VisualScene]):
        self.project.state.scenes = scenes
        self._refresh_scenes_ui(scenes)
        self.tab4_render.load_project_data(
            scenes=scenes,
            audio_path=self.project.state.audio_file_path,
            narration=self.project.state.narration
        )

    def _on_render_completed(self, output_video_path: str):
        self.project.state.final_video_path = output_video_path

    # =========================================================================
    # PRESETS & SETTINGS
    # =========================================================================

    def _populate_project_combo(self):
        """Dynamically populate top project combo from actual projects on disk."""
        self.project_select_combo.blockSignals(True)
        self.project_select_combo.clear()
        try:
            projs = ProjectManager.scan_projects(self.settings.workspace_directory)
            for p in projs:
                label = p["name"]
                if p.get("status_text"):
                    label = f"{p['name']} ({p['status_text']})"
                self.project_select_combo.addItem(label, p["dir_path"])
        except Exception as e:
            print(f"[TopProjectCombo] scan notice: {e}")

        # If current project has a dir, select it
        if hasattr(self, "project") and self.project:
            for i in range(self.project_select_combo.count()):
                if self.project_select_combo.itemData(i) == self.project.project_dir:
                    self.project_select_combo.setCurrentIndex(i)
                    break
        self.project_select_combo.addItem("+ New Project...", None)
        self.project_select_combo.blockSignals(False)

    def _populate_subject_combo(self):
        self.subject_combo.clear()
        subjects = PresetManager.get_all_subjects()
        for s in subjects:
            self.subject_combo.addItem(s.name, s.id)
        idx = self.subject_combo.findData(self.settings.default_subject_id)
        if idx >= 0:
            self.subject_combo.setCurrentIndex(idx)

    def _populate_voice_combo(self):
        """Audio-first workflow: Self-TTS is disabled."""
        self.voice_combo.clear()
        self.voice_combo.addItem("🎙️ External Audio File", "audio")

    def _on_preview_voice(self):
        pass

    def _on_chk_modal_tts_toggled(self, checked: bool):
        if not hasattr(self, "balance_guard_badge"):
            return
        self.balance_guard_badge.setText("🛡️ Kokoro Local Engine (100% Offline)")
        self.balance_guard_badge.setStyleSheet("color: #34D399; font-weight: 600; font-size: 10px; background-color: rgba(16, 185, 129, 0.15); padding: 4px 8px; border-radius: 6px; border: 1px solid #10B981;")

    def _on_open_style_preview(self):
        current_id = self.active_style_profile.id if hasattr(self, "active_style_profile") else "century_old_photograph"
        dlg = StylePreviewDialog(current_id, self)
        if dlg.exec():
            selected = dlg.selected_style_id
            if selected:
                from app.core.style_manager import StyleManager
                self.active_style_profile = StyleManager.set_active_style(selected)
                if hasattr(self, "lbl_main_active_style"):
                    self.lbl_main_active_style.setText(f"🎨 Active Style: {self.active_style_profile.name}")

    def _on_subject_changed(self, index: int):
        subject_id = self.subject_combo.currentData()
        self.custom_subject_edit.setVisible(subject_id == "custom")
        self.project.config.subject_id = subject_id or "history"

    def _on_style_changed(self, index: int = 0, style_id: Optional[str] = None):
        target_id = style_id or (self.active_style_profile.id if hasattr(self, "active_style_profile") else "century_old_photograph")
        self.custom_style_edit.setVisible(target_id == "custom")
        self.project.config.visual_style_id = target_id
        if hasattr(self.project, "state") and self.project.state:
            try:
                self.project.state.style_preset_id = target_id
            except Exception:
                pass
            th_h = getattr(self.project.state, "thumbnail_headline", "") or ""
            if th_h:
                from app.core.thumbnail_prompt_builder import build_viral_thumbnail_prompt
                new_th = build_viral_thumbnail_prompt(headline=th_h, style_id=target_id)
                self.project.state.thumbnail_prompt = new_th
                if hasattr(self, "tab4_render"):
                    self.tab4_render.set_thumbnail_prompt(new_th)

        preset = PresetManager.get_style_by_id(target_id)
        if preset and hasattr(self, "style_preview_name"):
            self.style_preview_name.setText(preset.name)
            self.style_preview_desc.setText(preset.description)

    def _on_count_mode_toggled(self, checked: bool):
        is_exact = self.count_exact_radio.isChecked()
        self.target_count_spin.setEnabled(is_exact)
        self.project.config.image_count_mode = ImageCountMode.EXACT if is_exact else ImageCountMode.AUTOMATIC

    # =========================================================================
    # PROJECT LIFECYCLE
    # =========================================================================

    def _on_project_name_changed(self, text: str):
        if not getattr(self, "_auto_naming", False):
            self._user_custom_proj_name = True
        self.project.name = text.strip() or "Project"

    def _on_new_project(self):
        from datetime import datetime
        self._user_custom_proj_name = False
        default_name = f"Project_{datetime.now().strftime('%Y%m%d_%H%M')}"
        self.project = StoryProject(
            name=default_name,
            config=ProjectConfig(),
            state=ProjectState()
        )
        self.proj_name_input.setText(default_name)
        self.story_text_edit.setPlainText("")
        self.story_stats_lbl.setText("0 words | 0 characters")
        self.ts_path_lbl.setText("Word Timestamps: Auto-generated on Voiceover")
        self._refresh_scenes_ui([])
        self.tab3_images.load_scenes([])
        self.tab4_render.load_project_data([])
        self.scene_editor.load_scene(None)
        QMessageBox.information(self, "New Project", "Created new blank project.")

    def _apply_project_to_ui(self, proj: StoryProject, auto_switch_tab: bool = True):
        """Applies a loaded StoryProject completely to all tabs and UI components."""
        self.project = proj
        # Sync top project combo to reflect this project
        self._populate_project_combo()
        self.proj_name_input.setText(proj.name)

        # 1. Populate Story & TTS in Tab 1
        raw_story = proj.state.story_raw_text or ""
        self.story_text_edit.setPlainText(raw_story)
        if hasattr(self, "audio_panel"):
            self.audio_panel.set_story_text(raw_story)
        words = len(raw_story.split()) if raw_story else 0
        chars = len(raw_story)
        self.story_stats_lbl.setText(f"{words} words | {chars} characters")

        # 2. Populate Voiceover Narration & Timestamps
        if proj.state.audio_file_path and Path(proj.state.audio_file_path).exists() and proj.state.narration:
            if hasattr(self, "audio_panel"):
                self.audio_panel.load_audio_narration(proj.state.audio_file_path, proj.state.narration)
            self.ts_path_lbl.setText(f"✅ Voiceover Narration ({proj.state.narration.duration_seconds:.1f}s, {len(proj.state.narration.words)} words)")
            self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        elif proj.state.narration:
            self.ts_path_lbl.setText(f"✅ Timestamps loaded ({proj.state.narration.duration_seconds:.1f}s, {len(proj.state.narration.words)} words)")
            self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
        else:
            self.ts_path_lbl.setText("Word Timestamps: Auto-generated on Voiceover")
            self.ts_path_lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")

        # 3. Populate Preset & Config settings
        s_idx = self.subject_combo.findData(proj.config.subject_id)
        if s_idx >= 0:
            self.subject_combo.setCurrentIndex(s_idx)
        if proj.config.custom_subject:
            self.custom_subject_edit.setPlainText(proj.config.custom_subject)

        if proj.config.visual_style_id:
            from app.core.style_manager import StyleManager
            self.active_style_profile = StyleManager.set_active_style(proj.config.visual_style_id)
            if hasattr(self, "lbl_main_active_style"):
                self.lbl_main_active_style.setText(f"🎨 Active Style: {self.active_style_profile.name}")
        if proj.config.custom_style:
            self.custom_style_edit.setPlainText(proj.config.custom_style)

        if proj.config.image_count_mode == ImageCountMode.EXACT:
            self.count_exact_radio.setChecked(True)
            self.target_count_spin.setValue(proj.config.target_image_count)
        else:
            self.count_auto_radio.setChecked(True)

        if proj.config.timing_mode == TimingMode.TARGET:
            self.timing_combo.setCurrentIndex(1)
        elif proj.config.timing_mode == TimingMode.FIXED:
            self.timing_combo.setCurrentIndex(2)
        else:
            self.timing_combo.setCurrentIndex(0)

        self.min_dur_spin.setValue(proj.config.min_duration)
        self.pref_dur_spin.setValue(proj.config.preferred_duration)
        self.max_dur_spin.setValue(proj.config.max_duration)

        # 4. Populate Scenes in Tab 2, Tab 3, Tab 4 and Scene Sync
        self._refresh_scenes_ui(proj.state.scenes)
        self.tab3_images.load_scenes(proj.state.scenes, proj.state.images_folder_path)
        self.tab4_render.load_project_data(
            scenes=proj.state.scenes,
            audio_path=proj.state.audio_file_path,
            narration=proj.state.narration
        )
        if hasattr(self, "images_panel"):
            if proj.project_dir:
                self.images_panel.set_project(Path(proj.project_dir))
            self.images_panel.set_audio_path(proj.state.audio_file_path or "")
            if proj.state.narration:
                self.images_panel.narration = proj.state.narration
            if proj.state.scenes:
                self.images_panel.scenes = proj.state.scenes
                self.images_panel._populate_table()
                self.images_panel._refresh_generate_btn_label()
                self.images_panel._scan_images()

        # 5. Auto-switch to the most appropriate tab
        if auto_switch_tab:
            if proj.state.scenes and len(proj.state.scenes) > 0:
                self.sidebar.set_current_page("studio")
            elif proj.state.audio_file_path and proj.state.narration:
                self.sidebar.set_current_page("studio")
            else:
                self.sidebar.set_current_page("production")

        if proj.project_dir:
            SettingsManager.add_recent_project(proj.project_dir)

    def _on_open_project_history(self):
        """Opens the Project History & Workspace browser dialog."""
        dlg = ProjectHistoryDialog(self.settings.workspace_directory, self)
        if dlg.exec() and dlg.selected_project_dir:
            try:
                proj = ProjectManager.load_project(dlg.selected_project_dir)
                self._apply_project_to_ui(proj, auto_switch_tab=True)
                QMessageBox.information(
                    self,
                    "Project Loaded",
                    f"Loaded project '{proj.name}' successfully!\n"
                    f"• Story text: {len(proj.state.story_raw_text or '')} chars\n"
                    f"• Scenes: {len(proj.state.scenes)}\n"
                    f"• Voiceover: {'Ready' if proj.state.audio_file_path else 'Not generated yet'}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Open Failed", f"Could not load project: {str(e)}")

    def _on_open_project(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Open Project Directory", self.settings.workspace_directory)
        if not dir_path:
            return

        try:
            proj = ProjectManager.load_project(dir_path)
            self._apply_project_to_ui(proj, auto_switch_tab=True)
            QMessageBox.information(self, "Project Loaded", f"Loaded project '{proj.name}' ({len(proj.state.scenes)} scenes).")
        except Exception as e:
            QMessageBox.critical(self, "Open Failed", f"Could not load project: {str(e)}")

    def _on_voiceover_generated(self, audio_path: str, narration: Optional[NarrationData] = None):
        """Called immediately when voiceover synthesis finishes."""
        self.project.state.audio_file_path = audio_path
        self.project.state.narration = narration
        self.project.state.story_raw_text = self.story_text_edit.toPlainText().strip()

        # Update labels
        if narration:
            self.ts_path_lbl.setText(f"✅ Voiceover Narration ({narration.duration_seconds:.1f}s, {len(narration.words)} words)")
            self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        else:
            self.ts_path_lbl.setText("⚠️ Voiceover audio ready without word timestamps.")
            self.ts_path_lbl.setStyleSheet("color: #f59e0b; font-size: 11px; font-weight: bold;")

        # Automatically create and save project directory
        if not self.project.project_dir:
            proj_name = self.proj_name_input.text().strip() or "Untitled_Project"
            proj = ProjectManager.create_project(proj_name, self.settings.workspace_directory, add_unique_id=True)
            self.project.project_dir = proj.project_dir
            self.project.name = proj.name
            self.proj_name_input.setText(proj.name)

        try:
            self._sync_project_config()
            ProjectManager.save_project(self.project)
            SettingsManager.add_recent_project(self.project.project_dir)
            self._populate_project_combo()  # Refresh combo after project dir created
        except Exception as e:
            print(f"Notice saving project on voiceover generation: {e}")

        # Auto-advance directly to Tab 2 for Storyboard generation
        self.tabs.setCurrentIndex(1)

    def _on_save_project(self):
        if not self.project.project_dir:
            self._on_save_project_as()
            return

        try:
            self._sync_project_config()
            ProjectManager.save_project(self.project)
            QMessageBox.information(self, "Saved", f"Project '{self.project.name}' saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"Could not save project: {str(e)}")

    def _on_save_project_as(self):
        parent_dir = QFileDialog.getExistingDirectory(self, "Select Parent Directory for Project", self.settings.workspace_directory)
        if not parent_dir:
            return

        name = self.proj_name_input.text().strip() or "Untitled_Project"
        proj_dir = Path(parent_dir) / name
        self.project.name = name
        self.project.project_dir = str(proj_dir.resolve())

        self._on_save_project()

    def _sync_project_config(self):
        self.project.state.story_raw_text = self.story_text_edit.toPlainText().strip()
        self.project.config.subject_id = self.subject_combo.currentData() or "history"
        self.project.config.custom_subject = self.custom_subject_edit.toPlainText().strip() or None
        self.project.config.visual_style_id = self.active_style_profile.id if hasattr(self, "active_style_profile") else "century_old_photograph"
        self.project.config.custom_style = self.custom_style_edit.toPlainText().strip() or None
        self.project.config.image_count_mode = ImageCountMode.EXACT if self.count_exact_radio.isChecked() else ImageCountMode.AUTOMATIC
        self.project.config.target_image_count = self.target_count_spin.value()

        t_idx = self.timing_combo.currentIndex()
        if t_idx == 1:
            self.project.config.timing_mode = TimingMode.TARGET
        elif t_idx == 2:
            self.project.config.timing_mode = TimingMode.FIXED
        else:
            self.project.config.timing_mode = TimingMode.AI_DECIDES

        self.project.config.min_duration = self.min_dur_spin.value()
        self.project.config.preferred_duration = self.pref_dur_spin.value()
        self.project.config.max_duration = self.max_dur_spin.value()

    # =========================================================================
    # FILE LOADERS
    # =========================================================================

    def _on_browse_story(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Story Narration File",
            "",
            "Text Files (*.txt *.md *.text *.markdown);;All Files (*.*)"
        )
        if not file_path:
            return

        try:
            raw_text, meta = StoryLoader.load_from_file(file_path)
            self.project.state.story_raw_text = raw_text
            self.project.state.story_file_path = file_path
            self.story_text_edit.setPlainText(raw_text)
            self.story_stats_lbl.setText(f"{meta['word_count']} words | {meta['character_count']} chars (Est: {meta['estimated_spoken_duration_seconds']}s)")
        except Exception as e:
            QMessageBox.critical(self, "Story Load Error", str(e))

    def _on_browse_timestamps(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Word-Level Timestamp JSON File",
            "",
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not file_path:
            return

        try:
            narration_data = TimestampParser.load_from_json(file_path)
            self.project.state.narration = narration_data
            self.project.state.timestamps_file_path = file_path
            self.ts_path_lbl.setText(f"✅ Imported: {Path(file_path).name} ({narration_data.duration_seconds:.1f}s)")
            self.ts_path_lbl.setStyleSheet("color: #4ade80; font-size: 11px;")
        except Exception as e:
            QMessageBox.critical(self, "Timestamp Load Error", str(e))

    # =========================================================================
    # AI GENERATION PIPELINE
    # =========================================================================

    def _on_start_generation(self):
        story_text = self.story_text_edit.toPlainText().strip()
        if not story_text:
            cur_aud = getattr(self.audio_panel, "_current_audio_path", None)
            if not cur_aud or not Path(cur_aud).exists():
                QMessageBox.warning(self, "Missing Audio File", "Please select a voiceover audio file (.wav, .mp3) to begin.")
                return
            # Audio is loaded but not yet transcribed: trigger transcription first
            self.statusBar().showMessage("Transcribing audio narration with Whisper...", 5000)
            self.audio_panel._on_start_transcription()
            return

        self._sync_project_config()

        selected_voice = self.voice_combo.currentData() or "audio"
        selected_speed = self.voice_speed_spin.value()
        should_synthesize_voice = False

        if not should_synthesize_voice:
            if not self.project.state.narration:
                estimated = TimestampParser.estimate_from_text(story_text)
                self.project.state.narration = estimated

        active_style = getattr(self, "active_style_profile", None)
        active_style_id = active_style.id if active_style else (getattr(self.project.state, "style_preset_id", None) or self.project.config.visual_style_id or "default")
        self.project.config.visual_style_id = active_style_id
        self.project.config.visual_style_id = active_style
        if hasattr(self.project, "state") and self.project.state:
            self.project.state.style_preset_id = active_style

        req = JobCreateRequest(
            project_name=self.project.name,
            story_text=story_text,
            narration=self.project.state.narration,
            subject_id=self.project.config.subject_id,
            custom_subject=self.project.config.custom_subject,
            visual_style_id=active_style,
            custom_style=self.project.config.custom_style,
            image_count_mode=self.project.config.image_count_mode,
            target_image_count=self.project.config.target_image_count,
            timing_mode=TimingMode.AI_DECIDES,
            min_duration=self.project.config.min_duration,
            preferred_duration=self.project.config.preferred_duration,
            max_duration=self.project.config.max_duration,
            batch_size=self.settings.default_batch_size,
            model_name=self.settings.default_model
        )

        self.start_gen_btn.setEnabled(False)
        self.plan_approval_banner.setVisible(False)
        self.progress_panel.set_running_state(True)

        # Resolve project directory before proceeding (never Untitled_Project)
        if not self.project.project_dir or self.project.name in ("Untitled_Project", ""):
            proj_name = (
                self.proj_name_input.text().strip()
                if hasattr(self, "proj_name_input") and self.proj_name_input.text().strip()
                else self.project.name
            )
            if not proj_name or proj_name == "Untitled_Project":
                proj_name = self.generate_project_name_from_script(story_text)
            from app.config.settings import resolve_project_dir
            proj_dir = resolve_project_dir(self.settings.workspace_directory, proj_name)
            self.project.project_dir = str(proj_dir.resolve())
            self.project.name = proj_name
            if hasattr(self, "proj_name_input"):
                self.proj_name_input.setText(proj_name)
            if hasattr(self, "active_project_lbl"):
                self.active_project_lbl.setText(f"📁 {proj_name}")

        # Synchronize audio & narration from workflow_panel if available
        if hasattr(self, "workflow_panel"):
            if getattr(self.workflow_panel, "audio_path", None):
                self.project.state.audio_file_path = self.workflow_panel.audio_path
            if getattr(self.workflow_panel, "narration", None):
                self.project.state.narration = self.workflow_panel.narration

        # User provides storyboard scenes: take from workflow_panel.scenes or parse text
        planned_scenes = []
        if hasattr(self, "workflow_panel") and self.workflow_panel.scenes:
            planned_scenes = self.workflow_panel.scenes
        elif hasattr(self, "workflow_panel") and self.workflow_panel.txt_visual_ideas.toPlainText().strip():
            self.workflow_panel._on_align_visuals()
            planned_scenes = self.workflow_panel.scenes
        elif hasattr(self.project, "state") and self.project.state.scenes:
            planned_scenes = self.project.state.scenes

        if not planned_scenes:
            QMessageBox.warning(
                self,
                "No Storyboard Scenes",
                "Please provide or load storyboard scenes in Step 2 before proceeding.\n"
                "(Storyboards are provided by you, not generated by AI)."
            )
            self.start_gen_btn.setEnabled(True)
            return

        # User provides production bibles: take from workflow_panel.bibles or load default
        bibles = getattr(self.workflow_panel, "bibles", None) or getattr(self.project.state, "extracted_bibles", None)
        if not bibles and hasattr(self.workflow_panel, "loaded_bibles") and self.workflow_panel.loaded_bibles:
            first_key = list(self.workflow_panel.loaded_bibles.keys())[0]
            bibles = self.workflow_panel.loaded_bibles[first_key]
            self.workflow_panel._apply_active_bible(first_key)

        if not bibles:
            from app.core.bible_manager import BibleManager
            sample_b = Path("audio and files/Bibles.txt")
            if sample_b.exists():
                loaded_map = BibleManager.load_bibles_from_path(sample_b)
                if loaded_map:
                    bibles = list(loaded_map.values())[0]

        if not bibles:
            bibles = {
                "story_bible": {"logline": story_text[:500] if story_text else "Production Story"},
                "characters": [{"name": "Protagonist", "physical_description": "Survivor", "clothing_and_gear": "Standard clothing"}],
                "environments": [{"name": "Setting", "architectural_style": "Story environment"}],
                "objects_and_props": [],
                "visual_style_bible": {"color_palette": f"Consistent with {active_style}"}
            }

        self.project.state.extracted_bibles = bibles

        from app.models.planning import StoryIntelligence
        story_intel = StoryIntelligence(
            story_summary=bibles.get("story_bible", {}).get("logline", "") or (story_text[:500] if story_text else ""),
            characters=bibles.get("characters", []),
            environments=bibles.get("environments", []),
            objects_and_props=bibles.get("key_objects", []),
            visual_style_bible=bibles.get("visual_style_bible", {})
        )
        analysis = story_intel.to_story_analysis()
        self.project.state.analysis = analysis

        # Immediately populate Studio with user storyboard and bibles (No Groq generation)
        self._on_parallel_pipeline_ready(
            analysis,
            planned_scenes,
            self.project.state.audio_file_path,
            self.project.state.narration
        )

        # Navigate to Studio
        if hasattr(self, "sidebar"):
            self.sidebar.set_current_page("studio")

    def _on_parallel_pipeline_ready(self, analysis, planned_scenes, audio_path, narration):
        if audio_path:
            self.project.state.audio_file_path = audio_path
        if narration:
            self.project.state.narration = narration
            self.ts_path_lbl.setText(f"✅ Voiceover Audio ({narration.duration_seconds:.1f}s, {len(narration.words)} words)")
            self.ts_path_lbl.setStyleSheet("color: #B8F24A; font-size: 11px; font-weight: bold;")

        # FIRST: update UI with planned scenes so Studio and story bible are ALWAYS populated
        self._on_job_planning_ready(analysis, planned_scenes)

        # Ensure production bibles are saved to disk
        if self.project.project_dir and analysis:
            try:
                from backend.pipeline.stage1_story_intelligence import save_story_bibles
                save_story_bibles(Path(self.project.project_dir), analysis)
            except Exception as e:
                print(f"[MainWindow] save_story_bibles notice: {e}")

        # Update deliverables tab
        try:
            self.tab4_render.load_project_data(
                scenes=planned_scenes,
                audio_path=self.project.state.audio_file_path,
                narration=self.project.state.narration
            )
        except Exception as e:
            print(f"[MainWindow] tab4_render load notice: {e}")

        # Update images panel & scene sync panel with audio & project folder
        try:
            if hasattr(self, "images_panel"):
                if self.project.project_dir:
                    self.images_panel.set_project(Path(self.project.project_dir))
                if audio_path:
                    self.images_panel.set_audio_path(audio_path)
                if narration:
                    self.images_panel.narration = narration
                if planned_scenes:
                    self.images_panel.scenes = planned_scenes
                    self.images_panel._populate_table()
                    self.images_panel._refresh_generate_btn_label()
        except Exception as e:
            print(f"[MainWindow] images_panel notice: {e}")

    def _on_job_planning_ready(self, analysis, planned_scenes):
        self.start_gen_btn.setEnabled(True)
        self.start_gen_btn.setText("➔ Next: Proceed to Studio (Review & Expand Prompts)")
        self.progress_panel.set_running_state(False)

        self.project.state.analysis = analysis
        self.project.state.scenes = planned_scenes
        if analysis:
            th_head = getattr(analysis, "thumbnail_headline", "") or ""
            sid = getattr(self.project.config, "visual_style_id", "3d_red_blue_stickman") or "3d_red_blue_stickman"
            from app.core.thumbnail_prompt_builder import build_viral_thumbnail_prompt
            th_prompt = getattr(analysis, "thumbnail_prompt", "") or ""
            is_stick = "stickman" in sid.lower() or "3d_red_blue" in sid.lower()
            prompt_is_stick = "stickman" in th_prompt.lower() if th_prompt else False
            if not th_prompt or (is_stick != prompt_is_stick) or (th_head and th_head not in th_prompt):
                th_prompt = build_viral_thumbnail_prompt(headline=th_head or "CONSTANTINOPLE FALLS", style_id=sid)
            self.project.state.thumbnail_headline = th_head
            self.project.state.thumbnail_prompt = th_prompt
            if hasattr(self, "tab4_render"):
                self.tab4_render.set_thumbnail_headline(th_head)
                self.tab4_render.set_thumbnail_prompt(th_prompt)
        self._update_story_bible_ui(analysis)
        self._refresh_scenes_ui(planned_scenes)

        total = len(planned_scenes)
        durs = [s.duration for s in planned_scenes] if planned_scenes else [4.0]
        min_d = min(durs)
        max_d = max(durs)
        avg_d = sum(durs) / total if total > 0 else 4.0

        mod_counts = {}
        for s in planned_scenes:
            m = getattr(s, "visual_modality", "cinematic_scene") or "cinematic_scene"
            mod_counts[m] = mod_counts.get(m, 0) + 1

        cine = mod_counts.get("cinematic_scene", 0)
        chart = mod_counts.get("infographic_chart", 0)
        blue = mod_counts.get("blueprint_schematic", 0)
        doc = mod_counts.get("historical_document", 0)
        other = total - (cine + chart + blue + doc)

        parts = []
        if cine > 0: parts.append(f"{cine} Cinematic")
        if chart > 0: parts.append(f"{chart} Charts")
        if blue > 0: parts.append(f"{blue} Blueprints")
        if doc > 0: parts.append(f"{doc} Documents")
        if other > 0: parts.append(f"{other} Graphics")
        mix_str = ", ".join(parts) if parts else f"{total} Scenes"

        audio_info = ""
        if self.project.state.audio_file_path and Path(self.project.state.audio_file_path).exists():
            fname = Path(self.project.state.audio_file_path).name
            dur = self.project.state.narration.duration_seconds if self.project.state.narration else 0.0
            audio_info = f" &nbsp;|&nbsp; <span style='color: #B8F24A;'>Audio: {fname} ({dur:.1f}s)</span>"

        self.plan_summary_lbl.setText(
            f"<b>{total} Storyboard Scenes Synchronized</b> &nbsp;|&nbsp; "
            f"<span style='color: #38BDF8;'>Pacing: {min_d:.1f}s - {max_d:.1f}s (Avg {avg_d:.1f}s, Hard Cap 8.0s)</span> &nbsp;|&nbsp; "
            f"<span style='color: #CBD5E1;'>Mix: {mix_str}</span> &nbsp;|&nbsp; "
            f"<span style='color: #4ADE80; font-weight: bold;'>🛡️ User Storyboard & Bibles Ready</span>"
            f"{audio_info}"
        )
        self.plan_approval_banner.setVisible(True)
        if hasattr(self, "main_scroll"):
            self.main_scroll.ensureWidgetVisible(self.plan_approval_banner)

    def _on_approve_plan(self):
        self.start_gen_btn.setEnabled(False)
        self.progress_panel.set_running_state(True)
        if hasattr(self, "approve_plan_btn"):
            self.approve_plan_btn.setEnabled(False)
            self.approve_plan_btn.setText("⏳ Generating Prompts with Groq AI...")
        if hasattr(self, "replan_btn"):
            self.replan_btn.setEnabled(False)
        if hasattr(self, "plan_summary_lbl"):
            self.plan_summary_lbl.setText("⚡ <b>Sending requests to Groq API...</b> Initializing batch prompt generation...")
        if hasattr(self, "plan_approval_banner"):
            self.plan_approval_banner.setVisible(True)

        if hasattr(self, "prompt_grid"):
            self.prompt_grid.show_progress("⚡ Sending requests to Groq API... Initializing batch prompt generation...", 15.0)

        # Authoritatively resolve active visual style from user selection
        active_style = getattr(self, "active_style_profile", None)
        active_style_id = active_style.id if active_style else (getattr(self.project.state, "style_preset_id", None) or self.project.config.visual_style_id or "default")
        self.project.config.visual_style_id = active_style_id

        self.project.config.visual_style_id = active_style
        if hasattr(self.project, "state") and self.project.state:
            self.project.state.style_preset_id = active_style

        story_text = self.story_text_edit.toPlainText().strip() or getattr(self.project.state, "story_raw_text", "")
        req = JobCreateRequest(
            project_name=self.project.name,
            story_text=story_text,
            narration=self.project.state.narration,
            subject_id=self.project.config.subject_id,
            custom_subject=self.project.config.custom_subject,
            visual_style_id=active_style,
            custom_style=self.project.config.custom_style,
            image_count_mode=self.project.config.image_count_mode,
            target_image_count=self.project.config.target_image_count,
            timing_mode=TimingMode.AI_DECIDES,
            min_duration=self.project.config.min_duration,
            preferred_duration=self.project.config.preferred_duration,
            max_duration=self.project.config.max_duration,
            batch_size=self.settings.default_batch_size,
            model_name=self.settings.default_model
        )

        # Phase 2: Generate prompts scene-by-scene with strict adherence to active_style
        self.active_job_worker = CloudJobWorker(
            base_url=self.settings.modal_api_url,
            request=req,
            api_key=self.settings.api_key,
            auth_token=self.settings.auth_token,
            plan_only=False,
            scenes_to_generate=self.project.state.scenes,
            analysis=self.project.state.analysis,
            bibles=getattr(self.project.state, "extracted_bibles", None)
        )
        if not hasattr(self, "_active_workers"):
            self._active_workers = []
        self._active_workers.append(self.active_job_worker)
        self.active_job_worker.finished.connect(
            lambda w=self.active_job_worker: self._active_workers.remove(w) if w in self._active_workers else None
        )
        self.active_job_worker.progress_updated.connect(self._on_job_progress_updated)
        self.active_job_worker.stage_changed.connect(self._on_job_stage_changed)
        self.active_job_worker.scenes_streamed.connect(self._on_job_scenes_streamed)
        self.active_job_worker.job_completed.connect(self._on_job_completed)
        self.active_job_worker.job_failed.connect(self._on_job_failed)
        self.active_job_worker.job_cancelled.connect(self._on_job_cancelled)
        self.active_job_worker.start()

    def _on_job_stage_changed(self, msg: str):
        if hasattr(self, "plan_summary_lbl"):
            self.plan_summary_lbl.setText(f"⚡ <b>Groq AI Pipeline:</b> {msg}")
        if hasattr(self, "prompt_grid"):
            curr_val = self.prompt_grid.live_progress_bar.value() if hasattr(self.prompt_grid, "live_progress_bar") else 30
            self.prompt_grid.update_progress(msg, curr_val)

    def _on_replan(self):
        self.plan_approval_banner.setVisible(False)
        self.start_gen_btn.setEnabled(True)
        self.start_gen_btn.setText("➔ Next: Proceed to Studio (Review & Expand Prompts)")
        self.progress_panel.set_running_state(False)
        if hasattr(self, "prompt_grid"):
            self.prompt_grid.hide_progress()

    def _on_job_progress_updated(self, progress: ProgressInfo):
        self.progress_panel.update_progress(progress)
        if hasattr(self, "plan_summary_lbl"):
            self.plan_summary_lbl.setText(f"⚡ <b>Generating prompts ({int(progress.progress_percentage)}%):</b> {progress.message}")
        if hasattr(self, "prompt_grid"):
            self.prompt_grid.update_progress(progress.message, progress.progress_percentage)
        if self.active_job_worker and self.active_job_worker.job_id:
            ProjectManager.mark_job_in_progress(
                self.project,
                self.active_job_worker.job_id,
                progress.stage.value,
                progress.progress_percentage
            )

    def _on_job_scenes_streamed(self, scenes: List[VisualScene]):
        self.project.state.scenes = scenes
        self._refresh_scenes_ui(scenes)
        self.tab3_images.load_scenes(scenes)
        self.tab4_render.load_project_data(
            scenes=scenes,
            audio_path=self.project.state.audio_file_path,
            narration=self.project.state.narration
        )

    def _update_story_bible_ui(self, analysis=None):
        """Dynamically populates the Story Bible cards from Stage 1 analysis."""
        if not hasattr(self, "bible_cards_layout"):
            return

        try:
            # Clear existing card widgets
            while self.bible_cards_layout.count():
                item = self.bible_cards_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        except Exception:
            return

        if not analysis:
            self.bible_badge.setText("READY FOR STORY ANALYSIS")
            self.bible_badge.setStyleSheet("background-color: #1e1b4b; color: #a5b4fc; font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 3px; border: 1px solid #4338ca;")
            
            c = QFrame()
            c.setStyleSheet("background-color: #0a0e16; border: 1px solid #202a3e; border-radius: 4px; padding: 6px 10px;")
            c_lay = QVBoxLayout(c)
            l = QLabel("<span style='color: #64748b; font-size: 11px;'>Enter or load your story narration above and start generation. Stage 1 Analysis will automatically extract recurring characters, environments, and continuity constraints here.</span>")
            l.setWordWrap(True)
            c_lay.addWidget(l)
            self.bible_cards_layout.addWidget(c)
            return

        chars = getattr(analysis.continuity, "characters", []) if hasattr(analysis, "continuity") and analysis.continuity else []
        locs = getattr(analysis.continuity, "locations", []) if hasattr(analysis, "continuity") and analysis.continuity else []
        objs = getattr(analysis.continuity, "key_objects", []) if hasattr(analysis, "continuity") and analysis.continuity else []
        total = len(chars) + len(locs) + len(objs)

        if total > 0:
            self.bible_badge.setText(f"{total} CONTINUITY ENTITIES ACTIVE")
            self.bible_badge.setStyleSheet("background-color: #3b0764; color: #d8b4fe; font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 3px; border: 1px solid #7e22ce;")
        else:
            self.bible_badge.setText("CONTINUITY & STYLE ACTIVE")
            self.bible_badge.setStyleSheet("background-color: #1e1b4b; color: #a5b4fc; font-size: 10px; font-weight: bold; padding: 2px 6px; border-radius: 3px; border: 1px solid #4338ca;")

        # Character cards (up to 2)
        for c in chars[:2]:
            c_frame = QFrame()
            c_frame.setStyleSheet("background-color: #0a0e16; border: 1px solid #202a3e; border-radius: 4px; padding: 6px 10px;")
            c_lay = QVBoxLayout(c_frame)
            desc = c.physical_description
            if getattr(c, "clothing", None):
                desc += f" | Attire: {c.clothing}"
            lbl = QLabel(f"<b>👤 {c.name} ({getattr(c, 'role', 'Character') or 'Character'})</b><br><span style='color: #94a3b8; font-size: 11px;'>{desc}</span>")
            lbl.setWordWrap(True)
            c_lay.addWidget(lbl)
            self.bible_cards_layout.addWidget(c_frame)

        # Location cards (up to 2)
        for loc in locs[:2]:
            l_frame = QFrame()
            l_frame.setStyleSheet("background-color: #0a0e16; border: 1px solid #202a3e; border-radius: 4px; padding: 6px 10px;")
            l_lay = QVBoxLayout(l_frame)
            desc = loc.environmental_features
            if getattr(loc, "architectural_style", None):
                desc += f" | Style: {loc.architectural_style}"
            lbl = QLabel(f"<b>🏛️ {loc.name}</b><br><span style='color: #94a3b8; font-size: 11px;'>{desc}</span>")
            lbl.setWordWrap(True)
            l_lay.addWidget(lbl)
            self.bible_cards_layout.addWidget(l_frame)

        # Object cards (up to 1)
        for obj in objs[:1]:
            o_frame = QFrame()
            o_frame.setStyleSheet("background-color: #0a0e16; border: 1px solid #202a3e; border-radius: 4px; padding: 6px 10px;")
            o_lay = QVBoxLayout(o_frame)
            lbl = QLabel(f"<b>📦 {obj.name}</b><br><span style='color: #94a3b8; font-size: 11px;'>{obj.visual_description}</span>")
            lbl.setWordWrap(True)
            o_lay.addWidget(lbl)
            self.bible_cards_layout.addWidget(o_frame)

        # Negative Prompt / Consistency Guard Card
        guard_frame = QFrame()
        guard_frame.setStyleSheet("background-color: #0a0e16; border: 1px solid #202a3e; border-radius: 4px; padding: 6px 10px;")
        g_lay = QVBoxLayout(guard_frame)
        tone = getattr(analysis, "visual_tone_notes", None) or "Period authenticity, contextual lighting, consistent visual style."
        lbl = QLabel(f"<b>🛡️ Consistency & Tone Guard</b><br><span style='color: #94a3b8; font-size: 11px;'>{tone}</span>")
        lbl.setWordWrap(True)
        g_lay.addWidget(lbl)
        self.bible_cards_layout.addWidget(guard_frame)

    def _on_job_analysis_ready(self, analysis):
        self.project.state.analysis = analysis
        self._update_story_bible_ui(analysis)

    def closeEvent(self, event):
        """Ensures background threads are safely terminated and app + CMD window exit immediately."""
        try:
            if hasattr(self, "parallel_worker") and self.parallel_worker and self.parallel_worker.isRunning():
                self.parallel_worker.cancel()
            if hasattr(self, "active_job_worker") and self.active_job_worker and self.active_job_worker.isRunning():
                self.active_job_worker.cancel()
        except Exception:
            pass
        event.accept()
        QApplication.instance().quit()
        os._exit(0)

    def _on_job_completed(self, resp):
        try:
            # Clear focus cleanly to prevent Qt6 dangling pointer focus restore crashes
            fw = QApplication.focusWidget()
            if fw:
                fw.clearFocus()
            self.setFocus()

            self.start_gen_btn.setEnabled(True)
            self.progress_panel.set_running_state(False)
            if hasattr(self, "prompt_grid"):
                self.prompt_grid.hide_progress()
            if resp.scenes:
                self.project.state.scenes = resp.scenes
                self._refresh_scenes_ui(resp.scenes, select_first=True)
                self.tab3_images.load_scenes(resp.scenes)
                self.tab4_render.load_project_data(
                    scenes=resp.scenes,
                    audio_path=self.project.state.audio_file_path,
                    narration=self.project.state.narration
                )

            # 1. Save clean prompts_only.txt & image_prompts.txt to data/ subfolder
            try:
                prompts_clean = StoryboardExporter.export_to_prompts_only(resp.scenes)
                data_dir = Path("data")
                data_dir.mkdir(parents=True, exist_ok=True)
                with open(data_dir / "prompts_only.txt", "w", encoding="utf-8") as pf:
                    pf.write(prompts_clean + "\n")
                full_text = StoryboardExporter.export_to_text(resp.scenes)
                with open(data_dir / "image_prompts.txt", "w", encoding="utf-8") as ipf:
                    ipf.write(full_text)

                # Automatically copy canonical image generation payload to Windows clipboard
                if prompts_clean and prompts_clean.strip():
                    from app.core.export import copy_to_system_clipboard
                    copy_to_system_clipboard(prompts_clean.strip())
                    print(f"[Clipboard] Automatically copied {len(resp.scenes)} scene prompts to Windows clipboard.")
            except Exception as e:
                print(f"[Export] Root export notice: {e}")

            # 2. Resolve project directory and auto-configure project images folder
            if not self.project.project_dir:
                p_name = self.proj_name_input.text().strip() if hasattr(self, "proj_name_input") and self.proj_name_input.text().strip() else self.project.name
                if not p_name or p_name == "Untitled_Project":
                    raw_s = self.story_text_edit.toPlainText().strip() if hasattr(self, "story_text_edit") else ""
                    p_name = self.generate_project_name_from_script(raw_s) if raw_s else f"Project_{datetime.now().strftime('%Y%m%d_%H%M')}"
                from app.config.settings import resolve_project_dir
                self.project.project_dir = str(resolve_project_dir(self.settings.workspace_directory, p_name).resolve())

            p_dir = Path(self.project.project_dir)
            p_dir.mkdir(parents=True, exist_ok=True)
            img_dir = p_dir / "images"
            img_dir.mkdir(parents=True, exist_ok=True)

            try:
                with open(p_dir / "prompts_only.txt", "w", encoding="utf-8") as pf:
                    pf.write(prompts_clean + "\n")
                with open(p_dir / "prompts.txt", "w", encoding="utf-8") as pf:
                    pf.write(prompts_clean + "\n")
                with open(p_dir / "image_prompts.txt", "w", encoding="utf-8") as pf:
                    pf.write(full_text)

                # Save Character Bible, Location Bible, Time & Period Bible
                if self.project.state.analysis:
                    from backend.pipeline.stage1_story_intelligence import save_story_bibles
                    save_story_bibles(p_dir, self.project.state.analysis)
            except Exception as e:
                print(f"[Export] Project export notice: {e}")

            # If items exceed 100, generate individual batch files for each Google account (e.g. 1-100, 101-200)
            try:
                batch_size = 100
                all_prompt_lines = [ln.strip() for ln in prompts_clean.splitlines() if ln.strip()]
                total_items = len(all_prompt_lines)
                if total_items > batch_size:
                    for b_idx in range(0, total_items, batch_size):
                        acc_num = (b_idx // batch_size) + 1
                        sub_lines = all_prompt_lines[b_idx : b_idx + batch_size]
                        start_num = b_idx + 1
                        end_num = min(b_idx + batch_size, total_items)
                        batch_fname = f"prompts_acc{acc_num}_{start_num:03d}_to_{end_num:03d}.txt"
                        batch_content = "\n".join(sub_lines) + "\n"
                        with open(p_dir / batch_fname, "w", encoding="utf-8") as bpf:
                            bpf.write(batch_content)
                        # Also place directly in account image subfolder
                        acc_img_dir = img_dir if acc_num == 1 else (img_dir / f"acc{acc_num}")
                        acc_img_dir.mkdir(parents=True, exist_ok=True)
                        with open(acc_img_dir / batch_fname, "w", encoding="utf-8") as bpf:
                            bpf.write(batch_content)
            except Exception as e:
                print(f"[Export] Multi-account batch files notice: {e}")

            # Auto-set download directory for Chrome profiles acc1..acc6
            for i in range(1, 7):
                acc_name = f"acc{i}"
                acc_img_dir = img_dir if i == 1 else (img_dir / acc_name)
                acc_img_dir.mkdir(parents=True, exist_ok=True)
                BrowserProfilesManager.configure_download_dir(acc_name, acc_img_dir)

            # Refresh top project combo to include this project
            self._populate_project_combo()

            # Auto-select images folder in Images Panel (Step 3) & Scene Sync Panel
            if hasattr(self, "images_panel"):
                self.images_panel.set_project(p_dir)
                self.images_panel.scenes = resp.scenes
                self.images_panel.set_audio_path(self.project.state.audio_file_path or "")
                if self.project.state.narration:
                    self.images_panel.narration = self.project.state.narration
                self.images_panel._populate_table()
                self.images_panel._refresh_generate_btn_label()
                self.images_panel._scan_images()

            # Persist project state
            try:
                ProjectManager.save_project(self.project)
            except Exception as e:
                print(f"[ProjectManager] Save notice: {e}")

            ProjectManager.clear_job_in_progress(self.project)

            # Elegant Non-Modal In-App Success Banner (Prevents any modal focus collapse)
            error_count = sum(1 for s in resp.scenes if getattr(s, "generation_status", "") == "error" or not (s.prompt and s.prompt.strip()))
            if error_count > 0:
                self.plan_summary_lbl.setText(
                    f"⚠️ <b>Generation Complete with Retries Needed!</b> &nbsp;|&nbsp; "
                    f"<span style='color: #B8F24A;'>{len(resp.scenes) - error_count}/{len(resp.scenes)} Prompts Ready</span> &nbsp;|&nbsp; "
                    f"<span style='color: #EF4444;'>{error_count} Scene(s) Skipped (click 'Retry Prompt' in red below)</span> &nbsp;|&nbsp; "
                    f"<span style='color: #38BDF8;'>Saved to {p_dir.name}</span>"
                )
            else:
                self.plan_summary_lbl.setText(
                    f"🎉 <b>Generation Complete!</b> &nbsp;|&nbsp; "
                    f"<span style='color: #B8F24A;'>{len(resp.scenes)} Scenes Validated</span> &nbsp;|&nbsp; "
                    f"<span style='color: #38BDF8;'>Saved to {p_dir.name} & prompts_only.txt</span> &nbsp;|&nbsp; "
                    f"<span style='color: #A3E635;'>Images Folder: {p_dir.name}/images/</span>"
                )
            self.plan_approval_banner.setStyleSheet("""
                QFrame {
                    background-color: #0b1f14;
                    border: 1px solid #10b981;
                    border-radius: 6px;
                    padding: 8px 12px;
                }
            """)
            self.approve_plan_btn.setText("➔ Next Step: Generate Images (Go to Images)")
            self.approve_plan_btn.setEnabled(True)
            self.approve_plan_btn.setStyleSheet("""
                QPushButton {
                    background-color: #10b981;
                    color: #0b0f19;
                    font-weight: 800;
                    font-size: 12px;
                    border-radius: 4px;
                    padding: 0 16px;
                }
                QPushButton:hover {
                    background-color: #34d399;
                }
            """)
            try:
                self.approve_plan_btn.clicked.disconnect()
            except Exception:
                pass
            self.approve_plan_btn.clicked.connect(lambda: self.sidebar.set_current_page("images"))
            self.plan_approval_banner.setVisible(True)

            self.statusBar().showMessage(
                f"🎉 Generation Complete: {len(resp.scenes)} storyboard scenes ready! Prompts saved to {p_dir.name}.",
                15000
            )

            # Switch cleanly to Studio page so user sees prompt matrix immediately
            self.sidebar.set_current_page("studio")

        except Exception as e:
            print(f"[MainWindow] Error in _on_job_completed: {e}")

    def _on_job_failed(self, user_msg: str, tech_details: str):
        self.start_gen_btn.setEnabled(True)
        self.progress_panel.set_running_state(False)
        if hasattr(self, "prompt_grid"):
            self.prompt_grid.hide_progress()
        msg_box = QMessageBox(self)
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Generation Error")
        msg_box.setText(user_msg)
        if tech_details:
            msg_box.setDetailedText(tech_details)
        msg_box.exec()

    def _on_cancel_generation(self):
        if hasattr(self, "parallel_worker") and self.parallel_worker and self.parallel_worker.isRunning():
            self.parallel_worker.cancel()
        if self.active_job_worker:
            self.active_job_worker.cancel()
            self.progress_panel.mark_cancelled()
        self.start_gen_btn.setEnabled(True)
        self.progress_panel.set_running_state(False)
        if hasattr(self, "prompt_grid"):
            self.prompt_grid.hide_progress()

    def _on_job_cancelled(self):
        self.start_gen_btn.setEnabled(True)
        self.progress_panel.set_running_state(False)
        if hasattr(self, "prompt_grid"):
            self.prompt_grid.hide_progress()
        ProjectManager.clear_job_in_progress(self.project)
        QMessageBox.information(self, "Cancelled", "Cloud generation job was cancelled.")

    # =========================================================================
    # SCENE LIST & PROMPT INSPECTOR
    # =========================================================================

    def _refresh_scenes_ui(self, scenes: List[VisualScene], select_first: bool = False):
        if hasattr(self, "prompt_grid"):
            self.prompt_grid.load_scenes(scenes)
        self.scene_count_badge.setText(f"{len(scenes)} Prompts")
        self.scenes_list.blockSignals(True)
        self.scenes_list.clear()
        filter_text = self.scene_search_input.text().strip().lower()

        for sc in scenes:
            if filter_text and filter_text not in sc.narration_excerpt.lower() and filter_text not in sc.visual_subject.lower() and filter_text not in sc.prompt.lower():
                continue

            item = QListWidgetItem()
            edit_mark = " [EDITED]" if sc.is_manually_edited else ""
            prompt_preview = (sc.prompt[:75] + "...") if len(sc.prompt) > 75 else sc.prompt
            item.setText(
                f"Image #{sc.scene_id} [{sc.start:.1f}s - {sc.end:.1f}s | {sc.duration:.1f}s]{edit_mark}\n"
                f"  Subject: {sc.visual_subject}\n"
                f"  Prompt: {prompt_preview}"
            )
            item.setData(Qt.ItemDataRole.UserRole, sc.scene_id)
            self.scenes_list.addItem(item)

        self.scenes_list.blockSignals(False)

        if select_first and scenes:
            self.scenes_list.setCurrentRow(0)
            self.scene_editor.load_scene(scenes[0])

    def _on_scene_list_item_selected(self, current: Optional[QListWidgetItem]):
        if not current:
            return
        scene_id = current.data(Qt.ItemDataRole.UserRole)
        sc = next((s for s in self.project.state.scenes if s.scene_id == scene_id), None)
        if sc:
            self.scene_editor.load_scene(sc)

    def _on_scene_search_changed(self, text: str):
        self._refresh_scenes_ui(self.project.state.scenes)

    def _on_quick_copy_all(self):
        if not self.project.state.scenes:
            QMessageBox.warning(self, "No Prompts", "No generated prompts to copy.")
            return

        clean_payload = StoryboardExporter.export_to_prompts_only(self.project.state.scenes)
        from app.core.export import copy_to_system_clipboard
        copy_to_system_clipboard(clean_payload)

        prompt_count = len([l for l in clean_payload.splitlines() if l.strip()])
        QMessageBox.information(self, "Copied", f"Copied {prompt_count} image prompts to clipboard (1 prompt per scene).")

    def _select_scene_by_id(self, scene_id: str):
        for i in range(self.scenes_list.count()):
            item = self.scenes_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == scene_id:
                self.scenes_list.setCurrentRow(i)
                break
        sc = next((s for s in self.project.state.scenes if s.scene_id == scene_id), None)
        if sc:
            self.scene_editor.load_scene(sc)

    def _on_scene_edited(self, updated_scene: VisualScene):
        idx = next((i for i, s in enumerate(self.project.state.scenes) if s.scene_id == updated_scene.scene_id), None)
        if idx is not None:
            self.project.state.scenes[idx] = updated_scene
            healed = TimelineEngine.heal_gaps_and_overlaps(self.project.state.scenes)
            self.project.state.scenes = healed
            self._refresh_scenes_ui(healed)
            self.tab3_images.load_scenes(healed)
            self.tab4_render.load_project_data(healed, self.project.state.audio_file_path, self.project.state.narration)
            self._select_scene_by_id(updated_scene.scene_id)

    def _on_regenerate_single_scene(self, scene_id: str):
        sc = next((s for s in self.project.state.scenes if s.scene_id == scene_id), None)
        if not sc:
            return

        req = SceneRegenerateRequest(
            scene=sc,
            story_summary=self.project.state.analysis.story_summary if self.project.state.analysis else "",
            subject_id=self.project.config.subject_id,
            custom_subject=self.project.config.custom_subject,
            visual_style_id=self.project.config.visual_style_id,
            custom_style=self.project.config.custom_style,
            continuity=self.project.state.analysis.continuity if self.project.state.analysis else None,
            model_name=self.settings.default_model
        )

        self.scene_editor.setEnabled(False)
        self.active_regen_worker = SceneRegenerateWorker(
            base_url=self.settings.modal_api_url,
            request=req,
            api_key=self.settings.api_key,
            auth_token=self.settings.auth_token
        )
        self.active_regen_worker.regenerate_completed.connect(self._on_single_scene_regenerated)
        self.active_regen_worker.regenerate_failed.connect(self._on_single_scene_regen_failed)
        self.active_regen_worker.start()

    def _on_single_scene_regenerated(self, resp):
        self.scene_editor.setEnabled(True)
        idx = next((i for i, s in enumerate(self.project.state.scenes) if s.scene_id == resp.scene_id), None)
        if idx is not None:
            sc = self.project.state.scenes[idx]
            sc.prompt = resp.prompt
            sc.visual_subject = resp.visual_subject
            sc.is_validated = True
            self.scene_editor.load_scene(sc)
            self._refresh_scenes_ui(self.project.state.scenes)
            QMessageBox.information(self, "Regeneration Complete", f"Scene #{resp.scene_id} regenerated successfully.")

    def _on_single_scene_regen_failed(self, msg: str, tech: str):
        self.scene_editor.setEnabled(True)
        QMessageBox.critical(self, "Regeneration Failed", f"{msg}\n\n{tech}")

    def _on_split_scene(self, scene_id: str, split_time: float):
        try:
            updated_list = TimelineEngine.split_scene(
                self.project.state.scenes,
                scene_id,
                split_time,
                self.project.state.narration
            )
            self.project.state.scenes = updated_list
            self._refresh_scenes_ui(updated_list, select_first=False)
            self.tab3_images.load_scenes(updated_list)
            self.tab4_render.load_project_data(updated_list, self.project.state.audio_file_path, self.project.state.narration)
            self._select_scene_by_id(scene_id)
            QMessageBox.information(self, "Scene Split", f"Scene #{scene_id} split into two scenes at {split_time:.2f}s.")
        except Exception as e:
            QMessageBox.critical(self, "Split Failed", str(e))

    def _on_merge_scene(self, scene_id: str):
        scenes = self.project.state.scenes
        idx = next((i for i, s in enumerate(scenes) if s.scene_id == scene_id), None)
        if idx is None or idx >= len(scenes) - 1:
            QMessageBox.warning(self, "Merge Not Possible", "No subsequent scene available to merge with.")
            return

        next_id = scenes[idx + 1].scene_id
        try:
            updated_list = TimelineEngine.merge_scenes(
                scenes,
                scene_id,
                next_id,
                self.project.state.narration
            )
            self.project.state.scenes = updated_list
            self._refresh_scenes_ui(updated_list)
            self.tab3_images.load_scenes(updated_list)
            self.tab4_render.load_project_data(updated_list, self.project.state.audio_file_path, self.project.state.narration)
            self._select_scene_by_id(scene_id)
            QMessageBox.information(self, "Scene Merged", f"Merged Scene #{scene_id} and #{next_id}.")
        except Exception as e:
            QMessageBox.critical(self, "Merge Failed", str(e))

    def _on_delete_scene(self, scene_id: str):
        try:
            updated_list = TimelineEngine.delete_scene(
                self.project.state.scenes,
                scene_id,
                self.project.state.narration
            )
            self.project.state.scenes = updated_list
            self._refresh_scenes_ui(updated_list, select_first=True)
            self.tab3_images.load_scenes(updated_list)
            self.tab4_render.load_project_data(updated_list, self.project.state.audio_file_path, self.project.state.narration)
            if not updated_list:
                self.scene_editor.load_scene(None)
        except Exception as e:
            QMessageBox.critical(self, "Delete Failed", str(e))

    def _on_retry_single_scene(self, scene: VisualScene):
        """Retries prompt generation for a single skipped or failed scene via Groq."""
        active_style = getattr(self.project.state, "style_preset_id", None) or self.project.config.visual_style_id or "zombie_survival"
        self.statusBar().showMessage(f"⏳ Generating prompt for Scene #{scene.scene_id} via Groq...", 6000)

        from app.ai.schemas import SceneRegenerateRequest
        target_model = getattr(self.settings, "default_model", None) or "qwen/qwen3.8-27b"
        if "qwen3-30b" in target_model.lower():
            target_model = "qwen/qwen3.8-27b"

        req = SceneRegenerateRequest(
            scene=scene,
            story_summary=getattr(self.project.state, "story_raw_text", "")[:800],
            visual_style_id=active_style,
            custom_style=getattr(self.project.config, "custom_style", ""),
            model_name=target_model
        )
        worker = SceneRegenerateWorker(request=req)
        if not hasattr(self, "_single_retry_workers"):
            self._single_retry_workers = []
        self._single_retry_workers.append(worker)

        def _on_success(resp):
            scene.prompt = resp.prompt
            scene.generation_status = "generated"
            scene.is_validated = True
            if hasattr(self, "prompt_grid"):
                self.prompt_grid.update_single_scene(scene)
            self.statusBar().showMessage(f"✅ Scene #{scene.scene_id} prompt generated successfully!", 6000)

            # Update exports and save project
            try:
                prompts_clean = StoryboardExporter.export_to_prompts_only(self.project.state.scenes)
                data_dir = Path("data")
                data_dir.mkdir(parents=True, exist_ok=True)
                with open(data_dir / "prompts_only.txt", "w", encoding="utf-8") as pf:
                    pf.write(prompts_clean + "\n")
                if self.project.project_dir:
                    p_dir = Path(self.project.project_dir)
                    with open(p_dir / "prompts_only.txt", "w", encoding="utf-8") as pf:
                        pf.write(prompts_clean + "\n")
                    with open(p_dir / "prompts.txt", "w", encoding="utf-8") as pf:
                        pf.write(prompts_clean + "\n")
                ProjectManager.save_project(self.project)
            except Exception as ex:
                print(f"[RetryScene] Export save notice: {ex}")

            if worker in self._single_retry_workers:
                self._single_retry_workers.remove(worker)

        def _on_failed(user_msg, tech):
            scene.generation_status = "error"
            if hasattr(self, "prompt_grid"):
                self.prompt_grid.update_single_scene(scene)
            self.statusBar().showMessage(f"❌ Retry failed for Scene #{scene.scene_id}: {user_msg}", 8000)
            if worker in self._single_retry_workers:
                self._single_retry_workers.remove(worker)

        worker.regenerate_completed.connect(_on_success)
        worker.regenerate_failed.connect(_on_failed)
        worker.start()

    def _on_add_manual_scene(self):
        scenes = self.project.state.scenes
        new_id = str(len(scenes) + 1)
        start_time = scenes[-1].end if scenes else 0.0
        end_time = round(start_time + 5.0, 2)
        new_scene = VisualScene(
            scene_id=new_id,
            start=round(start_time, 2),
            end=end_time,
            duration=5.0,
            narration_excerpt=f"Scene {new_id} narration excerpt...",
            visual_subject=f"Key Subject for Scene #{new_id}",
            framing_suggestion="Eye-level medium shot, cinematic lighting",
            prompt=f"A cinematic studio-quality visualization of Scene #{new_id}, highly detailed, volumetric lighting, photorealistic, 8k resolution.",
            is_manually_edited=True
        )
        scenes.append(new_scene)
        self.project.state.scenes = scenes
        self._refresh_scenes_ui(scenes)
        self.tab3_images.load_scenes(scenes)
        self.tab4_render.load_project_data(scenes, self.project.state.audio_file_path, self.project.state.narration)
        self._select_scene_by_id(new_id)

    # =========================================================================
    # CLOUD CONNECTION & RECOVERY
    # =========================================================================

    def _check_cloud_connection(self):
        if hasattr(self, "modal_url_lbl"):
            self.modal_url_lbl.setText("⚡ Engine: Groq Multi-Key AI & Whisper (11 Keys Active)")
        self.cloud_status_lbl.setText("⏳ Checking Groq...")
        self.cloud_status_lbl.setStyleSheet("color: #38bdf8; font-size: 11px;")

        if self._cloud_worker and self._cloud_worker.isRunning():
            return

        self._cloud_worker = ConnectionCheckWorker(
            "groq",
            self.settings.api_key,
            self.settings.auth_token,
            timeout=15.0,
            parent=self
        )
        self._cloud_worker.result_ready.connect(self._on_cloud_check_finished)
        self._cloud_worker.start()

    def _on_cloud_check_finished(self, resp):
        if resp.connected:
            model = resp.backend_info.model if resp.backend_info else "qwen/qwen3.8-27b"
            gpu = resp.backend_info.gpu if resp.backend_info else "Groq LPU Array"
            self.status_banner.set_connected(
                model,
                gpu,
                resp.latency_ms
            )
            self.cloud_status_lbl.setText(f"✅ Groq Ready ({resp.latency_ms:.0f}ms)")
            self.cloud_status_lbl.setStyleSheet("color: #4ade80; font-size: 11px; font-weight: bold;")
        else:
            self.status_banner.set_offline(resp.message)
            self.cloud_status_lbl.setText("⚠️ Groq Offline")
            self.cloud_status_lbl.setStyleSheet("color: #f87171; font-size: 11px;")

    def _check_project_recovery(self):
        unfinished = ProjectManager.check_for_unfinished_jobs(self.settings.workspace_directory)
        if unfinished:
            name, proj_dir, info = unfinished[0]
            dlg = RecoveryDialog(name, info.get("stage", "Unknown"), float(info.get("progress", 0.0)), self)
            if dlg.exec():
                if dlg.action in ("resume", "open"):
                    try:
                        proj = ProjectManager.load_project(proj_dir)
                        self._apply_project_to_ui(proj, auto_switch_tab=True)
                        if dlg.action == "resume":
                            if proj.state.story_raw_text:
                                self._on_start_generation()
                            else:
                                QMessageBox.information(self, "Story Required", "Please enter or check the story script in Tab 1, then click Generate Storyboard.")
                    except Exception as e:
                        QMessageBox.warning(self, "Recovery Notice", f"Could not load project: {str(e)}")
                elif dlg.action == "discard":
                    ProjectManager.clear_job_in_progress(proj_dir)

    def _on_open_settings(self):
        dlg = SettingsDialog(self)
        if dlg.exec():
            self.settings = SettingsManager.load_settings()
            self._check_cloud_connection()

    def _on_open_preset_editor(self):
        dlg = PresetEditorDialog(self)
        if dlg.exec():
            self._populate_subject_combo()

    def _on_export_dialog(self):
        if not self.project.state.scenes:
            QMessageBox.warning(self, "No Scenes", "There are no generated scenes to export yet.")
            return

        p_dir = getattr(self.project, "project_dir", None)
        dlg = ExportDialog(
            self.project.state.scenes,
            self.project.state.narration,
            project_dir=p_dir,
            parent=self
        )
        dlg.exec()

    def _on_open_prompts_file(self):
        """Directly opens prompts_only.txt in the system's default text editor (Notepad)."""
        target_path = None
        if hasattr(self, "project") and self.project and self.project.project_dir:
            p_file = Path(self.project.project_dir) / "prompts_only.txt"
            if p_file.exists():
                target_path = p_file
        if not target_path and Path("prompts_only.txt").exists():
            target_path = Path("prompts_only.txt")

        # If not yet on disk but scenes exist in memory, export immediately
        if hasattr(self, "project") and self.project and self.project.state.scenes:
            content = StoryboardExporter.export_to_prompts_only(self.project.state.scenes)
            if not target_path:
                target_path = Path("prompts_only.txt")
            target_path.write_text(content, encoding="utf-8")

        if target_path and target_path.exists():
            import os
            import subprocess
            if sys.platform == "win32":
                os.startfile(str(target_path))
            else:
                subprocess.Popen(["xdg-open", str(target_path)])
            self.statusBar().showMessage(f"📄 Opened {target_path.name}", 4000)
        else:
            QMessageBox.warning(self, "File Not Found", "No prompts_only.txt file found. Please generate storyboard prompts first.")