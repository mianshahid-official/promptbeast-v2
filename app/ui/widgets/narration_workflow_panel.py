"""
Production Narration-Aware Studio Workflow Panel for PromptBeast.
Implements Steps 1 to 5:
- Step 1: Import Script JSON & Audio File
- Step 2: Narration and Visual Ideas Input
- Step 3: Story Intelligence & Dynamic Bibles Extraction
- Step 4: Visual Style Selection & Target Settings
- Step 5: Groq Image Prompt Generation & Expansion
"""
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QPlainTextEdit,
    QLineEdit,
    QFileDialog,
    QMessageBox,
    QFrame,
    QTabWidget,
    QProgressBar,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QApplication,
)

from app.models.scene import VisualScene
from app.models.timeline import NarrationData
from app.core.narration_workflow import (
    import_script_json,
    parse_visual_breakdown_text,
    align_visuals_with_timestamps,
    parse_custom_bible_text,
)
from app.ai.narration_worker import BibleExtractionWorker, BatchPromptGenerationWorker
from app.ui.widgets.style_thumbnail_matrix import StyleThumbnailMatrix
from app.presets.loader import PresetManager


STYLE_PROMPT_MAP = {
    "zombie_survival": "01_zombie_survival_ai_prompt.txt",
    "whatif_flat_vector": "02_whatif_flat_vector_ai_prompt.txt",
    "century_old_photograph": "03_century_old_photograph_ai_prompt.txt",
    "bw_vintage_historical_photography": "04_bw_vintage_historical_photography_ai_prompt.txt",
    "3d_red_blue_stickman": "05_3d_red_blue_stickman_ai_prompt.txt",
    "2d_stickman": "06_2d_stickman_ai_prompt.txt",
    "3d_stickman": "07_3d_stickman_ai_prompt.txt",
    "mixed_media_collage": "08_mixed_media_collage_ai_prompt.txt",
    "paper_cutout": "09_paper_cutout_ai_prompt.txt",
    "paper_quilling": "10_paper_quilling_ai_prompt.txt",
    "vintage_photograph": "11_vintage_photograph_ai_prompt.txt",
}

STYLE_FEATURE_HIGHLIGHTS = {
    "zombie_survival": "🔒 Zombie Survival Directives: Locked Hero 'YOU' (Backwards Red Cap #D63A2A, Charcoal Hoodie #333333, Pure White T-Shirt #FFFFFF, Brown Crossbody Strap #4A2E1B) | Locked Horde (Uniform Olive-Green #8A9A66 Skin, White Eyes with Dot Pupils, Sharp Jagged Teeth, Tattered Brown Jacket #5C4033) | Thick Black #1A1A1A Outlines | Zero Blood/Gore.",
    "whatif_flat_vector": "🔒 What-If Flat Vector Directives: Minimalist Black Stick Figure | Strictly Locked Vibrant Bright RED Hoodie (#FF0000) across all scenes | Exactly Uniform Limb Thickness | Midnight Navy Night-World (#12182E) with Warm Golden Light Pools (#F59E0B) | Zero Gradients.",
    "century_old_photograph": "📜 Century-Old Photograph Directives: Authentic Damaged Archival Print | Faded Warm Sepia, Aged Albumen, or Silver Gelatin Monochrome | Naturally Torn & Chipped Paper Edges | Random Dust Specks & Thin Vertical Scratches | STRICTLY ZERO Modern RGB Colors.",
    "bw_vintage_historical_photography": "📷 B&W Historical Photography Directives: Genuine Archival Documentary Photography | Realistic Photographic Anatomy & Authentic Period Uniforms | Natural Monochrome Grayscale Tonal Range & Subtle Film Grain | Realistic Animals & Architecture | No Caricature / No 3D.",
    "3d_red_blue_stickman": "⚔️ 3D Red & Blue Stickman Directives: Strict Faction Separation | Attacking Forces = 100% Solid Bright Red 3D Stick Figures | Defending Forces = 100% Solid Blue 3D Stick Figures | No Color Mixing on Tasks | Featureless Smooth Spherical Heads | Rounded 3D Block Geometry | Zero Realistic Anatomy.",
    "2d_stickman": "🎨 2D Stickman Directives: Inspired by OverSimplified Cartoon Aesthetic | Bold Black Outlines & Smooth Flat Pastel/Primary Fills | Large Spherical Heads with Expressive Comedic Line Mouths & Dot Eyes | Flat Theatrical Staging | Zero 3D Shading.",
    "3d_stickman": "🧸 3D Stickman Directives: Soft-Toy Low-Poly CGI Hybrid | Tactile Matte Surface Finishes | Spherical Heads with Animated Eyebrow Shifts | Dimensional Tactile Environments (Wood, Stone, Cloth) | Dynamic Volumetric Lighting & Ambient Occlusion.",
    "mixed_media_collage": "✂️ Mixed-Media Collage Directives: Sophisticated Editorial Collage | Layered Photographic Cutouts, Archival Newspapers, Maps, Blueprints & Halftone Prints | Visible Cut Edges & Paper Texture | Dynamic Script-Driven Material Integration.",
    "paper_cutout": "📄 Paper-Cutout Directives: 2D Layered Cardstock Animation Style | Physically Cut Cardstock Shapes in Staggered Depth Planes | Visible Paper & Fiber Textures | Soft Delicate Drop Shadows Between Layers | Muted Historical Earth & Parchment Palette.",
    "paper_quilling": "🌀 Paper Quilling Directives: 100% Handcrafted Paper Quilling | Every Character, Tree, Cloud & Architecture Constructed from Narrow Standing Paper Strips Rolled, Curled & Coiled on Edge | Directional Raking Light & Tactile Relief Shadows.",
    "vintage_photograph": "🎞️ Vintage Photograph Directives: Timeless Analog Archival Print | Aged Warm Sepia & Silver Gelatin Tones | Natural Paper Grain, Corner Vignetting & Subtle Emulsion Hairlines | Period-Accurate Technology & Historic Staging.",
}


class PromptViewerDialog(QDialog):
    """Dialog showing the complete Master AI Prompt with copy functionality."""
    def __init__(self, style_name: str, prompt_text: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"PromptBeast Master AI Prompt — {style_name}")
        self.resize(840, 620)
        self.setStyleSheet("background-color: #0A0E17; color: #E2E8F0;")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 18, 18, 18)
        lay.setSpacing(12)

        hdr = QLabel(f"📋 Master AI Prompt for External LLM — {style_name}")
        hdr.setStyleSheet("font-size: 14px; font-weight: 700; color: #10B981;")
        lay.addWidget(hdr)

        desc = QLabel(
            "Instructions: Copy this complete prompt and send it to ChatGPT, Claude, or Gemini. "
            "The AI will acknowledge that it is ready, ask for your story script, and return your "
            "visual storyboard in PromptBeast's exact synchronized format. When you ask for bibles, "
            "it will provide the complete production bibles matching this visual style."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #94A3B8; font-size: 11px; line-height: 1.4;")
        lay.addWidget(desc)

        txt = QPlainTextEdit()
        txt.setPlainText(prompt_text)
        txt.setReadOnly(True)
        txt.setStyleSheet("""
            QPlainTextEdit {
                background-color: #05080E;
                color: #CBD5E1;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                border: 1px solid #1E293B;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        lay.addWidget(txt)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("📋 Copy Entire Prompt to Clipboard")
        btn_copy.setFixedHeight(36)
        btn_copy.setCursor(Qt.PointingHandCursor)
        btn_copy.setStyleSheet("""
            QPushButton {
                background-color: #10B981;
                color: #05080E;
                font-weight: 700;
                font-size: 12px;
                border-radius: 6px;
                padding: 0 20px;
            }
            QPushButton:hover {
                background-color: #34D399;
            }
        """)
        btn_close = QPushButton("Close")
        btn_close.setFixedHeight(36)
        btn_close.setCursor(Qt.PointingHandCursor)
        btn_close.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #94A3B8;
                font-size: 12px;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: #334155;
                color: #FFFFFF;
            }
        """)

        def _do_copy():
            QApplication.clipboard().setText(prompt_text)
            btn_copy.setText("✓ Copied to Clipboard!")
            btn_copy.setStyleSheet("background-color: #3B82F6; color: #FFFFFF; font-weight: 700; border-radius: 6px; padding: 0 20px;")

        btn_copy.clicked.connect(_do_copy)
        btn_close.clicked.connect(self.accept)

        btn_row.addWidget(btn_copy)
        btn_row.addStretch()
        btn_row.addWidget(btn_close)
        lay.addLayout(btn_row)


class NarrationWorkflowPanel(QWidget):
    """Unified Single-Page Narration Workflow Studio."""

    # Signals
    story_script_imported = Signal(str, object)   # (story_text, narration_data)
    visual_scenes_aligned = Signal(list)          # (List[VisualScene])
    bibles_updated = Signal(dict)                 # (extracted bibles dict)
    prompts_generated = Signal(list)              # (List[VisualScene])
    navigate_requested = Signal(str)              # page key e.g. "studio", "scene_sync", "deliverables"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.story_text: str = ""
        self.narration: Optional[NarrationData] = None
        self.scenes: List[VisualScene] = []
        self.bibles: Dict[str, Any] = {}
        self.json_path: Optional[str] = None
        self.audio_path: Optional[str] = None

        self._bible_worker: Optional[BibleExtractionWorker] = None
        self._prompt_worker: Optional[BatchPromptGenerationWorker] = None

        self._init_ui()
        self._auto_load_default_assets()

    def _auto_load_default_assets(self):
        """Auto-detects and preloads audio and files/speech.json and sample data.txt if present."""
        sample_json = Path("audio and files/speech.json")
        if sample_json.exists():
            self.load_json_file(str(sample_json.resolve()))

        sample_txt = Path("sample data.txt")
        if sample_txt.exists():
            try:
                txt_content = sample_txt.read_text(encoding="utf-8")
                self.txt_visual_ideas.setPlainText(txt_content)
                self._on_align_visuals()
            except Exception as e:
                print(f"[NarrationWorkflow] Auto-load notice: {e}")

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 10, 12, 14)
        main_layout.setSpacing(14)

        # ---------------------------------------------------------------------
        # STEP 1: IMPORT SCRIPT JSON & AUDIO
        # ---------------------------------------------------------------------
        step1_box = self._create_card("🎙️ STEP 1: Import Script JSON & Audio File")
        s1_lay = QVBoxLayout(step1_box)
        s1_lay.setSpacing(8)

        # File actions row
        f_row = QHBoxLayout()
        f_row.setSpacing(10)

        self.btn_browse_json = QPushButton("📁 Import Script JSON (speech.json)...")
        self.btn_browse_json.setFixedHeight(36)
        self.btn_browse_json.setCursor(Qt.PointingHandCursor)
        self.btn_browse_json.setStyleSheet("""
            QPushButton {
                background-color: #1A2433;
                color: #A3FF12;
                font-weight: 700;
                font-size: 12px;
                border: 1px solid rgba(163, 255, 18, 0.35);
                border-radius: 6px;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: rgba(163, 255, 18, 0.15);
                border-color: #A3FF12;
            }
        """)
        self.btn_browse_json.clicked.connect(self._on_browse_json)
        f_row.addWidget(self.btn_browse_json)

        self.btn_browse_audio = QPushButton("🎵 Select Audio File (speech.mp3)...")
        self.btn_browse_audio.setFixedHeight(36)
        self.btn_browse_audio.setCursor(Qt.PointingHandCursor)
        self.btn_browse_audio.setStyleSheet("""
            QPushButton {
                background-color: #1A2433;
                color: #38BDF8;
                font-weight: 700;
                font-size: 12px;
                border: 1px solid rgba(56, 189, 248, 0.35);
                border-radius: 6px;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 0.15);
                border-color: #38BDF8;
            }
        """)
        self.btn_browse_audio.clicked.connect(self._on_browse_audio)
        f_row.addWidget(self.btn_browse_audio)

        f_row.addStretch()
        s1_lay.addLayout(f_row)

        # Status row
        status_row = QHBoxLayout()
        self.lbl_json_status = QLabel("JSON: No script file loaded.")
        self.lbl_json_status.setStyleSheet("color: #94A3B8; font-size: 11px;")
        status_row.addWidget(self.lbl_json_status)

        status_row.addStretch()

        self.lbl_audio_status = QLabel("Audio: No audio file attached.")
        self.lbl_audio_status.setStyleSheet("color: #94A3B8; font-size: 11px;")
        status_row.addWidget(self.lbl_audio_status)
        s1_lay.addLayout(status_row)

        main_layout.addWidget(step1_box)

        # ---------------------------------------------------------------------
        # STEP 2: NARRATION & VISUAL IDEAS INPUT
        # ---------------------------------------------------------------------
        step2_box = self._create_card("📝 STEP 2: Narration & Visual Ideas Breakdown")
        s2_lay = QVBoxLayout(step2_box)
        s2_lay.setSpacing(8)

        s2_desc = QLabel("Paste script narration lines, corresponding visual ideas (e.g. Visual 1 Words: ... Visual: ...), scene descriptions, or creative instructions:")
        s2_desc.setStyleSheet("color: #94A3B8; font-size: 11px;")
        s2_lay.addWidget(s2_desc)

        self.txt_visual_ideas = QPlainTextEdit()
        self.txt_visual_ideas.setMinimumHeight(140)
        self.txt_visual_ideas.setMaximumHeight(260)
        self.txt_visual_ideas.setPlaceholderText(
            "Visual 1 Words: Sixty people were locked inside this mall on the first night. Visual: Wide cinematic shot of a massive shopping mall with terrified shoppers trapped behind locked steel shutters.\n\n"
            "Visual 2 Words: By Day 20, you will find proof that one of them is still alive. Visual: Dark underground mall corridor with mysterious fresh footprints in dusty concrete.\n..."
        )
        self.txt_visual_ideas.setStyleSheet("""
            QPlainTextEdit {
                background-color: #0B0F17;
                color: #E2E8F0;
                font-family: 'Consolas', 'Segoe UI', monospace;
                font-size: 11px;
                border: 1px solid #1E293B;
                border-radius: 6px;
                padding: 8px;
            }
            QPlainTextEdit:focus {
                border: 1px solid #10B981;
            }
        """)
        s2_lay.addWidget(self.txt_visual_ideas)

        # Alignment Action Row
        align_row = QHBoxLayout()
        self.btn_align_visuals = QPushButton("⚡ Parse & Synchronize Visual Ideas with Audio Timestamps")
        self.btn_align_visuals.setFixedHeight(34)
        self.btn_align_visuals.setCursor(Qt.PointingHandCursor)
        self.btn_align_visuals.setStyleSheet("""
            QPushButton {
                background-color: rgba(16, 185, 129, 0.2);
                color: #10B981;
                font-weight: 700;
                font-size: 12px;
                border: 1px solid #10B981;
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: rgba(16, 185, 129, 0.35);
            }
        """)
        self.btn_align_visuals.clicked.connect(self._on_align_visuals)
        align_row.addWidget(self.btn_align_visuals)

        self.btn_copy_ai_prompt_step2 = QPushButton("📋 Copy Master AI Prompt (External AI)")
        self.btn_copy_ai_prompt_step2.setFixedHeight(34)
        self.btn_copy_ai_prompt_step2.setCursor(Qt.PointingHandCursor)
        self.btn_copy_ai_prompt_step2.setStyleSheet("""
            QPushButton {
                background-color: rgba(168, 85, 247, 0.15);
                color: #C084FC;
                font-weight: 700;
                font-size: 11px;
                border: 1px solid rgba(168, 85, 247, 0.4);
                border-radius: 6px;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: rgba(168, 85, 247, 0.3);
                border-color: #C084FC;
            }
        """)
        self.btn_copy_ai_prompt_step2.clicked.connect(self._on_copy_style_prompt)
        align_row.addWidget(self.btn_copy_ai_prompt_step2)

        self.lbl_align_stats = QLabel("0 visual beats aligned.")
        self.lbl_align_stats.setStyleSheet("color: #38BDF8; font-size: 11px; font-weight: 600; margin-left: 10px;")
        align_row.addWidget(self.lbl_align_stats)

        align_row.addStretch()
        s2_lay.addLayout(align_row)

        main_layout.addWidget(step2_box)

        # ---------------------------------------------------------------------
        # STEP 3: STORY INTELLIGENCE & DYNAMIC BIBLES EXTRACTION
        # ---------------------------------------------------------------------
        step3_box = self._create_card("📖 STEP 3: Dynamic Story Bibles & Context Extraction")
        s3_lay = QVBoxLayout(step3_box)
        s3_lay.setSpacing(8)

        b_hdr = QHBoxLayout()
        self.btn_extract_bibles = QPushButton("🧠 Extract Story Bibles via Groq AI")
        self.btn_extract_bibles.setFixedHeight(34)
        self.btn_extract_bibles.setCursor(Qt.PointingHandCursor)
        self.btn_extract_bibles.setStyleSheet("""
            QPushButton {
                background-color: rgba(192, 132, 252, 0.18);
                color: #C084FC;
                font-weight: 700;
                font-size: 12px;
                border: 1px solid rgba(192, 132, 252, 0.5);
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background-color: rgba(192, 132, 252, 0.32);
            }
        """)
        self.btn_extract_bibles.clicked.connect(self._on_extract_bibles)
        b_hdr.addWidget(self.btn_extract_bibles)

        self.btn_paste_bibles = QPushButton("📋 Paste Custom Bibles (Text Format)")
        self.btn_paste_bibles.setFixedHeight(34)
        self.btn_paste_bibles.setCursor(Qt.PointingHandCursor)
        self.btn_paste_bibles.setStyleSheet("""
            QPushButton {
                background-color: rgba(56, 189, 248, 0.16);
                color: #38BDF8;
                font-weight: 700;
                font-size: 12px;
                border: 1px solid rgba(56, 189, 248, 0.45);
                border-radius: 6px;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: rgba(56, 189, 248, 0.28);
            }
        """)
        self.btn_paste_bibles.clicked.connect(self._on_paste_custom_bibles)
        b_hdr.addWidget(self.btn_paste_bibles)

        self.lbl_bible_status = QLabel("Dynamic bibles not yet extracted.")
        self.lbl_bible_status.setStyleSheet("color: #94A3B8; font-size: 11px;")
        b_hdr.addWidget(self.lbl_bible_status)

        b_hdr.addStretch()

        self.btn_save_bibles = QPushButton("💾 Save Edited Bibles")
        self.btn_save_bibles.setFixedHeight(30)
        self.btn_save_bibles.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #F8FAFC;
                font-size: 11px;
                border: 1px solid #334155;
                border-radius: 4px;
                padding: 0 12px;
            }
            QPushButton:hover {
                border-color: #10B981;
            }
        """)
        self.btn_save_bibles.clicked.connect(self._on_save_bibles_clicked)
        b_hdr.addWidget(self.btn_save_bibles)
        s3_lay.addLayout(b_hdr)

        # Tabbed Bible Editors
        self.bible_tabs = QTabWidget()
        self.bible_tabs.setFixedHeight(180)
        self.bible_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: 1px solid #1E293B;
                background: #0B0F17;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #111827;
                color: #94A3B8;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: 600;
                border: 1px solid #1E293B;
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #1E293B;
                color: #A3FF12;
                border-color: #10B981;
            }
        """)

        self.txt_story_bible = QPlainTextEdit()
        self.txt_story_bible.setPlaceholderText("Story Logline, Genre & Tone, Narrative Arc, Timeline Events...")
        self.bible_tabs.addTab(self.txt_story_bible, "Story Bible")

        self.txt_char_bible = QPlainTextEdit()
        self.txt_char_bible.setPlaceholderText("Characters, Appearances, Clothing, Age, Consistency Rules...")
        self.bible_tabs.addTab(self.txt_char_bible, "Character Bible")

        self.txt_env_bible = QPlainTextEdit()
        self.txt_env_bible.setPlaceholderText("Locations, Architecture, Atmospheric Lighting, Decay Progression...")
        self.bible_tabs.addTab(self.txt_env_bible, "Environment Bible")

        self.txt_objects_bible = QPlainTextEdit()
        self.txt_objects_bible.setPlaceholderText("Key Objects, Tools, Weapons, Survival Props, Significance...")
        self.bible_tabs.addTab(self.txt_objects_bible, "Key Objects & Props")

        self.txt_style_bible = QPlainTextEdit()
        self.txt_style_bible.setPlaceholderText("Color Palette, Lighting Rules, Camera Guidelines, Continuity Motifs...")
        self.bible_tabs.addTab(self.txt_style_bible, "Style & Cinematography")

        s3_lay.addWidget(self.bible_tabs)
        main_layout.addWidget(step3_box)

        # ---------------------------------------------------------------------
        # STEP 4: VISUAL STYLE & SETTINGS
        # ---------------------------------------------------------------------
        step4_box = self._create_card("🎨 STEP 4: Visual Art Style Presets & Output Settings")
        s4_lay = QVBoxLayout(step4_box)
        s4_lay.setSpacing(10)

        # Visual Style Matrix
        self.style_matrix = StyleThumbnailMatrix(self)
        self.style_matrix.style_selected.connect(self._on_style_selected)
        s4_lay.addWidget(self.style_matrix)

        # Style Directives & AI Master Prompt Bar
        style_prompt_bar = QHBoxLayout()
        style_prompt_bar.setSpacing(10)

        self.btn_copy_style_prompt = QPushButton("📋 Copy Master AI Prompt for External AI")
        self.btn_copy_style_prompt.setFixedHeight(32)
        self.btn_copy_style_prompt.setCursor(Qt.PointingHandCursor)
        self.btn_copy_style_prompt.setStyleSheet("""
            QPushButton {
                background-color: rgba(16, 185, 129, 0.15);
                color: #10B981;
                font-weight: 700;
                font-size: 11px;
                border: 1px solid rgba(16, 185, 129, 0.4);
                border-radius: 6px;
                padding: 0 14px;
            }
            QPushButton:hover {
                background-color: rgba(16, 185, 129, 0.28);
                border-color: #10B981;
            }
        """)
        self.btn_copy_style_prompt.clicked.connect(self._on_copy_style_prompt)
        style_prompt_bar.addWidget(self.btn_copy_style_prompt)

        self.btn_view_style_prompt = QPushButton("👁️ View Full Style Prompt & Directives")
        self.btn_view_style_prompt.setFixedHeight(32)
        self.btn_view_style_prompt.setCursor(Qt.PointingHandCursor)
        self.btn_view_style_prompt.setStyleSheet("""
            QPushButton {
                background-color: #1E293B;
                color: #94A3B8;
                font-size: 11px;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 0 12px;
            }
            QPushButton:hover {
                color: #F8FAFC;
                border-color: #38BDF8;
            }
        """)
        self.btn_view_style_prompt.clicked.connect(self._on_view_style_prompt)
        style_prompt_bar.addWidget(self.btn_view_style_prompt)

        self.lbl_prompt_copy_status = QLabel("")
        self.lbl_prompt_copy_status.setStyleSheet("color: #10B981; font-size: 11px; font-weight: 600;")
        style_prompt_bar.addWidget(self.lbl_prompt_copy_status)

        style_prompt_bar.addStretch()
        s4_lay.addLayout(style_prompt_bar)

        # Style Features Highlight Box
        self.lbl_style_feature_banner = QLabel()
        self.lbl_style_feature_banner.setWordWrap(True)
        self.lbl_style_feature_banner.setStyleSheet("""
            QLabel {
                background-color: #0B0F17;
                color: #38BDF8;
                font-size: 11px;
                font-weight: 500;
                border: 1px solid #1E293B;
                border-left: 3px solid #10B981;
                border-radius: 4px;
                padding: 8px 12px;
            }
        """)
        s4_lay.addWidget(self.lbl_style_feature_banner)

        # Initialize style features banner for default selection
        self._on_style_selected(self.style_matrix.current_style_id())

        # Options Row: Aspect Ratio, Model, Batch Size
        opt_row = QHBoxLayout()
        opt_row.setSpacing(12)

        lbl_ar = QLabel("Aspect Ratio:")
        lbl_ar.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 600;")
        opt_row.addWidget(lbl_ar)

        self.combo_aspect = QComboBox()
        self.combo_aspect.setFixedHeight(28)
        self.combo_aspect.addItems(["16:9", "9:16", "1:1"])
        opt_row.addWidget(self.combo_aspect)

        lbl_model = QLabel("Groq Model:")
        lbl_model.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 600; margin-left: 10px;")
        opt_row.addWidget(lbl_model)

        self.combo_model = QComboBox()
        self.combo_model.setFixedHeight(28)
        self.combo_model.addItems([
            "qwen/qwen3.8-27b",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b"
        ])
        opt_row.addWidget(self.combo_model)

        lbl_batch = QLabel("Batch Size:")
        lbl_batch.setStyleSheet("color: #94A3B8; font-size: 11px; font-weight: 600; margin-left: 10px;")
        opt_row.addWidget(lbl_batch)

        self.spin_batch = QSpinBox()
        self.spin_batch.setFixedHeight(28)
        self.spin_batch.setRange(1, 25)
        self.spin_batch.setValue(10)
        opt_row.addWidget(self.spin_batch)

        opt_row.addStretch()
        s4_lay.addLayout(opt_row)

        main_layout.addWidget(step4_box)

        # ---------------------------------------------------------------------
        # STEP 5: MASTER PROMPT EXPANSION VIA GROQ
        # ---------------------------------------------------------------------
        step5_box = self._create_card("🚀 STEP 5: Groq Image Prompt Generation & Expansion")
        s5_lay = QVBoxLayout(step5_box)
        s5_lay.setSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self.btn_generate_prompts = QPushButton("➔ Generate All Detailed Image Prompts with Groq")
        self.btn_generate_prompts.setFixedHeight(44)
        self.btn_generate_prompts.setCursor(Qt.PointingHandCursor)
        self.btn_generate_prompts.setStyleSheet("""
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:0.6 #34D399, stop:1 #A3FF12);
                color: #080D0B;
                font-weight: 800;
                font-size: 13px;
                border-radius: 8px;
                border: none;
                padding: 0 20px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #34D399, stop:0.6 #A3FF12, stop:1 #B8FF24);
            }
            QPushButton:disabled {
                background: #1E293B;
                color: #64748B;
            }
        """)
        self.btn_generate_prompts.clicked.connect(self._on_start_prompt_generation)
        btn_row.addWidget(self.btn_generate_prompts, 2)

        self.btn_stop_prompts = QPushButton("■ Stop")
        self.btn_stop_prompts.setFixedHeight(44)
        self.btn_stop_prompts.setEnabled(False)
        self.btn_stop_prompts.setCursor(Qt.PointingHandCursor)
        self.btn_stop_prompts.setStyleSheet("""
            QPushButton {
                background: rgba(239, 68, 68, 0.16);
                color: #EF4444;
                font-weight: 700;
                font-size: 13px;
                border: 1px solid rgba(239, 68, 68, 0.45);
                border-radius: 8px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: rgba(239, 68, 68, 0.32);
                color: #FFFFFF;
            }
        """)
        self.btn_stop_prompts.clicked.connect(self._on_stop_prompt_generation)
        btn_row.addWidget(self.btn_stop_prompts, 1)

        s5_lay.addLayout(btn_row)

        # Progress bar & status
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(12)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #0B0F17;
                border: 1px solid #1E293B;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #10B981, stop:1 #A3FF12);
                border-radius: 5px;
            }
        """)
        s5_lay.addWidget(self.progress_bar)

        self.lbl_progress_status = QLabel("Ready for prompt generation.")
        self.lbl_progress_status.setStyleSheet("color: #94A3B8; font-size: 11px;")
        s5_lay.addWidget(self.lbl_progress_status)

        # Navigation action to Studio
        nav_row = QHBoxLayout()
        self.btn_goto_studio = QPushButton("➔ Open Studio to Review & Edit Prompts (Stage 02)")
        self.btn_goto_studio.setFixedHeight(38)
        self.btn_goto_studio.setCursor(Qt.PointingHandCursor)
        self.btn_goto_studio.setStyleSheet("""
            QPushButton {
                background: rgba(163, 255, 18, 0.08);
                color: #A3FF12;
                font-size: 12px;
                font-weight: 700;
                border: 1px solid rgba(163, 255, 18, 0.35);
                border-radius: 6px;
                padding: 0 16px;
            }
            QPushButton:hover {
                background: rgba(163, 255, 18, 0.2);
            }
        """)
        self.btn_goto_studio.clicked.connect(lambda: self.navigate_requested.emit("studio"))
        nav_row.addWidget(self.btn_goto_studio)
        s5_lay.addLayout(nav_row)

        main_layout.addWidget(step5_box)

    def _create_card(self, title: str) -> QFrame:
        card = QFrame()
        card.setStyleSheet("""
            QFrame {
                background-color: #111827;
                border: 1px solid #1F2937;
                border-radius: 10px;
                padding: 12px;
            }
        """)
        # Card header title
        t_lbl = QLabel(title, card)
        t_lbl.setStyleSheet("color: #F8FAFC; font-size: 13px; font-weight: 700; border: none; background: transparent;")
        return card

    # =========================================================================
    # STEP 1 ACTIONS: SCRIPT JSON & AUDIO IMPORT
    # =========================================================================

    def _on_browse_json(self):
        start_dir = str(Path("audio and files").resolve()) if Path("audio and files").exists() else "."
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Script / Speech JSON File",
            start_dir,
            "JSON Files (*.json);;All Files (*.*)"
        )
        if not path:
            return
        self.load_json_file(path)

    def load_json_file(self, path: str):
        res = import_script_json(path)
        if not res.success:
            QMessageBox.critical(self, "JSON Import Error", res.message)
            return

        self.json_path = path
        self.story_text = res.story_text
        self.narration = res.narration

        dur_text = f"{res.narration.duration_seconds:.1f}s" if res.narration else "N/A"
        word_count = len(res.narration.words) if res.narration and res.narration.words else len(res.story_text.split())
        self.lbl_json_status.setText(f"✅ JSON: {Path(path).name} ({word_count} words · {dur_text})")
        self.lbl_json_status.setStyleSheet("color: #34D399; font-size: 11px; font-weight: 600;")

        # Auto-detect audio file in the same directory
        parent_dir = Path(path).parent
        for ext in [".mp3", ".wav", ".m4a"]:
            cand = parent_dir / f"{Path(path).stem}{ext}"
            if cand.exists():
                self.set_audio_file(str(cand))
                break
            # Or general speech.mp3
            speech_cand = parent_dir / f"speech{ext}"
            if speech_cand.exists():
                self.set_audio_file(str(speech_cand))
                break

        self.story_script_imported.emit(self.story_text, self.narration)

    def _on_browse_audio(self):
        start_dir = str(Path("audio and files").resolve()) if Path("audio and files").exists() else "."
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voiceover Audio File",
            start_dir,
            "Audio Files (*.mp3 *.wav *.m4a *.aac);;All Files (*.*)"
        )
        if path:
            self.set_audio_file(path)

    def set_audio_file(self, path: str):
        p = Path(path)
        if p.exists():
            self.audio_path = str(p.resolve())
            self.lbl_audio_status.setText(f"✅ Audio: {p.name}")
            self.lbl_audio_status.setStyleSheet("color: #38BDF8; font-size: 11px; font-weight: 600;")

            # Auto-catch matching JSON file in the same directory if not already loaded
            if not self.json_path or not Path(self.json_path).exists():
                parent_dir = p.parent
                candidates = [
                    parent_dir / f"{p.stem}.json",
                    parent_dir / "speech.json",
                    parent_dir / "narration.json",
                    parent_dir / "transcript.json"
                ]
                for cand in candidates:
                    if cand.exists() and cand.is_file():
                        self.load_json_file(str(cand.resolve()))
                        break

    # =========================================================================
    # STEP 2 ACTIONS: PARSE & ALIGN VISUAL IDEAS
    # =========================================================================

    def _on_align_visuals(self):
        raw_text = self.txt_visual_ideas.toPlainText().strip()
        if not raw_text:
            # Check if sample data.txt exists in root and offer to load it
            sample_file = Path("sample data.txt")
            if sample_file.exists():
                raw_text = sample_file.read_text(encoding="utf-8")
                self.txt_visual_ideas.setPlainText(raw_text)

        if not raw_text:
            QMessageBox.warning(self, "No Visual Ideas", "Please paste narration and visual ideas in the text field.")
            return

        items = parse_visual_breakdown_text(raw_text)
        if not items:
            QMessageBox.warning(self, "Parse Warning", "No visual breakdown lines could be extracted.")
            return

        total_audio_dur = self.narration.duration_seconds if self.narration else None
        self.scenes = align_visuals_with_timestamps(
            visual_items=items,
            narration=self.narration,
            total_audio_duration=total_audio_dur
        )

        self.lbl_align_stats.setText(f"✅ {len(self.scenes)} Visual Beats Aligned & Synchronized!")
        self.lbl_align_stats.setStyleSheet("color: #A3FF12; font-size: 11px; font-weight: 700;")

        self.visual_scenes_aligned.emit(self.scenes)

    # =========================================================================
    # STEP 3 ACTIONS: STORY INTELLIGENCE & DYNAMIC BIBLES
    # =========================================================================

    def _on_extract_bibles(self):
        script_text = self.story_text or self.txt_visual_ideas.toPlainText()
        if not script_text.strip():
            QMessageBox.warning(self, "Missing Script", "Please import a script JSON or paste narration first.")
            return

        self.btn_extract_bibles.setEnabled(False)
        self.lbl_bible_status.setText("Extracting dynamic story context & bibles via Groq...")
        self.lbl_bible_status.setStyleSheet("color: #F59E0B; font-size: 11px;")

        ideas_summary = self.txt_visual_ideas.toPlainText()[:4000]
        model = self.combo_model.currentText()

        self._bible_worker = BibleExtractionWorker(
            story_text=script_text,
            visual_ideas_summary=ideas_summary,
            model=model,
            parent=self
        )
        self._bible_worker.extraction_finished.connect(self._on_bibles_extracted)
        self._bible_worker.extraction_failed.connect(self._on_bibles_failed)
        self._bible_worker.start()

    def _on_bibles_extracted(self, bibles_data: dict):
        self.btn_extract_bibles.setEnabled(True)
        self.bibles = bibles_data
        self.lbl_bible_status.setText(f"✅ Bibles extracted ({len(bibles_data.get('characters', []))} characters, {len(bibles_data.get('environments', []))} environments)")
        self.lbl_bible_status.setStyleSheet("color: #34D399; font-size: 11px; font-weight: 600;")

        # Populate tabs
        story_b = bibles_data.get("story_bible", {})
        self.txt_story_bible.setPlainText(json.dumps(story_b, indent=2))

        chars_b = bibles_data.get("characters", [])
        self.txt_char_bible.setPlainText(json.dumps(chars_b, indent=2))

        envs_b = bibles_data.get("environments", [])
        self.txt_env_bible.setPlainText(json.dumps(envs_b, indent=2))

        objs_b = bibles_data.get("key_objects", [])
        self.txt_objects_bible.setPlainText(json.dumps(objs_b, indent=2))

        style_b = bibles_data.get("visual_style_bible", {})
        self.txt_style_bible.setPlainText(json.dumps(style_b, indent=2))

        self.bibles_updated.emit(self.bibles)

    def _on_bibles_failed(self, error_msg: str):
        self.btn_extract_bibles.setEnabled(True)
        self.lbl_bible_status.setText(f"❌ Extraction error: {error_msg[:60]}")
        self.lbl_bible_status.setStyleSheet("color: #EF4444; font-size: 11px;")
        QMessageBox.critical(self, "Bible Extraction Failed", f"Groq Bible extraction failed:\n{error_msg}")

    def _on_save_bibles_clicked(self):
        try:
            self.bibles["story_bible"] = json.loads(self.txt_story_bible.toPlainText() or "{}")
            self.bibles["characters"] = json.loads(self.txt_char_bible.toPlainText() or "[]")
            self.bibles["environments"] = json.loads(self.txt_env_bible.toPlainText() or "[]")
            self.bibles["key_objects"] = json.loads(self.txt_objects_bible.toPlainText() or "[]")
            self.bibles["visual_style_bible"] = json.loads(self.txt_style_bible.toPlainText() or "{}")
            self.lbl_bible_status.setText("✅ Edited bibles saved successfully!")
            self.lbl_bible_status.setStyleSheet("color: #A3FF12; font-size: 11px; font-weight: 600;")
            self.bibles_updated.emit(self.bibles)
        except Exception as e:
            QMessageBox.warning(self, "JSON Format Warning", f"Could not parse edited bibles as JSON:\n{e}")

    def _on_paste_custom_bibles(self):
        """Opens a dialog for the user to paste human-readable or structured production bibles."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Paste Custom Production Bibles (Character, Environment, Props, Style)")
        dlg.resize(750, 550)
        dlg.setStyleSheet("""
            QDialog {
                background-color: #0F172A;
                color: #F8FAFC;
            }
            QPlainTextEdit {
                background-color: #0B0F19;
                color: #E2E8F0;
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                border: 1px solid #334155;
                border-radius: 6px;
                padding: 8px;
            }
            QPushButton {
                background-color: #38BDF8;
                color: #0F172A;
                font-weight: 700;
                font-size: 12px;
                border-radius: 6px;
                padding: 6px 16px;
            }
            QPushButton:hover {
                background-color: #7DD3FC;
            }
        """)
        d_lay = QVBoxLayout(dlg)
        d_lay.setSpacing(10)

        lbl = QLabel(
            "<b>Paste your formatted Production Bibles below</b> (supports CHARACTER BIBLE, ENVIRONMENT BIBLE, "
            "OBJECT AND PROP BIBLE, CINEMATIC STYLE BIBLE, and CONTINUITY RULES):",
            dlg
        )
        lbl.setStyleSheet("color: #94A3B8; font-size: 12px;")
        d_lay.addWidget(lbl)

        txt_input = QPlainTextEdit(dlg)
        txt_input.setPlaceholderText(
            "1. CHARACTER BIBLE\nCHARACTER 1: THE PROTAGONIST\nRole: Main survivor\nAppearance: ...\n\n"
            "2. ENVIRONMENT BIBLE\nLOCATION 1: THE SHOPPING MALL\nArchitecture: ...\n\n"
            "3. OBJECT AND PROP BIBLE\nYellow notepad\nFlashlight\n..."
        )
        d_lay.addWidget(txt_input)

        btn_box = QHBoxLayout()
        btn_box.addStretch()

        btn_cancel = QPushButton("Cancel", dlg)
        btn_cancel.setStyleSheet("background-color: #334155; color: #F8FAFC;")
        btn_cancel.clicked.connect(dlg.reject)
        btn_box.addWidget(btn_cancel)

        btn_apply = QPushButton("Apply Bibles", dlg)
        btn_apply.clicked.connect(dlg.accept)
        btn_box.addWidget(btn_apply)

        d_lay.addLayout(btn_box)

        if dlg.exec() == QDialog.Accepted:
            raw_text = txt_input.toPlainText().strip()
            if not raw_text:
                return

            parsed = parse_custom_bible_text(raw_text)
            self._on_bibles_extracted(parsed)
            self.lbl_bible_status.setText(
                f"✅ Custom Bibles Loaded ({len(parsed.get('characters', []))} characters, "
                f"{len(parsed.get('environments', []))} locations, "
                f"{len(parsed.get('key_objects', []))} props)"
            )
            self.lbl_bible_status.setStyleSheet("color: #38BDF8; font-size: 11px; font-weight: 700;")

    # =========================================================================
    # STEP 5 ACTIONS: BATCH PROMPT GENERATION
    # =========================================================================

    def _on_start_prompt_generation(self):
        if not self.scenes:
            self._on_align_visuals()

        if not self.scenes:
            QMessageBox.warning(self, "No Scenes", "Please align visual ideas with timestamps before generating prompts.")
            return

        style_id = self.style_matrix.current_style_id()
        aspect_ratio = self.combo_aspect.currentText()
        model = self.combo_model.currentText()
        batch_size = self.spin_batch.value()

        # If bibles not yet extracted, construct a dynamic base
        if not self.bibles:
            self.bibles = {
                "story_bible": {"logline": self.story_text[:160]},
                "characters": [{"name": "Protagonist", "physical_description": "Survivor", "clothing_and_gear": "Standard survival gear"}],
                "environments": [{"name": "Primary Setting", "architectural_style": "Story environment"}],
                "visual_style_bible": {"color_palette": "Consistent with selected style"}
            }

        self.btn_generate_prompts.setEnabled(False)
        self.btn_stop_prompts.setEnabled(True)
        self.progress_bar.setValue(0)
        self.lbl_progress_status.setText("Starting Groq prompt expansion worker...")

        self._prompt_worker = BatchPromptGenerationWorker(
            scenes=self.scenes,
            bibles=self.bibles,
            story_context=self.story_text or "Story Narrative",
            visual_style_id=style_id,
            aspect_ratio=aspect_ratio,
            model=model,
            batch_size=batch_size,
            force_regenerate=False,
            parent=self
        )
        self._prompt_worker.progress_updated.connect(self._on_prompt_progress)
        self._prompt_worker.generation_finished.connect(self._on_prompt_generation_finished)
        self._prompt_worker.generation_failed.connect(self._on_prompt_generation_failed)
        self._prompt_worker.start()

    def _on_stop_prompt_generation(self):
        if self._prompt_worker:
            self._prompt_worker.cancel()
            self.lbl_progress_status.setText("Generation stopped by user.")
            self.btn_generate_prompts.setEnabled(True)
            self.btn_stop_prompts.setEnabled(False)

    def _on_prompt_progress(self, current: int, total: int, status_msg: str):
        pct = int((current / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_progress_status.setText(status_msg)

    def _on_prompt_generation_finished(self, updated_scenes: List[VisualScene]):
        self.scenes = updated_scenes
        self.btn_generate_prompts.setEnabled(True)
        self.btn_stop_prompts.setEnabled(False)
        self.progress_bar.setValue(100)
        self.lbl_progress_status.setText(f"✅ Generated {len(updated_scenes)} detailed image prompts matching selected style!")
        self.lbl_progress_status.setStyleSheet("color: #34D399; font-size: 11px; font-weight: 600;")
        self.prompts_generated.emit(self.scenes)

    def _on_prompt_generation_failed(self, error_msg: str):
        self.btn_generate_prompts.setEnabled(True)
        self.btn_stop_prompts.setEnabled(False)
        self.lbl_progress_status.setText(f"❌ Generation error: {error_msg[:60]}")
        self.lbl_progress_status.setStyleSheet("color: #EF4444; font-size: 11px;")
        QMessageBox.critical(self, "Prompt Generation Error", f"Prompt generation error:\n{error_msg}")

    def _get_prompt_path_for_style(self, style_id: str) -> Optional[Path]:
        """Resolves the dedicated Master AI Prompt file for a given style ID."""
        prompts_dir = Path(__file__).resolve().parent.parent.parent.parent / "config" / "ai_prompts"
        norm_id = style_id.lower().strip()
        filename = STYLE_PROMPT_MAP.get(norm_id, "01_zombie_survival_ai_prompt.txt")
        target_path = prompts_dir / filename
        if target_path.exists():
            return target_path
        return None

    def _on_style_selected(self, style_id: str):
        """Updates feature highlights banner and button text when user selects a visual style."""
        norm_id = style_id.lower().strip()
        features_text = STYLE_FEATURE_HIGHLIGHTS.get(
            norm_id,
            f"Active Style: {style_id}. Visual directives and constraints loaded from configuration."
        )
        if hasattr(self, "lbl_style_feature_banner"):
            self.lbl_style_feature_banner.setText(features_text)

        style_preset = PresetManager.get_style_by_id(norm_id)
        style_name = style_preset.name if style_preset else norm_id.replace("_", " ").title()

        if hasattr(self, "btn_copy_style_prompt"):
            self.btn_copy_style_prompt.setText(f"📋 Copy Master AI Prompt ({style_name})")
        if hasattr(self, "lbl_prompt_copy_status"):
            self.lbl_prompt_copy_status.setText("")

    def _on_copy_style_prompt(self):
        """Copies the dedicated Master AI Prompt for the active style to the system clipboard."""
        style_id = self.style_matrix.current_style_id()
        p = self._get_prompt_path_for_style(style_id)
        style_preset = PresetManager.get_style_by_id(style_id)
        style_name = style_preset.name if style_preset else style_id.replace("_", " ").title()

        if p and p.exists():
            try:
                content = p.read_text(encoding="utf-8")
                QApplication.clipboard().setText(content)
                if hasattr(self, "lbl_prompt_copy_status"):
                    self.lbl_prompt_copy_status.setText(f"✓ Copied {style_name} prompt to clipboard! Paste into ChatGPT/Claude.")
                    self.lbl_prompt_copy_status.setStyleSheet("color: #10B981; font-size: 11px; font-weight: 600;")
            except Exception as e:
                QMessageBox.warning(self, "Copy Error", f"Could not read prompt file:\n{e}")
        else:
            QMessageBox.warning(self, "Prompt Not Found", f"Prompt file for style '{style_name}' was not found at:\n{p}")

    def _on_view_style_prompt(self):
        """Opens a viewer dialog allowing inspection and full copy of the active style's Master AI Prompt."""
        style_id = self.style_matrix.current_style_id()
        p = self._get_prompt_path_for_style(style_id)
        style_preset = PresetManager.get_style_by_id(style_id)
        style_name = style_preset.name if style_preset else style_id.replace("_", " ").title()

        if p and p.exists():
            try:
                content = p.read_text(encoding="utf-8")
                dialog = PromptViewerDialog(style_name=style_name, prompt_text=content, parent=self)
                dialog.exec()
            except Exception as e:
                QMessageBox.warning(self, "Viewer Error", f"Could not open prompt viewer:\n{e}")
        else:
            QMessageBox.warning(self, "Prompt Not Found", f"Prompt file for style '{style_name}' was not found at:\n{p}")
