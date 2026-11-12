"""
New 3-File Narration Workflow Panel for PromptBeast-V2.
The user provides ONLY 3 files:
1. Script (.txt)
2. Audio (.mp3 / .wav)
3. Timestamp JSON (.json)

Executes Stages 0 to 5:
- Step 0: Load & validate (fails loudly, audits word coverage and audio duration)
- Step 1: Deterministic scene splitter (writes 01_chunks.json)
- Step 2: Story bibles (Groq, writes 02_raw_entities.json & 03_master_bible.json)
- Step 3: Storyboard beats (Groq, writes 04_storyboard_beats.json)
- Step 4: Image prompts (Groq, 180-220 words, writes 05_final_prompts.json)
- Step 5: Prompt export (writes prompts/prompt_XXX.txt, prompts_only.txt, prompts_index.json)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app.core.bible import generate_story_bible
from app.core.exporter import export_prompts
from app.core.groq_engine import GroqEngine
from app.core.loader import LoadedInputs, load_inputs
from app.core.project import ProjectCheckpointManager
from app.core.prompt_generator import generate_image_prompts
from app.core.splitter import split_scenes
from app.core.storyboard import plan_storyboard_beats
from app.core.validator import ValidationError


class PipelineExecutionWorker(QThread):
    """Background worker executing pipeline stages 0 through 5."""
    step_progress = Signal(int, str)
    stage_finished = Signal(str, object)
    pipeline_completed = Signal(dict)
    pipeline_failed = Signal(str)

    def __init__(
        self,
        script_path: str,
        audio_path: str,
        timestamp_path: str,
        project_dir: str,
        style_id: str,
        model_name: Optional[str] = None,
        parent=None
    ):
        super().__init__(parent)
        self.script_path = script_path
        self.audio_path = audio_path
        self.timestamp_path = timestamp_path
        self.project_dir = Path(project_dir).resolve()
        self.style_id = style_id
        self.model_name = model_name

    def run(self):
        try:
            dirs = ProjectCheckpointManager.get_project_dirs(self.project_dir)
            engine = GroqEngine(default_model=self.model_name or "openai/gpt-oss-120b")

            # Check existing checkpoints to support resume-on-crash
            cp_05 = ProjectCheckpointManager.load_checkpoint(self.project_dir, "05_final_prompts.json")
            cp_04 = ProjectCheckpointManager.load_checkpoint(self.project_dir, "04_storyboard_beats.json")
            cp_03 = ProjectCheckpointManager.load_checkpoint(self.project_dir, "03_master_bible.json")
            cp_01 = ProjectCheckpointManager.load_checkpoint(self.project_dir, "01_chunks.json")

            # Step 0: Load & Validate (always verified to guarantee input integrity)
            self.step_progress.emit(5, "Step 0: Validating script, audio, and word timestamps...")
            inputs = load_inputs(self.script_path, self.audio_path, self.timestamp_path)
            self.step_progress.emit(15, f"Step 0 Passed: {inputs.total_words} words validated, audio duration {inputs.audio_duration:.2f}s.")

            # Step 1: Split scenes
            if cp_01:
                self.step_progress.emit(25, "Step 1: Loaded existing scene chunks from 01_chunks.json.")
                scenes = cp_01
            else:
                self.step_progress.emit(20, "Step 1: Deterministically splitting scenes (2s-5s)...")
                scenes = split_scenes(inputs.words)
                ProjectCheckpointManager.save_checkpoint(self.project_dir, "01_chunks.json", scenes)
                self.step_progress.emit(30, f"Step 1 Complete: {len(scenes)} visual scenes generated.")
            self.stage_finished.emit("scenes", scenes)

            # Step 2: Story Bibles
            if cp_03:
                self.step_progress.emit(45, "Step 2: Loaded locked Master Story Bible from 03_master_bible.json.")
                master_bible = cp_03
            else:
                self.step_progress.emit(35, "Step 2: Extracting entities and merging Master Story Bible...")
                raw_entities_p = self.project_dir / "02_raw_entities.json"
                master_bible_p = self.project_dir / "03_master_bible.json"
                _, master_bible = generate_story_bible(
                    scenes,
                    engine,
                    model=self.model_name,
                    raw_entities_path=raw_entities_p,
                    master_bible_path=master_bible_p
                )
                self.step_progress.emit(50, f"Step 2 Complete: Master Bible locked ({len(master_bible.get('characters', []))} chars, {len(master_bible.get('locations', []))} locs).")
            self.stage_finished.emit("bible", master_bible)

            # Step 3: Storyboard Beats
            if cp_04:
                self.step_progress.emit(65, "Step 3: Loaded storyboard beats from 04_storyboard_beats.json.")
                beats_data = cp_04
                beats = beats_data.get("beats", beats_data) if isinstance(beats_data, dict) else beats_data
            else:
                self.step_progress.emit(55, "Step 3: Planning storyboard visual beats with Groq...")
                beats_cp = self.project_dir / "04_storyboard_beats.json"
                from app.core.style_manager import StyleManager
                style_profile = StyleManager.get_style_profile(self.style_id)
                style_anchor = style_profile.style_prompt if style_profile else "Cinematic visual style"
                beats, warnings = plan_storyboard_beats(
                    scenes,
                    master_bible,
                    style_anchor,
                    engine,
                    model=self.model_name,
                    checkpoint_path=beats_cp
                )
                self.step_progress.emit(70, f"Step 3 Complete: {len(beats)} beats planned.")
            self.stage_finished.emit("beats", beats)

            # Step 4: Image Prompts
            if cp_05:
                self.step_progress.emit(85, "Step 4: Loaded final prompts from 05_final_prompts.json.")
                prompts = cp_05
            else:
                self.step_progress.emit(75, "Step 4: Generating 180-220 word image prompts with Groq...")
                prompts_cp = self.project_dir / "05_final_prompts.json"
                prompts, still_failed = generate_image_prompts(
                    beats,
                    master_bible,
                    self.style_id,
                    engine,
                    model=self.model_name,
                    checkpoint_path=prompts_cp
                )
                self.step_progress.emit(88, f"Step 4 Complete: {len(prompts)} prompts generated ({len(still_failed)} failed).")
            self.stage_finished.emit("prompts", prompts)

            # Step 5: Export Prompts
            self.step_progress.emit(92, "Step 5: Exporting prompts files (prompts_only.txt, prompts_index.json)...")
            index_map = export_prompts(prompts, dirs["prompts"])

            self.step_progress.emit(100, f"Pipeline complete! {len(prompts)} prompts ready in {dirs['prompts']}.")
            self.pipeline_completed.emit({
                "inputs": inputs,
                "scenes": scenes,
                "master_bible": master_bible,
                "beats": beats,
                "prompts": prompts,
                "index_map": index_map,
                "prompts_dir": dirs["prompts"],
                "images_dir": dirs["images"],
                "output_dir": dirs["output"]
            })
        except Exception as e:
            self.pipeline_failed.emit(str(e))


class NarrationWorkflowPanel(QWidget):
    """Studio Workflow Panel implementing the strict 3-file user input flow."""
    pipeline_succeeded = Signal(dict)
    navigate_to_images = Signal()

    def __init__(self, parent=None, active_style_id: str = "default"):
        super().__init__(parent)
        self.active_style_id = active_style_id
        self.active_project_dir: Path = Path("data/project").resolve()
        self.worker: Optional[PipelineExecutionWorker] = None
        self._init_ui()

    def set_active_style(self, style_id: str):
        self.active_style_id = style_id

    def set_project_dir(self, p_dir: Path):
        self.active_project_dir = Path(p_dir).resolve()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(16, 16, 16, 16)

        # Title / Description Banner
        header = QFrame()
        header.setStyleSheet("background: #18181b; border-radius: 8px; padding: 12px; border: 1px solid #27272a;")
        h_layout = QVBoxLayout(header)
        h_title = QLabel("🎬 Production Pipeline — 3 Input Files Only")
        h_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #f4f4f5;")
        h_desc = QLabel(
            "Provide your Script (.txt), Narration Audio (.mp3/.wav), and Word Timestamp JSON. "
            "PromptBeast deterministically splits scenes, extracts the story bible, plans storyboard beats, "
            "and generates 180–220 word image prompts with Groq."
        )
        h_desc.setStyleSheet("color: #a1a1aa; font-size: 12px;")
        h_desc.setWordWrap(True)
        h_layout.addWidget(h_title)
        h_layout.addWidget(h_desc)
        layout.addWidget(header)

        # 3 Files Input Box
        files_group = QGroupBox("1. Project Input Files")
        files_group.setStyleSheet("QGroupBox { font-weight: bold; color: #38bdf8; }")
        fg_layout = QVBoxLayout(files_group)
        fg_layout.setSpacing(8)

        # Script File Row
        s_row = QHBoxLayout()
        s_lbl = QLabel("Script (.txt):")
        s_lbl.setFixedWidth(120)
        self.txt_script_path = QLineEdit()
        self.txt_script_path.setPlaceholderText("Path to narration script (.txt, UTF-8)...")
        btn_browse_script = QPushButton("Browse...")
        btn_browse_script.clicked.connect(self._browse_script)
        s_row.addWidget(s_lbl)
        s_row.addWidget(self.txt_script_path, 1)
        s_row.addWidget(btn_browse_script)
        fg_layout.addLayout(s_row)

        # Audio File Row
        a_row = QHBoxLayout()
        a_lbl = QLabel("Audio (.mp3/.wav):")
        a_lbl.setFixedWidth(120)
        self.txt_audio_path = QLineEdit()
        self.txt_audio_path.setPlaceholderText("Path to narration audio (.mp3, .wav)...")
        btn_browse_audio = QPushButton("Browse...")
        btn_browse_audio.clicked.connect(self._browse_audio)
        a_row.addWidget(a_lbl)
        a_row.addWidget(self.txt_audio_path, 1)
        a_row.addWidget(btn_browse_audio)
        fg_layout.addLayout(a_row)

        # Timestamps File Row
        t_row = QHBoxLayout()
        t_lbl = QLabel("Timestamps (.json):")
        t_lbl.setFixedWidth(120)
        self.txt_ts_path = QLineEdit()
        self.txt_ts_path.setPlaceholderText("Path to word-level timestamps JSON...")
        btn_browse_ts = QPushButton("Browse...")
        btn_browse_ts.clicked.connect(self._browse_ts)
        t_row.addWidget(t_lbl)
        t_row.addWidget(self.txt_ts_path, 1)
        t_row.addWidget(btn_browse_ts)
        fg_layout.addLayout(t_row)

        # Auto-load defaults if available in workspace
        self._populate_default_paths()

        layout.addWidget(files_group)

        # Action Buttons
        act_row = QHBoxLayout()
        self.btn_run_pipeline = QPushButton("🚀 Run Pipeline (Steps 0–5)")
        self.btn_run_pipeline.setStyleSheet("""
            QPushButton {
                background: #2563eb;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                border-radius: 6px;
            }
            QPushButton:hover { background: #1d4ed8; }
            QPushButton:disabled { background: #3f3f46; color: #71717a; }
        """)
        self.btn_run_pipeline.clicked.connect(self._on_run_pipeline)
        act_row.addWidget(self.btn_run_pipeline)

        layout.addLayout(act_row)

        # Progress Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 4px;
                text-align: center;
                color: #f4f4f5;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # Activity Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setStyleSheet("background: #09090b; color: #e4e4e7; font-family: monospace; font-size: 11px; border: 1px solid #27272a;")
        self.log_box.setPlaceholderText("Pipeline execution log will appear here...")
        layout.addWidget(self.log_box, 1)

    def _populate_default_paths(self):
        """Auto-populates paths if sample/test files exist."""
        script_cand = Path("audio and files/script only.txt")
        audio_cand = Path("audio and files/speech.mp3")
        ts_cand = Path("audio and files/speech.json")

        if script_cand.exists():
            self.txt_script_path.setText(str(script_cand.resolve()))
        if audio_cand.exists():
            self.txt_audio_path.setText(str(audio_cand.resolve()))
        if ts_cand.exists():
            self.txt_ts_path.setText(str(ts_cand.resolve()))

    def _browse_script(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Script File", "", "Text Files (*.txt);;All Files (*)")
        if f:
            self.txt_script_path.setText(f)

    def _browse_audio(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Narration Audio", "", "Audio Files (*.mp3 *.wav);;All Files (*)")
        if f:
            self.txt_audio_path.setText(f)

    def _browse_ts(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Word Timestamp JSON", "", "JSON Files (*.json);;All Files (*)")
        if f:
            self.txt_ts_path.setText(f)

    def _on_run_pipeline(self):
        script_p = self.txt_script_path.text().strip()
        audio_p = self.txt_audio_path.text().strip()
        ts_p = self.txt_ts_path.text().strip()

        if not script_p or not audio_p or not ts_p:
            QMessageBox.warning(self, "Missing Files", "Please provide all 3 input files: Script (.txt), Audio (.mp3/.wav), and Timestamp JSON.")
            return

        self.btn_run_pipeline.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_box.clear()
        self._append_log("Starting PromptBeast-V2 7-stage production pipeline...")

        self.worker = PipelineExecutionWorker(
            script_path=script_p,
            audio_path=audio_p,
            timestamp_path=ts_p,
            project_dir=str(self.active_project_dir),
            style_id=self.active_style_id,
            parent=self
        )
        self.worker.step_progress.connect(self._on_worker_progress)
        self.worker.pipeline_completed.connect(self._on_pipeline_completed)
        self.worker.pipeline_failed.connect(self._on_pipeline_failed)
        self.worker.start()

    def _on_worker_progress(self, val: int, msg: str):
        self.progress_bar.setValue(val)
        self._append_log(msg)

    def _on_pipeline_completed(self, result: Dict[str, Any]):
        self.btn_run_pipeline.setEnabled(True)
        self.progress_bar.setValue(100)
        self._append_log("✅ Pipeline Steps 0–5 Completed Successfully.")
        self._append_log("Prompts exported. Switching to Images tab for Step 5 manual image drop...")
        self.pipeline_succeeded.emit(result)
        self.navigate_to_images.emit()

    def _on_pipeline_failed(self, error_msg: str):
        self.btn_run_pipeline.setEnabled(True)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #ef4444; }")
        self._append_log(f"❌ PIPELINE ERROR: {error_msg}")
        QMessageBox.critical(self, "Pipeline Failure", f"Pipeline failed with error:\n\n{error_msg}")

    def _append_log(self, text: str):
        self.log_box.append(text)
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
