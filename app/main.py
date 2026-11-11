"""
PromptBeast Desktop Application Entry Point.
"""
import sys
import os
import argparse
import traceback
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from PySide6.QtCore import Qt, QCoreApplication
from PySide6.QtGui import QIcon, QFont
from PySide6.QtWidgets import QApplication
from app.ui.main_window import MainWindow
from app.presets.loader import PresetManager
from app.core.style_manager import StyleManager


def global_exception_handler(exctype, value, tb):
    """Global exception handler to prevent silent application crashes and log tracebacks."""
    err_msg = "".join(traceback.format_exception(exctype, value, tb))
    print(f"\n[CRITICAL UNCAUGHT EXCEPTION]\n{err_msg}", file=sys.stderr)
    try:
        data_dir = BASE_DIR / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "crash.log", "a", encoding="utf-8") as f:
            f.write(f"\n--- Crash Log: {exctype.__name__} ---\n{err_msg}\n")
    except Exception:
        pass


def main():
    sys.excepthook = global_exception_handler

    # Parse CLI Arguments for Per-Style Launchers & Standalone Tools
    parser = argparse.ArgumentParser(description="PromptBeast Production Studio")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="studio",
        choices=["studio", "scene_sync"],
        help="App execution mode: 'studio' (default) or 'scene_sync'"
    )
    parser.add_argument(
        "--style", "-s",
        type=str,
        default="default",
        help="Visual style identifier to pre-load (e.g. zombie_survival, what_if, ink, default)"
    )
    args, _ = parser.parse_known_args()

    # Configure High-DPI attributes
    QCoreApplication.setApplicationName("PromptBeast")
    QCoreApplication.setOrganizationName("PromptBeast Studio")
    QCoreApplication.setOrganizationDomain("promptbeast.ai")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    # Never terminate when a modal QMessageBox, file dialog, or preview window closes
    app.setQuitOnLastWindowClosed(True)

    # Set default modern font
    font = QFont("Segoe UI", 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)

    # Initialize Presets
    PresetManager.initialize()

    # Mode Dispatch: Standalone Scene Sync Tool vs. Style Studio
    if args.mode == "scene_sync":
        print("[PromptBeast] Launching Standalone Scene Sync Tool...", flush=True)
        from app.ui.scene_sync_window import SceneSyncWindow
        window = SceneSyncWindow()
        window.show()
    else:
        active_profile = StyleManager.set_active_style(args.style)
        print(f"[PromptBeast] Active Visual Style: {active_profile.name} ({active_profile.id})", flush=True)
        window = MainWindow(style_id=active_profile.id)
        window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
