"""
Visual Style & Reference Thumbnail Matrix for PromptBeast.
Presents the visual art style presets as rich, perfectly scaled visual cards.
Supports Zombie Survival, What If Flat Vector, Stickman, Century-Old Photo, etc.
"""

from pathlib import Path
from typing import Optional, List
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QSizePolicy,
    QScrollArea
)

from app.presets.loader import PresetManager


ALLOWED_STYLE_ORDER = [
    ("zombie_survival", "Zombie Survival", "2D Animation / Survival"),
    ("whatif_flat_vector", "What If Flat Vector", "2D Vector"),
    ("3d_red_blue_stickman", "3D Red & Blue Stickman", "3D Animation"),
    ("2d_stickman", "2D Stickman", "2D Animation"),
    ("3d_stickman", "3D Stickman", "3D Animation"),
    ("century_old_photograph", "Century-Old Photograph", "Historical Archival"),
    ("bw_vintage_historical_photography", "B&W Historical Photography", "Archival Monochrome"),
    ("paper_quilling", "Paper Quilling", "Papercraft"),
    ("mixed_media_collage", "Mixed-Media Collage", "Editorial Collage"),
    ("paper_cutout", "Paper-Cutout", "Cut-Paper Craft"),
    ("vintage_photograph", "Vintage Photograph", "Archival"),
]


class StyleThumbnailCard(QFrame):
    """Clickable responsive visual card representing a visual art style."""

    selected = Signal(str)

    def __init__(self, style_id: str, name: str, category: str, image_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.style_id = style_id
        self.style_name = name
        self.category = category
        self.image_path = image_path
        self._pixmap: Optional[QPixmap] = None
        if self.image_path and Path(self.image_path).exists():
            p = QPixmap(self.image_path)
            if not p.isNull():
                self._pixmap = p

        self._is_active = False
        self.setObjectName("StyleThumbnailCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.setFixedWidth(145)
        self.setFixedHeight(168)
        self._init_ui()

    def _init_ui(self):
        self.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Image thumbnail container box
        self.img_lbl = QLabel()
        self.img_lbl.setFixedHeight(108)
        self.img_lbl.setFixedWidth(131)
        self.img_lbl.setAlignment(Qt.AlignCenter)
        self.img_lbl.setStyleSheet("""
            background-color: #0B0E14;
            border-radius: 4px;
            padding: 2px;
        """)

        if self._pixmap:
            self.img_lbl.setPixmap(
                self._pixmap.scaled(131, 108, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
        else:
            self.img_lbl.setText("🎨\n" + self.category)
            self.img_lbl.setStyleSheet("background: #111A24; color: #64748B; font-size: 11px; border-radius: 4px;")

        layout.addWidget(self.img_lbl)

        # Style Name
        self.name_lbl = QLabel(self.style_name)
        self.name_lbl.setWordWrap(True)
        self.name_lbl.setAlignment(Qt.AlignCenter)
        self.name_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #CBD5E1; max-height: 38px;")
        layout.addWidget(self.name_lbl)

        self._update_style()

    def set_active(self, active: bool):
        self._is_active = active
        self._update_style()

    def _update_style(self):
        if self._is_active:
            self.setStyleSheet("""
                #StyleThumbnailCard {
                    background-color: rgba(20, 32, 24, 0.95);
                    border: 2px solid #A3FF12;
                    border-radius: 8px;
                }
            """)
            self.name_lbl.setStyleSheet("font-size: 11px; font-weight: 700; color: #A3FF12;")
        else:
            self.setStyleSheet("""
                #StyleThumbnailCard {
                    background-color: rgba(27, 41, 31, 0.5);
                    border: 1px solid rgba(163, 255, 18, 0.12);
                    border-radius: 8px;
                }
                #StyleThumbnailCard:hover {
                    border-color: rgba(163, 255, 18, 0.4);
                    background-color: rgba(40, 57, 43, 0.7);
                }
            """)
            self.name_lbl.setStyleSheet("font-size: 11px; font-weight: 600; color: #B1BDB3;")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.selected.emit(self.style_id)
        super().mousePressEvent(event)


class StyleThumbnailMatrix(QWidget):
    """Clean responsive horizontal slider displaying the curated visual style cards."""

    style_selected = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._selected_id: str = "zombie_survival"
        self._cards: List[StyleThumbnailCard] = []
        self._init_ui()

    def _init_ui(self):
        self.setMinimumWidth(0)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(6)

        # Header row
        header_row = QHBoxLayout()
        header_title = QLabel("🎨 Visual Art Style Presets")
        header_title.setStyleSheet("font-size: 13px; font-weight: 700; color: #F8FAFC;")

        self.active_style_badge = QLabel("Zombie Survival Active")
        self.active_style_badge.setStyleSheet("""
            font-size: 11px;
            font-weight: 600;
            color: #34D399;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid #10B981;
            border-radius: 6px;
            padding: 3px 8px;
        """)

        header_row.addWidget(header_title)
        header_row.addStretch()
        header_row.addWidget(self.active_style_badge)
        main_layout.addLayout(header_row)

        # Scroll Area for all style cards
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(186)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                background: #090D14;
                border: 1px solid #1F2836;
                border-radius: 8px;
            }
            QScrollBar:horizontal {
                height: 8px;
                background: #090D14;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal {
                background: #2D3A4F;
                border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #10B981;
            }
        """)

        self.cards_frame = QWidget()
        self.cards_row_layout = QHBoxLayout(self.cards_frame)
        self.cards_row_layout.setContentsMargins(8, 6, 8, 6)
        self.cards_row_layout.setSpacing(10)
        self.cards_row_layout.setAlignment(Qt.AlignLeft)

        self.scroll_area.setWidget(self.cards_frame)
        main_layout.addWidget(self.scroll_area)

        self._populate_styles()

    def _populate_styles(self):
        PresetManager.initialize()
        artworks_dir = Path(__file__).resolve().parent.parent.parent.parent / "assets" / "style_previews"

        for sid, default_name, default_cat in ALLOWED_STYLE_ORDER:
            candidates = [
                artworks_dir / f"{sid}.png",
                artworks_dir / f"{sid}.jpg",
            ]
            if sid == "century_old_photograph":
                candidates.append(artworks_dir / "vintage_photograph.png")

            img_p = None
            for c in candidates:
                if c.exists():
                    img_p = str(c.resolve())
                    break

            card = StyleThumbnailCard(
                style_id=sid,
                name=default_name,
                category=default_cat,
                image_path=img_p,
                parent=self.cards_frame
            )
            card.selected.connect(self.select_style)
            self._cards.append(card)
            self.cards_row_layout.addWidget(card)

        # Default selection: Zombie Survival if available, else first card
        default_sel = "zombie_survival" if any(c.style_id == "zombie_survival" for c in self._cards) else "century_old_photograph"
        self.select_style(default_sel)

    def select_style(self, style_id: str):
        sid = style_id.lower().strip()
        if sid in ("zombie", "zombies", "zombie_survival"):
            sid = "zombie_survival"
        elif sid in ("whatif", "what_if", "whatif_flat_vector", "flat_vector"):
            sid = "whatif_flat_vector"
        elif sid in ("vintage_photograph", "vintage_photo", "century_old_photographs"):
            sid = "century_old_photograph"
        elif sid in ("stick_figure", "stickman", "red_blue_stickman_3d", "2d_stick_figure"):
            sid = "3d_red_blue_stickman"
        elif sid in ("black_white_historical_photography", "bw_historical_photography"):
            sid = "bw_vintage_historical_photography"
        elif sid in ("quilling", "paper_quill", "paper_quilling_illustration"):
            sid = "paper_quilling"
        elif sid in ("collage", "mixed_media", "mixed_media_collage_illustration"):
            sid = "mixed_media_collage"
        elif sid in ("cutout", "paper_cut", "paper_cutout_illustration"):
            sid = "paper_cutout"

        self._selected_id = sid
        for c in self._cards:
            is_match = (c.style_id == sid)
            c.set_active(is_match)
            if is_match:
                self.active_style_badge.setText(f"★ {c.style_name} Active")

        self.style_selected.emit(sid)

    def current_style_id(self) -> str:
        return self._selected_id
