# Copyright 2026 Maxime SAUVAYRE
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# "Commons Clause" License Condition v1.0
#
# The Software is provided to you by the Licensor under the License, as
# defined above, subject to the following condition.
#
# Without limiting other conditions in the License, the grant of rights under
# the License will not include, and the License does not grant to you, the
# right to Sell the Software.
#
# For purposes of the foregoing, "Sell" means practicing any or all of the
# rights granted to you under the License to provide to third parties, for a
# fee or other consideration (including without limitation fees for hosting
# or consulting/support services related to the Software), a product or
# service whose value derives, entirely or substantially, from the
# functionality of the Software. Any license notice or attribution required
# by the License must also include this Commons Clause License Condition
# notice.
#
# Software: picturesRC
# License: Apache License 2.0
# Licensor: Maxime SAUVAYRE

"""Fenêtre principale de picturesRC.

Parcours prévu :
    1. Ajouter un dossier (ou glisser-déposer) : l'analyse démarre en fond.
    2. Les photos à confiance élevée sont validées d'office (option).
    3. On ne revoit que les orange / rouges : flèches pour ajuster,
       tracé à la souris le long de l'horizon, Entrée pour valider.
    4. Si besoin, ajuster le cadre de rognage à la souris (bords et coins,
       Ctrl+glisser pour le déplacer, Maj+glisser pour en tracer un nouveau).
    5. Exporter : copies corrigées + fichier JSON de traçabilité.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLibraryInfo, QLocale, QSettings, QSize, Qt, QThreadPool, QTimer, QTranslator
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import __version__, i18n
from ..core.imageio import SUPPORTED_EXTENSIONS, is_supported, load_preview
from ..core.crop import pixel_crop_box, to_norm
from ..core.orientation import (
    LEFT,
    RIGHT,
    normalize_quarter,
    rotate_array_quarter,
    rotate_norm_box,
    rotate_subject,
)
from ..core.inplace import ORIGINALS_DIRNAME, InPlaceSettings
from ..core.pipeline import crops_to_subject, is_modified
from ..core.straighten import rotate_array
from ..core.subject import auto_box
from ..i18n import tr
from .viewer import ImageViewer, numpy_to_qimage
from .workers import THUMB_SIDE, AnalyzeTask, ExportTask, PhotoItem

PREVIEW_SIDE = 1600
PREVIEW_CACHE = 6
MAX_ANGLE = 15.0

CROP_KEYS = ("subject", "aspect", "max_area", "none")  # libellés : clés « crop.<mode> » des catalogues
LEVEL_COLORS = {"high": QColor(46, 160, 67), "medium": QColor(230, 150, 20), "low": QColor(210, 50, 50)}
LEVEL_RANK = {"low": 0, "medium": 1, "high": 2}


class MainWindow(QMainWindow):
    def __init__(self, settings: QSettings | None = None):
        super().__init__()
        self.setWindowTitle(tr("app.window_title", version=__version__))
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.settings = settings or QSettings("picturesRC", "picturesRC")
        self.items: list[PhotoItem] = []
        self.index_by_path: dict[Path, int] = {}
        self.current: int = -1
        self.preview_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        self.pool = QThreadPool.globalInstance()
        self.pool.setMaxThreadCount(max(2, (os.cpu_count() or 2) - 1))
        self._tasks: set = set()  # garde les QRunnable (et leurs signaux) en vie
        self._export_task: ExportTask | None = None
        self._export_unchanged = 0
        self._export_counts: dict[str, int] = {}
        self._updating = False
        # None = sous-dossier « originaux » du dossier de chaque photo
        self.originals_dir: Path | None = None

        self._build_ui()
        self._build_shortcuts()
        self._update_controls()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        tb = QToolBar(tr("toolbar.name"))
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.addToolBar(tb)

        act_folder = QAction(tr("toolbar.add_folder"), self)
        act_folder.setShortcut(QKeySequence("Ctrl+O"))
        act_folder.triggered.connect(self.add_folder_dialog)
        act_files = QAction(tr("toolbar.add_files"), self)
        act_files.setShortcut(QKeySequence("Ctrl+Shift+O"))
        act_files.triggered.connect(self.add_files_dialog)
        act_clear = QAction(tr("toolbar.clear"), self)
        act_clear.triggered.connect(self.clear_items)
        tb.addAction(act_folder)
        tb.addAction(act_files)
        tb.addAction(act_clear)
        tb.addSeparator()

        self.crop_combo = QComboBox()
        for key in CROP_KEYS:
            self.crop_combo.addItem(tr(f"crop.{key}"), key)
        saved_crop = self.settings.value("crop_mode", "subject")
        self.crop_combo.setCurrentIndex(max(0, self.crop_combo.findData(saved_crop)))
        self.crop_combo.currentIndexChanged.connect(self._on_crop_mode)
        tb.addWidget(self.crop_combo)

        self.auto_validate = QCheckBox(tr("toolbar.auto_validate"))
        self.auto_validate.setChecked(self.settings.value("auto_validate", True, type=bool))
        self.auto_validate.toggled.connect(lambda v: self.settings.setValue("auto_validate", v))
        tb.addWidget(self.auto_validate)
        tb.addSeparator()

        self.write_json = QCheckBox(tr("toolbar.write_json"))
        self.write_json.setToolTip(tr("toolbar.write_json.tip"))
        self.write_json.setChecked(self.settings.value("write_json", True, type=bool))
        self.write_json.toggled.connect(lambda v: self.settings.setValue("write_json", v))
        tb.addWidget(self.write_json)

        act_out = QAction(tr("toolbar.originals"), self)
        act_out.setToolTip(tr("toolbar.originals.tip", dirname=ORIGINALS_DIRNAME))
        act_out.triggered.connect(self.choose_originals_dir)
        tb.addAction(act_out)
        self.act_export = QAction(tr("toolbar.export"), self)
        self.act_export.setToolTip(tr("toolbar.export.tip"))
        self.act_export.setShortcut(QKeySequence("Ctrl+E"))
        self.act_export.triggered.connect(self.export_validated)
        tb.addAction(self.act_export)

        # Langue de l'interface (appliquée au prochain démarrage), calée à droite
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        tb.addWidget(spacer)
        self.lang_combo = QComboBox()
        self.lang_combo.setToolTip(tr("toolbar.language.tip"))
        self.lang_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self.lang_combo.addItem("🌐 " + tr("language.auto"), "")
        for code, name in i18n.available_languages().items():
            self.lang_combo.addItem(f"🌐 {name}", code)
        self.lang_combo.setCurrentIndex(max(0, self.lang_combo.findData(self.settings.value("language", ""))))
        self.lang_combo.currentIndexChanged.connect(self._on_language)
        self.lang_combo.setMinimumWidth(self.lang_combo.sizeHint().width() + 12)  # marge pour l'icône 🌐
        tb.addWidget(self.lang_combo)

        # Liste de miniatures
        self.list = QListWidget()
        self.list.setViewMode(QListView.ViewMode.IconMode)
        self.list.setFlow(QListView.Flow.LeftToRight)
        self.list.setWrapping(True)
        self.list.setResizeMode(QListView.ResizeMode.Adjust)
        self.list.setMovement(QListView.Movement.Static)
        self.list.setIconSize(QSize(THUMB_SIDE // 2 + 30, THUMB_SIDE // 2 + 30))
        self.list.setSpacing(6)
        self.list.setUniformItemSizes(True)
        self.list.setWordWrap(True)
        self.list.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # les flèches servent à l'angle
        self.list.currentRowChanged.connect(self.select)
        self.list.setMinimumWidth(300)

        # Aperçu
        self.viewer = ImageViewer()
        self.viewer.line_drawn.connect(self._on_line_drawn)
        self.viewer.crop_changed.connect(self._on_crop_changed)
        self.viewer.crop_points_state.connect(self._on_crop_points)
        self.viewer.set_crop_mode(self.crop_mode)

        # Barre de réglage
        self.spin = QDoubleSpinBox()
        self.spin.setRange(-45.0, 45.0)
        self.spin.setDecimals(2)
        self.spin.setSingleStep(0.05)
        self.spin.setSuffix(" °")
        self.spin.setKeyboardTracking(False)
        self.spin.valueChanged.connect(self._on_spin)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(int(-MAX_ANGLE * 10), int(MAX_ANGLE * 10))
        self.slider.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.slider.valueChanged.connect(self._on_slider)

        self.btn_left = QPushButton(tr("btn.rotate_left"))
        self.btn_left.setToolTip(tr("btn.rotate_left.tip"))
        self.btn_left.clicked.connect(lambda: self.rotate_quarter(LEFT))
        self.btn_right = QPushButton(tr("btn.rotate_right"))
        self.btn_right.setToolTip(tr("btn.rotate_right.tip"))
        self.btn_right.clicked.connect(lambda: self.rotate_quarter(RIGHT))
        self.btn_auto = QPushButton(tr("btn.auto"))
        self.btn_auto.clicked.connect(self.reset_auto)
        self.btn_zero = QPushButton(tr("btn.zero"))
        self.btn_zero.clicked.connect(lambda: self.set_angle(0.0))
        self.btn_crop = QPushButton(tr("btn.crop_auto"))
        self.btn_crop.setToolTip(tr("btn.crop_auto.tip"))
        self.btn_crop.clicked.connect(self.viewer.reset_crop)
        self.btn_orig = QPushButton(tr("btn.original"))
        self.btn_orig.setCheckable(True)
        self.btn_orig.toggled.connect(self.viewer.set_show_original)
        self.btn_validate = QPushButton(tr("btn.validate"))
        self.btn_validate.setDefault(True)
        self.btn_validate.clicked.connect(self.validate_and_next)
        self.info = QLabel()
        self.info.setMinimumWidth(260)

        controls = QHBoxLayout()
        controls.addWidget(self.btn_left)
        controls.addWidget(self.btn_right)
        controls.addWidget(QLabel(tr("label.rotation")))
        controls.addWidget(self.spin)
        controls.addWidget(self.slider, 1)
        for b in (self.btn_auto, self.btn_zero, self.btn_crop, self.btn_orig, self.btn_validate):
            controls.addWidget(b)

        hint = QLabel(tr("hint"))
        hint.setTextFormat(Qt.TextFormat.RichText)
        hint.setStyleSheet("color: gray;")

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.viewer, 1)
        rl.addLayout(controls)
        rl.addWidget(self.info)
        rl.addWidget(hint)

        splitter = QSplitter()
        splitter.addWidget(self.list)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([380, 1020])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    def _build_shortcuts(self) -> None:
        def sc(keys: str, fn) -> None:
            s = QShortcut(QKeySequence(keys), self)
            s.setContext(Qt.ShortcutContext.WindowShortcut)
            s.activated.connect(fn)

        sc("Left", lambda: self.nudge(-0.1))
        sc("Right", lambda: self.nudge(+0.1))
        sc("Shift+Left", lambda: self.nudge(-1.0))
        sc("Shift+Right", lambda: self.nudge(+1.0))
        sc("Ctrl+Left", lambda: self.rotate_quarter(LEFT))
        sc("Ctrl+Right", lambda: self.rotate_quarter(RIGHT))
        sc("Up", lambda: self.move_selection(-1))
        sc("Down", lambda: self.move_selection(+1))
        sc("Return", self.validate_and_next)
        sc("Enter", self.validate_and_next)
        sc("A", self.reset_auto)
        sc("R", lambda: self.set_angle(0.0))
        sc("O", self.btn_orig.toggle)
        sc("G", self.toggle_grid)
        sc("C", self.viewer.reset_crop)
        sc("Escape", self.viewer.cancel_crop_points)
        sc("Space", self.toggle_validated)

    # ------------------------------------------------------------ propriétés
    @property
    def crop_mode(self) -> str:
        return self.crop_combo.currentData()

    def _item(self) -> PhotoItem | None:
        return self.items[self.current] if 0 <= self.current < len(self.items) else None

    # ------------------------------------------------------------- ajout
    def add_folder_dialog(self) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        d = QFileDialog.getExistingDirectory(self, tr("dialog.choose_folder"), start)
        if d:
            self.settings.setValue("last_dir", d)
            self.add_paths([Path(d)])

    def add_files_dialog(self) -> None:
        start = self.settings.value("last_dir", str(Path.home()))
        pattern = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXTENSIONS))
        files, _ = QFileDialog.getOpenFileNames(self, tr("dialog.choose_photos"), start,
                                                tr("dialog.images_filter", pattern=pattern))
        if files:
            self.settings.setValue("last_dir", str(Path(files[0]).parent))
            self.add_paths([Path(f) for f in files])

    def add_paths(self, paths: list[Path]) -> None:
        new: list[Path] = []
        for p in paths:
            if p.is_dir():
                new.extend(sorted(c for c in p.iterdir() if c.is_file() and is_supported(c)))
            elif p.is_file() and is_supported(p):
                new.append(p)
        # Fichiers temporaires d'un enregistrement en cours ou interrompu : ignorés
        new = [p for p in new if p not in self.index_by_path and ".picturesrc-tmp" not in p.name]
        if not new:
            self.statusBar().showMessage(tr("status.no_new_images"), 4000)
            return

        for p in new:
            self.index_by_path[p] = len(self.items)
            self.items.append(PhotoItem(p))
            li = QListWidgetItem(self._placeholder_icon(), f"{p.name}\n{tr('list.analyzing')}")
            li.setSizeHint(QSize(THUMB_SIDE // 2 + 50, THUMB_SIDE // 2 + 70))
            li.setToolTip(str(p))
            self.list.addItem(li)
            task = AnalyzeTask(p, MAX_ANGLE)
            task.signals.done.connect(self._on_analyzed)
            task.signals.failed.connect(self._on_analyze_failed)
            self._tasks.add(task)
            self.pool.start(task)

        if self.current < 0:
            self.list.setCurrentRow(0)
        self._update_status()

    def clear_items(self) -> None:
        if self._export_task is not None:
            return
        self.pool.clear()
        self.items.clear()
        self.index_by_path.clear()
        self.preview_cache.clear()
        self._tasks.clear()
        self.current = -1
        self.list.clear()
        self.viewer.set_source(None)
        self._update_controls()
        self._update_status()

    # ------------------------------------------------------ retours d'analyse
    def _on_analyzed(self, path: Path, analysis, thumb: np.ndarray) -> None:
        self._forget_task(path)
        idx = self.index_by_path.get(path)
        if idx is None:
            return
        it = self.items[idx]
        it.status, it.analysis, it.thumb_src, it.subject = "ready", analysis, thumb, analysis.subject
        it.estimate = analysis.estimate_for(self.crop_mode)
        it.angle = it.estimate.angle
        if self.auto_validate.isChecked() and self._level(it) == "high":
            it.validated = True
        self._refresh_list_item(idx)
        if idx == self.current:
            self._load_current()
        self._update_status()

    def _on_analyze_failed(self, path: Path, msg: str) -> None:
        self._forget_task(path)
        idx = self.index_by_path.get(path)
        if idx is None:
            return
        it = self.items[idx]
        it.status, it.error = "error", msg
        self._refresh_list_item(idx)
        self._update_status()

    def _forget_task(self, path: Path) -> None:
        self._tasks = {t for t in self._tasks if getattr(t, "path", None) != path}

    # ------------------------------------------------------------ sélection
    def select(self, row: int) -> None:
        self.current = row
        self.btn_orig.setChecked(False)
        self._load_current()

    def move_selection(self, step: int) -> None:
        if self.items:
            self.list.setCurrentRow(max(0, min(len(self.items) - 1, self.current + step)))

    def _get_preview(self, path: Path) -> np.ndarray:
        if path in self.preview_cache:
            self.preview_cache.move_to_end(path)
            return self.preview_cache[path]
        arr = load_preview(path, PREVIEW_SIDE)
        self.preview_cache[path] = arr
        while len(self.preview_cache) > PREVIEW_CACHE:
            self.preview_cache.popitem(last=False)
        return arr

    def _load_current(self) -> None:
        it = self._item()
        if it is None:
            self.viewer.set_source(None)
        elif it.status == "error":
            self.viewer.set_source(None)
            self.statusBar().showMessage(tr("status.error_on", name=it.path.name, error=it.error), 8000)
        else:
            try:
                QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
                # Après enregistrement, l'aperçu vient de l'original déplacé : la
                # photo du dossier source est déjà corrigée (pas de double correction).
                self.viewer.set_source(rotate_array_quarter(self._get_preview(it.original or it.path), it.quarter))
            except Exception as exc:  # noqa: BLE001
                it.status, it.error = "error", str(exc)
                self.viewer.set_source(None)
            finally:
                QApplication.restoreOverrideCursor()
            self.viewer.set_angle(it.angle)
            self.viewer.set_subject(it.subject)
            self.viewer.set_crop(it.crop)
        self._update_controls()

    # ---------------------------------------------------------------- angle
    def set_angle(self, angle: float) -> None:
        it = self._item()
        if it is None or it.status != "ready":
            return
        it.angle = round(max(-45.0, min(45.0, angle)), 2)
        self.viewer.set_angle(it.angle)
        self._refresh_list_item(self.current)
        self._update_controls()

    def rotate_quarter(self, direction: int) -> None:
        """Quart de tour de la photo courante (``LEFT`` = 1, ``RIGHT`` = -1).

        L'angle fin est conservé tel quel ; le cadre manuel et l'objet
        détecté sont transposés dans le nouveau repère (pas de nouvelle analyse).
        """
        it = self._item()
        if it is None or it.status != "ready":
            return
        it.quarter = normalize_quarter(it.quarter + direction)
        it.subject = rotate_subject(it.subject, direction)
        if it.analysis is not None:
            it.analysis.subject = rotate_subject(it.analysis.subject, direction)
        if it.crop is not None:
            it.crop = rotate_norm_box(it.crop, direction)
        self._load_current()
        self._refresh_list_item(self.current)
        self._update_status()

    def nudge(self, delta: float) -> None:
        it = self._item()
        if it is not None:
            self.set_angle(it.angle + delta)

    def reset_auto(self) -> None:
        it = self._item()
        if it is not None and it.estimate is not None:
            self.set_angle(it.estimate.angle)

    def _on_line_drawn(self, delta: float) -> None:
        it = self._item()
        if it is not None:
            self.set_angle(it.angle + delta)

    def _on_crop_points(self, state: int) -> None:
        if state == 1:
            self.statusBar().showMessage(tr("status.crop_first_corner"))
        elif state == -1:
            self.statusBar().showMessage(tr("status.crop_too_close"), 4000)
        else:
            self._update_status()

    def _on_crop_changed(self, norm) -> None:
        it = self._item()
        if it is None or it.status != "ready":
            return
        it.crop = None if norm is None else tuple(float(v) for v in norm)
        self._refresh_list_item(self.current)
        self._update_controls()

    def _on_spin(self, value: float) -> None:
        if not self._updating:
            self.set_angle(value)

    def _on_slider(self, value: int) -> None:
        if not self._updating:
            self.set_angle(value / 10.0)

    def _on_crop_mode(self) -> None:
        self.settings.setValue("crop_mode", self.crop_mode)
        # L'angle automatique dépend du mode (contour de l'objet en mode « objet ») :
        # on bascule sur la nouvelle estimation, sauf si l'angle a été retouché.
        for it in self.items:
            if it.status != "ready" or it.analysis is None:
                continue
            new = it.analysis.estimate_for(self.crop_mode)
            if new is it.estimate:
                continue
            if not it.manual:
                it.angle = new.angle
            it.estimate = new
        cur = self._item()
        if cur is not None and cur.status == "ready":
            self.viewer.set_angle(cur.angle)
        self.viewer.set_crop_mode(self.crop_mode)
        for i in range(len(self.items)):
            self._refresh_list_item(i)
        self._update_controls()

    def _on_language(self) -> None:
        """Mémorise la langue choisie ; l'interface est construite une fois pour
        toutes, la nouvelle langue s'applique donc au prochain démarrage."""
        code = self.lang_combo.currentData()
        self.settings.setValue("language", code)
        lang = i18n.match_language(code) or i18n.resolve_language(None)
        QMessageBox.information(self, i18n.translate(lang, "language.changed.title"),
                                i18n.translate(lang, "language.changed.text"))

    def toggle_grid(self) -> None:
        self.viewer.show_grid = not self.viewer.show_grid
        self.viewer.viewport().update()

    # ----------------------------------------------------------- validation
    def toggle_validated(self) -> None:
        it = self._item()
        if it is not None and it.status == "ready":
            it.validated = not it.validated
            self._refresh_list_item(self.current)
            self._update_controls()
            self._update_status()

    def validate_and_next(self) -> None:
        it = self._item()
        if it is None or it.status != "ready":
            return
        it.validated = True
        self._refresh_list_item(self.current)
        self._update_status()
        # Saute à la prochaine photo non validée (en bouclant)
        n = len(self.items)
        for k in range(1, n + 1):
            j = (self.current + k) % n
            if not self.items[j].validated and self.items[j].status != "error":
                self.list.setCurrentRow(j)
                return
        self._update_controls()
        self.statusBar().showMessage(tr("status.all_validated"), 6000)

    # --------------------------------------------------------------- export
    def _originals_label(self) -> str:
        return str(self.originals_dir) if self.originals_dir else tr("originals.default", dirname=ORIGINALS_DIRNAME)

    def choose_originals_dir(self) -> None:
        start = str(self.originals_dir or self.settings.value("last_dir", str(Path.home())))
        d = QFileDialog.getExistingDirectory(self, tr("dialog.originals_dir"), start)
        if d:
            self.originals_dir = Path(d)
            self._update_status()

    def export_validated(self) -> None:
        """Enregistre les photos validées dans leur dossier : la version corrigée
        remplace l'original, qui est déplacé dans le dossier des originaux."""
        if self._export_task is not None:
            return
        validated = [it for it in self.items if it.validated and it.status == "ready"]
        if not validated:
            QMessageBox.information(self, tr("export.title"), tr("export.none_validated"))
            return
        crop_on = self.crop_mode != "none"
        jobs, unchanged = [], 0
        for it in validated:
            crop = it.crop if crop_on else None
            changed = is_modified(it.angle, crop, crop is None and crops_to_subject(self.crop_mode, it.subject), it.quarter)
            if not changed and it.original is None:
                unchanged += 1  # rien à faire : la photo reste telle quelle
                continue
            jobs.append((it.path, it.angle, it.estimate, crop, it.subject, it.quarter, it.original))
        self._export_unchanged = unchanged
        self._export_counts = {"corrigee": 0, "restauree": 0}
        if not jobs:
            QMessageBox.information(
                self, tr("export.title"), tr("export.none_modified"),
            )
            return
        pending = sum(1 for it in self.items if it.status == "pending")
        parts = [tr("export.confirm", count=len(jobs), dest=self._originals_label())]
        if pending:
            parts.append(tr("export.confirm_pending", count=pending))
        parts.append(tr("export.confirm_continue"))
        if QMessageBox.question(self, tr("export.confirm_title"), "\n\n".join(parts)) != QMessageBox.StandardButton.Yes:
            return
        settings = InPlaceSettings(
            originals_dir=self.originals_dir,
            crop_mode=self.crop_mode,
            write_sidecar=self.write_json.isChecked(),
        )
        task = ExportTask(jobs, settings)
        task.signals.progress.connect(self._on_export_progress)
        task.signals.item_failed.connect(lambda p, m: self.statusBar().showMessage(
            tr("status.export_failed", name=Path(p).name, error=m), 8000))
        task.signals.finished.connect(self._on_export_finished)
        self._export_task = task
        self.progress.setRange(0, len(jobs))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.act_export.setEnabled(False)
        self.pool.start(task)

    def _on_export_progress(self, done: int, total: int, payload) -> None:
        self.progress.setValue(done)
        path, result = payload
        idx = self.index_by_path.get(path)
        if idx is not None and result is not None:
            it = self.items[idx]
            it.original = result.original
            it.exported_to = result.path if result.action == "corrigee" else None
            self._export_counts[result.action] = self._export_counts.get(result.action, 0) + 1
            self._refresh_list_item(idx)
            if idx == self.current:
                self._update_controls()

    def _on_export_finished(self, ok: int, ko: int) -> None:
        self._export_task = None
        self.progress.setVisible(False)
        self.act_export.setEnabled(True)
        fixed = self._export_counts.get("corrigee", 0)
        restored = self._export_counts.get("restauree", 0)
        parts = [tr("export.done_fixed", count=fixed, dest=self._originals_label())]
        if restored:
            parts.append(tr("export.done_restored", count=restored))
        if self._export_unchanged:
            parts.append(tr("export.done_unchanged", count=self._export_unchanged))
        if ko:
            parts.append(tr("export.done_failed", count=ko))
        QMessageBox.information(self, tr("export.done_title"), "\n\n".join(parts))
        self._update_status()

    # ------------------------------------------------------- affichage liste
    def _placeholder_icon(self) -> QIcon:
        pm = QPixmap(THUMB_SIDE // 2 + 30, THUMB_SIDE // 2 + 30)
        pm.fill(QColor(70, 70, 70))
        return QIcon(pm)

    def _render_thumb(self, it: PhotoItem) -> QIcon:
        side = THUMB_SIDE // 2 + 30
        canvas = QPixmap(side, side)
        canvas.fill(Qt.GlobalColor.transparent)
        p = QPainter(canvas)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if it.thumb_src is not None:
            arr = rotate_array(rotate_array_quarter(it.thumb_src, it.quarter), it.angle)
            if self.crop_mode != "none":
                h, w = arr.shape[:2]
                if it.crop is not None:
                    x0, y0, x1, y1 = pixel_crop_box(w, h, it.angle, it.crop)
                else:
                    box = auto_box(w, h, it.angle, self.crop_mode, it.subject)
                    x0, y0, x1, y1 = pixel_crop_box(w, h, it.angle, to_norm(box, w, h))
                arr = arr[y0:y1, x0:x1]
            pm = QPixmap.fromImage(numpy_to_qimage(arr)).scaled(
                side, side, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
            )
            p.drawPixmap((side - pm.width()) // 2, (side - pm.height()) // 2, pm)
        else:
            p.fillRect(0, 0, side, side, QColor(90, 30, 30))

        # Pastille de confiance (en haut à gauche)
        level = self._level(it)
        p.setPen(QColor(255, 255, 255))
        p.setBrush(LEVEL_COLORS[level])
        p.drawEllipse(4, 4, 16, 16)
        # Coche de validation (en haut à droite)
        if it.validated:
            p.setBrush(QColor(30, 110, 220))
            p.drawEllipse(side - 22, 4, 18, 18)
            p.setPen(QColor(255, 255, 255))
            p.drawText(side - 22, 4, 18, 18, Qt.AlignmentFlag.AlignCenter, "✓")
        p.end()
        return QIcon(canvas)

    def _level(self, it: PhotoItem) -> str:
        """Confiance globale : la plus faible entre l'angle et, en mode
        « objet », la détection de l'objet (aucun objet trouvé = moyenne)."""
        level = it.estimate.level if it.estimate else "low"
        if self.crop_mode == "subject" and it.crop is None:
            sub = it.subject.level if it.subject else "medium"
            level = min(level, sub, key=LEVEL_RANK.__getitem__)
        return level

    def _refresh_list_item(self, idx: int) -> None:
        li = self.list.item(idx)
        if li is None:
            return
        it = self.items[idx]
        if it.status == "error":
            li.setText(f"{it.path.name}\n{tr('list.error')}")
            li.setIcon(self._render_thumb(it))
            li.setToolTip(f"{it.path}\n{it.error}")
            return
        if it.status == "pending":
            return
        tag = {1: "  ↺", 2: "  ⟲180", 3: "  ↻"}.get(it.quarter, "")
        tag += "  ✎" if it.manual else ""
        tag += "  ✂" if it.crop is not None and self.crop_mode != "none" else ""
        tag += "  ⇪" if it.exported_to else ""
        li.setText(f"{it.path.name}\n{it.angle:+.2f}°{tag}")
        li.setIcon(self._render_thumb(it))

    def _update_controls(self) -> None:
        it = self._item()
        ready = it is not None and it.status == "ready"
        for w in (self.spin, self.slider, self.btn_left, self.btn_right, self.btn_auto, self.btn_zero,
                  self.btn_orig, self.btn_validate):
            w.setEnabled(ready)
        self.btn_crop.setEnabled(ready and it.crop is not None and self.crop_mode != "none")
        self._updating = True
        self.spin.setValue(it.angle if ready else 0.0)
        self.slider.setValue(int(round((it.angle if ready else 0.0) * 10)))
        self._updating = False
        if ready and it.estimate:
            e = it.estimate
            method = tr("method.contour" if e.source == "contour" else "method.lines")
            parts = [tr("info.auto", method=method, angle=e.angle, level=tr(f"level.{e.level}"),
                        confidence=e.confidence)]
            if it.crop is not None and self.crop_mode != "none":
                parts.append(tr("info.crop_manual"))
            elif self.crop_mode == "subject":
                parts.append(
                    tr("info.subject", level=tr(f"level.{it.subject.level}"), confidence=it.subject.confidence)
                    if it.subject
                    else tr("info.subject_missing")
                )
            if it.quarter:
                parts.append(tr("info.rotated", quarter=tr(f"quarter.{it.quarter}")))
            if it.validated:
                parts.append(tr("info.validated"))
            self.info.setText(f"<b>{it.path.name}</b> — " + " · ".join(parts))
        elif it is not None and it.status == "pending":
            self.info.setText(f"<b>{it.path.name}</b> — {tr('info.analyzing')}")
        else:
            self.info.setText("")

    def _update_status(self) -> None:
        n = len(self.items)
        if not n:
            self.statusBar().showMessage(tr("status.ready"))
            return
        ready = sum(1 for i in self.items if i.status == "ready")
        val = sum(1 for i in self.items if i.validated)
        to_review = sum(1 for i in self.items if i.status == "ready" and not i.validated)
        self.statusBar().showMessage(tr("status.summary", n=n, ready=ready, validated=val, review=to_review,
                                        originals=self._originals_label()))

    # --------------------------------------------------------- glisser-déposer
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_paths(paths)

    def closeEvent(self, event):
        if self._export_task is not None:
            if QMessageBox.question(self, tr("quit.title"), tr("quit.text")) != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self._export_task.cancelled = True
        self.pool.clear()
        self.pool.waitForDone(5000)
        super().closeEvent(event)


def install_qt_translator(app: QApplication, lang: str) -> None:
    """Traduit les textes propres à Qt (boutons Oui / Non, dialogues de fichiers…)
    s'ils sont disponibles pour cette langue ; sinon Qt reste en anglais."""
    translator = QTranslator(app)
    if translator.load(QLocale(lang), "qtbase", "_", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath)):
        app.installTranslator(translator)
        app._qt_translator = translator  # garde une référence


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("picturesRC")
    app.setOrganizationName("picturesRC")
    settings = QSettings("picturesRC", "picturesRC")
    lang = i18n.setup(settings.value("language", "") or None)
    install_qt_translator(app, lang)
    win = MainWindow(settings)
    win.show()
    # Photos ou dossiers passés en argument (ex. « Ouvrir avec… »)
    args = [Path(a) for a in app.arguments()[1:]]
    if args:
        QTimer.singleShot(0, lambda: win.add_paths(args))
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
