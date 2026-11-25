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

"""Zone d'aperçu : image pivotée, grille de contrôle, cadre de rognage
réglable et outil « tracer l'horizon ».

Souris (bouton gauche) :
    * deux clics simples (sans glisser) : coin supérieur gauche puis coin
      inférieur droit du cadre de rognage, appliqué aussitôt (Échap annule
      après le premier clic) ;
    * sur un bord ou un coin du cadre : redimensionner le cadre ;
    * Ctrl + glisser dans le cadre     : déplacer le cadre ;
    * Maj + glisser                    : tracer un nouveau cadre ;
    * ailleurs                         : tracer l'horizon (ou une verticale).
Le cadre ne peut jamais sortir de la zone utile de l'image pivotée : il
bute contre les coins vides et glisse le long de leur bord.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from ..core.crop import (
    box_from_drag,
    from_norm,
    move_box,
    refit_box,
    resize_box,
    to_norm,
)
from ..core.straighten import measure_line_angle, rotate_array
from ..core.subject import SubjectEstimate, auto_box
from ..i18n import tr

HANDLE_TOL = 8  # tolérance (pixels écran) pour attraper un bord du cadre
# En deçà (pixels écran), un appui-relâché est un clic, pas un glisser. On prend
# le seuil du système (QApplication.startDragDistance, ~10 px) s'il est plus grand :
# avec un pavé tactile ou une souris sensible, la main bouge un peu pendant le clic.
CLICK_TOL = 4
MIN_POINTS_BOX = 15  # taille minimale (pixels écran) d'un cadre posé en deux clics
MIN_CROP_FRACTION = 0.03  # taille minimale du cadre (fraction du grand côté)

_HANDLE_CURSORS = {
    "l": Qt.CursorShape.SizeHorCursor,
    "r": Qt.CursorShape.SizeHorCursor,
    "t": Qt.CursorShape.SizeVerCursor,
    "b": Qt.CursorShape.SizeVerCursor,
    "tl": Qt.CursorShape.SizeFDiagCursor,
    "br": Qt.CursorShape.SizeFDiagCursor,
    "tr": Qt.CursorShape.SizeBDiagCursor,
    "bl": Qt.CursorShape.SizeBDiagCursor,
}


def numpy_to_qimage(arr: np.ndarray) -> QImage:
    arr = np.ascontiguousarray(arr)
    h, w = arr.shape[:2]
    if arr.ndim == 2:
        return QImage(arr.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QImage(arr.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ImageViewer(QGraphicsView):
    """Affiche l'aperçu pivoté.

    Signaux :
        ``line_drawn(delta)``   — après un tracé d'horizon ;
        ``crop_changed(norm)``  — cadre modifié par l'utilisateur
        (``norm`` normalisé entre 0 et 1, ou ``None`` = cadre automatique).
    """

    line_drawn = Signal(float)
    crop_changed = Signal(object)
    # Cadre en deux clics : 1 = premier coin posé, 0 = terminé ou annulé,
    # -1 = second coin trop proche du premier (rien n'est appliqué)
    crop_points_state = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._item = QGraphicsPixmapItem()
        self._item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self.scene().addItem(self._item)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(40, 40, 40))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)  # curseur adapté au survol des bords du cadre

        self._source: np.ndarray | None = None
        self._angle = 0.0
        self._crop_mode = "subject"
        self._subject: SubjectEstimate | None = None
        self._manual_box: tuple[float, float, float, float] | None = None  # pixels de l'aperçu
        self._show_original = False
        self.show_grid = True
        self._drag_start: QPointF | None = None
        self._drag_end: QPointF | None = None
        # Opération en cours sur le cadre : ("resize", poignée) | ("move",) | ("draw",)
        self._crop_op: tuple | None = None
        self._crop_op_start: QPointF | None = None
        self._crop_op_box: tuple[float, float, float, float] | None = None
        self._crop_op_had_manual = False
        self._crop_op_manual: tuple[float, float, float, float] | None = None
        self._press_view: QPointF | None = None  # position écran de l'appui (détection des clics)
        self._first_point: QPointF | None = None  # premier coin posé (cadre en deux clics)
        self._hover_scene: QPointF | None = None
        self._placeholder = tr("viewer.placeholder")

    # --- API ---------------------------------------------------------------
    def set_source(self, image: np.ndarray | None) -> None:
        """Nouvelle image ; le cadre repasse en automatique (cf. ``set_crop``,
        ``set_subject``)."""
        self._source = image
        self._manual_box = None
        self._subject = None
        self._cancel_points(emit=True)
        self._render()
        self.fit()

    def set_angle(self, angle: float) -> None:
        """Change l'angle affiché ; un cadre manuel devenu invalide est réajusté
        (``crop_changed`` est alors émis)."""
        self._angle = angle
        if self._manual_box is not None and self._source is not None:
            h, w = self._source.shape[:2]
            fitted = refit_box(w, h, angle, self._manual_box)
            if fitted != self._manual_box:
                self._manual_box = fitted
                self.crop_changed.emit(self.crop_norm())
        self._render()

    def set_subject(self, subject: SubjectEstimate | None) -> None:
        """Objet détecté sur la photo affichée (cadre automatique du mode « objet »)."""
        self._subject = subject
        self.viewport().update()

    def set_crop_mode(self, mode: str) -> None:
        self._crop_mode = mode
        if mode == "none":
            self.cancel_crop_points()
        self.viewport().update()

    def set_crop(self, norm: tuple | None) -> None:
        """Impose le cadre (normalisé) sans émettre de signal ; ``None`` = automatique."""
        if norm is None or self._source is None:
            self._manual_box = None
        else:
            h, w = self._source.shape[:2]
            self._manual_box = refit_box(w, h, self._angle, from_norm(norm, w, h))
        self.viewport().update()

    def crop_norm(self) -> tuple | None:
        if self._manual_box is None or self._source is None:
            return None
        h, w = self._source.shape[:2]
        return to_norm(self._manual_box, w, h)

    def reset_crop(self) -> None:
        """Revient au cadre automatique."""
        if self._manual_box is not None:
            self._manual_box = None
            self.viewport().update()
            self.crop_changed.emit(None)

    def set_show_original(self, on: bool) -> None:
        self._show_original = on
        if on:
            self.cancel_crop_points()
        self._render()

    # --- cadre en deux clics -------------------------------------------------
    def crop_points_pending(self) -> bool:
        """Vrai si le premier coin est posé et que l'on attend le second."""
        return self._first_point is not None

    def cancel_crop_points(self) -> None:
        """Abandonne un cadre en deux clics commencé (touche Échap)."""
        self._cancel_points(emit=True)

    def _cancel_points(self, emit: bool) -> None:
        had = self._first_point is not None
        self._first_point = self._hover_scene = None
        self.viewport().update()
        if had and emit:
            self.crop_points_state.emit(0)

    def _points_box(self, p1: QPointF, p2: QPointF) -> tuple[float, float, float, float]:
        """Cadre défini par deux coins opposés, borné à la zone utile.

        Les coins sont attendus haut-gauche puis bas-droit, mais l'ordre
        inverse (ou anti-diagonal) donne le même cadre.
        """
        h, w = self._source.shape[:2]
        return box_from_drag(w, h, self._angle, (p1.x(), p1.y()), p2.x(), p2.y())

    def _on_click(self, view_pos: QPointF) -> None:
        """Simple clic gauche sur l'aperçu : pose un coin du cadre."""
        if not self._crop_editable():
            return
        scene = self.mapToScene(view_pos.toPoint())
        if self._first_point is None:
            self._first_point = scene
            self._hover_scene = scene
            self.viewport().update()
            self.crop_points_state.emit(1)
            return
        p1 = self._first_point
        self._first_point = self._hover_scene = None
        a, b = self.mapFromScene(p1), self.mapFromScene(scene)
        if abs(b.x() - a.x()) < MIN_POINTS_BOX or abs(b.y() - a.y()) < MIN_POINTS_BOX:
            self.viewport().update()
            self.crop_points_state.emit(-1)
            return
        self._manual_box = self._points_box(p1, scene)
        self.viewport().update()
        self.crop_points_state.emit(0)
        self.crop_changed.emit(self.crop_norm())

    def fit(self) -> None:
        if self._source is not None:
            self.fitInView(self._item, Qt.AspectRatioMode.KeepAspectRatio)

    # --- rendu -------------------------------------------------------------
    def _effective_angle(self) -> float:
        return 0.0 if self._show_original else self._angle

    def _render(self) -> None:
        if self._source is None:
            self._item.setPixmap(QPixmap())
            self.scene().setSceneRect(QRectF())
        else:
            rotated = rotate_array(self._source, self._effective_angle())
            self._item.setPixmap(QPixmap.fromImage(numpy_to_qimage(rotated)))
            self.scene().setSceneRect(self._item.boundingRect())
        self.viewport().update()

    def _crop_editable(self) -> bool:
        return self._source is not None and self._crop_mode != "none" and not self._show_original

    def _current_box(self) -> tuple[float, float, float, float] | None:
        """Cadre affiché, en pixels de l'aperçu (manuel ou automatique)."""
        if not self._crop_editable():
            return None
        if self._manual_box is not None:
            return self._manual_box
        h, w = self._source.shape[:2]
        return auto_box(w, h, self._angle, self._crop_mode, self._subject)

    def _crop_rect(self) -> QRectF | None:
        box = self._current_box()
        if box is None:
            return None
        l, t, r, b = box
        return QRectF(l, t, r - l, b - t)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fit()

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        if self._source is None:
            painter.resetTransform()
            painter.setPen(QColor(170, 170, 170))
            painter.drawText(self.viewport().rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        full = self._item.boundingRect()
        crop = self._crop_rect()
        area = crop or full

        if crop is not None:  # assombrit ce qui sera rogné
            path = QPainterPath()
            path.addRect(full)
            inner = QPainterPath()
            inner.addRect(crop)
            painter.fillPath(path.subtracted(inner), QColor(0, 0, 0, 150))
            manual = self._manual_box is not None
            # Cadre manuel en orange, automatique en blanc
            color = QColor(255, 170, 40, 235) if manual else QColor(255, 255, 255, 220)
            pen = QPen(color, 2 if manual else 0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawRect(crop)
            self._draw_handles(painter, crop, color)

        if self.show_grid:
            pen = QPen(QColor(255, 255, 0, 110), 0)
            pen.setCosmetic(True)
            painter.setPen(pen)
            step = area.height() / 8
            y = area.top() + step
            while y < area.bottom() - 1:
                painter.drawLine(QLineF(area.left(), y, area.right(), y))
                y += step
            x = area.left() + step
            while x < area.right() - 1:
                painter.drawLine(QLineF(x, area.top(), x, area.bottom()))
                x += step

        if self._first_point is not None:
            self._draw_points_preview(painter)

        if self._drag_start is not None and self._drag_end is not None:
            pen = QPen(QColor(0, 200, 255), 2)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawLine(QLineF(self._drag_start, self._drag_end))

    def _draw_points_preview(self, painter: QPainter) -> None:
        """Premier coin (cible) et cadre qui suit la souris, en pointillés."""
        scale = self.transform().m11() or 1.0
        color = QColor(0, 220, 255)
        if self._hover_scene is not None:
            l, t, r, b = self._points_box(self._first_point, self._hover_scene)
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(l, t, r - l, b - t))
        p = self._first_point
        rad = 7.0 / scale
        pen = QPen(color, 2)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawEllipse(p, rad, rad)
        painter.drawLine(QLineF(p.x() - 2 * rad, p.y(), p.x() + 2 * rad, p.y()))
        painter.drawLine(QLineF(p.x(), p.y() - 2 * rad, p.x(), p.y() + 2 * rad))

    def _draw_handles(self, painter: QPainter, crop: QRectF, color: QColor) -> None:
        """Petites poignées carrées (taille constante à l'écran) aux coins et milieux."""
        scale = self.transform().m11() or 1.0
        s = 7.0 / scale
        painter.setBrush(color)
        pen = QPen(QColor(0, 0, 0, 160), 0)
        pen.setCosmetic(True)
        painter.setPen(pen)
        cx, cy = crop.center().x(), crop.center().y()
        for x, y in (
            (crop.left(), crop.top()), (cx, crop.top()), (crop.right(), crop.top()),
            (crop.left(), cy), (crop.right(), cy),
            (crop.left(), crop.bottom()), (cx, crop.bottom()), (crop.right(), crop.bottom()),
        ):
            painter.drawRect(QRectF(x - s / 2, y - s / 2, s, s))
        painter.setBrush(Qt.BrushStyle.NoBrush)

    # --- interactions --------------------------------------------------------
    def _hit_handle(self, view_pos: QPointF) -> str | None:
        """Bord ou coin du cadre sous le curseur (coordonnées écran), sinon ``None``."""
        crop = self._crop_rect()
        if crop is None:
            return None
        tl = self.mapFromScene(crop.topLeft())
        br = self.mapFromScene(crop.bottomRight())
        x, y = view_pos.x(), view_pos.y()
        if not (tl.x() - HANDLE_TOL <= x <= br.x() + HANDLE_TOL and tl.y() - HANDLE_TOL <= y <= br.y() + HANDLE_TOL):
            return None
        handle = ""
        if abs(y - tl.y()) <= HANDLE_TOL:
            handle += "t"
        elif abs(y - br.y()) <= HANDLE_TOL:
            handle += "b"
        if abs(x - tl.x()) <= HANDLE_TOL:
            handle += "l"
        elif abs(x - br.x()) <= HANDLE_TOL:
            handle += "r"
        return handle or None

    def _inside_crop(self, scene_pos: QPointF) -> bool:
        crop = self._crop_rect()
        return crop is not None and crop.contains(scene_pos)

    def _update_hover_cursor(self, view_pos: QPointF, modifiers) -> None:
        if self._crop_editable():
            handle = self._hit_handle(view_pos)
            if handle:
                self.viewport().setCursor(_HANDLE_CURSORS[handle])
                return
            if modifiers & Qt.KeyboardModifier.ControlModifier and self._inside_crop(self.mapToScene(view_pos.toPoint())):
                self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)
                return
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._source is not None:
            pos = event.position()
            scene = self.mapToScene(pos.toPoint())
            mods = event.modifiers()
            self._press_view = QPointF(pos)
            if self._first_point is not None:
                # Second coin attendu : le clic est réservé au cadre en deux clics
                event.accept()
                return
            if self._crop_editable():
                handle = self._hit_handle(pos)
                op = None
                if handle:
                    op = ("resize", handle)
                elif mods & Qt.KeyboardModifier.ControlModifier and self._inside_crop(scene):
                    op = ("move",)
                elif mods & Qt.KeyboardModifier.ShiftModifier:
                    op = ("draw",)
                if op is not None:
                    h, w = self._source.shape[:2]
                    self._crop_op = op
                    self._crop_op_start = scene
                    self._crop_op_had_manual = self._manual_box is not None
                    self._crop_op_manual = self._manual_box
                    # Point de départ garanti valide (le cadre auto est arrondi au pixel)
                    self._crop_op_box = refit_box(w, h, self._angle, self._current_box())
                    event.accept()
                    return
            self._drag_start = scene
            self._drag_end = self._drag_start
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._first_point is not None:
            self._hover_scene = self.mapToScene(event.position().toPoint())
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self.viewport().update()
            return
        if self._crop_op is not None:
            self._apply_crop_op(self.mapToScene(event.position().toPoint()))
            self.viewport().update()
            return
        if self._drag_start is not None:
            self._drag_end = self.mapToScene(event.position().toPoint())
            self.viewport().update()
            return
        self._update_hover_cursor(event.position(), event.modifiers())
        super().mouseMoveEvent(event)

    def _is_click(self, event) -> bool:
        if self._press_view is None:
            return False
        d = event.position() - self._press_view
        tol = max(CLICK_TOL, QApplication.startDragDistance())
        return abs(d.x()) + abs(d.y()) <= tol

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._first_point is not None:
            self._on_click(event.position())
            return
        if (
            self._crop_op is not None
            and event.button() == Qt.MouseButton.LeftButton
            and self._crop_op[0] == "resize"
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
            and self._is_click(event)
        ):
            # Clic sur un bord du cadre sans glisser : c'est un coin du cadre en deux clics
            self._manual_box = self._crop_op_manual
            self._crop_op = self._crop_op_start = self._crop_op_box = None
            self._on_click(event.position())
            return
        if self._crop_op is not None and event.button() == Qt.MouseButton.LeftButton:
            before = self._crop_op_box
            self._apply_crop_op(self.mapToScene(event.position().toPoint()))
            op = self._crop_op
            self._crop_op = self._crop_op_start = self._crop_op_box = None
            if op[0] == "draw" and self._manual_box is not None:
                l, t, r, b = self._manual_box
                p0, p1 = self.mapFromScene(QPointF(l, t)), self.mapFromScene(QPointF(r, b))
                if abs(p1.x() - p0.x()) < 15 or abs(p1.y() - p0.y()) < 15:
                    # Simple clic avec Maj : on garde le cadre précédent
                    self._manual_box = before if self._crop_op_had_manual else None
            self.viewport().update()
            self.crop_changed.emit(self.crop_norm())
            return
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            start, end = self._drag_start, self.mapToScene(event.position().toPoint())
            self._drag_start = self._drag_end = None
            self.viewport().update()
            if self._is_click(event) and event.modifiers() == Qt.KeyboardModifier.NoModifier:
                self._on_click(event.position())
                return
            # Ignore les tracés trop courts à l'écran
            p0, p1 = self.mapFromScene(start), self.mapFromScene(end)
            if (p1 - p0).manhattanLength() >= 25:
                delta = measure_line_angle(start.x(), start.y(), end.x(), end.y())
                self.line_drawn.emit(0.0 if self._show_original else delta)
            return
        super().mouseReleaseEvent(event)

    def _apply_crop_op(self, scene_pos: QPointF) -> None:
        """Met à jour le cadre manuel pendant un glisser, sans sortir de la zone utile."""
        h, w = self._source.shape[:2]
        kind = self._crop_op[0]
        start, box = self._crop_op_start, self._crop_op_box
        x, y = scene_pos.x(), scene_pos.y()
        if kind == "resize":
            min_size = MIN_CROP_FRACTION * max(w, h)
            self._manual_box = resize_box(w, h, self._angle, box, self._crop_op[1], x, y, min_size)
        elif kind == "move":
            self._manual_box = move_box(w, h, self._angle, box, x - start.x(), y - start.y())
        elif kind == "draw":
            self._manual_box = box_from_drag(w, h, self._angle, (start.x(), start.y()), x, y)
