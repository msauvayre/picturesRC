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

"""Test de fumée de l'interface, sans écran (plateforme Qt « offscreen »)."""

from __future__ import annotations

import os
import time

import pytest
from PIL import Image

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from test_straighten import make_scene  # noqa: E402


def _wait(app, cond, timeout=30.0):
    t0 = time.time()
    while not cond() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    assert cond(), "délai dépassé"


def test_analyse_review_and_export(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from picturesrc.gui import app as gui_app

    src = tmp_path / "photos"
    src.mkdir()
    for i, tilt in enumerate([2.5, -4.0]):
        make_scene(1200, 800, seed=i).rotate(tilt, resample=Image.Resampling.BICUBIC).save(src / f"p{i}.jpg")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)

    # Réglages isolés : ne touche pas à ceux de l'utilisateur
    win = gui_app.MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    assert win.crop_mode == "subject"  # défaut : autour de l'objet
    # Scènes « ville » plein cadre, sans objet central : surface maximale
    win.crop_combo.setCurrentIndex(win.crop_combo.findData("max_area"))
    win.show()
    win.add_paths([src])
    _wait(app, lambda: all(it.status != "pending" for it in win.items))

    assert [it.angle for it in win.items] == [pytest.approx(-2.5, abs=0.2), pytest.approx(4.0, abs=0.2)]
    assert all(it.validated for it in win.items)  # confiance élevée -> validées d'office

    win.list.setCurrentRow(0)
    before = win.items[0].angle
    win.nudge(0.1)
    assert win.items[0].angle == pytest.approx(before + 0.1)
    assert win.items[0].manual

    win._on_line_drawn(-0.1)  # simulation d'un tracé d'horizon
    assert win.items[0].angle == pytest.approx(before)

    # Une 3e photo validée mais laissée à 0° : laissée telle quelle
    flat = src / "p2_inchangee.jpg"
    Image.new("RGB", (600, 400), (120, 120, 120)).save(flat)
    win.add_paths([flat])
    _wait(app, lambda: all(it.status != "pending" for it in win.items))
    win.items[2].angle = 0.0

    before = {p.name: p.read_bytes() for p in src.iterdir()}
    for it in win.items:
        it.validated = True
    win.export_validated()
    _wait(app, lambda: win._export_task is None, timeout=60)

    # Dossier source : les mêmes noms, les photos corrigées à la place des originaux
    assert sorted(p.name for p in src.iterdir() if p.is_file()) == ["p0.jpg", "p1.jpg", "p2_inchangee.jpg"]
    assert (src / "p0.jpg").read_bytes() != before["p0.jpg"]
    assert flat.read_bytes() == before[flat.name]  # non modifiée : intacte, non déplacée
    assert win._export_unchanged == 1
    # Originaux déplacés (seulement ceux des photos corrigées), JSON à côté
    originals = src / "originaux"
    assert sorted(p.name for p in originals.iterdir()) == [
        "p0.jpg", "p0.jpg.picturesrc.json", "p1.jpg", "p1.jpg.picturesrc.json"]
    assert (originals / "p0.jpg").read_bytes() == before["p0.jpg"]
    assert win.items[0].original == originals / "p0.jpg"

    # L'aperçu vient désormais de l'original déplacé (pas de double correction)
    win.list.setCurrentRow(1)
    win.list.setCurrentRow(0)
    app.processEvents()
    assert win.preview_cache and next(reversed(win.preview_cache)) == originals / "p0.jpg"

    # Ré-enregistrement sans JSON : recalcul depuis les originaux, JSON retirés
    win.write_json.setChecked(False)
    win.export_validated()
    _wait(app, lambda: win._export_task is None, timeout=60)
    assert sorted(p.name for p in originals.iterdir()) == ["p0.jpg", "p1.jpg"]
    assert (originals / "p0.jpg").read_bytes() == before["p0.jpg"]
    win.close()


def test_crop_frame_mouse_interactions(tmp_path, monkeypatch):
    """Cadre de rognage manipulé à la souris : bord, Ctrl+glisser, Maj+glisser, C, export."""
    from PySide6.QtCore import QPoint, QPointF, QSettings, Qt
    from PySide6.QtTest import QTest

    from picturesrc.core.crop import crop_is_valid, from_norm
    from picturesrc.gui import app as gui_app

    src = tmp_path / "photos"
    src.mkdir()
    make_scene(1200, 800, seed=5).rotate(3.0, resample=Image.Resampling.BICUBIC).save(src / "c.jpg")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    win = gui_app.MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    win.resize(1400, 900)
    win.crop_combo.setCurrentIndex(win.crop_combo.findData("max_area"))
    win.show()
    win.add_paths([src])
    _wait(app, lambda: all(it.status != "pending" for it in win.items))
    win.list.setCurrentRow(0)
    app.processEvents()

    viewer, it = win.viewer, win.items[0]
    vp = viewer.viewport()
    assert it.crop is None

    def view_pt(x, y) -> QPoint:
        return viewer.mapFromScene(QPointF(x, y))

    def drag(p0: QPoint, p1: QPoint, mods=Qt.KeyboardModifier.NoModifier):
        QTest.mousePress(vp, Qt.MouseButton.LeftButton, mods, p0)
        for k in range(1, 6):  # quelques mouvements intermédiaires
            QTest.mouseMove(vp, p0 + (p1 - p0) * (k / 5))
        QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, mods, p1)
        app.processEvents()

    h, w = viewer._source.shape[:2]
    l, t, r, b = viewer._current_box()

    # 1) Tirer le bord droit vers la gauche : cadre manuel, plus étroit
    mid_y = (t + b) / 2
    drag(view_pt(r, mid_y), view_pt(r - 0.3 * w, mid_y))
    assert it.crop is not None
    box = from_norm(it.crop, w, h)
    assert box[2] < r - 0.2 * w and box[0] == pytest.approx(l, abs=2)
    assert win.items[0].angle == pytest.approx(-3.0, abs=0.2)  # l'angle n'a pas bougé

    # 2) Ctrl+glisser vers la gauche, très loin : le cadre bute contre la zone utile
    c = view_pt((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    drag(c, c - QPoint(2000, 0), Qt.KeyboardModifier.ControlModifier)
    moved = from_norm(it.crop, w, h)
    assert moved[0] < box[0] and moved[2] - moved[0] == pytest.approx(box[2] - box[0], abs=1)
    assert crop_is_valid(w, h, it.angle, moved)

    # 3) Maj+glisser : nouveau cadre tracé
    drag(view_pt(0.5 * w, 0.4 * h), view_pt(0.8 * w, 0.7 * h), Qt.KeyboardModifier.ShiftModifier)
    drawn = from_norm(it.crop, w, h)
    assert drawn[0] == pytest.approx(0.5 * w, abs=3) and drawn[3] == pytest.approx(0.7 * h, abs=3)

    # 4) Glisser ailleurs (sans modificateur, hors bords) : tracé d'horizon, le cadre reste
    before = it.crop
    drag(view_pt(0.1 * w, 0.2 * h), view_pt(0.4 * w, 0.2 * h + 0.3 * w * 0.0175))
    assert it.crop == before
    assert it.angle != pytest.approx(-3.0, abs=0.05)

    # 5) Changer l'angle garde un cadre valide ; puis export avec le cadre manuel
    win.set_angle(-3.0)
    assert crop_is_valid(w, h, -3.0, from_norm(it.crop, w, h))
    assert "✂" in win.list.item(0).text()
    it.validated = True
    win.export_validated()
    _wait(app, lambda: win._export_task is None, timeout=60)
    out = src / "c.jpg"
    with Image.open(out) as im:
        assert im.width == pytest.approx(0.3 * 1200, abs=12) and im.height == pytest.approx(0.3 * 800, abs=12)
    meta = __import__("json").loads((src / "originaux" / "c.jpg.picturesrc.json").read_text(encoding="utf-8"))
    assert meta["transformation"]["rognage"] == "manuel"

    # 6) C : retour au cadre automatique
    QTest.keyClick(win, Qt.Key.Key_C)
    app.processEvents()
    assert it.crop is None and "✂" not in win.list.item(0).text()
    win.close()


def test_subject_mode_uses_outline_angle(tmp_path, monkeypatch):
    """Mode « objet » : angle d'après le contour du sachet ; changer de mode
    bascule sur l'angle « lignes », sauf si l'angle a été retouché."""
    from PySide6.QtCore import QSettings

    from picturesrc.gui import app as gui_app
    from test_subject import make_bag_with_skewed_content

    src = tmp_path / "scelles"
    src.mkdir()
    make_bag_with_skewed_content(3.0, 7.0).save(src / "a.jpg", quality=92)
    make_bag_with_skewed_content(-2.0, 7.0, seed=4).save(src / "b.jpg", quality=92)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    win = gui_app.MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    win.show()
    win.add_paths([src])
    _wait(app, lambda: all(it.status != "pending" for it in win.items))

    a, b = win.items
    assert win.crop_mode == "subject"
    assert a.estimate.source == "contour" and a.angle == pytest.approx(-3.0, abs=0.25)
    assert b.angle == pytest.approx(2.0, abs=0.25)
    assert a.validated and b.validated  # angle et objet sûrs : validées d'office

    b.angle += 0.5  # retouche manuelle sur b
    win.crop_combo.setCurrentIndex(win.crop_combo.findData("max_area"))
    assert a.estimate.source == "lignes" and a.angle == pytest.approx(a.estimate.angle)
    assert b.estimate.source == "lignes" and b.angle == pytest.approx(2.5, abs=0.25)  # retouche conservée

    win.crop_combo.setCurrentIndex(win.crop_combo.findData("subject"))
    assert a.angle == pytest.approx(-3.0, abs=0.25)
    win.close()


def test_crop_with_two_clicks(tmp_path, monkeypatch):
    """Cadre posé en deux clics : coin haut-gauche puis bas-droit, appliqué aussitôt."""
    from PySide6.QtCore import QPointF, QSettings, Qt
    from PySide6.QtTest import QTest

    from picturesrc.core.crop import crop_is_valid, from_norm
    from picturesrc.gui import app as gui_app

    src = tmp_path / "photos"
    src.mkdir()
    make_scene(1200, 800, seed=7).rotate(3.0, resample=Image.Resampling.BICUBIC).save(src / "d.jpg")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    win = gui_app.MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    win.resize(1400, 900)
    win.show()
    win.add_paths([src])
    _wait(app, lambda: all(it.status != "pending" for it in win.items))
    win.list.setCurrentRow(0)
    app.processEvents()

    viewer, it = win.viewer, win.items[0]
    vp = viewer.viewport()
    h, w = viewer._source.shape[:2]

    def click(x, y):
        QTest.mouseClick(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                         viewer.mapFromScene(QPointF(x, y)))
        app.processEvents()

    # 1) Deux clics à l'intérieur de l'image : le cadre est exactement celui des deux coins
    click(0.30 * w, 0.25 * h)
    assert viewer.crop_points_pending() and it.crop is None  # rien n'est appliqué au 1er clic
    assert "inférieur droit" in win.statusBar().currentMessage()
    click(0.70 * w, 0.80 * h)
    assert not viewer.crop_points_pending()
    l, t, r, b = from_norm(it.crop, w, h)
    tol = 0.01 * w  # précision du clic à l'écran
    assert (l, t, r, b) == (pytest.approx(0.30 * w, abs=tol), pytest.approx(0.25 * h, abs=tol),
                            pytest.approx(0.70 * w, abs=tol), pytest.approx(0.80 * h, abs=tol))
    assert "✂" in win.list.item(0).text()

    # 2) Coins dans les zones vides (coins noirs) : cadre borné à la zone utile
    click(1, 1)
    click(w - 2, h - 2)
    box = from_norm(it.crop, w, h)
    assert crop_is_valid(w, h, it.angle, box)
    assert box[2] - box[0] > 0.85 * w and box[3] - box[1] > 0.8 * h

    # 3) Ordre inversé (bas-droit puis haut-gauche) : même cadre
    click(0.70 * w, 0.80 * h)
    click(0.30 * w, 0.25 * h)
    l2, t2, r2, b2 = from_norm(it.crop, w, h)
    assert (l2, r2) == (pytest.approx(l, abs=tol), pytest.approx(r, abs=tol))

    # 4) Échap après le premier clic : annulé, cadre inchangé
    before = it.crop
    click(0.1 * w, 0.1 * h)
    QTest.keyClick(win, Qt.Key.Key_Escape)
    app.processEvents()
    assert not viewer.crop_points_pending() and it.crop == before

    # 5) Deux clics au même endroit : refusé, cadre inchangé
    click(0.5 * w, 0.5 * h)
    click(0.5 * w + 2, 0.5 * h + 2)
    assert it.crop == before and "trop proches" in win.statusBar().currentMessage()

    # 5 bis) Clic « tremblé » (la souris bouge de 6 px entre l'appui et le relâché) : reste un clic
    from PySide6.QtCore import QPoint
    p0 = viewer.mapFromScene(QPointF(0.3 * w, 0.3 * h))
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0)
    QTest.mouseMove(vp, p0 + QPoint(4, 2))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, p0 + QPoint(4, 2))
    app.processEvents()
    assert viewer.crop_points_pending()
    viewer.cancel_crop_points()

    # 6) Un clic sur un bord du cadre (sans glisser) pose aussi un coin
    click(l, 0.5 * h)
    assert viewer.crop_points_pending()
    viewer.cancel_crop_points()

    # 7) Un glisser trace toujours l'horizon (pas de coin posé)
    angle0 = it.angle
    QTest.mousePress(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, viewer.mapFromScene(QPointF(0.2 * w, 0.5 * h)))
    QTest.mouseMove(vp, viewer.mapFromScene(QPointF(0.5 * w, 0.5 * h + 8)))
    QTest.mouseRelease(vp, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, viewer.mapFromScene(QPointF(0.5 * w, 0.5 * h + 8)))
    app.processEvents()
    assert not viewer.crop_points_pending() and it.angle != pytest.approx(angle0, abs=0.01)
    win.close()


def test_quarter_turns_left_and_right(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings

    from picturesrc.core.orientation import rotate_norm_box
    from picturesrc.gui import app as gui_app

    src = tmp_path / "photos"
    src.mkdir()
    make_scene(1200, 800, seed=3).rotate(2.0, resample=Image.Resampling.BICUBIC).save(src / "paysage.jpg")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *a, **k: QtWidgets.QMessageBox.StandardButton.Yes)
    win = gui_app.MainWindow(QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat))
    win.show()
    win.add_paths([src])
    _wait(app, lambda: all(it.status != "pending" for it in win.items))
    win.list.setCurrentRow(0)
    it = win.items[0]
    angle = it.angle
    h, w = win.viewer._source.shape[:2]
    assert w > h

    it.crop = (0.2, 0.25, 0.7, 0.8)
    win.rotate_quarter(gui_app.RIGHT)
    assert it.quarter == 3
    assert win.viewer._source.shape[:2] == (w, h)  # l'aperçu est passé en portrait
    assert it.angle == pytest.approx(angle)  # l'angle fin est conservé
    assert it.crop == pytest.approx(rotate_norm_box((0.2, 0.25, 0.7, 0.8), gui_app.RIGHT))
    assert "↻" in win.list.item(0).text()

    win.rotate_quarter(gui_app.LEFT)
    assert it.quarter == 0
    assert it.crop == pytest.approx((0.2, 0.25, 0.7, 0.8))

    # Export : sans rognage ni rotation fine, 90° à droite -> dimensions échangées
    it.crop = None
    win.rotate_quarter(gui_app.RIGHT)
    win.crop_combo.setCurrentIndex(win.crop_combo.findData("none"))
    win.set_angle(0.0)
    it.validated = True
    win.export_validated()
    _wait(app, lambda: win._export_task is None, timeout=60)
    assert it.original == src / "originaux" / "paysage.jpg"  # un quart de tour seul est une modification
    out = Image.open(src / "paysage.jpg")
    assert out.size == (800, 1200)
    win.close()
