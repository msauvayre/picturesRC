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

"""Rognage autour de l'objet central (scellé posé sur un bureau)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from picturesrc.core.crop import crop_is_valid
from picturesrc.core.pipeline import ExportSettings, export_image, straighten_image
from picturesrc.core.straighten import estimate_tilt
from picturesrc.core.subject import _rotate_points, detect_subject, subject_crop_box

W, H = 1200, 1600
# Sachet : rectangle (gauche, haut, droite, bas) dans la scène non penchée
BAG = (220, 260, 980, 1300)


def make_desk_scene(tilt: float = 0.0, seed: int = 0, glare: bool = True) -> Image.Image:
    """Sachet blanc imprimé, posé de travers sur un bureau en bois, avec un
    passe-câble gris en bas et un reflet clair collé au haut du sachet."""
    rng = np.random.default_rng(seed)
    # Bois : orange-brun, veinage vertical, bruit
    x = np.arange(W)[None, :, None]
    grain = 10 * np.sin(x / 9.0 + rng.normal(0, 0.3, (H, 1, 1))) + rng.normal(0, 5, (H, W, 1))
    desk = np.clip(np.array([196, 132, 82])[None, None, :] + grain, 0, 255).astype(np.uint8)
    img = Image.fromarray(desk)
    d = ImageDraw.Draw(img)
    if glare:  # reflet de lumière sur le bureau, collé au sachet
        d.ellipse([560, 170, 680, 300], fill=(226, 190, 150))
    d.ellipse([480, 1380, 640, 1540], fill=(150, 150, 155), outline=(90, 90, 90), width=10)  # passe-câble

    bag = Image.new("RGBA", (BAG[2] - BAG[0], BAG[3] - BAG[1]), (236, 236, 240, 255))
    bd = ImageDraw.Draw(bag)
    bw, bh = bag.size
    bd.rectangle([0, 0, bw - 1, 60], fill=(200, 30, 40, 255))
    for yy in range(120, bh - 40, 70):  # formulaire
        bd.line([40, yy, bw - 40, yy], fill=(40, 60, 150, 255), width=3)
        bd.text((60, yy + 15), "Scelle n° 12345 - objet", fill=(30, 30, 30, 255))
    bd.rectangle([0, 0, bw - 1, bh - 1], outline=(120, 130, 170, 255), width=4)
    bag = bag.rotate(tilt, resample=Image.Resampling.BICUBIC, expand=True)
    cx, cy = (BAG[0] + BAG[2]) / 2, (BAG[1] + BAG[3]) / 2
    img.paste(bag, (int(cx - bag.width / 2), int(cy - bag.height / 2)), bag)
    return img.filter(ImageFilter.GaussianBlur(0.7))


def _bag_box_after_straightening(tilt: float, angle: float) -> tuple[float, float, float, float]:
    """Boîte attendue du sachet dans l'image redressée de ``angle``."""
    l, t, r, b = BAG
    cx, cy = (l + r) / 2, (t + b) / 2
    corners = np.array([[l, t], [r, t], [r, b], [l, b]], float)
    # Sachet penché de ``tilt`` (antihoraire) autour de son centre…
    a = np.radians(tilt)
    c, s = np.cos(a), np.sin(a)
    d = corners - (cx, cy)
    tilted = np.column_stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]) + (cx, cy)
    # …puis toute la photo redressée de ``angle`` autour du centre de l'image
    pts = _rotate_points(tilted, W, H, angle)
    return (*pts.min(axis=0), *pts.max(axis=0))


@pytest.mark.parametrize("tilt", [0.0, 3.0, -4.0])
def test_detects_bag_and_ignores_desk_clutter(tilt):
    scene = np.asarray(make_desk_scene(tilt, seed=int(tilt * 10) % 7))
    est = estimate_tilt(scene)
    sub = detect_subject(scene, est.angle)
    assert sub is not None and sub.level == "high"

    l, t, r, b = subject_crop_box(W, H, est.angle, sub)
    el, et, er, eb = _bag_box_after_straightening(tilt, est.angle)
    tol = 0.035 * (eb - et)  # marge de 2 % + imprécision du redressement
    assert l == pytest.approx(el, abs=tol) and r == pytest.approx(er, abs=tol)
    assert t == pytest.approx(et, abs=tol)  # le reflet au-dessus est écarté
    assert b == pytest.approx(eb, abs=tol)  # le passe-câble en dessous aussi
    assert crop_is_valid(W, H, est.angle, (l, t, r, b))


def test_no_central_object_gives_none_or_low_confidence():
    rng = np.random.default_rng(3)
    flat = np.clip(rng.normal(128, 4, (800, 1200, 3)), 0, 255).astype(np.uint8)
    sub = detect_subject(flat)
    assert sub is None or sub.level != "high"


def test_object_touching_border_is_flagged():
    img = make_desk_scene(0.0, glare=False)
    # On recadre pour que le sachet sorte de la photo par la gauche
    arr = np.asarray(img.crop((400, 0, W, H)))
    sub = detect_subject(arr)
    assert sub is not None and sub.level != "high"


def test_export_crops_around_subject_even_without_rotation(tmp_path):
    src = tmp_path / "scelle.jpg"
    make_desk_scene(0.0, glare=False).save(src, quality=92)
    preview = np.asarray(Image.open(src))
    sub = detect_subject(preview, 0.0)
    out = export_image(src, 0.0, ExportSettings(output_dir=tmp_path / "out", crop_mode="subject"), subject=sub)
    with Image.open(out) as im:
        bw, bh = BAG[2] - BAG[0], BAG[3] - BAG[1]
        assert im.width == pytest.approx(bw * 1.04, rel=0.03) and im.height == pytest.approx(bh * 1.04, rel=0.03)
    meta = json.loads(out.with_name(out.name + ".picturesrc.json").read_text(encoding="utf-8"))
    assert meta["transformation"]["rognage"] == "objet"
    assert meta["detection_objet"]["confiance"] == sub.confidence


def test_subject_mode_without_subject_falls_back(tmp_path):
    img = Image.new("L", (1200, 800), 255)
    out, box = straighten_image(img, 5.0, "subject", None, None)
    assert np.asarray(out).min() > 200  # surface maximale, pas de coin vide
    # Aucune rotation, aucun objet : copie identique à l'export
    src = tmp_path / "plat.jpg"
    Image.new("RGB", (600, 400), (120, 120, 120)).save(src)
    res = export_image(src, 0.0, ExportSettings(output_dir=tmp_path / "out", crop_mode="subject"), subject=None)
    assert res.read_bytes() == src.read_bytes()


def make_bag_with_skewed_content(bag_tilt: float, content_tilt: float, seed: int = 1) -> Image.Image:
    """Sachet penché de ``bag_tilt`` contenant un appareil sombre très
    contrasté, posé de travers (``content_tilt`` en plus) : le piège de
    l'estimation « lignes »."""
    rng = np.random.default_rng(seed)
    x = np.arange(W)[None, :, None]
    grain = 10 * np.sin(x / 9.0 + rng.normal(0, 0.3, (H, 1, 1))) + rng.normal(0, 5, (H, W, 1))
    img = Image.fromarray(np.clip(np.array([196, 132, 82])[None, None, :] + grain, 0, 255).astype(np.uint8))
    bw, bh = BAG[2] - BAG[0], BAG[3] - BAG[1]
    bag = Image.new("RGBA", (bw, bh), (232, 232, 236, 255))
    dev = Image.new("RGBA", (int(bw * 0.7), int(bh * 0.45)), (40, 40, 45, 255))
    dd = ImageDraw.Draw(dev)
    for k in range(6):  # nervures / étiquettes : nombreuses droites, toutes de travers
        y = 20 + k * (dev.height - 40) // 5
        dd.rectangle([15, y, dev.width - 15, y + 8], fill=(220, 220, 220, 255))
    dev = dev.rotate(content_tilt, resample=Image.Resampling.BICUBIC, expand=True)
    bag.paste(dev, ((bw - dev.width) // 2, (bh - dev.height) // 2), dev)
    bag = bag.rotate(bag_tilt, resample=Image.Resampling.BICUBIC, expand=True)
    cx, cy = (BAG[0] + BAG[2]) / 2, (BAG[1] + BAG[3]) / 2
    img.paste(bag, (int(cx - bag.width / 2), int(cy - bag.height / 2)), bag)
    return img.filter(ImageFilter.GaussianBlur(0.7))


@pytest.mark.parametrize("bag_tilt", [-2.0, 3.0])
def test_outline_angle_ignores_skewed_content(bag_tilt):
    from picturesrc.core.subject import analyze_photo, estimate_outline_tilt

    scene = np.asarray(make_bag_with_skewed_content(bag_tilt, 7.0))
    out = estimate_outline_tilt(scene)
    assert out is not None and out.source == "contour"
    assert out.angle == pytest.approx(-bag_tilt, abs=0.25)
    assert out.level == "high"

    analysis = analyze_photo(scene)
    # Le contenu de travers trompe l'estimation « lignes »…
    assert abs(analysis.lines.angle - (-bag_tilt)) > 1.0
    # …mais en mode « objet », c'est le contour du sachet qui fait foi
    est = analysis.estimate_for("subject")
    assert est.source == "contour" and est.angle == pytest.approx(-bag_tilt, abs=0.25)
    assert analysis.estimate_for("max_area") is analysis.lines


def test_outline_unusable_without_object():
    from picturesrc.core.subject import analyze_photo

    rng = np.random.default_rng(2)
    flat = np.clip(rng.normal(128, 4, (800, 1200, 3)), 0, 255).astype(np.uint8)
    analysis = analyze_photo(flat)
    assert analysis.estimate_for("subject") is analysis.lines
