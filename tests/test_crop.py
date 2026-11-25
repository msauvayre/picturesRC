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

"""Cadre de rognage réglable : jamais de coin vide, quel que soit le geste."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from picturesrc.core.crop import (
    auto_crop_box,
    box_from_drag,
    crop_is_valid,
    move_box,
    pixel_crop_box,
    refit_box,
    resize_box,
    to_norm,
)
from picturesrc.core.pipeline import ExportSettings, export_image, is_modified, straighten_image
from test_straighten import make_scene

W, H = 1200, 800
ANGLES = [-12.0, -2.5, 0.0, 0.7, 6.0]


def _min_after_crop(angle: float, norm) -> int:
    """Pixel le plus sombre d'une image blanche pivotée puis rognée (0 = coin vide)."""
    out, _ = straighten_image(Image.new("L", (W, H), 255), angle, "max_area", norm)
    return int(np.asarray(out).min())


def _small_box(angle: float):
    l, t, r, b = refit_box(W, H, angle, auto_crop_box(W, H, angle, "max_area"))
    return (l + 0.3 * (r - l), t + 0.3 * (b - t), r - 0.3 * (r - l), b - 0.3 * (b - t))


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("dx, dy", [(-5000, -5000), (5000, -5000), (5000, 5000), (-5000, 5000), (-5000, 0), (0, 5000)])
def test_move_stays_inside(angle, dx, dy):
    box = move_box(W, H, angle, _small_box(angle), dx, dy)
    assert crop_is_valid(W, H, angle, box)
    assert _min_after_crop(angle, to_norm(box, W, H)) > 200


@pytest.mark.parametrize("angle", ANGLES)
@pytest.mark.parametrize("handle", ["l", "r", "t", "b", "tl", "tr", "bl", "br"])
def test_resize_outward_stays_inside_and_keeps_opposite_edges(angle, handle):
    start = _small_box(angle)
    box = resize_box(W, H, angle, start, handle, -5000 if "l" in handle else 5000, -5000 if "t" in handle else 5000)
    assert crop_is_valid(W, H, angle, box)
    assert _min_after_crop(angle, to_norm(box, W, H)) > 200
    for i, side in enumerate("ltrb"):
        if side not in handle:
            assert box[i] == pytest.approx(start[i])  # les bords non saisis ne bougent pas
    # Le cadre a bien grandi du côté saisi
    assert (box[2] - box[0]) * (box[3] - box[1]) > (start[2] - start[0]) * (start[3] - start[1])


def test_resize_cannot_flip_or_collapse():
    start = _small_box(3.0)
    box = resize_box(W, H, 3.0, start, "l", start[2] + 500, 0, min_size=30)
    assert box[2] - box[0] == pytest.approx(30)


@pytest.mark.parametrize("angle", ANGLES)
def test_drawn_box_from_empty_corner(angle):
    # Tracé qui part d'un coin vide et finit au coin opposé : cadre valide et non dégénéré
    box = box_from_drag(W, H, angle, (0, 0), W, H)
    assert crop_is_valid(W, H, angle, box)
    assert box[2] - box[0] > 0.8 * W and box[3] - box[1] > 0.6 * H
    assert _min_after_crop(angle, to_norm(box, W, H)) > 200


def test_refit_after_angle_change():
    # Cadre collé au coin à 0°, puis on penche de 5° : il est réajusté sans coin vide
    box = (0.0, 0.0, 600.0, 400.0)
    assert crop_is_valid(W, H, 0.0, box)
    fitted = refit_box(W, H, 5.0, box)
    assert crop_is_valid(W, H, 5.0, fitted)
    assert fitted[0] < 300 and fitted[1] < 200  # reste du côté du coin choisi
    # Un cadre encore valide n'est pas touché
    inner = (500.0, 300.0, 700.0, 500.0)
    assert refit_box(W, H, 5.0, inner) == inner


def test_pixel_box_matches_normalized_crop_at_full_resolution():
    norm = (0.1, 0.2, 0.6, 0.9)
    assert pixel_crop_box(W, H, 0.0, norm) == (120, 160, 720, 720)
    # Même cadre normalisé, autre résolution : mêmes proportions
    l, t, r, b = pixel_crop_box(4800, 3200, 0.0, norm)
    assert (l, t, r, b) == (480, 640, 2880, 2880)


def test_manual_crop_without_rotation_is_exported(tmp_path):
    src = tmp_path / "a.jpg"
    make_scene(900, 600).save(src, quality=92)
    norm = (0.25, 0.25, 0.75, 0.75)
    assert is_modified(0.0, norm)
    out = export_image(src, 0.0, ExportSettings(output_dir=tmp_path / "out"), crop=norm)
    with Image.open(out) as im:
        assert im.size == (450, 300)
    meta = json.loads(out.with_name(out.name + ".picturesrc.json").read_text(encoding="utf-8"))
    t = meta["transformation"]
    assert t["rognage"] == "manuel"
    assert t["boite_rognage_px"] == [225, 150, 675, 450]
    assert t["cadre_manuel_normalise"] == [0.25, 0.25, 0.75, 0.75]


def test_manual_crop_ignored_in_no_crop_mode(tmp_path):
    src = tmp_path / "a.jpg"
    make_scene(600, 400).save(src)
    out = export_image(src, 0.0, ExportSettings(output_dir=tmp_path / "out", crop_mode="none"), crop=(0.1, 0.1, 0.5, 0.5))
    assert out.read_bytes() == src.read_bytes()  # copie identique : le cadre est ignoré


def test_manual_crop_with_rotation_has_no_empty_corner(tmp_path):
    src = tmp_path / "blanc.png"
    Image.new("RGB", (W, H), (255, 255, 255)).save(src)
    # Cadre demandé volontairement trop grand : il doit être réajusté à l'export
    out = export_image(src, 8.0, ExportSettings(output_dir=tmp_path / "out", write_sidecar=False), crop=(0.0, 0.0, 0.5, 1.0))
    with Image.open(out) as im:
        arr = np.asarray(im.convert("L"))
    assert arr.min() > 200
    assert arr.shape[1] < W // 2 + 1
