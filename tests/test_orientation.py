"""Rotations de 90° (quarts de tour) : cohérence entre aperçu, données et export."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from picturesrc.core.crop import to_norm
from picturesrc.core.orientation import (
    LEFT,
    RIGHT,
    normalize_quarter,
    rotate_array_quarter,
    rotate_image_quarter,
    rotate_norm_box,
    rotate_norm_points,
    rotate_subject,
)
from picturesrc.core.pipeline import ExportSettings, export_image, is_modified
from picturesrc.core.subject import detect_subject, subject_crop_box
from test_subject import make_desk_scene


def _gradient(w=7, h=4) -> np.ndarray:
    return (np.arange(w * h, dtype=np.uint8).reshape(h, w) * 3)


@pytest.mark.parametrize("q", [0, 1, 2, 3])
def test_numpy_and_pil_turn_the_same_way(q):
    arr = _gradient()
    via_pil = np.asarray(rotate_image_quarter(Image.fromarray(arr), q))
    assert np.array_equal(rotate_array_quarter(arr, q), via_pil)


def test_left_is_counterclockwise_and_right_clockwise():
    arr = np.zeros((4, 6), np.uint8)
    arr[0, 5] = 255  # coin haut-droit
    # 90° à gauche (antihoraire) : le coin haut-droit passe en haut-gauche
    assert rotate_array_quarter(arr, LEFT)[0, 0] == 255
    # 90° à droite (horaire) : il passe en bas-droit
    assert rotate_array_quarter(arr, RIGHT)[-1, -1] == 255


@pytest.mark.parametrize("q", [1, 2, 3, -1])
def test_norm_points_follow_pixels(q):
    h, w = 40, 70
    arr = np.zeros((h, w), np.uint8)
    y, x = 9, 52
    arr[y, x] = 255
    rot = rotate_array_quarter(arr, q)
    ny, nx = np.argwhere(rot == 255)[0]
    # centre du pixel en coordonnées normalisées
    (px, py), = rotate_norm_points(np.array([[(x + 0.5) / w, (y + 0.5) / h]]), q)
    assert px * rot.shape[1] == pytest.approx(nx + 0.5)
    assert py * rot.shape[0] == pytest.approx(ny + 0.5)


def test_four_turns_is_identity_and_box_stays_ordered():
    box = (0.1, 0.2, 0.6, 0.9)
    b = box
    for _ in range(4):
        b = rotate_norm_box(b, RIGHT)
        assert b[0] < b[2] and b[1] < b[3]
    assert b == pytest.approx(box)
    assert normalize_quarter(-1) == 3 and normalize_quarter(5) == 1


@pytest.mark.parametrize("q", [LEFT, RIGHT, 2])
def test_subject_box_commutes_with_quarter_turn(q):
    """Transposer l'objet détecté donne le même cadre que tourner le cadre."""
    img = np.asarray(make_desk_scene(tilt=4.0, seed=2, glare=False).convert("RGB"))
    angle = -4.0
    subject = detect_subject(img, angle)
    assert subject is not None
    h, w = img.shape[:2]
    expected = rotate_norm_box(to_norm(subject_crop_box(w, h, angle, subject), w, h), q)
    rh, rw = rotate_array_quarter(img, q).shape[:2]
    got = to_norm(subject_crop_box(rw, rh, angle, rotate_subject(subject, q)), rw, rh)
    assert got == pytest.approx(expected, abs=0.01)


def test_quarter_alone_counts_as_modification():
    assert not is_modified(0.0)
    assert is_modified(0.0, quarter=RIGHT)
    assert not is_modified(0.0, quarter=4)


@pytest.mark.parametrize("q,label", [(LEFT, "90° à gauche"), (RIGHT, "90° à droite"), (2, "180°")])
def test_export_quarter_is_lossless_without_other_change(tmp_path, q, label):
    src = tmp_path / "photo.png"  # PNG : comparaison exacte, sans compression avec perte
    arr = np.random.default_rng(0).integers(0, 255, (60, 90, 3), dtype=np.uint8)
    Image.fromarray(arr).save(src)

    out = export_image(src, 0.0, ExportSettings(output_dir=tmp_path / "out", crop_mode="none"), quarter=q)
    assert np.array_equal(np.asarray(Image.open(out)), rotate_array_quarter(arr, q))
    meta = json.loads(out.with_name(out.name + ".picturesrc.json").read_text(encoding="utf-8"))
    assert meta["transformation"]["quart_de_tour"] == label
    assert meta["transformation"]["quart_de_tour_deg_antihoraire"] == 90 * normalize_quarter(q)


def test_export_quarter_then_fine_rotation(tmp_path):
    src = tmp_path / "photo.png"
    Image.new("RGB", (300, 200), (200, 200, 200)).save(src)
    out = export_image(src, 3.0, ExportSettings(output_dir=tmp_path / "out", crop_mode="aspect"), quarter=RIGHT)
    res = Image.open(out)
    assert res.height > res.width  # paysage devenu portrait
    assert np.asarray(res).min() > 150  # pas de coin vide après la rotation fine
