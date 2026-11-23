"""Tests du moteur sur des scènes synthétiques inclinées d'un angle connu."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from picturesrc.core.pipeline import ExportSettings, export_image, straighten_image
from picturesrc.core.straighten import estimate_tilt, inscribed_crop_size, measure_line_angle


def make_scene(w: int = 1600, h: int = 1067, seed: int = 0) -> Image.Image:
    """Scène « urbaine » : ciel, horizon, bâtiments, fenêtres, un peu de bruit."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), (150, 190, 230))
    d = ImageDraw.Draw(img)
    horizon = int(h * 0.6)
    d.rectangle([0, horizon, w, h], fill=(90, 110, 70))
    x = 0
    while x < w:
        bw = int(rng.integers(w // 14, w // 7))
        bh = int(rng.integers(h // 6, h // 2))
        color = tuple(int(c) for c in rng.integers(60, 200, 3))
        d.rectangle([x, horizon - bh, x + bw, horizon], fill=color, outline=(30, 30, 30), width=3)
        for wy in range(horizon - bh + 20, horizon - 30, 45):
            for wx in range(x + 12, x + bw - 25, 35):
                d.rectangle([wx, wy, wx + 16, wy + 22], fill=(40, 40, 60))
        x += bw + int(rng.integers(5, 40))
    noise = rng.normal(0, 6, (h, w, 3))
    arr = np.clip(np.asarray(img, dtype=np.float64) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).filter(ImageFilter.GaussianBlur(0.8))


@pytest.mark.parametrize("true_tilt", [-7.5, -3.2, -0.8, -0.3, 0.0, 0.4, 1.4, 4.0, 9.0])
def test_estimate_recovers_known_tilt(true_tilt):
    scene = make_scene(seed=int(abs(true_tilt) * 10))
    # On penche la scène de true_tilt : la correction attendue est -true_tilt
    tilted = scene.rotate(true_tilt, resample=Image.Resampling.BICUBIC, fillcolor=(0, 0, 0))
    cw, ch = inscribed_crop_size(*scene.size, true_tilt)
    l, t = (scene.width - cw) // 2, (scene.height - ch) // 2
    tilted = tilted.crop((l, t, l + cw, t + ch))  # comme une vraie photo : pas de coins noirs

    est = estimate_tilt(np.asarray(tilted))
    assert est.angle == pytest.approx(-true_tilt, abs=0.15)
    assert est.level == "high"


def test_featureless_image_has_low_confidence():
    rng = np.random.default_rng(1)
    flat = np.clip(rng.normal(128, 4, (800, 1200, 3)), 0, 255).astype(np.uint8)
    est = estimate_tilt(flat)
    assert est.level == "low"


def test_measure_line_angle_convention():
    # Ligne qui « monte » vers la droite (y vers le bas) -> rotation horaire (négative)
    assert measure_line_angle(0, 100, 1000, 100 - 52.4) == pytest.approx(-3.0, abs=0.01)
    # Verticale dont le haut penche à droite -> rotation antihoraire
    assert measure_line_angle(0, 1000, 52.4, 0) == pytest.approx(3.0, abs=0.01)


@pytest.mark.parametrize("mode", ["aspect", "max_area"])
@pytest.mark.parametrize("angle", [-12.0, -2.5, 0.7, 6.0])
def test_crop_has_no_empty_corners(mode, angle):
    w, h = 1200, 800
    white = Image.new("L", (w, h), 255)
    out, box = straighten_image(white, angle, mode)
    arr = np.asarray(out)
    assert arr.min() > 200, "zone vide (noire) dans l'image rognée"
    if mode == "aspect":
        assert abs(out.width / out.height - w / h) < 0.01


def test_export_writes_new_file_and_sidecar(tmp_path):
    src = tmp_path / "photo é.jpg"  # accent volontaire (chemins Windows)
    make_scene(800, 533).save(src, quality=92)
    before = src.read_bytes()

    out = export_image(src, -2.0, ExportSettings(output_dir=tmp_path / "out"))
    assert out.exists() and out != src
    assert src.read_bytes() == before, "l'original ne doit jamais être modifié"

    meta = json.loads(out.with_name(out.name + ".picturesrc.json").read_text(encoding="utf-8"))
    assert meta["transformation"]["rotation_deg_antihoraire"] == -2.0
    assert len(meta["source"]["sha256"]) == 64

    # Un second export ne doit pas écraser le premier
    out2 = export_image(src, -2.0, ExportSettings(output_dir=tmp_path / "out"))
    assert out2 != out


def test_cli_saves_in_place_and_moves_originals(tmp_path):
    from picturesrc.cli import main

    src = tmp_path / "photos"
    src.mkdir()
    make_scene(1000, 667, seed=3).rotate(3.0, resample=Image.Resampling.BICUBIC).save(src / "penchee.jpg")
    Image.new("RGB", (600, 400), (120, 120, 120)).save(src / "uniforme.jpg")  # aucune ligne -> 0°
    before = {p.name: p.read_bytes() for p in src.iterdir()}

    assert main([str(src), "--no-json", "--crop", "aspect"]) == 0
    # Dossier source : la photo corrigée a pris la place de l'original, l'autre est intacte
    assert sorted(p.name for p in src.iterdir() if p.is_file()) == ["penchee.jpg", "uniforme.jpg"]
    assert (src / "penchee.jpg").read_bytes() != before["penchee.jpg"]
    assert (src / "uniforme.jpg").read_bytes() == before["uniforme.jpg"]
    # Seul l'original de la photo corrigée est déplacé
    originals = src / "originaux"
    assert sorted(p.name for p in originals.iterdir()) == ["penchee.jpg"]
    assert (originals / "penchee.jpg").read_bytes() == before["penchee.jpg"]


def test_unmodified_sidecar_marks_identical_copy(tmp_path):
    src = tmp_path / "a.jpg"
    make_scene(600, 400).save(src)
    out = export_image(src, 0.0, ExportSettings(output_dir=tmp_path / "out"))
    meta = json.loads(out.with_name(out.name + ".picturesrc.json").read_text(encoding="utf-8"))
    assert meta["transformation"]["copie_identique"] is True
    assert meta["source"]["sha256"] == meta["sortie"]["sha256"]
