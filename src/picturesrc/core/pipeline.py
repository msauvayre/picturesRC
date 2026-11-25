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

"""Application d'une correction en pleine résolution et export.

L'original n'est jamais modifié. Chaque export peut être accompagné d'un
fichier ``<nom>.picturesrc.json`` qui décrit la transformation appliquée
(angle, rognage, empreintes SHA-256), pour pouvoir la vérifier ou la rejouer.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from .. import __version__
from .crop import Box, pixel_crop_box, to_norm
from .imageio import load_image, save_image
from .orientation import QUARTER_LABELS, normalize_quarter, rotate_image_quarter
from .straighten import TiltEstimate, inscribed_crop_size
from .subject import SubjectEstimate, subject_crop_box

# "subject" = autour de l'objet détecté au centre (repli : "max_area")
CROP_MODES = ("subject", "aspect", "max_area", "none")

# En dessous de cet angle (degrés), la photo est considérée comme non
# modifiée : elle est copiée telle quelle (octet pour octet, sans réencodage).
MIN_ROTATION = 0.01


def is_modified(angle: float, crop: Box | None = None, subject: bool = False, quarter: int = 0) -> bool:
    """Vrai si la correction change réellement l'image : rotation non nulle,
    quart de tour, cadre de rognage choisi à la main, ou rognage autour d'un
    objet détecté (``subject=True``)."""
    return (
        abs(angle) >= MIN_ROTATION - 1e-9
        or crop is not None
        or subject
        or normalize_quarter(quarter) != 0
    )


def crops_to_subject(crop_mode: str, subject: SubjectEstimate | None) -> bool:
    """Vrai si l'image sera rognée autour d'un objet détecté."""
    return crop_mode == "subject" and subject is not None


@dataclass
class ExportSettings:
    output_dir: Path
    crop_mode: str = "max_area"  # "subject" | "aspect" | "max_area" | "none"
    jpeg_quality: int = 95
    suffix: str = ""  # ex. "_redresse" ; vide = même nom que l'original
    write_sidecar: bool = True


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    i = 1
    while True:
        candidate = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def straighten_image(
    img: Image.Image,
    angle: float,
    crop_mode: str = "max_area",
    crop: Box | None = None,
    subject: SubjectEstimate | None = None,
) -> tuple[Image.Image, tuple[int, int, int, int] | None]:
    """Fait pivoter ``img`` de ``angle`` puis la rogne.

    ``crop`` : cadre choisi par l'utilisateur, normalisé entre 0 et 1
    (``(gauche, haut, droite, bas)`` dans l'image pivotée). S'il est fourni,
    il remplace le cadre automatique ; il est réajusté si besoin pour ne
    jamais inclure de coin vide. Il est ignoré en mode ``"none"``.

    ``subject`` : objet détecté (mode ``"subject"``) ; sans objet, ce mode
    se replie sur ``"max_area"``.

    Renvoie ``(image, crop_box)``. ``crop_box`` = ``(gauche, haut, droite, bas)``
    en pixels dans l'image pivotée, ou ``None`` sans rognage.
    """
    if crop_mode not in CROP_MODES:
        raise ValueError(f"mode de rognage inconnu : {crop_mode!r}")
    if crop_mode == "none":
        crop = None
    if crop_mode == "subject" and subject is None:
        crop_mode = "max_area"
    rotate = abs(angle) >= 1e-6
    if not rotate and crop is None and crop_mode != "subject":
        return img.copy(), None

    if rotate:
        fill = (0, 0, 0, 0) if img.mode == "RGBA" else 0
        rotated = img.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=fill)
    else:
        rotated = img
    if crop_mode == "none":
        return rotated, None

    w, h = img.size
    if crop is not None:
        box = pixel_crop_box(w, h, angle, crop)
    elif crop_mode == "subject":
        box = pixel_crop_box(w, h, angle, to_norm(subject_crop_box(w, h, angle, subject), w, h))
    else:
        cw, ch = inscribed_crop_size(w, h, angle, crop_mode)
        left = (w - cw) // 2
        top = (h - ch) // 2
        box = (left, top, left + cw, top + ch)
    return rotated.crop(box), box


def build_sidecar(
    source: Path,
    out_path: Path,
    original_size: tuple[int, int],
    out_size: tuple[int, int],
    angle: float,
    quarter: int,
    crop: Box | None,
    to_subject: bool,
    crop_mode: str,
    box: tuple[int, int, int, int] | None,
    modified: bool,
    estimate: TiltEstimate | None,
    subject: SubjectEstimate | None,
    source_sha256: str | None = None,
) -> dict:
    """Contenu du fichier ``.picturesrc.json`` (traçabilité d'une correction)."""
    return {
        "logiciel": f"picturesRC {__version__}",
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {
            "chemin": str(source.resolve()),
            "sha256": source_sha256 or sha256_file(source),
            "taille_px": list(original_size),
        },
        "sortie": {
            "chemin": str(out_path.resolve()),
            "sha256": sha256_file(out_path),
            "taille_px": list(out_size),
        },
        "transformation": {
            "orientation_exif_appliquee": True,
            # Ordre : orientation EXIF, quart de tour, rotation fine, rognage
            "quart_de_tour": QUARTER_LABELS[quarter],
            "quart_de_tour_deg_antihoraire": 90 * quarter,
            "rotation_deg_antihoraire": round(angle, 3),
            "interpolation": "bicubique",
            "rognage": "manuel" if crop is not None else ("objet" if to_subject else
                        ("max_area" if crop_mode == "subject" else crop_mode)),
            "boite_rognage_px": list(box) if box else None,
            "cadre_manuel_normalise": [round(v, 6) for v in crop] if crop is not None else None,
        }
        if modified
        else {"copie_identique": True, "rotation_deg_antihoraire": 0.0},
        "estimation_auto": None
        if estimate is None
        else {
            "methode": estimate.source,  # "lignes" ou "contour" (bords de l'objet)
            "angle_deg": estimate.angle,
            "confiance": estimate.confidence,
            "segments_concordants": estimate.n_support,
            "segments_candidats": estimate.n_lines,
            "modifie_manuellement": abs(estimate.angle - angle) > 1e-3,
        },
        "detection_objet": None
        if subject is None or crop_mode != "subject"
        else {
            "confiance": subject.confidence,
            "part_de_l_image": subject.area_fraction,
        },
    }


def export_image(
    source: Path,
    angle: float,
    settings: ExportSettings,
    estimate: TiltEstimate | None = None,
    crop: Box | None = None,
    subject: SubjectEstimate | None = None,
    quarter: int = 0,
) -> Path:
    """Corrige ``source`` et l'écrit dans ``settings.output_dir``. Renvoie le chemin produit.

    ``quarter`` : quarts de tour antihoraires (1 = 90° à gauche, 3 = 90° à
    droite, cf. ``orientation``), appliqués avant tout le reste. ``angle``,
    ``crop`` et ``subject`` sont exprimés dans le repère de l'image tournée.

    ``crop`` : cadre de rognage choisi à la main (normalisé, cf.
    ``straighten_image``) ; ``None`` = cadre automatique selon
    ``settings.crop_mode``.
    ``subject`` : objet détecté, utilisé en mode ``"subject"``.

    Si la photo n'est pas modifiée (rotation nulle, pas de cadre manuel), le fichier est copié
    à l'identique (``shutil.copy2``) : aucun réencodage, aucune perte,
    même empreinte SHA-256 que l'original.
    """
    source = Path(source)
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = _unique_path(out_dir / f"{source.stem}{settings.suffix}{source.suffix}")
    if out_path.resolve() == source.resolve():  # ceinture et bretelles
        raise ValueError("Le fichier de sortie ne peut pas remplacer l'original.")

    if settings.crop_mode == "none":
        crop = None
    to_subject = crop is None and crops_to_subject(settings.crop_mode, subject)
    quarter = normalize_quarter(quarter)
    modified = is_modified(angle, crop, to_subject, quarter)
    if modified:
        img, exif_bytes, icc = load_image(source)
        original_size = img.size
        img = rotate_image_quarter(img, quarter)
        result, box = straighten_image(img, angle, settings.crop_mode, crop, subject)
        save_image(result, out_path, exif_bytes, icc, settings.jpeg_quality)
        out_size = result.size
    else:
        shutil.copy2(source, out_path)
        with Image.open(source) as im:
            original_size = out_size = im.size
        box = None

    if settings.write_sidecar:
        sidecar = build_sidecar(
            source, out_path, original_size, out_size, angle, quarter, crop, to_subject,
            settings.crop_mode, box, modified, estimate, subject,
        )
        sidecar_path = out_path.with_name(out_path.name + ".picturesrc.json")
        sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")

    return out_path
