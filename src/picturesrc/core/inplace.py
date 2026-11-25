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

"""Enregistrement « sur place » : la photo corrigée prend la place de
l'original dans le dossier source, et l'original est déplacé.

Par défaut l'original va dans un sous-dossier ``originaux`` du dossier de
la photo ; un dossier unique peut être imposé (``originals_dir``). Le
fichier ``.picturesrc.json`` (optionnel) est écrit à côté de l'original
déplacé : le dossier source ne contient que les photos.

Ordre des opérations (aucune étape ne laisse la photo sans version complète) :
    1. la correction est écrite dans un fichier temporaire du dossier source ;
    2. l'original est déplacé (renommage, ou copie vérifiée par SHA-256 puis
       suppression si le dossier des originaux est sur un autre disque) ;
    3. le fichier temporaire est renommé au nom de l'original.
En cas d'échec, l'original est remis à sa place et le temporaire supprimé.

Ré-enregistrement d'une photo déjà traitée (``original`` fourni) : la
correction est recalculée depuis l'original déplacé, jamais depuis la
photo déjà corrigée (pas de double rotation). Si la photo n'est plus
modifiée, l'original est remis à sa place.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..i18n import tr
from .crop import Box
from .imageio import load_image, save_image
from .orientation import normalize_quarter, rotate_image_quarter
from .pipeline import _unique_path, build_sidecar, crops_to_subject, is_modified, sha256_file, straighten_image
from .straighten import TiltEstimate
from .subject import SubjectEstimate

ORIGINALS_DIRNAME = "originaux"
SIDECAR_SUFFIX = ".picturesrc.json"


@dataclass
class InPlaceSettings:
    # None = sous-dossier « originaux » du dossier de chaque photo
    originals_dir: Path | None = None
    crop_mode: str = "max_area"
    jpeg_quality: int = 95
    write_sidecar: bool = True


@dataclass
class InPlaceResult:
    path: Path  # la photo dans son dossier source
    original: Path | None  # emplacement actuel de l'original (None = à sa place)
    action: str  # "corrigee" | "inchangee" | "restauree"


def originals_dir_for(path: Path, originals_dir: Path | None = None) -> Path:
    return Path(originals_dir) if originals_dir is not None else Path(path).parent / ORIGINALS_DIRNAME


def _tmp_path(path: Path) -> Path:
    # Même extension : save_image en déduit le format
    return _unique_path(path.with_name(f".{path.stem}.picturesrc-tmp{path.suffix}"))


def _move_verified(src: Path, dst: Path) -> None:
    """Déplace ``src`` vers ``dst`` (qui ne doit pas exister).

    Même disque : simple renommage (atomique). Disque différent : copie,
    vérification SHA-256, puis suppression de la source.
    """
    try:
        os.rename(src, dst)
        return
    except OSError:
        if dst.exists() or not src.exists():
            raise
    shutil.copy2(src, dst)
    if sha256_file(src) != sha256_file(dst):
        dst.unlink(missing_ok=True)
        raise OSError(tr("error.copy_mismatch", src=src, dst=dst))
    src.unlink()


def _render(pixels_from: Path, out_path: Path, angle, settings, crop, subject, quarter):
    img, exif_bytes, icc = load_image(pixels_from)
    original_size = img.size
    img = rotate_image_quarter(img, quarter)
    result, box = straighten_image(img, angle, settings.crop_mode, crop, subject)
    save_image(result, out_path, exif_bytes, icc, settings.jpeg_quality)
    return original_size, result.size, box


def _sidecar_path(original: Path) -> Path:
    return original.with_name(original.name + SIDECAR_SUFFIX)


def apply_in_place(
    path: Path,
    angle: float,
    settings: InPlaceSettings,
    estimate: TiltEstimate | None = None,
    crop: Box | None = None,
    subject: SubjectEstimate | None = None,
    quarter: int = 0,
    original: Path | None = None,
) -> InPlaceResult:
    """Remplace ``path`` par sa version corrigée et déplace l'original.

    ``original`` : emplacement de l'original si la photo a déjà été traitée
    (renvoyé par un appel précédent) ; ``None`` = ``path`` est l'original.
    Les autres paramètres sont ceux de ``pipeline.export_image``.
    """
    path = Path(path)
    if settings.crop_mode == "none":
        crop = None
    to_subject = crop is None and crops_to_subject(settings.crop_mode, subject)
    quarter = normalize_quarter(quarter)
    modified = is_modified(angle, crop, to_subject, quarter)

    if original is not None:
        original = Path(original)
        if not original.exists():
            raise FileNotFoundError(tr("error.original_missing", path=original))

    # --- rien à faire, ou retour à l'original --------------------------------
    if not modified:
        if original is None:
            return InPlaceResult(path, None, "inchangee")
        tmp = _tmp_path(path)
        try:
            shutil.copy2(original, tmp)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        if sha256_file(path) != sha256_file(original):
            raise OSError(tr("error.restore_mismatch", path=path))
        original.unlink()
        _sidecar_path(original).unlink(missing_ok=True)
        return InPlaceResult(path, None, "restauree")

    # --- correction -----------------------------------------------------------
    tmp = _tmp_path(path)
    moved_to: Path | None = None
    try:
        pixels_from = original if original is not None else path
        original_size, out_size, box = _render(pixels_from, tmp, angle, settings, crop, subject, quarter)
        if original is None:
            dest_dir = originals_dir_for(path, settings.originals_dir)
            dest_dir.mkdir(parents=True, exist_ok=True)
            moved_to = _unique_path(dest_dir / path.name)
            _move_verified(path, moved_to)
            original = moved_to
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        if moved_to is not None and moved_to.exists() and not path.exists():
            _move_verified(moved_to, path)  # l'original reprend sa place
        raise

    sidecar = _sidecar_path(original)
    if settings.write_sidecar:
        data = build_sidecar(
            original, path, original_size, out_size, angle, quarter, crop, to_subject,
            settings.crop_mode, box, True, estimate, subject,
        )
        data["enregistrement"] = {
            "mode": "sur_place",
            "original_deplace_vers": str(original.resolve()),
            "photo_corrigee": str(path.resolve()),
        }
        sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        sidecar.unlink(missing_ok=True)  # pas de JSON périmé d'un enregistrement précédent
    return InPlaceResult(path, original, "corrigee")
