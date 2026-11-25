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

"""Enregistrement sur place : photo corrigée dans le dossier source, original déplacé."""

from __future__ import annotations

import errno
import json
import os

import numpy as np
import pytest
from PIL import Image

from picturesrc.core import inplace
from picturesrc.core.inplace import InPlaceSettings, apply_in_place
from picturesrc.core.orientation import RIGHT
from test_straighten import make_scene


@pytest.fixture
def photo(tmp_path):
    src = tmp_path / "photos"
    src.mkdir()
    p = src / "scène 1.jpg"  # accent et espace volontaires
    make_scene(900, 600, seed=4).rotate(3.0, resample=Image.Resampling.BICUBIC).save(p, quality=92)
    return p


def _files(d):
    return sorted(x.name for x in d.iterdir() if x.is_file())


def test_corrected_photo_replaces_original_which_is_moved(photo):
    original_bytes = photo.read_bytes()
    res = apply_in_place(photo, -3.0, InPlaceSettings(crop_mode="aspect"))

    assert res.action == "corrigee" and res.path == photo
    assert res.original == photo.parent / "originaux" / photo.name
    assert res.original.read_bytes() == original_bytes  # original intact, déplacé
    assert photo.read_bytes() != original_bytes  # version corrigée à sa place
    assert _files(photo.parent) == [photo.name]  # ni temporaire ni JSON dans le dossier source

    meta = json.loads((res.original.parent / (photo.name + ".picturesrc.json")).read_text(encoding="utf-8"))
    assert meta["source"]["chemin"] == str(res.original.resolve())
    assert meta["sortie"]["chemin"] == str(photo.resolve())
    assert meta["enregistrement"]["mode"] == "sur_place"


def test_unmodified_photo_is_left_alone(photo):
    before = photo.read_bytes()
    res = apply_in_place(photo, 0.0, InPlaceSettings(crop_mode="none"))
    assert res.action == "inchangee" and res.original is None
    assert photo.read_bytes() == before
    assert not (photo.parent / "originaux").exists()


def test_custom_originals_dir_and_name_clash(photo, tmp_path):
    store = tmp_path / "archives"
    store.mkdir()
    (store / photo.name).write_bytes(b"autre fichier")  # nom déjà pris
    res = apply_in_place(photo, -3.0, InPlaceSettings(originals_dir=store, write_sidecar=False))
    assert res.original.parent == store and res.original.name != photo.name
    assert (store / photo.name).read_bytes() == b"autre fichier"  # jamais écrasé
    assert not list(store.glob("*.json"))


def test_resave_starts_again_from_original(photo):
    """Ré-enregistrer : recalcul depuis l'original déplacé, pas de double rotation."""
    s = InPlaceSettings(crop_mode="aspect")
    first = apply_in_place(photo, -3.0, s)
    once = np.asarray(Image.open(photo)).astype(int)
    second = apply_in_place(photo, -3.0, s, original=first.original)
    assert second.original == first.original
    assert len(list(first.original.parent.glob("*.jpg"))) == 1  # pas de 2e « original »
    again = np.asarray(Image.open(photo)).astype(int)
    assert once.shape == again.shape and np.abs(once - again).mean() < 2  # même résultat (au JPEG près)


def test_resave_without_correction_restores_original(photo):
    original_bytes = photo.read_bytes()
    first = apply_in_place(photo, 0.0, InPlaceSettings(), quarter=RIGHT)
    res = apply_in_place(photo, 0.0, InPlaceSettings(), original=first.original)
    assert res.action == "restauree" and res.original is None
    assert photo.read_bytes() == original_bytes
    assert not first.original.exists()
    assert not list(first.original.parent.iterdir())  # JSON retiré aussi


def test_failure_puts_original_back(photo, monkeypatch):
    original_bytes = photo.read_bytes()

    def broken_replace(a, b):
        raise PermissionError("fichier ouvert dans une autre application")

    monkeypatch.setattr(inplace.os, "replace", broken_replace)
    with pytest.raises(PermissionError):
        apply_in_place(photo, -3.0, InPlaceSettings())
    monkeypatch.undo()
    assert photo.read_bytes() == original_bytes  # l'original a repris sa place
    assert _files(photo.parent) == [photo.name]  # temporaire supprimé
    assert not list((photo.parent / "originaux").iterdir())


def test_move_to_other_disk_is_verified_copy(photo, monkeypatch, tmp_path):
    """Dossier des originaux sur un autre disque : copie + contrôle SHA-256 + suppression."""
    original_bytes = photo.read_bytes()

    def cross_device(a, b):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(inplace.os, "rename", cross_device)
    res = apply_in_place(photo, -3.0, InPlaceSettings(originals_dir=tmp_path / "autre_disque"))
    assert res.original.read_bytes() == original_bytes
    assert photo.exists() and photo.read_bytes() != original_bytes
