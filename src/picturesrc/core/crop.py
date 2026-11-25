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

"""Cadre de rognage réglable par l'utilisateur.

Après une rotation de ``angle`` (``expand=False``, même convention que le
reste du projet), seule une partie de l'image contient des pixels : le
rectangle d'origine pivoté autour du centre. Un cadre de rognage est
« valide » si ses quatre coins tombent dans cette zone ; il n'y a alors
aucun coin vide dans l'image exportée.

Représentations :
    * ``box``  = ``(gauche, haut, droite, bas)`` en pixels (flottants) dans
      l'image pivotée, qui a la même taille ``w × h`` que l'originale ;
    * ``norm`` = la même boîte divisée par ``(w, h, w, h)``, entre 0 et 1.
      C'est la forme mémorisée pour chaque photo : elle sert telle quelle
      sur l'aperçu réduit et sur l'image en pleine résolution.

La zone valide est convexe. Quand l'utilisateur déplace ou redimensionne
le cadre, on part donc d'un cadre valide et on avance vers le cadre demandé
aussi loin que possible (dichotomie) : le cadre « glisse » contre le bord
de la zone utile au lieu de la dépasser.
"""

from __future__ import annotations

import math

from .straighten import inscribed_crop_size

Box = tuple[float, float, float, float]

# Marge (pixels de l'image traitée) entre le cadre et le bord de la zone
# utile : évite les liserés sombres dus à l'interpolation bicubique.
DEFAULT_MARGIN = 1.0


def _inside(w: float, h: float, angle: float, x: float, y: float, margin: float) -> bool:
    """Vrai si le point ``(x, y)`` de l'image pivotée provient de l'image d'origine.

    Reprend la transformation inverse de ``PIL.Image.rotate`` (centre
    ``(w/2, h/2)``) : on calcule d'où vient le point, puis on vérifie qu'il
    tombe dans l'original, à ``margin`` près.
    """
    t = -math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    cx, cy = w / 2.0, h / 2.0
    dx, dy = x - cx, y - cy
    sx = c * dx + s * dy + cx
    sy = -s * dx + c * dy + cy
    eps = 1e-6
    return margin - eps <= sx <= w - margin + eps and margin - eps <= sy <= h - margin + eps


def crop_is_valid(w: float, h: float, angle: float, box: Box, margin: float = DEFAULT_MARGIN) -> bool:
    """Vrai si ``box`` ne contient aucun coin vide après rotation de ``angle``."""
    l, t, r, b = box
    if r < l or b < t:
        return False
    if abs(angle) < 1e-6:
        margin = 0.0  # pas de rotation, donc pas d'interpolation ni de liseré
    return all(_inside(w, h, angle, x, y, margin) for x, y in ((l, t), (r, t), (r, b), (l, b)))


def auto_crop_box(w: int, h: int, angle: float, mode: str = "max_area") -> Box:
    """Cadre automatique centré (``"aspect"`` ou ``"max_area"``), en pixels."""
    cw, ch = inscribed_crop_size(w, h, angle, mode)
    left, top = (w - cw) // 2, (h - ch) // 2
    return (float(left), float(top), float(left + cw), float(top + ch))


def _seed_box(w: float, h: float, angle: float, margin: float) -> Box:
    """Cadre automatique « surface maximale », garanti valide (point de départ sûr)."""
    box = auto_crop_box(int(round(w)), int(round(h)), angle, "max_area")
    if crop_is_valid(w, h, angle, box, margin):
        return box
    l, t, r, b = box
    k = margin + 0.5
    shrunk = (l + k, t + k, r - k, b - k)
    if crop_is_valid(w, h, angle, shrunk, margin):
        return shrunk
    cx, cy = w / 2.0, h / 2.0
    return (cx, cy, cx, cy)


def _max_step(w: float, h: float, angle: float, box: Box, delta: Box, margin: float) -> float:
    """Plus grand ``t`` de [0, 1] tel que ``box + t·delta`` reste valide.

    ``box`` doit être valide. La zone valide étant convexe, l'ensemble des
    ``t`` acceptables est un intervalle [0, t*] : une dichotomie suffit.
    """

    def at(t: float) -> Box:
        return tuple(v + t * d for v, d in zip(box, delta))  # type: ignore[return-value]

    if crop_is_valid(w, h, angle, at(1.0), margin):
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        if crop_is_valid(w, h, angle, at(mid), margin):
            lo = mid
        else:
            hi = mid
    return lo


def constrain_box(
    w: float,
    h: float,
    angle: float,
    old: Box,
    new: Box,
    margin: float = DEFAULT_MARGIN,
    rigid: bool = False,
) -> Box:
    """Rapproche le cadre valide ``old`` de ``new`` sans jamais sortir de la zone utile.

    On avance d'abord sur tous les bords à la fois (le cadre garde la forme
    voulue tant que possible), puis séparément, en deux passes :
        * ``rigid=True`` (déplacement) : axe par axe, les bords opposés
          restant solidaires — le cadre glisse le long de la zone utile ;
        * ``rigid=False`` (redimensionnement, réajustement) : bord par bord —
          un bord qui bute n'empêche pas les autres d'avancer.
    """
    box = tuple(old)
    full = tuple(n - o for n, o in zip(new, box))
    t = _max_step(w, h, angle, box, full, margin)  # type: ignore[arg-type]
    box = tuple(v + t * d for v, d in zip(box, full))
    groups = ((0, 2), (1, 3)) if rigid else ((0,), (2,), (1,), (3,))
    for _ in range(2):
        for group in groups:
            delta = [0.0, 0.0, 0.0, 0.0]
            for i in group:
                delta[i] = new[i] - box[i]
            if not any(delta):
                continue
            t = _max_step(w, h, angle, box, tuple(delta), margin)  # type: ignore[arg-type]
            box = tuple(v + t * d for v, d in zip(box, delta))
    return box  # type: ignore[return-value]


def move_box(w: float, h: float, angle: float, box: Box, dx: float, dy: float, margin: float = DEFAULT_MARGIN) -> Box:
    """Translation du cadre, bloquée au bord de la zone utile."""
    l, t, r, b = box
    return constrain_box(w, h, angle, box, (l + dx, t + dy, r + dx, b + dy), margin, rigid=True)


def resize_box(
    w: float,
    h: float,
    angle: float,
    box: Box,
    handle: str,
    x: float,
    y: float,
    min_size: float = 1.0,
    margin: float = DEFAULT_MARGIN,
) -> Box:
    """Déplace le bord ou le coin ``handle`` du cadre vers ``(x, y)``.

    ``handle`` combine ``"l"``, ``"r"``, ``"t"``, ``"b"`` (ex. ``"tl"`` pour le
    coin haut-gauche). Les bords opposés restent fixes ; le cadre ne peut
    pas se retourner ni devenir plus petit que ``min_size``.
    """
    l, t, r, b = box
    nl, nt, nr, nb = l, t, r, b
    if "l" in handle:
        nl = min(x, r - min_size)
    if "r" in handle:
        nr = max(x, l + min_size)
    if "t" in handle:
        nt = min(y, b - min_size)
    if "b" in handle:
        nb = max(y, t + min_size)
    return constrain_box(w, h, angle, box, (nl, nt, nr, nb), margin)


def box_from_drag(
    w: float,
    h: float,
    angle: float,
    anchor: tuple[float, float],
    x: float,
    y: float,
    margin: float = DEFAULT_MARGIN,
) -> Box:
    """Cadre tracé à la souris depuis ``anchor`` jusqu'à ``(x, y)``.

    Le point de départ est d'abord ramené dans le cadre automatique
    « surface maximale » (toujours valide), puis le coin opposé suit la
    souris, borné par la zone utile.
    """
    al, at, ar, ab = _seed_box(w, h, angle, margin)
    ax = min(max(anchor[0], al), ar)
    ay = min(max(anchor[1], at), ab)
    seed = (ax, ay, ax, ay)
    target = (min(ax, x), min(ay, y), max(ax, x), max(ay, y))
    return constrain_box(w, h, angle, seed, target, margin)


def refit_box(w: float, h: float, angle: float, box: Box, margin: float = DEFAULT_MARGIN) -> Box:
    """Adapte un cadre après un changement d'angle.

    Un cadre encore valide est conservé tel quel. Sinon, on part du cadre
    automatique (valide pour le nouvel angle) et on s'approche le plus
    possible du cadre choisi par l'utilisateur.
    """
    if crop_is_valid(w, h, angle, box, margin):
        return box
    seed = _seed_box(w, h, angle, margin)
    return constrain_box(w, h, angle, seed, box, margin)


def to_norm(box: Box, w: float, h: float) -> Box:
    l, t, r, b = box
    return (l / w, t / h, r / w, b / h)


def from_norm(norm: Box, w: float, h: float) -> Box:
    l, t, r, b = norm
    return (l * w, t * h, r * w, b * h)


def pixel_crop_box(w: int, h: int, angle: float, norm: Box, margin: float = DEFAULT_MARGIN) -> tuple[int, int, int, int]:
    """Convertit un cadre normalisé en boîte entière valide pour l'image ``w × h``.

    Le cadre est d'abord réajusté si besoin (arrondis entre l'aperçu et la
    pleine résolution), puis arrondi vers l'intérieur : un rectangle inclus
    dans un cadre valide est lui aussi valide.
    """
    box = refit_box(w, h, angle, from_norm(norm, w, h), margin)
    l, t, r, b = box
    left = max(0, math.ceil(l - 1e-6))
    top = max(0, math.ceil(t - 1e-6))
    right = min(w, math.floor(r + 1e-6))
    bottom = min(h, math.floor(b + 1e-6))
    right = max(right, left + 1)
    bottom = max(bottom, top + 1)
    return left, top, right, bottom
