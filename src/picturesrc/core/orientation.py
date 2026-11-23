"""Rotations par quarts de tour (90° à gauche, 90° à droite, 180°).

Le quart de tour est appliqué juste après l'orientation EXIF, AVANT la
rotation fine et le rognage. Tout le reste du traitement (angle fin,
cadre, objet détecté) travaille donc dans le repère de l'image déjà
tournée.

Convention : ``quarter`` = nombre de quarts de tour ANTIHORAIRES, ramené
dans 0..3 (même sens positif que ``angle``) :
    0 = aucun, 1 = 90° à gauche, 2 = 180°, 3 = 90° à droite.

Un quart de tour et la rotation fine (autour du centre) commutent : un
angle fin, un cadre normalisé ou l'objet détecté se transposent donc
exactement, sans refaire l'analyse.
"""

from __future__ import annotations

import dataclasses

import numpy as np
from PIL import Image

from .crop import Box
from .subject import SubjectEstimate

LEFT = 1  # 90° à gauche (antihoraire)
RIGHT = -1  # 90° à droite (horaire)

QUARTER_LABELS = {0: None, 1: "90° à gauche", 2: "180°", 3: "90° à droite"}

_PIL_TRANSPOSE = {
    1: Image.Transpose.ROTATE_90,  # PIL : antihoraire
    2: Image.Transpose.ROTATE_180,
    3: Image.Transpose.ROTATE_270,
}


def normalize_quarter(quarter: int) -> int:
    return int(quarter) % 4


def rotate_image_quarter(img: Image.Image, quarter: int) -> Image.Image:
    """Quart(s) de tour sans perte (permutation de pixels, pas d'interpolation)."""
    q = normalize_quarter(quarter)
    return img if q == 0 else img.transpose(_PIL_TRANSPOSE[q])


def rotate_array_quarter(arr: np.ndarray, quarter: int) -> np.ndarray:
    """Même chose pour un tableau numpy (aperçus, miniatures) ; résultat contigu."""
    q = normalize_quarter(quarter)
    return arr if q == 0 else np.ascontiguousarray(np.rot90(arr, q))


def rotate_norm_points(pts: np.ndarray, quarter: int) -> np.ndarray:
    """Points normalisés (N × 2, entre 0 et 1) -> repère de l'image tournée."""
    q = normalize_quarter(quarter)
    x, y = pts[:, 0], pts[:, 1]
    for _ in range(q):  # un quart de tour antihoraire : (x, y) -> (y, 1 - x)
        x, y = y, 1.0 - x
    return np.column_stack([x, y])


def rotate_norm_box(box: Box, quarter: int) -> Box:
    """Cadre normalisé ``(gauche, haut, droite, bas)`` -> repère de l'image tournée."""
    l, t, r, b = box
    corners = rotate_norm_points(np.array([[l, t], [r, b]], dtype=np.float64), quarter)
    (x0, y0), (x1, y1) = corners
    return (float(min(x0, x1)), float(min(y0, y1)), float(max(x0, x1)), float(max(y0, y1)))


def rotate_subject(subject: SubjectEstimate | None, quarter: int) -> SubjectEstimate | None:
    """Objet détecté -> repère de l'image tournée (nouvel objet, l'ancien est intact)."""
    if subject is None or normalize_quarter(quarter) == 0:
        return subject
    return dataclasses.replace(subject, hull=rotate_norm_points(subject.hull, quarter))
