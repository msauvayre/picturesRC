"""Estimation de l'inclinaison d'une photo et calcul du rognage après rotation.

Convention d'angle (partout dans le projet) :
    ``angle`` = rotation à APPLIQUER à l'image, en degrés,
    positive = sens antihoraire, à l'écran.
C'est la convention de ``PIL.Image.rotate`` et de ``cv2.getRotationMatrix2D``.

Principe de l'estimation (cas « horizon penché ») :
    1. Réduction de l'image (~1000 px), niveaux de gris, léger flou.
    2. Contours (Canny, seuils issus d'Otsu).
    3. Segments de droite (Hough probabiliste).
    4. On garde les segments presque horizontaux ou presque verticaux
       (écart < ``max_angle``). Chacun vote pour l'écart à son axe,
       avec un poids égal à sa longueur.
    5. Le pic d'une densité à noyau gaussien donne l'angle. La part des
       votes concordants donne un indice de confiance entre 0 et 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

ANALYSIS_MAX_SIDE = 1000

# Seuils de confiance, utilisés par l'interface pour les badges.
CONFIDENCE_HIGH = 0.45
CONFIDENCE_MEDIUM = 0.25


@dataclass
class TiltEstimate:
    """Résultat de l'estimation automatique."""

    angle: float  # rotation à appliquer (degrés, antihoraire positif)
    confidence: float  # 0 = aucune information, 1 = très fiable
    n_lines: int  # segments candidats retenus
    n_support: int  # segments concordants avec l'angle retenu
    # Segments concordants, en coordonnées de l'image analysée (debug/overlay)
    support_segments: np.ndarray = field(default_factory=lambda: np.empty((0, 4)))
    analysis_scale: float = 1.0  # facteur image analysée / image fournie
    # Méthode : "lignes" (droites de toute la photo) ou "contour" (bords de l'objet central)
    source: str = "lignes"

    @property
    def level(self) -> str:
        """``"high"``, ``"medium"`` ou ``"low"``."""
        if self.confidence >= CONFIDENCE_HIGH:
            return "high"
        if self.confidence >= CONFIDENCE_MEDIUM:
            return "medium"
        return "low"


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _segment_deviation(dx: np.ndarray, dy: np.ndarray, max_angle: float):
    """Écart (degrés) de chaque segment à l'axe horizontal ou vertical le plus proche.

    Repère image (y vers le bas). Renvoie ``(deviation, mask)`` : ``deviation``
    est directement l'angle de rotation à appliquer pour remettre le segment
    d'aplomb, ``mask`` indique les segments assez proches d'un axe.
    """
    theta = np.degrees(np.arctan2(dy, dx))
    # Une droite n'a pas de sens : on ramène theta dans ]-90, 90]
    theta = np.where(theta > 90, theta - 180, theta)
    theta = np.where(theta <= -90, theta + 180, theta)

    horizontal = np.abs(theta) < max_angle
    vertical = np.abs(theta) > (90 - max_angle)
    deviation = np.where(horizontal, theta, np.where(theta > 0, theta - 90, theta + 90))
    return deviation, horizontal | vertical


def measure_line_angle(x1: float, y1: float, x2: float, y2: float) -> float:
    """Angle à appliquer pour rendre horizontal OU vertical un segment tracé.

    Sert à l'outil « tracer l'horizon » de l'interface. L'axe visé est
    choisi automatiquement : celui dont le segment est le plus proche.
    """
    dev, _ = _segment_deviation(np.array([x2 - x1]), np.array([y2 - y1]), 45.0)
    return float(dev[0])


def _refine_segments(seg: np.ndarray, grad_mag: np.ndarray, half_band: int = 3) -> np.ndarray:
    """Réajuste chaque segment de Hough avec une précision sous-pixel.

    Les extrémités renvoyées par ``HoughLinesP`` sont entières et, pour une
    droite peu inclinée (< 1-2°), les pixels de contour forment des
    « marches d'escalier » : l'angle mesuré est biaisé vers 0°.
    Ici, pour chaque colonne (segment presque horizontal) ou chaque ligne
    (segment presque vertical) traversée, on cherche le maximum du gradient
    perpendiculairement au segment, on l'interpole par une parabole
    (position sous-pixel), puis on ajuste une droite par moindres carrés
    pondérés par l'intensité du gradient.

    Les segments qu'on ne peut pas réajuster de façon fiable sont renvoyés
    à ``NaN`` (et donc écartés) : leur angle brut est justement biaisé.
    """
    refined = np.full_like(seg, np.nan)
    offsets = np.arange(-half_band, half_band + 1)
    for i, (x1, y1, x2, y2) in enumerate(seg):
        horizontal = abs(x2 - x1) >= abs(y2 - y1)
        # On travaille dans un repère (u = axe principal, v = axe transverse)
        if horizontal:
            u1, v1, u2, v2, img = x1, y1, x2, y2, grad_mag
        else:
            u1, v1, u2, v2, img = y1, x1, y2, x2, grad_mag.T
        if u2 < u1:
            u1, v1, u2, v2 = u2, v2, u1, v1
        n_u, n_v = img.shape[1], img.shape[0]
        us = np.arange(int(math.ceil(u1)), int(math.floor(u2)) + 1)
        if len(us) < 10:
            continue
        v_line = v1 + (us - u1) * (v2 - v1) / (u2 - u1)
        vs = np.rint(v_line).astype(int)[:, None] + offsets[None, :]
        ok = (vs.min(axis=1) >= 0) & (vs.max(axis=1) < n_v) & (us >= 0) & (us < n_u)
        us, vs = us[ok], vs[ok]
        if len(us) < 10:
            continue
        prof = img[vs, us[:, None]]  # profils transverses
        k = np.argmax(prof, axis=1)
        inner = (k > 0) & (k < len(offsets) - 1)
        rows = np.nonzero(inner)[0]
        if len(rows) < 10:
            continue
        k = k[rows]
        a, b, c = prof[rows, k - 1], prof[rows, k], prof[rows, k + 1]
        denom = a - 2 * b + c
        delta = np.where(np.abs(denom) > 1e-9, 0.5 * (a - c) / denom, 0.0)
        v_sub = vs[rows, k] + np.clip(delta, -0.5, 0.5)
        u_sub = us[rows].astype(np.float64)
        wgt = b
        # Moindres carrés pondérés : v = p*u + q
        W = wgt.sum()
        if W <= 0:
            continue
        um, vm = (wgt * u_sub).sum() / W, (wgt * v_sub).sum() / W
        var_u = (wgt * (u_sub - um) ** 2).sum()
        if var_u <= 0:
            continue
        p = (wgt * (u_sub - um) * (v_sub - vm)).sum() / var_u
        q = vm - p * um
        nu1, nu2 = float(u1), float(u2)
        nv1, nv2 = p * nu1 + q, p * nu2 + q
        refined[i] = (nu1, nv1, nu2, nv2) if horizontal else (nv1, nu1, nv2, nu2)
    return refined


def estimate_tilt(image: np.ndarray, max_angle: float = 15.0) -> TiltEstimate:
    """Estime la rotation à appliquer pour redresser ``image``.

    ``image`` : tableau numpy RGB, RGBA ou niveaux de gris (une version
    réduite suffit : elle est de toute façon ramenée à ~1000 px).
    ``max_angle`` : inclinaison maximale recherchée, en degrés.
    """
    gray = _to_gray(image)
    h, w = gray.shape
    scale = min(1.0, ANALYSIS_MAX_SIDE / max(h, w))
    if scale < 1.0:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = gray.shape

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(gx, gy)
    otsu, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    high = max(otsu, 30.0)
    edges = cv2.Canny(gray, 0.5 * high, high, L2gradient=True)

    min_len = max(20, int(0.06 * max(h, w)))
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 1800,  # résolution 0,1°
        threshold=max(30, min_len // 2),
        minLineLength=min_len,
        maxLineGap=max(3, min_len // 10),
    )
    empty = TiltEstimate(0.0, 0.0, 0, 0, analysis_scale=scale)
    if lines is None:
        return empty

    seg = _refine_segments(lines.reshape(-1, 4).astype(np.float64), grad_mag)
    seg = seg[~np.isnan(seg).any(axis=1)]
    if len(seg) == 0:
        return empty
    dx = seg[:, 2] - seg[:, 0]
    dy = seg[:, 3] - seg[:, 1]
    length = np.hypot(dx, dy)
    deviation, keep = _segment_deviation(dx, dy, max_angle)
    if not np.any(keep):
        return empty

    dev, wgt, seg = deviation[keep], length[keep], seg[keep]

    # Densité à noyau gaussien pondérée par la longueur des segments
    sigma = 0.3
    grid = np.arange(-max_angle, max_angle + 1e-9, 0.05)
    density = (wgt[None, :] * np.exp(-0.5 * ((grid[:, None] - dev[None, :]) / sigma) ** 2)).sum(axis=1)
    peak = grid[int(np.argmax(density))]

    near = np.abs(dev - peak) <= 2 * sigma
    angle = float(np.average(dev[near], weights=wgt[near]))

    support = float(wgt[near].sum() / wgt.sum())
    n_support = int(near.sum())
    # Peu de segments concordants = indice peu fiable, même s'ils sont unanimes
    confidence = support * min(1.0, n_support / 5.0)

    return TiltEstimate(
        angle=round(angle, 2),
        confidence=round(confidence, 3),
        n_lines=int(len(dev)),
        n_support=n_support,
        support_segments=seg[near],
        analysis_scale=scale,
    )


def inscribed_crop_size(width: int, height: int, angle: float, mode: str = "aspect") -> tuple[int, int]:
    """Taille du plus grand rectangle sans coin vide après rotation de ``angle``.

    ``mode`` :
        ``"aspect"``   — conserve les proportions de l'image d'origine ;
        ``"max_area"`` — maximise la surface (les proportions changent).
    Le rectangle est centré. Une marge de 1 px de chaque côté évite les
    liserés dus à l'interpolation.
    """
    if abs(angle) < 1e-6:
        return width, height

    a = math.radians(abs(angle) % 180)
    sin_a, cos_a = abs(math.sin(a)), abs(math.cos(a))

    if mode == "aspect":
        s = min(
            width / (width * cos_a + height * sin_a),
            height / (width * sin_a + height * cos_a),
        )
        cw, ch = width * s, height * s
    elif mode == "max_area":
        long_side, short_side = (width, height) if width >= height else (height, width)
        if short_side <= 2.0 * sin_a * cos_a * long_side or abs(sin_a - cos_a) < 1e-10:
            x = 0.5 * short_side
            cw, ch = (x / sin_a, x / cos_a) if width >= height else (x / cos_a, x / sin_a)
        else:
            cos_2a = cos_a * cos_a - sin_a * sin_a
            cw = (width * cos_a - height * sin_a) / cos_2a
            ch = (height * cos_a - width * sin_a) / cos_2a
    else:
        raise ValueError(f"mode inconnu : {mode!r}")

    cw = min(width, int(math.floor(cw)) - 2)
    ch = min(height, int(math.floor(ch)) - 2)
    return max(1, cw), max(1, ch)


def rotate_array(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotation rapide d'un tableau numpy (aperçus), même convention d'angle."""
    if abs(angle) < 1e-6:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2.0, (h - 1) / 2.0), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
