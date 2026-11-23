"""Détection de l'objet d'intérêt posé au centre de la photo, pour rogner autour.

Cas visé : un objet (scellé, appareil, document…) photographié au centre,
sur un fond assez uniforme (bureau, table, drap). On retire le « contour »
inutile autour de l'objet.

Principe (sur une version réduite, ~600 px, déjà redressée) :
    1. Modèle de couleur du fond, appris sur une bande le long des bords
       (espace Lab, distance de Mahalanobis robuste : 80 % des pixels du
       bord les plus typiques).
    2. Les pixels éloignés de ce modèle forment le masque « objet » ;
       nettoyage morphologique, trous bouchés.
    3. Parmi les zones détectées, on garde la plus grande et la plus
       centrale (un passe-câble ou une tache au bord est ignoré).
    4. Les excroissances (reflet sur le bureau collé à l'objet…) sont
       retirées : on ne garde que les lignes et colonnes « bien remplies ».
    5. Le résultat est mémorisé sous forme d'enveloppe convexe, en
       coordonnées normalisées de l'image NON pivotée : on peut ainsi
       recalculer le cadre pour n'importe quel angle sans refaire la
       détection.

L'objet sert aussi à redresser la photo (``estimate_outline_tilt``) : ses
bords sont ajustés par des droites, ce qui ignore le contenu éventuellement
posé de travers à l'intérieur.

Une confiance (0-1) signale les cas douteux : objet qui touche le bord de
la photo, objet minuscule ou occupant toute l'image, forme peu compacte,
contraste faible avec le fond.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .crop import Box, auto_crop_box, refit_box
from .straighten import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    TiltEstimate,
    estimate_tilt,
    inscribed_crop_size,
    measure_line_angle,
    rotate_array,
)

DETECT_MAX_SIDE = 600
# Marge laissée autour de l'objet, en fraction de sa taille (de chaque côté)
SUBJECT_MARGIN = 0.02


@dataclass
class SubjectEstimate:
    """Objet détecté. ``hull`` : enveloppe convexe (N × 2), coordonnées
    normalisées (0-1) dans l'image d'origine, non pivotée."""

    hull: np.ndarray = field(repr=False)
    confidence: float
    area_fraction: float  # part de l'image occupée par l'objet

    @property
    def level(self) -> str:
        if self.confidence >= CONFIDENCE_HIGH:
            return "high"
        if self.confidence >= CONFIDENCE_MEDIUM:
            return "medium"
        return "low"


# --- géométrie : même transformation que PIL.Image.rotate (centre w/2, h/2) ---
def _rotate_points(pts: np.ndarray, w: float, h: float, angle: float) -> np.ndarray:
    """Image d'origine -> image pivotée de ``angle`` (antihoraire positif)."""
    t = -math.radians(angle)
    c, s = math.cos(t), math.sin(t)
    d = pts - (w / 2.0, h / 2.0)
    x = c * d[:, 0] - s * d[:, 1]
    y = s * d[:, 0] + c * d[:, 1]
    return np.column_stack([x, y]) + (w / 2.0, h / 2.0)


def _unrotate_points(pts: np.ndarray, w: float, h: float, angle: float) -> np.ndarray:
    """Image pivotée -> image d'origine (inverse de ``_rotate_points``)."""
    return _rotate_points(pts, w, h, -angle)


# --- détection ------------------------------------------------------------
def _background_distance(lab: np.ndarray, sample_mask: np.ndarray) -> np.ndarray:
    samples = lab[sample_mask]
    med = np.median(samples, axis=0)
    d0 = np.linalg.norm(samples - med, axis=1)
    core = samples[d0 <= np.quantile(d0, 0.8)]
    mu = core.mean(axis=0)
    cov = np.cov(core.T) + np.eye(3) * 4.0  # plancher : fond très uniforme
    inv = np.linalg.inv(cov)
    diff = lab - mu
    return np.sqrt(np.einsum("...i,ij,...j->...", diff, inv, diff))


def _trim_protrusions(mask: np.ndarray, keep: float = 0.5) -> tuple[int, int, int, int] | None:
    """Boîte (x0, y0, x1, y1) de la partie « pleine » du masque.

    Une ligne (resp. colonne) est conservée si elle contient au moins
    ``keep`` fois la largeur typique de l'objet. Un reflet collé au bord du
    sachet n'occupe qu'une fraction de la largeur : il est écarté.
    """
    rows = mask.sum(axis=1).astype(np.float64)
    cols = mask.sum(axis=0).astype(np.float64)
    if rows.max() == 0:
        return None
    ref_r = np.percentile(rows[rows > 0], 75)
    ref_c = np.percentile(cols[cols > 0], 75)
    ys = np.nonzero(rows >= keep * ref_r)[0]
    xs = np.nonzero(cols >= keep * ref_c)[0]
    if len(ys) == 0 or len(xs) == 0:
        return None
    return int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1


@dataclass
class _ObjectMask:
    solid: np.ndarray  # composante retenue, trous bouchés (uint8 0/1), image réduite et pivotée
    valid: np.ndarray  # zone utile (hors coins vides)
    dist: np.ndarray  # distance au modèle de fond
    width: int
    height: int


def _object_mask(image: np.ndarray, angle: float, threshold: float) -> _ObjectMask | None:
    """Masque de l'objet central, sur l'image réduite (~600 px) pivotée de ``angle``."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    h0, w0 = image.shape[:2]
    s = min(1.0, DETECT_MAX_SIDE / max(h0, w0))
    small = cv2.resize(image, (max(1, round(w0 * s)), max(1, round(h0 * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    rot = rotate_array(small, angle)

    # Zone utile (hors coins vides après rotation)
    valid = rotate_array(np.full((sh, sw), 255, np.uint8), angle) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    lab = cv2.cvtColor(cv2.GaussianBlur(rot, (5, 5), 0), cv2.COLOR_RGB2LAB).astype(np.float32)
    band = max(4, int(0.04 * max(sh, sw)))
    border = np.zeros((sh, sw), bool)
    border[:band] = border[-band:] = True
    border[:, :band] = border[:, -band:] = True
    sample = border & valid
    if sample.sum() < 50:
        return None
    dist = _background_distance(lab, sample)

    fg = ((dist > threshold) & valid).astype(np.uint8) * 255
    k = max(3, int(0.02 * max(sh, sw)) | 1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

    n, lbl, stats, cent = cv2.connectedComponentsWithStats(fg)
    best, best_score = 0, 0.0
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 0.01 * sh * sw:
            continue
        dc = math.hypot((cent[i][0] - sw / 2) / sw, (cent[i][1] - sh / 2) / sh)
        score = area * (1.0 - min(dc, 0.7))
        if score > best_score:
            best, best_score = i, score
    if best == 0:
        return None

    # Composante retenue, trous bouchés (zones transparentes du sachet…)
    comp = (lbl == best).astype(np.uint8)
    contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(comp)
    cv2.drawContours(solid, contours, -1, 1, thickness=cv2.FILLED)
    return _ObjectMask(solid, valid, dist, sw, sh)


def detect_subject(image: np.ndarray, angle: float = 0.0, threshold: float = 6.0) -> SubjectEstimate | None:
    """Détecte l'objet central de ``image`` (RGB, image d'origine non pivotée).

    ``angle`` : rotation qui sera appliquée (redressement) ; la détection se
    fait sur l'image redressée, où l'objet est aligné sur les axes.
    Renvoie ``None`` si rien d'exploitable n'est trouvé.
    """
    om = _object_mask(image, angle, threshold)
    if om is None:
        return None
    solid, valid, dist, sw, sh = om.solid, om.valid, om.dist, om.width, om.height

    box = _trim_protrusions(solid > 0)
    if box is None:
        return None
    x0, y0, x1, y1 = box
    trimmed = np.zeros_like(solid)
    trimmed[y0:y1, x0:x1] = solid[y0:y1, x0:x1]
    pts = cv2.findNonZero(trimmed)
    if pts is None:
        return None
    hull = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64) + 0.5

    # --- confiance -----------------------------------------------------
    bw, bh = x1 - x0, y1 - y0
    area_frac = bw * bh / float(valid.sum())
    fill = trimmed.sum() / float(max(1, bw * bh))  # compacité (objet ~ rectangle)
    # L'objet touche-t-il le bord de la zone utile ? (photo trop serrée)
    edge = cv2.dilate((~valid).astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    edge[:3] = edge[-3:] = True
    edge[:, :3] = edge[:, -3:] = True
    touch = float((trimmed.astype(bool) & edge).sum()) / max(1.0, 2.0 * (bw + bh))
    inside = dist[trimmed > 0]
    contrast = float(np.median(inside)) / threshold if inside.size else 0.0

    conf = 1.0
    conf *= min(1.0, max(0.0, (fill - 0.55) / 0.3))  # 0,85 et plus = rectangle plein
    conf *= max(0.0, 1.0 - 4.0 * touch)  # quelques % du périmètre au bord : pénalisé
    conf *= 1.0 if 0.05 <= area_frac <= 0.92 else 0.3
    conf *= min(1.0, contrast / 1.5)

    # Retour en coordonnées normalisées de l'image d'origine (non pivotée)
    orig = _unrotate_points(hull, sw, sh, angle) / (sw, sh)
    return SubjectEstimate(hull=orig, confidence=round(conf, 3), area_fraction=round(area_frac, 3))


# --- angle d'après le contour de l'objet -----------------------------------
SIDE_AGREEMENT = 1.0  # écart max (degrés) entre côtés jugés concordants


def _robust_line(u: np.ndarray, v: np.ndarray) -> tuple[float, float, int, float]:
    """Droite ``v = p·u + q`` ajustée en écartant les bosses (rejet itératif
    des résidus > 2,5 écarts absolus médians). Renvoie ``(p, q, n_gardés, écart-type)``."""
    keep = np.ones(len(u), bool)
    for _ in range(4):
        p, q = np.polyfit(u[keep], v[keep], 1)
        res = v - (p * u + q)
        r = res[keep]
        mad = 1.4826 * np.median(np.abs(r - np.median(r)))
        new_keep = np.abs(res) <= max(1.0, 2.5 * mad)
        if new_keep.sum() < 10 or np.array_equal(new_keep, keep):
            break
        keep = new_keep
    p, q = np.polyfit(u[keep], v[keep], 1)
    sd = float(np.std((v - (p * u + q))[keep]))
    return float(p), float(q), int(keep.sum()), sd


def _side_angles(mask: np.ndarray) -> list[tuple[float, float]]:
    """Pour chacun des 4 côtés du masque : ``(angle à appliquer, poids)``.

    Le bord est relevé ligne par ligne (côtés gauche/droit) ou colonne par
    colonne (haut/bas), sur les 76 % centraux pour éviter les coins
    arrondis ou froissés, puis ajusté par une droite robuste. Un côté
    plaqué contre le bord de la photo (objet coupé) est ignoré.
    """
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    h, w = mask.shape
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    rows = np.arange(int(y0 + 0.12 * (y1 - y0)), int(y1 - 0.12 * (y1 - y0)))
    cols = np.arange(int(x0 + 0.12 * (x1 - x0)), int(x1 - 0.12 * (x1 - x0)))
    out: list[tuple[float, float]] = []
    if len(rows) >= 20:
        sub = mask[rows] > 0
        ok = sub.any(axis=1)
        left = np.argmax(sub, axis=1)
        right = w - 1 - np.argmax(sub[:, ::-1], axis=1)
        for v, limit in ((left, 0), (right, w - 1)):
            u, vv = rows[ok].astype(float), v[ok].astype(float)
            if len(u) < 20 or np.median(np.abs(vv - limit)) <= 2:
                continue
            p, q, n, sd = _robust_line(u, vv)
            a, b = u[0], u[-1]
            out.append((measure_line_angle(p * a + q, a, p * b + q, b), n / (1.0 + sd)))
    if len(cols) >= 20:
        sub = mask[:, cols] > 0
        ok = sub.any(axis=0)
        top = np.argmax(sub, axis=0)
        bottom = h - 1 - np.argmax(sub[::-1], axis=0)
        for v, limit in ((top, 0), (bottom, h - 1)):
            u, vv = cols[ok].astype(float), v[ok].astype(float)
            if len(u) < 20 or np.median(np.abs(vv - limit)) <= 2:
                continue
            p, q, n, sd = _robust_line(u, vv)
            a, b = u[0], u[-1]
            out.append((measure_line_angle(a, p * a + q, b, p * b + q), n / (1.0 + sd)))
    return out


def estimate_outline_tilt(image: np.ndarray, max_angle: float = 15.0, threshold: float = 6.0) -> TiltEstimate | None:
    """Angle de redressement d'après les bords de l'objet central.

    Plus fiable que l'estimation « lignes » quand le contenu de l'objet est
    lui-même de travers (appareils posés en biais dans un sachet, étiquette
    collée de travers…). Les côtés d'un sachet souple ne sont jamais
    parfaitement parallèles : on retient ceux qui concordent à
    ``SIDE_AGREEMENT`` près, pondérés par la netteté de leur bord.
    Deux passes : la seconde mesure l'écart résiduel une fois l'objet redressé.
    """
    om = _object_mask(image, 0.0, threshold)
    if om is None:
        return None
    mask = om.solid * 255
    total = 0.0
    sides: list[tuple[float, float]] = []
    agree = np.zeros(0, bool)
    for _ in range(2):
        sides = _side_angles(rotate_array(mask, total) if total else mask)
        if len(sides) < 2:
            return None
        a = np.array([x[0] for x in sides])
        wgt = np.array([x[1] for x in sides])
        order = np.argsort(a)
        cum = np.cumsum(wgt[order])
        median = a[order][min(len(a) - 1, int(np.searchsorted(cum, cum[-1] / 2)))]
        agree = np.abs(a - median) <= SIDE_AGREEMENT
        total += float(np.average(a[agree], weights=wgt[agree]))
    if abs(total) > max_angle:
        return None
    wgt = np.array([x[1] for x in sides])
    confidence = float(wgt[agree].sum() / wgt.sum()) * min(1.0, agree.sum() / 3.0)
    return TiltEstimate(
        angle=round(total, 2),
        confidence=round(confidence, 3),
        n_lines=len(sides),
        n_support=int(agree.sum()),
        source="contour",
    )


@dataclass
class PhotoAnalysis:
    """Résultat complet de l'analyse d'une photo (cf. ``analyze_photo``)."""

    lines: TiltEstimate  # angle d'après les lignes droites de toute la photo
    outline: TiltEstimate | None  # angle d'après le contour de l'objet central
    subject: SubjectEstimate | None  # objet central détecté

    @property
    def outline_usable(self) -> bool:
        return (
            self.outline is not None
            and self.outline.confidence >= CONFIDENCE_MEDIUM
            and self.subject is not None
            and self.subject.level != "low"
        )

    def estimate_for(self, crop_mode: str) -> TiltEstimate:
        """Estimation d'angle à utiliser : en mode « objet », celle du
        contour de l'objet si elle est exploitable, sinon celle des lignes."""
        if crop_mode == "subject" and self.outline_usable:
            return self.outline  # type: ignore[return-value]
        return self.lines


def analyze_photo(image: np.ndarray, max_angle: float = 15.0) -> PhotoAnalysis:
    """Analyse une photo (aperçu RGB) : angle « lignes », angle « contour »
    et objet central. L'objet est détecté sur l'image redressée selon
    l'angle qui sera proposé en mode « objet »."""
    lines = estimate_tilt(image, max_angle=max_angle)
    outline = estimate_outline_tilt(image, max_angle=max_angle)
    angle = outline.angle if outline is not None and outline.confidence >= CONFIDENCE_MEDIUM else lines.angle
    subject = detect_subject(image, angle)
    analysis = PhotoAnalysis(lines, outline, subject)
    if outline is not None and not analysis.outline_usable and abs(angle - lines.angle) > 1e-6:
        # Contour inexploitable : on recadre d'après l'angle « lignes »
        analysis.subject = detect_subject(image, lines.angle)
    return analysis


# --- cadre ----------------------------------------------------------------
def subject_crop_box(
    w: int, h: int, angle: float, subject: SubjectEstimate, margin: float = SUBJECT_MARGIN
) -> Box:
    """Cadre (pixels, image pivotée) qui entoure l'objet avec une petite marge,
    sans jamais inclure de coin vide."""
    pts = _rotate_points(subject.hull * (w, h), w, h, angle)
    l, t = pts.min(axis=0)
    r, b = pts.max(axis=0)
    mx, my = margin * (r - l), margin * (b - t)
    box = (max(0.0, l - mx), max(0.0, t - my), min(float(w), r + mx), min(float(h), b + my))
    return refit_box(w, h, angle, box)


def auto_box(w: int, h: int, angle: float, mode: str, subject: SubjectEstimate | None = None) -> Box | None:
    """Cadre automatique pour un mode de rognage donné (``None`` = pas de rognage).

    Mode ``"subject"`` sans objet détecté : repli sur « surface maximale ».
    """
    if mode == "none":
        return None
    if mode == "subject":
        if subject is not None:
            return subject_crop_box(w, h, angle, subject)
        mode = "max_area"
    if mode == "aspect":
        cw, ch = inscribed_crop_size(w, h, angle, mode)
        left, top = (w - cw) // 2, (h - ch) // 2
        return (float(left), float(top), float(left + cw), float(top + ch))
    return auto_crop_box(w, h, angle, mode)
