"""Tâches de fond (analyse et export) exécutées dans un QThreadPool.

Règle Qt : ces tâches ne manipulent jamais de QPixmap ni de widget ;
elles renvoient des tableaux numpy via des signaux, convertis côté
fil principal.
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal

from ..core.imageio import load_preview
from ..core.inplace import InPlaceSettings, apply_in_place
from ..core.straighten import ANALYSIS_MAX_SIDE, TiltEstimate
from ..core.subject import PhotoAnalysis, SubjectEstimate, analyze_photo

THUMB_SIDE = 220


@dataclass
class PhotoItem:
    path: Path
    status: str = "pending"  # pending | ready | error
    analysis: PhotoAnalysis | None = None  # toutes les estimations
    estimate: TiltEstimate | None = None  # estimation retenue pour le mode de rognage courant
    subject: SubjectEstimate | None = None  # objet central détecté (mode « objet »)
    angle: float = 0.0
    # Quarts de tour antihoraires (0..3) appliqués avant tout le reste ;
    # angle, cadre et objet sont exprimés dans le repère de l'image tournée.
    quarter: int = 0
    # Cadre de rognage choisi à la main, normalisé (gauche, haut, droite, bas)
    # entre 0 et 1 dans l'image pivotée ; None = cadre automatique.
    crop: tuple[float, float, float, float] | None = None
    validated: bool = False
    exported_to: Path | None = None
    # Après enregistrement : emplacement de l'original déplacé (None = à sa place)
    original: Path | None = None
    error: str = ""
    thumb_src: np.ndarray | None = field(default=None, repr=False)

    @property
    def manual(self) -> bool:
        return self.estimate is not None and abs(self.angle - self.estimate.angle) > 1e-3


class AnalyzeSignals(QObject):
    done = Signal(object, object, object)  # path, PhotoAnalysis, thumbnail (np.ndarray)
    failed = Signal(object, str)  # path, message


class AnalyzeTask(QRunnable):
    def __init__(self, path: Path, max_angle: float):
        super().__init__()
        self.path = path
        self.max_angle = max_angle
        self.signals = AnalyzeSignals()

    def run(self) -> None:
        try:
            preview = load_preview(self.path, ANALYSIS_MAX_SIDE)
            analysis = analyze_photo(preview, max_angle=self.max_angle)
            h, w = preview.shape[:2]
            s = THUMB_SIDE / max(h, w)
            thumb = cv2.resize(preview, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
            self.signals.done.emit(self.path, analysis, np.ascontiguousarray(thumb))
        except Exception as exc:  # noqa: BLE001 — on remonte tout à l'interface
            self.signals.failed.emit(self.path, f"{type(exc).__name__}: {exc}")


class ExportSignals(QObject):
    progress = Signal(int, int, object)  # fait, total, (chemin, InPlaceResult | None)
    item_failed = Signal(object, str)
    finished = Signal(int, int)  # réussis, échecs


class ExportTask(QRunnable):
    """Enregistre les photos une par une, sur place (une seule pleine résolution
    en mémoire). Cf. ``core.inplace.apply_in_place``."""

    def __init__(
        self,
        jobs: list[
            tuple[
                Path,
                float,
                TiltEstimate | None,
                tuple[float, float, float, float] | None,
                SubjectEstimate | None,
                int,
                Path | None,
            ]
        ],
        settings: InPlaceSettings,
    ):
        super().__init__()
        self.jobs = jobs
        self.settings = settings
        self.signals = ExportSignals()
        self.cancelled = False

    def run(self) -> None:
        ok = ko = 0
        total = len(self.jobs)
        for i, (path, angle, est, crop, subject, quarter, original) in enumerate(self.jobs, start=1):
            if self.cancelled:
                break
            try:
                result = apply_in_place(path, angle, self.settings, est, crop, subject, quarter, original)
                ok += 1
                self.signals.progress.emit(i, total, (path, result))
            except Exception as exc:  # noqa: BLE001
                ko += 1
                traceback.print_exc()
                self.signals.item_failed.emit(path, f"{type(exc).__name__}: {exc}")
                self.signals.progress.emit(i, total, (path, None))
        self.signals.finished.emit(ok, ko)
