"""Cœur de traitement, sans aucune dépendance à l'interface graphique."""

from .straighten import TiltEstimate, estimate_tilt, inscribed_crop_size, measure_line_angle
from .crop import crop_is_valid, pixel_crop_box
from .imageio import load_image, load_preview, save_image
from .subject import PhotoAnalysis, SubjectEstimate, analyze_photo, detect_subject, estimate_outline_tilt
from .pipeline import ExportSettings, export_image

__all__ = [
    "TiltEstimate",
    "estimate_tilt",
    "inscribed_crop_size",
    "measure_line_angle",
    "crop_is_valid",
    "pixel_crop_box",
    "load_image",
    "load_preview",
    "save_image",
    "SubjectEstimate",
    "detect_subject",
    "estimate_outline_tilt",
    "analyze_photo",
    "PhotoAnalysis",
    "ExportSettings",
    "export_image",
]
