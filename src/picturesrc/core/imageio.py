"""Lecture / écriture des images.

Tout passe par Pillow plutôt que ``cv2.imread`` / ``cv2.imwrite`` :
    * chemins non ASCII (accents) fiables sous Windows ;
    * prise en compte de l'orientation EXIF ;
    * conservation des métadonnées EXIF et du profil ICC à l'export.

Limite de la v0.1 : les images 16 bits sont ramenées en 8 bits.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
_JPEG_EXT = {".jpg", ".jpeg"}
_ORIENTATION_TAG = 0x0112

# Les photos de plus de ~89 Mpx déclenchent sinon un avertissement Pillow
Image.MAX_IMAGE_PIXELS = 400_000_000


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _normalize_mode(img: Image.Image) -> Image.Image:
    if img.mode in ("RGB", "RGBA", "L"):
        return img
    if img.mode in ("P", "PA", "LA"):
        return img.convert("RGBA" if "transparency" in img.info or "A" in img.mode else "RGB")
    if img.mode == "CMYK":
        return img.convert("RGB")
    if img.mode.startswith("I") or img.mode == "F":
        # 16/32 bits niveaux de gris -> 8 bits
        arr = np.asarray(img, dtype=np.float64)
        lo, hi = float(arr.min()), float(arr.max())
        arr = (arr - lo) / (hi - lo or 1.0) * 255.0
        return Image.fromarray(arr.astype(np.uint8), "L")
    return img.convert("RGB")


def load_image(path: Path) -> tuple[Image.Image, bytes | None, bytes | None]:
    """Charge l'image en pleine résolution, orientation EXIF appliquée.

    Renvoie ``(image, exif_bytes, icc_profile)``. ``exif_bytes`` a son
    tag Orientation remis à 1, puisque l'orientation est déjà appliquée.
    """
    with Image.open(path) as src:
        src.load()
        icc = src.info.get("icc_profile")
        exif = src.getexif()
        img = ImageOps.exif_transpose(src)
    exif_bytes = None
    if exif:
        exif[_ORIENTATION_TAG] = 1
        exif_bytes = exif.tobytes()
    return _normalize_mode(img), exif_bytes, icc


def load_preview(path: Path, max_side: int) -> np.ndarray:
    """Charge rapidement une version réduite (RGB uint8) pour l'analyse et l'affichage.

    Pour les JPEG, ``draft`` laisse le décodeur réduire l'image dès la
    lecture : bien plus rapide sur des photos de 20 Mpx et plus.
    """
    with Image.open(path) as src:
        if src.format == "JPEG":
            src.draft("RGB", (max_side, max_side))
        src.load()
        img = ImageOps.exif_transpose(src)
    img = _normalize_mode(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return np.asarray(img)


def save_image(
    img: Image.Image,
    path: Path,
    exif_bytes: bytes | None = None,
    icc: bytes | None = None,
    jpeg_quality: int = 95,
) -> None:
    """Enregistre ``img`` ; le format est déduit de l'extension de ``path``."""
    ext = path.suffix.lower()
    params: dict = {}
    if icc:
        params["icc_profile"] = icc
    if exif_bytes and ext in (_JPEG_EXT | {".tif", ".tiff", ".webp", ".png"}):
        params["exif"] = exif_bytes
    if ext in _JPEG_EXT:
        if img.mode == "RGBA":
            img = img.convert("RGB")
        params.update(quality=jpeg_quality, optimize=True)
    elif ext in (".tif", ".tiff"):
        params["compression"] = "tiff_lzw"
    elif ext == ".webp":
        params.update(quality=jpeg_quality, method=4)
    img.save(path, **params)
