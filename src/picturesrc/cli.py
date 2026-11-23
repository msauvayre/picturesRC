"""Interface en ligne de commande : analyse et enregistrement par lot.

Les photos corrigées remplacent les originaux dans leur dossier ; les
originaux sont déplacés (par défaut dans <dossier>/originaux).

Exemples :
    picturesrc-cli D:/photos --dry-run
    picturesrc-cli D:/photos --crop subject
    picturesrc-cli D:/photos --originals E:/archives/originaux
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.imageio import is_supported, load_preview
from .core.inplace import ORIGINALS_DIRNAME, InPlaceSettings, apply_in_place
from .core.pipeline import CROP_MODES
from .core.straighten import ANALYSIS_MAX_SIDE, CONFIDENCE_MEDIUM
from .core.subject import analyze_photo


def collect_images(sources: list[Path]) -> list[Path]:
    files: list[Path] = []
    for src in sources:
        if src.is_dir():
            files.extend(sorted(p for p in src.iterdir()
                                if p.is_file() and is_supported(p) and ".picturesrc-tmp" not in p.name))
        elif src.is_file() and is_supported(src):
            files.append(src)
    return files


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="picturesrc-cli", description="Redresse l'horizon et rogne des photos.")
    parser.add_argument("sources", nargs="+", type=Path, help="fichiers ou dossiers")
    parser.add_argument("--originals", type=Path,
                        help=f"où déplacer les originaux des photos corrigées "
                             f"(défaut : <dossier de la photo>/{ORIGINALS_DIRNAME})")
    parser.add_argument("--crop", choices=CROP_MODES, default="subject",
                        help="mode de rognage : subject = autour de l'objet central (défaut), "
                             "aspect, max_area, none")
    parser.add_argument("--max-angle", type=float, default=15.0, help="inclinaison maximale recherchée en degrés")
    parser.add_argument("--min-confidence", type=float, default=CONFIDENCE_MEDIUM,
                        help="en dessous, l'image n'est pas corrigée et reste telle quelle (défaut : %(default)s)")
    parser.add_argument("--quality", type=int, default=95, help="qualité JPEG/WebP (défaut : 95)")
    parser.add_argument("--no-json", action="store_true", help="n'écrit pas les fichiers .picturesrc.json")
    parser.add_argument("--dry-run", action="store_true", help="analyse seulement, n'écrit rien")
    args = parser.parse_args(argv)

    files = collect_images(args.sources)
    if not files:
        print("Aucune image prise en charge trouvée.", file=sys.stderr)
        return 1

    settings = InPlaceSettings(originals_dir=args.originals, crop_mode=args.crop, jpeg_quality=args.quality,
                               write_sidecar=not args.no_json)

    for path in files:
        analysis = analyze_photo(load_preview(path, ANALYSIS_MAX_SIDE), max_angle=args.max_angle)
        est = analysis.estimate_for(args.crop)
        angle = est.angle if est.confidence >= args.min_confidence else 0.0
        line = (f"{path.name:40.40s}  angle {est.angle:+6.2f}° ({est.source:7s})"
                f"  confiance {est.confidence:4.2f} ({est.level})")
        subject = None
        if args.crop == "subject":
            subject = analysis.subject
            if subject is not None and subject.confidence < args.min_confidence:
                subject = None  # objet douteux : repli sur la surface maximale
            line += "  objet " + (f"{subject.confidence:4.2f}" if subject else " -- ")
        if args.dry_run:
            print(line)
            continue
        res = apply_in_place(path, angle, settings, est, subject=subject)
        if res.action == "corrigee":
            print(f"{line}  corrigée, original -> {res.original}")
        else:
            print(f"{line}  [sans correction, laissée telle quelle]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
