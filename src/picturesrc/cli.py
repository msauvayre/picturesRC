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

"""Interface en ligne de commande : analyse et enregistrement par lot.

Les photos corrigées remplacent les originaux dans leur dossier ; les
originaux sont déplacés (par défaut dans <dossier>/originaux).

Les messages suivent la langue du système, sauf ``--lang`` ou la variable
d'environnement ``PICTURESRC_LANG`` (voir ``picturesrc.i18n``).

Exemples :
    picturesrc-cli D:/photos --dry-run
    picturesrc-cli D:/photos --crop subject
    picturesrc-cli D:/photos --originals E:/archives/originaux
    picturesrc-cli D:/photos --lang en
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import i18n
from .core.imageio import is_supported, load_preview
from .core.inplace import ORIGINALS_DIRNAME, InPlaceSettings, apply_in_place
from .core.pipeline import CROP_MODES
from .core.straighten import ANALYSIS_MAX_SIDE, CONFIDENCE_MEDIUM
from .core.subject import analyze_photo
from .i18n import tr


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
    languages = list(i18n.available_languages())
    # La langue doit être connue avant de construire l'aide : lecture préalable de --lang
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--lang")
    lang = pre.parse_known_args(argv)[0].lang
    if lang:
        i18n.set_language(lang)  # option explicite : prime sur PICTURESRC_LANG
    else:
        i18n.setup()

    parser = argparse.ArgumentParser(prog="picturesrc-cli", description=tr("cli.description"))
    parser.add_argument("sources", nargs="+", type=Path, help=tr("cli.sources"))
    parser.add_argument("--originals", type=Path, help=tr("cli.originals", dirname=ORIGINALS_DIRNAME))
    parser.add_argument("--crop", choices=CROP_MODES, default="subject", help=tr("cli.crop"))
    parser.add_argument("--max-angle", type=float, default=15.0, help=tr("cli.max_angle"))
    parser.add_argument("--min-confidence", type=float, default=CONFIDENCE_MEDIUM, help=tr("cli.min_confidence"))
    parser.add_argument("--quality", type=int, default=95, help=tr("cli.quality"))
    parser.add_argument("--no-json", action="store_true", help=tr("cli.no_json"))
    parser.add_argument("--dry-run", action="store_true", help=tr("cli.dry_run"))
    parser.add_argument("--lang", metavar="CODE", help=tr("cli.lang", languages=", ".join(languages)))
    args = parser.parse_args(argv)

    files = collect_images(args.sources)
    if not files:
        print(tr("cli.no_images"), file=sys.stderr)
        return 1

    settings = InPlaceSettings(originals_dir=args.originals, crop_mode=args.crop, jpeg_quality=args.quality,
                               write_sidecar=not args.no_json)

    for path in files:
        analysis = analyze_photo(load_preview(path, ANALYSIS_MAX_SIDE), max_angle=args.max_angle)
        est = analysis.estimate_for(args.crop)
        angle = est.angle if est.confidence >= args.min_confidence else 0.0
        line = tr("cli.line", name=path.name, angle=est.angle, source=tr("cli.source.contour" if est.source == "contour" else "cli.source.lines"),
                  confidence=est.confidence, level=tr(f"level.{est.level}"))
        subject = None
        if args.crop == "subject":
            subject = analysis.subject
            if subject is not None and subject.confidence < args.min_confidence:
                subject = None  # objet douteux : repli sur la surface maximale
            line += f"  {tr('cli.object')} " + (f"{subject.confidence:4.2f}" if subject else " -- ")
        if args.dry_run:
            print(line)
            continue
        res = apply_in_place(path, angle, settings, est, subject=subject)
        if res.action == "corrigee":
            print(f"{line}  {tr('cli.fixed', original=res.original)}")
        else:
            print(f"{line}  {tr('cli.unchanged')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
