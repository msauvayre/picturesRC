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

"""Construit l'application autonome (dossier dist/picturesRC) avec PyInstaller.

À lancer depuis la racine du projet, sur l'OS cible (Windows OU Debian :
PyInstaller ne fait pas de compilation croisée) :

    python packaging/build.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

cmd = [
    sys.executable, "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--windowed",  # pas de console sous Windows
    "--name", "picturesRC",
    "--paths", str(ROOT / "src"),
    # Modules Qt inutiles : allège nettement le paquet
    "--exclude-module", "PySide6.QtWebEngineCore",
    "--exclude-module", "PySide6.QtWebEngineWidgets",
    "--exclude-module", "PySide6.Qt3DCore",
    "--exclude-module", "PySide6.QtQuick",
    "--exclude-module", "PySide6.QtQml",
    "--exclude-module", "PySide6.QtMultimedia",
    "--exclude-module", "tkinter",
    # Catalogues de traduction (un fichier JSON par langue)
    "--add-data", f"{ROOT / 'src' / 'picturesrc' / 'i18n' / 'locales'}{os.pathsep}picturesrc/i18n/locales",
    str(ROOT / "packaging" / "launcher.py"),
]
print(" ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
