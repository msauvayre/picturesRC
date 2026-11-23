"""Construit l'application autonome (dossier dist/picturesRC) avec PyInstaller.

À lancer depuis la racine du projet, sur l'OS cible (Windows OU Debian :
PyInstaller ne fait pas de compilation croisée) :

    python packaging/build.py
"""

from __future__ import annotations

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
    str(ROOT / "packaging" / "launcher.py"),
]
print(" ".join(cmd))
raise SystemExit(subprocess.call(cmd, cwd=ROOT))
