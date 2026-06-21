"""Pone la raíz del proyecto en el sys.path para reutilizar el código original.

El sistema multiagente NO duplica la CNN ni los servicios: los importa del
proyecto base (``config``, ``src``, ``app.services``). Importar este módulo
(o llamar a :func:`ensure_paths`) deja todo listo para esos imports.
"""
from __future__ import annotations

import sys
from pathlib import Path

# app/multiagente/core/bootstrap.py -> parents[3] = raíz del proyecto.
ROOT_DIR = Path(__file__).resolve().parents[3]
OUTPUTS_DIR = Path(__file__).resolve().parents[1] / "outputs"


def ensure_paths() -> Path:
    """Garantiza que la raíz del proyecto sea importable. Devuelve la raíz."""
    root = str(ROOT_DIR)
    if root not in sys.path:
        sys.path.insert(0, root)
    cfg_dir = str(ROOT_DIR / "config")
    if cfg_dir not in sys.path:
        sys.path.insert(0, cfg_dir)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    return ROOT_DIR


ensure_paths()
