"""Utilidades transversales: semillas, dispositivo, AMP e IO de JSON."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Fija las semillas de Python, NumPy y PyTorch para reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    """Devuelve la GPU si está disponible, si no la CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def device_info(device: torch.device) -> str:
    """Cadena legible con info del dispositivo (para logs)."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        cap = torch.cuda.get_device_capability(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"{name} (sm_{cap[0]}{cap[1]}, {total:.1f} GB) | torch {torch.__version__}"
    return f"CPU | torch {torch.__version__}"


def make_amp(device: torch.device, enabled: bool = True):
    """Crea el GradScaler y una factoría de autocast para precisión mixta.

    Devuelve (scaler, autocast_factory). AMP solo se activa en CUDA.
    """
    use_amp = enabled and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.amp.autocast("cuda", enabled=use_amp)

    return scaler, autocast


def save_json(obj: Any, path: str | Path) -> None:
    """Guarda un objeto serializable a JSON (UTF-8, indentado)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Carga un objeto desde un archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def to_py(obj: Any) -> Any:
    """Convierte recursivamente tipos de NumPy/Torch a primitivos de Python.

    Necesario para que las salidas sean serializables a JSON sin pasos extra.
    """
    if isinstance(obj, dict):
        return {k: to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_py(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj
