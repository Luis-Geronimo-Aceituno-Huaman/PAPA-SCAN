"""Servicio de diagnóstico: envuelve el modelo AgriVision ya entrenado.

Reutiliza `AgriVisionPredictor` (CNN + Grad-CAM en el mismo forward/backward).
El modelo se carga UNA vez (singleton) y no se reentrena: solo se consume.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

import config as cfg  # noqa: E402
from src.inference.predict import AgriVisionPredictor  # noqa: E402

_predictor: AgriVisionPredictor | None = None


def _latest_model() -> str:
    ckpts = sorted(cfg.MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f"No hay modelo entrenado en {cfg.MODELS_DIR}. Entrena con scripts/01_train.py")
    return str(ckpts[-1])


def get_predictor() -> AgriVisionPredictor:
    """Devuelve el predictor cargado (lo crea la primera vez)."""
    global _predictor
    if _predictor is None:
        _predictor = AgriVisionPredictor(_latest_model())
    return _predictor


def diagnose(pil_image) -> dict:
    """Ejecuta CNN + Grad-CAM y devuelve el resultado (con arrays en memoria)."""
    predictor = get_predictor()
    return predictor.predict(pil_image, save_outputs=False)


def describe_gradcam_zone(cam: np.ndarray) -> str:
    """Describe en palabras dónde se concentró la atención del modelo.

    Para el LLM (campo `zona_gradcam`): el modelo también ve el mapa, pero esta
    pista textual ayuda a respuestas más concretas.
    """
    if cam is None or cam.max() <= 0:
        return "sin una zona de atención clara"

    h, w = cam.shape
    # Centro de masa de la activación.
    ys, xs = np.mgrid[0:h, 0:w]
    total = cam.sum() + 1e-8
    cy = float((ys * cam).sum() / total) / h   # 0 arriba, 1 abajo
    cx = float((xs * cam).sum() / total) / w   # 0 izq, 1 der

    vert = "parte superior" if cy < 0.4 else ("parte inferior" if cy > 0.6 else "parte central")
    horiz = "izquierda" if cx < 0.4 else ("derecha" if cx > 0.6 else "centro")

    # ¿Concentrada (manchas) o difusa?
    high = (cam >= 0.5).mean()
    forma = "manchas concentradas" if high < 0.25 else "una región amplia"

    return f"{forma} en la {vert} de la hoja, hacia la {horiz}"
