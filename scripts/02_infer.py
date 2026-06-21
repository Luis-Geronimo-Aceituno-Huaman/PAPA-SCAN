"""Punto de entrada de inferencia (CLI).

Esta es la CAPA DE PRESENTACIÓN: carga el predictor, llama a la función pura
`predict()` y muestra el JSON resultante. La lógica de impresión vive aquí, no
en el módulo de inferencia (que permanece desacoplado de toda presentación).

Uso:
  python scripts/02_infer.py --image ruta/a/hoja.jpg
  python scripts/02_infer.py --image hoja.jpg --model artifacts/models/agrivision_v1_2026-06-17.pt
  python scripts/02_infer.py --image hoja.jpg --no-save     # no escribe PNGs
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

import config as cfg  # noqa: E402
from src.inference.predict import AgriVisionPredictor  # noqa: E402
from src.utils.helpers import to_py  # noqa: E402


def _latest_model() -> str:
    """Devuelve el checkpoint .pt más reciente en artifacts/models/."""
    ckpts = sorted(cfg.MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(
            f"No hay modelos en {cfg.MODELS_DIR}. Entrena primero con scripts/01_train.py")
    return str(ckpts[-1])


def main():
    ap = argparse.ArgumentParser(description="Inferencia de AgriVision")
    ap.add_argument("--image", required=True, help="ruta a la imagen de hoja")
    ap.add_argument("--model", default=None, help="ruta al checkpoint .pt (por defecto, el más reciente)")
    ap.add_argument("--no-save", action="store_true", help="no escribir overlay/heatmap a disco")
    args = ap.parse_args()

    model_path = args.model or _latest_model()
    predictor = AgriVisionPredictor(model_path)
    try:
        result = predictor.predict(args.image, save_outputs=not args.no_save)
    finally:
        predictor.close()

    # Quitar arrays de numpy (si --no-save) para imprimir solo el JSON limpio.
    printable = {k: v for k, v in result.items() if not k.endswith("_array")}
    print(json.dumps(to_py(printable), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
