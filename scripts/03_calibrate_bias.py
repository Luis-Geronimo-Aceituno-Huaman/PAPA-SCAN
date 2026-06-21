"""Calibra el umbral de la alerta de sesgo (masa del Grad-CAM sobre fondo).

Mide, sobre una muestra estratificada de imágenes de ENTRENAMIENTO que el modelo
clasifica correctamente, qué fracción de la masa del Grad-CAM cae sobre fondo
(no-hoja). Fija el umbral en un percentil alto (p90 por defecto): así "sesgo"
significa estadísticamente atípico respecto a predicciones correctas. NO usa el
set de test (evita cualquier contaminación). Actualiza los metadatos del .pt.

Uso:
  python scripts/03_calibrate_bias.py
  python scripts/03_calibrate_bias.py --per-class 60 --percentile 90
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

import config as cfg  # noqa: E402
from src.data.dataset import scan_split  # noqa: E402
from src.inference.gradcam import background_mass, leaf_mask_from_input  # noqa: E402
from src.inference.predict import AgriVisionPredictor  # noqa: E402
from src.utils.helpers import load_json, save_json, to_py  # noqa: E402


def _latest_model() -> str:
    ckpts = sorted(cfg.MODELS_DIR.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    if not ckpts:
        raise FileNotFoundError(f"No hay modelos en {cfg.MODELS_DIR}")
    return str(ckpts[-1])


def _stratified_sample(paths, labels, per_class, seed=cfg.RANDOM_SEED):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    sel = []
    for c in range(len(cfg.CLASSES)):
        idx = np.where(labels == c)[0]
        take = rng.choice(idx, size=min(per_class, len(idx)), replace=False)
        sel.extend(take.tolist())
    return [paths[i] for i in sel], labels[np.asarray(sel)]


def main():
    ap = argparse.ArgumentParser(description="Calibrar umbral de alerta de sesgo")
    ap.add_argument("--model", default=None)
    ap.add_argument("--per-class", type=int, default=60)
    ap.add_argument("--percentile", type=float, default=90.0)
    args = ap.parse_args()

    model_path = args.model or _latest_model()
    print(f"[modelo] {model_path}")
    predictor = AgriVisionPredictor(model_path)

    paths, labels = scan_split(cfg.TRAIN_VALID_DIR)
    s_paths, s_labels = _stratified_sample(paths, labels, args.per_class)
    print(f"[muestra] {len(s_paths)} imágenes de entrenamiento "
          f"({args.per_class}/clase)")

    vals, n_correct = [], 0
    for p, lbl in zip(s_paths, s_labels):
        pil = Image.open(p).convert("RGB")
        x = predictor.transform(pil).unsqueeze(0).to(predictor.device)
        _, cam, pred_idx = predictor.gradcam(x, class_idx=None)
        if pred_idx == int(lbl):  # solo imágenes bien clasificadas
            n_correct += 1
            mask = leaf_mask_from_input(x)
            vals.append(background_mass(cam, mask))
    predictor.close()

    vals = np.asarray(vals)
    threshold = float(np.percentile(vals, args.percentile))
    print(f"[stats] correctas={n_correct}/{len(s_paths)} "
          f"| masa_fondo: media={vals.mean():.3f} p50={np.median(vals):.3f} "
          f"p{int(args.percentile)}={threshold:.3f} max={vals.max():.3f}")

    # Actualizar metadatos del checkpoint y del JSON.
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    ckpt["metadata"]["umbral_sesgo_fondo"] = threshold
    ckpt["metadata"]["sesgo_calibracion"] = {
        "percentil": args.percentile,
        "n_muestras": int(n_correct),
        "media_masa_fondo": float(vals.mean()),
    }
    torch.save(ckpt, model_path)

    json_path = Path(model_path).with_name(Path(model_path).stem + "_metadata.json")
    if json_path.exists():
        meta = load_json(json_path)
        meta["umbral_sesgo_fondo"] = threshold
        meta["sesgo_calibracion"] = ckpt["metadata"]["sesgo_calibracion"]
        save_json(to_py(meta), json_path)

    print(f"[guardado] umbral_sesgo_fondo={threshold:.3f} -> {model_path}")


if __name__ == "__main__":
    main()
