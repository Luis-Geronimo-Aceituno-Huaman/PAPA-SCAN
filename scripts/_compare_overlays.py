"""Fase 4 — Compara overlays Grad-CAM v1 (línea base) vs v2 (fondo enmascarado).

Para las mismas imágenes muestreadas, arma un panel [original | v1 | v2] y reporta
si la alerta de sesgo cambia. Los overlays v1 ya están en outputs/baseline_roi/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))
import config as cfg  # noqa: E402
from src.inference.predict import AgriVisionPredictor  # noqa: E402

BASE = ROOT / "outputs" / "baseline_roi"
OUT = ROOT / "outputs" / "v2_compare"
OUT.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 5
SZ = 224

ckpt = sorted((ROOT / "artifacts" / "models").glob("*.pt"))[-1]
print(f"Modelo v2: {ckpt.name}\n")
predictor = AgriVisionPredictor(str(ckpt))


def _sq(bgr):
    return cv2.resize(bgr, (SZ, SZ))


print(f"{'imagen':46s} {'pred':14s} {'conf':>6s} {'sesgo_v2':>9s}")
for cls_dir in sorted(cfg.TEST_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    for img_path in sorted(cls_dir.glob("*"))[:N_PER_CLASS]:
        pil = Image.open(img_path).convert("RGB")
        tag = f"{cls_dir.name}__{img_path.stem}"
        r = predictor.predict(pil, image_id=tag, save_outputs=False)
        v2 = cv2.cvtColor(r["overlay_array"], cv2.COLOR_RGB2BGR)

        orig = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        v1_path = BASE / f"overlay_{tag}.png"
        v1 = cv2.imread(str(v1_path)) if v1_path.exists() else np.zeros((SZ, SZ, 3), np.uint8)

        panel = np.hstack([_sq(orig), _sq(v1), _sq(v2)])
        cv2.imwrite(str(OUT / f"cmp_{tag}.png"), panel)
        print(f"{img_path.name:46s} {r['clase_predicha']:14s} {r['confianza']:6.2f} "
              f"{str(r['alerta_sesgo']):>9s}")

predictor.close()
print(f"\nPaneles [original | v1 | v2] en: {OUT}")
