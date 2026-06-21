"""Fase 0 — Línea base de Grad-CAM + diagnóstico del recorte HSV ACTUAL.

Para cada imagen muestreada de Test/:
  - corre el modelo actual y guarda el overlay Grad-CAM (estado de hoy),
  - calcula la fracción foliar y el bbox del HSV actual, y guarda un preview del
    recorte que HARÍA hoy (para evidenciar si corta lesión necrótica — Riesgo B).

Salidas en outputs/baseline_roi/. No modifica nada del pipeline.
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
from src.data.roi import _largest_component_bbox, foliar_fraction, leaf_mask  # noqa: E402
from src.inference.predict import AgriVisionPredictor  # noqa: E402

OUT = ROOT / "outputs" / "baseline_roi"
OUT.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 5

ckpt = sorted((ROOT / "artifacts" / "models").glob("*.pt"))[-1]
print(f"Modelo: {ckpt.name}")
predictor = AgriVisionPredictor(str(ckpt))

print(f"{'imagen':40s} {'pred':14s} {'conf':>6s} {'sesgo':>6s} {'foliar%':>8s} {'recorte_bbox'}")
for cls_dir in sorted(cfg.TEST_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    for img_path in sorted(cls_dir.glob("*"))[:N_PER_CLASS]:
        pil = Image.open(img_path).convert("RGB")
        r = predictor.predict(pil, image_id=f"{cls_dir.name}__{img_path.stem}", save_outputs=False)

        # Overlay Grad-CAM actual.
        over = cv2.cvtColor(r["overlay_array"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(OUT / f"overlay_{cls_dir.name}__{img_path.stem}.png"), over)

        # Cómo recortaría el HSV ACTUAL (sobre la imagen original).
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        ff = foliar_fraction(bgr)
        bbox = _largest_component_bbox(leaf_mask(bgr))
        # Preview: imagen original con el bbox dibujado + máscara al lado.
        prev = bgr.copy()
        if bbox is not None:
            x, y, w, h = bbox
            cv2.rectangle(prev, (x, y), (x + w, y + h), (0, 0, 255), 3)
        mask3 = cv2.cvtColor(leaf_mask(bgr), cv2.COLOR_GRAY2BGR)
        prev = cv2.resize(prev, (256, 256)); mask3 = cv2.resize(mask3, (256, 256))
        cv2.imwrite(str(OUT / f"hsvcrop_{cls_dir.name}__{img_path.stem}.png"),
                    np.hstack([prev, mask3]))

        print(f"{img_path.name:40s} {r['clase_predicha']:14s} {r['confianza']:6.2f} "
              f"{str(r['alerta_sesgo']):>6s} {ff*100:7.1f}% {bbox}")

predictor.close()
print(f"\nOK. Overlays y previews de recorte HSV en: {OUT}")
