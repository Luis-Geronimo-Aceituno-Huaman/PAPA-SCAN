"""Fase 1 — Preview de la nueva segmentación (NO reentrena, NO toca el modelo).

Para imágenes muestreadas de Test/ genera un panel:
  [ original | máscara silueta | fondo enmascarado | enmascarado + recortado ]
así se verifica que la máscara disease-agnostic NO corta lesiones necróticas y
que el fondo queda limpio. Salidas en outputs/roi_preview/.
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
from src.data.roi import apply_roi, foliar_fraction, leaf_mask  # noqa: E402

OUT = ROOT / "outputs" / "roi_preview"
OUT.mkdir(parents=True, exist_ok=True)
N_PER_CLASS = 5
SZ = 224


def _sq(img_bgr):
    return cv2.resize(img_bgr, (SZ, SZ))


print(f"{'imagen':46s} {'foliar%':>8s}")
for cls_dir in sorted(cfg.TEST_DIR.iterdir()):
    if not cls_dir.is_dir():
        continue
    for img_path in sorted(cls_dir.glob("*"))[:N_PER_CLASS]:
        pil = Image.open(img_path).convert("RGB")
        bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        ff = foliar_fraction(bgr)

        mask = leaf_mask(bgr)
        masked = apply_roi(pil, crop=False, bg_mask=True)
        both = apply_roi(pil, crop=True, bg_mask=True)

        panel = np.hstack([
            _sq(bgr),
            _sq(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)),
            _sq(cv2.cvtColor(np.asarray(masked), cv2.COLOR_RGB2BGR)),
            _sq(cv2.cvtColor(np.asarray(both), cv2.COLOR_RGB2BGR)),
        ])
        cv2.imwrite(str(OUT / f"roi_{cls_dir.name}__{img_path.stem}.png"), panel)
        print(f"{img_path.name:46s} {ff*100:7.1f}%")

print(f"\nOK. Paneles [orig | máscara | enmascarado | +recorte] en: {OUT}")
