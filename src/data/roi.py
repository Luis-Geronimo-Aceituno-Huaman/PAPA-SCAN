"""Paso 2 — Segmentación de hoja: recorte ROI y/o enmascarado de fondo.

Estrategia determinística (idéntica en entrenamiento e inferencia):

1. Segmentación disease-agnostic: se une la región verde/amarilla (hoja sana,
   halo clorótico) con la marrón/necrótica (lesiones de tizón). Incluir el marrón
   es CRÍTICO: con solo verde/amarillo se recortaría/enmascararía la evidencia de
   la enfermedad (Riesgo B) y la auditoría de sesgo daría falsos positivos.
2. Limpieza morfológica + RELLENO de huecos + componente conexo más grande →
   silueta sólida de la hoja (las lesiones internas cuentan como hoja).
3. Según config: recorte al bounding box (con margen) y/o enmascarado del fondo
   (los píxeles no-hoja se ponen al color medio de ImageNet → ~0 tras normalizar,
   neutro). Enmascarar el fondo mata el atajo de fondo que aprende el modelo.

Fallback seguro: si no se detecta hoja suficiente (`ROI_MIN_FOLIAR_FRAC`), se
devuelve la imagen original sin tocar. NO usa segmentación semántica aprendida
(añadiría dependencias y latencia, y podría fallar offline en campo).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

# Rangos HSV (OpenCV: H in [0,179], S/V in [0,255]).
# Verde/amarillo: hoja sana y halo clorótico.
_GREEN_LOW = np.array([20, 30, 30], dtype=np.uint8)
_GREEN_HIGH = np.array([90, 255, 255], dtype=np.uint8)
# Marrón/necrótico: lesiones de tizón (naranjas-marrones, baja saturación/valor).
# Cubre H rojo-naranja [0..25] con S y V moderados, evitando negros puros (sombra).
_BROWN_LOW = np.array([0, 25, 25], dtype=np.uint8)
_BROWN_HIGH = np.array([25, 255, 220], dtype=np.uint8)

# Color de relleno del fondo = media de ImageNet en 0-255 (tras Normalize → ~0).
_FILL_RGB = tuple(int(round(255 * m)) for m in cfg.IMAGENET_MEAN)  # (124, 116, 104)


def _raw_leaf_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Unión verde/amarillo ∪ marrón, con limpieza morfológica (uint8 0/255)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, _GREEN_LOW, _GREEN_HIGH)
    brown = cv2.inRange(hsv, _BROWN_LOW, _BROWN_HIGH)
    mask = cv2.bitwise_or(green, brown)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Rellena los huecos internos de la máscara (lesiones rodeadas de hoja)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return filled


def _largest_component_bbox(mask: np.ndarray):
    """Bounding box (x, y, w, h) del componente conexo más grande, o None."""
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num <= 1:  # solo fondo
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = 1 + int(np.argmax(areas))
    x = stats[largest, cv2.CC_STAT_LEFT]
    y = stats[largest, cv2.CC_STAT_TOP]
    w = stats[largest, cv2.CC_STAT_WIDTH]
    h = stats[largest, cv2.CC_STAT_HEIGHT]
    return int(x), int(y), int(w), int(h)


def leaf_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Silueta sólida de la hoja (uint8 0/255): componente mayor con huecos rellenos.

    Disease-agnostic (verde/amarillo ∪ marrón necrótico). Es la máscara que usan
    tanto el recorte/enmascarado como la auditoría de sesgo del Grad-CAM, de modo
    que las lesiones marrones cuentan como HOJA, no como fondo.
    """
    raw = _raw_leaf_mask(img_bgr)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(raw, connectivity=8)
    if num <= 1:
        return np.zeros_like(raw)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    solid = np.where(labels == largest, 255, 0).astype(np.uint8)
    return _fill_holes(solid)


def foliar_fraction(img_bgr: np.ndarray) -> float:
    """Fracción del área de la imagen ocupada por la silueta de la hoja."""
    return float((leaf_mask(img_bgr) > 0).mean())


def apply_roi(img: Image.Image,
              crop: bool | None = None,
              bg_mask: bool | None = None,
              margin: float = cfg.ROI_MARGIN) -> Image.Image:
    """Aplica recorte y/o enmascarado de fondo de forma determinística.

    `crop`/`bg_mask` por defecto leen los flags de config. Si no se detecta hoja
    suficiente (`ROI_MIN_FOLIAR_FRAC`), devuelve la imagen original (fallback).
    """
    crop = cfg.ROI_CROP_ENABLED if crop is None else crop
    bg_mask = cfg.ROI_BG_MASK_ENABLED if bg_mask is None else bg_mask
    if not (crop or bg_mask):
        return img

    rgb = np.asarray(img.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = leaf_mask(bgr)

    if (mask > 0).mean() < cfg.ROI_MIN_FOLIAR_FRAC:
        return img  # fallback: no se detectó hoja fiable, no tocar

    if bg_mask:
        m = (mask > 0)[:, :, None]
        rgb = np.where(m, rgb, np.array(_FILL_RGB, dtype=np.uint8)).astype(np.uint8)

    if crop:
        bbox = _largest_component_bbox(mask)
        if bbox is not None:
            x, y, w, h = bbox
            H, W = mask.shape
            mx, my = int(w * margin), int(h * margin)
            x0, y0 = max(0, x - mx), max(0, y - my)
            x1, y1 = min(W, x + w + mx), min(H, y + h + my)
            if x1 > x0 and y1 > y0:
                rgb = rgb[y0:y1, x0:x1]

    return Image.fromarray(rgb)


def crop_to_leaf(img: Image.Image, margin: float = cfg.ROI_MARGIN) -> Image.Image:
    """Recorta a la región foliar (sin enmascarar). Compat retro / uso directo."""
    return apply_roi(img, crop=True, bg_mask=False, margin=margin)


def maybe_apply_roi(img: Image.Image) -> Image.Image:
    """Aplica recorte/enmascarado según los flags de config (usado en transforms)."""
    return apply_roi(img)


# Alias retro-compatible (transforms antiguos importaban este nombre).
maybe_crop_to_leaf = maybe_apply_roi
