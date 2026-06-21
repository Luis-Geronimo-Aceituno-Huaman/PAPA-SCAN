"""Estimación de severidad a partir del área foliar lesionada.

Idea: la máscara de tejido sano (verde/amarillo) tiene "huecos" donde hay
lesiones necróticas (marrón) que no son verdes. Si rellenamos la silueta de la
hoja y restamos el tejido sano, los huecos resultantes son el tejido lesionado.
La fracción lesionada se mapea a leve / moderada / severa.

También calcula el IoU entre la zona de alta activación del Grad-CAM y la zona
lesionada: un IoU alto confirma que el modelo miró la lesión real (no el fondo).
"""
from __future__ import annotations

import cv2
import numpy as np

# Rango HSV de tejido sano (verde/amarillo), igual que el módulo ROI del modelo.
_HSV_LOW = np.array([20, 30, 30], dtype=np.uint8)
_HSV_HIGH = np.array([90, 255, 255], dtype=np.uint8)

# Umbrales de fracción lesionada -> nivel de severidad.
LEVE_MAX = 0.08
MODERADA_MAX = 0.25


def _healthy_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _leaf_silhouette(healthy: np.ndarray) -> np.ndarray:
    """Silueta de la hoja completa: cierra y rellena el contorno mayor."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(healthy, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(healthy)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(sil, [biggest], -1, 255, thickness=cv2.FILLED)
    return sil


def lesion_fraction(img) -> tuple[float, np.ndarray]:
    """Fracción de área foliar lesionada y la máscara de lesión (0/255)."""
    img_rgb = np.asarray(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    healthy = _healthy_mask(img_bgr)
    silhouette = _leaf_silhouette(healthy)

    leaf_area = float((silhouette > 0).sum())
    if leaf_area < 1:
        return 0.0, np.zeros_like(healthy)

    # Lesión = dentro de la hoja pero NO tejido sano.
    lesion = cv2.bitwise_and(silhouette, cv2.bitwise_not(healthy))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, kernel)
    frac = float((lesion > 0).sum()) / leaf_area
    return min(frac, 1.0), lesion


def activation_lesion_iou(cam: np.ndarray, lesion_mask: np.ndarray,
                          cam_thresh: float = 0.5) -> float:
    """IoU entre la zona de alta activación del Grad-CAM y la zona lesionada."""
    if cam is None or lesion_mask is None:
        return 0.0
    cam_resized = cv2.resize(cam.astype(np.float32), lesion_mask.shape[::-1])
    cam_bin = (cam_resized >= cam_thresh).astype(np.uint8)
    lesion_bin = (lesion_mask > 0).astype(np.uint8)
    inter = float(np.logical_and(cam_bin, lesion_bin).sum())
    union = float(np.logical_or(cam_bin, lesion_bin).sum()) + 1e-8
    return inter / union


def estimate_severity(img, diagnostico_idx: int | None = None,
                      cam: np.ndarray | None = None,
                      is_healthy: bool = False) -> dict:
    """Estima la severidad. Devuelve nivel, fracción lesionada e IoU de activación."""
    if is_healthy:
        return {"nivel": "ninguna", "fraccion_lesion": 0.0, "iou_activacion": 0.0}

    frac, lesion = lesion_fraction(img)
    if frac < LEVE_MAX:
        nivel = "leve"
    elif frac < MODERADA_MAX:
        nivel = "moderada"
    else:
        nivel = "severa"

    iou = activation_lesion_iou(cam, lesion) if cam is not None else 0.0
    return {
        "nivel": nivel,
        "fraccion_lesion": round(frac, 3),
        "iou_activacion": round(iou, 3),
    }
