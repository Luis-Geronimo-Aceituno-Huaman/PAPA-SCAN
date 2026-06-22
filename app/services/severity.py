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
# Rango HSV de tejido necrótico (marrón de lesión), igual que el módulo ROI.
# Incluir el marrón en la SILUETA es crítico: con solo verde, una lesión del BORDE
# de la hoja queda fuera del contorno y el daño se subestima (medía ~1 % en lesiones
# de borde grandes). Con verde ∪ marrón la silueta encierra la hoja completa.
_BROWN_LOW = np.array([0, 25, 25], dtype=np.uint8)
_BROWN_HIGH = np.array([25, 255, 220], dtype=np.uint8)

# Umbrales de fracción lesionada -> nivel de severidad.
LEVE_MAX = 0.08
MODERADA_MAX = 0.25

# Cruce de seguridad (Capa 1 anti-falso-negativo): si el modelo dice "sana" pero
# el área lesionada medida por color supera este umbral, se sospecha un FALSO
# NEGATIVO y el caso se degrada a baja confianza y se deriva a un técnico.
# Ver `AgenteSeveridad._chequeo_falso_negativo`.
CONFLICTO_SANA_MIN = 0.10


def _healthy_mask(img_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _HSV_LOW, _HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _necrotic_mask(img_bgr: np.ndarray) -> np.ndarray:
    """Tejido marrón/necrótico (lesión), para que la silueta incluya el borde dañado."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _BROWN_LOW, _BROWN_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def _leaf_silhouette(tissue: np.ndarray) -> np.ndarray:
    """Silueta de la hoja completa: cierra y rellena el contorno mayor.

    `tissue` es la unión de tejido sano (verde) y necrótico (marrón), de modo que
    el contorno encierra la hoja entera y no recorta las lesiones del borde.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sil = np.zeros_like(tissue)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        cv2.drawContours(sil, [biggest], -1, 255, thickness=cv2.FILLED)
    return sil


def lesion_fraction(img) -> tuple[float, np.ndarray]:
    """Fracción de área foliar lesionada y la máscara de lesión (0/255)."""
    img_rgb = np.asarray(img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    healthy = _healthy_mask(img_bgr)
    # Silueta disease-agnostic: verde (sano) ∪ marrón (necrótico). Así el contorno
    # encierra la hoja completa, incluidas las lesiones del borde.
    tissue = cv2.bitwise_or(healthy, _necrotic_mask(img_bgr))
    silhouette = _leaf_silhouette(tissue)

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
    """Estima la severidad. Devuelve nivel, fracción lesionada e IoU de activación.

    El área lesionada se mide SIEMPRE, también cuando el diagnóstico es "sana":
    sirve de chequeo cruzado para detectar falsos negativos (lesión presente pese
    a un veredicto "sana"). Antes esta rama devolvía 0 sin mirar la imagen, lo que
    desactivaba la única herramienta capaz de delatar ese error.
    """
    frac, lesion = lesion_fraction(img)
    iou = activation_lesion_iou(cam, lesion) if cam is not None else 0.0

    if is_healthy:
        nivel = "ninguna"
    elif frac < LEVE_MAX:
        nivel = "leve"
    elif frac < MODERADA_MAX:
        nivel = "moderada"
    else:
        nivel = "severa"

    return {
        "nivel": nivel,
        "fraccion_lesion": round(frac, 3),
        "iou_activacion": round(iou, 3),
    }
