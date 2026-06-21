"""Validación y control de calidad de la imagen de entrada.

Antes de diagnosticar, se verifica que la imagen sea utilizable: formato válido,
resolución suficiente, nitidez (varianza del Laplaciano) e iluminación (brillo
medio). Devuelve un reporte que la GUI muestra como "Calidad de imagen".
"""
from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

MIN_RESOLUTION = 64          # px lado menor (coincide con la limpieza del dataset)
BLUR_THRESHOLD = 40.0        # varianza del Laplaciano; menor = borrosa
DARK_THRESHOLD = 40          # brillo medio (0-255) mínimo
BRIGHT_THRESHOLD = 225       # brillo medio máximo


def load_image(data: bytes) -> Image.Image:
    """Carga bytes a PIL.Image RGB; lanza ValueError si no es imagen válida."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # valida integridad
    except (UnidentifiedImageError, OSError):
        raise ValueError("El archivo no es una imagen válida.")
    # verify() invalida el objeto: reabrir.
    return Image.open(io.BytesIO(data)).convert("RGB")


def assess_quality(img: Image.Image) -> dict:
    """Evalúa resolución, nitidez e iluminación. Devuelve un reporte estructurado."""
    w, h = img.size
    arr = np.asarray(img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())

    checks = {
        "resolucion": {
            "ok": min(w, h) >= MIN_RESOLUTION,
            "valor": f"{w}x{h}",
        },
        "nitidez": {
            "ok": sharpness >= BLUR_THRESHOLD,
            "valor": round(sharpness, 1),
            "etiqueta": "OK" if sharpness >= BLUR_THRESHOLD else "borrosa",
        },
        "iluminacion": {
            "ok": DARK_THRESHOLD <= brightness <= BRIGHT_THRESHOLD,
            "valor": round(brightness, 1),
            "etiqueta": ("OK" if DARK_THRESHOLD <= brightness <= BRIGHT_THRESHOLD
                         else ("oscura" if brightness < DARK_THRESHOLD else "sobreexpuesta")),
        },
    }
    aprobada = all(c["ok"] for c in checks.values())
    return {
        "aprobada": aprobada,
        "checks": checks,
        "resumen": _summary(checks),
    }


def _summary(checks: dict) -> str:
    """Texto corto tipo 'Resolucion 224x224 · nitidez OK · iluminacion OK'."""
    return (f"Resolucion {checks['resolucion']['valor']} · "
            f"nitidez {checks['nitidez']['etiqueta']} · "
            f"iluminacion {checks['iluminacion']['etiqueta']}")
