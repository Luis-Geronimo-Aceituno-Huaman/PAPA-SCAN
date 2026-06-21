"""Herramienta de SEVERIDAD: reutiliza `app.services.severity`.

Estima la fracción de área foliar lesionada y el IoU entre la zona caliente del
Grad-CAM y la lesión (confirma que el modelo miró la lesión real, no el fondo).
"""
from __future__ import annotations

from multiagente.core import bootstrap  # noqa: F401
from multiagente.tools.registry import registry


@registry.register(
    "estimar_severidad",
    "Estima la severidad (ninguna/leve/moderada/severa) por el área foliar "
    "lesionada y el IoU activación-lesión. Requiere la imagen y el mapa Grad-CAM.",
)
def estimar_severidad(pil_image, cam=None, is_healthy: bool = False) -> dict:
    from app.services import severity
    return severity.estimate_severity(pil_image, cam=cam, is_healthy=is_healthy)
