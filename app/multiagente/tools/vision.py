"""Herramientas de PERCEPCIÓN: reutilizan la CNN AgriVision + Grad-CAM.

Envuelven `app.services.diagnosis` (que a su vez carga `AgriVisionPredictor`,
el modelo entrenado del proyecto). No se reentrena nada: solo se consume el .pt.

La herramienta NO escribe archivos: devuelve los escalares del diagnóstico y los
arrays en memoria (foto, enmascarada, overlay, heatmap, CAM). Quién persista los
medios lo decide el agente vía un MediaSink (web vs CLI).
"""
from __future__ import annotations

from app.multiagente.core import bootstrap  # noqa: F401  (configura sys.path)
from app.multiagente.tools.registry import registry


@registry.register(
    "diagnosticar_imagen",
    "Clasifica una hoja de papa (sana / tizón tardío / tizón temprano) con la CNN "
    "y genera el Grad-CAM. Devuelve clase, confianza calibrada, probabilidades, "
    "estado de confianza, alerta de sesgo, capa usada y los arrays de imagen.",
)
def diagnosticar_imagen(image) -> dict:
    """`image` puede ser una ruta (str/Path) o un PIL.Image ya cargado."""
    from pathlib import Path

    from PIL import Image

    from app.services import diagnosis

    pil = (Image.open(image).convert("RGB")
           if isinstance(image, (str, Path)) else image.convert("RGB"))
    pred = diagnosis.diagnose(pil)               # CNN + Grad-CAM (arrays en memoria)
    zona = diagnosis.describe_gradcam_zone(pred.get("cam"))

    return {
        "clase": pred["clase_predicha"],
        "confianza": pred["confianza"],
        "estado_confianza": pred["estado_confianza"],
        "umbral_clase": pred["umbral_clase"],
        "probabilidades": pred["probabilidades"],
        "alerta_sesgo": bool(pred["alerta_sesgo"]),
        "zona_gradcam": zona,
        "capa_usada": pred["capa_usada"],
        "metrica_critica_clase": pred.get("metrica_critica_clase", {}),
        "version_modelo": pred["version_modelo"],
        # arrays (se consumen por severidad y por el MediaSink; no se serializan):
        "_pil": pil,
        "_cam": pred.get("cam"),
        "_entrada": pred.get("entrada_array"),    # imagen enmascarada que vio el modelo
        "_overlay": pred.get("overlay_array"),
        "_heat": pred.get("heatmap_array"),
    }
