"""Herramientas de PERCEPCIÓN: reutilizan la CNN AgriVision + Grad-CAM.

Envuelven `app.services.diagnosis` (que a su vez carga `AgriVisionPredictor`,
el modelo entrenado del proyecto). No se reentrena nada: solo se consume el .pt.
"""
from __future__ import annotations

from multiagente.core import bootstrap  # noqa: F401  (configura sys.path)
from multiagente.core.bootstrap import OUTPUTS_DIR
from multiagente.tools.registry import registry


@registry.register(
    "diagnosticar_imagen",
    "Clasifica una hoja de papa (sana / tizón tardío / tizón temprano) con la CNN "
    "y genera el Grad-CAM. Devuelve clase, confianza calibrada, probabilidades, "
    "estado de confianza, alerta de sesgo y rutas de la foto/heatmap.",
)
def diagnosticar_imagen(image_path: str, case_id: str = "caso") -> dict:
    import cv2
    from PIL import Image

    from app.services import diagnosis

    pil = Image.open(image_path).convert("RGB")
    pred = diagnosis.diagnose(pil)               # CNN + Grad-CAM (arrays en memoria)
    zona = diagnosis.describe_gradcam_zone(pred.get("cam"))

    # Persistimos foto y heatmap para que el agente Explicador (VLM) los lea.
    foto_path = OUTPUTS_DIR / f"{case_id}_foto.jpg"
    heatmap_path = OUTPUTS_DIR / f"{case_id}_heatmap.png"
    overlay_path = OUTPUTS_DIR / f"{case_id}_overlay.png"
    pil.save(foto_path, format="JPEG", quality=92)
    cv2.imwrite(str(heatmap_path), cv2.cvtColor(pred["heatmap_array"], cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(overlay_path), cv2.cvtColor(pred["overlay_array"], cv2.COLOR_RGB2BGR))

    return {
        "clase": pred["clase_predicha"],
        "confianza": pred["confianza"],
        "estado_confianza": pred["estado_confianza"],
        "umbral_clase": pred["umbral_clase"],
        "probabilidades": pred["probabilidades"],
        "alerta_sesgo": pred["alerta_sesgo"],
        "zona_gradcam": zona,
        "version_modelo": pred["version_modelo"],
        "foto_path": str(foto_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        # arrays para la herramienta de severidad (no se serializan al reporte):
        "_cam": pred.get("cam"),
        "_pil": pil,
    }
