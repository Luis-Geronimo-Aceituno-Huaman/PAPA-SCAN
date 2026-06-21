"""Orquestador del flujo de diagnóstico + explicación (brief §4).

Encadena: validación → CNN → Grad-CAM → severidad → motor de reglas → LLM.
La explicación del LLM es OPCIONAL: si falla, el resto se entrega igual.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

import config as cfg  # noqa: E402
from app.services import diagnosis, llm_client, rules_engine, severity, validation  # noqa: E402
from app.settings import HEATMAPS_DIR, UPLOADS_DIR  # noqa: E402


def _save_rgb(array, path: Path) -> None:
    cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))


def run(image_bytes: bytes, prefix: str, with_llm: bool = True) -> dict:
    """Ejecuta el flujo completo y devuelve un dict con todos los resultados.

    prefix: identificador único del caso para nombrar los archivos.
    """
    # 1-2) Validación y calidad.
    pil = validation.load_image(image_bytes)
    calidad = validation.assess_quality(pil)

    # 3-4) Diagnóstico (CNN) + Grad-CAM (mismo forward/backward).
    pred = diagnosis.diagnose(pil)
    clase = pred["clase_predicha"]
    es_sana = clase == "sana"

    # 5) Severidad.
    sev = severity.estimate_severity(pil, cam=pred.get("cam"), is_healthy=es_sana)

    # Guardar archivos locales (original, imagen enmascarada, overlay, heatmap).
    foto_path = UPLOADS_DIR / f"{prefix}.jpg"
    pil.save(foto_path, format="JPEG", quality=92)
    masked_path = HEATMAPS_DIR / f"masked_{prefix}.png"
    overlay_path = HEATMAPS_DIR / f"overlay_{prefix}.png"
    heatmap_path = HEATMAPS_DIR / f"heatmap_{prefix}.png"
    _save_rgb(pred["entrada_array"], masked_path)
    _save_rgb(pred["overlay_array"], overlay_path)
    _save_rgb(pred["heatmap_array"], heatmap_path)

    # 6) Zona del Grad-CAM en palabras (para el LLM).
    zona = diagnosis.describe_gradcam_zone(pred.get("cam"))

    # 7) Motor de reglas (la recomendación NUNCA viene del LLM).
    rec = rules_engine.recommend(clase, severidad=sev["nivel"])
    rec_texto = rules_engine.recommendation_text(rec)

    # 8) LLM (opcional): explica los resultados.
    if with_llm:
        explic, disponible = llm_client.explain(
            foto_path=str(foto_path), heatmap_path=str(heatmap_path),
            diagnostico_nombre=rec["enfermedad"], confianza=pred["confianza"],
            severidad=sev["nivel"], zona_gradcam=zona, recomendacion_texto=rec_texto,
        )
    else:
        explic = llm_client.fallback_explanation(
            rec["enfermedad"], pred["confianza"], sev["nivel"], zona, rec_texto)
        disponible = False

    return {
        "prefix": prefix,
        "clase": clase,
        "diagnostico": rec["enfermedad"],
        "confianza": pred["confianza"],
        "estado_confianza": pred["estado_confianza"],
        "probabilidades": pred["probabilidades"],
        "severidad": sev["nivel"],
        "severidad_detalle": sev,
        "calidad": calidad,
        "capa_usada": pred["capa_usada"],
        "alerta_sesgo": pred["alerta_sesgo"],
        "zona_gradcam": zona,
        "version_modelo": pred["version_modelo"],
        "metrica_critica_clase": pred.get("metrica_critica_clase", {}),
        "foto_path": str(foto_path),
        "masked_path": str(masked_path),
        "overlay_path": str(overlay_path),
        "heatmap_path": str(heatmap_path),
        "recomendacion": rec,
        "recomendacion_texto": rec_texto,
        "explicacion": explic,
        "explicacion_disponible": disponible,
    }
