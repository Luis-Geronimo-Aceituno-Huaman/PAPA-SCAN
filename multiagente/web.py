"""Puente entre la app web (FastAPI) y el sistema multiagente.

La app web NO cambia su API ni su BD: este módulo ejecuta el flujo multiagente
y lo mapea a las mismas estructuras que antes producía `app.services.orchestrator`.
Así, "el cerebro" de PapaScan pasa a ser multiagente sin tocar el frontend.

Tres entradas, espejo de los routers:
- :func:`diagnosticar`  → POST /api/diagnose  (percepción + validación + severidad + agrónomo)
- :func:`explicar`      → POST /api/explain    (explicador + validación final)
- :func:`responder`     → POST /api/chat       (agente conversacional)
"""
from __future__ import annotations

from multiagente.core import bootstrap  # noqa: F401  (configura sys.path)
from multiagente.agents import ConversationalAgent, ExplainerAgent, ValidatorAgent
from multiagente.core.blackboard import Blackboard
from multiagente.core.media import DirMediaSink
from multiagente.coordinator import Coordinator


def _web_sink() -> DirMediaSink:
    """Sink que escribe en las carpetas de la app (igual que el orquestador viejo)."""
    from app.settings import HEATMAPS_DIR, UPLOADS_DIR
    return DirMediaSink(
        foto_dir=UPLOADS_DIR, media_dir=HEATMAPS_DIR,
        foto_tpl="{p}.jpg", masked_tpl="masked_{p}.png",
        overlay_tpl="overlay_{p}.png", heatmap_tpl="heatmap_{p}.png",
    )


# --------------------------------------------------------------------------- #
# 1) Diagnóstico (POST /api/diagnose)
# --------------------------------------------------------------------------- #
def diagnosticar(image_bytes: bytes, prefix: str, with_llm: bool = False) -> dict:
    """Ejecuta el flujo multiagente y devuelve el dict legado que espera el router.

    Mantiene EXACTAMENTE las mismas claves que el antiguo `orchestrator.run`.
    """
    from app.services import llm_client, validation

    pil = validation.load_image(image_bytes)          # valida y normaliza
    calidad = validation.assess_quality(pil)

    bb = Blackboard(case_id=prefix, image_path="")
    bb.set("_pil_in", pil)
    bb.set("media_sink", _web_sink())

    coord = Coordinator(usar_llm=with_llm)
    coord.fase_diagnostico(bb)

    if with_llm:
        coord.fase_explicacion(bb)
    else:
        # /diagnose es rápido y sin LLM: explicación determinística de respaldo
        # (el router la ignora, pero mantenemos la clave por compatibilidad).
        perc = bb.get("percepcion", {})
        rec = bb.get("recomendacion", {})
        sev = bb.get("severidad", {})
        bb.set("explicacion", llm_client.build_explanation(
            rec.get("enfermedad", perc.get("clase", "")), perc.get("confianza", 0.0),
            sev.get("nivel", "moderada"), perc.get("zona_gradcam", ""),
            bb.get("recomendacion_texto", "")))
        bb.set("explicacion_disponible", False)

    return _to_legacy(bb, calidad)


def _to_legacy(bb: Blackboard, calidad: dict) -> dict:
    """Mapea la pizarra al dict que consumían los routers (forma idéntica)."""
    perc = bb.get("percepcion", {})
    sev = bb.get("severidad", {})
    rec = bb.get("recomendacion", {})
    return {
        "prefix": bb.case_id,
        "clase": perc.get("clase"),
        "diagnostico": rec.get("enfermedad", perc.get("clase")),
        "confianza": perc.get("confianza"),
        "estado_confianza": perc.get("estado_confianza"),
        "probabilidades": perc.get("probabilidades", {}),
        "severidad": sev.get("nivel", "ninguna"),
        "severidad_detalle": sev,
        "calidad": calidad,
        "capa_usada": perc.get("capa_usada", ""),
        "alerta_sesgo": bool(perc.get("alerta_sesgo", False)),
        "zona_gradcam": perc.get("zona_gradcam", ""),
        "version_modelo": perc.get("version_modelo", ""),
        "metrica_critica_clase": perc.get("metrica_critica_clase", {}),
        "foto_path": perc.get("foto_path"),
        "masked_path": perc.get("masked_path", ""),
        "overlay_path": perc.get("overlay_path"),
        "heatmap_path": perc.get("heatmap_path"),
        "recomendacion": rec,
        "recomendacion_texto": bb.get("recomendacion_texto", ""),
        "explicacion": bb.get("explicacion", {}),
        "explicacion_disponible": bb.get("explicacion_disponible", False),
        # trazabilidad multiagente (nuevo; los routers pueden ignorarlo):
        "derivar_a_tecnico": bb.is_flagged("derivar_a_tecnico"),
        "_transcript": bb.transcript(),
    }


# --------------------------------------------------------------------------- #
# 2) Explicación (POST /api/explain/{case_id}) — sobre un caso ya guardado
# --------------------------------------------------------------------------- #
def explicar(*, diagnostico_nombre: str, confianza: float, severidad: str,
             zona_gradcam: str, foto_path: str, heatmap_path: str,
             recomendacion: dict, recomendacion_texto: str,
             estado_confianza: str = "alta", alerta_sesgo: bool = False):
    """Corre el AgenteExplicador + AgenteValidador sobre un caso reconstruido.

    Devuelve (explicacion_dict, disponible) — la misma forma que antes.
    """
    bb = Blackboard(case_id="explain", image_path="")
    bb.set("percepcion", {
        "clase": diagnostico_nombre, "confianza": confianza,
        "estado_confianza": estado_confianza, "alerta_sesgo": alerta_sesgo,
        "zona_gradcam": zona_gradcam, "foto_path": foto_path, "heatmap_path": heatmap_path,
    })
    bb.set("severidad", {"nivel": severidad})
    bb.set("recomendacion", recomendacion)
    bb.set("recomendacion_texto", recomendacion_texto)

    ExplainerAgent(usar_llm=True).act(bb)
    ValidatorAgent().act(bb)   # post-control anti-alucinación
    return bb.get("explicacion", {}), bb.get("explicacion_disponible", False)


# --------------------------------------------------------------------------- #
# 3) Chat de seguimiento (POST /api/chat)
# --------------------------------------------------------------------------- #
def responder(historial: list[dict], contexto: str, mensaje: str,
              system_prompt: str | None = None):
    """Delegado en el AgenteConversacional. Devuelve (respuesta, disponible)."""
    from app.services import llm_client
    sp = system_prompt or llm_client.CHAT_SYSTEM_PROMPT
    return ConversationalAgent().preguntar_libre(mensaje, historial, contexto, sp)
