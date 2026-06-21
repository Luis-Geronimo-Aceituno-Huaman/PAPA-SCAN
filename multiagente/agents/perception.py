"""Agente de Percepción: el "ojo" del sistema.

Responsable de mirar la imagen y producir el diagnóstico visual con la CNN y el
Grad-CAM. Es el primer agente que actúa: sin su salida, el resto no puede operar.
Persiste los medios (foto, enmascarada, overlay, heatmap) mediante el MediaSink
que el orquestador deje en la pizarra (web → carpetas de la app; CLI → outputs/).
"""
from __future__ import annotations

from multiagente.core.agent import Agent
from multiagente.core.blackboard import Blackboard
from multiagente.core.bootstrap import OUTPUTS_DIR
from multiagente.core.media import DirMediaSink
from multiagente.tools.registry import registry

# Sink por defecto (CLI): todo en multiagente/outputs/ con nombres legibles.
_DEFAULT_SINK = DirMediaSink(
    OUTPUTS_DIR, foto_tpl="{p}_foto.jpg", masked_tpl="{p}_masked.png",
    overlay_tpl="{p}_overlay.png", heatmap_tpl="{p}_heatmap.png",
)


class PerceptionAgent(Agent):
    name = "AgentePercepcion"
    role = "Diagnostica la hoja con la CNN y genera el mapa de calor (Grad-CAM)."
    tools = ["diagnosticar_imagen"]

    def act(self, bb: Blackboard) -> None:
        self.say(bb, "Analizando la imagen con la CNN AgriVision...")

        # La fuente puede ser un PIL ya cargado (web) o la ruta (CLI).
        fuente = bb.get("_pil_in") or bb.image_path
        out = registry.get("diagnosticar_imagen")(fuente)

        # Arrays internos (no van al reporte JSON).
        pil = out.pop("_pil")
        cam = out.pop("_cam")
        entrada = out.pop("_entrada")
        overlay = out.pop("_overlay")
        heat = out.pop("_heat")
        bb.set("_cam", cam)
        bb.set("_pil", pil)

        # Persistir medios con el sink inyectado (o el de la CLI por defecto).
        sink = bb.get("media_sink") or _DEFAULT_SINK
        rutas = sink.save(bb.case_id, pil, entrada, overlay, heat)
        out.update(rutas)

        bb.set("percepcion", out)
        self.say(
            bb,
            f"Diagnóstico: {out['clase']} (confianza {out['confianza']*100:.0f} %, "
            f"estado {out['estado_confianza']}). Atención del modelo: {out['zona_gradcam']}.",
        )
