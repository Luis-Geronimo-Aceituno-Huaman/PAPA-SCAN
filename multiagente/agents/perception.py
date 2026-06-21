"""Agente de Percepción: el "ojo" del sistema.

Responsable de mirar la imagen y producir el diagnóstico visual con la CNN y el
Grad-CAM. Es el primer agente que actúa: sin su salida, el resto no puede operar.
"""
from __future__ import annotations

from multiagente.core.agent import Agent
from multiagente.core.blackboard import Blackboard
from multiagente.tools.registry import registry


class PerceptionAgent(Agent):
    name = "AgentePercepcion"
    role = "Diagnostica la hoja con la CNN y genera el mapa de calor (Grad-CAM)."
    tools = ["diagnosticar_imagen"]

    def act(self, bb: Blackboard) -> None:
        self.say(bb, "Analizando la imagen con la CNN AgriVision...")
        out = registry.get("diagnosticar_imagen")(bb.image_path, bb.case_id)

        # Guardamos arrays internos aparte (no van al reporte final).
        bb.set("_cam", out.pop("_cam"))
        bb.set("_pil", out.pop("_pil"))
        bb.set("percepcion", out)

        self.say(
            bb,
            f"Diagnóstico: {out['clase']} (confianza {out['confianza']*100:.0f} %, "
            f"estado {out['estado_confianza']}). Atención del modelo: {out['zona_gradcam']}.",
        )
