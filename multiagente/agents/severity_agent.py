"""Agente de Severidad: cuantifica la gravedad del daño foliar.

Decide de forma autónoma: si la hoja es sana, no hay severidad que estimar y lo
informa; si está enferma, mide el área lesionada y verifica (vía IoU) que el
modelo haya mirado la lesión real y no el fondo.
"""
from __future__ import annotations

from multiagente.core.agent import Agent
from multiagente.core.blackboard import Blackboard
from multiagente.tools.registry import registry


class SeverityAgent(Agent):
    name = "AgenteSeveridad"
    role = "Estima la severidad (ninguna/leve/moderada/severa) del daño foliar."
    tools = ["estimar_severidad"]

    def act(self, bb: Blackboard) -> None:
        perc = bb.get("percepcion", {})
        es_sana = perc.get("clase") == "sana"

        if es_sana:
            self.say(bb, "La hoja es sana: no corresponde estimar severidad.")
        else:
            self.say(bb, "Midiendo el área foliar lesionada...")

        sev = registry.get("estimar_severidad")(
            bb.get("_pil"), cam=bb.get("_cam"), is_healthy=es_sana,
        )
        bb.set("severidad", sev)

        if not es_sana:
            self.say(
                bb,
                f"Severidad: {sev['nivel']} (área lesionada {sev['fraccion_lesion']*100:.0f} %, "
                f"IoU activación-lesión {sev['iou_activacion']}).",
            )
