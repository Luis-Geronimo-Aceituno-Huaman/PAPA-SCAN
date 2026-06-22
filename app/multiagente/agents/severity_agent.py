"""Agente de Severidad: cuantifica la gravedad del daño foliar.

Decide de forma autónoma: si la hoja es sana, no hay severidad que estimar y lo
informa; si está enferma, mide el área lesionada y verifica (vía IoU) que el
modelo haya mirado la lesión real y no el fondo.
"""
from __future__ import annotations

from app.multiagente.core.agent import Agent
from app.multiagente.core.blackboard import Blackboard
from app.multiagente.tools.registry import registry


class SeverityAgent(Agent):
    name = "AgenteSeveridad"
    role = "Estima la severidad (ninguna/leve/moderada/severa) del daño foliar."
    tools = ["estimar_severidad"]

    def act(self, bb: Blackboard) -> None:
        perc = bb.get("percepcion", {})
        es_sana = perc.get("clase") == "sana"

        if es_sana:
            self.say(bb, "La hoja es sana: verifico que no quede lesión sin atender...")
        else:
            self.say(bb, "Midiendo el área foliar lesionada...")

        sev = registry.get("estimar_severidad")(
            bb.get("_pil"), cam=bb.get("_cam"), is_healthy=es_sana,
        )
        bb.set("severidad", sev)

        if es_sana:
            self._chequeo_falso_negativo(bb, perc, sev)
        else:
            self.say(
                bb,
                f"Severidad: {sev['nivel']} (área lesionada {sev['fraccion_lesion']*100:.0f} %, "
                f"IoU activación-lesión {sev['iou_activacion']}).",
            )

    # ------------------------------------------------------------------ #
    def _chequeo_falso_negativo(self, bb: Blackboard, perc: dict, sev: dict) -> None:
        """Cruza el veredicto "sana" con el área lesionada medida por color.

        Si hay tejido lesionado considerable pese al "sana", se sospecha un FALSO
        NEGATIVO: se degrada la confianza (para que la app encienda la alerta) y se
        deriva a un técnico. Es la red de seguridad por reglas — el modelo no manda
        solo —, y atrapa el caso en que el modelo miró el verde sano e ignoró la
        lesión (Grad-CAM fuera de la zona dañada, IoU bajo).
        """
        from app.services.severity import CONFLICTO_SANA_MIN

        frac = sev.get("fraccion_lesion", 0.0)
        iou = sev.get("iou_activacion", 0.0)

        if frac < CONFLICTO_SANA_MIN:
            self.say(bb, "Sin lesión significativa: el diagnóstico 'sana' es consistente.")
            return

        # Conflicto: "sana" con lesión visible -> baja confianza + derivar a técnico.
        bb.set("conflicto_sana", {"fraccion_lesion": frac, "iou_activacion": iou})
        perc["estado_confianza"] = "baja_confianza"
        bb.set("percepcion", perc)
        bb.flag("derivar_a_tecnico", True)

        motivo = (f"el modelo dice 'sana' pero se detecta {frac*100:.0f} % de tejido "
                  f"lesionado sin atender (posible falso negativo)")
        vc = bb.get("validacion_confianza", {}) or {}
        vc["derivar"] = True
        vc["motivos"] = list(vc.get("motivos", [])) + [motivo]
        bb.set("validacion_confianza", vc)

        self.say(
            bb,
            f"⚠ CONFLICTO: veredicto 'sana' con {frac*100:.0f} % de área lesionada "
            f"(IoU atención-lesión {iou}). Degrado la confianza y marco DERIVAR A TÉCNICO.",
        )
