"""Agente Validador (crítico): la red de seguridad del sistema.

Cumple dos controles, en dos momentos:

1. PRE-control (antes de explicar): revisa la confianza y la alerta de sesgo del
   Grad-CAM. Si la confianza es baja o el modelo miró el fondo, marca el caso
   para DERIVAR A UN TÉCNICO (human-in-the-loop).

2. POST-control (después de explicar): verifica el invariante de seguridad — que
   el "qué hacer" venga EXACTAMENTE del motor de reglas y que el LLM no haya
   colado dosis/productos inventados en el campo que sí redacta.

Es idempotente: el Coordinador puede invocarlo dos veces; cada control corre una
sola vez gracias a los flags en la pizarra.
"""
from __future__ import annotations

import re

from multiagente.core.agent import Agent
from multiagente.core.blackboard import Blackboard

# Patrón de "dosis": número seguido de unidad agronómica (g/L, ml, kg, %, cc...).
_DOSIS_RE = re.compile(r"\d+([.,]\d+)?\s*(g/?l|g|ml|cc|kg|l|%)\b", re.IGNORECASE)


class ValidatorAgent(Agent):
    name = "AgenteValidador"
    role = "Controla confianza/sesgo (deriva a humano) y la seguridad de la explicación."
    tools = []

    def act(self, bb: Blackboard) -> None:
        if bb.has("percepcion") and not bb.has("validacion_confianza"):
            self._precheck(bb)
        if bb.has("explicacion") and not bb.has("validacion_explicacion"):
            self._postcheck(bb)

    # ------------------------------------------------------------------ #
    def _precheck(self, bb: Blackboard) -> None:
        from app.settings import LOW_CONFIDENCE_THRESHOLD

        perc = bb.get("percepcion", {})
        conf = perc.get("confianza", 0.0)
        baja = (perc.get("estado_confianza") == "baja_confianza"
                or conf < LOW_CONFIDENCE_THRESHOLD)
        sesgo = bool(perc.get("alerta_sesgo"))

        motivos = []
        if baja:
            motivos.append(f"confianza baja ({conf*100:.0f} %)")
        if sesgo:
            motivos.append("el Grad-CAM sugiere atención en el fondo (posible sesgo)")

        derivar = bool(motivos)
        bb.flag("derivar_a_tecnico", derivar)
        bb.set("validacion_confianza", {"derivar": derivar, "motivos": motivos})

        if derivar:
            self.say(bb, "⚠ Caso marcado para DERIVAR A TÉCNICO: " + "; ".join(motivos) + ".")
        else:
            self.say(bb, "Confianza y atención del modelo OK: no requiere derivación.")

    # ------------------------------------------------------------------ #
    def _postcheck(self, bb: Blackboard) -> None:
        explic = bb.get("explicacion", {})
        rec_texto = bb.get("recomendacion_texto", "")
        problemas = []

        # Invariante 1: el "qué hacer" debe ser EXACTAMENTE el texto de reglas.
        if explic.get("que_hacer", "") != rec_texto:
            problemas.append("el campo 'qué hacer' no coincide con el motor de reglas")

        # Invariante 2: el único campo redactado por el LLM no debe traer dosis.
        observado = explic.get("que_observo_el_modelo", "")
        if _DOSIS_RE.search(observado):
            problemas.append("el LLM incluyó algo que parece una dosis en su descripción")

        ok = not problemas
        bb.set("validacion_explicacion", {"segura": ok, "problemas": problemas})

        if ok:
            self.say(bb, "Explicación verificada: respeta el invariante de seguridad.")
        else:
            self.say(bb, "⚠ Explicación corregida al respaldo seguro: " + "; ".join(problemas) + ".")
            # Acción correctiva: forzar el respaldo determinístico seguro.
            from app.services import llm_client
            perc = bb.get("percepcion", {})
            rec = bb.get("recomendacion", {})
            sev = bb.get("severidad", {})
            bb.set("explicacion", llm_client.build_explanation(
                rec.get("enfermedad", perc.get("clase", "")),
                perc.get("confianza", 0.0), sev.get("nivel", "moderada"),
                perc.get("zona_gradcam", ""), rec_texto,
            ))
            bb.set("explicacion_disponible", False)
