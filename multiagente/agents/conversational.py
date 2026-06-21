"""Agente Conversacional: atiende preguntas de seguimiento.

Dos modos:
- por CASO: el contexto sale de la pizarra (diagnóstico ya hecho); usa el system
  prompt de caso.
- LIBRE: sin diagnóstico (pestaña Asistente), con un contexto general y el system
  prompt general.

La guarda off-topic vive en `llm_client.chat`. Construye el contexto para que las
respuestas estén ancladas al caso.
"""
from __future__ import annotations

from multiagente.core.agent import Agent
from multiagente.core.blackboard import Blackboard
from multiagente.tools.registry import registry


class ConversationalAgent(Agent):
    name = "AgenteConversacional"
    role = "Responde preguntas de seguimiento del agricultor, ancladas al caso."
    tools = ["responder_chat"]

    def contexto(self, bb: Blackboard) -> str:
        perc = bb.get("percepcion", {})
        rec = bb.get("recomendacion", {})
        sev = bb.get("severidad", {})
        return (
            f"Diagnóstico: {rec.get('enfermedad', perc.get('clase', ''))}. "
            f"Confianza: {perc.get('confianza', 0)*100:.0f} %. "
            f"Severidad: {sev.get('nivel', 'n/d')}. "
            f"Recomendación: {bb.get('recomendacion_texto', '')}"
        )

    def preguntar(self, bb: Blackboard, mensaje: str, historial: list[dict] | None = None):
        """Responde una pregunta sobre el caso. Devuelve (respuesta, disponible)."""
        respuesta, disponible = registry.get("responder_chat")(
            historial or [], self.contexto(bb), mensaje,
        )
        self.say(bb, f"Pregunta: «{mensaje}» -> respondida (LLM disponible: {disponible}).")
        return respuesta, disponible

    def preguntar_libre(self, mensaje: str, historial: list[dict], contexto: str,
                        system_prompt: str):
        """Chat libre (sin caso). Devuelve (respuesta, disponible)."""
        return registry.get("responder_chat")(
            historial, contexto, mensaje, system_prompt=system_prompt,
        )

    def act(self, bb: Blackboard) -> None:
        # En el flujo de diagnóstico no actúa solo; se invoca bajo demanda.
        return
