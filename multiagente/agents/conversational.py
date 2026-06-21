"""Agente Conversacional: atiende preguntas de seguimiento sobre el caso.

Usa el LLM local restringido al tema del cultivo (la guarda off-topic vive en
`llm_client.chat`). Construye el contexto del caso desde la pizarra para que las
respuestas estén ancladas al diagnóstico ya hecho.
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
        """Responde una pregunta puntual. Devuelve (respuesta, disponible)."""
        historial = historial or []
        respuesta, disponible = registry.get("responder_chat")(
            historial, self.contexto(bb), mensaje,
        )
        self.say(bb, f"Pregunta: «{mensaje}» -> respondida (LLM disponible: {disponible}).")
        return respuesta, disponible

    def act(self, bb: Blackboard) -> None:
        # En el flujo de diagnóstico no actúa solo; se invoca bajo demanda
        # (modo --ask de la CLI) vía preguntar().
        return
