"""Agente Explicador: la voz del sistema hacia el agricultor.

Usa el LLM multimodal local (Ollama / qwen3-vl) para interpretar el mapa de
calor y reformular la recomendación en lenguaje simple. Es el único agente que
genera texto libre, y aun así está acotado: el "qué hacer" lo fija el motor de
reglas; el LLM solo describe lo que se ve. Si Ollama no responde, entrega un
texto de respaldo determinístico (el sistema nunca queda mudo).
"""
from __future__ import annotations

from app.multiagente.core.agent import Agent
from app.multiagente.core.blackboard import Blackboard
from app.multiagente.tools.registry import registry


class ExplainerAgent(Agent):
    name = "AgenteExplicador"
    role = "Explica el caso en lenguaje simple usando el LLM local (solo lee el heatmap)."
    tools = ["llm_disponible", "explicar_caso"]

    def __init__(self, usar_llm: bool = True) -> None:
        self.usar_llm = usar_llm

    def act(self, bb: Blackboard) -> None:
        perc = bb.get("percepcion", {})
        rec = bb.get("recomendacion", {})
        sev = bb.get("severidad", {})

        if not self.usar_llm:
            self.say(bb, "LLM desactivado: usaré explicación determinística de respaldo.")
        elif not registry.get("llm_disponible")():
            self.say(bb, "Ollama no responde: usaré explicación determinística de respaldo.")
            self.usar_llm = False

        if self.usar_llm:
            self.say(bb, "Pidiendo al LLM que lea el mapa de calor...")
            explic, disponible = registry.get("explicar_caso")(
                foto_path=perc.get("foto_path"),
                heatmap_path=perc.get("heatmap_path"),
                diagnostico_nombre=rec.get("enfermedad", perc.get("clase", "")),
                confianza=perc.get("confianza", 0.0),
                severidad=sev.get("nivel", "moderada"),
                zona_gradcam=perc.get("zona_gradcam", ""),
                recomendacion_texto=bb.get("recomendacion_texto", ""),
            )
        else:
            # Respaldo 100 % determinístico (misma estructura de seis claves).
            from app.services import llm_client
            explic = llm_client.build_explanation(
                rec.get("enfermedad", perc.get("clase", "")),
                perc.get("confianza", 0.0), sev.get("nivel", "moderada"),
                perc.get("zona_gradcam", ""), bb.get("recomendacion_texto", ""),
            )
            disponible = False

        bb.set("explicacion", explic)
        bb.set("explicacion_disponible", disponible)
        self.say(bb, f"Explicación lista (LLM disponible: {disponible}).")
