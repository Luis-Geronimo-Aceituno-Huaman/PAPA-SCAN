"""Coordinador (Supervisor): el cerebro que reparte el trabajo entre agentes.

A diferencia del pipeline rígido anterior, el Coordinador DECIDE el plan y
despacha a cada agente especialista, enrutando según lo que aparece en la
pizarra. Cada agente es autónomo en su tarea; el Coordinador orquesta y sintetiza.

Expone el flujo en dos fases (igual que la API web): el DIAGNÓSTICO (rápido, sin
LLM) y la EXPLICACIÓN (capa LLM opcional). La CLI ejecuta ambas de corrido.

El plan es determinístico a propósito (robusto incluso con un LLM pequeño): el
"agente" que planifica es este supervisor con reglas de enrutamiento claras, un
patrón estándar de sistemas multiagente (supervisor + especialistas).
"""
from __future__ import annotations

from multiagente.agents import (
    AgronomistAgent, ConversationalAgent, ExplainerAgent,
    PerceptionAgent, SeverityAgent, ValidatorAgent,
)
from multiagente.core.blackboard import Blackboard
from multiagente.tools import registry  # noqa: F401  (registra herramientas)


class Coordinator:
    name = "Coordinador"

    def __init__(self, usar_llm: bool = True) -> None:
        self.percepcion = PerceptionAgent()
        self.severidad = SeverityAgent()
        self.agronomo = AgronomistAgent()
        self.explicador = ExplainerAgent(usar_llm=usar_llm)
        self.validador = ValidatorAgent()
        self.conversacional = ConversationalAgent()

    # ------------------------------------------------------------------ #
    # Fase 1: diagnóstico (sin LLM). Percepción → validación → severidad → agrónomo.
    # ------------------------------------------------------------------ #
    def fase_diagnostico(self, bb: Blackboard) -> Blackboard:
        bb.post(self.name, "→ Despacho AgentePercepcion.")
        self.percepcion.act(bb)

        bb.post(self.name, "→ Despacho AgenteValidador (pre-control de confianza/sesgo).")
        self.validador.act(bb)

        bb.post(self.name, "→ Despacho AgenteSeveridad.")
        self.severidad.act(bb)

        bb.post(self.name, "→ Despacho AgenteAgronomo.")
        self.agronomo.act(bb)
        return bb

    # ------------------------------------------------------------------ #
    # Fase 2: explicación (capa LLM opcional). Explicador → validación final.
    # ------------------------------------------------------------------ #
    def fase_explicacion(self, bb: Blackboard) -> Blackboard:
        bb.post(self.name, "→ Despacho AgenteExplicador.")
        self.explicador.act(bb)

        bb.post(self.name, "→ Despacho AgenteValidador (post-control anti-alucinación).")
        self.validador.act(bb)
        return bb

    # ------------------------------------------------------------------ #
    def diagnosticar(self, image_path: str, case_id: str = "caso") -> Blackboard:
        """Ejecuta el flujo multiagente completo (diagnóstico + explicación)."""
        bb = Blackboard(case_id=case_id, image_path=image_path)
        bb.post(self.name, f"Nuevo caso «{case_id}». Plan: percepción → validación → "
                           "severidad → agrónomo → explicación → verificación final.")
        self.fase_diagnostico(bb)
        self.fase_explicacion(bb)
        bb.post(self.name, "Caso resuelto. Sintetizando reporte.")
        return bb

    # ------------------------------------------------------------------ #
    @staticmethod
    def reporte(bb: Blackboard) -> dict:
        """Sintetiza el reporte final desde la pizarra (sin arrays internos)."""
        return {
            "case_id": bb.case_id,
            "diagnostico": bb.get("percepcion", {}).get("clase"),
            "confianza": bb.get("percepcion", {}).get("confianza"),
            "estado_confianza": bb.get("percepcion", {}).get("estado_confianza"),
            "probabilidades": bb.get("percepcion", {}).get("probabilidades"),
            "version_modelo": bb.get("percepcion", {}).get("version_modelo"),
            "zona_gradcam": bb.get("percepcion", {}).get("zona_gradcam"),
            "severidad": bb.get("severidad", {}),
            "recomendacion": bb.get("recomendacion", {}),
            "explicacion": bb.get("explicacion", {}),
            "explicacion_por_llm": bb.get("explicacion_disponible", False),
            "derivar_a_tecnico": bb.is_flagged("derivar_a_tecnico"),
            "validacion_confianza": bb.get("validacion_confianza", {}),
            "validacion_explicacion": bb.get("validacion_explicacion", {}),
            "archivos": {
                "foto": bb.get("percepcion", {}).get("foto_path"),
                "heatmap": bb.get("percepcion", {}).get("heatmap_path"),
                "overlay": bb.get("percepcion", {}).get("overlay_path"),
            },
        }
