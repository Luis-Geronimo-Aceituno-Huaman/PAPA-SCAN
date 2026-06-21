"""Coordinador (Supervisor): el cerebro que reparte el trabajo entre agentes.

A diferencia del pipeline rígido del proyecto base, aquí el Coordinador DECIDE
el plan y despacha a cada agente especialista, enrutando según lo que aparece en
la pizarra (p. ej. si la confianza es baja, marca derivación; si la hoja es
sana, la severidad se resuelve sola). Cada agente es autónomo en su tarea; el
Coordinador solo orquesta y, al final, sintetiza el reporte.

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
    def diagnosticar(self, image_path: str, case_id: str = "caso") -> Blackboard:
        """Ejecuta el flujo multiagente completo sobre una imagen."""
        bb = Blackboard(case_id=case_id, image_path=image_path)
        bb.post(self.name, f"Nuevo caso «{case_id}». Plan: percepción → validación → "
                           "severidad → agrónomo → explicación → verificación final.")

        # 1) Percepción (CNN + Grad-CAM).
        bb.post(self.name, "→ Despacho AgentePercepcion.")
        self.percepcion.act(bb)

        # 2) Validación temprana (confianza / sesgo → ¿derivar a humano?).
        bb.post(self.name, "→ Despacho AgenteValidador (pre-control).")
        self.validador.act(bb)

        # 3) Severidad (el propio agente resuelve el caso 'sana').
        bb.post(self.name, "→ Despacho AgenteSeveridad.")
        self.severidad.act(bb)

        # 4) Agrónomo (recomendación anclada a la KB).
        bb.post(self.name, "→ Despacho AgenteAgronomo.")
        self.agronomo.act(bb)

        # 5) Explicación (LLM multimodal; con respaldo determinístico).
        bb.post(self.name, "→ Despacho AgenteExplicador.")
        self.explicador.act(bb)

        # 6) Verificación final de seguridad de la explicación.
        bb.post(self.name, "→ Despacho AgenteValidador (post-control).")
        self.validador.act(bb)

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
