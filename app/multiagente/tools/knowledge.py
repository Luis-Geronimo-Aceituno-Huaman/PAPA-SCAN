"""Herramientas de CONOCIMIENTO: reutilizan el motor de reglas + la KB curada.

REGLA DE SEGURIDAD (heredada del proyecto base): la recomendación proviene
EXCLUSIVAMENTE de la base de conocimiento curada (`fito_kb.json`), nunca del
LLM. El agente Agrónomo solo puede CONSULTAR estas herramientas; no inventa.
"""
from __future__ import annotations

from app.multiagente.core import bootstrap  # noqa: F401
from app.multiagente.tools.registry import registry


@registry.register(
    "recomendar_tratamiento",
    "Devuelve la recomendación estructurada (controles, urgencia, disclaimer) "
    "anclada al diagnóstico y la severidad. Fuente: KB curada, sin inventar dosis.",
)
def recomendar_tratamiento(clase: str, severidad: str = "moderada",
                           estadio: str | None = None, region: str | None = None) -> dict:
    from app.services import rules_engine
    return rules_engine.recommend(clase, severidad=severidad, estadio=estadio, region=region)


@registry.register(
    "aplanar_recomendacion",
    "Convierte la recomendación estructurada en texto plano (lo que el LLM debe "
    "transmitir TAL CUAL, sin alterar productos ni dosis).",
)
def aplanar_recomendacion(rec: dict) -> str:
    from app.services import rules_engine
    return rules_engine.recommendation_text(rec)
