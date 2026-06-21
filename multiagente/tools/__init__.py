"""Herramientas que los agentes invocan.

Cada herramienta envuelve una capacidad REAL del proyecto base (la CNN, el
Grad-CAM, el motor de severidad, el motor de reglas + KB, el LLM). Importar
este paquete registra todas las herramientas en el :data:`registry`.
"""
from multiagente.tools import explanation, knowledge, severity, vision  # noqa: F401
from multiagente.tools.registry import registry  # noqa: F401
