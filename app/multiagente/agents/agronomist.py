"""Agente Agrónomo: decide la recomendación de manejo.

CLAVE DE SEGURIDAD: este agente NO inventa tratamientos. Su única fuente es el
motor de reglas + la base de conocimiento curada (herramientas). Consulta la KB
anclando la recomendación al diagnóstico y a la severidad, y la deja lista para
que el Explicador la transmita en lenguaje simple (sin alterar dosis ni productos).
"""
from __future__ import annotations

from app.multiagente.core.agent import Agent
from app.multiagente.core.blackboard import Blackboard
from app.multiagente.tools.registry import registry


class AgronomistAgent(Agent):
    name = "AgenteAgronomo"
    role = "Recomienda el manejo consultando SOLO la base de conocimiento curada."
    tools = ["recomendar_tratamiento", "aplanar_recomendacion"]

    def act(self, bb: Blackboard) -> None:
        clase = bb.get("percepcion", {}).get("clase", "")
        nivel = bb.get("severidad", {}).get("nivel", "moderada")
        # Para una hoja sana la KB usa la severidad 'ninguna' si existe; si no,
        # el motor de reglas igual responde con cuidados preventivos.
        severidad_kb = nivel if nivel not in ("ninguna",) else "leve"

        self.say(bb, f"Consultando la KB curada para «{clase}» (severidad {nivel})...")
        rec = registry.get("recomendar_tratamiento")(clase, severidad=severidad_kb)
        texto = registry.get("aplanar_recomendacion")(rec)

        bb.set("recomendacion", rec)
        bb.set("recomendacion_texto", texto)
        self.say(
            bb,
            f"Recomendación anclada a «{rec['enfermedad']}» (urgencia: "
            f"{rec.get('urgencia', 'n/d')}). Fuente: motor de reglas, sin inventar dosis.",
        )
