"""Clase base de todos los agentes.

Un agente tiene: un NOMBRE, un ROL (qué hace), una lista de HERRAMIENTAS que
puede usar, y un método :meth:`act` que opera sobre el :class:`Blackboard`
compartido. Cada agente es autónomo dentro de su rol: decide qué hacer con la
información disponible y publica sus resultados en la pizarra.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from multiagente.core.blackboard import Blackboard


class Agent(ABC):
    """Agente autónomo especializado."""

    #: Nombre legible del agente (se usa en logs y mensajes).
    name: str = "Agente"
    #: Descripción corta de su responsabilidad.
    role: str = ""
    #: Nombres de las herramientas que utiliza (solo informativo/trazabilidad).
    tools: list[str] = []

    def say(self, bb: Blackboard, mensaje: str) -> None:
        """Publica un mensaje del agente en la pizarra."""
        bb.post(self.name, mensaje)

    @abstractmethod
    def act(self, bb: Blackboard) -> None:
        """Ejecuta la tarea del agente leyendo/escribiendo en la pizarra."""
        raise NotImplementedError
