"""Registro de herramientas.

Patrón clásico de sistemas con agentes: las capacidades se exponen como
"herramientas" con nombre y descripción, y los agentes las invocan por nombre.
Esto desacopla a los agentes de la implementación concreta (la CNN, las reglas,
el LLM...) y hace explícito qué puede hacer cada quién.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    description: str
    func: Callable

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class ToolRegistry:
    """Catálogo de herramientas disponibles para los agentes."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, name: str, description: str) -> Callable:
        """Decorador para registrar una función como herramienta."""
        def deco(func: Callable) -> Callable:
            self._tools[name] = Tool(name=name, description=description, func=func)
            return func
        return deco

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"Herramienta no registrada: {name}")
        return self._tools[name]

    def names(self) -> list[str]:
        return sorted(self._tools)

    def catalog(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())


#: Registro global único.
registry = ToolRegistry()
