"""Blackboard: la memoria compartida del sistema multiagente.

Todos los agentes leen y escriben sobre la misma pizarra. Es el mecanismo de
comunicación (en lugar de llamadas directas entre agentes): el Coordinador
despacha agentes, cada agente publica sus hallazgos aquí, y los siguientes
agentes los consumen. También guarda un registro de mensajes (trazabilidad).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Message:
    """Un mensaje publicado por un agente en la pizarra (para trazabilidad)."""
    agente: str
    contenido: str


@dataclass
class Blackboard:
    """Estado compartido de un caso de diagnóstico."""
    case_id: str
    image_path: str
    data: dict[str, Any] = field(default_factory=dict)
    flags: dict[str, bool] = field(default_factory=dict)
    messages: list[Message] = field(default_factory=list)

    # -- escritura ---------------------------------------------------------- #
    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def flag(self, name: str, value: bool = True) -> None:
        self.flags[name] = value

    def post(self, agente: str, contenido: str) -> None:
        """Registra un mensaje de un agente (se ve en el log de ejecución)."""
        self.messages.append(Message(agente=agente, contenido=contenido))

    # -- lectura ------------------------------------------------------------ #
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self.data

    def is_flagged(self, name: str) -> bool:
        return self.flags.get(name, False)

    def transcript(self) -> str:
        """Devuelve el diálogo entre agentes como texto (para depurar/mostrar)."""
        return "\n".join(f"[{m.agente}] {m.contenido}" for m in self.messages)
