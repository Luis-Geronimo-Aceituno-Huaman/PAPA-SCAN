"""Agentes especializados del sistema PapaScan Multiagente."""
from multiagente.agents.agronomist import AgronomistAgent
from multiagente.agents.conversational import ConversationalAgent
from multiagente.agents.explainer import ExplainerAgent
from multiagente.agents.perception import PerceptionAgent
from multiagente.agents.severity_agent import SeverityAgent
from multiagente.agents.validator import ValidatorAgent

__all__ = [
    "AgronomistAgent", "ConversationalAgent", "ExplainerAgent",
    "PerceptionAgent", "SeverityAgent", "ValidatorAgent",
]
