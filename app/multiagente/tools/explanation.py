"""Herramientas de LENGUAJE (LLM local): reutilizan `app.services.llm_client`.

El LLM (Ollama / qwen3-vl) SOLO interpreta el mapa de calor y reformula la
recomendación en lenguaje simple. Si Ollama no está disponible, el cliente
devuelve un texto determinístico de respaldo (el sistema nunca depende del LLM).
"""
from __future__ import annotations

from app.multiagente.core import bootstrap  # noqa: F401
from app.multiagente.tools.registry import registry


@registry.register(
    "llm_disponible",
    "Indica si el LLM local (Ollama) responde y el modelo está cargado.",
)
def llm_disponible() -> bool:
    from app.services import llm_client
    return llm_client.is_available()


@registry.register(
    "explicar_caso",
    "Genera la explicación de seis claves (resumen, qué observó el modelo, nivel "
    "de confianza, qué hacer, alerta, siguiente paso). El LLM aporta la lectura "
    "multimodal del heatmap; lo demás se arma desde reglas. Devuelve (dict, disponible).",
)
def explicar_caso(*, foto_path: str, heatmap_path: str, diagnostico_nombre: str,
                  confianza: float, severidad: str, zona_gradcam: str,
                  recomendacion_texto: str):
    from app.services import llm_client
    return llm_client.explain(
        foto_path=foto_path, heatmap_path=heatmap_path,
        diagnostico_nombre=diagnostico_nombre, confianza=confianza,
        severidad=severidad, zona_gradcam=zona_gradcam,
        recomendacion_texto=recomendacion_texto,
    )


@registry.register(
    "responder_chat",
    "Responde una pregunta de seguimiento del agricultor, en español, restringida "
    "a la salud del cultivo. Acepta un system_prompt (caso o chat libre). "
    "Devuelve (respuesta, disponible).",
)
def responder_chat(historial: list[dict], contexto_caso: str, mensaje_usuario: str,
                   system_prompt: str | None = None):
    from app.services import llm_client
    sp = system_prompt or llm_client.CHAT_SYSTEM_PROMPT
    return llm_client.chat(historial, contexto_caso, mensaje_usuario, system_prompt=sp)
