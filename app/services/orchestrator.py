"""Orquestador del flujo de diagnóstico + explicación.

⚙️ EL CEREBRO AHORA ES MULTIAGENTE. Esta función conserva la MISMA firma y la
misma estructura de salida que la versión anterior (pipeline fijo), pero por
dentro delega en el sistema multiagente (`multiagente/`): un Coordinador despacha
agentes especialistas (Percepción, Validador, Severidad, Agrónomo, Explicador)
sobre una pizarra compartida. El frontend, la API y la BD no cambian.

La versión pipeline anterior queda en el historial de git.
"""
from __future__ import annotations

from multiagente import web as _ma_web


def run(image_bytes: bytes, prefix: str, with_llm: bool = True) -> dict:
    """Ejecuta el flujo multiagente y devuelve el dict de resultados.

    prefix: identificador único del caso para nombrar los archivos.
    with_llm: si True, también corre la fase de explicación (LLM); el router de
    /diagnose lo llama con False (la explicación es una capa aparte en /explain).
    """
    return _ma_web.diagnosticar(image_bytes, prefix, with_llm=with_llm)
