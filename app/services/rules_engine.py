"""Motor de reglas: produce la recomendación desde la base de conocimiento.

REGLA DE SEGURIDAD CLAVE (brief §7): la recomendación proviene EXCLUSIVAMENTE de
esta base curada, nunca del LLM. El LLM solo la reformula en lenguaje simple.
"""
from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))

import config as cfg  # noqa: E402
from app.settings import APP_DIR  # noqa: E402

_KB_PATH = APP_DIR / "knowledge_base" / "fito_kb.json"


@lru_cache(maxsize=1)
def load_kb() -> dict:
    """Carga la base de conocimiento (cacheada)."""
    with open(_KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def recommend(diagnostico_clase: str, severidad: str = "moderada",
              estadio: str | None = None, region: str | None = None) -> dict:
    """Devuelve la recomendación estructurada para la clase diagnosticada.

    diagnostico_clase: una de cfg.CLASSES ('sana'|'tizon_tardio'|'tizon_temprano').
    """
    kb = load_kb()
    enf = kb["enfermedades"].get(diagnostico_clase)
    if enf is None:
        raise ValueError(f"Clase desconocida en la KB: {diagnostico_clase}")

    nota_sev = kb["modificadores_severidad"].get(severidad, "")
    trazabilidad = (
        f"Recomendación anclada al diagnóstico ({enf['nombre']}) y a la severidad "
        f"{severidad}. Fuente: {kb['version']}. Generada sin conexión a Internet."
    )

    return {
        "enfermedad": enf["nombre"],
        "patogeno": enf["patogeno"],
        "severidad": severidad,
        "urgencia": enf.get("urgencia", ""),
        "estadio_fenologico": estadio or enf.get("estadio_fenologico_tipico", ""),
        "region": region or kb.get("regiones_por_defecto", ""),
        "nota_severidad": nota_sev,
        "controles": enf["controles"],
        "trazabilidad": trazabilidad,
        "disclaimer": kb["disclaimer"],
    }


def recommendation_text(rec: dict) -> str:
    """Aplana la recomendación a un texto plano para pasársela al LLM.

    El LLM debe transmitir esto TAL CUAL (no inventar dosis ni productos).
    """
    partes = [f"{c['titulo']}: {c['contenido']}" for c in rec["controles"]]
    if rec.get("nota_severidad"):
        partes.append(rec["nota_severidad"])
    return " ".join(partes)
