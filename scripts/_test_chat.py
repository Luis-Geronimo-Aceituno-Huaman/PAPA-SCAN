import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import ollama
from app.settings import LLM_MODEL, OLLAMA_HOST, LLM_TEMPERATURE
from app.services.llm_client import CHAT_SYSTEM_PROMPT, _strip_think

ctx = "diagnostico: Tizón tardío; confianza: 1.0; severidad: leve"
for pregunta in ["Y como lo evito?", "donde nacio Tupac amaru?"]:
    intro = f"{CHAT_SYSTEM_PROMPT}\n\nDatos del caso:\n{ctx}"
    messages = [{"role": "user", "content": intro},
                {"role": "assistant", "content": "Entendido. ¿Cuál es tu pregunta?"},
                {"role": "user", "content": f"{pregunta}\n\n(Responde en español, breve y directo.)"}]
    resp = ollama.Client(host=OLLAMA_HOST, timeout=120).chat(
        model=LLM_MODEL, messages=messages, options={"temperature": LLM_TEMPERATURE})
    raw = resp["message"]["content"]
    print(f"\n=== {pregunta!r} ===")
    print(f"content_len_crudo={len(raw)} | tras_strip_think={len(_strip_think(raw))!r}")
    print("respuesta:", _strip_think(raw)[:200] or "(VACÍO)")
