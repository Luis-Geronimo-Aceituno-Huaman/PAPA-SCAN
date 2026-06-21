"""Integración con el LLM multimodal local (Ollama / qwen3-vl).

El LLM SOLO EXPLICA: no diagnostica ni recomienda por su cuenta (brief §7). Recibe
las dos imágenes (foto + mapa de calor) y los datos estructurados, y devuelve la
explicación en JSON con seis claves fijas.

Si el LLM no está disponible o falla, `explain()` devuelve una explicación de
respaldo determinística (construida desde el diagnóstico y la recomendación de
reglas) y marca `disponible=False`. El diagnóstico NUNCA depende del LLM.
"""
from __future__ import annotations

import json
import logging
import re

import ollama

_log = logging.getLogger("papascan.llm")

from app.settings import (
    LLM_MODEL, LLM_TEMPERATURE, LLM_TIMEOUT_S, LOW_CONFIDENCE_THRESHOLD, OLLAMA_HOST,
)

_SIX_KEYS = ["resumen", "que_observo_el_modelo", "nivel_confianza",
             "que_hacer", "alerta", "siguiente_paso"]

# --------------------------------------------------------------------------- #
# Prompt (brief §5): PERSONA + CONTEXTO + TAREA + EJEMPLO + FORMATO
# --------------------------------------------------------------------------- #
SYSTEM_PROMPT = """\
[PERSONA]
Eres «AgroAsistente», un asistente experto en fitopatología (enfermedades de
plantas) que acompaña a agricultores y técnicos agrícolas en zonas rurales del
Perú. Hablas español claro, corto y respetuoso. NO eres quien diagnostica: un
modelo de IA ya hizo el diagnóstico y un motor de reglas ya definió la
recomendación. Tu única función es EXPLICAR esos resultados de forma sencilla.
Nunca los cambias ni los inventas.

[CONTEXTO]
Recibes, para cada caso:
- Una foto de la hoja del cultivo.
- Un mapa de calor (Grad-CAM): las zonas en rojo/amarillo son las que el modelo
  miró para decidir; las azules las ignoró.
- El diagnóstico del modelo: enfermedad y confianza (0-100 %).
- La severidad estimada.
- La recomendación del motor de reglas (de una base de conocimiento curada).
El usuario suele tener poca formación técnica y puede hablar quechua o aymara
como primera lengua: evita tecnicismos y frases largas.

[TAREA]
1. Explica en palabras simples qué enfermedad se detectó y con qué seguridad.
2. Interpreta el mapa de calor: di en qué parte de la hoja (bordes, centro,
   puntas, manchas) se concentró la atención del modelo.
3. Transmite la recomendación del motor de reglas TAL CUAL. No agregues
   productos, dosis ni tratamientos que no estén en ella.
4. Si la confianza es menor a 60 %, avísalo con claridad y recomienda consultar
   a un técnico agrónomo o a SENASA antes de actuar.
5. Si te preguntan algo fuera de la salud de cultivos, redirige con amabilidad.
PROHIBIDO inventar diagnósticos, dosis de agroquímicos o datos. Ante la duda,
recomienda consultar a un experto humano.

[EJEMPLO]
Entrada:
- diagnostico: "Tizón tardío (Phytophthora infestans)"
- confianza: 0.91
- severidad: "moderada"
- zona_gradcam: "manchas oscuras de los bordes en la mitad inferior de la hoja"
- recomendacion_reglas: "Retirar y quemar las hojas afectadas. Aplicar fungicida
  a base de cobre según la dosis de la guía. Mejorar la ventilación entre plantas."
Salida esperada (un único JSON):
{"resumen": "La hoja muestra signos de tizón tardío, una enfermedad común de la papa.",
 "que_observo_el_modelo": "El sistema se fijó sobre todo en las manchas oscuras del borde, en la parte de abajo de la hoja.",
 "nivel_confianza": "alto",
 "que_hacer": "Retira y quema las hojas enfermas, aplica un fungicida de cobre con la dosis de la guía y deja más espacio entre las plantas para que circule el aire.",
 "alerta": "",
 "siguiente_paso": "Revisa las plantas vecinas en los próximos días; si aparecen más manchas, avisa a tu técnico agrónomo."}

[FORMATO]
Responde SIEMPRE con un único objeto JSON válido, sin texto adicional, con estas
claves exactas: resumen, que_observo_el_modelo, nivel_confianza ("alto"/"medio"/"bajo"),
que_hacer, alerta ("" si no aplica), siguiente_paso.
"""

CHAT_SYSTEM_PROMPT = """\
Eres «AgroAsistente», asistente de fitopatología para agricultores del Perú.
Respondes preguntas de seguimiento sobre un caso ya diagnosticado. Hablas español
sencillo, en pocas frases, sin JSON. NO cambias el diagnóstico ni inventas dosis
o productos: te limitas a la información del caso y de la recomendación dada. Si
no tienes el dato, dilo y sugiere consultar a un técnico agrónomo o a SENASA.
"""

# Prompt para el chat LIBRE (pestaña Asistente, sin diagnóstico): asistente general
# sobre el cultivo de papa. NO está atado a un caso, así que SÍ puede dar información
# educativa general (síntomas, prevención, manejo), pero sin inventar dosis exactas.
GENERAL_CHAT_SYSTEM_PROMPT = """\
Eres «AgroAsistente», asistente de fitopatología para agricultores del Perú.
Respondes dudas GENERALES sobre el cultivo de papa y sus enfermedades: tizón tardío
(Phytophthora infestans), tizón temprano (Alternaria solani) y hojas sanas —síntomas,
prevención y manejo general. Hablas español sencillo, en pocas frases, sin JSON.
Puedes explicar de forma general, pero NO inventes dosis exactas de agroquímicos:
ante decisiones de tratamiento sugiere confirmar con un técnico agrónomo o SENASA.
"""


def _client() -> ollama.Client:
    return ollama.Client(host=OLLAMA_HOST, timeout=LLM_TIMEOUT_S)


def is_available() -> bool:
    """True si Ollama responde y el modelo está disponible."""
    try:
        models = _client().list().get("models", [])
        names = {m.get("model", m.get("name", "")) for m in models}
        return any(LLM_MODEL.split(":")[0] in n for n in names)
    except Exception:
        return False


def _coerce_six_keys(data: dict) -> dict:
    """Garantiza las seis claves exactas (rellena las faltantes con '')."""
    return {k: str(data.get(k, "")).strip() for k in _SIX_KEYS}


def _extract_json(text: str) -> dict:
    """Extrae el primer objeto JSON del texto del modelo.

    qwen3-vl no admite la gramática `format=json` con imágenes (devuelve vacío),
    así que se pide JSON en el prompt y se parsea aquí el bloque {...}.
    """
    text = text.strip()
    # Quitar fences de código si los hubiera.
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(text[start:end + 1])
    raise ValueError("No se encontró JSON en la respuesta del modelo")


def _nivel_confianza(conf: float) -> str:
    if conf >= 0.85:
        return "alto"
    if conf >= LOW_CONFIDENCE_THRESHOLD:
        return "medio"
    return "bajo"


def _strip_think(text: str) -> str:
    """Quita bloques <think>...</think> y fences que algunos modelos emiten."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    return text.strip()


def build_explanation(diagnostico_nombre: str, confianza: float, severidad: str,
                      zona_gradcam: str, recomendacion_texto: str,
                      que_observo: str | None = None) -> dict:
    """Arma la explicación de seis claves de forma determinística y SEGURA.

    `que_observo` puede provenir del LLM (interpretación multimodal del mapa de
    calor). Los demás campos —en especial `que_hacer`— se construyen desde el
    diagnóstico y la recomendación de reglas, NUNCA inventados por el LLM.
    """
    nivel = _nivel_confianza(confianza)
    baja = confianza < LOW_CONFIDENCE_THRESHOLD
    alerta = ("La confianza del diagnóstico es baja. Antes de actuar, consulta a un "
              "técnico agrónomo o a SENASA.") if baja else ""
    return {
        "resumen": f"El sistema detectó {diagnostico_nombre} con una confianza del "
                   f"{confianza*100:.0f} %.",
        "que_observo_el_modelo": que_observo or f"El modelo se fijó en {zona_gradcam}.",
        "nivel_confianza": nivel,
        "que_hacer": recomendacion_texto,
        "alerta": alerta,
        "siguiente_paso": "Vuelve a revisar el cultivo en los próximos días y, ante "
                          "dudas, consulta a tu técnico agrónomo.",
    }


# Alias retro-compatible.
fallback_explanation = build_explanation


# Prompt enfocado en lo que el VLM hace bien: leer el mapa de calor.
# (qwen3-vl:2b deja `content` vacío con system prompts largos o format=json; con
# una instrucción CONCRETA en el mensaje del usuario sí responde en `content`.)
def _interp_prompt(diagnostico_nombre: str) -> str:
    return (
        "Eres «AgroAsistente», un asistente agrícola que habla español claro y simple. "
        "Tienes dos imágenes de una hoja de papa: la FOTO original y un MAPA DE CALOR "
        "donde las zonas rojas/amarillas son las que el modelo miró y las azules las "
        f"ignoró. El modelo diagnosticó «{diagnostico_nombre}». "
        "En 2 o 3 frases simples, di qué se observa en la hoja (manchas, color) y en "
        "qué parte (bordes, centro, puntas) se concentró el color rojo del mapa de "
        "calor. No des diagnósticos nuevos, ni dosis, ni recomendaciones."
    )


def explain(*, foto_path: str, heatmap_path: str, diagnostico_nombre: str,
            confianza: float, severidad: str, zona_gradcam: str,
            recomendacion_texto: str) -> tuple[dict, bool]:
    """Genera la explicación. Devuelve (dict de seis claves, disponible).

    El LLM aporta la interpretación multimodal del mapa de calor
    (`que_observo_el_modelo`); el resto se arma desde reglas/diagnóstico. Si el
    LLM no responde, todo es determinístico y `disponible=False`.
    """
    try:
        resp = _client().chat(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": _interp_prompt(diagnostico_nombre),
                       "images": [foto_path, heatmap_path]}],
            options={"temperature": LLM_TEMPERATURE},
        )
        interp = _strip_think(resp["message"]["content"])
        if not interp:
            raise ValueError("El modelo no devolvió texto utilizable.")
        return build_explanation(diagnostico_nombre, confianza, severidad,
                                 zona_gradcam, recomendacion_texto,
                                 que_observo=interp), True
    except Exception as e:
        _log.warning("LLM explain() no disponible, usando respaldo: %s: %s",
                     type(e).__name__, e)
        return build_explanation(diagnostico_nombre, confianza, severidad,
                                 zona_gradcam, recomendacion_texto), False


def chat(historial: list[dict], contexto_caso: str, mensaje_usuario: str,
         system_prompt: str = CHAT_SYSTEM_PROMPT) -> tuple[str, bool]:
    """Responde una pregunta de seguimiento (texto plano). Devuelve (respuesta, disponible).

    historial: lista de {"role": "user"|"assistant", "content": str}.
    system_prompt: persona/instrucciones (CHAT_SYSTEM_PROMPT por caso;
    GENERAL_CHAT_SYSTEM_PROMPT para el chat libre sin diagnóstico).
    """
    # Contexto y persona en el primer turno de usuario (más fiable que system
    # largo con este modelo); historial previo como turnos; pregunta al final.
    intro = (f"{system_prompt}\n\nContexto:\n{contexto_caso}")
    messages = [{"role": "user", "content": intro},
                {"role": "assistant", "content": "Entendido. ¿Cuál es tu pregunta?"}]
    messages += historial
    messages.append({"role": "user", "content": (
        f"Pregunta del agricultor: {mensaje_usuario}\n\n"
        "Reglas de tu respuesta: en español, breve y directa, y SOLO sobre la salud "
        "del cultivo, el diagnóstico, la enfermedad o su tratamiento. Si la pregunta es "
        "de cualquier otro tema (historia, personas, geografía, etc.), responde ÚNICAMENTE "
        "esta frase, sin agregar nada más: «Solo puedo ayudarte con preguntas sobre la "
        "salud del cultivo y este diagnóstico.»")})
    def _ask() -> str:
        resp = _client().chat(model=LLM_MODEL, messages=messages,
                              options={"temperature": LLM_TEMPERATURE})
        return _strip_think(resp["message"]["content"])

    try:
        texto = _ask()
        if not texto:
            # qwen3-vl:2b deja `content` vacío de forma intermitente (mete todo en
            # `thinking`). Un reintento suele resolverlo en preguntas legítimas.
            _log.info("LLM chat() devolvió vacío; reintentando una vez.")
            texto = _ask()
    except Exception as e:
        # Error real de conexión/servidor: Ollama no responde.
        _log.warning("LLM chat() no disponible (conexión): %s: %s", type(e).__name__, e)
        return ("Ahora mismo no puedo responder preguntas adicionales (el asistente "
                "de explicación no está disponible). El diagnóstico y la recomendación "
                "siguen siendo válidos; ante dudas, consulta a un técnico agrónomo."), False

    if not texto:
        # Sigue vacío tras el reintento: se redirige con amabilidad (sin aparentar
        # que el asistente está caído, porque Ollama SÍ respondió).
        _log.info("LLM chat() vacío tras reintento; redirigiendo al tema del cultivo.")
        return ("Solo puedo ayudarte con preguntas sobre la salud del cultivo y este "
                "diagnóstico. Pregúntame sobre la enfermedad detectada, el tratamiento "
                "o los cuidados de la planta."), True
    return texto, True
