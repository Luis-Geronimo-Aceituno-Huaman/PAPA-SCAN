# PapaScan Multiagente 🥔🤖

Versión **multiagente** del sistema de diagnóstico de papa. Vive en su propia
carpeta y **no toca** el proyecto base: reutiliza la CNN AgriVision (Grad-CAM) y
los servicios del proyecto (severidad, reglas + KB, LLM) **como herramientas**.

Mientras el proyecto base ejecuta un **pipeline fijo** (un orquestador llama a
cada paso en orden), aquí hay **agentes autónomos** coordinados por un
**supervisor** que reparte el trabajo y enruta según el caso, comunicándose por
una **pizarra (blackboard)** compartida.

## Arquitectura

```
                       ┌───────────────────────┐
   Imagen  ───────────▶│   Coordinador         │  decide el plan y despacha
                       │   (Supervisor)        │  cada agente; sintetiza el reporte
                       └──┬─────┬─────┬─────┬───┘
        ┌─────────────────┘     │     │     └─────────────────┐
        ▼                       ▼     ▼                       ▼
 AgentePercepcion     AgenteSeveridad   AgenteAgronomo   AgenteExplicador
 (CNN + Grad-CAM)     (área lesionada)  (reglas + KB)    (LLM: lee el heatmap)
        │                                                      │
        └──────────────▶ AgenteValidador (crítico) ◀───────────┘
              pre: confianza/sesgo → ¿derivar a humano?
              post: verifica que el LLM no inventó dosis (anti-alucinación)

        AgenteConversacional  → preguntas de seguimiento (bajo demanda)
```

### Los agentes

| Agente | Rol | Herramientas (código real reutilizado) |
|---|---|---|
| **Coordinador** | Planifica y despacha; arma el reporte | — |
| **AgentePercepcion** | Diagnostica con la CNN + Grad-CAM | `diagnosticar_imagen` → `app.services.diagnosis` |
| **AgenteSeveridad** | Estima gravedad del daño foliar | `estimar_severidad` → `app.services.severity` |
| **AgenteAgronomo** | Recomienda manejo (solo desde la KB) | `recomendar_tratamiento` → `app.services.rules_engine` |
| **AgenteExplicador** | Explica en lenguaje simple (LLM) | `explicar_caso` → `app.services.llm_client` |
| **AgenteValidador** | Crítico: deriva a humano y bloquea alucinaciones | reglas deterministas |
| **AgenteConversacional** | Chat de seguimiento del caso | `responder_chat` → `app.services.llm_client` |

### Conceptos multiagente implementados

- **Supervisor + especialistas**: el Coordinador enruta; cada agente es autónomo
  en su rol (p. ej. el de Severidad resuelve solo el caso "hoja sana").
- **Blackboard**: memoria compartida; los agentes publican y consumen resultados
  ahí (no se llaman entre sí directamente). El diálogo queda registrado.
- **Herramientas (tools)**: las capacidades se exponen por nombre en un registro;
  los agentes las invocan sin conocer la implementación.
- **Agente crítico / guardrail**: el Validador deriva a técnico ante baja
  confianza o sesgo, y **verifica que la recomendación venga del motor de reglas**
  y que el LLM no haya colado dosis inventadas (seguridad heredada del brief).
- **Degradación elegante**: si Ollama no está, el Explicador usa un respaldo
  determinístico. El diagnóstico **nunca** depende del LLM.

## Requisitos

Usa el **mismo entorno** que el proyecto base (no agrega dependencias nuevas):
`torch`, `opencv-python`, `pillow`, `numpy`, `ollama`. Necesita el **modelo
entrenado** en `artifacts/models/` (ya incluido en el repo) y, opcionalmente,
**Ollama** corriendo con `qwen3-vl:2b` para la explicación con LLM.

## Uso

Desde la raíz del proyecto (con el venv activado):

```bash
# Diagnóstico completo (usa LLM si Ollama está disponible)
python -m multiagente.run --image "Test/Potato___Early_blight/alguna.JPG"

# Sin LLM (offline puro, respaldo determinístico)
python -m multiagente.run --image hoja.jpg --no-llm

# Salida JSON (para integrar con otra cosa)
python -m multiagente.run --image hoja.jpg --json

# Con una pregunta de seguimiento al agente conversacional
python -m multiagente.run --image hoja.jpg --ask "¿es contagioso?"
```

La salida muestra el **diálogo entre agentes** (trazabilidad), el **reporte final**
(diagnóstico, severidad, recomendación, explicación, verificación de seguridad) y
las rutas de la foto/heatmap/overlay generados en `multiagente/outputs/`.

## Relación con el proyecto base

- **No reemplaza** a la app PapaScan (`app/`): es una arquitectura alternativa.
- **Reutiliza** el modelo y la lógica probada; si cambias un servicio en `app/`,
  el sistema multiagente lo hereda (las herramientas son finas envolturas).
- Pensado para demostrar el enfoque **multiagente** sin perder el trabajo previo.

## Estructura

```
multiagente/
  core/         blackboard.py · agent.py · bootstrap.py
  tools/        registry.py · vision · severity · knowledge · explanation
  agents/       perception · severity_agent · agronomist · explainer
                validator · conversational
  coordinator.py   Supervisor que orquesta el flujo
  run.py           CLI
  outputs/         foto/heatmap/overlay generados (no se versiona)
```
