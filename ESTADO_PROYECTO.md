# Estado del proyecto — Punto de retomar

> Documento de contexto para continuar otro día. Última actualización: 2026-06-17.

---

## 1. Visión general

Sistema de **3 capas** para diagnóstico de enfermedades foliares de papa, **100 % local y offline**:

1. **AgriVision** (modelo) — CNN (EfficientNet-B0) + Grad-CAM. Diagnostica 3 clases y explica visualmente. ✅ **Entrenado y guardado.**
2. **PapaScan** (capa de explicación + web) — orquestador FastAPI que encadena CNN → Grad-CAM → severidad → motor de reglas → LLM, con interfaz web. ✅ **Construido y probado end-to-end.**
3. *(futuro)* LLM agrónomo de recomendaciones — la recomendación hoy la da un **motor de reglas** (no el LLM, por seguridad).

Specs: `agrivision_prompt (1).md` (modelo) y `brief_capa_explicacion.md` (capa LLM). Prototipos UI en `PROTOTIPOS DE INTERFAZ/`.

---

## 2. Qué está HECHO ✅

### Modelo AgriVision
- **VIGENTE: `artifacts/models/agrivision_v1_2026-06-18.pt` (v2, fondo enmascarado).** El anterior `..._2026-06-17.pt` (v1) queda como respaldo. La app toma el .pt más reciente por fecha.
- **Por qué v2:** el Grad-CAM de v1 reveló un **atajo de fondo** (en hojas sanas/tizón temprano el calor caía FUERA de la hoja). Se activó el enmascarado de fondo disease-agnostic (`ROI_BG_MASK_ENABLED=True`) y se reentrenó. Ahora el calor cae sobre la hoja/lesión (verificado: `outputs/v2_compare/`, `outputs/baseline_roi/`).
- **Test v2 (300 imgs, una vez): MCC=0.985, recall tizón tardío=0.97 (≥0.90 ✓), `listo_produccion=True`.** CV 5-fold MCC=1.000±0.000. 5 gráficos en `resultados/`.
- **No reentrenar para usar**: la inferencia solo carga el `.pt`.
- Hiperparámetros v2 (Optuna 40 trials, 13 pruned): lr=0.000599, batch=32, dropout=0.133, wd=0.00625, 5 bloques descongelados (98% de la red). Calibración T=0.5. Umbral sesgo=0.0 (con fondo enmascarado la masa-en-fondo de aciertos es ~0).

### App PapaScan (`app/`)
- **Backend FastAPI** completo: auth (JWT), diagnose, explain, chat, history.
- **PostgreSQL** BD `papascan` (tablas: users, cases, chat_messages). Columna `cases.masked_path` añadida con migración idempotente en `database.py` (`ALTER TABLE ... ADD COLUMN IF NOT EXISTS`).
- **LLM**: Ollama + `qwen3-vl:2b` (offline). Conexión por `127.0.0.1` (no `localhost`).
- **Frontend SPA** (`app/templates/index.html`, `app/static/`) — rediseñado como flujo tipo asistente:
  - Menú con 3 entradas: **Nuevo análisis** · **Asistente** · **Historial** (antes 5 pestañas que salían vacías).
  - Flujo: Nuevo análisis (subir + contenido educativo) → Diagnóstico (botón ← volver) → opciones **Explicabilidad** y **Recomendación** (cada una con ← volver).
  - **Explicabilidad** muestra 3 imágenes: original | **enmascarada** (lo que ve el modelo) | Grad-CAM. La enmascarada se genera en `predict.py` (`input_rgb`) y se sirve vía `/media`.
  - **Chat de seguimiento** en Explicabilidad y Recomendación (sincronizados, mismo caso) + pestaña **Asistente** = chat libre sin diagnóstico (`case_id` opcional, historial del cliente, no persistido). Cache-bust de estáticos con `?v=N` en index.html (subir N al editar css/js).
- **Motor de reglas** + KB curada (`app/knowledge_base/fito_kb.json`).
- **Probado end-to-end OK**: auth → diagnóstico → explicación LLM (multimodal, ~11s) → chat (con caso y libre) → historial.

---

## 3. Cómo arrancar todo (paso a paso)

```powershell
# 1. PostgreSQL (servicio) — normalmente ya arranca solo con Windows
Get-Service postgresql-x64-17    # debe estar Running

# 2. Ollama (servidor del LLM) — suele arrancar con el sistema
#    Verificar: debe responder en http://localhost:11434
ollama list                       # debe listar qwen3-vl:2b

# 3. App PapaScan  (NO fijar OLLAMA_HOST: el default 127.0.0.1 es offline-robusto)
cd "C:\Users\USER\Desktop\PROYECTO FINAL IA"
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Abrir en el navegador: **http://localhost:8000**
- Desde otro dispositivo en la LAN: `http://<IP-de-la-PC>:8000`
- Usuario de prueba: `agricultor_demo` / `papa1234` (o crear cuenta nueva).

### Reentrenar el modelo (solo si se quiere otra versión)
```powershell
python scripts/01_train.py                 # pipeline completo (~45 min)
python scripts/01_train.py --smoke          # prueba rápida
python scripts/02_infer.py --image ruta.jpg # inferencia por CLI
python scripts/03_calibrate_bias.py         # recalibrar umbral de alerta de sesgo
```

---

## 4. Estructura del proyecto

```
config/config.py            Config del modelo (clases, rutas, umbrales)
src/                        Código del modelo AgriVision
  data/ models/ training/ evaluation/ inference/ utils/
scripts/                    01_train.py · 02_infer.py · 03_calibrate_bias.py
artifacts/models/           Modelo entrenado + metadatos
resultados/                 5 gráficos de evaluación
Train_Valid/  Test/         Dataset (1200 / 300 imágenes)

app/                        App PapaScan (capa de explicación + web)
  main.py settings.py database.py models.py schemas.py security.py deps.py
  services/   validation diagnosis severity rules_engine llm_client orchestrator
  routers/    auth diagnose explain chat history
  knowledge_base/fito_kb.json
  templates/index.html  static/css/styles.css  static/js/app.js
storage/                    uploads/ heatmaps/ (generados en runtime)
```

---

## 5. Configuración / credenciales (local)

| Qué | Valor |
|---|---|
| PostgreSQL | BD `papascan`, user `postgres`, pass `postgres`, puerto 5432 |
| Ollama | `http://localhost:11434`, modelo `qwen3-vl:2b` |
| App | puerto 8000 |
| GPU | RTX 5060 (Blackwell sm_120), torch nightly cu132 |

Todo configurable por variables de entorno (`PAPASCAN_*`, `OLLAMA_HOST`) — ver `app/settings.py`.

---

## 6. Decisiones y gotchas importantes (para no repetir investigación)

- **GPU RTX 50 (Blackwell)**: requiere torch **nightly cu132** (sm_120). NO instalar torch estable de PyPI.
- **bcrypt 5 ↔ passlib**: incompatibles. `app/security.py` usa **bcrypt directo** (no passlib).
- **qwen3-vl:2b quirk**: deja `content` VACÍO con system prompts largos o `format="json"` (mete todo en `thinking`). Solución aplicada: instrucción CONCRETA en el mensaje de USUARIO, sin `format=json`. El LLM **solo interpreta el mapa de calor**; los demás campos del JSON se arman determinísticamente (más fiable y más seguro).
- **Seguridad (brief)**: las recomendaciones vienen SOLO del motor de reglas, nunca del LLM. El LLM explica, no diagnostica ni receta.
- **Diagnóstico vs explicación separados**: `/diagnose` es rápido (sin LLM); `/explain/{id}` llama al LLM bajo demanda (capa opcional). Si el LLM falla, hay fallback determinístico.
- **`localhost` vs `127.0.0.1` (offline)**: usar SIEMPRE `127.0.0.1` para Ollama y PostgreSQL. En Windows `localhost` resuelve primero a IPv6 `::1` (donde Ollama no escucha): añade ~2 s de rodeo con red y, al desconectar el wifi, la conexión a `::1` deja de rechazarse limpio y el cliente HTTP falla → la explicación cae al respaldo determinístico (`disponible=False`). Los defaults en `app/settings.py` ya son `127.0.0.1`; el error del LLM ahora se registra (logger `papascan.llm`) en vez de tragarse.
- **Segmentación de hoja (roi.py) disease-agnostic**: la máscara une verde/amarillo ∪ marrón necrótico y rellena huecos (silueta sólida), para NO recortar la lesión. Sirve para el enmascarado de fondo (v2) y para la auditoría de sesgo. Flags en `config.py`: `ROI_BG_MASK_ENABLED=True`, `ROI_CROP_ENABLED=False`.
- **alerta_sesgo** del Grad-CAM: con v2 (fondo enmascarado) la masa-en-fondo de aciertos es ~0 → `umbral_sesgo_fondo=0.0`; la alerta casi no dispara (auditoría algo redundante). En v1 era 0.697.
- **Chat (`llm_client.chat`)**: el `content` vacío de qwen3-vl:2b es intermitente → se REINTENTA una vez; si sigue vacío se redirige (no se dice "caído"). Guardia off-topic con frase de rechazo FIJA en el turno de usuario. Dos prompts: `CHAT_SYSTEM_PROMPT` (chat de un caso) y `GENERAL_CHAT_SYSTEM_PROMPT` (pestaña Asistente, sin diagnóstico).

---

## 7. Pendientes / ideas para próximos días

- [ ] Verificar el flujo completo en el **navegador** (captura real de diagnóstico/explicabilidad) y pulir detalles visuales de las secciones vs. prototipos.
- [ ] **Validar la base de conocimiento `fito_kb.json`** (dosis/productos) con un ingeniero agrónomo / SENASA antes de uso real. Hoy son valores de referencia con disclaimer.
- [ ] Exportar reporte a PDF (hoy usa `window.print()` del navegador).
- [ ] Considerar `qwen3-vl:4b` si se quiere mejor calidad de explicación (cabe en la GPU).
- [ ] Opcional: estadio fenológico/región configurables por el usuario en la UI (hoy salen de la KB por defecto).
- [ ] Tests automatizados (pytest) del backend.

---

## 8. Estado al cerrar la sesión (2026-06-18)

Sesión del 18-jun-2026 (resumen de cambios):
1. **Fix offline**: Ollama/PostgreSQL ahora por `127.0.0.1` (el `localhost`→IPv6 rompía la explicación sin wifi). Error del LLM logueado, no tragado.
2. **Modelo v2** (`agrivision_v1_2026-06-18`): reentrenado con **enmascarado de fondo** porque el Grad-CAM de v1 reveló un atajo de fondo. Test MCC=0.985, recall tardío=0.97, listo_produccion=True. v1 conservado como respaldo.
3. **UI**: 3ª imagen (enmascarada) en Explicabilidad; rediseño a flujo asistente (Nuevo análisis → Diagnóstico → Explicabilidad/Recomendación con botones ← volver); contenido educativo; chat en Explicabilidad+Recomendación (sincronizado) y pestaña **Asistente** (chat libre).
4. **Chat**: manejo de respuesta vacía (reintento+redirección), guardia off-topic, prompt general para el chat libre.

- Todo probado end-to-end (vía API). Servidor y Ollama detenidos al cerrar (relanzar con la sección 3; NO fijar `OLLAMA_HOST` a localhost).
- Sin tareas a medio hacer ni código roto.
- Scripts de diagnóstico temporales en `scripts/_*.py` (baseline_roi, roi_preview, ejemplo_recorte, compare_overlays, test_chat) — auditables/borrables.
- **Pendiente sugerido**: ejecutar el flujo real en el navegador (recargar con caché limpia) para validación visual final.
