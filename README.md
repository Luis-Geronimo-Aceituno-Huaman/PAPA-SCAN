# AgriVision + PapaScan 🥔🔬

Sistema **100 % local y offline** para diagnóstico de enfermedades foliares de papa
(*Solanum tuberosum*), en dos capas:

1. **AgriVision** — modelo de visión: CNN (EfficientNet-B0) + **Grad-CAM**. Clasifica
   3 clases y explica visualmente dónde mira. ✅ ya entrenado (incluido en el repo).
2. **PapaScan** — app web (FastAPI) con login, historial y chat. La recomendación la
   da un motor de reglas, no el LLM (por seguridad). **Su motor interno es
   multiagente**: un Coordinador despacha agentes especialistas (Percepción,
   Validador, Severidad, Agrónomo, Explicador) sobre una pizarra compartida
   (ver [`multiagente/`](multiagente/)). El frontend y la API no cambian.

## Clases

| Clase | Carpeta dataset | Patógeno |
|---|---|---|
| `sana` | `Potato___healthy` | — |
| `tizon_tardio` | `Potato___Late_blight` | *Phytophthora infestans* |
| `tizon_temprano` | `Potato___Early_blight` | *Alternaria solani* |

---

## ⚡ Cómo correrlo en otra PC (cualquier SO)

Necesitás 3 cosas: **Python 3.11+**, **PostgreSQL** (vía Docker, lo más fácil) y
**Ollama** (para la capa de explicación con LLM). El diagnóstico funciona aunque
Ollama no esté: en ese caso la explicación cae a un texto determinístico.

### 1. Clonar e instalar dependencias

```bash
git clone <URL-DE-TU-REPO>.git
cd "PROYECTO FINAL IA"

# Crear entorno virtual
python -m venv .venv

# Activarlo:
#   Windows (PowerShell):  .venv\Scripts\Activate.ps1
#   Windows (CMD):         .venv\Scripts\activate.bat
#   macOS / Linux:         source .venv/bin/activate

pip install -r requirements.txt
```

> **GPU NVIDIA RTX 50 (Blackwell, p.ej. RTX 5060):** instalá torch aparte ANTES del
> paso anterior (ver nota en `requirements.txt`). Para solo **correr la app** no hace
> falta GPU: la inferencia anda perfecto en CPU.

### 2. Base de datos PostgreSQL

**Opción A — Docker (recomendada, idéntica en todo SO):**

```bash
docker compose up -d
```

Esto crea la BD `papascan` con usuario/clave `postgres`/`postgres` (ya coinciden con
los defaults de la app). Para detenerla: `docker compose down`.

**Opción B — PostgreSQL instalado nativamente:** creá una base llamada `papascan`.
Si tu usuario/clave/puerto difieren, copiá `.env.example` a `.env` y ajustalos.
Las tablas se crean solas al arrancar la app.

### 3. Ollama (LLM local, opcional pero recomendado)

Instalá [Ollama](https://ollama.com/download) (Windows/macOS/Linux) y descargá el modelo:

```bash
ollama pull qwen3-vl:2b
```

Ollama queda escuchando en `http://127.0.0.1:11434`.

### 4. Arrancar la app

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Abrí **http://localhost:8000**
- Usuario demo: `agricultor_demo` / `papa1234` (o registrá una cuenta nueva).
- Desde otro dispositivo en la misma red: `http://<IP-de-la-PC>:8000`

---

## Configuración

Todo se configura por variables de entorno con defaults para uso local. Copiá
`.env.example` a `.env` solo si querés cambiar credenciales de BD, la clave JWT
o el modelo de LLM. Ver detalle en `app/settings.py`.

---

## Estructura

```
config/config.py        Config del modelo (clases, rutas, umbrales)
src/                    Código de AgriVision (data, models, training, evaluation,
                        inference, utils)
scripts/                01_train.py · 02_infer.py · 03_calibrate_bias.py
artifacts/models/       Modelo entrenado (.pt) + metadatos  ← incluido en el repo
resultados/             Gráficos de evaluación
app/                    App PapaScan (FastAPI)
  main.py settings.py database.py models.py security.py ...
  services/  routers/  knowledge_base/  templates/  static/
storage/                uploads/ heatmaps/ reports/ (se generan en runtime)
docker-compose.yml      PostgreSQL para levantar la BD en un comando
```

---

## (Opcional) Re-entrenar el modelo

El repo ya trae el modelo entrenado, así que esto **no es necesario** para usar la app.
Para re-entrenar hace falta el dataset (no se versiona por tamaño): imágenes de papa
de **PlantVillage** (carpetas `Potato___healthy`, `Potato___Late_blight`,
`Potato___Early_blight`) repartidas en `Train_Valid/` (1200) y `Test/` (300).

```bash
python scripts/01_train.py            # pipeline completo (~45 min con GPU)
python scripts/01_train.py --smoke    # prueba rápida
python scripts/02_infer.py --image ruta.jpg   # inferencia por CLI
```

---

## Métricas del modelo incluido (v2)

- Test (300 imgs, una sola vez): **MCC = 0.985**, recall tizón tardío = 0.97 (≥ 0.90 ✓).
- CV 5-fold: MCC = 1.000 ± 0.000. `listo_produccion = True`.
- Backbone EfficientNet-B0 con enmascarado de fondo (mata el atajo de fondo que
  delató el Grad-CAM en v1).
