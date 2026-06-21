"""
Configuración central de AgriVision.

Único lugar donde viven rutas, nombres de clase, mapeos de carpeta, constantes
de preprocesamiento y umbrales/objetivos de métricas definidos en el prompt.
Todos los demás módulos importan desde aquí para garantizar consistencia.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Rutas del proyecto
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent

TRAIN_VALID_DIR = ROOT_DIR / "Train_Valid"   # 1200 imágenes (400/clase)
TEST_DIR = ROOT_DIR / "Test"                  # 300 imágenes (100/clase) — se mira UNA vez

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"         # checkpoints + metadatos
OPTUNA_DIR = ARTIFACTS_DIR / "optuna"         # estudios/db de Optuna
RESULTADOS_DIR = ROOT_DIR / "resultados"      # los 5 .png de evaluación
OUTPUTS_DIR = ROOT_DIR / "outputs"            # overlays/heatmaps de inferencia

# --------------------------------------------------------------------------- #
# Clases y mapeo de carpetas (PlantVillage -> nomenclatura del prompt)
# --------------------------------------------------------------------------- #
# El ORDEN define el índice de cada clase en el vector de salida del modelo.
CLASSES: list[str] = ["sana", "tizon_tardio", "tizon_temprano"]

CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS: dict[int, str] = {i: c for c, i in CLASS_TO_IDX.items()}

# Nombre de carpeta en disco -> clase canónica del proyecto.
FOLDER_TO_CLASS: dict[str, str] = {
    "Potato___healthy": "sana",
    "Potato___Late_blight": "tizon_tardio",     # Phytophthora infestans
    "Potato___Early_blight": "tizon_temprano",  # Alternaria solani
}

# Etiquetas legibles para gráficos.
CLASS_LABELS_ES: dict[str, str] = {
    "sana": "Sana",
    "tizon_tardio": "Tizón tardío",
    "tizon_temprano": "Tizón temprano",
}

# --------------------------------------------------------------------------- #
# Preprocesamiento (debe ser idéntico en train e inferencia)
# --------------------------------------------------------------------------- #
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentación (solo train). Parámetros exactos del Paso 4 del prompt.
AUG_ROTATION_DEG = 30
AUG_RRC_SCALE = (0.8, 1.0)
AUG_RRC_RATIO = (0.9, 1.1)
AUG_JITTER_BRIGHTNESS = 0.2
AUG_JITTER_CONTRAST = 0.2
AUG_JITTER_SATURATION = 0.1   # leve; >0.3 está prohibido
AUG_JITTER_HUE = 0.0          # prohibido modificar el tono (señal diagnóstica)

# Recorte ROI + enmascarado de fondo (Paso 2). En PlantVillage la hoja llena el
# cuadro, pero el modelo aprende un ATAJO de fondo (el Grad-CAM lo delata: en hojas
# sanas/temprano el calor cae fuera de la hoja). Enmascarar el fondo a un color
# neutro fuerza al modelo a mirar la hoja y lo hace robusto a fotos de campo.
# La segmentación es disease-agnostic (incluye marrón necrótico) para NO recortar
# la evidencia. Debe ser IDÉNTICO en train e inferencia (los transforms lo aplican
# en ambos pipelines), así que cambiar estos flags exige reentrenar.
ROI_CROP_ENABLED = False      # recorta al bounding box de la hoja (+ margen)
ROI_BG_MASK_ENABLED = True    # pone el fondo (no-hoja) a color neutro -> mata el atajo
ROI_MARGIN = 0.10             # margen del 10% alrededor del bbox de la hoja
ROI_MIN_FOLIAR_FRAC = 0.03    # si se detecta menos hoja que esto -> NO tocar (fallback)
ROI_MIN_RESOLUTION = 64       # px; imágenes menores se descartan (Paso 1)
ROI_MAX_NONFOLIAR_FRAC = 0.60 # >60% de área no foliar -> descartar/recortar

# --------------------------------------------------------------------------- #
# Objetivos de métricas por clase (del prompt). Sirven como criterio de
# "listo para producción" y como línea punteada en los gráficos.
# --------------------------------------------------------------------------- #
# metrica_principal: la métrica que decide si la clase cumple.
# minimo: umbral mínimo aceptable (None = sin mínimo duro definido).
# resumen: métrica de resumen reportada para la clase.
CLASS_METRIC_TARGETS: dict[str, dict] = {
    "tizon_tardio": {
        "metrica_principal": "recall",
        "minimo": 0.90,
        "resumen": "f2",
        "sensibilidad": "Sensibilidad 1 — detección de tizón tardío",
    },
    "tizon_temprano": {
        "metrica_principal": "recall",
        "minimo": 0.80,
        "resumen": "f1",
        "sensibilidad": "Sensibilidad 2 — detección de tizón temprano",
    },
    "sana": {
        "metrica_principal": "precision",
        "minimo": None,
        "resumen": "f1",
        "sensibilidad": "Sensibilidad 3 — reconocimiento de hoja sana",
    },
}

# Umbral de confianza por defecto antes de calibración (se reemplaza por el
# umbral por clase elegido en entrenamiento vía curvas ROC).
DEFAULT_CONFIDENCE_THRESHOLD = 0.60

# --------------------------------------------------------------------------- #
# Validación / HPO
# --------------------------------------------------------------------------- #
N_FOLDS = 5                  # validación cruzada estratificada final
HPO_VAL_SPLIT = 0.20         # 80/20 solo para búsqueda de hiperparámetros
N_OPTUNA_TRIALS = 40         # 30–50 recomendado por el prompt
RANDOM_SEED = 42

# Épocas por fase (HPO usa pocas + pruning; final/CV entrenan más).
HPO_EPOCHS = 8               # trials cortos: comparar, no converger
CV_EPOCHS = 20               # cada fold de la validación cruzada
FINAL_EPOCHS = 30            # modelo final con las 1200 imágenes
EARLY_STOP_PATIENCE = 6
NUM_WORKERS = 4              # workers del DataLoader (CPU prepara batches)
MONITOR_METRIC = "mcc"       # métrica de early stopping / selección

# Backbone: liviano para GPU de 8 GB (RTX 5060) y dataset de 1200 imágenes.
BACKBONE = "efficientnet_b0"   # alternativa: "mobilenet_v2"

# --------------------------------------------------------------------------- #
# Versión del modelo (prefijo de artefactos y gráficos)
# --------------------------------------------------------------------------- #
MODEL_NAME = "agrivision"
MODEL_VERSION = "v1"


def make_version_id(date_str: str) -> str:
    """Construye el identificador de versión, p.ej. 'agrivision_v1_2026-06-17'.

    La fecha se pasa explícitamente (no se llama a datetime aquí) para que el
    identificador sea reproducible y controlado por el script que entrena.
    """
    return f"{MODEL_NAME}_{MODEL_VERSION}_{date_str}"
