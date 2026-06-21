"""Configuración de la aplicación PapaScan (capa de explicación + API).

Lee de variables de entorno con valores por defecto razonables para el
despliegue local offline. Nada de esto debe apuntar a servicios en la nube.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

# --------------------------------------------------------------------------- #
# Base de datos (PostgreSQL local)
# --------------------------------------------------------------------------- #
DB_USER = os.getenv("PAPASCAN_DB_USER", "postgres")
DB_PASSWORD = os.getenv("PAPASCAN_DB_PASSWORD", "postgres")
DB_HOST = os.getenv("PAPASCAN_DB_HOST", "127.0.0.1")  # IPv4 directo (ver nota OLLAMA_HOST)
DB_PORT = os.getenv("PAPASCAN_DB_PORT", "5432")
DB_NAME = os.getenv("PAPASCAN_DB_NAME", "papascan")
DATABASE_URL = os.getenv(
    "PAPASCAN_DATABASE_URL",
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
)

# --------------------------------------------------------------------------- #
# Autenticación (JWT)
# --------------------------------------------------------------------------- #
# En producción real, fijar PAPASCAN_SECRET_KEY por entorno (no usar el default).
SECRET_KEY = os.getenv("PAPASCAN_SECRET_KEY", "cambia-esta-clave-en-produccion-32+chars")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("PAPASCAN_TOKEN_MINUTES", "720"))  # 12 h

# --------------------------------------------------------------------------- #
# LLM local (Ollama, VLM multimodal)
# --------------------------------------------------------------------------- #
# 127.0.0.1 (IPv4) en vez de "localhost": en Windows "localhost" resuelve primero
# a IPv6 (::1), donde Ollama NO escucha, lo que añade ~2 s de rodeo con red y FALLA
# al desconectar la red (la pila IPv6 deja de rechazar limpio). 127.0.0.1 es loopback
# IPv4 directo: instantáneo y 100 % offline.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("PAPASCAN_LLM_MODEL", "qwen3-vl:2b")
LLM_TEMPERATURE = float(os.getenv("PAPASCAN_LLM_TEMPERATURE", "0.2"))
LLM_TIMEOUT_S = float(os.getenv("PAPASCAN_LLM_TIMEOUT", "120"))

# --------------------------------------------------------------------------- #
# Almacenamiento local de archivos (imágenes, mapas de calor, reportes)
# --------------------------------------------------------------------------- #
STORAGE_DIR = Path(os.getenv("PAPASCAN_STORAGE", str(ROOT_DIR / "storage")))
UPLOADS_DIR = STORAGE_DIR / "uploads"
HEATMAPS_DIR = STORAGE_DIR / "heatmaps"
REPORTS_DIR = STORAGE_DIR / "reports"

# Umbral de confianza por debajo del cual se exige derivar a un técnico (brief).
LOW_CONFIDENCE_THRESHOLD = float(os.getenv("PAPASCAN_LOW_CONF", "0.60"))

for _d in (STORAGE_DIR, UPLOADS_DIR, HEATMAPS_DIR, REPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
