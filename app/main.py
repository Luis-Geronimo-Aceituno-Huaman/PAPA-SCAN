"""Aplicación FastAPI de PapaScan — servidor local offline.

Monta la API (auth, diagnóstico, chat, historial), sirve los archivos de medios
(imágenes y mapas de calor) y el frontend estático. Pensado para correr en una
laptop/mini-PC y ser accedido por la red local (WiFi/LAN), sin internet.
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import init_db
from app.routers import auth, chat, diagnose, explain, history
from app.services import llm_client
from app.settings import APP_DIR, LLM_MODEL, STORAGE_DIR

app = FastAPI(
    title="PapaScan — Diagnóstico inteligente de papa",
    description="Capa de explicación con LLM local sobre el diagnóstico CNN + Grad-CAM.",
    version="1.0.0",
)

# Acceso desde la red local (celular/tablet). No exponer a internet sin auth.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(diagnose.router)
app.include_router(explain.router)
app.include_router(chat.router)
app.include_router(history.router)

# Archivos de medios (imágenes y mapas de calor generados) y frontend estático.
app.mount("/media", StaticFiles(directory=str(STORAGE_DIR)), name="media")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


@app.on_event("startup")
def _startup():
    init_db()


@app.get("/api/health")
def health():
    """Estado del sistema: BD viva, LLM disponible (informativo)."""
    return {
        "status": "ok",
        "llm_model": LLM_MODEL,
        "llm_disponible": llm_client.is_available(),
        "modo": "local-offline",
    }


@app.get("/")
def index():
    """Sirve la app PapaScan."""
    return FileResponse(str(APP_DIR / "templates" / "index.html"))
