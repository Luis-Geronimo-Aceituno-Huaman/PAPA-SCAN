"""Esquemas Pydantic para entrada/salida de la API."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


# --------------------------- Autenticación --------------------------- #
class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=4, max_length=128)
    full_name: str = ""
    role: str = "agricultor"


class UserOut(BaseModel):
    id: int
    username: str
    full_name: str
    role: str

    class Config:
        from_attributes = True


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserOut


# --------------------------- Explicación (LLM) --------------------------- #
class Explicacion(BaseModel):
    """Salida estructurada del LLM (seis claves fijas del brief)."""
    resumen: str = ""
    que_observo_el_modelo: str = ""
    nivel_confianza: str = ""        # alto | medio | bajo
    que_hacer: str = ""
    alerta: str = ""                 # "" si no aplica
    siguiente_paso: str = ""


# --------------------------- Recomendación (reglas) --------------------------- #
class ControlItem(BaseModel):
    titulo: str
    contenido: str
    fuente: str = ""


class Recomendacion(BaseModel):
    enfermedad: str
    severidad: str
    estadio_fenologico: str = ""
    region: str = ""
    controles: list[ControlItem] = []
    trazabilidad: str = ""


# --------------------------- Diagnóstico (respuesta completa) --------------------------- #
class DiagnoseResponse(BaseModel):
    case_id: int
    diagnostico: str
    confianza: float
    estado_confianza: str
    probabilidades: dict[str, float]
    severidad: str
    imagen_url: str
    masked_url: str = ""        # imagen enmascarada (lo que vio el modelo)
    heatmap_url: str
    overlay_url: str
    capa_usada: str
    alerta_sesgo: bool
    zona_gradcam: str
    recomendacion: Recomendacion
    version_modelo: str


class ExplainResponse(BaseModel):
    """Respuesta del endpoint de explicación (capa LLM opcional)."""
    case_id: int
    explicacion: Explicacion
    explicacion_disponible: bool          # False si el LLM no respondió (fallback)


# --------------------------- Chat de seguimiento --------------------------- #
class ChatTurn(BaseModel):
    role: str                            # "user" | "assistant"
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    case_id: int | None = None           # None = chat libre (sin diagnóstico)
    mensaje: str = Field(min_length=1, max_length=1000)
    historial: list[ChatTurn] = []       # solo se usa en chat libre (no persistido)


class ChatResponse(BaseModel):
    respuesta: str
    disponible: bool


# --------------------------- Historial --------------------------- #
class CaseSummary(BaseModel):
    id: int
    created_at: datetime
    diagnostico: str
    confianza: float
    severidad: str
    imagen_url: str

    class Config:
        from_attributes = True
