"""Modelos ORM: usuarios e historial de casos."""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    """Usuario del sistema (agricultor o técnico agrícola)."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str] = mapped_column(String(128), default="")
    role: Mapped[str] = mapped_column(String(32), default="agricultor")  # agricultor|tecnico
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    cases: Mapped[list["Case"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Case(Base):
    """Un caso de diagnóstico guardado en el historial."""
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    # Resultado del diagnóstico (CNN)
    diagnostico: Mapped[str] = mapped_column(String(64))
    confianza: Mapped[float] = mapped_column(Float)
    estado_confianza: Mapped[str] = mapped_column(String(32))
    severidad: Mapped[str] = mapped_column(String(32), default="")
    zona_gradcam: Mapped[str] = mapped_column(String(256), default="")

    # Rutas a archivos locales
    imagen_path: Mapped[str] = mapped_column(String(512))
    masked_path: Mapped[str] = mapped_column(String(512), default="")  # imagen enmascarada (lo que vio el modelo)
    heatmap_path: Mapped[str] = mapped_column(String(512), default="")
    overlay_path: Mapped[str] = mapped_column(String(512), default="")

    # Datos estructurados (JSON serializado como texto)
    probabilidades_json: Mapped[str] = mapped_column(Text, default="{}")
    recomendacion_json: Mapped[str] = mapped_column(Text, default="{}")
    explicacion_json: Mapped[str] = mapped_column(Text, default="{}")
    version_modelo: Mapped[str] = mapped_column(String(64), default="")

    user: Mapped["User"] = relationship(back_populates="cases")
    messages: Mapped[list["ChatMessage"]] = relationship(
        back_populates="case", cascade="all, delete-orphan", order_by="ChatMessage.created_at")


class ChatMessage(Base):
    """Mensaje de la conversación de seguimiento sobre un caso."""
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user|assistant
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    case: Mapped["Case"] = relationship(back_populates="messages")
