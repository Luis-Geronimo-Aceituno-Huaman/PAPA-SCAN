"""Motor y sesión de SQLAlchemy (PostgreSQL local)."""
from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.settings import DATABASE_URL

engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


class Base(DeclarativeBase):
    """Base declarativa para los modelos ORM."""
    pass


def get_db():
    """Dependencia de FastAPI: entrega una sesión y la cierra al terminar."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Crea las tablas si no existen y aplica migraciones ligeras idempotentes."""
    from app import models  # noqa: F401  (registra los modelos en Base)
    Base.metadata.create_all(bind=engine)
    # Migración: columna añadida después de la creación inicial de la tabla.
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE cases ADD COLUMN IF NOT EXISTS masked_path VARCHAR(512) DEFAULT ''"))
