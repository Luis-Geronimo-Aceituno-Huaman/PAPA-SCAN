"""Router de historial: lista y detalle de casos del usuario."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Case, User
from app.schemas import CaseSummary

router = APIRouter(prefix="/api/history", tags=["history"])


def _media_url(path: str) -> str:
    from app.routers.diagnose import media_url
    try:
        return media_url(path)
    except Exception:
        return ""


@router.get("", response_model=list[CaseSummary])
def list_cases(db: Session = Depends(get_db), current: User = Depends(get_current_user)):
    cases = (db.query(Case).filter(Case.user_id == current.id)
             .order_by(Case.created_at.desc()).limit(100).all())
    return [
        CaseSummary(id=c.id, created_at=c.created_at, diagnostico=c.diagnostico,
                    confianza=c.confianza, severidad=c.severidad,
                    imagen_url=_media_url(c.imagen_path))
        for c in cases
    ]


@router.get("/{case_id}")
def case_detail(case_id: int, db: Session = Depends(get_db),
                current: User = Depends(get_current_user)):
    case = db.query(Case).filter(Case.id == case_id,
                                 Case.user_id == current.id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Caso no encontrado.")
    return {
        "id": case.id,
        "created_at": case.created_at,
        "diagnostico": case.diagnostico,
        "confianza": case.confianza,
        "estado_confianza": case.estado_confianza,
        "severidad": case.severidad,
        "imagen_url": _media_url(case.imagen_path),
        "masked_url": _media_url(case.masked_path),
        "heatmap_url": _media_url(case.heatmap_path),
        "overlay_url": _media_url(case.overlay_path),
        "probabilidades": json.loads(case.probabilidades_json or "{}"),
        "recomendacion": json.loads(case.recomendacion_json or "{}"),
        "explicacion": json.loads(case.explicacion_json or "{}"),
        "version_modelo": case.version_modelo,
        "mensajes": [{"role": m.role, "content": m.content,
                      "created_at": m.created_at} for m in case.messages],
    }
