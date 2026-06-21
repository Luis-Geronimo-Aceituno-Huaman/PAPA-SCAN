"""Router de explicación: capa LLM OPCIONAL sobre un caso ya diagnosticado.

Separada del diagnóstico (brief §3-4): el diagnóstico nunca depende del LLM. Si
el LLM falla, devuelve una explicación de respaldo y `explicacion_disponible=False`.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Case, User
from app.schemas import ExplainResponse, Explicacion
from app.services import rules_engine
from app.multiagente import web as ma_web

router = APIRouter(prefix="/api", tags=["explain"])


@router.post("/explain/{case_id}", response_model=ExplainResponse)
def explain(case_id: int, db: Session = Depends(get_db),
            current: User = Depends(get_current_user)):
    case = db.query(Case).filter(Case.id == case_id,
                                 Case.user_id == current.id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Caso no encontrado.")

    rec = json.loads(case.recomendacion_json or "{}")
    rec_texto = rules_engine.recommendation_text(rec)

    # El AgenteExplicador (LLM multimodal) + el AgenteValidador (post-control
    # anti-alucinación) corren sobre el caso ya diagnosticado.
    explic, disponible = ma_web.explicar(
        diagnostico_nombre=case.diagnostico,
        confianza=case.confianza,
        severidad=case.severidad,
        zona_gradcam=case.zona_gradcam or "la zona resaltada en el mapa de calor",
        foto_path=case.imagen_path,
        heatmap_path=case.heatmap_path,
        recomendacion=rec,
        recomendacion_texto=rec_texto,
        estado_confianza=case.estado_confianza or "alta",
    )

    # Guardar la explicación en el caso.
    case.explicacion_json = json.dumps(explic, ensure_ascii=False)
    db.commit()

    return ExplainResponse(case_id=case.id, explicacion=Explicacion(**explic),
                           explicacion_disponible=disponible)
