"""Router de diagnóstico: recibe una imagen y orquesta el flujo completo."""
from __future__ import annotations

import json
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Case, User
from app.schemas import ControlItem, DiagnoseResponse, Recomendacion
from app.services import orchestrator
from app.settings import STORAGE_DIR

router = APIRouter(prefix="/api", tags=["diagnose"])

_MAX_BYTES = 12 * 1024 * 1024  # 12 MB


def media_url(path: str) -> str:
    """Convierte una ruta dentro de STORAGE_DIR en una URL servible (/media/...)."""
    rel = Path(path).resolve().relative_to(STORAGE_DIR.resolve())
    return "/media/" + str(rel).replace("\\", "/")


def _build_recomendacion(rec: dict) -> Recomendacion:
    return Recomendacion(
        enfermedad=rec["enfermedad"],
        severidad=rec["severidad"],
        estadio_fenologico=rec.get("estadio_fenologico", ""),
        region=rec.get("region", ""),
        controles=[ControlItem(titulo=c["titulo"], contenido=c["contenido"],
                               fuente=c.get("fuente", "")) for c in rec["controles"]],
        trazabilidad=rec.get("trazabilidad", ""),
    )


@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(file: UploadFile = File(...),
                   db: Session = Depends(get_db),
                   current: User = Depends(get_current_user)):
    data = await file.read()
    if len(data) > _MAX_BYTES:
        raise HTTPException(status_code=413, detail="La imagen supera el tamaño máximo (12 MB).")
    if not data:
        raise HTTPException(status_code=400, detail="Archivo vacío.")

    prefix = uuid.uuid4().hex[:12]
    try:
        # Sin LLM: diagnóstico + Grad-CAM + reglas (rápido). La explicación del
        # LLM es una capa opcional bajo demanda (POST /api/explain/{case_id}).
        result = orchestrator.run(data, prefix, with_llm=False)
    except ValueError as e:  # imagen inválida
        raise HTTPException(status_code=400, detail=str(e))

    # Persistir el caso en el historial.
    case = Case(
        user_id=current.id,
        diagnostico=result["diagnostico"],
        confianza=result["confianza"],
        estado_confianza=result["estado_confianza"],
        severidad=result["severidad"],
        zona_gradcam=result["zona_gradcam"],
        imagen_path=result["foto_path"],
        masked_path=result["masked_path"],
        heatmap_path=result["heatmap_path"],
        overlay_path=result["overlay_path"],
        probabilidades_json=json.dumps(result["probabilidades"], ensure_ascii=False),
        recomendacion_json=json.dumps(result["recomendacion"], ensure_ascii=False),
        version_modelo=result["version_modelo"],
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    return DiagnoseResponse(
        case_id=case.id,
        diagnostico=result["diagnostico"],
        confianza=result["confianza"],
        estado_confianza=result["estado_confianza"],
        probabilidades=result["probabilidades"],
        severidad=result["severidad"],
        imagen_url=media_url(result["foto_path"]),
        masked_url=media_url(result["masked_path"]),
        heatmap_url=media_url(result["heatmap_path"]),
        overlay_url=media_url(result["overlay_path"]),
        capa_usada=result["capa_usada"],
        alerta_sesgo=result["alerta_sesgo"],
        zona_gradcam=result["zona_gradcam"],
        recomendacion=_build_recomendacion(result["recomendacion"]),
        version_modelo=result["version_modelo"],
    )
