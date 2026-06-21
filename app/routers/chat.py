"""Router de preguntas de seguimiento sobre un caso (chat, sin JSON)."""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.database import get_db
from app.deps import get_current_user
from app.models import Case, ChatMessage, User
from app.schemas import ChatRequest, ChatResponse
from app.services import llm_client

router = APIRouter(prefix="/api", tags=["chat"])


def _case_context(case: Case) -> str:
    """Arma el contexto textual del caso para el LLM."""
    rec = json.loads(case.recomendacion_json or "{}")
    controles = " ".join(f"{c['titulo']}: {c['contenido']}"
                         for c in rec.get("controles", []))
    return (
        f"Diagnóstico: {case.diagnostico} (confianza {case.confianza*100:.0f} %). "
        f"Severidad: {case.severidad}. "
        f"Recomendación del motor de reglas: {controles}"
    )


_GENERAL_CONTEXT = (
    "Consulta general sobre el cultivo de papa y sus enfermedades (tizón tardío, "
    "tizón temprano, hojas sanas). El usuario aún no ha analizado ninguna hoja."
)
_MAX_FREE_HISTORY = 20  # turnos máximos aceptados del cliente en chat libre


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db),
         current: User = Depends(get_current_user)):
    # --- Chat libre (pestaña Asistente): sin caso, historial del cliente, no se persiste. ---
    if payload.case_id is None:
        historial = [{"role": t.role, "content": t.content}
                     for t in payload.historial[-_MAX_FREE_HISTORY:]]
        respuesta, disponible = llm_client.chat(
            historial, _GENERAL_CONTEXT, payload.mensaje,
            system_prompt=llm_client.GENERAL_CHAT_SYSTEM_PROMPT)
        return ChatResponse(respuesta=respuesta, disponible=disponible)

    # --- Chat de un caso diagnosticado: contexto e historial desde la BD, se persiste. ---
    case = db.query(Case).filter(Case.id == payload.case_id,
                                 Case.user_id == current.id).first()
    if case is None:
        raise HTTPException(status_code=404, detail="Caso no encontrado.")

    historial = [{"role": m.role, "content": m.content} for m in case.messages]
    contexto = _case_context(case)
    respuesta, disponible = llm_client.chat(historial, contexto, payload.mensaje)

    # Persistir la conversación (pregunta + respuesta).
    db.add(ChatMessage(case_id=case.id, role="user", content=payload.mensaje))
    db.add(ChatMessage(case_id=case.id, role="assistant", content=respuesta))
    db.commit()

    return ChatResponse(respuesta=respuesta, disponible=disponible)
