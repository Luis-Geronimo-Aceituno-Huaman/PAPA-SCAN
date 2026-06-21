"""Dependencias compartidas de FastAPI (usuario autenticado)."""
from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.security import decode_access_token

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

_CREDENTIALS_EXC = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Credenciales inválidas o token expirado",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)) -> User:
    """Valida el JWT y devuelve el usuario; lanza 401 si no es válido."""
    payload = decode_access_token(token)
    if not payload or "sub" not in payload:
        raise _CREDENTIALS_EXC
    user = db.query(User).filter(User.username == payload["sub"]).first()
    if user is None:
        raise _CREDENTIALS_EXC
    return user
