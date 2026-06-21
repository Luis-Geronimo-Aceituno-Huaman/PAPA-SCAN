"""Seguridad: hashing de contraseñas (bcrypt) y tokens JWT."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from app.settings import ACCESS_TOKEN_EXPIRE_MINUTES, JWT_ALGORITHM, SECRET_KEY

# Se usa bcrypt directamente (passlib 1.7.x es incompatible con bcrypt >= 5).
# bcrypt opera sobre los primeros 72 bytes de la contraseña; se trunca acorde.
def _to_bytes(password: str) -> bytes:
    return password.encode("utf-8")[:72]


def hash_password(password: str) -> str:
    return bcrypt.hashpw(_to_bytes(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(_to_bytes(plain), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_access_token(subject: str, extra: dict | None = None) -> str:
    """Genera un JWT firmado con expiración."""
    now = datetime.now(timezone.utc)
    payload = {"sub": subject, "iat": now,
               "exp": now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)}
    if extra:
        payload.update(extra)
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict | None:
    """Decodifica y valida un JWT; devuelve el payload o None si es inválido."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])
    except JWTError:
        return None
