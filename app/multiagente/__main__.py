"""Permite ejecutar el sistema multiagente con `python -m app.multiagente`.

Equivale a `python -m app.multiagente.run` (ver run.py para las opciones).
"""
from app.multiagente.run import main

if __name__ == "__main__":
    raise SystemExit(main())
