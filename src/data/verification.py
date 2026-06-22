"""Paso 5 — Verificaciones de consistencia antes de entrenar.

Comprobaciones programáticas que deben pasar ANTES del primer entrenamiento:
1. Guardar una grilla de un batch aumentado (auditar que la augmentación no
   destruye las señales diagnósticas).
2. Confirmar que val/test NO aplican augmentación (transform determinística).
3. Confirmar que la normalización deja el primer batch con media≈0, std≈1.
4. Verificar balance exacto por clase en cada fold (80/clase con 5 folds).

Estas funciones devuelven datos/flags; no imprimen ni hacen plt.show().
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import RandomResizedCrop, RandomRotation, ColorJitter
from torchvision.utils import make_grid, save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

from .transforms import denormalize  # noqa: E402

# Transformaciones aleatorias que NUNCA deben estar en val/test.
_RANDOM_TFMS = (RandomResizedCrop, RandomRotation, ColorJitter)


def save_augmented_batch_grid(loader, out_path: str | Path, n: int = 16) -> str:
    """Guarda una grilla de `n` imágenes aumentadas (des-normalizadas) a disco.

    Verificación visual #1. Devuelve la ruta como string. No llama a plt.show().
    """
    images, _ = next(iter(loader))
    images = images[:n]
    grid = make_grid(denormalize(images), nrow=4, padding=2)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, str(out_path))
    return str(out_path)


def eval_transform_is_deterministic(transform) -> bool:
    """Verificación #2: el transform de val/test no contiene aleatoriedad."""
    members = getattr(transform, "transforms", [transform])
    return not any(isinstance(t, _RANDOM_TFMS) for t in members)


def normalization_stats(loader) -> dict:
    """Verificación #3: media/std por canal del primer batch post-normalización.

    Deben estar cercanas a (0, 1). Devuelve los valores y un flag `ok`.
    """
    images, _ = next(iter(loader))
    mean = images.mean(dim=(0, 2, 3))
    std = images.std(dim=(0, 2, 3))
    ok = bool((mean.abs() < 0.6).all() and ((std > 0.5) & (std < 1.6)).all())
    return {
        "media_por_canal": [float(x) for x in mean],
        "std_por_canal": [float(x) for x in std],
        "ok": ok,
    }


def verify_fold_balance(labels, fold_indices, expected_per_class: int = 80) -> dict:
    """Verificación #4: cada fold tiene exactamente `expected_per_class` por clase.

    fold_indices : lista de arrays de índices de validación, uno por fold.
    Devuelve por fold el conteo por clase y un flag global `balanceado`.
    """
    labels = np.asarray(labels)
    per_fold = []
    balanced = True
    for k, idx in enumerate(fold_indices):
        counts = Counter(int(labels[i]) for i in idx)
        by_class = {cfg.IDX_TO_CLASS[c]: counts.get(c, 0) for c in range(len(cfg.CLASSES))}
        ok = all(v == expected_per_class for v in by_class.values())
        balanced = balanced and ok
        per_fold.append({"fold": k, "por_clase": by_class, "ok": ok})
    return {"por_fold": per_fold, "balanceado": balanced}
