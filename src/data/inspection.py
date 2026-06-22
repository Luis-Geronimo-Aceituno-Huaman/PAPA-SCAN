"""Paso 1 — Inspección y limpieza del dataset original.

Detecta problemas que contaminarían el entrenamiento ANTES de cualquier
transformación: duplicados/casi-duplicados (fuga de datos), imágenes de baja
resolución, imágenes con exceso de fondo no foliar y sesgos de resolución entre
clases. Se ejecuta una sola vez; produce un reporte y una lista de archivos
marcados. No borra nada por sí solo (la decisión queda registrada y es
auditable).
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

from . import roi  # noqa: E402


def average_hash(path: str, hash_size: int = 8) -> int:
    """Hash perceptual aHash de 64 bits para detectar (casi) duplicados."""
    img = Image.open(path).convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    bits = (arr > arr.mean()).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def find_near_duplicates(paths, labels, max_distance: int = 5) -> list[dict]:
    """Encuentra pares de imágenes casi idénticas (hamming <= max_distance).

    Marca con `fuga` los pares que cruzan clases distintas (peligro de fuga).
    """
    hashes = [(p, average_hash(p)) for p in paths]
    label_of = {p: int(l) for p, l in zip(paths, labels)}
    dups: list[dict] = []
    n = len(hashes)
    for i in range(n):
        pi, hi = hashes[i]
        for j in range(i + 1, n):
            pj, hj = hashes[j]
            dist = _hamming(hi, hj)
            if dist <= max_distance:
                dups.append({
                    "a": pi, "b": pj, "distancia": dist,
                    "clase_a": cfg.IDX_TO_CLASS[label_of[pi]],
                    "clase_b": cfg.IDX_TO_CLASS[label_of[pj]],
                    "fuga": label_of[pi] != label_of[pj],
                })
    return dups


def inspect_dataset(paths, labels) -> dict:
    """Inspecciona el dataset y devuelve un reporte estructurado.

    Reporta: baja resolución, exceso de fondo no foliar, estadísticas de tamaño
    por clase y duplicados/casi-duplicados. No modifica archivos.
    """
    low_res: list[str] = []
    high_background: list[dict] = []
    sizes_by_class: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for p, lbl in zip(paths, labels):
        cls = cfg.IDX_TO_CLASS[int(lbl)]
        img = Image.open(p)
        w, h = img.size
        sizes_by_class[cls].append((w, h))

        if min(w, h) < cfg.ROI_MIN_RESOLUTION:
            low_res.append(p)

        img_bgr = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2BGR)
        foliar = roi.foliar_fraction(img_bgr)
        if (1.0 - foliar) > cfg.ROI_MAX_NONFOLIAR_FRAC:
            high_background.append({"path": p, "fraccion_foliar": round(foliar, 3)})

    # Estadísticas de resolución por clase (para detectar sesgo sistemático).
    size_stats: dict[str, dict] = {}
    for cls, sizes in sizes_by_class.items():
        arr = np.asarray(sizes, dtype=np.float32)
        size_stats[cls] = {
            "n": int(len(sizes)),
            "ancho_medio": float(arr[:, 0].mean()),
            "alto_medio": float(arr[:, 1].mean()),
            "ancho_std": float(arr[:, 0].std()),
            "alto_std": float(arr[:, 1].std()),
        }

    duplicates = find_near_duplicates(paths, labels)
    cross_class_leaks = [d for d in duplicates if d["fuga"]]

    return {
        "n_imagenes": int(len(paths)),
        "baja_resolucion": low_res,
        "exceso_fondo": high_background,
        "estadisticas_tamano_por_clase": size_stats,
        "duplicados_total": len(duplicates),
        "duplicados_cruzando_clase": len(cross_class_leaks),
        "duplicados_detalle": duplicates,
        "resumen": {
            "n_baja_resolucion": len(low_res),
            "n_exceso_fondo": len(high_background),
            "n_fuga_potencial": len(cross_class_leaks),
            "limpio": (len(low_res) == 0 and len(cross_class_leaks) == 0),
        },
    }
