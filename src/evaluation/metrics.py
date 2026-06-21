"""Métricas de evaluación con costo asimétrico de errores.

Diseñadas según el prompt:
- `tizon_tardio`: principal = recall (mín 0.90), resumen = F2 (penaliza FN).
- `tizon_temprano`: principal = recall (mín 0.80), resumen = F1.
- `sana`: principal = precisión (evitar FP).
- Globales: MCC (principal), Macro-F1, Weighted-F1, matriz de confusión.

Accuracy global NO se reporta como principal por ser engañosa con costos
asimétricos.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    fbeta_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

_LABELS = list(range(len(cfg.CLASSES)))  # [0, 1, 2]


def per_class_metrics(y_true, y_pred) -> dict:
    """Precisión, recall, F1 y F2 por clase (clave = nombre de clase)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, zero_division=0
    )
    f2 = fbeta_score(y_true, y_pred, beta=2, labels=_LABELS, average=None, zero_division=0)

    out: dict[str, dict] = {}
    for idx, cls in cfg.IDX_TO_CLASS.items():
        out[cls] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1[idx]),
            "f2": float(f2[idx]),
            "support": int(support[idx]),
        }
    return out


def global_metrics(y_true, y_pred) -> dict:
    """MCC, Macro-F1, Weighted-F1 y accuracy (esta última solo informativa)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    _, _, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=_LABELS, average="weighted", zero_division=0
    )
    return {
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "accuracy": float((y_true == y_pred).mean()),  # informativa, no principal
    }


def confusion(y_true, y_pred, normalize: str | None = None) -> np.ndarray:
    """Matriz de confusión (filas=verdadero, columnas=predicho).

    normalize: None | 'true' (normaliza por fila) | 'pred' | 'all'.
    """
    return confusion_matrix(y_true, y_pred, labels=_LABELS, normalize=normalize)


def primary_class_value(per_class: dict, cls: str) -> float:
    """Valor de la métrica PRINCIPAL definida para una clase."""
    metric_name = cfg.CLASS_METRIC_TARGETS[cls]["metrica_principal"]
    return per_class[cls][metric_name]


def production_check(per_class: dict) -> dict:
    """Verifica si cada clase cumple su mínimo aceptable.

    Devuelve {clase: {metrica, valor, minimo, cumple}} y un flag global
    `listo_produccion` (todas las clases con mínimo definido lo cumplen).
    """
    result: dict = {}
    ready = True
    for cls, target in cfg.CLASS_METRIC_TARGETS.items():
        metric_name = target["metrica_principal"]
        minimo = target["minimo"]
        valor = per_class[cls][metric_name]
        cumple = True if minimo is None else bool(valor >= minimo)
        if minimo is not None and not cumple:
            ready = False
        result[cls] = {
            "metrica": metric_name,
            "valor": float(valor),
            "minimo": minimo,
            "cumple": cumple,
        }
    return {"por_clase": result, "listo_produccion": ready}


def full_report(y_true, y_pred) -> dict:
    """Reporte completo: por clase + globales + chequeo de producción."""
    pc = per_class_metrics(y_true, y_pred)
    return {
        "por_clase": pc,
        "global": global_metrics(y_true, y_pred),
        "produccion": production_check(pc),
    }


def optuna_objective_value(y_true, y_pred) -> float:
    """Valor a maximizar en HPO: MCC (resumen principal del sistema).

    El MCC integra VP/VN/FP/FN de todas las clases sin sesgo por clase
    dominante, alineado con el costo asimétrico declarado en el prompt.
    """
    return float(matthews_corrcoef(np.asarray(y_true), np.asarray(y_pred)))
