"""Calibración de confianza y selección de umbrales por clase.

Dos piezas, ambas ajustadas sobre un conjunto de validación (nunca el test):

1. **Temperature scaling**: aprende un único escalar T que divide los logits
   antes del softmax, de modo que la confianza refleje la probabilidad real de
   acierto (el softmax crudo suele estar sobreconfiado). T se ajusta
   minimizando la NLL con LBFGS.

2. **Umbral de confianza por clase**: a partir de la curva ROC one-vs-rest
   (sobre probabilidades ya calibradas) se elige el punto de operación por
   clase. El umbral se usa en inferencia: si la prob de la clase ganadora cae
   por debajo, el resultado se marca `baja_confianza`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402


# Rango sensato para la temperatura. Evita el caso degenerado T→0 que aparece
# cuando el conjunto de calibración es perfectamente separable (la NLL empuja T
# a cero, sobre-afilando las probabilidades). Calibrar nunca debería volver al
# modelo arbitrariamente más confiado.
T_MIN, T_MAX = 0.5, 5.0


def fit_temperature(logits: np.ndarray, labels: np.ndarray, max_iter: int = 100) -> float:
    """Ajusta la temperatura T minimizando la NLL sobre validación.

    Devuelve T acotada a [T_MIN, T_MAX]. Se aplica como softmax(logits / T).
    """
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T) > 0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([log_T], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / log_T.exp(), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    T = float(log_T.exp().item())
    return float(min(max(T, T_MIN), T_MAX))


def calibrated_probs(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Aplica temperature scaling y devuelve probabilidades (softmax)."""
    logits_t = torch.tensor(logits, dtype=torch.float32) / float(temperature)
    return torch.softmax(logits_t, dim=1).numpy()


def select_thresholds(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Elige el umbral de confianza por clase desde la ROC one-vs-rest.

    Punto de operación = índice de Youden (máximo tpr - fpr). Para las clases
    con recall mínimo exigido, se respeta ese mínimo si el punto de Youden no lo
    alcanza (se baja el umbral hasta cumplir el recall objetivo).

    Devuelve {clase: {umbral, auc, tpr, fpr, recall_objetivo}} y, por clase, los
    puntos de la curva ROC (fpr, tpr, thresholds) para el gráfico.
    """
    y_true = np.asarray(y_true)
    result: dict = {}
    roc_data: dict = {}

    for cls, idx in cfg.CLASS_TO_IDX.items():
        binary = (y_true == idx).astype(int)
        scores = probs[:, idx]
        fpr, tpr, thr = roc_curve(binary, scores)
        auc = float(roc_auc_score(binary, scores))

        # Punto de Youden (mejor equilibrio tpr/fpr).
        j = tpr - fpr
        best = int(np.argmax(j))
        threshold = float(thr[best])
        op_tpr, op_fpr = float(tpr[best]), float(fpr[best])

        # Respetar el recall mínimo de la clase si aplica.
        minimo = cfg.CLASS_METRIC_TARGETS[cls]["minimo"]
        metric = cfg.CLASS_METRIC_TARGETS[cls]["metrica_principal"]
        if minimo is not None and metric == "recall" and op_tpr < minimo:
            # Bajar el umbral hasta el punto donde el recall (tpr) alcanza el mínimo.
            feasible = np.where(tpr >= minimo)[0]
            if len(feasible) > 0:
                pick = feasible[0]  # menor umbral que cumple (thr es decreciente)
                threshold = float(thr[pick])
                op_tpr, op_fpr = float(tpr[pick]), float(fpr[pick])

        # roc_curve puede devolver un umbral inicial > 1 (inf); acotar a [0, 1].
        threshold = float(min(max(threshold, 0.0), 1.0))

        result[cls] = {
            "umbral": threshold,
            "auc": auc,
            "tpr_operacion": op_tpr,
            "fpr_operacion": op_fpr,
            "recall_objetivo": minimo,
        }
        roc_data[cls] = {
            "fpr": [float(x) for x in fpr],
            "tpr": [float(x) for x in tpr],
            "thresholds": [float(min(max(x, 0.0), 1.0)) for x in thr],
            "auc": auc,
            "umbral_elegido": threshold,
            "op_fpr": op_fpr,
            "op_tpr": op_tpr,
        }

    return {"umbrales": result, "roc": roc_data}


def calibrate(logits: np.ndarray, labels: np.ndarray) -> dict:
    """Pipeline de calibración completo sobre validación.

    Devuelve temperatura, umbrales por clase y datos ROC, listo para guardar en
    los metadatos del modelo.
    """
    temperature = fit_temperature(logits, labels)
    probs = calibrated_probs(logits, temperature)
    thr = select_thresholds(labels, probs)
    return {
        "temperatura": temperature,
        "umbrales_por_clase": {c: thr["umbrales"][c]["umbral"] for c in cfg.CLASSES},
        "umbrales_detalle": thr["umbrales"],
        "roc": thr["roc"],
    }
