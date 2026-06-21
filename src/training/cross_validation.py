"""Validación cruzada estratificada de 5 folds sobre las 1 200 imágenes.

Con los mejores hiperparámetros de Optuna, entrena un modelo por fold y reporta
media ± desviación estándar de las métricas clave por clase. Lo importante no es
solo el promedio sino la VARIANZA entre folds: si un fold da recall 0.95 en
tizón tardío y otro 0.70, el modelo es frágil.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import LeafDataset, build_loader, subset  # noqa: E402
from src.data.transforms import build_train_transforms, build_eval_transforms  # noqa: E402
from src.evaluation import metrics as M  # noqa: E402
from src.models.architecture import build_model  # noqa: E402
from src.training.trainer import train_model  # noqa: E402


def get_fold_indices(labels, n_folds: int = cfg.N_FOLDS):
    """Devuelve la lista de (train_idx, val_idx) por fold (estratificado)."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cfg.RANDOM_SEED)
    dummy = np.zeros(len(labels))
    return list(skf.split(dummy, labels))


def run_cross_validation(paths, labels, hp: dict, device,
                         n_folds: int = cfg.N_FOLDS, verbose: bool = True) -> dict:
    """Entrena un modelo por fold y agrega métricas (media ± std).

    Devuelve métricas por fold (para boxplot), el agregado y los índices de
    validación de cada fold (para verificar balance 80/clase).
    """
    folds = get_fold_indices(labels, n_folds)

    per_fold: list[dict] = []
    val_indices: list[np.ndarray] = []

    for k, (tr_idx, va_idx) in enumerate(folds):
        val_indices.append(va_idx)
        tr_paths, tr_labels = subset(paths, labels, tr_idx)
        va_paths, va_labels = subset(paths, labels, va_idx)

        train_ds = LeafDataset(tr_paths, tr_labels, transform=build_train_transforms())
        val_ds = LeafDataset(va_paths, va_labels, transform=build_eval_transforms())
        train_loader = build_loader(train_ds, hp["batch_size"], device, shuffle=True)
        val_loader = build_loader(val_ds, hp["batch_size"], device, shuffle=False)

        model = build_model(dropout=hp["dropout"],
                            num_trainable_blocks=hp["num_trainable_blocks"])
        result = train_model(
            model, train_loader, val_loader, device=device,
            lr=hp["lr"], weight_decay=hp["weight_decay"],
            epochs=cfg.CV_EPOCHS, patience=cfg.EARLY_STOP_PATIENCE,
            monitor=cfg.MONITOR_METRIC, verbose=False,
        )
        val = result["val_eval"]
        report = M.full_report(val["y_true"], val["y_pred"])
        fold_metrics = {
            "fold": k,
            "mcc": report["global"]["mcc"],
            "macro_f1": report["global"]["macro_f1"],
            "recall": {c: report["por_clase"][c]["recall"] for c in cfg.CLASSES},
            "precision": {c: report["por_clase"][c]["precision"] for c in cfg.CLASSES},
            "por_clase": report["por_clase"],
        }
        per_fold.append(fold_metrics)
        if verbose:
            print(f"  fold {k+1}/{n_folds} | MCC={fold_metrics['mcc']:.3f} "
                  f"| recall_tardio={fold_metrics['recall']['tizon_tardio']:.3f}")

    aggregate = _aggregate(per_fold)
    return {"por_fold": per_fold, "agregado": aggregate, "val_indices": val_indices}


def _aggregate(per_fold: list[dict]) -> dict:
    """Media ± std de MCC, macro-F1 y recall/precisión por clase entre folds."""
    def mean_std(values):
        arr = np.asarray(values, dtype=np.float64)
        return {"media": float(arr.mean()), "std": float(arr.std())}

    agg = {
        "mcc": mean_std([f["mcc"] for f in per_fold]),
        "macro_f1": mean_std([f["macro_f1"] for f in per_fold]),
        "recall": {c: mean_std([f["recall"][c] for f in per_fold]) for c in cfg.CLASSES},
        "precision": {c: mean_std([f["precision"][c] for f in per_fold]) for c in cfg.CLASSES},
    }
    return agg
