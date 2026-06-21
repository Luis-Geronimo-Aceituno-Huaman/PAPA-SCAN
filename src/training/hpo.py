"""Búsqueda de hiperparámetros con Optuna (sampler TPE + pruning).

Cada trial es un entrenamiento corto sobre el split estratificado 80/20 (solo
para HPO). El TPE aprende de trials anteriores; el MedianPruner corta trials
que van claramente peor a mitad de camino. Se maximiza el MCC de validación.

Los trials se corren EN SECUENCIA (una sola GPU): correr varios en paralelo
haría que compitan por memoria y resultaría más lento.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.dataset import LeafDataset, build_loader, subset  # noqa: E402
from src.data.transforms import build_train_transforms, build_eval_transforms  # noqa: E402
from src.models.architecture import build_model  # noqa: E402
from src.training.trainer import train_model  # noqa: E402


def _make_loaders(tr_paths, tr_labels, va_paths, va_labels, batch_size, device):
    """Construye los DataLoaders de train (con augmentación) y val (determinístico)."""
    train_ds = LeafDataset(tr_paths, tr_labels, transform=build_train_transforms())
    val_ds = LeafDataset(va_paths, va_labels, transform=build_eval_transforms())
    train_loader = build_loader(train_ds, batch_size, device, shuffle=True)
    val_loader = build_loader(val_ds, batch_size, device, shuffle=False)
    return train_loader, val_loader


def _pruning_callback(trial):
    """Callback de época: reporta MCC intermedio y aborta si Optuna lo decide."""
    def cb(epoch, val):
        trial.report(val["mcc"], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return cb


def make_objective(paths, labels, device):
    """Crea la función objetivo de Optuna sobre el split 80/20 estratificado."""
    idx = np.arange(len(labels))
    tr_idx, va_idx = train_test_split(
        idx, test_size=cfg.HPO_VAL_SPLIT, stratify=labels, random_state=cfg.RANDOM_SEED
    )
    tr_paths, tr_labels = subset(paths, labels, tr_idx)
    va_paths, va_labels = subset(paths, labels, va_idx)

    def objective(trial: optuna.Trial) -> float:
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.1, 0.6)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        num_trainable_blocks = trial.suggest_int("num_trainable_blocks", 0, 5)

        train_loader, val_loader = _make_loaders(
            tr_paths, tr_labels, va_paths, va_labels, batch_size, device
        )
        model = build_model(dropout=dropout, num_trainable_blocks=num_trainable_blocks)
        result = train_model(
            model, train_loader, val_loader,
            device=device, lr=lr, weight_decay=weight_decay,
            epochs=cfg.HPO_EPOCHS, patience=cfg.HPO_EPOCHS,  # sin early-stop interno en HPO
            monitor=cfg.MONITOR_METRIC,
            epoch_callback=_pruning_callback(trial),
        )
        return result["best_metric"]

    return objective


def run_hpo(paths, labels, device, n_trials: int = cfg.N_OPTUNA_TRIALS,
            study_name: str = "agrivision_hpo", storage: str | None = None) -> dict:
    """Ejecuta la búsqueda y devuelve los mejores hiperparámetros y el estudio."""
    sampler = optuna.samplers.TPESampler(seed=cfg.RANDOM_SEED)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    study = optuna.create_study(
        direction="maximize", sampler=sampler, pruner=pruner,
        study_name=study_name, storage=storage,
        load_if_exists=storage is not None,
    )
    objective = make_objective(paths, labels, device)
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)

    return {
        "best_params": study.best_params,
        "best_value": float(study.best_value),
        "n_trials": len(study.trials),
        "n_pruned": len([t for t in study.trials
                         if t.state == optuna.trial.TrialState.PRUNED]),
        "study": study,
    }
