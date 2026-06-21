"""Bucle de entrenamiento con precisión mixta (AMP) y early stopping.

Desacoplado de Optuna: para el pruning, `train_model` acepta un callback
`epoch_callback(epoch, val_metrics)` que el módulo de HPO usa para reportar la
métrica intermedia y abortar el trial (lanzando la excepción que decida).
"""
from __future__ import annotations

import copy
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation import metrics as M  # noqa: E402
from src.utils.helpers import make_amp  # noqa: E402


@dataclass
class Hyperparams:
    """Hiperparámetros optimizados por Optuna."""
    lr: float = 1e-3
    batch_size: int = 32
    dropout: float = 0.3
    weight_decay: float = 1e-4
    num_trainable_blocks: int = 2

    def to_dict(self) -> dict:
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "dropout": self.dropout,
            "weight_decay": self.weight_decay,
            "num_trainable_blocks": self.num_trainable_blocks,
        }


def build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    """AdamW solo sobre los parámetros entrenables (backbone descongelado + head)."""
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate(model, loader, device, autocast, criterion=None) -> dict:
    """Evalúa el modelo y devuelve loss, etiquetas, predicciones, probabilidades y logits."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with autocast():
            logits = model(images)
            if criterion is not None:
                loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                n += images.size(0)
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels)
    probs = torch.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)

    y_true = labels.numpy()
    y_pred = preds.numpy()
    report = {
        "loss": (total_loss / n) if n else None,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": probs.numpy(),
        "logits": logits.numpy(),
        "mcc": M.optuna_objective_value(y_true, y_pred),
        "macro_f1": M.global_metrics(y_true, y_pred)["macro_f1"],
        "recall_tardio": M.per_class_metrics(y_true, y_pred)["tizon_tardio"]["recall"],
    }
    return report


def train_one_epoch(model, loader, optimizer, criterion, device, scaler, autocast) -> float:
    """Una época de entrenamiento. Devuelve la loss media."""
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast():
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.detach().item() * images.size(0)
        n += images.size(0)
    return total_loss / n


def train_model(
    model,
    train_loader,
    val_loader,
    *,
    device,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int = 5,
    monitor: str = "mcc",
    use_amp: bool = True,
    epoch_callback=None,
    verbose: bool = False,
) -> dict:
    """Entrena con early stopping sobre `monitor` (val). Devuelve historial y mejor estado.

    epoch_callback(epoch:int, val_metrics:dict) -> None : se llama al final de
        cada época; puede lanzar una excepción para abortar (pruning de Optuna).
    """
    model.to(device)
    scaler, autocast = make_amp(device, enabled=use_amp)
    criterion = nn.CrossEntropyLoss()  # dataset balanceado (400/clase)
    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {"train_loss": [], "val_loss": [], "val_mcc": [],
               "val_macro_f1": [], "val_recall_tardio": []}
    best_metric = -np.inf
    best_state = copy.deepcopy(model.state_dict())
    best_val = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler, autocast
        )
        val = evaluate(model, val_loader, device, autocast, criterion)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mcc"].append(val["mcc"])
        history["val_macro_f1"].append(val["macro_f1"])
        history["val_recall_tardio"].append(val["recall_tardio"])

        current = val[monitor]
        if verbose:
            print(f"  época {epoch+1}/{epochs} | train_loss={train_loss:.4f} "
                  f"| val_loss={val['loss']:.4f} | {monitor}={current:.4f}")

        if epoch_callback is not None:
            epoch_callback(epoch, val)  # puede lanzar TrialPruned

        if current > best_metric:
            best_metric = current
            best_state = copy.deepcopy(model.state_dict())
            best_val = val
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if verbose:
                    print(f"  early stopping en época {epoch+1}")
                break

    model.load_state_dict(best_state)
    return {
        "history": history,
        "best_state": best_state,
        "best_metric": float(best_metric),
        "monitor": monitor,
        "val_eval": best_val,
    }
