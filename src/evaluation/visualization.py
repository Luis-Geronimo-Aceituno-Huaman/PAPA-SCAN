"""Los 5 gráficos de evaluación, guardados como .png en resultados/.

Cada gráfico responde una pregunta distinta (necesarios y suficientes):
- curvas_entrenamiento : ¿sobreajuste/subentrenamiento?
- matriz_confusion     : ¿hay celdas tizon_tardio -> sana? (alerta crítica)
- sensibilidades       : ¿cumple los mínimos operativos por clase?
- boxplot_cv           : ¿es estable o frágil entre folds?
- curvas_roc           : apoyo a la elección de umbral por clase

Ninguna función llama a plt.show(); todas guardan a disco con el prefijo de la
versión del modelo y devuelven la ruta como string.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend sin ventana (no requiere display)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402


def _path(version_id: str, name: str) -> str:
    cfg.RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
    return str(cfg.RESULTADOS_DIR / f"{version_id}_{name}.png")


def _labels_es(classes=cfg.CLASSES):
    return [cfg.CLASS_LABELS_ES[c] for c in classes]


def plot_training_curves(history: dict, version_id: str) -> str:
    """curvas_entrenamiento.png — loss y métricas de train vs validación."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(epochs, history["train_loss"], "o-", label="train loss")
    ax1.plot(epochs, history["val_loss"], "s-", label="val loss")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss")
    ax1.set_title("Curva de pérdida"); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history["val_mcc"], "o-", label="val MCC")
    ax2.plot(epochs, history["val_macro_f1"], "s-", label="val macro-F1")
    ax2.plot(epochs, history["val_recall_tardio"], "^-", label="val recall tizón tardío")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Métrica")
    ax2.set_title("Métricas de validación"); ax2.legend(); ax2.grid(alpha=0.3)

    fig.suptitle(f"{version_id} — Curvas de entrenamiento")
    fig.tight_layout()
    out = _path(version_id, "curvas_entrenamiento")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def plot_confusion_matrix(cm: np.ndarray, version_id: str) -> str:
    """matriz_confusion.png — heatmap normalizado con énfasis en tizon_tardio->sana."""
    labels = _labels_es()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=1,
                xticklabels=labels, yticklabels=labels, ax=ax,
                cbar_kws={"label": "Proporción (normalizado por fila)"})
    ax.set_xlabel("Predicho"); ax.set_ylabel("Verdadero")
    ax.set_title(f"{version_id} — Matriz de confusión")

    # Resaltar la celda crítica: tizon_tardio (verdadero) -> sana (predicho).
    i = cfg.CLASS_TO_IDX["tizon_tardio"]; j = cfg.CLASS_TO_IDX["sana"]
    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False, edgecolor="red", lw=3))

    fig.tight_layout()
    out = _path(version_id, "matriz_confusion")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def plot_sensitivities(per_class: dict, version_id: str) -> str:
    """sensibilidades_y_precision.png — barras de las 3 sensibilidades + mínimos."""
    specs = [
        ("tizon_tardio", "recall", "S1: recall\ntizón tardío"),
        ("tizon_temprano", "recall", "S2: recall\ntizón temprano"),
        ("sana", "precision", "S3: precisión\nsana"),
    ]
    values = [per_class[c][m] for c, m, _ in specs]
    minimos = [cfg.CLASS_METRIC_TARGETS[c]["minimo"] for c, _, _ in specs]
    names = [s for _, _, s in specs]
    colors = ["#c0392b", "#e67e22", "#27ae60"]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, values, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontweight="bold")
    # Líneas punteadas de mínimo aceptable por clase.
    for i, m in enumerate(minimos):
        if m is not None:
            ax.hlines(m, i - 0.4, i + 0.4, colors="black", linestyles="dashed")
            ax.text(i, m + 0.005, f"mín {m:.2f}", ha="center", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Valor de la métrica")
    ax.set_title(f"{version_id} — Sensibilidades y precisión por clase")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = _path(version_id, "sensibilidades_y_precision")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def plot_cv_boxplot(per_fold: list[dict], version_id: str) -> str:
    """boxplot_cv.png — distribución del recall por clase y del MCC entre folds."""
    data, names = [], []
    for c in cfg.CLASSES:
        data.append([f["recall"][c] for f in per_fold])
        names.append(f"recall\n{cfg.CLASS_LABELS_ES[c]}")
    data.append([f["mcc"] for f in per_fold])
    names.append("MCC")

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, labels=names, patch_artist=True, showmeans=True)
    palette = ["#27ae60", "#c0392b", "#e67e22", "#2980b9"]
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color); patch.set_alpha(0.6)
    # superponer puntos de cada fold
    for i, vals in enumerate(data, start=1):
        ax.scatter(np.full(len(vals), i), vals, color="black", s=15, zorder=3, alpha=0.6)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Valor")
    ax.set_title(f"{version_id} — Estabilidad entre los {len(per_fold)} folds")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out = _path(version_id, "boxplot_cv")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out


def plot_roc_curves(roc_data: dict, version_id: str) -> str:
    """curvas_roc.png — ROC one-vs-rest por clase, con AUC y umbral elegido."""
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"sana": "#27ae60", "tizon_tardio": "#c0392b", "tizon_temprano": "#e67e22"}

    for cls in cfg.CLASSES:
        d = roc_data[cls]
        emphasis = cls == "tizon_tardio"
        ax.plot(d["fpr"], d["tpr"], color=colors[cls],
                lw=3.0 if emphasis else 1.8,
                label=f"{cfg.CLASS_LABELS_ES[cls]} (AUC={d['auc']:.3f})")
        # Marcador sobre el punto de operación (umbral elegido).
        ax.scatter([d["op_fpr"]], [d["op_tpr"]], color=colors[cls],
                   s=90, edgecolor="black", zorder=5,
                   marker="*" if emphasis else "o")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="azar")
    ax.set_xlabel("Tasa de falsos positivos (FPR)")
    ax.set_ylabel("Tasa de verdaderos positivos (TPR / recall)")
    ax.set_title(f"{version_id} — Curvas ROC one-vs-rest")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)

    fig.tight_layout()
    out = _path(version_id, "curvas_roc")
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
    return out
