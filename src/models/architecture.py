"""Definición de la CNN (transfer learning) y localización de la capa Grad-CAM.

Backbone liviano (EfficientNet-B0 o MobileNetV2) apropiado para 1 200 imágenes
y una GPU de 8 GB. Se congela todo el backbone y se descongelan las últimas N
"capas" (bloques) según el hiperparámetro `num_trainable_blocks` (0–5). El
clasificador final siempre es entrenable.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402


def build_model(
    num_classes: int = len(cfg.CLASSES),
    backbone: str = cfg.BACKBONE,
    dropout: float = 0.3,
    num_trainable_blocks: int = 2,
    pretrained: bool = True,
) -> nn.Module:
    """Construye la CNN con transfer learning.

    num_trainable_blocks : nº de bloques finales del backbone a descongelar
        (0–5). El resto del backbone queda congelado; el clasificador siempre
        entrena.
    """
    num_trainable_blocks = int(max(0, min(5, num_trainable_blocks)))

    if backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes),
        )
        feature_blocks = model.features  # Sequential de 9 bloques (0..8)

    elif backbone == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
        feature_blocks = model.features

    else:
        raise ValueError(f"Backbone no soportado: {backbone}")

    # 1) Congelar todo el backbone.
    for p in feature_blocks.parameters():
        p.requires_grad = False

    # 2) Descongelar los últimos N bloques.
    if num_trainable_blocks > 0:
        for block in feature_blocks[-num_trainable_blocks:]:
            for p in block.parameters():
                p.requires_grad = True

    return model


def get_gradcam_layer(model: nn.Module) -> tuple[nn.Module, str]:
    """Devuelve (módulo, nombre) de la última capa convolucional para Grad-CAM.

    Para EfficientNet-B0 y MobileNetV2 es el último bloque de `features`.
    """
    layer = model.features[-1]
    return layer, "features[-1]"


def trainable_summary(model: nn.Module) -> dict:
    """Resumen de parámetros entrenables vs totales (para logs)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": int(total),
        "trainable_params": int(trainable),
        "trainable_pct": round(100.0 * trainable / total, 2),
    }
