"""Pasos 3 y 4 — Preprocesamiento y augmentación.

Coherencia train/inferencia (Paso 3): el redimensionado y la normalización son
IDÉNTICOS en ambos pipelines. El recorte ROI (Paso 2), si está habilitado, va
primero y es determinístico, así que también es idéntico.

Augmentación (Paso 4): solo en entrenamiento. Prioriza geometría y permite
brillo/contraste leve; saturación leve (0.1) y hue=0.0 (modificar el tono está
prohibido porque el color es la señal diagnóstica). ToTensor/Normalize van
siempre al final.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

from .roi import maybe_apply_roi  # noqa: E402  (recorte y/o enmascarado de fondo)


def build_train_transforms(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    """Pipeline de entrenamiento (Paso 2 opcional + Paso 4 + Paso 3)."""
    return transforms.Compose([
        transforms.Lambda(maybe_apply_roi),                    # ROI determinístico
        transforms.RandomResizedCrop(
            image_size, scale=cfg.AUG_RRC_SCALE, ratio=cfg.AUG_RRC_RATIO
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=cfg.AUG_ROTATION_DEG),
        transforms.ColorJitter(
            brightness=cfg.AUG_JITTER_BRIGHTNESS,
            contrast=cfg.AUG_JITTER_CONTRAST,
            saturation=cfg.AUG_JITTER_SATURATION,
            hue=cfg.AUG_JITTER_HUE,
        ),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])


def build_eval_transforms(image_size: int = cfg.IMAGE_SIZE) -> transforms.Compose:
    """Pipeline determinístico para validación / test / inferencia.

    EXACTAMENTE: (ROI opcional) → Resize(224×224) → ToTensor → Normalize.
    Sin augmentación, sin recortes aleatorios, sin flips.
    """
    return transforms.Compose([
        transforms.Lambda(maybe_apply_roi),                    # ROI determinístico
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(cfg.IMAGENET_MEAN, cfg.IMAGENET_STD),
    ])


# La inferencia usa exactamente el mismo pipeline determinístico que validación.
build_inference_transforms = build_eval_transforms


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Revierte la normalización ImageNet para overlay/visualización ([0,1]).

    Acepta CHW o BCHW y devuelve la misma forma con valores en [0, 1].
    """
    mean = torch.tensor(cfg.IMAGENET_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(cfg.IMAGENET_STD, device=tensor.device).view(-1, 1, 1)
    if tensor.dim() == 4:
        mean = mean.unsqueeze(0)
        std = std.unsqueeze(0)
    return (tensor * std + mean).clamp(0, 1)
