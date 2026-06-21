"""Grad-CAM calculado en el MISMO forward/backward pass que la predicción.

Registra hooks sobre la última capa convolucional para capturar activaciones
(forward) y gradientes (backward). El método principal hace un único forward,
retropropaga el score de la clase objetivo y construye el mapa de calor a partir
de las activaciones ponderadas por el gradiente medio. Así el diagnóstico y su
explicación provienen exactamente del mismo cómputo (sin inconsistencias).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

from src.data.transforms import denormalize  # noqa: E402


class GradCAM:
    """Grad-CAM con hooks persistentes sobre una capa objetivo."""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fh = target_layer.register_forward_hook(self._save_activation)
        self._bh = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def __call__(self, input_tensor: torch.Tensor, class_idx: int | None = None):
        """Forward + backward del score de `class_idx` (o la clase predicha).

        Devuelve (logits.detach() [1, C], cam [H, W] en [0, 1]).
        El input debe ser un batch de tamaño 1.
        """
        self.model.zero_grad(set_to_none=True)
        # Forzar la construcción del grafo aunque el backbone esté congelado.
        input_tensor = input_tensor.clone().requires_grad_(True)

        logits = self.model(input_tensor)            # [1, C]
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        score = logits[0, class_idx]
        score.backward()

        # Ponderación: promedio espacial del gradiente por canal (alpha_k).
        grads = self.gradients                       # [1, K, h, w]
        acts = self.activations                      # [1, K, h, w]
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)  # [1, 1, h, w]
        cam = F.relu(cam)

        # Reescalar al tamaño de entrada y normalizar a [0, 1].
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().float()
        cam -= cam.min()
        max_v = cam.max()
        if max_v > 0:
            cam /= max_v
        return logits.detach(), cam.cpu().numpy(), class_idx

    def remove(self):
        """Quita los hooks (llamar al terminar de usar el objeto)."""
        self._fh.remove()
        self._bh.remove()


def heatmap_rgb(cam: np.ndarray) -> np.ndarray:
    """Convierte el CAM [0,1] en un heatmap RGB uint8 (colormap JET)."""
    cam_uint8 = np.uint8(255 * np.clip(cam, 0, 1))
    bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def input_rgb(input_tensor: torch.Tensor) -> np.ndarray:
    """Imagen EXACTA que vio el modelo (des-normalizada): RGB uint8 [H,W,3].

    Con el enmascarado de fondo activado, esto muestra la hoja sobre fondo neutro
    —lo que el CNN realmente procesó— para auditarlo visualmente en la app.
    """
    base = denormalize(input_tensor.squeeze(0).detach().cpu())  # CHW en [0,1]
    return np.uint8(255 * base.permute(1, 2, 0).numpy())         # HWC RGB


def overlay_rgb(input_tensor: torch.Tensor, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Superpone el heatmap sobre la imagen original (des-normalizada)."""
    base = input_rgb(input_tensor)                               # HWC RGB
    heat = heatmap_rgb(cam)
    overlay = np.uint8((1 - alpha) * base + alpha * heat)
    return overlay


def leaf_mask_from_input(input_tensor: torch.Tensor) -> np.ndarray:
    """Máscara binaria de hoja (1=hoja, 0=fondo) alineada con el CAM.

    Se calcula sobre la MISMA imagen 224×224 que vio el modelo (des-normalizada),
    reutilizando la segmentación HSV verde/amarillo del módulo ROI. Así la máscara
    y el mapa de calor comparten exactamente la misma rejilla de píxeles.
    """
    from src.data.roi import leaf_mask  # import local para evitar ciclo

    base = denormalize(input_tensor.squeeze(0).detach().cpu())  # CHW [0,1]
    rgb = np.uint8(255 * base.permute(1, 2, 0).numpy())          # HWC RGB
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return (leaf_mask(bgr) > 0).astype(np.uint8)


def background_mass(cam: np.ndarray, leaf_mask: np.ndarray) -> float:
    """Fracción de la masa del Grad-CAM que cae sobre FONDO (no-hoja).

    Mide directamente lo que importa para auditar sesgo: ¿el modelo se apoya en
    píxeles que no son la hoja? Independiente de si el foco está al centro o al
    borde (una lesión real cerca del borde NO penaliza).
    """
    total = float(cam.sum()) + 1e-8
    bg = float((cam * (leaf_mask == 0)).sum())
    return bg / total


def bias_alert(cam: np.ndarray, leaf_mask: np.ndarray | None = None,
               threshold: float = 0.5) -> bool:
    """True si demasiada masa del CAM cae sobre fondo (posible atajo del dataset).

    Si se pasa `leaf_mask`, usa la fracción de masa sobre fondo (preferido) con el
    `threshold` calibrado. Sin máscara, cae al método antiguo basado en bordes.
    """
    if leaf_mask is not None:
        return bool(background_mass(cam, leaf_mask) > threshold)

    # Fallback (sin máscara disponible): masa fuera del recuadro central.
    h, w = cam.shape
    bh, bw = int(h * 0.15), int(w * 0.15)
    total = cam.sum() + 1e-8
    interior = cam[bh:h - bh, bw:w - bw].sum()
    return bool(1.0 - float(interior / total) > threshold)
