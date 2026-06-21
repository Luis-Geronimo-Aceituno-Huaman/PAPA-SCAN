"""Fase de inferencia — función pura y autocontenida.

`AgriVisionPredictor.predict()` es la ÚNICA puerta de entrada al modelo. Recibe
una ruta o imagen ya cargada y devuelve siempre el mismo objeto estructurado,
serializable a JSON. No imprime, no muestra ventanas, no llama a plt.show().
Toda presentación es responsabilidad de quien la consuma (CLI, GUI, Agente 3).

El diagnóstico (CNN), la confianza calibrada y el Grad-CAM se calculan en el
mismo forward/backward pass (ver gradcam.GradCAM).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.transforms import build_inference_transforms  # noqa: E402
from src.inference.gradcam import (  # noqa: E402
    GradCAM, bias_alert, heatmap_rgb, input_rgb, leaf_mask_from_input, overlay_rgb,
)
from src.models.architecture import build_model, get_gradcam_layer  # noqa: E402
from src.utils.helpers import get_device  # noqa: E402


def save_checkpoint(path, model, metadata: dict) -> None:
    """Guarda el modelo final junto con sus metadatos (formato fijo)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, str(path))


class AgriVisionPredictor:
    """Carga un modelo entrenado + metadatos y produce diagnósticos calibrados."""

    def __init__(self, checkpoint_path: str, device: torch.device | None = None):
        self.device = device or get_device()
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.meta = ckpt["metadata"]

        self.model = build_model(
            backbone=self.meta.get("backbone", cfg.BACKBONE),
            dropout=self.meta["hiperparametros"]["dropout"],
            num_trainable_blocks=self.meta["hiperparametros"]["num_trainable_blocks"],
            pretrained=False,
        )
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(self.device).eval()

        layer, layer_name = get_gradcam_layer(self.model)
        self.gradcam = GradCAM(self.model, layer)
        self.layer_name = self.meta.get("capa_gradcam", layer_name)

        self.temperature = float(self.meta.get("temperatura", 1.0))
        self.thresholds = self.meta["umbrales_por_clase"]
        # Umbral calibrado de masa-en-fondo para la alerta de sesgo (default 0.5).
        self.bias_threshold = float(self.meta.get("umbral_sesgo_fondo", 0.5))
        self.version = self.meta["version_modelo"]
        self.train_metrics = self.meta.get("metricas_entrenamiento_por_clase", {})
        self.transform = build_inference_transforms()

    # ---------------------------------------------------------------- #
    def _metrica_critica(self, cls: str) -> dict:
        target = cfg.CLASS_METRIC_TARGETS[cls]
        return {
            "nombre": target["sensibilidad"],
            "tipo": target["metrica_principal"],
            "valor_train": float(self.train_metrics.get(cls, float("nan"))),
        }

    def predict(self, image, image_id: str | None = None, save_outputs: bool = True) -> dict:
        """Diagnostica una imagen y devuelve el objeto estructurado.

        image : ruta (str/Path) o PIL.Image ya cargada.
        save_outputs : si False, no escribe PNGs; devuelve los arrays en memoria.
        """
        # 1) Cargar y preprocesar (mismo pipeline que validación).
        if isinstance(image, (str, Path)):
            pil = Image.open(image).convert("RGB")
            if image_id is None:
                image_id = Path(image).stem
        else:
            pil = image.convert("RGB")
            if image_id is None:
                image_id = "muestra"
        x = self.transform(pil).unsqueeze(0).to(self.device)

        # 2-5) Diagnóstico + Grad-CAM en el MISMO forward/backward pass.
        logits, cam, pred_idx = self.gradcam(x, class_idx=None)

        # 3) Confianza calibrada (temperature scaling).
        probs = torch.softmax(logits[0] / self.temperature, dim=0).cpu().numpy()
        clase_predicha = cfg.IDX_TO_CLASS[pred_idx]
        confianza = float(probs[pred_idx])

        # 4) Umbral por clase -> estado de confianza.
        umbral = float(self.thresholds.get(clase_predicha, cfg.DEFAULT_CONFIDENCE_THRESHOLD))
        estado = "alta" if confianza >= umbral else "baja_confianza"

        # 6) Heatmap + overlay.
        heat = heatmap_rgb(cam)
        over = overlay_rgb(x, cam)

        # Alerta de sesgo: ¿cuánta masa del Grad-CAM cae sobre fondo (no-hoja)?
        mask = leaf_mask_from_input(x)
        alerta = bias_alert(cam, leaf_mask=mask, threshold=self.bias_threshold)

        overlay_path, heatmap_path = None, None
        if save_outputs:
            cfg.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
            overlay_path = str(cfg.OUTPUTS_DIR / f"overlay_{image_id}.png")
            heatmap_path = str(cfg.OUTPUTS_DIR / f"heatmap_{image_id}.png")
            cv2.imwrite(overlay_path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
            cv2.imwrite(heatmap_path, cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))

        # 7) Empaquetar en un único objeto estructurado (serializable a JSON).
        resultado = {
            "clase_predicha": clase_predicha,
            "confianza": round(confianza, 4),
            "probabilidades": {
                cfg.IDX_TO_CLASS[i]: round(float(probs[i]), 4) for i in range(len(cfg.CLASSES))
            },
            "estado_confianza": estado,
            "umbral_clase": round(umbral, 4),
            "metrica_critica_clase": self._metrica_critica(clase_predicha),
            "overlay_path": overlay_path,
            "heatmap_path": heatmap_path,
            "capa_usada": self.layer_name,
            "alerta_sesgo": alerta,
            "version_modelo": self.version,
        }
        if not save_outputs:
            # La GUI/orquestador puede renderizar o guardar los arrays donde quiera,
            # y usar el CAM y la máscara para severidad/auditoría.
            resultado["entrada_array"] = input_rgb(x)   # imagen enmascarada que vio el modelo
            resultado["overlay_array"] = over
            resultado["heatmap_array"] = heat
            resultado["cam"] = cam
            resultado["leaf_mask"] = mask
        return resultado

    def close(self):
        """Libera los hooks del Grad-CAM."""
        self.gradcam.remove()
