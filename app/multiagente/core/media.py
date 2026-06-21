"""Persistencia de medios (foto, enmascarada, overlay, heatmap).

El AgentePercepcion produce arrays en memoria; DÓNDE se guardan depende de quién
use el sistema: la CLI escribe en ``app/multiagente/outputs/``; la web escribe en las
carpetas de la app (``storage/uploads`` y ``storage/heatmaps``). Para no acoplar
el agente a esa decisión, se le inyecta un *MediaSink* por la pizarra.
"""
from __future__ import annotations

from pathlib import Path

import cv2


class DirMediaSink:
    """Guarda los cuatro medios en directorios y con nombres configurables.

    foto_dir / media_dir : carpetas destino (la foto puede ir aparte).
    *_tpl                 : plantillas de nombre; ``{p}`` = prefijo del caso.
    """

    def __init__(self, foto_dir, media_dir=None,
                 foto_tpl: str = "{p}.jpg", masked_tpl: str = "masked_{p}.png",
                 overlay_tpl: str = "overlay_{p}.png", heatmap_tpl: str = "heatmap_{p}.png",
                 jpg_quality: int = 92) -> None:
        self.foto_dir = Path(foto_dir)
        self.media_dir = Path(media_dir) if media_dir is not None else Path(foto_dir)
        self.foto_tpl = foto_tpl
        self.masked_tpl = masked_tpl
        self.overlay_tpl = overlay_tpl
        self.heatmap_tpl = heatmap_tpl
        self.jpg_quality = jpg_quality
        self.foto_dir.mkdir(parents=True, exist_ok=True)
        self.media_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _save_rgb(array, path: Path) -> None:
        cv2.imwrite(str(path), cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def save(self, prefix: str, foto_pil, masked_arr, overlay_arr, heat_arr) -> dict:
        """Escribe los medios y devuelve sus rutas absolutas (como str)."""
        foto_path = self.foto_dir / self.foto_tpl.format(p=prefix)
        masked_path = self.media_dir / self.masked_tpl.format(p=prefix)
        overlay_path = self.media_dir / self.overlay_tpl.format(p=prefix)
        heatmap_path = self.media_dir / self.heatmap_tpl.format(p=prefix)

        foto_pil.convert("RGB").save(foto_path, format="JPEG", quality=self.jpg_quality)
        if masked_arr is not None:
            self._save_rgb(masked_arr, masked_path)
        self._save_rgb(overlay_arr, overlay_path)
        self._save_rgb(heat_arr, heatmap_path)

        return {
            "foto_path": str(foto_path),
            "masked_path": str(masked_path) if masked_arr is not None else "",
            "overlay_path": str(overlay_path),
            "heatmap_path": str(heatmap_path),
        }
