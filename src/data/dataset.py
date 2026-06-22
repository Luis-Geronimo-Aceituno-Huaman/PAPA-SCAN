"""Dataset de hojas de papa y escaneo de carpetas.

Las carpetas en disco usan la nomenclatura de PlantVillage
(`Potato___healthy`, etc.); aquí se mapean a las clases canónicas del proyecto
(`sana`, `tizon_tardio`, `tizon_temprano`) y a su índice fijo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "config"))
import config as cfg  # noqa: E402

_VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def scan_split(split_dir: str | Path) -> tuple[list[str], np.ndarray]:
    """Escanea un split (Train_Valid/ o Test/) y devuelve (rutas, etiquetas).

    Las etiquetas son índices enteros según `config.CLASS_TO_IDX`.
    """
    split_dir = Path(split_dir)
    if not split_dir.exists():
        raise FileNotFoundError(f"No existe el directorio: {split_dir}")

    paths: list[str] = []
    labels: list[int] = []
    for folder_name, class_name in cfg.FOLDER_TO_CLASS.items():
        class_dir = split_dir / folder_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Falta la carpeta de clase: {class_dir}")
        idx = cfg.CLASS_TO_IDX[class_name]
        for p in sorted(class_dir.iterdir()):
            if p.suffix.lower() in _VALID_EXT:
                paths.append(str(p))
                labels.append(idx)

    if not paths:
        raise RuntimeError(f"No se encontraron imágenes en {split_dir}")
    return paths, np.asarray(labels, dtype=np.int64)


class LeafDataset(Dataset):
    """Dataset a partir de listas paralelas de rutas y etiquetas.

    Trabajar con listas (en vez de ImageFolder) permite hacer splits
    estratificados y validación cruzada por índices sobre el mismo conjunto.

    Parameters
    ----------
    paths : lista de rutas a imágenes.
    labels : etiquetas enteras (mismo largo que paths).
    transform : transformación torchvision a aplicar.
    return_path : si True, __getitem__ devuelve (img, label, path).
    """

    def __init__(self, paths, labels, transform=None, return_path: bool = False):
        assert len(paths) == len(labels), "paths y labels deben tener igual largo"
        self.paths = list(paths)
        self.labels = list(labels)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        label = int(self.labels[i])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        if self.return_path:
            return img, label, path
        return img, label


def subset(paths, labels, indices):
    """Devuelve sublistas (paths, labels) para los índices dados."""
    paths = list(paths)
    labels = np.asarray(labels)
    idx = np.asarray(indices)
    return [paths[i] for i in idx], labels[idx]


def build_loader(dataset, batch_size, device, shuffle: bool) -> DataLoader:
    """Crea un DataLoader con ajustes de rendimiento coherentes.

    En CUDA usa pin_memory y, si hay workers, persistent_workers (evita el
    costoso respawn de procesos en cada época, importante en Windows).
    """
    pin = device.type == "cuda"
    workers = cfg.NUM_WORKERS
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, pin_memory=pin,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
    )
