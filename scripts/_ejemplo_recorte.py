import sys
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "config"))
import config as cfg
from src.data.roi import apply_roi

img = next((cfg.TEST_DIR / "Potato___Late_blight").glob("00695906*"))
pil = Image.open(img).convert("RGB")
out = ROOT / "outputs" / "roi_preview"
out.mkdir(parents=True, exist_ok=True)

pil.save(out / "ejemplo_original.png")
apply_roi(pil, crop=True, bg_mask=False).save(out / "ejemplo_recorte_solo.png")
apply_roi(pil, crop=True, bg_mask=True).save(out / "ejemplo_recorte_y_mascara.png")

print("Imagen base:", img.name, "tam:", pil.size)
print("Guardado en:", out)
for f in ("ejemplo_original.png", "ejemplo_recorte_solo.png", "ejemplo_recorte_y_mascara.png"):
    print(f"  - {f}  ->", Image.open(out / f).size)
