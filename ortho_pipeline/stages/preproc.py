# ortho_pipeline/stages/preproc.py
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling, calculate_default_transform
from pathlib import Path
from ..base import Stage

# İstediğimiz Sentinel-2 bantları ve dosya etiketleri
BANDS = {
    "B02": "B02_10m.jp2",  # Blue
    "B03": "B03_10m.jp2",  # Green
    "B04": "B04_10m.jp2",  # Red
    "B08": "B08_10m.jp2",  # NIR
}

def find_band(safe_dir: Path, tag: str) -> Path:
    """SAFE klasörü içinde band dosyasını bulur."""
    files = list(safe_dir.rglob(f"*_{tag}"))
    if not files:
        raise FileNotFoundError(f"{tag} not found in {safe_dir}")
    return files[0]

def warp_one(src_path, profile, dst_array, idx, resampling):
    """Tek bandı hedef grid'e yeniden projekte eder."""
    with rasterio.open(src_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_array[idx],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=getattr(Resampling, resampling.lower()),
        )

class Preproc(Stage):
    """
    .SAFE klasörlerini hedef CRS/pikselde 4-bant orto GeoTIFF’e çevirir.
    cfg:
      inputs: [".SAFE", ".SAFE", ...]
      target_crs: EPSG:32635
      pixel_size: 10
      resampling: cubic
      out_dir: C:/.../ortho
    """
    def process(self, data=None):
        out_dir = Path(self.cfg["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        ortho_list = []

        for safe in self.cfg["inputs"]:
            safe_dir = Path(safe)
            dst_path = out_dir / f"{safe_dir.stem}_ortho.tif"

            # Referans bandı (Red) ile hedef grid tanımı
            with rasterio.open(find_band(safe_dir, BANDS["B04"])) as ref:
                trf, w, h = calculate_default_transform(
                    ref.crs, self.cfg["target_crs"],
                    ref.width, ref.height, *ref.bounds,
                    resolution=self.cfg["pixel_size"]
                )
                profile = ref.profile.copy()
                profile.update({
                    "crs": self.cfg["target_crs"],
                    "transform": trf,
                    "width": w,
                    "height": h,
                    "count": 4,
                    "dtype": "uint16",
                    "driver": "GTiff",
                    "compress": "DEFLATE",
                })

            # 4 bant için boş raster
            stack = np.zeros((4, h, w), dtype="uint16")

            for i, tag in enumerate(BANDS.values()):
                warp_one(find_band(safe_dir, tag),
                         profile, stack, i,
                         self.cfg.get("resampling", "cubic"))

            with rasterio.open(dst_path, "w", **profile) as dst:
                dst.write(stack)

            ortho_list.append(dst_path.as_posix())

        # Sonraki stage’lere bilgiyi ilet
        return {
            "images": ortho_list,
            "crs": self.cfg["target_crs"],
            "pixel": self.cfg["pixel_size"],
        }
