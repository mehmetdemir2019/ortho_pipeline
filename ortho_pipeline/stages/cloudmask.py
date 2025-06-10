# ortho_pipeline/stages/cloudmask.py
import numpy as np
import rasterio
from pathlib import Path
from skimage import morphology as morph
from ..base import Stage


class CloudMask(Stage):
    """Sentinel-2 .SAFE klasöründen basit bulut maskesi üretir."""

    def _read_band(self, safe_dir: Path, band_tag: str):
        """
        band_tag  →  örn. 'B02_10m.jp2'
        SAFE/GRANULE/IMG_DATA altındaki ilk eşleşen dosyayı döndürür.
        """
        jp2_list = list(safe_dir.rglob(f"*_{band_tag}"))
        if not jp2_list:
            raise FileNotFoundError(f"{band_tag} not found in {safe_dir}")
        with rasterio.open(jp2_list[0]) as src:
            return src.read(1).astype("f4"), src.profile

    def process(self, data):
        out_masks = []

        # Config’te inputs varsa onu, yoksa önceki stage’den gelen 'images' listesini kullan
        inputs = self.cfg.get("inputs",
                              data.get("images", []) if data else [])

        for safe in inputs:
            safe_dir = Path(safe)

            # 10 m bantları oku
            b02, prof = self._read_band(safe_dir, "B02_10m.jp2")  # Blue
            _,   _    = self._read_band(safe_dir, "B03_10m.jp2")  # Green (gerekmiyor)
            b04, _    = self._read_band(safe_dir, "B04_10m.jp2")  # Red
            b08, _    = self._read_band(safe_dir, "B08_10m.jp2")  # NIR

            # Basit bulut kuralı: haze (mavi/kırmızı) ve negatif NDVI
            ndvi = (b08 - b04) / (b08 + b04 + 1e-4)
            haze = b02 / np.maximum(b04, 1)
            cloud = (haze > 1.1) | (ndvi < -0.05)

            # Küçük gürültülü pikselleri at
            cloud = morph.remove_small_objects(cloud, 64)

            # Maske dosyasını kaydet
            mask_path = safe_dir.with_suffix(".cloud.tif")
            prof.update(driver="GTiff", count=1,
                        dtype="uint8", compress="DEFLATE")
            with rasterio.open(mask_path, "w", **prof) as dst:
                dst.write(cloud.astype("uint8"), 1)

            out_masks.append(mask_path.as_posix())

        data = data or {}
        data["masks"] = out_masks
        return data
