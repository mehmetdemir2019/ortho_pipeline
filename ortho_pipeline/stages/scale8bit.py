# ortho_pipeline/stages/scale8bit.py  ▸ rev-AUTO

import numpy as np, rasterio
from pathlib import Path
from ..base import Stage

class Scale8bit(Stage):
    def process(self, data):
        in_path  = Path(data[self.cfg.get("input_key", "mosaic")])
        out_path = in_path.with_name(in_path.stem + "_8bit.tif")

        with rasterio.open(in_path) as src:
            arr = src.read().astype("float32")       # 4×H×W
            profile = src.profile

            # otomatik %2–%98 percentil
            mins, maxs = [], []
            for b in range(src.count):
                band = arr[b].ravel()
                mins.append(np.percentile(band, 2))
                maxs.append(np.percentile(band, 98))

        # manuel değer tanımlıysa baskın gelsin
        man_min = self.cfg.get("min_clip"); man_max = self.cfg.get("max_clip")
        if man_min is not None and man_max is not None:
            mins = [man_min]*arr.shape[0]; maxs = [man_max]*arr.shape[0]

        scaled = np.empty_like(arr, dtype="uint8")
        for i in range(arr.shape[0]):
            scaled[i] = np.clip(
                (arr[i]-mins[i]) / (maxs[i]-mins[i]) * 255, 0, 255
            )

        profile.update(dtype="uint8",
                       compress="DEFLATE",
                       nodata=0,
                       photometric="RGB")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(scaled)

        data["scaled"] = out_path.as_posix()
        return data
