# ortho_pipeline/stages/mosaic.py
import rasterio
from rasterio.merge import merge
from pathlib import Path
from ..base import Stage

class Mosaic(Stage):
    """Birden çok ortho‐TIF’i tek GeoTIFF’e birleştirir."""

    def process(self, data):
        ortho_files = data["images"]
        srcs = [rasterio.open(p) for p in ortho_files]

        arr, tr = merge(srcs, nodata=0)
        profile = srcs[0].profile
        profile.update(
            height=arr.shape[1],
            width=arr.shape[2],
            transform=tr,
            count=arr.shape[0],
            dtype=arr.dtype,
            compress="DEFLATE",
        )

        out = Path(self.cfg["out_path"])
        out.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(arr)

        for s in srcs:
            s.close()

        data["mosaic"] = out.as_posix()
        return data
