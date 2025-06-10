from pathlib import Path
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def warp_to(dst_path: Path, src_path: Path,
            crs: str, pixel_size: float, resampling: str = "cubic",
            dem_path: Path | None = None, rpc: bool = True) -> Path:
    """Tek görüntüyü hedef CRS & piksele yeniden örnekler."""
    resampling_enum = Resampling[resampling]
    with rasterio.open(src_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, crs, src.width, src.height,
            *src.bounds, resolution=pixel_size)
        kwargs = src.meta.copy()
        kwargs.update(dict(crs=crs, transform=transform,
                           width=width, height=height))
        with rasterio.open(dst_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=crs,
                    resampling=resampling_enum)
    return dst_path
