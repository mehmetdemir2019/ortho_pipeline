# ortho_pipeline/stages/cutline.py
import numpy as np
import rasterio
from shapely.geometry import box
from skimage.graph import route_through_array
from pathlib import Path
from ..base import Stage


class Cutline(Stage):
    def _seam_and_bounds(self, fp_a, fp_b):
        with rasterio.open(fp_a) as A, rasterio.open(fp_b) as B:
            inter = box(*A.bounds).intersection(box(*B.bounds))
            if inter.is_empty:
                return None, None

            wa = rasterio.windows.from_bounds(*inter.bounds, transform=A.transform).round_offsets().round_lengths()
            wb = rasterio.windows.from_bounds(*inter.bounds, transform=B.transform).round_offsets().round_lengths()

            a = A.read(1, window=wa).astype("f4")
            b = B.read(1, window=wb).astype("f4")
        cost = np.abs(a - b)
        seam, _ = route_through_array(cost, (0, 0), (cost.shape[0]-1, cost.shape[1]-1))
        mask = np.zeros_like(cost, dtype=bool)
        for r, c in seam: mask[r, c] = True
        return mask, inter.bounds

    def process(self, data):
        pairs = list(zip(data["images"][:-1], data["images"][1:]))
        seams   = []
        bboxes  = []
        for fp_a, fp_b in pairs:
            msk, bbox = self._seam_and_bounds(fp_a, fp_b)
            if msk is not None:
                seams.append(msk)
                bboxes.append(bbox)

        data["seam_mask_list"]   = seams
        data["cutline_bounds_list"] = bboxes
        return data
