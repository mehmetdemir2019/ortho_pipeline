# ortho_pipeline/stages/blend.py   ◂ rev-D  (boyut sabit & güvenli)

import numpy as np, cv2, rasterio
from pathlib import Path
from ..base import Stage


class Blend(Stage):
    def process(self, data):

        imgs    = data["images"]
        seams   = data["seam_mask_list"]
        bounds  = data["cutline_bounds_list"]
        feather = self.cfg.get("feather", 25)

        # 1) taban raster = ilk ortho
        base_fp = Path(imgs[0])
        with rasterio.open(base_fp) as A:
            base    = A.read().astype("f4")          # 4×H×W
            profile = A.profile
            base_tr = A.transform

        # 2) diğer sahneleri sırayla ekle
        for idx, fp_b in enumerate(imgs[1:]):
            seam = seams[idx]
            bbox = bounds[idx]

            with rasterio.open(fp_b) as B:
                win_b = rasterio.windows.from_bounds(*bbox, transform=B.transform)\
                                         .round_offsets().round_lengths()
                win_a = rasterio.windows.from_bounds(*bbox, transform=base_tr)\
                                         .round_offsets().round_lengths()

                arrB  = B.read(window=win_b).astype("f4")          # 4×hb×wb

            # --- ortak boyut ---
            h = min(seam.shape[0], arrB.shape[1], win_a.height)
            w = min(seam.shape[1], arrB.shape[2], win_a.width)

            # Seam & alpha
            seam_crop = seam[:h, :w]
            dist  = cv2.distanceTransform(~seam_crop.astype("uint8"),
                                          cv2.DIST_L2, 3)
            alpha = np.clip(dist / feather, 0, 1)[None, ...]       # 1×h×w

            # Pencere koordinatları
            r0, c0 = win_a.row_off, win_a.col_off
            r1, c1 = r0 + h, c0 + w

            # Ayıklanmış dilimler
            base_cut = base[:, r0:r1, c0:c1]       # 4×h×w
            B_cut    = arrB[:, :h, :w]             # 4×h×w

            # Feather-blend
            base[:, r0:r1, c0:c1] = alpha * base_cut + (1 - alpha) * B_cut

        # 3) çıktı
        out_fp = base_fp.with_name("blend_full.tif")
        profile.update(compress="DEFLATE")
        with rasterio.open(out_fp, "w", **profile) as dst:
            dst.write(base.astype(profile["dtype"]))

        data["images"] = [out_fp.as_posix()]
        data["blend"]  = out_fp.as_posix()
        return data
