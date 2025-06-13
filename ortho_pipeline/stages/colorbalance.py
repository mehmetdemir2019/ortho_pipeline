import rasterio
import numpy as np
from pathlib import Path
from itertools import combinations
from sklearn.linear_model import LinearRegression
from shapely.geometry import box
from ..base import Stage


def read_raster(fp):
    with rasterio.open(fp) as src:
        arr = src.read().astype("float32")
        geom = box(*src.bounds)
        profile = src.profile
    return arr, geom, profile


def overlap_window(src1, src2):
    # rasterio.windows.intersection yerine kendi geometri kesişimini kullan
    inter = src1["geom"].intersection(src2["geom"])
    if inter.is_empty:
        return None
    win1 = rasterio.windows.from_bounds(*inter.bounds,
                                        src1["profile"]["transform"],
                                        height=src1["profile"]["height"],
                                        width=src1["profile"]["width"])
    win2 = rasterio.windows.from_bounds(*inter.bounds,
                                        src2["profile"]["transform"],
                                        height=src2["profile"]["height"],
                                        width=src2["profile"]["width"])
    return win1.round_offsets().round_lengths(), win2.round_offsets().round_lengths()


class ColorBalance(Stage):
    """
    cfg:
      sample_per_edge: 10000       # her komşu çift için örnek piksel sayısı
      solver: "lstsq"              # global çözüm Y = A·gain + bias
    """

    def process(self, data):
        ortho_files = data["images"]
        scenes = []

        # 1) Tüm ortho'ları oku ve geometrilerini sakla
        for fp in ortho_files:
            arr, geom, prof = read_raster(fp)
            scenes.append({"fp": fp, "arr": arr, "geom": geom, "profile": prof})

        bands = scenes[0]["arr"].shape[0]
        pairs = list(combinations(range(len(scenes)), 2))

        # 2) Her komşu çift için örnek pikseller topla
        samples = {b: [] for b in range(bands)}   # {band: [(x1,x2),...]}
        for i, j in pairs:
            win = overlap_window(scenes[i], scenes[j])
            if win is None:
                continue
            w1, w2 = win
            # rastgele örnek piksel çek
            a = scenes[i]["arr"][:, w1.row_off:w1.row_off+w1.height,
                                   w1.col_off:w1.col_off+w1.width]
            b = scenes[j]["arr"][:, w2.row_off:w2.row_off+w2.height,
                                   w2.col_off:w2.col_off+w2.width]
            h, w = a.shape[1:]

            n = min(self.cfg.get("sample_per_edge", 10000), h * w)
            idx = np.random.choice(h * w, n, replace=False)
            yy, xx = np.unravel_index(idx, (h, w))

            for band in range(bands):
                samples[band].append((a[band, yy, xx], b[band, yy, xx]))

        # 3) Band bazlı gain & bias çöz (X=g1*Y+b1, Y=g2*X+b2 vs.)
        gains, biases = [], []
        for band in range(bands):
            # Tüm örnek çiftlerini tek diziye birleştir
            src_pix = np.concatenate([s[0] for s in samples[band]])
            dst_pix = np.concatenate([s[1] for s in samples[band]])
            model = LinearRegression().fit(src_pix.reshape(-1, 1), dst_pix)
            gains.append(model.coef_[0])
            biases.append(model.intercept_)

        # 4) Katsayıları tüm sahnelere uygula (basit global)
        balanced_paths = []
        for sc in scenes:
            arr = sc["arr"].copy()
            for b in range(bands):
                arr[b] = arr[b] * gains[b] + biases[b]
            out_fp = Path(sc["fp"]).with_name(Path(sc["fp"]).stem + "_bal.tif")
            with rasterio.open(out_fp, "w", **sc["profile"]) as dst:
                dst.write(arr.astype(sc["profile"]["dtype"]))
            balanced_paths.append(out_fp.as_posix())

        # 5) Pipeline’a yeni balanced dosyaları ver
        data["images"] = balanced_paths
        data["colorbalanced"] = True
        return data
