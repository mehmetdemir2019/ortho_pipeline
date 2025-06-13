# ortho_pipeline/stages/cogify.py
import subprocess
from pathlib import Path
from ..base import Stage

class Cogify(Stage):
    """
    Mevcut rasterı Cloud-Optimized GeoTIFF'e çevirir.
    cfg:
      input_key: "mosaic"      # pipeline içinde hangi anahtarı oku
      out_suffix: "_cog.tif"  # marmara_mosaic + _cog.tif
      blocksize: 512
      compress: DEFLATE
    """
    def process(self, data):
        in_path = Path(data[self.cfg.get("input_key", "mosaic")])
        out_path = in_path.with_name(in_path.stem + self.cfg.get("out_suffix", "_cog.tif"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "gdal_translate",
            "-of", "COG",
            "-co", f"COMPRESS={self.cfg.get('compress','DEFLATE')}",
            "-co", f"BLOCKSIZE={self.cfg.get('blocksize',512)}",
            str(in_path),
            str(out_path)
        ]
        subprocess.run(cmd, check=True)
        data["cog"] = out_path.as_posix()
        return data
