import streamlit as st, yaml, tempfile, pathlib, subprocess
from ortho_pipeline.pipeline import Pipeline

st.title("🔍 Sentinel-2 Mozaik Oluşturucu")

# 1️⃣ Kullanıcı SAFE klasör yol(lar)ını girsin (satır satır)
paths_txt = st.text_area(
    "SAFE klasörü yollari (her satıra bir yol yapıştır)",
    placeholder=r"C:\data\S2A_MSIL2A_20240814T085601…\SAFE",
    height=120,
)

# 2️⃣ Parametreler
pct_low  = st.slider("8-bit %Low", 0, 10, 2)
pct_high = st.slider("8-bit %High", 90, 100, 98)

if st.button("Mozaik Oluştur"):
    input_paths = [p.strip() for p in paths_txt.splitlines() if p.strip()]
    if not input_paths:
        st.error("En az bir SAFE yolu girmelisiniz!")
        st.stop()

    # ✔️ Yol doğrulama
    missing = [p for p in input_paths if not pathlib.Path(p).is_dir()]
    if missing:
        st.error(f"Bulunamadı / klasör değil:\n{chr(10).join(missing)}")
        st.stop()

    # 3️⃣ Geçici config.yml yaz
    cfg = {
        "order": ["preproc", "colorbalance", "mosaic", "scale8bit"],
        "preproc": {
            "inputs": input_paths,
            "target_crs": "EPSG:32635",
            "pixel_size": 10,
            "out_dir": "ortho",
        },
        "colorbalance": {"sample_per_edge": 5000},
        "mosaic": {"out_path": "mosaic/marmara_mosaic.tif"},
        "scale8bit": {
            "input_key": "mosaic",
            "pct_low": pct_low,
            "pct_high": pct_high,
            "out_suffix": "_8bit.tif",
            "nodata": 0,
        },
    }

    with tempfile.NamedTemporaryFile(
        delete=False, suffix=".yml", mode="w", encoding="utf-8"
    ) as tmp_cfg:
        yaml.safe_dump(cfg, tmp_cfg)
        cfg_path = tmp_cfg.name

    st.info("🛠 İşlem başlıyor, lütfen bekleyin…")
    try:
        result = Pipeline(cfg_path).run()
        st.success("✅ Tamamlandı!")
        st.json(result)

        img_path = pathlib.Path(result["scaled"])
        if img_path.exists():
            st.image(str(img_path), caption="8-bit Mozaik Ön-izleme", use_column_width=True)
    except Exception as e:
        st.error(f"🚨 Hata: {e}")
