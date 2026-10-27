"""Aplicatia Streamlit: clasificarea unei imagini cu un semn de circulatie (GTSRB)."""
import time

import streamlit as st

from traffic_signs import config
from traffic_signs.classes import CLASS_NAMES
from traffic_signs.inference import (
    InvalidImageError, available_models, classify, format_percent, is_uncertain,
    meta_image_path, read_image, sample_class_id, sample_images,
)
from traffic_signs.models import load_trained_model

TITLE = "Recunoașterea semnelor de circulație"
DESCRIPTIONS = {
    "cnn": "CNN propriu, 1,3 milioane de parametri, imagini 48×48",
    "resnet50": "ResNet50 pre-antrenat pe ImageNet, 24 milioane de parametri, imagini 96×96",
}
PROMOTE_CMD = "uv run python scripts/promote_model.py --run-name resnet50_conv4 --as resnet50_conv4"
NO_SAMPLE = "(niciunul)"


@st.cache_resource(show_spinner="Se încarcă modelul…")
def get_model(path: str):
    """Modelul se incarca o singura data per cale (cache)."""
    return load_trained_model(path)


st.set_page_config(page_title=TITLE, page_icon="🚦", layout="wide")
st.title(f"🚦 {TITLE}")
st.write("Încarcă o fotografie cu un semn de circulație și modelul îi spune ce semn este.")

# directoarele se citesc la rulare (nu la import), ca testele sa le poata inlocui
models_dir = config.MODELS_DIR
assets_dir = config.APP_ASSETS_DIR

specs = available_models(models_dir)
if not specs:
    st.error("Nu există niciun model antrenat în folderul `models/`.")
    st.stop()

with st.sidebar:
    label = st.radio("Model", [s.label for s in specs])
    spec = next(s for s in specs if s.label == label)
    st.caption(DESCRIPTIONS[spec.key])
    if len(specs) == 1:
        st.info(f"Modelul ResNet50 lipsește. Adaugă-l cu:\n\n`{PROMOTE_CMD}`")

uploaded = st.file_uploader(
    "Încarcă o imagine cu un semn", type=["png", "jpg", "jpeg", "ppm", "bmp", "webp"]
)
samples = {CLASS_NAMES[sample_class_id(p)]: p for p in sample_images(assets_dir)}
choice = st.selectbox("…sau alege un exemplu", [NO_SAMPLE, *samples])

source, true_id = None, None
if uploaded is not None:  # fisierul incarcat are prioritate
    source = uploaded
elif choice != NO_SAMPLE:
    source = samples[choice]
    true_id = sample_class_id(source)

if source is None:
    st.info("Merge cel mai bine când semnul ocupă cea mai mare parte a imaginii.")
else:
    try:
        image = read_image(source)
    except InvalidImageError as err:
        st.error(str(err))
    else:
        model = get_model(str(models_dir / spec.filename))
        start = time.perf_counter()
        top = classify(model, image, spec.image_size, k=3)
        elapsed_ms = (time.perf_counter() - start) * 1000
        best = top[0]

        col_img, col_pred, col_meta = st.columns(3)
        with col_img:
            st.image(image, caption="Imaginea încărcată", use_container_width=True)
        with col_pred:
            st.subheader(best.name)
            st.metric("Încredere", format_percent(best.confidence))
            for p in top:
                st.progress(min(max(p.confidence, 0.0), 1.0), text=f"{p.name} — {format_percent(p.confidence)}")
            st.caption(f"Timp de inferență: {elapsed_ms:.0f} ms")
            if is_uncertain(best):
                st.warning(
                    "Modelul nu e sigur. Poate semnul nu e unul dintre cele 43 cunoscute "
                    "sau imaginea nu e clară."
                )
            if true_id is not None:
                mark = "✅" if best.class_id == true_id else "❌"
                st.write(f"Clasa reală: **{CLASS_NAMES[true_id]}** {mark}")
        with col_meta:
            st.markdown("**Semnul oficial**")
            meta = meta_image_path(best.class_id, assets_dir)
            if meta is not None:
                st.image(str(meta), use_container_width=True)

st.caption("Modelul cunoaște doar cele 43 de semne din setul german GTSRB.")
