"""Aplicatia Streamlit: clasificarea unei imagini cu un semn de circulatie (GTSRB)."""
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

from traffic_signs import config
from traffic_signs.classes import CLASS_NAMES
from traffic_signs.inference import (
    InvalidImageError, available_models, classify, format_percent, is_uncertain,
    meta_image_path, read_image, sample_class_id, sample_images,
)
from traffic_signs.models import load_trained_model
from traffic_signs.scene import (
    PipelineConfig, SceneSign, annotate, crop_with_margin, filter_by_conf, load_pipeline_config,
    load_scene_gt, match, run_pipeline, scene_samples, sign_statuses, warm_up_yolo,
)

TITLE = "Recunoașterea semnelor de circulație"
DESCRIPTIONS = {
    "cnn": "CNN propriu, 1,3 milioane de parametri, imagini 48×48",
    "resnet50": "ResNet50 pre-antrenat pe ImageNet, 24 milioane de parametri, imagini 96×96",
}
PROMOTE_CMD = "uv run python scripts/promote_model.py --run-name resnet50_conv4 --as resnet50_conv4"
NO_SAMPLE = "(niciunul)"
PROMOTE_YOLO_CMD = "uv run python scripts/evaluate_yolo.py --run-name <nume> --promote"
TUNE_CMD = "uv run python scripts/tune_pipeline.py"


@st.cache_resource(show_spinner="Se încarcă modelul…")
def get_model(path: str):
    """Modelul se incarca o singura data per cale (cache)."""
    return load_trained_model(path)


@st.cache_resource(show_spinner="Se încarcă detectorul YOLO…")
def get_yolo(path: str):
    """Detectorul se incarca o singura data; ultralytics se importa doar aici (lenes)."""
    from PIL import Image

    original_open = Image.open
    from ultralytics import YOLO

    # ultralytics inlocuieste global PIL.Image.open: la un fisier invalid incearca
    # `pip install pi-heif` si arunca ModuleNotFoundError, nu mesajul nostru prietenos
    Image.open = original_open
    yolo = YOLO(path)
    warm_up_yolo(yolo)  # incalzire in spinner: timpul afisat nu include initializarea
    return yolo


@st.cache_data(show_spinner="Caut semnele în poză…", max_entries=16)
def detect_and_classify(image: np.ndarray, yolo_path: str, model_path: str,
                        cfg: PipelineConfig) -> tuple[list[SceneSign], dict[str, float]]:
    """YOLO + CNN o singura data per poza (cheia = octetii imaginii + caile modelelor + cfg).

    Detectia se face la pragul minim; glisorul doar filtreaza rezultatul din cache.
    """
    return run_pipeline(get_yolo(yolo_path), get_model(model_path), image, cfg, device="cpu")


def scene_metrics_sentence(reports_dir: Path) -> str | None:
    path = Path(reports_dir) / "metrics.json"
    if not path.is_file():
        return None
    try:
        m = json.loads(path.read_text())
        return (f"Pe {m['n_images']} de poze de test (GTSDB), lanțul YOLO + CNN a găsit și a numit "
                f"corect {format_percent(m['e2e_recall'])} din cele {m['n_gt']} de semne reale.")
    except (ValueError, KeyError, TypeError):
        return None


def render_sign_card(sign: SceneSign, image: np.ndarray, margin: float, assets_dir: Path,
                     verdict: str | None) -> None:
    best = sign.best
    with st.container(border=True):
        col_crop, col_meta = st.columns(2)
        with col_crop:
            st.image(crop_with_margin(image, sign.box, margin), caption="Din poză", width="stretch")
        with col_meta:
            meta = meta_image_path(best.class_id, assets_dir)
            if meta is not None:
                st.image(str(meta), caption="Semnul oficial", width="stretch")
        st.markdown(f"**{sign.number}. {best.name}**")
        st.write(f"Detector (YOLO): {format_percent(sign.det_score)}")
        st.write(f"Clasificator (CNN): {format_percent(best.confidence)}")
        if is_uncertain(best):
            st.warning("Clasificatorul nu e sigur de numele acestui semn.")
        if verdict:
            st.write(verdict)


def render_scene_tab(models_dir: Path, assets_dir: Path) -> None:
    st.write("Încarcă o poză de stradă: detectorul YOLO găsește semnele, iar CNN-ul le numește.")
    # caile se citesc la rulare, ca testele sa le poata inlocui
    yolo_path, cfg_path = Path(config.YOLO_WEIGHTS), Path(config.PIPELINE_CONFIG)
    if not yolo_path.is_file():
        st.info(f"Lipsește detectorul YOLO (`{yolo_path.name}`). Adaugă-l cu:\n\n`{PROMOTE_YOLO_CMD}`")
        return
    if not cfg_path.is_file():
        st.info(f"Lipsește configurația lanțului (`{cfg_path.name}`). Creeaz-o cu:\n\n`{TUNE_CMD}`")
        return
    try:
        cfg = load_pipeline_config(cfg_path)
    except (ValueError, KeyError, TypeError):  # JSONDecodeError e subclasa de ValueError
        st.error(f"Configurația lanțului (`{cfg_path.name}`) este invalidă. "
                 f"Recreeaz-o cu:\n\n`{TUNE_CMD}`")
        return
    model_path = Path(models_dir) / cfg.classifier
    if not model_path.is_file():
        st.info(f"Lipsește clasificatorul `{cfg.classifier}` din folderul `models/`.")
        return

    sentence = scene_metrics_sentence(config.M8_REPORTS_DIR)
    if sentence:
        st.caption(sentence)

    uploaded = st.file_uploader(
        "Încarcă o poză de stradă", type=["png", "jpg", "jpeg", "ppm", "bmp", "webp"],
        key="scene_upload",
    )
    samples = {f"Poza {p.stem}": p for p in scene_samples(assets_dir)}
    choice = st.selectbox("…sau alege un exemplu", [NO_SAMPLE, *samples], key="scene_sample")

    source, sample_stem = None, None
    if uploaded is not None:  # fisierul incarcat are prioritate
        source = uploaded
        if choice != NO_SAMPLE:
            st.caption("Se folosește poza încărcată; exemplul ales e ignorat.")
    elif choice != NO_SAMPLE:
        source, sample_stem = samples[choice], samples[choice].stem

    if source is None:
        st.info("Alege o poză ca să încep. Modelele se încarcă abia atunci.")
        return
    try:
        image = read_image(source)
    except InvalidImageError as err:
        st.error(str(err))
        return

    conf = st.slider("Prag de încredere (detecție)", 0.05, 0.95, float(cfg.conf), 0.05,
                     key="scene_conf")
    gts = load_scene_gt(assets_dir).get(sample_stem) if sample_stem else None
    show_truth = gts is not None and st.toggle("Arată răspunsul corect", key="scene_truth")

    try:
        all_signs, timings = detect_and_classify(image, str(yolo_path), str(model_path), cfg)
    except Exception:  # fisier corupt/incompatibil: mesaj prietenos, fara traceback
        st.error("Modelele nu au putut fi încărcate. Verifică fișierele din folderul `models/`.")
        return
    signs = filter_by_conf(all_signs, conf)

    statuses, missed, verdicts = None, None, {}
    if show_truth:
        m = match(signs, gts)
        statuses = sign_statuses(signs, gts, m)
        missed = [gts[i] for i in m.missed]
        for si, gi in m.pairs:
            s, real = signs[si], CLASS_NAMES[gts[gi].class_id]
            verdicts[s.number] = ("✅ Găsit și numit corect" if statuses[s.number] == "correct"
                                  else f"❌ Nume greșit — clasa reală: **{real}**")
        for si in m.false_signs:
            verdicts[signs[si].number] = "❌ Detecție falsă (nu e semn acolo)"

    st.image(annotate(image, signs, statuses, missed), width="stretch")
    st.caption(f"Timp: detecție {timings['detect_ms']:.0f} ms + "
               f"clasificare {timings['classify_ms']:.0f} ms")
    if missed:
        names = ", ".join(CLASS_NAMES[g.class_id] for g in missed)
        st.warning(f"Semne ratate de detector (pătrat punctat): {names}")

    if not signs:
        st.info("Nu am găsit niciun semn peste pragul ales. Încearcă un prag mai mic.")
        return
    for row_start in range(0, len(signs), 3):
        cols = st.columns(3)
        for col, sign in zip(cols, signs[row_start:row_start + 3]):
            with col:
                render_sign_card(sign, image, cfg.margin, assets_dir, verdicts.get(sign.number))


st.set_page_config(page_title=TITLE, page_icon="🚦", layout="wide")
st.title(f"🚦 {TITLE}")

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
    if not any(s.key == "resnet50" for s in specs):
        st.info(f"Modelul ResNet50 lipsește. Adaugă-l cu:\n\n`{PROMOTE_CMD}`")
    st.caption("Modelul ales se aplică tabului «Un semn».")

tab_single, tab_scene = st.tabs(["Un semn", "Scenă de stradă"])

with tab_single:
    st.write("Încarcă o fotografie cu un semn de circulație și modelul îi spune ce semn este.")
    uploaded = st.file_uploader(
        "Încarcă o imagine cu un semn", type=["png", "jpg", "jpeg", "ppm", "bmp", "webp"]
    )
    samples = {CLASS_NAMES[sample_class_id(p)]: p for p in sample_images(assets_dir)}
    choice = st.selectbox("…sau alege un exemplu", [NO_SAMPLE, *samples])

    source, true_id = None, None
    if uploaded is not None:  # fisierul incarcat are prioritate
        source = uploaded
        if choice != NO_SAMPLE:
            st.caption("Se folosește imaginea încărcată; exemplul ales e ignorat.")
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
            try:
                model = get_model(str(models_dir / spec.filename))
            except Exception:  # fisier corupt/incompatibil: mesaj prietenos, fara traceback
                st.error("Modelul nu a putut fi încărcat. Verifică fișierul din folderul `models/`.")
                st.stop()
            start = time.perf_counter()
            top = classify(model, image, spec.image_size, k=3)
            elapsed_ms = (time.perf_counter() - start) * 1000
            best = top[0]

            col_img, col_pred, col_meta = st.columns(3)
            with col_img:
                st.image(image, caption="Imaginea încărcată", width="stretch")
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
                    st.image(str(meta), width="stretch")

with tab_scene:
    render_scene_tab(models_dir, assets_dir)

st.caption("Modelul cunoaște doar cele 43 de semne din setul german GTSRB.")
