import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path

# ---------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------
MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")
IMAGE_SIZE = (224, 224)

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------
# Modernes Dark-Mode-Design
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at 15% 10%, rgba(99,102,241,.14), transparent 28%),
            radial-gradient(circle at 85% 20%, rgba(139,92,246,.12), transparent 28%),
            #09090b;
        color: #f4f4f5;
    }

    .block-container {
        max-width: 850px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }

    .hero {
        text-align: center;
        margin-bottom: 2rem;
    }

    .badge {
        display: inline-block;
        padding: .45rem .8rem;
        border: 1px solid rgba(139,92,246,.35);
        border-radius: 999px;
        background: rgba(139,92,246,.10);
        color: #c4b5fd;
        font-size: .78rem;
        font-weight: 600;
        letter-spacing: .03em;
        margin-bottom: 1rem;
    }

    .hero h1 {
        font-size: clamp(2.2rem, 7vw, 4rem);
        line-height: 1.05;
        font-weight: 800;
        margin: 0;
        letter-spacing: -.055em;
        background: linear-gradient(90deg, #fff, #c4b5fd);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        color: #a1a1aa;
        font-size: 1.05rem;
        margin-top: .9rem;
    }

    [data-testid="stFileUploader"] {
        background: rgba(24,24,27,.72);
        border: 1px dashed #52525b;
        border-radius: 22px;
        padding: 1.2rem;
        transition: .2s ease;
    }

    [data-testid="stFileUploader"]:hover {
        border-color: #8b5cf6;
        background: rgba(39,39,42,.8);
    }

    .result-card {
        margin-top: 1.5rem;
        padding: 1.7rem;
        border-radius: 24px;
        background: linear-gradient(
            145deg,
            rgba(39,39,42,.90),
            rgba(24,24,27,.90)
        );
        border: 1px solid #3f3f46;
        box-shadow: 0 20px 60px rgba(0,0,0,.25);
    }

    .result-label {
        color: #a1a1aa;
        font-size: .85rem;
        margin-bottom: .4rem;
    }

    .prediction {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -.035em;
        margin-bottom: .3rem;
    }

    .confidence {
        color: #a78bfa;
        font-weight: 600;
        font-size: 1rem;
    }

    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-top: 1.8rem;
        margin-bottom: .8rem;
    }

    .prob-row {
        margin-bottom: .9rem;
    }

    .prob-header {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: .35rem;
        font-size: .9rem;
    }

    .prob-name {
        color: #e4e4e7;
        font-weight: 500;
    }

    .prob-value {
        color: #a1a1aa;
        font-variant-numeric: tabular-nums;
    }

    .bar-bg {
        height: 8px;
        width: 100%;
        border-radius: 999px;
        background: #27272a;
        overflow: hidden;
    }

    .bar-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #7c3aed, #a78bfa);
    }

    .footer {
        text-align: center;
        color: #52525b;
        font-size: .78rem;
        margin-top: 2.5rem;
    }

    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #3f3f46;
        background: #18181b;
        color: #f4f4f5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modell nicht gefunden: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data
def load_labels():
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Labels nicht gefunden: {LABELS_PATH}")

    labels = []
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Unterstützt z.B.:
            # 0 Kleidungsstücke
            # 1 Kleidungsstücke
            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1].strip())
            else:
                labels.append(line)

    return labels


def predict(image, model, labels):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    # Standardmäßige Bildvorverarbeitung für ein Keras-Klassifikationsmodell.
    # Falls dein Modell beim Training eine andere Normalisierung verwendet,
    # kann preprocess() hier angepasst werden.
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    predictions = model.predict(array, verbose=0)
    predictions = np.asarray(predictions).squeeze()

    # Falls das Modell Logits statt Wahrscheinlichkeiten ausgibt.
    if np.any(predictions < 0) or not np.isclose(np.sum(predictions), 1.0, atol=0.05):
        exp = np.exp(predictions - np.max(predictions))
        predictions = exp / np.sum(exp)

    predictions = np.clip(predictions, 0, 1)
    if predictions.sum() > 0:
        predictions = predictions / predictions.sum()

    count = min(len(labels), len(predictions))
    results = [
        (labels[i], float(predictions[i]))
        for i in range(count)
    ]
    results.sort(key=lambda x: x[1], reverse=True)

    return results


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="badge">✦ AI IMAGE CLASSIFIER</div>
    <h1>Was ist auf dem Bild?</h1>
    <p>Lade ein Foto hoch und lass dein trainiertes KI-Modell es erkennen.</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Bild hochladen",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible",
)

if uploaded_file is None:
    st.info("📷 Ziehe ein Bild hierher oder wähle eine Datei aus.", icon="ℹ️")

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Hochgeladenes Bild",
            use_container_width=True,
        )

        with st.spinner("🧠 KI analysiert das Bild..."):
            model = load_model()
            labels = load_labels()
            results = predict(image, model, labels)

        if not results:
            st.error("Das Modell hat keine Klassen zurückgegeben.")
            st.stop()

        best_label, best_probability = results[0]

        st.markdown(f"""
        <div class="result-card">
            <div class="result-label">WAHRSCHEINLICHSTE ERKENNUNG</div>
            <div class="prediction">{best_label}</div>
            <div class="confidence">{best_probability * 100:.1f}% Wahrscheinlichkeit</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title">Alle Ergebnisse</div>',
            unsafe_allow_html=True
        )

        for label, probability in results:
            percentage = probability * 100

            st.markdown(f"""
            <div class="prob-row">
                <div class="prob-header">
                    <span class="prob-name">{label}</span>
                    <span class="prob-value">{percentage:.1f}%</span>
                </div>
                <div class="bar-bg">
                    <div class="bar-fill" style="width:{percentage:.2f}%"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        st.error(
            "Beim Laden des Modells oder bei der Analyse ist ein Fehler aufgetreten."
        )
        st.exception(e)

st.markdown(
    '<div class="footer">Powered by your trained Keras model · Streamlit</div>',
    unsafe_allow_html=True
)
