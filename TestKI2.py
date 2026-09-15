import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
from pathlib import Path


MODEL_PATH = Path("keras_model.h5")
LABELS_PATH = Path("labels.txt")
IMAGE_SIZE = (224, 224)


st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="centered"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


@st.cache_data
def load_labels():
    labels = []

    with open(LABELS_PATH, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            parts = line.split(maxsplit=1)

            if len(parts) == 2 and parts[0].isdigit():
                labels.append(parts[1])
            else:
                labels.append(line)

    return labels


def predict(image, model, labels):

    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    # Falls das Modell Logits liefert
    if (
        np.any(prediction < 0)
        or not np.isclose(
            np.sum(prediction),
            1.0,
            atol=0.05
        )
    ):
        prediction = tf.nn.softmax(prediction).numpy()

    results = []

    for i in range(
        min(len(labels), len(prediction))
    ):
        results.append(
            (
                labels[i],
                float(prediction[i])
            )
        )

    results.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return results


# -----------------------------
# DESIGN
# -----------------------------

st.markdown("""
<style>

.stApp {
    background:
        radial-gradient(
            circle at top left,
            rgba(124,58,237,.16),
            transparent 30%
        ),
        radial-gradient(
            circle at top right,
            rgba(99,102,241,.12),
            transparent 30%
        ),
        #09090b;
}

.block-container {
    max-width: 850px;
    padding-top: 60px;
}

.hero {
    text-align: center;
    margin-bottom: 35px;
}

.badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 30px;
    background: rgba(124,58,237,.12);
    border: 1px solid rgba(167,139,250,.3);
    color: #c4b5fd;
    font-size: 13px;
    font-weight: 600;
}

.hero h1 {
    font-size: 55px;
    font-weight: 800;
    letter-spacing: -3px;
    margin-bottom: 10px;
    color: white;
}

.hero p {
    color: #a1a1aa;
    font-size: 17px;
}

.result {
    margin-top: 25px;
    padding: 28px;
    border-radius: 24px;
    background: rgba(24,24,27,.8);
    border: 1px solid #3f3f46;
}

.result-small {
    color: #a1a1aa;
    font-size: 13px;
    font-weight: 600;
}

.result-name {
    font-size: 34px;
    font-weight: 800;
    color: white;
    margin-top: 5px;
}

.result-confidence {
    color: #a78bfa;
    font-weight: 600;
    margin-top: 5px;
}

.section {
    color: white;
    font-size: 19px;
    font-weight: 700;
    margin-top: 30px;
    margin-bottom: 15px;
}

.bar-background {
    width: 100%;
    height: 8px;
    background: #27272a;
    border-radius: 20px;
    overflow: hidden;
}

.bar {
    height: 100%;
    border-radius: 20px;
    background: linear-gradient(
        90deg,
        #7c3aed,
        #a78bfa
    );
}

.category {
    color: #e4e4e7;
    font-size: 14px;
    font-weight: 500;
}

.percent {
    color: #a1a1aa;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# HEADER
# -----------------------------

st.markdown("""
<div class="hero">

<div class="badge">
✦ AI IMAGE CLASSIFIER
</div>

<h1>Was ist auf dem Bild?</h1>

<p>
Lade ein Foto hoch und lass die KI erkennen,
was darauf zu sehen ist.
</p>

</div>
""", unsafe_allow_html=True)


# -----------------------------
# UPLOAD
# -----------------------------

uploaded_file = st.file_uploader(
    "📷 Bild auswählen",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        use_container_width=True
    )

    try:

        with st.spinner(
            "🧠 KI analysiert das Bild..."
        ):

            model = load_model()
            labels = load_labels()

            results = predict(
                image,
                model,
                labels
            )


        if not results:
            st.error(
                "Das Modell hat keine Ergebnisse geliefert."
            )
            st.stop()


        best_label = results[0][0]
        best_probability = results[0][1]


        # -----------------------------
        # BESTES ERGEBNIS
        # -----------------------------

        st.markdown(
            f"""
            <div class="result">

                <div class="result-small">
                    WAHRSCHEINLICHSTE ERKENNUNG
                </div>

                <div class="result-name">
                    {best_label}
                </div>

                <div class="result-confidence">
                    {best_probability * 100:.1f}% Wahrscheinlichkeit
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        # -----------------------------
        # ALLE KLASSEN
        # -----------------------------

        st.markdown(
            '<div class="section">Alle Kategorien</div>',
            unsafe_allow_html=True
        )


        for label, probability in results:

            percentage = probability * 100

            st.markdown(
                f"""
                <div style="margin-bottom:16px">

                    <div style="
                        display:flex;
                        justify-content:space-between;
                        margin-bottom:6px;
                    ">

                        <span class="category">
                            {label}
                        </span>

                        <span class="percent">
                            {percentage:.1f}%
                        </span>

                    </div>

                    <div class="bar-background">

                        <div
                            class="bar"
                            style="width:{percentage:.2f}%"
                        ></div>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    except Exception as error:

        st.error(
            "Das Modell konnte nicht geladen werden."
        )

        st.code(
            str(error)
        )
