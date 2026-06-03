import streamlit as st
import cv2
import numpy as np
import tempfile
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="Deepfake Video Detection",
    page_icon="🎭",
    layout="wide"
)

st.markdown("""
<style>

.main {
    background-color: #0B1120;
}

h1 {
    text-align:center;
}

.result-card-real {
    background: linear-gradient(135deg,#11998e,#38ef7d);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
}

.result-card-fake {
    background: linear-gradient(135deg,#cb2d3e,#ef473a);
    padding:40px;
    border-radius:20px;
    text-align:center;
    color:white;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
}

.big-result {
    font-size:42px;
    font-weight:bold;
}

.confidence {
    font-size:28px;
    margin-top:15px;
}

.about-box {
    padding:20px;
    border-radius:15px;
    background-color:#111827;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1>🎭 Deepfake Video Detection</h1>
<p style='text-align:center;font-size:18px;'>
Upload a video and detect whether it is REAL or FAKE
</p>
""", unsafe_allow_html=True)

st.divider()

st.markdown("""
<div class="about-box">

### 📌 About This Project

This Deepfake Detection System uses:

✅ TensorFlow

✅ OpenCV

✅ Deep Learning

The model extracts frames from uploaded videos and predicts whether the content is manipulated or authentic.

</div>
""", unsafe_allow_html=True)

st.divider()


@st.cache_resource
def load_my_model():
    return load_model("deepfake_model.h5")

model = load_my_model()

IMG_SIZE = 64

def extract_frames(video_path):

    frames = []

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % 20 == 0:

            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame / 255.0

            frames.append(frame)

        frame_count += 1

    cap.release()

    return np.array(frames)

def predict_video(video_path):

    frames = extract_frames(video_path)

    if len(frames) == 0:
        return "ERROR", 0

    predictions = model.predict(frames, verbose=0)

    avg_prediction = np.mean(predictions, axis=0)

    real_score = float(avg_prediction[0])
    fake_score = float(avg_prediction[1])

    if real_score > fake_score:
        return "REAL VIDEO", real_score * 100
    else:
        return "FAKE VIDEO", fake_score * 100


col1, col2 = st.columns([1.2, 1])

with col1:

    uploaded_file = st.file_uploader(
        "Upload Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:

        st.video(uploaded_file)

with col2:

    st.write("")
    st.write("")
    st.write("")

    detect = st.button(
        "🔍 Detect Deepfake",
        use_container_width=True
    )

if uploaded_file and detect:

    temp_file = tempfile.NamedTemporaryFile(delete=False)

    temp_file.write(uploaded_file.read())

    video_path = temp_file.name

    with st.spinner("Analyzing video..."):

        result, confidence = predict_video(video_path)

    st.markdown("<br>", unsafe_allow_html=True)

    if result == "REAL VIDEO":

        st.markdown(f"""
        <div class="result-card-real">

        <div class="big-result">
        ✅ REAL VIDEO
        </div>

        <div class="confidence">
        Confidence: {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

    elif result == "FAKE VIDEO":

        st.markdown(f"""
        <div class="result-card-fake">

        <div class="big-result">
        ⚠️ FAKE VIDEO
        </div>

        <div class="confidence">
        Confidence: {confidence:.2f}%
        </div>

        </div>
        """, unsafe_allow_html=True)

st.divider()

st.markdown(
    """
    <center>
    <h4>Deepfake Video Detection System</h4>
    <p>Built with TensorFlow • OpenCV • Streamlit</p>
    </center>
    """,
    unsafe_allow_html=True
)