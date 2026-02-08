import os
import numpy as np
import librosa
import tensorflow as tf
import requests
from flask import Flask, request, render_template

app = Flask(__name__)

MODEL_PATH = "model_v2.h5"
MODEL_URL = "https://huggingface.co/getsuck/water_leak_ai/resolve/main/model_v2.h5"

model = None  # ยังไม่โหลดตอนเปิดเว็บ


# 📥 โหลดโมเดลเมื่อจำเป็น
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("📥 Downloading model from HuggingFace...")
        r = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("✅ Model downloaded")


def load_model():
    global model
    if model is None:
        download_model()
        print("🧠 Loading AI model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    return model


# 🎵 แปลงเสียงเป็น features แบบเดียวกับที่เทรน
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_scaled, axis=0)


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = "temp.wav"
            file.save(filepath)

            features = extract_features(filepath)
            ai_model = load_model()
            result = ai_model.predict(features)

            classes = ["No Leak", "Leak"]
            prediction = classes[np.argmax(result)]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
