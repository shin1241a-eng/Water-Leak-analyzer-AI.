import os
import urllib.request
import numpy as np
import librosa
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL_PATH = "model_v2.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download(
        "GOOGLE_DRIVE_FILE_URL",
        MODEL_PATH,
        quiet=False
    )

model = load_model(MODEL_PATH)

# โหลดโมเดลครั้งเดียวตอนเริ่มเซิร์ฟเวอร์
if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from HuggingFace...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

print("🧠 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, -1)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    features = extract_features(filepath)
    prediction = model.predict(features)
    label = int(np.argmax(prediction))

    classes = ["leak", "no_leak"]
    return jsonify({"prediction": classes[label]})

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
