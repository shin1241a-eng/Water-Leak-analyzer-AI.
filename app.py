import os
import numpy as np
import librosa
import tensorflow as tf
import gdown
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ================== โหลดโมเดลจาก Google Drive ==================
MODEL_PATH = "model.h5"
FILE_ID = "1qEYZdn-Zm8PhfwaTib2dYlgU9DDajn8w"  # 🔥 ใส่ Google Drive FILE ID ตรงนี้
MODEL_URL = "https://drive.google.com/uc?id=1qEYZdn-Zm8PhfwaTib2dYlgU9DDajn8w"


if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("Loading AI model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

# ================== ฟังก์ชันแปลงเสียง ==================
def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=22050)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled.reshape(1, -1)

# ================== หน้าเว็บหลัก ==================
@app.route("/")
def home():
    return render_template("index.html")

# ================== วิเคราะห์เสียง ==================
@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    features = extract_features(filepath)
    prediction = model.predict(features)[0][0]

    os.remove(filepath)

    result = "Leak Detected" if prediction > 0.5 else "No Leak Detected"
    confidence = float(prediction)

    return jsonify({
        "result": result,
        "confidence": round(confidence * 100, 2)
    })

# ================== รันบน Render ==================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)

