from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import tensorflow as tf
import gdown
import os

app = Flask(__name__)

# ================== โหลดโมเดลจาก Google Drive ==================
MODEL_PATH = "model_v2.h5"

if not os.path.exists(MODEL_PATH):
    print("📥 Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1qEYZdn-Zm8PhfwaTib2dYlgU9DDajn8w"

    gdown.download(url, MODEL_PATH, quiet=False)

print("🤖 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!")

# ================== ฟังก์ชันแปลงเสียง ==================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled.reshape(1, -1)

# ================== หน้าเว็บหลัก ==================
@app.route("/")
def home():
    return render_template("index.html")

# ================== ทำนายเสียง ==================
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    features = extract_features(filepath)
    prediction = model.predict(features)[0][0]

    os.remove(filepath)

    result = "Leak Detected 🚨" if prediction > 0.5 else "No Leak ✅"
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(debug=True)
