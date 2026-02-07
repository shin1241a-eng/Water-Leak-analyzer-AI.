from flask import Flask, render_template, request, jsonify
import numpy as np
import librosa
import os

app = Flask(__name__)

model = None  # ยังไม่โหลดตอนเปิดเซิร์ฟเวอร์

# 🔥 ฟังก์ชันโหลดโมเดลเมื่อจำเป็นเท่านั้น
def get_model():
    global model
    if model is None:
        print("📥 Loading model for first time...")
        from tensorflow.keras.models import load_model
        model = load_model("model_v2.h5")
        print("✅ Model loaded!")
    return model


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    filepath = "temp.wav"
    file.save(filepath)

    # โหลดเสียง
    y, sr = librosa.load(filepath, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)

    # 🔥 โหลดโมเดลตอนนี้แหละ
    model = get_model()

    prediction = model.predict(np.expand_dims(mfcc, axis=0))
    result = "Leak Detected" if prediction[0][0] > 0.5 else "No Leak"

    os.remove(filepath)
    return jsonify({"result": result})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
