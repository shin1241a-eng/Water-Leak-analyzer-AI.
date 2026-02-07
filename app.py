from flask import Flask, render_template, request, jsonify
import os
import librosa
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# โฟลเดอร์เก็บไฟล์ที่อัปโหลด
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# โหลดโมเดล AI ครั้งเดียวตอนเปิดเว็บ
model = load_model("model_v2.h5")


# ================= AI PART =================
SR = 16000
DURATION = 3
SAMPLES = SR * DURATION

def preprocess_wav(path):
    y, sr = librosa.load(path, sr=SR, mono=True)

    # ทำให้ยาว 3 วินาทีเท่าตอนเทรน
    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    mel = librosa.power_to_db(mel)  # เอา ref=np.max ออกให้เหมือนตอนเทรน

    # fix time axis = 94 เหมือนใน Notebook
    if mel.shape[1] < 94:
        mel = np.pad(mel, ((0, 0), (0, 94 - mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :94]

    return mel.astype(np.float32)


def predict_wav(path):
    mel = preprocess_wav(path)
    mel = mel[np.newaxis, ..., np.newaxis]  # เพิ่ม batch + channel
    prob = model.predict(mel)[0][0]
    return prob


def run_ai_model(filepath):
    score = predict_wav(filepath)

    if score > 0.5:
        return "พบเสียงน้ำรั่ว (LEAK)"
    else:
        return "ไม่พบเสียงน้ำรั่ว (NORMAL)"


# ================= WEB PART =================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    files = request.files.getlist("files")  # รับหลายไฟล์
    results = []

    for file in files:
        if file.filename == "":
            continue

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        result = run_ai_model(filepath)

        results.append({
            "filename": file.filename,
            "result": result
        })

    return jsonify(results)



# รันเว็บ
if __name__ == "__main__":
    app.run(debug=True)
