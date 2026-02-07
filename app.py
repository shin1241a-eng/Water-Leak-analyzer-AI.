from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import librosa
import os
import gdown

app = Flask(__name__)

MODEL_PATH = "model_v2.h5"

# ดาวน์โหลดโมเดลจาก Google Drive ถ้ายังไม่มี
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    url = "https://drive.google.com/uc?id=1qEYZdn-Zm8PhfwaTib2dYlgU9DDajn8w"
    gdown.download(url, MODEL_PATH, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc.reshape(1, -1)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    filepath = "temp.wav"
    file.save(filepath)

    features = extract_features(filepath)
    prediction = model.predict(features)[0][0]

    if prediction > 0.5:
        result = f"พบเสียงน้ำรั่ว (ความมั่นใจ {prediction:.2f})"
    else:
        result = f"ไม่พบเสียงน้ำรั่ว (ความมั่นใจ {prediction:.2f})"

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
