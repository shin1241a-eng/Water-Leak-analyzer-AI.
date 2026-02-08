import os
import numpy as np
import librosa
import soundfile as sf
from flask import Flask, request, render_template
import tensorflow as tf

app = Flask(__name__)

# ===================== โหลดโมเดล TFLite =====================
interpreter = tf.lite.Interpreter(model_path="model_v2.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ===================== ฟังก์ชันแปลงเสียงเป็น Features =====================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    return mfcc_scaled

# ===================== ฟังก์ชันทำนาย =====================
def predict_audio(features):
    input_data = np.array(features, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return output_data

# ===================== หน้าเว็บหลัก =====================
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", prediction="No file uploaded")

        file = request.files["file"]

        if file.filename == "":
            return render_template("index.html", prediction="No selected file")

        filepath = os.path.join("temp.wav")
        file.save(filepath)

        try:
            features = extract_features(filepath)
            result = predict_audio(features)

            class_index = np.argmax(result)

            if class_index == 0:
                prediction = "🚰 Leak Detected"
            else:
                prediction = "✅ No Leak"

        except Exception as e:
            prediction = f"Error: {str(e)}"

        if os.path.exists(filepath):
            os.remove(filepath)

    return render_template("index.html", prediction=prediction)

# ===================== รันเซิร์ฟเวอร์ =====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
