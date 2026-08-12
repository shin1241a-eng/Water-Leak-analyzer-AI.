import os
import tempfile

import librosa
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

MODEL_PATH = "model_v2.h5"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded!")


def preprocess_wav(path):

    y, sr = librosa.load(
        path,
        sr=16000,
        mono=True
    )

    # 3 seconds
    samples = 16000 * 3

    if len(y) < samples:
        y = np.pad(
            y,
            (0, samples - len(y))
        )
    else:
        y = y[:samples]

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=16000,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )

    # Fix shape = 128 x 94
    if mel.shape[1] < 94:
        mel = np.pad(
            mel,
            ((0, 0), (0, 94 - mel.shape[1])),
            mode="constant"
        )
    else:
        mel = mel[:, :94]

    return mel.astype(np.float32)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({
            "error": "No file uploaded"
        }), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "error": "No file selected"
        }), 400

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp:

            file.save(temp.name)
            temp_path = temp.name

        # Preprocess
        mel = preprocess_wav(temp_path)

        # (128,94) → (1,128,94,1)
        mel = mel[np.newaxis, ..., np.newaxis]

        # Predict
        score = float(
            model.predict(mel, verbose=0)[0][0]
        )

        if score > 0.5:
            prediction = "leak"
            confidence = score
        else:
            prediction = "no_leak"
            confidence = 1 - score

        return jsonify({
            "prediction": prediction,
            "confidence": confidence
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
