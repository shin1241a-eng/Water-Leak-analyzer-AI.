import os
import tempfile

import gdown
import librosa
import numpy as np
import tensorflow as tf

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS


# =========================
# Flask
# =========================

app = Flask(__name__)
CORS(app)


# =========================
# Model
# =========================

MODEL_PATH = "model_v2.h5"

GOOGLE_DRIVE_FILE_ID = "1qEYZdn-Zm8PhfwaTib2dYlgU9DDajn8w"


# Download model if it does not exist
if not os.path.exists(MODEL_PATH):

    print("📥 Downloading AI model from Google Drive...")

    gdown.download(
        id=GOOGLE_DRIVE_FILE_ID,
        output=MODEL_PATH,
        quiet=False
    )

    print("✅ Model downloaded!")


# Load model
print("🧠 Loading AI model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("✅ AI model loaded!")


# =========================
# Preprocess WAV
# =========================

def preprocess_wav(path):

    SR = 16000
    DURATION = 3
    SAMPLES = SR * DURATION

    # Load audio
    y, sr = librosa.load(
        path,
        sr=SR,
        mono=True
    )

    # Make audio exactly 3 seconds
    if len(y) < SAMPLES:

        y = np.pad(
            y,
            (0, SAMPLES - len(y))
        )

    else:

        y = y[:SAMPLES]


    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )

    # Convert to dB
    mel = librosa.power_to_db(
        mel,
        ref=np.max
    )


    # Fix time dimension to 94
    if mel.shape[1] < 94:

        mel = np.pad(
            mel,
            (
                (0, 0),
                (0, 94 - mel.shape[1])
            ),
            mode="constant"
        )

    else:

        mel = mel[:, :94]


    return mel.astype(np.float32)


# =========================
# Home page
# =========================

@app.route("/")
def home():

    return render_template("index.html")


# =========================
# AI Prediction
# =========================

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

        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav"
        ) as temp:

            file.save(temp.name)

            temp_path = temp.name


        # Preprocess
        mel = preprocess_wav(temp_path)


        # Add batch + channel dimension
        # (128, 94)
        #      ↓
        # (1, 128, 94, 1)

        mel = mel[
            np.newaxis,
            ...,
            np.newaxis
        ]


        # AI prediction
        score = float(
            model.predict(
                mel,
                verbose=0
            )[0][0]
        )


        # Classification
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

        print("❌ Prediction error:", e)

        return jsonify({

            "error": str(e)

        }), 500


    finally:

        # Delete temporary file
        if (
            temp_path
            and os.path.exists(temp_path)
        ):

            os.remove(temp_path)


# =========================
# Start server
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(

        host="0.0.0.0",

        port=port

    )
