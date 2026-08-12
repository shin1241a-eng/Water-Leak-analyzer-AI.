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

import os
import librosa
import numpy as np

SR = 16000
DURATION = 3        # วินาที
SAMPLES = SR * DURATION
def load_audio(path):
    y, sr = librosa.load(path, sr=SR, mono=True)

    if len(y) < SAMPLES:
        y = np.pad(y, (0, SAMPLES - len(y)))
    else:
        y = y[:SAMPLES]

    return y
def extract_mel(y):
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=SR,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel)
    return mel_db
X = []
y = []

base_path = "dataset/all"

label_map = {
    "normal": 0,   # No Leak (MIMII Valve)
    "leak": 1      # Leak
}

for label_name, label in label_map.items():
    folder = os.path.join(base_path, label_name)

    for file in os.listdir(folder):
        if file.endswith(".wav"):
            path = os.path.join(folder, file)
            audio = load_audio(path)
            feature = extract_mel(audio)

            X.append(feature)
            y.append(label)
X = np.array(X)
y = np.array(y)

# เพิ่ม channel dimension (CNN ต้องใช้)
X = X[..., np.newaxis]

print(X.shape)  # (samples, 128, time, 1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=X_train.shape[1:]),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation="relu"),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),

    Dense(1, activation="sigmoid")   # ❗ 1 neuron = Leak / No Leak
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()
history = model.fit(
    X_train, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2
)
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)
import librosa
import numpy as np

def preprocess_wav(path):
    y, sr = librosa.load(path, sr=16000, mono=True)

    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        n_fft=1024,
        hop_length=512
    )
    mel = librosa.power_to_db(mel, ref=np.max)

    # fix time axis = 94
    if mel.shape[1] < 94:
        mel = np.pad(mel, ((0,0),(0,94-mel.shape[1])), mode='constant')
    else:
        mel = mel[:, :94]

    return mel.astype(np.float32)def predict_wav(path, model):
    mel = preprocess_wav(path)
    mel = mel[np.newaxis, ..., np.newaxis]  # (1,128,94,1)
    prob = model.predict(mel)[0][0]
    return probscore = predict_wav(r, model)

if score > 0.5:
    print("Prediction: LEAK")
else:
    print("Prediction: NO LEAK")

print("Confidence:", score)
