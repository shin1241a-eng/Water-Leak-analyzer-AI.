import tensorflow as tf
import numpy as np
import librosa
import os
import gdown
import tensorflow as tf

MODEL_PATH = "model_v2.h5"

# ถ้ายังไม่มีไฟล์โมเดล ให้โหลดจาก Google Drive
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    url = "https://drive.google.com/uc?id=1qEYZdn-Zm8PhfwaTib2dYlgU9DDajn8w"
    gdown.download(url, MODEL_PATH, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

model = tf.keras.models.load_model("model_v2.h5")

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

    try:
        features = extract_features(filepath)
        prediction = model.predict(features)
        score = float(prediction[0][0])

        if score > 0.5:
            result = f"พบเสียงน้ำรั่ว (ความมั่นใจ {score:.2f})"
        else:
            result = f"ไม่พบเสียงน้ำรั่ว (ความมั่นใจ {score:.2f})"

        return jsonify({"result": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    # บันทึกไฟล์ชั่วคราว
    filepath = os.path.join("temp.wav")
    file.save(filepath)

    # ====== ตรงนี้ใส่โค้ด AI ของเธอ ======
    # เช่น โหลดไฟล์เสียง → extract feature → predict

    result = "Water leak detected"  # ตอนนี้ใช้ผลลัพธ์ตัวอย่างไปก่อน

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)




# รันเว็บ
if __name__ == "__main__":
    app.run(debug=True)




