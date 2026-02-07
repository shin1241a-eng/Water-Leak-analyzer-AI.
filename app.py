from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)  # อนุญาตให้เว็บเรียกข้ามโดเมนได้

@app.route('/')
def home():
    return "Water Leak AI server is running"

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

