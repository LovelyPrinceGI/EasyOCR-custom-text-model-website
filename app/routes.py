from flask import render_template, request, jsonify
from app import app
# import ฟังก์ชันที่เราสร้างไว้
from app.ocr_model import predict_text

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # ตรวจสอบว่ามีไฟล์ถูกส่งมาใน request หรือไม่
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'})
            
        if file:
            # ส่งข้อมูลไฟล์ (bytes) ไปให้ฟังก์ชันของเราประมวลผล
            try:
                predicted_text = predict_text(file.stream)
                return jsonify({'prediction': predicted_text})
            except Exception as e:
                return jsonify({'error': str(e)})

    # ถ้าเป็น GET request ก็แสดงหน้าเว็บปกติ
    return render_template('index.html')