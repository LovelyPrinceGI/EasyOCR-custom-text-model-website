from flask import Flask

# สร้าง Instance หลักของ Flask Application ของเรา
# นี่คือ 'app' ที่ run.py ตามหาอยู่ค่ะ!
# โค้ดใหม่
app = Flask(__name__, template_folder='../templates', static_folder='../static')

# บอกให้ app ของเรารู้จักกับไฟล์อื่นๆ ที่สำคัญ
# เราจะ import เข้ามาหลังสุดเพื่อป้องกันปัญหา Circular Import นะคะ
from app import routes, ocr_model