import os
import torch
import importlib
from PIL import Image
from torchvision import transforms
from collections import OrderedDict
from easyocr.utils import CTCLabelConverter
import torch.nn.functional as F

ignore_idx = []

# --- Step 1: ฟังก์ชันสำหรับโหลด Character Set จากไฟล์ ---
def load_character_set(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"!!! CRITICAL ERROR: Character set file not found at '{file_path}'")
        return None

# --- Step 2: สร้างการตั้งค่าตามคู่มือ ---
char_file_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'custom_char.txt')
character_set = load_character_set(char_file_path)

if character_set is None:
    raise SystemExit("Application cannot start without the character set file.")

converter = CTCLabelConverter(character_set)
num_class = len(converter.character)

network_params = {
    'input_channel': 1,
    'output_channel': 512,
    'hidden_size': 512,
    'num_class': num_class
}

# --- Step 3: สร้างโมเดลและโหลดไฟล์ .pth (Final Version) ---
model_pkg = importlib.import_module("easyocr.model.model")
model = model_pkg.Model(**network_params)

model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'thai.pth')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"กำลังโหลดโมเดล thai.pth บน {device}...")

# โหลด state_dict จากไฟล์
saved_state_dict = torch.load(model_path, map_location=device)

# --- ส่วนที่แก้ไข ---
# สร้าง state_dict ใหม่เพื่อ "ลอกป้ายชื่อ" module. ออก
# ซึ่งเป็นขั้นตอนที่จำเป็นเมื่อโมเดลถูกเทรนบนหลาย GPU
new_state_dict = OrderedDict()
for k, v in saved_state_dict.items():
    name = k[7:] if k.startswith('module.') else k
    new_state_dict[name] = v
# --------------------

# # ลบเลเยอร์สุดท้ายที่มีขนาดไม่ตรงกันออกจาก state_dict ใหม่
# new_state_dict.pop('Prediction.weight', None)
# new_state_dict.pop('Prediction.bias', None)

# โหลด weights ที่จัดการแล้วเข้าโมเดล
model.load_state_dict(new_state_dict, strict=False)

model = model.to(device)
model.eval()
print("โหลดโมเดลสำเร็จ!")

# --- Step 4: สร้างฟังก์ชันสำหรับทำนายผล ---
# สังเกตว่าขนาด Resize เปลี่ยนเป็น (64, 600) ตามไฟล์ Inference แล้วค่ะ
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 600)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# --- Step 4: สร้างฟังก์ชันสำหรับทำนายผล (Final Version) ---
# --- Step 4: สร้างฟังก์ชันสำหรับทำนายผล (The Ultimate Final Version) ---
def predict_text(image_bytes):
    """
    ใช้โมเดลที่เราโหลดขึ้นมาเพื่ออ่านตัวอักษรจากรูปภาพ
    (เวอร์ชันสมบูรณ์ที่มีขั้นตอน Cleansing)
    """
    image = Image.open(image_bytes).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    batch_size = image_tensor.size(0)
    batch_max_length = 25

    with torch.no_grad():
        text_for_pred = torch.LongTensor(batch_size, batch_max_length + 1).fill_(0).to(device)
        preds = model(image_tensor, text_for_pred)

        probs = F.softmax(preds, dim=2)

        # --- คาถาชำระล้างที่เพิ่มเข้ามา ---
        probs[:, :, ignore_idx] = 0.0
        norm = probs.sum(dim=2, keepdim=True)
        # ป้องกันการหารด้วยศูนย์
        if (norm == 0).any():
            norm[norm == 0] = 1
        probs = probs / norm
        # --------------------------------

        _, preds_index = probs.max(2)
        preds_size = torch.IntTensor([preds.size(1)] * batch_size)
        preds_str = converter.decode_greedy(preds_index.contiguous().cpu().numpy(), preds_size.cpu().numpy())

    return preds_str[0] if preds_str else "ไม่พบข้อความในรูปภาพค่ะ"