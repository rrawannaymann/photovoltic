from ultralytics import YOLO
from PIL import Image
from torchvision import transforms
import numpy as np
import os
from fpdf import FPDF

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

def generate_report(image_path, model_choice): 
    if not os.path.exists(image_path):
        return f"File not found: {image_path}"

    # تحديد اسم الموديل
    model_path = 'models/best_multi.pt' if model_choice == 'multi' else 'models/best.pt'
    
    # تحميل النموذج
    model = YOLO(model_path)

    # تحميل الصورة الأصلية
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    # التنبؤ
    results = model(image_tensor)

    # الحصول على صورة التتبع
    tracked_image = results[0].plot()
    tracked_image_pil = Image.fromarray(tracked_image)

    # تعديل حجم الصورة لتكون بنفس حجم الصورة الأصلية
    tracked_image_pil = tracked_image_pil.resize(image.size)

    # حفظ الصورة
    filename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(filename)[0]
    output_image_name = f"{name_without_ext}_tracked.jpg"
    output_image_path = os.path.join("static/outputs", output_image_name)
    tracked_image_pil.save(output_image_path)

    # إعداد التقرير
    report = f"The tracked image has been saved at {output_image_path}"
    original_image_path = image_path.replace("static/", "")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.multi_cell(0, 10, report)
    pdf_bytes = pdf.output(dest='S').encode('latin1')

    return report, original_image_path, output_image_path.replace("static/", "")