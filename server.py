from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch
import sqlite3
import io
from typing import Optional

app = FastAPI()

# CORS sozlamalari
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model va processorni yuklab olish
model = ViTForImageClassification.from_pretrained(
    "Hemg/New-plant-diseases-classification",
    token="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"
)

processor = ViTImageProcessor.from_pretrained(
    "Hemg/New-plant-diseases-classification",
    token="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"
)

# Kasallik bashorati endpoint
@app.post("/predict_plant_disease")
async def predict_disease(file: UploadFile = File(...)):
    try:
        # Faylni o‘qish va rasmga aylantirish
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Modelga tayyorlash
        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            predicted_label = model.config.id2label[predicted_class_idx]

        # Ma’lumotlar bazasidan ma’lumot olish
        return controllers.image_about(predicted_label)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Xatolik: {str(e)}")


# Kasallik haqida ma’lumot olish uchun controller
class controllers:

    @staticmethod
    def image_about(bar_code: str) -> Optional[dict]:
        """ Model qaytargan label asosida bazadan ma’lumot olish """
        try:
            conn = sqlite3.connect('data/About_plant_qq.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM diseases WHERE name = ?", (bar_code,))
            result = cursor.fetchone()
            conn.close()

            if result:
                return controllers.format_image_about_result(result)
            return {"message": f"{bar_code} bo‘yicha hech qanday ma’lumot topilmadi."}
        except Exception as e:
            return {"error": f"Bazada izlashda xatolik: {str(e)}"}

    @staticmethod
    def format_image_about_result(result):
        """ Natijani JSON formatga keltirish """
        return {
            'name': result[0].replace("(", " ").replace(")", " ").replace("_", " "),
            'About_disease': result[1],
            'Origin': result[2],
            'Suggestions': result[3],
            'Prevent': result[4],
        }
