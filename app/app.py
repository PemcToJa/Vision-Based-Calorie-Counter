import json
import os
import pathlib
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import cv2
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from ultralytics import YOLO
import mimetypes

mimetypes.add_type('text/css', '.css')
mimetypes.add_type('application/javascript', '.js')

from src.training.food_Quantity_Classifier import FoodMassFusion

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_FILE_PATH = pathlib.Path(__file__).resolve()
APP_DIR = CURRENT_FILE_PATH.parent
PROJECT_ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR
MODEL_DIR = PROJECT_ROOT / "src" / "training" / "models"
NUTRITION_CSV = APP_DIR / "nutrition.csv"

if not STATIC_DIR.exists():
    print(f"Folder dose not exists: {STATIC_DIR}")
else:
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

def load_nutrition_db(csv_path):
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return {}

    df = pd.read_csv(csv_path)

    df.columns = [c.replace('"', '').strip() for c in df.columns]

    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.replace('"', '', regex=False).str.strip()

    numeric_cols = ['weight', 'calories', 'protein', 'carbohydrates', 'fats']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.dropna(subset=['weight', 'calories'])
    df = df[df['weight'] > 0]

    df['cal_1g'] = df['calories'] / df['weight']
    df['cal_1g'] = df['calories'] / df['weight']
    df['prot_1g'] = df['protein'] / df['weight']
    df['carb_1g'] = df['carbohydrates'] / df['weight']
    df['fat_1g'] = df['fats'] / df['weight']

    db = df.groupby('label').agg({
        'cal_1g': 'mean',
        'prot_1g': 'mean',
        'carb_1g': 'mean',
        'fat_1g': 'mean'
    }).to_dict('index')

    return db

class FoodPredictor:
    def __init__(self, classifier_path, fusion_path, class_mapping, nutrition_db):
        self.device = DEVICE
        self.class_mapping = class_mapping
        self.nutrition_db = nutrition_db

        self.yolo_model = YOLO('yolov8n-seg.pt').to(self.device)

        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device).eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

        self.classifier = models.efficientnet_v2_s(weights=None)
        in_features = self.classifier.classifier[1].in_features
        self.classifier.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, 101)
        )
        self.classifier.load_state_dict(torch.load(classifier_path, map_location=self.device))
        self.classifier.eval().to(self.device)

        self.fusion_model = FoodMassFusion()
        self.fusion_model.load_state_dict(torch.load(fusion_path, map_location=self.device))
        self.fusion_model.eval().to(self.device)

        self.final_transform = A.Compose([
            A.LongestMaxSize(max_size=480),
            A.PadIfNeeded(min_height=480, min_width=480, border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ], additional_targets={'depth_image': 'image'})

    def _smart_crop(self, img_bgr):
        h, w, _ = img_bgr.shape
        results = self.yolo_model.predict(img_bgr, classes=[45, 61], conf=0.2, verbose=False)
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            best_idx = torch.argmax(results[0].boxes.conf).item()
            x1, y1, x2, y2 = results[0].boxes.xyxy[best_idx].tolist()
            pw, ph = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
            return img_bgr[max(0, int(y1 - ph)):min(h, int(y2 + ph)), max(0, int(x1 - pw)):min(w, int(x2 + pw))]
        return img_bgr[int(h * 0.1):int(h * 0.9), int(w * 0.1):int(w * 0.9)]

    @torch.no_grad()
    def analyze_image(self, rgb_cv):
        img_bgr = self._smart_crop(rgb_cv)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        input_batch = self.midas_transforms(img_rgb).to(self.device)
        prediction = self.midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1), size=img_rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze().cpu().numpy()

        depth_min, depth_max = prediction.min(), prediction.max()
        depth_norm = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)
        depth_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        transformed = self.final_transform(image=img_rgb, depth_image=depth_rgb)
        rgb_t = transformed["image"].unsqueeze(0).to(self.device)
        depth_t = transformed["depth_image"].unsqueeze(0).to(self.device)

        class_out = self.classifier(rgb_t)
        p_idx = torch.max(class_out, 1)[1].item()
        label = self.class_mapping.get(p_idx, "Unknown")

        mass_out = self.fusion_model(rgb_t, depth_t)
        mass = torch.expm1(mass_out).item()

        nutr = self.nutrition_db.get(label, {"cal_1g": 0, "prot_1g": 0, "carb_1g": 0, "fat_1g": 0})

        return {
            "label": label,
            "mass": round(mass, 1),
            "calories": round(mass * nutr['cal_1g'], 1),
            "protein": round(mass * nutr['prot_1g'], 1),
            "carbs": round(mass * nutr['carb_1g'], 1),
            "fat": round(mass * nutr['fat_1g'], 1)
        }


def get_mappings():
    mapping_path = APP_DIR / "label_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError("No file label_mapping.json")

    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return {int(k): v for k, v in data.items()}


mappings = get_mappings()
nutr_db = load_nutrition_db(NUTRITION_CSV)
predictor = FoodPredictor(
    MODEL_DIR / "Food_Classifier.pth",
    MODEL_DIR / "Fusion-CalorieNet.pth",
    mappings,
    nutr_db
)

@app.get("/")
async def serve_index():
    return FileResponse(APP_DIR / "index.html")

@app.post("/analyze")
async def analyze(
        model_type: str = Form(...),
        rgb_image: UploadFile = File(...)
):
    try:
        rgb_bytes = await rgb_image.read()
        nparr = np.frombuffer(rgb_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="file extenstion not compatable")

        results = predictor.analyze_image(img)

        return {
            "calories": results["calories"],
            "fat": results["fat"],
            "carbs": results["carbs"],
            "protein": results["protein"],
            "product_type": results["label"],
            "mass": results["mass"]
        }
    except Exception as e:
        print(f"error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)