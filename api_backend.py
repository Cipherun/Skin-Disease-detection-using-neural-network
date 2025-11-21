# api_backend.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn
import torch
import timm
import numpy as np
from PIL import Image
import json
import joblib
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ----------------- Config -----------------
MODEL_PATH = "best_skin_model.pth"
CLASSES_JSON = "classes.json"
LABEL_ENCODER_PKL = "label_encoder.pkl"
IMG_SIZE = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------- Load classes -----------------
with open(CLASSES_JSON, "r") as f:
    classes = json.load(f)  # ['akiec','bcc','bkl','df','mel','nv','vasc']

# (Optional) label encoder – in case you need it later
le = joblib.load(LABEL_ENCODER_PKL)

# ----------------- Load model -----------------
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

# ----------------- Preprocessing -----------------
transform = Compose([
    Resize(IMG_SIZE, IMG_SIZE),
    Normalize(mean=(0.485, 0.456, 0.406),
              std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ----------------- Disease Info (HAM10000) -----------------
DISEASE_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Bowen’s Disease",
        "description": "Precancerous or in-situ squamous cell carcinoma of the skin.",
        "cause": "Often caused by long-term sun (UV) exposure damaging skin cells.",
        "prevention": "Use sunscreen, avoid intense sun, wear protective clothing, and get regular skin checks."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "The most common type of skin cancer, usually slow growing and locally invasive.",
        "cause": "Chronic UV exposure, fair skin, history of sunburns, older age.",
        "prevention": "Limit sun exposure, use broad-spectrum sunscreen, avoid tanning beds, and monitor new lesions."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "description": "Includes seborrheic keratoses, solar lentigines, and similar benign lesions.",
        "cause": "Aging skin, sun exposure, genetic factors.",
        "prevention": "Sun protection, regular skin self-exams to monitor changes."
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Benign fibrous skin nodule, usually small and firm.",
        "cause": "Often follows minor skin trauma or insect bites; exact cause unclear.",
        "prevention": "No specific prevention; avoid scratching/trauma to skin when possible."
    },
    "mel": {
        "name": "Melanoma",
        "description": "A serious, potentially life-threatening skin cancer of melanocytes.",
        "cause": "Intense or intermittent UV exposure, sunburns, genetic predisposition, many or atypical moles.",
        "prevention": "Strong sun protection, avoid tanning beds, early detection (ABCDE rule), regular dermatologist checks."
    },
    "nv": {
        "name": "Melanocytic Nevus (Mole)",
        "description": "Common pigmented benign moles arising from melanocytes.",
        "cause": "Normal skin development, genetics, and some UV influence.",
        "prevention": "General sun safety, monitor for changes in size, shape, color, or symptoms."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "Includes angiomas, angiokeratomas, pyogenic granulomas, and hemorrhages.",
        "cause": "Abnormal growths of blood vessels, trauma, or congenital factors.",
        "prevention": "No specific prevention, but protecting skin from trauma may help reduce some lesions."
    }
}

# ----------------- FastAPI App -----------------
app = FastAPI(title="Skin Disease Detection API",
              description="HAM10000-based skin lesion classifier. Not a medical diagnosis.",
              version="1.0")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and decode image
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        # Preprocess
        img_t = transform(image=img_np)["image"].unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            out = model(img_t)
            pred_idx = int(out.argmax(1).item())
            conf = float(torch.softmax(out, 1)[0][pred_idx].item())

        class_id = classes[pred_idx]
        info = DISEASE_INFO.get(class_id, {
            "name": class_id,
            "description": "Information not available.",
            "cause": "Unknown.",
            "prevention": "Please consult a dermatologist."
        })

        return {
            "class_id": class_id,
            "label": info["name"],
            "confidence": conf,
            "description": info["description"],
            "cause": info["cause"],
            "prevention": info["prevention"]
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# Needed for BytesIO
from io import BytesIO

if __name__ == "__main__":
    uvicorn.run("api_backend:app", host="127.0.0.1", port=8000, reload=True)
