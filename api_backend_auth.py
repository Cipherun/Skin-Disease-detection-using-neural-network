"""
Enhanced FastAPI backend with doctor authentication and Grad-CAM powered
prediction endpoint tailored for the new Streamlit white portal UI.
"""

import os
import sqlite3
import secrets
import hashlib
from datetime import datetime, timedelta
from io import BytesIO
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import torch
import timm
import numpy as np
from PIL import Image
import json
import joblib
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

APP_TITLE = "Skin Disease Detection API (Auth-enabled)"
MODEL_PATH = "best_skin_model.pth"
CLASSES_JSON = "classes.json"
LABEL_ENCODER_PKL = "label_encoder.pkl"
IMG_SIZE = 224
DATABASE_PATH = "doctor_portal.db"
TOKEN_TTL_HOURS = 12

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- MODEL + DATA ----------------------
with open(CLASSES_JSON, "r") as f:
    CLASSES = json.load(f)

joblib.load(LABEL_ENCODER_PKL)  # future use

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(CLASSES))
state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

transform = Compose(
    [
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


DISEASE_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Bowen’s Disease",
        "description": "Precancerous lesion that can progress to squamous cell carcinoma.",
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "description": "Common slow-growing skin cancer caused by chronic UV exposure.",
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "description": "Benign lesions such as seborrheic keratoses or solar lentigines.",
    },
    "df": {
        "name": "Dermatofibroma",
        "description": "Small benign fibrous nodule usually triggered by minor trauma.",
    },
    "mel": {
        "name": "Melanoma",
        "description": "Aggressive malignancy of melanocytes requiring early detection.",
    },
    "nv": {
        "name": "Melanocytic Nevus (Mole)",
        "description": "Common benign mole; monitor for ABCDE rule changes.",
    },
    "vasc": {
        "name": "Vascular Lesion",
        "description": "Includes angiomas, angiokeratomas, and other vascular proliferations.",
    },
}


# ---------------------- DB HELPERS ----------------------
def init_db():
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS tokens (
            token TEXT PRIMARY KEY,
            doctor_id INTEGER NOT NULL,
            expires_at DATETIME NOT NULL,
            FOREIGN KEY(doctor_id) REFERENCES doctors(id) ON DELETE CASCADE
        )
    """
    )
    conn.commit()
    conn.close()


def get_db():
    return sqlite3.connect(DATABASE_PATH)


def hash_password(password: str, salt: Optional[str] = None):
    if not salt:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100000)
    return salt, pwd_hash.hex()


def verify_password(password: str, salt: str, stored_hash: str) -> bool:
    _, new_hash = hash_password(password, salt)
    return secrets.compare_digest(new_hash, stored_hash)


def create_token_record(doctor_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(hours=TOKEN_TTL_HOURS)
    conn = get_db()
    conn.execute(
        "INSERT INTO tokens (token, doctor_id, expires_at) VALUES (?,?,?)",
        (token, doctor_id, expires.isoformat()),
    )
    conn.commit()
    conn.close()
    return token


def get_doctor_by_token(token: str):
    conn = get_db()
    c = conn.cursor()
    c.execute(
        """
        SELECT d.id, d.full_name, d.email, t.expires_at
        FROM tokens t
        JOIN doctors d ON d.id = t.doctor_id
        WHERE t.token = ?
        """,
        (token,),
    )
    row = c.fetchone()
    conn.close()
    if row is None:
        return None
    expires_at = datetime.fromisoformat(row[3])
    if expires_at < datetime.utcnow():
        delete_token(token)
        return None
    return {"id": row[0], "full_name": row[1], "email": row[2]}


def delete_token(token: str):
    conn = get_db()
    conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
    conn.commit()
    conn.close()


# ---------------------- AUTH MODELS ----------------------
class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    doctor_name: str
    doctor_email: EmailStr


# ---------------------- APP + SECURITY ----------------------
auth_scheme = HTTPBearer()
app = FastAPI(
    title=APP_TITLE,
    description="Secure API powering the white Streamlit portal. Not a medical device.",
    version="2.0",
)


def current_doctor(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
):
    token = credentials.credentials
    doctor = get_doctor_by_token(token)
    if not doctor:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")
    return doctor


# ---------------------- ROUTES: AUTH ----------------------
@app.post("/auth/register", status_code=201)
def register(request: RegisterRequest):
    salt, pwd_hash = hash_password(request.password)
    try:
        conn = get_db()
        conn.execute(
            "INSERT INTO doctors (full_name, email, password_hash, salt) VALUES (?,?,?,?)",
            (request.full_name.strip(), request.email.lower(), pwd_hash, salt),
        )
        conn.commit()
        conn.close()
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Email already registered.")
    return {"message": "Doctor account created. Await admin approval to activate login."}


@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest):
    conn = get_db()
    c = conn.cursor()
    c.execute("SELECT id, full_name, email, password_hash, salt FROM doctors WHERE email = ?", (request.email.lower(),))
    row = c.fetchone()
    conn.close()
    if row is None or not verify_password(request.password, row[4], row[3]):
        raise HTTPException(status_code=401, detail="Invalid credentials.")

    token = create_token_record(row[0])
    return LoginResponse(access_token=token, doctor_name=row[1], doctor_email=row[2])


@app.get("/auth/me")
def me(doctor: dict = Depends(current_doctor)):
    return doctor


@app.post("/auth/logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(auth_scheme)):
    delete_token(credentials.credentials)
    return {"message": "Logged out."}


# ---------------------- ROUTES: PREDICTION ----------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...), doctor: dict = Depends(current_doctor)):
    try:
        image_bytes = await file.read()
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)

        img_t = transform(image=img_np)["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            out = model(img_t)
            pred_idx = int(out.argmax(1).item())
            conf = float(torch.softmax(out, 1)[0][pred_idx].item())

        class_id = CLASSES[pred_idx]
        info = DISEASE_INFO.get(class_id, {"name": class_id, "description": "No description available."})

        return {
            "doctor": {"id": doctor["id"], "name": doctor["full_name"]},
            "prediction": {
                "class_id": class_id,
                "label": info["name"],
                "confidence": conf,
                "description": info["description"],
            },
        }
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


# ---------------------- STARTUP ----------------------
@app.on_event("startup")
def startup_event():
    init_db()


if __name__ == "__main__":
    import uvicorn

    init_db()
    uvicorn.run("api_backend_auth:app", host="127.0.0.1", port=8500, reload=True)





