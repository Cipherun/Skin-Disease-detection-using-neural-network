import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import json
import joblib
import cv2
import requests
import sqlite3
import hashlib
import secrets
from datetime import datetime
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2


# ---------------------- APP CONFIG ----------------------
st.set_page_config(
    page_title="Skin Disease Detection – Modern UI",
    page_icon="🩺",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    section[data-testid="stSidebar"] {
        display: none !important;
    }
    body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background: #050505 !important;
    }
    body, p, div, span, label {
        color: #f5f5f5 !important;
    }
    .hero {
        padding: 3rem;
        border-radius: 18px;
        background: linear-gradient(135deg,#141414 0%,#050505 100%);
        border: 1px solid #1f2933;
    }
    .panel {
        padding: 2rem;
        border-radius: 18px;
        border: 1px solid #1f2933;
        background: #0f0f0f;
        box-shadow: 0 6px 30px rgba(0,0,0,0.65);
    }
    .pill {
        padding: 0.35rem 1rem;
        border-radius: 999px;
        border: 1px solid #333;
        font-size: 0.85rem;
        color: #a5b4fc;
        background: #1f1f1f;
    }
    .nav-btn {
        border-radius: 999px !important;
        border: 1px solid #27272a;
        background: #0f0f0f;
        color: #f5f5f5;
        font-weight: 600;
        width: 100%;
    }
    .nav-btn.active {
        background: #ffffff;
        color: #050505;
        border-color: #ffffff;
    }
    .quick-access button {
        border-radius: 18px !important;
        padding: 1rem !important;
        border: 1px solid #27272a !important;
        background: #111 !important;
        color: #f5f5f5 !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------- SESSION STATE ----------------------
if "active_page" not in st.session_state:
    st.session_state.active_page = "Home"
if "doctor_options_open" not in st.session_state:
    st.session_state.doctor_options_open = False
if "doctor_authenticated" not in st.session_state:
    st.session_state.doctor_authenticated = False
if "recent_predictions" not in st.session_state:
    st.session_state.recent_predictions = []
if "auth_token" not in st.session_state:
    st.session_state.auth_token = None
if "doctor_info" not in st.session_state:
    st.session_state.doctor_info = None

# Backend API URL
API_BASE_URL = "http://127.0.0.1:8500"
LOCAL_DB_PATH = "doctors_local.db"

# Initialize local database for fallback authentication
def init_local_auth_db():
    conn = sqlite3.connect(LOCAL_DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS doctors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

init_local_auth_db()


# ---------------------- MODEL + GRADCAM ----------------------
IMG_SIZE = 224


@st.cache_resource
def load_model_and_utils():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open("classes.json", "r") as f:
        classes = json.load(f)

    label_encoder = joblib.load("label_encoder.pkl")
    model = timm.create_model(
        "efficientnet_b0", pretrained=False, num_classes=len(classes)
    )
    state_dict = torch.load("best_skin_model.pth", map_location=device, weights_only=True)
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

    return model, device, classes, transform, label_encoder


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, _, __, output):
        self.activations = output.detach()

    def _backward_hook(self, _, __, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):
        outputs = self.model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        pred_idx = int(outputs.argmax(dim=1).item())
        confidence = float(probabilities[0, pred_idx].item())

        score = outputs[0, pred_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        act = self.activations[0]
        grad = self.gradients[0]
        weights = grad.mean(dim=(1, 2))

        cam = (weights.view(-1, 1, 1) * act).sum(dim=0)
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() + 1e-8)

        return cam, pred_idx, confidence


@st.cache_resource
def get_gradcam_bundle():
    model, device, classes, transform, _ = load_model_and_utils()
    target_layer = model.conv_head
    gradcam = GradCAM(model, target_layer)
    return gradcam, model, device, classes, transform


DISEASE_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Bowen's Disease",
        "desc": "Precancerous or in-situ squamous cell carcinoma of the skin.",
        "cause": "Long-term cumulative sun (UV) exposure damaging skin cells over years. More common in fair-skinned individuals, older adults (50+), and those with history of sunburns.",
        "prevention": "Use broad-spectrum sunscreen (SPF 30+), avoid peak sun hours (10 AM - 4 PM), wear protective clothing and wide-brimmed hats, regular dermatological skin checks every 6-12 months.",
        "prevalence": "Affects approximately 5-10% of adults over 40, more common in sunny climates.",
        "symptoms": "Rough, scaly patches on sun-exposed areas (face, scalp, hands, arms), may be pink, red, or brown, usually 2-6mm in size.",
        "treatment": "Cryotherapy, topical medications (5-FU, imiquimod), photodynamic therapy, or surgical removal. Regular monitoring is essential.",
        "severity": "Low to Moderate - Can progress to squamous cell carcinoma if untreated."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "desc": "Most common skin cancer, usually slow-growing and locally invasive. Rarely metastasizes but can cause significant local damage.",
        "cause": "Chronic UV exposure (especially in childhood), fair skin type, history of sunburns, older age (60+), genetic predisposition, exposure to arsenic or radiation.",
        "prevention": "Limit sun exposure, use broad-spectrum sunscreen daily, avoid tanning beds, wear sun-protective clothing, perform monthly self-exams, annual dermatologist visits.",
        "prevalence": "Most common cancer worldwide - affects ~3.3 million people annually in the US alone. More common in men and older adults.",
        "symptoms": "Pearly or waxy bump, flat flesh-colored or brown scar-like lesion, bleeding or scabbing sore that heals and returns, pink growth with raised border.",
        "treatment": "Surgical excision (Mohs surgery for high-risk cases), cryotherapy, topical treatments, radiation therapy. 95%+ cure rate with early treatment.",
        "severity": "Low - Rarely fatal but requires treatment to prevent local tissue destruction."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "desc": "Includes seborrheic keratoses, solar lentigines, and similar benign lesions. Non-cancerous growths.",
        "cause": "Aging skin (most common), cumulative sun exposure, genetic factors, hormonal changes. Not related to skin cancer.",
        "prevention": "Sun protection throughout life, regular skin self-exams to monitor changes, avoid excessive sun exposure.",
        "prevalence": "Very common - affects 80-90% of people over 50. Multiple lesions are typical.",
        "symptoms": "Waxy, stuck-on appearance, brown to black color, rough texture, usually 1-3cm, can appear anywhere on body except palms/soles.",
        "treatment": "Usually no treatment needed. Can be removed for cosmetic reasons via cryotherapy, curettage, or laser. Biopsy if diagnosis uncertain.",
        "severity": "Benign - No health risk, purely cosmetic concern."
    },
    "df": {
        "name": "Dermatofibroma",
        "desc": "Benign fibrous skin nodule, usually small and firm. Common harmless skin growth.",
        "cause": "Often follows minor skin trauma or insect bites. Exact cause unclear - may be reactive process. More common in women, ages 20-50.",
        "prevention": "No specific prevention. Avoid repeated trauma to skin when possible. Not related to sun exposure.",
        "prevalence": "Common - affects approximately 2-3% of adults. Most people have 1-2 lesions.",
        "symptoms": "Small (3-10mm), firm bump, usually on legs or arms, brown to purple color, dimples when pinched (Fitzpatrick sign), may be tender.",
        "treatment": "No treatment needed unless symptomatic or for cosmetic reasons. Surgical excision if desired. Recurrence possible if incompletely removed.",
        "severity": "Benign - Completely harmless, no health implications."
    },
    "mel": {
        "name": "Melanoma",
        "desc": "Serious, potentially life-threatening skin cancer of melanocytes. Most dangerous form of skin cancer.",
        "cause": "Intense or intermittent UV exposure, history of severe sunburns (especially in childhood), fair skin, many moles (50+), atypical moles, family history, genetic mutations (CDKN2A, BRAF), immunosuppression.",
        "prevention": "Strong sun protection (SPF 50+), avoid tanning beds completely, seek shade, wear protective clothing, perform monthly ABCDE self-exams, annual full-body dermatologist exams, early detection is critical.",
        "prevalence": "Less common but most dangerous - ~100,000 new cases annually in US. Incidence increasing 3% per year. More common in men and older adults.",
        "symptoms": "ABCDE rule: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving/changing. New or changing mole, irregular borders, multiple colors, itching or bleeding.",
        "treatment": "Surgical excision (wide local excision), sentinel lymph node biopsy, immunotherapy, targeted therapy, chemotherapy for advanced cases. Early detection = 98% 5-year survival.",
        "severity": "High - Can be fatal if not detected early. Metastasis to other organs is possible."
    },
    "nv": {
        "name": "Melanocytic Nevus (Mole)",
        "desc": "Common benign pigmented moles. Normal skin development.",
        "cause": "Normal skin development, genetics, some UV influence. Most appear in childhood/early adulthood. Hormonal changes (pregnancy) can darken existing moles.",
        "prevention": "General sun safety, monitor for changes using ABCDE rule, avoid excessive sun exposure, regular self-exams.",
        "prevalence": "Extremely common - average adult has 10-40 moles. Some individuals have 50+ (atypical mole syndrome).",
        "symptoms": "Small (usually <5mm), round or oval, uniform color (brown, tan, black), smooth borders, may be flat or raised, typically stable over time.",
        "treatment": "No treatment needed unless changing, symptomatic, or for cosmetic reasons. Biopsy if suspicious changes. Surgical removal if indicated.",
        "severity": "Benign - Normal skin feature. Monitor for changes that could indicate melanoma."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "desc": "Includes angiomas, angiokeratomas, pyogenic granulomas, hemorrhages, and other vascular proliferations.",
        "cause": "Abnormal growths of blood vessels, trauma, congenital factors, hormonal changes, aging. Some types are genetic.",
        "prevention": "No specific prevention. Protect skin from trauma. Some types are congenital and cannot be prevented.",
        "prevalence": "Common - cherry angiomas affect 75% of adults over 75. Pyogenic granulomas are less common but well-recognized.",
        "symptoms": "Red to purple color, may bleed easily, various sizes and shapes, can be flat or raised, may grow rapidly (pyogenic granuloma).",
        "treatment": "Observation if asymptomatic. Treatment options include laser therapy, cryotherapy, electrocautery, or surgical removal if bleeding or cosmetic concern.",
        "severity": "Benign - Usually harmless. May bleed if traumatized. Rare malignant forms exist."
    },
}


def predict_with_gradcam(pil_image: Image.Image):
    gradcam, model, device, classes, transform = get_gradcam_bundle()
    rgb_np = np.array(pil_image.convert("RGB"))
    tensor = transform(image=rgb_np)["image"].unsqueeze(0).to(device)

    cam, pred_idx, conf = gradcam.generate(tensor)
    class_id = classes[pred_idx]
    disease = DISEASE_INFO.get(
        class_id,
        {"name": class_id, "desc": "No description available."},
    )

    cam_uint8 = np.uint8(cam * 255)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (rgb_np.shape[1], rgb_np.shape[0]))
    overlay = (0.45 * heatmap + 0.55 * rgb_np).astype(np.uint8)

    return {
        "class_id": class_id,
        "confidence": conf,
        "disease": disease,
        "original": rgb_np,
        "heatmap": heatmap,
        "overlay": overlay,
    }


# ---------------------- UI HELPERS ----------------------
def navigation_bar():
    col1, col2, col3 = st.columns([1, 1, 1])
    mapping = [("Home", col1), ("Analyzer", col2), ("Doctor", col3)]
    for label, col in mapping:
        with col:
            is_active = st.session_state.active_page == label
            classes = "nav-btn active" if is_active else "nav-btn"
            if st.button(label, key=f"nav-{label}", use_container_width=True):
                st.session_state.active_page = label
                if label != "Doctor":
                    st.session_state.doctor_options_open = False
                if label != "Doctor Panel":
                    st.session_state.doctor_authenticated = False
        st.markdown(
            f"<style>div[data-testid='stButton'][key='nav-{label}'] button{{}} .nav-btn{{}}</style>",
            unsafe_allow_html=True,
        )


def home_page():
    st.markdown(
        """
        <div class="hero">
            <div class="pill">AI Dermatology Assistant</div>
            <h1 style="margin-top:1rem;margin-bottom:1rem;color:#fafafa;">Diagnose Skin Lesions with Explainable AI</h1>
            <p style="font-size:1.05rem;max-width:720px;color:#d1d5db;">
                Upload dermoscopic images, get real-time predictions, and visualize Grad-CAM heatmaps that explain model focus areas.
                This interface keeps everything clean, minimal, and laser-focused on medical insight.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Quick access")
    st.markdown("<div class='quick-access'>", unsafe_allow_html=True)
    qa_cols = st.columns(3)
    buttons = [
        ("Run Analyzer", "Analyzer"),
        ("Doctor Portal", "Doctor"),
        ("Doctor Panel", "Doctor Panel"),
    ]
    for (label, target), col in zip(buttons, qa_cols):
        with col:
            if st.button(label, key=f"home-{target}", use_container_width=True):
                st.session_state.active_page = target
                if target != "Doctor":
                    st.session_state.doctor_options_open = False
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Why clinicians like this view")
    col1, col2, col3 = st.columns(3)
    col1.metric("7+", "Skin diseases covered")
    col2.metric("Explainability", "Grad-CAM overlays")
    col3.metric("Latency", "< 3s on CPU / GPU")


def analyzer_page():
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("🧪 Upload and Analyze")
    uploaded = st.file_uploader("Upload a dermoscopic lesion image", type=["jpg", "jpeg", "png"])

    if uploaded:
        pil_image = Image.open(uploaded).convert("RGB")
        st.image(pil_image, caption="Original upload", use_column_width=True)

        if st.button("Run Grad-CAM Inference", type="primary"):
            with st.spinner("Analyzing image with EfficientNet + Grad-CAM..."):
                outputs = predict_with_gradcam(pil_image)

            st.success("Prediction ready")
            
            # Main prediction info
            col_info1, col_info2 = st.columns([2, 1])
            with col_info1:
                st.markdown(f"### {outputs['disease']['name']}")
                st.write(f"**Class ID:** `{outputs['class_id']}` | **Confidence:** {outputs['confidence']*100:.2f}%")
            with col_info2:
                severity_color = {"High": "🔴", "Low to Moderate": "🟡", "Low": "🟢", "Benign": "🟢"}
                severity_text = outputs['disease'].get('severity', 'Unknown')
                severity_icon = severity_color.get(severity_text.split(' - ')[0], "⚪")
                st.markdown(f"**Severity:** {severity_icon} {severity_text}")
            
            st.markdown("---")
            
            # Disease information in expandable sections
            st.markdown("#### 📋 Disease Information")
            
            with st.expander("📖 Description", expanded=True):
                st.write(outputs["disease"]["desc"])
            
            with st.expander("🔍 Symptoms"):
                st.write(outputs["disease"].get("symptoms", "Symptoms vary by individual. Consult a dermatologist for evaluation."))
            
            with st.expander("⚠️ Causes & Risk Factors"):
                st.write(outputs["disease"].get("cause", "Causes may vary. Consult medical literature for detailed information."))
            
            with st.expander("🛡️ Prevention"):
                st.write(outputs["disease"].get("prevention", "General sun protection and regular skin checks are recommended."))
            
            with st.expander("📊 Prevalence & Statistics"):
                st.write(outputs["disease"].get("prevalence", "Prevalence data varies by population and geographic region."))
            
            with st.expander("💊 Treatment Options"):
                st.write(outputs["disease"].get("treatment", "Treatment should be determined by a qualified dermatologist based on individual case."))
            
            st.markdown("---")
            st.markdown("#### 🖼️ Grad-CAM Visualization")
            c1, c2, c3 = st.columns(3)
            c1.image(outputs["original"], caption="Original Image", use_column_width=True)
            c2.image(outputs["heatmap"], caption="Grad-CAM Heatmap", use_column_width=True)
            c3.image(outputs["overlay"], caption="Overlay Visualization", use_column_width=True)
            
            st.info("⚠️ **Medical Disclaimer:** This is an AI-assisted tool for educational purposes only. Always consult a certified dermatologist for professional medical diagnosis and treatment.")

            st.session_state.recent_predictions.insert(
                0,
                {
                    "disease": outputs["disease"]["name"],
                    "class_id": outputs["class_id"],
                    "confidence": f"{outputs['confidence']*100:.2f}%",
                },
            )
            st.session_state.recent_predictions = st.session_state.recent_predictions[:5]

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- AUTH HELPERS ----------------------
def hash_password_local(password: str, salt: str = None):
    """Hash password for local storage"""
    if not salt:
        salt = secrets.token_hex(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 100000)
    return salt, pwd_hash.hex()

def verify_password_local(password: str, salt: str, stored_hash: str) -> bool:
    """Verify password against stored hash"""
    _, new_hash = hash_password_local(password, salt)
    return secrets.compare_digest(new_hash, stored_hash)

def register_doctor_local(full_name: str, email: str, password: str):
    """Register doctor in local database"""
    try:
        salt, pwd_hash = hash_password_local(password)
        conn = sqlite3.connect(LOCAL_DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO doctors (full_name, email, password_hash, salt) VALUES (?,?,?,?)",
            (full_name.strip(), email.lower(), pwd_hash, salt)
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully! You can now login."
    except sqlite3.IntegrityError:
        return False, "Email already registered. Please use a different email or login."
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_doctor_local(email: str, password: str):
    """Login doctor from local database"""
    try:
        conn = sqlite3.connect(LOCAL_DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id, full_name, email, password_hash, salt FROM doctors WHERE email = ?", (email.lower(),))
        row = c.fetchone()
        conn.close()
        if row is None or not verify_password_local(password, row[4], row[3]):
            return False, "Invalid email or password."
        return True, {
            "access_token": f"local_token_{row[0]}_{secrets.token_urlsafe(16)}",
            "doctor_name": row[1],
            "doctor_email": row[2]
        }
    except Exception as e:
        return False, f"Error: {str(e)}"

def register_doctor(full_name: str, email: str, password: str):
    """Register a new doctor account - tries API first, falls back to local"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/register",
            json={"full_name": full_name, "email": email, "password": password},
            timeout=5
        )
        if response.status_code == 201:
            return True, response.json().get("message", "Account created successfully!")
        else:
            error_msg = response.json().get("detail", "Registration failed")
            return False, error_msg
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Fallback to local database
        return register_doctor_local(full_name, email, password)
    except Exception as e:
        return False, f"Error: {str(e)}"

def login_doctor(email: str, password: str):
    """Login doctor - tries API first, falls back to local"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"email": email, "password": password},
            timeout=5
        )
        if response.status_code == 200:
            data = response.json()
            return True, data
        else:
            error_msg = response.json().get("detail", "Login failed")
            return False, error_msg
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        # Fallback to local database
        return login_doctor_local(email, password)
    except Exception as e:
        return False, f"Error: {str(e)}"


def doctor_portal():
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("👨‍⚕️ Doctor Portal")
    if st.button("Doctor", use_container_width=True):
        st.session_state.doctor_options_open = not st.session_state.doctor_options_open

    if st.session_state.doctor_options_open and not st.session_state.doctor_authenticated:
        tab1, tab2 = st.tabs(["Create new account", "Login"])

        with tab1:
            st.write("### Register as a new doctor")
            st.caption("Fill in your details to create an account")
            
            full_name = st.text_input("Full Name", key="reg_name", placeholder="Dr. John Smith")
            email = st.text_input("Email", key="reg_email", placeholder="doctor@example.com")
            password = st.text_input("Password", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
            
            if st.button("Create Account", key="reg_btn", type="primary"):
                if not full_name or not email or not password:
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long")
                else:
                    with st.spinner("Creating account..."):
                        success, message = register_doctor(full_name, email, password)
                        if success:
                            st.success(message)
                            st.info("You can now login with your email and password")
                        else:
                            st.error(message)

        with tab2:
            st.write("### Login to Doctor Panel")
            email = st.text_input("Email", key="doc_email", placeholder="doctor@example.com")
            password = st.text_input("Password", type="password", key="doc_pass")
            if st.button("Login", key="doc_login_btn", type="primary"):
                if not email or not password:
                    st.error("Please enter both email and password")
                else:
                    with st.spinner("Logging in..."):
                        success, result = login_doctor(email, password)
                        if success:
                            st.session_state.auth_token = result["access_token"]
                            st.session_state.doctor_info = {
                                "name": result["doctor_name"],
                                "email": result["doctor_email"]
                            }
                            st.session_state.doctor_authenticated = True
                            st.session_state.active_page = "Doctor Panel"
                            st.success(f"Welcome back, {result['doctor_name']}!")
                            st.rerun()
                        else:
                            st.error(result)

    if st.session_state.active_page == "Doctor Panel" and st.session_state.doctor_authenticated:
        doctor_panel()

    st.markdown("</div>", unsafe_allow_html=True)


def doctor_panel():
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("📊 Doctor Panel — Recent AI Reads")
    
    if st.session_state.doctor_info:
        st.write(f"**Logged in as:** {st.session_state.doctor_info['name']} ({st.session_state.doctor_info['email']})")
    
    if st.button("Logout", key="logout_btn"):
        st.session_state.doctor_authenticated = False
        st.session_state.auth_token = None
        st.session_state.doctor_info = None
        st.session_state.active_page = "Doctor"
        st.rerun()

    if not st.session_state.recent_predictions:
        st.info("Run a prediction in Analyzer to populate recent cases.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    for idx, record in enumerate(st.session_state.recent_predictions, start=1):
        st.write(f"**Case {idx}** — {record['disease']} ({record['class_id']}) · Confidence {record['confidence']}")

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- ROUTER ----------------------
navigation_bar()

page = st.session_state.active_page

if page == "Home":
    home_page()
elif page == "Analyzer":
    analyzer_page()
elif page == "Doctor":
    doctor_portal()
elif page == "Doctor Panel":
    doctor_panel()


