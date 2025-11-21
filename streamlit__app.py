import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import json
import joblib
import cv2
import os
import time
import sqlite3
import pandas as pd
import smtplib
from email.message import EmailMessage
from fpdf import FPDF
from io import BytesIO
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ---------------------- BASIC CONFIG ----------------------
st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

# Hide sidebar + default footer/menu
st.markdown("""
<style>
section[data-testid="stSidebar"] {display: none !important;}
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 🔐 SMTP CONFIG (EDIT THESE FOR REAL EMAIL USAGE)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SENDER_EMAIL = "your_email@example.com"        # TODO: change to your email
SENDER_PASSWORD = "your_app_password_here"     # TODO: use app password, NOT real password


# ---------------------- SESSION STATE ----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"
if "doctor_dark" not in st.session_state:
    st.session_state.doctor_dark = False
if "doctor_logged_in" not in st.session_state:
    st.session_state.doctor_logged_in = False

# ---------------------- BACKGROUND + THEME CSS ----------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1580281657527-47d5a47ead7f');
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
.glass-card {
    margin-top: 80px;
    padding: 40px;
    border-radius: 18px;
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(16px);
    box-shadow: 0 12px 30px rgba(0,0,0,0.18);
    animation: fadein 0.6s ease-in-out;
}
.dark-card {
    background: rgba(15,23,42,0.95) !important;
    color: #e5e7eb !important;
}
.title-text {
    text-align: center;
    font-size: 40px;
    font-weight: 700;
    color: #123c69;
    margin-bottom: 10px;
}
.subtitle-text {
    text-align: center;
    font-size: 18px;
    color: #333333;
    margin-bottom: 25px;
}
.disclaimer-box {
    background: #f2f7ff;
    padding: 15px 20px;
    border-radius: 12px;
    border-left: 5px solid #2d8cf0;
    margin-top: 20px;
    font-size: 15px;
}
@keyframes fadein {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}
</style>
""", unsafe_allow_html=True)


# ---------------------- DB INIT ----------------------
def init_db():
    os.makedirs("uploaded", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    # NOTE: if you had an older table without 'email', you may need to delete patients.db once
    c.execute("""
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        email TEXT,
        predicted TEXT,
        confidence REAL,
        image_path TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)
    conn.commit()
    conn.close()

init_db()


# ---------------------- MODEL + GRADCAM ----------------------
IMG_SIZE = 224

@st.cache_resource
def load_model_and_utils():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("classes.json", "r") as f:
        classes = json.load(f)

    le = joblib.load("label_encoder.pkl")

    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
    # If torch version complains about weights_only, remove that argument
    state_dict = torch.load("best_skin_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    transform = Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=(0.485, 0.456, 0.406),
                  std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return model, device, classes, transform


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):
        out = self.model(input_tensor)
        probs = torch.softmax(out, dim=1)
        pred_idx = int(out.argmax(dim=1).item())
        conf = float(probs[0, pred_idx].item())

        score = out[0, pred_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        activs = self.activations[0]  # (C,H,W)
        grads = self.gradients[0]     # (C,H,W)
        weights = grads.mean(dim=(1, 2))  # (C,)

        cam = (weights.view(-1, 1, 1) * activs).sum(dim=0)  # (H,W)
        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam, pred_idx, conf


@st.cache_resource
def get_gradcam():
    model, device, classes, transform = load_model_and_utils()
    target_layer = model.conv_head
    gradcam = GradCAM(model, target_layer)
    return gradcam, model, device, classes, transform


# ---------------------- DISEASE INFO ----------------------
DISEASE_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Bowen’s Disease",
        "desc": "Precancerous or in-situ squamous cell carcinoma of the skin.",
        "cause": "Long-term sun (UV) exposure damaging skin cells.",
        "prev": "Use sunscreen, avoid strong sun, protective clothing, regular checks."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "desc": "Most common skin cancer, usually slow-growing and locally invasive.",
        "cause": "Chronic UV exposure, fair skin, history of sunburns, older age.",
        "prev": "Limit sun, use sunscreen, avoid tanning beds, monitor lesions."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "desc": "Includes seborrheic keratoses, solar lentigines, and similar benign lesions.",
        "cause": "Aging, sun exposure, genetics.",
        "prev": "Sun protection and regular skin self-exams."
    },
    "df": {
        "name": "Dermatofibroma",
        "desc": "Benign fibrous skin nodule, usually small and firm.",
        "cause": "Often follows minor trauma or insect bites.",
        "prev": "No specific prevention; avoid repeated trauma."
    },
    "mel": {
        "name": "Melanoma",
        "desc": "Serious, potentially life-threatening skin cancer of melanocytes.",
        "cause": "Intense UV exposure, sunburns, genetic predisposition, many or atypical moles.",
        "prev": "Strong sun protection, avoid tanning beds, early detection, dermatologist visits."
    },
    "nv": {
        "name": "Melanocytic Nevus (Mole)",
        "desc": "Common benign pigmented moles.",
        "cause": "Normal skin development, genetics, some UV influence.",
        "prev": "General sun safety and monitoring for changes."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "desc": "Includes angiomas, angiokeratomas, pyogenic granulomas, hemorrhages.",
        "cause": "Abnormal blood vessel growth, trauma, congenital factors.",
        "prev": "No specific prevention; protect skin from trauma."
    }
}


# ---------------------- HELPER FUNCTIONS ----------------------
def predict_and_explain(pil_image: Image.Image):
    gradcam, model, device, classes, transform = get_gradcam()
    img_np = np.array(pil_image.convert("RGB"))
    img_t = transform(image=img_np)["image"].unsqueeze(0).to(device)

    cam, pred_idx, conf = gradcam.generate(img_t)
    class_id = classes[pred_idx]
    info = DISEASE_INFO.get(class_id, {
        "name": class_id,
        "desc": "No description available.",
        "cause": "Unknown.",
        "prev": "Please consult a dermatologist."
    })

    cam_uint8 = np.uint8(cam * 255)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    overlay = (0.45 * heatmap + 0.55 * img_np).astype(np.uint8)

    return class_id, conf, info, img_np, heatmap, overlay


def save_patient_record(name, age, gender, email, predicted, confidence, image: Image.Image):
    filename = f"{name}_{int(time.time())}.jpg".replace(" ", "_")
    os.makedirs("uploaded", exist_ok=True)
    save_path = os.path.join("uploaded", filename)
    image.save(save_path)

    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO results (name, age, gender, email, predicted, confidence, image_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (name, age, gender, email, predicted, float(confidence), save_path))
    conn.commit()
    conn.close()
    return save_path


def load_all_records():
    conn = sqlite3.connect("patients.db")
    df = pd.read_sql_query("SELECT * FROM results ORDER BY timestamp DESC", conn)
    conn.close()
    return df


def update_record(record_id, name, age, gender, email, predicted, confidence):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("""
        UPDATE results
        SET name = ?, age = ?, gender = ?, email = ?, predicted = ?, confidence = ?
        WHERE id = ?
    """, (name, age, gender, email, predicted, float(confidence), int(record_id)))
    conn.commit()
    conn.close()


def delete_record(record_id):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("DELETE FROM results WHERE id = ?", (int(record_id),))
    conn.commit()
    conn.close()


# def generate_pdf_report(row, info):
#     pdf = FPDF()
#     pdf.add_page()
#     pdf.set_auto_page_break(auto=True, margin=15)

#     pdf.set_font("Arial", "B", 16)
#     pdf.cell(0, 10, "Skin Disease Detection Report", ln=True, align="C")
#     pdf.ln(5)

#     pdf.set_font("Arial", size=12)
#     pdf.cell(0, 8, f"Patient: {row['name']}", ln=True)
#     pdf.cell(0, 8, f"Age: {row['age']}   Gender: {row['gender']}", ln=True)
#     pdf.cell(0, 8, f"Email: {row.get('email', '')}", ln=True)
#     pdf.cell(0, 8, f"Date: {row['timestamp']}", ln=True)
#     pdf.ln(5)

#     pdf.cell(0, 8, f"Disease: {info['name']} ({row['predicted']})", ln=True)
#     pdf.cell(0, 8, f"Model Confidence: {row['confidence']*100:.2f}%", ln=True)
#     pdf.ln(4)

#     pdf.multi_cell(0, 7, f"Description: {info['desc']}")
#     pdf.ln(3)
#     pdf.multi_cell(0, 7, f"Possible Causes: {info['cause']}")
#     pdf.ln(3)
#     pdf.multi_cell(0, 7, f"Prevention / Care: {info['prev']}")
#     pdf.ln(5)

#     if os.path.exists(row["image_path"]):
#         try:
#             pdf.ln(4)
#             pdf.set_font("Arial", "B", 12)
#             pdf.cell(0, 8, "Lesion Image:", ln=True)
#             pdf.image(row["image_path"], w=80)
#         except Exception:
#             pass

#     buffer = BytesIO()
#     pdf.output(buffer)
#     buffer.seek(0)
#     return buffer


# def send_email_report(to_email, row, info, pdf_bytes):
#     msg = EmailMessage()
#     msg["Subject"] = f"Skin Disease Report - {row['name']}"
#     msg["From"] = SENDER_EMAIL
#     msg["To"] = to_email

#     body = f"""
# Skin Disease Detection Report

# Patient: {row['name']} ({row['age']} yrs, {row['gender']})
# Predicted disease: {info['name']} ({row['predicted']})
# Model confidence: {row['confidence']*100:.2f}%

# Description:
# {info['desc']}

# Possible Causes:
# {info['cause']}

# Prevention / Care:
# {info['prev']}

# This report is generated automatically for educational purposes only.
# Please consult a dermatologist for real medical advice.
# """
#     msg.set_content(body)

#     msg.add_attachment(pdf_bytes.read(), maintype="application", subtype="pdf",
#                        filename=f"report_{row['id']}.pdf")

#     try:
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls()
#             server.login(SENDER_EMAIL, SENDER_PASSWORD)
#             server.send_message(msg)
#         return True, "Email sent successfully."
#     except Exception as e:
#         return False, str(e)


# ---------------------- UI PAGES ----------------------
def home_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='title-text'>🩺 Skin Disease Detection App</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle-text'>Deep Learning-based analysis of dermoscopic images using the HAM10000 dataset.</div>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("👤 Patient Mode", use_container_width=True):
            st.session_state.page = "patient"
    with col2:
        if st.button("🎯 Try Without Login", use_container_width=True):
            st.session_state.page = "try"
    with col3:
        if st.button("👨‍⚕️ Doctor Panel", use_container_width=True):
            st.session_state.page = "doctor_login"

    st.markdown("""
    <div class='disclaimer-box'>
    <b>Disclaimer:</b> This tool is for <b>educational and research purposes only</b>. 
    It is not a substitute for professional medical diagnosis. Please consult a certified dermatologist.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def patient_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("👤 Patient Mode")

    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    email = st.text_input("Email (for sending report, optional)")
    uploaded = st.file_uploader("Upload a dermoscopic skin lesion image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze with Grad-CAM"):
            if not name.strip():
                st.warning("Please enter patient name.")
            else:
                with st.spinner("Analyzing image with AI model..."):
                    class_id, conf, info, img_np, heatmap, overlay = predict_and_explain(image)
                    save_patient_record(name, age, gender, email, class_id, conf, image)

                st.success("Analysis Completed ✅")

                st.write(f"**Patient:** {name}, {age} yrs, {gender}")
                st.write(f"**Predicted Class ID:** `{class_id}`")
                st.write(f"**Predicted Disease:** **{info['name']}**")
                st.write(f"**Model Confidence:** `{conf*100:.2f}%`")

                st.write("### Grad-CAM Explainability")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.image(img_np, caption="Original", use_column_width=True)
                with c2:
                    st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
                with c3:
                    st.image(overlay, caption="Overlay", use_column_width=True)

                st.write("### About this disease")
                st.write(f"**Description:** {info['desc']}")
                st.write(f"**Possible Causes:** {info['cause']}")
                st.write(f"**Prevention / Care:** {info['prev']}")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)


def try_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎯 Try Without Login")

    uploaded = st.file_uploader("Upload a dermoscopic skin lesion image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Quick Analyze with Grad-CAM"):
            with st.spinner("Analyzing image with AI model..."):
                class_id, conf, info, img_np, heatmap, overlay = predict_and_explain(image)

            st.success("Analysis Completed ✅")

            st.write(f"**Predicted Class ID:** `{class_id}`")
            st.write(f"**Predicted Disease:** **{info['name']}**")
            st.write(f"**Model Confidence:** `{conf*100:.2f}%`")

            st.write("### Grad-CAM Explainability")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.image(img_np, caption="Original", use_column_width=True)
            with c2:
                st.image(heatmap, caption="Grad-CAM Heatmap", use_column_width=True)
            with c3:
                st.image(overlay, caption="Overlay", use_column_width=True)

            st.write("### About this disease")
            st.write(f"**Description:** {info['desc']}")
            st.write(f"**Possible Causes:** {info['cause']}")
            st.write(f"**Prevention / Care:** {info['prev']}")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)


def doctor_login_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("👨‍⚕️ Doctor Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "doctor" and password == "admin123":
            st.success("Login successful ✅")
            st.session_state.doctor_logged_in = True
            st.session_state.page = "doctor"
        else:
            st.error("Invalid credentials")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)


def doctor_page():
    if not st.session_state.doctor_logged_in:
        st.warning("Please login as doctor first.")
        st.session_state.page = "doctor_login"
        return

    card_class = "glass-card"
    col_theme, _ = st.columns([1, 3])
    with col_theme:
        st.session_state.doctor_dark = st.checkbox("🌙 Dark theme for dashboard", value=st.session_state.doctor_dark)
    if st.session_state.doctor_dark:
        card_class += " dark-card"

    st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
    st.subheader("👨‍⚕️ Doctor Dashboard")

    df = load_all_records()
    if df.empty:
        st.info("No patient records yet.")
    else:
        # Search & sort controls
        st.write("### 🔎 Search & Sort")
        search_text = st.text_input("Search by name, email or disease (class id):", "")
        sort_col = st.selectbox("Sort by", ["timestamp", "name", "age", "predicted", "confidence"])
        sort_order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)

        filtered_df = df.copy()
        if search_text.strip():
            s = search_text.lower()
            mask = (
                filtered_df["name"].str.lower().str.contains(s)
                | filtered_df["email"].fillna("").str.lower().str.contains(s)
                | filtered_df["predicted"].str.lower().str.contains(s)
            )
            filtered_df = filtered_df[mask]

        ascending = (sort_order == "Ascending")
        filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)

        st.write("### 📝 Patient Records")
        st.dataframe(filtered_df, use_container_width=True)

        # Admin panel
        st.write("### 🛠 Admin Panel")
        if not filtered_df.empty:
            record_ids = filtered_df["id"].tolist()
            selected_id = st.selectbox("Select record ID to manage", record_ids)

            row = filtered_df[filtered_df["id"] == selected_id].iloc[0]
            info = DISEASE_INFO.get(row["predicted"], {
                "name": row["predicted"],
                "desc": "No description available.",
                "cause": "Unknown.",
                "prev": "Please consult a dermatologist."
            })

            st.write(f"**Selected Patient:** {row['name']} | {row['age']} yrs | {row['gender']}")

            c1, c2 = st.columns(2)
            with c1:
                st.write("**Edit Fields**")
                new_name = st.text_input("Name", value=row["name"])
                new_age = st.number_input("Age", min_value=1, max_value=120, value=int(row["age"]))
                new_gender = st.selectbox("Gender", ["Male", "Female", "Other"], index=["Male","Female","Other"].index(row["gender"]) if row["gender"] in ["Male","Female","Other"] else 0)
                new_email = st.text_input("Email", value=row.get("email", "") or "")
                new_pred = st.text_input("Predicted Class ID", value=row["predicted"])
                new_conf = st.number_input("Confidence (0-1)", min_value=0.0, max_value=1.0, value=float(row["confidence"]))

                if st.button("💾 Save Changes"):
                    update_record(selected_id, new_name, new_age, new_gender, new_email, new_pred, new_conf)
                    st.success("Record updated.")
                    st.experimental_rerun()

                if st.button("🗑 Delete Record"):
                    delete_record(selected_id)
                    st.success("Record deleted.")
                    st.experimental_rerun()

            with c2:
                st.write("**Image & Report**")
                if os.path.exists(row["image_path"]):
                    st.image(row["image_path"], caption="Patient Image", width=250)
                else:
                    st.warning("Image file not found.")

                # pdf_buffer = generate_pdf_report(row, info)

                # st.download_button(
                #     label="📄 Download PDF Report",
                #     data=pdf_buffer,
                #     file_name=f"report_{row['id']}.pdf",
                #     mime="application/pdf"
                # )

                # email_target = st.text_input("Send report to email", value=row.get("email", "") or "")
                # if st.button("📧 Email Report"):
                #     if not email_target.strip():
                #         st.warning("Please enter a valid email.")
                #     else:
                #         pdf_buffer_for_email = generate_pdf_report(row, info)
                #         ok, msg = send_email_report(email_target, row, info, pdf_buffer_for_email)
                #         if ok:
                #             st.success(msg)
                        # else:
                        #     st.error(f"Email error: {msg}")

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- ROUTER ----------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "patient":
    patient_page()
elif st.session_state.page == "try":
    try_page()
elif st.session_state.page == "doctor_login":
    doctor_login_page()
elif st.session_state.page == "doctor":
    doctor_page()
