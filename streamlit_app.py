# # # # streamlit_app.py
# # # import streamlit as st
# # # import requests
# # # from io import BytesIO
# # # from PIL import Image

# # # # # ----------------- Config -----------------
# # # # API_URL = "http://127.0.0.1:8000/predict"

# # # # st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺")

# # # # # ----------------- UI -----------------
# # # # st.title("🩺 Skin Disease Detection")
# # # # st.write("""
# # # # This app uses a deep learning model trained on the HAM10000 dataset  
# # # # to predict the type of skin lesion from a dermoscopic image.

# # # # > ⚠️ **Important:** This is **not** a medical diagnosis.  
# # # # Always consult a qualified dermatologist for any skin concerns.
# # # # """)

# # # # uploaded_file = st.file_uploader("Upload a skin lesion image (JPG/PNG)", type=["jpg", "jpeg", "png"])

# # # # if uploaded_file is not None:
# # # #     # Show uploaded image
# # # #     image = Image.open(uploaded_file).convert("RGB")
# # # #     st.image(image, caption="Uploaded Image", use_column_width=True)

# # # #     if st.button("🔍 Analyze Image"):
# # # #         with st.spinner("Sending image to model..."):
# # # #             # Prepare file for API
# # # #             img_bytes = BytesIO()
# # # #             image.save(img_bytes, format="JPEG")
# # # #             img_bytes.seek(0)

# # # #             files = {"file": ("image.jpg", img_bytes, "image/jpeg")}

# # # #             try:
# # # #                 response = requests.post(API_URL, files=files, timeout=30)
# # # #                 if response.status_code == 200:
# # # #                     data = response.json()

# # # #                     st.success("Prediction completed")

# # # #                     st.subheader("Prediction Result")
# # # #                     st.markdown(f"**Predicted Disease:** `{data['label']}`  (`{data['class_id']}`)")
# # # #                     st.markdown(f"**Confidence:** `{data['confidence']*100:.2f}%`")

# # # #                     st.subheader("About this disease")
# # # #                     st.markdown(f"**Description:** {data['description']}")
# # # #                     st.markdown(f"**Possible Causes:** {data['cause']}")
# # # #                     st.markdown(f"**Prevention / Care:** {data['prevention']}")

# # # #                     st.info("⚠ These details are for educational purposes only. Please consult a dermatologist for real medical advice.")
# # # #                 else:
# # # #                     st.error(f"API Error: {response.status_code} - {response.text}")

# # # #             except Exception as e:
# # # #                 st.error(f"Failed to contact API: {e}")
# # # # else:
# # # #     st.info("Please upload an image to start the prediction.")





# # # # st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺")

# # # # st.title("🩺 Skin Disease Detection App")

# # # # st.write("""
# # # # Welcome to the **Skin Disease Detection System**.

# # # # Choose a mode from the left side menu:

# # # # - 👨‍⚕️ **Doctor Login**
# # # # - 👤 **Patient Mode**
# # # # - 🎯 **Try Without Login**
# # # # """)

# # # # st.info("This app uses a Deep Learning model trained on HAM10000 dataset.")

# # # # streamlit_app.py

# # # import streamlit as st

# # # st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺")

# # # st.title("🩺 Skin Disease Detection App")

# # # st.write("""
# # # Welcome to the **Skin Disease Detection System** powered by Deep Learning.

# # # Navigate using the left sidebar:

# # # - 👨‍⚕️ **Doctor Login**
# # # - 👤 **Patient Mode**
# # # - 🎯 **Try Without Login**
# # # """)

# # # st.info("This app uses a Deep Learning model trained on the HAM10000 dataset for educational purposes.")


# # # streamlit_app.py
# # import streamlit as st

# # # ----------------- Page Config -----------------
# # st.set_page_config(
# #     page_title="Skin Disease Detection",
# #     page_icon="🩺",
# #     layout="wide"
# # )

# # # ----------------- Custom CSS -----------------
# # st.markdown("""
# # <style>
# #     .main-title {
# #         font-size: 42px !important;
# #         font-weight: 700 !important;
# #         text-align: center;
# #         color: #1f4e79;
# #         margin-top: -20px;
# #     }
# #     .sub-text {
# #         text-align: center;
# #         font-size: 18px;
# #         color: #444444;
# #         margin-bottom: 20px;
# #     }
# #     .info-box {
# #         background-color: #edf7ff;
# #         padding: 20px;
# #         border-radius: 12px;
# #         border-left: 6px solid #2d8cf0;
# #         font-size: 17px;
# #     }
# #     .footer {
# #         text-align: center;
# #         color: #666;
# #         margin-top: 50px;
# #         font-size: 14px;
# #     }
# # </style>
# # """, unsafe_allow_html=True)

# # # ----------------- Hero Section -----------------
# # st.markdown("<h1 class='main-title'>🩺 Skin Disease Detection App</h1>", unsafe_allow_html=True)

# # st.markdown("""
# # <p class='sub-text'>
# # A powerful AI system trained on the <b>HAM10000</b> dermatology dataset 
# # to help detect and classify skin lesions from dermoscopic images.
# # </p>
# # """, unsafe_allow_html=True)

# # # Divider
# # st.markdown("---")

# # # ----------------- Info Card -----------------
# # st.markdown("""
# # <div class="info-box">
# # This platform uses a state-of-the-art deep learning model to analyze skin lesion images
# # and classify them into one of 7 medical categories such as Melanoma, Nevus, BCC, and more.<br><br>

# # <b>⚠ Disclaimer:</b>  
# # This tool is intended <b>for educational and research purposes only</b>.
# # It is <b>not a medical diagnosis</b>. Please consult a certified dermatologist for professional evaluation.
# # </div>
# # """, unsafe_allow_html=True)

# # # Footer
# # st.markdown("""
# # <p class='footer'>
# # Made with ❤️ using Streamlit & FastAPI | Powered by Deep Learning (EfficientNet)
# # </p>
# # """, unsafe_allow_html=True)

# import streamlit as st
# from PIL import Image

# # ---------------------- PAGE CONFIG ----------------------
# st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

# # ---------------------- SESSION STATE ----------------------
# if "page" not in st.session_state:
#     st.session_state.page = "home"

# if "theme" not in st.session_state:
#     st.session_state.theme = "light"

# if "language" not in st.session_state:
#     st.session_state.language = "English"

# if "show_doctor_login" not in st.session_state:
#     st.session_state.show_doctor_login = False


# # ---------------------- BACKGROUND IMAGE ----------------------
# st.markdown("""
# <style>
# [data-testid="stAppViewContainer"] {
#     background-image: url('https://images.unsplash.com/photo-1580281657527-47d5a47ead7f');
#     background-size: cover;
#     background-repeat: no-repeat;
#     background-attachment: fixed;
# }
# [data-testid="stHeader"] {background: rgba(0,0,0,0);}
# .main-block {
#     margin-top: 80px;
#     padding: 40px;
#     border-radius: 15px;
#     backdrop-filter: blur(20px);
#     background: rgba(255,255,255,0.7);
#     animation: fadein 1s ease-in-out;
# }
# .dark-theme .main-block {
#     background: rgba(0,0,0,0.6);
#     color: white;
# }
# @keyframes fadein {
#     from {opacity: 0; transform: translateY(20px);}
#     to {opacity: 1; transform: translateY(0);}
# }
# </style>
# """, unsafe_allow_html=True)


# # ---------------------- LANGUAGE PACK ----------------------
# LANG = {
#     "English": {
#         "title": "🩺 Skin Disease Detection",
#         "desc": "AI-powered system trained on HAM10000 dataset to detect skin lesions.",
#         "get_started": "Get Started",
#         "doctor_login": "Doctor Login",
#         "patient_mode": "Patient Mode",
#         "try_now": "Try Without Login",
#         "close": "Close",
#         "enter_username": "Enter Username",
#         "enter_password": "Enter Password",
#         "login_button": "Login"
#     },
#     "Kannada": {
#         "title": "🩺 ಚರ್ಮ ರೋಗ ಪತ್ತೆ",
#         "desc": "HAM10000 ಡೇಟಾಸೆಟ್ ಆಧಾರಿತ AI ಚರ್ಮ ರೋಗ ಪತ್ತೆ ವ್ಯವಸ್ಥೆ.",
#         "get_started": "ಪ್ರಾರಂಭಿಸಿ",
#         "doctor_login": "ವೈದ್ಯರ ಲಾಗಿನ್",
#         "patient_mode": "ರೋಗಿ ಮೋಡ್",
#         "try_now": "ಲಾಗಿನ್ ಇಲ್ಲದೆ ಪ್ರಯತ್ನಿಸಿ",
#         "close": "ಮುಚ್ಚಿ",
#         "enter_username": "ಬಳಕೆದಾರ ಹೆಸರು",
#         "enter_password": "ಪಾಸ್ವರ್ಡ್",
#         "login_button": "ಲಾಗಿನ್"
#     },
#     "Hindi": {
#         "title": "🩺 त्वचा रोग पहचान",
#         "desc": "HAM10000 डेटासेट आधारित AI त्वचा रोग पहचान प्रणाली।",
#         "get_started": "शुरू करें",
#         "doctor_login": "डॉक्टर लॉगिन",
#         "patient_mode": "मरीज मोड",
#         "try_now": "बिना लॉगिन प्रयास करें",
#         "close": "बंद करें",
#         "enter_username": "उपयोगकर्ता नाम दर्ज करें",
#         "enter_password": "पासवर्ड दर्ज करें",
#         "login_button": "लॉगिन"
#     }
# }

# TEXT = LANG[st.session_state.language]


# # ---------------------- DARK/LIGHT TOGGLE ----------------------
# toggle = st.checkbox("🌙 Dark Mode")
# if toggle:
#     st.session_state.theme = "dark"
#     st.markdown("<style>html {filter: invert(1) hue-rotate(180deg);}</style>", unsafe_allow_html=True)


# # ---------------------- LANGUAGE DROPDOWN ----------------------
# st.session_state.language = st.selectbox("🌐 Language", ["English", "Kannada", "Hindi"])
# TEXT = LANG[st.session_state.language]


# # ---------------------- HOME PAGE ----------------------
# if st.session_state.page == "home":

#     st.markdown("<div class='main-block'>", unsafe_allow_html=True)

#     st.markdown(f"<h1 style='text-align:center;'>{TEXT['title']}</h1>", unsafe_allow_html=True)
#     st.markdown(f"<p style='text-align:center;font-size:20px;'>{TEXT['desc']}</p>", unsafe_allow_html=True)

#     st.write("")

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         if st.button(f"👨‍⚕️ {TEXT['doctor_login']}", use_container_width=True):
#             st.session_state.show_doctor_login = True

#     with col2:
#         if st.button(f"👤 {TEXT['patient_mode']}", use_container_width=True):
#             st.session_state.page = "patient"

#     with col3:
#         if st.button(f"🎯 {TEXT['try_now']}", use_container_width=True):
#             st.session_state.page = "try"

#     st.markdown("</div>", unsafe_allow_html=True)


# # ---------------------- DOCTOR LOGIN POPUP ----------------------
# if st.session_state.show_doctor_login:
#     st.markdown("""
#     <style>
#     .popup {
#         position: fixed;
#         top: 20%;
#         left: 50%;
#         transform: translate(-50%, 0);
#         padding: 30px;
#         background: white;
#         border-radius: 12px;
#         box-shadow: 0 5px 25px rgba(0,0,0,0.3);
#         width: 350px;
#         z-index: 9999;
#         animation: fadein 0.5s ease-in-out;
#     }
#     </style>
#     """, unsafe_allow_html=True)

#     with st.container():
#         st.markdown("<div class='popup'>", unsafe_allow_html=True)
#         st.subheader(TEXT["doctor_login"])
#         username = st.text_input(TEXT["enter_username"])
#         password = st.text_input(TEXT["enter_password"], type="password")

#         if st.button(TEXT["login_button"]):
#             if username == "doctor" and password == "admin123":
#                 st.success("Login successful")
#                 st.session_state.page = "doctor"
#                 st.session_state.show_doctor_login = False
#             else:
#                 st.error("Invalid credentials")

#         if st.button(TEXT["close"]):
#             st.session_state.show_doctor_login = False

#         st.markdown("</div>", unsafe_allow_html=True)


# # ---------------------- SWITCH PAGES ----------------------
# if st.session_state.page == "doctor":
#     st.write("Doctor Panel Page Coming Here...")

# elif st.session_state.page == "patient":
#     st.write("Patient Mode Page Coming Here...")

# elif st.session_state.page == "try":
#     st.write("Try Without Login Page Coming Here...")

import streamlit as st
from PIL import Image
import numpy as np
import torch
import timm
import json
import joblib
import cv2
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(page_title="Skin Disease Detection", page_icon="🩺", layout="wide")

# ---------------------- HIDE SIDEBAR ----------------------
st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {display: none !important;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- SESSION STATE ----------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------------------- BACKGROUND + THEME CSS ----------------------
st.markdown(
    """
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
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(16px);
        box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        animation: fadein 0.8s ease-in-out;
    }
    @keyframes fadein {
        from {opacity: 0; transform: translateY(20px);}
        to {opacity: 1; transform: translateY(0);}
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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------- MODEL + GRADCAM LOADING ----------------------

IMG_SIZE = 224

@st.cache_resource
def load_model_and_utils():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load classes
    with open("classes.json", "r") as f:
        classes = json.load(f)

    # Load label encoder (optional, not strictly needed here)
    le = joblib.load("label_encoder.pkl")

    # Create and load model
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=len(classes))
    state_dict = torch.load("best_skin_model.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Albumentations transform
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

        # Forward hook
        self.target_layer.register_forward_hook(self._forward_hook)
        # Backward hook
        self.target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach()

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor):
        # Forward pass
        out = self.model(input_tensor)  # shape: (1, num_classes)
        probs = torch.softmax(out, dim=1)
        pred_idx = int(out.argmax(dim=1).item())
        conf = float(probs[0, pred_idx].item())

        score = out[0, pred_idx]
        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Gradients & activations
        # activations shape: (1, C, H, W)
        activs = self.activations[0]        # (C, H, W)
        grads = self.gradients[0]           # (C, H, W)

        # Global average pooling of gradients
        weights = grads.mean(dim=(1, 2))    # (C,)

        # Weighted combination
        cam = (weights.view(-1, 1, 1) * activs).sum(dim=0)  # (H, W)

        cam = cam.cpu().numpy()
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)      # normalize 0-1

        return cam, pred_idx, conf


@st.cache_resource
def get_gradcam():
    model, device, classes, transform = load_model_and_utils()
    target_layer = model.conv_head
    gradcam = GradCAM(model, target_layer)
    return gradcam, model, device, classes, transform


# ---------------------- DISEASE INFO (HAM10000) ----------------------
DISEASE_INFO = {
    "akiec": {
        "name": "Actinic Keratoses / Bowen’s Disease",
        "desc": "Precancerous or in-situ squamous cell carcinoma of the skin.",
        "cause": "Long-term sun (UV) exposure damaging skin cells.",
        "prev": "Use sunscreen, avoid intense sun, wear protective clothing, regular skin checks."
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "desc": "Most common skin cancer, usually slow-growing and locally invasive.",
        "cause": "Chronic UV exposure, fair skin, history of sunburns, older age.",
        "prev": "Limit sun, wear sunscreen, avoid tanning beds, monitor new lesions."
    },
    "bkl": {
        "name": "Benign Keratosis-like Lesion",
        "desc": "Includes seborrheic keratoses, solar lentigines, and similar benign lesions.",
        "cause": "Aging, sun exposure, genetics.",
        "prev": "Sun protection and routine skin self-exams."
    },
    "df": {
        "name": "Dermatofibroma",
        "desc": "Benign fibrous skin nodule, usually small and firm.",
        "cause": "Often after minor trauma or insect bites; exact cause unclear.",
        "prev": "No specific prevention; avoid scratching/trauma to skin when possible."
    },
    "mel": {
        "name": "Melanoma",
        "desc": "Serious, potentially life-threatening skin cancer of melanocytes.",
        "cause": "Intense UV exposure, sunburns, genetic predisposition, many or atypical moles.",
        "prev": "Strong sun protection, avoid tanning beds, early detection, dermatologist checks."
    },
    "nv": {
        "name": "Melanocytic Nevus (Mole)",
        "desc": "Common benign pigmented moles.",
        "cause": "Normal skin development, genetics, some UV influence.",
        "prev": "General sun safety and monitoring for changes using ABCDE rule."
    },
    "vasc": {
        "name": "Vascular Lesion",
        "desc": "Includes angiomas, angiokeratomas, pyogenic granulomas, hemorrhages.",
        "cause": "Abnormal blood vessel growth, trauma, or congenital factors.",
        "prev": "No specific prevention; protect skin from trauma."
    }
}


# ---------------------- HELPER: RUN PREDICTION + GRADCAM ----------------------
def predict_and_explain(pil_image: Image.Image):
    gradcam, model, device, classes, transform = get_gradcam()

    # Original image as numpy
    img_np = np.array(pil_image.convert("RGB"))

    # Preprocess for model
    img_t = transform(image=img_np)["image"].unsqueeze(0).to(device)

    # GradCAM generate
    cam, pred_idx, conf = gradcam.generate(img_t)

    class_id = classes[pred_idx]
    info = DISEASE_INFO.get(class_id, {
        "name": class_id,
        "desc": "No description available.",
        "cause": "Unknown.",
        "prev": "Please consult a dermatologist."
    })

    # Resize CAM & overlay
    cam_uint8 = np.uint8(cam * 255)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))

    overlay = (0.45 * heatmap + 0.55 * img_np).astype(np.uint8)

    return class_id, conf, info, img_np, heatmap, overlay


# ---------------------- UI PAGES ----------------------
def home_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.markdown("<div class='title-text'>🩺 Skin Disease Detection App</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='subtitle-text'>Deep Learning based analysis of dermoscopic images using the HAM10000 dataset.</div>",
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
        if st.button("👨‍⚕️ Doctor (coming soon)", use_container_width=True):
            st.session_state.page = "doctor"

    st.markdown(
        """
        <div class='disclaimer-box'>
        <b>Disclaimer:</b> This tool is for <b>educational and research purposes only</b>. 
        It is <b>not</b> a medical diagnosis. For any skin concern, always consult a certified dermatologist.
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("</div>", unsafe_allow_html=True)


def patient_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("👤 Patient Mode")

    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120, value=25)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    uploaded = st.file_uploader("Upload a dermoscopic skin image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze (with Grad-CAM)"):
            if not name.strip():
                st.warning("Please enter patient name.")
            else:
                with st.spinner("Analyzing image with AI model..."):
                    class_id, conf, info, img_np, heatmap, overlay = predict_and_explain(image)

                st.success("Analysis Completed")

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

    uploaded = st.file_uploader("Upload a dermoscopic skin image", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze (Quick)"):
            with st.spinner("Analyzing image with AI model..."):
                class_id, conf, info, img_np, heatmap, overlay = predict_and_explain(image)

            st.success("Analysis Completed")

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


def doctor_page():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("👨‍⚕️ Doctor Panel (Coming Soon)")
    st.info("Doctor dashboard, patient records, and advanced analytics can be added here later.")
    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------- PAGE ROUTER ----------------------
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "patient":
    patient_page()
elif st.session_state.page == "try":
    try_page()
elif st.session_state.page == "doctor":
    doctor_page()
