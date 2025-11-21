import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import sqlite3
import os

API_URL = "http://127.0.0.1:8000/predict"

st.title("👤 Patient Diagnosis")

name = st.text_input("Name")
age = st.number_input("Age", min_value=1, max_value=120)
gender = st.selectbox("Gender", ["Male", "Female"])

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file and name and age:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    if st.button("Analyze"):
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
        r = requests.post(API_URL, files=files)

        if r.status_code == 200:
            data = r.json()

            st.success("Prediction complete!")
            st.write("### Result")
            st.write(f"**Disease:** {data['label']}")
            st.write(f"**Confidence:** {data['confidence']:.2f}")
            st.write(f"**Cause:** {data['cause']}")
            st.write(f"**Prevention:** {data['prevention']}")

            # Save to DB
            conn = sqlite3.connect("patients.db")
            c = conn.cursor()

            image_path = f"uploaded/{name}_{uploaded_file.name}"
            os.makedirs("uploaded", exist_ok=True)
            Image.open(uploaded_file).save(image_path)

            c.execute("""
                INSERT INTO results (name, age, gender, predicted, confidence, image_path)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, age, gender, data["label"], data["confidence"], image_path))

            conn.commit()
            conn.close()
