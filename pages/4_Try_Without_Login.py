# import streamlit as st
# import requests
# from PIL import Image
# from io import BytesIO

# API_URL = "http://127.0.0.1:8000/predict"

# st.title("🎯 Try Without Login")

# uploaded_file = st.file_uploader("Upload an Image", type=["jpg","png","jpeg"])

# if uploaded_file:
#     img = Image.open(uploaded_file)
#     st.image(img, use_column_width=True)

#     if st.button("Analyze"):
#         img_bytes = BytesIO()
#         img.save(img_bytes, format="JPEG")
#         img_bytes.seek(0)

#         files = {"file": ("image.jpg", img_bytes, "image/jpeg")}
#         r = requests.post(API_URL, files=files)

#         if r.status_code == 200:
#             data = r.json()

#             st.success("Prediction Result")
#             st.write(f"**Disease:** {data['label']}")
#             st.write(f"**Confidence:** {data['confidence']:.2f}")
#             st.write(f"**Cause:** {data['cause']}")
#             st.write(f"**Prevention:** {data['prevention']}")


# pages/3_Try_Without_Login.py
import streamlit as st
import requests
from io import BytesIO
from PIL import Image

API_URL = "http://127.0.0.1:8000/predict"

st.title("🎯 Try Without Login")

uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Analyze"):
        img_bytes = BytesIO()
        image.save(img_bytes, format="JPEG")
        img_bytes.seek(0)

        try:
            response = requests.post(API_URL, files={"file": ("image.jpg", img_bytes, "image/jpeg")})

            if response.status_code == 200:
                data = response.json()

                st.success("Prediction Completed")

                st.write(f"**Predicted Disease:** `{data['label']}`")
                st.write(f"**Class ID:** `{data['class_id']}`")
                st.write(f"**Confidence:** `{data['confidence'] * 100:.2f}%`")

                st.subheader("About the Disease")
                st.write(f"**Description:** {data['description']}")
                st.write(f"**Causes:** {data['cause']}")
                st.write(f"**Prevention:** {data['prevention']}")

            else:
                st.error(f"API Error: {response.status_code}")

        except Exception as e:
            st.error(f"Error contacting API: {e}")

else:
    st.info("Please upload an image to analyze.")
