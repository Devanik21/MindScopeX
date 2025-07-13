# app.py
import streamlit as st
from PIL import Image
import google.generativeai as genai
import io
import base64

# Streamlit page config
st.set_page_config(
    page_title="🧠 NeuroSight AI",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🧠 NeuroSight AI - Medical MRI Image Analysis")
st.markdown(
    "Upload brain MRI images and let Gemini provide advanced diagnostic insights."
)

# Sidebar: Gemini API Key Input
st.sidebar.header("🔐 API Configuration")
gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password"
)

if not gemini_api_key:
    st.sidebar.warning("Please enter your Gemini API key to continue.")
    st.stop()

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Load Gemini model
model = genai.GenerativeModel("gemini-2.5-flash")

# File uploader for MRI images
uploaded_files = st.file_uploader(
    "📤 Upload MRI images (JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.header("🖼 Uploaded Images & AI Diagnosis")
    for uploaded_file in uploaded_files:
        st.subheader(f"🧾 Image: {uploaded_file.name}")

        # Load image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        # Convert image to bytes and base64 encode
        image_bytes = uploaded_file.read()

        # Prompt for Gemini (you can modify this)
        prompt = (
            "This is a brain MRI image of a patient suspected to have a brain tumor or glioma.\n"
            "Provide detailed insights: detect any abnormalities, potential tumor regions, size estimation if possible, "
            "and medical interpretation useful for doctors. Be precise and medical-grade in your explanation."
        )

        # Send to Gemini
        with st.spinner("Analyzing image with Gemini..."):
            try:
                response = model.generate_content([
                    prompt,
                    {
                        "mime_type": "image/jpeg",
                        "data": image_bytes
                    }
                ])
                st.markdown("### 🧠 AI Diagnosis")
                st.write(response.text)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
else:
    st.info("Upload up to four MRI images to begin analysis.")
