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
model = genai.GenerativeModel("gemini-1.5-flash")

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

        # Load and display image
        image = Image.open(uploaded_file)
        st.image(image, use_column_width=True)

        # Convert image to base64 for Gemini
        image_bytes = uploaded_file.read()
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        markdown_img = f"![mri](data:image/jpeg;base64,{b64})"

        # Construct chat prompt with inline image
        prompt = (
            "You are a medical AI assistant. Analyze the following brain MRI image for tumor or glioma:\n\n"
            f"{markdown_img}\n\n"
            "Please provide:\n"
            "1. Any detected abnormalities or tumor regions.\n"
            "2. Approximate size or location.\n"
            "3. Medical interpretation helpful for clinicians."
        )

        # Send to Gemini via chat API
        with st.spinner("Analyzing image with Gemini..."):
            try:
                response = model.chat([
                    {"author": "user", "content": prompt}
                ])
                st.markdown("### 🧠 AI Diagnosis")
                st.write(response.last.content)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
else:
    st.info("Upload up to four MRI images to begin analysis.")
