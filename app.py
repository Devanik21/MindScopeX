# app.py
import streamlit as st
from PIL import Image
import google.generativeai as genai
import base64
import io

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

# File uploader for MRI images
uploaded_files = st.file_uploader(
    "📤 Upload MRI images (JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.header("🖼 Uploaded Images & AI Diagnosis")
    for uploaded_file in uploaded_files:
        file_bytes = uploaded_file.read()
        # Display image
        image = Image.open(io.BytesIO(file_bytes))
        st.subheader(f"🧾 Image: {uploaded_file.name}")
        st.image(image, use_column_width=True)

        # Prepare base64-encoded image for prompt
        b64 = base64.b64encode(file_bytes).decode("utf-8")
        markdown_img = f"![mri](data:image/jpeg;base64,{b64})"

        # Construct chat prompt
        prompt = (
            "You are a medical AI assistant. Analyze the following brain MRI image for tumor or glioma:\n\n"
            f"{markdown_img}\n\n"
            "Please provide:\n"
            "1. Any detected abnormalities or tumor regions.\n"
            "2. Approximate size or location.\n"
            "3. Medical interpretation helpful for clinicians."
        )

        # Send to Gemini via genai.chat
        with st.spinner("Analyzing image with Gemini..."):
            try:
                response = genai.chat(
                    model="gemini-1.5-flash",
                    messages=[{"author": "user", "content": prompt}]
                )
                # Display AI diagnosis
                st.markdown("### 🧠 AI Diagnosis")
                # Depending on response structure
                content = None
                if hasattr(response, 'last') and hasattr(response.last, 'content'):
                    content = response.last.content
                elif 'choices' in response and response['choices']:
                    content = response['choices'][0]['message']['content']
                st.write(content)
            except Exception as e:
                st.error(f"❌ Analysis failed: {e}")
else:
    st.info("Upload up to four MRI images to begin analysis.")
