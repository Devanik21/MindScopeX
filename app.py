# app.py
import streamlit as st
from PIL import Image
import google.generativeai as genai
import io

# Streamlit page config
st.set_page_config(
    page_title="Medical Image Analytics",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🧠 Medical Image Advanced Analytics")
st.markdown(
    "Upload brain MRI images below and let Gemini provide advanced insights to assist cancer patients."
)

# Sidebar: Gemini API Key Input
st.sidebar.header("Configuration")
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
    "Upload MRI images (JPEG, PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    st.header("Uploaded Images & Analysis")
    for uploaded_file in uploaded_files:
        # Display image
        st.subheader(f"Image: {uploaded_file.name}")
        image = Image.open(io.BytesIO(uploaded_file.read()))
        st.image(image, use_column_width=True)

        # Call Gemini API for advanced analytics
        with st.spinner("Analyzing image with Gemini..."):
            try:
                # Example image analysis call; adjust features as needed
                analysis = genai.analyze(
                    model="gemini-2.5-flash",
                    image=image,
                    features=[
                        "LABEL_DETECTION",
                        "OBJECT_DETECTION",
                        "TEXT_DETECTION"
                    ]
                )
                st.json(analysis)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
else:
    st.info("Upload up to four MRI images to begin analysis.")
