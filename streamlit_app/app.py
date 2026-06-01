import sys
import os
import time
from datetime import datetime
import pandas as pd

# Fix import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import numpy as np
from src.pipeline import process_image
from src.ocr import PlateOCR

# -----------------------------
# PAGE SETUP & STYLING
# -----------------------------
st.set_page_config(
    page_title="ANPR Dashboard",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Premium Glassmorphic CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

/* Main font and styling overrides */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
}

.stApp {
    background: radial-gradient(circle at 10% 20%, rgb(17, 24, 39) 0%, rgb(9, 12, 22) 100%);
    color: #E2E8F0;
}

/* Header Text styling */
.gradient-title {
    background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 50%, #1D4ED8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    text-align: center;
}

.gradient-subtitle {
    font-size: 1.1rem;
    color: #94A3B8;
    text-align: center;
    margin-bottom: 2rem;
}

/* Glassmorphic Container */
.glass-container {
    background: rgba(30, 41, 59, 0.45);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
}

/* Number Plate Badge */
.plate-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.0rem;
    font-weight: 700;
    color: #10B981;
    background: rgba(16, 185, 129, 0.1);
    border: 2px solid rgba(16, 185, 129, 0.4);
    border-radius: 12px;
    padding: 10px 20px;
    display: inline-block;
    letter-spacing: 3px;
    margin: 15px 0;
    text-shadow: 0 0 10px rgba(16, 185, 129, 0.2);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    text-align: center;
}

/* Warning / Status Card */
.status-card {
    background: rgba(245, 158, 11, 0.08);
    border-left: 4px solid #F59E0B;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 10px 0;
    color: #FBBF24;
}

/* Success Card */
.success-card {
    background: rgba(16, 185, 129, 0.08);
    border-left: 4px solid #10B981;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 10px 0;
    color: #34D399;
}

/* Error Card */
.error-card {
    background: rgba(239, 68, 68, 0.08);
    border-left: 4px solid #EF4444;
    padding: 12px 16px;
    border-radius: 4px;
    margin: 10px 0;
    color: #FCA5A5;
}

/* Custom styles for sidebar controls */
[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.7) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# CACHED ENGINES & HELPERS
# -----------------------------
@st.cache_resource
def load_ocr_engine(languages_tuple):
    # Tuple keys can be cached by st.cache_resource
    return PlateOCR(langs=list(languages_tuple))

def hex_to_bgr(hex_str):
    hex_str = hex_str.lstrip('#')
    rgb = tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0]) # BGR format

# Initialize session state for history and tracking
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_processed' not in st.session_state:
    st.session_state.last_processed = None
if 'current_outputs' not in st.session_state:
    st.session_state.current_outputs = []

# -----------------------------
# SIDEBAR CONTROL PANEL
# -----------------------------
st.sidebar.image("https://img.icons8.com/nolan/96/car.png", width=70)
st.sidebar.title("ANPR Settings")

# 1. Image Selection Mode
input_mode = st.sidebar.radio("Image Input Mode", ["Upload Image", "Use Sample Image"])

uploaded_file = None
selected_sample = None

if input_mode == "Upload Image":
    uploaded_file = st.sidebar.file_uploader("Upload vehicle image", type=["jpg", "png", "jpeg"])
else:
    samples_dict = {
        "test1.jpg (Car - Straight View)": "data/samples/test1.jpg",
        "test2.jpg (Car - Devanagari/Hindi Plate)": "data/samples/test2.jpg",
        "test3.jpg (Plate Crop - Small)": "data/samples/test3.jpg",
        "test4.jpg (Car - Angled Plate)": "data/samples/test4.jpg",
        "test5.png (Car - High Contrast Shadow)": "data/samples/test5.png",
        "test6.png (Sedan - Night View)": "data/samples/test6.png",
        "test7.jpg (Extreme Close-up)": "data/samples/test7.jpg"
    }
    selected_sample = st.sidebar.selectbox("Select sample image", list(samples_dict.keys()))
    sample_path = samples_dict[selected_sample]

st.sidebar.markdown("---")
st.sidebar.subheader("Recognition Tuning")

# 2. OCR Language selector
ocr_lang_option = st.sidebar.selectbox(
    "OCR Engine Language",
    ["English Only (Recommended)", "English + Hindi"]
)
ocr_langs = ('en',) if ocr_lang_option.startswith("English Only") else ('en', 'hi')
ocr_engine = load_ocr_engine(ocr_langs)

# 3. Confidence Thresholds
det_thresh = st.sidebar.slider("Min Detection Conf", 0.10, 1.00, 0.45, 0.05)
ocr_thresh = st.sidebar.slider("Min OCR Conf", 0.10, 1.00, 0.30, 0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Bounding Box Styling")

# 4. Box Rendering Customization
box_color = st.sidebar.color_picker("Box & Outline Color", "#10B981")
line_thickness = st.sidebar.slider("Line Thickness", 1, 5, 2, 1)
font_scale = st.sidebar.slider("Text Font Scale", 0.4, 1.5, 0.7, 0.1)

# -----------------------------
# MAIN APP BODY
# -----------------------------
st.markdown('<div class="gradient-title">🚗 Automatic Number Plate Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="gradient-subtitle">State-of-the-Art Deep Learning License Plate Detection & Character Recognition Dashboard</div>', unsafe_allow_html=True)

# Determine the image to read
img = None
current_file_name = ""

if input_mode == "Upload Image" and uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    current_file_name = uploaded_file.name
elif input_mode == "Use Sample Image":
    img = cv2.imread(sample_path)
    current_file_name = selected_sample

if img is None:
    # Landing Page - Dashboard Explanation
    st.markdown("""
    <div class="glass-container">
        <h3>👋 Welcome to the ANPR Dashboard!</h3>
        <p>To begin vehicle plate text extraction, please use one of the following methods:</p>
        <ol>
            <li><b>Select 'Use Sample Image'</b> in the sidebar to try pre-configured test plates immediately.</li>
            <li><b>Upload your own image</b> by choosing 'Upload Image' in the settings panel.</li>
        </ol>
        <hr style="border-color: rgba(255,255,255,0.05)"/>
        <h4>🧠 How it works under the hood:</h4>
        <ul>
            <li><b>Detection:</b> A custom-trained YOLOv8 convolutional neural network identifies license plate bounding boxes in real-time.</li>
            <li><b>OCR Preprocessing:</b> Plate crops are automatically normalized, upscaled, and enhanced using adaptive binarization (Otsu's thresholding) and CLAHE contrast adjustments.</li>
            <li><b>OCR Text Extraction:</b> The EasyOCR neural engine parses Latin and Hindi characters, followed by a smart position-based pattern mapping post-processor for plate correction.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
else:
    # Pipeline Processing (Instantly cached if the file remains the same)
    if st.session_state.last_processed != current_file_name:
        with st.spinner("Processing image through YOLOv8 and OCR engines..."):
            outputs = process_image(img, ocr_engine=ocr_engine)
            st.session_state.current_outputs = outputs
            st.session_state.last_processed = current_file_name
            
            # Log results in history
            for out in outputs:
                st.session_state.history.append({
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Source Image": current_file_name,
                    "Detected Plate": out["text"],
                    "Detection Conf": f"{out['det_conf']*100:.1f}%",
                    "OCR Conf": f"{out['ocr_conf']*100:.1f}%"
                })
    else:
        outputs = st.session_state.current_outputs

    # Filter outputs based on selected thresholds
    valid_outputs = [
        out for out in outputs 
        if out["det_conf"] >= det_thresh and out["ocr_conf"] >= ocr_thresh
    ]

    # Two column layout: Image Visualization & Extracted Results
    col_vis, col_res = st.columns([3, 2])

    with col_vis:
        st.markdown('<div class="glass-container"><h4>🖼️ Detection Visualization</h4>', unsafe_allow_html=True)
        
        img_draw = img.copy()
        bgr_color = hex_to_bgr(box_color)

        for out in valid_outputs:
            x1, y1, x2, y2 = out["bbox"]
            text = out["text"]
            det_conf = out["det_conf"]

            # Draw styled bounding box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), bgr_color, line_thickness)
            
            # Draw Outline/Shadow Label for high legibility
            label = f"{text if text else 'Plate'} ({det_conf*100:.1f}%)"
            cv2.putText(
                img_draw, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (0, 0, 0), 
                line_thickness + 2,
                cv2.LINE_AA
            )
            cv2.putText(
                img_draw, 
                label, 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                bgr_color, 
                line_thickness,
                cv2.LINE_AA
            )

        st.image(img_draw, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_res:
        st.markdown('<div class="glass-container"><h4>📊 Recognition Statistics</h4>', unsafe_allow_html=True)

        if not outputs:
            st.markdown('<div class="error-card">⚠️ No license plates were detected in this image. Try lowering the "Min Detection Conf" slider in the sidebar.</div>', unsafe_allow_html=True)
        elif not valid_outputs:
            st.markdown('<div class="status-card">⚠️ Plates were detected, but they fall below the active confidence thresholds. Please lower the thresholds in the sidebar.</div>', unsafe_allow_html=True)
        else:
            for idx, out in enumerate(valid_outputs):
                x1, y1, x2, y2 = out["bbox"]
                text = out["text"]
                det_conf = out["det_conf"]
                ocr_conf = out["ocr_conf"]
                crop = out["crop"]

                st.markdown(f"##### Plate #{idx+1}")
                
                # Side-by-side: crop image and plate text
                col_crop, col_badge = st.columns([1, 1])
                with col_crop:
                    st.image(crop, channels="BGR", caption="Plate Crop", use_container_width=True)
                with col_badge:
                    plate_display = text if text else "NOT CLEAR"
                    st.markdown(f'<div class="plate-badge">{plate_display}</div>', unsafe_allow_html=True)

                # Metric values and progress indicators
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    st.metric("Detection Confidence", f"{det_conf*100:.1f}%")
                    st.progress(float(det_conf))
                with col_m2:
                    st.metric("OCR Confidence", f"{ocr_conf*100:.1f}%")
                    st.progress(float(ocr_conf))

                # Interactive warnings and quality alerts
                if ocr_conf < 0.15:
                    st.markdown('<div class="error-card">❌ OCR failed to extract reliable characters. Please try another image.</div>', unsafe_allow_html=True)
                elif ocr_conf < 0.35:
                    st.markdown('<div class="status-card">⚠️ Low OCR confidence. Certain characters may be mistranscribed.</div>', unsafe_allow_html=True)
                
                if crop.shape[0] < 45 or crop.shape[1] < 90:
                    st.markdown('<div class="status-card">⚠️ Bounding box resolution is very small. OCR accuracy might be impaired.</div>', unsafe_allow_html=True)

                if idx < len(valid_outputs) - 1:
                    st.markdown('<hr style="border-color: rgba(255,255,255,0.05); margin: 15px 0;"/>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# SESSION DETECTION LOG
# -----------------------------
if st.session_state.history:
    st.markdown('<div class="glass-container"><h4>📋 Session History Log</h4>', unsafe_allow_html=True)
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist, use_container_width=True)
    
    # Export log to CSV
    csv_data = df_hist.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Export History to CSV",
        data=csv_data,
        file_name=f"anpr_session_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
    st.markdown('</div>', unsafe_allow_html=True)