import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import pandas as pd
import sqlite3
import re 
import matplotlib.cm as cm
from streamlit_lottie import st_lottie
from database import create_db, add_user, login_user, save_history, get_history

# --- 1. INITIAL SETUP & DATABASE ---
create_db()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- 2. AUTHENTICATION STATE ---
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_phone = ""
    st.session_state.user_name = ""

def logout():
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# --- 3. PAGE CONFIG & MODERN CSS ---
st.set_page_config(page_title="MEDIVISION PLUS | Explainable AI Portal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #1E293B; border-right: 1px solid #334155; }
    .medical-card { 
        background-color: #1E293B; 
        padding: 25px; 
        border-radius: 15px; 
        border: 1px solid #3B82F6; 
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
    }
    h1, h2, h3 { color: #3B82F6 !important; font-weight: 700; }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        border-radius: 8px;
        width: 100%;
        font-weight: bold;
        height: 3.2em;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. SMART GRAD-CAM & PROCESSING ---
@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def generate_gradcam(img_array, model, pred_index=None):
    """Automatically finds the last 4D conv layer to generate attention heatmap."""
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if len(layer.output_shape) == 4:
            last_conv_layer_name = layer.name
            break
    
    if not last_conv_layer_name:
        return None

    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def apply_heatmap(heatmap, original_img):
    """Overlays the heatmap onto the diagnostic image."""
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_img.shape[1], original_img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * 0.4 + original_img
    return tf.keras.preprocessing.image.array_to_img(superimposed_img)

def is_medical_scan(img):
    """Detects grayscale/medical imaging vs consumer photos."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return np.mean(hsv[:, :, 1]) < 40 

def preprocess_image(uploaded_file, target_size, model_type):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (target_size, target_size))
    if model_type == "Brain":
        img_array = img_resized.astype('float32') / 255.0
    else:
        img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
    return img, np.expand_dims(img_array, axis=0)

# --- 5. VISUAL LOGIN GATEWAY ---
if not st.session_state.logged_in:
    col_v, col_a = st.columns([1.3, 1], gap="large")
    with col_v:
        st.markdown("<br><h1 style='color: #3B82F6; font-size: 3.5rem;'>🏥 MEDIVISION PLUS</h1>", unsafe_allow_html=True)
        st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 30px;">
                <div style="background-color: #1E293B; padding: 25px; border-radius: 15px; border-top: 5px solid #60A5FA; text-align: center;">
                    <h1 style="margin:0;">🧠</h1><b style="color: #60A5FA; letter-spacing: 1px;">BRAIN MRI</b>
                </div>
                <div style="background-color: #1E293B; padding: 25px; border-radius: 15px; border-top: 5px solid #34D399; text-align: center;">
                    <h1 style="margin:0;">🫁</h1><b style="color: #34D399; letter-spacing: 1px;">LUNG CT</b>
                </div>
                <div style="background-color: #1E293B; padding: 25px; border-radius: 15px; border-top: 5px solid #F87171; text-align: center;">
                    <h1 style="margin:0;">🔬</h1><b style="color: #F87171; letter-spacing: 1px;">SKIN CANCER</b>
                </div>
                <div style="background-color: #1E293B; padding: 25px; border-radius: 15px; border-top: 5px solid #A78BFA; text-align: center;">
                    <h1 style="margin:0;">👁️</h1><b style="color: #A78BFA; letter-spacing: 1px;">GRAD-CAM XAI</b>
                </div>
            </div>
        """, unsafe_allow_html=True)
        lottie_med = load_lottieurl("https://lottie.host/8664188b-8772-4d2c-8097-40d164d1f56a/I9QG6E3KOn.json")
        if lottie_med: st_lottie(lottie_med, height=350, key="login_anim")
    
    with col_a:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>PORTAL ACCESS</h2>", unsafe_allow_html=True)
            t1, t2 = st.tabs(["🔒 LOGIN", "➕ REGISTER"])
            with t1:
                l_phone = st.text_input("Admin/Patient ID", placeholder="8638968521", key="l_phone")
                l_pass = st.text_input("Security Key", type="password", key="l_pass")
                if st.button("AUTHORIZE SESSION", use_container_width=True):
                    user = login_user(l_phone, l_pass)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_phone = str(user[0]); st.session_state.user_name = user[2]
                        st.rerun()
                    else: st.error("Authentication Denied")
            with t2:
                r_name = st.text_input("Patient Full Name")
                r_phone = st.text_input("Mobile (10 Digits)")
                r_age = st.number_input("Age", 1, 120, 21)
                r_pass = st.text_input("Create Password", type="password")
                if st.button("INITIALIZE PROFILE", use_container_width=True):
                    if bool(re.match(r"^[A-Za-z\s]+$", r_name)) and bool(re.match(r"^\d{10}$", r_phone)):
                        if add_user(r_phone, r_pass, r_name, r_age): st.success("Success! Please Login.")
                        else: st.error("Profile already exists.")
                    else: st.error("Invalid Input: Use alphabets for name and 10 digits for phone.")
    st.stop()

# --- 6. MAIN INTERFACE ---
st.sidebar.markdown(f"### 👤 Profile: \n**{st.session_state.user_name}**")
menu = st.sidebar.radio("Navigation", ["Diagnostic Hub", "History", "Logout"])

if menu == "Logout": logout()
elif menu == "History":
    st.title("📋 Diagnostic History")
    df = get_history(st.session_state.user_phone)
    if not df.empty: st.dataframe(df, use_container_width=True)
    else: st.info("No records found.")

elif menu == "Diagnostic Hub":
    st.title("🩺 XAI Diagnostic Hub")
    config = {
        "Brain Tumor (MRI)": {"size": 64, "type": "Brain", "path": os.path.join(MODELS_DIR, "BrainTumor_New.h5"), "labels": ["Tumor Detected", "Normal"]},
        "Skin Cancer (Dermoscopy)": {"size": 128, "type": "Skin", "path": os.path.join(MODELS_DIR, "mobilenetv2_fast_highacc.keras"), "labels": ["Benign", "Malignant"]},
        "Lung Tumor (CT Scan)": {"size": 224, "type": "Lung", "path": os.path.join(MODELS_DIR, "lung_model.keras"), "labels": ["Normal", "Tumor Detected"]}
    }
    module = st.selectbox("Select Suite", list(config.keys()))
    current = config[module]
    uploader = st.file_uploader(f"Upload {module} scan", type=["jpg", "png", "jpeg"])

    if uploader:
        col1, col2 = st.columns(2)
        raw_img, proc_img = preprocess_image(uploader, current["size"], current["type"])
        with col1:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.image(raw_img, caption="Input Scan", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            if st.button("🚀 EXECUTE AI ANALYSIS"):
                if not is_medical_scan(raw_img): st.error("❌ Invalid Image Source.")
                else:
                    with st.spinner("Analyzing..."):
                        if os.path.exists(current["path"]):
                            model = load_selected_model(current["path"])
                            pred = model.predict(proc_img)
                            idx = np.argmax(pred); label = current["labels"][idx]; conf = float(np.max(pred) * 100)
                            if conf < 75.0: st.warning("⚠️ Ambiguous Imaging.")
                            else:
                                color = "#F87171" if "Tumor" in label or "Malignant" in label else "#34D399"
                                st.markdown(f"### Assessment: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)
                                st.metric("AI Confidence", f"{conf:.2f}%")
                                save_history(st.session_state.user_phone, module, label, conf)
                                # Heatmap Logic
                                st.divider()
                                heatmap = generate_gradcam(proc_img, model)
                                if heatmap is not None:
                                    cam_img = apply_heatmap(heatmap, raw_img)
                                    st.image(cam_img, caption="Red indicates high-priority diagnostic features.", use_container_width=True)
                        else: st.error("Model Engine Offline.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 7. ADMIN DASHBOARD ---
ADMIN_PHONE = "8638968521"
if st.session_state.user_phone == ADMIN_PHONE:
    st.sidebar.divider()
    if st.sidebar.checkbox("🔓 Developer Tools"):
        st.header("📊 Global System Logs")
        conn = sqlite3.connect("patients.db")
        st.dataframe(pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn), use_container_width=True)
        conn.close()

st.sidebar.caption("MEDIVISION PLUS v2.0 | ADTU 2026")