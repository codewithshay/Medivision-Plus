import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import pandas as pd
import sqlite3
import re 
from tensorflow.keras.utils import normalize
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
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200: return None
        return r.json()
    except: return None

# --- 3. PAGE CONFIG & THEME ---
st.set_page_config(page_title="MEDIVISION PLUS | AI Diagnostic Platform", layout="wide")

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
    }
    </style>
    """, unsafe_allow_html=True)

# --- 4. UTILITIES & SMART IMAGE VALIDATION ---
@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def is_valid_medical_image(img, scan_type):
    """
    Smarter Validation:
    - Brain/Lung (MRI/CT): Must be grayscale (Low saturation)
    - Skin (Dermoscopy): Must have color (Higher saturation)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    
    if scan_type in ["Brain", "Lung"]:
        return avg_saturation < 40 # MRI/CT should be grayscale
    elif scan_type == "Skin":
        return avg_saturation > 5 # Skin scans must have color
    return True

def preprocess_image(uploaded_file, target_size, model_type):
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None: return None, None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (target_size, target_size))
        
        if model_type == "Brain":
            # Matching your training: normalize(axis=1)
            img_array = np.array(img_resized)
            img_array = normalize(img_array, axis=1)
        else:
            # Using MobileNetV2 preprocessing for Lung/Skin
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(img_resized))
            
        return img, np.expand_dims(img_array, axis=0)
    except:
        return None, None

# --- 5. LOGIN / SIGNUP SCREEN ---
if not st.session_state.logged_in:
    col_visual, col_auth = st.columns([1.3, 1], gap="large")

    with col_visual:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: left; color: #3B82F6; font-size: 3rem;'>🏥 MEDIVISION PLUS</h1>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 25px;">
                <div style="background-color: #1E293B; padding: 25px; border-radius: 12px; border-top: 5px solid #60A5FA; text-align: center;">
                    <h1 style="margin:0; font-size: 2.5rem;">🧠</h1>
                    <b style="color: #60A5FA; font-size: 1rem; letter-spacing: 1px;">BRAIN MRI</b>
                </div>
                <div style="background-color: #1E293B; padding: 25px; border-radius: 12px; border-top: 5px solid #34D399; text-align: center;">
                    <h1 style="margin:0; font-size: 2.5rem;">🫁</h1>
                    <b style="color: #34D399; font-size: 1rem; letter-spacing: 1px;">LUNG CT</b>
                </div>
                <div style="background-color: #1E293B; padding: 25px; border-radius: 12px; border-top: 5px solid #F87171; text-align: center;">
                    <h1 style="margin:0; font-size: 2.5rem;">🔬</h1>
                    <b style="color: #F87171; font-size: 1rem; letter-spacing: 1px;">SKIN CANCER</b>
                </div>
                <div style="background-color: #1E293B; padding: 25px; border-radius: 12px; border-top: 5px solid #A78BFA; text-align: center;">
                    <h1 style="margin:0; font-size: 2.5rem;">📱</h1>
                    <b style="color: #A78BFA; font-size: 1rem; letter-spacing: 1px;">OMNI-DEVICE</b>
                </div>
            </div>
        """, unsafe_allow_html=True)

        lottie_med = load_lottieurl("https://lottie.host/8664188b-8772-4d2c-8097-40d164d1f56a/I9QG6E3KOn.json")
        if lottie_med: st_lottie(lottie_med, height=320, key="login_anim")

    with col_auth:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>CLINICAL GATEWAY</h3>", unsafe_allow_html=True)
            tab1, tab2 = st.tabs(["🔒 LOGIN", "➕ REGISTER"])
            
            with tab1:
                l_phone = st.text_input("User ID", placeholder="8638968521", key="login_phone")
                l_pass = st.text_input("Security Key", type="password", key="login_pass")
                if st.button("AUTHORIZE ACCESS", use_container_width=True):
                    user = login_user(l_phone, l_pass)
                    if user:
                        st.session_state.logged_in = True
                        st.session_state.user_phone = str(user[0])
                        st.session_state.user_name = user[2]
                        st.rerun()
                    else: st.error("Authentication Denied")

            with tab2:
                r_name = st.text_input("Full Patient Name")
                r_phone = st.text_input("Mobile (10 digits)")
                r_age = st.number_input("Age", 1, 120, 21)
                r_pass = st.text_input("Password", type="password")
                if st.button("INITIALIZE PROFILE", use_container_width=True):
                    if bool(re.match(r"^[A-Za-z\s]+$", r_name)) and bool(re.match(r"^\d{10}$", r_phone)):
                        if add_user(r_phone, r_pass, r_name, r_age): st.success("Account Ready!")
                        else: st.error("Profile already exists.")
                    else: st.error("Check inputs (Name: Alphabets, Phone: 10 digits).")
    st.stop()

# --- 6. MAIN APP INTERFACE ---
st.sidebar.markdown(f"### 👤 Profile: \n**{st.session_state.user_name}**")
menu = st.sidebar.radio("Navigation", ["Diagnostic Hub", "My History", "Logout"])

if menu == "Logout": logout()

elif menu == "My History":
    st.title("📋 Diagnostic Records")
    history_df = get_history(st.session_state.user_phone)
    if not history_df.empty: st.dataframe(history_df, use_container_width=True)
    else: st.info("No records found.")

elif menu == "Diagnostic Hub":
    st.title("🩺 AI Diagnostic Module")
    config = {
        "Brain Tumor (MRI)": {
            "size": 64, 
            "type": "Brain", 
            "path": os.path.join(MODELS_DIR, "BrainTumor_New.h5"), 
            "labels": ["Normal", "Tumor Detected"] # Matches CATEGORIES = ['no_tumor', 'tumor']
        },
        "Skin Cancer (Dermoscopy)": {
            "size": 128, 
            "type": "Skin", 
            "path": os.path.join(MODELS_DIR, "mobilenetv2_fast_highacc.keras"), 
            "labels": ["Benign", "Malignant"]
        },
        "Lung Tumor (CT Scan)": {
            "size": 224, 
            "type": "Lung", 
            "path": os.path.join(MODELS_DIR, "lung_model.keras"), 
            "labels": ["Normal", "Tumor Detected"]
        }
    }
    module = st.selectbox("Select Diagnostic Suite", list(config.keys()))
    current = config[module]
    uploader = st.file_uploader(f"Upload {module} scan", type=["jpg", "png", "jpeg"])

    if uploader:
        c1, c2 = st.columns(2)
        raw_img, proc_img = preprocess_image(uploader, current["size"], current["type"])
        with c1:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            st.image(raw_img, caption="Diagnostic Input", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            if st.button("🚀 EXECUTE AI ANALYSIS"):
                if not is_valid_medical_image(raw_img, current["type"]):
                    st.error(f"❌ Invalid Image for {current['type']} scan.")
                else:
                    with st.spinner("Processing Clinical Data..."):
                        if os.path.exists(current["path"]):
                            model = load_selected_model(current["path"])
                            pred = model.predict(proc_img)
                            idx = np.argmax(pred)
                            label = current["labels"][idx]
                            conf = float(np.max(pred) * 100)

                            if conf < 75.0: st.warning("⚠️ Inconclusive Analysis.")
                            else:
                                color = "#F87171" if "Tumor" in label or "Malignant" in label else "#34D399"
                                st.markdown(f"### Assessment: <span style='color:{color};'>{label}</span>", unsafe_allow_html=True)
                                st.metric("AI Confidence", f"{conf:.2f}%")
                                save_history(st.session_state.user_phone, module, label, conf)
                        else: st.error("Model Engine missing.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 7. SECURE ADMIN ANALYTICS ---
ADMIN_PHONE = "8638968521"
if st.session_state.user_phone == ADMIN_PHONE:
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("🔓 Developer Analytics"):
        conn = sqlite3.connect("patients.db")
        st.dataframe(pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn), use_container_width=True)
        conn.close()

st.sidebar.markdown("---")
st.sidebar.caption("MEDIVISION PLUS v2.0 | ADTU 2026")