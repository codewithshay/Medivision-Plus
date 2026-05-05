import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
import requests
import pandas as pd
import sqlite3
import re 
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

# --- 4. UTILITIES & IMAGE VALIDATION ---
@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

def is_medical_scan(img):
    """Checks if the image is grayscale enough to be an MRI/CT scan."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    avg_saturation = np.mean(hsv[:, :, 1])
    # Standard medical scans have very low saturation
    return avg_saturation < 40 

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

# --- 5. LOGIN / SIGNUP SCREEN ---
if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.title("🏥 MEDIVISION PLUS")
        st.subheader("Advanced Multi-Organ Diagnostic AI")
        
        lottie_med = load_lottieurl("https://lottie.host/8664188b-8772-4d2c-8097-40d164d1f56a/I9QG6E3KOn.json")
        if lottie_med: st_lottie(lottie_med, height=200, key="med_anim")
        else: st.markdown("### 🩺")
        
        tab1, tab2 = st.tabs(["Secure Login", "Patient Registration"])
        
        with tab1:
            l_phone = st.text_input("Phone Number", key="login_phone")
            l_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login to Portal"):
                user = login_user(l_phone, l_pass)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_phone = str(user[0])
                    st.session_state.user_name = user[2]
                    st.rerun()
                else:
                    st.error("Access Denied: Invalid credentials.")

        with tab2:
            r_name = st.text_input("Patient Full Name")
            r_phone = st.text_input("Phone Number (10 digits)", key="reg_phone")
            r_age = st.number_input("Age", 1, 120)
            r_pass = st.text_input("Create Password", type="password", key="reg_pass")
            
            if st.button("Initialize Account"):
                is_name_valid = bool(re.match(r"^[A-Za-z\s]+$", r_name))
                is_phone_valid = bool(re.match(r"^\d{10}$", r_phone))
                
                if not r_name or not r_phone:
                    st.warning("Please fill in all clinical fields.")
                elif not is_name_valid:
                    st.error("Invalid Name: Use alphabets only.")
                elif not is_phone_valid:
                    st.error("Invalid Phone: Must be 10 digits.")
                elif len(r_pass) < 4:
                    st.error("Security Alert: Password must be at least 4 characters.")
                else:
                    if add_user(r_phone, r_pass, r_name, r_age):
                        st.success(f"Registered {r_name}! Switch to Login tab.")
                    else:
                        st.error("Identity Error: Phone number already exists.")
    st.stop()

# --- 6. MAIN APP INTERFACE ---
st.sidebar.markdown(f"### 👤 Patient Profile: \n**{st.session_state.user_name}**")
st.sidebar.markdown("---")
menu = st.sidebar.radio("Navigation", ["Diagnostic Hub", "My History", "Logout"])

if menu == "Logout":
    logout()

elif menu == "My History":
    st.title("📋 Patient Diagnostic Records")
    history_df = get_history(st.session_state.user_phone)
    if not history_df.empty:
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No clinical history found.")

elif menu == "Diagnostic Hub":
    st.title("🩺 AI Diagnostic Module")
    config = {
        "Brain Tumor (MRI)": {"size": 64, "type": "Brain", "path": os.path.join(MODELS_DIR, "BrainTumor_New.h5"), "labels": ["Tumor Detected", "Normal"]},
        "Skin Cancer (Dermoscopy)": {"size": 128, "type": "Skin", "path": os.path.join(MODELS_DIR, "mobilenetv2_fast_highacc.keras"), "labels": ["Benign", "Malignant"]},
        "Lung Tumor (CT Scan)": {"size": 224, "type": "Lung", "path": os.path.join(MODELS_DIR, "lung_model.keras"), "labels": ["Normal", "Tumor Detected"]}
    }
    module = st.selectbox("Select Target Organ", list(config.keys()))
    current = config[module]
    uploader = st.file_uploader(f"Upload {module} scan", type=["jpg", "png", "jpeg"])

    if uploader:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            raw_img, proc_img = preprocess_image(uploader, current["size"], current["type"])
            st.image(raw_img, caption="Original Input", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            if st.button("🚀 START AI ANALYSIS"):
                # VALIDATION CHECK: Is it a grayscale medical image?
                if not is_medical_scan(raw_img):
                    st.error("❌ Non-Medical Image Detected. Please upload a valid grayscale MRI or CT scan.")
                else:
                    with st.spinner("Analyzing Imaging Data..."):
                        if os.path.exists(current["path"]):
                            model = load_selected_model(current["path"])
                            pred = model.predict(proc_img)
                            idx = np.argmax(pred)
                            label = current["labels"][idx]
                            conf = float(np.max(pred) * 100)

                            if conf < 75.0:
                                st.warning("⚠️ Low AI Confidence: The image does not clearly match medical patterns. Please provide a high-quality scan.")
                            else:
                                color = "#F87171" if "Tumor" in label or "Malignant" in label else "#34D399"
                                st.markdown(f"### Diagnosis Outcome:")
                                st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
                                st.metric("Clinical Confidence Level", f"{conf:.2f}%")
                                save_history(st.session_state.user_phone, module, label, conf)
                        else: st.error("Model not found.")
            st.markdown('</div>', unsafe_allow_html=True)

# --- 7. SECURE ADMIN ANALYTICS ---
ADMIN_PHONE = "8638968521"
if st.session_state.user_phone == ADMIN_PHONE:
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("🔓 Developer Analytics"):
        st.divider()
        st.header("📊 Admin Dashboard")
        try:
            conn = sqlite3.connect("patients.db")
            df_users = pd.read_sql_query("SELECT name, phone, age FROM users", conn)
            df_logs = pd.read_sql_query("SELECT * FROM history ORDER BY date DESC", conn)
            conn.close()
            st.metric("Total Patients", len(df_users))
            st.dataframe(df_users, use_container_width=True)
            st.subheader("Global System Logs")
            st.dataframe(df_logs, use_container_width=True)
        except Exception as e: st.error(f"DB Error: {e}")

st.sidebar.markdown("---")
st.sidebar.caption("MEDIVISION PLUS v2.0 | ADTU 2026")