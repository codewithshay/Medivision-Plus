import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
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
    st.session_state.logged_in = False
    st.rerun()

# --- 3. PAGE CONFIG & THEME ---
st.set_page_config(page_title="NeuroScan AI | Diagnostic Portal", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0F172A; color: #E2E8F0; }
    [data-testid="stSidebar"] { background-color: #1E293B; border-right: 1px solid #334155; }
    .medical-card { background-color: #1E293B; padding: 25px; border-radius: 12px; border: 1px solid #334155; }
    </style>
    """, unsafe_allow_html=True)

# --- 4. AI MODEL UTILITIES ---
@st.cache_resource
def load_selected_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

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
        st.title("🩺 NeuroScan AI Portal")
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            # Added unique key="login_phone"
            l_phone = st.text_input("Phone Number", key="login_phone")
            # Added unique key="login_pass"
            l_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Login"):
                user = login_user(l_phone, l_pass)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_phone = user[0]
                    st.session_state.user_name = user[2]
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

        with tab2:
            r_name = st.text_input("Patient Full Name")
            # Added unique key="reg_phone"
            r_phone = st.text_input("Phone Number", key="reg_phone")
            r_age = st.number_input("Age", 1, 120)
            # Added unique key="reg_pass"
            r_pass = st.text_input("Create Password", type="password", key="reg_pass")
            if st.button("Create Account"):
                if add_user(r_phone, r_pass, r_name, r_age):
                    st.success("Registration successful! Go to Login tab.")
                else:
                    st.error("User already exists.")
    st.stop()

# --- 6. MAIN APP (LOGGED IN ONLY) ---
st.sidebar.title(f"Patient: {st.session_state.user_name}")
menu = st.sidebar.radio("Navigation", ["Diagnostic Hub", "My History", "Logout"])

if menu == "Logout":
    logout()

elif menu == "My History":
    st.title("📋 Your Diagnostic History")
    history_df = get_history(st.session_state.user_phone)
    if not history_df.empty:
        st.dataframe(history_df, width="stretch")
    else:
        st.info("No records found. Visit the Diagnostic Hub to start.")

elif menu == "Diagnostic Hub":
    st.title("🩺 AI Diagnostic Module")
    
    config = {
        "Brain Tumor (MRI)": {
            "size": 64, "type": "Brain",
            "path": os.path.join(MODELS_DIR, "BrainTumor_New.h5"), 
            "labels": ["Tumor Detected", "Normal"]
        },
        "Skin Cancer (Dermoscopy)": {
            "size": 128, "type": "Skin",
            "path": os.path.join(MODELS_DIR, "mobilenetv2_fast_highacc.keras"), 
            "labels": ["Benign", "Malignant"]
        },
        "Lung Tumor (CT Scan)": {
            "size": 224, "type": "Lung",
            "path": os.path.join(MODELS_DIR, "lung_model.keras"), 
            "labels": ["Normal", "Tumor Detected"]
        }
    }

    module = st.selectbox("Select Organ for Scan Analysis", list(config.keys()))
    current = config[module]
    
    uploader = st.file_uploader(f"Upload {module} scan", type=["jpg", "png", "jpeg"])

    if uploader:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            raw_img, proc_img = preprocess_image(uploader, current["size"], current["type"])
            st.image(raw_img, caption="Uploaded Scan", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            if st.button("🚀 INITIATE ANALYSIS"):
                if os.path.exists(current["path"]):
                    model = load_selected_model(current["path"])
                    pred = model.predict(proc_img)
                    idx = np.argmax(pred)
                    label = current["labels"][idx]
                    conf = float(np.max(pred) * 100)

                    color = "#F87171" if "Tumor" in label or "Malignant" in label else "#34D399"
                    st.markdown(f"<h2 style='color: {color};'>{label}</h2>", unsafe_allow_html=True)
                    st.metric("AI Confidence Level", f"{conf:.2f}%")

                    save_history(st.session_state.user_phone, module, label, conf)
                    st.success("Diagnosis recorded in your history.")
                else:
                    st.error("Model file missing.")
            st.markdown('</div>', unsafe_allow_html=True)