import streamlit as st
import cv2
import time
import os
import pygame
import yaml
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
import threading 

# ==========================================================
# MODULE 1: EXTERNAL DEPENDENCIES & CORE LOGIC
# ==========================================================
from database import init_db, save_detection_to_db, fetch_history_from_db
from alerts import send_whatsapp_message, trigger_alarm
from yolo_model import load_yolo_model

# ==========================================================
# MODULE 2: THREADED CAMERA ENGINE (Zero-Lag Handling)
# ==========================================================
class ThreadedCamera:
    def __init__(self, source):
        self.capture = cv2.VideoCapture(source)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.status, self.frame = self.capture.read()
        self.is_running = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        while self.is_running:
            if self.capture.isOpened():
                self.status, self.frame = self.capture.read()
            time.sleep(0.01)

    def read(self):
        return self.status, self.frame

    def stop(self):
        self.is_running = False
        if self.thread.is_alive(): self.thread.join()
        if self.capture is not None: self.capture.release()

# ==========================================================
# MODULE 3: APP INITIALIZATION & AUTHENTICATION
# ==========================================================
st.set_page_config(page_title="Fire Command Center", layout="wide")

def load_auth():
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
    return stauth.Authenticate(
        config['credentials'], config['cookie']['name'],
        config['cookie']['key'], config['cookie']['expiry_days']
    )

authenticator = load_auth()
authenticator.login()

# ==========================================================
# MODULE 4: MAIN INTERFACE & CONTROL LOOP
# ==========================================================
if st.session_state.get("authentication_status"):
    # UI Sidebar
    st.sidebar.markdown(f"### 👤 {st.session_state.get('name')}")
    authenticator.logout('Logout', 'sidebar')

    # Initializing Backend
    init_db()
    model = load_yolo_model()
    SAVE_DIR = os.path.join(os.getcwd(), "detected_fires")
    os.makedirs(SAVE_DIR, exist_ok=True)

    # State Management
    if 'system_active' not in st.session_state: st.session_state.system_active = False
    
    # UI Controls
    if st.session_state.get("username") == "admin":
        st.sidebar.title("⚙️ Admin Controls")
        conf = st.sidebar.slider("Confidence", 0.0, 1.0, 0.65)
        src = st.sidebar.text_input("📹 Source", "0")
        if st.sidebar.button("🟢 Start"): st.session_state.system_active = True
        if st.sidebar.button("🔴 Stop"): st.session_state.system_active = False
    
    # Core Processing
    st.title("🔥 Smart Fire Detection Command Center")
    video_placeholder = st.empty()
    
    if st.session_state.system_active:
        # Camera Setup
        if st.session_state.get('camera') is None:
            st.session_state.camera = ThreadedCamera(int(src) if src.isdigit() else src)
        
        # Inference Loop
        while st.session_state.system_active:
            ret, frame = st.session_state.camera.read()
            if not ret: continue
            
            # Logic
            results = model(frame, conf=conf)
            # ... (detection code) ...
            video_placeholder.image(cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB), use_container_width=True)
            
    else:
        video_placeholder.info("System Offline.")