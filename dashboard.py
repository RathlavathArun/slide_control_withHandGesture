# app.py
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from collections import deque
import time

st.set_page_config(layout="wide", page_title="Gesture-Controlled Slide Presenter")

# -------------------------
# Sidebar / Controls
# -------------------------
st.sidebar.title("🎮 Controls")
start_btn = st.sidebar.button("▶️ Start Presentation")
stop_btn = st.sidebar.button("⏹ Stop Presentation")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Gestures:**")
st.sidebar.write("•  Wave Left -> Previous Slide")
st.sidebar.write("•  Wave Right -> Next Slide")
st.sidebar.write("• (2 fingers) -> Zoom In")
st.sidebar.write("•  (3 fingers) -> Zoom Out")
st.sidebar.write("•  (little finger only) -> Close Presentation")
st.sidebar.markdown("---")
pdf_path = st.sidebar.file_uploader("Upload slides (PDF)", type=["pdf"])
st.sidebar.write("If none uploaded, sample placeholder will be used.")

# -------------------------
# Layout
# -------------------------
col1, col2 = st.columns([1, 2])
with col1:
    st.markdown("### 📷 Webcam Feed")
    webcam_placeholder = st.empty()
with col2:
    st.markdown("### 🖼 Slide Display")
    slide_placeholder = st.empty()
    status_placeholder = st.empty()

# -------------------------
# Helper: PDF -> Image
# -------------------------
def render_pdf_page(doc, page_index, zoom, crop_center=None):
    page = doc.load_page(page_index)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    if crop_center:
        cx, cy = crop_center
        w, h = img.size
        crop_w, crop_h = 800, 600
        left = max(0, int(cx - crop_w // 2))
        top = max(0, int(cy - crop_h // 2))
        right = min(w, left + crop_w)
        bottom = min(h, top + crop_h)
        img = img.crop((left, top, right, bottom))
    return img

# -------------------------
# Hand detection utils
# -------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def fingers_up(hand_landmarks):
    tips_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
    fingers = []
    wrist_x = coords[0][0]
    thumb_tip_x = coords[4][0]
    thumb_ip_x = coords[3][0]
    index_x = coords[8][0]
    pinky_x = coords[20][0]
    hand_is_right = index_x < pinky_x
    if hand_is_right:
        thumb_extended = thumb_tip_x < thumb_ip_x
    else:
        thumb_extended = thumb_tip_x > thumb_ip_x
    fingers.append(thumb_extended)
    for t, p in zip(tips_ids[1:], pip_ids[1:]):
        fingers.append(coords[t][1] < coords[p][1])
    return fingers

# -------------------------
# Gesture detection state
# -------------------------
movement_history = deque(maxlen=12)
last_action_time = 0
action_cooldown = 1.0
running = False

# -------------------------
# Load PDF or placeholder
# -------------------------
if pdf_path is not None:
    pdf_bytes = pdf_path.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
else:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 100), "No PDF uploaded.\nPlease upload a PDF in the sidebar.", fontsize=20)
    for i in range(2):
        p = doc.new_page()
        p.insert_text((50, 100), f"Sample Slide {i+2}", fontsize=60)

num_pages = doc.page_count
current_page = 0
zoom_factor = 1.0
zoom_center = None

# -------------------------
# Start / Stop handlers
# -------------------------
if "presenting" not in st.session_state:
    st.session_state.presenting = False
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = False

if start_btn:
    st.session_state.presenting = True
    st.session_state.stop_flag = False
if stop_btn:
    st.session_state.presenting = False
    st.session_state.stop_flag = True

# -------------------------
# Main loop
# -------------------------
if st.session_state.presenting:
    status_placeholder.info("Presentation running. Place your hand in the webcam box for gestures.")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_placeholder.error("Cannot open webcam.")
    else:
        with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.5, max_num_hands=1) as hands:
            try:
                while st.session_state.presenting and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = hands.process(frame_rgb)
                    h, w, _ = frame.shape
                    gesture_text = "Detected Gesture: None"

                    if results.multi_hand_landmarks:
                        hand_landmarks = results.multi_hand_landmarks[0]
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        wrist = hand_landmarks.landmark[0]
                        wrist_x_px = wrist.x * w
                        wrist_y_px = wrist.y * h
                        movement_history.append(wrist_x_px)
                        dx = movement_history[-1] - movement_history[0] if len(movement_history) >= 6 else 0
                        fingers = fingers_up(hand_landmarks)
                        count_up = sum(fingers)
                        pinky_up = fingers[4] and not any(fingers[:4])
                        two_fingers = (count_up == 2)
                        three_fingers = (count_up == 3)
                        now = time.time()

                        if now - last_action_time > action_cooldown:
                            if dx > 80:
                                current_page = min(num_pages - 1, current_page + 1)
                                gesture_text = "Detected Gesture: Wave Right -> Next Slide"
                                last_action_time = now
                                movement_history.clear()
                            elif dx < -80:
                                current_page = max(0, current_page - 1)
                                gesture_text = "Detected Gesture: Wave Left -> Prev Slide"
                                last_action_time = now
                                movement_history.clear()
                            elif two_fingers:
                                zoom_factor = min(3.0, zoom_factor + 0.2)
                                zoom_center = (int(wrist_x_px * zoom_factor), int(wrist_y_px * zoom_factor))
                                gesture_text = "Detected Gesture: 2 Fingers -> Zoom In"
                                last_action_time = now
                            elif three_fingers:
                                zoom_factor = max(0.5, zoom_factor - 0.2)
                                zoom_center = (int(wrist_x_px * zoom_factor), int(wrist_y_px * zoom_factor))
                                gesture_text = "Detected Gesture: 3 Fingers -> Zoom Out"
                                last_action_time = now
                            elif pinky_up:
                                gesture_text = "Detected Gesture: Pinky Only -> Stop Presentation"
                                st.session_state.presenting = False
                                st.session_state.stop_flag = True
                                last_action_time = now

                        xs = [lm.x for lm in hand_landmarks.landmark]
                        ys = [lm.y for lm in hand_landmarks.landmark]
                        x_min = int(min(xs) * w) - 10
                        x_max = int(max(xs) * w) + 10
                        y_min = int(min(ys) * h) - 10
                        y_max = int(max(ys) * h) + 10
                        cv2.rectangle(frame, (max(0, x_min), max(0, y_min)), (min(w, x_max), min(h, y_max)), (0, 255, 0), 2)
                    else:
                        movement_history.clear()

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    webcam_placeholder.image(frame, channels="RGB")

                    try:
                        slide_img = render_pdf_page(doc, current_page, zoom_factor, crop_center=zoom_center)
                    except Exception:
                        slide_img = Image.new("RGB", (800, 600), color=(255, 255, 255))
                    slide_placeholder.image(slide_img, use_column_width=True)
                    status_placeholder.info(gesture_text)
                    time.sleep(0.03)
                    if st.session_state.stop_flag:
                        st.session_state.presenting = False
                        break
            finally:
                cap.release()
                status_placeholder.info("Presentation stopped.")
else:
    status_placeholder.info("Presentation stopped. Click 'Start Presentation' to begin.")
    try:
        preview_img = render_pdf_page(doc, current_page, 1.0)
        slide_placeholder.image(preview_img, use_column_width=True)
    except Exception:
        slide_placeholder.text("Upload a PDF and click Start Presentation to begin.")
    webcam_placeholder.image(np.zeros((480, 640, 3), dtype=np.uint8))
