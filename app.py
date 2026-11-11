"""
Drowsiness Detection — Streamlit App
Real-time driver monitoring: EAR + optional CNN (eye state & yawn).
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
from collections import deque
from datetime import datetime
from scipy.spatial import distance
from imutils import face_utils
import dlib

MODEL_PATH = "models/shape_predictor_68_face_landmarks.dat"

# ── Page config (must be first Streamlit call) ───────────
st.set_page_config(
    page_title="DrowsyGuard — Drowsiness Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Theme ────────────────────────────────────────────────
DARK = st.sidebar.toggle("Dark mode", value=True)

if DARK:
    C = {
        "bg_grad": "linear-gradient(160deg, #020617 0%, #0c2340 45%, #0a3d6e 100%)",
        "sidebar_grad": "linear-gradient(180deg, #071525 0%, #0c2d4f 100%)",
        "hero_grad": "linear-gradient(135deg, #0a2540 0%, #0d4a7a 50%, #0a3060 100%)",
        "card_grad": "linear-gradient(145deg, rgba(13, 58, 107, 0.9) 0%, rgba(8, 35, 68, 0.95) 100%)",
        "card": "rgba(10, 45, 82, 0.85)",
        "border": "rgba(56, 189, 248, 0.25)",
        "text": "#e0f2fe",
        "muted": "#7dd3fc",
        "accent": "#38bdf8",
        "accent2": "#0ea5e9",
        "green": "#34d399",
        "warn": "#7dd3fc",
        "red": "#f87171",
        "info": "#60a5fa",
        "shadow": "0 8px 32px rgba(14, 116, 178, 0.25)",
        "badge_bg": "rgba(56, 189, 248, 0.18)",
        "video_bg": (64, 34, 10),       # #0a2240 BGR
        "header_bar": (90, 45, 12),     # #0c2d5a BGR
        "landmark_eye": (248, 189, 56), # sky blue BGR
        "landmark_mouth": (250, 165, 96),
        "video_text": (248, 226, 224),
        "box_ok": (153, 217, 52),
        "box_alert": (113, 113, 248),
        "btn_secondary_bg": "rgba(14, 116, 178, 0.35)",
        "btn_secondary_border": "rgba(56, 189, 248, 0.55)",
        "btn_secondary_text": "#e0f2fe",
        "btn_secondary_hover": "rgba(14, 165, 233, 0.55)",
    }
else:
    C = {
        "bg_grad": "linear-gradient(160deg, #e0f4ff 0%, #bae6fd 40%, #dbeafe 100%)",
        "sidebar_grad": "linear-gradient(180deg, #f0f9ff 0%, #dbeafe 100%)",
        "hero_grad": "linear-gradient(135deg, #ffffff 0%, #e0f2fe 50%, #bae6fd 100%)",
        "card_grad": "linear-gradient(145deg, rgba(255,255,255,0.92) 0%, rgba(224,242,254,0.88) 100%)",
        "card": "rgba(255, 255, 255, 0.82)",
        "border": "rgba(14, 165, 233, 0.35)",
        "text": "#0c4a6e",
        "muted": "#0369a1",
        "accent": "#0284c7",
        "accent2": "#0ea5e9",
        "green": "#059669",
        "warn": "#0284c7",
        "red": "#dc2626",
        "info": "#2563eb",
        "shadow": "0 8px 32px rgba(14, 165, 233, 0.18)",
        "badge_bg": "rgba(14, 165, 233, 0.15)",
        "video_bg": (254, 234, 219),    # #dbeafe BGR
        "header_bar": (254, 219, 191),  # #bfdbfe BGR
        "landmark_eye": (199, 132, 2),  # #0284c7 BGR
        "landmark_mouth": (235, 105, 37),
        "video_text": (110, 74, 12),
        "box_ok": (105, 150, 5),
        "box_alert": (38, 38, 220),
        "btn_secondary_bg": "rgba(255, 255, 255, 0.75)",
        "btn_secondary_border": "rgba(14, 165, 233, 0.45)",
        "btn_secondary_text": "#0369a1",
        "btn_secondary_hover": "rgba(186, 230, 253, 0.95)",
    }

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

[data-testid="stAppViewContainer"] {{
    background: {C["bg_grad"]} !important;
}}

.stApp {{
    background: transparent !important;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

[data-testid="stSidebar"] {{
    background: {C["sidebar_grad"]} !important;
    border-right: 1px solid {C["border"]} !important;
    backdrop-filter: blur(16px);
}}

[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
    color: {C["text"]} !important;
}}

[data-testid="stSidebar"] .block-container {{
    padding-top: 1.5rem;
}}

.block-container {{
    padding-top: 1.5rem;
    max-width: 1400px;
}}

p, span, label, h1, h2, h3, h4, h5, h6 {{
    color: {C["text"]};
}}

/* Hide Streamlit footer */
footer {{ visibility: hidden; }}

/* Hero header */
.hero {{
    background: {C["hero_grad"]};
    border: 1px solid {C["border"]};
    border-radius: 16px;
    padding: 24px 28px;
    margin-bottom: 24px;
    box-shadow: {C["shadow"]};
}}
.hero h1 {{
    font-size: 1.75rem;
    font-weight: 700;
    color: {C["text"]};
    margin: 0 0 6px 0;
    letter-spacing: -0.02em;
}}
.hero p {{
    color: {C["muted"]};
    margin: 0;
    font-size: 0.95rem;
}}
.badge {{
    display: inline-block;
    background: {C["badge_bg"]};
    color: {C["accent"]};
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 999px;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 10px;
}}

/* Section labels */
.section-label {{
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: {C["muted"]};
    margin: 20px 0 10px 0;
}}

/* Metric cards */
.card {{
    background: {C["card_grad"]};
    border: 1px solid {C["border"]};
    border-radius: 12px;
    padding: 18px 16px;
    text-align: center;
    height: 100%;
    box-shadow: {C["shadow"]};
    backdrop-filter: blur(10px);
}}
.card-value {{
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
}}
.card-label {{
    font-size: 0.75rem;
    font-weight: 500;
    color: {C["muted"]};
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}
.card-sm .card-value {{ font-size: 1.4rem; }}

/* Gauge */
.gauge-track {{
    background: {C["border"]};
    border-radius: 999px;
    height: 8px;
    margin-top: 10px;
    overflow: hidden;
}}
.gauge-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width 0.3s ease;
}}

/* Alert banner */
.alert-banner {{
    background: linear-gradient(90deg, #dc2626, #ef4444);
    color: white;
    border-radius: 10px;
    padding: 14px 20px;
    font-weight: 600;
    text-align: center;
    margin: 12px 0;
    animation: pulse 1.2s ease-in-out infinite alternate;
}}
@keyframes pulse {{
    from {{ opacity: 0.85; }}
    to   {{ opacity: 1; }}
}}

/* Log table */
.log-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}}
.log-table th {{
    color: {C["muted"]};
    font-weight: 600;
    text-align: left;
    padding: 8px 6px;
    border-bottom: 1px solid {C["border"]};
}}
.log-table td {{
    padding: 7px 6px;
    color: {C["text"]};
    border-bottom: 1px solid {C["border"]};
}}

/* Buttons */
div.stButton > button[kind="primary"] {{
    background: linear-gradient(90deg, {C["accent2"]}, {C["accent"]}) !important;
    border: none !important;
    font-weight: 600 !important;
    color: white !important;
}}
div.stButton > button[kind="primary"]:hover {{
    filter: brightness(1.08);
    color: white !important;
}}
div.stButton > button:not([kind="primary"]) {{
    background: {C["btn_secondary_bg"]} !important;
    border: 1px solid {C["btn_secondary_border"]} !important;
    color: {C["btn_secondary_text"]} !important;
    font-weight: 600 !important;
}}
div.stButton > button:not([kind="primary"]):hover {{
    background: {C["btn_secondary_hover"]} !important;
    border-color: {C["accent"]} !important;
    color: {C["btn_secondary_text"]} !important;
}}
div.stButton > button {{
    border-radius: 8px !important;
}}

/* Video container */
.video-wrap {{
    border: 1px solid {C["border"]};
    border-radius: 12px;
    overflow: hidden;
    background: {C["card"]};
}}
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────

def eye_aspect_ratio(eye):
    a = distance.euclidean(eye[1], eye[5])
    b = distance.euclidean(eye[2], eye[4])
    c = distance.euclidean(eye[0], eye[3])
    return (a + b) / (2.0 * c)


def crop_region(gray, points, size=64, pad=8):
    pts = np.array(points, dtype=np.int32)
    x1 = max(pts[:, 0].min() - pad, 0)
    y1 = max(pts[:, 1].min() - pad, 0)
    x2 = min(pts[:, 0].max() + pad, gray.shape[1])
    y2 = min(pts[:, 1].max() + pad, gray.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    roi = gray[y1:y2, x1:x2]
    return cv2.resize(roi, (size, size))


def draw_eye_contours(frame, left_eye, right_eye, color=None):
    color = color or C["landmark_eye"]
    for eye in (left_eye, right_eye):
        hull = cv2.convexHull(eye)
        cv2.drawContours(frame, [hull], -1, color, 1)
        for (x, y) in eye:
            cv2.circle(frame, (x, y), 2, color, -1)


def draw_mouth_contour(frame, mouth, color=None):
    color = color or C["landmark_mouth"]
    hull = cv2.convexHull(mouth)
    cv2.drawContours(frame, [hull], -1, color, 1)


def metric_card(value, label, color=None, gauge_pct=None, gauge_color=None, small=False):
    cls = "card card-sm" if small else "card"
    col = color or C["accent"]
    gauge = ""
    if gauge_pct is not None:
        gauge = (
            f'<div class="gauge-track">'
            f'<div class="gauge-fill" style="width:{gauge_pct}%;background:{gauge_color or col}"></div>'
            f'</div>'
        )
    return (
        f'<div class="{cls}">'
        f'<div class="card-value" style="color:{col}">{value}</div>'
        f'<div class="card-label">{label}</div>'
        f'{gauge}'
        f'</div>'
    )


def render_idle_frame(width):
    h = 360
    img = np.full((h, width, 3), C["video_bg"], dtype=np.uint8)
    cv2.rectangle(img, (0, 0), (width, 40), C["header_bar"], -1)
    cv2.putText(img, "DrowsyGuard", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, C["video_text"], 1)

    cx, cy = width // 2, h // 2 + 20
    ring = C["landmark_eye"]
    cv2.circle(img, (cx, cy), 48, ring, 2)
    cv2.ellipse(img, (cx, cy), (36, 14), 0, 0, 360, ring, 2)
    cv2.circle(img, (cx, cy), 10, ring, -1)

    txt = "Press Start to begin"
    tw = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)[0][0]
    cv2.putText(img, txt, ((width - tw) // 2, cy + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, C["video_text"], 1)
    return img


# ── Model loaders ────────────────────────────────────────

@st.cache_resource
def load_dlib(model_path):
    if not os.path.exists(model_path):
        return None, None
    return dlib.get_frontal_face_detector(), dlib.shape_predictor(model_path)


@st.cache_resource
def load_cnn_models():
    eye_path = "models/eye_state_model.h5"
    yawn_path = "models/yawn_model.h5"
    if not os.path.exists(eye_path) or not os.path.exists(yawn_path):
        return None, None
    try:
        from tensorflow.keras.models import load_model
        return load_model(eye_path), load_model(yawn_path)
    except Exception:
        return None, None


def predict_cnn(model, roi, threshold=0.75):
    if model is None or roi is None:
        return None, False
    x = roi.astype("float32") / 255.0
    x = x.reshape(1, 64, 64, 1)
    prob = float(model.predict(x, verbose=0)[0][0])
    return prob, prob >= threshold


def smooth_ear(raw_ear, history):
    """Average recent EAR values to ignore quick blinks."""
    if raw_ear is None:
        return None
    history.append(raw_ear)
    return float(np.mean(history))


def eyes_closed(ear, ear_thresh, eye_prob=None, use_cnn=False, cnn_thresh=0.75):
    """True only when EAR is low; CNN must confirm if enabled."""
    if ear is None or ear >= ear_thresh:
        return False
    if use_cnn and eye_prob is not None:
        return eye_prob >= cnn_thresh
    return True


# ── Session state ────────────────────────────────────────

for key, val in {
    "running": False,
    "eyes_closed_since": None,
    "yawn_since": None,
    "ear_smooth_buf": deque(maxlen=5),
    "ear_history": [],
    "alerts": [],
    "total_frames": 0,
    "alert_count": 0,
    "session_start": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Sidebar ──────────────────────────────────────────────

with st.sidebar:
    st.markdown("### DrowsyGuard")
    st.caption("Real-time drowsiness monitoring")

    st.markdown('<div class="section-label">Detection</div>', unsafe_allow_html=True)
    EAR_THRESH = st.slider("Eye sensitivity", 0.15, 0.35, 0.22, 0.01,
                           help="Lower = more sensitive.")
    CLOSED_EYE_SECS = st.slider("Alert after closed eyes (sec)", 1.0, 8.0, 3.0, 0.5,
                                help="Alert only if eyes stay closed this long (blinks are ~0.2s)")
    YAWN_SECS = st.slider("Yawn alert after (sec)", 0.5, 5.0, 2.0, 0.5,
                          help="How long a yawn must last before alert")

    eye_model, yawn_model = load_cnn_models()
    cnn_ok = eye_model is not None and yawn_model is not None
    USE_CNN = st.toggle("Use CNN models", value=False, disabled=not cnn_ok,
                        help="Optional. Uses EAR by default — more reliable for blinks.")
    if not cnn_ok:
        st.caption("CNN models not found in `models/`")

    st.markdown('<div class="section-label">Display</div>', unsafe_allow_html=True)
    SHOW_LANDMARKS = st.toggle("Eye landmarks", value=True)
    SHOW_MOUTH = st.toggle("Mouth contour", value=True)
    SHOW_EAR_TEXT = st.toggle("Show values on video", value=False)
    FRAME_WIDTH = st.select_slider("Camera width", [320, 450, 640], value=450)

    st.divider()
    st.markdown(f"""
<div style="font-size:0.78rem;color:{C['muted']};line-height:1.6">
Alert after <b>{CLOSED_EYE_SECS:.1f}s</b> with eyes closed.<br>
Normal blinks are ignored.
</div>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────

cnn_badge = "EAR + CNN" if (USE_CNN and cnn_ok) else "EAR only"
st.markdown(f"""
<div class="hero">
    <div class="badge">{cnn_badge} · Live monitoring</div>
    <h1>DrowsyGuard</h1>
    <p>Real-time driver drowsiness detection using facial landmarks and deep learning</p>
</div>
""", unsafe_allow_html=True)

col_video, col_stats = st.columns([3, 2], gap="large")

with col_video:
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        start_btn = st.button("Start", type="primary", use_container_width=True)
    with bc2:
        stop_btn = st.button("Stop", use_container_width=True)
    with bc3:
        reset_btn = st.button("Reset", use_container_width=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.session_start = datetime.now()
        st.session_state.eyes_closed_since = None
        st.session_state.yawn_since = None
        st.session_state.ear_smooth_buf.clear()
    if stop_btn:
        st.session_state.running = False
    if reset_btn:
        for k in ("eyes_closed_since", "yawn_since"):
            st.session_state[k] = None
        st.session_state.ear_smooth_buf = deque(maxlen=5)
        st.session_state.ear_history = []
        st.session_state.alerts = []
        st.session_state.total_frames = 0
        st.session_state.alert_count = 0
        st.session_state.session_start = None

    st.markdown('<div class="section-label">Live feed</div>', unsafe_allow_html=True)
    video_ph = st.empty()
    alert_ph = st.empty()

with col_stats:
    st.markdown('<div class="section-label">Status</div>', unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    ear_ph = m1.empty()
    status_ph = m2.empty()

    st.markdown('<div class="section-label">Activity</div>', unsafe_allow_html=True)
    chart_ph = st.empty()

    if USE_CNN and cnn_ok:
        st.markdown('<div class="section-label">AI analysis</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        eye_cnn_ph = c1.empty()
        yawn_cnn_ph = c2.empty()

    st.markdown('<div class="section-label">Summary</div>', unsafe_allow_html=True)
    sum_c1, sum_c2 = st.columns(2)
    alert_sum_ph = sum_c1.empty()
    time_sum_ph = sum_c2.empty()

    st.markdown('<div class="section-label">Alert log</div>', unsafe_allow_html=True)
    log_ph = st.empty()


def update_log():
    if st.session_state.alerts:
        rows = "".join(
            f"<tr><td>{a['time'].strftime('%H:%M:%S')}</td>"
            f"<td style='color:{C['red']}'>"
            f"{'Eyes closed' if a['type'] == 'Drowsy' else 'Yawn detected'}</td></tr>"
            for a in reversed(st.session_state.alerts[-10:])
        )
        log_ph.markdown(f"""
<table class="log-table">
    <thead><tr><th>Time</th><th>Alert</th></tr></thead>
    <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)
    else:
        log_ph.markdown(f"<p style='color:{C['muted']};font-size:0.85rem'>No alerts yet.</p>",
                        unsafe_allow_html=True)


def update_stats(elapsed):
    alert_sum_ph.markdown(
        metric_card(st.session_state.alert_count, "Alerts", C["red"], small=True),
        unsafe_allow_html=True,
    )
    time_sum_ph.markdown(
        metric_card(elapsed or "00:00", "Time", C["accent2"], small=True),
        unsafe_allow_html=True,
    )


def render_paused():
    video_ph.image(render_idle_frame(FRAME_WIDTH), channels="BGR", use_container_width=True)
    if st.session_state.total_frames == 0:
        ear_ph.markdown(metric_card("—", "Eye level", C["muted"]), unsafe_allow_html=True)
        status_ph.markdown(metric_card("Idle", "Status", C["muted"]), unsafe_allow_html=True)
        return

    last_ear = st.session_state.ear_history[-1] if st.session_state.ear_history else None
    if last_ear is not None:
        pct = min(int(last_ear / 0.45 * 100), 100)
        col = C["red"] if last_ear < EAR_THRESH else C["green"]
        ear_ph.markdown(metric_card(f"{last_ear:.3f}", "Eye level", col, pct, col),
                        unsafe_allow_html=True)
    else:
        ear_ph.markdown(metric_card("—", "No face", C["warn"]), unsafe_allow_html=True)

    status_ph.markdown(metric_card("Paused", "Status", C["warn"]), unsafe_allow_html=True)

    if st.session_state.ear_history:
        df = pd.DataFrame({"EAR": st.session_state.ear_history})
        chart_ph.line_chart(df, height=140, use_container_width=True)

    elapsed = ""
    if st.session_state.session_start:
        s = int((datetime.now() - st.session_state.session_start).total_seconds())
        elapsed = f"{s // 60:02d}:{s % 60:02d}"
    update_stats(elapsed)
    update_log()


# ── Main loop ────────────────────────────────────────────

if not st.session_state.running:
    render_paused()
else:
    detector, predictor = load_dlib(MODEL_PATH)
    if detector is None:
        st.error(f"Landmark model not found: `{MODEL_PATH}`")
        st.session_state.running = False
        st.stop()

    (l_start, l_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (r_start, r_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
    (m_start, m_end) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        st.error("Cannot open webcam. Check permissions and close other apps using the camera.")
        st.session_state.running = False
        st.stop()

    fps_t = time.time()
    eye_prob = yawn_prob = None
    closed_duration = 0.0

    try:
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Webcam read failed.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, 360))
            frame = cv2.flip(frame, 1)  # mirror view like a selfie camera
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray, 0)

            current_ear = None
            smooth_ear_val = None
            is_drowsy = False
            is_yawning = False
            eye_prob = yawn_prob = None
            closed_duration = 0.0
            now = time.time()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 40), C["header_bar"], -1)
            frame = cv2.addWeighted(overlay, 0.85, frame, 0.15, 0)

            for face in faces:
                shape = face_utils.shape_to_np(predictor(gray, face))
                left_eye = shape[l_start:l_end]
                right_eye = shape[r_start:r_end]
                mouth = shape[m_start:m_end]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                current_ear = (left_ear + right_ear) / 2.0
                smooth_ear_val = smooth_ear(current_ear, st.session_state.ear_smooth_buf)

                if SHOW_LANDMARKS:
                    draw_eye_contours(frame, left_eye, right_eye)
                if SHOW_MOUTH:
                    draw_mouth_contour(frame, mouth)

                eye_prob = yawn_prob = None
                if USE_CNN and cnn_ok:
                    eye_roi = crop_region(gray, np.vstack([left_eye, right_eye]))
                    mouth_roi = crop_region(gray, mouth)
                    eye_prob, _ = predict_cnn(eye_model, eye_roi)
                    yawn_prob, yawning_now = predict_cnn(yawn_model, mouth_roi)

                    if yawning_now:
                        if st.session_state.yawn_since is None:
                            st.session_state.yawn_since = now
                        elif now - st.session_state.yawn_since >= YAWN_SECS:
                            is_yawning = True
                    else:
                        st.session_state.yawn_since = None

                closed = eyes_closed(
                    smooth_ear_val, EAR_THRESH,
                    eye_prob=eye_prob, use_cnn=USE_CNN and cnn_ok,
                )
                if closed:
                    if st.session_state.eyes_closed_since is None:
                        st.session_state.eyes_closed_since = now
                    closed_duration = now - st.session_state.eyes_closed_since
                    if closed_duration >= CLOSED_EYE_SECS:
                        is_drowsy = True
                else:
                    st.session_state.eyes_closed_since = None
                    closed_duration = 0.0

                x1, y1 = face.left(), face.top()
                x2, y2 = face.right(), face.bottom()
                alert_active = is_drowsy or is_yawning
                box_col = C["box_alert"] if alert_active else C["box_ok"]
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, 2)

                ear_display = smooth_ear_val if smooth_ear_val is not None else current_ear
                if SHOW_EAR_TEXT and ear_display is not None:
                    cv2.putText(frame, f"EAR {ear_display:.3f}", (x1, max(y1 - 8, 14)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_col, 1)

            if not faces:
                st.session_state.eyes_closed_since = None
                st.session_state.yawn_since = None
                st.session_state.ear_smooth_buf.clear()

            if is_drowsy or is_yawning:
                cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 360), C["box_alert"], 3)
                msg = "DROWSINESS ALERT" if is_drowsy else "YAWN DETECTED"
                tw = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)[0][0]
                cv2.putText(frame, msg, ((FRAME_WIDTH - tw) // 2, 350),
                            cv2.FONT_HERSHEY_DUPLEX, 0.75, C["box_alert"], 2)

            fps = 1.0 / max(time.time() - fps_t, 1e-9)
            fps_t = time.time()
            if is_drowsy:
                status_txt = "DROWSY"
            elif is_yawning:
                status_txt = "YAWNING"
            elif closed_duration > 0:
                status_txt = f"Closed {closed_duration:.1f}s"
            elif faces:
                status_txt = "Monitoring"
            else:
                status_txt = "No face"
            cv2.putText(frame, f"DrowsyGuard  ·  {status_txt}  ·  {fps:.0f} fps",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C["video_text"], 1)

            st.session_state.total_frames += 1
            chart_ear = smooth_ear_val if smooth_ear_val is not None else current_ear
            if chart_ear is not None:
                st.session_state.ear_history.append(chart_ear)
                if len(st.session_state.ear_history) > 150:
                    st.session_state.ear_history.pop(0)

            if (is_drowsy or is_yawning) and (
                not st.session_state.alerts
                or (datetime.now() - st.session_state.alerts[-1]["time"]).seconds > 3
            ):
                st.session_state.alert_count += 1
                alert_type = "Drowsy" if is_drowsy else "Yawn"
                detail = f"EAR {current_ear:.3f}" if current_ear else "—"
                if is_yawning and yawn_prob is not None:
                    detail = f"Yawn {yawn_prob:.0%}"
                st.session_state.alerts.append({
                    "time": datetime.now(),
                    "type": alert_type,
                    "detail": detail,
                })
                if len(st.session_state.alerts) > 20:
                    st.session_state.alerts.pop(0)
                if os.name == "posix":
                    os.system("afplay /System/Library/Sounds/Glass.aiff &")

            video_ph.image(frame, channels="BGR", use_container_width=True)

            if is_drowsy:
                alert_ph.markdown('<div class="alert-banner">Drowsiness detected — take a break!</div>',
                                  unsafe_allow_html=True)
            elif is_yawning:
                alert_ph.markdown('<div class="alert-banner">Yawn detected — you may be tired!</div>',
                                  unsafe_allow_html=True)
            else:
                alert_ph.empty()

            if current_ear is not None:
                display_ear = smooth_ear_val if smooth_ear_val is not None else current_ear
                pct = min(int(display_ear / 0.45 * 100), 100)
                col = C["red"] if display_ear < EAR_THRESH else C["green"]
                label = (f"Closed {closed_duration:.1f}s / {CLOSED_EYE_SECS:.1f}s"
                         if closed_duration > 0 else "Eye level")
                ear_ph.markdown(
                    metric_card(f"{display_ear:.3f}", label, col, pct, col),
                    unsafe_allow_html=True,
                )
            else:
                ear_ph.markdown(metric_card("—", "No face", C["warn"]), unsafe_allow_html=True)

            if is_drowsy or is_yawning:
                sc, sl = C["red"], "Alert"
            elif closed_duration > 0:
                sc, sl = C["warn"], f"{closed_duration:.1f}s"
            elif faces:
                sc, sl = C["green"], "Awake"
            else:
                sc, sl = C["warn"], "Searching"
            status_ph.markdown(metric_card(sl, "Status", sc), unsafe_allow_html=True)

            if st.session_state.ear_history:
                chart_ph.line_chart(pd.DataFrame({"EAR": st.session_state.ear_history}),
                                    height=140, use_container_width=True)

            if USE_CNN and cnn_ok:
                if eye_prob is not None:
                    eye_lbl = "Closed" if eye_prob >= 0.75 else "Open"
                    eye_cnn_ph.markdown(
                        metric_card(f"{eye_prob:.0%}", f"Eye: {eye_lbl}",
                                    C["red"] if eye_prob >= 0.75 else C["green"], small=True),
                        unsafe_allow_html=True)
                else:
                    eye_cnn_ph.markdown(metric_card("—", "Eye CNN", C["muted"], small=True),
                                        unsafe_allow_html=True)

                if yawn_prob is not None:
                    yawn_lbl = "Yawn" if yawn_prob >= 0.75 else "Normal"
                    yawn_cnn_ph.markdown(
                        metric_card(f"{yawn_prob:.0%}", f"Mouth: {yawn_lbl}",
                                    C["red"] if yawn_prob >= 0.75 else C["green"], small=True),
                        unsafe_allow_html=True)
                else:
                    yawn_cnn_ph.markdown(metric_card("—", "Yawn CNN", C["muted"], small=True),
                                         unsafe_allow_html=True)

            elapsed = ""
            if st.session_state.session_start:
                s = int((datetime.now() - st.session_state.session_start).total_seconds())
                elapsed = f"{s // 60:02d}:{s % 60:02d}"
            update_stats(elapsed)
            update_log()

    finally:
        cap.release()
