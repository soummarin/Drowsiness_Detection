"""
=======================================================
  Drowsiness Detection — Streamlit App  (app.py)
  
  Features:
    • Live webcam feed with face landmark overlay
    • Real-time EAR (Eye Aspect Ratio) gauge
    • Yawn detection (CNN model if available)
    • Configurable alert thresholds
    • Session log
    • Dark, modern UI
=======================================================
"""

import streamlit as st
import cv2
import numpy as np
import time
from datetime import datetime
from scipy.spatial import distance
from imutils import face_utils
import dlib

# Optional CNN models (loaded only when available)
CNN_AVAILABLE = False


# ─────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DrowsyGuard — Real-time Drowsiness Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
#  CUSTOM CSS  (dark glassmorphism theme)
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Base ─────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1535 50%, #0a0a1a 100%);
        color: #e2e8f0;
    }

    /* ── Sidebar ──────────────────────────────── */
    [data-testid="stSidebar"] {
        background: rgba(13, 21, 53, 0.95) !important;
        border-right: 1px solid rgba(99, 179, 237, 0.15);
    }
    [data-testid="stSidebar"] .block-container { padding-top: 1rem; }

    /* ── Cards / metric boxes ─────────────────── */
    .metric-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(99,179,237,0.2);
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: transform 0.2s, border-color 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99,179,237,0.5);
    }
    .metric-value { font-size: 2.4rem; font-weight: 700; line-height: 1; margin-bottom: 4px; }
    .metric-label { font-size: 0.78rem; color: #94a3b8; letter-spacing: 0.08em; text-transform: uppercase; }

    /* Status badge */
    .status-awake  { color: #48bb78; }
    .status-drowsy { color: #fc8181; }
    .status-warn   { color: #f6ad55; }

    /* Alert box */
    .alert-box {
        background: rgba(252,129,129,0.12);
        border: 1px solid rgba(252,129,129,0.5);
        border-radius: 12px;
        padding: 16px 20px;
        margin: 12px 0;
        font-weight: 600;
        font-size: 1.05rem;
        color: #fc8181;
        animation: pulse 1s ease-in-out infinite alternate;
    }
    @keyframes pulse { from { opacity: 0.8; } to { opacity: 1; } }

    /* Log table */
    .log-table { font-size: 0.82rem; color: #94a3b8; }

    /* Section header */
    .section-header {
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #63b3ed;
        margin-bottom: 8px;
        margin-top: 20px;
    }

    /* Gauge bar */
    .gauge-outer {
        background: rgba(255,255,255,0.07);
        border-radius: 999px;
        height: 12px;
        overflow: hidden;
        margin: 6px 0;
    }
    .gauge-inner {
        height: 100%;
        border-radius: 999px;
        transition: width 0.3s ease;
    }

    /* Hide Streamlit default elements */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
    div[data-testid="stToolbar"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────

def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_ear_color(ear, thresh):
    ratio = ear / 0.45
    if ear < thresh:
        return "#fc8181"  # red
    elif ear < thresh + 0.05:
        return "#f6ad55"  # orange
    else:
        return "#68d391"  # green


def draw_eye_contours(frame, left_eye, right_eye):
    lh = cv2.convexHull(left_eye)
    rh = cv2.convexHull(right_eye)
    cv2.drawContours(frame, [lh], -1, (0, 220, 100), 1)
    cv2.drawContours(frame, [rh], -1, (0, 220, 100), 1)
    for (x, y) in np.concatenate((left_eye, right_eye), axis=0):
        cv2.circle(frame, (x, y), 2, (0, 200, 255), -1)


# ─────────────────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────────────────
defaults = {
    "running":      False,
    "flag":         0,
    "ear_history":  [],
    "alerts":       [],
    "total_frames": 0,
    "alert_count":  0,
    "session_start": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────
#  SIDEBAR — SETTINGS
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 👁️ DrowsyGuard")
    st.markdown("<p style='color:#94a3b8;font-size:0.85rem;margin-top:-8px'>Real-time Drowsiness Detection</p>",
                unsafe_allow_html=True)
    st.divider()

    st.markdown("<div class='section-header'>Detection Settings</div>", unsafe_allow_html=True)
    EAR_THRESH   = st.slider("EAR Threshold",   0.15, 0.35, 0.25, 0.01,
                              help="Eye Aspect Ratio below this triggers alert")
    FRAME_CHECK  = st.slider("Consecutive Frames", 5, 40, 20,
                              help="How many frames of closed eyes before alert fires")

    st.markdown("<div class='section-header'>Display</div>", unsafe_allow_html=True)
    SHOW_LANDMARKS = st.toggle("Show Eye Landmarks", value=True)
    SHOW_EAR_TEXT  = st.toggle("Show EAR on Frame",  value=True)
    FRAME_WIDTH    = st.select_slider("Camera Width", [320, 450, 640], value=450)

    st.divider()
    st.markdown("<div class='section-header'>Landmark Model</div>", unsafe_allow_html=True)
    MODEL_PATH = st.text_input("dlib .dat path", value="models/shape_predictor_68_face_landmarks.dat")

    st.divider()
    st.markdown("""
<div style='color:#64748b;font-size:0.75rem;line-height:1.6'>
Algorithm: <b>Eye Aspect Ratio (EAR)</b><br>
EAR = (‖p2-p6‖ + ‖p3-p5‖) / (2·‖p1-p4‖)<br><br>
If EAR &lt; threshold for N consecutive frames → <b>DROWSY ALERT</b>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  MAIN LAYOUT
# ─────────────────────────────────────────────────────────
st.markdown("# 👁️&nbsp;&nbsp;DrowsyGuard — Drowsiness Detection")
st.markdown("<p style='color:#94a3b8;margin-top:-12px'>Real-time driver drowsiness monitoring using facial landmark analysis</p>",
            unsafe_allow_html=True)

col_video, col_stats = st.columns([3, 2], gap="large")

with col_video:
    # Controls row
    c1, c2, c3 = st.columns(3)
    with c1:
        start_btn = st.button("▶ Start Detection", type="primary", use_container_width=True)
    with c2:
        stop_btn  = st.button("⏹ Stop",            use_container_width=True)
    with c3:
        reset_btn = st.button("🔄 Reset Stats",     use_container_width=True)

    if start_btn:
        st.session_state.running = True
        st.session_state.session_start = datetime.now()
    if stop_btn:
        st.session_state.running = False
    if reset_btn:
        for k in ["flag", "ear_history", "alerts", "total_frames", "alert_count"]:
            st.session_state[k] = [] if k in ("ear_history", "alerts") else 0
        st.session_state.session_start = None

    st.markdown("<div class='section-header'>Live Camera Feed</div>", unsafe_allow_html=True)
    video_placeholder = st.empty()
    alert_placeholder = st.empty()

with col_stats:
    st.markdown("<div class='section-header'>Real-time Metrics</div>", unsafe_allow_html=True)
    ear_col, status_col = st.columns(2)
    ear_metric_ph   = ear_col.empty()
    status_ph       = status_col.empty()

    st.markdown("<div class='section-header'>EAR Over Time</div>", unsafe_allow_html=True)
    chart_ph = st.empty()

    st.markdown("<div class='section-header'>Session Statistics</div>", unsafe_allow_html=True)
    stats_ph = st.empty()

    st.markdown("<div class='section-header'>Alert Log</div>", unsafe_allow_html=True)
    log_ph = st.empty()


# ─────────────────────────────────────────────────────────
#  LOAD DLIB MODELS (cached)
# ─────────────────────────────────────────────────────────
@st.cache_resource
def load_dlib_models(model_path):
    if not __import__('os').path.exists(model_path):
        return None, None
    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model_path)
    return detector, predictor


def render_idle_frame():
    img = np.zeros((360, FRAME_WIDTH, 3), dtype=np.uint8)
    img[:] = (15, 20, 40)
    txt  = "Press 'Start Detection'"
    txt2 = "to begin monitoring"
    fw = FRAME_WIDTH
    x1 = (fw - cv2.getTextSize(txt,  cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0][0]) // 2
    x2 = (fw - cv2.getTextSize(txt2, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0]) // 2
    cv2.putText(img, txt,  (x1, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7,  (80, 120, 200), 1)
    cv2.putText(img, txt2, (x2, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (60, 90, 160), 1)
    # Draw eye icon
    cv2.ellipse(img, (fw//2, 280), (50, 20), 0, 0, 360, (0, 180, 120), 2)
    cv2.circle(img, (fw//2, 280), 8, (0, 200, 150), -1)
    return img


# ─────────────────────────────────────────────────────────
#  DETECTION LOOP
# ─────────────────────────────────────────────────────────
if not st.session_state.running:
    idle = render_idle_frame()
    video_placeholder.image(idle, channels="BGR", use_container_width=True)
    # Render last known state if available
    if st.session_state.total_frames > 0:
        last_ear = st.session_state.ear_history[-1] if st.session_state.ear_history else None
        if last_ear is not None:
            ear_pct = min(int(last_ear / 0.45 * 100), 100)
            ec = "#fc8181" if last_ear < EAR_THRESH else "#68d391"
            ear_metric_ph.markdown(f"""
<div class='metric-card'>
    <div class='metric-value' style='color:{ec}'>{last_ear:.3f}</div>
    <div class='metric-label'>Eye Aspect Ratio</div>
    <div class='gauge-outer' style='margin-top:8px'>
        <div class='gauge-inner' style='width:{ear_pct}%;background:{ec}'></div>
    </div>
</div>""", unsafe_allow_html=True)
        else:
            ear_metric_ph.markdown("""
<div class='metric-card'>
    <div class='metric-value' style='color:#718096'>—</div>
    <div class='metric-label'>No Face Detected</div>
</div>""", unsafe_allow_html=True)

        status_ph.markdown("""
<div class='metric-card'>
    <div class='metric-value' style='color:#f6ad55'>⏸️</div>
    <div class='metric-label' style='color:#f6ad55;font-weight:600'>PAUSED</div>
</div>""", unsafe_allow_html=True)

        if st.session_state.ear_history:
            import pandas as pd
            chart_ph.line_chart(pd.DataFrame({"EAR": st.session_state.ear_history}), height=130, use_container_width=True)
            
        stats_ph.markdown(f"""
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.6rem;color:#63b3ed'>{st.session_state.total_frames}</div>
        <div class='metric-label'>Frames</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.6rem;color:#fc8181'>{st.session_state.alert_count}</div>
        <div class='metric-label'>Alerts</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.6rem;color:#68d391'>—</div>
        <div class='metric-label'>Faces</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.3rem;color:#a78bfa'>PAUSED</div>
        <div class='metric-label'>Session</div>
    </div>
</div>""", unsafe_allow_html=True)

        if st.session_state.alerts:
            rows = "".join([f"<tr><td>{a['time'].strftime('%H:%M:%S')}</td><td style='color:#fc8181'>{a['ear']}</td></tr>" for a in reversed(st.session_state.alerts[-10:])])
            log_ph.markdown(f"<table class='log-table' style='width:100%;border-collapse:collapse'><thead><tr style='color:#63b3ed;font-weight:600'><th align='left'>Time</th><th align='left'>EAR</th></tr></thead><tbody>{rows}</tbody></table>", unsafe_allow_html=True)
        else:
            log_ph.markdown("<p style='color:#4a5568;font-size:0.85rem'>No alerts yet.</p>", unsafe_allow_html=True)

    else:
        # Static default metrics
        ear_metric_ph.markdown("""
<div class='metric-card'>
    <div class='metric-value' style='color:#63b3ed'>—</div>
    <div class='metric-label'>Eye Aspect Ratio</div>
</div>""", unsafe_allow_html=True)
        status_ph.markdown("""
<div class='metric-card'>
    <div class='metric-value'>💤</div>
    <div class='metric-label'>Status</div>
</div>""", unsafe_allow_html=True)

else:
    detector, predictor = load_dlib_models(MODEL_PATH)

    if detector is None:
        st.error(f"❌ Model file not found: `{MODEL_PATH}`\n\nPlease check the path in the sidebar.")
        st.session_state.running = False
        st.stop()

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        st.error("❌ Cannot open webcam. Make sure it's connected and not in use by another app.")
        st.session_state.running = False
        st.stop()

    try:
        fps_time = time.time()
        frame_count_local = 0

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Cannot read from webcam.")
                break

            frame = cv2.resize(frame, (FRAME_WIDTH, 360))
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            subjects = detector(gray, 0)

            current_ear = None
            is_drowsy   = False

            # ── Draw overlay header bar ────────────────
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (FRAME_WIDTH, 36), (10, 15, 40), -1)
            frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)

            # ── Process each detected face ─────────────
            for subject in subjects:
                shape = predictor(gray, subject)
                shape = face_utils.shape_to_np(shape)

                left_eye  = shape[lStart:lEnd]
                right_eye = shape[rStart:rEnd]
                leftEAR   = eye_aspect_ratio(left_eye)
                rightEAR  = eye_aspect_ratio(right_eye)
                current_ear = (leftEAR + rightEAR) / 2.0

                # Landmarks
                if SHOW_LANDMARKS:
                    draw_eye_contours(frame, left_eye, right_eye)

                if current_ear < EAR_THRESH:
                    st.session_state.flag += 1
                    if st.session_state.flag >= FRAME_CHECK:
                        is_drowsy = True
                else:
                    st.session_state.flag = 0

                # Face bounding box
                x1, y1 = subject.left(), subject.top()
                x2, y2 = subject.right(), subject.bottom()
                box_color = (80, 80, 220) if not is_drowsy else (60, 60, 220)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

                # EAR text on frame
                if SHOW_EAR_TEXT and current_ear is not None:
                    ear_txt = f"EAR: {current_ear:.3f}"
                    cv2.putText(frame, ear_txt, (x1, y1 - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                (0, 220, 120) if not is_drowsy else (60, 60, 220), 1)

            # ── Alert overlay ──────────────────────────
            if is_drowsy:
                cv2.rectangle(frame, (0, 0), (FRAME_WIDTH, 360), (0, 0, 200), 4)
                txt = "⚠ DROWSINESS ALERT"
                tw  = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 0.85, 2)[0][0]
                cv2.putText(frame, txt, ((FRAME_WIDTH - tw) // 2, 335),
                            cv2.FONT_HERSHEY_DUPLEX, 0.85, (60, 60, 255), 2)

            # Header info
            fps_now = 1.0 / max(time.time() - fps_time, 1e-9)
            fps_time = time.time()
            status_txt  = "ALERT" if is_drowsy else ("Monitoring" if subjects else "No Face")
            status_color = (60, 60, 220) if is_drowsy else (0, 200, 100) if subjects else (180, 140, 0)
            cv2.putText(frame, f"DrowsyGuard  |  {status_txt}  |  FPS:{fps_now:.0f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

            # ── Session history ────────────────────────
            frame_count_local += 1
            st.session_state.total_frames += 1
            if current_ear is not None:
                st.session_state.ear_history.append(current_ear)
                if len(st.session_state.ear_history) > 150:
                    st.session_state.ear_history.pop(0)
            if is_drowsy and (not st.session_state.alerts or
               (datetime.now() - st.session_state.alerts[-1]["time"]).seconds > 3):
                st.session_state.alert_count += 1
                # Play alert sound on macOS in background
                import os
                os.system("afplay /System/Library/Sounds/Glass.aiff &")
                st.session_state.alerts.append({
                    "time": datetime.now(),
                    "ear":  round(current_ear, 3) if current_ear else 0,
                })
                if len(st.session_state.alerts) > 20:
                    st.session_state.alerts.pop(0)

            # ── Update UI ──────────────────────────────
            video_placeholder.image(frame, channels="BGR", use_container_width=True)

            # Alert banner
            if is_drowsy:
                alert_placeholder.markdown(
                    "<div class='alert-box'>⚠️ DROWSINESS DETECTED — Please take a break!</div>",
                    unsafe_allow_html=True)
            else:
                alert_placeholder.empty()

            # EAR metric
            if current_ear is not None:
                ear_pct = min(int(current_ear / 0.45 * 100), 100)
                ec = "#fc8181" if current_ear < EAR_THRESH else "#68d391"
                ear_metric_ph.markdown(f"""
<div class='metric-card'>
    <div class='metric-value' style='color:{ec}'>{current_ear:.3f}</div>
    <div class='metric-label'>Eye Aspect Ratio</div>
    <div class='gauge-outer' style='margin-top:8px'>
        <div class='gauge-inner' style='width:{ear_pct}%;background:{ec}'></div>
    </div>
</div>""", unsafe_allow_html=True)
            else:
                ear_metric_ph.markdown("""
<div class='metric-card'>
    <div class='metric-value' style='color:#718096'>—</div>
    <div class='metric-label'>No Face Detected</div>
</div>""", unsafe_allow_html=True)

            # Status
            if is_drowsy:
                sc, sl, si = "#fc8181", "DROWSY", "😴"
            elif subjects:
                sc, sl, si = "#68d391", "AWAKE",  "👀"
            else:
                sc, sl, si = "#f6ad55", "NO FACE","🔍"
            status_ph.markdown(f"""
<div class='metric-card'>
    <div class='metric-value' style='color:{sc}'>{si}</div>
    <div class='metric-label' style='color:{sc};font-weight:600'>{sl}</div>
</div>""", unsafe_allow_html=True)

            # EAR chart
            if st.session_state.ear_history:
                import pandas as pd
                df = pd.DataFrame({"EAR": st.session_state.ear_history})
                chart_ph.line_chart(df, height=130, use_container_width=True)

            # Stats
            elapsed = ""
            if st.session_state.session_start:
                secs = int((datetime.now() - st.session_state.session_start).total_seconds())
                elapsed = f"{secs//60:02d}:{secs%60:02d}"
            stats_ph.markdown(f"""
<div style='display:grid;grid-template-columns:1fr 1fr;gap:10px'>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.6rem;color:#63b3ed'>{st.session_state.total_frames}</div>
        <div class='metric-label'>Frames</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.6rem;color:#fc8181'>{st.session_state.alert_count}</div>
        <div class='metric-label'>Alerts</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.6rem;color:#68d391'>{len(subjects)}</div>
        <div class='metric-label'>Faces</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value' style='font-size:1.3rem;color:#a78bfa'>{elapsed or '—'}</div>
        <div class='metric-label'>Session</div>
    </div>
</div>""", unsafe_allow_html=True)

            # Alert log
            if st.session_state.alerts:
                rows = "".join([
                    f"<tr><td>{a['time'].strftime('%H:%M:%S')}</td><td style='color:#fc8181'>{a['ear']}</td></tr>"
                    for a in reversed(st.session_state.alerts[-10:])
                ])
                log_ph.markdown(f"""
<table class='log-table' style='width:100%;border-collapse:collapse'>
    <thead><tr style='color:#63b3ed;font-weight:600'>
        <th align='left'>Time</th><th align='left'>EAR</th>
    </tr></thead>
    <tbody>{rows}</tbody>
</table>""", unsafe_allow_html=True)
            else:
                log_ph.markdown("<p style='color:#4a5568;font-size:0.85rem'>No alerts yet.</p>",
                                unsafe_allow_html=True)

    finally:
        cap.release()
