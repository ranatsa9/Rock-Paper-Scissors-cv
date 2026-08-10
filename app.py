from pathlib import Path
from threading import Lock
import time

import av
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from ultralytics import YOLO


st.set_page_config(
    page_title="RPS Vision Arena",
    page_icon="✊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at 12% 4%, rgba(61, 218, 235, .18), transparent 27%),
            radial-gradient(circle at 88% 12%, rgba(239, 66, 126, .17), transparent 25%),
            radial-gradient(circle at 72% 88%, rgba(255, 208, 73, .10), transparent 28%),
            linear-gradient(145deg, #081329 0%, #0d1d3d 52%, #111a38 100%);
        color: #f6f8ff;
    }
    [data-testid="stSidebar"] {
        background: rgba(8, 18, 40, .72);
        border-right: 1px solid rgba(127, 220, 255, .16);
        box-shadow: 18px 0 50px rgba(0, 0, 0, .18);
        backdrop-filter: blur(28px);
    }
    [data-testid="stSidebar"] * { color: #f5feff; }
    [data-testid="stSidebar"] [data-baseweb="slider"] * { color: #ffffff; }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { max-width: 1240px; padding-top: 2rem; padding-bottom: 3rem; }
    h1, h2, h3 { color: #ffffff !important; letter-spacing: -.025em; }
    p, label, [data-testid="stMarkdownContainer"] { color: #cbd7f2; }
    .eyebrow {
        display: inline-flex;
        align-items: center;
        gap: .45rem;
        padding: .45rem .8rem;
        margin-bottom: 1rem;
        border-radius: 999px;
        background: rgba(74, 221, 235, .10);
        border: 1px solid rgba(74, 221, 235, .28);
        color: #73edf2;
        font-size: .78rem;
        font-weight: 800;
        letter-spacing: .09em;
        text-transform: uppercase;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 3.2rem 3.4rem;
        border: 1px solid rgba(144, 225, 255, .20);
        border-radius: 36px;
        background: linear-gradient(130deg, rgba(29, 53, 98, .72), rgba(17, 36, 76, .48));
        box-shadow: 0 28px 80px rgba(0, 0, 0, .30), inset 0 1px 0 rgba(255,255,255,.10);
        backdrop-filter: blur(30px);
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #ffffff !important; font-size: clamp(2.5rem, 5vw, 4.8rem); margin: 0; line-height: .98; max-width: 850px; }
    .hero h1 em { color: #55e5eb; font-style: normal; text-shadow: 0 0 28px rgba(85,229,235,.18); }
    .hero p { color: #b9c9eb; font-size: 1.15rem; margin: 1.1rem 0 0; max-width: 700px; }
    .gesture-card {
        text-align: center;
        padding: 1.1rem;
        border-radius: 21px;
        background: linear-gradient(145deg, rgba(31, 53, 95, .72), rgba(17, 34, 72, .50));
        border: 1px solid rgba(157, 214, 255, .15);
        box-shadow: 0 16px 40px rgba(0, 0, 0, .20), inset 0 1px 0 rgba(255,255,255,.08);
        font-size: 1.05rem;
        transition: transform .2s ease, box-shadow .2s ease;
    }
    .gesture-card:hover { transform: translateY(-5px); box-shadow: 0 22px 48px rgba(0, 0, 0, .27); }
    .gesture-card span { display: flex; align-items: center; justify-content: center; width: 82px; height: 82px; margin: 0 auto .7rem; border-radius: 50%; font-size: 2.5rem; }
    .gesture-card b { color: #ffffff; }
    .rock { border-bottom: 5px solid #ef427e; }
    .rock span { background: #ef427e; box-shadow: 0 0 32px rgba(239,66,126,.30); }
    .paper { border-bottom: 5px solid #ffd049; }
    .paper span { background: #ffd049; box-shadow: 0 0 32px rgba(255,208,73,.25); }
    .scissors { border-bottom: 5px solid #4dddeb; }
    .scissors span { background: #4dddeb; box-shadow: 0 0 32px rgba(77,221,235,.27); }
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: .8rem;
        background: rgba(18, 37, 78, .62);
        border: 1px solid rgba(139, 213, 255, .15);
        padding: .7rem;
        border-radius: 24px;
        margin-top: 1.5rem;
        box-shadow: 0 18px 45px rgba(0, 0, 0, .20), inset 0 1px 0 rgba(255,255,255,.07);
        backdrop-filter: blur(24px);
    }
    [data-testid="stTabs"] button {
        width: 100%;
        min-height: 76px;
        justify-content: center;
        border-radius: 18px;
        color: #cbd7f2;
        font-size: 1.02rem;
        font-weight: 800;
        padding: 1rem 1.2rem;
        background: rgba(31, 54, 98, .62);
        border: 1px solid rgba(146, 215, 255, .13);
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: white;
        background: linear-gradient(110deg, #ef427e, #9a54dc);
        box-shadow: 0 12px 28px rgba(239,66,126,.20);
    }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none; }
    [data-testid="stMetric"] {
        background: rgba(28, 50, 91, .66);
        border: 1px solid rgba(139, 213, 255, .14);
        border-radius: 18px;
        padding: .9rem;
    }
    .stButton > button, [data-testid="stBaseButton-primary"] {
        border-radius: 999px;
        font-weight: 700;
        border: 0;
        background: linear-gradient(110deg, #ff9838, #ffbf4b);
        color: #17394a;
        box-shadow: 0 10px 24px rgba(255, 153, 56, .25);
    }
    [data-testid="stFileUploader"] {
        background: rgba(25, 47, 87, .68);
        border: 2px dashed rgba(77, 221, 235, .38);
        padding: .8rem;
        border-radius: 22px;
    }
    [data-testid="stCameraInput"] {
        padding: 1rem;
        border-radius: 28px;
        border: 1px solid rgba(120, 219, 255, .22);
        background: linear-gradient(135deg, rgba(31,54,98,.72), rgba(15,34,73,.58));
        box-shadow: 0 26px 70px rgba(0, 0, 0, .28), inset 0 1px 0 rgba(255,255,255,.10);
        backdrop-filter: blur(18px);
    }
    [data-testid="stCustomComponentV1"] {
        padding: 16px;
        min-height: 650px;
        border-radius: 34px;
        border: 1px solid rgba(130, 224, 255, .28);
        background: linear-gradient(135deg, rgba(45,74,126,.60), rgba(13,31,67,.66));
        box-shadow: 0 30px 85px rgba(0, 0, 0, .38), inset 0 1px 0 rgba(255,255,255,.13), 0 0 55px rgba(77,221,235,.08);
        backdrop-filter: blur(30px);
        overflow: hidden;
    }
    [data-testid="stCustomComponentV1"] iframe { min-height: 615px !important; }
    .battle-guide {
        display: grid;
        grid-template-columns: 1fr auto 1fr;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.2rem;
        margin: .8rem 0 1.1rem;
        border-radius: 20px;
        border: 1px solid rgba(148, 215, 255, .16);
        background: rgba(19, 38, 79, .62);
        box-shadow: inset 0 1px 0 rgba(255,255,255,.08);
        backdrop-filter: blur(22px);
        color: white;
        font-weight: 800;
    }
    .battle-guide .p1 { color: #ff79a9; text-align: left; }
    .battle-guide .p2 { color: #65e8f0; text-align: right; }
    .battle-guide .vs { color: #ffd049; font-size: .82rem; letter-spacing: .12em; }
    [data-testid="stAlert"] { border-radius: 16px; }
    iframe { border-radius: 22px !important; }
    @media (max-width: 760px) {
        .hero { padding: 2rem 1.5rem; }
        [data-testid="stTabs"] [data-baseweb="tab-list"] { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path(__file__).parent / "model" / "best.pt"
MODEL_LOCK = Lock()


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    loaded_model = YOLO(str(MODEL_PATH))
    if torch.cuda.is_available():
        device = 0
        device_label = "NVIDIA GPU"
    elif torch.backends.mps.is_available():
        device = "mps"
        device_label = "Apple Silicon GPU"
    else:
        device = "cpu"
        device_label = "CPU"

    loaded_model.predict(
        source=np.zeros((320, 320, 3), dtype=np.uint8),
        imgsz=320,
        device=device,
        verbose=False,
    )
    return loaded_model, device, device_label


model, device, device_label = load_model()

CLASS_COLORS = {
    "rock": (126, 66, 239),
    "paper": (73, 208, 255),
    "scissors": (235, 221, 77),
}


def extract_detections(result):
    """Convert YOLO boxes into plain data for custom drawing and game logic."""
    detections = []
    if result.boxes is None:
        return detections

    names = result.names
    for xyxy, class_id, confidence in zip(
        result.boxes.xyxy.cpu().numpy(),
        result.boxes.cls.cpu().numpy(),
        result.boxes.conf.cpu().numpy(),
    ):
        x1, y1, x2, y2 = (int(value) for value in xyxy)
        class_name = str(names[int(class_id)]).lower()
        detections.append(
            {
                "box": (x1, y1, x2, y2),
                "name": class_name,
                "confidence": float(confidence),
                "center_x": (x1 + x2) / 2,
            }
        )
    return detections


def draw_corner_box(image, box, color, length=28, thickness=4):
    """Draw modern corner markers instead of a distracting full rectangle."""
    x1, y1, x2, y2 = box
    cv2.line(image, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(image, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(image, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(image, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - length), color, thickness)


def draw_detection_overlay(image_bgr, result, show_confidence=True):
    """Draw labels inside the frame so they never disappear off an edge."""
    output = image_bgr.copy()
    height, width = output.shape[:2]
    detections = extract_detections(result)

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        x1, x2 = max(0, x1), min(width - 1, x2)
        y1, y2 = max(0, y1), min(height - 1, y2)
        color = CLASS_COLORS.get(detection["name"], (235, 221, 77))
        draw_corner_box(output, (x1, y1, x2, y2), color)

        label = detection["name"].title()
        if show_confidence:
            label += f'  {detection["confidence"]:.0%}'
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = max(0.58, min(0.88, width / 1100))
        text_size, _ = cv2.getTextSize(label, font, font_scale, 2)
        label_width = text_size[0] + 24
        label_height = text_size[1] + 20
        label_x = min(max(8, x1 + 8), max(8, width - label_width - 8))
        # Always place the label inside the detected object and inside the frame.
        label_y = min(max(8, y1 + 8), max(8, height - label_height - 8))

        overlay = output.copy()
        cv2.rectangle(
            overlay,
            (label_x, label_y),
            (label_x + label_width, label_y + label_height),
            color,
            -1,
        )
        cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
        cv2.putText(
            output,
            label,
            (label_x + 12, label_y + label_height - 10),
            font,
            font_scale,
            (8, 16, 35),
            2,
            cv2.LINE_AA,
        )

    return output, detections


def round_winner(player_one, player_two):
    if player_one == player_two:
        return "DRAW"
    wins_against = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    if wins_against.get(player_one) == player_two:
        return "PLAYER 1 WINS"
    return "PLAYER 2 WINS"


def draw_game_hud(image_bgr, detections):
    """Show a two-player result using the two strongest left/right detections."""
    valid = [item for item in detections if item["name"] in CLASS_COLORS]
    height, width = image_bgr.shape[:2]
    hud_height = max(104, int(height * 0.17))
    # Keep the result above native fullscreen controls and outside crop-prone edges.
    safe_bottom = max(48, int(height * 0.07))
    hud_bottom = height - safe_bottom
    hud_top = hud_bottom - hud_height
    overlay = image_bgr.copy()
    cv2.rectangle(overlay, (0, hud_top), (width, hud_bottom), (8, 16, 38), -1)
    cv2.addWeighted(overlay, 0.88, image_bgr, 0.12, 0, image_bgr)

    font = cv2.FONT_HERSHEY_DUPLEX
    if len(valid) < 2:
        message = "BATTLE MODE  -  Waiting for two hands"
        size, _ = cv2.getTextSize(message, font, 0.78, 2)
        cv2.putText(
            image_bgr,
            message,
            ((width - size[0]) // 2, hud_top + hud_height // 2 + size[1] // 2),
            font,
            0.78,
            (235, 221, 77),
            2,
            cv2.LINE_AA,
        )
        return image_bgr

    players = sorted(sorted(valid, key=lambda item: item["confidence"], reverse=True)[:2], key=lambda item: item["center_x"])
    player_one, player_two = players
    result_text = round_winner(player_one["name"], player_two["name"])
    player_scale = max(0.62, min(0.86, width / 1100))
    result_scale = max(0.65, min(0.95, width / 950))

    # Strong on-hand identity badges make left/right roles unmistakable.
    for player, badge, badge_color in (
        (player_one, "P1", (126, 66, 239)),
        (player_two, "P2", (235, 221, 77)),
    ):
        x1, y1, x2, _ = player["box"]
        badge_x = min(max(34, x2 - 34), width - 34)
        badge_y = min(max(52, y1 + 58), hud_top - 34)
        cv2.circle(image_bgr, (badge_x, badge_y), 30, (8, 16, 38), -1, cv2.LINE_AA)
        cv2.circle(image_bgr, (badge_x, badge_y), 26, badge_color, -1, cv2.LINE_AA)
        badge_size, _ = cv2.getTextSize(badge, font, 0.65, 2)
        cv2.putText(image_bgr, badge, (badge_x - badge_size[0] // 2, badge_y + badge_size[1] // 2), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    # Tint each side of the score bar with its player's permanent color.
    score_overlay = image_bgr.copy()
    cv2.rectangle(score_overlay, (0, hud_top), (width // 3, hud_bottom), (70, 25, 100), -1)
    cv2.rectangle(score_overlay, (width * 2 // 3, hud_top), (width, hud_bottom), (94, 75, 20), -1)
    cv2.addWeighted(score_overlay, 0.42, image_bgr, 0.58, 0, image_bgr)

    cv2.putText(image_bgr, f'PLAYER 1  |  {player_one["name"].upper()}', (24, hud_top + hud_height // 2 + 8), font, player_scale, (255, 255, 255), 2, cv2.LINE_AA)
    right_text = f'{player_two["name"].upper()}  |  PLAYER 2'
    right_size, _ = cv2.getTextSize(right_text, font, player_scale, 2)
    cv2.putText(image_bgr, right_text, (width - right_size[0] - 24, hud_top + hud_height // 2 + 8), font, player_scale, (255, 255, 255), 2, cv2.LINE_AA)

    result_size, _ = cv2.getTextSize(result_text, font, result_scale, 2)
    result_color = (73, 208, 255) if result_text == "DRAW" else (235, 221, 77)
    cv2.putText(image_bgr, result_text, ((width - result_size[0]) // 2, hud_top + hud_height // 2 + 10), font, result_scale, result_color, 2, cv2.LINE_AA)
    return image_bgr


def motion_score(previous_gray, current_bgr):
    """Estimate whole-frame motion cheaply before running the YOLO model."""
    small = cv2.resize(current_bgr, (160, 90), interpolation=cv2.INTER_AREA)
    current_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.GaussianBlur(current_gray, (7, 7), 0)
    if previous_gray is None:
        return 255.0, current_gray
    difference = cv2.absdiff(previous_gray, current_gray)
    return float(np.mean(difference)), current_gray


def draw_hold_prompt(image_bgr, stable_frames, required_frames):
    """Tell players to hold still while the final gesture is being locked."""
    output = image_bgr.copy()
    height, width = output.shape[:2]
    progress = min(1.0, stable_frames / required_frames)
    box_width = min(width - 40, 520)
    box_height = 68
    left = (width - box_width) // 2
    top = 24

    overlay = output.copy()
    cv2.rectangle(overlay, (left, top), (left + box_width, top + box_height), (8, 16, 38), -1)
    cv2.addWeighted(overlay, 0.88, output, 0.12, 0, output)
    message = "HOLD YOUR MOVE" if stable_frames == 0 else "LOCKING GESTURE..."
    font = cv2.FONT_HERSHEY_DUPLEX
    text_size, _ = cv2.getTextSize(message, font, 0.72, 2)
    cv2.putText(
        output,
        message,
        ((width - text_size[0]) // 2, top + 31),
        font,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    bar_left = left + 22
    bar_right = left + box_width - 22
    bar_top = top + 45
    cv2.rectangle(output, (bar_left, bar_top), (bar_right, bar_top + 8), (48, 58, 87), -1)
    cv2.rectangle(output, (bar_left, bar_top), (bar_left + int((bar_right - bar_left) * progress), bar_top + 8), (235, 221, 77), -1)
    return output


def draw_countdown(image_bgr, text, accent=(235, 221, 77)):
    """Draw a large, readable round prompt over the live camera."""
    output = image_bgr.copy()
    height, width = output.shape[:2]
    overlay = output.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (8, 16, 38), -1)
    cv2.addWeighted(overlay, 0.44, output, 0.56, 0, output)

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = max(0.9, min(4.0, width / max(280, len(text) * 32)))
    thickness = max(3, int(scale * 2))
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    x = max(16, (width - text_size[0]) // 2)
    y = (height + text_size[1]) // 2
    cv2.putText(output, text, (x, y), font, scale, (8, 16, 38), thickness + 8, cv2.LINE_AA)
    cv2.putText(output, text, (x, y), font, scale, accent, thickness, cv2.LINE_AA)
    return output


@st.cache_resource
def get_round_state():
    return {
        "lock": Lock(),
        "status": "idle",
        "detect_at": 0.0,
        "result_frame": None,
    }


ROUND_STATE = get_round_state()

with st.sidebar:
    st.title("🎮 Arena controls")
    live_confidence = st.slider(
        "Live confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Raise this if the camera makes incorrect guesses.",
    )
    live_size = st.select_slider(
        "Live quality",
        options=[256, 320, 416],
        value=320,
        help="256 is fastest. 416 can be more accurate but slower.",
    )
    mirror_camera = st.toggle(
        "Mirror camera",
        value=True,
        help="Makes the camera behave like a mirror and keeps game sides intuitive.",
    )
    battle_mode = st.toggle(
        "Two-player battle mode",
        value=False,
        help="Player 1 stands on the left and Player 2 stands on the right.",
    )
    st.divider()
    st.caption(f"⚡ Running on: **{device_label}**")
    st.caption("Tip: keep one hand clearly visible and use even lighting.")

st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">● Live computer vision</div>
        <h1>Play with your hands.<br><em>Let AI call the move.</em></h1>
        <p>A fast, colorful Rock–Paper–Scissors detector built for live camera play and instant image testing.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

rock_col, paper_col, scissors_col = st.columns(3)
with rock_col:
    st.markdown('<div class="gesture-card rock"><span>✊</span><b>Rock</b></div>', unsafe_allow_html=True)
with paper_col:
    st.markdown('<div class="gesture-card paper"><span>✋</span><b>Paper</b></div>', unsafe_allow_html=True)
with scissors_col:
    st.markdown('<div class="gesture-card scissors"><span>✌️</span><b>Scissors</b></div>', unsafe_allow_html=True)

live_tab, camera_tab, upload_tab = st.tabs(
    ["📹  LIVE CAMERA", "📸  TAKE A PHOTO", "🖼️  UPLOAD IMAGE"]
)


def detect_still_image(rgb_image, confidence):
    input_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    with MODEL_LOCK:
        result = model.predict(
            source=input_bgr,
            conf=float(confidence),
            imgsz=640,
            device=device,
            max_det=10,
            verbose=False,
        )[0]
    annotated_bgr, _ = draw_detection_overlay(input_bgr, result)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    return result, annotated_rgb


def show_still_result(rgb_image, confidence):
    with st.spinner("Looking for a gesture…"):
        result, annotated_rgb = detect_still_image(rgb_image, confidence)

    before_col, after_col = st.columns(2)
    with before_col:
        st.image(rgb_image, caption="Your photo", use_container_width=True)
    with after_col:
        st.image(annotated_rgb, caption="AI detection", use_container_width=True)

    detection_count = len(result.boxes)
    st.metric("Gestures detected", detection_count)
    if detection_count == 0:
        st.warning("No confident gesture was found. Try lowering confidence slightly or retaking the photo.")

with live_tab:
    st.subheader("Three-second game round")
    st.write("Start the camera, press **START ROUND**, then reveal your gesture when the countdown reaches **SHOW!**")
    if battle_mode:
        st.markdown(
            '<div class="battle-guide"><span class="p1">● PLAYER 1 — LEFT</span><span class="vs">VS</span><span class="p2">RIGHT — PLAYER 2 ●</span></div>',
            unsafe_allow_html=True,
        )

    if st.button("▶ START ROUND", type="primary", use_container_width=True):
        with ROUND_STATE["lock"]:
            ROUND_STATE["status"] = "countdown"
            ROUND_STATE["detect_at"] = time.monotonic() + 3.0
            ROUND_STATE["result_frame"] = None

    def video_frame_callback(frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        if mirror_camera:
            image_bgr = cv2.flip(image_bgr, 1)

        with ROUND_STATE["lock"]:
            status = ROUND_STATE["status"]
            detect_at = ROUND_STATE["detect_at"]
            frozen_result = ROUND_STATE["result_frame"]

        if status == "idle":
            waiting = draw_countdown(image_bgr, "PRESS START ROUND", (235, 221, 77))
            return av.VideoFrame.from_ndarray(waiting, format="bgr24")

        if status == "result" and frozen_result is not None:
            return av.VideoFrame.from_ndarray(frozen_result, format="bgr24")

        remaining = detect_at - time.monotonic()
        if remaining > 0:
            countdown_number = str(max(1, int(np.ceil(remaining))))
            countdown_frame = draw_countdown(image_bgr, countdown_number)
            return av.VideoFrame.from_ndarray(countdown_frame, format="bgr24")

        # The countdown has finished: run YOLO once on this final frame.
        with ROUND_STATE["lock"]:
            if ROUND_STATE["status"] != "countdown":
                frozen_result = ROUND_STATE["result_frame"]
                if frozen_result is not None:
                    return av.VideoFrame.from_ndarray(frozen_result, format="bgr24")
                show_frame = draw_countdown(image_bgr, "SHOW!", (73, 208, 255))
                return av.VideoFrame.from_ndarray(show_frame, format="bgr24")
            ROUND_STATE["status"] = "detecting"

        with MODEL_LOCK:
            result = model.predict(
                source=image_bgr,
                conf=float(live_confidence),
                imgsz=int(live_size),
                device=device,
                max_det=5,
                iou=0.45,
                verbose=False,
            )[0]

        annotated_bgr, detections = draw_detection_overlay(image_bgr, result, show_confidence=False)
        if battle_mode:
            annotated_bgr = draw_game_hud(annotated_bgr, detections)
        if not detections:
            annotated_bgr = draw_countdown(image_bgr, "NO GESTURE", (126, 66, 239))

        with ROUND_STATE["lock"]:
            ROUND_STATE["result_frame"] = annotated_bgr.copy()
            ROUND_STATE["status"] = "result"
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")

    webrtc_streamer(
        key="rps-live-camera",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration={
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        },
        media_stream_constraints={"video": True, "audio": False},
        video_html_attrs={
            "style": {
                "width": "100%",
                "height": "560px",
                "objectFit": "contain",
                "borderRadius": "24px",
                "backgroundColor": "#060d20",
            },
            "autoPlay": True,
            "playsInline": True,
            "muted": True,
        },
        async_processing=True,
    )
    st.info("The AI runs once after each three-second countdown. Press START ROUND again to play another round.")

with camera_tab:
    st.subheader("Take a photo and detect your move")
    st.write("Pose first, press **Take Photo**, and the AI will analyze the captured frame.")
    camera_confidence = st.slider(
        "Camera photo confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    camera_photo = st.camera_input("Camera snapshot")

    if camera_photo is not None:
        camera_rgb = np.array(Image.open(camera_photo).convert("RGB"))
        if mirror_camera:
            camera_rgb = cv2.flip(camera_rgb, 1)
        show_still_result(camera_rgb, camera_confidence)

with upload_tab:
    st.subheader("Upload a photo")
    upload_confidence = st.slider(
        "Upload confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )
    uploaded_file = st.file_uploader("Choose a JPG, JPEG, or PNG", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        rgb_image = np.array(Image.open(uploaded_file).convert("RGB"))
        show_still_result(rgb_image, upload_confidence)

with st.expander("💡 How to get the best detection"):
    st.markdown(
        "Keep your full hand visible, use even lighting and a simple background, "
        "and stay around 40–80 cm from the camera. Raise confidence to reduce "
        "wrong guesses or lower it to reduce missed detections."
    )

st.caption("RPS Vision Arena • YOLO gesture detection • Real-time camera processing")
