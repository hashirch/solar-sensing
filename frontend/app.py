"""Local web UI for comparing the trained solar-panel segmentation models."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from otp_service import OTPChallenge, otp_service


ROOT = Path(__file__).resolve().parents[1]
CNN_WEIGHTS = ROOT / "results/weights/best_cnn_unet.pth"
RCNN_WEIGHTS = ROOT / "results/weights/mask_rcnn_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OTP_REQUIRED = os.getenv("SOLARSIGHT_OTP_REQUIRED", "").lower() == "true"


st.set_page_config(
    page_title="SolarSight | Panel Intelligence",
    page_icon="◒",
    layout="wide",
    initial_sidebar_state="expanded",
)


def require_email_verification() -> None:
    """Gate the local workspace with a visible two-step email login."""
    if not OTP_REQUIRED:
        return

    challenge: OTPChallenge | None = st.session_state.get("otp_challenge")
    verified_email = st.session_state.get("verified_email")
    if verified_email:
        with st.sidebar:
            st.success(f"Verified: {verified_email}")
            if st.button("Sign out", key="otp_sign_out"):
                st.session_state.pop("verified_email", None)
                st.session_state.pop("otp_challenge", None)
                st.session_state.pop("otp_challenge_email", None)
                st.rerun()
        return

    st.markdown(
        '<div class="login-shell"><div class="login-card">'
        '<div class="eyebrow">SolarSight · Secure access</div>'
        '<h1 class="login-title">Welcome back.</h1>'
        '<p class="login-copy">Verify your email to open the local segmentation workspace.</p>',
        unsafe_allow_html=True,
    )
    email = st.text_input(
        "Email address",
        value="hinarashid431@gmail.com",
        key="otp_email",
        disabled=challenge is not None,
    )
    if challenge is None:
        if st.button("Continue with email", type="primary"):
            if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
                st.error("Enter a valid email address.")
            elif otp_service.send_verification_email(
                "SolarSight user", email, (new := otp_service.create_challenge()).code
            ):
                st.session_state.otp_challenge = new
                st.session_state.otp_challenge_email = email.strip().lower()
                st.rerun()
            else:
                st.error("Email delivery is unavailable. Check SMTP settings.")
        st.markdown("</div></div>", unsafe_allow_html=True)
        st.stop()

    challenge_email = st.session_state.otp_challenge_email
    st.caption(f"Code sent to {challenge_email}")
    code = st.text_input("Verification code", max_chars=6, key="otp_code")
    if st.button("Verify and open workspace", type="primary"):
        if otp_service.verify(challenge, code):
            st.session_state.verified_email = challenge_email
            st.session_state.pop("otp_challenge", None)
            st.session_state.pop("otp_challenge_email", None)
            st.session_state.pop("otp_code", None)
            st.rerun()
        st.error("That code is invalid or expired.")

    if st.button("Resend code"):
        new_challenge = otp_service.create_challenge()
        if otp_service.send_resend_email("SolarSight user", challenge_email, new_challenge.code):
            st.session_state.otp_challenge = new_challenge
            st.session_state.otp_challenge_email = challenge_email
            st.rerun()
        st.error("Email delivery is unavailable. Check SMTP settings.")
    st.markdown("</div></div>", unsafe_allow_html=True)
    st.stop()


require_email_verification()

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;600;700&display=swap');

    :root {
        --ink: #14342b;
        --muted: #647a70;
        --leaf: #24744d;
        --leaf-dark: #143f31;
        --mint: #eaf5ee;
        --line: #d7e7dc;
        --gold: #b97d25;
        --paper: #f7fbf8;
        --white: #ffffff;
    }

    .stApp {
        background: var(--paper);
        color: var(--ink);
        font-family: 'DM Sans', sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: #edf6ef;
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
    .block-container { max-width: 1440px; padding: 2.5rem 4rem 4rem; }
    h1, h2, h3, p { color: var(--ink); }
    h1 { font-size: clamp(2.2rem, 4vw, 4.25rem); letter-spacing: -0.06em; line-height: .98; }
    h2 { letter-spacing: -0.04em; }
    .mono { font-family: 'DM Mono', monospace; }

    .brand {
        display: flex; align-items: center; gap: .75rem; margin-bottom: 3rem;
    }
    .brand-mark {
        width: 2.25rem; height: 2.25rem; display: grid; place-items: center;
        border-radius: 12px 12px 12px 4px; color: white; background: var(--leaf-dark);
        font-size: 1.15rem; transform: rotate(-8deg);
    }
    .brand-name { font-size: 1.05rem; font-weight: 700; letter-spacing: -.03em; }
    .brand-subtitle { color: var(--muted); font-size: .72rem; margin-top: .1rem; }
    .eyebrow {
        color: var(--leaf); font: 500 .72rem 'DM Mono', monospace;
        letter-spacing: .12em; text-transform: uppercase; margin-bottom: 1rem;
    }
    .hero-copy { max-width: 720px; margin-bottom: 2.5rem; }
    .hero-copy p { color: var(--muted); font-size: 1.05rem; line-height: 1.65; max-width: 610px; }
    .accent { color: var(--leaf); }
    .login-shell { max-width: 540px; margin: 8vh auto 0; }
    .login-card { background: var(--white); border: 1px solid var(--line); border-radius: 20px; padding: 2rem; box-shadow: 0 10px 35px rgba(20, 63, 49, .06); }
    .login-title { margin: .4rem 0 .7rem; font-size: clamp(2.2rem, 5vw, 3.8rem); }
    .login-copy { color: var(--muted); margin: 0 0 1.5rem; }

    .stat-card, .model-card, .result-card, .upload-panel {
        background: var(--white); border: 1px solid var(--line); border-radius: 20px;
        box-shadow: 0 10px 35px rgba(20, 63, 49, .045);
    }
    .stat-card { padding: 1.1rem 1.25rem; height: 100%; }
    .stat-label { color: var(--muted); font-size: .78rem; }
    .stat-value { color: var(--ink); font-size: 1.35rem; font-weight: 700; margin-top: .4rem; }
    .stat-note { color: var(--leaf); font-size: .72rem; margin-top: .25rem; }

    .section-label {
        color: var(--muted); font: 500 .72rem 'DM Mono', monospace;
        letter-spacing: .08em; text-transform: uppercase; margin: 2.75rem 0 1rem;
    }
    .model-card { padding: 1.2rem; min-height: 135px; }
    .model-card.active { border-color: #9ac9ac; background: #f5fbf6; }
    .model-head { display: flex; justify-content: space-between; gap: 1rem; }
    .model-name { font-weight: 700; }
    .model-kind { color: var(--muted); font-size: .8rem; margin-top: .35rem; }
    .pill {
        display: inline-flex; align-items: center; gap: .35rem; color: var(--leaf);
        background: var(--mint); border-radius: 999px; padding: .3rem .55rem;
        font: 500 .68rem 'DM Mono', monospace; white-space: nowrap;
    }
    .pill::before {
        content: ''; width: 6px; height: 6px; background: #3aa66a; border-radius: 50%;
    }

    .upload-panel { padding: 1.4rem; margin-top: 1.5rem; }
    .upload-title { font-size: 1.1rem; font-weight: 700; margin-bottom: .35rem; }
    .upload-copy { color: var(--muted); font-size: .9rem; margin-bottom: 1rem; }
    [data-testid="stFileUploaderDropzone"] {
        background: #f6fbf7; border: 1px dashed #9bc9aa; border-radius: 14px;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--leaf); }
    .result-card { padding: 1rem; margin-top: 1rem; }
    .result-heading {
        display: flex; align-items: center; justify-content: space-between;
        margin-bottom: .8rem;
    }
    .result-name { font-weight: 700; overflow-wrap: anywhere; }
    .result-meta { color: var(--muted); font: .7rem 'DM Mono', monospace; }
    .empty-state {
        text-align: center; border: 1px dashed #a7cbb2; border-radius: 20px;
        background: rgba(255,255,255,.55); padding: 3.5rem 1rem; margin-top: 1.5rem;
    }
    .empty-icon { color: var(--leaf); font-size: 2.5rem; margin-bottom: .65rem; }
    .empty-state p { color: var(--muted); margin: .35rem auto; max-width: 480px; }
    .footer {
        color: var(--muted); border-top: 1px solid var(--line); margin-top: 4rem;
        padding-top: 1.25rem; font-size: .75rem;
    }
    button:focus, input:focus, [tabindex="0"]:focus {
        outline: 3px solid rgba(185, 125, 37, .45) !important; outline-offset: 3px !important;
    }
    @media (max-width: 800px) {
        .block-container { padding: 1.5rem 1.25rem 3rem; }
        .brand { margin-bottom: 2rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_cnn():
    from src.solar_segmentation.cnn import build_model, build_transforms

    model = build_model(None).to(DEVICE)
    model.load_state_dict(
        torch.load(CNN_WEIGHTS, map_location=DEVICE, weights_only=True)
    )
    model.eval()
    return model, build_transforms(train=False)


@st.cache_resource
def load_rcnn():
    from detectron2 import model_zoo
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = str(RCNN_WEIGHTS)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = str(DEVICE)
    cfg.INPUT.MIN_SIZE_TEST = 432
    cfg.INPUT.MAX_SIZE_TEST = 512
    return DefaultPredictor(cfg)


def cnn_prediction(image: Image.Image) -> Image.Image:
    model, transform = load_cnn()
    image_array = np.asarray(image.convert("RGB"))
    transformed = transform(image=image_array, mask=np.zeros(image_array.shape[:2]))
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        mask = torch.sigmoid(model(tensor))[0, 0].cpu().numpy() > 0.5

    resized_mask = cv2.resize(
        mask.astype(np.uint8),
        (image.width, image.height),
        interpolation=cv2.INTER_NEAREST,
    )
    return overlay_mask(image, resized_mask, (0, 255, 0))


def rcnn_prediction(image: Image.Image) -> Image.Image:
    predictor = load_rcnn()
    bgr_image = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    output = predictor(bgr_image)
    result = bgr_image.copy()
    instances = output["instances"].to("cpu")
    if instances.has("pred_masks"):
        for mask in instances.pred_masks.numpy():
            result[mask] = (0.55 * result[mask] + 0.45 * np.array([0, 0, 255])).astype(
                np.uint8
            )
    if instances.has("pred_boxes"):
        for box in instances.pred_boxes.tensor.numpy().astype(int):
            x1, y1, x2, y2 = box
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 2)
    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))


def overlay_mask(
    image: Image.Image, mask: np.ndarray, color: tuple[int, int, int]
) -> Image.Image:
    base = np.asarray(image.convert("RGB")).copy()
    base[mask.astype(bool)] = (
        0.55 * base[mask.astype(bool)] + 0.45 * np.asarray(color)
    ).astype(np.uint8)
    return Image.fromarray(base)


with st.sidebar:
    st.markdown(
        """
        <div class="brand">
            <div class="brand-mark">◒</div>
            <div><div class="brand-name">SolarSight</div>
            <div class="brand-subtitle">Panel intelligence workspace</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Analysis setup")
    st.caption("Choose the segmentation engines you want to compare.")
    models = st.multiselect(
        "Models to run",
        ["CNN/U-Net", "Mask R-CNN"],
        default=["CNN/U-Net", "Mask R-CNN"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("### Run environment")
    st.markdown(
        f'<span class="pill">{"GPU" if DEVICE.type == "cuda" else "CPU"} ready</span>',
        unsafe_allow_html=True,
    )
    st.caption(f"PyTorch inference device: `{DEVICE}`")
    st.caption("Results are processed locally. Images are not uploaded to a remote service.")


st.markdown('<div class="eyebrow">Solar panel segmentation · 01</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-copy"><h1>See the surface.<br><span class="accent">Measure the potential.</span></h1>'
    "<p>Compare two computer-vision models on satellite imagery and surface the solar panels that matter for your next decision.</p></div>",
    unsafe_allow_html=True,
)

stat_columns = st.columns(3)
stats = [
    ("2", "Segmentation engines", "CNN/U-Net + Mask R-CNN"),
    ("Local", "Privacy-first workflow", "No remote image uploads"),
    ("2021—25", "Multi-temporal dataset", "Built for change over time"),
]
for column, (value, label, note) in zip(stat_columns, stats):
    with column:
        st.markdown(
            f'<div class="stat-card"><div class="stat-label">{label}</div>'
            f'<div class="stat-value">{value}</div><div class="stat-note">{note}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-label">Available engines</div>', unsafe_allow_html=True)
model_columns = st.columns(2)
for column, name, kind, description in [
    (model_columns[0], "CNN / U-Net", "Pixel-level segmentation", "Fast, dense masks for a complete surface view."),
    (model_columns[1], "Mask R-CNN", "Instance segmentation", "Distinct detections with individual bounding boxes."),
]:
    with column:
        active = name.replace(" / ", "/") in models or name in models
        st.markdown(
            f'<div class="model-card {"active" if active else ""}"><div class="model-head">'
            f'<div><div class="model-name">{name}</div><div class="model-kind">{kind}</div></div>'
            f'<span class="pill">{"Selected" if active else "Available"}</span></div>'
            f'<div class="model-kind" style="margin-top:1rem">{description}</div></div>',
            unsafe_allow_html=True,
        )

st.markdown('<div class="section-label">Start a comparison</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="upload-panel"><div class="upload-title">Bring in satellite imagery</div>'
    '<div class="upload-copy">Upload JPG, PNG, or WebP files. Add multiple images to compare a batch in one run.</div>',
    unsafe_allow_html=True,
)
photos = st.file_uploader(
    "Upload satellite photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)
st.markdown("</div>", unsafe_allow_html=True)

if not models:
    st.info("Select at least one model from the analysis setup to begin.")
elif photos:
    st.markdown(
        f'<div class="section-label">Comparison results · {len(photos)} image{"s" if len(photos) != 1 else ""}</div>',
        unsafe_allow_html=True,
    )
    for photo in photos:
        image = Image.open(photo).convert("RGB")
        st.markdown(
            f'<div class="result-card"><div class="result-heading"><div class="result-name">{photo.name}</div>'
            f'<div class="result-meta">{image.width} × {image.height} px</div></div>',
            unsafe_allow_html=True,
        )
        columns = st.columns(len(models))
        for column, model_name in zip(columns, models):
            with column:
                st.markdown(f"**{model_name}**")
                try:
                    with st.spinner(f"Running {model_name}..."):
                        prediction = (
                            cnn_prediction(image)
                            if model_name == "CNN/U-Net"
                            else rcnn_prediction(image)
                        )
                    st.image(prediction, use_container_width=True)
                    st.caption("Overlay generated locally · Ready for review")
                except Exception as error:
                    st.error(f"{type(error).__name__}: {error}")
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.markdown(
        '<div class="empty-state"><div class="empty-icon">◒</div><h3>Your workspace is ready</h3>'
        "<p>Upload an aerial image above to reveal the model overlays and compare panel coverage side by side.</p></div>",
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer"><span class="mono">SOLARSIGHT / LOCAL INFERENCE</span>'
    " · A visual comparison tool for the solar-panel segmentation research workflow.</div>",
    unsafe_allow_html=True,
)
