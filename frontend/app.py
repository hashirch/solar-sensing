"""Local web UI for comparing the trained solar-panel segmentation models."""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from frontend.cnn_model import build_model, build_transforms
from otp_service import OTPChallenge, otp_service


FRONTEND_ROOT = Path(__file__).resolve().parent
CNN_WEIGHTS = FRONTEND_ROOT / "models/best_cnn_unet.pth"
RCNN_WEIGHTS = FRONTEND_ROOT / "models/mask_rcnn_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OTP_REQUIRED = os.getenv("SOLARSIGHT_OTP_REQUIRED", "").lower() == "true"


st.set_page_config(page_title="SolarSight", layout="wide")


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

    st.title("SolarSight")
    st.caption("Verify your email to open the local segmentation workspace.")
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
    st.stop()


require_email_verification()

@st.cache_resource
def load_cnn():
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
    models = st.multiselect(
        "Models to run",
        ["CNN/U-Net", "Mask R-CNN"],
        default=["CNN/U-Net", "Mask R-CNN"],
    )

st.caption("Compare solar-panel segmentation models on local images.")
st.subheader("Upload images")
photos = st.file_uploader(
    "Upload satellite photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True,
)

if not models:
    st.info("Select at least one model from the analysis setup to begin.")
elif photos:
    st.subheader("Results")
    for photo in photos:
        image = Image.open(photo).convert("RGB")
        st.markdown(f"**{photo.name}** — {image.width} × {image.height} px")
        columns = st.columns(len(models))
        for column, model_name in zip(columns, models):
            with column:
                st.write(model_name)
                try:
                    with st.spinner(f"Running {model_name}..."):
                        prediction = (
                            cnn_prediction(image)
                            if model_name == "CNN/U-Net"
                            else rcnn_prediction(image)
                        )
                    st.image(prediction, use_container_width=True)
                except Exception as error:
                    st.error(f"{type(error).__name__}: {error}")
else:
    st.info("Upload an image to see model predictions.")
