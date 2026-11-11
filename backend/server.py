"""Inference API server connecting the real PyTorch models to the React frontend."""

import base64
import io
import json
import hashlib
import logging
import re
import secrets
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from backend import db
from backend.cnn_model import build_model, build_transforms
from backend.otp_service import otp_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = Path(__file__).resolve().parent.parent
CNN_WEIGHTS = ROOT_DIR / "backend/models/best_cnn_unet.pth"
RCNN_WEIGHTS = ROOT_DIR / "backend/models/mask_rcnn_final.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Global model references
cnn_model = None
cnn_transform = None
rcnn_predictor = None


def init_models():
    global cnn_model, cnn_transform, rcnn_predictor

    logging.info(f"Loading CNN / U-Net from {CNN_WEIGHTS} on {DEVICE}...")
    cnn_model = build_model(None).to(DEVICE)
    cnn_model.load_state_dict(
        torch.load(CNN_WEIGHTS, map_location=DEVICE, weights_only=True)
    )
    cnn_model.eval()
    cnn_transform = build_transforms(train=False)
    logging.info("CNN / U-Net loaded successfully.")

    logging.info(f"Loading Detectron2 Mask R-CNN from {RCNN_WEIGHTS} on {DEVICE}...")
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
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.45  # Filter low-confidence false positives
    rcnn_predictor = DefaultPredictor(cfg)
    logging.info("Mask R-CNN loaded successfully.")


def run_cnn_inference(image: Image.Image, extras: dict | None = None):
    """Executes true CNN U-Net prediction using best_cnn_unet.pth."""
    w, h = image.size
    image_array = np.asarray(image.convert("RGB"))
    transformed = cnn_transform(image=image_array, mask=np.zeros(image_array.shape[:2]))
    tensor = transformed["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        raw_pred = torch.sigmoid(cnn_model(tensor))[0, 0].cpu().numpy()
        mask_binary = (raw_pred > 0.5).astype(np.uint8)

    resized_mask = cv2.resize(
        mask_binary,
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )

    # Green translucent overlay matching Streamlit frontend (0.55 * base + 0.45 * [0, 255, 0])
    base = image_array.copy()
    has_panels = resized_mask.astype(bool)
    base[has_panels] = (
        0.55 * base[has_panels] + 0.45 * np.array([34, 197, 94], dtype=np.uint8)
    ).astype(np.uint8)

    pil_result = Image.fromarray(base)
    buffer = io.BytesIO()
    pil_result.save(buffer, format="JPEG", quality=92)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_str}"

    if extras is not None:
        extras["cnn_mask"] = has_panels
    total_panel_pixels = int(np.sum(has_panels))
    return data_url, total_panel_pixels, (w, h)


def run_rcnn_inference(image: Image.Image, min_conf=0.50, extras: dict | None = None):
    """Executes true Detectron2 Mask R-CNN prediction using mask_rcnn_final.pth with tiny, clean percentage tags."""
    w, h = image.size
    rgb_arr = np.asarray(image.convert("RGB"))
    bgr_image = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)

    output = rcnn_predictor(bgr_image)
    instances = output["instances"].to("cpu")

    result = bgr_image.copy()
    instances_data = []
    kept = []  # (mask or None, box, score) for the combined view

    has_masks = instances.has("pred_masks")
    has_boxes = instances.has("pred_boxes")
    has_scores = instances.has("scores")

    num_instances = len(instances)

    for i in range(num_instances):
        score = float(instances.scores[i].item()) if has_scores else 0.9
        if score < min_conf:
            continue

        box = instances.pred_boxes.tensor[i].numpy().astype(int) if has_boxes else None
        kept.append((instances.pred_masks[i].numpy() if has_masks else None, box, score))

        if has_masks:
            mask = instances.pred_masks[i].numpy()
            # Red/coral mask overlay: (0.55 * result + 0.45 * [0, 0, 239])
            result[mask] = (
                0.55 * result[mask] + 0.45 * np.array([0, 0, 239])
            ).astype(np.uint8)

        if box is not None:
            x1, y1, x2, y2 = box
            # 1px thin yellow bounding box for clean presentation
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 1)

            # Tiny, minimal percentage badge (e.g. "99%")
            pct_label = f"{int(round(score * 100))}%"
            font_scale = 0.25  # Very small micro-font
            (tw, th), baseline = cv2.getTextSize(
                pct_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            badge_h = th + 3
            badge_w = tw + 4
            badge_y = max(badge_h, y1)

            # Dark micro background
            cv2.rectangle(
                result,
                (x1, badge_y - badge_h),
                (x1 + badge_w, badge_y),
                (15, 23, 42),
                -1,
            )
            # Yellow text
            cv2.putText(
                result,
                pct_label,
                (x1 + 2, badge_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

            bw = int(x2 - x1)
            bh = int(y2 - y1)
            area_m2 = round(bw * bh * 0.09, 1)

            instances_data.append(
                {
                    "id": len(instances_data) + 1,
                    "bbox": [int(x1), int(y1), bw, bh],
                    "confidence": round(score, 3),
                    "areaM2": area_m2,
                }
            )

    if extras is not None:
        extras["rcnn_instances"] = kept

    annotated_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    pil_result = Image.fromarray(annotated_rgb)
    buffer = io.BytesIO()
    pil_result.save(buffer, format="JPEG", quality=92)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_str}"

    return data_url, instances_data


def render_combined(image: Image.Image, cnn_mask, rcnn_instances) -> str:
    """CNN dense coverage (green fill) with Mask R-CNN panel outlines, boxes and scores on top."""
    rgb = np.asarray(image.convert("RGB")).copy()
    if cnn_mask is not None and cnn_mask.any():
        rgb[cnn_mask] = (0.6 * rgb[cnn_mask] + 0.4 * np.array([34, 197, 94], dtype=np.uint8)).astype(np.uint8)
    out = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for mask, box, score in rcnn_instances:
        if mask is not None:
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, (60, 60, 239), 1, cv2.LINE_AA)  # coral outline (BGR)
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 1)
            label = f"{int(round(score * 100))}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.25, 1)
            by = max(th + 3, y1)
            cv2.rectangle(out, (x1, by - th - 3), (x1 + tw + 4, by), (15, 23, 42), -1)
            cv2.putText(out, label, (x1 + 2, by - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 255, 255), 1, cv2.LINE_AA)

    buffer = io.BytesIO()
    Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)).save(buffer, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


# ─── Account auth: password (hashed) + email one-time code ───
OTP_MAX_ATTEMPTS = 5
OTP_RESEND_COOLDOWN_S = 30
LOGIN_MAX_FAILS = 5
LOGIN_WINDOW_S = 900
_otp_challenges: dict = {}  # email -> {"code_hash", "expires_at", "attempts", "sent"}
_pending: dict = {}  # email -> password step passed, waiting for the code
_login_fails: dict = {}  # email -> [timestamps of failed password attempts]
_DUMMY_HASH = db.hash_password("not-a-real-password")  # keeps timing equal for unknown emails


_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]{2,}$")


def _valid_email(email: str) -> bool:
    return len(email) <= 254 and bool(_EMAIL_RE.match(email))


def _sha(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _send_code(email: str, name: str, enforce_cooldown: bool = False, key: str = ""):
    key = key or email
    existing = _otp_challenges.get(key)
    if enforce_cooldown and existing and time.time() - existing["sent"] < OTP_RESEND_COOLDOWN_S:
        wait = int(OTP_RESEND_COOLDOWN_S - (time.time() - existing["sent"]))
        return 429, {"success": False, "error": f"Please wait {wait}s before requesting a new code."}

    challenge = otp_service.create_challenge()
    _otp_challenges[key] = {
        "code_hash": _sha(challenge.code),  # the code itself is never kept
        "expires_at": challenge.expires_at,
        "attempts": 0,
        "sent": time.time(),
    }
    if otp_service.smtp_username and otp_service.smtp_password:
        purpose = "password" if key.startswith("pwchange:") else "verify"
        if not otp_service.send_verification_email(name, email, challenge.code, purpose):
            _otp_challenges.pop(key, None)
            return 502, {"success": False, "error": "Could not send the verification email."}
    else:
        logging.warning("SMTP not configured. Verification code for %s: %s", email, challenge.code)
    return 200, {"success": True, "expiresInMinutes": otp_service.expiry_minutes}


def auth_register(name: str, email: str, password: str):
    name = " ".join(name.split())[:60]
    if not name:
        return 400, {"success": False, "error": "Please enter your name."}
    if not _valid_email(email):
        return 400, {"success": False, "error": "Please enter a valid email address."}
    if not 8 <= len(password) <= 128:
        return 400, {"success": False, "error": "Password must be 8-128 characters."}
    existing = db.get_user_by_email(email)
    if existing and existing["password_hash"]:
        # Already registered: verify ownership with a code and sign in. The existing
        # name and password are left untouched, and the response is identical to a new
        # sign-up, so this does not reveal whether an account exists.
        status, body = _send_code(email, existing["display_name"], enforce_cooldown=True)
        if status == 200:
            _pending[email] = {"kind": "login", "user_id": existing["id"]}
        return status, body
    status, body = _send_code(email, name, enforce_cooldown=True)
    if status == 200:
        _pending[email] = {"kind": "register", "name": name, "pw_hash": db.hash_password(password)}
    return status, body


def auth_login(email: str, password: str):
    now = time.time()
    fails = [t for t in _login_fails.get(email, []) if now - t < LOGIN_WINDOW_S]
    if len(fails) >= LOGIN_MAX_FAILS:
        return 429, {"success": False, "error": "Too many failed attempts. Try again in 15 minutes."}
    user = db.get_user_by_email(email) if _valid_email(email) else None
    stored = user["password_hash"] if user and user["password_hash"] else None
    ok = db.verify_password(password, stored or _DUMMY_HASH)
    if not (ok and stored):
        fails.append(now)
        _login_fails[email] = fails
        return 401, {"success": False, "error": "Incorrect email or password."}
    _login_fails.pop(email, None)
    db.touch_login(user["id"])
    token = secrets.token_urlsafe(32)
    db.create_session(user["id"], token)
    return 200, {"success": True, "token": token, "user": _public_user(db.get_user(user["id"]))}


def otp_resend(email: str):
    pending = _pending.get(email)
    if not pending:
        return 400, {"success": False, "error": "Start by signing in again."}
    name = pending.get("name") or (db.get_user(pending["user_id"]) or {}).get("display_name", "there")
    return _send_code(email, name, enforce_cooldown=True)


def _check_code(key: str, code: str):
    """Validate a one-time code. Returns None on success, else (status, body)."""
    entry = _otp_challenges.get(key)
    if not entry:
        return 400, {"success": False, "error": "No active code. Request a new one."}
    if datetime.now(timezone.utc) >= entry["expires_at"]:
        _otp_challenges.pop(key, None)
        return 400, {"success": False, "error": "This code has expired. Request a new one."}
    if entry["attempts"] >= OTP_MAX_ATTEMPTS:
        _otp_challenges.pop(key, None)
        return 429, {"success": False, "error": "Too many attempts. Request a new code."}
    entry["attempts"] += 1
    code = code.strip()
    if not (len(code) == 6 and code.isdecimal() and secrets.compare_digest(_sha(code), entry["code_hash"])):
        left = OTP_MAX_ATTEMPTS - entry["attempts"]
        return 401, {"success": False, "error": f"Incorrect code. {left} attempt(s) left."}
    _otp_challenges.pop(key, None)  # single use
    return None


def otp_verify(email: str, code: str):
    pending = _pending.get(email)
    if not pending:
        return 400, {"success": False, "error": "No active code. Sign in again."}
    failure = _check_code(email, code)
    if failure:
        if failure[0] == 429:
            _pending.pop(email, None)
        return failure
    _pending.pop(email, None)
    if pending["kind"] == "register":
        user = db.create_or_claim_account(email, pending["name"], pending["pw_hash"])
    else:
        user = db.get_user(pending["user_id"])
    db.touch_login(user["id"])
    token = secrets.token_urlsafe(32)
    db.create_session(user["id"], token)
    return 200, {"success": True, "token": token, "user": _public_user(db.get_user(user["id"]))}


def password_change_request(user: dict, current: str, new: str):
    if not user["password_hash"] or not db.verify_password(current, user["password_hash"]):
        return 401, {"success": False, "error": "Current password is incorrect."}
    if not 8 <= len(new) <= 128:
        return 400, {"success": False, "error": "New password must be 8-128 characters."}
    if db.verify_password(new, user["password_hash"]):
        return 400, {"success": False, "error": "New password must be different from the current one."}
    key = f"pwchange:{user['id']}"
    status, body = _send_code(user["email"], user["display_name"], enforce_cooldown=True, key=key)
    if status == 200:
        _pending[key] = {"pw_hash": db.hash_password(new)}
    return status, body


def password_change_confirm(user: dict, code: str, token: str):
    key = f"pwchange:{user['id']}"
    pending = _pending.get(key)
    if not pending:
        return 400, {"success": False, "error": "No password change in progress."}
    failure = _check_code(key, code)
    if failure:
        if failure[0] == 429:
            _pending.pop(key, None)
        return failure
    _pending.pop(key, None)
    db.set_password(user["id"], pending["pw_hash"], keep_token=token)
    return 200, {"success": True}


def _public_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "email": user["email"],
        "displayName": user["display_name"],
        "role": user["role"],
        "createdAt": user["created_at"],
        "lastLoginAt": user["last_login_at"],
    }


class InferenceHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header(
            "Access-Control-Allow-Headers", "Content-Type, Authorization, X-Requested-With"
        )

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def _token(self) -> str:
        header = self.headers.get("Authorization", "")
        return header[7:].strip() if header.startswith("Bearer ") else ""

    def _user(self):
        return db.get_user_by_token(self._token())

    def do_DELETE(self):
        user = self._user()
        if not user:
            return self._json(401, {"success": False, "error": "Not signed in."})
        if self.path.startswith("/api/analyses/") and self.path.rsplit("/", 1)[-1].isdigit():
            ok = db.delete_analysis(user["id"], int(self.path.rsplit("/", 1)[-1]))
            return self._json(200 if ok else 404, {"success": ok})
        self._json(404, {"success": False})

    def do_GET(self):
        if self.path in ("/api/me", "/api/analyses") or self.path.startswith("/api/analyses/"):
            user = self._user()
            if not user:
                return self._json(401, {"success": False, "error": "Not signed in."})
            if self.path == "/api/me":
                return self._json(200, {"success": True, "user": _public_user(user)})
            if self.path == "/api/analyses":
                return self._json(200, {"success": True, "analyses": db.list_analyses(user["id"])})
            tail = self.path.rsplit("/", 1)[-1]
            item = db.get_analysis(user["id"], int(tail)) if tail.isdigit() else None
            return self._json(200 if item else 404, {"success": bool(item), "analysis": item})

        if self.path in ("/api/health", "/health"):
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            resp = {
                "status": "online",
                "device": str(DEVICE),
                "models": ["best_cnn_unet.pth", "mask_rcnn_final.pth"],
            }
            self.wfile.write(json.dumps(resp).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, status: int, body: dict):
        self.send_response(status)
        self._send_cors_headers()
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(body).encode("utf-8"))

    def do_POST(self):
        if self.path == "/api/session":
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(min(length, 4096)).decode("utf-8"))
                user = db.get_user_by_token(str(data.get("token", "")))
            except Exception:
                user = None
            body = {"success": bool(user)}
            if user:
                body["user"] = _public_user(user)
            return self._json(200 if user else 401, body)

        if self.path == "/api/logout":
            if self._token():
                db.delete_session(self._token())
            return self._json(200, {"success": True})

        if self.path == "/api/me":
            user = self._user()
            if not user:
                return self._json(401, {"success": False, "error": "Not signed in."})
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(min(length, 4096)).decode("utf-8"))
                name = " ".join(str(data.get("displayName", "")).split())[:60]
            except Exception:
                name = ""
            if not name:
                return self._json(400, {"success": False, "error": "Display name is required."})
            updated = db.update_display_name(user["id"], name)
            return self._json(200, {"success": True, "user": _public_user(updated)})

        if self.path in ("/api/auth/register", "/api/auth/login", "/api/otp/request", "/api/otp/verify"):
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(min(length, 8192)).decode("utf-8"))
                email = str(data.get("email", "")).strip().lower()
                if self.path.endswith("register"):
                    status, body = auth_register(str(data.get("name", "")), email, str(data.get("password", "")))
                elif self.path.endswith("login"):
                    status, body = auth_login(email, str(data.get("password", "")))
                elif self.path.endswith("request"):
                    status, body = otp_resend(email)
                else:
                    status, body = otp_verify(email, str(data.get("code", "")))
            except Exception:
                logging.exception("Auth request failed")
                status, body = 400, {"success": False, "error": "Invalid request."}
            return self._json(status, body)

        if self.path in ("/api/me/password/request", "/api/me/password/confirm"):
            user = self._user()
            if not user:
                return self._json(401, {"success": False, "error": "Not signed in."})
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(min(length, 8192)).decode("utf-8"))
                if self.path.endswith("request"):
                    status, body = password_change_request(
                        user, str(data.get("currentPassword", "")), str(data.get("newPassword", ""))
                    )
                else:
                    status, body = password_change_confirm(user, str(data.get("code", "")), self._token())
            except Exception:
                logging.exception("Password change failed")
                status, body = 400, {"success": False, "error": "Invalid request."}
            return self._json(status, body)

        if self.path in ("/api/predict", "/predict"):
            start_time = time.time()
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)

            try:
                min_conf = 0.50
                payload: dict = {}
                try:
                    payload = json.loads(body.decode("utf-8"))
                    raw_data = payload.get("image", "")
                    if "min_confidence" in payload:
                        min_conf = float(payload["min_confidence"])
                    if "," in raw_data:
                        raw_data = raw_data.split(",", 1)[1]
                    image_bytes = base64.b64decode(raw_data)
                except Exception:
                    image_bytes = body

                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                w, h = pil_image.size

                # Run TRUE model inference with small, refined percentage tags
                extras: dict = {}
                cnn_url, panel_pixels, dims = run_cnn_inference(pil_image, extras)
                rcnn_url, instances = run_rcnn_inference(pil_image, min_conf=min_conf, extras=extras)
                combined_url = render_combined(
                    pil_image, extras.get("cnn_mask"), extras.get("rcnn_instances", [])
                )

                total_pixels = w * h
                coverage_pct = (
                    round((panel_pixels / total_pixels) * 100, 1)
                    if total_pixels > 0
                    else 0.0
                )
                total_area_m2 = round(panel_pixels * 0.09, 1)
                est_capacity_kw = round(total_area_m2 * 0.2, 1)

                avg_conf = (
                    round(
                        sum(inst["confidence"] for inst in instances)
                        / len(instances)
                        * 100,
                        1,
                    )
                    if instances
                    else 0.0
                )

                elapsed_ms = int((time.time() - start_time) * 1000)

                response_data = {
                    "success": True,
                    "model_source": "pytorch_trained_weights",
                    "device": str(DEVICE),
                    "cnnResultUrl": cnn_url,
                    "rcnnResultUrl": rcnn_url,
                    "combinedResultUrl": combined_url,
                    "metrics": {
                        "panelCount": len(instances),
                        "totalAreaM2": total_area_m2,
                        "estimatedCapacityKw": est_capacity_kw,
                        "avgConfidence": avg_conf,
                        "inferenceTimeMs": elapsed_ms,
                        "coveragePercent": coverage_pct,
                        "instances": instances,
                    },
                    "imageDimensions": {"width": w, "height": h},
                }

                user = self._user()
                if user:
                    try:
                        response_data["analysisId"] = db.save_analysis(
                            user["id"],
                            str(payload.get("filename", "image")),
                            response_data["model_source"],
                            response_data["imageDimensions"],
                            response_data["metrics"],
                        )
                    except Exception:
                        logging.exception("Could not save analysis")

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response_data).encode("utf-8"))

            except Exception as e:
                logging.exception("Error during inference")
                self.send_response(500)
                self._send_cors_headers()
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(
                    json.dumps({"success": False, "error": str(e)}).encode("utf-8")
                )
        else:
            self.send_response(404)
            self.end_headers()


def start_server(port=8000):
    db.init()
    init_models()
    server_address = ("", port)
    httpd = HTTPServer(server_address, InferenceHandler)
    logging.info(f"Solar Sensing PyTorch Inference API running on http://localhost:{port}")
    httpd.serve_forever()


if __name__ == "__main__":
    start_server(8000)
