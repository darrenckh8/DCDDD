"""
Live drowsiness inference
=========================

Runs the trained model against a webcam or video file in real time.
Mirrors the exact preprocessing pipeline used in trainer.py.

Usage
-----
  # Webcam (default)
  python live_inference.py

  # Video file
  python live_inference.py --source path/to/video.mp4

  # Use TFLite instead of Keras
  python live_inference.py --model drowsiness_mv3_lstm_f16.tflite

  # Calibrate norm params from a short alert clip first
  python live_inference.py --alert-clip path/to/alert.mp4

Files expected in the same directory as this script
----------------------------------------------------
  drowsiness_mv3_lstm.keras   (or a .tflite variant)
  deploy_config.json
  norm_global_params.json
  face_landmarker.task
"""

import argparse
import collections
import json
import os
import sys
import time

from pathlib import Path

try:
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    import cv2
    import mediapipe as mp
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. "
        "Activate your project virtualenv and install required packages."
    ) from exc

# ─────────────────────────── Paths ───────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL   = BASE_DIR / "drowsiness_mv3_lstm.keras"
DEPLOY_CFG_PATH = BASE_DIR / "deploy_config.json"
GLOBAL_NORM_PATH = BASE_DIR / "norm_global_params.json"
MP_MODEL_PATH   = str(BASE_DIR / "face_landmarker.task")

FEATURE_NAMES = [
    "EAR_Left", "EAR_Right", "EAR_Avg", "MAR",
    "PUC_Left", "PUC_Right", "MUC", "Pitch", "Yaw", "Roll",
]
NUM_RAW_FEATURES = len(FEATURE_NAMES)

# Matches extractor.py's default MAX_GAP_FILL. Real-time inference cannot use
# future frames for interpolation, so this is a causal forward-fill equivalent.
DEFAULT_MAX_GAP_FILL = 15

# trainer.py evaluates/tunes thresholds on windows spaced by EVAL_STEP.
DEFAULT_PREDICT_STRIDE = 15

# ─────────────────────────── MediaPipe landmarks ─────────────────────────────

RIGHT_EYE       = [33, 160, 158, 133, 153, 144]
LEFT_EYE        = [263, 387, 385, 362, 380, 373]
RIGHT_EYE_UC, RIGHT_EYE_LC = 159, 145
LEFT_EYE_UC,  LEFT_EYE_LC  = 386, 374
MOUTH_LEFT,   MOUTH_RIGHT   = 61, 291
MOUTH_UPPER   = [82, 13, 312]
MOUTH_LOWER   = [87, 14, 317]
POSE_LM_IDS   = [1, 152, 263, 33, 61, 291]
MODEL_3D = np.array([
    ( 0.0,    0.0,    0.0),
    ( 0.0,  -63.6,  -12.5),
    (-43.3,  32.7,  -26.0),
    ( 43.3,  32.7,  -26.0),
    (-28.9, -28.9,  -24.1),
    ( 28.9, -28.9,  -24.1),
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

# ─────────────────────────── Feature extraction ──────────────────────────────

def _dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return np.sqrt(dx * dx + dy * dy)

def compute_ear(lm, idx):
    v1 = _dist(lm[idx[1]], lm[idx[5]])
    v2 = _dist(lm[idx[2]], lm[idx[4]])
    h  = _dist(lm[idx[0]], lm[idx[3]])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0

def compute_puc(lm, uc, lc, ca, cb):
    v = _dist(lm[uc], lm[lc])
    h = _dist(lm[ca], lm[cb])
    return v / h if h > 1e-6 else 0.0

def compute_mar(lm):
    h = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    if h < 1e-6:
        return 0.0
    v = sum(_dist(lm[u], lm[l]) for u, l in zip(MOUTH_UPPER, MOUTH_LOWER))
    return v / (3.0 * h)

def compute_muc(lm):
    v = _dist(lm[13], lm[14])
    h = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    return v / h if h > 1e-6 else 0.0

def compute_head_pose(lm, W, H):
    pts = np.array([lm[i] for i in POSE_LM_IDS], dtype=np.float64)
    focal = float(W)
    cam   = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]], dtype=np.float64)
    ok, rvec, _ = cv2.solvePnP(MODEL_3D, pts, cam, DIST_COEFFS,
                                flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2( R[2,1], R[2,2]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = np.degrees(np.arctan2( R[1,0], R[0,0]))
    else:
        pitch = np.degrees(np.arctan2(-R[1,2], R[1,1]))
        yaw   = np.degrees(np.arctan2(-R[2,0], sy))
        roll  = 0.0
    return pitch, yaw, roll

def extract_features(fl, W, H):
    """Extract 10 raw features from a FaceLandmarker result."""
    lm = np.array([(p.x * W, p.y * H) for p in fl], dtype=np.float64)
    ear_l = compute_ear(lm, LEFT_EYE)
    ear_r = compute_ear(lm, RIGHT_EYE)
    puc_l = compute_puc(lm, LEFT_EYE_UC, LEFT_EYE_LC, LEFT_EYE[0], LEFT_EYE[3])
    puc_r = compute_puc(lm, RIGHT_EYE_UC, RIGHT_EYE_LC, RIGHT_EYE[0], RIGHT_EYE[3])
    mar   = compute_mar(lm)
    muc   = compute_muc(lm)
    pitch, yaw, roll = compute_head_pose(lm, W, H)
    return np.array([ear_l, ear_r, (ear_l + ear_r) / 2.0,
                     mar, puc_l, puc_r, muc,
                     pitch, yaw, roll], dtype=np.float32)


class LiveFeatureState:
    """Causal live equivalent of extractor gap handling and angle unwrapping."""

    def __init__(self, max_gap_fill=DEFAULT_MAX_GAP_FILL):
        self.max_gap_fill = max(0, int(max_gap_fill))
        self.last_raw = None
        self.prev_angles = None
        self.gap_count = 0

    def _unwrap_angles(self, angles):
        angles = angles.astype(np.float32, copy=True)
        if self.prev_angles is None:
            unwrapped = angles
        else:
            unwrapped = angles + 360.0 * np.round((self.prev_angles - angles) / 360.0)
        self.prev_angles = unwrapped.astype(np.float32)
        return self.prev_angles

    def update(self, detected_raw):
        """Return a raw 10-feature frame, or None until the first face appears."""
        if detected_raw is not None:
            raw = detected_raw.astype(np.float32, copy=True)
            raw[7:10] = self._unwrap_angles(raw[7:10])
            self.last_raw = raw
            self.gap_count = 0
            return raw

        if self.last_raw is None:
            return None

        self.gap_count += 1
        if self.gap_count <= self.max_gap_fill:
            return self.last_raw.copy()

        # extractor.py leaves long gaps as zeros after limited filling. Reset
        # angle continuity so a new face after a long gap starts a fresh track.
        self.last_raw = np.zeros(NUM_RAW_FEATURES, dtype=np.float32)
        self.prev_angles = np.zeros(3, dtype=np.float32)
        return self.last_raw.copy()

# ─────────────────────────── Enhanced features (mirrors trainer.py) ──────────

def apply_enhanced_features(X_raw, rolling_window, perclos_threshold):
    """(1, T, 10) → (1, T, 33): raw + Δ + ΔΔ + [EAR-var, head-move, PERCLOS]."""
    N, T, _ = X_raw.shape
    X_proc  = X_raw.astype(np.float32, copy=True)

    delta  = np.zeros_like(X_proc)
    delta[:, 1:] = X_proc[:, 1:] - X_proc[:, :-1]
    ddelta = np.zeros_like(delta)
    ddelta[:, 1:] = delta[:, 1:] - delta[:, :-1]

    ear  = X_proc[:, :, 2]
    pd_  = delta[:, :, 7]

    ear_pad = np.pad(ear, ((0, 0), (rolling_window - 1, 0)), mode="edge")
    ear_var = np.var(
        sliding_window_view(ear_pad, rolling_window, axis=1), axis=2
    ).astype(np.float32)

    t_idx  = np.arange(T)
    starts = np.maximum(t_idx - rolling_window + 1, 0)
    wlens  = (t_idx - starts + 1).astype(np.float32)
    cs     = np.cumsum(np.abs(pd_), axis=1)
    cs_pad = np.concatenate([np.zeros((N, 1), dtype=np.float32), cs], axis=1)
    head_move = ((cs_pad[:, t_idx + 1] - cs_pad[:, starts]) / wlens).astype(np.float32)

    closed = (ear < perclos_threshold).astype(np.float32)
    c_pad  = np.pad(closed, ((0, 0), (rolling_window - 1, 0)), mode="edge")
    perclos = np.mean(
        sliding_window_view(c_pad, rolling_window, axis=1), axis=2
    ).astype(np.float32)

    temporal = np.stack([ear_var, head_move, perclos], axis=-1)
    return np.concatenate([X_proc, delta, ddelta, temporal], axis=-1)

# ─────────────────────────── Temporal smoothing ──────────────────────────────

def smooth_predict(probs, threshold, K):
    binary   = (probs >= threshold).astype(np.int32)
    if K <= 1:
        return binary
    smoothed = np.zeros_like(binary)
    for i in range(K - 1, len(binary)):
        window      = binary[i - K + 1 : i + 1]
        smoothed[i] = 1 if window.sum() == K else 0
    return smoothed

# ─────────────────────────── Normalisation ───────────────────────────────────

def normalise_window(raw_window, norm_params):
    """Apply per-feature median/IQR normalisation to a (T, 10) array."""
    out = raw_window.copy()
    for i, feat in enumerate(FEATURE_NAMES):
        med = norm_params[feat]["median"]
        iqr = norm_params[feat]["iqr"]
        if abs(iqr) < 1e-6:
            iqr = 1.0
        out[:, i] = (out[:, i] - med) / iqr
    return out

def normalise_frame(raw_frame, norm_params):
    """Apply per-feature median/IQR normalisation to a single 10-feature frame."""
    return normalise_window(raw_frame[np.newaxis, :], norm_params)[0]

def compute_norm_params_from_clip(clip_path, landmarker):
    """Compute per-subject norm params from a short alert clip."""
    print(f"Calibrating norm params from: {clip_path}")
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open alert clip: {clip_path}")

    raw_features = []
    feature_state = LiveFeatureState()
    ts_ms = 0.0
    fps   = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(img, int(ts_ms))
        detected = extract_features(result.face_landmarks[0], W, H) if result.face_landmarks else None
        raw = feature_state.update(detected)
        if raw is not None:
            raw_features.append(raw)
        ts_ms += 1000.0 / fps

    cap.release()
    if len(raw_features) < 30:
        raise ValueError(f"Alert clip too short: only {len(raw_features)} frames detected")

    arr = np.array(raw_features)
    params = {}
    for i, feat in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        med = float(np.median(col))
        iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
        if iqr < 1e-6:
            iqr = 1.0
        params[feat] = {"median": med, "iqr": iqr}
    print(f"  Calibrated on {len(raw_features)} alert frames.")
    return params

# ─────────────────────────── Model loading ───────────────────────────────────

def load_model(model_path):
    path = str(model_path)
    if path.endswith(".tflite"):
        import tensorflow as tf
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()
        out_det = interp.get_output_details()
        print(f"Loaded TFLite model: {path}")
        return ("tflite", interp, inp_det, out_det)
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(path, compile=False)
        print(f"Loaded Keras model: {path}")
        return ("keras", model)

def run_model(model_bundle, X):
    """X: (1, SEQ_LEN, 33) float32 → probability float"""
    kind = model_bundle[0]
    if kind == "keras":
        _, model = model_bundle
        return float(model.predict(X, verbose=0).flatten()[0])
    else:
        _, interp, inp_det, out_det = model_bundle
        inp_dtype = inp_det[0]["dtype"]
        if inp_dtype == np.int8:
            scale, zero = inp_det[0]["quantization"]
            if scale <= 0:
                raise ValueError("Invalid int8 quantization scale for TFLite input.")
            X_in = np.clip(np.round(X / scale + zero), -128, 127).astype(np.int8)
        else:
            X_in = X.astype(inp_dtype)
        interp.set_tensor(inp_det[0]["index"], X_in)
        interp.invoke()
        out = interp.get_tensor(out_det[0]["index"]).flatten()
        if out_det[0]["dtype"] == np.int8:
            scale, zero = out_det[0]["quantization"]
            return float((out[0] - zero) * scale)
        return float(out[0])

# ─────────────────────────── Display ─────────────────────────────────────────

def draw_overlay(frame, prob, label, threshold, frame_count, fps_display):
    H, W = frame.shape[:2]

    # Colour: green=alert, red=drowsy, grey=warming up
    if label is None:
        colour = (120, 120, 120)
        status = "Warming up..."
    elif label == 1:
        colour = (0, 0, 220)
        status = "DROWSY"
    else:
        colour = (0, 180, 0)
        status = "ALERT"

    # Status banner
    cv2.rectangle(frame, (0, 0), (W, 56), (20, 20, 20), -1)
    cv2.putText(frame, status, (12, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, colour, 2, cv2.LINE_AA)

    # Probability bar
    if prob is not None:
        bar_w = int((W - 24) * prob)
        cv2.rectangle(frame, (12, 62), (W - 12, 82), (50, 50, 50), -1)
        cv2.rectangle(frame, (12, 62), (12 + bar_w, 82), colour, -1)
        thr_x = 12 + int((W - 24) * threshold)
        cv2.line(frame, (thr_x, 58), (thr_x, 86), (255, 255, 255), 2)
        cv2.putText(frame, f"p={prob:.2f}  thr={threshold:.2f}", (12, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS
    cv2.putText(frame, f"{fps_display:.0f} fps  frame {frame_count}",
                (W - 200, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return frame

# ─────────────────────────── Main inference loop ─────────────────────────────

def run_inference(args):
    # ── Load config ──────────────────────────────────────────────────────────
    with open(DEPLOY_CFG_PATH) as f:
        cfg = json.load(f)

    cfg_features = cfg.get("feature_names")
    if cfg_features is not None and cfg_features != FEATURE_NAMES:
        sys.exit(
            "deploy_config.json feature_names do not match live_inference.py: "
            f"{cfg_features} != {FEATURE_NAMES}"
        )

    threshold   = float(cfg["threshold"])
    smooth_k    = max(1, int(cfg["smooth_k"]))
    seq_len     = int(cfg["sequence_length"])
    rolling_win = int(cfg["rolling_window"])
    perclos_thr = float(cfg["perclos_ear_iqr_threshold"])
    predict_stride = args.predict_stride
    if predict_stride is None:
        predict_stride = int(cfg.get("eval_step", DEFAULT_PREDICT_STRIDE))
    predict_stride = max(1, int(predict_stride))
    max_gap_fill = int(cfg.get("max_gap_fill", DEFAULT_MAX_GAP_FILL))
    temporal_pad = rolling_win - 1
    padded_len   = seq_len + temporal_pad

    # ── Load norm params ─────────────────────────────────────────────────────
    with open(GLOBAL_NORM_PATH) as f:
        global_norm = json.load(f)

    # ── MediaPipe landmarker ─────────────────────────────────────────────────
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MP_MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ── Per-subject calibration (optional) ───────────────────────────────────
    norm_params = global_norm
    if args.alert_clip:
        norm_params = compute_norm_params_from_clip(args.alert_clip, landmarker)
        landmarker.close()
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    # ── Load model ───────────────────────────────────────────────────────────
    model_bundle = load_model(args.model)

    # ── Open video source ────────────────────────────────────────────────────
    source_raw = str(args.source).strip()
    source = source_raw
    if not Path(source_raw).exists():
        try:
            source = int(source_raw)
        except ValueError:
            source = source_raw
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        sys.exit(f"Cannot open source: {args.source}")

    cam_fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    print(f"\nRunning at ~{cam_fps:.0f} fps  |  "
          f"seq_len={seq_len}  |  stride={predict_stride}  |  "
          f"threshold={threshold:.3f}  |  K={smooth_k}")
    print("Press Q to quit.\n")

    # ── Buffers ───────────────────────────────────────────────────────────────
    # ring_buf: stores the last padded_len normalised frames
    ring_buf    = collections.deque(maxlen=padded_len)
    # prob_buf: stores the last smooth_k predictions for smoothing
    prob_buf    = collections.deque(maxlen=smooth_k)
    # state used to fill short gaps and unwrap head-pose angles
    feature_state = LiveFeatureState(max_gap_fill=max_gap_fill)

    frame_count = 0
    feature_count = 0
    ts_ms       = 0.0
    label       = None
    prob        = None
    t_prev      = time.perf_counter()
    fps_display = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        frame_count += 1

        # FPS measurement
        now = time.perf_counter()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(now - t_prev, 1e-6))
        t_prev = now

        # ── MediaPipe detection ───────────────────────────────────────────────
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect_for_video(img, int(ts_ms))
        ts_ms += 1000.0 / cam_fps

        detected = extract_features(result.face_landmarks[0], W, H) if result.face_landmarks else None
        raw = feature_state.update(detected)

        # ── Normalise and buffer ──────────────────────────────────────────────
        if raw is not None:
            ring_buf.append(normalise_frame(raw, norm_params))
            feature_count += 1

        # ── Run model once we have enough frames ──────────────────────────────
        should_predict = (
            len(ring_buf) >= seq_len and
            (feature_count - seq_len) % predict_stride == 0
        )
        if should_predict:
            window = np.array(ring_buf, dtype=np.float32)   # (<= padded_len, 10)
            if len(window) < padded_len:
                pad_n = padded_len - len(window)
                window = np.pad(window, ((pad_n, 0), (0, 0)), mode="edge")

            # Enhanced features and trim temporal pad
            X_enh = apply_enhanced_features(
                window[np.newaxis],   # (1, padded_len, 10)
                rolling_win,
                perclos_thr,
            )
            X_enh = X_enh[:, temporal_pad:, :]   # (1, seq_len, 33)

            prob = run_model(model_bundle, X_enh.astype(np.float32))
            prob_buf.append(prob)

            # Temporal smoothing: need K consecutive windows
            preds = smooth_predict(np.array(list(prob_buf)), threshold, smooth_k)
            label = int(preds[-1])

        # ── Draw and show ─────────────────────────────────────────────────────
        frame = draw_overlay(frame, prob, label, threshold, frame_count, fps_display)
        cv2.imshow("Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q'), 27):
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()
    print("Done.")

# ─────────────────────────── CLI ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Live drowsiness inference")
    p.add_argument("--source",      default="0",
                   help="Webcam index (0) or path to video file")
    p.add_argument("--model",       default=str(DEFAULT_MODEL),
                   help="Path to .keras or .tflite model")
    p.add_argument("--alert-clip",  default=None,
                   help="Path to a short alert video for per-subject calibration")
    p.add_argument("--predict-stride", type=int, default=None,
                   help=("Run the model every N accepted frames. Defaults to "
                         "deploy_config eval_step, or 15 to match trainer.py."))
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not Path(DEPLOY_CFG_PATH).exists():
        sys.exit(f"Missing deploy_config.json at {DEPLOY_CFG_PATH}")
    if not Path(GLOBAL_NORM_PATH).exists():
        sys.exit(f"Missing norm_global_params.json at {GLOBAL_NORM_PATH}")
    if not Path(MP_MODEL_PATH).exists():
        sys.exit(f"Missing face_landmarker.task at {MP_MODEL_PATH}")
    if not Path(args.model).exists():
        sys.exit(f"Model not found: {args.model}")

    run_inference(args)
