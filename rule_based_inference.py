#!/usr/bin/env python3
"""
Rule-based drowsiness inference
===============================
Standalone live detector using only:
  - EAR
  - MAR
  - head pose
  - PERCLOS

No Keras/TFLite drowsiness model is loaded. MediaPipe is used only to obtain
face landmarks for the measurements.

Usage
-----
  python rule_based_inference.py
  python rule_based_inference.py --source path/to/video.mp4
  python rule_based_inference.py --ear-threshold 0.20 --perclos-threshold 0.35
"""

import argparse
import collections
import time
from pathlib import Path

try:
    import cv2
    import mediapipe as mp
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. "
        "Activate your project virtualenv and install required packages."
    ) from exc

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ModuleNotFoundError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parent
MP_MODEL_PATH = BASE_DIR / "face_landmarker.task"

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_UPPER = [82, 13, 312]
MOUTH_LOWER = [87, 14, 317]
POSE_LM_IDS = [1, 152, 263, 33, 61, 291]

MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


def _dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return float(np.sqrt(dx * dx + dy * dy))


def compute_ear(lm, eye_idx):
    v1 = _dist(lm[eye_idx[1]], lm[eye_idx[5]])
    v2 = _dist(lm[eye_idx[2]], lm[eye_idx[4]])
    h = _dist(lm[eye_idx[0]], lm[eye_idx[3]])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def compute_mar(lm):
    h = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    if h < 1e-6:
        return 0.0
    v = sum(_dist(lm[u], lm[l]) for u, l in zip(MOUTH_UPPER, MOUTH_LOWER))
    return v / (3.0 * h)


def compute_head_pose(lm, width, height):
    pts = np.array([lm[i] for i in POSE_LM_IDS], dtype=np.float64)
    focal = float(width)
    camera = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D,
        pts,
        camera,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0
    rot, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(rot[2, 1], rot[2, 2]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))
    else:
        pitch = np.degrees(np.arctan2(-rot[1, 2], rot[1, 1]))
        yaw = np.degrees(np.arctan2(-rot[2, 0], sy))
        roll = 0.0
    return float(pitch), float(yaw), float(roll)


def extract_measurements(face_landmarks, width, height):
    lm = np.array([(p.x * width, p.y * height) for p in face_landmarks], dtype=np.float64)
    ear_l = compute_ear(lm, LEFT_EYE)
    ear_r = compute_ear(lm, RIGHT_EYE)
    ear = (ear_l + ear_r) / 2.0
    mar = compute_mar(lm)
    pitch, yaw, roll = compute_head_pose(lm, width, height)
    return {
        "ear_l": ear_l,
        "ear_r": ear_r,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


def camera_index(source):
    raw = str(source).strip()
    if Path(raw).exists():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class Picamera2Capture:
    def __init__(self, index=0, width=640, height=480, fps=30):
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError("Picamera2 is not installed.")
        self.fps = float(fps) if fps else 30.0
        self.loop_source = False
        self.camera = Picamera2(camera_num=int(index))
        cfg = self.camera.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"},
        )
        self.camera.configure(cfg)
        try:
            self.camera.set_controls({"FrameRate": self.fps})
        except Exception:
            pass
        self.camera.start()

    def read(self):
        frame = self.camera.capture_array()
        if frame is None:
            return False, None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return True, np.ascontiguousarray(frame)

    def reset(self):
        return None

    def release(self):
        try:
            self.camera.stop()
        finally:
            self.camera.close()


class OpenCVVideoCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open source: {source}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.loop_source = not isinstance(source, int)

    def read(self):
        return self.cap.read()

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self.cap.release()


def open_source(source, width, height, fps):
    idx = camera_index(source)
    if idx is not None:
        return Picamera2Capture(idx, width, height, fps)
    return OpenCVVideoCapture(str(source))


def rotate_frame(frame, rotation):
    rotation = int(rotation) % 360
    if rotation == 0:
        return frame
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotation}")


class RuleState:
    def __init__(self, args, fps):
        self.args = args
        self.fps = max(1.0, float(fps))
        perclos_len = max(1, int(args.perclos_window * self.fps))
        self.closed_window = collections.deque(maxlen=perclos_len)
        self.closed_frames = 0
        self.yawn_frames = 0
        self.pose_frames = 0
        self.missing_frames = 0
        self.label = 0
        self.reason = "OK"

    def reset_face_state(self):
        self.closed_frames = 0
        self.yawn_frames = 0
        self.pose_frames = 0
        self.label = 0
        self.reason = "NO FACE"

    def update(self, measurements, face_ok):
        if not face_ok:
            self.missing_frames += 1
            if self.missing_frames >= int(self.args.no_face_reset * self.fps):
                self.closed_window.clear()
                self.reset_face_state()
            return {
                "label": 0,
                "reason": "NO FACE",
                "perclos": 0.0,
                "closed": False,
                "yawn": False,
                "bad_pose": False,
            }

        self.missing_frames = 0
        ear = measurements["ear"]
        mar = measurements["mar"]
        pitch = measurements["pitch"]
        yaw = measurements["yaw"]
        roll = measurements["roll"]

        closed = ear < self.args.ear_threshold
        yawn = mar > self.args.mar_threshold
        bad_pose = (
            abs(pitch) > self.args.pitch_threshold or
            abs(yaw) > self.args.yaw_threshold or
            abs(roll) > self.args.roll_threshold
        )

        self.closed_window.append(1 if closed else 0)
        perclos = float(np.mean(self.closed_window)) if self.closed_window else 0.0
        self.closed_frames = self.closed_frames + 1 if closed else 0
        self.yawn_frames = self.yawn_frames + 1 if yawn else 0
        self.pose_frames = self.pose_frames + 1 if bad_pose else 0

        closed_long = self.closed_frames >= int(self.args.eye_closed_seconds * self.fps)
        yawn_long = self.yawn_frames >= int(self.args.yawn_seconds * self.fps)
        pose_long = self.pose_frames >= int(self.args.pose_seconds * self.fps)
        perclos_high = (
            len(self.closed_window) >= int(min(self.args.perclos_window, 3.0) * self.fps) and
            perclos >= self.args.perclos_threshold
        )

        reasons = []
        if closed_long:
            reasons.append("EYES CLOSED")
        if perclos_high:
            reasons.append("HIGH PERCLOS")
        if yawn_long:
            reasons.append("YAWN")
        if pose_long:
            reasons.append("HEAD POSE")

        self.label = 1 if reasons else 0
        self.reason = " + ".join(reasons) if reasons else "OK"
        return {
            "label": self.label,
            "reason": self.reason,
            "perclos": perclos,
            "closed": closed,
            "yawn": yawn,
            "bad_pose": bad_pose,
        }


def create_landmarker():
    if not MP_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing MediaPipe model: {MP_MODEL_PATH}")
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MP_MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def draw_overlay(frame, measurements, state, rule, fps_display, frame_count):
    height, width = frame.shape[:2]
    label = rule["label"]
    face_ok = measurements is not None
    status = rule["reason"] if face_ok else "NO FACE"
    color = (0, 0, 220) if label else ((0, 170, 255) if not face_ok else (0, 170, 0))

    cv2.rectangle(frame, (0, 0), (width, 122), (15, 15, 15), -1)
    cv2.putText(frame, status, (14, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"FPS {fps_display:.0f}  Frame {frame_count}",
        (width - 210, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    if measurements is None:
        cv2.putText(
            frame,
            "Waiting for face",
            (14, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 170, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    lines = [
        f"EAR {measurements['ear']:.3f} < {state.args.ear_threshold:.3f}   "
        f"PERCLOS {rule['perclos']:.2f} > {state.args.perclos_threshold:.2f}",
        f"MAR {measurements['mar']:.3f} > {state.args.mar_threshold:.3f}   "
        f"Pose P/Y/R {measurements['pitch']:+.1f}/{measurements['yaw']:+.1f}/{measurements['roll']:+.1f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (14, 68 + 28 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    bar_x, bar_y = 14, 108
    bar_w = max(120, width - 28)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (70, 70, 70), -1)
    cv2.rectangle(
        frame,
        (bar_x, bar_y),
        (bar_x + int(bar_w * min(1.0, rule["perclos"])), bar_y + 8),
        color,
        -1,
    )
    return frame


def run(args):
    cap = open_source(args.source, args.camera_width, args.camera_height, args.camera_fps)
    landmarker = create_landmarker()
    state = RuleState(args, cap.fps)

    ts_ms = 0.0
    frame_count = 0
    t_prev = time.perf_counter()
    fps_display = 0.0
    last_measurements = None

    print("Rule-based drowsiness detector")
    print(f"  EAR threshold:      {args.ear_threshold:.3f}")
    print(f"  MAR threshold:      {args.mar_threshold:.3f}")
    print(f"  PERCLOS threshold:  {args.perclos_threshold:.2f} over {args.perclos_window:.1f}s")
    print("Press Q or Esc to quit.\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if cap.loop_source:
                    cap.reset()
                    continue
                break

            frame = rotate_frame(frame, args.rotation)
            height, width = frame.shape[:2]
            frame_count += 1

            now = time.perf_counter()
            fps_display = 0.9 * fps_display + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(image, int(ts_ms))
            ts_ms += 1000.0 / max(cap.fps, 1.0)

            face_ok = bool(result.face_landmarks)
            if face_ok:
                last_measurements = extract_measurements(result.face_landmarks[0], width, height)
                rule = state.update(last_measurements, True)
            else:
                last_measurements = None
                rule = state.update(None, False)

            draw_overlay(frame, last_measurements, state, rule, fps_display, frame_count)
            cv2.imshow("Rule-Based Drowsiness Detector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        landmarker.close()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based drowsiness detector")
    parser.add_argument("--source", default="0", help="Picamera2 index or video file path")
    parser.add_argument("--camera-width", type=int, default=640, help="Picamera2 capture width")
    parser.add_argument("--camera-height", type=int, default=480, help="Picamera2 capture height")
    parser.add_argument("--camera-fps", type=int, default=30, help="Picamera2 target FPS")
    parser.add_argument(
        "--rotation",
        type=int,
        default=180,
        choices=[0, 90, 180, 270],
        help="Rotate frames before feature extraction and display",
    )
    parser.add_argument("--ear-threshold", type=float, default=0.20, help="EAR below this means eyes closed")
    parser.add_argument("--mar-threshold", type=float, default=0.06, help="MAR above this means yawning")
    parser.add_argument("--perclos-threshold", type=float, default=0.35, help="Closed-eye fraction threshold")
    parser.add_argument("--perclos-window", type=float, default=20.0, help="Rolling PERCLOS window in seconds")
    parser.add_argument("--eye-closed-seconds", type=float, default=1.5, help="Continuous eye closure trigger")
    parser.add_argument("--yawn-seconds", type=float, default=1.0, help="Continuous yawn trigger")
    parser.add_argument("--pose-seconds", type=float, default=1.2, help="Continuous bad head-pose trigger")
    parser.add_argument("--pitch-threshold", type=float, default=28.0, help="Absolute pitch threshold in degrees")
    parser.add_argument("--yaw-threshold", type=float, default=38.0, help="Absolute yaw threshold in degrees")
    parser.add_argument("--roll-threshold", type=float, default=35.0, help="Absolute roll threshold in degrees")
    parser.add_argument("--no-face-reset", type=float, default=0.5, help="Seconds without face before clearing state")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
