#!/usr/bin/env python3
"""
DrowsGuard — Advanced Dashcam HUD
===================================
Full-screen touchscreen UI for the drowsiness detection system.
Designed for Raspberry Pi 5 (Bookworm) + capacitive display (1024×600).

Install
-------
  sudo apt install python3-pyqt5 python3-picamera2
  # or: pip install PyQt5 --break-system-packages

Usage
-----
  python ui.py
  python ui.py --source 0 --model drowsiness_mv3_lstm.keras
  python ui.py --no-fullscreen   # for dev on desktop
"""

import argparse
import collections
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path

try:
    import cv2
    import numpy as np
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. "
        "Activate your project virtualenv and install required packages."
    ) from exc

for _qt_env_name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_PLUGIN_PATH"):
    if "cv2" in os.environ.get(_qt_env_name, "").lower():
        os.environ.pop(_qt_env_name, None)

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ModuleNotFoundError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False

try:
    from PyQt5.QtCore import (
        Qt, QThread, QTimer, pyqtSignal, QObject, QRect, QLibraryInfo
    )
    from PyQt5.QtGui import (
        QImage, QPixmap, QPainter, QColor, QFont
    )
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
        QSizePolicy, QProgressBar, QListWidget, QListWidgetItem
    )
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. "
        "Install PyQt5 or activate your project virtualenv."
    ) from exc

# ── Try to import inference module ────────────────────────────────────────────
try:
    from live_inference import (
        extract_features, apply_enhanced_features,
        compute_norm_params_from_clip,
        load_model, run_model, smooth_predict,
        normalise_frame, LiveFeatureState,
        BASE_DIR, DEPLOY_CFG_PATH, GLOBAL_NORM_PATH, MP_MODEL_PATH,
        DEFAULT_MODEL, FEATURE_NAMES, DEFAULT_PREDICT_STRIDE,
        DEFAULT_MAX_GAP_FILL,
    )
    import mediapipe as mp
    INFERENCE_AVAILABLE = True
except (ImportError, SystemExit) as _e:
    warnings.warn(f"live_inference not importable ({_e}); running in DEMO mode.")
    INFERENCE_AVAILABLE = False
    BASE_DIR      = Path(__file__).resolve().parent
    DEFAULT_MODEL = BASE_DIR / "drowsiness_mv3_lstm.keras"
    DEPLOY_CFG_PATH  = BASE_DIR / "deploy_config.json"
    GLOBAL_NORM_PATH = BASE_DIR / "norm_global_params.json"
    MP_MODEL_PATH    = str(BASE_DIR / "face_landmarker.task")
    FEATURE_NAMES = [
        "EAR_Left", "EAR_Right", "EAR_Avg", "MAR",
        "PUC_Left", "PUC_Right", "MUC", "Pitch", "Yaw", "Roll",
    ]
    DEFAULT_PREDICT_STRIDE = 15
    DEFAULT_MAX_GAP_FILL = 15


# ═══════════════════════════════════════════════════════════════════════════════
#  PALETTE
# ═══════════════════════════════════════════════════════════════════════════════

P = {
    "bg":         "#202124",
    "bg2":        "#262a2e",
    "bg3":        "#30353a",
    "surface":    "#2b3035",
    "border":     "#3d444b",
    "border_hi":  "#555e66",
    "green":      "#2e7d32",
    "green_dim":  "#1f5f25",
    "green_mid":  "#43a047",
    "red":        "#c62828",
    "red_dim":    "#8e1b1b",
    "amber":      "#f9a825",
    "amber_dim":  "#8a6518",
    "cyan":       "#1976d2",
    "white":      "#f1f3f4",
    "dim":        "#a8b0b8",
    "dimmer":     "#7b848d",
    "scanline":   "#17191c",
}

def configure_qt_environment(qt_platform=None):
    """Keep OpenCV's bundled Qt plugins from hijacking PyQt startup."""
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR", "QT_PLUGIN_PATH"):
        if "cv2" in os.environ.get(name, "").lower():
            os.environ.pop(name, None)

    plugin_path = QLibraryInfo.location(QLibraryInfo.PluginsPath)
    if plugin_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = plugin_path

    if qt_platform:
        os.environ["QT_QPA_PLATFORM"] = qt_platform
    elif "QT_QPA_PLATFORM" not in os.environ:
        if os.environ.get("WAYLAND_DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "wayland"
        elif os.environ.get("DISPLAY"):
            os.environ["QT_QPA_PLATFORM"] = "xcb"
        else:
            os.environ["QT_QPA_PLATFORM"] = "linuxfb"


# ═══════════════════════════════════════════════════════════════════════════════
#  FRAME SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

def _camera_index(source):
    raw = str(source).strip()
    if Path(raw).exists():
        return None
    try:
        return int(raw)
    except ValueError:
        return None


class Picamera2Capture:
    """BGR frame source backed by Picamera2 for Raspberry Pi cameras."""

    def __init__(self, camera_index=0, width=640, height=480, fps=30):
        if not PICAMERA2_AVAILABLE:
            raise RuntimeError(
                "Picamera2 is not installed. On Raspberry Pi OS Bookworm, run: "
                "sudo apt install python3-picamera2"
            )
        self.fps = float(fps) if fps else 30.0
        self.loop_source = False
        self._camera = Picamera2(camera_num=int(camera_index))
        config = self._camera.create_video_configuration(
            main={"size": (int(width), int(height)), "format": "RGB888"},
        )
        self._camera.configure(config)
        try:
            self._camera.set_controls({"FrameRate": self.fps})
        except Exception:
            pass
        self._camera.start()

    def read(self):
        frame = self._camera.capture_array()
        if frame is None:
            return False, None
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame

    def reset(self):
        return None

    def release(self):
        try:
            self._camera.stop()
        finally:
            self._camera.close()


class OpenCVVideoCapture:
    """BGR frame source backed by OpenCV for video files/devices."""

    def __init__(self, source):
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open: {source}")
        self.fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.loop_source = not isinstance(source, int)

    def read(self):
        return self._cap.read()

    def reset(self):
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def release(self):
        self._cap.release()


def open_frame_source(source, camera_width=640, camera_height=480, camera_fps=30):
    index = _camera_index(source)
    if index is not None:
        return Picamera2Capture(index, camera_width, camera_height, camera_fps)
    return OpenCVVideoCapture(str(source))


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceSignals(QObject):
    frame_ready = pyqtSignal(object, object)   # (frame_bgr ndarray, metrics dict)
    error       = pyqtSignal(str)


class DemoWorker(QThread):
    """Generates synthetic data when live_inference is unavailable."""
    def __init__(self, source="0", camera_width=640, camera_height=480, camera_fps=30, parent=None):
        super().__init__(parent)
        self.source = source
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.signals  = InferenceSignals()
        self._running = True
        self._paused  = False
        self.threshold = 0.5

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False; self.wait(2000)

    def run(self):
        t = 0.0
        fc = 0
        try:
            cap = open_frame_source(
                self.source,
                self.camera_width,
                self.camera_height,
                self.camera_fps,
            )
        except Exception:
            cap = None
        while self._running:
            if self._paused:
                time.sleep(0.05); continue
            t += 0.033; fc += 1
            ok, frame = cap.read() if cap is not None else (False, None)
            if not ok:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO CAMERA — DEMO MODE",
                            (80, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 200, 100), 2)

            prob = 0.5 + 0.45 * math.sin(t * 0.18)
            metrics = {
                "prob":         prob,
                "label":        int(prob >= 0.5),
                "ear_l":        0.32 - 0.12 * abs(math.sin(t * 0.3)),
                "ear_r":        0.30 - 0.10 * abs(math.sin(t * 0.31)),
                "ear_avg":      0.31 - 0.11 * abs(math.sin(t * 0.305)),
                "mar":          0.08 + 0.06 * abs(math.sin(t * 0.22)),
                "puc_l":        0.9 - 0.4 * abs(math.sin(t * 0.3)),
                "puc_r":        0.88 - 0.38 * abs(math.sin(t * 0.31)),
                "muc":          0.04 + 0.04 * abs(math.sin(t * 0.22)),
                "pitch":        5.0 * math.sin(t * 0.07),
                "yaw":          3.0 * math.sin(t * 0.05),
                "roll":         2.0 * math.sin(t * 0.04),
                "fps":          28.5 + 2.0 * math.sin(t),
                "frame_count":  fc,
                "perclos":      max(0.0, 0.15 * math.sin(t * 0.2)),
                "face_detected": True,
                "threshold":    self.threshold,
                "warmup":       fc < 30,
            }
            self.signals.frame_ready.emit(frame, metrics)
            time.sleep(0.033)
        if cap is not None:
            cap.release()


class InferenceWorker(QThread):
    def __init__(
        self,
        source,
        model_path,
        alert_clip=None,
        predict_stride=None,
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        parent=None,
    ):
        super().__init__(parent)
        self.source      = source
        self.model_path  = model_path
        self.alert_clip  = alert_clip
        self.predict_stride = predict_stride
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.signals     = InferenceSignals()
        self._running    = True
        self._paused     = False
        self.threshold   = None

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False; self.wait(3000)

    def run(self):
        # ── Config ────────────────────────────────────────────────────────────
        try:
            with open(DEPLOY_CFG_PATH) as f:
                cfg = json.load(f)
        except Exception as e:
            self.signals.error.emit(f"Config: {e}"); return

        cfg_features = cfg.get("feature_names")
        if cfg_features is not None and cfg_features != FEATURE_NAMES:
            self.signals.error.emit(
                "Config feature_names do not match live inference pipeline."
            )
            return

        threshold = float(cfg["threshold"] if self.threshold is None else self.threshold)
        self.threshold = threshold
        smooth_k     = max(1, int(cfg["smooth_k"]))
        seq_len      = int(cfg["sequence_length"])
        rolling_win  = int(cfg["rolling_window"])
        perclos_thr  = float(cfg["perclos_ear_iqr_threshold"])
        predict_stride = self.predict_stride
        if predict_stride is None:
            predict_stride = int(cfg.get("eval_step", DEFAULT_PREDICT_STRIDE))
        predict_stride = max(1, int(predict_stride))
        max_gap_fill = int(cfg.get("max_gap_fill", DEFAULT_MAX_GAP_FILL))
        temporal_pad = rolling_win - 1
        padded_len   = seq_len + temporal_pad

        try:
            with open(GLOBAL_NORM_PATH) as f:
                norm_params = json.load(f)
        except Exception as e:
            self.signals.error.emit(f"Norm params: {e}"); return

        # ── MediaPipe ─────────────────────────────────────────────────────────
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=MP_MODEL_PATH),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)

        if self.alert_clip:
            try:
                norm_params = compute_norm_params_from_clip(self.alert_clip, landmarker)
                landmarker.close()
                landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
            except Exception as e:
                self.signals.error.emit(f"Alert calibration: {e}")
                landmarker.close(); return

        try:
            model_bundle = load_model(self.model_path)
        except Exception as e:
            self.signals.error.emit(f"Model: {e}")
            landmarker.close(); return

        # ── Open source ───────────────────────────────────────────────────────
        try:
            cap = open_frame_source(
                self.source,
                self.camera_width,
                self.camera_height,
                self.camera_fps,
            )
        except Exception as e:
            self.signals.error.emit(f"Source: {e}")
            landmarker.close(); return

        cam_fps = cap.fps

        ring_buf    = collections.deque(maxlen=padded_len)
        prob_buf    = collections.deque(maxlen=smooth_k)
        ear_history = collections.deque(maxlen=rolling_win)
        feature_state = LiveFeatureState(max_gap_fill=max_gap_fill)
        display_raw = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        frame_count = 0
        feature_count = 0
        ts_ms       = 0.0
        label       = None
        prob        = None
        t_prev      = time.perf_counter()
        fps_disp    = 0.0

        while self._running:
            if self._paused:
                time.sleep(0.05); continue

            ok, frame = cap.read()
            if not ok:
                if cap.loop_source:
                    cap.reset()
                    ring_buf.clear()
                    prob_buf.clear()
                    ear_history.clear()
                    feature_state = LiveFeatureState(max_gap_fill=max_gap_fill)
                    display_raw = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
                    feature_count = 0
                    label = None
                    prob = None
                else:
                    time.sleep(0.02)
                continue

            H, W  = frame.shape[:2]
            frame_count += 1
            now   = time.perf_counter()
            fps_disp = 0.9 * fps_disp + 0.1 / max(now - t_prev, 1e-6)
            t_prev = now

            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(img, int(ts_ms))
            ts_ms += 1000.0 / cam_fps

            face_ok = bool(result.face_landmarks)
            detected = extract_features(result.face_landmarks[0], W, H) if face_ok else None
            raw = feature_state.update(detected)

            if raw is not None:
                display_raw = raw
                normed = normalise_frame(raw, norm_params)
                ring_buf.append(normed)
                ear_history.append(float(normed[2]))
                feature_count += 1

            should_predict = (
                len(ring_buf) >= seq_len and
                (feature_count - seq_len) % predict_stride == 0
            )
            if should_predict:
                win  = np.array(ring_buf, dtype=np.float32)
                if len(win) < padded_len:
                    pad_n = padded_len - len(win)
                    win = np.pad(win, ((pad_n, 0), (0, 0)), mode="edge")
                Xenh = apply_enhanced_features(win[np.newaxis], rolling_win, perclos_thr)
                Xenh = Xenh[:, temporal_pad:, :]
                prob = run_model(model_bundle, Xenh.astype(np.float32))
                prob_buf.append(prob)

                preds = smooth_predict(np.array(list(prob_buf)), threshold, smooth_k)
                label = int(preds[-1])

            perclos_est = float(
                np.mean(np.array(list(ear_history), dtype=np.float32) < perclos_thr)
            ) if ear_history else 0.0

            metrics = {
                "prob": prob, "label": label,
                "ear_l": float(display_raw[0]), "ear_r": float(display_raw[1]),
                "ear_avg": float(display_raw[2]), "mar": float(display_raw[3]),
                "puc_l": float(display_raw[4]), "puc_r": float(display_raw[5]),
                "muc": float(display_raw[6]), "pitch": float(display_raw[7]),
                "yaw": float(display_raw[8]), "roll": float(display_raw[9]),
                "fps": fps_disp, "frame_count": frame_count,
                "perclos": perclos_est, "face_detected": face_ok,
                "threshold": threshold, "warmup": prob is None,
            }
            self.signals.frame_ready.emit(frame, metrics)

        cap.release()
        landmarker.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEST DASHCAM UI
# ═══════════════════════════════════════════════════════════════════════════════

class SectionHeader(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("sectionHeader")


class CornerFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("panel")
        self.setFrameShape(QFrame.StyledPanel)


class DrowsinessGauge(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(145)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.status_lbl = QLabel("Initializing")
        self.status_lbl.setObjectName("driverStatus")
        self.status_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.status_lbl)

        self.prob_lbl = QLabel("Risk 0%")
        self.prob_lbl.setObjectName("probLabel")
        self.prob_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.prob_lbl)

        self.prob_bar = QProgressBar()
        self.prob_bar.setRange(0, 100)
        self.prob_bar.setTextVisible(False)
        self.prob_bar.setFixedHeight(14)
        lay.addWidget(self.prob_bar)

        self.threshold_lbl = QLabel("Threshold --")
        self.threshold_lbl.setObjectName("mutedLabel")
        self.threshold_lbl.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.threshold_lbl)

    def set_data(self, prob, label, thresh):
        risk = max(0.0, min(1.0, prob if prob is not None else 0.0))
        if label is None:
            status, color = "Warming up", P["dimmer"]
        elif label == 1:
            status, color = "Drowsiness detected", P["red"]
        else:
            status, color = "Driver alert", P["green"]

        self.status_lbl.setText(status)
        self.status_lbl.setStyleSheet(f"color: {color};")
        self.prob_lbl.setText(f"Risk {risk * 100:.0f}%")
        self.threshold_lbl.setText(f"Threshold {thresh:.2f}")
        self.prob_bar.setValue(int(round(risk * 100)))
        self.prob_bar.setStyleSheet(
            "QProgressBar { background: %s; border: 1px solid %s; border-radius: 4px; }"
            "QProgressBar::chunk { background: %s; border-radius: 4px; }"
            % (P["bg3"], P["border"], color)
        )


class GaugeBar(QWidget):
    def __init__(self, label, lo=0.0, hi=1.0, danger_lo=None, danger_hi=None, parent=None):
        super().__init__(parent)
        self.lo = lo
        self.hi = hi
        self.danger_lo = danger_lo
        self.danger_hi = danger_hi
        self.setFixedHeight(30)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)

        self.name_lbl = QLabel(label)
        self.name_lbl.setObjectName("metricName")
        self.name_lbl.setFixedWidth(76)
        lay.addWidget(self.name_lbl)

        self.bar = QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        lay.addWidget(self.bar, 1)

        self.value_lbl = QLabel("0.000")
        self.value_lbl.setObjectName("metricValue")
        self.value_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_lbl.setFixedWidth(58)
        lay.addWidget(self.value_lbl)

    def set_value(self, v):
        if v is None or not np.isfinite(v):
            v = 0.0
        clamped = max(self.lo, min(self.hi, float(v)))
        ratio = (clamped - self.lo) / max(self.hi - self.lo, 1e-6)
        alert = (
            (self.danger_lo is not None and v < self.danger_lo) or
            (self.danger_hi is not None and v > self.danger_hi)
        )
        color = P["red"] if alert else P["cyan"]
        self.value_lbl.setText(f"{v:.3f}")
        self.value_lbl.setStyleSheet(f"color: {color};")
        self.bar.setValue(int(max(0.0, min(1.0, ratio)) * 1000))
        self.bar.setStyleSheet(
            "QProgressBar { background: %s; border: none; border-radius: 4px; }"
            "QProgressBar::chunk { background: %s; border-radius: 4px; }"
            % (P["bg3"], color)
        )


class HeadPoseWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setHorizontalSpacing(12)
        lay.setVerticalSpacing(4)
        self.values = {}
        for col, name in enumerate(("Pitch", "Yaw", "Roll")):
            title = QLabel(name)
            title.setObjectName("metricName")
            value = QLabel("+0.0 deg")
            value.setObjectName("poseValue")
            value.setAlignment(Qt.AlignCenter)
            lay.addWidget(title, 0, col, Qt.AlignCenter)
            lay.addWidget(value, 1, col, Qt.AlignCenter)
            self.values[name.lower()] = value

    def set_pose(self, pitch, yaw, roll):
        self.values["pitch"].setText(f"{pitch:+.1f} deg")
        self.values["yaw"].setText(f"{yaw:+.1f} deg")
        self.values["roll"].setText(f"{roll:+.1f} deg")


class AlertLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._start_time = time.time()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        lay.addWidget(SectionHeader("Events"))
        self.list_widget = QListWidget()
        self.list_widget.setObjectName("eventList")
        lay.addWidget(self.list_widget)

    def add_event(self, msg, level=1):
        elapsed = time.time() - self._start_time
        ts = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
        item = QListWidgetItem(f"{ts}  {msg}")
        item.setForeground(QColor({0: P["green"], 1: P["amber"], 2: P["red"]}.get(level, P["white"])))
        self.list_widget.insertItem(0, item)
        while self.list_widget.count() > 80:
            self.list_widget.takeItem(self.list_widget.count() - 1)


class FaceStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)
        self.fps_lbl = QLabel("FPS --")
        self.frames_lbl = QLabel("Frames 0")
        self.face_lbl = QLabel("Face no")
        for label in (self.fps_lbl, self.frames_lbl, self.face_lbl):
            label.setObjectName("statusItem")
            lay.addWidget(label)
        lay.addStretch()

    def update_data(self, fps, frames, face):
        self.fps_lbl.setText(f"FPS {fps:.1f}")
        self.frames_lbl.setText(f"Frames {frames:,}")
        self.face_lbl.setText("Face yes" if face else "Face no")
        self.face_lbl.setStyleSheet(f"color: {P['green'] if face else P['amber']};")


class CameraWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._label = None
        self._warmup = True
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(420, 300)

    def set_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    def set_status(self, label, warmup):
        self._label = label
        self._warmup = warmup
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#050505"))

        if self._pixmap:
            scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ox = (w - scaled.width()) // 2
            oy = (h - scaled.height()) // 2
            p.drawPixmap(ox, oy, scaled)

        p.fillRect(0, 0, w, 34, QColor(0, 0, 0, 145))
        p.setFont(QFont("DejaVu Sans", 10, QFont.Bold))
        p.setPen(QColor("#ffffff"))
        p.drawText(12, 22, time.strftime("%Y-%m-%d  %H:%M:%S"))
        p.setPen(QColor(P["red"]))
        p.drawText(w - 58, 22, "REC")

        if self._warmup:
            banner_text = "Warming up..."
            banner_col = QColor(P["white"])
            bg_col = QColor(0, 0, 0, 150)
        elif self._label == 1:
            banner_text = "Drowsiness detected"
            banner_col = QColor("#ffffff")
            bg_col = QColor(P["red"])
        else:
            banner_text = None
            banner_col = None
            bg_col = None

        if banner_text:
            p.fillRect(0, h - 38, w, 38, bg_col)
            p.setFont(QFont("DejaVu Sans", 12, QFont.Bold))
            p.setPen(banner_col)
            p.drawText(QRect(0, h - 38, w, 38), Qt.AlignCenter, banner_text)
        p.end()


GLOBAL_QSS = f"""
QMainWindow, QWidget {{
    background: {P['bg']};
    color: {P['white']};
    font-family: 'DejaVu Sans', Arial, sans-serif;
    font-size: 11px;
}}
QFrame#panel {{
    background: {P['surface']};
    border: 1px solid {P['border']};
    border-radius: 6px;
}}
QLabel#sectionHeader {{
    color: {P['dim']};
    font-size: 10px;
    font-weight: bold;
    padding-bottom: 2px;
}}
QLabel#driverStatus {{
    font-size: 22px;
    font-weight: bold;
}}
QLabel#probLabel {{
    color: {P['white']};
    font-size: 16px;
    font-weight: bold;
}}
QLabel#mutedLabel, QLabel#metricName, QLabel#statusItem {{
    color: {P['dim']};
}}
QLabel#metricValue, QLabel#poseValue {{
    color: {P['white']};
    font-weight: bold;
}}
QPushButton {{
    background: {P['bg3']};
    color: {P['white']};
    border: 1px solid {P['border_hi']};
    border-radius: 5px;
    padding: 0px 14px;
    min-height: 38px;
    font-weight: bold;
}}
QPushButton:pressed {{
    background: {P['border']};
}}
QPushButton#danger {{
    background: {P['red_dim']};
    border-color: {P['red']};
}}
QPushButton#pause_active {{
    background: {P['amber_dim']};
    border-color: {P['amber']};
}}
QListWidget#eventList {{
    background: {P['bg2']};
    border: 1px solid {P['border']};
    border-radius: 4px;
    padding: 4px;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._session_start = time.time()
        self._alert_count = 0
        self._was_drowsy = False
        self._paused = False

        self.setStyleSheet(GLOBAL_QSS)
        self.setWindowTitle("DrowsGuard Dashcam")
        self._build_ui()
        self._start_worker()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(8)
        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(10)
        self.cam_widget = CameraWidget()
        body.addWidget(self.cam_widget, 1)
        body.addWidget(self._build_right_panel(), 0)
        root.addLayout(body, 1)
        root.addWidget(self._build_footer())

    def _build_header(self):
        bar = CornerFrame()
        bar.setFixedHeight(48)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)
        lay.setSpacing(16)

        title = QLabel("DrowsGuard Dashcam")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        lay.addWidget(title)

        self.status_chip = QLabel("Initializing")
        self.status_chip.setStyleSheet(f"color: {P['dim']}; font-weight: bold;")
        lay.addWidget(self.status_chip)
        lay.addStretch()

        self.alert_badge = QLabel("0 alerts")
        self.alert_badge.setObjectName("mutedLabel")
        lay.addWidget(self.alert_badge)

        self.clock_lbl = QLabel("00:00:00")
        self.clock_lbl.setObjectName("mutedLabel")
        lay.addWidget(self.clock_lbl)
        return bar

    def _panel_with_layout(self):
        frame = CornerFrame()
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(8)
        return frame, lay

    def _build_right_panel(self):
        panel = QWidget()
        panel.setFixedWidth(330)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(10)

        frame, flay = self._panel_with_layout()
        flay.addWidget(SectionHeader("Driver status"))
        self.main_gauge = DrowsinessGauge()
        flay.addWidget(self.main_gauge)
        lay.addWidget(frame)

        frame, flay = self._panel_with_layout()
        flay.addWidget(SectionHeader("Eye and mouth"))
        self.bar_ear_l = GaugeBar("EAR left", 0.0, 0.5, danger_lo=0.18)
        self.bar_ear_r = GaugeBar("EAR right", 0.0, 0.5, danger_lo=0.18)
        self.bar_mar = GaugeBar("MAR", 0.0, 0.8, danger_hi=0.5)
        self.bar_perclos = GaugeBar("PERCLOS", 0.0, 1.0, danger_hi=0.35)
        for bar in (self.bar_ear_l, self.bar_ear_r, self.bar_mar, self.bar_perclos):
            flay.addWidget(bar)
        lay.addWidget(frame)

        frame, flay = self._panel_with_layout()
        flay.addWidget(SectionHeader("Head pose"))
        self.pose_widget = HeadPoseWidget()
        flay.addWidget(self.pose_widget)
        lay.addWidget(frame)

        frame, flay = self._panel_with_layout()
        flay.addWidget(SectionHeader("Camera"))
        self.face_strip = FaceStatusWidget()
        flay.addWidget(self.face_strip)
        lay.addWidget(frame)

        frame, flay = self._panel_with_layout()
        self.alert_log = AlertLogWidget()
        flay.addWidget(self.alert_log)
        lay.addWidget(frame, 1)
        return panel

    def _build_footer(self):
        bar = CornerFrame()
        bar.setFixedHeight(54)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 7, 12, 7)
        lay.setSpacing(10)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.setFixedWidth(120)
        self.pause_btn.clicked.connect(self._toggle_pause)
        lay.addWidget(self.pause_btn)

        self.thresh_lbl = QLabel("Threshold --")
        self.thresh_lbl.setObjectName("mutedLabel")
        lay.addWidget(self.thresh_lbl)
        lay.addStretch()

        model_name = Path(getattr(self.args, "model", "")).name or "--"
        model_lbl = QLabel(f"Model: {model_name}")
        model_lbl.setObjectName("mutedLabel")
        lay.addWidget(model_lbl)

        quit_btn = QPushButton("Quit")
        quit_btn.setObjectName("danger")
        quit_btn.setFixedWidth(100)
        quit_btn.clicked.connect(self.close)
        lay.addWidget(quit_btn)
        return bar

    def _start_worker(self):
        source = getattr(self.args, 'source', '0')
        camera_width = getattr(self.args, 'camera_width', 640)
        camera_height = getattr(self.args, 'camera_height', 480)
        camera_fps = getattr(self.args, 'camera_fps', 30)
        if INFERENCE_AVAILABLE:
            model = getattr(self.args, 'model', str(DEFAULT_MODEL))
            clip = getattr(self.args, 'alert_clip', None)
            stride = getattr(self.args, 'predict_stride', None)
            self.worker = InferenceWorker(
                source, model, clip, stride, camera_width, camera_height, camera_fps
            )
        else:
            self.worker = DemoWorker(source, camera_width, camera_height, camera_fps)

        self.worker.signals.frame_ready.connect(self._on_frame)
        self.worker.signals.error.connect(self._on_error)
        self.worker.start()
        self.alert_log.add_event("System started", 0)

    def _on_frame(self, frame, metrics):
        self.cam_widget.set_frame(frame)
        self.cam_widget.set_status(metrics["label"], metrics["warmup"])
        self.main_gauge.set_data(metrics["prob"], metrics["label"], metrics["threshold"])
        self.bar_ear_l.set_value(metrics["ear_l"])
        self.bar_ear_r.set_value(metrics["ear_r"])
        self.bar_mar.set_value(metrics["mar"])
        self.bar_perclos.set_value(metrics["perclos"])
        self.pose_widget.set_pose(metrics["pitch"], metrics["yaw"], metrics["roll"])
        self.face_strip.update_data(metrics["fps"], metrics["frame_count"], metrics["face_detected"])
        self.thresh_lbl.setText(f"Threshold {metrics['threshold']:.2f}")

        if metrics["warmup"]:
            self._set_chip("Warming up", P["dimmer"])
        elif metrics["label"] == 1:
            self._set_chip("Drowsiness detected", P["red"])
        else:
            self._set_chip("Driver alert", P["green"])

        if metrics["label"] == 1 and not self._was_drowsy:
            self._was_drowsy = True
            self._alert_count += 1
            prob = metrics["prob"] if metrics["prob"] is not None else 0.0
            self.alert_log.add_event(f"Drowsiness detected (p={prob:.2f})", 2)
            self._update_alert_badge()
        elif metrics["label"] == 0 and self._was_drowsy:
            self._was_drowsy = False
            self.alert_log.add_event("Driver alert cleared", 0)

    def _on_error(self, msg):
        self._set_chip("Error", P["red"])
        self.alert_log.add_event(f"Error: {msg[:60]}", 2)
        self.cam_widget.set_status(None, False)

    def _set_chip(self, text, color):
        self.status_chip.setText(text)
        self.status_chip.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _update_alert_badge(self):
        suffix = "s" if self._alert_count != 1 else ""
        self.alert_badge.setText(f"{self._alert_count} alert{suffix}")
        color = P["red"] if self._alert_count else P["dim"]
        self.alert_badge.setStyleSheet(f"color: {color};")

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.worker.pause()
            self.pause_btn.setText("Resume")
            self.pause_btn.setObjectName("pause_active")
            self.alert_log.add_event("Session paused", 1)
        else:
            self.worker.resume()
            self.pause_btn.setText("Pause")
            self.pause_btn.setObjectName("")
            self.alert_log.add_event("Session resumed", 0)
        self.pause_btn.setStyleSheet("")
        self.setStyleSheet(GLOBAL_QSS)

    def _tick_clock(self):
        elapsed = int(time.time() - self._session_start)
        self.clock_lbl.setText(f"{elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")

    def closeEvent(self, event):
        if hasattr(self, "worker"):
            self.worker.stop()
        event.accept()


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    ap = argparse.ArgumentParser(description="DrowsGuard Dashcam UI")
    ap.add_argument("--source",        default="0",
                    help=("Picamera2 camera index, e.g. 0, or a video file path "
                          "for OpenCV playback"))
    ap.add_argument("--model",         default=str(DEFAULT_MODEL),
                    help="Path to .keras or .tflite model")
    ap.add_argument("--alert-clip",    default=None, dest="alert_clip",
                    help="Short alert clip for per-subject calibration")
    ap.add_argument("--predict-stride", type=int, default=None,
                    help=("Run the model every N accepted frames. Defaults to "
                          "deploy_config eval_step, or 15 to match trainer.py."))
    ap.add_argument("--camera-width", type=int, default=640,
                    help="Picamera2 capture width for numeric camera sources")
    ap.add_argument("--camera-height", type=int, default=480,
                    help="Picamera2 capture height for numeric camera sources")
    ap.add_argument("--camera-fps", type=int, default=30,
                    help="Requested Picamera2 frame rate for numeric camera sources")
    ap.add_argument("--qt-platform", default=None,
                    choices=["xcb", "wayland", "eglfs", "linuxfb", "offscreen", "minimal"],
                    help="Override Qt platform plugin if auto-detection is wrong")
    ap.add_argument("--no-fullscreen", action="store_true",
                    help="Run in a window instead of full screen")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    configure_qt_environment(args.qt_platform)

    app = QApplication(sys.argv)
    app.setApplicationName("DrowsGuard")

    win = MainWindow(args)
    if args.no_fullscreen:
        win.resize(1024, 600)
        win.show()
    else:
        win.showFullScreen()

    sys.exit(app.exec_())
