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

try:
    from picamera2 import Picamera2
    PICAMERA2_AVAILABLE = True
except ModuleNotFoundError:
    Picamera2 = None
    PICAMERA2_AVAILABLE = False

try:
    from PyQt5.QtCore import (
        Qt, QThread, QTimer, pyqtSignal, QObject, QRect, QPointF,
        QRectF, QSizeF
    )
    from PyQt5.QtGui import (
        QImage, QPixmap, QPainter, QColor, QFont, QPen, QBrush,
        QLinearGradient, QConicalGradient, QRadialGradient, QPainterPath,
        QFontMetrics
    )
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton,
        QVBoxLayout, QHBoxLayout, QGridLayout, QFrame,
        QSizePolicy, QStackedWidget, QScrollArea
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
    "bg":         "#06100c",
    "bg2":        "#0b1a14",
    "bg3":        "#0f211a",
    "surface":    "#122118",
    "border":     "#1e3329",
    "border_hi":  "#2a4a3a",
    "green":      "#00e87a",
    "green_dim":  "#007a3f",
    "green_mid":  "#00b05c",
    "red":        "#ff2d2d",
    "red_dim":    "#7a1515",
    "amber":      "#ffb300",
    "amber_dim":  "#7a5500",
    "cyan":       "#00d4e8",
    "white":      "#d8ede6",
    "dim":        "#4a6a5a",
    "dimmer":     "#2a3d34",
    "scanline":   "#00150d",
}

def qc(key):
    return QColor(P[key])


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
#  CUSTOM WIDGETS
# ═══════════════════════════════════════════════════════════════════════════════

class SectionHeader(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setStyleSheet(f"""
            color: {P['dim']};
            font-size: 8px;
            font-weight: bold;
            font-family: 'DejaVu Sans Mono', monospace;
            letter-spacing: 2px;
            padding-bottom: 2px;
        """)


class CornerFrame(QWidget):
    """Dark frame with corner bracket decorations."""
    def __init__(self, parent=None):
        super().__init__(parent)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        w, h = self.width(), self.height()
        cs   = 8   # corner size

        p.fillRect(0, 0, w, h, qc("bg2"))

        # Subtle border
        p.setPen(QPen(qc("border"), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        # Corner accents
        p.setPen(QPen(qc("green_dim"), 1))
        for (x, y, dx, dy) in [
            (0, 0,  1, 1), (w-1, 0,  -1, 1),
            (0, h-1, 1, -1), (w-1, h-1, -1, -1)
        ]:
            p.drawLine(x, y, x + dx * cs, y)
            p.drawLine(x, y, x, y + dy * cs)
        p.end()


class DrowsinessGauge(QWidget):
    """Large arc gauge showing drowsiness probability."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.prob   = 0.0
        self.label  = None
        self.thresh = 0.5
        self._pulse = 0.0
        self.setMinimumSize(150, 150)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(40)

    def set_data(self, prob, label, thresh):
        self.prob   = prob if prob is not None else 0.0
        self.label  = label
        self.thresh = thresh
        self.update()

    def _tick(self):
        if self.label == 1:
            self._pulse = (self._pulse + 0.12) % (2 * math.pi)
            self.update()
        elif self._pulse != 0.0:
            self._pulse = 0.0
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        r  = min(w, h) / 2 - 14

        # ── Pulsing outer ring (drowsy state) ─────────────────────────────────
        if self.label == 1:
            alpha = int(80 + 60 * abs(math.sin(self._pulse)))
            pulse_pen = QPen(QColor(255, 45, 45, alpha), 2)
            pulse_pen.setStyle(Qt.DashLine)
            p.setPen(pulse_pen)
            p.setBrush(Qt.NoBrush)
            pr = r + 10 + 3 * abs(math.sin(self._pulse))
            p.drawEllipse(QRectF(cx - pr, cy - pr, pr * 2, pr * 2))

        # ── Track arc ─────────────────────────────────────────────────────────
        track_pen = QPen(qc("bg3"), 10, Qt.SolidLine, Qt.RoundCap)
        p.setPen(track_pen)
        p.setBrush(Qt.NoBrush)
        p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2),
                  int(225 * 16), int(-270 * 16))

        # ── Filled arc ────────────────────────────────────────────────────────
        if self.label is None:
            arc_col = QColor(P["dim"])
        elif self.label == 1:
            arc_col = QColor(P["red"])
        elif self.prob > 0.65:
            arc_col = QColor(P["amber"])
        elif self.prob > 0.4:
            # blend amber → green
            t = (self.prob - 0.4) / 0.25
            arc_col = QColor(
                int(255 * t), int(179 - 179 * t + 184 * (1 - t)), 0
            )
        else:
            arc_col = QColor(P["green"])

        fill_pen = QPen(arc_col, 10, Qt.SolidLine, Qt.RoundCap)
        p.setPen(fill_pen)
        span = -int(270 * 16 * self.prob)
        if span != 0:
            p.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2),
                      int(225 * 16), span)

        # ── Threshold tick ────────────────────────────────────────────────────
        thr_ang = math.radians(225 - 270 * self.thresh)
        tick_r1, tick_r2 = r - 6, r + 6
        p.setPen(QPen(QColor(P["white"]), 1))
        p.drawLine(
            QPointF(cx + tick_r1 * math.cos(thr_ang), cy - tick_r1 * math.sin(thr_ang)),
            QPointF(cx + tick_r2 * math.cos(thr_ang), cy - tick_r2 * math.sin(thr_ang)),
        )

        # ── Inner circle ──────────────────────────────────────────────────────
        inner = r - 16
        p.setPen(Qt.NoPen)
        radial = QRadialGradient(cx, cy, inner)
        radial.setColorAt(0.0, QColor(P["bg3"]))
        radial.setColorAt(1.0, QColor(P["bg2"]))
        p.setBrush(QBrush(radial))
        p.drawEllipse(QRectF(cx - inner, cy - inner, inner * 2, inner * 2))

        # ── Status text ───────────────────────────────────────────────────────
        if self.label is None:
            txt, col, fsz = "INIT", P["dim"], 11
        elif self.label == 1:
            txt, col, fsz = "DROWSY", P["red"], 13
        else:
            txt, col, fsz = "ALERT", P["green"], 14

        p.setFont(QFont("DejaVu Sans Mono", fsz, QFont.Bold))
        p.setPen(QColor(col))
        p.drawText(QRectF(cx - inner, cy - 16, inner * 2, 28),
                   Qt.AlignCenter, txt)

        # ── Probability value ─────────────────────────────────────────────────
        p.setFont(QFont("DejaVu Sans Mono", 8))
        p.setPen(QColor(P["dim"]))
        p.drawText(QRectF(cx - inner, cy + 10, inner * 2, 16),
                   Qt.AlignCenter, f"p = {self.prob:.3f}")

        p.end()


class GaugeBar(QWidget):
    """Compact horizontal gauge with bracket label."""
    def __init__(self, label, lo=0.0, hi=1.0, danger_lo=None, danger_hi=None, parent=None):
        super().__init__(parent)
        self.label      = label
        self.lo, self.hi = lo, hi
        self.danger_lo  = danger_lo
        self.danger_hi  = danger_hi
        self.value      = (lo + hi) / 2
        self.alert      = False
        self.setFixedHeight(26)

    def set_value(self, v):
        self.value = max(self.lo, min(self.hi, v))
        self.alert = (
            (self.danger_lo is not None and v < self.danger_lo) or
            (self.danger_hi is not None and v > self.danger_hi)
        )
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        lw    = 64
        bar_x = lw + 4
        bar_w = w - bar_x - 40
        bar_h = 5
        bar_y = h // 2 - bar_h // 2

        # Label
        p.setFont(QFont("DejaVu Sans Mono", 7, QFont.Bold))
        p.setPen(qc("dim"))
        p.drawText(QRect(0, 0, lw, h), Qt.AlignVCenter | Qt.AlignLeft, self.label)

        # Track
        p.setPen(Qt.NoPen)
        p.setBrush(qc("bg3"))
        p.drawRoundedRect(bar_x, bar_y, bar_w, bar_h, 2, 2)

        # Fill
        ratio = (self.value - self.lo) / max(self.hi - self.lo, 1e-6)
        fw    = int(bar_w * ratio)
        color = P["red"] if self.alert else P["green"]
        if fw > 0:
            grad = QLinearGradient(bar_x, 0, bar_x + fw, 0)
            if self.alert:
                grad.setColorAt(0, QColor(P["red_dim"]))
                grad.setColorAt(1, QColor(P["red"]))
            else:
                grad.setColorAt(0, QColor(P["green_dim"]))
                grad.setColorAt(1, QColor(P["green_mid"]))
            p.setBrush(QBrush(grad))
            p.drawRoundedRect(bar_x, bar_y, fw, bar_h, 2, 2)

        # Value
        p.setFont(QFont("DejaVu Sans Mono", 8))
        p.setPen(QColor(color if self.alert else P["white"]))
        p.drawText(QRect(bar_x + bar_w + 4, 0, 36, h),
                   Qt.AlignVCenter | Qt.AlignRight, f"{self.value:.3f}")

        p.end()


class HeadPoseWidget(QWidget):
    """Three mini axis bars: Pitch / Yaw / Roll."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.pitch = self.yaw = self.roll = 0.0
        self.setFixedHeight(62)

    def set_pose(self, pitch, yaw, roll):
        self.pitch, self.yaw, self.roll = pitch, yaw, roll
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        col_w = w // 3

        for i, (lbl, val) in enumerate([("PITCH", self.pitch),
                                         ("YAW",   self.yaw),
                                         ("ROLL",  self.roll)]):
            cx = col_w * i + col_w // 2
            bw = col_w - 14
            bh = 5
            by = h // 2 - bh // 2 + 6

            # Label
            p.setFont(QFont("DejaVu Sans Mono", 7, QFont.Bold))
            p.setPen(qc("dim"))
            p.drawText(QRect(col_w * i, 0, col_w, 14), Qt.AlignCenter, lbl)

            # Track
            p.setPen(Qt.NoPen)
            p.setBrush(qc("bg3"))
            p.drawRoundedRect(cx - bw // 2, by, bw, bh, 2, 2)

            # Centered fill
            ratio  = max(-1.0, min(1.0, val / 45.0))
            half   = bw // 2
            fw     = int(half * abs(ratio))
            color  = P["amber"] if abs(val) > 25 else P["cyan"]
            p.setBrush(QColor(color))
            if fw > 0:
                if ratio < 0:
                    p.drawRoundedRect(cx - fw, by, fw, bh, 2, 2)
                else:
                    p.drawRoundedRect(cx, by, fw, bh, 2, 2)

            # Center notch
            p.setPen(QPen(qc("border_hi"), 1))
            p.drawLine(cx, by - 2, cx, by + bh + 2)

            # Value
            p.setFont(QFont("DejaVu Sans Mono", 7))
            p.setPen(QColor(color))
            p.drawText(QRect(col_w * i, by + bh + 3, col_w, 14),
                       Qt.AlignCenter, f"{val:+.1f}°")

        p.end()


class AlertLogWidget(QWidget):
    """Scrolling event log panel."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.entries     = []
        self._start_time = time.time()
        self.setMinimumHeight(70)

    def add_event(self, msg, level=1):
        elapsed = time.time() - self._start_time
        ts      = f"{int(elapsed)//60:02d}:{int(elapsed)%60:02d}"
        self.entries.append((ts, msg, level))
        if len(self.entries) > 60:
            self.entries.pop(0)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, qc("bg2"))
        p.setPen(QPen(qc("border"), 1))
        p.drawRect(0, 0, w - 1, h - 1)

        # Header
        p.setFont(QFont("DejaVu Sans Mono", 7, QFont.Bold))
        p.setPen(qc("dim"))
        p.drawText(6, 13, "EVENT LOG")

        lh      = 13
        visible = max(0, (h - 18) // lh)
        shown   = self.entries[-visible:]

        level_colors = {0: P["green"], 1: P["amber"], 2: P["red"]}
        for i, (ts, msg, lv) in enumerate(reversed(shown)):
            y = h - 5 - i * lh
            p.setFont(QFont("DejaVu Sans Mono", 7))
            p.setPen(QColor(P["dimmer"])); p.drawText(6,  y, ts)
            p.setPen(QColor(level_colors.get(lv, P["white"])))
            # Truncate long messages
            fm  = QFontMetrics(p.font())
            txt = fm.elidedText(msg, Qt.ElideRight, w - 58)
            p.drawText(50, y, txt)

        p.end()


class FaceStatusWidget(QWidget):
    """Compact face detection + FPS strip."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.fps    = 0.0
        self.frames = 0
        self.face   = False
        self.setFixedHeight(32)

    def update_data(self, fps, frames, face):
        self.fps = fps; self.frames = frames; self.face = face
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, qc("bg2"))

        def tag(x, label, val, color):
            p.setFont(QFont("DejaVu Sans Mono", 7, QFont.Bold))
            p.setPen(qc("dim"))
            p.drawText(x, 0, 50, h, Qt.AlignVCenter | Qt.AlignLeft, label)
            p.setFont(QFont("DejaVu Sans Mono", 9))
            p.setPen(QColor(color))
            p.drawText(x + 48, 0, 70, h, Qt.AlignVCenter | Qt.AlignLeft, val)

        tag(6,         "FPS",    f"{self.fps:.1f}",   P["white"])
        tag(w // 3,    "FRAMES", f"{self.frames:,}",  P["white"])
        tag(2*w//3,    "FACE",   "◈ YES" if self.face else "○ NO",
            P["green"] if self.face else P["amber"])

        p.end()


# ═══════════════════════════════════════════════════════════════════════════════
#  CAMERA WIDGET  (with HUD overlay)
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWidget(QWidget):
    """Displays the camera frame with a subtle scan-line / corner HUD overlay."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap    = None
        self._label     = None   # None | 0 | 1
        self._warmup    = True
        self._anim      = 0.0
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def set_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qi   = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    def set_status(self, label, warmup):
        self._label  = label
        self._warmup = warmup
        self.update()

    def _tick(self):
        if self._label == 1:
            self._anim = (self._anim + 0.15) % (2 * math.pi)
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()

        # Camera frame
        if self._pixmap:
            scaled = self._pixmap.scaled(
                w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            ox = (w - scaled.width())  // 2
            oy = (h - scaled.height()) // 2
            p.drawPixmap(ox, oy, scaled)
        else:
            p.fillRect(0, 0, w, h, QColor("#000"))

        # Scan-line overlay
        p.setPen(Qt.NoPen)
        for y in range(0, h, 4):
            p.fillRect(0, y, w, 1, QColor(0, 0, 0, 18))

        # Corner brackets
        cs  = 20
        brd = 8
        col = P["red"] if self._label == 1 else P["green_dim"]
        if self._label == 1:
            alpha = int(160 + 80 * abs(math.sin(self._anim)))
            col_q = QColor(P["red"])
            col_q.setAlpha(alpha)
        else:
            col_q = QColor(col)

        pen = QPen(col_q, 2)
        p.setPen(pen)
        for (x, y, sx, sy) in [
            (brd, brd, 1, 1), (w - brd, brd, -1, 1),
            (brd, h - brd, 1, -1), (w - brd, h - brd, -1, -1)
        ]:
            p.drawLine(x, y, x + sx * cs, y)
            p.drawLine(x, y, x, y + sy * cs)

        # Status overlay banner
        if self._warmup:
            banner_text = "⬡  WARMING UP..."
            banner_col  = QColor(P["dim"])
            bg_col      = QColor(0, 0, 0, 140)
        elif self._label == 1:
            pulse_a = int(180 + 60 * abs(math.sin(self._anim)))
            banner_text = "⚠  DROWSINESS DETECTED"
            banner_col  = QColor(P["red"])
            bg_col      = QColor(80, 0, 0, pulse_a)
        else:
            banner_text = None
            bg_col      = None

        if banner_text:
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(bg_col))
            p.drawRect(0, h - 34, w, 34)
            p.setFont(QFont("DejaVu Sans Mono", 11, QFont.Bold))
            p.setPen(banner_col)
            p.drawText(QRect(0, h - 34, w, 34), Qt.AlignCenter, banner_text)

        p.end()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN WINDOW
# ═══════════════════════════════════════════════════════════════════════════════

GLOBAL_QSS = f"""
QMainWindow, QWidget {{
    background: {P['bg']};
    color: {P['white']};
    font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
}}
QPushButton {{
    background: {P['bg2']};
    color: {P['green']};
    border: 1px solid {P['green_dim']};
    border-radius: 3px;
    padding: 0px 14px;
    font-family: 'DejaVu Sans Mono', monospace;
    font-size: 10px;
    font-weight: bold;
    min-height: 40px;
    letter-spacing: 1px;
}}
QPushButton:pressed {{
    background: {P['bg3']};
    border-color: {P['green']};
    color: {P['green']};
}}
QPushButton#danger {{
    color: {P['red']};
    border-color: {P['red_dim']};
}}
QPushButton#danger:pressed {{
    background: {P['bg3']};
    border-color: {P['red']};
}}
QPushButton#pause_active {{
    color: {P['amber']};
    border-color: {P['amber_dim']};
}}
QLabel {{
    background: transparent;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args          = args
        self._session_start = time.time()
        self._alert_count  = 0
        self._was_drowsy   = False
        self._paused       = False

        self.setStyleSheet(GLOBAL_QSS)
        self.setWindowTitle("DrowsGuard")
        self._build_ui()
        self._start_worker()

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setContentsMargins(6, 5, 6, 5)
        body.setSpacing(6)

        self.cam_widget = CameraWidget()
        body.addWidget(self.cam_widget, 3)
        body.addLayout(self._build_right_panel(), 2)

        root.addLayout(body, 1)
        root.addWidget(self._build_footer())

    def _build_header(self):
        bar = QWidget()
        bar.setFixedHeight(46)
        bar.setStyleSheet(f"""
            background: {P['bg2']};
            border-bottom: 1px solid {P['border']};
        """)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(14, 0, 14, 0)

        # Logo
        logo = QLabel("◈  DROWSGUARD  v2")
        logo.setStyleSheet(f"""
            color: {P['green']};
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 3px;
        """)
        lay.addWidget(logo)
        lay.addStretch()

        # Status chip
        self.status_chip = QLabel("● INITIALIZING")
        self.status_chip.setStyleSheet(
            f"color: {P['dim']}; font-size: 11px; font-weight: bold;"
        )
        lay.addWidget(self.status_chip)
        lay.addStretch()

        # Alert badge
        self.alert_badge = QLabel("0 ALERTS")
        self.alert_badge.setStyleSheet(
            f"color: {P['dim']}; font-size: 10px;"
        )
        lay.addWidget(self.alert_badge)

        # Separator
        sep = QLabel("  │  ")
        sep.setStyleSheet(f"color: {P['dimmer']};")
        lay.addWidget(sep)

        # Session clock
        self.clock_lbl = QLabel("00:00:00")
        self.clock_lbl.setStyleSheet(
            f"color: {P['dim']}; font-size: 10px; font-weight: bold;"
        )
        lay.addWidget(self.clock_lbl)

        return bar

    def _build_right_panel(self):
        lay = QVBoxLayout()
        lay.setSpacing(5)

        # ── Drowsiness gauge ──────────────────────────────────────────────────
        g_frame = CornerFrame()
        gf_lay  = QVBoxLayout(g_frame)
        gf_lay.setContentsMargins(8, 5, 8, 6)
        gf_lay.setSpacing(2)
        gf_lay.addWidget(SectionHeader("DROWSINESS INDEX"))
        self.main_gauge = DrowsinessGauge()
        gf_lay.addWidget(self.main_gauge, 0, Qt.AlignCenter)
        lay.addWidget(g_frame)

        # ── Ocular metrics ────────────────────────────────────────────────────
        e_frame = CornerFrame()
        ef_lay  = QVBoxLayout(e_frame)
        ef_lay.setContentsMargins(8, 5, 8, 5)
        ef_lay.setSpacing(2)
        ef_lay.addWidget(SectionHeader("OCULAR METRICS"))

        self.bar_ear_l   = GaugeBar("EAR  L",   0.0, 0.5, danger_lo=0.18)
        self.bar_ear_r   = GaugeBar("EAR  R",   0.0, 0.5, danger_lo=0.18)
        self.bar_mar     = GaugeBar("MAR",       0.0, 0.8, danger_hi=0.5)
        self.bar_perclos = GaugeBar("PERCLOS",   0.0, 1.0, danger_hi=0.35)
        for b in (self.bar_ear_l, self.bar_ear_r, self.bar_mar, self.bar_perclos):
            ef_lay.addWidget(b)
        lay.addWidget(e_frame)

        # ── Head pose ─────────────────────────────────────────────────────────
        p_frame = CornerFrame()
        pf_lay  = QVBoxLayout(p_frame)
        pf_lay.setContentsMargins(8, 5, 8, 5)
        pf_lay.setSpacing(2)
        pf_lay.addWidget(SectionHeader("HEAD POSE"))
        self.pose_widget = HeadPoseWidget()
        pf_lay.addWidget(self.pose_widget)
        lay.addWidget(p_frame)

        # ── Face / FPS strip ──────────────────────────────────────────────────
        self.face_strip = FaceStatusWidget()
        self.face_strip.setStyleSheet(
            f"border: 1px solid {P['border']}; border-radius: 3px;"
        )
        lay.addWidget(self.face_strip)

        # ── Alert log ─────────────────────────────────────────────────────────
        self.alert_log = AlertLogWidget()
        lay.addWidget(self.alert_log, 1)

        return lay

    def _build_footer(self):
        bar = QWidget()
        bar.setFixedHeight(54)
        bar.setStyleSheet(
            f"background: {P['bg2']}; border-top: 1px solid {P['border']};"
        )
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(10, 7, 10, 7)
        lay.setSpacing(8)

        self.pause_btn = QPushButton("⏸   PAUSE")
        self.pause_btn.setFixedWidth(130)
        self.pause_btn.clicked.connect(self._toggle_pause)
        lay.addWidget(self.pause_btn)

        # Threshold display
        self.thresh_lbl = QLabel("THR: —")
        self.thresh_lbl.setStyleSheet(
            f"color: {P['dim']}; font-size: 9px; font-weight: bold;"
        )
        lay.addWidget(self.thresh_lbl)
        lay.addStretch()

        # Model name
        mdl = getattr(self.args, 'model', '')
        short = Path(mdl).name if mdl else "—"
        mdl_lbl = QLabel(f"MODEL: {short}")
        mdl_lbl.setStyleSheet(f"color: {P['dimmer']}; font-size: 8px;")
        lay.addWidget(mdl_lbl)
        lay.addStretch()

        quit_btn = QPushButton("✕   QUIT")
        quit_btn.setObjectName("danger")
        quit_btn.setFixedWidth(110)
        quit_btn.clicked.connect(self.close)
        lay.addWidget(quit_btn)

        return bar

    # ── Worker lifecycle ──────────────────────────────────────────────────────

    def _start_worker(self):
        source = getattr(self.args, 'source', '0')
        camera_width = getattr(self.args, 'camera_width', 640)
        camera_height = getattr(self.args, 'camera_height', 480)
        camera_fps = getattr(self.args, 'camera_fps', 30)
        if INFERENCE_AVAILABLE:
            model  = getattr(self.args, 'model', str(DEFAULT_MODEL))
            clip   = getattr(self.args, 'alert_clip', None)
            stride = getattr(self.args, 'predict_stride', None)
            self.worker = InferenceWorker(
                source,
                model,
                clip,
                stride,
                camera_width,
                camera_height,
                camera_fps,
            )
        else:
            self.worker = DemoWorker(source, camera_width, camera_height, camera_fps)

        self.worker.signals.frame_ready.connect(self._on_frame)
        self.worker.signals.error.connect(self._on_error)
        self.worker.start()
        self.alert_log.add_event("System started", 0)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_frame(self, frame, metrics):
        # Camera view
        self.cam_widget.set_frame(frame)
        self.cam_widget.set_status(metrics["label"], metrics["warmup"])

        # Drowsiness gauge
        self.main_gauge.set_data(metrics["prob"], metrics["label"], metrics["threshold"])

        # Bars
        self.bar_ear_l.set_value(metrics["ear_l"])
        self.bar_ear_r.set_value(metrics["ear_r"])
        self.bar_mar.set_value(metrics["mar"])
        self.bar_perclos.set_value(metrics["perclos"])

        # Head pose
        self.pose_widget.set_pose(metrics["pitch"], metrics["yaw"], metrics["roll"])

        # Face strip
        self.face_strip.update_data(metrics["fps"], metrics["frame_count"],
                                    metrics["face_detected"])

        # Threshold label
        self.thresh_lbl.setText(f"THR: {metrics['threshold']:.2f}")

        # Header status
        if metrics["warmup"]:
            self._set_chip("● WARMING UP", P["dim"])
        elif metrics["label"] == 1:
            self._set_chip("▲  DROWSY", P["red"])
        elif metrics["label"] == 0:
            self._set_chip("●  ALERT", P["green"])

        # Transitions
        if metrics["label"] == 1 and not self._was_drowsy:
            self._was_drowsy = True
            self._alert_count += 1
            self.alert_log.add_event(
                f"DROWSINESS DETECTED  (p={metrics['prob']:.2f})", 2
            )
            self._update_alert_badge()
        elif metrics["label"] == 0 and self._was_drowsy:
            self._was_drowsy = False
            self.alert_log.add_event("Driver alert — cleared", 0)

    def _on_error(self, msg):
        self._set_chip("● ERROR", P["red"])
        self.alert_log.add_event(f"ERR: {msg[:40]}", 2)
        self.cam_widget.set_status(None, False)

    def _set_chip(self, text, color):
        self.status_chip.setText(text)
        self.status_chip.setStyleSheet(
            f"color: {color}; font-size: 11px; font-weight: bold;"
        )

    def _update_alert_badge(self):
        n    = self._alert_count
        s    = f"{n} ALERT{'S' if n != 1 else ''}"
        col  = P["red"] if n > 0 else P["dim"]
        self.alert_badge.setText(s)
        self.alert_badge.setStyleSheet(f"color: {col}; font-size: 10px; font-weight: bold;")

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.worker.pause()
            self.pause_btn.setText("▶   RESUME")
            self.pause_btn.setObjectName("pause_active")
            self.alert_log.add_event("Session paused", 1)
        else:
            self.worker.resume()
            self.pause_btn.setText("⏸   PAUSE")
            self.pause_btn.setObjectName("")
            self.alert_log.add_event("Session resumed", 0)
        self.pause_btn.setStyleSheet("")   # force QSS re-eval
        self.setStyleSheet(GLOBAL_QSS)

    def _tick_clock(self):
        e = int(time.time() - self._session_start)
        self.clock_lbl.setText(f"{e//3600:02d}:{(e%3600)//60:02d}:{e%60:02d}")

    def closeEvent(self, event):
        if hasattr(self, 'worker'):
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
    ap.add_argument("--no-fullscreen", action="store_true",
                    help="Run in a window instead of full screen")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Wayland / EGLFS on Pi — force xcb or linuxfb
    import os
    if "DISPLAY" not in os.environ and "WAYLAND_DISPLAY" not in os.environ:
        os.environ.setdefault("QT_QPA_PLATFORM", "linuxfb")

    app = QApplication(sys.argv)
    app.setApplicationName("DrowsGuard")

    win = MainWindow(args)
    if args.no_fullscreen:
        win.resize(1024, 600)
        win.show()
    else:
        win.showFullScreen()

    sys.exit(app.exec_())
