#!/usr/bin/env python3
"""
DrowsGuard Dashcam UI
=====================
Camera-first UI for the drowsiness detection system.
Designed for Raspberry Pi 5 (Bookworm) + capacitive display.

Install
-------
  sudo apt install python3-pyqt5 python3-picamera2
  pip install piper-tts              # optional neural voice prompts
  sudo apt install flite espeak-ng   # fallback voice prompts
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
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import warnings
import wave
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
        QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy
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

DEFAULT_CALIBRATION_FRAMES = 90
PIPER_MODEL_CANDIDATES = (
    "piper_voice.onnx",
    "voice.onnx",
    "en_US-amy-medium.onnx",
    "en_US-lessac-medium.onnx",
    "en_US-ryan-medium.onnx",
)


class AudioNotifier:
    """Small non-blocking voice/tone helper for dashcam prompts."""

    def __init__(
        self,
        enabled=True,
        alert_interval=8.0,
        engine="piper",
        voice=None,
        rate=132,
    ):
        self.enabled = bool(enabled)
        self.alert_interval = max(1.0, float(alert_interval))
        self.engine = engine
        self.voice = voice
        self.rate = max(80, min(260, int(rate)))
        self._last_spoken = {}
        self._busy = False
        self._lock = threading.Lock()
        self._piper_model = None
        self._piper_config = None
        self._warned = set()
        self._speaker = self._find_speaker()

    def _find_speaker(self):
        fallback = ("piper", "flite", "espeak-ng", "espeak", "say", "spd-say")
        if self.engine and self.engine != "auto":
            candidates = (self.engine,) + tuple(c for c in fallback if c != self.engine)
        else:
            candidates = fallback
        for cmd in candidates:
            path = shutil.which(cmd)
            if not path:
                continue
            if cmd == "piper":
                model = self._find_piper_model()
                if not model:
                    self._warn_once(
                        "piper_model",
                        "Audio: Piper found, but no Piper .onnx voice model with matching config was found.",
                    )
                    continue
                player = self._audio_player_cmd("__probe__.wav")
                if not player:
                    self._warn_once(
                        "piper_player",
                        "Audio: Piper voice found, but no WAV player found. Install alsa-utils for aplay.",
                    )
                    continue
                self._piper_model = model
            return path
        return None

    def _find_piper_model(self):
        candidates = []
        if self.voice:
            candidates.extend(self._piper_model_candidates_from_path(Path(self.voice).expanduser()))
        env_model = os.environ.get("PIPER_MODEL")
        if env_model:
            candidates.extend(self._piper_model_candidates_from_path(Path(env_model).expanduser()))
        candidates.extend(BASE_DIR / name for name in PIPER_MODEL_CANDIDATES)
        candidates.extend(sorted(BASE_DIR.glob("*.onnx")))
        for folder in (BASE_DIR / "voices", BASE_DIR / "piper_voices"):
            if folder.exists():
                candidates.extend(sorted(folder.glob("**/*.onnx")))

        for candidate in candidates:
            if not candidate.exists() or candidate.suffix != ".onnx":
                continue
            config = self._piper_config_for(candidate)
            if config:
                self._piper_config = config
                return str(candidate)

        for root in (Path("/usr/share/piper-voices"), Path("/usr/local/share/piper-voices")):
            if not root.exists():
                continue
            for name in PIPER_MODEL_CANDIDATES[2:]:
                matches = list(root.glob(f"**/{name}"))
                if matches:
                    config = self._piper_config_for(matches[0])
                    if config:
                        self._piper_config = config
                        return str(matches[0])
            for match in sorted(root.glob("**/*.onnx")):
                config = self._piper_config_for(match)
                if config:
                    self._piper_config = config
                    return str(match)
        return None

    def _piper_model_candidates_from_path(self, path):
        if path.is_dir():
            return sorted(path.glob("**/*.onnx"))
        return [path]

    def _piper_config_for(self, model_path):
        model_path = Path(model_path)
        configs = (
            Path(str(model_path) + ".json"),
            model_path.with_suffix(".json"),
        )
        for config in configs:
            if config.exists():
                return str(config)
        return None

    def _warn_once(self, key, message):
        if key in self._warned:
            return
        self._warned.add(key)
        print(message, file=sys.stderr)

    def _audio_player_cmd(self, wav_path):
        players = (
            ("aplay", ["-q", wav_path]),
            ("paplay", [wav_path]),
            ("pw-play", [wav_path]),
            ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet", wav_path]),
            ("afplay", [wav_path]),
        )
        for name, args in players:
            path = shutil.which(name)
            if path:
                return [path] + args
        return None

    def say(self, text, key=None, min_interval=0.0, warning_tone=False):
        if not self.enabled:
            return

        now = time.monotonic()
        key = key or text
        last = self._last_spoken.get(key, -1e9)
        if now - last < float(min_interval):
            return
        self._last_spoken[key] = now

        if not self._speaker and not warning_tone:
            return

        thread = threading.Thread(
            target=self._speak,
            args=(text, warning_tone),
            daemon=True,
        )
        thread.start()

    def _speak(self, text, warning_tone=False):
        with self._lock:
            if self._busy:
                return
            self._busy = True
        try:
            if warning_tone:
                self._play_warning_tone_blocking()
            if not self._speaker:
                return
            name = Path(self._speaker).name
            if name == "piper":
                wav_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                        wav_path = fh.name
                    cmd = [
                        self._speaker,
                        "--model", self._piper_model,
                        "--config", self._piper_config,
                        "--output_file", wav_path,
                    ]
                    result = subprocess.run(
                        cmd,
                        input=f"{text}\n".encode("utf-8"),
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=10,
                        check=False,
                    )
                    if result.returncode != 0 or not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
                        self._warn_once("piper_run", "Audio: Piper failed to synthesize speech.")
                        return
                    player = self._audio_player_cmd(wav_path)
                    if player:
                        result = subprocess.run(
                            player,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL,
                            timeout=8,
                            check=False,
                        )
                        if result.returncode != 0:
                            self._warn_once("piper_play", "Audio: Piper generated speech, but playback failed.")
                    else:
                        self._warn_once("piper_player_runtime", "Audio: No WAV player available for Piper output.")
                finally:
                    if wav_path:
                        try:
                            os.unlink(wav_path)
                        except OSError:
                            pass
                return
            if name == "flite":
                voice = self.voice or "slt"
                cmd = [self._speaker, "-voice", voice, "-t", text]
            elif name in ("espeak", "espeak-ng"):
                voice = self.voice or "en-us+f4"
                cmd = [
                    self._speaker, "-v", voice,
                    "-s", str(self.rate),
                    "-p", "42",
                    "-g", "5",
                    text,
                ]
            elif name == "say":
                cmd = [self._speaker]
                if self.voice:
                    cmd.extend(["-v", self.voice])
                cmd.extend(["-r", str(self.rate), text])
            elif name == "spd-say":
                cmd = [self._speaker, "-r", "-45", text]
            else:
                cmd = [self._speaker, text]
            subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=6,
                check=False,
            )
        except Exception:
            return
        finally:
            with self._lock:
                self._busy = False

    def _play_warning_tone_blocking(self):
        wav_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                wav_path = fh.name
            self._write_warning_tone(wav_path)
            player = self._audio_player_cmd(wav_path)
            if player:
                result = subprocess.run(
                    player,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=4,
                    check=False,
                )
                if result.returncode == 0:
                    return
            QApplication.beep()
        except Exception:
            QApplication.beep()
        finally:
            if wav_path:
                try:
                    os.unlink(wav_path)
                except OSError:
                    pass

    def _write_warning_tone(self, wav_path):
        sample_rate = 22050
        segments = (
            (880.0, 0.16),
            (0.0, 0.06),
            (880.0, 0.16),
            (0.0, 0.06),
            (660.0, 0.22),
        )
        amplitude = 0.35
        samples = []
        for freq, duration in segments:
            count = int(sample_rate * duration)
            for i in range(count):
                if freq <= 0.0:
                    value = 0.0
                else:
                    fade = min(1.0, i / 120.0, (count - i) / 120.0)
                    value = amplitude * fade * math.sin(2.0 * math.pi * freq * i / sample_rate)
                samples.append(struct.pack("<h", int(max(-1.0, min(1.0, value)) * 32767)))
        with wave.open(wav_path, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(b"".join(samples))


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
    raise ValueError(f"Unsupported frame rotation: {rotation}")


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
        # Picamera2's RGB888 stream arrives in the channel order expected by
        # OpenCV/MediaPipe here. Swapping it again makes the UI look blue.
        if frame.ndim == 3 and frame.shape[2] == 4:
            frame = frame[:, :, :3]
        frame = np.ascontiguousarray(frame)
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


def stabilise_norm_params(params, fallback_params):
    stable = {}
    for feat in FEATURE_NAMES:
        fallback = fallback_params.get(feat, {"median": 0.0, "iqr": 1.0})
        med = float(params.get(feat, {}).get("median", fallback["median"]))
        iqr = float(params.get(feat, {}).get("iqr", fallback["iqr"]))
        fallback_iqr = abs(float(fallback.get("iqr", 1.0))) or 1.0
        if not np.isfinite(med):
            med = float(fallback.get("median", 0.0))
        if not np.isfinite(iqr) or iqr < fallback_iqr:
            iqr = fallback_iqr
        stable[feat] = {"median": med, "iqr": iqr}
    return stable


def compute_norm_params_from_frames(raw_features, fallback_params):
    arr = np.asarray(raw_features, dtype=np.float32)
    params = {}
    for i, feat in enumerate(FEATURE_NAMES):
        col = arr[:, i]
        med = float(np.median(col))
        iqr = float(np.percentile(col, 75) - np.percentile(col, 25))
        if iqr < 1e-6:
            iqr = 1.0
        params[feat] = {"median": med, "iqr": iqr}
    return stabilise_norm_params(params, fallback_params)


# ═══════════════════════════════════════════════════════════════════════════════
#  INFERENCE WORKER
# ═══════════════════════════════════════════════════════════════════════════════

class InferenceSignals(QObject):
    frame_ready = pyqtSignal(object, object)   # (frame_bgr ndarray, metrics dict)
    error       = pyqtSignal(str)


class DemoWorker(QThread):
    """Generates synthetic data when live_inference is unavailable."""
    def __init__(
        self,
        source="0",
        camera_width=640,
        camera_height=480,
        camera_fps=30,
        frame_rotation=0,
        parent=None,
    ):
        super().__init__(parent)
        self.source = source
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera_fps = camera_fps
        self.frame_rotation = int(frame_rotation) % 360
        self.signals  = InferenceSignals()
        self._running = True
        self._paused  = False
        self.threshold = 0.5
        self._calibrated = False
        self._calibration_requested = False

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False; self.wait(2000)
    def start_calibration(self):
        self._calibration_requested = True

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
            frame = rotate_frame(frame, self.frame_rotation)

            if not self._calibrated:
                if self._calibration_requested:
                    self._calibrated = True
                    state = "ready"
                    progress = 1.0
                else:
                    state = "waiting"
                    progress = 0.0
                self.signals.frame_ready.emit(frame, {
                    "prob": None,
                    "label": None,
                    "fps": 28.5,
                    "frame_count": fc,
                    "face_detected": True,
                    "threshold": self.threshold,
                    "warmup": True,
                    "calibrated": self._calibrated,
                    "calibration_state": state,
                    "calibration_progress": progress,
                })
                time.sleep(0.033)
                continue

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
                "calibrated":   True,
                "calibration_state": "ready",
                "calibration_progress": 1.0,
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
        calibration_frames=DEFAULT_CALIBRATION_FRAMES,
        frame_rotation=0,
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
        self.calibration_frames = max(30, int(calibration_frames))
        self.frame_rotation = int(frame_rotation) % 360
        self.signals     = InferenceSignals()
        self._running    = True
        self._paused     = False
        self.threshold   = None
        self._calibration_requested = False
        self._calibrated = bool(alert_clip)

    def pause(self):  self._paused = True
    def resume(self): self._paused = False
    def stop(self):   self._running = False; self.wait(3000)
    def start_calibration(self):
        self._calibration_requested = True

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
                global_norm = json.load(f)
                norm_params = global_norm
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
                norm_params = stabilise_norm_params(
                    compute_norm_params_from_clip(self.alert_clip, landmarker),
                    global_norm,
                )
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
        calibration_raw = []
        calibration_state = "ready" if self._calibrated else "waiting"
        calibration_feature_state = LiveFeatureState(max_gap_fill=0)

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
            frame = rotate_frame(frame, self.frame_rotation)

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

            if not self._calibrated:
                if self._calibration_requested:
                    if face_ok:
                        raw = calibration_feature_state.update(detected)
                        if raw is not None:
                            calibration_raw.append(raw)
                            display_raw = raw
                        calibration_state = "calibrating"
                        if len(calibration_raw) >= self.calibration_frames:
                            norm_params = compute_norm_params_from_frames(calibration_raw, global_norm)
                            self._calibrated = True
                            calibration_state = "ready"
                            ring_buf.clear()
                            prob_buf.clear()
                            ear_history.clear()
                            feature_state = LiveFeatureState(max_gap_fill=max_gap_fill)
                            display_raw = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
                            feature_count = 0
                            label = None
                            prob = None
                    else:
                        calibration_state = "no_face"
                else:
                    calibration_state = "waiting"

                self.signals.frame_ready.emit(frame, {
                    "prob": None, "label": None,
                    "ear_l": float(display_raw[0]), "ear_r": float(display_raw[1]),
                    "ear_avg": float(display_raw[2]), "mar": float(display_raw[3]),
                    "puc_l": float(display_raw[4]), "puc_r": float(display_raw[5]),
                    "muc": float(display_raw[6]), "pitch": float(display_raw[7]),
                    "yaw": float(display_raw[8]), "roll": float(display_raw[9]),
                    "fps": fps_disp, "frame_count": frame_count,
                    "perclos": 0.0, "face_detected": face_ok,
                    "threshold": threshold, "warmup": True,
                    "calibrated": self._calibrated,
                    "calibration_state": calibration_state,
                    "calibration_progress": min(1.0, len(calibration_raw) / self.calibration_frames),
                })
                continue

            if not face_ok:
                ring_buf.clear()
                prob_buf.clear()
                ear_history.clear()
                feature_state = LiveFeatureState(max_gap_fill=max_gap_fill)
                display_raw = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
                feature_count = 0
                label = None
                prob = None
                self.signals.frame_ready.emit(frame, {
                    "prob": None, "label": None,
                    "ear_l": 0.0, "ear_r": 0.0,
                    "ear_avg": 0.0, "mar": 0.0,
                    "puc_l": 0.0, "puc_r": 0.0,
                    "muc": 0.0, "pitch": 0.0,
                    "yaw": 0.0, "roll": 0.0,
                    "fps": fps_disp, "frame_count": frame_count,
                    "perclos": 0.0, "face_detected": False,
                    "threshold": threshold, "warmup": True,
                    "calibrated": True,
                    "calibration_state": "ready",
                    "calibration_progress": 1.0,
                })
                continue

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
                "calibrated": True,
                "calibration_state": "ready",
                "calibration_progress": 1.0,
            }
            self.signals.frame_ready.emit(frame, metrics)

        cap.release()
        landmarker.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHCAM UI
# ═══════════════════════════════════════════════════════════════════════════════

class CameraWidget(QWidget):
    calibration_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._calibrate_rect = QRect()
        self._metrics = {
            "prob": None,
            "label": None,
            "fps": 0.0,
            "face_detected": False,
            "warmup": True,
            "paused": False,
            "error": "",
            "calibrated": False,
            "calibration_state": "waiting",
            "calibration_progress": 0.0,
        }
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 180)

    def set_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        qi = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qi)
        self.update()

    def set_metrics(self, metrics):
        self._metrics = dict(metrics)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#050505"))

        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            ox = (w - scaled.width()) // 2
            oy = (h - scaled.height()) // 2
            p.drawPixmap(ox, oy, scaled)

        compact = w < 720 or h < 430
        tiny = w < 360 or h < 260
        margin = 8 if compact else 14
        top_h = 24 if tiny else (28 if compact else 38)
        bottom_h = 34 if tiny else (42 if compact else 58)
        label = self._metrics.get("label")
        warmup = bool(self._metrics.get("warmup"))
        paused = bool(self._metrics.get("paused"))
        error = str(self._metrics.get("error") or "")
        calibrated = bool(self._metrics.get("calibrated"))
        calibration_state = str(self._metrics.get("calibration_state") or "waiting")
        calibration_progress = float(self._metrics.get("calibration_progress") or 0.0)
        prob = self._metrics.get("prob")
        risk = max(0.0, min(1.0, prob if prob is not None else 0.0))
        face_ok = bool(self._metrics.get("face_detected"))
        fps = float(self._metrics.get("fps") or 0.0)

        p.fillRect(0, 0, w, top_h, QColor(0, 0, 0, 150))
        p.setFont(QFont("DejaVu Sans", 8 if tiny else (9 if compact else 11), QFont.Bold))
        p.setPen(QColor("#ffffff"))
        timestamp = time.strftime("%H:%M:%S") if tiny else time.strftime("%Y-%m-%d  %H:%M:%S")
        p.drawText(QRect(margin, 0, w - 82, top_h), Qt.AlignVCenter | Qt.AlignLeft, timestamp)
        p.setPen(QColor(P["red"]))
        rec_w = 46 if tiny else (54 if compact else 66)
        p.drawText(QRect(w - rec_w - margin, 0, rec_w, top_h), Qt.AlignVCenter | Qt.AlignRight, "REC")

        p.fillRect(0, h - bottom_h, w, bottom_h, QColor(0, 0, 0, 165))
        if not calibrated:
            status = "CAL" if tiny else "Normalize"
            status_color = QColor(P["dimmer"])
        elif not face_ok:
            status = "NO FACE" if tiny else "No face"
            status_color = QColor(P["amber"])
        elif warmup:
            status = "WAIT" if tiny else "Warming up"
            status_color = QColor(P["dimmer"])
        elif label == 1:
            status = "ALERT" if tiny else ("Drowsy" if compact else "Drowsiness detected")
            status_color = QColor(P["red"])
        else:
            status = "OK" if tiny else "Driver alert"
            status_color = QColor(P["green"])

        p.setFont(QFont("DejaVu Sans", 11 if tiny else (12 if compact else 17), QFont.Bold))
        p.setPen(status_color)
        risk_text = "--" if not calibrated or not face_ok else (f"{risk * 100:.0f}%" if tiny else f"Risk {risk * 100:.0f}%")
        risk_w = 44 if tiny else (78 if compact else 104)
        status_rect = QRect(margin, h - bottom_h + 2, w - (2 * margin) - risk_w - 12, bottom_h - 12)
        p.drawText(status_rect, Qt.AlignVCenter | Qt.AlignLeft, status)

        p.setFont(QFont("DejaVu Sans", 9 if tiny else (10 if compact else 13), QFont.Bold))
        p.setPen(QColor("#ffffff"))
        risk_rect = QRect(w - risk_w - margin, h - bottom_h + 2, risk_w, bottom_h - 12)
        p.drawText(risk_rect, Qt.AlignVCenter | Qt.AlignRight, risk_text)

        bar_x = margin
        bar_w = max(80, w - (2 * margin) - risk_w - 18)
        bar_y = h - 8 if tiny else (h - 12 if compact else h - 16)
        bar_h = 4 if tiny else (5 if compact else 7)
        p.fillRect(bar_x, bar_y, bar_w, bar_h, QColor(90, 90, 90, 170))
        if calibrated and face_ok:
            p.fillRect(bar_x, bar_y, int(bar_w * risk), bar_h, status_color)

        if not tiny:
            small = f"FPS {fps:.0f}   Face {'yes' if face_ok else 'no'}"
            if not compact:
                frame_count = int(self._metrics.get("frame_count") or 0)
                small += f"   Frame {frame_count:,}"
            p.setFont(QFont("DejaVu Sans", 8 if compact else 10))
            p.setPen(QColor(P["dim"]))
            p.drawText(margin, h - bottom_h + (12 if compact else 17), small)

        if error:
            err_h = 34 if compact else 46
            err_y = top_h + 8
            p.fillRect(0, err_y, w, err_h, QColor(120, 0, 0, 210))
            p.setFont(QFont("DejaVu Sans", 10 if compact else 13, QFont.Bold))
            p.setPen(QColor("#ffffff"))
            p.drawText(QRect(margin, err_y, w - 2 * margin, err_h), Qt.AlignVCenter | Qt.AlignLeft, error)

        if paused:
            pause_h = 44 if compact else 64
            pause_y = max(top_h + 6, h // 2 - pause_h // 2)
            p.fillRect(0, pause_y, w, pause_h, QColor(0, 0, 0, 185))
            p.setFont(QFont("DejaVu Sans", 16 if compact else 24, QFont.Bold))
            p.setPen(QColor("#ffffff"))
            p.drawText(QRect(0, pause_y, w, pause_h), Qt.AlignCenter, "PAUSED")

        if not calibrated:
            self._draw_calibration_overlay(
                p, w, h, compact, tiny, face_ok,
                calibration_state, calibration_progress,
            )

        p.end()

    def _draw_calibration_overlay(self, p, w, h, compact, tiny, face_ok, state, progress):
        p.fillRect(0, 0, w, h, QColor(0, 0, 0, 150))
        panel_w = min(w - 28, 420 if not compact else 320)
        panel_h = 170 if not tiny else 132
        panel_x = max(14, (w - panel_w) // 2)
        panel_y = max(38, (h - panel_h) // 2)
        panel = QRect(panel_x, panel_y, panel_w, panel_h)
        p.fillRect(panel, QColor(10, 10, 10, 215))

        title = "NORMALIZE" if tiny else "Normalize Driver"
        p.setFont(QFont("DejaVu Sans", 15 if tiny else 20, QFont.Bold))
        p.setPen(QColor("#ffffff"))
        p.drawText(QRect(panel_x, panel_y + 14, panel_w, 34), Qt.AlignCenter, title)

        if state == "waiting":
            message = "Face forward, eyes open"
        elif state == "no_face":
            message = "Face not found"
        else:
            message = "Hold still"
        p.setFont(QFont("DejaVu Sans", 9 if tiny else 12, QFont.Bold))
        p.setPen(QColor(P["green"] if face_ok else P["amber"]))
        p.drawText(QRect(panel_x + 10, panel_y + 50, panel_w - 20, 28), Qt.AlignCenter, message)

        button_w = min(panel_w - 36, 300)
        button_h = 42 if tiny else 50
        button_x = panel_x + (panel_w - button_w) // 2
        button_y = panel_y + panel_h - button_h - 18
        button = QRect(button_x, button_y, button_w, button_h)

        if state == "waiting":
            self._calibrate_rect = button
            p.fillRect(button, QColor(P["green"]))
            p.setFont(QFont("DejaVu Sans", 12 if tiny else 15, QFont.Bold))
            p.setPen(QColor("#ffffff"))
            p.drawText(button, Qt.AlignCenter, "START")
            return

        self._calibrate_rect = QRect()
        bar = QRect(button_x, button_y + button_h // 2 - 4, button_w, 8)
        p.fillRect(bar, QColor(80, 80, 80, 220))
        p.fillRect(bar.x(), bar.y(), int(bar.width() * max(0.0, min(1.0, progress))), bar.height(), QColor(P["green"]))
        pct = int(max(0.0, min(1.0, progress)) * 100)
        p.setFont(QFont("DejaVu Sans", 10 if tiny else 12, QFont.Bold))
        p.setPen(QColor("#ffffff"))
        p.drawText(QRect(button_x, button_y - 18, button_w, 18), Qt.AlignCenter, f"{pct}%")

    def mousePressEvent(self, event):
        if self._calibrate_rect.isValid() and self._calibrate_rect.contains(event.pos()):
            self.calibration_requested.emit()
            return
        super().mousePressEvent(event)


GLOBAL_QSS = f"""
QMainWindow, QWidget {{
    background: #000000;
    color: {P['white']};
    font-family: 'DejaVu Sans', Arial, sans-serif;
}}
"""


class MainWindow(QMainWindow):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self._paused = False
        self._last_error = ""
        self._last_audio_calibration_state = None
        self._last_audio_calibrated = False
        self._last_audio_face_ok = None
        self._last_audio_label = None
        self.audio = AudioNotifier(
            enabled=not getattr(args, "mute_audio", False),
            alert_interval=getattr(args, "audio_alert_interval", 8.0),
            engine=getattr(args, "audio_engine", "piper"),
            voice=getattr(args, "audio_voice", None),
            rate=getattr(args, "audio_rate", 132),
        )

        self.setStyleSheet(GLOBAL_QSS)
        self.setWindowTitle("DrowsGuard Dashcam")
        self._build_ui()
        self._start_worker()
        QTimer.singleShot(700, self._play_startup_instruction)

        self._clock_timer = QTimer(self)
        self._clock_timer.timeout.connect(self._tick_clock)
        self._clock_timer.start(1000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.cam_widget = CameraWidget()
        self.cam_widget.calibration_requested.connect(self._start_normalization)
        root.addWidget(self.cam_widget)

    def _play_startup_instruction(self):
        if getattr(self.args, "alert_clip", None):
            self.audio.say("Calibration loaded. Monitoring will begin shortly.", key="startup")
        else:
            self.audio.say(
                "Please face forward, then tap start.",
                key="cal_wait",
                min_interval=12.0,
            )

    def _start_worker(self):
        source = getattr(self.args, 'source', '0')
        camera_width = getattr(self.args, 'camera_width', 640)
        camera_height = getattr(self.args, 'camera_height', 480)
        camera_fps = getattr(self.args, 'camera_fps', 30)
        calibration_frames = getattr(self.args, 'calibration_frames', DEFAULT_CALIBRATION_FRAMES)
        frame_rotation = getattr(self.args, 'rotation', 180)
        if INFERENCE_AVAILABLE:
            model = getattr(self.args, 'model', str(DEFAULT_MODEL))
            clip = getattr(self.args, 'alert_clip', None)
            stride = getattr(self.args, 'predict_stride', None)
            self.worker = InferenceWorker(
                source, model, clip, stride, camera_width, camera_height, camera_fps,
                calibration_frames, frame_rotation
            )
        else:
            self.worker = DemoWorker(
                source, camera_width, camera_height, camera_fps, frame_rotation
            )

        self.worker.signals.frame_ready.connect(self._on_frame)
        self.worker.signals.error.connect(self._on_error)
        self.worker.start()

    def _start_normalization(self):
        self._last_error = ""
        if hasattr(self.worker, "start_calibration"):
            self.worker.start_calibration()
        self.audio.say("Calibrating. Please hold still.", key="normalize_start", min_interval=2.0)
        metrics = dict(self.cam_widget._metrics)
        metrics.update({
            "prob": None,
            "label": None,
            "calibrated": False,
            "calibration_state": "calibrating",
            "calibration_progress": 0.0,
        })
        self.cam_widget.set_metrics(metrics)

    def _on_frame(self, frame, metrics):
        self.cam_widget.set_frame(frame)
        metrics = dict(metrics)
        metrics["paused"] = self._paused
        if self._last_error:
            metrics["error"] = self._last_error
        self.cam_widget.set_metrics(metrics)
        self._handle_audio(metrics)

    def _handle_audio(self, metrics):
        if self._paused:
            return

        calibrated = bool(metrics.get("calibrated"))
        calibration_state = str(metrics.get("calibration_state") or "waiting")
        face_ok = bool(metrics.get("face_detected"))
        warmup = bool(metrics.get("warmup"))
        label = metrics.get("label")

        if not calibrated:
            if calibration_state != self._last_audio_calibration_state:
                if calibration_state == "waiting":
                    self.audio.say(
                        "Please face forward, then tap start.",
                        key="cal_wait",
                        min_interval=12.0,
                    )
                elif calibration_state == "calibrating":
                    self.audio.say(
                        "Hold still.",
                        key="cal_active",
                        min_interval=5.0,
                    )
                elif calibration_state == "no_face":
                    self.audio.say(
                        "Driver not visible.",
                        key="cal_no_face",
                        min_interval=4.0,
                    )
            elif calibration_state == "no_face":
                self.audio.say(
                    "Driver not visible.",
                    key="cal_no_face",
                    min_interval=6.0,
                )
            self._last_audio_calibration_state = calibration_state
            self._last_audio_calibrated = False
            return

        if calibrated and not self._last_audio_calibrated:
            self.audio.say("Calibration complete. Monitoring active.", key="cal_done")
        self._last_audio_calibrated = True
        self._last_audio_calibration_state = calibration_state

        if not face_ok:
            self.audio.say(
                "Driver not visible.",
                key="no_face",
                min_interval=7.0,
            )
            self._last_audio_face_ok = False
            self._last_audio_label = None
            return

        if self._last_audio_face_ok is False:
            self.audio.say("Driver visible. Monitoring resumed.", key="face_back", min_interval=5.0)
        self._last_audio_face_ok = True

        if label == 1 and not warmup:
            self.audio.say(
                "Attention. Drowsiness detected.",
                key="drowsy_alert",
                min_interval=self.audio.alert_interval,
                warning_tone=True,
            )
        elif label == 0 and self._last_audio_label == 1:
            self.audio.say("Driver alert.", key="alert_clear", min_interval=5.0)
        self._last_audio_label = label

    def _on_error(self, msg):
        self._last_error = msg[:80]
        self.audio.say("System alert.", key="system_error", min_interval=10.0)
        self.cam_widget.set_metrics({
            "prob": None,
            "label": None,
            "fps": 0.0,
            "face_detected": False,
            "warmup": False,
            "paused": self._paused,
            "error": self._last_error,
            "calibrated": bool(self.cam_widget._metrics.get("calibrated")),
            "calibration_state": self.cam_widget._metrics.get("calibration_state", "waiting"),
            "calibration_progress": self.cam_widget._metrics.get("calibration_progress", 0.0),
        })

    def _toggle_pause(self):
        self._paused = not self._paused
        if self._paused:
            self.worker.pause()
            self.audio.say("Paused.", key="paused")
        else:
            self.worker.resume()
            self.audio.say("Resumed.", key="resumed")
        metrics = dict(self.cam_widget._metrics)
        metrics["paused"] = self._paused
        self.cam_widget.set_metrics(metrics)

    def _tick_clock(self):
        self.cam_widget.update()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        elif key in (Qt.Key_Space, Qt.Key_P):
            self._toggle_pause()
        else:
            super().keyPressEvent(event)

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
    ap.add_argument("--rotation", type=int, default=180,
                    choices=[0, 90, 180, 270],
                    help="Rotate camera/video frames before inference and display")
    ap.add_argument("--calibration-frames", type=int, default=DEFAULT_CALIBRATION_FRAMES,
                    help="Detected face frames to collect for startup normalization")
    ap.add_argument("--mute-audio", action="store_true",
                    help="Disable voice prompts and drowsiness warning tone")
    ap.add_argument("--audio-alert-interval", type=float, default=8.0,
                    help="Minimum seconds between repeated drowsiness audio alerts")
    ap.add_argument("--audio-engine", default="piper",
                    choices=["auto", "piper", "flite", "espeak-ng", "espeak", "say", "spd-say"],
                    help="Voice prompt engine. Default uses Piper neural TTS")
    ap.add_argument("--audio-voice", default=None,
                    help=("Optional voice. For Piper, pass a .onnx voice model path; "
                          "for espeak-ng, use names like 'en-us+f4'."))
    ap.add_argument("--piper-model", dest="audio_voice", default=None,
                    help="Alias for --audio-voice when using Piper")
    ap.add_argument("--audio-rate", type=int, default=132,
                    help="Voice speaking rate for espeak/say engines")
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
