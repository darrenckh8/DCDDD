#!/usr/bin/env python3
"""
Rule-based dashcam UI
=====================
Standalone drowsiness detector using only EAR, MAR, head pose, and PERCLOS.
No Keras/TFLite model is loaded.

Usage
-----
  python rule_based_inference.py
  python rule_based_inference.py --source path/to/video.mp4 --no-fullscreen
  python rule_based_inference.py --ear-threshold 0.20 --perclos-threshold 0.35
"""

import argparse
import collections
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
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
    from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal, QObject, QRect, QLibraryInfo
    from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor, QFont
    from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QSizePolicy
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Missing dependency '{exc.name}'. "
        "Install PyQt5 or activate your project virtualenv."
    ) from exc


BASE_DIR = Path(__file__).resolve().parent
MP_MODEL_PATH = BASE_DIR / "face_landmarker.task"

RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_UPPER = [82, 13, 312]
MOUTH_LOWER = [87, 14, 317]
POSE_LM_IDS = [1, 152, 263, 33, 61, 291]
MOUTH_OUTLINE = [61, 82, 13, 312, 291, 317, 14, 87]

MODEL_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -63.6, -12.5),
    (-43.3, 32.7, -26.0),
    (43.3, 32.7, -26.0),
    (-28.9, -28.9, -24.1),
    (28.9, -28.9, -24.1),
], dtype=np.float64)
DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)

P = {
    "green": "#2e7d32",
    "red": "#c62828",
    "amber": "#f9a825",
    "white": "#f1f3f4",
    "dim": "#a8b0b8",
    "dimmer": "#7b848d",
}

PIPER_MODEL_CANDIDATES = (
    "piper_voice.onnx",
    "voice.onnx",
    "en_US-amy-medium.onnx",
    "en_US-lessac-medium.onnx",
    "en_US-ryan-medium.onnx",
)

AUDIO_PRELOAD_PHRASES = (
    "Attention. Drowsiness detected.",
    "Driver not visible.",
    "Please face forward, then tap start.",
    "Calibrating. Please hold still.",
    "Hold still.",
    "Calibration complete. Monitoring active.",
    "Driver visible. Monitoring resumed.",
    "Driver alert.",
    "System alert.",
    "Paused.",
    "Resumed.",
    "Calibrating. Please face forward and hold still.",
)


class AudioNotifier:
    """Small non-blocking voice/tone helper matching ui.py."""

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
        self._tone_busy = False
        self._tone_lock = threading.Lock()
        self._cache_dir = tempfile.TemporaryDirectory(prefix="rule_dashcam_tts_")
        self._speech_cache = {}
        self._speech_cache_pending = set()
        self._cache_lock = threading.Lock()
        self._synth_lock = threading.Lock()
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

    def preload(self, phrases):
        if not self.enabled or not self._can_cache_speech():
            return
        thread = threading.Thread(
            target=self._preload_worker,
            args=(tuple(dict.fromkeys(phrases)),),
            daemon=True,
        )
        thread.start()

    def _preload_worker(self, phrases):
        for text in phrases:
            self._ensure_cached_speech(text)

    def _can_cache_speech(self):
        return (
            self._speaker is not None and
            Path(self._speaker).name == "piper" and
            self._piper_model is not None and
            self._piper_config is not None
        )

    def _cached_speech_path(self, text):
        with self._cache_lock:
            path = self._speech_cache.get(text)
        if path and os.path.exists(path):
            return path
        return None

    def _ensure_cached_speech(self, text):
        if not self._can_cache_speech():
            return None

        cached = self._cached_speech_path(text)
        if cached:
            return cached

        with self._cache_lock:
            if text in self._speech_cache_pending:
                return None
            self._speech_cache_pending.add(text)

        try:
            index = len(self._speech_cache)
            wav_path = os.path.join(self._cache_dir.name, f"tts_{index:02d}.wav")
            if self._synthesize_piper_to_file(text, wav_path):
                with self._cache_lock:
                    self._speech_cache[text] = wav_path
                return wav_path
        finally:
            with self._cache_lock:
                self._speech_cache_pending.discard(text)
        return None

    def _synthesize_piper_to_file(self, text, wav_path):
        with self._synth_lock:
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
            return False
        return True

    def _play_cached_speech_async(self, wav_path):
        thread = threading.Thread(
            target=self._play_wav_file,
            args=(wav_path, 8),
            daemon=True,
        )
        thread.start()

    def _play_wav_file(self, wav_path, timeout=8):
        player = self._audio_player_cmd(wav_path)
        if player:
            result = subprocess.run(
                player,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=timeout,
                check=False,
            )
            if result.returncode != 0:
                self._warn_once("wav_play", "Audio: WAV playback failed.")
            return
        self._warn_once("wav_player_runtime", "Audio: No WAV player available for speech output.")

    def say(self, text, key=None, min_interval=0.0, warning_tone=False):
        if not self.enabled:
            return

        now = time.monotonic()
        key = key or text
        last = self._last_spoken.get(key, -1e9)
        if now - last < float(min_interval):
            return

        tone_started = False
        if warning_tone:
            tone_started = self._play_warning_tone_async()

        if not self._speaker:
            if tone_started or not warning_tone:
                self._last_spoken[key] = now
            return

        cached_path = self._cached_speech_path(text)
        if cached_path and warning_tone:
            self._last_spoken[key] = now
            self._play_cached_speech_async(cached_path)
            return

        with self._lock:
            if self._busy:
                if tone_started or not warning_tone:
                    self._last_spoken[key] = now
                return
            self._busy = True

        self._last_spoken[key] = now
        thread = threading.Thread(
            target=self._speak,
            args=(text, cached_path),
            daemon=True,
        )
        thread.start()

    def _play_warning_tone_async(self):
        with self._tone_lock:
            if self._tone_busy:
                return False
            self._tone_busy = True

        thread = threading.Thread(target=self._tone_worker, daemon=True)
        thread.start()
        return True

    def _tone_worker(self):
        try:
            self._play_warning_tone_blocking()
        finally:
            with self._tone_lock:
                self._tone_busy = False

    def _speak(self, text, cached_path=None):
        if not self._speaker:
            return
        try:
            name = Path(self._speaker).name
            if cached_path:
                self._play_wav_file(cached_path, timeout=8)
                return
            if name == "piper":
                wav_path = None
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fh:
                        wav_path = fh.name
                    if not self._synthesize_piper_to_file(text, wav_path):
                        return
                    self._play_wav_file(wav_path, timeout=8)
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


def _dist(a, b):
    dx, dy = a[0] - b[0], a[1] - b[1]
    return float(np.sqrt(dx * dx + dy * dy))


def wrap_angle_deg(value):
    wrapped = (np.asarray(value, dtype=np.float64) + 180.0) % 360.0 - 180.0
    if wrapped.shape == ():
        return float(wrapped)
    return wrapped


def circular_mean_deg(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    radians = np.deg2rad(arr)
    mean_sin = np.mean(np.sin(radians), axis=0)
    mean_cos = np.mean(np.cos(radians), axis=0)
    return wrap_angle_deg(np.rad2deg(np.arctan2(mean_sin, mean_cos)))


def angle_delta_deg(value, baseline):
    return wrap_angle_deg(float(value) - float(baseline))


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


def compute_landmark_head_pose(lm):
    eye_left = lm[33]
    eye_right = lm[263]
    nose = lm[1]
    chin = lm[152]
    mouth_center = (lm[MOUTH_LEFT] + lm[MOUTH_RIGHT]) / 2.0
    eye_center = (eye_left + eye_right) / 2.0
    eye_width = max(_dist(eye_left, eye_right), 1.0)
    face_height = max(_dist(eye_center, chin), eye_width)

    face_center = (eye_center * 0.65) + (mouth_center * 0.35)
    yaw = ((nose[0] - face_center[0]) / eye_width) * 85.0
    pitch = ((nose[1] - eye_center[1]) / face_height) * 110.0
    roll = np.degrees(np.arctan2(eye_right[1] - eye_left[1], eye_right[0] - eye_left[0]))
    roll = wrap_angle_deg(roll)
    if roll > 90.0:
        roll -= 180.0
    elif roll < -90.0:
        roll += 180.0
    return float(pitch), float(yaw), float(roll)


def solve_pnp_pose(lm, width, height):
    pts = np.array([lm[i] for i in POSE_LM_IDS], dtype=np.float64)
    focal = float(width)
    camera = np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    ok, rvec, tvec = cv2.solvePnP(
        MODEL_3D,
        pts,
        camera,
        DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0, None, None, camera
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
    return float(pitch), float(yaw), float(roll), rvec, tvec, camera


def compute_head_pose(lm, width, height):
    return compute_landmark_head_pose(lm)


def extract_measurements(face_landmarks, width, height):
    lm = np.array([(p.x * width, p.y * height) for p in face_landmarks], dtype=np.float64)
    ear_l = compute_ear(lm, LEFT_EYE)
    ear_r = compute_ear(lm, RIGHT_EYE)
    ear = (ear_l + ear_r) / 2.0
    mar = compute_mar(lm)
    pitch, yaw, roll = compute_landmark_head_pose(lm)
    return {
        "ear_l": ear_l,
        "ear_r": ear_r,
        "ear": ear,
        "mar": mar,
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
        "landmarks": lm,
    }


def _pixel_point(lm, idx, width, height):
    x = int(np.clip(round(float(lm[idx][0])), 0, width - 1))
    y = int(np.clip(round(float(lm[idx][1])), 0, height - 1))
    return x, y


def _draw_landmark_group(frame, lm, indexes, color, closed=True, thickness=2):
    height, width = frame.shape[:2]
    pts = np.array([_pixel_point(lm, idx, width, height) for idx in indexes], dtype=np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts.reshape((-1, 1, 2))], closed, color, thickness, cv2.LINE_AA)
    for point in pts:
        cv2.circle(frame, tuple(point), max(2, thickness + 1), color, -1, cv2.LINE_AA)


def draw_feature_overlay(frame, measurements, rule):
    lm = measurements.get("landmarks")
    if lm is None:
        return frame

    height, width = frame.shape[:2]
    closed = bool(rule.get("closed"))
    yawn = bool(rule.get("yawn"))
    bad_pose = bool(rule.get("bad_pose"))

    eye_color = (0, 0, 255) if closed else (0, 220, 80)
    mouth_color = (0, 0, 255) if yawn else (0, 200, 255)
    pose_color = (0, 0, 255) if bad_pose else (255, 200, 0)
    thickness = max(1, round(min(width, height) / 320))

    for eye in (LEFT_EYE, RIGHT_EYE):
        _draw_landmark_group(frame, lm, eye, eye_color, closed=True, thickness=thickness)
        p0 = _pixel_point(lm, eye[0], width, height)
        p3 = _pixel_point(lm, eye[3], width, height)
        p15 = _pixel_point(lm, eye[1], width, height)
        p54 = _pixel_point(lm, eye[5], width, height)
        p24 = _pixel_point(lm, eye[2], width, height)
        p42 = _pixel_point(lm, eye[4], width, height)
        cv2.line(frame, p0, p3, eye_color, thickness, cv2.LINE_AA)
        cv2.line(frame, p15, p54, eye_color, thickness, cv2.LINE_AA)
        cv2.line(frame, p24, p42, eye_color, thickness, cv2.LINE_AA)

    _draw_landmark_group(frame, lm, MOUTH_OUTLINE, mouth_color, closed=True, thickness=thickness)
    cv2.line(frame, _pixel_point(lm, MOUTH_LEFT, width, height), _pixel_point(lm, MOUTH_RIGHT, width, height),
             mouth_color, thickness, cv2.LINE_AA)
    for upper, lower in zip(MOUTH_UPPER, MOUTH_LOWER):
        cv2.line(frame, _pixel_point(lm, upper, width, height), _pixel_point(lm, lower, width, height),
                 mouth_color, thickness, cv2.LINE_AA)

    nose = _pixel_point(lm, 1, width, height)
    axis_len = max(35.0, min(width, height) * 0.12)
    roll = float(measurements.get("roll", 0.0))
    roll_rad = np.deg2rad(roll)
    x_vec = np.array([np.cos(roll_rad), np.sin(roll_rad)], dtype=np.float64) * axis_len
    y_vec = np.array([-np.sin(roll_rad), np.cos(roll_rad)], dtype=np.float64) * axis_len
    z_vec = np.array([
        float(rule.get("yaw_delta") or 0.0) * 1.7,
        float(rule.get("pitch_delta") or 0.0) * 1.7,
    ], dtype=np.float64)
    if np.linalg.norm(z_vec) < axis_len * 0.35:
        z_vec = np.array([0.0, -axis_len * 0.35], dtype=np.float64)

    def clipped_axis_point(offset):
        point = np.array(nose, dtype=np.float64) + offset
        return (
            int(np.clip(round(float(point[0])), 0, width - 1)),
            int(np.clip(round(float(point[1])), 0, height - 1)),
        )

    cv2.arrowedLine(frame, nose, clipped_axis_point(x_vec), (0, 0, 255), thickness + 1, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(frame, nose, clipped_axis_point(y_vec), (0, 220, 80), thickness + 1, cv2.LINE_AA, tipLength=0.25)
    cv2.arrowedLine(frame, nose, clipped_axis_point(z_vec), pose_color, thickness + 1, cv2.LINE_AA, tipLength=0.25)

    text = (
        f"EAR {float(measurements.get('ear', 0.0)):.2f}  "
        f"MAR {float(measurements.get('mar', 0.0)):.2f}  "
        f"dP/Y/R {float(rule.get('pitch_delta') or 0.0):+.0f}/"
        f"{float(rule.get('yaw_delta') or 0.0):+.0f}/"
        f"{float(rule.get('roll_delta') or 0.0):+.0f}"
    )
    label_anchor = _pixel_point(lm, 10, width, height)
    text_x_max = max(6, width - 220)
    text_pos = (int(np.clip(label_anchor[0] - 90, 6, text_x_max)), max(24, label_anchor[1] - 20))
    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), thickness + 1, cv2.LINE_AA)
    return frame


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
        self.pose_baseline = None
        self.pose_samples = []

    def start_calibration(self):
        self.closed_window.clear()
        self.reset_face_state()
        self.missing_frames = 0
        self.pose_baseline = None
        self.pose_samples = []

    def reset_face_state(self):
        self.closed_frames = 0
        self.yawn_frames = 0
        self.pose_frames = 0

    @property
    def calibration_required(self):
        return max(0, int(self.args.pose_calibration_seconds * self.fps))

    @property
    def calibrated(self):
        return self.pose_baseline is not None or self.calibration_required == 0

    def _update_pose_baseline(self, measurements):
        if self.calibrated:
            return True
        self.pose_samples.append([measurements["pitch"], measurements["yaw"], measurements["roll"]])
        if len(self.pose_samples) >= self.calibration_required:
            self.pose_baseline = circular_mean_deg(np.array(self.pose_samples, dtype=np.float32))
            return True
        return False

    def waiting_rule(self, face_ok):
        return {
            "label": None,
            "reason": "WAITING",
            "perclos": 0.0,
            "closed": False,
            "yawn": False,
            "bad_pose": False,
            "calibrated": self.calibrated,
            "calibration_state": "ready" if self.calibrated else "waiting",
            "calibration_progress": self.calibration_progress(),
            "pitch_delta": 0.0,
            "yaw_delta": 0.0,
            "roll_delta": 0.0,
            "warmup": True,
        }

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
                "calibrated": self.calibrated,
                "calibration_state": "ready" if self.calibrated else "no_face",
                "calibration_progress": self.calibration_progress(),
                "pitch_delta": 0.0,
                "yaw_delta": 0.0,
                "roll_delta": 0.0,
                "warmup": True,
            }

        self.missing_frames = 0
        if not self._update_pose_baseline(measurements):
            return {
                "label": None,
                "reason": "CALIBRATING",
                "perclos": 0.0,
                "closed": False,
                "yawn": False,
                "bad_pose": False,
                "calibrated": False,
                "calibration_state": "calibrating",
                "calibration_progress": self.calibration_progress(),
                "pitch_delta": 0.0,
                "yaw_delta": 0.0,
                "roll_delta": 0.0,
                "warmup": True,
            }

        ear = measurements["ear"]
        mar = measurements["mar"]
        if self.pose_baseline is None:
            pitch_delta = wrap_angle_deg(measurements["pitch"])
            yaw_delta = wrap_angle_deg(measurements["yaw"])
            roll_delta = wrap_angle_deg(measurements["roll"])
        else:
            pitch_delta = angle_delta_deg(measurements["pitch"], self.pose_baseline[0])
            yaw_delta = angle_delta_deg(measurements["yaw"], self.pose_baseline[1])
            roll_delta = angle_delta_deg(measurements["roll"], self.pose_baseline[2])

        closed = ear < self.args.ear_threshold
        yawn = mar > self.args.mar_threshold
        bad_pose = (
            abs(pitch_delta) > self.args.pitch_threshold or
            abs(yaw_delta) > self.args.yaw_threshold or
            abs(roll_delta) > self.args.roll_threshold
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

        return {
            "label": 1 if reasons else 0,
            "reason": " + ".join(reasons) if reasons else "OK",
            "perclos": perclos,
            "closed": closed,
            "yawn": yawn,
            "bad_pose": bad_pose,
            "calibrated": True,
            "calibration_state": "ready",
            "calibration_progress": 1.0,
            "pitch_delta": float(pitch_delta),
            "yaw_delta": float(yaw_delta),
            "roll_delta": float(roll_delta),
            "warmup": False,
        }

    def calibration_progress(self):
        required = self.calibration_required
        if required <= 0:
            return 1.0
        return min(1.0, len(self.pose_samples) / required)


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


class RuleSignals(QObject):
    frame_ready = pyqtSignal(object, object)
    error = pyqtSignal(str)


class RuleWorker(QThread):
    def __init__(self, args, parent=None):
        super().__init__(parent)
        self.args = args
        self.signals = RuleSignals()
        self._running = True
        self._paused = False
        self._calibration_requested = bool(getattr(args, "auto_calibrate", False))
        self._calibration_reset_pending = False

    def pause(self):
        self._paused = True

    def resume(self):
        self._paused = False

    def start_calibration(self):
        self._calibration_requested = True
        self._calibration_reset_pending = True

    def stop(self):
        self._running = False
        self.wait(3000)

    def run(self):
        cap = None
        landmarker = None
        try:
            cap = open_source(
                self.args.source,
                self.args.camera_width,
                self.args.camera_height,
                self.args.camera_fps,
            )
            landmarker = create_landmarker()
            state = RuleState(self.args, cap.fps)
            ts_ms = 0.0
            frame_count = 0
            t_prev = time.perf_counter()
            fps_display = 0.0

            while self._running:
                if self._paused:
                    time.sleep(0.05)
                    continue
                if self._calibration_reset_pending:
                    state.start_calibration()
                    self._calibration_reset_pending = False

                ok, frame = cap.read()
                if not ok:
                    if cap.loop_source:
                        cap.reset()
                        continue
                    time.sleep(0.02)
                    continue

                frame = rotate_frame(frame, self.args.rotation)
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
                    measurements = extract_measurements(result.face_landmarks[0], width, height)
                else:
                    measurements = {}

                if not state.calibrated and not self._calibration_requested:
                    rule = state.waiting_rule(face_ok)
                elif face_ok:
                    rule = state.update(measurements, True)
                else:
                    rule = state.update(None, False)

                if face_ok and not getattr(self.args, "hide_face_overlay", False):
                    draw_feature_overlay(frame, measurements, rule)

                metrics = {
                    "label": rule["label"],
                    "reason": rule["reason"],
                    "perclos": rule["perclos"],
                    "closed": rule["closed"],
                    "yawn": rule["yawn"],
                    "bad_pose": rule["bad_pose"],
                    "calibrated": rule["calibrated"],
                    "calibration_state": rule["calibration_state"],
                    "calibration_progress": rule["calibration_progress"],
                    "warmup": rule["warmup"],
                    "face_detected": face_ok,
                    "fps": fps_display,
                    "frame_count": frame_count,
                    "ear": float(measurements.get("ear", 0.0)),
                    "ear_l": float(measurements.get("ear_l", 0.0)),
                    "ear_r": float(measurements.get("ear_r", 0.0)),
                    "mar": float(measurements.get("mar", 0.0)),
                    "pitch": float(measurements.get("pitch", 0.0)),
                    "yaw": float(measurements.get("yaw", 0.0)),
                    "roll": float(measurements.get("roll", 0.0)),
                    "pitch_delta": rule["pitch_delta"],
                    "yaw_delta": rule["yaw_delta"],
                    "roll_delta": rule["roll_delta"],
                }
                self.signals.frame_ready.emit(frame, metrics)
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            if cap is not None:
                cap.release()
            if landmarker is not None:
                landmarker.close()


class CameraWidget(QWidget):
    calibration_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap = None
        self._calibrate_rect = QRect()
        self._metrics = {
            "label": None,
            "reason": "Starting",
            "perclos": 0.0,
            "face_detected": False,
            "warmup": True,
            "fps": 0.0,
            "frame_count": 0,
            "ear": 0.0,
            "mar": 0.0,
            "pitch_delta": 0.0,
            "yaw_delta": 0.0,
            "roll_delta": 0.0,
            "calibrated": False,
            "calibration_state": "waiting",
            "calibration_progress": 0.0,
            "paused": False,
            "error": "",
        }
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(240, 180)

    def set_frame(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, width, height, 3 * width, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(image)
        self.update()

    def set_metrics(self, metrics):
        self._metrics = dict(metrics)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        width, height = self.width(), self.height()
        painter.fillRect(0, 0, width, height, QColor("#050505"))

        if self._pixmap is not None and not self._pixmap.isNull():
            scaled = self._pixmap.scaled(width, height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            x = (width - scaled.width()) // 2
            y = (height - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

        compact = width < 720 or height < 430
        tiny = width < 360 or height < 260
        margin = 8 if compact else 14
        top_h = 24 if tiny else (28 if compact else 38)
        bottom_h = 34 if tiny else (42 if compact else 58)

        label = self._metrics.get("label")
        reason = str(self._metrics.get("reason") or "OK")
        face_ok = bool(self._metrics.get("face_detected"))
        warmup = bool(self._metrics.get("warmup"))
        calibrated = bool(self._metrics.get("calibrated"))
        calibration_state = str(self._metrics.get("calibration_state") or "waiting")
        progress = float(self._metrics.get("calibration_progress") or 0.0)
        paused = bool(self._metrics.get("paused"))
        error = str(self._metrics.get("error") or "")
        perclos = max(0.0, min(1.0, float(self._metrics.get("perclos") or 0.0)))
        fps = float(self._metrics.get("fps") or 0.0)

        painter.fillRect(0, 0, width, top_h, QColor(0, 0, 0, 150))
        painter.setFont(QFont("DejaVu Sans", 8 if tiny else (9 if compact else 11), QFont.Bold))
        painter.setPen(QColor("#ffffff"))
        timestamp = time.strftime("%H:%M:%S") if tiny else time.strftime("%Y-%m-%d  %H:%M:%S")
        painter.drawText(QRect(margin, 0, width - 82, top_h), Qt.AlignVCenter | Qt.AlignLeft, timestamp)
        painter.setPen(QColor(P["red"]))
        rec_w = 46 if tiny else (54 if compact else 66)
        painter.drawText(QRect(width - rec_w - margin, 0, rec_w, top_h), Qt.AlignVCenter | Qt.AlignRight, "REC")

        if error:
            status = "Error"
            color = QColor(P["red"])
            right_text = "--"
        elif not calibrated:
            status = "CAL" if tiny else "Calibrate"
            color = QColor(P["dimmer"])
            right_text = f"{progress * 100:.0f}%"
        elif not face_ok:
            status = "NO FACE" if tiny else "No face"
            color = QColor(P["amber"])
            right_text = "--"
        elif warmup:
            status = "WAIT" if tiny else "Warming up"
            color = QColor(P["dimmer"])
            right_text = "--"
        elif label == 1:
            status = "ALERT" if tiny else reason
            color = QColor(P["red"])
            right_text = f"PERCLOS {perclos * 100:.0f}%"
        else:
            status = "OK" if tiny else "Driver alert"
            color = QColor(P["green"])
            right_text = f"PERCLOS {perclos * 100:.0f}%"

        painter.fillRect(0, height - bottom_h, width, bottom_h, QColor(0, 0, 0, 165))
        painter.setFont(QFont("DejaVu Sans", 11 if tiny else (12 if compact else 17), QFont.Bold))
        painter.setPen(color)
        right_w = 74 if tiny else (112 if compact else 150)
        status_rect = QRect(margin, height - bottom_h + 2, width - (2 * margin) - right_w - 12, bottom_h - 12)
        painter.drawText(status_rect, Qt.AlignVCenter | Qt.AlignLeft, status)

        painter.setPen(QColor("#ffffff"))
        painter.setFont(QFont("DejaVu Sans", 9 if tiny else (10 if compact else 13), QFont.Bold))
        right_rect = QRect(width - right_w - margin, height - bottom_h + 2, right_w, bottom_h - 12)
        painter.drawText(right_rect, Qt.AlignVCenter | Qt.AlignRight, right_text)

        bar_x = margin
        bar_w = max(80, width - (2 * margin) - right_w - 18)
        bar_y = height - 8 if tiny else (height - 12 if compact else height - 16)
        bar_h = 4 if tiny else (5 if compact else 7)
        painter.fillRect(bar_x, bar_y, bar_w, bar_h, QColor(90, 90, 90, 170))
        fill = progress if not calibrated else perclos
        if not calibrated or face_ok:
            painter.fillRect(bar_x, bar_y, int(bar_w * max(0.0, min(1.0, fill))), bar_h, color)

        if not tiny:
            small = f"FPS {fps:.0f}   Face {'yes' if face_ok else 'no'}"
            if not compact:
                small += (
                    f"   EAR {float(self._metrics.get('ear') or 0.0):.3f}"
                    f"   MAR {float(self._metrics.get('mar') or 0.0):.3f}"
                    f"   dPose {float(self._metrics.get('pitch_delta') or 0.0):+.0f}/"
                    f"{float(self._metrics.get('yaw_delta') or 0.0):+.0f}/"
                    f"{float(self._metrics.get('roll_delta') or 0.0):+.0f}"
                )
            painter.setFont(QFont("DejaVu Sans", 8 if compact else 10))
            painter.setPen(QColor(P["dim"]))
            painter.drawText(margin, height - bottom_h + (12 if compact else 17), small)

        if error:
            err_h = 34 if compact else 46
            err_y = top_h + 8
            painter.fillRect(0, err_y, width, err_h, QColor(120, 0, 0, 210))
            painter.setFont(QFont("DejaVu Sans", 10 if compact else 13, QFont.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRect(margin, err_y, width - 2 * margin, err_h), Qt.AlignVCenter | Qt.AlignLeft, error)

        if paused:
            pause_h = 44 if compact else 64
            pause_y = max(top_h + 6, height // 2 - pause_h // 2)
            painter.fillRect(0, pause_y, width, pause_h, QColor(0, 0, 0, 185))
            painter.setFont(QFont("DejaVu Sans", 16 if compact else 24, QFont.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(QRect(0, pause_y, width, pause_h), Qt.AlignCenter, "PAUSED")

        if not calibrated:
            self._draw_calibration_overlay(
                painter, width, height, compact, tiny, face_ok,
                calibration_state, progress,
            )

        painter.end()

    def _draw_calibration_overlay(self, painter, width, height, compact, tiny, face_ok, state, progress):
        painter.fillRect(0, 0, width, height, QColor(0, 0, 0, 150))
        panel_w = min(width - 28, 420 if not compact else 320)
        panel_h = 170 if not tiny else 132
        panel_x = max(14, (width - panel_w) // 2)
        panel_y = max(38, (height - panel_h) // 2)
        panel = QRect(panel_x, panel_y, panel_w, panel_h)
        painter.fillRect(panel, QColor(10, 10, 10, 215))

        title = "CALIBRATE" if tiny else "Calibrate Driver"
        painter.setFont(QFont("DejaVu Sans", 15 if tiny else 20, QFont.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRect(panel_x, panel_y + 14, panel_w, 34), Qt.AlignCenter, title)

        if state == "waiting":
            message = "Face forward, eyes open"
        elif state == "no_face":
            message = "Face not found"
        else:
            message = "Hold still"
        painter.setFont(QFont("DejaVu Sans", 9 if tiny else 12, QFont.Bold))
        painter.setPen(QColor(P["green"] if face_ok else P["amber"]))
        painter.drawText(QRect(panel_x + 10, panel_y + 50, panel_w - 20, 28), Qt.AlignCenter, message)

        button_w = min(panel_w - 36, 300)
        button_h = 42 if tiny else 50
        button_x = panel_x + (panel_w - button_w) // 2
        button_y = panel_y + panel_h - button_h - 18
        button = QRect(button_x, button_y, button_w, button_h)

        if state == "waiting":
            self._calibrate_rect = button
            painter.fillRect(button, QColor(P["green"]))
            painter.setFont(QFont("DejaVu Sans", 12 if tiny else 15, QFont.Bold))
            painter.setPen(QColor("#ffffff"))
            painter.drawText(button, Qt.AlignCenter, "START")
            return

        self._calibrate_rect = QRect()
        bar = QRect(button_x, button_y + button_h // 2 - 4, button_w, 8)
        clamped = max(0.0, min(1.0, progress))
        painter.fillRect(bar, QColor(80, 80, 80, 220))
        painter.fillRect(bar.x(), bar.y(), int(bar.width() * clamped), bar.height(), QColor(P["green"]))
        painter.setFont(QFont("DejaVu Sans", 10 if tiny else 12, QFont.Bold))
        painter.setPen(QColor("#ffffff"))
        painter.drawText(QRect(button_x, button_y - 18, button_w, 18), Qt.AlignCenter, f"{int(clamped * 100)}%")

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
        self.audio.preload(AUDIO_PRELOAD_PHRASES)
        self.setStyleSheet(GLOBAL_QSS)
        self.setWindowTitle("Rule-Based Dashcam")
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
        self.cam_widget.calibration_requested.connect(self._start_calibration)
        root.addWidget(self.cam_widget)

    def _play_startup_instruction(self):
        if getattr(self.args, "auto_calibrate", False):
            self.audio.say("Calibrating. Please face forward and hold still.", key="startup")
        else:
            self.audio.say(
                "Please face forward, then tap start.",
                key="cal_wait",
                min_interval=12.0,
            )

    def _start_worker(self):
        self.worker = RuleWorker(self.args)
        self.worker.signals.frame_ready.connect(self._on_frame)
        self.worker.signals.error.connect(self._on_error)
        self.worker.start()

    def _start_calibration(self):
        self._last_error = ""
        if hasattr(self.worker, "start_calibration"):
            self.worker.start_calibration()
        self.audio.say("Calibrating. Please hold still.", key="cal_start", min_interval=2.0)
        metrics = dict(self.cam_widget._metrics)
        metrics.update({
            "label": None,
            "reason": "CALIBRATING",
            "calibrated": False,
            "calibration_state": "calibrating",
            "calibration_progress": 0.0,
            "warmup": True,
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

    def _on_error(self, message):
        self._last_error = message[:100]
        self.audio.say("System alert.", key="system_error", min_interval=10.0)
        metrics = dict(self.cam_widget._metrics)
        metrics.update({
            "label": None,
            "reason": "Error",
            "face_detected": False,
            "paused": self._paused,
            "error": self._last_error,
        })
        self.cam_widget.set_metrics(metrics)

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


def parse_args():
    parser = argparse.ArgumentParser(description="Rule-based dashcam UI")
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
    parser.add_argument("--pose-calibration-seconds", type=float, default=2.0,
                        help="Seconds used to learn neutral head pose after START")
    parser.add_argument("--pitch-threshold", type=float, default=28.0, help="Relative pitch threshold in degrees")
    parser.add_argument("--yaw-threshold", type=float, default=38.0, help="Relative yaw threshold in degrees")
    parser.add_argument("--roll-threshold", type=float, default=35.0, help="Relative roll threshold in degrees")
    parser.add_argument("--no-face-reset", type=float, default=0.5, help="Seconds without face before clearing state")
    parser.add_argument("--hide-face-overlay", action="store_true",
                        help="Hide EAR/MAR landmarks and head-pose axes on the camera feed")
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Start head-pose calibration immediately without tapping START")
    parser.add_argument("--mute-audio", action="store_true",
                        help="Disable voice prompts and drowsiness warning tone")
    parser.add_argument("--audio-alert-interval", type=float, default=8.0,
                        help="Minimum seconds between repeated drowsiness audio alerts")
    parser.add_argument("--audio-engine", default="piper",
                        choices=["auto", "piper", "flite", "espeak-ng", "espeak", "say", "spd-say"],
                        help="Voice prompt engine. Default uses Piper neural TTS")
    parser.add_argument("--audio-voice", default=None,
                        help=("Optional voice. For Piper, pass a .onnx voice model path; "
                              "for espeak-ng, use names like 'en-us+f4'."))
    parser.add_argument("--piper-model", dest="audio_voice", default=None,
                        help="Alias for --audio-voice when using Piper")
    parser.add_argument("--audio-rate", type=int, default=132,
                        help="Voice speaking rate for espeak/say engines")
    parser.add_argument("--qt-platform", default=None,
                        choices=["xcb", "wayland", "eglfs", "linuxfb", "offscreen", "minimal"],
                        help="Override Qt platform plugin if auto-detection is wrong")
    parser.add_argument("--no-fullscreen", action="store_true", help="Run in a window instead of full screen")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    configure_qt_environment(args.qt_platform)

    app = QApplication(sys.argv)
    app.setApplicationName("Rule-Based Dashcam")

    window = MainWindow(args)
    if args.no_fullscreen:
        window.resize(1024, 600)
        window.show()
    else:
        window.showFullScreen()

    sys.exit(app.exec_())
