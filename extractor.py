"""
Feature extractor for UTA-RLDD drowsiness detection dataset.

Extracts per-frame facial features from video using MediaPipe FaceMesh
(478 landmarks with iris refinement) and cv2.solvePnP for head pose.

Features (10):
  EAR_Left, EAR_Right, EAR_Avg  — Eye Aspect Ratio (Soukupová & Čech)
  MAR                            — Mouth Aspect Ratio (3 vertical pairs)
  PUC_Left, PUC_Right            — Eyelid opening at pupil center / eye width
  MUC                            — Mouth center opening / mouth width
  Pitch, Yaw, Roll               — Head pose Euler angles (degrees, solvePnP)

Dataset layout:
  dataset/{01..60}/{0,5,10}.{mov,mp4}
  0 = alert (→ label 0), 5 = low drowsy (skipped), 10 = drowsy (→ label 1)

Output:
  uta_rldd_features.csv
  Columns: Subject, Video_File, Frame, Label, <10 features>

Usage:
  python extractor.py
"""

import os
import re
import time

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
OUTPUT_CSV = BASE_DIR / "uta_rldd_features.csv"
MODEL_PATH = str(BASE_DIR / "face_landmarker.task")

# Filename stem → output label  (skip "5")
LABEL_MAP = {"0": 0, "10": 1}
VIDEO_EXTS = {".mov", ".mp4", ".avi", ".mkv", ".m4v"}

# Regex to parse stems like "0", "10", "10_1", "0_2"
_STEM_RE = re.compile(r'^(0|10)(?:_(\d+))?$')

CPU_COUNT = max(1, os.cpu_count() or 1)
NUM_WORKERS = int(os.environ.get("EXTRACTOR_WORKERS", str(CPU_COUNT)))
NUM_WORKERS = max(1, NUM_WORKERS)

# MediaPipe FaceMesh
MP_DET_CONF = 0.5
MP_TRACK_CONF = 0.5

# Output columns
COLUMNS = [
    'Subject', 'Video_File', 'Frame', 'Label',
    'EAR_Left', 'EAR_Right', 'EAR_Avg', 'MAR',
    'PUC_Left', 'PUC_Right', 'MUC',
    'Pitch', 'Yaw', 'Roll',
]
FEATURE_NAMES = COLUMNS[4:]


# ---------------------------------------------------------------------------
# MediaPipe FaceMesh landmark indices
# ---------------------------------------------------------------------------
# Eye landmarks for EAR (6-point: outer, up1, up2, inner, low2, low1)
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [263, 387, 385, 362, 380, 373]

# Upper/lower lid center for PUC
RIGHT_EYE_UC, RIGHT_EYE_LC = 159, 145
LEFT_EYE_UC, LEFT_EYE_LC = 386, 374

# Mouth landmarks for MAR (3 vertical pairs) and MUC (center pair)
MOUTH_LEFT, MOUTH_RIGHT = 61, 291
MOUTH_UPPER = [82, 13, 312]
MOUTH_LOWER = [87, 14, 317]

# Head pose: solvePnP landmark indices and 3D model (mm)
POSE_LM_IDS = [1, 152, 263, 33, 61, 291]
MODEL_3D = np.array([
    (0.0,    0.0,    0.0),      # nose tip
    (0.0,  -63.6,  -12.5),     # chin
    (-43.3,  32.7,  -26.0),    # left eye outer corner
    (43.3,  32.7,  -26.0),     # right eye outer corner
    (-28.9, -28.9,  -24.1),    # left mouth corner
    (28.9, -28.9,  -24.1),     # right mouth corner
], dtype=np.float64)

DIST_COEFFS = np.zeros((4, 1), dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _dist(a, b):
    """Euclidean distance between two 2D points (inline-friendly)."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    return np.sqrt(dx * dx + dy * dy)


def compute_ear(lm, eye_idx):
    """Eye Aspect Ratio: (v1 + v2) / (2 * h).

    6 landmarks: [outer, upper1, upper2, inner, lower2, lower1]
    """
    v1 = _dist(lm[eye_idx[1]], lm[eye_idx[5]])
    v2 = _dist(lm[eye_idx[2]], lm[eye_idx[4]])
    h = _dist(lm[eye_idx[0]], lm[eye_idx[3]])
    return (v1 + v2) / (2.0 * h) if h > 1e-6 else 0.0


def compute_puc(lm, uc, lc, corner_a, corner_b):
    """Eyelid opening at pupil center, normalised by eye width."""
    v = _dist(lm[uc], lm[lc])
    h = _dist(lm[corner_a], lm[corner_b])
    return v / h if h > 1e-6 else 0.0


def compute_mar(lm):
    """Mouth Aspect Ratio: mean of 3 vertical distances / horizontal."""
    h = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    if h < 1e-6:
        return 0.0
    v = sum(_dist(lm[u], lm[l]) for u, l in zip(MOUTH_UPPER, MOUTH_LOWER))
    return v / (3.0 * h)


def compute_muc(lm):
    """Mouth center opening / width."""
    v = _dist(lm[13], lm[14])
    h = _dist(lm[MOUTH_LEFT], lm[MOUTH_RIGHT])
    return v / h if h > 1e-6 else 0.0


def compute_head_pose(lm, frame_w, frame_h):
    """Pitch, yaw, roll (degrees) via solvePnP with approximate camera model."""
    pts_2d = np.array([lm[i] for i in POSE_LM_IDS], dtype=np.float64)

    focal = float(frame_w)
    cam_matrix = np.array([
        [focal, 0,     frame_w / 2.0],
        [0,     focal, frame_h / 2.0],
        [0,     0,     1.0],
    ], dtype=np.float64)

    ok, rvec, _ = cv2.solvePnP(
        MODEL_3D, pts_2d, cam_matrix, DIST_COEFFS,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return 0.0, 0.0, 0.0

    R, _ = cv2.Rodrigues(rvec)

    # ZYX Tait-Bryan decomposition → pitch (X), yaw (Y), roll (Z)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:   # gimbal lock (yaw ≈ ±90°, rare in normal head poses)
        pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        yaw = np.degrees(np.arctan2(-R[2, 0], sy))
        roll = 0.0

    return pitch, yaw, roll


# ---------------------------------------------------------------------------
# Per-video extraction  (runs in worker process)
# ---------------------------------------------------------------------------

def extract_video(video_paths, subject_id, label, video_file_key):
    """Extract per-frame features from one (possibly multi-part) video.

    video_paths: list of file-path strings (sorted by part number).
    Returns a list of row-tuples matching COLUMNS.
    """
    # Prevent OpenCV from spawning internal threads (we parallelise at process level)
    cv2.setNumThreads(1)

    # Probe resolution from first part
    probe = cv2.VideoCapture(video_paths[0])
    if not probe.isOpened():
        print(f"  [ERROR] Cannot open: {video_paths[0]}")
        return []
    W = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    probe.release()

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=MP_DET_CONF,
        min_face_presence_confidence=MP_DET_CONF,
        min_tracking_confidence=MP_TRACK_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)

    features = []      # list of 10-element lists (or NaN list)
    n_detected = 0
    fidx = 0
    timestamp_ms = 0.0

    # Iterate over all parts (usually just one)
    for part_path in video_paths:
        cap = cv2.VideoCapture(part_path)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open part: {part_path}")
            continue

        part_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_step_ms = 1000.0 / float(part_fps)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect_for_video(mp_image, int(timestamp_ms))

            if result.face_landmarks:
                fl = result.face_landmarks[0]
                lm = np.array(
                    [(p.x * W, p.y * H) for p in fl],
                    dtype=np.float64,
                )

                ear_l = compute_ear(lm, LEFT_EYE)
                ear_r = compute_ear(lm, RIGHT_EYE)
                puc_l = compute_puc(lm, LEFT_EYE_UC, LEFT_EYE_LC,
                                    LEFT_EYE[0], LEFT_EYE[3])
                puc_r = compute_puc(lm, RIGHT_EYE_UC, RIGHT_EYE_LC,
                                    RIGHT_EYE[0], RIGHT_EYE[3])
                mar = compute_mar(lm)
                muc = compute_muc(lm)
                pitch, yaw, roll_val = compute_head_pose(lm, W, H)

                features.append([
                    ear_l, ear_r, (ear_l + ear_r) / 2.0,
                    mar, puc_l, puc_r, muc,
                    pitch, yaw, roll_val,
                ])
                n_detected += 1
            else:
                features.append([np.nan] * 10)

            fidx += 1
            timestamp_ms += frame_step_ms

        cap.release()

    landmarker.close()

    if fidx == 0:
        parts_str = ', '.join(video_paths)
        print(f"  [WARN] 0 frames read: {parts_str}")
        return []

    # --- Fill missing detections: forward → backward → zero ---
    df_feat = pd.DataFrame(features, columns=FEATURE_NAMES)
    df_feat = df_feat.ffill().bfill().fillna(0.0)

    det_pct = n_detected / fidx * 100
    print(f"  Subject {subject_id:>2} | {video_file_key:>16s} | "
          f"{fidx:>6d} frames | {det_pct:5.1f}% detected")

    # --- Build output rows ---
    vals = df_feat.values
    rows = []
    for i in range(fidx):
        rows.append((
            subject_id, video_file_key, i, label,
            vals[i, 0], vals[i, 1], vals[i, 2], vals[i, 3],
            vals[i, 4], vals[i, 5], vals[i, 6], vals[i, 7],
            vals[i, 8], vals[i, 9],
        ))

    return rows


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------

def discover_videos():
    """Find all alert (0) and drowsy (10) videos in dataset/.

    Split files (e.g. 10_1.mp4, 10_2.mp4) are grouped into a single task
    with parts ordered by part number.
    """
    tasks = []
    for subj_dir in sorted(DATASET_DIR.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj_name = subj_dir.name           # "01", "02", …
        try:
            subj_int = int(subj_name)
        except ValueError:
            continue

        # Group files by label: {label_str: [(part_num, fpath), ...]}
        label_parts = defaultdict(list)
        for fpath in subj_dir.iterdir():
            ext = fpath.suffix.lower()
            if ext not in VIDEO_EXTS:
                continue
            m = _STEM_RE.match(fpath.stem)
            if not m:
                continue
            label_str = m.group(1)          # "0" or "10"
            part_num = int(m.group(2)) if m.group(2) else 0
            label_parts[label_str].append((part_num, fpath))

        for label_str, parts in label_parts.items():
            label = LABEL_MAP[label_str]
            parts.sort()  # sort by part number
            paths = [str(fp) for _, fp in parts]
            # Canonical key: "32/10.mp4" even if split into 10_1, 10_2
            first_ext = parts[0][1].suffix
            video_key = f"{subj_name}/{label_str}{first_ext}"
            if len(parts) > 1:
                names = ', '.join(fp.name for _, fp in parts)
                print(f"  [INFO] Subject {subj_name}: merged split files "
                      f"for label {label_str}: {names}")
            tasks.append((paths, subj_int, label, video_key))

    return tasks


# ---------------------------------------------------------------------------
# Parallel wrapper  (picklable top-level function for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker(args):
    """Unpack args and call extract_video. Top-level for Windows pickling."""
    try:
        return extract_video(*args)
    except Exception as e:
        print(f"  [ERROR] {args[0]}: {e}")
        return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    t_start = time.perf_counter()

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model file: {MODEL_PATH}. "
            "Place face_landmarker.task in the project root."
        )

    tasks = discover_videos()
    n_subjects = len({t[1] for t in tasks})
    print(f"Found {len(tasks)} videos from {n_subjects} subjects "
          f"(labels: alert=0, drowsy=1, skipping low-drowsy=5)")
    print(f"Using {NUM_WORKERS} worker processes (cpus={CPU_COUNT})\n")

    all_rows = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(_worker, t): t for t in tasks}
        done_count = 0
        for future in as_completed(futures):
            rows = future.result()
            all_rows.extend(rows)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.perf_counter() - t_start
                rate = done_count / elapsed
                eta = (len(tasks) - done_count) / rate if rate > 0 else 0
                print(f"  Progress: {done_count}/{len(tasks)} videos | "
                      f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")

    if not all_rows:
        print("ERROR: No features extracted. Check dataset path and video files.")
        return

    # Build DataFrame, sort, and save
    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df = df.sort_values(
        ['Subject', 'Video_File', 'Frame']
    ).reset_index(drop=True)

    df.to_csv(str(OUTPUT_CSV), index=False)

    elapsed = time.perf_counter() - t_start
    n_alert = (df['Label'] == 0).sum()
    n_drowsy = (df['Label'] == 1).sum()
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total frames:  {len(df):,}")
    print(f"  Alert (0):     {n_alert:,}")
    print(f"  Drowsy (1):    {n_drowsy:,}")
    print(f"  Subjects:      {df['Subject'].nunique()}")
    print(f"  Videos:        {df['Video_File'].nunique()}")
    print(f"  Saved to:      {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
