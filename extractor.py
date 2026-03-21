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
import csv
import atexit

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor


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


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        val = int(raw)
    except ValueError:
        print(f"  [WARN] Invalid {name}={raw!r}; using {default}")
        val = int(default)
    return max(minimum, val)

CPU_COUNT = max(1, os.cpu_count() or 1)
NUM_WORKERS = _env_int("EXTRACTOR_WORKERS", CPU_COUNT, 1)
MAX_GAP_FILL = _env_int("EXTRACTOR_MAX_GAP_FILL", 15, 0)

# Worker/runtime performance controls.
EXTRACTOR_MP_DELEGATE = os.environ.get("EXTRACTOR_MP_DELEGATE", "auto").strip().lower()
EXTRACTOR_WORKER_THREADS = _env_int("EXTRACTOR_WORKER_THREADS", 1, 1)
EXTRACTOR_OMP_THREADS = _env_int("EXTRACTOR_OMP_THREADS", EXTRACTOR_WORKER_THREADS, 1)
EXTRACTOR_CHUNKS_PER_WORKER = _env_int("EXTRACTOR_CHUNKS_PER_WORKER", 4, 1)

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


# Worker-local resources (one landmarker reused across many videos in process).
_WORKER_LANDMARKER = None
_WORKER_DELEGATE = "cpu"
_WORKER_TS_MS = 0


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


def _safe_part_fps(cap):
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 1e-6:
        return 30.0
    return fps


def _fill_feature_gaps(features):
    """Interpolate short detector dropouts; leave long gaps obvious."""
    df_feat = pd.DataFrame(features, columns=FEATURE_NAMES, dtype=np.float64)
    if MAX_GAP_FILL > 0:
        df_feat = df_feat.interpolate(
            method='linear',
            limit=MAX_GAP_FILL,
            limit_direction='both',
            limit_area='inside',
        )
        df_feat = df_feat.ffill(limit=MAX_GAP_FILL).bfill(limit=MAX_GAP_FILL)
    df_feat = df_feat.fillna(0.0)

    # Prevent ±180° wrap discontinuities from appearing as large fake jumps.
    ang_cols = ["Pitch", "Yaw", "Roll"]
    ang = df_feat[ang_cols].to_numpy(dtype=np.float64, copy=True)
    df_feat.loc[:, ang_cols] = np.rad2deg(np.unwrap(np.deg2rad(ang), axis=0))
    return df_feat


# ---------------------------------------------------------------------------
# Per-video extraction  (runs in worker process)
# ---------------------------------------------------------------------------

def _close_worker_resources():
    global _WORKER_LANDMARKER, _WORKER_TS_MS
    if _WORKER_LANDMARKER is not None:
        _WORKER_LANDMARKER.close()
        _WORKER_LANDMARKER = None
    _WORKER_TS_MS = 0


def _create_landmarker_with_delegate(delegate_name: str):
    delegate_name = delegate_name.lower()
    if delegate_name == "gpu":
        delegate = mp.tasks.BaseOptions.Delegate.GPU
    elif delegate_name == "cpu":
        delegate = mp.tasks.BaseOptions.Delegate.CPU
    else:
        raise ValueError(f"Unsupported delegate: {delegate_name}")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=MODEL_PATH, delegate=delegate),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=MP_DET_CONF,
        min_face_presence_confidence=MP_DET_CONF,
        min_tracking_confidence=MP_TRACK_CONF,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(options)


def _init_worker(
    delegate_mode: str | None = None,
    cv_threads: int | None = None,
    omp_threads: int | None = None,
):
    """Initialise heavy worker-local resources once per process."""
    global _WORKER_LANDMARKER, _WORKER_DELEGATE

    cv_threads = EXTRACTOR_WORKER_THREADS if cv_threads is None else max(1, int(cv_threads))
    omp_threads = EXTRACTOR_OMP_THREADS if omp_threads is None else max(1, int(omp_threads))
    cv2.setNumThreads(cv_threads)
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(omp_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(omp_threads)

    prefer = EXTRACTOR_MP_DELEGATE if delegate_mode is None else delegate_mode.strip().lower()
    if prefer not in {"auto", "cpu", "gpu"}:
        print(f"  [WARN] Unknown EXTRACTOR_MP_DELEGATE={prefer!r}; using auto")
        prefer = "auto"

    delegate_order = ["gpu", "cpu"] if prefer == "auto" else [prefer]
    last_err = None
    for delegate_name in delegate_order:
        try:
            _WORKER_LANDMARKER = _create_landmarker_with_delegate(delegate_name)
            _WORKER_DELEGATE = delegate_name
            if prefer == "auto" and delegate_name == "cpu":
                print("  [INFO] MediaPipe GPU delegate unavailable; using CPU delegate")
            break
        except Exception as exc:
            last_err = exc
            _WORKER_LANDMARKER = None
            continue

    if _WORKER_LANDMARKER is None:
        raise RuntimeError(f"Unable to initialize MediaPipe landmarker: {last_err}")

    atexit.register(_close_worker_resources)


def _detect_for_video_with_retry(landmarker, mp_image, ts_ms: int):
    """Retry once with a bumped timestamp if MediaPipe rejects monotonicity."""
    global _WORKER_TS_MS
    try:
        return landmarker.detect_for_video(mp_image, ts_ms), ts_ms
    except Exception as exc:
        if "monotonically increasing" not in str(exc):
            raise
        retry_ts = max(int(_WORKER_TS_MS) + 1, int(ts_ms) + 1)
        result = landmarker.detect_for_video(mp_image, retry_ts)
        return result, retry_ts

def extract_video(video_paths, subject_id, label, video_file_key):
    """Extract per-frame features from one (possibly multi-part) video.

    video_paths: list of file-path strings (sorted by part number).
    Returns a list of row-tuples matching COLUMNS.
    """
    global _WORKER_LANDMARKER, _WORKER_TS_MS
    if _WORKER_LANDMARKER is None:
        _init_worker()
    landmarker = _WORKER_LANDMARKER

    features = []      # list of 10-element lists (or NaN list)
    n_detected = 0
    fidx = 0
    # MediaPipe VIDEO mode requires strictly increasing timestamps for the
    # lifetime of a landmarker instance. Because we reuse one instance per
    # worker, continue timestamps across videos handled by this worker.
    timestamp_ms = float(_WORKER_TS_MS + 1)

    # Iterate over all parts (usually just one)
    for part_path in video_paths:
        cap = cv2.VideoCapture(part_path)
        if not cap.isOpened():
            print(f"  [ERROR] Cannot open part: {part_path}")
            continue

        part_fps = _safe_part_fps(cap)
        frame_step_ms = 1000.0 / float(part_fps)

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            h_cur, w_cur = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(timestamp_ms)
            result, ts_used = _detect_for_video_with_retry(landmarker, mp_image, ts_ms)

            if result.face_landmarks:
                fl = result.face_landmarks[0]
                lm = np.array(
                    [(p.x * w_cur, p.y * h_cur) for p in fl],
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
                pitch, yaw, roll_val = compute_head_pose(lm, w_cur, h_cur)

                features.append([
                    ear_l, ear_r, (ear_l + ear_r) / 2.0,
                    mar, puc_l, puc_r, muc,
                    pitch, yaw, roll_val,
                ])
                n_detected += 1
            else:
                features.append([np.nan] * 10)

            fidx += 1
            _WORKER_TS_MS = int(ts_used)
            timestamp_ms += frame_step_ms
            if int(timestamp_ms) <= _WORKER_TS_MS:
                timestamp_ms = float(_WORKER_TS_MS + 1)

        cap.release()

    if fidx == 0:
        parts_str = ', '.join(video_paths)
        print(f"  [WARN] 0 frames read: {parts_str}")
        return []

    if n_detected == 0:
        print(f"  [WARN] No face landmarks detected for {video_file_key}; skipping video")
        return []

    # --- Fill short missing-detection gaps without hallucinating long spans ---
    df_feat = _fill_feature_gaps(features)

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

    tasks.sort(key=lambda t: (t[1], t[3]))
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

    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(
            f"Missing MediaPipe model file: {MODEL_PATH}. "
            "Place face_landmarker.task in the project root."
        )

    tasks = discover_videos()
    if not tasks:
        print(f"ERROR: No videos found under {DATASET_DIR}")
        return
    worker_count = min(NUM_WORKERS, len(tasks))
    chunk_size = max(1, len(tasks) // max(worker_count * EXTRACTOR_CHUNKS_PER_WORKER, 1))
    if EXTRACTOR_MP_DELEGATE == "auto":
        # Multi-process extraction scales better with CPU delegate; one-worker mode
        # can opportunistically use GPU when available.
        effective_delegate = "cpu" if worker_count > 1 else "auto"
    else:
        effective_delegate = EXTRACTOR_MP_DELEGATE
    n_subjects = len({t[1] for t in tasks})
    print(f"Found {len(tasks)} videos from {n_subjects} subjects "
          f"(labels: alert=0, drowsy=1, skipping low-drowsy=5)")
    print(
        f"Using {worker_count} worker processes (cpus={CPU_COUNT}, "
        f"opencv_threads/worker={EXTRACTOR_WORKER_THREADS}, omp_threads/worker={EXTRACTOR_OMP_THREADS}, "
        f"delegate={effective_delegate}, chunksize={chunk_size})\n"
    )

    tmp_csv = OUTPUT_CSV.with_suffix(OUTPUT_CSV.suffix + ".tmp")
    total_frames = 0
    n_alert = 0
    n_drowsy = 0
    written_videos = 0
    written_subjects = set()

    with ProcessPoolExecutor(
        max_workers=worker_count,
        initializer=_init_worker,
        initargs=(effective_delegate, EXTRACTOR_WORKER_THREADS, EXTRACTOR_OMP_THREADS),
    ) as pool, tmp_csv.open('w', newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS)
        done_count = 0
        for rows in pool.map(_worker, tasks, chunksize=chunk_size):
            if rows:
                writer.writerows(rows)
                frame_count = len(rows)
                total_frames += frame_count
                written_videos += 1
                written_subjects.add(rows[0][0])
                if rows[0][3] == 0:
                    n_alert += frame_count
                else:
                    n_drowsy += frame_count
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                elapsed = time.perf_counter() - t_start
                rate = done_count / elapsed
                eta = (len(tasks) - done_count) / rate if rate > 0 else 0
                print(f"  Progress: {done_count}/{len(tasks)} videos | "
                      f"{elapsed:.0f}s elapsed | ~{eta:.0f}s remaining")

    if total_frames == 0:
        tmp_csv.unlink(missing_ok=True)
        print("ERROR: No features extracted. Check dataset path and video files.")
        return

    tmp_csv.replace(OUTPUT_CSV)

    elapsed = time.perf_counter() - t_start
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Total frames:  {total_frames:,}")
    print(f"  Alert (0):     {n_alert:,}")
    print(f"  Drowsy (1):    {n_drowsy:,}")
    print(f"  Subjects:      {len(written_subjects)}")
    print(f"  Videos:        {written_videos}")
    print(f"  Saved to:      {OUTPUT_CSV}")


if __name__ == '__main__':
    main()
