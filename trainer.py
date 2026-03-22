import os
import json
import shutil
import time
import warnings
import re

CPU_COUNT = max(1, os.cpu_count() or 1)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(CPU_COUNT))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(CPU_COUNT))
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("MKL_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("NUMEXPR_NUM_THREADS", str(CPU_COUNT))

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import mixed_precision

try:
    from scipy.signal import savgol_filter as _savgol
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def _tf_version_at_least(major: int, minor: int) -> bool:
    m = re.match(r"^\s*(\d+)\.(\d+)", str(tf.__version__))
    if not m:
        return False
    cur = (int(m.group(1)), int(m.group(2)))
    return cur >= (major, minor)


_USE_STRING_HARD_ACTS = _tf_version_at_least(2, 13)
HARD_SWISH_ACT = "hard_swish" if _USE_STRING_HARD_ACTS else tf.nn.hard_swish
HARD_SIGMOID_ACT = "hard_sigmoid" if _USE_STRING_HARD_ACTS else tf.keras.activations.hard_sigmoid

try:
    tf.config.threading.set_inter_op_parallelism_threads(CPU_COUNT)
    tf.config.threading.set_intra_op_parallelism_threads(CPU_COUNT)
except RuntimeError as exc:
    # This can happen if TensorFlow runtime was already initialized upstream.
    print(f"WARNING: unable to set TF thread pools at import time: {exc}")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "off", "no"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return max(minimum, int(default))
    try:
        val = int(raw)
    except ValueError:
        print(f"WARNING: invalid {name}={raw!r}; using {default}")
        val = int(default)
    return max(minimum, val)


def _configure_tf_runtime() -> dict:
    """Configure TensorFlow runtime for maximum device utilisation."""
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except Exception:
            pass

    enable_xla = _env_flag("TRAINER_ENABLE_XLA", True)
    try:
        tf.config.optimizer.set_jit(enable_xla)
    except Exception:
        pass

    mp_mode = os.environ.get("TRAINER_MIXED_PRECISION", "auto").strip().lower()
    if mp_mode in {"off", "false", "0"}:
        policy = "float32"
    elif mp_mode in {"bf16", "mixed_bfloat16"}:
        policy = "mixed_bfloat16"
    elif mp_mode in {"on", "true", "1", "fp16", "mixed_float16"}:
        policy = "mixed_float16"
    else:  # auto
        policy = "mixed_float16" if gpus else "float32"
    if not gpus and policy == "mixed_float16":
        print("WARNING: mixed_float16 requested without a GPU; falling back to float32")
        policy = "float32"
    mixed_precision.set_global_policy(policy)

    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        strategy_name = f"MirroredStrategy({len(gpus)} GPUs)"
    elif len(gpus) == 1:
        strategy = tf.distribute.OneDeviceStrategy(device="/GPU:0")
        strategy_name = "OneDeviceStrategy(GPU)"
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/CPU:0")
        strategy_name = "OneDeviceStrategy(CPU)"

    replicas = int(strategy.num_replicas_in_sync)
    runtime_batch = _env_int(
        "TRAINER_RUNTIME_BATCH_SIZE",
        BATCH_SIZE * max(replicas, 1),
        1,
    )

    return {
        "strategy": strategy,
        "strategy_name": strategy_name,
        "replicas": replicas,
        "runtime_batch_size": runtime_batch,
        "enable_xla": enable_xla,
        "mixed_precision_policy": mixed_precision.global_policy().name,
        "gpu_count": len(gpus),
    }

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    average_precision_score,
    precision_score, recall_score,
    confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)


# ───────────────────────────── Configuration ─────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE  = os.path.join(BASE_DIR, "uta_rldd_features.csv")

FEATURES = [
    "EAR_Left", "EAR_Right", "EAR_Avg", "MAR",
    "PUC_Left", "PUC_Right", "MUC",
    "Pitch", "Yaw", "Roll",
]
NUM_RAW      = len(FEATURES)        # 10
NUM_FEATURES = NUM_RAW * 3 + 3      # 33  (raw + delta + ddelta + 3 temporal)

ROLLING_WINDOW = 30                 # 1 s at 30 fps
TEMPORAL_PAD   = ROLLING_WINDOW - 1 # 29 cold-start frames

# PERCLOS proxy: EAR_Avg below this z-score → eye closed
PERCLOS_EAR_Z_THRESHOLD = -0.25

SAVGOL_WINDOW = 7
SAVGOL_POLY   = 2

# Keep preprocessing consistent between train/eval. Default is disabled to
# avoid lookahead leakage from symmetric Savitzky-Golay windows.
SAVGOL_ENABLED  = False
SAVGOL_ON_TRAIN = SAVGOL_ENABLED
SAVGOL_ON_EVAL  = SAVGOL_ENABLED

SEQ_LEN    = 150  # 5 s at 30 fps
TRAIN_STEP = 20
EVAL_STEP  = 15   # denser than TRAIN_STEP for stable per-epoch metrics

# ── MobileNetV3-1D backbone ──
# (expand_ch, out_ch, kernel, stride, use_se, use_hswish)
MBCONV_CONFIG = [
    ( 32,  16, 3, 1, True,  False),   # B0: 150→150, 16ch
    ( 48,  24, 3, 2, False, False),   # B1: 150→75,  24ch
    ( 48,  24, 3, 1, True,  False),   # B2: 75→75,   24ch
    ( 64,  32, 5, 2, True,  True),    # B3: 75→38,   32ch
    ( 96,  32, 5, 1, True,  True),    # B4: 38→38,   32ch
    ( 96,  48, 5, 1, True,  True),    # B5: 38→38,   48ch
    (144,  64, 5, 2, True,  True),    # B6: 38→19,   64ch
    (192,  64, 5, 1, True,  True),    # B7: 19→19,   64ch
]
STEM_FILTERS = 16
NECK_DIM     = 128
SE_RATIO     = 4

# ── BiLSTM ──
LSTM_UNITS = 64     # per direction → 128 total
ATTN_UNITS = 64

# ── Training ──
DROPOUT       = 0.35
EPOCHS        = 100
BATCH_SIZE    = 256
PEAK_LR       = 1e-3
WEIGHT_DECAY  = 1e-4
WARMUP_EPOCHS = 5
MIN_LR        = 1e-6

# ES_PATIENCE must exceed the first cosine restart period (T₀=20).
# A value of 3 caused EarlyStopping to fire before warmup even completed.
ES_PATIENCE = 25

FOCAL_GAMMA = 2.0
# FOCAL_ALPHA is computed per-run from actual class balance (_compute_focal_alpha).
# This constant is only the fallback default.
FOCAL_ALPHA = 0.5

AUGMENT_ROUNDS  = 3
MIXUP_ALPHA     = 0.3
TTA_ROUNDS      = 4

# Deployment/evaluation parity:
# tune threshold and report primary metrics using the same inference mode that
# will be used in deployment.
DEPLOY_USE_TTA = False

# LABEL_SMOOTH kept for reference only — not applied during augmentation.
# MixUp already provides soft targets; applying smoothing before MixUp
# creates double-smoothed labels with unpredictable effective smoothing.
LABEL_SMOOTH    = 0.05

MIXUP_FRACTION  = 0.35
MIXUP_CHUNK     = 16384
MIN_ALERT_FRAMES_FOR_ANCHOR = 100

# F_BETA < 1 weights precision more heavily than recall in threshold search.
# 0.5 penalises false positives twice as much as false negatives, which
# corrects the over-triggering seen with F_BETA=1.
F_BETA           = 0.5

FIXED_THRESHOLD  = None   # set float in [0,1] to lock deployment threshold
N_FOLDS          = 3

# Set to False for the final run to get an honest generalisation estimate.
# True is acceptable during development iteration to save time.
SKIP_CV          = False

SEED             = 42
DETERMINISTIC    = False

# K=2: require 2 consecutive windows above threshold before classifying drowsy.
# Cuts ~63% of false positives (noise spikes) while losing almost no recall,
# because genuine drowsiness spans multiple consecutive windows.
SMOOTH_K = 2

# INT8 export mode:
# False -> hybrid quantization (float32 I/O, int8 internal kernels)
# True  -> full int8 I/O
INT8_FULL_IO = False

MODEL_NAME = "drowsiness_mv3_lstm"

# Runtime-compute setup (devices, precision, XLA, effective batch size).
_RUNTIME = _configure_tf_runtime()
STRATEGY = _RUNTIME["strategy"]
STRATEGY_NAME = _RUNTIME["strategy_name"]
NUM_REPLICAS = _RUNTIME["replicas"]
RUNTIME_BATCH_SIZE = _RUNTIME["runtime_batch_size"]
ENABLE_XLA = _RUNTIME["enable_xla"]
MIXED_PRECISION_POLICY = _RUNTIME["mixed_precision_policy"]
GPU_COUNT = _RUNTIME["gpu_count"]


# ─────────────────────────── Temporal smoothing ──────────────────────────


def smooth_predict(probs: np.ndarray, threshold: float, K: int = SMOOTH_K) -> np.ndarray:
    """Require K consecutive windows above threshold to classify as drowsy.

    K=1 → original single-window behaviour (no smoothing).
    K=2 → requires 2 consecutive windows; eliminates brief noise spikes.
    """
    binary = (probs >= threshold).astype(np.int32)
    if K <= 1:
        return binary
    smoothed = np.zeros_like(binary)
    for i in range(K - 1, len(binary)):
        window = binary[i - K + 1 : i + 1]
        smoothed[i] = 1 if window.sum() == K else 0
    return smoothed


def smooth_predict_grouped(
    probs: np.ndarray,
    threshold: float,
    groups: np.ndarray | None,
    K: int = SMOOTH_K,
) -> np.ndarray:
    """Apply temporal smoothing independently within each contiguous group."""
    if groups is None:
        return smooth_predict(probs, threshold, K=K)
    if len(groups) != len(probs):
        raise ValueError(f"groups length ({len(groups)}) != probs length ({len(probs)})")

    out = np.zeros(len(probs), dtype=np.int32)
    start = 0
    while start < len(probs):
        end = start + 1
        while end < len(probs) and groups[end] == groups[start]:
            end += 1
        out[start:end] = smooth_predict(probs[start:end], threshold, K=K)
        start = end
    return out


# ─────────────────────────── Data Loading ────────────────────────────────


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["Label"].isin([0, 1])].copy()
    df = df.sort_values(["Subject", "Video_File", "Frame"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} frames from {df['Subject'].nunique()} subjects")
    print(f"  Alert: {(df['Label']==0).sum():,}  Drowsy: {(df['Label']==1).sum():,}")
    return df


def sanitize_features(df: pd.DataFrame, tag: str = "features") -> pd.DataFrame:
    """Replace NaN/inf in the feature block so split stats stay well-defined."""
    if df.empty:
        return df.copy()
    df = df.copy()
    nans = int(df[FEATURES].isna().sum().sum())
    infs = int(np.isinf(df[FEATURES].values).sum())
    if nans or infs:
        print(f"  WARNING [{tag}]: {nans} NaN, {infs} inf — replacing with 0")
        df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def _compute_feature_stats(df: pd.DataFrame) -> dict:
    """Median/IQR stats for a single subject or global fallback block."""
    params: dict = {}
    for feat in FEATURES:
        med = float(df[feat].median())
        iqr = float(df[feat].quantile(0.75) - df[feat].quantile(0.25))
        if not np.isfinite(med):
            med = 0.0
        if not np.isfinite(iqr) or iqr < 1e-6:
            iqr = 1.0
        params[feat] = {"median": med, "iqr": iqr}
    return params


def fit_subject_norm_params(
    df: pd.DataFrame,
    min_alert_frames: int = MIN_ALERT_FRAMES_FOR_ANCHOR,
) -> tuple[dict, dict]:
    """Fit train-only per-subject params plus a global fallback for unseen subjects."""
    if df.empty:
        raise ValueError("Cannot fit normalisation params on an empty dataframe.")

    subject_params: dict = {}
    for subj, grp in df.groupby("Subject", sort=False):
        alert = grp[grp["Label"] == 0]
        anchor = alert if len(alert) >= min_alert_frames else grp
        subject_params[str(subj)] = _compute_feature_stats(anchor)

    global_params = {
        feat: {
            "median": float(np.median([subject_params[s][feat]["median"] for s in subject_params])),
            "iqr": float(np.median([subject_params[s][feat]["iqr"] for s in subject_params])),
        }
        for feat in FEATURES
    }
    return subject_params, global_params


def apply_norm_params(
    df: pd.DataFrame,
    subject_params: dict,
    global_params: dict,
    use_subject_params: bool,
    tag: str,
) -> pd.DataFrame:
    """Normalise a split with train-subject params or the global fallback."""
    if df.empty:
        return df.copy()

    groups = []
    fallback_subjects = 0
    for subj, grp in df.groupby("Subject", sort=False):
        grp = grp.copy()
        params = subject_params.get(str(subj)) if use_subject_params else None
        if params is None:
            params = global_params
            fallback_subjects += 1
        for feat in FEATURES:
            grp[feat] = (grp[feat] - params[feat]["median"]) / params[feat]["iqr"]
        groups.append(grp)

    print(f"  {tag}: global fallback norm for {fallback_subjects}/{len(groups)} subjects")
    out = pd.concat(groups)
    return out.sort_values(["Subject", "Video_File", "Frame"]).reset_index(drop=True)


def prepare_split_features(
    df: pd.DataFrame,
    subject_params: dict,
    global_params: dict,
    use_subject_params: bool,
    tag: str,
    apply_savgol_filter: bool,
) -> pd.DataFrame:
    """Apply split-safe normalisation, denoise per video, and sanitise outputs."""
    df = apply_norm_params(df, subject_params, global_params, use_subject_params, tag)
    df = apply_savgol(df, enabled=apply_savgol_filter, tag=tag)
    return sanitize_features(df, tag=f"{tag} post-savgol")


def apply_savgol(df: pd.DataFrame, enabled: bool = True, tag: str = "") -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if not enabled:
        if tag:
            print(f"  {tag}: skipping Savitzky-Golay (leakage-safe mode)")
        return df
    if not _HAS_SCIPY:
        print("  WARNING: scipy missing — skipping Savitzky-Golay denoising")
        return df
    parts = []
    for _, grp in df.groupby(["Subject", "Video_File"], sort=False):
        g = grp.copy()
        if len(g) >= SAVGOL_WINDOW:
            for feat in FEATURES:
                g[feat] = _savgol(g[feat].values, SAVGOL_WINDOW, SAVGOL_POLY)
        parts.append(g)
    out = pd.concat(parts)
    return out.sort_values(["Subject", "Video_File", "Frame"]).reset_index(drop=True)


# ─────────────────────────── Feature Engineering ─────────────────────────


def apply_enhanced_features(X_raw: np.ndarray) -> np.ndarray:
    """(N, T, 10) → (N, T, 33): raw + Δ + ΔΔ + [EAR-var, head-move, PERCLOS]."""
    N, T, _ = X_raw.shape

    # Angle unwrapping is handled during extraction (before normalisation),
    # so these values are already in consistent units for temporal deltas.
    X_proc = X_raw.astype(np.float32, copy=True)

    delta = np.zeros_like(X_proc)
    delta[:, 1:] = X_proc[:, 1:] - X_proc[:, :-1]
    ddelta = np.zeros_like(delta)
    ddelta[:, 1:] = delta[:, 1:] - delta[:, :-1]

    ear   = X_proc[:, :, 2]   # EAR_Avg (index 2)
    pd_   = delta[:, :, 7]   # Pitch delta (index 7)

    # Rolling EAR variance
    ear_pad = np.pad(ear, ((0, 0), (ROLLING_WINDOW - 1, 0)), mode="edge")
    ear_var = np.var(
        sliding_window_view(ear_pad, ROLLING_WINDOW, axis=1), axis=2
    ).astype(np.float32)

    # Rolling mean absolute pitch delta (head movement proxy)
    t_idx  = np.arange(T)
    starts = np.maximum(t_idx - ROLLING_WINDOW + 1, 0)
    wlens  = (t_idx - starts + 1).astype(np.float32)
    cs     = np.cumsum(np.abs(pd_), axis=1)
    cs_pad = np.concatenate([np.zeros((N, 1), dtype=np.float32), cs], axis=1)
    head_move = (
        (cs_pad[:, t_idx + 1] - cs_pad[:, starts]) / wlens
    ).astype(np.float32)

    # PERCLOS proxy: rolling fraction of frames with EAR below closure threshold
    closed = (ear < PERCLOS_EAR_Z_THRESHOLD).astype(np.float32)
    closed_pad = np.pad(closed, ((0, 0), (ROLLING_WINDOW - 1, 0)), mode="edge")
    perclos = np.mean(
        sliding_window_view(closed_pad, ROLLING_WINDOW, axis=1), axis=2
    ).astype(np.float32)

    temporal = np.stack([ear_var, head_move, perclos], axis=-1)
    return np.concatenate([X_proc, delta, ddelta, temporal], axis=-1)


# ─────────────────────────── Sequence Creation ───────────────────────────


def _create_raw_sequences(
    df: pd.DataFrame,
    seq_len: int,
    step: int,
    temporal_pad: int = 0,
    return_groups: bool = False,
):
    """Sliding-window sequences (raw 10-feature) grouped by video."""
    padded_len = seq_len + temporal_pad
    X, y = [], []
    groups = []
    for (subject, video_file), grp in df.groupby(["Subject", "Video_File"], sort=False):
        raw    = grp[FEATURES].values.astype(np.float32)
        labels = grp["Label"].values
        if len(raw) < seq_len:
            continue
        group_id = f"{subject}|{video_file}"
        for i in range(0, len(raw) - seq_len + 1, step):
            ps    = max(0, i - temporal_pad)
            chunk = raw[ps : i + seq_len]
            if chunk.shape[0] < padded_len:
                pad_n = padded_len - chunk.shape[0]
                chunk = np.pad(chunk, ((pad_n, 0), (0, 0)), mode="edge")
            X.append(chunk)
            y.append(labels[i + seq_len - 1])
            groups.append(group_id)
    if not X:
        empty_X = np.empty((0, padded_len, NUM_RAW), dtype=np.float32)
        empty_y = np.empty((0,), dtype=np.int32)
        if return_groups:
            return empty_X, empty_y, np.empty((0,), dtype=object)
        return empty_X, empty_y

    X_arr = np.array(X, dtype=np.float32)
    y_arr = np.array(y, dtype=np.int32)
    if return_groups:
        return X_arr, y_arr, np.array(groups, dtype=object)
    return X_arr, y_arr


def create_sequences(
    df: pd.DataFrame,
    seq_len: int,
    step: int,
    temporal_pad: int = 0,
    return_groups: bool = False,
):
    """Full 33-feature sequences: raw → enhanced features → trim padding."""
    if return_groups:
        X_raw, y, groups = _create_raw_sequences(
            df, seq_len, step, temporal_pad, return_groups=True
        )
    else:
        X_raw, y = _create_raw_sequences(df, seq_len, step, temporal_pad)
        groups = None
    if len(X_raw) == 0:
        empty_X = np.empty((0, seq_len, NUM_FEATURES), dtype=np.float32)
        if return_groups:
            return empty_X, y, np.empty((0,), dtype=object)
        return empty_X, y
    X = apply_enhanced_features(X_raw)
    if temporal_pad > 0:
        X = X[:, temporal_pad:, :]
    if return_groups:
        return X, y, groups
    return X, y


# ─────────────────────────── Augmentation ────────────────────────────────


def _time_warp(seq: np.ndarray, rng: np.random.Generator, sigma: float = 0.2) -> np.ndarray:
    T = seq.shape[0]
    warp = np.cumsum(np.maximum(rng.normal(1.0, sigma, T), 0.1))
    warp = np.clip(warp / warp[-1] * (T - 1), 0, T - 1)
    orig = np.arange(T, dtype=np.float64)
    out  = np.empty_like(seq)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(warp, orig, seq[:, c])
    return out


def _augment_batch(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    Xa = X.copy()
    n  = len(Xa)

    # Gaussian noise
    mask = rng.random(n) < 0.5
    if mask.any():
        Xa[mask] += rng.normal(0, 0.05, Xa[mask].shape).astype(np.float32)

    # Amplitude scaling
    mask = rng.random(n) < 0.3
    if mask.any():
        Xa[mask] *= rng.uniform(0.8, 1.2, (mask.sum(), 1, 1)).astype(np.float32)

    # Feature masking (exclude EAR_Avg at index 2 to protect PERCLOS)
    maskable = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    mask = rng.random(n) < 0.2
    for i in np.where(mask)[0]:
        fi = rng.choice(maskable, rng.integers(1, 3), replace=False)
        Xa[i, :, fi] = 0.0

    # Time warp
    mask = rng.random(n) < 0.3
    for i in np.where(mask)[0]:
        Xa[i] = _time_warp(Xa[i], rng, sigma=0.2)

    return Xa, y


def _apply_inplace_mixup(
    X: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    alpha: float    = MIXUP_ALPHA,
    fraction: float = MIXUP_FRACTION,
    chunk_size: int = MIXUP_CHUNK,
) -> tuple:
    """Mix a random subset of rows in-place to keep peak memory bounded.

    All source rows are snapshotted before the loop begins so that a sample
    appearing as both target and source across chunks is not corrupted.
    """
    n = len(X)
    if n < 2 or fraction <= 0.0:
        return X, y

    k   = max(1, int(n * fraction))
    tgt = rng.choice(n, size=k, replace=False)
    src = rng.permutation(n)[:k]

    # Snapshot all source rows before any in-place modification
    X_src = X[src].copy()
    y_src = y[src].copy()

    for s in range(0, k, chunk_size):
        e   = min(s + chunk_size, k)
        t   = tgt[s:e]
        lam = rng.beta(alpha, alpha, (len(t), 1, 1)).astype(np.float32)
        X[t] = lam * X[t] + (1 - lam) * X_src[s:e]
        y[t] = lam.reshape(-1) * y[t] + (1 - lam.reshape(-1)) * y_src[s:e]

    return X, y


def _apply_mixup_with_global_partners(
    X: np.ndarray,
    y: np.ndarray,
    X_raw_pool: np.ndarray,
    y_pool: np.ndarray,
    rng: np.random.Generator,
    temporal_pad: int = 0,
    alpha: float = MIXUP_ALPHA,
    fraction: float = MIXUP_FRACTION,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix a subset of the batch with partners sampled from the full dataset."""
    n = len(X)
    pool_n = len(X_raw_pool)
    if n < 1 or pool_n < 1 or fraction <= 0.0:
        return X, y

    k = max(1, int(n * fraction))
    tgt = rng.choice(n, size=k, replace=False)
    partner_idx = rng.integers(0, pool_n, size=k)

    X_partner = apply_enhanced_features(X_raw_pool[partner_idx])
    if temporal_pad > 0:
        X_partner = X_partner[:, temporal_pad:, :]
    X_partner = np.ascontiguousarray(X_partner, dtype=np.float32)
    y_partner = np.ascontiguousarray(y_pool[partner_idx].astype(np.float32), dtype=np.float32)

    lam = rng.beta(alpha, alpha, (k, 1, 1)).astype(np.float32)
    X[tgt] = lam * X[tgt] + (1.0 - lam) * X_partner
    y[tgt] = lam.reshape(-1) * y[tgt] + (1.0 - lam.reshape(-1)) * y_partner
    return X, y


def build_augmented_dataset(
    X_raw: np.ndarray,
    y: np.ndarray,
    rng: np.random.Generator,
    n_rounds: int = AUGMENT_ROUNDS,
    temporal_pad: int = 0,
) -> tuple:
    """Augment raw seqs → compute enhanced features per copy → mixup.

    Each augmented copy is feature-engineered and released independently
    to avoid holding n_rounds+1 raw copies in memory simultaneously.
    Augmentation happens on raw sequences before feature engineering so that
    PERCLOS (a rolling threshold-derived feature) is recomputed from the
    augmented EAR values rather than being linearly blended directly.
    """
    X_base_enh = apply_enhanced_features(X_raw)
    if temporal_pad > 0:
        X_base_enh = X_base_enh[:, temporal_pad:, :]

    parts_X = [X_base_enh]
    parts_y = [y.astype(np.float32)]

    for _ in range(n_rounds):
        Xa, ya = _augment_batch(X_raw, y, rng)
        Xa_enh = apply_enhanced_features(Xa)
        if temporal_pad > 0:
            Xa_enh = Xa_enh[:, temporal_pad:, :]
        parts_X.append(Xa_enh)
        parts_y.append(ya.astype(np.float32))
        del Xa, Xa_enh

    X_all = np.concatenate(parts_X, axis=0)
    y_all = np.concatenate(parts_y, axis=0)
    del parts_X, parts_y

    perm       = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    X_all, y_all = _apply_inplace_mixup(X_all, y_all, rng)

    return X_all, y_all


class AugmentedSequence(keras.utils.Sequence):
    """On-the-fly augmentation + feature engineering to keep memory bounded.

    Not multiprocessing-safe: this class keeps mutable epoch state
    (`_epoch_seed`, `_sample_idx`, `_round_ids`) and should be used with
    `use_multiprocessing=False`.
    """

    def __init__(
        self,
        X_raw: np.ndarray,
        y: np.ndarray,
        batch_size: int = RUNTIME_BATCH_SIZE,
        n_rounds: int = AUGMENT_ROUNDS,
        temporal_pad: int = 0,
        seed: int = SEED,
    ):
        super().__init__()
        self.X_raw = X_raw
        self.y = y.astype(np.float32)
        self.batch_size = batch_size
        self.n_rounds = n_rounds
        self.temporal_pad = temporal_pad
        self.seed = seed
        self.sample_count = len(self.X_raw) * (self.n_rounds + 1)
        self._epoch = 0
        self._epoch_seed = seed
        self._sample_idx = np.empty((0,), dtype=np.int32)
        self._round_ids = np.empty((0,), dtype=np.int8)
        self.on_epoch_end()

    def __len__(self) -> int:
        return (self.sample_count + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self) -> None:
        base_idx = np.arange(len(self.X_raw), dtype=np.int32)
        sample_idx = np.tile(base_idx, self.n_rounds + 1)
        round_ids = np.repeat(np.arange(self.n_rounds + 1, dtype=np.int8), len(base_idx))
        self._epoch_seed = self.seed + self._epoch * 100_003
        rng = np.random.default_rng(self._epoch_seed)
        perm = rng.permutation(len(sample_idx))
        self._sample_idx = sample_idx[perm]
        self._round_ids = round_ids[perm]
        self._epoch += 1

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        s = idx * self.batch_size
        e = min(s + self.batch_size, self.sample_count)
        batch_idx = self._sample_idx[s:e]
        batch_rounds = self._round_ids[s:e]
        X_batch = self.X_raw[batch_idx].copy()
        y_batch = self.y[batch_idx].copy()
        rng = np.random.default_rng(self._epoch_seed + idx + 1)

        aug_mask = batch_rounds > 0
        if aug_mask.any():
            X_aug, y_aug = _augment_batch(X_batch[aug_mask], y_batch[aug_mask], rng)
            X_batch[aug_mask] = X_aug
            y_batch[aug_mask] = y_aug

        X_feat = apply_enhanced_features(X_batch)
        if self.temporal_pad > 0:
            X_feat = X_feat[:, self.temporal_pad:, :]

        X_feat = np.ascontiguousarray(X_feat, dtype=np.float32)
        y_batch = np.ascontiguousarray(y_batch, dtype=np.float32)
        X_feat, y_batch = _apply_mixup_with_global_partners(
            X_feat,
            y_batch,
            self.X_raw,
            self.y,
            rng,
            temporal_pad=self.temporal_pad,
        )
        return X_feat, y_batch


# ─────────────────────────── Class balance helper ────────────────────────


def _compute_focal_alpha(y_hard: np.ndarray) -> float:
    """Derive focal-loss alpha from actual training class proportions.

    Alpha = alert fraction (inverse-frequency weighting):
      α > 0.5 → drowsy up-weighted (drowsy is minority)
      α = 0.5 → balanced
      α < 0.5 → alert up-weighted  (alert is minority)
    Clipped to [0.2, 0.8] to prevent extreme weighting.
    """
    total        = max(len(y_hard), 1)
    drowsy_count = int((y_hard == 1).sum())
    alert_count  = total - drowsy_count
    alpha        = float(np.clip(alert_count / total, 0.2, 0.8))
    print(f"  Class balance → alert:{alert_count:,}  drowsy:{drowsy_count:,}  "
          f"focal_alpha={alpha:.3f}")
    return alpha


# ─────────────────────────── Threshold ───────────────────────────────────


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray | None = None,
    f_beta: float = F_BETA,
) -> float:
    """Sweep thresholds ∈ [0.10, 0.90] to maximise Fβ using smooth_predict.

    The search uses smooth_predict (K=SMOOTH_K) so the threshold found is
    optimal for the same smoothed classifier used at evaluation time.
    Previously the search used raw (y_prob >= thr) while _eval_metrics used
    smooth_predict — a mismatch that caused the tuned threshold to be
    suboptimal for the actual deployed classifier.
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_score, best_thr = -1.0, 0.5
    beta_sq = f_beta ** 2
    for thr in np.linspace(0.10, 0.90, 161):
        pred = smooth_predict_grouped(y_prob, thr, groups=groups, K=SMOOTH_K)
        tp = int(np.sum((pred == 1) & (y_true == 1)))
        fp = int(np.sum((pred == 1) & (y_true == 0)))
        fn = int(np.sum((pred == 0) & (y_true == 1)))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        fb = (1 + beta_sq) * pr * rc / (beta_sq * pr + rc) if (pr + rc) > 0 else 0.0
        if fb > best_score:
            best_score, best_thr = fb, thr
    return float(best_thr)


def resolve_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray | None = None,
) -> float:
    """Return FIXED_THRESHOLD if configured, otherwise Fβ-optimal threshold."""
    if FIXED_THRESHOLD is not None:
        thr = float(FIXED_THRESHOLD)
        if not (0.0 <= thr <= 1.0):
            raise ValueError(f"FIXED_THRESHOLD must be in [0, 1], got {thr}")
        return thr
    return find_optimal_threshold(y_true, y_prob, groups=groups)


# ─────────────────────────── Metrics Helpers ─────────────────────────────


def _eval_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thr: float,
    tag: str,
    groups: np.ndarray | None = None,
) -> dict:
    """Compute and print a full metric block; return dict of scalar results."""
    y_pred  = smooth_predict_grouped(y_prob, thr, groups=groups, K=SMOOTH_K)
    y_pred5 = smooth_predict_grouped(y_prob, 0.5, groups=groups, K=SMOOTH_K)

    auc    = roc_auc_score(y_true, y_prob)           if len(np.unique(y_true)) > 1 else 0.0
    prauc  = average_precision_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    acc    = accuracy_score(y_true, y_pred)
    f1     = f1_score(y_true, y_pred,  zero_division=0)
    prec   = precision_score(y_true, y_pred,  zero_division=0)
    rec    = recall_score(y_true, y_pred,  zero_division=0)
    cm     = confusion_matrix(y_true, y_pred)

    acc5   = accuracy_score(y_true, y_pred5)
    f1_5   = f1_score(y_true, y_pred5, zero_division=0)
    rec5   = recall_score(y_true, y_pred5, zero_division=0)

    print(f"\n  {tag} @thr={thr:.3f}  (vs 0.5 in parentheses)")
    print(f"    AUC:   {auc:.4f}        PR-AUC: {prauc:.4f}")
    print(f"    Acc:   {acc:.4f}  ({acc5:.4f})")
    print(f"    F1:    {f1:.4f}  ({f1_5:.4f})")
    print(f"    Prec:  {prec:.4f}")
    print(f"    Rec:   {rec:.4f}  ({rec5:.4f})")
    print(f"    CM:\n{cm}")
    print(classification_report(y_true, y_pred, target_names=["Alert", "Drowsy"]))

    return {
        "auc": auc, "prauc": prauc,
        "accuracy": acc, "f1": f1, "precision": prec, "recall": rec,
        "confusion_matrix": cm, "threshold": thr,
        "acc_05": acc5, "f1_05": f1_5, "rec_05": rec5,
    }


# ─────────────────────────── Model ───────────────────────────────────────


def _mbconv1d(x, expand, out, kernel, stride, use_se, use_hs, name="mb"):
    """MobileNetV3-style inverted residual block (1-D).

    expand → depthwise → [SE] → project, with optional identity residual.
    """
    in_ch = x.shape[-1]

    # Expand
    if expand != in_ch:
        h = keras.layers.Conv1D(expand, 1, use_bias=False,
                                kernel_initializer="he_normal", name=f"{name}_exp")(x)
        h = keras.layers.BatchNormalization(name=f"{name}_exp_bn")(h)
        if use_hs:
            h = keras.layers.Activation(HARD_SWISH_ACT, name=f"{name}_exp_act")(h)
        else:
            h = keras.layers.ReLU(name=f"{name}_exp_act")(h)
    else:
        h = x

    # Depthwise
    h = keras.layers.Conv1D(expand, kernel, strides=stride, padding="same",
                            groups=expand, use_bias=False,
                            kernel_initializer="he_normal", name=f"{name}_dw")(h)
    h = keras.layers.BatchNormalization(name=f"{name}_dw_bn")(h)
    if use_hs:
        h = keras.layers.Activation(HARD_SWISH_ACT, name=f"{name}_dw_act")(h)
    else:
        h = keras.layers.ReLU(name=f"{name}_dw_act")(h)

    # Squeeze-and-Excitation
    if use_se:
        mid = max(expand // SE_RATIO, 8)
        se  = keras.layers.GlobalAveragePooling1D(keepdims=True, name=f"{name}_se_gap")(h)
        se  = keras.layers.Dense(mid, activation="relu",
                                 kernel_initializer="he_normal", name=f"{name}_se_r")(se)
        se  = keras.layers.Dense(expand, kernel_initializer="he_normal", name=f"{name}_se_e")(se)
        se  = keras.layers.Activation(HARD_SIGMOID_ACT, name=f"{name}_se_hs")(se)
        h   = keras.layers.Multiply(name=f"{name}_se_m")([h, se])

    # Project (linear bottleneck — no activation)
    h = keras.layers.Conv1D(out, 1, use_bias=False,
                            kernel_initializer="he_normal", name=f"{name}_proj")(h)
    h = keras.layers.BatchNormalization(name=f"{name}_proj_bn")(h)

    if stride == 1 and in_ch == out:
        h = keras.layers.Add(name=f"{name}_res")([h, x])
    return h


def _attention_pool(x):
    """Learnable attention pooling over the time axis."""
    score = keras.layers.Dense(ATTN_UNITS, activation="tanh",
                               kernel_initializer="he_normal")(x)
    score = keras.layers.Dense(1, kernel_initializer="he_normal")(score)
    alpha = keras.layers.Softmax(axis=1, name="attn_weights")(score)
    alpha_t = keras.layers.Permute((2, 1), name="attn_weights_t")(alpha)
    ctx = keras.layers.Dot(axes=(2, 1), name="attn_pool")([alpha_t, x])
    return keras.layers.Flatten(name="attn_ctx")(ctx)


def build_model(seq_len: int = SEQ_LEN, num_features: int = NUM_FEATURES) -> keras.Model:
    """MobileNetV3-Small 1D → BiLSTM → Attention → Dense → sigmoid.

    Backbone : Stem + 8 MBConv1D blocks (3 stride-2 stages → 150→75→38→19)
    Temporal : BiLSTM(128) over 19 compressed timesteps
    Pooling  : Learnable attention
    Head     : Dense(128, BN, h-swish) → Dense(64, relu) → Dense(1, sigmoid)
    """
    inp = keras.layers.Input(shape=(seq_len, num_features), name="input")

    # Stem
    x = keras.layers.Conv1D(STEM_FILTERS, 3, padding="same", use_bias=False,
                            kernel_initializer="he_normal", name="stem_conv")(inp)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = keras.layers.Activation(HARD_SWISH_ACT, name="stem_act")(x)

    # MobileNetV3 inverted residual blocks
    for i, (exp, out, ks, s, se, hs) in enumerate(MBCONV_CONFIG):
        x = _mbconv1d(x, exp, out, ks, s, se, hs, name=f"mb{i}")

    # Neck: 1×1 conv to expand channels before LSTM
    x = keras.layers.Conv1D(NECK_DIM, 1, use_bias=False,
                            kernel_initializer="he_normal", name="neck_conv")(x)
    x = keras.layers.BatchNormalization(name="neck_bn")(x)
    x = keras.layers.Activation(HARD_SWISH_ACT, name="neck_act")(x)

    # recurrent_dropout=0.1 regularises hidden-to-hidden paths.
    # unroll=True keeps TFLite compatibility.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(LSTM_UNITS, return_sequences=True,
                          dropout=0.1, recurrent_dropout=0.1, unroll=True),
        name="bilstm",
    )(x)
    x = keras.layers.LayerNormalization(name="lstm_ln")(x)

    # Attention pooling
    x = _attention_pool(x)

    # Head: BN before activation stabilises scale and speeds convergence.
    # use_bias=False on Dense because BN has its own bias (beta).
    x   = keras.layers.Dense(128, use_bias=False,
                              kernel_initializer="he_normal", name="head_fc1")(x)
    x   = keras.layers.BatchNormalization(name="head_bn1")(x)
    x   = keras.layers.Activation(HARD_SWISH_ACT, name="head_act")(x)
    x   = keras.layers.Dropout(DROPOUT, name="head_drop1")(x)
    x   = keras.layers.Dense(64, activation="relu",
                              kernel_initializer="he_normal", name="head_fc2")(x)
    x   = keras.layers.Dropout(DROPOUT * 0.5, name="head_drop2")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    return keras.Model(inp, out, name=MODEL_NAME)


# ─────────────────────────── Training Helpers ────────────────────────────


def cosine_lr_schedule(epoch: int, _lr: float = None) -> float:
    """Linear warm-up followed by cosine decay with warm restarts (T₀=20, T_mult=2).

    _lr is accepted but ignored — Keras's LearningRateScheduler always passes
    the current learning rate as the second argument, so the parameter must
    exist but should not be used to drive schedule logic.
    """
    total_epochs = EPOCHS
    warmup       = WARMUP_EPOCHS
    peak         = PEAK_LR
    minimum      = MIN_LR

    if epoch < warmup:
        return minimum + (peak - minimum) * epoch / max(warmup, 1)
    t = epoch - warmup
    T_cur = 20
    while t >= T_cur:
        t -= T_cur
        T_cur = min(T_cur * 2, total_epochs)
    progress = t / max(T_cur, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + np.cos(np.pi * progress))


class HardBinaryAccuracy(keras.metrics.Metric):
    """BinaryAccuracy with hard-thresholded labels for MixUp-style soft targets."""

    def __init__(self, name: str = "hard_accuracy", threshold: float = 0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = float(threshold)
        self._inner = keras.metrics.BinaryAccuracy(threshold=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_f = tf.cast(y_true, tf.float32)
        y_true_hard = tf.cast(y_true_f >= 0.5, y_pred.dtype)
        return self._inner.update_state(y_true_hard, y_pred, sample_weight=sample_weight)

    def result(self):
        return self._inner.result()

    def reset_state(self):
        self._inner.reset_state()

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"threshold": self.threshold})
        return cfg


def compile_model(model: keras.Model, focal_alpha: float = FOCAL_ALPHA) -> keras.Model:
    """Compile with data-driven focal alpha and class balancing enabled."""
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=PEAK_LR,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=True,
            gamma=FOCAL_GAMMA,
            alpha=focal_alpha,
            from_logits=False,
        ),
        metrics=[
            HardBinaryAccuracy(name="hard_accuracy", threshold=0.5),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def predict_with_tta(
    model: keras.Model,
    X: np.ndarray,
    n_rounds: int = TTA_ROUNDS,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Average predictions over original + lightly augmented copies (TTA)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    preds = [model.predict(X, verbose=0, batch_size=RUNTIME_BATCH_SIZE).flatten()]
    for r in range(n_rounds):
        X_a  = X + rng.normal(0, 0.02, X.shape).astype(np.float32)
        if r % 2 == 0:
            X_a *= rng.uniform(0.95, 1.05, (len(X_a), 1, 1)).astype(np.float32)
        preds.append(model.predict(X_a, verbose=0, batch_size=RUNTIME_BATCH_SIZE).flatten())
    return np.mean(preds, axis=0)


def predict_probabilities(
    model: keras.Model,
    X: np.ndarray,
    use_tta: bool,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Predict with deployment-selected inference mode (single-pass or TTA)."""
    if use_tta:
        return predict_with_tta(model, X, rng=rng)
    return model.predict(X, verbose=0, batch_size=RUNTIME_BATCH_SIZE).flatten()


def get_callbacks(ckpt_path: str) -> list:
    """EarlyStopping and ModelCheckpoint both monitor val_auc.

    Aligning both on the same metric prevents the scenario where val_loss
    plateaus (triggering ES) while val_auc is still improving.
    ES_PATIENCE=25 ensures training survives the full first cosine cycle
    (T₀=20 epochs) before ES is allowed to fire.
    """
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_auc",
            patience=ES_PATIENCE,
            mode="max",
            restore_best_weights=False,   # checkpoint handles weight restoration
            verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
            verbose=0,
        ),
        keras.callbacks.LearningRateScheduler(cosine_lr_schedule, verbose=0),
    ]


# ─────────────────────────── Cross-Validation ────────────────────────────


def run_cross_validation(df: pd.DataFrame) -> list:
    df = sanitize_features(df, tag="raw input")
    subjects = df["Subject"].values
    # StratifiedGroupKFold preserves subject integrity while balancing class
    # distribution across folds, avoiding the skewed ratios that GroupKFold
    # produces when alert and drowsy videos differ in length.
    gkf          = StratifiedGroupKFold(n_splits=N_FOLDS)
    fold_results: list = []
    best_auc, best_fold = -1.0, -1
    best_tmp_path = os.path.join(BASE_DIR, "_best_fold_tmp.keras")
    if os.path.exists(best_tmp_path):
        os.remove(best_tmp_path)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(df, df["Label"], groups=subjects)
    ):
        tf.keras.backend.clear_session()
        print(f"\n{'='*60}\nFOLD {fold+1}/{N_FOLDS}\n{'='*60}")

        train_subj = df.iloc[train_idx]["Subject"].unique()
        test_subj  = df.iloc[test_idx]["Subject"].unique()

        rng_split = np.random.default_rng(SEED + fold)

        if len(train_subj) < 2:
            print("  SKIP: not enough training subjects to create a validation split")
            tf.keras.backend.clear_session()
            continue
        n_val = max(1, len(train_subj) // 10)
        n_val = min(n_val, len(train_subj) - 1)
        val_subj   = rng_split.choice(train_subj, n_val, replace=False)
        act_train  = np.array([s for s in train_subj if s not in val_subj])

        print(f"  Train: {len(act_train)}  Val: {len(val_subj)}  "
              f"Test: {len(test_subj)} → {sorted(test_subj)}")

        df_tr = df[df["Subject"].isin(act_train)]
        df_va = df[df["Subject"].isin(val_subj)]
        df_te = df[df["Subject"].isin(test_subj)]

        norm_params, global_norm = fit_subject_norm_params(df_tr)
        df_tr = prepare_split_features(
            df_tr, norm_params, global_norm, True, "train", apply_savgol_filter=SAVGOL_ON_TRAIN
        )
        df_va = prepare_split_features(
            df_va, norm_params, global_norm, False, "val", apply_savgol_filter=SAVGOL_ON_EVAL
        )
        df_te = prepare_split_features(
            df_te, norm_params, global_norm, False, "test", apply_savgol_filter=SAVGOL_ON_EVAL
        )

        X_raw, y_tr  = _create_raw_sequences(df_tr, SEQ_LEN, TRAIN_STEP, TEMPORAL_PAD)
        X_val, y_val, g_val = create_sequences(
            df_va, SEQ_LEN, EVAL_STEP, TEMPORAL_PAD, return_groups=True
        )
        X_te, y_te, g_te = create_sequences(
            df_te, SEQ_LEN, EVAL_STEP, TEMPORAL_PAD, return_groups=True
        )

        print(f"  Sequences → train: {len(X_raw)}, val: {len(X_val)}, test: {len(X_te)}")
        if len(X_raw) == 0 or len(X_val) == 0 or len(X_te) == 0:
            print("  SKIP: insufficient sequences")
            tf.keras.backend.clear_session()
            continue

        focal_alpha = _compute_focal_alpha(y_tr)
        train_seq = AugmentedSequence(
            X_raw,
            y_tr,
            batch_size=RUNTIME_BATCH_SIZE,
            n_rounds=AUGMENT_ROUNDS,
            temporal_pad=TEMPORAL_PAD,
            seed=SEED + fold + 100,
        )
        print(f"  Training windows / epoch (base+aug): {train_seq.sample_count:,}")

        with STRATEGY.scope():
            model = compile_model(build_model(), focal_alpha=focal_alpha)
        if fold == 0:
            model.summary()

        ckpt_path = os.path.join(BASE_DIR, f"_fold{fold}_ckpt.weights.h5")
        # Keep default single-process generator consumption. AugmentedSequence
        # has mutable epoch state and is not safe with use_multiprocessing=True.
        model.fit(train_seq, validation_data=(X_val, y_val), epochs=EPOCHS,
                  callbacks=get_callbacks(ckpt_path), verbose=1)

        if os.path.exists(ckpt_path):
            model.load_weights(ckpt_path)
            os.remove(ckpt_path)

        # Threshold tuning and evaluation use deployment-selected inference mode.
        eval_use_tta = DEPLOY_USE_TTA
        val_rng = np.random.default_rng(SEED + fold + 200) if eval_use_tta else None
        te_rng  = np.random.default_rng(SEED + fold + 300) if eval_use_tta else None
        y_val_prob = predict_probabilities(model, X_val, use_tta=eval_use_tta, rng=val_rng)
        y_te_prob  = predict_probabilities(model, X_te, use_tta=eval_use_tta, rng=te_rng)
        thr = resolve_threshold(y_val, y_val_prob, groups=g_val)

        result = _eval_metrics(y_te, y_te_prob, thr, tag=f"Fold {fold+1}", groups=g_te)
        result["fold"] = fold
        result["test_subjects"] = sorted(test_subj.tolist())
        fold_results.append(result)

        if result["auc"] > best_auc:
            best_auc, best_fold = result["auc"], fold
            model.save(best_tmp_path)

        del model
        tf.keras.backend.clear_session()

    if not fold_results:
        if os.path.exists(best_tmp_path):
            os.remove(best_tmp_path)
        print("\nNo valid folds to summarise (all skipped).")
        return []

    # ── CV summary ──
    print(f"\n{'='*60}\nCV SUMMARY\n{'='*60}")
    for m in ["accuracy", "f1", "auc", "prauc", "precision", "recall"]:
        v = [r[m] for r in fold_results]
        print(f"  {m:>10}: {np.mean(v):.4f} ± {np.std(v):.4f}  "
              f"(min={np.min(v):.4f}  max={np.max(v):.4f})")

    thrs = [r["threshold"] for r in fold_results]
    print(f"\n  Thresholds: {', '.join(f'{t:.3f}' for t in thrs)}")
    print(f"  Mean threshold: {np.mean(thrs):.3f}")

    print("\n  --- vs default 0.5 ---")
    for m, dk in [("accuracy","acc_05"),("f1","f1_05"),("recall","rec_05")]:
        vo = np.mean([r[m]  for r in fold_results])
        vd = np.mean([r[dk] for r in fold_results])
        d  = vo - vd
        sign = "+" if d >= 0 else ""
        print(f"  {m:>10}: {vd:.4f} → {vo:.4f}  (Δ{sign}{d:.4f})")

    total_cm = sum(r["confusion_matrix"] for r in fold_results)
    print(f"\nAggregate CM:\n{total_cm}")

    header = f"{'Fold':>4} | {'Thr':>5} | {'Acc':>7} | {'F1':>7} | {'AUC':>7} | {'PR-AUC':>7} | {'Prec':>7} | {'Rec':>7} | Subjects"
    print(f"\n{header}")
    print("-" * len(header))
    for r in fold_results:
        ss = ",".join(str(s) for s in r["test_subjects"][:6])
        if len(r["test_subjects"]) > 6:
            ss += "..."
        print(f"  {r['fold']+1:>2} | {r['threshold']:.3f} | "
              f"{r['accuracy']:.4f}  | {r['f1']:.4f}  | {r['auc']:.4f}  | "
              f"{r['prauc']:.4f}  | {r['precision']:.4f}  | {r['recall']:.4f}  | {ss}")

    if best_fold >= 0 and os.path.exists(best_tmp_path):
        p = os.path.join(BASE_DIR, "best_fold_model.keras")
        shutil.copy2(best_tmp_path, p)
        os.remove(best_tmp_path)
        print(f"\nBest fold model (fold {best_fold+1}, AUC={best_auc:.4f}): {p}")

    return fold_results


# ─────────────────────────── Final Training ──────────────────────────────


def train_final_model(df: pd.DataFrame) -> None:
    df = sanitize_features(df, tag="raw input")
    subjects   = df["Subject"].unique()
    n_subjects = len(subjects)

    if n_subjects < 4:
        raise ValueError(
            f"Need ≥4 subjects for train/val/thr/report split, got {n_subjects}."
        )

    rng_s = np.random.default_rng(SEED + 1000)

    # Three non-overlapping holdout groups:
    #   val_subj    → EarlyStopping & ModelCheckpoint  (tainted by model selection)
    #   thr_subj    → threshold tuning                 (tainted by threshold selection)
    #   report_subj → final metric reporting            (NEVER seen during training)
    n_each   = max(1, n_subjects // 10)
    n_hold   = min(n_subjects - 2, n_each * 3)
    n_val    = max(1, n_hold // 3)
    n_thr    = max(1, (n_hold - n_val) // 2)
    n_report = n_hold - n_val - n_thr   # ≥ 0; may be 0 for tiny datasets
    has_report = n_report > 0

    held_out    = rng_s.choice(subjects, n_hold, replace=False)
    val_subj    = held_out[:n_val]
    thr_subj    = held_out[n_val : n_val + n_thr]
    report_subj = held_out[n_val + n_thr :]
    train_subj  = np.array([s for s in subjects if s not in held_out])

    print(f"\nFinal model splits: {len(train_subj)} train | {len(val_subj)} val | "
          f"{len(thr_subj)} thr-tune | {len(report_subj)} report subjects")

    df_tr   = df[df["Subject"].isin(train_subj)]
    df_va   = df[df["Subject"].isin(val_subj)]
    df_thr  = df[df["Subject"].isin(thr_subj)]
    df_rep  = df[df["Subject"].isin(report_subj)]

    norm_params, global_norm = fit_subject_norm_params(df_tr)
    save_norm_params(norm_params, global_norm)

    df_tr = prepare_split_features(
        df_tr, norm_params, global_norm, True, "train", apply_savgol_filter=SAVGOL_ON_TRAIN
    )
    df_va = prepare_split_features(
        df_va, norm_params, global_norm, False, "val", apply_savgol_filter=SAVGOL_ON_EVAL
    )
    df_thr = prepare_split_features(
        df_thr, norm_params, global_norm, False, "thr", apply_savgol_filter=SAVGOL_ON_EVAL
    )
    df_rep = prepare_split_features(
        df_rep, norm_params, global_norm, False, "report", apply_savgol_filter=SAVGOL_ON_EVAL
    )

    X_raw, y_tr  = _create_raw_sequences(df_tr, SEQ_LEN, TRAIN_STEP, TEMPORAL_PAD)
    X_val, y_val, g_val = create_sequences(
        df_va, SEQ_LEN, EVAL_STEP, TEMPORAL_PAD, return_groups=True
    )

    if len(X_raw) == 0:
        raise ValueError("No training sequences created for final model.")
    if len(X_val) == 0:
        raise ValueError("No validation sequences created for final model.")

    # Compute focal_alpha from hard integer labels before augmentation
    focal_alpha = _compute_focal_alpha(y_tr)
    train_seq = AugmentedSequence(
        X_raw,
        y_tr,
        batch_size=RUNTIME_BATCH_SIZE,
        n_rounds=AUGMENT_ROUNDS,
        temporal_pad=TEMPORAL_PAD,
        seed=SEED,
    )
    if train_seq.sample_count == 0:
        raise ValueError("Augmented training sequence is empty.")

    print(f"  Train windows / epoch: {train_seq.sample_count:,}  Val: {len(X_val):,}")

    with STRATEGY.scope():
        model = compile_model(build_model(), focal_alpha=focal_alpha)
    model.summary()

    ckpt_path = os.path.join(BASE_DIR, "_final_ckpt.weights.h5")
    # Keep default single-process generator consumption. AugmentedSequence
    # has mutable epoch state and is not safe with use_multiprocessing=True.
    model.fit(train_seq, validation_data=(X_val, y_val), epochs=EPOCHS,
              callbacks=get_callbacks(ckpt_path), verbose=1)

    if os.path.exists(ckpt_path):
        model.load_weights(ckpt_path)
        os.remove(ckpt_path)
        print("Restored best checkpoint (val_auc)")

    keras_path = os.path.join(BASE_DIR, f"{MODEL_NAME}.keras")
    model.save(keras_path)
    print(f"\nKeras model saved: {keras_path}")

    # ── Threshold tuning on separate holdout ──
    X_thr, y_thr, g_thr = create_sequences(
        df_thr, SEQ_LEN, EVAL_STEP, TEMPORAL_PAD, return_groups=True
    )
    print(f"  Threshold holdout: {len(X_thr):,} sequences")

    deploy_rng_thr = np.random.default_rng(SEED + 999) if DEPLOY_USE_TTA else None
    if len(X_thr) > 0:
        y_thr_prob = predict_probabilities(model, X_thr, use_tta=DEPLOY_USE_TTA, rng=deploy_rng_thr)
        opt_thr    = resolve_threshold(y_thr, y_thr_prob, groups=g_thr)
    else:
        print("  WARNING: threshold holdout is empty — using 0.5")
        opt_thr    = 0.5
        y_thr_prob = None

    print(f"  Deployment threshold: {opt_thr:.3f}")

    # ── Final metrics ──
    print("\n--- Final Model Metrics ---")

    # Primary honest block — report_subj never seen during training or tuning
    X_rep, y_rep, g_rep = create_sequences(
        df_rep, SEQ_LEN, EVAL_STEP, TEMPORAL_PAD, return_groups=True
    )
    if has_report and len(X_rep) > 0:
        deploy_rng_rep = np.random.default_rng(SEED + 1002) if DEPLOY_USE_TTA else None
        y_rep_prob = predict_probabilities(model, X_rep, use_tta=DEPLOY_USE_TTA, rng=deploy_rng_rep)
        _eval_metrics(y_rep, y_rep_prob, opt_thr,
                      groups=g_rep,
                      tag="Held-out report set (HONEST — unseen during all training)")
    else:
        reason = "no report subjects (dataset too small)" if not has_report else "report set produced no sequences"
        print(f"  WARNING: skipping honest report block — {reason}")

    # Supplementary blocks — labelled so their taint is obvious
    deploy_rng_val = np.random.default_rng(SEED + 1001) if DEPLOY_USE_TTA else None
    y_val_prob = predict_probabilities(model, X_val, use_tta=DEPLOY_USE_TTA, rng=deploy_rng_val)
    _eval_metrics(y_val, y_val_prob, opt_thr,
                  groups=g_val,
                  tag="Validation set (tainted: checkpoint was selected on val_auc)")

    if y_thr_prob is not None:
        _eval_metrics(y_thr, y_thr_prob, opt_thr,
                      groups=g_thr,
                      tag="Threshold holdout (tainted: threshold was tuned on this set)")

    # ── Deploy config ──
    deploy_cfg = {
        "threshold":               opt_thr,
        "smooth_k":                SMOOTH_K,
        "deploy_use_tta":          DEPLOY_USE_TTA,
        "fixed_threshold":         FIXED_THRESHOLD,
        "int8_full_io":            INT8_FULL_IO,
        "runtime_batch_size":      RUNTIME_BATCH_SIZE,
        "runtime_strategy":        STRATEGY_NAME,
        "runtime_replicas":        NUM_REPLICAS,
        "runtime_xla_enabled":     ENABLE_XLA,
        "runtime_mp_policy":       MIXED_PRECISION_POLICY,
        "focal_alpha_used":        focal_alpha,
        "normalization_mode":      "train_subject_anchor_with_global_fallback_for_unseen_subjects",
        "savgol_on_train":         SAVGOL_ON_TRAIN,
        "savgol_on_eval":          SAVGOL_ON_EVAL,
        "sequence_length":         SEQ_LEN,
        "num_raw_features":        NUM_RAW,
        "num_features":            NUM_FEATURES,
        "feature_names":           FEATURES,
        "rolling_window":          ROLLING_WINDOW,
        "perclos_ear_z_threshold": PERCLOS_EAR_Z_THRESHOLD,
    }
    cfg_path = os.path.join(BASE_DIR, "deploy_config.json")
    with open(cfg_path, "w") as fh:
        json.dump(deploy_cfg, fh, indent=2)
    print(f"Deploy config saved: {cfg_path}")

    # ── TFLite export ──
    # X_val is used for INT8 calibration — it is clean and unaugmented,
    # matching the real inference distribution.
    cal_rng = np.random.default_rng(SEED)
    cal_idx = cal_rng.permutation(len(X_val))[:min(300, len(X_val))]
    X_cal   = X_val[cal_idx].astype(np.float32)

    sm_dir = os.path.join(BASE_DIR, "_saved_model_tmp")
    model.export(sm_dir)

    # Float16
    conv_f16 = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv_f16.optimizations = [tf.lite.Optimize.DEFAULT]
    conv_f16.target_spec.supported_types = [tf.float16]
    tfl_f16   = conv_f16.convert()
    f16_path  = os.path.join(BASE_DIR, f"{MODEL_NAME}_f16.tflite")
    with open(f16_path, "wb") as fh:
        fh.write(tfl_f16)
    print(f"Float16 TFLite: {f16_path} ({len(tfl_f16)/1024:.1f} KB)")

    # INT8 PTQ
    conv_i8 = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv_i8.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset():
        for i in range(len(X_cal)):
            yield [X_cal[i : i + 1]]

    conv_i8.representative_dataset = representative_dataset
    conv_i8.target_spec.supported_ops  = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    if INT8_FULL_IO:
        conv_i8.inference_input_type = tf.int8
        conv_i8.inference_output_type = tf.int8
        int8_mode = "full-int8-io"
    else:
        conv_i8.inference_input_type = tf.float32
        conv_i8.inference_output_type = tf.float32
        int8_mode = "hybrid-f32-io"
    tfl_i8   = conv_i8.convert()
    i8_path  = os.path.join(BASE_DIR, f"{MODEL_NAME}_int8.tflite")
    with open(i8_path, "wb") as fh:
        fh.write(tfl_i8)
    print(f"INT8 TFLite ({int8_mode}): {i8_path} ({len(tfl_i8)/1024:.1f} KB)")

    shutil.rmtree(sm_dir, ignore_errors=True)

    def _prepare_tflite_input(sample_f32: np.ndarray, input_detail: dict) -> np.ndarray:
        dtype = input_detail["dtype"]
        if dtype == np.int8:
            scale, zero = input_detail["quantization"]
            if scale <= 0:
                raise ValueError("Invalid int8 quantization scale for TFLite input.")
            q = np.round(sample_f32 / scale + zero)
            return np.clip(q, -128, 127).astype(np.int8)
        return sample_f32.astype(dtype, copy=False)

    # ── Benchmark ──
    print("\n--- Inference Benchmark ---")
    sample = X_cal[0:1]
    int8_label = "INT8" if INT8_FULL_IO else "INT8-hybrid"
    for label, path in [("Float16", f16_path), (int8_label, i8_path)]:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()
        sample_in = _prepare_tflite_input(sample, inp_det[0])

        for _ in range(20):
            interp.set_tensor(inp_det[0]["index"], sample_in)
            interp.invoke()

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            interp.set_tensor(inp_det[0]["index"], sample_in)
            interp.invoke()
            times.append(time.perf_counter() - t0)

        ms = np.array(times) * 1000
        print(f"  {label:>8}: avg={np.mean(ms):.2f}ms  "
              f"p50={np.median(ms):.2f}ms  p95={np.percentile(ms,95):.2f}ms")


# ─────────────────────────── Norm Params ─────────────────────────────────


def save_norm_params(params: dict, global_params: dict | None = None) -> None:
    """Save train-subject norm params plus a deployment-safe global fallback."""
    if not params:
        raise ValueError("Cannot save empty normalisation params.")
    if global_params is None:
        global_params = {
            feat: {
                "median": float(np.median([params[s][feat]["median"] for s in params])),
                "iqr": float(np.median([params[s][feat]["iqr"] for s in params])),
            }
            for feat in FEATURES
        }
    p = os.path.join(BASE_DIR, "norm_params.json")
    with open(p, "w") as fh:
        json.dump(params, fh, indent=2)
    print(f"Norm params saved:        {p}")

    pg = os.path.join(BASE_DIR, "norm_global_params.json")
    with open(pg, "w") as fh:
        json.dump(global_params, fh, indent=2)
    print(f"Global norm params saved: {pg}")


# ─────────────────────────── Main ────────────────────────────────────────


def main() -> None:
    dev = f"GPU x{GPU_COUNT}" if GPU_COUNT > 0 else "CPU"
    print(f"TF {tf.__version__} | {dev} | strategy={STRATEGY_NAME} | "
          f"replicas={NUM_REPLICAS} | batch={RUNTIME_BATCH_SIZE} | "
          f"xla={ENABLE_XLA} | mp={MIXED_PRECISION_POLICY} | "
          f"inter={tf.config.threading.get_inter_op_parallelism_threads()} "
          f"intra={tf.config.threading.get_intra_op_parallelism_threads()} "
          f"cpus={CPU_COUNT}")

    tf.keras.utils.set_random_seed(SEED)
    if DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()

    print("\nLoading data...")
    df = load_data(CSV_FILE)

    if not SKIP_CV:
        print("\n" + "=" * 60)
        print("PHASE 1: Cross-Validation")
        print("=" * 60)
        run_cross_validation(df)
    else:
        print("\nSkipping cross-validation (SKIP_CV=True)")

    print("\n" + "=" * 60)
    print("PHASE 2: Final Model & Export")
    print("=" * 60)
    train_final_model(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
