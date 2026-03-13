"""
Drowsiness detection model trainer v4 — MobileNetV3-1D + BiLSTM.

Architecture:  MobileNetV3-Small (1D) backbone → BiLSTM → Attention pooling
Rationale:
  - MobileNetV3 inverted residual blocks with SE + h-swish are parameter-efficient
  - Depthwise separable convolutions learn per-feature temporal patterns
  - 3 stride-2 stages compress 90→12 timesteps, extracting hierarchical features
  - BiLSTM integrates compressed features across time
  - Attention pooling focuses on the most informative compressed timesteps
  - ~260K params — efficient for TFLite deployment

Key features:
  1. MobileNetV3-style inverted residual blocks (depthwise + SE + h-swish)
  2. BiLSTM for temporal modeling on compressed features
    3. Cosine LR with warm restarts + optional SWA + TTA
  4. Robust anchored normalisation + Savitzky-Golay denoising
    5. 33 engineered features: raw(10) + delta(10) + ddelta(10) + temporal(3)
  6. Grouped 5-fold CV by subject (no leakage)
  7. F-beta (F2) threshold optimisation (recall-biased for safety)
    8. Focal loss + optional mixup + time-warp augmentation
  9. TFLite Float16 + INT8 (PTQ) export via SavedModel

Pipeline:
  1. Load CSV → filter binary labels
  2. Anchored normalisation (median/IQR from alert video per subject)
  3. Savitzky-Golay smoothing per video
  4. Compute enhanced features (deltas + rolling temporal)
  5. Create sequences (90 frames + 29 cold-start padding)
  6. Augment (jitter + scale + mask + time-warp + mixup)
  7. 5-fold GroupKFold CV → evaluate with F2-optimal threshold
  8. Final model on all data → Keras + TFLite export
"""

import os
import json
import shutil
import time
import warnings

CPU_COUNT = max(1, os.cpu_count() or 1)
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", str(CPU_COUNT))
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(CPU_COUNT))

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
import tensorflow as tf
from tensorflow import keras

try:
    from scipy.signal import savgol_filter as _savgol
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

tf.config.threading.set_inter_op_parallelism_threads(CPU_COUNT)
tf.config.threading.set_intra_op_parallelism_threads(CPU_COUNT)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix,
    classification_report,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# ───────────────────────────── Configuration ─────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE = os.path.join(BASE_DIR, "uta_rldd_features.csv")

FEATURES = [
    "EAR_Left", "EAR_Right", "EAR_Avg", "MAR",
    "PUC_Left", "PUC_Right", "MUC",
    "Pitch", "Yaw", "Roll",
]
NUM_RAW = len(FEATURES)                     # 10
NUM_FEATURES = NUM_RAW * 3 + 3              # 33

ROLLING_WINDOW = 30                         # 1 s at 30 fps
TEMPORAL_PAD = ROLLING_WINDOW - 1           # 29 cold-start frames

# PERCLOS proxy on anchored-normalized EAR_Avg: EAR below this threshold
# is treated as eye-closure for rolling closure ratio.
PERCLOS_EAR_Z_THRESHOLD = -0.25

SAVGOL_WINDOW = 7
SAVGOL_POLY = 2

SEQ_LEN = 90                               # 3 s at 30 fps
TRAIN_STEP = 15
EVAL_STEP = 30

# ── MobileNetV3-1D backbone config ──
# (expand_channels, output_channels, kernel_size, stride, use_se, use_hswish)
MBCONV_CONFIG = [
    (32,  16, 3, 1, True,  False),          # B0: 90→90, 16ch
    (48,  24, 3, 2, False, False),          # B1: 90→45, 24ch
    (48,  24, 3, 1, True,  False),          # B2: 45→45, 24ch
    (64,  32, 5, 2, True,  True),           # B3: 45→23, 32ch
    (96,  32, 5, 1, True,  True),           # B4: 23→23, 32ch
    (96,  48, 5, 1, True,  True),           # B5: 23→23, 48ch
    (144, 64, 5, 2, True,  True),           # B6: 23→12, 64ch
    (192, 64, 5, 1, True,  True),           # B7: 12→12, 64ch
]
STEM_FILTERS = 16
NECK_DIM = 128
SE_RATIO = 4

# ── LSTM ──
LSTM_UNITS = 64                             # per direction → 128 total
ATTN_UNITS = 64

# ── Classifier head ──
DROPOUT = 0.35

# ── Training ──
EPOCHS = 100
BATCH_SIZE = 256
PEAK_LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 5
MIN_LR = 1e-6
SWA_START_FRAC = 0.75

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.5                           # balanced (F2 threshold handles recall)

AUGMENT_ROUNDS = 3
MIXUP_ALPHA = 0.3
TTA_ROUNDS = 4
LABEL_SMOOTH = 0.05

USE_MIXUP = True                            # mixup applied post-enhancement if True
USE_SWA = False                             # SWA disabled to avoid EarlyStopping conflict
MIXUP_FRACTION = 0.35                       # in-place mixup on subset to limit memory
MIXUP_CHUNK = 16384                         # chunk size for memory-safe in-place mixup

F_BETA = 2.0
N_FOLDS = 5
SKIP_CV = False
SEED = 42
DETERMINISTIC = False

ES_PATIENCE = 25                            # increased to survive LR warm restarts

MODEL_NAME = "drowsiness_mv3_lstm"


# ─────────────────────────── Data Loading ────────────────────────────────


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["Label"].isin([0, 1])].copy()
    df = df.sort_values(["Subject", "Video_File", "Frame"]).reset_index(drop=True)
    print(f"Loaded {len(df):,} frames from {df['Subject'].nunique()} subjects")
    print(f"  Alert: {(df['Label']==0).sum():,}  Drowsy: {(df['Label']==1).sum():,}")
    return df


def anchored_normalize(df):
    """Per-subject normalisation: median/IQR from alert video."""
    norm_params = {}
    groups = []
    for subj, grp in df.groupby("Subject"):
        grp = grp.copy()
        alert = grp[grp["Label"] == 0]
        if len(alert) < 100:
            alert = grp
        params = {}
        for f in FEATURES:
            med = float(alert[f].median())
            iqr = float(alert[f].quantile(0.75) - alert[f].quantile(0.25))
            if iqr < 1e-6:
                iqr = 1.0
            grp[f] = (grp[f] - med) / iqr
            params[f] = {"median": med, "iqr": iqr}
        norm_params[str(subj)] = params
        groups.append(grp)
    return pd.concat(groups), norm_params


def apply_savgol(df):
    if not _HAS_SCIPY:
        print("  WARNING: scipy missing — skipping Savitzky-Golay")
        return df
    parts = []
    for _, grp in df.groupby("Video_File"):
        g = grp.copy()
        if len(g) >= SAVGOL_WINDOW:
            for f in FEATURES:
                g[f] = _savgol(g[f].values, SAVGOL_WINDOW, SAVGOL_POLY)
        parts.append(g)
    return pd.concat(parts)


# ─────────────────────────── Feature Engineering ─────────────────────────


def apply_enhanced_features(X_raw):
    """Vectorised: (N, T, 10) → (N, T, 33) [raw + delta + ddelta + temporal]."""
    N, T, _ = X_raw.shape

    delta = np.zeros_like(X_raw)
    delta[:, 1:] = X_raw[:, 1:] - X_raw[:, :-1]
    ddelta = np.zeros_like(delta)
    ddelta[:, 1:] = delta[:, 1:] - delta[:, :-1]

    ear = X_raw[:, :, 2]   # EAR_Avg
    pd_ = delta[:, :, 7]   # Pitch delta

    ear_pad = np.pad(ear, ((0, 0), (ROLLING_WINDOW - 1, 0)), mode="edge")
    ear_win = sliding_window_view(ear_pad, ROLLING_WINDOW, axis=1)
    ear_var = np.var(ear_win, axis=2).astype(np.float32)

    t_idx = np.arange(T)
    starts = np.maximum(t_idx - ROLLING_WINDOW + 1, 0)
    wlens = (t_idx - starts + 1).astype(np.float32)
    abs_pd = np.abs(pd_)
    cs = np.cumsum(abs_pd, axis=1)
    cs_pad = np.concatenate([np.zeros((N, 1), dtype=np.float32), cs], axis=1)
    head_move = ((cs_pad[:, t_idx + 1] - cs_pad[:, starts]) / wlens).astype(
        np.float32
    )

    closed = (ear < PERCLOS_EAR_Z_THRESHOLD).astype(np.float32)
    closed_pad = np.pad(
        closed, ((0, 0), (ROLLING_WINDOW - 1, 0)), mode="edge"
    )
    closed_win = sliding_window_view(closed_pad, ROLLING_WINDOW, axis=1)
    perclos = np.mean(closed_win, axis=2).astype(np.float32)

    temporal = np.stack([ear_var, head_move, perclos], axis=-1)
    return np.concatenate([X_raw, delta, ddelta, temporal], axis=-1)


# ─────────────────────────── Sequence Creation ───────────────────────────


def create_raw_sequences(df, seq_len, step, temporal_pad=0):
    """Raw 10-feature sequences grouped by video, with cold-start padding."""
    padded_len = seq_len + temporal_pad
    X, y = [], []
    for _, grp in df.groupby("Video_File"):
        raw = grp[FEATURES].values.astype(np.float32)
        labels = grp["Label"].values
        if len(raw) < seq_len:
            continue
        for i in range(0, len(raw) - seq_len + 1, step):
            ps = max(0, i - temporal_pad)
            chunk = raw[ps : i + seq_len]
            if chunk.shape[0] < padded_len:
                # Edge-pad instead of first-frame replication to reduce transients
                pad_len = padded_len - chunk.shape[0]
                chunk = np.pad(chunk, ((pad_len, 0), (0, 0)), mode="edge")
            X.append(chunk)
            y.append(labels[i + seq_len - 1])
    if not X:
        return (
            np.empty((0, padded_len, NUM_RAW), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)


def create_sequences(df, seq_len, step, temporal_pad=0):
    """Full 33-feature sequences: raw → enhanced → trim padding."""
    X_raw, y = create_raw_sequences(df, seq_len, step, temporal_pad)
    if len(X_raw) == 0:
        return np.empty((0, seq_len, NUM_FEATURES), dtype=np.float32), y
    X = apply_enhanced_features(X_raw)
    if temporal_pad > 0:
        X = X[:, temporal_pad:, :]
    return X, y


# ─────────────────────────── Augmentation ────────────────────────────────


def time_warp(seq, rng, sigma=0.2):
    T = seq.shape[0]
    warp = np.cumsum(np.maximum(rng.normal(1.0, sigma, T), 0.1))
    warp = warp / warp[-1] * (T - 1)
    warp = np.clip(warp, 0, T - 1)
    orig = np.arange(T, dtype=np.float64)
    out = np.empty_like(seq)
    for c in range(seq.shape[1]):
        out[:, c] = np.interp(warp, orig, seq[:, c])
    return out


def augment_batch(X, y, rng):
    Xa = X.copy()
    n = Xa.shape[0]

    m = rng.random(n) < 0.5
    if m.any():
        Xa[m] += rng.normal(0, 0.05, Xa[m].shape).astype(np.float32)

    m = rng.random(n) < 0.3
    if m.any():
        Xa[m] *= rng.uniform(0.8, 1.2, (m.sum(), 1, 1)).astype(np.float32)

    # Feature masking: exclude EAR_Avg (idx 2) to avoid corrupting PERCLOS
    maskable_features = [0, 1, 3, 4, 5, 6, 7, 8, 9]  # all except EAR_Avg
    m = rng.random(n) < 0.2
    for i in np.where(m)[0]:
        fi = rng.choice(maskable_features, rng.integers(1, 3), replace=False)
        Xa[i, :, fi] = 0.0

    m = rng.random(n) < 0.3
    for i in np.where(m)[0]:
        Xa[i] = time_warp(Xa[i], rng, sigma=0.2)

    return Xa, y


def mixup_batch(X, y, rng, alpha=MIXUP_ALPHA):
    n = len(X)
    idx = rng.permutation(n)
    lam = rng.beta(alpha, alpha, (n, 1, 1)).astype(np.float32)
    X_m = lam * X + (1 - lam) * X[idx]
    lam_f = lam.reshape(-1)
    y_m = lam_f * y.astype(np.float32) + (1 - lam_f) * y[idx].astype(np.float32)
    return X_m, y_m


def apply_inplace_mixup(X, y, rng, alpha=MIXUP_ALPHA,
                        fraction=MIXUP_FRACTION, chunk_size=MIXUP_CHUNK):
    """Apply mixup in place on a subset to avoid allocating a full extra tensor."""
    n = len(X)
    if n < 2 or fraction <= 0.0:
        return X, y

    k = int(n * fraction)
    if k <= 0:
        return X, y

    tgt = rng.choice(n, size=k, replace=False)
    src = rng.permutation(n)[:k]

    # Process in chunks to keep temporary arrays small.
    for s in range(0, k, chunk_size):
        e = min(s + chunk_size, k)
        t = tgt[s:e]
        p = src[s:e]

        lam = rng.beta(alpha, alpha, (len(t), 1, 1)).astype(np.float32)
        lam_f = lam.reshape(-1)

        x_t = X[t].copy()
        y_t = y[t].copy()
        X[t] = lam * x_t + (1 - lam) * X[p]
        y[t] = lam_f * y_t + (1 - lam_f) * y[p]

    return X, y


def build_augmented_dataset(X_raw, y, rng, n_rounds=AUGMENT_ROUNDS,
                            temporal_pad=0):
    """Augment raw seqs → compute enhanced features → trim padding.
    
    Mixup (if USE_MIXUP) is applied AFTER feature engineering to avoid
    corrupting threshold-based derived features like PERCLOS.
    """
    parts_X = [X_raw]
    parts_y = [y.astype(np.float32)]

    for _ in range(n_rounds):
        Xa, ya = augment_batch(X_raw, y, rng)
        parts_X.append(Xa)
        parts_y.append(ya.astype(np.float32))

    # Concatenate and apply label smoothing to hard-label samples
    X_all = np.concatenate(parts_X, axis=0)
    y_all = np.concatenate(parts_y, axis=0)
    y_all = y_all * (1 - 2 * LABEL_SMOOTH) + LABEL_SMOOTH

    # Shuffle before feature engineering
    perm = rng.permutation(len(X_all))
    X_all, y_all = X_all[perm], y_all[perm]

    # Compute enhanced features (including threshold-based PERCLOS)
    X_enh = apply_enhanced_features(X_all)
    if temporal_pad > 0:
        X_enh = X_enh[:, temporal_pad:, :]

    # Apply mixup AFTER feature engineering (on 33-feature vectors).
    # Done in place on a subset to avoid doubling RAM usage.
    if USE_MIXUP:
        X_enh, y_all = apply_inplace_mixup(X_enh, y_all, rng)

    return X_enh, y_all


# ─────────────────────────── Threshold ───────────────────────────────────


def find_optimal_threshold(y_true, y_prob, f_beta=F_BETA):
    """Sweep thresholds to maximise F-beta (F2 by default)."""
    if len(np.unique(y_true)) < 2:
        return 0.5
    best_score, best_thr = -1.0, 0.5
    beta_sq = f_beta ** 2
    for thr in np.linspace(0.10, 0.90, 161):
        pred = (y_prob >= thr).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        fn = np.sum((pred == 0) & (y_true == 1))
        pr = tp / (tp + fp) if (tp + fp) else 0.0
        rc = tp / (tp + fn) if (tp + fn) else 0.0
        if pr + rc > 0:
            fb = (1 + beta_sq) * pr * rc / (beta_sq * pr + rc)
        else:
            fb = 0.0
        if fb > best_score:
            best_score, best_thr = fb, thr
    return float(best_thr)


# ─────────────────────────── Model ───────────────────────────────────────


class HSwish(keras.layers.Layer):
    """Hard-swish: x * relu6(x + 3) / 6."""
    def call(self, x):
        return x * tf.nn.relu6(x + 3.0) * (1.0 / 6.0)


class HSigmoid(keras.layers.Layer):
    """Hard-sigmoid: relu6(x + 3) / 6."""
    def call(self, x):
        return tf.nn.relu6(x + 3.0) * (1.0 / 6.0)


class SumOverTime(keras.layers.Layer):
    """Sum over the time axis (axis=1). TFLite-safe."""
    def call(self, x):
        return tf.reduce_sum(x, axis=1)


def _mbconv1d(x, expand, out, kernel, stride, use_se, use_hs, name="mb"):
    """MobileNetV3-style inverted residual block (1D).

    expand → depthwise → [SE] → project, with optional residual.
    """
    in_ch = x.shape[-1]
    Act = HSwish if use_hs else keras.layers.ReLU

    # ── Expand phase ──
    if expand != in_ch:
        h = keras.layers.Conv1D(
            expand, 1, use_bias=False,
            kernel_initializer="he_normal", name=f"{name}_exp",
        )(x)
        h = keras.layers.BatchNormalization(name=f"{name}_exp_bn")(h)
        h = Act(name=f"{name}_exp_act")(h)
    else:
        h = x

    # ── Depthwise phase (grouped conv = depthwise when groups == channels) ──
    h = keras.layers.Conv1D(
        expand, kernel, strides=stride, padding="same",
        groups=expand, use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_dw",
    )(h)
    h = keras.layers.BatchNormalization(name=f"{name}_dw_bn")(h)
    h = Act(name=f"{name}_dw_act")(h)

    # ── Squeeze-and-Excitation ──
    if use_se:
        mid = max(expand // SE_RATIO, 8)
        se = keras.layers.GlobalAveragePooling1D(
            keepdims=True, name=f"{name}_se_gap",
        )(h)
        se = keras.layers.Dense(
            mid, activation="relu",
            kernel_initializer="he_normal", name=f"{name}_se_r",
        )(se)
        se = keras.layers.Dense(
            expand, kernel_initializer="he_normal", name=f"{name}_se_e",
        )(se)
        se = HSigmoid(name=f"{name}_se_hs")(se)
        h = keras.layers.Multiply(name=f"{name}_se_m")([h, se])

    # ── Project phase (linear bottleneck, no activation) ──
    h = keras.layers.Conv1D(
        out, 1, use_bias=False,
        kernel_initializer="he_normal", name=f"{name}_proj",
    )(h)
    h = keras.layers.BatchNormalization(name=f"{name}_proj_bn")(h)

    # ── Residual skip ──
    if stride == 1 and in_ch == out:
        h = keras.layers.Add(name=f"{name}_res")([h, x])

    return h


def _attention_pool(x):
    """Learnable attention pooling over timesteps."""
    score = keras.layers.Dense(ATTN_UNITS, activation="tanh",
                               kernel_initializer="he_normal")(x)
    score = keras.layers.Dense(1, kernel_initializer="he_normal")(score)
    alpha = keras.layers.Softmax(axis=1, name="attn_weights")(score)
    ctx = keras.layers.Multiply()([x, alpha])
    return SumOverTime()(ctx)


def build_model(seq_len=SEQ_LEN, num_features=NUM_FEATURES):
    """MobileNetV3-Small 1D → BiLSTM → Attention → Dense → sigmoid.

    Backbone: Stem + 8 MBConv1D blocks (3 stride-2 stages: 90→45→23→12)
    Temporal: BiLSTM(128) over 12 compressed timesteps
    Pooling:  Learnable attention
    Head:     Dense(128, h-swish) → Dense(64, relu) → Dense(1, sigmoid)
    ~260K parameters.
    """
    inp = keras.layers.Input(shape=(seq_len, num_features), name="input")

    # ── Stem ──
    x = keras.layers.Conv1D(
        STEM_FILTERS, 3, padding="same", use_bias=False,
        kernel_initializer="he_normal", name="stem_conv",
    )(inp)
    x = keras.layers.BatchNormalization(name="stem_bn")(x)
    x = HSwish(name="stem_act")(x)

    # ── MobileNetV3 inverted residual blocks ──
    for i, (exp, out, ks, s, se, hs) in enumerate(MBCONV_CONFIG):
        x = _mbconv1d(x, exp, out, ks, s, se, hs, name=f"mb{i}")

    # ── Neck (1×1 conv to expand before LSTM) ──
    x = keras.layers.Conv1D(
        NECK_DIM, 1, use_bias=False,
        kernel_initializer="he_normal", name="neck_conv",
    )(x)
    x = keras.layers.BatchNormalization(name="neck_bn")(x)
    x = HSwish(name="neck_act")(x)

    # ── BiLSTM (unroll=True for TFLite) ──
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(
            LSTM_UNITS, return_sequences=True,
            dropout=0.1, recurrent_dropout=0.0,
            unroll=True,
        ),
        name="bilstm",
    )(x)
    x = keras.layers.LayerNormalization(name="lstm_ln")(x)

    # ── Attention pooling ──
    x = _attention_pool(x)

    # ── Classification head ──
    x = keras.layers.Dense(128, kernel_initializer="he_normal",
                           name="head_fc1")(x)
    x = HSwish(name="head_act")(x)
    x = keras.layers.Dropout(DROPOUT, name="head_drop1")(x)
    x = keras.layers.Dense(64, activation="relu",
                           kernel_initializer="he_normal",
                           name="head_fc2")(x)
    x = keras.layers.Dropout(DROPOUT * 0.5, name="head_drop2")(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inp, out, name=MODEL_NAME)
    return model


# ─────────────────────────── Training Helpers ────────────────────────────


def cosine_lr_schedule(epoch, total_epochs=EPOCHS, warmup=WARMUP_EPOCHS,
                       peak=PEAK_LR, minimum=MIN_LR):
    """Cosine decay with linear warmup and warm restarts (T_0=20, T_mult=2)."""
    if epoch < warmup:
        return minimum + (peak - minimum) * epoch / max(warmup, 1)
    t = epoch - warmup
    T_cur = 20
    while t >= T_cur:
        t -= T_cur
        T_cur = min(T_cur * 2, total_epochs)
    progress = t / max(T_cur, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + np.cos(np.pi * progress))


def compile_model(model):
    class HardBinaryAccuracy(keras.metrics.Metric):
        """Binary accuracy that ignores mixed labels from mixup."""

        def __init__(self, threshold=0.5, name="hard_accuracy", **kwargs):
            super().__init__(name=name, **kwargs)
            self.threshold = threshold
            self.correct = self.add_weight(name="correct", initializer="zeros")
            self.count = self.add_weight(name="count", initializer="zeros")

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
            y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

            # Ignore labels in the ambiguous middle range introduced by mixup.
            hard_mask = tf.logical_or(y_true <= 0.25, y_true >= 0.75)
            y_true_h = tf.cast(y_true >= self.threshold, tf.float32)
            y_pred_h = tf.cast(y_pred >= self.threshold, tf.float32)
            matches = tf.cast(tf.equal(y_true_h, y_pred_h), tf.float32)

            matches = tf.boolean_mask(matches, hard_mask)
            batch_n = tf.cast(tf.size(matches), tf.float32)

            self.correct.assign_add(tf.reduce_sum(matches))
            self.count.assign_add(batch_n)

        def result(self):
            return tf.math.divide_no_nan(self.correct, self.count)

        def reset_state(self):
            self.correct.assign(0.0)
            self.count.assign(0.0)

    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=PEAK_LR,
            weight_decay=WEIGHT_DECAY,
            clipnorm=1.0,
        ),
        loss=keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            gamma=FOCAL_GAMMA,
            alpha=FOCAL_ALPHA,
            from_logits=False,
        ),
        metrics=[
            HardBinaryAccuracy(),
            keras.metrics.AUC(name="auc"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class SWACallback(keras.callbacks.Callback):
    """Stochastic Weight Averaging: average weights from later epochs."""
    def __init__(self, swa_start_epoch):
        super().__init__()
        self.swa_start = swa_start_epoch
        self.swa_weights = None
        self.swa_n = 0

    def on_epoch_end(self, epoch, logs=None):
        if epoch >= self.swa_start:
            w = self.model.get_weights()
            if self.swa_weights is None:
                self.swa_weights = [wi.copy() for wi in w]
            else:
                for i, wi in enumerate(w):
                    self.swa_weights[i] = (
                        self.swa_weights[i] * self.swa_n + wi
                    ) / (self.swa_n + 1)
            self.swa_n += 1

    def on_train_end(self, logs=None):
        if self.swa_weights is not None and self.swa_n >= 5:
            self.model.set_weights(self.swa_weights)
            print(f"    SWA: averaged weights from {self.swa_n} epochs")


def predict_with_tta(model, X, n_rounds=TTA_ROUNDS, rng=None):
    """Test-time augmentation: average over original + jittered/scaled copies."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    preds = [model.predict(X, verbose=0, batch_size=BATCH_SIZE).flatten()]
    for r in range(n_rounds):
        X_aug = X.copy()
        X_aug += rng.normal(0, 0.02, X_aug.shape).astype(np.float32)
        if r % 2 == 0:
            scale = rng.uniform(0.95, 1.05, (len(X_aug), 1, 1)).astype(np.float32)
            X_aug *= scale
        preds.append(model.predict(X_aug, verbose=0, batch_size=BATCH_SIZE).flatten())
    return np.mean(preds, axis=0)


def get_callbacks(total_epochs=EPOCHS):
    # Use val_loss for early stopping to avoid AUC instability on rare
    # single-class validation splits.
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=ES_PATIENCE,
            mode="min",
            # If SWA is enabled, avoid restoring ES best weights at train end,
            # because SWA also sets weights in on_train_end.
            restore_best_weights=(not USE_SWA),
            verbose=1,
        ),
        keras.callbacks.LearningRateScheduler(cosine_lr_schedule, verbose=0),
    ]
    if USE_SWA:
        swa_start = int(total_epochs * SWA_START_FRAC)
        callbacks.append(SWACallback(swa_start_epoch=swa_start))
    return callbacks


# ─────────────────────────── Cross-Validation ────────────────────────────


def run_cross_validation(df):
    subjects = df["Subject"].values
    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_results = []
    best_auc, best_fold = -1.0, -1
    best_tmp_path = os.path.join(BASE_DIR, "_best_fold_tmp.keras")
    if os.path.exists(best_tmp_path):
        os.remove(best_tmp_path)

    for fold, (train_idx, test_idx) in enumerate(
        gkf.split(df, df["Label"], groups=subjects)
    ):
        # Release graphs/state from previous fold before building next model.
        tf.keras.backend.clear_session()

        print(f"\n{'='*60}\nFOLD {fold+1}/{N_FOLDS}\n{'='*60}")

        train_subj = df.iloc[train_idx]["Subject"].unique()
        test_subj = df.iloc[test_idx]["Subject"].unique()

        rng_split = np.random.default_rng(SEED + fold)
        rng_aug = np.random.default_rng(SEED + fold + 100)

        n_val = max(2, len(train_subj) // 10)
        val_subj = rng_split.choice(train_subj, n_val, replace=False)
        act_train_subj = np.array([s for s in train_subj if s not in val_subj])

        print(f"  Train: {len(act_train_subj)}  Val: {len(val_subj)}  "
              f"Test: {len(test_subj)} → {sorted(test_subj)}")

        df_tr = df[df["Subject"].isin(act_train_subj)]
        df_va = df[df["Subject"].isin(val_subj)]
        df_te = df[df["Subject"].isin(test_subj)]

        X_raw, y_tr = create_raw_sequences(
            df_tr, SEQ_LEN, TRAIN_STEP, temporal_pad=TEMPORAL_PAD)
        X_val, y_val = create_sequences(
            df_va, SEQ_LEN, EVAL_STEP, temporal_pad=TEMPORAL_PAD)
        X_te, y_te = create_sequences(
            df_te, SEQ_LEN, EVAL_STEP, temporal_pad=TEMPORAL_PAD)

        print(f"  Sequences → train: {len(X_raw)}, val: {len(X_val)}, "
              f"test: {len(X_te)}")

        if len(X_raw) == 0 or len(X_val) == 0 or len(X_te) == 0:
            print("  SKIP: insufficient sequences")
            tf.keras.backend.clear_session()
            continue

        X_tr_full, y_tr_full = build_augmented_dataset(
            X_raw, y_tr, rng_aug, temporal_pad=TEMPORAL_PAD)
        print(f"  After augmentation: {len(X_tr_full):,}")

        model = build_model()
        model = compile_model(model)

        if fold == 0:
            model.summary()

        train_ds = (
            tf.data.Dataset.from_tensor_slices((X_tr_full, y_tr_full))
            .shuffle(min(50_000, len(X_tr_full)))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )
        val_ds = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE)
        )

        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=get_callbacks(),
            verbose=1,
        )

        # Evaluate with TTA
        tta_rng = np.random.default_rng(SEED + fold + 200)
        y_prob = predict_with_tta(model, X_te, rng=tta_rng)
        y_val_prob = predict_with_tta(model, X_val, rng=tta_rng)
        thr = find_optimal_threshold(y_val, y_val_prob)

        y_pred = (y_prob >= thr).astype(int)
        y_pred_d = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        auc = (roc_auc_score(y_te, y_prob)
               if len(np.unique(y_te)) > 1 else 0.0)
        prec = precision_score(y_te, y_pred, zero_division=0)
        rec = recall_score(y_te, y_pred, zero_division=0)
        cm = confusion_matrix(y_te, y_pred)

        result = {
            "fold": fold,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "precision": prec,
            "recall": rec,
            "confusion_matrix": cm,
            "threshold": thr,
            "test_subjects": sorted(test_subj),
            "acc_default": accuracy_score(y_te, y_pred_d),
            "f1_default": f1_score(y_te, y_pred_d, zero_division=0),
            "rec_default": recall_score(y_te, y_pred_d, zero_division=0),
        }
        fold_results.append(result)

        if auc > best_auc:
            best_auc, best_fold = auc, fold
            model.save(best_tmp_path)

        print(f"\n  Fold {fold+1} (thr={thr:.3f}):")
        print(f"    Acc:  {acc:.4f} (0.5: {result['acc_default']:.4f})")
        print(f"    F1:   {f1:.4f} (0.5: {result['f1_default']:.4f})")
        print(f"    AUC:  {auc:.4f}")
        print(f"    Prec: {prec:.4f}  Rec: {rec:.4f} "
              f"(0.5: {result['rec_default']:.4f})")
        print(f"    CM:\n{cm}")
        print(classification_report(y_te, y_pred,
                                    target_names=["Alert", "Drowsy"]))

        # Free model + graph memory for this fold.
        del model
        tf.keras.backend.clear_session()

    # ── Summary ──
    if not fold_results:
        if os.path.exists(best_tmp_path):
            os.remove(best_tmp_path)
        print("\nNo valid folds to summarize (all folds skipped).")
        return []

    print(f"\n{'='*60}\nCV SUMMARY\n{'='*60}")
    for m in ["accuracy", "f1", "auc", "precision", "recall"]:
        v = [r[m] for r in fold_results]
        print(f"  {m:>10}: {np.mean(v):.4f} ± {np.std(v):.4f}  "
              f"(min={np.min(v):.4f} max={np.max(v):.4f})")

    thrs = [r["threshold"] for r in fold_results]
    print(f"\n  Thresholds: {', '.join(f'{t:.3f}' for t in thrs)}")
    print(f"  Mean threshold: {np.mean(thrs):.3f}")

    print(f"\n  --- vs default 0.5 ---")
    for m, dk in [("accuracy", "acc_default"), ("f1", "f1_default"),
                  ("recall", "rec_default")]:
        vo = np.mean([r[m] for r in fold_results])
        vd = np.mean([r[dk] for r in fold_results])
        d = vo - vd
        print(f"  {m:>10}: {vd:.4f} → {vo:.4f} (Δ{'+' if d>=0 else ''}{d:.4f})")

    print(f"\n{'Fold':>4} | {'Thr':>5} | {'Acc':>7} | {'F1':>7} | "
          f"{'AUC':>7} | {'Prec':>7} | {'Rec':>7} | Subjects")
    print("-" * 90)
    for r in fold_results:
        ss = ",".join(str(s) for s in r["test_subjects"][:6])
        if len(r["test_subjects"]) > 6:
            ss += "..."
        print(f"  {r['fold']+1:>2} | {r['threshold']:.3f} | "
              f"{r['accuracy']:.4f}  | {r['f1']:.4f}  | {r['auc']:.4f}  | "
              f"{r['precision']:.4f}  | {r['recall']:.4f}  | {ss}")

    total_cm = sum(r["confusion_matrix"] for r in fold_results)
    print(f"\nAggregate CM:\n{total_cm}")

    if best_fold >= 0 and os.path.exists(best_tmp_path):
        p = os.path.join(BASE_DIR, "best_fold_model.keras")
        shutil.copy2(best_tmp_path, p)
        os.remove(best_tmp_path)
        print(f"\nBest fold model (fold {best_fold+1}, AUC={best_auc:.4f}): {p}")

    return fold_results


# ─────────────────────────── Final Training ──────────────────────────────


def train_final_model(df):
    subjects = df["Subject"].unique()
    n_subjects = len(subjects)
    if n_subjects < 3:
        raise ValueError(
            f"Need at least 3 subjects for train/val/threshold split, got {n_subjects}."
        )

    rng_s = np.random.default_rng(SEED + 1000)
    rng_a = np.random.default_rng(SEED)

    # Split: ~10% for ES validation, ~10% for threshold tuning (holdout)
    n_val = max(1, n_subjects // 10)
    n_thr = max(1, n_subjects // 10)
    n_hold = min(n_subjects - 1, n_val + n_thr)
    n_val = min(n_val, n_hold - 1)
    n_thr = n_hold - n_val

    held_out = rng_s.choice(subjects, n_hold, replace=False)
    val_subj = held_out[:n_val]
    thr_subj = held_out[n_val : n_val + n_thr]  # separate threshold holdout
    train_subj = np.array([s for s in subjects if s not in held_out])

    print(f"\nFinal model: {len(train_subj)} train, {len(val_subj)} val, "
          f"{len(thr_subj)} threshold-holdout")

    df_tr = df[df["Subject"].isin(train_subj)]
    df_va = df[df["Subject"].isin(val_subj)]
    df_thr = df[df["Subject"].isin(thr_subj)]

    X_raw, y_tr = create_raw_sequences(
        df_tr, SEQ_LEN, TRAIN_STEP, temporal_pad=TEMPORAL_PAD)
    X_val, y_val = create_sequences(
        df_va, SEQ_LEN, EVAL_STEP, temporal_pad=TEMPORAL_PAD)

    if len(X_raw) == 0:
        raise ValueError("No training sequences created for final model.")
    if len(X_val) == 0:
        raise ValueError("No validation sequences created for final model.")

    X_full, y_full = build_augmented_dataset(
        X_raw, y_tr, rng_a, temporal_pad=TEMPORAL_PAD)

    if len(X_full) == 0:
        raise ValueError("Augmented training set is empty.")

    print(f"  Train: {len(X_full):,}  Val: {len(X_val):,}")

    model = build_model()
    model = compile_model(model)
    model.summary()

    train_ds = (
        tf.data.Dataset.from_tensor_slices((X_full, y_full))
        .shuffle(min(50_000, len(X_full)))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices((X_val, y_val))
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=get_callbacks(),
        verbose=1,
    )

    # Save Keras model
    keras_path = os.path.join(BASE_DIR, f"{MODEL_NAME}.keras")
    model.save(keras_path)
    print(f"\nKeras model saved: {keras_path}")

    # Deployment threshold tuned on SEPARATE holdout (not ES validation set)
    X_thr, y_thr = create_sequences(
        df_thr, SEQ_LEN, EVAL_STEP, temporal_pad=TEMPORAL_PAD)
    print(f"  Threshold holdout: {len(X_thr):,} sequences")

    if len(X_thr) == 0:
        print("  WARNING: threshold holdout has 0 sequences; using default threshold=0.5")
        opt_thr = 0.5
    else:
        tta_rng = np.random.default_rng(SEED + 999)
        y_thr_prob = predict_with_tta(model, X_thr, rng=tta_rng)
        opt_thr = find_optimal_threshold(y_thr, y_thr_prob)
    print(f"  Deployment threshold: {opt_thr:.3f}")

    deploy_cfg = {
        "threshold": opt_thr,
        "sequence_length": SEQ_LEN,
        "num_raw_features": NUM_RAW,
        "num_features": NUM_FEATURES,
        "feature_names": FEATURES,
        "rolling_window": ROLLING_WINDOW,
        "perclos_ear_z_threshold": PERCLOS_EAR_Z_THRESHOLD,
    }
    cfg_path = os.path.join(BASE_DIR, "deploy_config.json")
    with open(cfg_path, "w") as f:
        json.dump(deploy_cfg, f, indent=2)
    print(f"Deploy config saved: {cfg_path}")

    # Enhanced sequences for TFLite calibration + benchmark
    X_tr_enh = apply_enhanced_features(X_raw)
    if TEMPORAL_PAD > 0:
        X_tr_enh = X_tr_enh[:, TEMPORAL_PAD:, :]

    # ── TFLite via SavedModel export (Keras 3 compatible) ──
    sm_dir = os.path.join(BASE_DIR, "_saved_model_tmp")
    model.export(sm_dir)

    # Float16
    conv = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    tfl_f16 = conv.convert()
    f16_path = os.path.join(BASE_DIR, f"{MODEL_NAME}_f16.tflite")
    with open(f16_path, "wb") as f:
        f.write(tfl_f16)
    print(f"Float16 TFLite: {f16_path} ({len(tfl_f16)/1024:.1f} KB)")

    # INT8 (PTQ)
    conv8 = tf.lite.TFLiteConverter.from_saved_model(sm_dir)
    conv8.optimizations = [tf.lite.Optimize.DEFAULT]
    rep_rng = np.random.default_rng(SEED)
    rep_idx = rep_rng.permutation(len(X_tr_enh))[:min(300, len(X_tr_enh))]

    def representative_dataset():
        for i in rep_idx:
            yield [X_tr_enh[i : i + 1].astype(np.float32)]

    conv8.representative_dataset = representative_dataset
    conv8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv8.inference_input_type = tf.float32
    conv8.inference_output_type = tf.float32
    tfl_int8 = conv8.convert()
    i8_path = os.path.join(BASE_DIR, f"{MODEL_NAME}_int8.tflite")
    with open(i8_path, "wb") as f:
        f.write(tfl_int8)
    print(f"INT8 TFLite: {i8_path} ({len(tfl_int8)/1024:.1f} KB)")

    shutil.rmtree(sm_dir, ignore_errors=True)

    # ── Benchmark ──
    print("\n--- Inference Benchmark ---")
    sample = X_tr_enh[0:1].astype(np.float32)
    for label, path in [("Float16", f16_path), ("INT8", i8_path)]:
        interp = tf.lite.Interpreter(model_path=path)
        interp.allocate_tensors()
        inp_det = interp.get_input_details()

        for _ in range(20):
            interp.set_tensor(inp_det[0]["index"], sample)
            interp.invoke()

        times = []
        for _ in range(200):
            t0 = time.perf_counter()
            interp.set_tensor(inp_det[0]["index"], sample)
            interp.invoke()
            times.append(time.perf_counter() - t0)

        ms = np.array(times) * 1000
        print(f"  {label}: avg={np.mean(ms):.2f}ms  "
              f"p50={np.median(ms):.2f}ms  p95={np.percentile(ms, 95):.2f}ms")


def save_norm_params(params):
    """Save per-subject norm params + global fallback for unseen subjects."""
    # Compute global fallback: median of all subject params
    global_params = {}
    for f in FEATURES:
        medians = [params[s][f]["median"] for s in params]
        iqrs = [params[s][f]["iqr"] for s in params]
        global_params[f] = {
            "median": float(np.median(medians)),
            "iqr": float(np.median(iqrs)),
        }

    # Save per-subject params
    p = os.path.join(BASE_DIR, "norm_params.json")
    with open(p, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Norm params saved: {p}")

    # Save global fallback for deployment on unseen subjects
    pg = os.path.join(BASE_DIR, "norm_global_params.json")
    with open(pg, "w") as f:
        json.dump(global_params, f, indent=2)
    print(f"Global norm params (fallback): {pg}")


# ─────────────────────────── Main ────────────────────────────────────────


def main():
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    dev = f"GPU x{gpu_count}" if gpu_count > 0 else "CPU"
    print(f"TF {tf.__version__} | {dev} | "
          f"inter={tf.config.threading.get_inter_op_parallelism_threads()} "
          f"intra={tf.config.threading.get_intra_op_parallelism_threads()} "
          f"cpus={CPU_COUNT}")

    tf.keras.utils.set_random_seed(SEED)
    if DETERMINISTIC:
        tf.config.experimental.enable_op_determinism()

    print("Loading data...")
    df = load_data(CSV_FILE)

    print("\nAnchored normalisation...")
    df, norm_params = anchored_normalize(df)
    save_norm_params(norm_params)

    print("\nSavitzky-Golay denoising...")
    df = apply_savgol(df)

    # Sanitize
    nans = df[FEATURES].isna().sum().sum()
    infs = int(np.isinf(df[FEATURES].values).sum())
    if nans or infs:
        print(f"  WARNING: {nans} NaN, {infs} inf → replacing with 0")
        df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    if not SKIP_CV:
        print("\n" + "=" * 60)
        print("PHASE 1: Cross-Validation")
        print("=" * 60)
        results = run_cross_validation(df)

        mean_thr = float(np.mean([r["threshold"] for r in results]))
        print(f"\nCV mean threshold: {mean_thr:.3f}")
    else:
        print("\nSkipping cross-validation (SKIP_CV=True)")

    print("\n" + "=" * 60)
    print("PHASE 2: Final Model & Export")
    print("=" * 60)
    train_final_model(df)

    print("\nDone.")


if __name__ == "__main__":
    main()
