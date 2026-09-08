import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path

BASE_DIR = Path(__file__).parent
SENSOR_FILE = BASE_DIR / "sensor_log.jsonl"
KEY_FILE = BASE_DIR / "key_log.jsonl"

WINDOW_MS = 400
TARGET_LENGTH = 50

SENSOR_COLUMNS_6 = [
    "rot_alpha", "rot_beta", "rot_gamma",
    "ori_alpha", "ori_beta", "ori_gamma",
]
SENSOR_COLUMNS = SENSOR_COLUMNS_6  # kept for backwards compatibility

REGION_MAP = {
    "q": 1, "w": 1, "e": 1, "r": 2, "t": 2, "y": 2, "u": 3, "i": 3, "o": 3, "p": 3,
    "a": 4, "s": 4, "d": 4, "f": 5, "g": 5, "h": 5, "j": 6, "k": 6, "l": 6,
    "z": 7, "x": 7, "c": 7, "v": 8, "b": 8, "n": 9, "m": 9,
}


# Data loading and windowing

def load_json(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def clean_sensor_data(df, columns=SENSOR_COLUMNS_6):
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    df[columns] = df[columns].apply(pd.to_numeric, errors="coerce")
    df["perf_time"] = pd.to_numeric(df["perf_time"], errors="coerce")
    return df.dropna(subset=["perf_time"]).sort_values("perf_time").reset_index(drop=True)


def clean_key_data(df):
    df = df.copy()
    df["perf_time"] = pd.to_numeric(df["perf_time"], errors="coerce")
    df["actual_key"] = df["actual_key"].astype(str).str.lower()
    df["region9"] = df["actual_key"].map(REGION_MAP)
    df = df.dropna(subset=["perf_time", "region9"])
    if "correct" in df.columns:
        df = df[df["correct"] == True]
    return df.sort_values("perf_time").reset_index(drop=True)


def resample_window(window_df, columns=SENSOR_COLUMNS_6, window_ms=WINDOW_MS):
    old_times = window_df["relative_time"].to_numpy()
    new_times = np.linspace(-window_ms, window_ms, TARGET_LENGTH)

    channels = []
    for col in columns:
        values = window_df[col].to_numpy(dtype=float)
        valid = ~np.isnan(values)

        if valid.sum() < 2:
            channels.append(np.zeros(TARGET_LENGTH))
            continue

        resampled = np.interp(new_times, old_times[valid], values[valid])
        resampled = resampled - resampled.mean()
        channels.append(resampled)

    return np.stack(channels, axis=0)


def build_sequence_dataset(sensor_df, key_df, columns=SENSOR_COLUMNS_6, window_ms=WINDOW_MS, label_column="region9"):
    X, y = [], []

    for _, key_row in key_df.iterrows():
        t = key_row["perf_time"]
        window = sensor_df[
            (sensor_df["perf_time"] >= t - window_ms)
            & (sensor_df["perf_time"] <= t + window_ms)
        ].copy()

        if len(window) < 8:
            continue

        window["relative_time"] = window["perf_time"] - t
        X.append(resample_window(window, columns, window_ms))
        label = key_row[label_column]
        y.append(int(label) if label_column == "region9" else label)

    return np.array(X, dtype=np.float32), np.array(y)


def load_dataset(columns=SENSOR_COLUMNS_6, window_ms=WINDOW_MS, label_column="region9",
                  sensor_file=None, key_file=None):
    sensor_file = sensor_file or SENSOR_FILE
    key_file = key_file or KEY_FILE
    sensor_df = clean_sensor_data(load_json(sensor_file), columns)
    key_df = clean_key_data(load_json(key_file))
    return build_sequence_dataset(sensor_df, key_df, columns, window_ms, label_column)


# Feature extraction

CHANNEL_INDEX = {name: i for i, name in enumerate(SENSOR_COLUMNS_6)}
LOBE_PAIRS = [("rot_beta", "rot_gamma"), ("ori_beta", "ori_gamma")]
ALL_PAIRS = [(a, b) for i, a in enumerate(SENSOR_COLUMNS_6) for b in SENSOR_COLUMNS_6[i + 1:]]


def correlation_features(window):
    features = []
    for channel_a, channel_b in ALL_PAIRS:
        a = window[CHANNEL_INDEX[channel_a]]
        b = window[CHANNEL_INDEX[channel_b]]
        if a.std() < 1e-8 or b.std() < 1e-8:
            features.append(0.0)
        else:
            features.append(np.corrcoef(a, b)[0, 1])
    return np.array(features, dtype=np.float32)


def jerk_features(signal):
    jerk = np.diff(signal)
    return [np.max(np.abs(jerk)), np.sum(jerk ** 2)]


def frequency_features(signal):
    spectrum = np.abs(np.fft.rfft(signal))
    dominant_freq = np.argmax(spectrum[1:]) + 1  # skip the DC component
    return [dominant_freq, spectrum.sum()]


def polynomial_features(signal, degree=3):
    t = np.linspace(-1, 1, len(signal))
    return np.polyfit(t, signal, degree).tolist()


def extended_lobe_angle_features(window, channel_a, channel_b):
    a = window[CHANNEL_INDEX[channel_a]]
    b = window[CHANNEL_INDEX[channel_b]]
    half = len(a) // 2

    upper_angle = np.arctan2(b[:half].mean(), a[:half].mean())
    lower_angle = np.arctan2(b[half:].mean(), a[half:].mean())
    upper_width = np.hypot(a[:half].std(), b[:half].std())
    lower_width = np.hypot(a[half:].std(), b[half:].std())

    return [upper_angle, lower_angle, upper_width, lower_width]


def basic_stats_single_channel(signal):
    half = len(signal) // 2
    first_half, second_half = signal[:half], signal[half:]
    peak = np.max(np.abs(signal))
    mean_abs = np.mean(np.abs(signal)) + 1e-6
    return [
        signal.mean(), signal.std(), peak, peak / mean_abs,
        np.sum(signal ** 2), np.argmax(np.abs(signal)) / len(signal),
        first_half.mean() - second_half.mean(),
    ]


def extended_extract_features(window):
    features = []
    for c in range(window.shape[0]):
        signal = window[c]
        features.extend(basic_stats_single_channel(signal))
        features.extend(jerk_features(signal))
        features.extend(frequency_features(signal))
        features.extend(polynomial_features(signal))

    for channel_a, channel_b in LOBE_PAIRS:
        features.extend(extended_lobe_angle_features(window, channel_a, channel_b))

    return np.array(features, dtype=np.float32)


def build_feature_sets(X_raw):
    correlation = np.stack([correlation_features(w) for w in X_raw])
    extended = np.stack([extended_extract_features(w) for w in X_raw])
    return {"combined": np.concatenate([correlation, extended], axis=1)}


# CNN model and augmentation

EPOCHS = 80
BATCH_SIZE = 32
PATIENCE = 15

NOISE_STD = 0.05
SCALE_RANGE = (0.9, 1.1)
MAX_SHIFT = 3


class CNN1D_Large(nn.Module):
    # larger capacity: 64 -> 128 -> 256 channels
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256), nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def augment_batch(x):
    batch_size = x.shape[0]
    noise = torch.randn_like(x) * NOISE_STD
    scale = torch.empty(batch_size, 1, 1, device=x.device).uniform_(*SCALE_RANGE)
    x = x * scale + noise

    shift = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1)
    if shift != 0:
        x = torch.roll(x, shifts=shift, dims=2)

    return x
