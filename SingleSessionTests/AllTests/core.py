import json
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).parent
SENSOR_FILE = BASE_DIR / "sensor_log.jsonl"
KEY_FILE = BASE_DIR / "key_log.jsonl"

WINDOW_MS = 400
TARGET_LENGTH = 50

SENSOR_COLUMNS_6 = [
    "rot_alpha", "rot_beta", "rot_gamma",
    "ori_alpha", "ori_beta", "ori_gamma",
]

SENSOR_COLUMNS_9 = SENSOR_COLUMNS_6 + ["acc_x", "acc_y", "acc_z"]

SENSOR_COLUMNS_12 = SENSOR_COLUMNS_6 + [
    "acc_x", "acc_y", "acc_z",
    "gacc_x", "gacc_y", "gacc_z",
]

# backwards compatibility
SENSOR_COLUMNS = SENSOR_COLUMNS_6

REGION_MAP = {
    "q": 1, "w": 1, "e": 1, "r": 2, "t": 2, "y": 2, "u": 3, "i": 3, "o": 3, "p": 3,
    "a": 4, "s": 4, "d": 4, "f": 5, "g": 5, "h": 5, "j": 6, "k": 6, "l": 6,
    "z": 7, "x": 7, "c": 7, "v": 8, "b": 8, "n": 9, "m": 9,
}


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
        # subtract resting orientation ti hightlight only deviations
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


def standardise_train_test(X_train, X_test):
    n_train, channels, timesteps = X_train.shape
    n_test = X_test.shape[0]

    scaler = StandardScaler()
    X_train_flat = X_train.transpose(0, 2, 1).reshape(-1, channels)
    X_test_flat = X_test.transpose(0, 2, 1).reshape(-1, channels)

    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    X_train_scaled = X_train_scaled.reshape(n_train, timesteps, channels).transpose(0, 2, 1)
    X_test_scaled = X_test_scaled.reshape(n_test, timesteps, channels).transpose(0, 2, 1)

    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier


SEED = 2133744
CHANNEL_INDEX = {name: i for i, name in enumerate(SENSOR_COLUMNS_6)}

# channel pairs treated like TouchLogger's pitch/roll plane
LOBE_PAIRS = [("rot_beta", "rot_gamma"), ("ori_beta", "ori_gamma")]

# all pairwise combinations among the 6 channels, for correlation features
ALL_PAIRS = [(a, b) for i, a in enumerate(SENSOR_COLUMNS_6) for b in SENSOR_COLUMNS_6[i + 1:]]


# Basic Feature Extraction

def basic_extract_features(window):
    features = []
    for c in range(window.shape[0]):
        signal = window[c]
        half = len(signal) // 2
        first_half, second_half = signal[:half], signal[half:]
        peak = np.max(np.abs(signal))
        mean_abs = np.mean(np.abs(signal)) + 1e-6

        features.extend([
            signal.mean(),
            signal.std(),
            peak,
            peak / mean_abs,# peak-to-average ratio
            np.sum(signal ** 2),# energy
            np.argmax(np.abs(signal)) / len(signal),# time-to-peak
            first_half.mean() - second_half.mean(),# half-window asymmetry
        ])
    return np.array(features, dtype=np.float32)

def lobe_area(points):
    area = 0.0
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2

def touchlogger_features(window):
    features = []
    for channel_a, channel_b in LOBE_PAIRS:
        a = window[CHANNEL_INDEX[channel_a]]
        b = window[CHANNEL_INDEX[channel_b]]
        half = len(a) // 2

        upper_pts = list(zip(a[:half], b[:half]))
        lower_pts = list(zip(a[half:], b[half:]))

        aub = np.arctan2(np.mean(b[:half]), np.mean(a[:half]))
        alb = np.arctan2(np.mean(b[half:]), np.mean(a[half:]))

        upper_mags = np.hypot(a[:half], b[:half])
        lower_mags = np.hypot(a[half:], b[half:])
        upper_edge_idx = np.argmax(upper_mags)
        lower_edge_idx = np.argmax(lower_mags)
        upper_edge_angle = np.arctan2(b[:half][upper_edge_idx], a[:half][upper_edge_idx])
        lower_edge_angle = np.arctan2(b[half:][lower_edge_idx], a[half:][lower_edge_idx])

        upper_width = np.hypot(a[:half].std(), b[:half].std())
        lower_width = np.hypot(a[half:].std(), b[half:].std())

        features.extend([
            aub, alb,
            upper_edge_angle, lower_edge_angle,
            upper_width, lower_width,
            lobe_area(upper_pts), lobe_area(lower_pts),
        ])
    return np.array(features, dtype=np.float32)

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
    coeffs = np.polyfit(t, signal, degree)
    return coeffs.tolist()

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
    sets = {
        "basic": np.stack([basic_extract_features(w) for w in X_raw]),
        "touchlogger": np.stack([touchlogger_features(w) for w in X_raw]),
        "correlation": np.stack([correlation_features(w) for w in X_raw]),
        "extended": np.stack([extended_extract_features(w) for w in X_raw]),
    }
    sets["combined"] = np.concatenate(
        [sets["correlation"], sets["extended"]], axis=1
    )
    return sets

def build_classifiers():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED),
        "XGBoost": XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss"),
        "MLP": MLPClassifier(hidden_layer_sizes=(256,), solver="lbfgs", max_iter=1000, random_state=SEED),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=SEED),
        "kNN": KNeighborsClassifier(n_neighbors=5),
        "GaussianNB": GaussianNB(),
    }

def evaluate(X, y, seed=SEED):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        yield X_train, y_train, X_test, y_test

def main():
    X_raw, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    print("Building feature sets...")
    feature_sets = build_feature_sets(X_raw)
    for name, feats in feature_sets.items():
        print(f"  {name}: {feats.shape[1]} features")
    print()

    results = {}

    for feat_name, X in feature_sets.items():
        for clf_name in build_classifiers().keys():
            scores = []
            for X_train, y_train, X_test, y_test in evaluate(X, y):
                clf = build_classifiers()[clf_name]  # fresh instance per fold
                clf.fit(X_train, y_train)
                scores.append(clf.score(X_test, y_test))

            key = f"{feat_name} + {clf_name}"
            results[key] = (np.mean(scores), np.std(scores))
            print(f"{key}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    print("\n Summary: ")
    for key, (mean, std) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{key}: {mean:.4f} ± {std:.4f}")

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier

from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 80
BATCH_SIZE = 32
PATIENCE = 15
SEED = 2133744

NOISE_STD = 0.05
SCALE_RANGE = (0.9, 1.1)
MAX_SHIFT = 3


class CNN1D_Original(nn.Module):
    # 32 -> 64 -> 128 channels
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D_Reduced(nn.Module):
    # smaller capacity: 16 -> 32 -> 64 channels
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(0.3),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN1D_Large(nn.Module):
    # larger capacity: 64 -> 128 -> 256 channels
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

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

def train_one_fold(model_class, use_augment, X_train, y_train, X_test, y_test, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    model = model_class(num_channels=X_train.shape[1], num_classes=num_classes).to(device)

    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_augment:
                xb = augment_batch(xb)

            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    assert best_state is not None, "training loop never ran"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    return test_acc


def run_experiment(model_class, use_augment, name):
    X, y_original = load_dataset()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\nFold {fold}/5")

        X_train, X_test = standardise_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        acc = train_one_fold(
            model_class, use_augment, X_train, y_train, X_test, y_test,
            num_classes=len(encoder.classes_),
        )
        scores.append(acc)
        print(f"Fold accuracy: {acc}")

    print(f"\n{name} accuracy:")
    print(f"{np.mean(scores)} ± {np.std(scores)}")

