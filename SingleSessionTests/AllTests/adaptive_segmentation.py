import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from core import load_json, clean_sensor_data, clean_key_data, resample_window, SENSOR_FILE, KEY_FILE, WINDOW_MS, build_feature_sets


SEED = 2133744
MAX_HALF_WINDOW_MS = 600   
BASELINE_THRESHOLD_STD = 0.5 


def resample_window_asymmetric(window_df, start_offset, end_offset, columns=None):
    from core import SENSOR_COLUMNS_6, TARGET_LENGTH
    columns = columns or SENSOR_COLUMNS_6

    old_times = window_df["relative_time"].to_numpy()

    new_times = np.linspace(start_offset, end_offset, TARGET_LENGTH)

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


def find_adaptive_boundaries(sensor_df, t, channel="rot_beta"):

    local = sensor_df[
        (sensor_df["perf_time"] >= t - MAX_HALF_WINDOW_MS)
        & (sensor_df["perf_time"] <= t + MAX_HALF_WINDOW_MS)
    ].dropna(subset=[channel])
    if len(local) < 8:
        return t - WINDOW_MS, t + WINDOW_MS  # use fixed window instead

    values = local[channel].to_numpy(dtype=float)
    times = local["perf_time"].to_numpy()
    baseline = np.abs(values).mean()
    threshold = baseline * BASELINE_THRESHOLD_STD

    before = local[local["perf_time"] < t]
    after = local[local["perf_time"] >= t]

    start = t - WINDOW_MS
    if len(before):
        b_values = before[channel].to_numpy(dtype=float)
        b_times = before["perf_time"].to_numpy()
        quiet = np.where(np.abs(b_values) < threshold)[0]
        if len(quiet):
            start = b_times[quiet[-1]]  # most recent quiet point before the keypress

    end = t + WINDOW_MS
    if len(after):
        a_values = after[channel].to_numpy(dtype=float)
        a_times = after["perf_time"].to_numpy()
        quiet = np.where(np.abs(a_values) < threshold)[0]
        if len(quiet):
            end = a_times[quiet[0]]  # first quiet point after the keypress

    return start, end


def build_adaptive_dataset(sensor_df, key_df):
    X, y = [], []

    for _, key_row in key_df.iterrows():
        t = key_row["perf_time"]
        start, end = find_adaptive_boundaries(sensor_df, t)

        window = sensor_df[(sensor_df["perf_time"] >= start) & (sensor_df["perf_time"] <= end)].copy()
        if len(window) < 8:
            continue

        window["relative_time"] = window["perf_time"] - t
        # use the real asymmetric bounds directly
        X.append(resample_window_asymmetric(window, start - t, end - t))
        y.append(int(key_row["region9"]))

    return np.array(X, dtype=np.float32), np.array(y)


def build_fixed_dataset(sensor_df, key_df):
    from core import build_sequence_dataset
    return build_sequence_dataset(sensor_df, key_df)


def evaluate(X_raw, y_original, name):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    features = build_feature_sets(X_raw)["combined"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    for train_idx, test_idx in cv.split(features, y):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


def main():
    sensor_df = clean_sensor_data(load_json(SENSOR_FILE))
    key_df = clean_key_data(load_json(KEY_FILE))

    X_fixed, y_fixed = build_fixed_dataset(sensor_df, key_df)

    X_adaptive, y_adaptive = build_adaptive_dataset(sensor_df, key_df)

    print(f"\nFixed-window samples: {len(X_fixed)}")
    print(f"Adaptive-window samples: {len(X_adaptive)}\n")

    evaluate(X_fixed, y_fixed, "Fixed 400ms window")
    evaluate(X_adaptive, y_adaptive, "Adaptive-duration window")


if __name__ == "__main__":
    main()
