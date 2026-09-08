import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from core import load_json, clean_sensor_data, clean_key_data, resample_window, SENSOR_FILE, KEY_FILE, WINDOW_MS, build_feature_sets


SEED = 2133744
DETECTION_TOLERANCE_MS = 150   # how close to a real keypress the guess has to be to match
MIN_SPIKE_DISTANCE_MS = 200    # minimum gap enforced between two detected spikes
RMS_WINDOW_MS = 100            # sliding window 



def compute_rms_signal(sensor_df, channels, window_ms=RMS_WINDOW_MS):
    valid_df = sensor_df.dropna(subset=list(channels))
    times = valid_df["perf_time"].to_numpy()
    magnitude = np.sqrt(sum(valid_df[c].to_numpy(dtype=float) ** 2 for c in channels))
    n = len(times)
    half_window = window_ms / 2

    left_idx = np.searchsorted(times, times - half_window, side="left")
    right_idx = np.searchsorted(times, times + half_window, side="right")

    rms = np.zeros(n)
    for i in range(n):
        segment = magnitude[left_idx[i]:right_idx[i]]
        rms[i] = np.sqrt(np.mean(segment ** 2)) if len(segment) else 0.0

    print(f"RMS signal stats ({'+'.join(channels)}): min={rms.min():.4f} mean={rms.mean():.4f} "
          f"median={np.median(rms):.4f} max={rms.max():.4f} std={rms.std():.4f}")

    return times, rms


def detect_spikes(times, rms, percentile):
    threshold = np.percentile(rms, percentile)
    candidate_idx = np.where(rms > threshold)[0]

    detected_times = []
    last_time = -np.inf
    for idx in candidate_idx:
        t = times[idx]
        if t - last_time >= MIN_SPIKE_DISTANCE_MS:
            detected_times.append(t)
            last_time = t

    return np.array(detected_times), threshold


def match_detections(detected_times, key_df):
    ground_truth_times = key_df["perf_time"].to_numpy()
    matched_pairs = []  
    used_gt = set()

    for dt in detected_times:
        diffs = np.abs(ground_truth_times - dt)
        idx = np.argmin(diffs)
        if diffs[idx] <= DETECTION_TOLERANCE_MS and idx not in used_gt:
            matched_pairs.append((dt, idx))
            used_gt.add(idx)

    return matched_pairs


def run_sweep(sensor_df, key_df, channels, label):
    print(f"\n=== Detection signal: {label} ===")
    times, rms = compute_rms_signal(sensor_df, channels)

    best_recall = -1
    best_matched_pairs = None
    best_percentile = None

    for percentile in [80, 85, 90, 93, 95, 97]:
        detected_times, threshold = detect_spikes(times, rms, percentile)
        matched_pairs = match_detections(detected_times, key_df)

        precision = len(matched_pairs) / len(detected_times) if len(detected_times) else 0
        recall = len(matched_pairs) / len(key_df)
        print(f"percentile={percentile:3d}  threshold={threshold:.4f}  "
              f"detected={len(detected_times):5d}  matched={len(matched_pairs):5d}  "
              f"precision={precision:.4f}  recall={recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            best_matched_pairs = matched_pairs
            best_percentile = percentile

    # remembers the highst recall for th results
    return best_matched_pairs


def classify_matched(sensor_df, key_df, matched_pairs, label):
    X, y = [], []
    for dt, gt_idx in matched_pairs:
        window = sensor_df[
            (sensor_df["perf_time"] >= dt - WINDOW_MS)
            & (sensor_df["perf_time"] <= dt + WINDOW_MS)
        ].copy()
        if len(window) < 8:
            continue
        window["relative_time"] = window["perf_time"] - dt
        X.append(resample_window(window))
        y.append(int(key_df.iloc[gt_idx]["region9"]))

    X = np.array(X, dtype=np.float32)
    y_original = np.array(y)
    print(f"Usable windows: {len(X)}")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    features = build_feature_sets(X)["combined"]

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

    print(f"{label} end-to-end accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


def main():
    sensor_df_orientation = clean_sensor_data(load_json(SENSOR_FILE))

    from core import SENSOR_COLUMNS_12
    sensor_df_full = clean_sensor_data(load_json(SENSOR_FILE), SENSOR_COLUMNS_12)
    key_df = clean_key_data(load_json(KEY_FILE))

    print(f"Ground-truth keypresses: {len(key_df)}")

    matched_orientation = run_sweep(sensor_df_orientation, key_df, ("rot_beta", "rot_gamma"), "orientation (rot_beta+rot_gamma)")
    matched_acceleration = run_sweep(sensor_df_full, key_df, ("acc_x", "acc_y", "acc_z"), "acceleration (acc_x+acc_y+acc_z)")

    print("\n Classification")
    classify_matched(sensor_df_orientation, key_df, matched_orientation, "Orientation-based detection")
    classify_matched(sensor_df_full, key_df, matched_acceleration, "Acceleration-based detection")

if __name__ == "__main__":
    main()
