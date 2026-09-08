import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from core import load_dataset, SENSOR_COLUMNS_6

SEED = 2133744


# ---- ACCessory-style features not yet in our pipeline ----

def accessory_features(window):
    """11 stats per channel, following Owusu et al.'s ACCessory feature table."""
    features = []
    names = []

    for c, channel_name in enumerate(SENSOR_COLUMNS_6):
        signal = window[c]
        rms = np.sqrt(np.mean(signal ** 2))
        rmse = np.sqrt(np.mean((signal - signal.mean()) ** 2))  # RMS of residual from the mean
        deltas = np.diff(signal)
        avg_delta = np.mean(np.abs(deltas))

        # local peaks/troughs: a point is a peak if higher than both neighbours
        is_peak = (signal[1:-1] > signal[:-2]) & (signal[1:-1] > signal[2:])
        is_trough = (signal[1:-1] < signal[:-2]) & (signal[1:-1] < signal[2:])
        num_max = is_peak.sum()
        num_min = is_trough.sum()

        peak_idx = np.where(is_peak)[0] + 1
        trough_idx = np.where(is_trough)[0] + 1
        all_idx = np.arange(len(signal))
        ttp = np.mean([np.min(np.abs(all_idx - p)) for p in peak_idx]) if len(peak_idx) else 0.0
        ttc = np.mean([np.min(np.abs(all_idx - t)) for t in trough_idx]) if len(trough_idx) else 0.0

        # RMS cross rate: how often the signal crosses its own RMS value
        above_rms = signal > rms
        rcr = np.sum(above_rms[:-1] != above_rms[1:]) / len(signal)

        sma = np.mean(np.abs(signal))  # signal magnitude area

        features.extend([rms, rmse, signal.min(), signal.max(), avg_delta,
                          num_max, num_min, ttp, ttc, rcr, sma])
        names.extend([f"{channel_name}_{stat}" for stat in
                      ["rms", "rmse", "min", "max", "avg_delta", "num_max", "num_min", "ttp", "ttc", "rcr", "sma"]])

    return np.array(features, dtype=np.float32), names


def build_kitchen_sink(X_raw):
    from phase1_grid import basic_extract_features, correlation_features, extended_extract_features, ALL_PAIRS

    all_features = []
    names = None

    for w in X_raw:
        basic = basic_extract_features(w)
        corr = correlation_features(w)
        extended = extended_extract_features(w)
        accessory, accessory_names = accessory_features(w)

        combined = np.concatenate([basic, corr, extended, accessory])
        all_features.append(combined)

        if names is None:
            basic_names = [f"{ch}_{stat}" for ch in SENSOR_COLUMNS_6
                            for stat in ["mean", "std", "peak", "peak_avg_ratio", "energy", "time_to_peak", "half_asymmetry"]]
            corr_names = [f"corr_{a}_{b}" for a, b in ALL_PAIRS]
            extended_names = [f"ext_{ch}_{i}" for ch in SENSOR_COLUMNS_6 for i in range(15)] + \
                              [f"ext_lobe_{i}" for i in range(8)]
            names = basic_names + corr_names + extended_names + accessory_names

    return np.stack(all_features), names


def main():
    X_raw, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    print("Building kitchen-sink feature set...")
    X, feature_names = build_kitchen_sink(X_raw)
    print(f"Total features: {X.shape[1]}\n")

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rf_scores, xgb_scores = [], []
    rf_importances = np.zeros(X.shape[1])
    xgb_importances = np.zeros(X.shape[1])

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED)
        rf.fit(X_train, y_train)
        rf_scores.append(rf.score(X_test, y_test))
        rf_importances += rf.feature_importances_

        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        xgb.fit(X_train, y_train)
        xgb_scores.append(xgb.score(X_test, y_test))
        xgb_importances += xgb.feature_importances_

        print(f"Fold {fold}: RF={rf_scores[-1]:.4f}  XGB={xgb_scores[-1]:.4f}")

    print(f"\nRandom Forest (kitchen sink): {np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    print(f"XGBoost (kitchen sink): {np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}")

    rf_importances /= 5
    xgb_importances /= 5

    print("\n=== Top 20 features by Random Forest importance ===")
    for i in np.argsort(-rf_importances)[:20]:
        print(f"{feature_names[i]}: {rf_importances[i]:.4f}")

    print("\n=== Top 20 features by XGBoost importance ===")
    for i in np.argsort(-xgb_importances)[:20]:
        print(f"{feature_names[i]}: {xgb_importances[i]:.4f}")


if __name__ == "__main__":
    main()
