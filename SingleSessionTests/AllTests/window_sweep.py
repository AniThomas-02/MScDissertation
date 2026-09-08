import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from core import load_dataset, build_feature_sets


SEEDS = [2133744, 42, 7]
WINDOW_SIZES_MS = [150, 200, 300, 400, 500]


def run_once(window_ms, seed):
    X_raw, y_original = load_dataset(window_ms=window_ms)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    # "combined" was the best-performing feature set from the earlier grid
    X = build_feature_sets(X_raw)["combined"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=seed, eval_metric="mlogloss")
        clf.fit(X_train, y_train)
        scores.append(clf.score(X_test, y_test))

    return np.mean(scores)


def main():
    results = {ms: [] for ms in WINDOW_SIZES_MS}

    for seed in SEEDS:
        print(f"Seed {seed}:")
        for window_ms in WINDOW_SIZES_MS:
            acc = run_once(window_ms, seed)
            results[window_ms].append(acc)
            print(f"±{window_ms}ms window: {acc:.4f}")
        print()

    print("Final results:")
    for window_ms, accs in results.items():
        print(f"±{window_ms}ms: {np.mean(accs):.4f} ± {np.std(accs):.4f}  (runs: {[round(a, 4) for a in accs]})")


if __name__ == "__main__":
    main()
