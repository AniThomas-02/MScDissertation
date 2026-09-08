import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from core import load_dataset
from all_features import build_kitchen_sink

SEED = 2133744

# union of RF's and XGBoost's top-20 important features from the large feature test
CURATED_FEATURES = [
    "rot_beta_half_asymmetry", "ext_rot_beta_6", "ext_rot_beta_13",
    "corr_rot_alpha_rot_beta", "ext_ori_gamma_12", "ext_ori_gamma_14",
    "ext_ori_gamma_11", "ext_rot_beta_11", "ext_lobe_1", "ext_rot_beta_3",
    "ext_lobe_0", "corr_rot_beta_rot_gamma", "ext_ori_alpha_14",
    "rot_beta_peak_avg_ratio", "rot_beta_sma", "ext_ori_alpha_12",
    "ext_rot_alpha_7", "ext_ori_beta_12", "ext_ori_beta_14", "rot_alpha_sma",
    "ori_beta_avg_delta", "ext_lobe_7", "rot_gamma_half_asymmetry", "ext_rot_alpha_8",
]


def evaluate(name, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rf_scores, xgb_scores = [], []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED)
        rf.fit(X_train, y_train)
        rf_scores.append(rf.score(X_test, y_test))

        xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        xgb.fit(X_train, y_train)
        xgb_scores.append(xgb.score(X_test, y_test))

    print(f"{name} ({X.shape[1]} features)")
    print(f"Random Forest:{np.mean(rf_scores):.4f} ± {np.std(rf_scores):.4f}")
    print(f"XGBoost:{np.mean(xgb_scores):.4f} ± {np.std(xgb_scores):.4f}\n")


def main():
    X_raw, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    print("Building kitchen-sink feature set...")
    X_full, feature_names = build_kitchen_sink(X_raw)

    curated_idx = [feature_names.index(name) for name in CURATED_FEATURES]
    X_curated = X_full[:, curated_idx]

    evaluate("Kitchen sink (all features)", X_full, y)
    evaluate("Curated (top-20 union)", X_curated, y)


if __name__ == "__main__":
    main()
