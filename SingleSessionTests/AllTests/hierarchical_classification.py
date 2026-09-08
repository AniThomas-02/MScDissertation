import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from core import load_dataset, build_feature_sets


SEED = 2133744


def main():
    X_raw, y_original = load_dataset()

    # regions 1-9 already form a 3x3 grid: row = (region-1)//3, col = (region-1)%3
    # e.g. region 1 (q/w/e) -> row 0, col 0;  region 5 (f/g/h) -> row 1, col 1
    rows = np.array([(r - 1) // 3 for r in y_original])
    cols = np.array([(r - 1) % 3 for r in y_original])

    encoder = LabelEncoder()
    y_flat = encoder.fit_transform(y_original)

    features = build_feature_sets(X_raw)["combined"]

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y_flat), 1)), y_flat)
    print(f"Baseline (9-way): {dummy.score(np.zeros((len(y_flat), 1)), y_flat)}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    flat_scores = []
    row_scores, col_scores, hierarchical_scores = [], [], []

    for train_idx, test_idx in cv.split(features, y_flat):
        X_train, X_test = features[train_idx], features[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # flat 9-way classifier
        flat_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        flat_clf.fit(X_train, y_flat[train_idx])
        flat_scores.append(flat_clf.score(X_test, y_flat[test_idx]))

        # stage 1: row classifier 
        row_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        row_clf.fit(X_train, rows[train_idx])
        row_pred = row_clf.predict(X_test)
        row_scores.append((row_pred == rows[test_idx]).mean())

        # stage 2: column classifier
        col_clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        col_clf.fit(X_train, cols[train_idx])
        col_pred = col_clf.predict(X_test)
        col_scores.append((col_pred == cols[test_idx]).mean())

        # combined hierarchical prediction: both row AND column must be correct
        both_correct = (row_pred == rows[test_idx]) & (col_pred == cols[test_idx])
        hierarchical_scores.append(both_correct.mean())

    print(f"Flat 9-way classifier:        {np.mean(flat_scores):.4f} ± {np.std(flat_scores):.4f}")
    print(f"Stage 1 (row, 3-way):         {np.mean(row_scores):.4f} ± {np.std(row_scores):.4f}")
    print(f"Stage 2 (column, 3-way):      {np.mean(col_scores):.4f} ± {np.std(col_scores):.4f}")
    print(f"Hierarchical (row AND column): {np.mean(hierarchical_scores):.4f} ± {np.std(hierarchical_scores):.4f}")
    print(f"\n(expected hierarchical accuracy if independent: {np.mean(row_scores) * np.mean(col_scores):.4f})")


if __name__ == "__main__":
    main()
