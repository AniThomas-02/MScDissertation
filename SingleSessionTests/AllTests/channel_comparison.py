import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from core import load_dataset, standardise_train_test, SENSOR_COLUMNS_6, SENSOR_COLUMNS_9,SENSOR_COLUMNS_12, train_one_fold, CNN1D_Large


SEED = 2133744


def run(columns, name):
    X, y_original = load_dataset(columns)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"\n{name} channels: {columns}")
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = standardise_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        acc = train_one_fold(
            CNN1D_Large, True, X_train, y_train, X_test, y_test,
            num_classes=len(encoder.classes_),
        )
        scores.append(acc)
        print(f"Fold {fold}/5: {acc}")

    print(f"{name} accuracy: {np.mean(scores)} ± {np.std(scores)}")
    return scores


def main():
    scores_6 = run(SENSOR_COLUMNS_6, "6-channel")
    scores_9 = run(SENSOR_COLUMNS_9, "9-channel")
    scores_12 = run(SENSOR_COLUMNS_12, "12-channel")

    print("\nSummary:")
    print(f"6-channel:  {np.mean(scores_6):.4f} ± {np.std(scores_6):.4f}")
    print(f"12-channel: {np.mean(scores_12):.4f} ± {np.std(scores_12):.4f}")


if __name__ == "__main__":
    main()
