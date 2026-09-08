import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

from core import load_dataset, build_feature_sets

SEED = 2133744

PARTICIPANT_FILES = {
    "participant_1": ("participant_1/sensor_log.jsonl", "participant_1/key_log.jsonl"),
    "participant_2": ("participant_2/sensor_log.jsonl", "participant_2/key_log.jsonl"),
    "participant_3": ("participant_3/sensor_log.jsonl", "participant_3/key_log.jsonl"),
    "participant_4": ("participant_4/sensor_log.jsonl", "participant_4/key_log.jsonl"),
    "participant_5": ("participant_5/sensor_log.jsonl", "participant_5/key_log.jsonl"), 
}


def load_all_participants():
    X_all, y_all = [], []
    for name, (sensor_file, key_file) in PARTICIPANT_FILES.items():
        X_raw, _ = load_dataset(sensor_file=sensor_file, key_file=key_file)
        features = build_feature_sets(X_raw)["combined"]
        X_all.append(features)
        y_all.extend([name] * len(features))
        print(f"  {name}: {len(features)} samples")
    return np.concatenate(X_all, axis=0), np.array(y_all)


def main():
    X, y_original = load_all_participants()

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    print(f"\nPeople: {list(encoder.classes_)}")
    print(f"Total samples: {len(y)}")

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y):.4f}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []
    all_true, all_pred = np.zeros(len(y), dtype=int), np.zeros(len(y), dtype=int)

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED)
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)

        acc = (preds == y_test).mean()
        scores.append(acc)
        all_true[test_idx] = y_test
        all_pred[test_idx] = preds
        print(f"Fold {fold}: {acc:.4f}")

    print(f"\nOverall accuracy: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

    print("\nPer-person recall:")
    cm = confusion_matrix(all_true, all_pred)
    for i, name in enumerate(encoder.classes_):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        print(f"  {name}: {recall:.4f}  ({cm[i].sum()} samples)")


if __name__ == "__main__":
    main()
