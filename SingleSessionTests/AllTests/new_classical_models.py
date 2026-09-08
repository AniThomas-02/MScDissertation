import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.dummy import DummyClassifier
from lightgbm import LGBMClassifier

from core import load_dataset, build_feature_sets
from all_features import build_kitchen_sink
from curated_features import CURATED_FEATURES

SEED = 2133744


def build_classifiers():
    return {
        "QDA": QuadraticDiscriminantAnalysis(reg_param=0.5),  
        "ExtraTrees": ExtraTreesClassifier(n_estimators=300, max_depth=8, random_state=SEED),
        "LightGBM": LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, verbosity=-1),
    }


def evaluate(feat_name, X, y, classifier_names=None):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    classifier_names = classifier_names or list(build_classifiers().keys())
    results = {name: [] for name in classifier_names}

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for clf_name in classifier_names:
            clf = build_classifiers()[clf_name]
            clf.fit(X_train, y_train)
            results[clf_name].append(clf.score(X_test, y_test))

    for clf_name, scores in results.items():
        print(f"{feat_name} + {clf_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    print()


def main():
    X_raw, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    combined = build_feature_sets(X_raw)["combined"]
    X_full, feature_names = build_kitchen_sink(X_raw)
    curated_idx = [feature_names.index(name) for name in CURATED_FEATURES]
    curated = X_full[:, curated_idx]

    # QDA is only run on 24 feautres to make sure it has enough samples
    evaluate("combined", combined, y, classifier_names=["ExtraTrees", "LightGBM"])
    evaluate("curated", curated, y)


if __name__ == "__main__":
    main()
