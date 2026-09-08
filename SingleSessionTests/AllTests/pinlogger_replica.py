import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier

from core import load_dataset, build_feature_sets

SEED = 2133744
HIDDEN_NODES = 1000

SOLVER = "lbfgs" # closest available equivalent to scaled conjugate-gradient backprop


def top_k_accuracy(model, X, y, k):
    probs = model.predict_proba(X)
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = [y[i] in top_k_preds[i] for i in range(len(y))]
    return np.mean(correct)


def main():
    X_raw, y_original = load_dataset()
    X = build_feature_sets(X_raw)["basic"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    # 70% train / 15% validation / 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=SEED
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y_train), 1)), y_train)
    print(f"Baseline: {dummy.score(np.zeros((len(y_test), 1)), y_test)}")

    model = MLPClassifier(
        hidden_layer_sizes=(HIDDEN_NODES,),
        solver=SOLVER,
        max_iter=1000,
        random_state=SEED,
    )

    best_val_acc = 0
    best_model = None

    for n_iter in range(50, 1001, 50):
        model.set_params(max_iter=n_iter)
        model.fit(X_train, y_train)
        val_acc = model.score(X_val, y_val)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    print(f"\nValidation accuracy: {best_val_acc}")

    top1 = best_model.score(X_test, y_test)
    top2 = top_k_accuracy(best_model, X_test, y_test, k=2)
    top3 = top_k_accuracy(best_model, X_test, y_test, k=3)

    print(f"\nTest Top-1: {top1}")
    print(f"Test Top-2: {top2}")
    print(f"Test Top-3: {top3}")


if __name__ == "__main__":
    main()
