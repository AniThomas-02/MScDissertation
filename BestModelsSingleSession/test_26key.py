import numpy as np
import torch

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

from core import load_dataset, standardise_train_test, build_feature_sets, CNN1D_Large, EPOCHS, BATCH_SIZE, PATIENCE, augment_batch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

SEED = 2133744


def top_k_accuracy(probs, y_true, k):
    top_k_preds = np.argsort(probs, axis=1)[:, -k:]
    correct = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return np.mean(correct)


def run_xgboost(X_raw, y):
    features = build_feature_sets(X_raw)["combined"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    top1_scores, top3_scores = [], []

    for train_idx, test_idx in cv.split(features, y):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
        clf.fit(X_train, y_train)
        probs = clf.predict_proba(X_test)

        top1_scores.append(clf.score(X_test, y_test))
        top3_scores.append(top_k_accuracy(probs, y_test, k=3))

    return top1_scores, top3_scores


def train_one_fold_with_probs(model_class, use_augment, X_train, y_train, X_test, y_test, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    model = model_class(num_channels=X_train.shape[1], num_classes=num_classes).to(device)
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)

    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    best_val_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if use_augment:
                xb = augment_batch(xb)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_acc = (model(X_val_t).argmax(dim=1) == y_val_t).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    assert best_state is not None, "training loop never ran"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()

    top1 = (probs.argmax(axis=1) == y_test).mean()
    return top1, probs


def run_cnn(X_raw, y, num_classes):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    top1_scores, top3_scores = [], []

    for train_idx, test_idx in cv.split(X_raw, y):
        X_train, X_test = standardise_train_test(X_raw[train_idx], X_raw[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        top1, probs = train_one_fold_with_probs(CNN1D_Large, True, X_train, y_train, X_test, y_test, num_classes)
        top1_scores.append(top1)
        top3_scores.append(top_k_accuracy(probs, y_test, k=3))

    return top1_scores, top3_scores


def main():
    print("26-key inference:")
    X_raw, y_original = load_dataset(label_column="actual_key")

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    num_classes = len(encoder.classes_)
    print(f"Classes: {num_classes}  Samples: {len(y)}")

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    print("Running XGBoost...")
    xgb_top1, xgb_top3 = run_xgboost(X_raw, y)
    print(f"XGBoost - 26-key Top-1: {np.mean(xgb_top1):.4f} ± {np.std(xgb_top1):.4f}")
    print(f"XGBoost - 26-key Top-3: {np.mean(xgb_top3):.4f} ± {np.std(xgb_top3):.4f}")

    print("\nRunning CNN...")
    cnn_top1, cnn_top3 = run_cnn(X_raw, y, num_classes)
    print(f"CNN 26-key Top-1: {np.mean(cnn_top1):.4f} ± {np.std(cnn_top1):.4f}")
    print(f"CNN 26-key Top-3: {np.mean(cnn_top3):.4f} ± {np.std(cnn_top3):.4f}")

if __name__ == "__main__":
    main()
