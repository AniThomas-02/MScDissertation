import copy
import random

import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset

from core import (load_dataset, build_feature_sets,
                   CNN1D_Large, augment_batch, EPOCHS, BATCH_SIZE, PATIENCE)

SEED = 2133744

PARTICIPANT_FILES = {
    "participant_1": ("participant_1/sensor_log.jsonl", "participant_1/key_log.jsonl"),
    "participant_2": ("participant_2/sensor_log.jsonl", "participant_2/key_log.jsonl"),
    "participant_3": ("participant_3/sensor_log.jsonl", "participant_3/key_log.jsonl"),
    "participant_4": ("participant_4/sensor_log.jsonl", "participant_4/key_log.jsonl"),
    "participant_5": ("participant_5/sensor_log.jsonl", "participant_5/key_log.jsonl"), 
}


def train_cnn(X_train_raw, y_train, X_test_raw, y_test, num_classes, seed=SEED):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw, y_train, test_size=0.15, stratify=y_train, random_state=seed
    )

    n_channels = X_tr_raw.shape[1]
    scaler = StandardScaler()

    def scale(X_raw, fit=False):
        n, c, t = X_raw.shape
        flat = X_raw.transpose(0, 2, 1).reshape(-1, c)
        flat = scaler.fit_transform(flat) if fit else scaler.transform(flat)
        return flat.reshape(n, t, c).transpose(0, 2, 1).astype(np.float32)

    X_tr = scale(X_tr_raw, fit=True)      
    X_val = scale(X_val_raw, fit=False)
    X_test = scale(X_test_raw, fit=False)

    model = CNN1D_Large(num_channels=n_channels, num_classes=num_classes).to(device)
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.2)
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
            best_state = copy.deepcopy(model.state_dict())  
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            break

    assert best_state is not None, "training loop never ran"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        return (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()


def train_xgboost(X_train_raw, y_train, X_test_raw, y_test):
    feats_train = build_feature_sets(X_train_raw)["combined"]
    feats_test = build_feature_sets(X_test_raw)["combined"]

    scaler = StandardScaler()
    feats_train = scaler.fit_transform(feats_train)
    feats_test = scaler.transform(feats_test)

    clf = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05, random_state=SEED, eval_metric="mlogloss")
    clf.fit(feats_train, y_train)
    return clf.score(feats_test, y_test)


def load_participant_data(label_column):
    data = {}
    for name, (sensor_file, key_file) in PARTICIPANT_FILES.items():
        X, y_original = load_dataset(sensor_file=sensor_file, key_file=key_file, label_column=label_column)
        data[name] = (X, y_original)
        print(f"  {name}: {len(X)} samples")
    return data


def test_pooled_mixed_model(label_column, label_name):
    print(f"\n Pooled mixed-model 5-fold CV: {label_name}")
    data = load_participant_data(label_column)

    X_all = np.concatenate([X for X, _ in data.values()], axis=0)
    y_all_original = np.concatenate([y for _, y in data.values()])

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_all_original)
    num_classes = len(encoder.classes_)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    cnn_scores, xgb_scores = [], []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_all, y), start=1):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        cnn_acc = train_cnn(X_train, y_train, X_test, y_test, num_classes)
        xgb_acc = train_xgboost(X_train, y_train, X_test, y_test)

        cnn_scores.append(cnn_acc)
        xgb_scores.append(xgb_acc)
        print(f"Fold {fold}: CNN={cnn_acc:.4f}  XGBoost={xgb_acc:.4f}")

    print(f"\nCNN pooled mixed-model:     {np.mean(cnn_scores):.4f} +/- {np.std(cnn_scores):.4f}")
    print(f"XGBoost pooled mixed-model: {np.mean(xgb_scores):.4f} +/- {np.std(xgb_scores):.4f}")

    return cnn_scores, xgb_scores


def main():
    cnn_9, xgb_9 = test_pooled_mixed_model("region9", "9-region")
    cnn_26, xgb_26 = test_pooled_mixed_model("actual_key", "26-key")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"9-region  CNN pooled mixed-model: {np.mean(cnn_9):.4f} +/- {np.std(cnn_9):.4f}")
    print(f"9-region  XGB pooled mixed-model: {np.mean(xgb_9):.4f} +/- {np.std(xgb_9):.4f}")
    print(f"26-key    CNN pooled mixed-model: {np.mean(cnn_26):.4f} +/- {np.std(cnn_26):.4f}")
    print(f"26-key    XGB pooled mixed-model: {np.mean(xgb_26):.4f} +/- {np.std(xgb_26):.4f}")


if __name__ == "__main__":
    main()
