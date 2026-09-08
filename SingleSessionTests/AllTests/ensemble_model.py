import numpy as np
import torch

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from core import load_dataset, standardise_train_test, build_feature_sets, CNN1D_Large, augment_batch, EPOCHS, BATCH_SIZE, PATIENCE

SEED = 2133744


def train_cnn_proba(X_train_raw, y_train, X_test_raw, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    X_train, X_test = standardise_train_test(X_train_raw, X_test_raw)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    model = CNN1D_Large(num_channels=X_train.shape[1], num_classes=num_classes).to(device)
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

    assert best_state is not None, "training loop never ran — check data/epochs"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(X_test_t), dim=1).cpu().numpy()

    return probs


def main():
    X_raw, y_original = load_dataset()
    X_features = build_feature_sets(X_raw)["extended"]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    num_classes = len(encoder.classes_)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_scores = {"cnn": [], "rf": [], "xgb": [], "mlp": [], "ensemble": []}

    for fold, (train_idx, test_idx) in enumerate(cv.split(X_raw, y), start=1):
        print(f"\nFold {fold}/5")

        y_train, y_test = y[train_idx], y[test_idx]

        # CNN uses the raw windows
        cnn_probs = train_cnn_proba(X_raw[train_idx], y_train, X_raw[test_idx], num_classes)

        # RF, XGBoost, MLP use the extended engineered features
        Xf_train, Xf_test = X_features[train_idx], X_features[test_idx]
        scaler = StandardScaler()
        Xf_train = scaler.fit_transform(Xf_train)
        Xf_test = scaler.transform(Xf_test)

        rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=SEED)
        rf.fit(Xf_train, y_train)
        rf_probs = rf.predict_proba(Xf_test)

        xgb = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            random_state=SEED, eval_metric="mlogloss",
        )
        xgb.fit(Xf_train, y_train)
        xgb_probs = xgb.predict_proba(Xf_test)

        mlp = MLPClassifier(hidden_layer_sizes=(1000,), solver="lbfgs", max_iter=1000, random_state=SEED)
        mlp.fit(Xf_train, y_train)
        mlp_probs = mlp.predict_proba(Xf_test)

        # average the four probability distributions (soft voting)
        ensemble_probs = (cnn_probs + rf_probs + xgb_probs + mlp_probs) / 4

        fold_scores["cnn"].append((cnn_probs.argmax(axis=1) == y_test).mean())
        fold_scores["rf"].append((rf_probs.argmax(axis=1) == y_test).mean())
        fold_scores["xgb"].append((xgb_probs.argmax(axis=1) == y_test).mean())
        fold_scores["mlp"].append((mlp_probs.argmax(axis=1) == y_test).mean())
        fold_scores["ensemble"].append((ensemble_probs.argmax(axis=1) == y_test).mean())

        print(f"  CNN: {fold_scores['cnn'][-1]:.4f}  RF: {fold_scores['rf'][-1]:.4f}  "
              f"XGB: {fold_scores['xgb'][-1]:.4f}  MLP: {fold_scores['mlp'][-1]:.4f}  "
              f"Ensemble: {fold_scores['ensemble'][-1]:.4f}")

    print("\nFinal results:")
    for name, scores in fold_scores.items():
        print(f"{name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")


if __name__ == "__main__":
    main()
