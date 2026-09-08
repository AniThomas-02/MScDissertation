import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from torch.utils.data import DataLoader, TensorDataset

from core import load_dataset, standardise_train_test, CNN1D_Large, augment_batch, EPOCHS, BATCH_SIZE, PATIENCE
from advanced_nn_models import TCN

SEEDS = [2133744, 42, 7]


def train_one_fold(model_class, X_train, y_train, X_test, y_test, num_classes, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed
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

    assert best_state is not None, "training loop not run"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    return test_acc


def run_experiment(model_class, X, y, num_classes, seed):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = []

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = standardise_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        acc = train_one_fold(model_class, X_train, y_train, X_test, y_test, num_classes, seed)
        scores.append(acc)

    return scores


def main():
    X, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    num_classes = len(encoder.classes_)

    dummy = DummyClassifier(strategy="stratified", random_state=SEEDS[0])
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    models = {"CNN (large+aug)": CNN1D_Large, "TCN": TCN}
    results = {name: [] for name in models}

    for seed in SEEDS:
        print(f"Seed {seed}:")
        for name, model_class in models.items():
            scores = run_experiment(model_class, X, y, num_classes, seed)
            fold_mean = np.mean(scores)
            results[name].append(fold_mean)
            print(f"{name}: {fold_mean:.4f}  (folds: {[round(s, 4) for s in scores]})")
        print()

    print("Results:")
    for name, seed_means in results.items():
        print(f"{name}: {np.mean(seed_means):.4f} ± {np.std(seed_means):.4f}  (per-seed means: {[round(m, 4) for m in seed_means]})")


if __name__ == "__main__":
    main()
