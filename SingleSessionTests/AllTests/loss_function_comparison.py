import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from torch.utils.data import DataLoader, TensorDataset
from core import load_dataset, standardise_train_test, CNN1D_Large, augment_batch, EPOCHS, BATCH_SIZE, PATIENCE


SEED = 2133744


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


LOSS_FUNCTIONS = {
    "CrossEntropy (no smoothing)": lambda: nn.CrossEntropyLoss(),
    "CrossEntropy (smoothing=0.05)": lambda: nn.CrossEntropyLoss(label_smoothing=0.05),
    "CrossEntropy (smoothing=0.1, current)": lambda: nn.CrossEntropyLoss(label_smoothing=0.1),
    "CrossEntropy (smoothing=0.2)": lambda: nn.CrossEntropyLoss(label_smoothing=0.2),
    "Focal loss (gamma=2.0)": lambda: FocalLoss(gamma=2.0),
}


def train_one_fold(loss_name, X_train, y_train, X_test, y_test, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    model = CNN1D_Large(num_channels=X_train.shape[1], num_classes=num_classes).to(device)
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = LOSS_FUNCTIONS[loss_name]()
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

    assert best_state is not None, "training loop never ran"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    return test_acc


def main():
    X, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    num_classes = len(encoder.classes_)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}\n")

    results = {}
    for loss_name in LOSS_FUNCTIONS:
        print(loss_name)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
            X_train, X_test = standardise_train_test(X[train_idx], X[test_idx])
            y_train, y_test = y[train_idx], y[test_idx]

            acc = train_one_fold(loss_name, X_train, y_train, X_test, y_test, num_classes)
            scores.append(acc)
            print(f"Fold {fold}: {acc:.4f}")

        results[loss_name] = (np.mean(scores), np.std(scores))
        print(f"{loss_name}: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")

    print("Summary")
    for name, (mean, std) in sorted(results.items(), key=lambda kv: -kv[1][0]):
        print(f"{name}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
