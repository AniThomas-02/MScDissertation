import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.dummy import DummyClassifier
from torch.utils.data import DataLoader, TensorDataset

from core import load_dataset, standardise_train_test, augment_batch, EPOCHS, BATCH_SIZE, PATIENCE

SEED = 2133744


# ---- Model 1: Temporal Convolutional Network (dilated convolutions) ----

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        padding = dilation * (3 - 1) // 2  # keep sequence length unchanged
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        # match channel count for the residual connection when it changes
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        out = self.dropout(self.relu(self.bn(self.conv(x))))
        return out + self.residual(x)


class TCN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        # dilations 1, 2, 4, 8 give an increasing receptive field with few parameters
        self.blocks = nn.Sequential(
            TCNBlock(num_channels, 32, dilation=1),
            TCNBlock(32, 32, dilation=2),
            TCNBlock(32, 64, dilation=4),
            TCNBlock(64, 64, dilation=8),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.head(self.pool(self.blocks(x)))


# ---- Model 2: CNN-GRU hybrid (CRNN) ----

class CRNN(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        # CNN front-end extracts local motion patterns from the raw window
        self.conv = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # GRU integrates the extracted features over time
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        features = self.conv(x)               # (batch, channels, time)
        features = features.transpose(1, 2)    # (batch, time, channels) for GRU
        _, h_n = self.gru(features)
        return self.head(self.dropout(h_n[-1]))


def train_one_fold(model_class, use_augment, X_train, y_train, X_test, y_test, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    model = model_class(num_channels=X_train.shape[1], num_classes=num_classes).to(device)

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

    assert best_state is not None, "training loop never ran — check data/epochs"
    model.load_state_dict(best_state)

    X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)
    model.eval()
    with torch.no_grad():
        test_acc = (model(X_test_t).argmax(dim=1) == y_test_t).float().mean().item()

    return test_acc


def run_experiment(model_class, name):
    X, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)

    dummy = DummyClassifier(strategy="stratified", random_state=SEED)
    dummy.fit(np.zeros((len(y), 1)), y)
    print(f"Baseline: {dummy.score(np.zeros((len(y), 1)), y)}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        X_train, X_test = standardise_train_test(X[train_idx], X[test_idx])
        y_train, y_test = y[train_idx], y[test_idx]

        acc = train_one_fold(model_class, True, X_train, y_train, X_test, y_test, num_classes=len(encoder.classes_))
        scores.append(acc)
        print(f"Fold {fold}/5: {acc:.4f}")

    print(f"{name} accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}\n")


if __name__ == "__main__":
    print("TCN (Temporal Convolutional Network)")
    run_experiment(TCN, "TCN")

    print("CRNN (CNN + GRU hybrid)")
    run_experiment(CRNN, "CRNN")
