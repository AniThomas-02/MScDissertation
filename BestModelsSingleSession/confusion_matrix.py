import numpy as np
import torch
import torch.nn as nn

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, TensorDataset
from core import load_dataset, standardise_train_test, CNN1D_Large, augment_batch, EPOCHS, BATCH_SIZE, PATIENCE


SEED = 2133744


def train_one_fold_predict(X_train, y_train, X_test, num_classes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, stratify=y_train, random_state=SEED
    )

    model = CNN1D_Large(num_channels=X_train.shape[1], num_classes=num_classes).to(device)
    train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr, dtype=torch.long))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)  # Table 7.5's validated winner - was SGD+momentum, which lost

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
    model.eval()
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).cpu().numpy()
    return preds


def main():
    X, y_original = load_dataset()
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_original)
    num_classes = len(encoder.classes_)
    class_names = encoder.classes_

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    all_true = np.zeros(len(y), dtype=int)
    all_pred = np.zeros(len(y), dtype=int)

    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = standardise_train_test(X[train_idx], X[test_idx])
        y_train = y[train_idx]

        preds = train_one_fold_predict(X_train, y_train, X_test, num_classes)
        all_true[test_idx] = y[test_idx]
        all_pred[test_idx] = preds

    cm = confusion_matrix(all_true, all_pred)
    print("Confusion matrix:")
    print("Region labels:", class_names)
    print()
    header = "      " + " ".join(f"{c:5d}" for c in class_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{class_names[i]:5d} " + " ".join(f"{v:5d}" for v in row))

    print("\nPer-region recall:")
    for i, name in enumerate(class_names):
        recall = cm[i, i] / cm[i].sum() if cm[i].sum() else 0
        print(f"  Region {name}: {recall:.4f}  ({cm[i].sum()} true samples)")

    print("\nMost confused pairs:")
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confusions.append((cm[i, j], class_names[i], class_names[j]))
    confusions.sort(reverse=True)
    for count, true_label, pred_label in confusions[:10]:
        true_row, true_col = (true_label - 1) // 3, (true_label - 1) % 3
        pred_row, pred_col = (pred_label - 1) // 3, (pred_label - 1) % 3
        same_row = "same row" if true_row == pred_row else "different row"
        same_col = "same col" if true_col == pred_col else "different col"
        print(f"  True {true_label} -> Predicted {pred_label}: {count} times ({same_row}, {same_col})")


if __name__ == "__main__":
    main()
