from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset


class CNN(nn.Module):
    def __init__(self, channels=2, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


def metric_dict(y_true, y_pred):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def tabular_baselines(feature_csv, seed=42):
    df = pd.read_csv(feature_csv).dropna()
    X = df.drop(columns=["subject_id", "severity"])
    y = df["severity"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, random_state=seed, class_weight="balanced"
        ),
        "SVM": SVC(C=1.0, kernel="rbf", class_weight="balanced"),
    }

    results = {}
    for name, model in models.items():
        train_x = X_train if name == "Random Forest" else X_train_scaled
        test_x = X_test if name == "Random Forest" else X_test_scaled
        model.fit(train_x, y_train)
        results[name] = metric_dict(y_test, model.predict(test_x))

    return results


def image_cnn(image_dir, epochs=50, seed=42):
    image_dir = Path(image_dir)
    labels = pd.read_csv(image_dir / "labels.csv")
    X = np.stack([
        np.load(image_dir / f"{sid}.npy").astype(np.float32)
        for sid in labels["subject_id"]
    ])
    y = labels["severity"].to_numpy(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
        batch_size=32,
        shuffle=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN(channels=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_test).to(device)).argmax(1).cpu().numpy()

    return metric_dict(y_test, pred)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/gait_features.csv")
    parser.add_argument("--gadf", default="data/gadf")
    parser.add_argument("--gasf", default="data/gasf")
    parser.add_argument("--out", default="results/baselines.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = tabular_baselines(args.features, seed=args.seed)

    if Path(args.gadf).exists():
        results["GADF CNN"] = image_cnn(args.gadf, seed=args.seed)
    if Path(args.gasf).exists():
        results["GASF CNN"] = image_cnn(args.gasf, seed=args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
