from pathlib import Path
import argparse
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from model import Generator, DiscriminatorClassifier


class GAFDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = Path(image_dir)
        self.labels = pd.read_csv(self.image_dir / "labels.csv")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        x = np.load(self.image_dir / f"{row.subject_id}.npy").astype(np.float32)
        return torch.from_numpy(x), torch.tensor(int(row.severity), dtype=torch.long)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    truth, pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, logits = model(x)
            pred.extend(logits.argmax(1).cpu().numpy())
            truth.extend(y.numpy())

    acc = accuracy_score(truth, pred)
    p, r, f1, _ = precision_recall_fscore_support(
        truth, pred, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def train(image_dir, out_dir, epochs=100, batch_size=32, latent_dim=128, seed=42):
    seed_all(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = GAFDataset(image_dir)

    n_test = max(1, int(0.2 * len(dataset)))
    n_val = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_set, val_set, test_set = random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    G = Generator(latent_dim=latent_dim).to(device)
    D = DiscriminatorClassifier().to(device)

    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    adv_loss = nn.BCEWithLogitsLoss()
    cls_loss = nn.CrossEntropyLoss()

    best_f1 = -1
    history = []

    for epoch in range(1, epochs + 1):
        G.train()
        D.train()

        for real, labels in train_loader:
            real = real.to(device)
            labels = labels.to(device)
            b = real.size(0)

            ones = torch.ones(b, 1, device=device)
            zeros = torch.zeros(b, 1, device=device)

            z = torch.randn(b, latent_dim, device=device)
            fake = G(z)

            opt_d.zero_grad()
            real_score, real_class = D(real)
            fake_score, _ = D(fake.detach())
            loss_d = (
                adv_loss(real_score, ones)
                + adv_loss(fake_score, zeros)
                + cls_loss(real_class, labels)
            )
            loss_d.backward()
            opt_d.step()

            opt_g.zero_grad()
            fake_score, _ = D(fake)
            loss_g = adv_loss(fake_score, ones)
            loss_g.backward()
            opt_g.step()

        metrics = evaluate(D, val_loader, device)
        metrics["epoch"] = epoch
        history.append(metrics)

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(D.state_dict(), out_dir / "discriminator_best.pt")
            torch.save(G.state_dict(), out_dir / "generator_best.pt")

        print(
            f"epoch {epoch:03d} | val acc {metrics['accuracy']:.3f} | "
            f"val f1 {metrics['f1']:.3f}"
        )

    D.load_state_dict(torch.load(out_dir / "discriminator_best.pt", map_location=device))
    test_metrics = evaluate(D, test_loader, device)

    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)
    (out_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    print("test:", test_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/gadf")
    parser.add_argument("--out-dir", default="results/stgan")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        image_dir=args.image_dir,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        latent_dim=args.latent_dim,
        seed=args.seed,
    )
