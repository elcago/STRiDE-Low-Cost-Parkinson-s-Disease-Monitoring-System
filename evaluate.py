from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from model import DiscriminatorClassifier


def evaluate(image_dir, checkpoint, out_dir):
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(image_dir / "labels.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DiscriminatorClassifier().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    truth, pred = [], []

    with torch.no_grad():
        for row in labels.itertuples():
            x = np.load(image_dir / f"{row.subject_id}.npy").astype(np.float32)
            x = torch.from_numpy(x).unsqueeze(0).to(device)
            _, logits = model(x)
            truth.append(int(row.severity))
            pred.append(int(logits.argmax(1).item()))

    p, r, f1, _ = precision_recall_fscore_support(
        truth, pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy_score(truth, pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
    }

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(confusion_matrix(truth, pred)).to_csv(
        out_dir / "confusion_matrix.csv", index=False
    )
    (out_dir / "classification_report.txt").write_text(
        classification_report(truth, pred, digits=4, zero_division=0)
    )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default="data/gadf")
    parser.add_argument("--checkpoint", default="results/stgan/discriminator_best.pt")
    parser.add_argument("--out-dir", default="results/evaluation")
    args = parser.parse_args()
    evaluate(args.image_dir, args.checkpoint, args.out_dir)
