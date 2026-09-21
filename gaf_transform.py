from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def resample_signal(x, size=64):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.zeros(size, dtype=np.float32)

    old = np.linspace(0, 1, len(x))
    new = np.linspace(0, 1, size)
    return np.interp(new, old, x).astype(np.float32)


def normalize_minus_one_one(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = np.min(x), np.max(x)
    if hi == lo:
        return np.zeros_like(x)
    return 2 * (x - lo) / (hi - lo) - 1


def gaf(x, method="gadf"):
    x = np.clip(normalize_minus_one_one(x), -1, 1)
    phi = np.arccos(x)

    if method.lower() == "gasf":
        return np.cos(phi[:, None] + phi[None, :]).astype(np.float32)
    if method.lower() == "gadf":
        return np.sin(phi[:, None] - phi[None, :]).astype(np.float32)
    raise ValueError("method must be 'gadf' or 'gasf'")


def subject_image(df, size=64, method="gadf"):
    channels = []

    for col in ["LTotal", "RTotal"]:
        if col not in df.columns:
            raise ValueError(f"missing {col}")
        signal = resample_signal(df[col].to_numpy(), size=size)
        channels.append(gaf(signal, method=method))

    return np.stack(channels, axis=0)


def build_images(processed_dir, out_dir, size=64, method="gadf"):
    processed_dir = Path(processed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(processed_dir / "manifest.csv")
    labels = []

    for record in manifest.to_dict("records"):
        df = pd.read_csv(processed_dir / record["processed_file"])
        image = subject_image(df, size=size, method=method)
        np.save(out_dir / f"{record['subject_id']}.npy", image)
        labels.append({
            "subject_id": record["subject_id"],
            "severity": int(record["severity"]),
        })

    pd.DataFrame(labels).to_csv(out_dir / "labels.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    parser.add_argument("--out-dir", default="data/gadf")
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--method", choices=["gadf", "gasf"], default="gadf")
    args = parser.parse_args()
    build_images(args.processed_dir, args.out_dir, args.size, args.method)
