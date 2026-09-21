from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd

SENSOR_COLS = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8",
    "R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8",
]


def clean_grf(df, sample_rate=100, trim_seconds=10, force_threshold=10.0):
    df = df.copy()
    n_trim = int(sample_rate * trim_seconds)

    if len(df) > 2 * n_trim:
        df = df.iloc[n_trim:-n_trim].copy()

    force_cols = [c for c in SENSOR_COLS if c in df.columns]
    total_cols = [c for c in ["LTotal", "RTotal"] if c in df.columns]

    for col in force_cols + total_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df.loc[df[col] < force_threshold, col] = 0.0

    return df.reset_index(drop=True)


def load_grf_txt(path):
    columns = ["time", *SENSOR_COLS, "LTotal", "RTotal"]
    df = pd.read_csv(path, sep=r"\s+", header=None)

    if df.shape[1] < len(columns):
        raise ValueError(f"{path} has {df.shape[1]} columns; expected at least {len(columns)}")

    df = df.iloc[:, : len(columns)]
    df.columns = columns
    return df


def process_dataset(raw_dir, metadata_path, out_dir, sample_rate=100):
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    required = {"subject_id", "file", "severity"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing columns: {sorted(missing)}")

    rows = []

    for record in metadata.to_dict("records"):
        source = raw_dir / record["file"]
        df = load_grf_txt(source)
        df = clean_grf(df, sample_rate=sample_rate)

        output = out_dir / f"{record['subject_id']}.csv"
        df.to_csv(output, index=False)

        rows.append({
            "subject_id": record["subject_id"],
            "severity": int(record["severity"]),
            "source_file": record["file"],
            "processed_file": output.name,
            "n_samples": len(df),
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)

    summary = {
        "sample_rate_hz": sample_rate,
        "trim_seconds_each_end": 10,
        "force_threshold_n": 10.0,
        "subjects": len(manifest),
    }
    (out_dir / "preprocessing.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--sample-rate", type=int, default=100)
    args = parser.parse_args()

    process_dataset(
        raw_dir=args.raw_dir,
        metadata_path=args.metadata,
        out_dir=args.out_dir,
        sample_rate=args.sample_rate,
    )
