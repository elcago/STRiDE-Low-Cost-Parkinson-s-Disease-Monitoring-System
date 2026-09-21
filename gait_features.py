from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def _cycle_stats(force, sample_rate):
    force = np.asarray(force, dtype=float)
    active = force > 10
    edges = np.diff(active.astype(int), prepend=0)
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]

    if len(ends) and len(starts) and ends[0] < starts[0]:
        ends = ends[1:]
    if len(starts) and len(ends) and starts[-1] > ends[-1]:
        starts = starts[:-1]

    n = min(len(starts), len(ends))
    starts = starts[:n]
    ends = ends[:n]

    stance = (ends - starts) / sample_rate if n else np.array([])
    stride = np.diff(starts) / sample_rate if len(starts) > 1 else np.array([])

    return starts, ends, stance, stride


def _cop(sensor_values):
    sensor_values = np.asarray(sensor_values, dtype=float)
    n_sensors = sensor_values.shape[1]

    x = np.linspace(-1.0, 1.0, n_sensors)
    total = sensor_values.sum(axis=1)
    total[total == 0] = np.nan
    cop = np.nansum(sensor_values * x[None, :], axis=1) / total
    return np.nan_to_num(cop)


def extract_features(df, sample_rate=100):
    left = df.get("LTotal", pd.Series(np.zeros(len(df)))).to_numpy(float)
    right = df.get("RTotal", pd.Series(np.zeros(len(df)))).to_numpy(float)

    l_starts, l_ends, l_stance, l_stride = _cycle_stats(left, sample_rate)
    r_starts, r_ends, r_stance, r_stride = _cycle_stats(right, sample_rate)

    duration_min = max(len(df) / sample_rate / 60.0, 1e-6)
    cadence = (len(l_starts) + len(r_starts)) / duration_min

    stride_values = np.concatenate([l_stride, r_stride]) if (len(l_stride) + len(r_stride)) else np.array([])
    stance_values = np.concatenate([l_stance, r_stance]) if (len(l_stance) + len(r_stance)) else np.array([])

    left_sensors = [c for c in df.columns if c.startswith("L") and c[1:].isdigit()]
    right_sensors = [c for c in df.columns if c.startswith("R") and c[1:].isdigit()]

    cop_left = _cop(df[left_sensors].to_numpy()) if left_sensors else np.zeros(len(df))
    cop_right = _cop(df[right_sensors].to_numpy()) if right_sensors else np.zeros(len(df))

    cop_left_len = np.abs(np.diff(cop_left)).sum()
    cop_right_len = np.abs(np.diff(cop_right)).sum()

    peak_left = float(np.max(left)) if len(left) else 0.0
    peak_right = float(np.max(right)) if len(right) else 0.0

    mean_force_left = float(np.mean(left))
    mean_force_right = float(np.mean(right))
    impulse_left = float(np.trapz(left, dx=1 / sample_rate))
    impulse_right = float(np.trapz(right, dx=1 / sample_rate))

    def asymmetry(a, b):
        denom = (abs(a) + abs(b)) / 2
        return 0.0 if denom == 0 else 100 * abs(a - b) / denom

    left_stride = float(np.mean(l_stride)) if len(l_stride) else np.nan
    right_stride = float(np.mean(r_stride)) if len(r_stride) else np.nan

    return {
        "cadence": cadence,
        "stride_time": float(np.nanmean(stride_values)) if len(stride_values) else np.nan,
        "stance_time": float(np.nanmean(stance_values)) if len(stance_values) else np.nan,
        "peak_force": (peak_left + peak_right) / 2,
        "average_force": (mean_force_left + mean_force_right) / 2,
        "force_impulse": (impulse_left + impulse_right) / 2,
        "cop_position_mean": float(np.mean(np.concatenate([cop_left, cop_right]))),
        "cop_trajectory_length": (cop_left_len + cop_right_len) / 2,
        "cop_asymmetry": asymmetry(cop_left_len, cop_right_len),
        "force_asymmetry": asymmetry(peak_left, peak_right),
        "step_time_asymmetry": asymmetry(left_stride, right_stride)
        if np.isfinite(left_stride) and np.isfinite(right_stride)
        else np.nan,
    }


def build_feature_table(processed_dir):
    processed_dir = Path(processed_dir)
    manifest = pd.read_csv(processed_dir / "manifest.csv")
    rows = []

    for record in manifest.to_dict("records"):
        df = pd.read_csv(processed_dir / record["processed_file"])
        row = extract_features(df)
        row["subject_id"] = record["subject_id"]
        row["severity"] = record["severity"]
        rows.append(row)

    features = pd.DataFrame(rows)
    features.to_csv(processed_dir / "gait_features.csv", index=False)
    return features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()
    build_feature_table(args.processed_dir)
