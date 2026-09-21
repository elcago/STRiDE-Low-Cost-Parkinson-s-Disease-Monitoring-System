from pathlib import Path
import argparse
import pandas as pd
from scipy.stats import f_oneway


def run_anova(feature_csv, out_csv):
    df = pd.read_csv(feature_csv)
    feature_cols = [c for c in df.columns if c not in {"subject_id", "severity"}]
    rows = []

    for feature in feature_cols:
        groups = [
            group[feature].dropna().to_numpy()
            for _, group in df.groupby("severity")
        ]
        groups = [g for g in groups if len(g) > 1]

        if len(groups) < 2:
            continue

        f_stat, p_value = f_oneway(*groups)
        rows.append({
            "feature": feature,
            "f_statistic": f_stat,
            "p_value": p_value,
        })

    results = pd.DataFrame(rows).sort_values("p_value")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/gait_features.csv")
    parser.add_argument("--out", default="results/feature_anova.csv")
    args = parser.parse_args()
    run_anova(args.features, args.out)
