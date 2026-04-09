"""
Recall@k 곡선 시각화 및 표 출력

사용법:
  python eval_recall.py --thr 0.001
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import results_dir, FEATURES_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="0.001")
    args = parser.parse_args()
    thr  = float(args.thr)
    rdir = results_dir(thr)

    # 데이터 로드
    qv_path = rdir / "test_qvalues.parquet"
    bc_path = rdir / "baseline_comparison.parquet"
    b11     = pd.read_parquet(FEATURES_DIR / "b_11yr.parquet")

    res = pd.read_parquet(qv_path)
    if bc_path.exists():
        bc  = pd.read_parquet(bc_path)
        res = res.merge(bc[["patent_id", "xgb_score", "rf_score"]], on="patent_id", how="left")
    res = res.merge(b11[["patent_id", "B_11yr"]], on="patent_id", how="left")
    res["B_11yr"] = res["B_11yr"].fillna(0)

    y         = res["psb"].values
    total_psb = int(y.sum())
    n         = len(y)

    print(f"Test: {n:,}  PSB=1: {total_psb}  (base rate: {total_psb/n*100:.3f}%)")
    print()

    models = {}
    for col, lbl in [("Q_35_pred", "Q_35 (BI)"),
                     ("Q_115_pred","Q_115 (BI)"),
                     ("xgb_score", "XGBoost"),
                     ("rf_score",  "Random Forest"),
                     ("B_11yr",    "B_11yr")]:
        if col in res.columns:
            models[lbl] = res[col].fillna(-999).values

    key_ks = [50, 100, 200, 300, 500, 700, 1000]
    header = f"{'k':>6} | " + " | ".join(f"{lbl[:10]:>10}" for lbl in models) + " | {'Random':>8}"
    print(header)
    print("-" * len(header))

    for k in key_ks:
        row = f"{k:>6} |"
        for lbl, scores in models.items():
            order = np.argsort(scores)[::-1]
            hit   = int(y[order[:k]].sum())
            recall = hit / total_psb if total_psb > 0 else 0
            row += f" {recall:>8.4f}({hit:2d})|"
        rand_recall = min(k * total_psb / n, total_psb) / total_psb if total_psb > 0 else 0
        row += f" {rand_recall:>8.4f}|"
        print(row)

    # 그래프
    k_list = list(range(10, 201, 10)) + list(range(250, n + 1, 100))
    colors = {"Q_35 (BI)": "red", "Q_115 (BI)": "darkred",
              "XGBoost": "blue", "Random Forest": "green", "B_11yr": "orange"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, max_k, title in [(axes[0], max(k_list), "Full Range"),
                              (axes[1], 500, "Top 500 Zoom")]:
        k_sub = [k for k in k_list if k <= max_k]
        for lbl, scores in models.items():
            order = np.argsort(scores)[::-1]
            rs = [int(y[order[:k]].sum()) / total_psb for k in k_sub]
            ax.plot(k_sub, rs, label=lbl, color=colors.get(lbl, "gray"), linewidth=2)
        rand = [min(k * total_psb / n, total_psb) / total_psb for k in k_sub]
        ax.plot(k_sub, rand, label="Random", color="gray", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Top-k"); ax.set_ylabel("Recall")
        ax.set_title(f"Recall@k — {title} (thr={thr})")
        ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = rdir / "recall_at_k.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_png.name}")


if __name__ == "__main__":
    main()
