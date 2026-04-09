"""
전체 평가: BI Q_115, XGBoost, RF, B_11yr, B_full 비교
(test_qvalues_v2.parquet 기준)

사용법:
  python eval_full.py --thr 0.001
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

from config import PSB_THRESHOLDS, results_dir, FEATURES_DIR


def evaluate_all(label, scores, y, base_rate, total_psb):
    scores = np.array(scores, dtype=float)
    y = np.array(y)
    order = np.argsort(scores)[::-1]

    print(f"[{label}]")
    print(f"  {'k':>5}  {'P@k':>7}  {'R@k':>7}  {'Lift':>6}  {'PSB hit':>8}")
    for k in [50, 100, 200, 500, 1000]:
        top_k = y[order[:k]]
        hit   = int(top_k.sum())
        pk    = hit / k
        rk    = hit / total_psb if total_psb > 0 else 0
        lift  = pk / base_rate if base_rate > 0 else 0
        print(f"  {k:>5}  {pk:>7.4f}  {rk:>7.4f}  {lift:>6.2f}x  {hit:>8}")

    ap = average_precision_score(y, scores)
    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = float("nan")
    prec, rec, _ = precision_recall_curve(y, scores)
    rec_at_p5 = rec[prec >= 0.05].max() if (prec >= 0.05).any() else 0.0

    print(f"  AP (AUCPR): {ap:.4f}")
    print(f"  AUROC:      {auc:.4f}")
    print(f"  Recall @ Precision>=5%: {rec_at_p5:.4f}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="0.001")
    args = parser.parse_args()
    thr  = float(args.thr)
    rdir = results_dir(thr)

    v2_path = rdir / "test_qvalues_v2.parquet"
    if not v2_path.exists():
        # fallback to standard qvalues
        v2_path = rdir / "test_qvalues.parquet"

    res = pd.read_parquet(v2_path)
    b11 = pd.read_parquet(FEATURES_DIR / "b_11yr.parquet")
    res = res.merge(b11[["patent_id", "B_11yr"]], on="patent_id", how="left")
    res["B_11yr"] = res["B_11yr"].fillna(0)

    y         = res["psb"].values
    total_psb = int(y.sum())
    n         = len(y)
    base_rate = total_psb / n

    print(f"Test: {n:,}  PSB=1: {total_psb}  base rate: {base_rate:.4f}")
    print()

    # 전 생애 Beauty(B, B_11yr)는 PSB 라벨 정의에 사용되므로 baseline 제외.
    models = {}
    for col, lbl in [("q_score",             "Q_115 (BI)"),
                     ("Q_35_pred",           "Q_35  (BI)"),
                     ("xgb_score",           "XGBoost"),
                     ("rf_score",            "Random Forest"),
                     ("beauty_early_score",  "Early Beauty B'")]:
        if col in res.columns:
            models[lbl] = res[col].values

    for label, scores in models.items():
        evaluate_all(label, scores, y, base_rate, total_psb)


if __name__ == "__main__":
    main()
