"""
Exp 4: PSB 샘플 가중치 민감도 분석

PSB_WEIGHTS = [50, 100, 200, 300, 500, 664, 1000, 2000] 각각에 대해
Backward Induction 전체 실행 → Precision@k, Recall@k, PR-AUC 비교

사용법:
  python exp4_weight_sensitivity.py
  python exp4_weight_sensitivity.py --thr 0.005
  python exp4_weight_sensitivity.py --thr all
"""

import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import xgboost as xgb
from config import (
    GAMMA, MAINT_COST, PSB_REWARD, PSB_THRESHOLDS,
    XGB_PARAMS, labels_file, results_dir,
)
from utils import load_and_merge, split, get_feature_cols

PSB_WEIGHTS = [50, 100, 200, 300, 500, 664, 1000, 2000]


def make_sw(psb: np.ndarray, w: float) -> np.ndarray:
    sw = np.ones(len(psb), dtype=float)
    sw[psb == 1] = w
    return sw


def run_bi(df35, df75, df115, psb_weight: float) -> pd.DataFrame:
    """단일 psb_weight로 Backward Induction 실행 → test Q_35 반환"""
    tr35,  val35,  te35  = split(df35.copy())
    tr75,  val75,  te75  = split(df75.copy())
    tr115, val115, te115 = split(df115.copy())

    # Step A: 11.5yr
    fc115 = [c for c in get_feature_cols(11.5) if c in tr115.columns]
    for d in [tr115, val115, te115]:
        d["Y_115"] = -MAINT_COST[11.5] + d["psb"] * PSB_REWARD
    m115 = xgb.XGBRegressor(**XGB_PARAMS)
    m115.fit(tr115[fc115].values, tr115["Y_115"].values,
             sample_weight=make_sw(tr115["psb"].values, psb_weight),
             eval_set=[(val115[fc115].values, val115["Y_115"].values)],
             verbose=False)
    for d in [tr115, val115, te115]:
        d["Q_115"] = m115.predict(d[fc115].values)

    # Step B: 7.5yr
    fc75 = [c for c in get_feature_cols(7.5) if c in tr75.columns]
    q115_map = {pid: q for d in [tr115, val115, te115]
                for pid, q in zip(d["patent_id"], d["Q_115"])}
    for d in [tr75, val75, te75]:
        d["Y_75"] = -MAINT_COST[7.5] + GAMMA * np.maximum(
            d["patent_id"].map(q115_map).fillna(0), 0)
    m75 = xgb.XGBRegressor(**XGB_PARAMS)
    m75.fit(tr75[fc75].values, tr75["Y_75"].values,
            sample_weight=make_sw(tr75["psb"].values, psb_weight),
            eval_set=[(val75[fc75].values, val75["Y_75"].values)],
            verbose=False)
    for d in [tr75, val75, te75]:
        d["Q_75"] = m75.predict(d[fc75].values)

    # Step C: 3.5yr
    fc35 = [c for c in get_feature_cols(3.5) if c in tr35.columns]
    q75_map = {pid: q for d in [tr75, val75, te75]
               for pid, q in zip(d["patent_id"], d["Q_75"])}
    for d in [tr35, val35, te35]:
        d["Y_35"] = -MAINT_COST[3.5] + GAMMA * np.maximum(
            d["patent_id"].map(q75_map).fillna(0), 0)
    m35 = xgb.XGBRegressor(**XGB_PARAMS)
    m35.fit(tr35[fc35].values, tr35["Y_35"].values,
            sample_weight=make_sw(tr35["psb"].values, psb_weight),
            eval_set=[(val35[fc35].values, val35["Y_35"].values)],
            verbose=False)
    te35["Q_35"] = m35.predict(te35[fc35].values)

    return te35[["patent_id", "psb", "Q_35"]].copy()


def evaluate(te: pd.DataFrame, weight: float) -> dict:
    psb_total = int(te["psb"].sum())
    n         = len(te)
    base_rate = psb_total / n
    sorted_df = te.sort_values("Q_35", ascending=False).reset_index(drop=True)

    row = {"weight": weight, "base_rate": round(base_rate, 5),
           "psb_total": psb_total, "n_test": n}

    for k in [50, 100, 200, 500, 1000]:
        topk = sorted_df.head(k)
        p    = topk["psb"].sum() / k
        r    = topk["psb"].sum() / psb_total if psb_total > 0 else 0
        lift = p / base_rate if base_rate > 0 else 0
        row[f"P@{k}"]    = round(p, 4)
        row[f"R@{k}"]    = round(r, 4)
        row[f"lift@{k}"] = round(lift, 1)

    row["PR_AUC"] = round(average_precision_score(te["psb"], te["Q_35"]), 4)

    maintain = te[te["Q_35"] > 0]
    row["maintain_n"]    = len(maintain)
    row["maintain_psb"]  = int(maintain["psb"].sum())
    row["maintain_prec"] = round(maintain["psb"].sum() / max(len(maintain), 1), 4)
    return row


def run_weight_sensitivity(thr: float):
    tag = f"thr={thr}"
    print(f"\n{'='*60}")
    print(f"  Exp 4: Weight Sensitivity  [{tag}]")
    print(f"{'='*60}")
    t_start = time.time()

    labels = pd.read_parquet(labels_file(thr))
    df35   = load_and_merge(3.5,  labels)
    df75   = load_and_merge(7.5,  labels)
    df115  = load_and_merge(11.5, labels)
    rdir   = results_dir(thr)

    rows = []
    for w in PSB_WEIGHTS:
        t0 = time.time()
        print(f"\n── weight={w} ──")
        te = run_bi(df35, df75, df115, w)
        row = evaluate(te, w)
        rows.append(row)
        print(f"  PR_AUC={row['PR_AUC']}  P@100={row['P@100']}  "
              f"lift@100={row['lift@100']}x  "
              f"maintain={row['maintain_n']}(prec={row['maintain_prec']})  "
              f"({time.time()-t0:.1f}s)")

    result_df = pd.DataFrame(rows)
    print(f"\n{'='*70}")
    print("Weight Sensitivity Results")
    print("="*70)
    print(result_df.to_string(index=False))

    out = rdir / "weight_sensitivity.csv"
    result_df.to_csv(out, index=False)
    print(f"\n  Saved: {out.name}")
    print(f"[{tag}] Done. ({time.time()-t_start:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all")
    args = parser.parse_args()
    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]
    for thr in thrs:
        run_weight_sensitivity(thr)


if __name__ == "__main__":
    main()
