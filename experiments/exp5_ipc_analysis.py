"""
Exp 5: IPC 서브클래스별 성능 분석

baseline_comparison.parquet (exp2 결과)를 IPC별로 집계하여
Backward Induction vs Baselines 성능을 기술 분야 단위로 비교

사용법:
  python exp5_ipc_analysis.py
  python exp5_ipc_analysis.py --thr 0.005
  python exp5_ipc_analysis.py --thr all
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.metrics import average_precision_score

from config import PSB_THRESHOLDS, results_dir

MIN_GROUP_SIZE = 50   # IPC 그룹 최소 특허 수


def evaluate_group(y_true: pd.Series, scores: pd.Series,
                   k_list=(100, 200, 500)) -> dict:
    combined = pd.DataFrame({"y": y_true, "score": scores}).dropna()
    if len(combined) == 0 or combined["y"].sum() == 0:
        return {"count": len(y_true), "psb_count": int(y_true.sum()),
                "average_precision": 0.0,
                **{f"precision@{k}": 0.0 for k in k_list}}
    combined = combined.sort_values("score", ascending=False)
    ap = average_precision_score(combined["y"], combined["score"])
    precs = {}
    for k in k_list:
        k_ = min(k, len(combined))
        precs[f"precision@{k}"] = combined.head(k_)["y"].sum() / k_ if k_ > 0 else 0.0
    return {"count": len(y_true), "psb_count": int(y_true.sum()),
            "average_precision": ap, **precs}


def run_ipc_analysis(thr: float):
    tag  = f"thr={thr}"
    rdir = results_dir(thr)
    print(f"\n{'='*60}")
    print(f"  Exp 5: IPC Analysis  [{tag}]")
    print(f"{'='*60}")

    bc_path = rdir / "baseline_comparison.parquet"
    if not bc_path.exists():
        print(f"  [WARN] {bc_path.name} not found -- run exp2 first.")
        return

    df = pd.read_parquet(bc_path).dropna(subset=["ipc_subclass"])
    print(f"  Loaded: {len(df):,} rows")

    # 전 생애 Beauty(B)는 PSB 라벨 정의에 사용되므로 baseline 제외.
    # 시간 대칭 버전(beauty_early_score)은 exp2 에서 생성되면 자동 포함.
    score_cols = {
        "bi":           "bi_policy_score",
        "q35":          "q35_score",
        "rf":           "rf_score",
        "xgb":          "xgb_score",
        "lr":           "lr_score",
        "beauty_early": "beauty_early_score",
    }

    records = []
    for ipc, grp in df.groupby("ipc_subclass"):
        if len(grp) < MIN_GROUP_SIZE:
            continue
        rec = {"ipc_subclass": ipc}
        for prefix, col in score_cols.items():
            if col not in grp.columns:
                continue
            m = evaluate_group(grp["psb"], grp[col])
            for k, v in m.items():
                rec[f"{prefix}_{k}"] = v
        records.append(rec)

    results_df = pd.DataFrame(records)
    results_df = results_df.sort_values("bi_psb_count", ascending=False)

    print("\n── Top 10 IPC Subclasses (by PSB count) ──")
    for _, row in results_df.head(10).iterrows():
        print(f"  {row['ipc_subclass']:6s}  "
              f"PSB={int(row['bi_psb_count'])}  "
              f"BI_AP={row['bi_average_precision']:.4f}  "
              f"RF_AP={row['rf_average_precision']:.4f}  "
              f"Q35_AP={row.get('q35_average_precision', 0):.4f}")

    print("\n── Overall Average AP ──")
    for prefix in score_cols:
        col = f"{prefix}_average_precision"
        if col in results_df.columns:
            print(f"  {prefix:8s}: {results_df[col].mean():.4f}")

    results_df.to_parquet(rdir / "ipc_analysis.parquet", index=False)
    results_df.to_csv(rdir / "ipc_analysis.csv", index=False)
    print(f"\n  Saved ipc_analysis.parquet / .csv  [{tag}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all")
    args = parser.parse_args()
    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]
    for thr in thrs:
        run_ipc_analysis(thr)


if __name__ == "__main__":
    main()
