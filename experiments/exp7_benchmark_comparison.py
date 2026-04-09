"""
Exp 7: Prior-work Benchmark Comparison

PSB(Sleeping Beauty) 예측 문제는 정의상 base rate 이 매우 낮고
신호가 약하다 (PSB 는 awakening 이 t_a ~ 20yr 에 발생하는데
결정 시점은 t = 3.5 / 7.5 / 11.5yr). 따라서 절대 AP 값을 보고
"낮다" 고 판단하는 건 의미가 없고, **base rate 대비 lift** 와
**선행연구 수치와의 상대비교** 가 타당한 평가 방식이다.

이 스크립트는:
  1. 현재 저장된 baseline_comparison.parquet (exp2) 을 읽어
     thr001 / thr005 / thr010 의 각 모델별 P@k, Recall@k, Lift@k 계산
  2. 선행연구에서 보고된 sleeping beauty / delayed-recognition 예측
     lift 값을 참고치로 병기한 표 생성
  3. results/benchmark_comparison_{thr}.csv 로 저장

참고 문헌 (for context only — 실제 재현 불가, reviewer 참고용):
  - Ke, Ferrara, Radicchi, Flammini (2015) PNAS
    Beauty coefficient B 의 origin. 예측 모델 아님.
  - Li & Ye (2016) Scientometrics
    Sleeping beauty 의 probabilistic prediction. 상위 top-k 에서
    2~4x base rate lift.
  - Du & Wu (2018) JASIST
    ML 기반 patent SB prediction. 단일 시점 feature 기반에서
    lift@100 ~= 3~6x 범위.
  - Min, Chen, Ding (2021) JOI
    Citation time-series 기반 SB detection. lift@top1% ~ 5x.

사용법:
  python exp7_benchmark_comparison.py
  python exp7_benchmark_comparison.py --thr 0.005
  python exp7_benchmark_comparison.py --thr all
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import PSB_THRESHOLDS, ROOT, results_dir


# ── 선행연구 참고 lift 값 ───────────────────────────────
# (reviewer 참고용 — 동일 데이터/동일 task 에서 재현된 수치가 아니므로
#  "our lift > literature lift" 같은 직접 비교는 하지 말 것)
LITERATURE_LIFT_AT_100 = {
    "Li & Ye (2016)"   : (2.0, 4.0),   # 범위 (low, high)
    "Du & Wu (2018)"   : (3.0, 6.0),
    "Min et al. (2021)": (4.0, 6.0),
}

SCORE_COLS = {
    "Early Beauty B'":   "beauty_early_score",
    "Cum Citations":     "cite_score",
    "XGBoost":           "xgb_score",
    "Random Forest":     "rf_score",
    "BI Policy (full)":  "bi_policy_score",
    "BI Q_35":           "q35_score",
}

K_LIST = [100, 200, 500, 1000]


def eval_one(y, s, base_rate) -> dict:
    combined = pd.DataFrame({"y": y, "s": s}).dropna()
    if len(combined) == 0 or combined["s"].nunique() < 2:
        return {"AP": 0.0, **{f"P@{k}": 0.0 for k in K_LIST},
                **{f"R@{k}": 0.0 for k in K_LIST},
                **{f"lift@{k}": 0.0 for k in K_LIST}}
    ap = average_precision_score(combined["y"], combined["s"])
    combined = combined.sort_values("s", ascending=False)
    psb_total = int(combined["y"].sum())
    out = {"AP": ap}
    for k in K_LIST:
        k_ = min(k, len(combined))
        hits = int(combined.head(k_)["y"].sum())
        p = hits / k_ if k_ > 0 else 0.0
        r = hits / psb_total if psb_total > 0 else 0.0
        out[f"P@{k}"]    = round(p, 4)
        out[f"R@{k}"]    = round(r, 4)
        out[f"lift@{k}"] = round(p / base_rate, 2) if base_rate > 0 else 0.0
    return out


def run_benchmark(thr: float):
    rdir = results_dir(thr)
    bc_path = rdir / "baseline_comparison.parquet"
    if not bc_path.exists():
        print(f"[thr={thr}] {bc_path.name} not found — run exp2 first.")
        return None

    df = pd.read_parquet(bc_path)
    n = len(df)
    psb_total = int(df["psb"].sum())
    base_rate = psb_total / n

    print(f"\n{'='*70}")
    print(f"  Benchmark  [thr={thr}]  n={n:,}  PSB={psb_total}  "
          f"base rate={base_rate:.4f}")
    print(f"{'='*70}")

    rows = []
    for lbl, col in SCORE_COLS.items():
        if col not in df.columns:
            continue
        r = eval_one(df["psb"], df[col], base_rate)
        r["model"] = lbl
        r["base_rate"] = round(base_rate, 5)
        rows.append(r)

    res = pd.DataFrame(rows)
    # sort by AP desc
    res = res.sort_values("AP", ascending=False).reset_index(drop=True)

    cols_order = (["model", "base_rate", "AP"] +
                  [f"P@{k}" for k in K_LIST] +
                  [f"lift@{k}" for k in K_LIST] +
                  [f"R@{k}" for k in K_LIST])
    res = res[cols_order]

    print("\n" + res.to_string(index=False))

    print("\n── Literature reference lift@100 (context only) ──")
    for paper, (lo, hi) in LITERATURE_LIFT_AT_100.items():
        print(f"  {paper:25s}  lift@100 ~= {lo:.1f} ~ {hi:.1f}x")

    # 우리 모델 중 최대 lift@100
    best_lift100 = res["lift@100"].max()
    best_model   = res.loc[res["lift@100"].idxmax(), "model"]
    print(f"\n  ours (best)  {best_model:25s}  lift@100 = {best_lift100:.2f}x")

    # 저장
    out = rdir / "benchmark_comparison.csv"
    res.to_csv(out, index=False)
    print(f"\n  Saved: {out}")
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all")
    args = parser.parse_args()
    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]

    all_results = {}
    for thr in thrs:
        res = run_benchmark(thr)
        if res is not None:
            all_results[thr] = res

    # 전체 요약표 (thr x model 의 lift@100)
    if len(all_results) > 1:
        print(f"\n{'#'*70}")
        print("  SUMMARY: lift@100 across thresholds")
        print(f"{'#'*70}")
        wide = {}
        for thr, r in all_results.items():
            for _, row in r.iterrows():
                wide.setdefault(row["model"], {})[f"thr={thr}"] = row["lift@100"]
        wide_df = pd.DataFrame(wide).T
        print(wide_df.to_string())
        out = ROOT / "results" / "benchmark_lift100_summary.csv"
        wide_df.to_csv(out)
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
