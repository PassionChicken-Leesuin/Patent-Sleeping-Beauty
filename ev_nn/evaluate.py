"""
ev_nn/evaluate.py — 심층 평가 스크립트
=========================================
train.py 가 저장한 results/*.parquet 을 불러와
ranking 지표 외의 추가 분석을 수행.

분석 항목:
  1. EV Calibration   — 예측 Q 구간별 실제 평균 reward
  2. Portfolio Profit — 상위 K 유지 시 실제 총 수익
  3. exp1(BI XGBoost) 와 직접 비교 (같은 test set 기준)

사용법:
  python evaluate.py                          # results/ 내 전체 파일
  python evaluate.py --file test_qvalues_thr001_weighted.parquet
  python evaluate.py --k 100 200 500 1000
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = Path(__file__).parent / "results"
EXP1_RESULTS_DIR = ROOT / "results"      # experiments/ 결과 (thr별 하위폴더)


# ══════════════════════════════════════════════════════
#  1. EV Calibration
# ══════════════════════════════════════════════════════
def ev_calibration(df: pd.DataFrame, score_col: str = "Q_35",
                   n_bins: int = 10) -> pd.DataFrame:
    """
    예측 Q 값을 n_bins 구간으로 나눠 각 구간 내
    실제 평균 reward (PSB=1 비율 × R - cost) 와 비교.

    실제 reward per sample:
      r_i = -3.85 (PSB=0)  or  +4.15 (PSB=1)
      단, 여기선 PSB=1 비율 × R - cost 로 집계
    """
    from config import MAINT_COST, PSB_REWARD
    cost = MAINT_COST[3.5]   # t=3.5yr 기준

    df = df[[score_col, "psb"]].dropna()
    df["actual_r"] = -cost + df["psb"] * PSB_REWARD

    df["bin"] = pd.qcut(df[score_col], q=n_bins, duplicates="drop")
    calib = (
        df.groupby("bin", observed=True)
        .agg(
            count        = ("psb", "count"),
            psb_rate     = ("psb", "mean"),
            mean_pred_Q  = (score_col, "mean"),
            mean_actual_r= ("actual_r", "mean"),
        )
        .reset_index()
    )
    calib["calibration_err"] = calib["mean_pred_Q"] - calib["mean_actual_r"]
    return calib


# ══════════════════════════════════════════════════════
#  2. Portfolio Profit @ K
# ══════════════════════════════════════════════════════
def portfolio_profit(df: pd.DataFrame, score_col: str = "Q_35",
                     k_list: list = None) -> pd.DataFrame:
    """
    score 기준 상위 K 특허를 유지할 때의 실제 총 수익.

    총 수익 = sum(-cost_35 + psb_i × R)  for top-K patents
    (유지 비용은 항상 지불, PSB=1이면 보상 수령)
    """
    from config import MAINT_COST, PSB_REWARD
    if k_list is None:
        k_list = [100, 200, 500, 1000]

    cost = MAINT_COST[3.5]
    df = df[[score_col, "psb"]].dropna().sort_values(score_col, ascending=False)

    rows = []
    for k in k_list:
        k_ = min(k, len(df))
        top = df.head(k_)
        profit = float((-cost + top["psb"] * PSB_REWARD).sum())
        psb_count = int(top["psb"].sum())
        rows.append({
            "K":          k,
            "n_selected": k_,
            "psb_count":  psb_count,
            "profit":     profit,
            "profit_per_patent": profit / k_ if k_ > 0 else 0.0,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
#  3. exp1 BI vs ev_nn 비교 테이블
# ══════════════════════════════════════════════════════
def compare_with_bi(ev_df: pd.DataFrame, thr: float = 0.001,
                    k_list: list = None) -> pd.DataFrame:
    """
    exp1(BI XGBoost) 결과 파일을 불러와 portfolio profit 비교.
    같은 patent_id / test set 기준.
    """
    if k_list is None:
        k_list = [100, 200, 500, 1000]

    from config import MAINT_COST, PSB_REWARD, thr_tag
    cost = MAINT_COST[3.5]

    bi_path = EXP1_RESULTS_DIR / thr_tag(thr) / "test_qvalues.parquet"
    if not bi_path.exists():
        print(f"  [compare] BI result not found: {bi_path}")
        return pd.DataFrame()

    bi_df = pd.read_parquet(bi_path)[["patent_id", "Q_35_pred", "psb"]]
    bi_df = bi_df.rename(columns={"Q_35_pred": "Q_35"})

    rows = []
    for name, df_, col in [
        ("ev_nn (Neural FQI)", ev_df, "Q_35"),
        ("exp1  (BI XGBoost)", bi_df, "Q_35"),
    ]:
        df_ = df_[[col, "psb"]].dropna().sort_values(col, ascending=False)
        for k in k_list:
            k_  = min(k, len(df_))
            top = df_.head(k_)
            profit = float((-cost + top["psb"] * PSB_REWARD).sum())
            rows.append({
                "model": name,
                "K": k,
                "psb_count":  int(top["psb"].sum()),
                "profit":     profit,
                "profit_per_patent": profit / k_ if k_ > 0 else 0.0,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--file", default=None,
                        help="특정 parquet 파일명 (없으면 results/ 전체)")
    parser.add_argument("--thr",  type=float, default=0.001,
                        help="BI 비교용 PSB threshold")
    parser.add_argument("--k",    nargs="+", type=int,
                        default=[100, 200, 500, 1000])
    parser.add_argument("--bins", type=int, default=10,
                        help="Calibration 구간 수")
    args, _ = parser.parse_known_args()

    # ── 대상 파일 결정 ────────────────────────────────
    if args.file:
        parq_files = [RESULTS_DIR / args.file]
    else:
        parq_files = sorted(RESULTS_DIR.glob("test_qvalues_*.parquet"))

    if not parq_files:
        print("No result files found in", RESULTS_DIR)
        return

    for parq_path in parq_files:
        print("\n" + "=" * 60)
        print(f"  Evaluating: {parq_path.name}")
        print("=" * 60)

        df = pd.read_parquet(parq_path)
        if "Q_35" not in df.columns:
            print("  [skip] Q_35 column not found.")
            continue

        # ── 1. EV Calibration ─────────────────────────
        print("\n── 1. EV Calibration (Q_35 예측값 vs 실제 reward) ──")
        calib = ev_calibration(df, score_col="Q_35", n_bins=args.bins)
        print(calib[["bin", "count", "psb_rate", "mean_pred_Q",
                      "mean_actual_r", "calibration_err"]].to_string(index=False))

        # ── 2. Portfolio Profit ────────────────────────
        print("\n── 2. Portfolio Profit @ K ──")
        profit_df = portfolio_profit(df, score_col="Q_35", k_list=args.k)
        print(profit_df.to_string(index=False))

        # ── 3. vs exp1 BI 비교 ─────────────────────────
        print(f"\n── 3. vs exp1 BI XGBoost  (thr={args.thr}) ──")
        cmp_df = compare_with_bi(df, thr=args.thr, k_list=args.k)
        if not cmp_df.empty:
            print(cmp_df.pivot_table(
                index="K",
                columns="model",
                values=["psb_count", "profit", "profit_per_patent"]
            ).to_string())

        # ── 저장 ──────────────────────────────────────
        stem     = parq_path.stem.replace("test_qvalues_", "")
        out_calib  = RESULTS_DIR / f"calibration_{stem}.csv"
        out_profit = RESULTS_DIR / f"profit_{stem}.csv"
        calib.to_csv(out_calib,  index=False)
        profit_df.to_csv(out_profit, index=False)
        print(f"\n  Saved: {out_calib.name},  {out_profit.name}")


if __name__ == "__main__":
    main()
