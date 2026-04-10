"""
Exp 8: BI vs Classifier head-to-head

핵심 질문: 동일한 피처·동일한 test set에서
  - Logistic Regression (t=3.5yr features, binary classifier)
  - XGBoost Classifier  (t=3.5yr features, binary classifier)
  - BI Q_35             (backward induction Q-value regressor)
중 어느 것이 더 나은가? 그리고 그 차이는 통계적으로 유의한가?

baseline_comparison.parquet (exp2 산출물) 을 읽어 재사용하므로
exp2 를 먼저 돌려야 한다.

평가 지표
  - AP (average precision)
  - P@100, lift@100
  - P@500, lift@500
  - P@1000, lift@1000

유의성 검정
  1. Bootstrap 95% CI for each metric  (n_boot=1000 resamples)
  2. Paired bootstrap test  delta(A, B) = metric(A) - metric(B)
     95% CI of delta; test whether CI excludes 0

사용법:
  python exp8_bi_vs_classifier.py
  python exp8_bi_vs_classifier.py --thr 0.005
  python exp8_bi_vs_classifier.py --thr all
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import PSB_THRESHOLDS, results_dir, ROOT


MODELS = {
    "BI Q_35":       "q35_score",
    "XGBoost":       "xgb_score",
    "LogisticReg":   "lr_score",
}

K_LIST   = [100, 500, 1000]
N_BOOT   = 1000
RNG_SEED = 42


def compute_metrics(y: np.ndarray, s: np.ndarray, base_rate: float) -> dict:
    out = {"AP": average_precision_score(y, s) if np.unique(s).size > 1 else 0.0}
    order = np.argsort(-s, kind="stable")
    y_sorted = y[order]
    for k in K_LIST:
        k_ = min(k, len(y))
        hits = int(y_sorted[:k_].sum())
        p = hits / k_
        out[f"P@{k}"]    = p
        out[f"hits@{k}"] = hits
        out[f"lift@{k}"] = p / base_rate if base_rate > 0 else 0.0
    return out


def bootstrap_metrics(y: np.ndarray, scores: dict, n_boot: int,
                       rng: np.random.Generator) -> dict:
    """각 모델에 대해 n_boot 개의 점 추정치 샘플 반환."""
    n = len(y)
    # 전체 test set resample — 모델 간 paired 비교가 가능하도록
    # 동일한 boot index 를 모든 모델에 적용
    base_rate_const = y.mean()
    draws = {
        m: {"AP": [], **{f"P@{k}": [] for k in K_LIST},
            **{f"lift@{k}": [] for k in K_LIST},
            **{f"hits@{k}": [] for k in K_LIST}}
        for m in scores
    }
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_b = y[idx]
        br_b = y_b.mean()
        for m, s in scores.items():
            s_b = s[idx]
            # AP
            if np.unique(s_b).size > 1 and y_b.sum() > 0:
                ap = average_precision_score(y_b, s_b)
            else:
                ap = 0.0
            draws[m]["AP"].append(ap)
            order = np.argsort(-s_b, kind="stable")
            y_sorted = y_b[order]
            for k in K_LIST:
                k_ = min(k, n)
                hits = int(y_sorted[:k_].sum())
                p = hits / k_
                draws[m][f"P@{k}"].append(p)
                draws[m][f"hits@{k}"].append(hits)
                # lift vs original base rate (constant), not boot-sample base rate
                draws[m][f"lift@{k}"].append(p / base_rate_const if base_rate_const > 0 else 0.0)
        if (b + 1) % 200 == 0:
            print(f"    boot {b+1}/{n_boot}", end="\r")
    print()
    return draws


def ci(arr: list, alpha: float = 0.05) -> tuple:
    a = np.asarray(arr)
    lo, hi = np.quantile(a, [alpha/2, 1 - alpha/2])
    return float(lo), float(hi), float(np.mean(a)), float(np.std(a))


def paired_delta_ci(draws_a: list, draws_b: list, alpha: float = 0.05) -> dict:
    a = np.asarray(draws_a)
    b = np.asarray(draws_b)
    d = a - b
    lo, hi = np.quantile(d, [alpha/2, 1 - alpha/2])
    p_gt0 = float((d > 0).mean())
    return {"mean_delta": float(d.mean()),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "frac_positive": p_gt0,
            "sig": bool(lo > 0 or hi < 0)}


def run_exp8(thr: float):
    print(f"\n{'='*70}")
    print(f"  Exp 8: BI vs Classifier  [thr={thr}]")
    print(f"{'='*70}")

    rdir = results_dir(thr)
    bc_path = rdir / "baseline_comparison.parquet"
    if not bc_path.exists():
        print(f"  [ERR] {bc_path} not found — run exp2 first.")
        return None

    bc = pd.read_parquet(bc_path)
    y = bc["psb"].values.astype(int)
    n = len(y)
    base_rate = y.mean()
    print(f"  n={n:,}  PSB={int(y.sum())}  base_rate={base_rate:.5f}")

    scores = {}
    for m_name, col in MODELS.items():
        if col not in bc.columns:
            print(f"  [WARN] {col} missing — skipped {m_name}")
            continue
        s = bc[col].fillna(0).values.astype(float)
        scores[m_name] = s

    # ── Point metrics ───────────────────────────────
    print("\n-- Point estimates --")
    rows = []
    for m, s in scores.items():
        met = compute_metrics(y, s, base_rate)
        met["model"] = m
        rows.append(met)
    point_df = pd.DataFrame(rows)
    cols = ["model", "AP",
            "P@100", "lift@100", "hits@100",
            "P@500", "lift@500", "hits@500",
            "P@1000", "lift@1000", "hits@1000"]
    point_df = point_df[cols]
    for c in ["AP", "P@100", "P@500", "P@1000"]:
        point_df[c] = point_df[c].round(4)
    for c in ["lift@100", "lift@500", "lift@1000"]:
        point_df[c] = point_df[c].round(2)
    print(point_df.to_string(index=False))

    # ── Bootstrap CIs ───────────────────────────────
    print(f"\n-- Bootstrap 95% CIs  (n_boot={N_BOOT}) --")
    rng = np.random.default_rng(RNG_SEED)
    draws = bootstrap_metrics(y, scores, n_boot=N_BOOT, rng=rng)

    ci_rows = []
    for m in scores:
        for metric in ["AP", "P@100", "lift@100", "P@500", "lift@500",
                       "P@1000", "lift@1000"]:
            lo, hi, mean, sd = ci(draws[m][metric])
            ci_rows.append({
                "model": m, "metric": metric,
                "mean": round(mean, 4),
                "ci_lo": round(lo, 4), "ci_hi": round(hi, 4),
                "sd": round(sd, 4),
            })
    ci_df = pd.DataFrame(ci_rows)
    print(ci_df.to_string(index=False))

    # ── Paired BI vs others ─────────────────────────
    print("\n-- Paired delta CIs: does BI Q_35 beat the classifier? --")
    paired_rows = []
    if "BI Q_35" in scores:
        for other in [m for m in scores if m != "BI Q_35"]:
            for metric in ["AP", "P@100", "lift@100",
                           "P@500", "lift@500", "P@1000", "lift@1000"]:
                r = paired_delta_ci(
                    draws["BI Q_35"][metric], draws[other][metric]
                )
                r["comparison"] = f"BI_Q_35 - {other}"
                r["metric"] = metric
                paired_rows.append(r)
    paired_df = pd.DataFrame(paired_rows)
    if len(paired_df) > 0:
        paired_df = paired_df[
            ["comparison", "metric", "mean_delta",
             "ci_lo", "ci_hi", "frac_positive", "sig"]
        ]
        for c in ["mean_delta", "ci_lo", "ci_hi"]:
            paired_df[c] = paired_df[c].round(4)
        paired_df["frac_positive"] = paired_df["frac_positive"].round(3)
        print(paired_df.to_string(index=False))

        # summary
        sig_wins = paired_df[paired_df["sig"] & (paired_df["mean_delta"] > 0)]
        sig_losses = paired_df[paired_df["sig"] & (paired_df["mean_delta"] < 0)]
        print(f"\n  BI_Q_35 significantly wins : {len(sig_wins)} metric-comparisons")
        print(f"  BI_Q_35 significantly loses: {len(sig_losses)} metric-comparisons")

    # ── Save ─────────────────────────────────────
    point_df.to_csv(rdir / "exp8_point_estimates.csv", index=False)
    ci_df.to_csv(rdir / "exp8_bootstrap_ci.csv", index=False)
    if len(paired_df) > 0:
        paired_df.to_csv(rdir / "exp8_paired_delta.csv", index=False)
    print(f"\n  Saved: exp8_*.csv to {rdir}")

    return point_df, ci_df, paired_df if len(paired_df) > 0 else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all")
    args = parser.parse_args()

    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]

    all_paired = []
    for thr in thrs:
        res = run_exp8(thr)
        if res is not None and res[2] is not None:
            p = res[2].copy()
            p["thr"] = thr
            all_paired.append(p)

    if all_paired:
        combined = pd.concat(all_paired, ignore_index=True)
        out = ROOT / "results" / "exp8_bi_vs_classifier_summary.csv"
        combined.to_csv(out, index=False)
        print(f"\n{'#'*70}")
        print(f"  OVERALL SUMMARY")
        print(f"{'#'*70}")
        pivot = combined.pivot_table(
            index=["comparison", "metric"],
            columns="thr",
            values=["mean_delta", "sig"],
            aggfunc="first",
        )
        print(pivot.to_string())
        print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
