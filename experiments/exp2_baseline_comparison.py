"""
Exp 2: Baseline 모델 비교

Baselines:
  1. Early-truncated Beauty B' (t<=3.5yr 구간만 사용, 시간 대칭)
  2. 누적 인용수 (t=3.5yr) 랭킹
  3. XGBoost Classifier (t=3.5yr 피처)
  4. Random Forest Classifier (t=3.5yr 피처)
  5. Backward Induction 결과 (exp1에서 생성된 test_qvalues.parquet)

※ 전 생애(age 0~25) 기반 Beauty Coefficient B는 PSB 라벨의 정의
   그 자체이므로 baseline에서 제외됨 (label leakage).
   대신 의사결정 시점과 같은 정보 범위를 쓰는 B'를 사용한다.

※ 모든 모델은 동일한 test set (1988-1989)에서 평가

사용법:
  python exp2_baseline_comparison.py              # thr=0.001
  python exp2_baseline_comparison.py --thr 0.005
  python exp2_baseline_comparison.py --thr all
"""

import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import PSB_THRESHOLDS, labels_file, models_dir, results_dir
from utils import (
    load_and_merge, split, get_feature_cols,
    train_xgb_classifier, train_rf_classifier,
    evaluate_ranking, save_model,
)


def _compute_early_beauty(mat: np.ndarray) -> np.ndarray:
    """
    Early-truncated Beauty Coefficient B'.

    mat : (N, T) forward citation counts for ages 0..T-1
    B'  = sum_{t=0..t_m} (l(t) - c(t)) / max(1, c(t))
          where t_m = argmax_t c(t) within observed window,
                l(t) = linear baseline from c(0) to c(t_m).

    원본 beauty_coefficient(step4)와 동일한 공식을
    관측 가능한 초기 구간에만 제한 적용한 버전.
    """
    N, T  = mat.shape
    out   = np.zeros(N, dtype=float)
    if T == 0:
        return out
    t_m   = mat.argmax(axis=1)                      # (N,)
    c0    = mat[:, 0]                                # (N,)
    c_tm  = mat[np.arange(N), t_m]                   # (N,)
    # 단일 값이면 (t_m==0) B'=0
    active = t_m > 0
    for i in np.where(active)[0]:
        tm_i = int(t_m[i])
        slope = (c_tm[i] - c0[i]) / tm_i
        s = 0.0
        for t in range(tm_i + 1):
            l_t = slope * t + c0[i]
            c_t = mat[i, t]
            s  += (l_t - c_t) / max(1.0, c_t)
        out[i] = s
    return out


def run_baseline_comparison(thr: float):
    tag = f"thr={thr}"
    print(f"\n{'='*60}")
    print(f"  Exp 2: Baseline Comparison  [{tag}]")
    print(f"{'='*60}")
    t_start = time.time()

    labels = pd.read_parquet(labels_file(thr))
    mdir   = models_dir(thr)
    rdir   = results_dir(thr)

    # 3.5yr 피처 로드 (baseline 모델 학습 기준)
    df35 = load_and_merge(3.5, labels)
    tr35, val35, te35 = split(df35)
    print(f"  Train={len(tr35):,}(PSB={tr35['psb'].sum()})  "
          f"Val={len(val35):,}(PSB={val35['psb'].sum()})  "
          f"Test={len(te35):,}(PSB={te35['psb'].sum()})")

    feat_cols = [c for c in get_feature_cols(3.5) if c in tr35.columns]
    X_tr  = tr35[feat_cols].values;  y_tr  = tr35["psb"].values
    X_val = val35[feat_cols].values; y_val = val35["psb"].values
    X_te  = te35[feat_cols].values;  y_te  = te35["psb"].values

    all_results = {"patent_id":   te35["patent_id"].values,
                   "psb":         te35["psb"].values,
                   "ipc_subclass":te35["ipc_subclass"].values}

    # ── Baseline 1: Early-truncated Beauty B' (t<=3.5yr) ──
    # 의사결정 시점과 동일한 정보 범위로 계산한 Beauty.
    # 전 생애 B는 PSB 라벨 정의에 쓰이므로 baseline 대상에서 제외됨.
    print("\n── Baseline 1: Early-truncated Beauty (t<=3.5yr) ──")
    early_cols = ["t35__cite_yr0", "t35__cite_yr1",
                  "t35__cite_yr2", "t35__cite_yr3"]
    early_cols = [c for c in early_cols if c in te35.columns]
    early_mat  = te35[early_cols].fillna(0).values.astype(float)
    b_prime    = _compute_early_beauty(early_mat)
    beauty_scores_early = pd.Series(b_prime, index=te35.index)
    evaluate_ranking(te35["psb"], beauty_scores_early,
                     label="Early Beauty B'")
    all_results["beauty_early_score"] = beauty_scores_early.values

    # ── Baseline 2: Cumulative Citations (3.5yr) ─────
    print("\n── Baseline 2: Cumulative Citations (3.5yr) ──")
    cite_col = "t35__cum_citations"
    cite_scores = te35[cite_col] if cite_col in te35.columns else pd.Series(0, index=te35.index)
    evaluate_ranking(te35["psb"], cite_scores, label="Cum Citations")
    all_results["cite_score"] = cite_scores.values

    # ── Baseline 3: XGBoost Classifier ──────────────
    print("\n── Baseline 3: XGBoost Classifier ──")
    xgb_model  = train_xgb_classifier(X_tr, y_tr, X_val, y_val, psb_tr=y_tr)
    xgb_scores = pd.Series(xgb_model.predict_proba(X_te)[:, 1], index=te35.index)
    evaluate_ranking(te35["psb"], xgb_scores, label="XGBoost")
    save_model(xgb_model, mdir / "baseline_xgb.pkl")
    all_results["xgb_score"] = xgb_scores.values

    # ── Baseline 4: Random Forest ────────────────────
    print("\n── Baseline 4: Random Forest ──")
    rf_model  = train_rf_classifier(X_tr, y_tr, X_val, y_val, psb_tr=y_tr)
    rf_scores = pd.Series(rf_model.predict_proba(X_te)[:, 1], index=te35.index)
    evaluate_ranking(te35["psb"], rf_scores, label="Random Forest")
    save_model(rf_model, mdir / "baseline_rf.pkl")
    all_results["rf_score"] = rf_scores.values

    # ── Baseline 5: Backward Induction (exp1 결과) ───
    print("\n── Baseline 5: Backward Induction (from exp1) ──")
    bi_path = rdir / "test_qvalues.parquet"
    if bi_path.exists():
        bi = pd.read_parquet(bi_path)
        te35_bi = te35[["patent_id"]].merge(
            bi[["patent_id", "policy_full_maintain", "Q_35_pred"]],
            on="patent_id", how="left",
        )
        # merge 후 index 리셋 → te35와 행 순서 일치 보장
        bi_policy_scores = pd.Series(te35_bi["policy_full_maintain"].fillna(0).values)
        bi_q35_scores    = pd.Series(te35_bi["Q_35_pred"].fillna(-999).values)
        psb_series       = pd.Series(te35["psb"].values)
        evaluate_ranking(psb_series, bi_policy_scores, label="BI Policy")
        evaluate_ranking(psb_series, bi_q35_scores,    label="BI Q_35")
        all_results["bi_policy_score"] = bi_policy_scores.values
        all_results["q35_score"]       = bi_q35_scores.values
    else:
        print(f"  [WARN] {bi_path.name} not found — run exp1 first.")
        all_results["bi_policy_score"] = np.zeros(len(te35))
        all_results["q35_score"]       = np.zeros(len(te35))

    # ── 결과 저장 ────────────────────────────────────
    out = pd.DataFrame(all_results)
    out_path = rdir / "baseline_comparison.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path.name}  ({len(out):,} rows)")
    print(f"[{tag}] Done. ({time.time()-t_start:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all")
    args = parser.parse_args()
    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]
    for thr in thrs:
        run_baseline_comparison(thr)


if __name__ == "__main__":
    main()
