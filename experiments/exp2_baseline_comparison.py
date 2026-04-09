"""
Exp 2: Baseline 모델 비교

Baselines:
  1. Beauty Coefficient B 랭킹
  2. 누적 인용수 (t=3.5yr) 랭킹
  3. XGBoost Classifier (t=3.5yr 피처)
  4. Random Forest Classifier (t=3.5yr 피처)
  5. Backward Induction 결과 (exp1에서 생성된 test_qvalues.parquet)

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
                   "B":           te35["B"].values,
                   "ipc_subclass":te35["ipc_subclass"].values}

    # ── Baseline 1: Beauty Coefficient ──────────────
    print("\n── Baseline 1: Beauty Coefficient ──")
    beauty_scores = te35["B"]
    evaluate_ranking(te35["psb"], beauty_scores, label="Beauty Coeff")
    all_results["beauty_score"] = beauty_scores.values

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
