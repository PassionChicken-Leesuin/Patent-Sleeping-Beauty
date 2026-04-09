"""
Exp 1: Backward Induction (Fitted Q-Iteration)

구조:
  t=11.5yr (terminal): Y = -cost_11.5 + PSB_REWARD * I(psb=1)
  t=7.5yr:             Y = -cost_7.5  + γ * max(Q_115(s'), 0)
  t=3.5yr:             Y = -cost_3.5  + γ * max(Q_75(s'),  0)

사용법:
  python exp1_backward_induction.py                  # thr=0.001
  python exp1_backward_induction.py --thr 0.005      # thr=0.005
  python exp1_backward_induction.py --thr all        # 3가지 전부
"""

import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from config import (
    GAMMA, MAINT_COST, PSB_REWARD, PSB_WEIGHT, PSB_THRESHOLDS,
    TRAIN_YEARS, VAL_YEARS, TEST_YEARS,
    labels_file, models_dir, results_dir,
)
from utils import (
    load_and_merge, split, encode_ipc_splits, print_split_stats,
    get_feature_cols, train_qmodel, evaluate_ranking, save_model,
)


def run_backward_induction(thr: float):
    tag = f"thr={thr}"
    print(f"\n{'='*60}")
    print(f"  Exp 1: Backward Induction  [{tag}]")
    print(f"{'='*60}")
    t_start = time.time()

    # ── 데이터 로드 ─────────────────────────────────
    labels = pd.read_parquet(labels_file(thr))
    print(f"Labels: {len(labels):,}  PSB=1: {labels['psb'].sum():,}")

    df35  = load_and_merge(3.5,  labels)
    df75  = load_and_merge(7.5,  labels)
    df115 = load_and_merge(11.5, labels)

    tr35,  val35,  te35  = split(df35)
    tr75,  val75,  te75  = split(df75)
    tr115, val115, te115 = split(df115)

    # IPC freq encoding: train fold 에서만 fit, val/test 에 transform
    tr35,  val35,  te35  = encode_ipc_splits(tr35,  val35,  te35)
    tr75,  val75,  te75  = encode_ipc_splits(tr75,  val75,  te75)
    tr115, val115, te115 = encode_ipc_splits(tr115, val115, te115)

    print("\nData split:")
    print_split_stats("3.5yr",  tr35,  val35,  te35)
    print_split_stats("7.5yr",  tr75,  val75,  te75)
    print_split_stats("11.5yr", tr115, val115, te115)

    mdir = models_dir(thr)
    rdir = results_dir(thr)

    # ════════════════════════════════════════════════
    #  Step A: t=11.5yr (terminal)
    #  Y = -cost_115 + PSB_REWARD * I(psb=1)
    # ════════════════════════════════════════════════
    print("\n── Step A: t=11.5yr (terminal) ──")
    cost_115 = MAINT_COST[11.5]
    for df_ in [tr115, val115, te115]:
        df_["Y_115"] = -cost_115 + df_["psb"] * PSB_REWARD

    fc115 = [c for c in get_feature_cols(11.5) if c in tr115.columns]
    model_115 = train_qmodel(
        tr115[fc115].values, tr115["Y_115"].values,
        val115[fc115].values, val115["Y_115"].values,
        tag="Q_11.5", psb_tr=tr115["psb"].values,
    )
    save_model(model_115, mdir / "q_model_115.pkl")

    for df_ in [tr115, val115, te115]:
        df_["Q_115_pred"] = model_115.predict(df_[fc115].values)

    # ════════════════════════════════════════════════
    #  Step B: t=7.5yr
    #  Y = -cost_75 + γ * max(Q_115(s'), 0)
    # ════════════════════════════════════════════════
    print("\n── Step B: t=7.5yr ──")
    cost_75 = MAINT_COST[7.5]
    q115_map = {pid: q for df_ in [tr115, val115, te115]
                for pid, q in zip(df_["patent_id"], df_["Q_115_pred"])}

    for df_ in [tr75, val75, te75]:
        q_next = df_["patent_id"].map(q115_map).fillna(0)
        df_["Y_75"] = -cost_75 + GAMMA * np.maximum(q_next, 0)

    fc75 = [c for c in get_feature_cols(7.5) if c in tr75.columns]
    model_75 = train_qmodel(
        tr75[fc75].values, tr75["Y_75"].values,
        val75[fc75].values, val75["Y_75"].values,
        tag="Q_7.5", psb_tr=tr75["psb"].values,
    )
    save_model(model_75, mdir / "q_model_75.pkl")

    for df_ in [tr75, val75, te75]:
        df_["Q_75_pred"] = model_75.predict(df_[fc75].values)

    # ════════════════════════════════════════════════
    #  Step C: t=3.5yr
    #  Y = -cost_35 + γ * max(Q_75(s'), 0)
    # ════════════════════════════════════════════════
    print("\n── Step C: t=3.5yr ──")
    cost_35 = MAINT_COST[3.5]
    q75_map = {pid: q for df_ in [tr75, val75, te75]
               for pid, q in zip(df_["patent_id"], df_["Q_75_pred"])}

    for df_ in [tr35, val35, te35]:
        q_next = df_["patent_id"].map(q75_map).fillna(0)
        df_["Y_35"] = -cost_35 + GAMMA * np.maximum(q_next, 0)

    fc35 = [c for c in get_feature_cols(3.5) if c in tr35.columns]
    model_35 = train_qmodel(
        tr35[fc35].values, tr35["Y_35"].values,
        val35[fc35].values, val35["Y_35"].values,
        tag="Q_3.5", psb_tr=tr35["psb"].values,
    )
    save_model(model_35, mdir / "q_model_35.pkl")

    for df_ in [tr35, val35, te35]:
        df_["Q_35_pred"] = model_35.predict(df_[fc35].values)

    # ════════════════════════════════════════════════
    #  결과 저장 (test set)
    # ════════════════════════════════════════════════
    print("\n── Test Set Evaluation ──")

    # policy: Q > 0 → maintain(1)
    te35["policy_35"]   = (te35["Q_35_pred"]   > 0).astype(int)
    te75["policy_75"]   = (te75["Q_75_pred"]   > 0).astype(int)
    te115["policy_115"] = (te115["Q_115_pred"] > 0).astype(int)

    results = (
        te35[["patent_id", "grant_year", "psb", "B", "t_a",
               "ipc_subclass", "Q_35_pred", "policy_35"]]
        .merge(te75[["patent_id",  "Q_75_pred",  "policy_75"]],  on="patent_id", how="left")
        .merge(te115[["patent_id", "Q_115_pred", "policy_115"]], on="patent_id", how="left")
    )
    results["policy_full_maintain"] = (
        (results["policy_35"] == 1) &
        (results["policy_75"] == 1) &
        (results["policy_115"] == 1)
    ).astype(int)

    out_path = rdir / "test_qvalues.parquet"
    results.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({len(results):,} rows)")

    # 정책별 평가
    total     = len(results)
    psb_total = results["psb"].sum()
    print(f"\n  Test total={total:,}  PSB={psb_total}")

    for col, lbl in [("policy_35",   "3.5yr"),
                     ("policy_75",   "7.5yr"),
                     ("policy_115",  "11.5yr"),
                     ("policy_full_maintain", "full")]:
        sel     = results[col].sum()
        hit     = results[results[col] == 1]["psb"].sum()
        prec    = hit / sel        if sel > 0        else 0
        recall  = hit / psb_total  if psb_total > 0  else 0
        print(f"  [{lbl}] maintain={sel:,}  PSB={hit}  "
              f"prec={prec:.4f}  recall={recall:.4f}")

    # Q_35 랭킹 평가
    print("\n── Q_35 Ranking ──")
    evaluate_ranking(results["psb"], results["Q_35_pred"], label="Q_35")

    print(f"\n[{tag}] Done. ({time.time()-t_start:.1f}s)")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all",
                        help="PSB threshold: 0.001 / 0.005 / 0.010 / all")
    args = parser.parse_args()

    if args.thr == "all":
        thrs = PSB_THRESHOLDS
    else:
        thrs = [float(args.thr)]

    for thr in thrs:
        run_backward_induction(thr)


if __name__ == "__main__":
    main()
