"""
Exp 3: PSB Reward 민감도 분석

PSB_REWARDS = [4, 8, 12, 16, 20, 24] 각각에 대해
전체 Backward Induction 재실행 → PSB 예측 + 경제성 평가

사용법:
  python exp3_reward_sensitivity.py
  python exp3_reward_sensitivity.py --thr 0.005
  python exp3_reward_sensitivity.py --thr all
"""

import sys
import argparse
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from config import (
    GAMMA, MAINT_COST, PSB_THRESHOLDS,
    labels_file, models_dir, results_dir,
)
from utils import (
    load_and_merge, split, encode_ipc_splits, get_feature_cols,
    train_qmodel, save_model,
)

PSB_REWARDS = [4.0, 8.0, 12.0, 16.0, 20.0, 24.0]


def run_bi_for_reward(reward: float, tr35, val35, te35,
                      tr75, val75, te75, tr115, val115, te115, mdir: Path):
    """단일 reward 값에 대해 Backward Induction 실행"""
    # Step A
    cost_115 = MAINT_COST[11.5]
    for d in [tr115, val115, te115]:
        d["Y_115"] = -cost_115 + d["psb"] * reward
    fc115 = [c for c in get_feature_cols(11.5) if c in tr115.columns]
    m115  = train_qmodel(tr115[fc115].values, tr115["Y_115"].values,
                         val115[fc115].values, val115["Y_115"].values,
                         tag=f"Q_11.5|r={reward}", psb_tr=tr115["psb"].values)
    for d in [tr115, val115, te115]:
        d["Q_115_pred"] = m115.predict(d[fc115].values)

    # Step B
    cost_75  = MAINT_COST[7.5]
    q115_map = {pid: q for d in [tr115, val115, te115]
                for pid, q in zip(d["patent_id"], d["Q_115_pred"])}
    for d in [tr75, val75, te75]:
        d["Y_75"] = -cost_75 + GAMMA * np.maximum(d["patent_id"].map(q115_map).fillna(0), 0)
    fc75 = [c for c in get_feature_cols(7.5) if c in tr75.columns]
    m75  = train_qmodel(tr75[fc75].values, tr75["Y_75"].values,
                        val75[fc75].values, val75["Y_75"].values,
                        tag=f"Q_7.5|r={reward}", psb_tr=tr75["psb"].values)
    for d in [tr75, val75, te75]:
        d["Q_75_pred"] = m75.predict(d[fc75].values)

    # Step C
    cost_35 = MAINT_COST[3.5]
    q75_map = {pid: q for d in [tr75, val75, te75]
               for pid, q in zip(d["patent_id"], d["Q_75_pred"])}
    for d in [tr35, val35, te35]:
        d["Y_35"] = -cost_35 + GAMMA * np.maximum(d["patent_id"].map(q75_map).fillna(0), 0)
    fc35 = [c for c in get_feature_cols(3.5) if c in tr35.columns]
    m35  = train_qmodel(tr35[fc35].values, tr35["Y_35"].values,
                        val35[fc35].values, val35["Y_35"].values,
                        tag=f"Q_3.5|r={reward}", psb_tr=tr35["psb"].values)
    te35["Q_35_pred"] = m35.predict(te35[fc35].values)

    # Policy & merge
    te35["policy_35"]   = (te35["Q_35_pred"]   > 0).astype(int)
    te75["policy_75"]   = (te75["Q_75_pred"]   > 0).astype(int)
    te115["policy_115"] = (te115["Q_115_pred"] > 0).astype(int)

    res = (
        te35[["patent_id", "psb", "Q_35_pred", "policy_35"]]
        .merge(te75[["patent_id",  "Q_75_pred",  "policy_75"]],  on="patent_id", how="left")
        .merge(te115[["patent_id", "Q_115_pred", "policy_115"]], on="patent_id", how="left")
    )
    res["policy_full_maintain"] = (
        (res["policy_35"] == 1) & (res["policy_75"] == 1) & (res["policy_115"] == 1)
    ).astype(int)

    # 모델 저장
    r_tag = str(reward).replace(".", "p")
    save_model(m35,  mdir / f"q_model_35_reward_{r_tag}.pkl")
    save_model(m75,  mdir / f"q_model_75_reward_{r_tag}.pkl")
    save_model(m115, mdir / f"q_model_115_reward_{r_tag}.pkl")

    return res


def compute_metrics(res: pd.DataFrame, reward: float) -> dict:
    y  = res["psb"]
    sc = res["policy_full_maintain"].astype(float)
    ap = average_precision_score(y, sc)

    metrics = {"psb_reward": reward, "average_precision": ap}
    for k in [100, 200, 500, 1000]:
        k_ = min(k, len(res))
        top = res.sort_values("policy_full_maintain", ascending=False).head(k_)
        metrics[f"precision@{k}"] = top["psb"].sum() / k_

    # 경제성
    maintained = res[res["policy_full_maintain"] == 1]
    psb_hits   = maintained["psb"].sum()
    total_cost = len(maintained) * sum(MAINT_COST.values())
    total_ben  = psb_hits * reward
    metrics.update({
        "total_maintained": len(maintained),
        "psb_hits":         psb_hits,
        "total_cost":       round(total_cost, 2),
        "total_benefit":    round(total_ben, 2),
        "net_benefit":      round(total_ben - total_cost, 2),
        "efficiency_ratio": round(total_ben / total_cost, 4) if total_cost > 0 else 0.0,
    })
    return metrics


def run_reward_sensitivity(thr: float):
    tag = f"thr={thr}"
    print(f"\n{'='*60}")
    print(f"  Exp 3: Reward Sensitivity  [{tag}]")
    print(f"{'='*60}")
    t_start = time.time()

    labels = pd.read_parquet(labels_file(thr))
    df35   = load_and_merge(3.5,  labels)
    df75   = load_and_merge(7.5,  labels)
    df115  = load_and_merge(11.5, labels)

    mdir = models_dir(thr)
    rdir = results_dir(thr)

    summary   = []
    all_parts = []

    for reward in PSB_REWARDS:
        print(f"\n── reward={reward} ──")
        tr35,  val35,  te35  = split(df35.copy())
        tr75,  val75,  te75  = split(df75.copy())
        tr115, val115, te115 = split(df115.copy())
        tr35,  val35,  te35  = encode_ipc_splits(tr35,  val35,  te35)
        tr75,  val75,  te75  = encode_ipc_splits(tr75,  val75,  te75)
        tr115, val115, te115 = encode_ipc_splits(tr115, val115, te115)

        res = run_bi_for_reward(
            reward,
            tr35, val35, te35,
            tr75, val75, te75,
            tr115, val115, te115,
            mdir,
        )
        m = compute_metrics(res, reward)
        summary.append(m)
        print(f"  AP={m['average_precision']:.4f}  "
              f"maintained={m['total_maintained']:,}  "
              f"PSB_hits={m['psb_hits']}  "
              f"net_benefit={m['net_benefit']:,.1f}")

        res["psb_reward"] = reward
        all_parts.append(res)

    # 저장
    pd.concat(all_parts, ignore_index=True).to_parquet(
        rdir / "reward_sensitivity.parquet", index=False)
    sum_df = pd.DataFrame(summary)
    sum_df.to_csv(rdir / "reward_sensitivity_summary.csv", index=False)

    print("\n── Summary ──")
    print(sum_df[["psb_reward", "average_precision", "total_maintained",
                  "psb_hits", "net_benefit", "efficiency_ratio"]].to_string(index=False))
    print(f"\n[{tag}] Done. ({time.time()-t_start:.1f}s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all")
    args = parser.parse_args()
    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]
    for thr in thrs:
        run_reward_sensitivity(thr)


if __name__ == "__main__":
    main()
