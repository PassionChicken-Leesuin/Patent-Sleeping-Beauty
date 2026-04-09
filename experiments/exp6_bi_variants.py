"""
Exp 6: Backward Induction - Signal Decay Variants

Q-value 신호 소멸 문제를 해결하는 4가지 변형 비교:

  [clip]            원본: Y = -cost + γ * max(Q_next, 0)
  [no_clip]         클리핑 제거: Y = -cost + γ * Q_next
  [softplus]        소프트 클리핑: Y = -cost + γ * softplus(Q_next)
  [reward_shaping]  중간 reward: Y = -cost + α*R*I(psb=1) + γ * max(Q_next, 0)

각 변형에 대해 Q_35 랭킹 AP를 비교 → 신호 보존 효과 측정.

사용법:
  python exp6_bi_variants.py                  # thr=0.001, 전체 변형
  python exp6_bi_variants.py --thr 0.005
  python exp6_bi_variants.py --thr all
  python exp6_bi_variants.py --variant no_clip --thr all
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
    load_and_merge, split, print_split_stats,
    get_feature_cols, train_qmodel, evaluate_ranking, save_model,
)


# ── 변형별 Bellman Backup 함수 ────────────────────────────────────────────────

def softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    """
    Softplus: log(1 + exp(beta*x)) / beta
    x > 0  → x에 수렴 (clipping과 동일)
    x < 0  → 0으로 수렴하지만 완전히 0이 되지 않음 (신호 보존)
    beta가 클수록 hard max와 유사, 작을수록 신호가 더 많이 전파
    """
    # overflow 방지: exp(beta*x) 가 너무 커지면 log(exp(bx)) = bx 로 처리
    return np.where(
        beta * x > 30,
        x,
        (1.0 / beta) * np.log1p(np.exp(beta * x))
    )


def bellman_clip(q_next: np.ndarray, cost: float) -> np.ndarray:
    """원본: max(Q_next, 0) clipping"""
    return -cost + GAMMA * np.maximum(q_next, 0)


def bellman_no_clip(q_next: np.ndarray, cost: float) -> np.ndarray:
    """클리핑 없음: Q_next 그대로 전파"""
    return -cost + GAMMA * q_next


def bellman_softplus(q_next: np.ndarray, cost: float,
                     beta: float = 1.0) -> np.ndarray:
    """Softplus 클리핑"""
    return -cost + GAMMA * softplus(q_next, beta=beta)


def bellman_reward_shaping(q_next: np.ndarray, cost: float,
                            psb: np.ndarray, alpha: float) -> np.ndarray:
    """
    중간 reward shaping:
    Y = -cost + alpha * PSB_REWARD * I(psb=1) + γ * max(Q_next, 0)
    alpha: 중간 단계 reward 비율 (0~1)
    """
    return (-cost
            + alpha * PSB_REWARD * psb
            + GAMMA * np.maximum(q_next, 0))


# ── 단일 변형에 대한 BI 실행 ─────────────────────────────────────────────────

def run_variant(thr: float, variant: str):
    """
    variant: 'clip' | 'no_clip' | 'softplus' | 'reward_shaping'
    """
    tag = f"thr={thr} | {variant}"
    print(f"\n{'='*60}")
    print(f"  Exp 6: BI Variant [{tag}]")
    print(f"{'='*60}")
    t_start = time.time()

    # ── 데이터 로드 ───────────────────────────────────
    labels = pd.read_parquet(labels_file(thr))
    df35  = load_and_merge(3.5,  labels)
    df75  = load_and_merge(7.5,  labels)
    df115 = load_and_merge(11.5, labels)

    tr35,  val35,  te35  = split(df35)
    tr75,  val75,  te75  = split(df75)
    tr115, val115, te115 = split(df115)

    print("\nData split:")
    print_split_stats("3.5yr",  tr35,  val35,  te35)

    mdir = models_dir(thr)
    rdir = results_dir(thr)

    # ═══════════════════════════════════════════════════
    #  Step A: t=11.5yr (terminal) — 모든 변형에서 동일
    #  Y_115 = -cost_115 + PSB_REWARD * I(psb=1)
    # ═══════════════════════════════════════════════════
    print(f"\n-- Step A: t=11.5yr (terminal) --")
    cost_115 = MAINT_COST[11.5]
    for df_ in [tr115, val115, te115]:
        df_["Y_115"] = -cost_115 + df_["psb"].values * PSB_REWARD

    fc115 = [c for c in get_feature_cols(11.5) if c in tr115.columns]
    model_115 = train_qmodel(
        tr115[fc115].values, tr115["Y_115"].values,
        val115[fc115].values, val115["Y_115"].values,
        tag=f"Q_115[{variant}]", psb_tr=tr115["psb"].values,
    )
    save_model(model_115, mdir / f"q_model_115_{variant}.pkl")

    for df_ in [tr115, val115, te115]:
        df_["Q_115_pred"] = model_115.predict(df_[fc115].values)

    # ═══════════════════════════════════════════════════
    #  Step B: t=7.5yr — 변형별 Bellman backup
    # ═══════════════════════════════════════════════════
    print(f"\n-- Step B: t=7.5yr [{variant}] --")
    cost_75 = MAINT_COST[7.5]
    q115_map = {pid: q for df_ in [tr115, val115, te115]
                for pid, q in zip(df_["patent_id"], df_["Q_115_pred"])}

    for df_ in [tr75, val75, te75]:
        q_next = df_["patent_id"].map(q115_map).fillna(0).values
        if variant == "clip":
            df_["Y_75"] = bellman_clip(q_next, cost_75)
        elif variant == "no_clip":
            df_["Y_75"] = bellman_no_clip(q_next, cost_75)
        elif variant == "softplus":
            df_["Y_75"] = bellman_softplus(q_next, cost_75, beta=1.0)
        elif variant == "reward_shaping":
            df_["Y_75"] = bellman_reward_shaping(
                q_next, cost_75, psb=df_["psb"].values, alpha=0.3
            )

    # 타겟 분포 진단
    y75_all = np.concatenate([tr75["Y_75"].values, val75["Y_75"].values])
    print(f"  Y_75 stats: mean={y75_all.mean():.4f}  "
          f"std={y75_all.std():.4f}  "
          f"min={y75_all.min():.4f}  "
          f"max={y75_all.max():.4f}  "
          f"unique={len(np.unique(np.round(y75_all, 4)))}")

    fc75 = [c for c in get_feature_cols(7.5) if c in tr75.columns]
    model_75 = train_qmodel(
        tr75[fc75].values, tr75["Y_75"].values,
        val75[fc75].values, val75["Y_75"].values,
        tag=f"Q_75[{variant}]", psb_tr=tr75["psb"].values,
    )
    save_model(model_75, mdir / f"q_model_75_{variant}.pkl")

    for df_ in [tr75, val75, te75]:
        df_["Q_75_pred"] = model_75.predict(df_[fc75].values)

    # ═══════════════════════════════════════════════════
    #  Step C: t=3.5yr — 변형별 Bellman backup
    # ═══════════════════════════════════════════════════
    print(f"\n-- Step C: t=3.5yr [{variant}] --")
    cost_35 = MAINT_COST[3.5]
    q75_map = {pid: q for df_ in [tr75, val75, te75]
               for pid, q in zip(df_["patent_id"], df_["Q_75_pred"])}

    for df_ in [tr35, val35, te35]:
        q_next = df_["patent_id"].map(q75_map).fillna(0).values
        if variant == "clip":
            df_["Y_35"] = bellman_clip(q_next, cost_35)
        elif variant == "no_clip":
            df_["Y_35"] = bellman_no_clip(q_next, cost_35)
        elif variant == "softplus":
            df_["Y_35"] = bellman_softplus(q_next, cost_35, beta=1.0)
        elif variant == "reward_shaping":
            df_["Y_35"] = bellman_reward_shaping(
                q_next, cost_35, psb=df_["psb"].values, alpha=0.1
            )

    # 타겟 분포 진단
    y35_all = np.concatenate([tr35["Y_35"].values, val35["Y_35"].values])
    print(f"  Y_35 stats: mean={y35_all.mean():.4f}  "
          f"std={y35_all.std():.4f}  "
          f"min={y35_all.min():.4f}  "
          f"max={y35_all.max():.4f}  "
          f"unique={len(np.unique(np.round(y35_all, 4)))}")

    fc35 = [c for c in get_feature_cols(3.5) if c in tr35.columns]
    model_35 = train_qmodel(
        tr35[fc35].values, tr35["Y_35"].values,
        val35[fc35].values, val35["Y_35"].values,
        tag=f"Q_35[{variant}]", psb_tr=tr35["psb"].values,
    )
    save_model(model_35, mdir / f"q_model_35_{variant}.pkl")

    for df_ in [tr35, val35, te35]:
        df_["Q_35_pred"] = model_35.predict(df_[fc35].values)

    # ═══════════════════════════════════════════════════
    #  결과 저장 및 평가
    # ═══════════════════════════════════════════════════
    print(f"\n-- Test Set Evaluation [{variant}] --")
    results = (
        te35[["patent_id", "grant_year", "psb", "B", "t_a",
               "ipc_subclass", "Q_35_pred"]]
        .merge(te75[["patent_id",  "Q_75_pred"]],  on="patent_id", how="left")
        .merge(te115[["patent_id", "Q_115_pred"]], on="patent_id", how="left")
    )
    results["variant"] = variant

    out_path = rdir / f"test_qvalues_{variant}.parquet"
    results.to_parquet(out_path, index=False)
    print(f"  Saved: {out_path.name}  ({len(results):,} rows)")

    # Q_35 랭킹 평가 (핵심)
    print(f"\n-- Q_35 Ranking [{variant}] --")
    ap_result = evaluate_ranking(
        results["psb"], results["Q_35_pred"], label=f"Q_35[{variant}]"
    )

    print(f"\n[{tag}] Done. ({time.time()-t_start:.1f}s)")
    return ap_result["average_precision"]


# ── 전체 변형 비교 ────────────────────────────────────────────────────────────

def run_all_variants(thr: float, variants: list):
    print(f"\n{'#'*60}")
    print(f"  Exp 6: All Variants  [thr={thr}]")
    print(f"{'#'*60}")

    summary = {}
    for v in variants:
        ap = run_variant(thr, v)
        summary[v] = ap

    # 비교 테이블
    print(f"\n{'='*50}")
    print(f"  SUMMARY: Q_35 AP by Variant  [thr={thr}]")
    print(f"{'='*50}")
    baseline = summary.get("clip", None)
    for v, ap in summary.items():
        delta = f" ({ap-baseline:+.4f} vs clip)" if baseline is not None and v != "clip" else ""
        marker = " <-- BEST" if ap == max(summary.values()) else ""
        print(f"  {v:<20} AP = {ap:.4f}{delta}{marker}")
    print(f"{'='*50}")

    return summary


# ── main ──────────────────────────────────────────────────────────────────────

VARIANTS = ["clip", "no_clip", "softplus", "reward_shaping"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", default="all",
                        help="PSB threshold: 0.001 / 0.005 / 0.010 / all")
    parser.add_argument("--variant", default="all",
                        help="Variant: clip / no_clip / softplus / reward_shaping / all")
    args = parser.parse_args()

    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]
    variants = VARIANTS if args.variant == "all" else [args.variant]

    all_results = {}
    for thr in thrs:
        all_results[thr] = run_all_variants(thr, variants)

    # 전체 총정리
    if len(thrs) > 1 and len(variants) > 1:
        print(f"\n{'#'*60}")
        print("  FINAL SUMMARY: Q_35 AP")
        print(f"  {'Variant':<20}", end="")
        for thr in thrs:
            print(f"  thr={thr:.3f}", end="")
        print()
        print(f"  {'-'*55}")
        for v in variants:
            print(f"  {v:<20}", end="")
            for thr in thrs:
                ap = all_results[thr].get(v, float("nan"))
                print(f"  {ap:.4f}  ", end="")
            print()
        print(f"{'#'*60}")


if __name__ == "__main__":
    main()
