"""
전체 실험 일괄 실행 스크립트

PSB_THRESHOLDS = [0.001, 0.005, 0.010] 각각에 대해
Exp 1~6를 순차 실행.

사용법:
  python run_experiments.py                   # 모든 threshold x 모든 실험
  python run_experiments.py --thr 0.001       # 단일 threshold
  python run_experiments.py --exp 1 2         # exp1, exp2만 실행
  python run_experiments.py --thr 0.005 --exp 1 3
  python run_experiments.py --exp 6           # 신호 소멸 변형 실험만
"""

import sys
import argparse
import time
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PSB_THRESHOLDS

HERE = Path(__file__).parent

EXPERIMENTS = {
    1: ("exp1_backward_induction.py",  "Backward Induction"),
    2: ("exp2_baseline_comparison.py", "Baseline Comparison"),
    3: ("exp3_reward_sensitivity.py",  "Reward Sensitivity"),
    4: ("exp4_weight_sensitivity.py",  "Weight Sensitivity"),
    5: ("exp5_ipc_analysis.py",        "IPC Analysis"),
    6: ("exp6_bi_variants.py",         "BI Signal Decay Variants"),
}


def run(script: str, thr: float) -> bool:
    result = subprocess.run(
        [sys.executable, str(HERE / script), "--thr", str(thr)],
        cwd=str(HERE),
    )
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run PSB experiments")
    parser.add_argument("--thr", default="all",
                        help="Threshold(s): 0.001 / 0.005 / 0.010 / all")
    parser.add_argument("--exp", nargs="+", type=int,
                        default=list(EXPERIMENTS.keys()),
                        help="Experiment numbers to run (e.g. --exp 1 2 3)")
    args = parser.parse_args()

    thrs = PSB_THRESHOLDS if args.thr == "all" else [float(args.thr)]
    exps = args.exp

    total_start = time.time()
    failures = []

    print("=" * 60)
    print("  PSB Experiment Runner")
    print(f"  Thresholds : {thrs}")
    print(f"  Experiments: {exps}")
    print("=" * 60)

    for thr in thrs:
        print(f"\n>> Threshold = {thr}")
        for exp_id in exps:
            script, desc = EXPERIMENTS[exp_id]
            print(f"\n  [{exp_id}] {desc}  (thr={thr})")
            t0 = time.time()
            ok = run(script, thr)
            elapsed = time.time() - t0
            status = "[OK]  " if ok else "[FAIL]"
            print(f"  {status}  ({elapsed:.1f}s)")
            if not ok:
                failures.append((thr, exp_id, desc))

    print(f"\n{'='*60}")
    print(f"  Total elapsed: {time.time()-total_start:.1f}s")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for thr, eid, desc in failures:
            print(f"    thr={thr}  Exp {eid}: {desc}")
    else:
        print("  All experiments completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
