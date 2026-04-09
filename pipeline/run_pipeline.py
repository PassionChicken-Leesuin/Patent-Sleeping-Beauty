"""
데이터 파이프라인 순차 실행 (Step 1~5 + B_11yr)

사용법:
  python run_pipeline.py               # 전체 실행
  python run_pipeline.py --step 1      # Step 1만
  python run_pipeline.py --step 1-3   # Step 1~3
  python run_pipeline.py --step 4 5   # Step 4, 5
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).parent

STEPS = [
    (1, "step1_patents_80s.py",       "1980-1989 특허 기본정보 추출"),
    (2, "step2_citations.py",          "Citation 데이터 처리"),
    (3, "step3_static_features.py",    "IPC / Assignee / Inventor 추출"),
    (4, "step4_beauty_coefficient.py", "Beauty Coefficient & PSB 라벨 (3가지 threshold)"),
    (5, "step5_build_features.py",     "3.5/7.5/11.5yr Feature 테이블 구축"),
    (6, "compute_b11yr.py",            "11.5yr 기준 Beauty Coefficient 계산"),
]


def run_step(step_id: int, script: str, desc: str) -> float:
    print(f"\n{'='*60}")
    print(f"  Step {step_id}: {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=str(HERE))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_id} failed (exit {result.returncode})")
        sys.exit(1)
    print(f"\n[OK] Step {step_id} done  ({elapsed:.1f}s)")
    return elapsed


def parse_steps(args) -> list:
    if not args.step:
        return [s[0] for s in STEPS]
    result = []
    for token in args.step:
        if "-" in token:
            s, e = token.split("-")
            result.extend(range(int(s), int(e) + 1))
        else:
            result.append(int(token))
    return sorted(set(result))


def main():
    parser = argparse.ArgumentParser(description="PSB Data Pipeline")
    parser.add_argument("--step", nargs="+",
                        help="Steps to run, e.g. --step 1 2 3  or  --step 1-5")
    args = parser.parse_args()
    steps_to_run = parse_steps(args)

    print(f"Running steps: {steps_to_run}")
    t_total = time.time()

    for step_id, script, desc in STEPS:
        if step_id in steps_to_run:
            run_step(step_id, script, desc)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete. Total: {time.time()-t_total:.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
