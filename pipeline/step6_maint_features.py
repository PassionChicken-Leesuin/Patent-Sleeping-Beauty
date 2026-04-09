"""
Step 6: Maintenance Fee Features

USPTO maintenance fee events (M1xx / M2xx) 를 각 특허의
실제 납부 이력으로 변환한다. 결과는 두 가지 방식으로 활용:

  (a) 후속 시점 피처  — 의사결정 시점 t 에서 "과거 납부 이력"을
      알 수 있으므로, t=7.5yr 모델은 paid_3_5 를, t=11.5yr 모델은
      paid_3_5 / paid_7_5 를 피처로 사용.  3.5yr 모델에서는 사용 ×.

  (b) 생존 필터  — BI 상태공간을 실제로 살아있는 특허로 제한할 때
      사용.  t=7.5yr 샘플은 paid_3_5=1 인 특허만,
      t=11.5yr 샘플은 paid_3_5=1 AND paid_7_5=1 인 특허만.

분류 규칙 (event_date - grant_date 기준):
    paid_3_5  :  3.0 ~ 5.0  yr 사이에 M-event 가 하나라도 존재
    paid_7_5  :  6.5 ~ 9.0  yr 사이에 M-event 가 하나라도 존재
    paid_11_5 : 10.5 ~ 13.5 yr 사이에 M-event 가 하나라도 존재

USPTO 실제 납부 window 는 각 시점의 "anniversary 6 months before /
6 months after" + 6-month grace period 이므로 위 범위는 모든 정상
납부 + 늦은 납부(late surcharge)까지 포섭한다.

Output:
    features/maint_features.parquet
        columns: patent_id, paid_3_5, paid_7_5, paid_11_5, lapsed_age
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
import pandas as pd

from config import (
    PATENTS_80S_FILE, MAINT_EVENTS_FILE, FEATURES_DIR, PATENT_TERM_YRS,
)


WIN_3_5  = (3.0, 5.0)
WIN_7_5  = (6.5, 9.0)
WIN_11_5 = (10.5, 13.5)


def main():
    t_start = time.time()

    print("Loading data...")
    pat = pd.read_parquet(PATENTS_80S_FILE, columns=["patent_id", "grant_date"])
    mf  = pd.read_parquet(MAINT_EVENTS_FILE)
    print(f"  patents   : {len(pat):,}")
    print(f"  maint evs : {len(mf):,}")

    # M-code 만 사용 (M1xx/M2xx = maintenance fee payment 관련)
    mf = mf[mf["event_code"].str.startswith(("M1", "M2"), na=False)].copy()
    print(f"  M-events  : {len(mf):,}")

    mf = mf.merge(pat, on="patent_id", how="inner")
    age_days   = (mf["event_date"] - mf["grant_date"]).dt.days
    mf["age"]  = age_days.astype(float) / 365.25

    def in_win(a, lo, hi):
        return (a >= lo) & (a <= hi)

    mf["w_3_5"]  = in_win(mf["age"], *WIN_3_5)
    mf["w_7_5"]  = in_win(mf["age"], *WIN_7_5)
    mf["w_11_5"] = in_win(mf["age"], *WIN_11_5)

    grp = mf.groupby("patent_id").agg(
        paid_3_5  = ("w_3_5",  "any"),
        paid_7_5  = ("w_7_5",  "any"),
        paid_11_5 = ("w_11_5", "any"),
    ).reset_index()

    # 전 특허로 확장 (M-event 자체가 없는 특허 → 전부 0)
    out = pat[["patent_id"]].merge(grp, on="patent_id", how="left")
    for c in ["paid_3_5", "paid_7_5", "paid_11_5"]:
        out[c] = out[c].fillna(False).astype(int)

    # lapsed_age: 3.5yr 에 lapsed → 3.5, 7.5 lapsed → 7.5, 11.5 lapsed → 11.5,
    #             전 기간 생존 → PATENT_TERM_YRS (17)
    lapsed_age = np.full(len(out), float(PATENT_TERM_YRS))
    lapsed_age[out["paid_11_5"] == 0] = 11.5
    lapsed_age[out["paid_7_5"]  == 0] = 7.5
    lapsed_age[out["paid_3_5"]  == 0] = 3.5
    out["lapsed_age"] = lapsed_age

    paid35  = int(out["paid_3_5"].sum())
    paid75  = int(out["paid_7_5"].sum())
    paid115 = int(out["paid_11_5"].sum())
    n       = len(out)
    print("\nMaint fee survival rates:")
    print(f"  paid_3_5   : {paid35:,} / {n:,}  ({paid35/n*100:5.2f}%)")
    print(f"  paid_7_5   : {paid75:,} / {n:,}  ({paid75/n*100:5.2f}%)")
    print(f"  paid_11_5  : {paid115:,} / {n:,}  ({paid115/n*100:5.2f}%)")

    out_path = FEATURES_DIR / "maint_features.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  Saved: {out_path.name}  ({n:,} rows)")
    print(f"Step 6 done. ({time.time()-t_start:.1f}s)")


if __name__ == "__main__":
    main()
