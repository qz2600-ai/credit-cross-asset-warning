import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from credit_warning.labels import future_trough, initial_stress_cutoff, add_stress_label

def test_future_trough_uses_only_following_10_rows_and_dates():
    idx=pd.bdate_range("2020-01-01",periods=15)
    p=pd.Series([100,99,98,97,96,95,94,93,92,91,90,1,1,1,1],index=idx,dtype=float)
    out=future_trough(p,10)
    assert np.isclose(out.iloc[0]["future_trough_return_10d"],-.10)
    assert out.iloc[0]["future_trough_date_10d"]==idx[10]
    assert out.iloc[0]["label_end_date"]==idx[10]
    assert pd.isna(out.iloc[5]["future_trough_return_10d"])

def test_initial_cutoff_only_eligible_initial_training():
    idx=pd.bdate_range("2020-01-01",periods=6)
    f=pd.DataFrame({"future_trough_return_10d":[-.1,-.2,-.3,-.9,-.8,-.7],
                    "label_end_date":[idx[1],idx[2],idx[3],idx[4],idx[5],idx[5]]},index=idx)
    cutoff=initial_stress_cutoff(f,str(idx[3].date()),.10)
    assert np.isclose(cutoff,np.quantile([-.1,-.2,-.3],.10))

def test_stress_label_numeric_cutoff():
    idx=pd.bdate_range("2020-01-01",periods=3)
    f=pd.DataFrame({"future_trough_return_10d":[-.2,-.1,np.nan]},index=idx)
    o=add_stress_label(f,-.15)
    assert o["stress_label"].iloc[0]==1 and o["stress_label"].iloc[1]==0 and pd.isna(o["stress_label"].iloc[2])


def test_group_stress_episodes_consecutive_positive_rows():
    from credit_warning.labels import group_stress_episodes
    idx=pd.bdate_range("2020-01-01",periods=7)
    f=pd.DataFrame({
      "stress_label":[1,1,0,1,1,1,0],
      "future_trough_return_10d":[-.2,-.3,-.01,-.15,-.4,-.25,-.02],
      "future_trough_date_10d":pd.to_datetime(["2020-01-20","2020-01-21","2020-01-22","2020-01-23","2020-01-24","2020-01-27","2020-01-28"])
    },index=idx)
    e=group_stress_episodes(f,"training")
    assert len(e)==2
    assert list(e["positive_label_days"])==[2,3]
    assert e.iloc[0]["episode_start"]==idx[0] and e.iloc[0]["episode_end"]==idx[1]
    assert np.isclose(e.iloc[1]["worst_future_trough_return"],-.4)
    assert e.iloc[1]["associated_trough_date"]==pd.Timestamp("2020-01-24")

def test_group_stress_episodes_does_not_bridge_ineligible_row_gap():
    from credit_warning.labels import group_stress_episodes
    idx=pd.to_datetime(["2020-01-02","2020-01-03","2020-01-07"])
    f=pd.DataFrame({"stress_label":[1,1,1],
      "future_trough_return_10d":[-.2,-.3,-.4],
      "future_trough_date_10d":pd.to_datetime(["2020-01-10","2020-01-13","2020-01-14"])},index=idx)
    # All rows supplied are eligible and adjacent in the eligible sample, so they form one episode.
    e=group_stress_episodes(f,"test")
    assert len(e)==1 and e.iloc[0]["positive_label_days"]==3
