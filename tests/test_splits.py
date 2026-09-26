import sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from credit_warning.splits import chronological_samples

def test_chronological_split_and_label_boundary_exclusion():
    idx=pd.to_datetime(["2016-12-29","2016-12-30","2017-01-02","2020-12-30","2020-12-31","2021-01-04"])
    f=pd.DataFrame({"label_end_date":pd.to_datetime(["2016-12-30","2017-01-05","2017-01-16","2020-12-31","2021-01-08","2021-01-15"])},index=idx)
    periods={"train_end":"2016-12-31","validation_start":"2017-01-01","validation_end":"2020-12-31","test_start":"2021-01-01","test_end":"2021-12-31"}
    s=chronological_samples(f,periods)
    assert list(s["training"].index)==[idx[0]]
    assert list(s["validation"].index)==[idx[2],idx[3]]
    assert list(s["test"].index)==[idx[5]]
    for name,end in [("training","2016-12-31"),("validation","2020-12-31"),("test","2021-12-31")]:
        assert (s[name]["label_end_date"]<=pd.Timestamp(end)).all()
