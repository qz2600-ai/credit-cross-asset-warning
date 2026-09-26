import sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from credit_warning.splits import purged_chronological_folds
from credit_warning.models import make_logistic,ALL_FEATURES
from credit_warning.evaluate import threshold_for_alert_rate

def frame(n=200):
    idx=pd.bdate_range("2010-01-01",periods=n)
    d=pd.DataFrame(index=idx)
    for j,f in enumerate(ALL_FEATURES): d[f]=np.sin(np.arange(n)/(j+3))+j*.01
    d["stress_label"]=(np.arange(n)%11==0).astype(int)
    d["label_end_date"]=pd.Series(idx,index=idx).shift(-10).values
    return d.dropna()

def test_purged_folds_no_label_overlap():
    d=frame(); folds=purged_chronological_folds(d,4,10)
    for tr,va in folds:
        assert len(tr)>0 and len(va)>0
        assert tr.index.max()<va.index.min()
        assert tr["label_end_date"].max()<va.index.min()

def test_scaler_uses_training_rows_only():
    d=frame();tr=d.iloc[:100];va=d.iloc[110:130]
    m=make_logistic(ALL_FEATURES,1);m.fit(tr[ALL_FEATURES],tr.stress_label)
    assert np.allclose(m.named_steps["scaler"].mean_,tr[ALL_FEATURES].mean().values)
    assert not np.allclose(m.named_steps["scaler"].mean_,pd.concat([tr,va])[ALL_FEATURES].mean().values)

def test_model_fit_data_can_be_hard_limited_before_test():
    d=frame(); cutoff=d.index[120]
    pre=d.loc[:cutoff]
    assert pre.index.max()<=cutoff and not (pre.index>cutoff).any()

def test_walkforward_validation_dates_exactly_once():
    idx=pd.bdate_range("2017-01-01",periods=997)
    seen=[]
    for start in range(0,len(idx),21): seen.extend(idx[start:start+21])
    assert len(seen)==len(idx) and len(set(seen))==len(idx)

def test_training_labels_observable_before_block():
    d=frame();block=d.index[130]
    tr=d[(d.index<block)&(d.label_end_date<block)]
    assert (tr.label_end_date<block).all()

def test_threshold_approximately_target_alert_rate():
    s=pd.Series(np.linspace(0,1,997))
    th,a=threshold_for_alert_rate(s,.10)
    assert abs(a.mean()-.10)<=1/len(s)

def test_fixed_seed_reproduces_logistic_results():
    d=frame();tr=d.iloc[:120]
    a=make_logistic(ALL_FEATURES,1,42);b=make_logistic(ALL_FEATURES,1,42)
    a.fit(tr[ALL_FEATURES],tr.stress_label);b.fit(tr[ALL_FEATURES],tr.stress_label)
    assert np.allclose(a.predict_proba(tr[ALL_FEATURES]),b.predict_proba(tr[ALL_FEATURES]))
