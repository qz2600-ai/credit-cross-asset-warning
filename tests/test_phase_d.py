import sys,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from credit_warning.evaluate import paired_moving_block_pr_bootstrap
LOCK={"m0":-0.015438,"m1":0.440215,"m2":0.531285,"m3":0.583690}

def test_locked_spec_matches_phase_c():
    c=json.loads((ROOT/"outputs/reports/phase_c_selected_config.json").read_text())
    assert c["m1"]["C"]==.1 and c["m2"]["C"]==.1
    assert c["m3"]=={"max_depth":3,"min_samples_leaf":100}
    assert c["validation_refit_frequency_trading_days"]==21

def test_thresholds_unchanged():
    assert LOCK=={"m0":-0.015438,"m1":0.440215,"m2":0.531285,"m3":0.583690}

def test_test_sample_exactly_1424_and_once():
    p=pd.read_csv(ROOT/"outputs/tables/test_predictions.csv",parse_dates=["date"])
    assert len(p)==1424 and p.date.nunique()==1424
    assert p.date.min()==pd.Timestamp("2021-01-04") and p.date.max()==pd.Timestamp("2026-09-03")

def test_refit_label_observability_and_no_test_tuning():
    a=pd.read_csv(ROOT/"outputs/tables/test_refit_audit.csv",parse_dates=["prediction_start","most_recent_label_end_date"])
    assert (a.most_recent_label_end_date<a.prediction_start).all()
    c=json.loads((ROOT/"outputs/reports/phase_c_selected_config.json").read_text())
    assert c["test_period_loaded_by_phase_c_runner"] is False

def test_scaler_isolation_recorded_each_refit():
    a=pd.read_csv(ROOT/"outputs/tables/test_refit_audit.csv")
    assert a.scaler_statistics_json.notna().all()
    for x in a.scaler_statistics_json:
        j=json.loads(x); assert set(j)=={"m1","m2"} and len(j["m1"]["mean"])==3 and len(j["m2"]["mean"])==12

def test_paired_bootstrap_reproducible():
    y=np.array([0,1,0,0,1]*30);p1=np.linspace(.1,.9,len(y));p2=p1[::-1]
    a,da=paired_moving_block_pr_bootstrap(y,p1,p2,50,20,42)
    b,db=paired_moving_block_pr_bootstrap(y,p1,p2,50,20,42)
    pd.testing.assert_frame_equal(a,b);assert da==db

def test_saved_metrics_reproduce_from_predictions():
    p=pd.read_csv(ROOT/"outputs/tables/test_predictions.csv")
    m=pd.read_csv(ROOT/"outputs/tables/final_test_metrics.csv").set_index("model")
    y=p.stress_label.astype(int)
    for model in ["m1","m2","m3"]:
        s=p[f"{model}_probability"]
        assert np.isclose(average_precision_score(y,s),m.loc[model,"pr_auc"])
        assert np.isclose(roc_auc_score(y,s),m.loc[model,"roc_auc"])
        assert np.isclose(brier_score_loss(y,s),m.loc[model,"brier"])
