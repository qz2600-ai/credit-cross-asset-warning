import json, subprocess, sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def test_phase_d_saved_prediction_metric_verification():
    """Canonical verifier reconstructs all locked metrics from saved Phase-D scores and Phase-B labels."""
    cp=subprocess.run([sys.executable,str(ROOT/"scripts/run_phase_d.py"),"--verify"],cwd=ROOT,capture_output=True,text=True,timeout=30)
    assert cp.returncode==0, cp.stderr
    payload=json.loads(cp.stdout)
    assert payload["verified"] is True
    assert payload["max_absolute_metric_error"] < 1e-12

def test_phase_d_reproduction_inputs_and_lock_are_present():
    assert (ROOT/"outputs/tables/phase_b_features_labels.csv").exists()
    c=json.loads((ROOT/"outputs/reports/phase_c_selected_config.json").read_text())
    assert c["m1"]=={"C":0.1} and c["m2"]=={"C":0.1}
    assert c["m3"]=={"max_depth":3,"min_samples_leaf":100}
    assert c["validation_refit_frequency_trading_days"]==21

import os,pytest
@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("RUN_SLOW_INTEGRATION")!="1",reason="set RUN_SLOW_INTEGRATION=1")
def test_phase_d_full_walkforward_reproduction():
 cp=subprocess.run([sys.executable,str(ROOT/"scripts/run_phase_d.py"),"--verify-full"],cwd=ROOT,capture_output=True,text=True,timeout=300)
 assert cp.returncode==0,cp.stderr
 x=json.loads(cp.stdout);assert x["verified_full"] and x["max_score_discrepancy"]<1e-12 and x["max_metric_discrepancy"]<1e-12 and x["test_rows"]==1424
