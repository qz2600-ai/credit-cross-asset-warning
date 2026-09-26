import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from credit_warning.data import audit_and_align

def test_gate_rejects_duplicate_and_nonpositive():
    idx = pd.date_range("2007-04-04", periods=4, freq="B")
    etf = pd.DataFrame({x:[100,101,102,103] for x in ["HYG","LQD","IEF","SPY"]}, index=idx)
    etf.loc[idx[2],"LQD"] = 0
    vix = pd.DataFrame({"VIXCLS":[15,16,17,18]}, index=idx)
    audit, aligned, pct, represented, criteria = audit_and_align(
        etf,vix,["HYG","LQD","IEF","SPY"],"VIXCLS","2007-04-04","2026-09-18")
    assert not criteria["no nonpositive values"]

def test_no_forward_fill_alignment():
    idx = pd.date_range("2007-04-04", periods=4, freq="B")
    etf = pd.DataFrame({x:[100,101,102,103] for x in ["HYG","LQD","IEF","SPY"]}, index=idx)
    etf.loc[idx[1],"IEF"] = np.nan
    vix = pd.DataFrame({"VIXCLS":[15,16,17,18]}, index=idx)
    _, aligned, _, _, _ = audit_and_align(
        etf,vix,["HYG","LQD","IEF","SPY"],"VIXCLS","2007-04-04","2026-09-18")
    assert idx[1] not in aligned.index


def _coverage_fixture():
    idx = pd.bdate_range("2007-04-02", "2007-04-20")
    etf = pd.DataFrame({x: 100.0 + np.arange(len(idx)) for x in ["HYG","LQD","IEF","SPY"]}, index=idx)
    # Configured start predates HYG's first actual observation.
    etf.loc[idx[idx < pd.Timestamp("2007-04-11")], "HYG"] = np.nan
    vix = pd.DataFrame({"VIXCLS": 15.0 + np.arange(len(idx)) / 10.0}, index=idx)
    return idx, etf, vix

def test_comparison_etfs_cover_hyg_actual_observations_passes():
    idx, etf, vix = _coverage_fixture()
    _, _, _, _, criteria = audit_and_align(
        etf, vix, ["HYG","LQD","IEF","SPY"], "VIXCLS", "2007-04-04", "2007-04-20")
    assert criteria["all comparison ETFs cover HYG actual observations"]

def test_comparison_etf_starting_after_hyg_fails():
    idx, etf, vix = _coverage_fixture()
    etf.loc[idx[idx <= pd.Timestamp("2007-04-11")], "LQD"] = np.nan
    _, _, _, _, criteria = audit_and_align(
        etf, vix, ["HYG","LQD","IEF","SPY"], "VIXCLS", "2007-04-04", "2007-04-20")
    assert not criteria["all comparison ETFs cover HYG actual observations"]

def test_comparison_etf_ending_before_hyg_fails():
    idx, etf, vix = _coverage_fixture()
    etf.loc[idx[-1], "IEF"] = np.nan
    _, _, _, _, criteria = audit_and_align(
        etf, vix, ["HYG","LQD","IEF","SPY"], "VIXCLS", "2007-04-04", "2007-04-20")
    assert not criteria["all comparison ETFs cover HYG actual observations"]
