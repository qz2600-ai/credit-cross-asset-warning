from pathlib import Path
import argparse, json, platform, sys, time, yaml
import numpy as np
import pandas as pd
from importlib.metadata import version, PackageNotFoundError
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from credit_warning.data import load_cached_raw, audit_and_align, data_fingerprint

def package_versions():
    names = ["pandas","numpy","scikit-learn","scipy","matplotlib","seaborn","PyYAML","yfinance","pandas-datareader","pytest","jupyter"]
    out = {}
    for n in names:
        try: out[n] = version(n)
        except PackageNotFoundError: out[n] = "not installed"
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    t0 = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    raw = ROOT/"data"/"raw"
    etfs, vix = load_cached_raw(raw, cfg["assets"]["etfs"], cfg["assets"]["vix_series"])
    audit, aligned, common_pct, represented, criteria = audit_and_align(
        etfs, vix, cfg["assets"]["etfs"], cfg["assets"]["vix_series"],
        cfg["project"]["analysis_start"], cfg["project"]["analysis_end"])
    out_tables, out_reports = ROOT/"outputs"/"tables", ROOT/"outputs"/"reports"
    out_tables.mkdir(parents=True, exist_ok=True); out_reports.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out_tables/"data_quality.csv", index=False)
    passed = all(criteria.values())
    lines = ["# Phase A Data Quality Report","",
             f"**Gate result: {'PASS' if passed else 'FAIL'}**","",
             f"Configured analysis window: {cfg['project']['analysis_start']} through {cfg['project']['analysis_end']}.",
             f"Common aligned HYG trading dates: {len(aligned)} ({common_pct:.3f}% of HYG dates).",
             "No ETF forward fill or VIX fill was performed; alignment uses an inner join after audit.","",
             "## Gate criteria",""]
    lines += [f"- [{'x' if ok else ' '}] {name}" for name,ok in criteria.items()]
    lines += ["","## Stress-period representation",""] + [f"- {k}: {v}" for k,v in represented.items()]
    vix_series = cfg["assets"]["vix_series"]
    vix_valid = vix[vix_series].dropna().loc[cfg["project"]["analysis_start"]:cfg["project"]["analysis_end"]]
    vix_log_change = np.log(vix_valid / vix_valid.shift(1))
    vix_review = pd.DataFrame({
        "date": vix_log_change.index,
        "vix_close": vix_valid.values,
        "daily_log_change": vix_log_change.values,
        "abs_daily_log_change": vix_log_change.abs().values,
    }).dropna().nlargest(10, "abs_daily_log_change")
    vix_review["daily_pct_change"] = np.expm1(vix_review["daily_log_change"]) * 100.0
    lines += [
        "","## Largest VIX daily changes","",
        "The 20% absolute daily log-return threshold is diagnostic only. "
        "Valid VIX observations are retained as observed; none are removed or winsorized.",
        "",
        vix_review[["date","vix_close","daily_log_change","daily_pct_change"]].to_markdown(index=False),
    ]
    lines += ["","## Series audit","", audit.to_markdown(index=False)]
    (out_reports/"data_quality.md").write_text("\n".join(lines))
    raw_paths = [raw/f"{x}.csv" for x in [*cfg["assets"]["etfs"], cfg["assets"]["vix_series"]]]
    meta = {
        "phase": "A" if not passed else "A_passed_ready_for_phase_B",
        "gate_passed": passed,
        "configuration": cfg,
        "data_fingerprint_sha256": data_fingerprint(raw_paths),
        "python": platform.python_version(),
        "packages": package_versions(),
        "runtime_seconds": time.time()-t0,
        "output_paths": [str(out_reports/"data_quality.md"), str(out_tables/"data_quality.csv")],
        "model_settings": None,
    }
    (out_reports/"run_metadata.json").write_text(json.dumps(meta, indent=2))
    if not passed:
        raise SystemExit("PHASE A GATE FAILED. Per work order, stopping before model construction.")
    print("PHASE A GATE PASSED. Model construction may proceed, but this Phase-A-only runner stops here.")

if __name__ == "__main__":
    main()
