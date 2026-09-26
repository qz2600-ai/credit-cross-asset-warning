from pathlib import Path
import argparse, json, platform, sys, time
from importlib.metadata import version, PackageNotFoundError
import pandas as pd, yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from credit_warning.data import load_cached_raw, audit_and_align, data_fingerprint
from credit_warning.features import make_features, feature_dictionary, FEATURE_COLUMNS
from credit_warning.labels import future_trough, initial_stress_cutoff, add_stress_label
from credit_warning.splits import chronological_samples

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",required=True); args=ap.parse_args()
    t0=time.time(); cfg=yaml.safe_load(Path(args.config).read_text())
    raw=ROOT/"data"/"raw"; tabs=ROOT/"outputs"/"tables"; reps=ROOT/"outputs"/"reports"
    tabs.mkdir(parents=True,exist_ok=True); reps.mkdir(parents=True,exist_ok=True)
    etfs,vix=load_cached_raw(raw,cfg["assets"]["etfs"],cfg["assets"]["vix_series"])
    audit,aligned,common_pct,represented,criteria=audit_and_align(etfs,vix,cfg["assets"]["etfs"],cfg["assets"]["vix_series"],
        cfg["project"]["analysis_start"],cfg["project"]["analysis_end"])
    if not all(criteria.values()): raise SystemExit("Phase A gate no longer passes; stop.")
    counts=[("aligned_phase_a_rows",len(aligned))]
    feats=make_features(aligned,cfg["assets"]["vix_series"]); counts.append(("feature_rows_before_complete_case_drop",len(feats)))
    missing=feats.isna().sum().rename("missing_count").to_frame()
    complete=feats.dropna(); counts.append(("rows_after_feature_complete_case_drop",len(complete)))
    labels=future_trough(aligned["HYG"],cfg["target"]["horizon_days"]); counts.append(("label_rows_constructed",len(labels)))
    frame=complete.join(labels,how="left")
    counts.append(("complete_feature_rows_with_observable_10d_label",int(frame["label_end_date"].notna().sum())))
    cutoff=initial_stress_cutoff(frame,cfg["periods"]["train_end"],cfg["target"]["training_quantile"])
    frame=add_stress_label(frame,cutoff)
    samples=chronological_samples(frame,cfg["periods"])
    for n,d in samples.items(): counts.append((f"{n}_rows_after_boundary_exclusion",len(d)))
    fd=feature_dictionary(); fd.to_csv(tabs/"feature_dictionary.csv",index=False)
    missing.to_csv(tabs/"feature_missing_counts.csv")
    stats=feats[FEATURE_COLUMNS].describe(percentiles=[.01,.05,.25,.5,.75,.95,.99]).T
    stats.to_csv(tabs/"feature_summary_statistics.csv")
    pd.DataFrame(counts,columns=["stage","row_count"]).to_csv(tabs/"phase_b_row_counts.csv",index=False)
    rates=pd.DataFrame([{"period":n,"rows":len(d),"stress_labels":int(d["stress_label"].sum()),
                         "stress_label_rate":float(d["stress_label"].mean()),
                         "start":str(d.index.min().date()),"end":str(d.index.max().date())} for n,d in samples.items()])
    rates.to_csv(tabs/"stress_label_rates.csv",index=False)
    # Audit-only derived data; no model calculations.
    frame.to_csv(tabs/"phase_b_features_labels.csv",index_label="date")
    boundary=[]
    specs={"training":(None,cfg["periods"]["train_end"]),
           "validation":(cfg["periods"]["validation_start"],cfg["periods"]["validation_end"]),
           "test":(cfg["periods"]["test_start"],cfg["periods"]["test_end"])}
    for n,(start,end) in specs.items():
        base=(frame.index<=pd.Timestamp(end))
        if start: base &= frame.index>=pd.Timestamp(start)
        crossing=base & frame["label_end_date"].notna() & (frame["label_end_date"]>pd.Timestamp(end))
        boundary.append({"period":n,"boundary_crossing_rows_excluded":int(crossing.sum())})
    pd.DataFrame(boundary).to_csv(tabs/"boundary_exclusions.csv",index=False)
    lines=["# Phase B Feature and Label Audit","",
      "**Phase A gate: PASS. Phase B completed. No models were constructed, fit, tuned, or evaluated.**","",
      f"Configured analysis window remains {cfg['project']['analysis_start']} through {cfg['project']['analysis_end']}.",
      f"Usable complete-feature date range: {complete.index.min().date()} through {complete.index.max().date()}.",
      f"Numerical initial-training 10th-percentile stress cutoff: `{cutoff:.12f}`.","",
      "## Row counts","",pd.DataFrame(counts,columns=["stage","row_count"]).to_markdown(index=False),"",
      "## Missing values by feature","",missing.reset_index(names="feature").to_markdown(index=False),"",
      "## Feature summary statistics","",stats.reset_index(names="feature").to_markdown(index=False),"",
      "## Stress-label rates by chronological sample","",rates.to_markdown(index=False),"",
      "## Label-horizon boundary exclusions","",pd.DataFrame(boundary).to_markdown(index=False),"",
      "Boundary-crossing labels were excluded from each chronological sample by requiring `label_end_date <= period end`.",
      "The stress cutoff was computed only from complete-feature initial-training rows with an observable 10-day label whose `label_end_date` is on or before 2016-12-31.",
      "No feature was removed or selected based on outcomes, and no model performance was inspected."]
    (reps/"phase_b_audit.md").write_text("\n".join(lines))
    meta={"phase":"B","phase_a_gate_passed":True,"models_constructed":False,"stress_cutoff":cutoff,
          "usable_feature_start":str(complete.index.min().date()),"usable_feature_end":str(complete.index.max().date()),
          "sample_rows":{n:len(d) for n,d in samples.items()},"runtime_seconds":time.time()-t0,
          "python":platform.python_version(),"configuration":cfg}
    (reps/"phase_b_run_metadata.json").write_text(json.dumps(meta,indent=2))
    print("PHASE B COMPLETE. No models constructed. Stopping after Phase B.")
if __name__=="__main__": main()
