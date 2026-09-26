#!/usr/bin/env python
from __future__ import annotations
import argparse, hashlib, json, os, platform, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, yaml
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
 precision_score, recall_score, f1_score, precision_recall_curve, roc_curve)
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from credit_warning.models import HYG_FEATURES,ALL_FEATURES,make_logistic,make_rf
from credit_warning.labels import group_stress_episodes
from credit_warning.evaluate import calibration_deciles,paired_moving_block_pr_bootstrap

LOCK={"m0":-0.015438,"m1":0.440215,"m2":0.531285,"m3":0.583690}
SEED=42

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def block_fit(bi,test,df):
    dates=test.index[bi*21:(bi+1)*21]; bs=dates[0]
    tr=df[(df.index<bs)&(df.label_end_date<bs)]
    row={"block_id":bi,"prediction_start":dates[0],"prediction_end":dates[-1],
         "training_start":tr.index.min(),"training_end":tr.index.max(),"training_rows":len(tr),
         "training_positives":int(tr.stress_label.sum()),"most_recent_label_end_date":tr.label_end_date.max(),
         "random_seed":SEED}
    preds=pd.DataFrame(index=dates); coefs=[]; stats={}
    for name,fs in [("m1",HYG_FEATURES),("m2",ALL_FEATURES)]:
        mod=make_logistic(fs,.1,SEED);mod.fit(tr[fs],tr.stress_label)
        preds[f"{name}_probability"]=mod.predict_proba(test.loc[dates,fs])[:,1]
        sc=mod.named_steps["scaler"];stats[name]={"features":fs,"mean":sc.mean_.tolist(),"scale":sc.scale_.tolist()}
        for f,c in zip(fs,mod.named_steps["model"].coef_[0]):
            coefs.append({"model":name,"block_id":bi,"block_start":bs,"feature":f,
                          "standardized_coefficient":float(c),"odds_ratio_1sd":float(np.exp(c))})
    # Keep the locked 500-tree RF; one job per forest because blocks are parallelized.
    rf=make_rf(3,100,SEED);rf.set_params(n_jobs=1);rf.fit(tr[ALL_FEATURES],tr.stress_label)
    preds["m3_probability"]=rf.predict_proba(test.loc[dates,ALL_FEATURES])[:,1]
    preds["block_id"]=bi;preds["m0_score"]=-test.loc[dates,"hyg_return_5d"]
    row["scaler_statistics_json"]=json.dumps(stats)
    return preds.reset_index(names="date"),row,coefs

def episode_table(out,df):
    eps=group_stress_episodes(out,"test"); rows=[]
    for _,e in eps.iterrows():
        r={"episode_start":e.episode_start,"episode_end":e.episode_end,"positive_label_days":e.positive_label_days,
           "worst_future_trough_return":e.worst_future_trough_return,"associated_trough_date":e.associated_trough_date}
        for m in ["m0","m1","m2","m3"]:
            alerts=out.index[out[f"{m}_alert"].eq(1)]
            during=alerts[(alerts>=e.episode_start)&(alerts<=e.episode_end)]
            before=alerts[(alerts<e.episode_start)&(alerts>=e.episode_start-pd.Timedelta(days=30))]
            after=alerts[(alerts>e.episode_end)&(alerts<=e.episode_end+pd.Timedelta(days=30))]
            first=during.min() if len(during) else pd.NaT
            lead=int(((df.index>=first)&(df.index<=e.associated_trough_date)).sum()-1) if pd.notna(first) else np.nan
            r.update({f"{m}_detected":bool(len(during)),f"{m}_first_alert":first,f"{m}_lead_trading_days":lead,
                      f"{m}_alerts_before_30cal_days":len(before),f"{m}_alerts_during":len(during),f"{m}_alerts_after_30cal_days":len(after)})
        rows.append(r)
    return pd.DataFrame(rows)

def metric_table(out,eps):
    y=out.stress_label.astype(int); years=(out.index.max()-out.index.min()).days/365.25; rows=[]
    for m in ["m0","m1","m2","m3"]:
        score=out.m0_score if m=="m0" else out[f"{m}_probability"]; al=out[f"{m}_alert"].astype(int)
        brier=np.nan if m=="m0" else brier_score_loss(y,score)
        leads=eps.loc[eps[f"{m}_detected"],f"{m}_lead_trading_days"]
        rows.append({"pr_auc":average_precision_score(y,score),"roc_auc":roc_auc_score(y,score),"brier":brier,
          "precision":precision_score(y,al,zero_division=0),"recall":recall_score(y,al,zero_division=0),
          "f1":f1_score(y,al,zero_division=0),"alert_rate":al.mean(),
          "false_alerts_per_year":int(((al==1)&(y==0)).sum())/years,
          "mean_future_trough_after_alert":out.loc[al.astype(bool),"future_trough_return_10d"].mean(),
          "median_future_trough_after_alert":out.loc[al.astype(bool),"future_trough_return_10d"].median(),
          "model":m,"alert_days":int(al.sum()),"episodes_detected":int(eps[f"{m}_detected"].sum()),
          "stress_episode_detection_rate":eps[f"{m}_detected"].mean(),
          "median_detected_episode_lead_time":leads.median()})
    return pd.DataFrame(rows)

def figures(out,metrics,cal,coefs,eps,annual,figs):
    figs.mkdir(parents=True,exist_ok=True); y=out.stress_label.astype(int)
    def save(name):
        plt.tight_layout();plt.savefig(figs/f"{name}.png",dpi=180,bbox_inches="tight");plt.savefig(figs/f"{name}.svg",bbox_inches="tight");plt.close()
    for m in ["m1","m2","m3"]:
        p,r,_=precision_recall_curve(y,out[f"{m}_probability"]);plt.plot(r,p,label=f"{m.upper()} AP={average_precision_score(y,out[f'{m}_probability']):.3f}")
    p,r,_=precision_recall_curve(y,out.m0_score);plt.plot(r,p,label=f"M0 AP={average_precision_score(y,out.m0_score):.3f}")
    plt.xlabel("Recall");plt.ylabel("Precision");plt.title("Locked Test Precision–Recall Curves");plt.legend();save("test_precision_recall_curves")
    for m,s in [("m0",out.m0_score),("m1",out.m1_probability),("m2",out.m2_probability),("m3",out.m3_probability)]:
        f,t,_=roc_curve(y,s);plt.plot(f,t,label=f"{m.upper()} AUC={roc_auc_score(y,s):.3f}")
    plt.plot([0,1],[0,1],"--");plt.xlabel("False-positive rate");plt.ylabel("True-positive rate");plt.title("Locked Test ROC Curves");plt.legend();save("test_roc_curves")
    for m in ["m1","m2","m3"]:
        q=cal[cal.model==m];plt.plot(q.mean_predicted_risk,q.observed_stress_rate,marker="o",label=m.upper())
    plt.plot([0,1],[0,1],"--");plt.xlabel("Mean model score");plt.ylabel("Observed stress rate");plt.title("Test Score Reliability by Decile");plt.legend();save("test_calibration_curves")
    plt.plot(out.index,out.m2_probability,label="M2 score")
    for _,e in eps.iterrows():plt.axvspan(e.episode_start,e.episode_end,alpha=.12)
    plt.axhline(LOCK["m2"],linestyle="--",label="Locked threshold");plt.ylabel("Model score");plt.title("M2 Test Score and Stress Episodes");plt.legend();save("m2_warning_probability")
    plt.plot(out.index,out.m1_probability,label="M1");plt.plot(out.index,out.m2_probability,label="M2");plt.ylabel("Model score");plt.title("M1 vs M2 Test Scores");plt.legend();save("m1_vs_m2_probabilities")
    for f,g in coefs[coefs.model=="m2"].groupby("feature"):plt.plot(g.block_start,g.standardized_coefficient,label=f)
    plt.axhline(0,linewidth=.8);plt.ylabel("Standardized coefficient");plt.title("M2 Walk-Forward Coefficient Stability");plt.legend(fontsize=6,ncol=2);save("m2_coefficient_stability")
    x=np.arange(len(eps));w=.35
    plt.bar(x-w/2,eps.m1_lead_trading_days.fillna(0),w,label="M1");plt.bar(x+w/2,eps.m2_lead_trading_days.fillna(0),w,label="M2")
    plt.xlabel("Stress episode");plt.ylabel("First-alert lead (trading days; 0 if undetected)");plt.title("Episode Detection and Lead Time");plt.legend();save("episode_detection_lead_time")
    a1=annual[annual.model=="m1"];a2=annual[annual.model=="m2"]
    plt.plot(a1.year,a1.pr_auc,marker="o",label="M1 PR-AUC");plt.plot(a2.year,a2.pr_auc,marker="o",label="M2 PR-AUC");plt.plot(a1.year,a1.stress_label_rate,marker="s",label="Stress-label prevalence")
    plt.xlabel("Year");plt.title("Annual Test Ranking and Stress Frequency");plt.legend();save("annual_test_performance")

def run(output_root=None,n_jobs=None):
    tabs=(Path(output_root)/"tables" if output_root else ROOT/"outputs/tables"); reps=(Path(output_root)/"reports" if output_root else ROOT/"outputs/reports"); figs=(Path(output_root)/"figures" if output_root else ROOT/"outputs/figures")
    tabs.mkdir(parents=True,exist_ok=True);reps.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(ROOT/"outputs/tables/phase_b_features_labels.csv",parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date")
    test=df[(df.index>=pd.Timestamp("2021-01-01"))&(df.label_end_date<=pd.Timestamp("2026-09-18"))].copy()
    assert len(test)==1424 and test.index.min()==pd.Timestamp("2021-01-04") and test.index.max()==pd.Timestamp("2026-09-03")
    nblocks=(len(test)+20)//21
    jobs=Parallel(n_jobs=n_jobs or min(5,os.cpu_count() or 1),prefer="processes")(delayed(block_fit)(b,test,df) for b in range(nblocks))
    pred=pd.concat([x[0] for x in jobs]).set_index("date").sort_index()
    audit=pd.DataFrame([x[1] for x in jobs]);coefs=pd.DataFrame([r for x in jobs for r in x[2]])
    assert pred.index.equals(test.index) and pred.index.is_unique
    out=test[["future_trough_return_10d","future_trough_date_10d","label_end_date","stress_label"]].join(pred)
    out["m0_alert"]=(test.hyg_return_5d<=LOCK["m0"]).astype(int)
    for m in ["m1","m2","m3"]:out[f"{m}_alert"]=(out[f"{m}_probability"]>=LOCK[m]).astype(int)
    eps=episode_table(out,df);metrics=metric_table(out,eps); mm=metrics.set_index("model")
    comparison=pd.DataFrame([{"comparison":"M2_minus_M1","pr_auc_absolute_difference":mm.loc["m2","pr_auc"]-mm.loc["m1","pr_auc"],
      "pr_auc_relative_difference":mm.loc["m2","pr_auc"]/mm.loc["m1","pr_auc"]-1,"roc_auc_difference":mm.loc["m2","roc_auc"]-mm.loc["m1","roc_auc"],
      "brier_difference":mm.loc["m2","brier"]-mm.loc["m1","brier"],"precision_difference":mm.loc["m2","precision"]-mm.loc["m1","precision"],
      "recall_difference":mm.loc["m2","recall"]-mm.loc["m1","recall"],"episode_detection_difference":mm.loc["m2","stress_episode_detection_rate"]-mm.loc["m1","stress_episode_detection_rate"],
      "median_lead_time_difference":mm.loc["m2","median_detected_episode_lead_time"]-mm.loc["m1","median_detected_episode_lead_time"]}])
    boot,discard=paired_moving_block_pr_bootstrap(out.stress_label,out.m1_probability,out.m2_probability,1000,20,SEED)
    bs=[]
    points={"m1_pr_auc":mm.loc["m1","pr_auc"],"m2_pr_auc":mm.loc["m2","pr_auc"],"m2_minus_m1_pr_auc":comparison.iloc[0].pr_auc_absolute_difference}
    for c in points:bs.append({"metric":c,"estimate":points[c],"ci_2_5":boot[c].quantile(.025),"ci_97_5":boot[c].quantile(.975),"valid_replicates":len(boot),"discarded_one_class":discard})
    bs=pd.DataFrame(bs)
    cal=pd.concat([calibration_deciles(out,f"{m}_probability").assign(model=m) for m in ["m1","m2","m3"]],ignore_index=True)
    stab=coefs.groupby(["model","feature"]).standardized_coefficient.agg(median="median",q25=lambda x:x.quantile(.25),q75=lambda x:x.quantile(.75),sign_consistency=lambda x:max((x>0).mean(),(x<0).mean())).reset_index()
    last=coefs.block_id.max();fc=coefs[coefs.block_id==last][["model","feature","standardized_coefficient","odds_ratio_1sd"]].rename(columns={"standardized_coefficient":"final_refit_coefficient","odds_ratio_1sd":"final_refit_odds_ratio_1sd"});stab=stab.merge(fc,on=["model","feature"])
    annual=[]
    for yr,g in out.groupby(out.index.year):
        for m in ["m0","m1","m2","m3"]:
            s=g.m0_score if m=="m0" else g[f"{m}_probability"]
            annual.append({"year":yr,"model":m,"observations":len(g),"positive_labels":int(g.stress_label.sum()),"stress_label_rate":g.stress_label.mean(),"pr_auc":average_precision_score(g.stress_label,s) if g.stress_label.nunique()>1 else np.nan,"alert_rate":g[f"{m}_alert"].mean()})
    annual=pd.DataFrame(annual)
    for name,obj in [("final_test_metrics.csv",metrics),("model_comparison.csv",comparison),("test_predictions.csv",out),("test_episode_results.csv",eps),("bootstrap_results.csv",boot),("bootstrap_summary.csv",bs),("coefficient_stability.csv",stab),("calibration_by_model.csv",cal),("annual_test_results.csv",annual),("test_refit_audit.csv",audit),("walkforward_test_logistic_coefficients.csv",coefs)]:
        obj.to_csv(tabs/name,index=(name=="test_predictions.csv"),index_label="date" if name=="test_predictions.csv" else None)
    figures(out,metrics,cal,coefs,eps,annual,figs)
    return metrics,comparison,bs

def verify():
    locked=pd.read_csv(ROOT/"outputs/tables/final_test_metrics.csv").set_index("model")
    out=pd.read_csv(ROOT/"outputs/tables/test_predictions.csv",parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date")
    df=pd.read_csv(ROOT/"outputs/tables/phase_b_features_labels.csv",parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date")
    eps=episode_table(out,df);recomputed=metric_table(out,eps).set_index("model")
    numeric=["pr_auc","roc_auc","brier","precision","recall","f1","alert_rate","false_alerts_per_year",
             "mean_future_trough_after_alert","median_future_trough_after_alert","alert_days","episodes_detected",
             "stress_episode_detection_rate","median_detected_episode_lead_time"]
    errors={}
    for m in locked.index:
        errors[m]={}
        for c in numeric:
            a=locked.loc[m,c];b=recomputed.loc[m,c]
            errors[m][c]=0.0 if pd.isna(a) and pd.isna(b) else float(abs(a-b))
    maxerr=max(v for d in errors.values() for v in d.values())
    ok=bool(maxerr<1e-12 and len(out)==1424 and out.index.is_unique)
    print(json.dumps({"verified":ok,"max_absolute_metric_error":maxerr,"metric_absolute_errors":errors},indent=2))
    return 0 if ok else 1

def verify_full():
    """Slow integration check: refit every locked walk-forward model from Phase-B data."""
    df=pd.read_csv(ROOT/"outputs/tables/phase_b_features_labels.csv",parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date")
    test=df[(df.index>=pd.Timestamp("2021-01-01"))&(df.label_end_date<=pd.Timestamp("2026-09-18"))].copy()
    nblocks=(len(test)+20)//21
    jobs=Parallel(n_jobs=16,prefer="threads")(delayed(block_fit)(b,test,df) for b in range(nblocks))
    pred=pd.concat([x[0] for x in jobs]).set_index("date").sort_index()
    out=test[["future_trough_return_10d","future_trough_date_10d","label_end_date","stress_label"]].join(pred)
    out["m0_alert"]=(test.hyg_return_5d<=LOCK["m0"]).astype(int)
    for m in ["m1","m2","m3"]:out[f"{m}_alert"]=(out[f"{m}_probability"]>=LOCK[m]).astype(int)
    eps=episode_table(out,df);metrics=metric_table(out,eps)
    import tempfile
    _tmp=tempfile.TemporaryDirectory();_tmpdir=Path(_tmp.name)
    out.to_csv(_tmpdir/"test_predictions.csv",index_label="date")
    metrics.to_csv(_tmpdir/"final_test_metrics.csv",index=False)
    lockedp=pd.read_csv(ROOT/"outputs/tables/test_predictions.csv",parse_dates=["date"],index_col="date")
    sc=["m0_score","m1_probability","m2_probability","m3_probability"]
    score_err=float(np.nanmax(np.abs(out[sc].to_numpy()-lockedp[sc].to_numpy())))
    lm=pd.read_csv(ROOT/"outputs/tables/final_test_metrics.csv").set_index("model");rm=metrics.set_index("model")
    cs=["pr_auc","roc_auc","brier","precision","recall","f1","alert_rate","false_alerts_per_year","mean_future_trough_after_alert","median_future_trough_after_alert","alert_days","episodes_detected","stress_episode_detection_rate","median_detected_episode_lead_time"]
    es=[]
    for m in lm.index:
        for c in cs:
            a,b=lm.loc[m,c],rm.loc[m,c];es.append(0. if pd.isna(a) and pd.isna(b) else abs(a-b))
    metric_err=float(max(es));ok=bool(score_err<1e-12 and metric_err<1e-12 and len(out)==1424)
    print(json.dumps({"verified_full":ok,"max_score_discrepancy":score_err,"max_metric_discrepancy":metric_err,"test_rows":len(out)},indent=2))
    return 0 if ok else 1

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--output-root");ap.add_argument("--jobs",type=int);ap.add_argument("--verify",action="store_true");ap.add_argument("--verify-full",action="store_true")
    a=ap.parse_args()
    if a.verify:raise SystemExit(verify())
    if a.verify_full:raise SystemExit(verify_full())
    run(a.output_root,a.jobs)
