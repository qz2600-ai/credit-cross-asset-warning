#!/usr/bin/env python
import argparse,json,sys,tempfile
from pathlib import Path
import numpy as np,pandas as pd,matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score,precision_score,recall_score
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/"src"))
from credit_warning.models import HYG_FEATURES,ALL_FEATURES,make_logistic
from credit_warning.labels import group_stress_episodes
from credit_warning.evaluate import paired_moving_block_pr_bootstrap
FILES=["rolling_5y_test_scores.csv","rolling_5y_robustness.csv","bootstrap_block_sensitivity.csv","equal_alert_budget_diagnostic.csv","stress_by_year.csv","m1_m2_evidence_by_stage.csv","training_feature_correlation.csv","feature_drift.csv","coefficient_sign_diagnostics.csv"]
def ep(f,a):
 e=group_stress_episodes(f,"test");ds=[];ls=[]
 for _,x in e.iterrows():
  h=f.index[(f.index>=x.episode_start)&(f.index<=x.episode_end)&f[a].eq(1)];ds.append(bool(len(h)))
  if len(h):ls.append(int(((f.index>=h.min())&(f.index<=x.associated_trough_date)).sum()-1))
 return sum(ds),np.mean(ds),np.median(ls) if ls else np.nan
def save(fig,d,n):
 for x in ["png","svg"]:fig.savefig(d/f"{n}.{x}",dpi=180 if x=="png" else None,bbox_inches="tight")
 plt.close(fig)
def gen(base):
 base=Path(base);t=base/"tables";g=base/"figures";q=base/"reports";[x.mkdir(parents=True,exist_ok=True) for x in [t,g,q]]
 d=pd.read_csv(R/"outputs/tables/phase_b_features_labels.csv",parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date")
 p=pd.read_csv(R/"outputs/tables/test_predictions.csv",parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date");test=d.loc[p.index];th={"m1":.440215,"m2":.531285};rows=[]
 for b in range((len(test)+20)//21):
  ix=test.index[b*21:(b+1)*21]
  if not len(ix):continue
  lo=ix[0]-pd.DateOffset(years=5);tr=d[(d.index>=lo)&(d.index<ix[0])&(d.label_end_date<ix[0])]
  for m,fs in [("m1",HYG_FEATURES),("m2",ALL_FEATURES)]:
   z=make_logistic(fs,.1,42);z.fit(tr[fs],tr.stress_label)
   for dt,s in zip(ix,z.predict_proba(test.loc[ix,fs])[:,1]):rows.append([dt,m,s,b,tr.index.min(),tr.index.max(),len(tr)])
 roll=pd.DataFrame(rows,columns=["date","model","score","block_id","training_start","training_end","training_rows"]);roll.to_csv(t/FILES[0],index=False)
 w=roll.pivot(index="date",columns="model",values="score").join(test[["stress_label","future_trough_return_10d","future_trough_date_10d"]]);rb=[]
 for m in ["m1","m2"]:
  al=(w[m]>=th[m]).astype(int);x=w.copy();x[m+"_alert"]=al;ed,er,lead=ep(x,m+"_alert")
  rb.append([m,average_precision_score(w.stress_label,w[m]),th[m],al.mean(),precision_score(w.stress_label,al),recall_score(w.stress_label,al),ed,er,int(((al==1)&(w.stress_label==0)).sum()),lead])
 rb=pd.DataFrame(rb,columns=["model","pr_auc","threshold","alert_rate","precision","recall","episodes_detected","episode_detection_rate","false_alert_days","median_lead_time"]);rb.to_csv(t/FILES[1],index=False)
 bs=[]
 for L in [10,20,40,60]:
  b,disc=paired_moving_block_pr_bootstrap(p.stress_label,p.m1_probability,p.m2_probability,1000,L,42);a=average_precision_score(p.stress_label,p.m1_probability);c=average_precision_score(p.stress_label,p.m2_probability)
  for n,v in [("m1_pr_auc",a),("m2_pr_auc",c),("m2_minus_m1_pr_auc",c-a)]:bs.append([L,n,v,b[n].quantile(.025),b[n].quantile(.975),len(b),disc])
 bs=pd.DataFrame(bs,columns=["block_days","metric","estimate","ci_2_5","ci_97_5","valid_replicates","discarded_one_class"]);bs.to_csv(t/FILES[2],index=False)
 eq=[];n=int(np.ceil(.1*len(p)))
 for m in ["m1","m2","m3"]:
  s=p[m+"_probability"];cut=s.nlargest(n).min();al=(s>=cut).astype(int);x=p.copy();x[m+"_a"]=al;ed,er,lead=ep(x,m+"_a")
  eq.append([m,n,int(al.sum()),cut,precision_score(p.stress_label,al),recall_score(p.stress_label,al),ed,er,int(((al==1)&(p.stress_label==0)).sum()),lead])
 eq=pd.DataFrame(eq,columns=["model","alert_budget_days","actual_alert_days","retrospective_score_cutoff","precision","recall","episodes_detected","episode_detection_rate","false_alert_days","median_lead_time"]);eq.to_csv(t/FILES[3],index=False)
 es=group_stress_episodes(p,"test");yr=pd.DataFrame({"positive_days":p.groupby(p.index.year).stress_label.sum().astype(int)});yr["episodes"]=pd.Series(es.groupby(es.episode_start.dt.year).size());yr=yr.fillna(0).astype(int);yr.index.name="year";yr.to_csv(t/FILES[4])
 cv=pd.read_csv(R/"outputs/tables/hyperparameter_selection.csv");vm=pd.read_csv(R/"outputs/tables/model_metrics_validation.csv").set_index("model");tm=pd.read_csv(R/"outputs/tables/final_test_metrics.csv").set_index("model")
 ev=pd.DataFrame([["initial_training_purged_cv",cv[(cv.model=="m1")&(cv.C==.1)].mean_pr_auc.iloc[0],cv[(cv.model=="m2")&(cv.C==.1)].mean_pr_auc.iloc[0]],["walk_forward_validation_2017_2020",vm.loc["m1","pr_auc"],vm.loc["m2","pr_auc"]],["locked_test_2021_2026",tm.loc["m1","pr_auc"],tm.loc["m2","pr_auc"]]],columns=["stage","m1_pr_auc","m2_pr_auc"]);ev["m2_minus_m1_pr_auc"]=ev.m2_pr_auc-ev.m1_pr_auc;ev.to_csv(t/FILES[5],index=False)
 tr=d[(d.index<=pd.Timestamp("2016-12-31"))&(d.label_end_date<=pd.Timestamp("2016-12-31"))];va=d[(d.index>="2017-01-01")&(d.label_end_date<="2020-12-31")];co=tr[ALL_FEATURES].corr();co.to_csv(t/FILES[6],index_label="feature");dr=[]
 for f in ALL_FEATURES:
  mu,sd=tr[f].mean(),tr[f].std(ddof=0)
  for n,x in [("training",tr),("validation",va),("test",test)]:dr.append([f,n,x[f].mean(),x[f].std(ddof=0),(x[f].mean()-mu)/sd,x[f].std(ddof=0)/sd])
 dr=pd.DataFrame(dr,columns=["feature","split","mean","std","standardized_mean_drift_vs_training","std_ratio_vs_training"]);dr.to_csv(t/FILES[7],index=False)
 fd=pd.read_csv(R/"outputs/tables/feature_dictionary.csv").set_index("feature");cf=pd.read_csv(R/"outputs/tables/walkforward_test_logistic_coefficients.csv");sr=[]
 for (m,f),x in cf.groupby(["model","feature"]):
  sv=np.sign(x.standardized_coefficient.to_numpy());med=float(np.median(x.standardized_coefficient));ex=fd.loc[f,"expected_sign"];sr.append([m,f,ex,med,int(np.sum(sv[1:]!=sv[:-1])),bool((ex=="positive" and med<0)or(ex=="negative" and med>0))])
 pd.DataFrame(sr,columns=["model","feature","expected_sign","median_coefficient","sign_reversals_across_refits","median_sign_economically_unexpected"]).to_csv(t/FILES[8],index=False)
 fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(co,vmin=-1,vmax=1,cmap="coolwarm");ax.set_xticks(range(12));ax.set_xticklabels(co.columns,rotation=70,ha="right",fontsize=7);ax.set_yticks(range(12));ax.set_yticklabels(co.index,fontsize=7);ax.set_title("Training-Only Feature Correlations");fig.colorbar(im,ax=ax);save(fig,g,"training_feature_correlation")
 pv=dr[dr.split!="training"].pivot(index="feature",columns="split",values="standardized_mean_drift_vs_training");fig,ax=plt.subplots(figsize=(10,6));xx=np.arange(12);ax.bar(xx-.19,pv.validation,.38,label="Validation");ax.bar(xx+.19,pv.test,.38,label="Test");ax.set_xticks(xx);ax.set_xticklabels(pv.index,rotation=70,ha="right",fontsize=7);ax.set_ylabel("Mean shift / training SD");ax.legend();save(fig,g,"feature_drift")
 ca=pd.read_csv(R/"outputs/tables/calibration_by_model.csv");fig,ax=plt.subplots()
 for m in ["m1","m2","m3"]:
  x=ca[ca.model==m];ax.plot(x.mean_predicted_risk,x.observed_stress_rate,marker="o",label=m.upper())
 ax.plot([0,1],[0,1],"--");ax.set_xlabel("Mean model score");ax.set_ylabel("Observed stress-label rate");ax.legend();save(fig,g,"phase_e_score_reliability")
 fig,ax=plt.subplots(figsize=(12,4));ax.plot(p.index,p.m2_probability,label="M2 score")
 for _,e in es.iterrows():ax.axvspan(e.episode_start,e.episode_end,alpha=.12)
 ax.set_ylabel("Model score");ax.legend();save(fig,g,"phase_e_m2_score_timeline")
 fig,ax=plt.subplots(figsize=(12,4));ax.plot(p.index,p.m1_probability,label="M1");ax.plot(p.index,p.m2_probability,label="M2");ax.set_ylabel("Model score");ax.legend();save(fig,g,"phase_e_m1_m2_scores")
 txt="# Phase E Robustness and Stability Addendum\n\nThe locked Phase D primary conclusion remains **No reliable improvement**. These analyses are secondary and do not redefine the primary result.\n\n## Five-year rolling window\n\n"+rb.to_markdown(index=False)+"\n\nThe rolling-window PR-AUC comparison remains unfavorable to M2. This does not replace the expanding-window primary test.\n\n## Bootstrap block sensitivity\n\n"+bs.to_markdown(index=False)+"\n\n## Retrospective equal-alert-budget diagnostic\n\nThe highest-risk 10% of locked test dates (143 dates) were selected retrospectively. These are not deployable thresholds.\n\n"+eq.to_markdown(index=False)+"\n\n## Regime concentration\n\n45 of 57 positive test days and 14 of 17 stress episodes occur in 2022.\n\n"+yr.reset_index().to_markdown(index=False)+"\n\n## Evidence by stage\n\n"+ev.to_markdown(index=False)+"\n\nM2 underperformed M1 in initial-training purged cross-validation, outperformed it during 2017–2020 validation, and underperformed it in the locked 2021–2026 test. Feature redundancy and regime instability are plausible explanations for the lack of generalization, not proven causal explanations.\n";(q/"phase_e_robustness_report.md").write_text(txt)
def verify():
 with tempfile.TemporaryDirectory() as z:
  gen(z);errs={}
  for n in FILES:
   a=pd.read_csv(R/"outputs/tables"/n);b=pd.read_csv(Path(z)/"tables"/n);mx=0.
   for c in a.columns:
    if pd.api.types.is_numeric_dtype(a[c]):
     x=a[c].to_numpy(float)-b[c].to_numpy(float);mx=max(mx,float(np.nanmax(np.abs(x))) if np.any(~np.isnan(x)) else 0.)
    else:assert a[c].astype(str).tolist()==b[c].astype(str).tolist()
   errs[n]=mx
  ok=max(errs.values())<1e-10;print(json.dumps({"verified":ok,"max_numerical_discrepancy":max(errs.values()),"by_file":errs},indent=2));return 0 if ok else 1
if __name__=="__main__":
 a=argparse.ArgumentParser();a.add_argument("--verify",action="store_true");a.add_argument("--output-root",default=str(R/"outputs/phase_e_regenerated"));x=a.parse_args()
 if x.verify: raise SystemExit(verify())
 gen(x.output_root);print("Phase E regenerated")
