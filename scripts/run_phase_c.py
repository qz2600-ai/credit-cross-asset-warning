from pathlib import Path
import argparse,json,sys,time
import numpy as np,pandas as pd,yaml
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"src"))
from credit_warning.models import *
from credit_warning.splits import purged_chronological_folds
from credit_warning.evaluate import *
from credit_warning.labels import group_stress_episodes

def ep_count(d):
    return len(group_stress_episodes(d,"fold"))

def select_hyperparams(train,kind,features=ALL_FEATURES):
    folds=purged_chronological_folds(train,4,10)
    audit=[]; candidates=([{"C":c} for c in [.1,1.,10.]] if kind=="logistic"
       else [{"max_depth":d,"min_samples_leaf":l} for d in [3,5,8] for l in [20,50,100]])
    vals=[]
    for fi,(tr,va) in enumerate(folds,1):
        if va.stress_label.sum()==0:
            raise RuntimeError(f"Purged fold {fi} has no positive observations; PR AUC unsupported.")
        audit.append({"fold":fi,"training_start":tr.index.min(),"training_end":tr.index.max(),
          "validation_start":va.index.min(),"validation_end":va.index.max(),"training_rows":len(tr),
          "validation_rows":len(va),"training_positives":int(tr.stress_label.sum()),
          "validation_positives":int(va.stress_label.sum()),"training_episodes":ep_count(tr),
          "validation_episodes":ep_count(va),
          "max_training_label_end":tr.label_end_date.max(),
          "purge_trading_rows":10})
    for c in candidates:
        scores=[]
        for tr,va in folds:
            if kind=="logistic":
                m=make_logistic(features,c["C"]); m.fit(tr[features],tr.stress_label)
                p=m.predict_proba(va[features])[:,1]
            else:
                m=make_rf(**c);m.fit(tr[ALL_FEATURES],tr.stress_label);p=m.predict_proba(va[ALL_FEATURES])[:,1]
            scores.append(average_precision_score(va.stress_label,p))
        vals.append((float(np.mean(scores)),c,scores))
    best=max(v[0] for v in vals)
    near=[v for v in vals if best-v[0]<=.01]
    # simpler/more regularized: logistic smallest C; RF shallowest depth then largest leaf.
    chosen=(min(near,key=lambda x:x[1]["C"]) if kind=="logistic"
            else min(near,key=lambda x:(x[1]["max_depth"],-x[1]["min_samples_leaf"])))
    return chosen[1],vals,pd.DataFrame(audit)

def wf_predict(all_pretest,val,params1,params2,params3,block=21):
    rows=[];coefs=[]
    idx=val.index
    for bi,start in enumerate(range(0,len(idx),block)):
        dates=idx[start:start+block]; block_start=dates[0]
        # strictly observable before prediction block
        tr=all_pretest[(all_pretest.index<block_start)&(all_pretest.label_end_date<block_start)]
        for name,features,params in [("m1",HYG_FEATURES,params1),("m2",ALL_FEATURES,params2)]:
            m=make_logistic(features,params["C"]);m.fit(tr[features],tr.stress_label)
            pred=m.predict_proba(val.loc[dates,features])[:,1]
            if name=="m1": p1=pred
            else: p2=pred
            coef=m.named_steps["model"].coef_[0]
            for f,c in zip(features,coef): coefs.append({"model":name,"block_id":bi,"block_start":block_start,"feature":f,"standardized_coefficient":c,"odds_ratio_1sd":np.exp(c)})
        rf=make_rf(**params3);rf.fit(tr[ALL_FEATURES],tr.stress_label);p3=rf.predict_proba(val.loc[dates,ALL_FEATURES])[:,1]
        for j,d in enumerate(dates):
            rows.append({"date":d,"block_id":bi,"training_start":tr.index.min(),"training_end":tr.index.max(),
              "max_training_label_end":tr.label_end_date.max(),"training_rows":len(tr),
              "m1_probability":p1[j],"m2_probability":p2[j],"m3_probability":p3[j]})
    return pd.DataFrame(rows).set_index("date"),pd.DataFrame(coefs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);a=ap.parse_args()
    cfg=yaml.safe_load((ROOT/a.config).read_text()); tabs=ROOT/"outputs/tables"; reps=ROOT/"outputs/reports"
    vend=pd.Timestamp(cfg["periods"]["validation_end"])
    # Hard Phase-C isolation: copy only CSV records whose date is <= validation_end.
    # Post-validation outcome fields are never parsed or loaded by Phase C.
    import csv, io
    source=tabs/"phase_b_features_labels.csv"
    with source.open(newline="") as fh:
        reader=csv.reader(fh); header=next(reader); kept=[header]
        for row in reader:
            if pd.Timestamp(row[0]) > vend:
                break
            kept.append(row)
    buf=io.StringIO()
    writer=csv.writer(buf); writer.writerows(kept); buf.seek(0)
    d=pd.read_csv(buf,parse_dates=["date","future_trough_date_10d","label_end_date"],index_col="date")
    train=d[(d.index<=pd.Timestamp(cfg["periods"]["train_end"]))&(d.label_end_date<=pd.Timestamp(cfg["periods"]["train_end"]))].copy()
    val=d[(d.index>=pd.Timestamp(cfg["periods"]["validation_start"]))&(d.index<=vend)&(d.label_end_date<=vend)].copy()
    # M0 fixed training threshold.
    m0_threshold=float(train["hyg_return_5d"].quantile(.10))
    p1,cv1,audit=select_hyperparams(train,"logistic",HYG_FEATURES)
    # Same logistic hyperparameter selection mechanics for M2, independently.
    # Evaluate candidates on its actual 12-feature representation.
    folds=purged_chronological_folds(train,4,10)
    vals2=[]
    for C in [.1,1.,10.]:
        sc=[]
        for tr,va in folds:
            m=make_logistic(ALL_FEATURES,C);m.fit(tr[ALL_FEATURES],tr.stress_label);sc.append(average_precision_score(va.stress_label,m.predict_proba(va[ALL_FEATURES])[:,1]))
        vals2.append((float(np.mean(sc)),{"C":C},sc))
    best=max(x[0] for x in vals2); near=[x for x in vals2 if best-x[0]<=.01]; p2=min(near,key=lambda x:x[1]["C"])[1]
    p3,cv3,_=select_hyperparams(train,"rf")
    pred,coefs=wf_predict(d,val,p1,p2,p3)
    out=val[["future_trough_return_10d","future_trough_date_10d","label_end_date","stress_label"]].join(pred)
    out["m0_score"]=-val["hyg_return_5d"]; out["m0_alert"]=(val["hyg_return_5d"]<=m0_threshold).astype(int)
    thresholds={"m0_hyg_return_threshold":m0_threshold}
    for m in ["m1","m2","m3"]:
        th,alerts=threshold_for_alert_rate(out[f"{m}_probability"],cfg["evaluation"]["target_alert_rate"])
        thresholds[f"{m}_probability_threshold"]=th;out[f"{m}_alert"]=alerts.astype(int)
    met=[]
    for m in ["m1","m2","m3"]:
        x=metrics(out.stress_label.astype(int),out[f"{m}_probability"],out[f"{m}_alert"],out.future_trough_return_10d,out.index);x["model"]=m;met.append(x)
    # M0 score is not probability: PR/ROC + threshold metrics; Brier/calibration not applicable.
    from sklearn.metrics import roc_auc_score,precision_score,recall_score,f1_score
    y=out.stress_label.astype(int); al=out.m0_alert
    met.append({"model":"m0","pr_auc":average_precision_score(y,out.m0_score),"roc_auc":roc_auc_score(y,out.m0_score),"brier":np.nan,
      "precision":precision_score(y,al,zero_division=0),"recall":recall_score(y,al,zero_division=0),"f1":f1_score(y,al,zero_division=0),
      "alert_rate":al.mean(),"false_alerts_per_year":int(((al==1)&(y==0)).sum())/((out.index.max()-out.index.min()).days/365.25),
      "mean_future_trough_after_alert":out.loc[al.astype(bool),"future_trough_return_10d"].mean(),
      "median_future_trough_after_alert":out.loc[al.astype(bool),"future_trough_return_10d"].median()})
    metricsdf=pd.DataFrame(met)
    # episode detection and lead time
    ep=group_stress_episodes(out,"validation"); erows=[]
    for _,e in ep.iterrows():
        g=out.loc[e.episode_start:e.episode_end]
        for m in ["m0","m1","m2","m3"]:
            hits=g.index[g[f"{m}_alert"].eq(1)]
            first=hits.min() if len(hits) else pd.NaT
            trough=e.associated_trough_date
            lead=(len(out.loc[first:trough])-1 if pd.notna(first) and trough in out.index else np.nan)
            erows.append({"model":m,"episode_start":e.episode_start,"episode_end":e.episode_end,"detected":bool(len(hits)),
                          "first_alert":first,"associated_trough_date":trough,"lead_trading_rows":lead})
    epdet=pd.DataFrame(erows)
    det=epdet.groupby("model").detected.mean().rename("stress_episode_detection_rate")
    metricsdf=metricsdf.merge(det,on="model",how="left")
    # calibration
    cal=pd.concat([calibration_deciles(out,f"{m}_probability").assign(model=m) for m in ["m1","m2","m3"]],ignore_index=True)
    # Validation permutation importance for selected M2/M3 configurations.
    # Models are fit on initial training only; validation is used only for permutation scoring.
    imps=[]
    for mname,params,kind in [("m2",p2,"log"),("m3",p3,"rf")]:
        model=make_logistic(ALL_FEATURES,params["C"]) if kind=="log" else make_rf(**params)
        model.fit(train[ALL_FEATURES],train.stress_label)
        pi=permutation_importance(model,val[ALL_FEATURES],val.stress_label,scoring="average_precision",
                                  n_repeats=5,random_state=42)
        for f,mean,std in zip(ALL_FEATURES,pi.importances_mean,pi.importances_std):
            imps.append({"model":mname,"feature":f,"importance_mean":mean,"importance_std":std})
    imp=pd.DataFrame(imps)
    # candidate audit
    cand=[]
    for model,vals in [("m1",cv1),("m2",vals2),("m3",cv3)]:
        for mean,param,scores in vals:cand.append({"model":model,**param,"mean_pr_auc":mean,**{f"fold_{i+1}_pr_auc":v for i,v in enumerate(scores)}})
    out.to_csv(tabs/"validation_predictions.csv",index_label="date");metricsdf.to_csv(tabs/"model_metrics_validation.csv",index=False)
    audit.to_csv(tabs/"purged_fold_audit.csv",index=False);pd.DataFrame(cand).to_csv(tabs/"hyperparameter_selection.csv",index=False)
    pd.DataFrame([thresholds]).to_csv(tabs/"locked_validation_thresholds.csv",index=False);coefs.to_csv(tabs/"logistic_coefficients_validation.csv",index=False)
    imp.to_csv(tabs/"permutation_importance_validation.csv",index=False);cal.to_csv(tabs/"calibration_validation.csv",index=False)
    epdet.to_csv(tabs/"validation_episode_alerts.csv",index=False)
    selected={"m1":p1,"m2":p2,"m3":p3,"thresholds":thresholds,"test_accessed":False}
    (reps/"phase_c_selected_config.json").write_text(json.dumps(selected,indent=2))
    comp=metricsdf.set_index("model").loc[["m1","m2"],["pr_auc","brier","precision","recall"]]
    report=["# Phase C Validation-Only Report","",
      "**No test predictions, test metrics, test feature importance, or test-period model results were generated. Phase C input was truncated at 2020-12-31 before model work.**","",
      "Hyperparameters were selected only with 10-trading-row-purged chronological folds inside initial training. Validation predictions are genuine 21-trading-day walk-forward predictions; each block uses only rows whose label_end_date precedes the block start.","",
      "## Selected hyperparameters and locked thresholds","",f"```json\n{json.dumps(selected,indent=2)}\n```","",
      "## Purged-fold audit","",audit.to_markdown(index=False),"",
      "## Validation metrics","",metricsdf.to_markdown(index=False),"",
      "## M2 versus M1","",comp.to_markdown(),"",
      "Given only eight validation stress episodes and their concentration in a few events, no claim of superiority is made from a small numerical difference alone. The locked configuration records the validation evidence without using test results.","",
      "## Episode alerts and first-alert lead time","",epdet.to_markdown(index=False)]
    (reps/"phase_c_validation_report.md").write_text("\n".join(report))
    print("PHASE C COMPLETE — validation only. No test-period model results generated.")
if __name__=="__main__":main()
