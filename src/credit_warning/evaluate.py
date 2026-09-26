from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss,precision_score,recall_score,f1_score

def threshold_for_alert_rate(scores: pd.Series, target=.10):
    n=len(scores); k=max(1,int(round(target*n)))
    ordered=scores.sort_values(ascending=False,kind="mergesort")
    threshold=float(ordered.iloc[k-1])
    alerts=scores>=threshold
    return threshold,alerts

def metrics(y,score,alert,trough,dates):
    years=max((dates.max()-dates.min()).days/365.25,1/365.25)
    false=int(((alert==1)&(y==0)).sum())
    alerted=trough[alert.astype(bool)]
    return {
      "pr_auc":average_precision_score(y,score),"roc_auc":roc_auc_score(y,score),
      "brier":brier_score_loss(y,score),"precision":precision_score(y,alert,zero_division=0),
      "recall":recall_score(y,alert,zero_division=0),"f1":f1_score(y,alert,zero_division=0),
      "alert_rate":float(np.mean(alert)),"false_alerts_per_year":false/years,
      "mean_future_trough_after_alert":float(alerted.mean()),"median_future_trough_after_alert":float(alerted.median())}

def calibration_deciles(frame,model):
    x=frame[[model,"stress_label"]].dropna().copy()
    x["risk_decile"]=pd.qcut(x[model].rank(method="first"),10,labels=False)+1
    return x.groupby("risk_decile").agg(n=("stress_label","size"),mean_predicted_risk=(model,"mean"),
      observed_stress_rate=("stress_label","mean")).reset_index()


def paired_moving_block_pr_bootstrap(y, p1, p2, repetitions=1000, block_days=20, seed=42):
    rng=np.random.default_rng(seed); y=np.asarray(y);p1=np.asarray(p1);p2=np.asarray(p2);n=len(y)
    rows=[];discarded=0
    for b in range(repetitions):
        ix=[]
        while len(ix)<n:
            s=int(rng.integers(0,n-block_days+1));ix.extend(range(s,s+block_days))
        ix=np.asarray(ix[:n]);yb=y[ix]
        if np.unique(yb).size<2: discarded+=1;continue
        a1=average_precision_score(yb,p1[ix]);a2=average_precision_score(yb,p2[ix])
        rows.append((b,a1,a2,a2-a1))
    return pd.DataFrame(rows,columns=["replication","m1_pr_auc","m2_pr_auc","m2_minus_m1_pr_auc"]),discarded
