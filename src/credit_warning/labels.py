from __future__ import annotations
import numpy as np
import pandas as pd

def future_trough(prices: pd.Series, horizon: int=10) -> pd.DataFrame:
    p = prices.astype(float)
    n=len(p)
    vals=np.full(n,np.nan); dates=np.full(n,np.datetime64("NaT"),dtype="datetime64[ns]")
    end_dates=np.full(n,np.datetime64("NaT"),dtype="datetime64[ns]")
    arr=p.to_numpy(); idx=p.index.to_numpy(dtype="datetime64[ns]")
    for i in range(n-horizon):
        future=arr[i+1:i+horizon+1]/arr[i]-1.0
        j=int(np.argmin(future))
        vals[i]=future[j]; dates[i]=idx[i+1+j]; end_dates[i]=idx[i+horizon]
    return pd.DataFrame({"future_trough_return_10d":vals,
                         "future_trough_date_10d":pd.to_datetime(dates),
                         "label_end_date":pd.to_datetime(end_dates)},index=p.index)

def initial_stress_cutoff(frame: pd.DataFrame, train_end: str, quantile: float=0.10) -> float:
    eligible=frame.loc[(frame.index<=pd.Timestamp(train_end)) &
                       frame["future_trough_return_10d"].notna() &
                       (frame["label_end_date"]<=pd.Timestamp(train_end)),
                       "future_trough_return_10d"]
    if eligible.empty: raise ValueError("No eligible initial-training labels.")
    return float(eligible.quantile(quantile))

def add_stress_label(frame: pd.DataFrame, cutoff: float) -> pd.DataFrame:
    out=frame.copy()
    out["stress_label"]=np.where(out["future_trough_return_10d"].notna(),
                                 (out["future_trough_return_10d"]<=cutoff).astype(int),np.nan)
    return out


def group_stress_episodes(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    """Group consecutive eligible positive-label rows by trading-row adjacency."""
    pos = frame.loc[frame["stress_label"].eq(1)].copy()
    cols=["split","episode_start","episode_end","positive_label_days",
          "worst_future_trough_return","associated_trough_date"]
    if pos.empty:
        return pd.DataFrame(columns=cols)
    eligible_positions = pd.Series(np.arange(len(frame)), index=frame.index)
    ppos = eligible_positions.loc[pos.index]
    group_id = ppos.diff().ne(1).cumsum()
    rows=[]
    for _, g in pos.groupby(group_id):
        worst_idx=g["future_trough_return_10d"].idxmin()
        rows.append({
            "split":split,
            "episode_start":g.index.min(),
            "episode_end":g.index.max(),
            "positive_label_days":len(g),
            "worst_future_trough_return":float(g["future_trough_return_10d"].min()),
            "associated_trough_date":g.loc[worst_idx,"future_trough_date_10d"],
        })
    return pd.DataFrame(rows,columns=cols)
