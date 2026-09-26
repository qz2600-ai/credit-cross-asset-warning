from __future__ import annotations
import pandas as pd
import numpy as np

def chronological_samples(frame: pd.DataFrame, periods: dict):
    # Boundary rule: the complete label horizon must remain within its own period.
    specs={
      "training": (None, periods["train_end"]),
      "validation": (periods["validation_start"], periods["validation_end"]),
      "test": (periods["test_start"], periods["test_end"]),
    }
    out={}
    for name,(start,end) in specs.items():
        mask=frame.index<=pd.Timestamp(end)
        if start is not None: mask &= frame.index>=pd.Timestamp(start)
        mask &= frame["label_end_date"].notna() & (frame["label_end_date"]<=pd.Timestamp(end))
        out[name]=frame.loc[mask].copy()
    return out


def purged_chronological_folds(frame: pd.DataFrame, n_folds: int=4, purge_rows: int=10):
    """Expanding chronological folds: four equal-ish validation blocks in the latter training sample."""
    n=len(frame)
    # Initial 40% train, remaining 60% split into n_folds chronological validation blocks.
    start=int(np.floor(n*0.40))
    edges=np.linspace(start,n,n_folds+1,dtype=int)
    folds=[]
    for i in range(n_folds):
        vs,ve=edges[i],edges[i+1]
        val=frame.iloc[vs:ve]
        train_end=max(0,vs-purge_rows)
        train=frame.iloc[:train_end]
        folds.append((train,val))
    return folds
