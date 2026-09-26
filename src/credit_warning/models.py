from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

HYG_FEATURES=["hyg_return_5d","hyg_volatility_20d","hyg_negative_share_10d"]
ALL_FEATURES=[
"hyg_return_5d","hyg_volatility_20d","hyg_negative_share_10d","hy_ig_gap_5d","hy_ig_gap_20d",
"hy_treasury_gap_10d","equity_adjusted_hyg_5d","hyg_spy_correlation_20d","hyg_ief_correlation_20d",
"cross_asset_dispersion_5d","vix_zscore_252d","vix_change_5d"]

def make_logistic(features,C,seed=42):
    return Pipeline([("scaler",StandardScaler()),
      ("model",LogisticRegression(C=C,penalty="l2",class_weight="balanced",max_iter=5000,random_state=seed))])

def make_rf(max_depth,min_samples_leaf,seed=42):
    return RandomForestClassifier(n_estimators=500,class_weight="balanced_subsample",max_features="sqrt",
      random_state=seed,n_jobs=-1,max_depth=max_depth,min_samples_leaf=min_samples_leaf)
