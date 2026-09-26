from __future__ import annotations
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "hyg_return_5d","hyg_volatility_20d","hyg_negative_share_10d",
    "hy_ig_gap_5d","hy_ig_gap_20d","hy_treasury_gap_10d",
    "equity_adjusted_hyg_5d","hyg_spy_correlation_20d","hyg_ief_correlation_20d",
    "cross_asset_dispersion_5d","vix_zscore_252d","vix_change_5d",
]

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1))

def trailing_log_return(r: pd.Series, k: int) -> pd.Series:
    return r.rolling(k, min_periods=k).sum()

def _equity_adjusted(hyg_r: pd.Series, spy_r: pd.Series, hyg5: pd.Series, spy5: pd.Series) -> pd.Series:
    # For date t, regress on exactly t-64..t-5: shift returns by 5, then take 60 rows.
    x = spy_r.shift(5)
    y = hyg_r.shift(5)
    mx = x.rolling(60, min_periods=60).mean()
    my = y.rolling(60, min_periods=60).mean()
    mxy = (x*y).rolling(60, min_periods=60).mean()
    mx2 = (x*x).rolling(60, min_periods=60).mean()
    cov = mxy - mx*my
    var = mx2 - mx*mx
    beta = cov / var
    alpha = my - beta*mx
    return hyg5 - (5.0*alpha + beta*spy5)

def _dispersion(prices_r: pd.DataFrame, ret5: pd.DataFrame) -> pd.Series:
    # sigma uses 60 daily returns ending at t-5, i.e. t-64..t-5.
    sigma = prices_r.shift(5).rolling(60, min_periods=60).std(ddof=1)
    standardized = ret5 / (sigma*np.sqrt(5.0))
    return standardized.std(axis=1, ddof=1)

def make_features(aligned: pd.DataFrame, vix_series: str="VIXCLS") -> pd.DataFrame:
    px = aligned[["HYG","LQD","IEF","SPY"]].astype(float)
    r = log_returns(px)
    r5 = r.rolling(5, min_periods=5).sum()
    r10 = r.rolling(10, min_periods=10).sum()
    r20 = r.rolling(20, min_periods=20).sum()
    f = pd.DataFrame(index=aligned.index)
    f["hyg_return_5d"] = r5["HYG"]
    f["hyg_volatility_20d"] = r["HYG"].rolling(20, min_periods=20).std(ddof=1)*np.sqrt(252.0)
    f["hyg_negative_share_10d"] = (r["HYG"] < 0).astype(float).where(r["HYG"].notna()).rolling(10,min_periods=10).mean()
    f["hy_ig_gap_5d"] = r5["HYG"] - r5["LQD"]
    f["hy_ig_gap_20d"] = r20["HYG"] - r20["LQD"]
    f["hy_treasury_gap_10d"] = r10["HYG"] - r10["IEF"]
    f["equity_adjusted_hyg_5d"] = _equity_adjusted(r["HYG"],r["SPY"],r5["HYG"],r5["SPY"])
    f["hyg_spy_correlation_20d"] = r["HYG"].rolling(20,min_periods=20).corr(r["SPY"])
    f["hyg_ief_correlation_20d"] = r["HYG"].rolling(20,min_periods=20).corr(r["IEF"])
    f["cross_asset_dispersion_5d"] = _dispersion(r, r5)
    v = aligned[vix_series].astype(float)
    benchmark_mean = v.shift(1).rolling(252,min_periods=252).mean()
    benchmark_std = v.shift(1).rolling(252,min_periods=252).std(ddof=1)
    f["vix_zscore_252d"] = (v-benchmark_mean)/benchmark_std
    f["vix_change_5d"] = np.log(v/v.shift(5))
    return f[FEATURE_COLUMNS]

def feature_dictionary() -> pd.DataFrame:
    rows = [
      ("hyg_return_5d","sum of HYG daily log returns over t-4..t","log return","negative","5 returns"),
      ("hyg_volatility_20d","std of HYG daily log returns t-19..t * sqrt(252)","annualized log-return volatility","positive","20 returns"),
      ("hyg_negative_share_10d","mean 1{HYG daily log return < 0} over t-9..t","share","positive","10 returns"),
      ("hy_ig_gap_5d","HYG 5d log return - LQD 5d log return","log-return gap","negative","5 returns"),
      ("hy_ig_gap_20d","HYG 20d log return - LQD 20d log return","log-return gap","negative","20 returns"),
      ("hy_treasury_gap_10d","HYG 10d log return - IEF 10d log return","log-return gap","negative","10 returns"),
      ("equity_adjusted_hyg_5d","HYG 5d return - (5*alpha + beta*SPY 5d); alpha,beta estimated on t-64..t-5","residual log return","negative","65 rows"),
      ("hyg_spy_correlation_20d","correlation of HYG and SPY daily log returns t-19..t","correlation","context-dependent","20 returns"),
      ("hyg_ief_correlation_20d","correlation of HYG and IEF daily log returns t-19..t","correlation","context-dependent","20 returns"),
      ("cross_asset_dispersion_5d","cross-sectional sample std of each asset 5d return/(60d sigma ending t-5 * sqrt(5))","standardized dispersion","positive","65 rows"),
      ("vix_zscore_252d","(VIX[t]-mean of t-252..t-1)/std of t-252..t-1","z-score","positive","252 prior observations"),
      ("vix_change_5d","log(VIX[t]/VIX[t-5])","log change","positive","5 rows"),
    ]
    return pd.DataFrame(rows,columns=["feature","formula","units","expected_sign","required_lookback"])
