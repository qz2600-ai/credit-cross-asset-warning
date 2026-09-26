import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/"src"))
from credit_warning.features import log_returns, make_features

def synthetic(n=320):
    idx=pd.bdate_range("2020-01-01",periods=n)
    x=np.arange(n,dtype=float)
    # deterministic non-degenerate positive prices and VIX
    return pd.DataFrame({
      "HYG":100*np.exp(.001*x + .0003*np.sin(x)),
      "LQD":90*np.exp(.0006*x + .0002*np.cos(x)),
      "IEF":80*np.exp(.0002*x + .00015*np.sin(x*.7)),
      "SPY":200*np.exp(.0012*x + .0004*np.cos(x*.5)),
      "VIXCLS":20+2*np.sin(x/13)+.01*x},index=idx)

def test_log_return_hand_calculated():
    p=pd.DataFrame({"HYG":[100.,110.,121.]},index=pd.bdate_range("2020-01-01",periods=3))
    got=log_returns(p)["HYG"].iloc[1:].to_numpy()
    assert np.allclose(got,[np.log(1.1),np.log(1.1)])

def test_return_features_and_gaps_exact():
    d=synthetic(); f=make_features(d); r=np.log(d[["HYG","LQD","IEF","SPY"]]/d[["HYG","LQD","IEF","SPY"]].shift(1))
    t=d.index[100]
    assert np.isclose(f.loc[t,"hyg_return_5d"],r["HYG"].loc[:t].tail(5).sum())
    assert np.isclose(f.loc[t,"hyg_volatility_20d"],r["HYG"].loc[:t].tail(20).std(ddof=1)*np.sqrt(252))
    assert np.isclose(f.loc[t,"hyg_negative_share_10d"],(r["HYG"].loc[:t].tail(10)<0).mean())
    assert np.isclose(f.loc[t,"hy_ig_gap_5d"],r["HYG"].loc[:t].tail(5).sum()-r["LQD"].loc[:t].tail(5).sum())
    assert np.isclose(f.loc[t,"hy_ig_gap_20d"],r["HYG"].loc[:t].tail(20).sum()-r["LQD"].loc[:t].tail(20).sum())
    assert np.isclose(f.loc[t,"hy_treasury_gap_10d"],r["HYG"].loc[:t].tail(10).sum()-r["IEF"].loc[:t].tail(10).sum())

def test_equity_adjusted_hyg_synthetic_linear_relationship():
    n=150; idx=pd.bdate_range("2020-01-01",periods=n)
    spy_r=.001+.0005*np.sin(np.arange(n-1))
    alpha=.0002; beta=.7; hyg_r=alpha+beta*spy_r
    spy=100*np.r_[1,np.exp(np.cumsum(spy_r))]
    hyg=100*np.r_[1,np.exp(np.cumsum(hyg_r))]
    d=pd.DataFrame({"HYG":hyg,"SPY":spy,"LQD":hyg*.9,"IEF":spy*.8,"VIXCLS":20+np.arange(n)*.01},index=idx)
    f=make_features(d); t=idx[100]
    assert abs(f.loc[t,"equity_adjusted_hyg_5d"]) < 1e-12

def test_correlations_exact():
    d=synthetic(); f=make_features(d); r=np.log(d[["HYG","LQD","IEF","SPY"]]/d[["HYG","LQD","IEF","SPY"]].shift(1)); t=d.index[100]
    assert np.isclose(f.loc[t,"hyg_spy_correlation_20d"],r["HYG"].loc[:t].tail(20).corr(r["SPY"].loc[:t].tail(20)))
    assert np.isclose(f.loc[t,"hyg_ief_correlation_20d"],r["HYG"].loc[:t].tail(20).corr(r["IEF"].loc[:t].tail(20)))

def test_cross_asset_dispersion_known_formula():
    d=synthetic(); f=make_features(d); r=np.log(d[["HYG","LQD","IEF","SPY"]]/d[["HYG","LQD","IEF","SPY"]].shift(1)); t=d.index[100]; pos=d.index.get_loc(t)
    vals=[]
    for a in ["HYG","LQD","IEF","SPY"]:
        ret5=r[a].iloc[pos-4:pos+1].sum()
        sigma=r[a].iloc[pos-64:pos-4].std(ddof=1)
        vals.append(ret5/(sigma*np.sqrt(5)))
    assert np.isclose(f.loc[t,"cross_asset_dispersion_5d"],np.std(vals,ddof=1))

def test_vix_zscore_excludes_current_and_vix_change():
    d=synthetic(); f=make_features(d); t=d.index[280]; pos=d.index.get_loc(t)
    prior=d["VIXCLS"].iloc[pos-252:pos]
    expected=(d.loc[t,"VIXCLS"]-prior.mean())/prior.std(ddof=1)
    assert np.isclose(f.loc[t,"vix_zscore_252d"],expected)
    assert np.isclose(f.loc[t,"vix_change_5d"],np.log(d["VIXCLS"].iloc[pos]/d["VIXCLS"].iloc[pos-5]))

def test_rolling_boundaries():
    d=synthetic(); f=make_features(d)
    assert f["hyg_return_5d"].first_valid_index()==d.index[5]
    assert f["hyg_volatility_20d"].first_valid_index()==d.index[20]
    assert f["equity_adjusted_hyg_5d"].first_valid_index()==d.index[65]
    assert f["cross_asset_dispersion_5d"].first_valid_index()==d.index[65]
    assert f["vix_zscore_252d"].first_valid_index()==d.index[252]
