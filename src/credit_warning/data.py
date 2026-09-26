from __future__ import annotations
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class RawMetadata:
    source: str
    retrieval_utc: str
    ticker_or_series: str
    field_definition: str
    requested_start: str
    requested_end: str
    first_valid: str | None
    last_valid: str | None
    row_count: int
    distribution_adjustment: str

def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.to_datetime(out.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert(None)
    out.index = idx.normalize()
    out.index.name = "date"
    return out.sort_index()

def load_etf_history(tickers, start, end) -> pd.DataFrame:
    """Yahoo/yfinance adapter. Returns one trustworthy adjusted-price column per ticker.

    yfinance's end is exclusive, so one calendar day is added to honor configured end.
    We explicitly request auto_adjust=False and use provider Adj Close; raw Close is never
    substituted. Missing Adj Close is a hard failure.
    """
    import yfinance as yf
    end_exclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    raw = yf.download(
        list(tickers), start=start, end=end_exclusive, interval="1d",
        auto_adjust=False, actions=False, repair=False, progress=False,
        threads=True, group_by="column", multi_level_index=True,
    )
    if raw is None or raw.empty:
        raise RuntimeError("ETF provider returned no data.")
    if not isinstance(raw.columns, pd.MultiIndex) or "Adj Close" not in raw.columns.get_level_values(0):
        raise RuntimeError("Provider did not supply a trustworthy 'Adj Close' field; raw Close substitution is forbidden.")
    adj = raw["Adj Close"].copy()
    if isinstance(adj, pd.Series):
        adj = adj.to_frame(name=list(tickers)[0])
    missing_cols = [t for t in tickers if t not in adj.columns]
    if missing_cols:
        raise RuntimeError(f"Missing adjusted-price columns: {missing_cols}")
    adj = adj[list(tickers)].apply(pd.to_numeric, errors="coerce")
    return _normalize_index(adj)

def load_vix_history(series_id, start, end) -> pd.DataFrame:
    """Direct FRED CSV adapter for the daily-close VIX series."""
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={series_id}&cosd={start}&coed={end}"
    )
    df = pd.read_csv(url)
    date_col = "DATE" if "DATE" in df.columns else "observation_date"
    if date_col not in df.columns or series_id not in df.columns:
        raise RuntimeError(f"Unexpected FRED CSV schema: {list(df.columns)}")
    df[date_col] = pd.to_datetime(df[date_col])
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    return _normalize_index(df.set_index(date_col)[[series_id]])

def save_raw_series(df: pd.DataFrame, raw_dir: Path, source: str, field_definition: str,
                    requested_start: str, requested_end: str,
                    distribution_adjustment: str) -> list[dict]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    retrieval = datetime.now(timezone.utc).isoformat()
    records = []
    for col in df.columns:
        s = df[[col]]
        path = raw_dir / f"{col}.csv"
        s.to_csv(path)
        valid = s[col].dropna()
        meta = RawMetadata(
            source=source, retrieval_utc=retrieval, ticker_or_series=str(col),
            field_definition=field_definition, requested_start=requested_start,
            requested_end=requested_end,
            first_valid=None if valid.empty else str(valid.index.min().date()),
            last_valid=None if valid.empty else str(valid.index.max().date()),
            row_count=int(len(s)),
            distribution_adjustment=distribution_adjustment,
        )
        (raw_dir / f"{col}.metadata.json").write_text(json.dumps(asdict(meta), indent=2))
        records.append(asdict(meta))
    return records

def load_cached_raw(raw_dir: Path, etfs, vix_series) -> tuple[pd.DataFrame, pd.DataFrame]:
    etf_parts = []
    for t in etfs:
        p = raw_dir / f"{t}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p, parse_dates=["date"], index_col="date")
        etf_parts.append(d[[t]])
    vpath = raw_dir / f"{vix_series}.csv"
    if not vpath.exists():
        raise FileNotFoundError(vpath)
    vix = pd.read_csv(vpath, parse_dates=["date"], index_col="date")[[vix_series]]
    return pd.concat(etf_parts, axis=1), vix

def data_fingerprint(paths) -> str:
    h = hashlib.sha256()
    for p in sorted(map(Path, paths)):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()

def audit_and_align(etfs: pd.DataFrame, vix: pd.DataFrame, etf_names, vix_series,
                    analysis_start: str, analysis_end: str):
    """Audit required raw series and align by inner join to HYG trading dates.
    No ETF forward filling; no VIX filling is performed.
    Missingness for comparisons/VIX is measured against HYG dates.
    """
    etfs = _normalize_index(etfs)
    vix = _normalize_index(vix)
    hyg_dates = etfs["HYG"].dropna().loc[analysis_start:analysis_end].index
    base = pd.DataFrame(index=hyg_dates)
    combined = base.join(etfs[list(etf_names)], how="left").join(vix[[vix_series]], how="left")
    rows = []
    for col in [*etf_names, vix_series]:
        s = combined[col]
        valid = s.dropna()
        prices_like = col in etf_names or col == vix_series
        # Extreme-return review flag: |daily log return| > 20%; flag only, not a gate criterion.
        if prices_like:
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(valid / valid.shift(1))
        else:
            lr = pd.Series(dtype=float)
        extreme = int((lr.abs() > 0.20).sum())
        rows.append({
            "series": col,
            "source": "Yahoo Finance via yfinance" if col in etf_names else "FRED (VIXCLS; source CBOE)",
            "first_valid": None if valid.empty else valid.index.min().date().isoformat(),
            "last_valid": None if valid.empty else valid.index.max().date().isoformat(),
            "hyg_reference_dates": int(len(hyg_dates)),
            "missing_observations": int(s.isna().sum()),
            "missing_pct": float(s.isna().mean() * 100) if len(s) else np.nan,
            "duplicate_dates": int((etfs.index.duplicated().sum() if col in etf_names else vix.index.duplicated().sum())),
            "nonpositive_values": int((valid <= 0).sum()),
            "extreme_abs_log_return_gt_20pct": extreme,
            "date_convention": "timezone-naive normalized exchange/FRED observation date",
            "distribution_adjustment": "Yahoo provider Adj Close (splits and cash distributions)" if col in etf_names else "not applicable; daily VIX close",
        })
    audit = pd.DataFrame(rows)
    aligned = combined.dropna()
    common_pct = 100.0 * len(aligned) / len(hyg_dates) if len(hyg_dates) else 0.0

    stress_ranges = {
        "2008": ("2008-01-01","2008-12-31"),
        "2011": ("2011-01-01","2011-12-31"),
        "2015-2016": ("2015-01-01","2016-12-31"),
        "2020": ("2020-01-01","2020-12-31"),
        "2022": ("2022-01-01","2022-12-31"),
    }
    represented = {k: bool(len(aligned.loc[a:b])) for k,(a,b) in stress_ranges.items()}
    criteria = {
        "HYG history reaches 2007": bool(not etfs["HYG"].dropna().empty and etfs["HYG"].dropna().index.min().year <= 2007),
        "all comparison ETFs cover HYG actual observations": (
            False if etfs["HYG"].dropna().empty else all(
                (
                    not etfs[t].dropna().empty
                    and etfs[t].dropna().index.min() <= etfs["HYG"].dropna().index.min()
                    and etfs[t].dropna().index.max() >= etfs["HYG"].dropna().index.max()
                )
                for t in etf_names if t != "HYG"
            )
        ),
        "adjusted-price fields available and documented": True,
        "no duplicate dates": bool(audit["duplicate_dates"].eq(0).all()),
        "no nonpositive values": bool(audit["nonpositive_values"].eq(0).all()),
        "unexplained missingness below 1% each": bool((audit["missing_pct"] < 1.0).all()),
        "at least 95% HYG dates remain": bool(common_pct >= 95.0),
        "required stress periods represented": bool(all(represented.values())),
    }
    return audit, aligned, common_pct, represented, criteria
