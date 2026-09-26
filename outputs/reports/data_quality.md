# Phase A Data Quality Report

**Gate result: PASS**

Configured analysis window: 2007-04-04 through 2026-09-18.
Common aligned HYG trading dates: 4892 (100.000% of HYG dates).
No ETF forward fill or VIX fill was performed; alignment uses an inner join after audit.

## Gate criteria

- [x] HYG history reaches 2007
- [x] all comparison ETFs cover HYG actual observations
- [x] adjusted-price fields available and documented
- [x] no duplicate dates
- [x] no nonpositive values
- [x] unexplained missingness below 1% each
- [x] at least 95% HYG dates remain
- [x] required stress periods represented

## Stress-period representation

- 2008: True
- 2011: True
- 2015-2016: True
- 2020: True
- 2022: True

## Largest VIX daily changes

The 20% absolute daily log-return threshold is diagnostic only. Valid VIX observations are retained as observed; none are removed or winsorized.

| date                |   vix_close |   daily_log_change |   daily_pct_change |
|:--------------------|------------:|-------------------:|-------------------:|
| 2018-02-05 00:00:00 |       37.32 |           0.768245 |           115.598  |
| 2024-12-18 00:00:00 |       27.62 |           0.55411  |            74.0391 |
| 2024-08-05 00:00:00 |       38.57 |           0.500166 |            64.8995 |
| 2021-01-27 00:00:00 |       37.21 |           0.480214 |            61.6421 |
| 2025-04-09 00:00:00 |       33.62 |          -0.442449 |           -35.7539 |
| 2021-11-26 00:00:00 |       28.62 |           0.43202  |            54.0366 |
| 2025-04-04 00:00:00 |       45.31 |           0.411664 |            50.9327 |
| 2011-08-08 00:00:00 |       48    |           0.405465 |            50      |
| 2016-06-24 00:00:00 |       25.76 |           0.401011 |            49.3333 |
| 2020-06-11 00:00:00 |       40.79 |           0.391709 |            47.9507 |

## Series audit

| series   | source                     | first_valid   | last_valid   |   hyg_reference_dates |   missing_observations |   missing_pct |   duplicate_dates |   nonpositive_values |   extreme_abs_log_return_gt_20pct | date_convention                                          | distribution_adjustment                                  |
|:---------|:---------------------------|:--------------|:-------------|----------------------:|-----------------------:|--------------:|------------------:|---------------------:|----------------------------------:|:---------------------------------------------------------|:---------------------------------------------------------|
| HYG      | Yahoo Finance via yfinance | 2007-04-11    | 2026-09-18   |                  4892 |                      0 |             0 |                 0 |                    0 |                                 0 | timezone-naive normalized exchange/FRED observation date | Yahoo provider Adj Close (splits and cash distributions) |
| LQD      | Yahoo Finance via yfinance | 2007-04-11    | 2026-09-18   |                  4892 |                      0 |             0 |                 0 |                    0 |                                 0 | timezone-naive normalized exchange/FRED observation date | Yahoo provider Adj Close (splits and cash distributions) |
| IEF      | Yahoo Finance via yfinance | 2007-04-11    | 2026-09-18   |                  4892 |                      0 |             0 |                 0 |                    0 |                                 0 | timezone-naive normalized exchange/FRED observation date | Yahoo provider Adj Close (splits and cash distributions) |
| SPY      | Yahoo Finance via yfinance | 2007-04-11    | 2026-09-18   |                  4892 |                      0 |             0 |                 0 |                    0 |                                 0 | timezone-naive normalized exchange/FRED observation date | Yahoo provider Adj Close (splits and cash distributions) |
| VIXCLS   | FRED (VIXCLS; source CBOE) | 2007-04-11    | 2026-09-18   |                  4892 |                      0 |             0 |                 0 |                    0 |                               118 | timezone-naive normalized exchange/FRED observation date | not applicable; daily VIX close                          |