# Phase B Feature and Label Audit

**Phase A gate: PASS. Phase B completed. No models were constructed, fit, tuned, or evaluated.**

Configured analysis window remains 2007-04-04 through 2026-09-18.
Usable complete-feature date range: 2008-04-10 through 2026-09-18.
Numerical initial-training 10th-percentile stress cutoff: `-0.030244246702`.

## Row counts

| stage                                           |   row_count |
|:------------------------------------------------|------------:|
| aligned_phase_a_rows                            |        4892 |
| feature_rows_before_complete_case_drop          |        4892 |
| rows_after_feature_complete_case_drop           |        4640 |
| label_rows_constructed                          |        4892 |
| complete_feature_rows_with_observable_10d_label |        4630 |
| training_rows_after_boundary_exclusion          |        2189 |
| validation_rows_after_boundary_exclusion        |         997 |
| test_rows_after_boundary_exclusion              |        1424 |

## Missing values by feature

| feature                   |   missing_count |
|:--------------------------|----------------:|
| hyg_return_5d             |               5 |
| hyg_volatility_20d        |              20 |
| hyg_negative_share_10d    |              10 |
| hy_ig_gap_5d              |               5 |
| hy_ig_gap_20d             |              20 |
| hy_treasury_gap_10d       |              10 |
| equity_adjusted_hyg_5d    |              65 |
| hyg_spy_correlation_20d   |              20 |
| hyg_ief_correlation_20d   |              20 |
| cross_asset_dispersion_5d |              65 |
| vix_zscore_252d           |             252 |
| vix_change_5d             |               5 |

## Feature summary statistics

| feature                   |   count |         mean |       std |        min |          1% |         5% |         25% |          50% |        75% |       95% |       99% |        max |
|:--------------------------|--------:|-------------:|----------:|-----------:|------------:|-----------:|------------:|-------------:|-----------:|----------:|----------:|-----------:|
| hyg_return_5d             |    4887 |  0.000944291 | 0.0150581 | -0.189292  | -0.0443794  | -0.018652  | -0.00376346 |  0.00160003  | 0.00624624 | 0.0201285 | 0.0388405 |  0.132605  |
| hyg_volatility_20d        |    4872 |  0.0781319   | 0.075063  |  0.0144916 |  0.0177687  |  0.024148  |  0.0377902  |  0.0559992   | 0.0910802  | 0.197765  | 0.398717  |  0.757279  |
| hyg_negative_share_10d    |    4882 |  0.452397    | 0.161074  |  0         |  0.1        |  0.2       |  0.3        |  0.4         | 0.6        | 0.7       | 0.8       |  0.9       |
| hy_ig_gap_5d              |    4887 |  0.000180037 | 0.0131169 | -0.123046  | -0.0399202  | -0.0186296 | -0.00530467 |  0.00080686  | 0.00615504 | 0.0181902 | 0.0330228 |  0.0897937 |
| hy_ig_gap_20d             |    4872 |  0.000717929 | 0.0235118 | -0.245421  | -0.0737346  | -0.0355582 | -0.00895234 |  0.00267432  | 0.0129487  | 0.0305683 | 0.053112  |  0.137234  |
| hy_treasury_gap_10d       |    4882 |  0.000681419 | 0.024916  | -0.216158  | -0.0790871  | -0.0358119 | -0.00812992 |  0.00255038  | 0.0113499  | 0.0337004 | 0.0634948 |  0.14854   |
| equity_adjusted_hyg_5d    |    4827 | -9.44685e-05 | 0.0110711 | -0.118107  | -0.030443   | -0.0143963 | -0.00401859 | -5.44029e-05 | 0.00404831 | 0.0134136 | 0.0292141 |  0.164774  |
| hyg_spy_correlation_20d   |    4872 |  0.691792    | 0.190734  | -0.586387  |  0.00718281 |  0.331368  |  0.612769   |  0.731307    | 0.825184   | 0.912098  | 0.948146  |  0.969592  |
| hyg_ief_correlation_20d   |    4872 | -0.0122119   | 0.478403  | -0.858356  | -0.764214   | -0.666772  | -0.437372   | -0.0947943   | 0.422914   | 0.784774  | 0.888476  |  0.957636  |
| cross_asset_dispersion_5d |    4827 |  0.722969    | 0.553619  |  0.0373288 |  0.104515   |  0.171276  |  0.368232   |  0.593082    | 0.91733    | 1.67401   | 2.73836   |  8.35173   |
| vix_zscore_252d           |    4640 | -0.0649984   | 1.3237    | -2.07558   | -1.74075    | -1.42213   | -0.9144     | -0.386774    | 0.399761   | 2.29672   | 4.9577    | 17.8937    |
| vix_change_5d             |    4887 |  0.000279285 | 0.151715  | -0.628654  | -0.350384   | -0.222737  | -0.0861699  | -0.00965026  | 0.0762771  | 0.24966   | 0.443843  |  1.14072   |

## Stress-label rates by chronological sample

| period     |   rows |   stress_labels |   stress_label_rate | start      | end        |
|:-----------|-------:|----------------:|--------------------:|:-----------|:-----------|
| training   |   2189 |             219 |           0.100046  | 2008-04-10 | 2016-12-15 |
| validation |    997 |              41 |           0.0411234 | 2017-01-03 | 2020-12-16 |
| test       |   1424 |              57 |           0.0400281 | 2021-01-04 | 2026-09-03 |

## Label-horizon boundary exclusions

| period     |   boundary_crossing_rows_excluded |
|:-----------|----------------------------------:|
| training   |                                10 |
| validation |                                10 |
| test       |                                 0 |

Boundary-crossing labels were excluded from each chronological sample by requiring `label_end_date <= period end`.
The stress cutoff was computed only from complete-feature initial-training rows with an observable 10-day label whose `label_end_date` is on or before 2016-12-31.
No feature was removed or selected based on outcomes, and no model performance was inspected.