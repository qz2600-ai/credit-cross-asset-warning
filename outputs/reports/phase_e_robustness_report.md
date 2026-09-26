# Phase E Robustness and Stability Addendum

The locked Phase D primary conclusion remains **No reliable improvement**. These analyses are secondary and do not redefine the primary result.

## Five-year rolling window

| model   |    pr_auc |   threshold |   alert_rate |   precision |   recall |   episodes_detected |   episode_detection_rate |   false_alert_days |   median_lead_time |
|:--------|----------:|------------:|-------------:|------------:|---------:|--------------------:|-------------------------:|-------------------:|-------------------:|
| m1      | 0.11305   |    0.440215 |     0.230337 |   0.131098  | 0.754386 |                  16 |                 0.941176 |                285 |                9.5 |
| m2      | 0.0619451 |    0.531285 |     0.104635 |   0.0872483 | 0.22807  |                   4 |                 0.235294 |                136 |                9   |

The rolling-window PR-AUC comparison remains unfavorable to M2. This does not replace the expanding-window primary test.

## Bootstrap block sensitivity

|   block_days | metric             |   estimate |     ci_2_5 |   ci_97_5 |   valid_replicates |   discarded_one_class |
|-------------:|:-------------------|-----------:|-----------:|----------:|-------------------:|----------------------:|
|           10 | m1_pr_auc          |  0.113573  |  0.048677  | 0.232197  |               1000 |                     0 |
|           10 | m2_pr_auc          |  0.0675418 |  0.0334252 | 0.126554  |               1000 |                     0 |
|           10 | m2_minus_m1_pr_auc | -0.0460314 | -0.137177  | 0.0151856 |               1000 |                     0 |
|           20 | m1_pr_auc          |  0.113573  |  0.0503158 | 0.246802  |               1000 |                     0 |
|           20 | m2_pr_auc          |  0.0675418 |  0.0318957 | 0.136686  |               1000 |                     0 |
|           20 | m2_minus_m1_pr_auc | -0.0460314 | -0.139598  | 0.0195515 |               1000 |                     0 |
|           40 | m1_pr_auc          |  0.113573  |  0.0409919 | 0.241127  |               1000 |                     0 |
|           40 | m2_pr_auc          |  0.0675418 |  0.0296216 | 0.133884  |               1000 |                     0 |
|           40 | m2_minus_m1_pr_auc | -0.0460314 | -0.144981  | 0.0196752 |               1000 |                     0 |
|           60 | m1_pr_auc          |  0.113573  |  0.0321739 | 0.213723  |               1000 |                     0 |
|           60 | m2_pr_auc          |  0.0675418 |  0.0260307 | 0.132676  |               1000 |                     0 |
|           60 | m2_minus_m1_pr_auc | -0.0460314 | -0.127767  | 0.0330128 |               1000 |                     0 |

## Retrospective equal-alert-budget diagnostic

The highest-risk 10% of locked test dates (143 dates) were selected retrospectively. These are not deployable thresholds.

| model   |   alert_budget_days |   actual_alert_days |   retrospective_score_cutoff |   precision |   recall |   episodes_detected |   episode_detection_rate |   false_alert_days |   median_lead_time |
|:--------|--------------------:|--------------------:|-----------------------------:|------------:|---------:|--------------------:|-------------------------:|-------------------:|-------------------:|
| m1      |                 143 |                 143 |                     0.537875 |   0.13986   | 0.350877 |                   7 |                 0.411765 |                123 |                  8 |
| m2      |                 143 |                 143 |                     0.486636 |   0.0769231 | 0.192982 |                   5 |                 0.294118 |                132 |                  9 |
| m3      |                 143 |                 143 |                     0.5781   |   0.146853  | 0.368421 |                   7 |                 0.411765 |                122 |                 10 |

## Regime concentration

45 of 57 positive test days and 14 of 17 stress episodes occur in 2022.

|   year |   positive_days |   episodes |
|-------:|----------------:|-----------:|
|   2021 |               0 |          0 |
|   2022 |              45 |         14 |
|   2023 |               4 |          2 |
|   2024 |               0 |          0 |
|   2025 |               8 |          1 |
|   2026 |               0 |          0 |

## Evidence by stage

| stage                             |   m1_pr_auc |   m2_pr_auc |   m2_minus_m1_pr_auc |
|:----------------------------------|------------:|------------:|---------------------:|
| initial_training_purged_cv        |    0.186694 |   0.113174  |           -0.0735201 |
| walk_forward_validation_2017_2020 |    0.250099 |   0.404139  |            0.154041  |
| locked_test_2021_2026             |    0.113573 |   0.0675418 |           -0.0460314 |

M2 underperformed M1 in initial-training purged cross-validation, outperformed it during 2017–2020 validation, and underperformed it in the locked 2021–2026 test. Feature redundancy and regime instability are plausible explanations for the lack of generalization, not proven causal explanations.
