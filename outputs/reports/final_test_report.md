# Final Locked Test Report

## Locked-test integrity
The Phase C specification was held fixed. The test contains 1,424 eligible observations from 2021-01-04 through 2026-09-03. Predictions were generated once in 21-trading-day blocks. Before every block, training rows were restricted to `label_end_date < block_start`; logistic scaling was refit on that training sample only. No test outcome was used for tuning or threshold selection.

## Primary result
**Predeclared classification: No reliable improvement.**

M1 test PR-AUC is 0.113573; M2 test PR-AUC is 0.067542. The M2-minus-M1 difference is -0.046031 (-40.5% relative). The paired 20-trading-day moving-block bootstrap 95% interval for the PR-AUC difference is [-0.139598, 0.019551]. Under the predeclared rule, M2 has no reliable improvement because its test PR-AUC is lower than M1's.

M2 has a lower Brier score (0.117772 versus 0.142252), but its locked-threshold recall and episode detection are materially lower. This does not override the primary PR-AUC comparison.

## Test metrics
|    pr_auc |   roc_auc |      brier |   precision |   recall |       f1 |   alert_rate |   false_alerts_per_year |   mean_future_trough_after_alert |   median_future_trough_after_alert | model   |   alert_days |   episodes_detected |   stress_episode_detection_rate |   median_detected_episode_lead_time |
|----------:|----------:|-----------:|------------:|---------:|---------:|-------------:|------------------------:|---------------------------------:|-----------------------------------:|:--------|-------------:|--------------------:|--------------------------------:|------------------------------------:|
| 0.113573  |  0.806273 |   0.142252 |   0.141818  | 0.684211 | 0.23494  |    0.193118  |                41.6823  |                      -0.0127324  |                        -0.00867103 | m1      |          275 |                  13 |                        0.764706 |                                   9 |
| 0.0675418 |  0.682876 |   0.117772 |   0.0841121 | 0.157895 | 0.109756 |    0.0751404 |                17.3088  |                      -0.00870842 |                        -0.00539613 | m2      |          107 |                   5 |                        0.294118 |                                   8 |
| 0.113244  |  0.807775 |   0.129682 |   0.140741  | 0.333333 | 0.197917 |    0.0948034 |                20.4879  |                      -0.0117729  |                        -0.00553995 | m3      |          135 |                   7 |                        0.411765 |                                  10 |
| 0.0709548 |  0.46316  | nan        |   0.140351  | 0.140351 | 0.140351 |    0.0400281 |                 8.65438 |                      -0.0129983  |                        -0.00977196 | m0      |           57 |                   3 |                        0.176471 |                                   7 |

## M2 versus M1
| comparison   |   pr_auc_absolute_difference |   pr_auc_relative_difference |   roc_auc_difference |   brier_difference |   precision_difference |   recall_difference |   episode_detection_difference |   median_lead_time_difference |
|:-------------|-----------------------------:|-----------------------------:|---------------------:|-------------------:|-----------------------:|--------------------:|-------------------------------:|------------------------------:|
| M2_minus_M1  |                   -0.0460314 |                    -0.405302 |            -0.123397 |         -0.0244799 |              -0.057706 |           -0.526316 |                      -0.470588 |                            -1 |

## Bootstrap
| metric             |   estimate |     ci_2_5 |   ci_97_5 |   valid_replicates |   discarded_one_class |
|:-------------------|-----------:|-----------:|----------:|-------------------:|----------------------:|
| m1_pr_auc          |  0.113573  |  0.0503158 | 0.246802  |               1000 |                     0 |
| m2_pr_auc          |  0.0675418 |  0.0318957 | 0.136686  |               1000 |                     0 |
| m2_minus_m1_pr_auc | -0.0460314 | -0.139598  | 0.0195515 |               1000 |                     0 |

## Episode dependence
There are 17 distinct test stress episodes. The three largest contain 29 of 57 positive-label days (50.9%), so daily labels should not be interpreted as independent crisis observations.

## Interpretation
This study evaluates early-warning discrimination for the specified HYG stress label. It does not predict defaults, establish causality, or demonstrate a profitable trading strategy.
