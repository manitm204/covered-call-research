# Full ablation sweep — 2000 configs (exploratory, NOT pre-registered)

- configs with trades: 2000; errored: 0; zero-trade: 0
- positive expectancy: 10/2000 (0.5%); bootstrap 95% CI entirely above zero: 4 (chance expectation under a no-edge null at 2.5% one-sided: ~50)
- expectancy per spread distribution: p5 -42.2, p25 -20.1, median -11.7, p75 -7.8, p95 -4.9 ($)

## Marginal: short_delta

| short_delta | n | median exp $ | mean exp $ | % positive |
|---|---|---|---|---|
| 0.05 | 250 | -5.85 | -6.01 | 4% |
| 0.08 | 250 | -7.77 | -9.01 | 0% |
| 0.1 | 250 | -9.61 | -11.16 | 0% |
| 0.12 | 250 | -11.48 | -14.22 | 0% |
| 0.15 | 250 | -14.27 | -17.14 | 0% |
| 0.2 | 250 | -15.10 | -20.05 | 0% |
| 0.25 | 250 | -19.21 | -24.46 | 0% |
| 0.3 | 250 | -22.01 | -28.16 | 0% |

## Marginal: width

| width | n | median exp $ | mean exp $ | % positive |
|---|---|---|---|---|
| 1.0 | 400 | -8.58 | -9.79 | 2% |
| 2.0 | 400 | -10.21 | -12.47 | 0% |
| 3.0 | 400 | -11.63 | -14.81 | 0% |
| 5.0 | 400 | -14.08 | -18.73 | 0% |
| 10.0 | 400 | -18.78 | -25.59 | 0% |

## Marginal: target_dte

| target_dte | n | median exp $ | mean exp $ | % positive |
|---|---|---|---|---|
| 7 | 400 | -9.99 | -12.76 | 0% |
| 14 | 400 | -11.93 | -15.88 | 1% |
| 21 | 400 | -10.29 | -13.86 | 1% |
| 30 | 400 | -14.37 | -18.91 | 0% |
| 45 | 400 | -13.29 | -19.98 | 1% |

## Marginal: entry_frequency

| entry_frequency | n | median exp $ | mean exp $ | % positive |
|---|---|---|---|---|
| monthly | 1000 | -13.60 | -18.90 | 1% |
| weekly | 1000 | -10.30 | -13.66 | 0% |

## Marginal: exit_style

| exit_style | n | median exp $ | mean exp $ | % positive |
|---|---|---|---|---|
| hold | 400 | -19.68 | -24.09 | 1% |
| pt25_sl2 | 400 | -8.65 | -9.99 | 0% |
| pt50 | 400 | -16.58 | -19.98 | 2% |
| pt50_sl2 | 400 | -10.05 | -12.18 | 0% |
| sl2 | 400 | -11.62 | -15.16 | 0% |

## Top 20 by expectancy (treat as hypotheses, not conclusions)

| config | n_trades | exp $ | 95% CI | win rate | max DD |
|---|---|---|---|---|---|
| d0.05_w1_dte45_monthly_hold | 41 | 7.87 | [0.15, 12.74] | 98% | -0.31% |
| d0.05_w1_dte14_monthly_pt50 | 30 | 7.54 | [4.20, 11.09] | 97% | -0.36% |
| d0.05_w1_dte21_monthly_hold | 29 | 6.46 | [0.56, 10.18] | 97% | -0.35% |
| d0.05_w10_dte14_monthly_pt50 | 86 | 5.44 | [-2.80, 11.53] | 95% | -0.37% |
| d0.05_w1_dte45_monthly_pt50 | 41 | 5.41 | [4.23, 6.71] | 93% | -0.36% |
| d0.05_w1_dte45_weekly_hold | 227 | 2.81 | [-6.97, 10.01] | 97% | -0.60% |
| d0.05_w5_dte14_monthly_pt50 | 86 | 1.76 | [-6.35, 7.74] | 95% | -0.45% |
| d0.05_w1_dte21_monthly_pt50 | 29 | 1.69 | [-3.74, 5.01] | 76% | -0.36% |
| d0.05_w2_dte21_monthly_hold | 78 | 0.18 | [-7.91, 6.36] | 95% | -0.29% |
| d0.05_w1_dte30_monthly_pt50 | 42 | 0.00 | [-11.27, 6.21] | 93% | -0.33% |
| d0.05_w1_dte21_weekly_hold | 116 | -0.20 | [-13.32, 7.97] | 94% | -0.48% |
| d0.05_w3_dte14_monthly_pt50 | 83 | -0.46 | [-9.60, 5.73] | 94% | -0.48% |
| d0.05_w1_dte14_monthly_pt50_sl2 | 30 | -0.66 | [-7.01, 5.58] | 70% | -0.40% |
| d0.08_w2_dte21_monthly_hold | 94 | -0.74 | [-10.62, 7.57] | 88% | -0.37% |
| d0.08_w1_dte21_monthly_hold | 88 | -0.86 | [-8.03, 5.01] | 85% | -0.44% |
| d0.05_w1_dte30_monthly_hold | 42 | -0.86 | [-19.01, 11.86] | 90% | -0.56% |
| d0.05_w2_dte14_monthly_pt50 | 73 | -0.99 | [-9.07, 4.73] | 82% | -0.50% |
| d0.05_w5_dte14_monthly_hold | 86 | -1.04 | [-15.24, 9.92] | 93% | -0.41% |
| d0.08_w5_dte21_monthly_hold | 95 | -1.06 | [-22.00, 15.34] | 88% | -0.38% |
| d0.05_w1_dte45_weekly_pt50 | 227 | -1.12 | [-8.14, 4.31] | 93% | -0.40% |

## Bottom 5 by expectancy

- d0.30_w10_dte45_monthly_hold: -145.94 $/spread over 95 trades
- d0.25_w10_dte45_monthly_hold: -130.76 $/spread over 95 trades
- d0.30_w10_dte45_monthly_pt50: -113.01 $/spread over 95 trades
- d0.20_w10_dte45_monthly_hold: -107.69 $/spread over 94 trades
- d0.30_w10_dte30_monthly_hold: -102.43 $/spread over 96 trades
