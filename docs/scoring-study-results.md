# Scoring sensitivity study: full output (scoring version 1.0)

This file is the unedited output of `python tools/scoring_sensitivity.py` for
scoring version 1.0 (progress lines removed). The analysis is in
[benchmark-methodology.md](benchmark-methodology.md), section 6. Scores are
family-balanced means over the public suite (seeds 0, 1, 2) and a generated
stochastic study suite (16 scenarios, seeds 0, 1, 2), unless a table says
otherwise. The adversarial policies are study instruments, not baselines.

### Raw components (hierarchical means over both suites; Safety parts and quality parts in [0, 1])

| policy | evac | casualty | stranding | Safety | time | coordination | hazard | path | replanning | EMBER |
|---|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.1 | 93.4 | 0.0 | n/a | n/a | n/a | n/a | 21.6 | 0.0 |
| random-legal | 0.1 | 0.0 | 95.5 | 0.0 | 23.3 | 79.0 | 91.9 | 41.1 | 21.7 | 0.0 |
| nearest-greedy | 57.6 | 30.3 | 92.7 | 26.8 | 85.5 | 65.9 | 86.4 | 100.0 | 27.3 | 25.1 |
| static-shortest | 67.0 | 39.5 | 100.0 | 36.5 | 86.6 | 76.1 | 87.1 | 100.0 | 27.3 | 34.7 |
| fire-aware | 73.2 | 43.0 | 100.0 | 39.2 | 83.7 | 74.2 | 88.1 | 97.3 | 78.7 | 36.5 |
| congestion-aware | 82.4 | 54.2 | 100.0 | 50.8 | 83.9 | 77.5 | 89.9 | 94.7 | 84.0 | 45.0 |
| adv-reckless-fast | 74.1 | 48.2 | 100.0 | 45.7 | 86.9 | 79.3 | 89.4 | 97.6 | 27.3 | 41.4 |
| adv-abandon-far | 61.0 | 25.9 | 95.8 | 15.7 | 83.9 | 77.4 | 91.1 | 95.1 | 58.5 | 13.6 |
| adv-safe-slow | 46.8 | 26.0 | 93.1 | 21.7 | 47.9 | 57.6 | 83.9 | 94.7 | 45.9 | 15.0 |
| adv-late-start | 47.3 | 33.3 | 100.0 | 32.6 | 44.4 | 75.9 | 83.0 | 93.4 | 69.9 | 24.7 |
| adv-jostler | 74.8 | 43.4 | 100.0 | 39.5 | 83.7 | 75.0 | 88.2 | 96.1 | 78.7 | 36.7 |
| adv-herd | 64.2 | 39.9 | 100.0 | 37.0 | 78.4 | 74.3 | 87.3 | 92.2 | 89.7 | 33.0 |
| adv-wanderer | 52.2 | 33.7 | 93.1 | 23.9 | 50.3 | 72.3 | 85.3 | 56.4 | 47.7 | 15.9 |
| adv-hider | 45.5 | 36.2 | 91.7 | 26.4 | 76.5 | 44.5 | 94.2 | 89.3 | 30.6 | 21.0 |

### Totals (people over all episodes)

| policy | evacuated | casualties | stranded | privileged max |
|---|---|---|---|---|
| always-stay | 0 | 1287 | 36 | 1131 |
| random-legal | 35 | 1261 | 27 | 1131 |
| nearest-greedy | 827 | 481 | 15 | 1131 |
| static-shortest | 880 | 443 | 0 | 1131 |
| fire-aware | 905 | 418 | 0 | 1131 |
| congestion-aware | 1005 | 318 | 0 | 1131 |
| adv-reckless-fast | 965 | 358 | 0 | 1131 |
| adv-abandon-far | 887 | 427 | 9 | 1131 |
| adv-safe-slow | 681 | 633 | 9 | 1131 |
| adv-late-start | 672 | 651 | 0 | 1131 |
| adv-jostler | 944 | 379 | 0 | 1131 |
| adv-herd | 800 | 523 | 0 | 1131 |
| adv-wanderer | 823 | 491 | 9 | 1131 |
| adv-hider | 643 | 653 | 27 | 1131 |

### Candidate formulas: overall scores per suite

| policy | official | additive | plain-product | no-incident-factor | arith-quality |
|---|---|---|---|---|---|
| always-stay | 0.0 | 1.1 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 17.1 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 25.1 | 73.2 | 37.9 | 30.2 | 25.2 |
| static-shortest | 34.7 | 81.7 | 47.3 | 40.1 | 34.9 |
| fire-aware | 36.5 | 84.5 | 50.2 | 43.5 | 36.9 |
| congestion-aware | 45.0 | 88.0 | 53.0 | 54.2 | 45.7 |
| adv-reckless-fast | 41.4 | 84.6 | 50.1 | 48.0 | 41.8 |
| adv-abandon-far | 13.6 | 80.8 | 44.1 | 22.7 | 13.9 |
| adv-safe-slow | 15.0 | 63.8 | 16.3 | 17.7 | 16.2 |
| adv-late-start | 24.7 | 58.4 | 17.2 | 27.0 | 25.8 |
| adv-jostler | 36.7 | 85.1 | 50.4 | 44.3 | 37.0 |
| adv-herd | 33.0 | 79.2 | 41.0 | 37.8 | 33.5 |
| adv-wanderer | 15.9 | 66.5 | 10.7 | 19.2 | 16.9 |
| adv-hider | 21.0 | 57.8 | 23.2 | 24.4 | 22.0 |

#### official score, public suite only

| policy | EMBER |
|---|---|
| always-stay | 0.0 |
| random-legal | 0.0 |
| nearest-greedy | 26.8 |
| static-shortest | 36.7 |
| fire-aware | 39.6 |
| congestion-aware | 49.6 |
| adv-reckless-fast | 44.5 |
| adv-abandon-far | 11.3 |
| adv-safe-slow | 16.4 |
| adv-late-start | 31.6 |
| adv-jostler | 39.7 |
| adv-herd | 36.0 |
| adv-wanderer | 17.2 |
| adv-hider | 23.0 |

#### official score, generated suite only

| policy | EMBER |
|---|---|
| always-stay | 0.0 |
| random-legal | 0.0 |
| nearest-greedy | 19.3 |
| static-shortest | 28.0 |
| fire-aware | 26.3 |
| congestion-aware | 29.6 |
| adv-reckless-fast | 31.2 |
| adv-abandon-far | 21.2 |
| adv-safe-slow | 10.6 |
| adv-late-start | 1.6 |
| adv-jostler | 26.4 |
| adv-herd | 23.2 |
| adv-wanderer | 11.7 |
| adv-hider | 14.5 |

### Requirement checks per candidate

| requirement | official | additive | plain-product | no-incident-factor | arith-quality |
|---|---|---|---|---|---|
| R1 one casualty in 100, otherwise perfect, scores <= 70 | pass | FAIL | FAIL | FAIL | pass |
| R2 safe run with Time 0.25 beats perfect run with 1/100 casualty | pass | FAIL | FAIL | FAIL | pass |
| R3 2 casualties < 1 casualty by more than 1 < 0 does | pass | pass | pass | pass | pass |
| R4 10/100 casualties still >= 5 (unsafe runs stay separable) | pass | pass | pass | pass | pass |
| R5 10/100 casualties <= 35 | pass | FAIL | FAIL | FAIL | pass |
| R6 one stranded in 100 <= 85 and above one casualty | pass | FAIL | FAIL | FAIL | pass |
| R7 10/100 stranded <= 50 | pass | FAIL | FAIL | FAIL | pass |
| R8 congestion rate 0.3 costs >= 10 points | pass | FAIL | pass | pass | pass |
| R9 50% losses score 0 (catastrophic gate) | pass | FAIL | FAIL | pass | pass |
| R10 non-compensation: Time 0.1 with everything else perfect scores <= 60 | pass | FAIL | pass | pass | FAIL |

### Adversarial checks (official vs candidates): policy must score below congestion-aware

| adversary - congestion-aware | official | additive | plain-product | no-incident-factor | arith-quality |
|---|---|---|---|---|---|
| adv-reckless-fast | -3.6 | -3.4 | -2.9 | -6.1 | -3.8 |
| adv-abandon-far | -31.4 | -7.2 | -9.0 | -31.5 | -31.8 |
| adv-safe-slow | -30.0 | -24.1 | -36.7 | -36.5 | -29.5 |
| adv-late-start | -20.3 | -29.6 | -35.9 | -27.2 | -19.9 |
| adv-jostler | -8.4 | -2.9 | -2.7 | -9.9 | -8.6 |
| adv-herd | -12.0 | -8.7 | -12.0 | -16.4 | -12.2 |
| adv-wanderer | -29.1 | -21.5 | -42.3 | -34.9 | -28.8 |
| adv-hider | -24.0 | -30.2 | -29.9 | -29.7 | -23.7 |

### Synthetic single-failure profile (official)

| case | official | additive | plain-product | no-incident-factor | arith-quality |
|---|---|---|---|---|---|
| perfect | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1 casualty / 100 | 56.5 | 99.4 | 99.0 | 94.1 | 56.5 |
| 2 casualties / 100 | 39.4 | 98.9 | 98.0 | 88.6 | 39.4 |
| 3 casualties / 100 | 30.0 | 98.3 | 97.0 | 83.3 | 30.0 |
| 5 casualties / 100 | 19.6 | 97.2 | 95.0 | 73.5 | 19.6 |
| 10 casualties / 100 | 9.1 | 94.4 | 90.0 | 53.1 | 9.1 |
| 25 casualties / 100 | 1.6 | 86.1 | 75.0 | 17.8 | 1.6 |
| 1 casualty / 4 | 10.7 | 86.1 | 75.0 | 17.8 | 10.7 |
| 1 stranded / 100 | 76.8 | 99.4 | 99.0 | 96.1 | 76.8 |
| 10 stranded / 100 | 30.3 | 94.4 | 90.0 | 65.6 | 30.3 |
| Time 0.5 | 77.1 | 91.7 | 50.0 | 77.1 | 81.2 |
| Time 0.25 | 59.5 | 87.5 | 25.0 | 59.5 | 71.9 |
| congestion rate 0.1 | 94.9 | 97.9 | 81.0 | 94.9 | 95.2 |
| congestion rate 0.3 | 83.7 | 94.3 | 49.0 | 83.7 | 87.2 |
| congestion rate 0.6 | 63.2 | 90.7 | 16.0 | 63.2 | 79.0 |

### Component correlation (Spearman, episodes with every component defined, moving policies)

|  | safety | time | coordination | hazard | path |
|---|---|---|---|---|---|
| safety | 1.00 | 0.30 | 0.33 | 0.13 | 0.09 |
| time | 0.30 | 1.00 | 0.37 | 0.14 | 0.48 |
| coordination | 0.33 | 0.37 | 1.00 | 0.08 | -0.01 |
| hazard | 0.13 | 0.14 | 0.08 | 1.00 | 0.05 |
| path | 0.09 | 0.48 | -0.01 | 0.05 | 1.00 |

(n = 908 episodes)

### Sensitivity grid: constants satisfying every requirement, and rank stability

feasible: 52 of 405 combinations
official constants {'casualty_incident_factor': '0.6', 'casualty_rate_exponent': '4', 'evacuation_exponent': '2', 'stranding_incident_factor': '0.8', 'congestion_exponent': '2'} feasible: True
Kendall tau vs official ranking over feasible combinations: min 0.956, mean 0.982
- casualty_incident_factor: 0.5: 16, 0.6: 36, 0.7: 0, 0.8: 0, 0.9: 0
- casualty_rate_exponent: 2: 24, 4: 16, 6: 12
- evacuation_exponent: 1: 20, 2: 16, 3: 16
- stranding_incident_factor: 0.7: 26, 0.8: 26, 0.9: 0
- congestion_exponent: 1: 0, 2: 26, 3: 26

### Quality-weight perturbation (each weight x0.5 and x1.5, renormalized)

| perturbation | Kendall tau | max |delta score| |
|---|---|---|
| time x0.5 | 0.978 | 1.9 |
| time x1.5 | 1.000 | 1.2 |
| coordination x0.5 | 1.000 | 0.7 |
| coordination x1.5 | 0.978 | 0.5 |
| hazard x0.5 | 1.000 | 0.7 |
| hazard x1.5 | 0.978 | 0.6 |
| path x0.5 | 1.000 | 0.3 |
| path x1.5 | 1.000 | 0.3 |
| replanning x0.5 | 1.000 | 0.1 |
| replanning x1.5 | 1.000 | 0.1 |

### Official ranking

| rank | policy | EMBER |
|---|---|---|
| 1 | congestion-aware | 45.0 |
| 2 | adv-reckless-fast | 41.4 |
| 3 | adv-jostler | 36.7 |
| 4 | fire-aware | 36.5 |
| 5 | static-shortest | 34.7 |
| 6 | adv-herd | 33.0 |
| 7 | nearest-greedy | 25.1 |
| 8 | adv-late-start | 24.7 |
| 9 | adv-hider | 21.0 |
| 10 | adv-wanderer | 15.9 |
| 11 | adv-safe-slow | 15.0 |
| 12 | adv-abandon-far | 13.6 |
| 13 | always-stay | 0.0 |
| 14 | random-legal | 0.0 |
