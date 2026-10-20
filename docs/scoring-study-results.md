# Scoring sensitivity study: full output (scoring version 1.0)

This file is the unedited output of `python tools/scoring_sensitivity.py` for
scoring version 1.0 (progress lines removed; headings demoted by one level).
The analysis is in [benchmark-methodology.md](benchmark-methodology.md),
section 6. Scores are family-balanced means over the public suite (seeds 0,
1, 2) and a generated stochastic study suite (16 scenarios, seeds 0, 1, 2),
unless a table says otherwise. The adversarial policies are study
instruments, not baselines. `F-official` is the official formula; the other
columns are the rejected designs of benchmark-methodology.md, section 5.

quality exponent beta = 0.736966; severe thresholds (component space) time 0.5000, coordination 0.5625, hazard 0.7500, path 0.5000, replanning 0.5000

### Raw components (hierarchical means over both suites; Safety parts and quality parts in [0, 1])

| policy | evac | casualty | stranding | Safety | time | coordination | hazard | path | replanning | EMBER |
|---|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.1 | 93.4 | 0.0 | n/a | n/a | n/a | n/a | 21.6 | 0.0 |
| random-legal | 0.1 | 0.0 | 95.5 | 0.0 | 23.3 | 79.0 | 91.9 | 41.1 | 21.7 | 0.0 |
| nearest-greedy | 57.6 | 30.3 | 92.7 | 26.8 | 95.2 | 65.9 | 86.4 | 100.0 | 27.3 | 25.2 |
| static-shortest | 67.0 | 39.5 | 100.0 | 36.5 | 95.6 | 76.1 | 87.1 | 100.0 | 27.3 | 34.8 |
| fire-aware | 73.2 | 43.0 | 100.0 | 39.2 | 92.7 | 74.2 | 88.1 | 97.3 | 78.7 | 35.7 |
| congestion-aware | 82.4 | 54.2 | 100.0 | 50.8 | 94.5 | 77.5 | 89.9 | 94.7 | 84.0 | 43.8 |
| adv-reckless-fast | 74.1 | 48.2 | 100.0 | 45.7 | 97.4 | 79.3 | 89.4 | 97.6 | 27.3 | 41.5 |
| adv-abandon-far | 61.0 | 25.9 | 95.8 | 15.7 | 93.4 | 77.4 | 91.1 | 95.1 | 58.5 | 12.7 |
| adv-safe-slow | 46.8 | 26.0 | 93.1 | 21.7 | 50.6 | 57.6 | 83.9 | 94.7 | 45.9 | 10.8 |
| adv-late-start | 47.3 | 33.3 | 100.0 | 32.6 | 50.4 | 75.9 | 83.0 | 93.4 | 69.9 | 22.0 |
| adv-jostler | 74.8 | 43.4 | 100.0 | 39.5 | 93.1 | 75.0 | 88.2 | 96.1 | 78.7 | 35.8 |
| adv-herd | 64.2 | 39.9 | 100.0 | 37.0 | 86.2 | 74.3 | 87.3 | 92.2 | 89.7 | 32.8 |
| adv-wanderer | 52.2 | 33.7 | 93.1 | 23.9 | 55.1 | 72.3 | 85.3 | 56.4 | 47.7 | 12.9 |
| adv-hider | 45.5 | 36.2 | 91.7 | 26.4 | 85.0 | 44.5 | 94.2 | 89.3 | 30.6 | 19.1 |
| adv-fire-hugger | 79.2 | 52.2 | 100.0 | 49.3 | 93.5 | 78.0 | 88.5 | 94.6 | 83.9 | 41.9 |

### Severe quality failures (share of applicable episodes, %)

| policy | time | coordination | hazard | path | replanning |
|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 37.2 |
| random-legal | 29.5 | 7.7 | 6.4 | 16.7 | 32.1 |
| nearest-greedy | 0.0 | 23.1 | 7.7 | 0.0 | 19.2 |
| static-shortest | 0.0 | 7.7 | 6.4 | 0.0 | 19.2 |
| fire-aware | 0.0 | 19.2 | 6.4 | 0.0 | 9.0 |
| congestion-aware | 0.0 | 16.7 | 6.4 | 0.0 | 9.0 |
| adv-reckless-fast | 0.0 | 7.7 | 2.6 | 0.0 | 19.2 |
| adv-abandon-far | 0.0 | 12.8 | 2.6 | 0.0 | 15.4 |
| adv-safe-slow | 51.3 | 39.7 | 15.4 | 0.0 | 23.1 |
| adv-late-start | 52.6 | 26.9 | 15.4 | 0.0 | 21.8 |
| adv-jostler | 0.0 | 16.7 | 6.4 | 0.0 | 9.0 |
| adv-herd | 9.0 | 19.2 | 11.5 | 5.1 | 7.7 |
| adv-wanderer | 28.2 | 20.5 | 21.8 | 23.1 | 12.8 |
| adv-hider | 3.8 | 60.3 | 1.3 | 0.0 | 16.7 |
| adv-fire-hugger | 0.0 | 11.5 | 7.7 | 0.0 | 10.3 |

### Totals (people over all episodes) and person-level attribution

| policy | evacuated | casualties | stranded | savable | avoidable cas. | draft rule | avoidable str. | draft rule |
|---|---|---|---|---|---|---|---|---|
| always-stay | 0 | 1287 | 36 | 1131 | 1095 | 1095 | 36 | 36 |
| random-legal | 35 | 1261 | 27 | 1131 | 1069 | 1069 | 27 | 27 |
| nearest-greedy | 827 | 481 | 15 | 1131 | 289 | 289 | 15 | 15 |
| static-shortest | 880 | 443 | 0 | 1131 | 251 | 251 | 0 | 0 |
| fire-aware | 905 | 418 | 0 | 1131 | 226 | 226 | 0 | 0 |
| congestion-aware | 1005 | 318 | 0 | 1131 | 126 | 126 | 0 | 0 |
| adv-reckless-fast | 965 | 358 | 0 | 1131 | 166 | 166 | 0 | 0 |
| adv-abandon-far | 887 | 427 | 9 | 1131 | 235 | 235 | 9 | 9 |
| adv-safe-slow | 681 | 633 | 9 | 1131 | 441 | 441 | 9 | 9 |
| adv-late-start | 672 | 651 | 0 | 1131 | 459 | 459 | 0 | 0 |
| adv-jostler | 944 | 379 | 0 | 1131 | 187 | 187 | 0 | 0 |
| adv-herd | 800 | 523 | 0 | 1131 | 331 | 331 | 0 | 0 |
| adv-wanderer | 823 | 491 | 9 | 1131 | 299 | 299 | 9 | 9 |
| adv-hider | 643 | 653 | 27 | 1131 | 461 | 461 | 27 | 27 |
| adv-fire-hugger | 997 | 326 | 0 | 1131 | 134 | 134 | 0 | 0 |

Episodes where the draft's aggregate rule differs from person-level attribution: 0 of 1170

### Candidate designs: overall scores (both suites)

| policy | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.1 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 17.1 | 0.0 |
| nearest-greedy | 25.2 | 25.4 | 25.4 | 25.8 | 26.7 | 27.0 | 25.6 | 74.7 | 39.7 |
| static-shortest | 34.8 | 35.1 | 35.0 | 35.5 | 36.5 | 36.7 | 35.4 | 83.2 | 49.3 |
| fire-aware | 35.7 | 36.9 | 36.8 | 37.4 | 39.1 | 38.6 | 36.4 | 86.0 | 52.1 |
| congestion-aware | 43.8 | 46.3 | 46.2 | 48.0 | 50.5 | 48.9 | 45.0 | 89.7 | 56.0 |
| adv-reckless-fast | 41.5 | 42.5 | 42.5 | 43.9 | 45.5 | 44.3 | 42.4 | 86.3 | 53.0 |
| adv-abandon-far | 12.7 | 13.9 | 13.9 | 14.4 | 15.6 | 16.0 | 13.2 | 82.4 | 46.3 |
| adv-safe-slow | 10.8 | 15.1 | 15.1 | 16.2 | 21.4 | 14.7 | 12.4 | 64.3 | 16.7 |
| adv-late-start | 22.0 | 25.0 | 25.0 | 26.5 | 32.2 | 24.7 | 24.4 | 59.3 | 18.1 |
| adv-jostler | 35.8 | 37.0 | 37.0 | 37.7 | 39.4 | 39.0 | 36.5 | 86.6 | 52.4 |
| adv-herd | 32.8 | 33.3 | 33.3 | 34.0 | 36.9 | 34.9 | 34.1 | 80.5 | 42.6 |
| adv-wanderer | 12.9 | 16.1 | 16.1 | 17.6 | 23.5 | 17.0 | 15.0 | 67.3 | 11.3 |
| adv-hider | 19.1 | 21.9 | 21.9 | 23.1 | 26.2 | 22.1 | 20.1 | 58.9 | 24.3 |
| adv-fire-hugger | 41.9 | 44.6 | 44.5 | 46.6 | 49.0 | 46.1 | 43.0 | 88.5 | 54.0 |

#### F-official and A-multiplicative, public suite only

| policy | F-official | A-multiplicative |
|---|---|---|
| always-stay | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 |
| nearest-greedy | 26.9 | 27.2 |
| static-shortest | 36.9 | 37.1 |
| fire-aware | 39.8 | 39.9 |
| congestion-aware | 49.1 | 51.2 |
| adv-reckless-fast | 44.5 | 45.8 |
| adv-abandon-far | 11.1 | 11.6 |
| adv-safe-slow | 11.2 | 16.5 |
| adv-late-start | 28.4 | 32.0 |
| adv-jostler | 39.9 | 40.1 |
| adv-herd | 37.0 | 36.3 |
| adv-wanderer | 13.8 | 17.4 |
| adv-hider | 22.5 | 24.1 |
| adv-fire-hugger | 46.1 | 48.8 |

#### F-official and A-multiplicative, generated suite only

| policy | F-official | A-multiplicative |
|---|---|---|
| always-stay | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 |
| nearest-greedy | 19.3 | 19.5 |
| static-shortest | 28.1 | 28.3 |
| fire-aware | 22.1 | 26.6 |
| congestion-aware | 26.0 | 30.1 |
| adv-reckless-fast | 31.4 | 31.6 |
| adv-abandon-far | 18.2 | 21.5 |
| adv-safe-slow | 9.3 | 10.7 |
| adv-late-start | 0.8 | 1.6 |
| adv-jostler | 22.2 | 26.7 |
| adv-herd | 18.9 | 23.4 |
| adv-wanderer | 9.9 | 11.8 |
| adv-hider | 8.1 | 14.6 |
| adv-fire-hugger | 27.9 | 30.4 |

### Requirement checks per candidate

| requirement | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| R1 one avoidable casualty in 100, otherwise perfect, scores <= 60 | pass | pass | pass | pass | pass | FAIL | pass | FAIL | FAIL |
| R2 Time is not irrelevant: Time 0.5 (not severe) costs >= 10 points | pass | pass | pass | FAIL | FAIL | pass | FAIL | FAIL | pass |
| R3 2 casualties score below 1 casualty | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| R4 10/100 casualties still >= 5 (unsafe runs stay separable) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| R5 10/100 casualties <= 35 | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R6 one stranded in 100 <= 85 and above one casualty | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R7 10/100 stranded <= 50 | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R8 congestion rate 0.3 costs >= 10 points | pass | pass | pass | FAIL | FAIL | pass | pass | FAIL | pass |
| R9 50% avoidable losses score 0 (catastrophic gate) | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R10 Time 0.1 with everything else perfect scores <= 60 | pass | pass | pass | FAIL | FAIL | pass | pass | FAIL | pass |
| R11 a savable casualty is not hidden by another person's unavoidable loss | pass | pass | pass | pass | pass | FAIL | pass | FAIL | FAIL |
| R12 no casualty + no severe failure beats perfect + 1 casualty (M* 2..1000, s 0..1) | pass | FAIL | FAIL | pass | pass | FAIL | pass | FAIL | FAIL |
| R13 severe bottleneck (rate 0.3), no loss: < 80 and >= 20 points lost | pass | FAIL | FAIL | FAIL | FAIL | pass | pass | FAIL | pass |
| R14 severe Time 0.4 / Hazard 0.3 / Replanning 0.4 / Path 0.4 alone: each < 80 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | pass | FAIL | pass |
| R15 failed / missing / timeout episodes score 0 and stay in every mean (structural; tests) | pass | pass | pass | pass | pass | pass | pass | pass | pass |

### Pre-registered ordering (left must score above right)

| ordering | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| zero-loss, Time 0.6 > perfect quality + 1 casualty | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| perfect > zero-loss but slow (Time 0.5) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, severe congestion (0.3) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, high exposure (0.3) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, inefficient routing (Path 0.6) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| replanning 0.9 > static-good, invalidation failure (0.3) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 1 stranded > 1 casualty | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| 1 casualty > 2 casualties | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| mild congestion (0.05) > severe congestion (0.4) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| mild delay (Time 0.8) > extreme delay (Time 0.1) | pass | pass | pass | pass | pass | pass | pass | pass | pass |

F-official values: zero-loss, Time 0.6 86.8 vs perfect quality + 1 casualty 56.5; perfect 100.0 vs zero-loss but slow (Time 0.5) 82.6; perfect 100.0 vs zero-loss, severe congestion (0.3) 61.1; perfect 100.0 vs zero-loss, high exposure (0.3) 69.9; perfect 100.0 vs zero-loss, inefficient routing (Path 0.6) 95.4; replanning 0.9 98.5 vs static-good, invalidation failure (0.3) 40.2; 1 stranded 76.8 vs 1 casualty 56.5; 1 casualty 56.5 vs 2 casualties 39.4; mild congestion (0.05) 98.1 vs severe congestion (0.4) 42.4; mild delay (Time 0.8) 94.0 vs extreme delay (Time 0.1) 8.5

### Adversarial checks: adversary minus congestion-aware (gaming check, not a selection criterion)

| adversary - congestion-aware | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| adv-reckless-fast | -2.3 | -3.7 | -3.7 | -4.1 | -5.0 | -4.6 | -2.6 | -3.4 | -3.0 |
| adv-abandon-far | -31.1 | -32.4 | -32.4 | -33.6 | -35.0 | -33.0 | -31.8 | -7.3 | -9.8 |
| adv-safe-slow | -33.0 | -31.2 | -31.1 | -31.9 | -29.2 | -34.3 | -32.6 | -25.5 | -39.3 |
| adv-late-start | -21.7 | -21.3 | -21.3 | -21.5 | -18.3 | -24.3 | -20.6 | -30.4 | -38.0 |
| adv-jostler | -8.0 | -9.3 | -9.2 | -10.3 | -11.1 | -10.0 | -8.5 | -3.1 | -3.6 |
| adv-herd | -11.0 | -12.9 | -12.9 | -14.0 | -13.7 | -14.1 | -10.9 | -9.2 | -13.4 |
| adv-wanderer | -30.8 | -30.2 | -30.1 | -30.4 | -27.0 | -32.0 | -30.0 | -22.5 | -44.8 |
| adv-hider | -24.6 | -24.4 | -24.3 | -24.9 | -24.4 | -26.9 | -24.9 | -30.9 | -31.7 |
| adv-fire-hugger | -1.9 | -1.7 | -1.7 | -1.4 | -1.5 | -2.8 | -2.0 | -1.3 | -2.1 |

### Synthetic single-failure profile

| case | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| perfect | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1 casualty / 100 | 56.5 | 56.5 | 56.5 | 56.5 | 56.5 | 63.0 | 56.5 | 99.4 | 99.0 |
| 2 casualties / 100 | 39.4 | 39.4 | 39.4 | 39.4 | 39.4 | 44.3 | 39.4 | 98.9 | 98.0 |
| 3 casualties / 100 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 33.7 | 30.0 | 98.3 | 97.0 |
| 5 casualties / 100 | 19.6 | 19.6 | 19.6 | 19.6 | 19.6 | 22.1 | 19.6 | 97.2 | 95.0 |
| 10 casualties / 100 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 10.2 | 9.1 | 94.4 | 90.0 |
| 25 casualties / 100 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.8 | 1.6 | 86.1 | 75.0 |
| 1 casualty / 4 | 10.7 | 10.7 | 10.7 | 10.7 | 10.7 | 12.0 | 10.7 | 86.1 | 75.0 |
| 1 stranded / 100 | 76.8 | 76.8 | 76.8 | 76.8 | 76.8 | 83.1 | 76.8 | 99.4 | 99.0 |
| 10 stranded / 100 | 30.3 | 30.3 | 30.3 | 30.3 | 30.3 | 34.1 | 30.3 | 94.4 | 90.0 |
| 1 savable dead + 1 unsavable stranded / 100 | 56.5 | 56.5 | 56.5 | 56.5 | 56.5 | 62.9 | 56.5 | 99.4 | 99.0 |
| Time 0.8 | 94.0 | 92.0 | 92.0 | 96.5 | 99.6 | 94.3 | 97.3 | 96.7 | 80.0 |
| Time 0.5 (threshold) | 82.6 | 77.1 | 77.1 | 90.0 | 98.9 | 69.5 | 92.0 | 91.7 | 50.0 |
| Time 0.4 (severe) | 49.7 | 70.9 | 70.9 | 87.3 | 98.5 | 56.7 | 57.3 | 90.0 | 40.0 |
| Time 0.25 | 27.3 | 59.5 | 59.5 | 82.4 | 98.0 | 35.7 | 33.8 | 87.5 | 25.0 |
| Time 0.1 | 8.5 | 42.2 | 42.2 | 74.8 | 97.1 | 14.3 | 12.1 | 85.0 | 10.0 |
| congestion rate 0.1 | 96.2 | 94.9 | 94.9 | 97.8 | 99.7 | 96.3 | 98.3 | 97.9 | 81.0 |
| congestion rate 0.25 (threshold) | 89.9 | 86.6 | 86.6 | 94.2 | 99.3 | 81.0 | 95.5 | 95.1 | 56.2 |
| congestion rate 0.3 (severe) | 61.1 | 83.7 | 80.0 | 92.9 | 99.2 | 73.0 | 65.8 | 94.3 | 49.0 |
| congestion rate 0.5 | 27.5 | 70.7 | 70.7 | 87.3 | 98.5 | 38.7 | 31.8 | 91.7 | 25.0 |
| avoidable hazard 0.1 | 98.1 | 97.4 | 97.4 | 98.9 | 99.9 | 98.5 | 99.2 | 98.9 | 90.0 |
| avoidable hazard 0.3 (severe) | 69.9 | 91.5 | 80.0 | 96.3 | 99.6 | 91.6 | 72.6 | 96.7 | 70.0 |
| Path 0.6 | 95.4 | 93.8 | 93.8 | 97.3 | 99.7 | 90.4 | 98.0 | 97.8 | 60.0 |
| Path 0.4 (severe) | 58.8 | 89.2 | 80.0 | 95.3 | 99.5 | 69.1 | 61.7 | 96.7 | 40.0 |
| Replanning 0.8 | 96.8 | 95.6 | 95.6 | 98.1 | 99.8 | 96.3 | 98.6 | 98.0 | 80.0 |
| Replanning 0.3 (severe) | 40.2 | 78.6 | 78.6 | 90.7 | 98.9 | 47.3 | 44.4 | 93.0 | 30.0 |
| every component at its threshold | 64.8 | 55.5 | 55.5 | 80.6 | 97.8 | 60.5 | 82.7 | 78.1 | 5.3 |

### Component correlation (Spearman, episodes with every component defined, moving policies)

|  | safety | time | coordination | hazard | path | replanning |
|---|---|---|---|---|---|---|
| safety | 1.00 | 0.05 | 0.23 | -0.08 | -0.34 | 0.49 |
| time | 0.05 | 1.00 | 0.20 | 0.05 | 0.55 | -0.12 |
| coordination | 0.23 | 0.20 | 1.00 | -0.23 | -0.15 | 0.35 |
| hazard | -0.08 | 0.05 | -0.23 | 1.00 | 0.24 | -0.51 |
| path | -0.34 | 0.55 | -0.15 | 0.24 | 1.00 | -0.56 |
| replanning | 0.49 | -0.12 | 0.35 | -0.51 | -0.56 | 1.00 |

(n = 220 episodes)

### Score spread (F-official)

moving policies: min 10.8, max 43.8, range 33.0, sd 11.9
A-multiplicative: range 32.4
C-safety-tiers: range 33.6
D-lexicographic: range 35.0
E-soft-min: range 34.3

### Per-family ranking (F-official; family means over seeds)

| policy | generated:single_exit_high_density | generated:single_exit_low_density | generated:split_flow_across_exits | public:central_ignition | public:crowd_100 | public:maze_obstacle_routing | public:multiple_exits_unequal_distance | public:multiple_ignitions | public:narrow_corridor_bottleneck | public:route_invalidation_replanning | public:single_exit_high_density | public:single_exit_low_density | public:threatened_nearest_exit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 18.4 | 17.5 | 22.0 | 4.1 | 0.0 | 0.0 | 99.0 | 9.0 | 24.8 | 32.6 | 0.4 | 99.4 | 0.0 |
| static-shortest | 18.5 | 38.1 | 27.9 | 4.1 | 0.0 | 99.3 | 99.0 | 9.0 | 24.8 | 32.6 | 0.4 | 99.4 | 0.0 |
| fire-aware | 7.5 | 33.9 | 24.9 | 4.1 | 0.0 | 99.3 | 99.0 | 9.0 | 24.8 | 44.0 | 0.4 | 99.4 | 18.3 |
| congestion-aware | 10.8 | 40.3 | 26.9 | 7.4 | 7.1 | 99.1 | 99.0 | 21.0 | 91.4 | 44.0 | 4.2 | 99.4 | 18.3 |
| adv-reckless-fast | 20.8 | 44.0 | 29.4 | 7.4 | 3.1 | 99.1 | 99.0 | 9.0 | 91.4 | 32.6 | 4.2 | 99.4 | 0.0 |
| adv-abandon-far | 7.1 | 32.0 | 15.6 | 7.4 | 0.4 | 32.5 | 15.7 | 9.1 | 25.0 | 4.9 | 1.2 | 10.7 | 3.5 |
| adv-safe-slow | 0.2 | 16.4 | 11.3 | 6.6 | 0.0 | 0.0 | 45.9 | 6.7 | 0.0 | 0.0 | 0.0 | 49.1 | 3.5 |
| adv-late-start | 0.0 | 0.7 | 1.7 | 0.0 | 1.3 | 90.8 | 60.7 | 0.0 | 0.0 | 70.1 | 1.5 | 59.7 | 0.0 |
| adv-jostler | 7.8 | 33.9 | 25.0 | 4.1 | 0.5 | 99.3 | 99.0 | 9.0 | 24.8 | 44.0 | 0.4 | 99.4 | 18.3 |
| adv-herd | 7.5 | 33.9 | 15.4 | 0.0 | 0.0 | 99.3 | 83.9 | 0.0 | 24.8 | 44.0 | 0.4 | 99.4 | 17.9 |
| adv-wanderer | 0.3 | 14.6 | 14.8 | 0.0 | 0.9 | 0.0 | 45.5 | 1.3 | 1.2 | 9.1 | 0.7 | 78.5 | 1.0 |
| adv-hider | 1.2 | 3.4 | 19.7 | 6.6 | 0.0 | 0.0 | 95.1 | 0.0 | 0.0 | 0.0 | 11.8 | 93.3 | 17.9 |
| adv-fire-hugger | 14.2 | 40.3 | 29.2 | 7.4 | 4.6 | 98.5 | 99.0 | 5.9 | 91.4 | 44.0 | 6.9 | 99.4 | 3.5 |
- generated:single_exit_high_density: adv-reckless-fast 20.8, static-shortest 18.5, nearest-greedy 18.4
- generated:single_exit_low_density: adv-reckless-fast 44.0, adv-fire-hugger 40.3, congestion-aware 40.3
- generated:split_flow_across_exits: adv-reckless-fast 29.4, adv-fire-hugger 29.2, static-shortest 27.9
- public:central_ignition: congestion-aware 7.4, adv-reckless-fast 7.4, adv-fire-hugger 7.4
- public:crowd_100: congestion-aware 7.1, adv-fire-hugger 4.6, adv-reckless-fast 3.1
- public:maze_obstacle_routing: static-shortest 99.3, fire-aware 99.3, adv-jostler 99.3
- public:multiple_exits_unequal_distance: nearest-greedy 99.0, static-shortest 99.0, fire-aware 99.0
- public:multiple_ignitions: congestion-aware 21.0, adv-abandon-far 9.1, fire-aware 9.0
- public:narrow_corridor_bottleneck: congestion-aware 91.4, adv-reckless-fast 91.4, adv-fire-hugger 91.4
- public:route_invalidation_replanning: adv-late-start 70.1, fire-aware 44.0, congestion-aware 44.0
- public:single_exit_high_density: adv-hider 11.8, adv-fire-hugger 6.9, congestion-aware 4.2
- public:single_exit_low_density: nearest-greedy 99.4, static-shortest 99.4, fire-aware 99.4
- public:threatened_nearest_exit: fire-aware 18.3, congestion-aware 18.3, adv-jostler 18.3

### Sensitivity grids: requirement feasibility and rank stability

#### grid safety: feasible 26 of 81
every combination: Kendall tau min 0.923, mean 0.979; max |delta score| 12.0
requirements failed (combinations): R1 27, R11 27, R4 15, R6 30, R7 9
official {'casualty_incident_factor': '0.6', 'casualty_rate_exponent': '4', 'evacuation_exponent': '2', 'stranding_incident_factor': '0.8'} feasible: True
Kendall tau vs official over feasible: min 0.923, mean 0.982; max |delta score| 7.8
- casualty_incident_factor: 0.5: 8, 0.6: 18, 0.7: 0
- casualty_rate_exponent: 2: 12, 4: 8, 6: 6
- evacuation_exponent: 1: 10, 2: 8, 3: 8
- stranding_incident_factor: 0.7: 13, 0.8: 13, 0.9: 0

#### grid gates: feasible 68 of 243
every combination: Kendall tau min 0.923, mean 0.963; max |delta score| 7.4
requirements failed (combinations): R13 81, R14 141, R8 27
official {'severe_factor': '0.8', 'severe_exponent': '1', 'severe_time': '1/2', 'severe_block_rate': '1/4', 'severe_hazard_rate': '1/4'} feasible: True
Kendall tau vs official over feasible: min 0.942, mean 0.972; max |delta score| 7.4
- severe_factor: 0.7: 24, 0.8: 24, 0.9: 20
- severe_exponent: 0.5: 20, 1: 24, 2: 24
- severe_time: 1/3: 0, 1/2: 34, 2/3: 34
- severe_block_rate: 1/5: 34, 1/4: 34, 1/3: 0
- severe_hazard_rate: 1/5: 36, 1/4: 32, 1/3: 0

### Quality-weight perturbation (each weight x0.5 and x1.5, renormalized)

| perturbation | Kendall tau | max |delta score| |
|---|---|---|
| time x0.5 | 1.000 | 1.1 |
| time x1.5 | 0.981 | 0.8 |
| coordination x0.5 | 0.981 | 0.4 |
| coordination x1.5 | 1.000 | 0.3 |
| hazard x0.5 | 0.981 | 0.4 |
| hazard x1.5 | 1.000 | 0.3 |
| path x0.5 | 1.000 | 0.2 |
| path x1.5 | 1.000 | 0.2 |
| replanning x0.5 | 1.000 | 0.1 |
| replanning x1.5 | 1.000 | 0.1 |

### Rank agreement between candidate designs (Kendall tau vs F-official)

| candidate | Kendall tau |
|---|---|
| F-official | 1.000 |
| A-multiplicative | 0.981 |
| B-hard-ceilings | 0.981 |
| C-safety-tiers | 0.962 |
| D-lexicographic | 0.942 |
| E-soft-min | 1.000 |
| F-strand-floor | 1.000 |
| additive | 0.808 |
| plain-product | 0.865 |

### Official ranking (F-official, both suites)

| rank | policy | EMBER |
|---|---|---|
| 1 | congestion-aware | 43.8 |
| 2 | adv-fire-hugger | 41.9 |
| 3 | adv-reckless-fast | 41.5 |
| 4 | adv-jostler | 35.8 |
| 5 | fire-aware | 35.7 |
| 6 | static-shortest | 34.8 |
| 7 | adv-herd | 32.8 |
| 8 | nearest-greedy | 25.2 |
| 9 | adv-late-start | 22.0 |
| 10 | adv-hider | 19.1 |
| 11 | adv-wanderer | 12.9 |
| 12 | adv-abandon-far | 12.7 |
| 13 | adv-safe-slow | 10.8 |
| 14 | always-stay | 0.0 |
| 15 | random-legal | 0.0 |
