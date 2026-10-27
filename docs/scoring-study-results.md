# Scoring sensitivity study: full output (scoring version 1.0)

> **Scope:** These are public development or archived research rules. They do
> not define the official hidden evaluation distribution. Previously disclosed
> recipes and study seeds are exposed research, not secret evaluation assets.
> See [Evaluation boundary](evaluation-boundary.md).


This file is the unedited output of `python tools/scoring_sensitivity.py` for
scoring version 1.0 (progress lines removed; headings demoted by one level).
The analysis is in [benchmark-methodology.md](benchmark-methodology.md),
sections 5.12 and 6. Unless a table says otherwise, scores are
family-balanced means over the two **study suites**: the public suite (seeds
0, 1, 2) and a generated stochastic study suite (16 scenarios, seeds 0, 1,
2). The **archived suite** is a second generated suite from the same config
and a fresh pre-registered generator seed (20261021, seeds 0, 1, 2); it was
not used to choose anything. The adversarial policies, including
`adv-block-scrub`, are study instruments, not baselines. `F-official` (and
`G5-official`) is the official formula; the other columns are the rejected
designs of benchmark-methodology.md, sections 5.9 and 5.12. Episodes where
nobody can escape (`not_applicable`) are excluded from every mean, as in the
score.

quality exponent beta = 0.736966; severe thresholds (component space) time 0.5000, hazard 0.7500, path 0.5000, replanning 0.5000

### Block-scrub reproduction (trajectory invariance)

| suite | episodes | identical state trajectories | identical trajectory evidence | identical official score | blocks (inner) | blocks (scrub) | evacuee blocks (inner) |
|---|---|---|---|---|---|---|---|
| public | 30 | 30 | 30 | 30 | 4449 | 0 | 4165 |
| generated | 48 | 48 | 48 | 48 | 844 | 0 | 246 |
| archived | 48 | 48 | 48 | 48 | 1211 | 0 | 507 |

Scrub gain (adv-block-scrub minus congestion-aware), per candidate and suite:

| candidate | public | generated | both | archived | max episode |delta| | R16 invariance |
|---|---|---|---|---|---|---|
| 1.0 | +4.52 | +7.35 | +5.17 | +3.07 | 70.85 | FAIL |
| G1 | +4.52 | +7.35 | +5.17 | +3.07 | 70.85 | FAIL |
| G4 | +0.00 | +0.00 | +0.00 | +0.00 | 0.00 | pass |
| G5-official | +0.00 | +0.00 | +0.00 | +0.00 | 0.00 | pass |
| G6 | +4.46 | +0.57 | +3.56 | +1.16 | 26.61 | FAIL |
| G7 | +0.07 | +0.12 | +0.08 | +0.17 | 3.16 | FAIL |
| T-wait | +0.00 | +0.00 | +0.00 | +0.00 | 0.00 | pass |

### Cut-capacity bound (Time reference) checks, every policy and suite

episodes with evacuees: 1773; cut bound <= person-time in all: True; cut bound > exit-capacity bound in 164; person-time equal to the cut bound in 789
- public: 411 episodes; cut > exit bound in 0; total capacity-forced delay 38484 person-steps of 138973
- generated: 676 episodes; cut > exit bound in 104; total capacity-forced delay 2044 person-steps of 52020
- archived: 686 episodes; cut > exit bound in 60; total capacity-forced delay 3761 person-steps of 56947

### Raw components (hierarchical means over both suites; Safety parts and quality parts in [0, 1])

| policy | evac | casualty | stranding | Safety | time | hazard | path | replanning | EMBER |
|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.1 | 93.4 | 0.0 | n/a | n/a | n/a | 21.6 | 0.0 |
| random-legal | 0.1 | 0.0 | 95.5 | 0.0 | 23.3 | 91.9 | 41.1 | 21.7 | 0.0 |
| nearest-greedy | 57.6 | 30.3 | 92.7 | 26.8 | 95.3 | 86.4 | 100.0 | 27.3 | 25.8 |
| static-shortest | 67.0 | 39.5 | 100.0 | 36.5 | 95.7 | 87.1 | 100.0 | 27.3 | 35.5 |
| fire-aware | 73.2 | 43.0 | 100.0 | 39.2 | 92.8 | 88.1 | 97.3 | 78.7 | 37.9 |
| congestion-aware | 82.4 | 54.2 | 100.0 | 50.8 | 94.6 | 89.9 | 94.7 | 84.0 | 48.6 |
| adv-reckless-fast | 74.1 | 48.2 | 100.0 | 45.7 | 97.5 | 89.4 | 97.6 | 27.3 | 44.4 |
| adv-abandon-far | 61.0 | 25.9 | 95.8 | 15.7 | 93.5 | 91.1 | 95.1 | 58.5 | 14.7 |
| adv-safe-slow | 46.8 | 26.0 | 93.1 | 21.7 | 50.7 | 83.9 | 94.7 | 45.9 | 10.6 |
| adv-late-start | 47.3 | 33.3 | 100.0 | 32.6 | 50.4 | 83.0 | 93.4 | 69.9 | 21.8 |
| adv-jostler | 74.8 | 43.4 | 100.0 | 39.5 | 93.2 | 88.2 | 96.1 | 78.7 | 38.2 |
| adv-herd | 64.2 | 39.9 | 100.0 | 37.0 | 86.2 | 87.3 | 92.2 | 89.7 | 34.6 |
| adv-wanderer | 52.2 | 33.7 | 93.1 | 23.9 | 55.2 | 85.3 | 56.4 | 47.7 | 12.6 |
| adv-hider | 45.5 | 36.2 | 91.7 | 26.4 | 85.1 | 94.2 | 89.3 | 30.6 | 24.4 |
| adv-fire-hugger | 79.2 | 52.2 | 100.0 | 49.3 | 93.6 | 88.5 | 94.6 | 83.9 | 47.3 |
| adv-block-scrub | 82.4 | 54.2 | 100.0 | 50.8 | 94.6 | 89.9 | 94.7 | 84.0 | 48.6 |

### Severe quality failures, study suites (share of applicable episodes, %; any = at least one gate)

| policy | time | hazard | path | replanning | any |
|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 37.2 | 37.2 |
| random-legal | 29.5 | 6.4 | 16.7 | 32.1 | 44.9 |
| nearest-greedy | 0.0 | 7.7 | 0.0 | 19.2 | 26.9 |
| static-shortest | 0.0 | 6.4 | 0.0 | 19.2 | 25.6 |
| fire-aware | 0.0 | 6.4 | 0.0 | 9.0 | 15.4 |
| congestion-aware | 0.0 | 6.4 | 0.0 | 9.0 | 15.4 |
| adv-reckless-fast | 0.0 | 2.6 | 0.0 | 19.2 | 21.8 |
| adv-abandon-far | 0.0 | 2.6 | 0.0 | 15.4 | 17.9 |
| adv-safe-slow | 51.3 | 15.4 | 0.0 | 23.1 | 64.1 |
| adv-late-start | 51.3 | 15.4 | 0.0 | 21.8 | 61.5 |
| adv-jostler | 0.0 | 6.4 | 0.0 | 9.0 | 15.4 |
| adv-herd | 9.0 | 11.5 | 5.1 | 7.7 | 24.4 |
| adv-wanderer | 26.9 | 21.8 | 23.1 | 12.8 | 50.0 |
| adv-hider | 3.8 | 1.3 | 0.0 | 16.7 | 17.9 |
| adv-fire-hugger | 0.0 | 7.7 | 0.0 | 10.3 | 17.9 |
| adv-block-scrub | 0.0 | 6.4 | 0.0 | 9.0 | 15.4 |

### Severe quality failures, archived suite (share of applicable episodes, %; any = at least one gate)

| policy | time | hazard | path | replanning | any |
|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 41.7 | 41.7 |
| random-legal | 33.3 | 8.3 | 27.1 | 33.3 | 52.1 |
| nearest-greedy | 0.0 | 0.0 | 0.0 | 12.5 | 12.5 |
| static-shortest | 0.0 | 0.0 | 0.0 | 12.5 | 12.5 |
| fire-aware | 0.0 | 2.1 | 0.0 | 16.7 | 16.7 |
| congestion-aware | 0.0 | 2.1 | 0.0 | 10.4 | 12.5 |
| adv-reckless-fast | 0.0 | 2.1 | 0.0 | 12.5 | 14.6 |
| adv-abandon-far | 0.0 | 2.1 | 0.0 | 8.3 | 10.4 |
| adv-safe-slow | 43.8 | 8.3 | 0.0 | 12.5 | 56.2 |
| adv-late-start | 64.6 | 8.3 | 0.0 | 25.0 | 75.0 |
| adv-jostler | 0.0 | 4.2 | 0.0 | 12.5 | 12.5 |
| adv-herd | 4.2 | 10.4 | 2.1 | 18.8 | 29.2 |
| adv-wanderer | 0.0 | 18.8 | 0.0 | 14.6 | 31.2 |
| adv-hider | 0.0 | 0.0 | 0.0 | 10.4 | 10.4 |
| adv-fire-hugger | 0.0 | 10.4 | 0.0 | 10.4 | 20.8 |
| adv-block-scrub | 0.0 | 2.1 | 0.0 | 10.4 | 12.5 |

### Congestion diagnostics (reported, never scored; study suites)

| policy | block rate | contested cell-steps / episode | hotspot cells / episode | peak contested | exit throughput |
|---|---|---|---|---|---|
| always-stay | n/a | 0.0 | 0.0 | 0 | n/a |
| random-legal | 0.275 | 24.1 | 17.0 | 18 | 0.156 |
| nearest-greedy | 0.557 | 30.1 | 7.2 | 13 | 0.842 |
| static-shortest | 0.491 | 24.2 | 7.1 | 13 | 0.840 |
| fire-aware | 0.488 | 25.6 | 8.3 | 13 | 0.789 |
| congestion-aware | 0.302 | 26.4 | 10.0 | 20 | 0.789 |
| adv-reckless-fast | 0.300 | 25.1 | 8.7 | 20 | 0.843 |
| adv-abandon-far | 0.367 | 24.6 | 8.2 | 22 | 0.786 |
| adv-safe-slow | 0.400 | 25.5 | 14.5 | 19 | 0.463 |
| adv-late-start | 0.366 | 21.1 | 7.9 | 20 | 0.841 |
| adv-jostler | 0.434 | 30.1 | 8.9 | 21 | 0.736 |
| adv-herd | 0.526 | 27.3 | 9.1 | 10 | 0.801 |
| adv-wanderer | 0.299 | 38.6 | 16.7 | 22 | 0.520 |
| adv-hider | 0.733 | 36.9 | 9.4 | 13 | 0.723 |
| adv-fire-hugger | 0.344 | 30.6 | 9.6 | 18 | 0.762 |
| adv-block-scrub | 0.000 | 0.0 | 0.0 | 0 | 0.789 |

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
| adv-block-scrub | 1005 | 318 | 0 | 1131 | 126 | 126 | 0 | 0 |

Episodes where the draft's aggregate rule differs from person-level attribution: 0 of 1248

### Candidate designs: overall scores (both suites)

| policy | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.1 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 5.4 | 0.1 |
| nearest-greedy | 25.8 | 26.0 | 26.0 | 26.2 | 26.8 | 27.2 | 26.1 | 75.8 | 54.0 |
| static-shortest | 35.5 | 35.6 | 35.6 | 35.9 | 36.5 | 36.9 | 35.8 | 84.2 | 62.4 |
| fire-aware | 37.9 | 38.0 | 37.9 | 38.3 | 39.2 | 40.0 | 38.4 | 87.5 | 67.1 |
| congestion-aware | 48.6 | 48.7 | 48.6 | 49.4 | 50.7 | 51.7 | 49.4 | 91.3 | 72.1 |
| adv-reckless-fast | 44.4 | 44.3 | 44.2 | 44.7 | 45.6 | 45.8 | 45.0 | 87.2 | 67.1 |
| adv-abandon-far | 14.7 | 14.9 | 14.8 | 15.1 | 15.6 | 16.9 | 14.9 | 83.1 | 59.9 |
| adv-safe-slow | 10.6 | 14.2 | 14.2 | 15.3 | 21.3 | 14.5 | 12.5 | 65.1 | 23.7 |
| adv-late-start | 21.8 | 23.7 | 23.7 | 25.2 | 32.2 | 23.9 | 24.7 | 57.9 | 24.2 |
| adv-jostler | 38.2 | 38.2 | 38.2 | 38.6 | 39.5 | 40.4 | 38.7 | 88.1 | 67.4 |
| adv-herd | 34.6 | 34.0 | 33.9 | 34.5 | 36.9 | 36.1 | 35.8 | 81.3 | 55.5 |
| adv-wanderer | 12.6 | 14.9 | 14.9 | 16.5 | 23.5 | 16.6 | 15.2 | 66.7 | 16.5 |
| adv-hider | 24.4 | 24.0 | 23.9 | 24.4 | 26.3 | 26.1 | 25.4 | 60.6 | 38.3 |
| adv-fire-hugger | 47.3 | 47.3 | 47.2 | 48.0 | 49.2 | 49.8 | 48.1 | 89.8 | 69.3 |
| adv-block-scrub | 48.6 | 48.7 | 48.6 | 49.4 | 50.7 | 51.7 | 49.4 | 91.3 | 72.1 |

#### F-official and A-multiplicative, public suite only

| policy | F-official | A-multiplicative |
|---|---|---|
| always-stay | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 |
| nearest-greedy | 27.5 | 27.7 |
| static-shortest | 37.4 | 37.5 |
| fire-aware | 40.4 | 40.3 |
| congestion-aware | 53.2 | 53.3 |
| adv-reckless-fast | 48.2 | 47.9 |
| adv-abandon-far | 12.0 | 12.1 |
| adv-safe-slow | 10.6 | 15.0 |
| adv-late-start | 28.1 | 30.3 |
| adv-jostler | 40.7 | 40.6 |
| adv-herd | 37.0 | 36.1 |
| adv-wanderer | 13.1 | 15.8 |
| adv-hider | 25.7 | 25.0 |
| adv-fire-hugger | 52.0 | 51.8 |
| adv-block-scrub | 53.2 | 53.3 |

#### F-official and A-multiplicative, generated suite only

| policy | F-official | A-multiplicative |
|---|---|---|
| always-stay | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 |
| nearest-greedy | 20.2 | 20.6 |
| static-shortest | 28.9 | 29.3 |
| fire-aware | 29.8 | 30.1 |
| congestion-aware | 33.1 | 33.3 |
| adv-reckless-fast | 31.9 | 32.2 |
| adv-abandon-far | 23.6 | 24.2 |
| adv-safe-slow | 10.7 | 11.4 |
| adv-late-start | 0.9 | 1.6 |
| adv-jostler | 29.8 | 30.1 |
| adv-herd | 26.6 | 26.9 |
| adv-wanderer | 10.9 | 11.9 |
| adv-hider | 20.2 | 20.5 |
| adv-fire-hugger | 31.6 | 32.3 |
| adv-block-scrub | 33.1 | 33.3 |

### Requirement checks per candidate

| requirement | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| R1 one avoidable casualty in 100, otherwise perfect, scores <= 60 | pass | pass | pass | pass | pass | FAIL | pass | FAIL | FAIL |
| R2 Time is not irrelevant: Time 0.5 (not severe) costs >= 10 points | pass | pass | pass | pass | FAIL | pass | pass | FAIL | pass |
| R3 2 casualties score below 1 casualty | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| R4 10/100 casualties still >= 5 (unsafe runs stay separable) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| R5 10/100 casualties <= 35 | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R6 one stranded in 100 <= 85 and above one casualty | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R7 10/100 stranded <= 50 | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R8 realized bottleneck (Time 0.5 vs the capacity bound, block rate 0.3) costs >= 10 points | pass | pass | pass | pass | FAIL | pass | pass | FAIL | pass |
| R9 50% avoidable losses score 0 (catastrophic gate) | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| R10 Time 0.1 with everything else perfect scores <= 60 | pass | pass | pass | FAIL | FAIL | pass | pass | FAIL | pass |
| R11 a savable casualty is not hidden by another person's unavoidable loss | pass | pass | pass | pass | pass | FAIL | pass | FAIL | FAIL |
| R12 no casualty + no severe failure beats perfect + 1 casualty (M* 2..1000, s 0..1) | pass | FAIL | FAIL | pass | pass | FAIL | pass | FAIL | FAIL |
| R13 severe realized bottleneck (Time 0.4, block rate 0.3), no loss: < 80 and >= 20 points lost | pass | pass | pass | FAIL | FAIL | pass | pass | FAIL | pass |
| R14 severe Time 0.4 / Hazard 0.3 / Replanning 0.4 / Path 0.4 alone: each < 80 | pass | FAIL | FAIL | FAIL | FAIL | FAIL | pass | FAIL | pass |
| R15 failed / missing / timeout episodes score 0 and stay in every mean (structural; tests) | pass | pass | pass | pass | pass | pass | pass | pass | pass |

### Pre-registered ordering (left must score above right)

| ordering | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| zero-loss, Time 0.6 > perfect quality + 1 casualty | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| perfect > zero-loss but slow (Time 0.5) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, severe realized bottleneck (Time 0.4) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, high exposure (0.3) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, inefficient routing (Path 0.6) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| replanning 0.9 > static-good, invalidation failure (0.3) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| 1 stranded > 1 casualty | pass | pass | pass | pass | pass | pass | pass | FAIL | FAIL |
| 1 casualty > 2 casualties | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| mild realized congestion (Time 0.9) > severe realized bottleneck (Time 0.4) | pass | pass | pass | pass | pass | pass | pass | pass | pass |
| mild delay (Time 0.8) > extreme delay (Time 0.1) | pass | pass | pass | pass | pass | pass | pass | pass | pass |

F-official values: zero-loss, Time 0.6 82.8 vs perfect quality + 1 casualty 56.5; perfect 100.0 vs zero-loss but slow (Time 0.5) 77.5; perfect 100.0 vs zero-loss, severe realized bottleneck (Time 0.4) 45.7; perfect 100.0 vs zero-loss, high exposure (0.3) 68.4; perfect 100.0 vs zero-loss, inefficient routing (Path 0.6) 93.9; replanning 0.9 98.1 vs static-good, invalidation failure (0.3) 38.5; 1 stranded 76.8 vs 1 casualty 56.5; 1 casualty 56.5 vs 2 casualties 39.4; mild realized congestion (Time 0.9) 96.2 vs severe realized bottleneck (Time 0.4) 45.7; mild delay (Time 0.8) 92.1 vs extreme delay (Time 0.1) 6.8

### Coordination candidates: overall scores (both study suites)

| policy | 1.0 | G1 | G4 | G5-official | G6 | G7 | T-wait |
|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 25.2 | 25.2 | 25.9 | 25.8 | 25.4 | 25.9 | 25.9 |
| static-shortest | 34.8 | 34.8 | 35.6 | 35.5 | 35.0 | 35.5 | 35.6 |
| fire-aware | 35.7 | 35.7 | 38.0 | 37.9 | 37.5 | 38.0 | 38.1 |
| congestion-aware | 43.8 | 43.8 | 48.8 | 48.6 | 45.4 | 48.9 | 48.9 |
| adv-reckless-fast | 41.5 | 41.5 | 44.6 | 44.4 | 41.6 | 44.6 | 44.7 |
| adv-abandon-far | 12.7 | 12.7 | 14.7 | 14.7 | 13.9 | 14.7 | 14.7 |
| adv-safe-slow | 10.8 | 10.8 | 10.0 | 10.6 | 11.1 | 11.2 | 7.0 |
| adv-late-start | 22.0 | 22.0 | 20.9 | 21.8 | 22.1 | 23.0 | 18.4 |
| adv-jostler | 35.8 | 35.8 | 38.3 | 38.2 | 37.5 | 38.2 | 38.3 |
| adv-herd | 32.8 | 32.8 | 34.5 | 34.6 | 34.5 | 35.0 | 35.1 |
| adv-wanderer | 12.9 | 12.9 | 12.2 | 12.6 | 13.3 | 13.6 | 13.6 |
| adv-hider | 19.1 | 19.1 | 24.3 | 24.4 | 22.0 | 24.8 | 24.9 |
| adv-fire-hugger | 41.9 | 41.9 | 47.5 | 47.3 | 42.7 | 47.5 | 47.6 |
| adv-block-scrub | 48.9 | 48.9 | 48.8 | 48.6 | 48.9 | 48.9 | 48.9 |

#### Coordination candidates, public suite only

| policy | 1.0 | G1 | G4 | G5-official | G6 | G7 | T-wait |
|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 26.9 | 26.9 | 27.6 | 27.5 | 27.0 | 27.6 | 27.6 |
| static-shortest | 36.9 | 36.9 | 37.5 | 37.4 | 36.9 | 37.5 | 37.5 |
| fire-aware | 39.8 | 39.8 | 40.4 | 40.4 | 39.9 | 40.5 | 40.5 |
| congestion-aware | 49.1 | 49.1 | 53.4 | 53.2 | 49.1 | 53.5 | 53.6 |
| adv-reckless-fast | 44.5 | 44.5 | 48.4 | 48.2 | 44.6 | 48.4 | 48.4 |
| adv-abandon-far | 11.1 | 11.1 | 12.0 | 12.0 | 11.1 | 12.0 | 12.0 |
| adv-safe-slow | 11.2 | 11.2 | 9.8 | 10.6 | 11.2 | 11.3 | 6.1 |
| adv-late-start | 28.4 | 28.4 | 26.9 | 28.1 | 28.4 | 29.5 | 23.8 |
| adv-jostler | 39.9 | 39.9 | 40.8 | 40.7 | 39.9 | 40.7 | 40.9 |
| adv-herd | 37.0 | 37.0 | 36.8 | 37.0 | 37.0 | 37.5 | 37.6 |
| adv-wanderer | 13.8 | 13.8 | 12.6 | 13.1 | 13.9 | 14.2 | 14.3 |
| adv-hider | 22.5 | 22.5 | 25.6 | 25.7 | 22.5 | 26.2 | 26.2 |
| adv-fire-hugger | 46.1 | 46.1 | 52.2 | 52.0 | 46.1 | 52.2 | 52.3 |
| adv-block-scrub | 53.6 | 53.6 | 53.4 | 53.2 | 53.6 | 53.6 | 53.6 |

#### Coordination candidates, generated suite only

| policy | 1.0 | G1 | G4 | G5-official | G6 | G7 | T-wait |
|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 19.3 | 19.3 | 20.3 | 20.2 | 20.1 | 20.3 | 20.3 |
| static-shortest | 28.1 | 28.1 | 29.0 | 28.9 | 28.7 | 29.0 | 29.0 |
| fire-aware | 22.1 | 22.1 | 29.9 | 29.8 | 29.6 | 29.9 | 29.9 |
| congestion-aware | 26.0 | 26.0 | 33.3 | 33.1 | 32.8 | 33.2 | 33.3 |
| adv-reckless-fast | 31.4 | 31.4 | 32.0 | 31.9 | 31.5 | 32.0 | 32.0 |
| adv-abandon-far | 18.2 | 18.2 | 23.8 | 23.6 | 23.4 | 23.8 | 23.8 |
| adv-safe-slow | 9.3 | 9.3 | 10.6 | 10.7 | 11.0 | 11.0 | 10.0 |
| adv-late-start | 0.8 | 0.8 | 0.9 | 0.9 | 1.0 | 1.0 | 0.6 |
| adv-jostler | 22.2 | 22.2 | 29.9 | 29.8 | 29.6 | 29.9 | 29.9 |
| adv-herd | 18.9 | 19.0 | 26.6 | 26.6 | 26.4 | 26.7 | 26.7 |
| adv-wanderer | 9.9 | 10.0 | 10.8 | 10.9 | 11.4 | 11.5 | 11.5 |
| adv-hider | 8.1 | 8.1 | 20.3 | 20.2 | 20.0 | 20.2 | 20.3 |
| adv-fire-hugger | 27.9 | 27.9 | 31.8 | 31.6 | 31.4 | 31.8 | 31.8 |
| adv-block-scrub | 33.4 | 33.4 | 33.3 | 33.1 | 33.4 | 33.4 | 33.3 |

#### Coordination candidates, archived suite

| policy | 1.0 | G1 | G4 | G5-official | G6 | G7 | T-wait |
|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 7.6 | 7.6 | 9.3 | 9.3 | 9.0 | 9.2 | 9.3 |
| static-shortest | 32.2 | 32.2 | 33.2 | 33.2 | 32.2 | 33.1 | 33.2 |
| fire-aware | 34.1 | 34.1 | 37.2 | 37.1 | 36.2 | 37.1 | 37.2 |
| congestion-aware | 37.7 | 37.7 | 40.7 | 40.5 | 39.6 | 40.6 | 40.7 |
| adv-reckless-fast | 30.8 | 30.8 | 31.6 | 31.6 | 30.7 | 31.5 | 31.6 |
| adv-abandon-far | 16.3 | 16.3 | 18.4 | 18.3 | 18.0 | 18.4 | 18.5 |
| adv-safe-slow | 3.2 | 3.2 | 3.5 | 3.6 | 3.5 | 3.7 | 2.7 |
| adv-late-start | 2.6 | 2.6 | 2.3 | 2.5 | 2.6 | 2.7 | 1.5 |
| adv-jostler | 34.0 | 34.0 | 37.1 | 37.1 | 36.2 | 37.0 | 37.1 |
| adv-herd | 25.2 | 25.2 | 27.4 | 27.6 | 27.1 | 27.7 | 27.8 |
| adv-wanderer | 11.3 | 11.3 | 10.7 | 11.0 | 11.5 | 11.6 | 11.7 |
| adv-hider | 1.0 | 1.0 | 5.5 | 5.5 | 5.4 | 5.5 | 5.5 |
| adv-fire-hugger | 36.2 | 36.2 | 38.1 | 38.0 | 37.0 | 38.1 | 38.2 |
| adv-block-scrub | 40.7 | 40.7 | 40.7 | 40.5 | 40.7 | 40.7 | 40.7 |

#### Coordination candidates: summary

| candidate | tau vs official (study) | tau vs official (archived) | tau study vs archived | top 3 (study) | congestion-aware rank study / archived | best adversary - congestion-aware |
|---|---|---|---|---|---|---|
| 1.0 | 0.966 | 0.966 | 0.782 | adv-block-scrub, congestion-aware, adv-fire-hugger | 2 / 2 | +5.2 |
| G1 | 0.966 | 0.966 | 0.782 | adv-block-scrub, congestion-aware, adv-fire-hugger | 2 / 2 | +5.2 |
| G4 | 1.000 | 1.000 | 0.814 | congestion-aware, adv-block-scrub, adv-fire-hugger | 1 / 1 | +0.0 |
| G5-official | 1.000 | 1.000 | 0.814 | congestion-aware, adv-block-scrub, adv-fire-hugger | 1 / 1 | +0.0 |
| G6 | 0.983 | 1.000 | 0.798 | adv-block-scrub, congestion-aware, adv-fire-hugger | 2 / 2 | +3.6 |
| G7 | 1.000 | 1.000 | 0.815 | adv-block-scrub, congestion-aware, adv-fire-hugger | 2 / 2 | +0.1 |
| T-wait | 1.000 | 1.000 | 0.814 | congestion-aware, adv-block-scrub, adv-fire-hugger | 1 / 1 | +0.0 |

#### Coordination candidates: requirement checks (synthetic profiles; every block belongs to an evacuee)

| requirement | 1.0 | G1 | G4 | G5-official | G6 | G7 | T-wait |
|---|---|---|---|---|---|---|---|
| R1 one avoidable casualty in 100, otherwise perfect, scores <= 60 | pass | pass | pass | pass | pass | pass | pass |
| R2 Time is not irrelevant: Time 0.5 (not severe) costs >= 10 points | pass | pass | pass | pass | pass | pass | FAIL |
| R3 2 casualties score below 1 casualty | pass | pass | pass | pass | pass | pass | pass |
| R4 10/100 casualties still >= 5 (unsafe runs stay separable) | pass | pass | pass | pass | pass | pass | FAIL |
| R5 10/100 casualties <= 35 | pass | pass | pass | pass | pass | pass | pass |
| R6 one stranded in 100 <= 85 and above one casualty | pass | pass | pass | pass | pass | pass | pass |
| R7 10/100 stranded <= 50 | pass | pass | pass | pass | pass | pass | pass |
| R8 realized bottleneck (Time 0.5 vs the capacity bound, block rate 0.3) costs >= 10 points | pass | pass | pass | pass | pass | pass | FAIL |
| R9 50% avoidable losses score 0 (catastrophic gate) | pass | pass | pass | pass | pass | pass | pass |
| R10 Time 0.1 with everything else perfect scores <= 60 | pass | pass | pass | pass | pass | pass | pass |
| R11 a savable casualty is not hidden by another person's unavoidable loss | pass | pass | pass | pass | pass | pass | pass |
| R12 no casualty + no severe failure beats perfect + 1 casualty (M* 2..1000, s 0..1) | pass | pass | pass | pass | pass | pass | pass |
| R13 severe realized bottleneck (Time 0.4, block rate 0.3), no loss: < 80 and >= 20 points lost | pass | pass | pass | pass | pass | pass | FAIL |
| R14 severe Time 0.4 / Hazard 0.3 / Replanning 0.4 / Path 0.4 alone: each < 80 | pass | pass | pass | pass | pass | pass | pass |
| R15 failed / missing / timeout episodes score 0 and stay in every mean (structural; tests) | pass | pass | pass | pass | pass | pass | pass |
| R8-old block rate 0.3 alone (no delay) costs >= 10 points | pass | pass | FAIL | FAIL | pass | pass | FAIL |
| R13-old block rate 0.3 alone: < 80 and >= 20 points lost | pass | pass | FAIL | FAIL | pass | pass | FAIL |
| R16 trajectory invariance (scrub delta exactly 0 in every episode) | FAIL | FAIL | pass | pass | FAIL | FAIL | pass |

| ordering | 1.0 | G1 | G4 | G5-official | G6 | G7 | T-wait |
|---|---|---|---|---|---|---|---|
| zero-loss, Time 0.6 > perfect quality + 1 casualty | pass | pass | pass | pass | pass | pass | FAIL |
| perfect > zero-loss but slow (Time 0.5) | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, severe realized bottleneck (Time 0.4) | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, high exposure (0.3) | pass | pass | pass | pass | pass | pass | pass |
| perfect > zero-loss, inefficient routing (Path 0.6) | pass | pass | pass | pass | pass | pass | FAIL |
| replanning 0.9 > static-good, invalidation failure (0.3) | pass | pass | pass | pass | pass | pass | pass |
| 1 stranded > 1 casualty | pass | pass | pass | pass | pass | pass | pass |
| 1 casualty > 2 casualties | pass | pass | pass | pass | pass | pass | pass |
| mild realized congestion (Time 0.9) > severe realized bottleneck (Time 0.4) | pass | pass | pass | pass | pass | pass | pass |
| mild delay (Time 0.8) > extreme delay (Time 0.1) | pass | pass | pass | pass | pass | pass | pass |

#### Does a Coordination component carry information beyond Time? (Spearman over episodes, moving policies)

|  | time | 1.0 coordination | G7 coordination | T-wait | path |
|---|---|---|---|---|---|
| time | 1.00 | 0.14 | 0.67 | 0.88 | 0.57 |
| 1.0 coordination | 0.14 | 1.00 | 0.31 | 0.20 | 0.01 |
| G7 coordination | 0.67 | 0.31 | 1.00 | 0.77 | 0.31 |
| T-wait | 0.88 | 0.20 | 0.77 | 1.00 | 0.23 |
| path | 0.57 | 0.01 | 0.31 | 0.23 | 1.00 |

(n = 1064 episodes)

### Adversarial checks: adversary minus congestion-aware (gaming check, not a selection criterion)

| adversary - congestion-aware | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| adv-reckless-fast | -4.2 | -4.4 | -4.4 | -4.7 | -5.1 | -5.8 | -4.4 | -4.1 | -5.0 |
| adv-abandon-far | -33.9 | -33.8 | -33.8 | -34.3 | -35.0 | -34.7 | -34.5 | -8.2 | -12.2 |
| adv-safe-slow | -38.0 | -34.5 | -34.5 | -34.0 | -29.4 | -37.2 | -36.9 | -26.2 | -48.4 |
| adv-late-start | -26.8 | -25.0 | -25.0 | -24.2 | -18.5 | -27.8 | -24.7 | -33.4 | -48.0 |
| adv-jostler | -10.4 | -10.5 | -10.5 | -10.8 | -11.2 | -11.3 | -10.7 | -3.2 | -4.7 |
| adv-herd | -14.0 | -14.7 | -14.7 | -14.8 | -13.8 | -15.6 | -13.6 | -10.0 | -16.6 |
| adv-wanderer | -36.0 | -33.8 | -33.8 | -32.9 | -27.2 | -35.1 | -34.2 | -24.6 | -55.6 |
| adv-hider | -24.1 | -24.7 | -24.7 | -24.9 | -24.4 | -25.6 | -24.0 | -30.7 | -33.9 |
| adv-fire-hugger | -1.3 | -1.4 | -1.4 | -1.4 | -1.5 | -1.9 | -1.3 | -1.5 | -2.8 |
| adv-block-scrub | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 |

### Synthetic single-failure profile

| case | F-official | A-multiplicative | B-hard-ceilings | C-safety-tiers | D-lexicographic | E-soft-min | F-strand-floor | additive | plain-product |
|---|---|---|---|---|---|---|---|---|---|
| perfect | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| 1 casualty / 100 | 56.5 | 56.5 | 56.5 | 56.5 | 56.5 | 62.6 | 56.5 | 99.4 | 99.0 |
| 2 casualties / 100 | 39.4 | 39.4 | 39.4 | 39.4 | 39.4 | 44.0 | 39.4 | 98.8 | 98.0 |
| 3 casualties / 100 | 30.0 | 30.0 | 30.0 | 30.0 | 30.0 | 33.5 | 30.0 | 98.1 | 97.0 |
| 5 casualties / 100 | 19.6 | 19.6 | 19.6 | 19.6 | 19.6 | 22.0 | 19.6 | 96.9 | 95.0 |
| 10 casualties / 100 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 10.2 | 9.1 | 93.8 | 90.0 |
| 25 casualties / 100 | 1.6 | 1.6 | 1.6 | 1.6 | 1.6 | 1.8 | 1.6 | 84.4 | 75.0 |
| 1 casualty / 4 | 10.7 | 10.7 | 10.7 | 10.7 | 10.7 | 11.9 | 10.7 | 84.4 | 75.0 |
| 1 stranded / 100 | 76.8 | 76.8 | 76.8 | 76.8 | 76.8 | 82.9 | 76.8 | 99.4 | 99.0 |
| 10 stranded / 100 | 30.3 | 30.3 | 30.3 | 30.3 | 30.3 | 33.9 | 30.3 | 93.8 | 90.0 |
| 1 savable dead + 1 unsavable stranded / 100 | 56.5 | 56.5 | 56.5 | 56.5 | 56.5 | 62.6 | 56.5 | 99.4 | 99.0 |
| Time 0.8 | 92.1 | 89.4 | 89.4 | 95.4 | 99.5 | 93.0 | 96.5 | 96.2 | 80.0 |
| Time 0.5 (threshold) | 77.5 | 70.7 | 70.7 | 87.3 | 98.5 | 66.6 | 89.4 | 90.6 | 50.0 |
| Time 0.4 (severe) | 45.7 | 63.2 | 63.2 | 84.0 | 98.2 | 54.0 | 55.2 | 88.8 | 40.0 |
| Time 0.25 | 24.0 | 50.0 | 50.0 | 78.2 | 97.5 | 34.0 | 32.0 | 85.9 | 25.0 |
| Time 0.1 | 6.8 | 31.6 | 31.6 | 70.2 | 96.6 | 13.6 | 11.0 | 83.1 | 10.0 |
| block rate 0.3, no delay (diagnostic only) | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| realized bottleneck: Time 0.5, block rate 0.3 | 77.5 | 70.7 | 70.7 | 87.3 | 98.5 | 66.6 | 89.4 | 90.6 | 50.0 |
| severe realized bottleneck: Time 0.4, block rate 0.3 | 45.7 | 63.2 | 63.2 | 84.0 | 98.2 | 54.0 | 55.2 | 88.8 | 40.0 |
| avoidable hazard 0.1 | 97.4 | 96.5 | 96.5 | 98.5 | 99.8 | 98.1 | 98.9 | 98.8 | 90.0 |
| avoidable hazard 0.3 (severe) | 68.4 | 88.8 | 80.0 | 95.1 | 99.4 | 89.9 | 71.9 | 96.2 | 70.0 |
| Path 0.6 | 93.9 | 91.8 | 91.8 | 96.5 | 99.6 | 88.5 | 97.3 | 97.5 | 60.0 |
| Path 0.4 (severe) | 57.2 | 85.8 | 80.0 | 93.8 | 99.3 | 66.1 | 60.9 | 96.2 | 40.0 |
| Replanning 0.8 | 96.0 | 94.6 | 94.6 | 97.6 | 99.7 | 95.5 | 98.2 | 97.8 | 80.0 |
| Replanning 0.3 (severe) | 38.5 | 74.0 | 74.0 | 88.7 | 98.7 | 45.3 | 43.6 | 92.2 | 30.0 |
| every component at its threshold | 64.7 | 55.3 | 55.3 | 80.6 | 97.8 | 59.8 | 82.7 | 80.6 | 9.4 |

### Component correlation (Spearman, episodes with every component defined, moving policies)

|  | safety | time | hazard | path | replanning |
|---|---|---|---|---|---|
| safety | 1.00 | 0.09 | -0.13 | -0.33 | 0.51 |
| time | 0.09 | 1.00 | 0.03 | 0.54 | -0.13 |
| hazard | -0.13 | 0.03 | 1.00 | 0.23 | -0.52 |
| path | -0.33 | 0.54 | 0.23 | 1.00 | -0.55 |
| replanning | 0.51 | -0.13 | -0.52 | -0.55 | 1.00 |

(n = 238 episodes)

### Score spread (F-official)

moving policies: min 10.6, max 48.6, range 38.0, sd 13.5
A-multiplicative: range 34.5
C-safety-tiers: range 34.3
D-lexicographic: range 35.0
E-soft-min: range 37.2

### Per-family ranking (F-official; family means over seeds)

| policy | generated:single_exit_high_density | generated:single_exit_low_density | generated:split_flow_across_exits | archived:single_exit_high_density | archived:single_exit_low_density | archived:split_flow_across_exits | public:central_ignition | public:crowd_100 | public:maze_obstacle_routing | public:multiple_exits_unequal_distance | public:multiple_ignitions | public:narrow_corridor_bottleneck | public:route_invalidation_replanning | public:single_exit_high_density | public:single_exit_low_density | public:threatened_nearest_exit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always-stay | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| random-legal | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| nearest-greedy | 20.1 | 18.0 | 22.7 | 8.3 | 1.0 | 18.5 | 4.1 | 0.9 | 0.0 | 100.0 | 9.0 | 26.0 | 32.9 | 3.2 | 99.1 | 0.0 |
| static-shortest | 20.1 | 38.4 | 28.2 | 8.0 | 51.0 | 40.7 | 4.1 | 0.9 | 99.0 | 100.0 | 9.0 | 26.0 | 32.9 | 3.2 | 99.1 | 0.0 |
| fire-aware | 20.5 | 38.4 | 30.4 | 7.9 | 57.1 | 46.3 | 4.1 | 1.0 | 99.0 | 100.0 | 9.0 | 26.0 | 44.2 | 3.2 | 99.1 | 18.0 |
| congestion-aware | 23.0 | 44.4 | 31.9 | 8.3 | 68.1 | 45.2 | 7.4 | 17.0 | 98.8 | 100.0 | 21.0 | 95.5 | 44.2 | 31.2 | 99.1 | 18.0 |
| adv-reckless-fast | 21.5 | 44.4 | 29.7 | 7.8 | 51.0 | 35.9 | 7.4 | 7.9 | 98.8 | 100.0 | 9.0 | 95.5 | 32.9 | 31.2 | 99.1 | 0.0 |
| adv-abandon-far | 19.1 | 33.4 | 18.4 | 8.0 | 30.0 | 16.9 | 7.4 | 2.0 | 32.4 | 15.7 | 9.0 | 26.2 | 4.8 | 7.9 | 10.7 | 3.4 |
| adv-safe-slow | 0.9 | 17.2 | 14.2 | 1.0 | 3.3 | 6.7 | 6.6 | 0.0 | 0.0 | 42.4 | 6.6 | 0.0 | 0.0 | 0.2 | 45.1 | 5.2 |
| adv-late-start | 0.0 | 0.7 | 2.1 | 0.0 | 1.8 | 5.7 | 0.0 | 3.3 | 87.9 | 57.2 | 0.0 | 0.0 | 68.0 | 9.3 | 55.1 | 0.0 |
| adv-jostler | 20.5 | 38.4 | 30.4 | 7.9 | 57.1 | 46.2 | 4.1 | 4.4 | 99.0 | 100.0 | 9.0 | 26.0 | 44.2 | 3.2 | 99.1 | 18.0 |
| adv-herd | 20.5 | 38.4 | 20.7 | 7.9 | 57.1 | 17.6 | 0.0 | 0.1 | 99.0 | 80.6 | 0.0 | 26.0 | 44.2 | 3.2 | 99.1 | 17.5 |
| adv-wanderer | 0.5 | 14.4 | 17.7 | 0.5 | 15.9 | 16.5 | 0.0 | 1.8 | 0.0 | 41.8 | 1.2 | 1.7 | 8.4 | 3.3 | 72.4 | 1.0 |
| adv-hider | 17.3 | 15.6 | 27.7 | 7.6 | 1.0 | 7.8 | 6.4 | 0.0 | 0.0 | 93.5 | 0.0 | 0.0 | 0.0 | 48.7 | 91.2 | 17.4 |
| adv-fire-hugger | 23.0 | 40.9 | 30.9 | 5.3 | 72.6 | 36.1 | 7.4 | 16.2 | 98.0 | 100.0 | 5.9 | 95.5 | 44.2 | 50.1 | 99.1 | 3.4 |
| adv-block-scrub | 23.0 | 44.4 | 31.9 | 8.3 | 68.1 | 45.2 | 7.4 | 17.0 | 98.8 | 100.0 | 21.0 | 95.5 | 44.2 | 31.2 | 99.1 | 18.0 |
- generated:single_exit_high_density: congestion-aware 23.0, adv-fire-hugger 23.0, adv-block-scrub 23.0
- generated:single_exit_low_density: congestion-aware 44.4, adv-reckless-fast 44.4, adv-block-scrub 44.4
- generated:split_flow_across_exits: congestion-aware 31.9, adv-block-scrub 31.9, adv-fire-hugger 30.9
- archived:single_exit_high_density: congestion-aware 8.3, adv-block-scrub 8.3, nearest-greedy 8.3
- archived:single_exit_low_density: adv-fire-hugger 72.6, congestion-aware 68.1, adv-block-scrub 68.1
- archived:split_flow_across_exits: fire-aware 46.3, adv-jostler 46.2, congestion-aware 45.2
- public:central_ignition: congestion-aware 7.4, adv-reckless-fast 7.4, adv-abandon-far 7.4
- public:crowd_100: congestion-aware 17.0, adv-block-scrub 17.0, adv-fire-hugger 16.2
- public:maze_obstacle_routing: static-shortest 99.0, fire-aware 99.0, adv-jostler 99.0
- public:multiple_exits_unequal_distance: nearest-greedy 100.0, static-shortest 100.0, fire-aware 100.0
- public:multiple_ignitions: congestion-aware 21.0, adv-block-scrub 21.0, nearest-greedy 9.0
- public:narrow_corridor_bottleneck: congestion-aware 95.5, adv-reckless-fast 95.5, adv-fire-hugger 95.5
- public:route_invalidation_replanning: adv-late-start 68.0, fire-aware 44.2, congestion-aware 44.2
- public:single_exit_high_density: adv-fire-hugger 50.1, adv-hider 48.7, congestion-aware 31.2
- public:single_exit_low_density: nearest-greedy 99.1, static-shortest 99.1, fire-aware 99.1
- public:threatened_nearest_exit: fire-aware 18.0, congestion-aware 18.0, adv-jostler 18.0

### Sensitivity grids: requirement feasibility and rank stability

#### grid safety: feasible 26 of 81
every combination: Kendall tau min 0.983, mean 0.996; max |delta score| 13.2
requirements failed (combinations): R1 27, R11 27, R4 15, R6 30, R7 9
official {'casualty_incident_factor': '0.6', 'casualty_rate_exponent': '4', 'evacuation_exponent': '2', 'stranding_incident_factor': '0.8'} feasible: True
Kendall tau vs official over feasible: min 0.983, mean 0.996; max |delta score| 8.4
- casualty_incident_factor: 0.5: 8, 0.6: 18, 0.7: 0
- casualty_rate_exponent: 2: 12, 4: 8, 6: 6
- evacuation_exponent: 1: 10, 2: 8, 3: 8
- stranding_incident_factor: 0.7: 13, 0.8: 13, 0.9: 0

#### grid gates: feasible 36 of 81
every combination: Kendall tau min 0.898, mean 0.971; max |delta score| 6.9
requirements failed (combinations): R13 27, R14 45
official {'severe_factor': '0.8', 'severe_exponent': '1', 'severe_time': '1/2', 'severe_hazard_rate': '1/4'} feasible: True
Kendall tau vs official over feasible: min 0.983, mean 0.999; max |delta score| 6.8
- severe_factor: 0.7: 12, 0.8: 12, 0.9: 12
- severe_exponent: 0.5: 12, 1: 12, 2: 12
- severe_time: 1/3: 0, 1/2: 18, 2/3: 18
- severe_hazard_rate: 1/5: 18, 1/4: 18, 1/3: 0

### Quality-weight perturbation (each weight x0.5 and x1.5, renormalized)

| perturbation | Kendall tau | max |delta score| |
|---|---|---|
| time x0.5 | 1.000 | 1.3 |
| time x1.5 | 1.000 | 0.7 |
| hazard x0.5 | 1.000 | 0.6 |
| hazard x1.5 | 1.000 | 0.5 |
| path x0.5 | 1.000 | 0.3 |
| path x1.5 | 1.000 | 0.3 |
| replanning x0.5 | 1.000 | 0.1 |
| replanning x1.5 | 1.000 | 0.1 |

### Rank agreement between candidate designs (Kendall tau vs F-official)

| candidate | Kendall tau |
|---|---|
| F-official | 1.000 |
| A-multiplicative | 0.983 |
| B-hard-ceilings | 0.983 |
| C-safety-tiers | 0.949 |
| D-lexicographic | 0.915 |
| E-soft-min | 1.000 |
| F-strand-floor | 0.966 |
| additive | 0.831 |
| plain-product | 0.898 |

### Official ranking (F-official, both study suites) and archived validation

| rank | policy | EMBER (study) | EMBER (archived) | rank (archived) |
|---|---|---|---|---|
| 1 | congestion-aware | 48.6 | 40.5 | 1 |
| 2 | adv-block-scrub | 48.6 | 40.5 | 2 |
| 3 | adv-fire-hugger | 47.3 | 38.0 | 3 |
| 4 | adv-reckless-fast | 44.4 | 31.6 | 7 |
| 5 | adv-jostler | 38.2 | 37.1 | 5 |
| 6 | fire-aware | 37.9 | 37.1 | 4 |
| 7 | static-shortest | 35.5 | 33.2 | 6 |
| 8 | adv-herd | 34.6 | 27.6 | 8 |
| 9 | nearest-greedy | 25.8 | 9.3 | 11 |
| 10 | adv-hider | 24.4 | 5.5 | 12 |
| 11 | adv-late-start | 21.8 | 2.5 | 14 |
| 12 | adv-abandon-far | 14.7 | 18.3 | 9 |
| 13 | adv-wanderer | 12.6 | 11.0 | 10 |
| 14 | adv-safe-slow | 10.6 | 3.6 | 13 |
| 15 | always-stay | 0.0 | 0.0 | 15 |
| 16 | random-legal | 0.0 | 0.0 | 16 |

Kendall tau, study vs archived ranking: 0.814
