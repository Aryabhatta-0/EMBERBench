# Results portal validation

The redesign changes presentation and the report-to-site export only. It does not modify scoring, simulator semantics, suites, pack hashes, reference results, model results, or `.claude/`. `data/results.js` is regenerated from the original run directories and adds 10 OpenRouter decision models (see below). The 15 models of the PR #36 snapshot keep identical scores, Safety, Quality and per-world outcomes.

## Scope and files

- Six entry points: overview, leaderboard, comparison, model profile, scenarios, methodology (`*.html` and their corresponding `*.js`).
- Shared shell/formatting in `common.js`, pure analysis in `analytics.js`, accessible SVG/bar/heatmap renderers in `charts.js`, and responsive themes in `style.css`.
- `tools/build_results_site.py`: fail-closed identity/provenance validation, unknown operational values preserved, official ranks assigned before rounding, schema version 2 metadata for future generation.
- `tests/test_results_site.py`, `tests/site.test.mjs`, and the Node job in `.github/workflows/tests.yml` cover the data and UI assumptions. `README.md` documents operation and limitations.

## Automated checks

- Full Python run: **1,681 passed, 5 skipped** before the final additional rounding regression. The five skips were two wheel tests without pip, one Windows symlink privilege check, the opt-in full-pack evaluation, and the opt-in release wrapper.
- Installed the pinned pip and reran the two wheel tests: **2 passed**.
- Final focused builder, release, and leak-guard tests: **118 passed, 1 skipped**. The skipped release wrapper is covered separately by the successful full release tool run. Builder subset: **40 passed**.
- Frontend Node tests: **16 passed**, including missing values, sorting, filters, selection limits, rank precision, trial-averaged pairing, incomplete pairs, cost frontier, escaping, and all six page/module contracts.
- `mypy --strict`: **74 source files passed**. Explicit strict checks on the builder, remote runner and builder tests also passed, with `MYPYPATH=src;tests;tools;tools/experiments/v1` for the runner's research helpers on Windows.
- `tools/check_release.py --dist dist --require-license`: **passed**. Two clean builds, matching normalized wheel/sdist members, byte-identical wheels, source/sdist wheel parity, clean install, version contracts, public suite, and 31 regression digest entries verified.
- `tools/check_public_assets.py --untracked --dist dist`: **0 violations** across checkout, wheel and sdist.
- JavaScript syntax and Git whitespace checks passed. No production dependencies or external asset requests were added.

## Added OpenRouter decision models

`tools/experiments/remote/run_remote.py --api decision --base-url https://openrouter.ai/api/alpha/decisions --provider openrouter`, adapter `r1`, the frozen public pack (digest `ca18c895…b506e`, 19 unique worlds) and the official scoring 1.0. `nace-ai/drex-v2.0` accepts at most 16 questions per request, so it ran with `--max-people 16`. Cost is the cost that OpenRouter reported.

| Model (resolved) | EMBER Score (95% CI) | Safety | Quality | Evacuated | Requests | Cost (USD) | Wall clock |
|---|---|---|---|---|---|---|---|
| `nace-ai/drex-v2.0-20261009` | 32.66 (30.79–35.42) | 0.3568 | 0.7554 | 195 | 725 | 0.118 | 721 s |
| `liquid/d1-ultra-20261015` | 32.39 (32.35–32.45) | 0.3339 | 0.7154 | 145 | 753 | 0.098 | 587 s |
| `respan/span-02-20261015` | 23.97 (23.93–24.03) | 0.2562 | 0.6853 | 159 | 730 | 0.098 | 647 s |
| `inception/mercury-decide-20260930` | 20.68 (20.64–20.75) | 0.2240 | 0.6602 | 145 | 803 | 0.058 | 700 s |
| `liquid/d1-20260930` | 19.94 (19.93–19.98) | 0.2119 | 0.6754 | 154 | 753 | 0.042 | 587 s |
| `togethercomputer/tev1-4b-experimental-20260923` | 19.84 (19.80–19.90) | 0.2231 | 0.7130 | 169 | 677 | 0.310 | 657 s |
| `perplexity/pplx-decider-flash-v1.2-20261016` | 18.87 (18.87–18.87) | 0.2225 | 0.6511 | 141 | 700 | 0.103 | 572 s |
| `cloudflare/clef-flash` | 13.54 (13.42–13.75) | 0.1414 | 0.7384 | 133 | 447 | 0.049 | 394 s |
| `cloudflare/clef-omni-20261009` | 11.77 (11.77–11.77) | 0.1197 | 0.7055 | 108 | 540 | 0.361 | 334 s |
| `upstage/solar-decide-flash-20261008` | 0.21 (0.21–0.21) | 0.0023 | 0.4169 | 67 | 559 | 0.442 | 768 s |

The `upstage/solar-decide-flash` answers were inspected: they are valid, varied moves with low model confidence, not a degenerate adapter output. Rebuild checks: builder and remote-runner tests **56 passed**, Node tests **16 passed**, `mypy --strict` on the builder and its tests passed, public-asset guard **0 violations**.

## Browser checks

Chromium covered all six routes at **320, 390, 768, 1024 and 1440 pixels**: 30 route/width combinations, with no page-level horizontal overflow. Tables and wide charts scroll within their own regions. Light/dark views and final screenshots were inspected.

Exercised search, entry/provider filters, reset, missing API sorting, header sort reversal, horizontal scroll and keyboard focus retention, 2–4 comparison choices, fifth-choice rejection, model browsing and invalid IDs, scenario deep links/filter/sorting, mobile menu navigation, all five scatter relationships, highlights, hover/focus tooltips and Escape dismissal, heatmap keyboard visibility, and methodology table access. No console errors were observed. Direct paths/query strings loaded beneath an `/EMBERBench/` static mount as well as the site root.

Independent scientific, UI/UX, security/leakage, and responsive/accessibility reviews were completed. Findings fixed included rounded-rank ambiguity for future exports, scenario URL escaping, focus/scroll loss during rerender, heatmap focus occlusion, mobile profile overflow, tooltip hover persistence, sticky-header anchor offsets, and keyboard access to methodology tables. Contrast samples passed in both themes. This was not a screen-reader certification or a complete automated WCAG audit.

## Explicit limits

- Run directories are not committed. The snapshot was regenerated from them with `tools/build_results_site.py`, never hand-edited.
- Nine of the new OpenRouter runs were resumed after an HTTP 402 (no credits) on a first key. Their first segment sent 2–27 accepted requests. Their API usage and cost cover the final invocation only, so they slightly undercount.
- No evaluation timestamps or official paired-inference output are available. There is no invented latest-model ordering, p-value, significance badge, or hidden-suite claim.
- Current reports contain one trial per model. Their intervals cannot measure unobserved between-trial variability.
- API usage may cover the final invocation of a resumed run. List-price cost estimates are marked `≈`; reference-agent local compute is unmeasured.
- GitHub Pages deployment runs after a maintainer merges the follow-up PR. No merge or production deployment was performed by this task.
