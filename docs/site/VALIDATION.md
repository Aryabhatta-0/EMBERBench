# Results portal validation

The redesign changes presentation and the report-to-site export only. It does not modify scoring, simulator semantics, suites, pack hashes, reference results, model results, or `.claude/`. `data/results.js` is byte-identical to the snapshot merged in PR #36.

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

## Browser checks

Chromium covered all six routes at **320, 390, 768, 1024 and 1440 pixels**: 30 route/width combinations, with no page-level horizontal overflow. Tables and wide charts scroll within their own regions. Light/dark views and final screenshots were inspected.

Exercised search, entry/provider filters, reset, missing API sorting, header sort reversal, horizontal scroll and keyboard focus retention, 2–4 comparison choices, fifth-choice rejection, model browsing and invalid IDs, scenario deep links/filter/sorting, mobile menu navigation, all five scatter relationships, highlights, hover/focus tooltips and Escape dismissal, heatmap keyboard visibility, and methodology table access. No console errors were observed. Direct paths/query strings loaded beneath an `/EMBERBench/` static mount as well as the site root.

Independent scientific, UI/UX, security/leakage, and responsive/accessibility reviews were completed. Findings fixed included rounded-rank ambiguity for future exports, scenario URL escaping, focus/scroll loss during rerender, heatmap focus occlusion, mobile profile overflow, tooltip hover persistence, sticky-header anchor offsets, and keyboard access to methodology tables. Contrast samples passed in both themes. This was not a screen-reader certification or a complete automated WCAG audit.

## Explicit limits

- Original official run directories are absent from this checkout. The existing generated snapshot is retained, not reverse-engineered or hand-edited. New provenance/schema fields appear only after regeneration from original reports; absent fields display `—` now.
- No evaluation timestamps or official paired-inference output are available. There is no invented latest-model ordering, p-value, significance badge, or hidden-suite claim.
- Current reports contain one trial per model. Their intervals cannot measure unobserved between-trial variability.
- API usage may cover the final invocation of a resumed run. List-price cost estimates are marked `≈`; reference-agent local compute is unmeasured.
- GitHub Pages deployment runs after a maintainer merges the follow-up PR. No merge or production deployment was performed by this task.
