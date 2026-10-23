# EMBERBench results portal

A dependency-free, GitHub Pages-compatible explorer of official **public development** score reports. The portal does not present a hidden-suite leaderboard.

## Pages

- `index.html`: overview, leaders, interactive EMBER bars, relationship selector, scenario heatmap, provenance.
- `leaderboard.html`: searchable, sortable table with type/provider filters, score CIs, API metrics and comparison selection. Query parameters preserve filters and sorting.
- `compare.html?models=ID,ID`: 2–4 model profiles, components, scenarios, world scores, robustness, congestion and API metrics. Exactly two entries also show descriptive paired world differences.
- `model.html?id=ID`: model profile with nine evidence sections and reproducibility metadata.
- `scenarios.html?scenario=ID`: documented public scenarios, generated populations/world counts, sortable scenario rankings and heatmap ordering.
- `methodology.html`: scoring, frozen environments, replay verification, uncertainty, public/hidden distinction and provenance.

All routes use relative assets and ordinary HTML paths. Model links stay in the same tab and support normal browser open-in-new-tab behavior. Color theme and up to four selected IDs are stored locally when browser storage is available.

## Data and scientific boundaries

`data/results.js` is generated exclusively by `tools/build_results_site.py`. Never enter results in UI source files. The module format avoids using a `.json` extension reserved by the public-asset guard for benchmark assets.

The redesign retains the existing generated snapshot byte-for-byte. Original run directories are not included in this checkout, so it cannot regenerate that snapshot or recover missing provenance. Missing constants/manifest/report digests display **—**, with an explicit notice. No evaluation date or latest-to-oldest order is inferred. The existing snapshot contains one trial per model; reported CIs do not capture between-trial variability from a single trial.

Future builds expose schema version 2, suite/manifest/constants digests, deterministic source report digests, trial counts, families, failure/component details, and authoritative ranks computed before display rounding. Legacy data uses the scoring tiebreak policy on the existing snapshot; its current score buckets and Safety differences are unambiguous. Rank is descriptive, never a significance claim.

Paired differences match `(index, scenario_id, seed)` and average repeated trials within each world. Their unweighted mean is descriptive and differs from the family-balanced aggregate score. Incomplete trial groups are excluded and coverage is reported. No official comparison inference is present in the snapshot.

API usage is operational metadata, not a score input. Missing usage/cost is not zero; reference-agent compute cost is unmeasured. Mean decision latency is not measured streaming time to first token. Resumed run metadata can cover only the final invocation and include provider waits. **≈** means estimated API cost at supplied list prices. Cost-frontier lines are descriptive and mix reported/estimated invocation metadata; they do not imply price guarantees.

## Generate data

Use Python 3.12+ and every original finished run directory, each containing `scores/` and a matching `summary.json` for API models:

```bash
python tools/build_results_site.py --run ../pack-run --run ../jev-run --run ../subq-run --out docs/site/data/results.js
```

Optional `--price MODEL_ID=INPUT,OUTPUT` supplies USD per million tokens where the API reported no cost. Estimation requires both token counts; measured zero cost is preserved. Current snapshot prices were supplied by the previous generation, not embedded in the UI.

Generation rejects nonofficial reports, mismatched packs, scoring versions, constants, suite/manifest hashes, scheduled worlds, model identities and summary provenance. It uses explicit field allowlists and emits no endpoint URLs, keys, cache paths or nondeterministic timestamps. This validates report consistency; it does not replace the evaluator's replay authentication.

## Preview and tests

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory docs/site
node --test tests/site.test.mjs
python -m pytest -q tests/test_results_site.py
python -m mypy --strict tools/build_results_site.py tests/test_results_site.py
python tools/check_public_assets.py --untracked
```

Open `http://127.0.0.1:8765/`. The site must be served over HTTP because it uses JavaScript modules. No application build step, external fonts, analytics, CDNs, or production JavaScript dependencies are required. CI runs the Node tests separately from the Python test matrix.

## Design research

The live Artificial Analysis leaderboard, model catalogue and model profile informed analytical hierarchy, filter discoverability, consistent chart controls and explicit unavailable values. EMBERBench uses its own content, structure, slate/teal/copper palette, typography, CSS and mark. No reference assets, branding, text or styles were copied.

Reference pages inspected: https://artificialanalysis.ai/leaderboards/models, https://artificialanalysis.ai/models, https://artificialanalysis.ai/models/muse-wonder-full.

## Browser verification inventory

Validate desktop, tablet and narrow mobile; light/dark themes; shared navigation and mobile menu; direct paths and query parameters; search/type/provider filter cycles; ascending/descending sorts; missing API values; table horizontal scroll/focus preservation; empty search; invalid model/metric IDs; 2–4 selections and fifth-selection rejection; profile navigation; all relationship charts, highlights, hover/focus tooltips and Escape; heatmap keyboard visibility; scenario navigation/sorting; and console errors.

Desktop Chromium checks and narrow viewport captures are recorded in the task's screenshot artifacts. Public development scope and one-trial uncertainty caveats remain visible throughout.
