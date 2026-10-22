# EMBERBench results website (first version)

A static website with the EMBER Scores of every evaluated model on the V1 public development pack.

- `index.html`: the leaderboard. A filter shows all models, decision models or reference agents.
  Select a model to open its metrics in a new window.
- `model.html?id=<model-id>`: one model. It shows the headline numbers, people outcomes, the score per
  scenario, the score components, the score per world, the robustness counts, congestion diagnostics
  and, for API models, the API use. Each chart has a hover tooltip and a table view.
- `data/results.js`: the data of both pages, made by `tools/build_results_site.py`. It is a JavaScript
  module, not a `.json` file, because the public-asset guard keeps `.json` files for benchmark assets.

The site has no build step and no external dependency. It is plain HTML, CSS and JavaScript modules,
so any static host can serve it (for example GitHub Pages). It must be served over HTTP: browsers do
not load JavaScript modules from files opened directly from disk.

## Update the data

Give every finished run directory (each with `scores/`, and `summary.json` for API runs):

```bash
python tools/build_results_site.py --run ../pack-run --run ../jev-run --run ../subq-run --run ../decision-run-1 --out docs/site/data/results.js
```

The build stops if a score report is not official or was scored against another environment pack.

## Preview

```bash
python -m http.server 8765 --directory docs/site
```

Then open `http://localhost:8765/`.

The suite is public, so every number on the site is a development result, not a hidden-suite
leaderboard score.
