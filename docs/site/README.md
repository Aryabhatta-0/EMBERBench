# EMBERBench results website (first version)

A static website with the EMBER Scores of every evaluated model on the V1 public development pack.

The pages are written for readers who do not know the benchmark: plain words, and charts for the
important numbers.

- `index.html`: the overall ranking as a bar chart, with the people saved and the cost of each model. A
  filter shows all models, AI decision models or reference programs. Two more charts compare the cost
  and the time to finish of the AI decision models. Select a model to open its page in a new window.
- `model.html?id=<model-id>`: one model. It shows the people saved, the EMBER Score, the cost, the time
  to finish, the response time and the tokens used. Comparison charts show this model in color and
  every other model in gray. A chart shows the score in each type of fire against the best AI decision
  model. The score parts, robustness, crowding, API use and every world are in the technical details.
  Every chart has a hover tooltip.
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

Cost is what the API reported (OpenRouter). For a run whose API reported no cost, give the published
list price in USD per million input and output tokens with `--price MODEL_ID=IN,OUT`. The site marks
that cost with ≈ and a striped bar. The current data uses `--price typesafe-jev-latest-choice-r1=0.042,0`
and `--price subq-subq-snap-preview-choice-r1=0.05,0`.

The response time is the mean wait for one decision. The decision APIs do not stream, so the first token
arrives with the whole answer: the response time is the time to first token. The time to finish is the
wall-clock time of the run. A run that was resumed records only its last part, and it includes the
provider's rate-limit waits.

## Preview

```bash
python -m http.server 8765 --directory docs/site
```

Then open `http://localhost:8765/`.

The suite is public, so every number on the site is a development result, not a hidden-suite
leaderboard score.
