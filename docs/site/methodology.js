import { loadResults, shell, esc, fmt } from "./common.js";

const REPO = "https://github.com/Aryabhatta-0/EMBERBench/blob/main/";
const source = (path, label) => `<a href="${REPO}${path}">${esc(label)}</a>`;

const components = [
  [
    "Evacuation",
    "Safety",
    "The fraction of individually savable people evacuated, squared. This factor is not the raw evacuation rate.",
  ],
  [
    "Casualty",
    "Safety",
    "A penalty factor for avoidable deaths, using both their count and their share of individually savable people.",
  ],
  [
    "Stranding",
    "Safety",
    "A penalty factor for savable people still active or stranded at the time limit. Deaths receive a stronger penalty than stranding.",
  ],
  [
    "Time",
    "Quality · weight 3/8",
    "A capacity-aware lower bound divided by total evacuation time, over evacuees. The bound accounts for unavoidable queues at exits and narrow passages.",
  ],
  [
    "Hazard",
    "Quality · weight 2/8",
    "Penalizes danger-zone person-states above the privileged minimum, normalized by evacuee person-time. The danger zone is within two Manhattan cells of fire; distance ignores walls.",
  ],
  [
    "Path",
    "Quality · weight 1/8",
    "The fewest fire-aware cell moves divided by actual cell moves, over evacuees. A detour made necessary by fire is not penalized.",
  ],
  [
    "Replanning",
    "Quality · weight 2/8",
    "The best possible remaining evacuation time divided by actual resolution time after recoverable route invalidation. Lost people are censored at the time limit. Undefined when no recoverable event occurs.",
  ],
];

async function render() {
  const data = await loadResults();
  shell(data, "methodology");
  const main = document.querySelector("#main");
  const worlds = new Set(
    data.models.flatMap((m) => m.episodes.map((e) => e.index)),
  ).size;
  const trialCounts = [
    ...new Set(
      data.models.map((m) => m.totals.trials).filter((n) => n != null),
    ),
  ].sort((a, b) => a - b);
  const trialCopy =
    trialCounts.length === 1
      ? `The published data contains ${esc(trialCounts[0])} trial${trialCounts[0] === 1 ? "" : "s"} per model.`
      : `Published trial counts are ${trialCounts.map((n) => esc(n)).join(", ")}; each model page reports its count.`;

  main.innerHTML = `
    <header class="page-head">
      <p class="eyebrow">Evaluation protocol</p>
      <h1>How EMBERBench measures decisions</h1>
      <p class="muted">A shared environment, authenticated trajectories, and a score that makes avoidable loss expensive.</p>
    </header>

    <section class="panel prose" aria-labelledby="process-title">
      <div class="section-head"><h2 id="process-title">From observation to evidence</h2></div>
      <p>A decision model chooses how each person moves through a grid as fire spreads. It receives the current state and known simulation rules, then returns actions for the next step. This repeats until the episode ends.</p>
      <ol class="pipeline" aria-label="Evaluation pipeline">
        <li><strong>Observation</strong><span>Current map, fire, and people</span></li>
        <li><strong>Model / agent</strong><span>Chooses the next move</span></li>
        <li><strong>Action</strong><span>Moves or stays for each person</span></li>
        <li><strong>Simulator</strong><span>Resolves movement and fire</span></li>
        <li><strong>Replay verification</strong><span>Authenticates the recorded trajectory</span></li>
        <li><strong>EMBER Score</strong><span>Scores the verified outcomes</span></li>
      </ol>
      <p class="muted">Observation → Model / agent → Action → Simulator repeats at every step. Replay verification and scoring follow the recorded episode.</p>
      <p>${source("docs/model-interface.md", "Model interface")} · ${source("docs/simulation-contract.md", "Simulation contract")} · ${source("docs/scoring.md", "Normative scoring definition")}</p>
    </section>

    <div class="grid2">
      <section class="panel prose" aria-labelledby="frozen-title">
        <h2 id="frozen-title">The same frozen worlds</h2>
        <p>Every model in this portal uses the same EnvironmentPack: scenario bytes, evaluation schedule, simulation versions, and fire-world digests are pinned before evaluation.</p>
        <p>Fire is exogenous: people and actions cannot change it. For the same scheduled world, every model encounters the same burning cells at the same step. The model receives neither the evaluator seed nor future fire states.</p>
        <p>People’s positions, collisions, queues, and evacuation outcomes depend on the actions. Congestion can therefore differ even when the fire world is identical.</p>
        <p>${source("docs/environment-pack.md", "EnvironmentPack protocol")} · ${source("docs/v1-fairness-spec.md", "Information and fairness rules")}</p>
      </section>
      <section class="panel prose" aria-labelledby="verified-title">
        <h2 id="verified-title">Replay before scoring</h2>
        <p>The evaluator replays every recorded run and recomputes its metrics. The manifest must target the suite, stored metrics must match the replay, and the fire trajectory must match the frozen pack. Inconsistent evidence is rejected.</p>
        <p>A scheduled episode with no authenticated result scores zero, including missing, invalid-output, model-error, and timeout failures. An episode in which nobody is individually savable is excluded for every model.</p>
        <p>The site builder accepts official pack-scored reports from one shared pack and scoring version. “Official” describes the scoring validation; these remain public-development results.</p>
        <p>${source("docs/integrity.md", "Integrity and provenance")} · ${source("tools/build_results_site.py", "Generated-data builder")}</p>
      </section>
    </div>

    <section class="panel prose" aria-labelledby="score-title">
      <div class="section-head"><h2 id="score-title">One score, seven visible components</h2><span class="badge">0–100 points · higher is better</span></div>
      <p>EMBER Score measures benchmark performance. It is not a percentage of people evacuated. Each episode combines Safety, Quality, and an additional ceiling for severe quality failures.</p>
      <p><strong>Episode score = 100 × Safety × Quality<sup>β</sup> × Severity</strong>, where β ≈ ${fmt(Math.log(0.6) / Math.log(0.5), 3)} under scoring version ${esc(data.scoring_version)}.</p>
      <div class="grid2">
        <div><h3>Safety</h3><p>Evacuation × Casualty × Stranding. Loss is attributed person by person against a relaxed, single-person escape optimum with full future-fire knowledge. Safety becomes zero when at least half of the individually savable people die or remain stranded.</p></div>
        <div><h3>Quality</h3><p>A weighted geometric mean of Time, Hazard, Path, and Replanning. Undefined components are omitted and their weights redistributed. The privileged references stay evaluator-side and never reach the agent.</p></div>
      </div>
      <p class="muted">Every component is a factor from 0 to 1; higher is better. A casualty factor near 1 means a smaller penalty, not more deaths. Coverage shows where a component is defined.</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Methodology reference table"><table>
        <caption>Score components and their role</caption>
        <thead><tr><th scope="col">Component</th><th scope="col">Role</th><th scope="col">What it measures</th></tr></thead>
        <tbody>${components.map(([name, role, description]) => `<tr><th scope="row">${esc(name)}</th><td>${esc(role)}</td><td>${esc(description)}</td></tr>`).join("")}</tbody>
      </table></div>
      <p>Severe failures apply an additional multiplier: Time, Path, or Replanning below 0.5, or an avoidable danger-zone rate above 25%. Congestion remains a diagnostic; its realized delay and hazard costs already affect the trajectory-based score. API cost, tokens, requests, and latency never change EMBER Score.</p>
      <p>${source("docs/scoring.md", "Full formulas, thresholds, and component coverage")} · ${source("docs/benchmark-methodology.md", "Methodology and design rationale")}</p>
    </section>

    <div class="grid2">
      <section class="panel prose" aria-labelledby="aggregation-title">
        <h2 id="aggregation-title">Reading scores and uncertainty</h2>
        <p>Episode trials are averaged first, followed by worlds within a scenario, scenarios within their primary family, and finally families. Each primary family has equal headline weight. Averaged Safety and Quality factors cannot be multiplied to reconstruct the overall score.</p>
        <p>The 95% interval is a stratified two-stage percentile bootstrap over scenarios and worlds within families. It describes this evaluation sample. A family with one scenario cannot reveal variation across unseen scenarios.</p>
        <p>${trialCopy} The protocol calls for at least three trials for a sampled model, with pinned inference settings; temperature zero alone does not establish determinism. The displayed interval does not establish repeat-run reliability.</p>
        <p>Comparison charts show descriptive differences. Official paired inference requires matched scoring versions, constants, manifests, packs, and applicable episodes. A “better” outcome also requires a paired interval excluding zero and at least a one-point difference. These official comparison outcomes are not included in the site data.</p>
        <p>${source("docs/scoring.md#6-aggregation", "Aggregation")} · ${source("docs/scoring.md#7-confidence-interval", "Confidence intervals")} · ${source("docs/scoring.md#8-comparison-and-leaderboard", "Paired comparison rules")}</p>
      </section>
      <section class="panel prose" aria-labelledby="public-development">
        <h2 id="public-development">What these results can establish</h2>
        <p><strong>Public development.</strong> These fixed diagnostic scenarios are available for development. Performance here does not establish generalization to hidden environments. The current pack schedules ${esc(worlds)} frozen worlds; deterministic scenarios are scheduled once and the stochastic scenario contributes distinct fire worlds.</p>
        <p><strong>Hidden evaluation.</strong> Held-out evaluation uses scenarios unavailable during development. This portal contains no hidden-suite results.</p>
        <p><strong>Decision models and reference agents.</strong> Decision models use external inference services. Reference agents are local benchmark programs with prescribed rules or planning algorithms; they do not use an LLM. Both choose actions through the same model interface.</p>
        <p><strong>API efficiency.</strong> Cost is provider-reported or estimated from token counts and published list prices, as labeled. Mean decision latency is the wait for one decision. Run wall time can include rate-limit waits; a resumed run may record only its last segment. Missing measurements remain unavailable.</p>
        <p>${source("docs/scenario-suite.md", "Public scenario suite")} · ${source("docs/baselines.md", "Reference agents")} · ${source("docs/site/README.md", "Site data and operational measurements")}</p>
      </section>
    </div>

    <section class="panel prose" aria-labelledby="provenance">
      <div class="section-head"><h2 id="provenance">Evaluation identity and provenance</h2><span class="badge">Generated official score data</span></div>
      <p>Every displayed result comes from the generated data module. The pack digest identifies the shared frozen environment. Scoring version ${esc(data.scoring_version)} defines the score arithmetic separately from simulation and metric versions.</p>
      <div class="table-wrap" tabindex="0" role="region" aria-label="Methodology reference table"><table>
        <caption>Identity attached to this published evaluation</caption>
        <tbody>
          <tr><th scope="row">Evaluation scope</th><td>${esc(data.label)}</td></tr>
          <tr><th scope="row">Suite ID</th><td><code>${esc(data.suite_id)}</code></td></tr>
          <tr><th scope="row">EnvironmentPack</th><td><code>${esc(data.pack_id)}</code></td></tr>
          <tr><th scope="row">Pack SHA-256</th><td><code style="overflow-wrap:anywhere">${esc(data.pack_digest)}</code></td></tr>
          <tr><th scope="row">Scoring version</th><td>${esc(data.scoring_version)}</td></tr>
          <tr><th scope="row">Data module</th><td><a href="data/results.js">Generated results.js</a></td></tr>
          <tr><th scope="row">Generation authority</th><td>${source("tools/build_results_site.py", "tools/build_results_site.py")}</td></tr>
        </tbody>
      </table></div>
      <p class="muted">The generated artifact contains no evaluation timestamp. No “last evaluated” date is inferred from model order or page load time.</p>
    </section>`;
}

render().catch(() => {
  const main = document.querySelector("#main");
  if (main)
    main.innerHTML =
      '<section class="panel empty"><h1>Methodology data unavailable</h1><p>The generated evaluation data could not be loaded. Reload the page or open the <a href="https://github.com/Aryabhatta-0/EMBERBench/blob/main/docs/scoring.md">scoring definition</a>.</p></section>';
});
