import {
  loadResults,
  shell,
  esc,
  fmt,
  count,
  duration,
  costText,
  modelLink,
  metricCard,
  section,
  table,
  kindBadge,
  provenance,
  errorPage,
  isDecision,
} from "./common.js";
import { sortModels, finite, leaderboardOrder } from "./analytics.js";
import { scoreBars, scatter, heatmap, legend } from "./charts.js";

async function main() {
  const data = await loadResults();
  shell(data, "overview");
  const models = leaderboardOrder(data.models),
    decisions = models.filter(isDecision),
    worlds = new Set(
      models[0].episodes.map((e) => `${e.index}/${e.scenario_id}/${e.seed}`),
    ).size;
  const leaders = [
    ["Highest EMBER Score", "score", fmt],
    ["Best Safety", "safety", (v) => fmt(v, 3)],
    ["Best Quality", "quality", (v) => fmt(v, 3)],
    ["Fewest avoidable casualties", "casualties", count],
    ["Fastest API decision latency", "latency", duration],
    ["Lowest API cost", "cost", null],
  ];
  const leaderCards = leaders
    .map(([label, key, display]) => {
      const candidates = sortModels(
          key === "latency" || key === "cost" ? decisions : models,
          key,
        ),
        best = candidates.find((m) =>
          finite(
            key === "cost"
              ? m.api?.cost_usd
              : key === "latency"
                ? m.api?.mean_act_latency_s
                : key === "casualties"
                  ? m.totals.avoidable_casualties
                  : m[key],
          ),
        );
      const value = best
        ? key === "cost"
          ? costText(best)
          : display(
              key === "latency"
                ? best.api.mean_act_latency_s
                : key === "casualties"
                  ? best.totals.avoidable_casualties
                  : best[key],
            )
        : "—";
      const ties = best
        ? candidates.filter(
            (m) =>
              (key === "cost"
                ? m.api?.cost_usd
                : key === "latency"
                  ? m.api?.mean_act_latency_s
                  : key === "casualties"
                    ? m.totals.avoidable_casualties
                    : m[key]) ===
              (key === "cost"
                ? best.api?.cost_usd
                : key === "latency"
                  ? best.api?.mean_act_latency_s
                  : key === "casualties"
                    ? best.totals.avoidable_casualties
                    : best[key]),
          ).length
        : 0;
      return `<article class="leader-card"><span class="metric-label">${esc(label)}</span><strong class="leader-value">${esc(value)}</strong>${best ? modelLink(best) : "<span>Not recorded</span>"}<span class="metric-note">${key === "cost" || key === "latency" ? "Decision models with recorded data" : `${ties > 1 ? `${ties} tied · ` : ""}all evaluated entries`}</span></article>`;
    })
    .join("");
  document.getElementById("main").innerHTML =
    `<header class="page-head overview-head"><div><p class="eyebrow">Public benchmark / Results explorer</p><h1>Decisions under pressure.<br><span class="heading-muted">Evidence, world by world.</span></h1><p class="muted">Compare how decision models and reference agents guide people through dynamic fire evacuations. Shared worlds. Replay-verified outcomes.</p></div><a class="button primary" href="leaderboard.html">Explore leaderboard <span aria-hidden="true">↗</span></a></header>
  <div class="metrics">${metricCard("Decision models", count(decisions.length), "Evaluated API decision policies")}${metricCard("Reference agents", count(models.length - decisions.length), "Rule-based comparison strategies")}${metricCard("Frozen unique worlds", count(worlds), `${models[0].scenarios.length} public scenarios`)}${metricCard("Scoring version", data.scoring_version, "EMBER points · 0–100")}</div>
  <div class="pack-strip"><span>EnvironmentPack <b>${esc(data.pack_id)}</b></span><a href="methodology.html#provenance">${esc(data.pack_digest.slice(0, 16))}… ↗</a></div>
  <section><div class="section-head"><h2>Leaders at a glance</h2><span class="muted">Descriptive best values, not significance claims</span></div><div class="leaders">${leaderCards}</div></section>
  <div class="grid2 overview-analysis">${section("EMBER Score by model", "Higher is better. All evaluated entries; reference agents are not LLMs.", `<div class="toolbar"><label class="field">Show<select id="top-kind"><option value="all">All entries</option><option value="Decision model">Decision models</option><option value="Reference agent">Reference agents</option></select></label></div><div id="top-chart"></div>${legend()}<a class="link" href="leaderboard.html">Open full leaderboard</a>`)}${section("Explore the trade-offs", "Use actual outcomes to understand score differences.", `<div class="toolbar"><label class="field">Relationship<select id="analysis-metric"><option value="quality">Safety vs Quality</option><option value="casualties">EMBER vs avoidable casualties</option><option value="congestion">EMBER vs congestion block rate</option><option value="latency">EMBER vs mean decision latency</option><option value="cost">EMBER vs API cost</option></select></label><label class="field">Highlight model<select id="highlight-model"><option value="">None</option>${models.map((m) => `<option value="${esc(m.id)}">${esc(m.name)}</option>`).join("")}</select></label></div><div id="scatter-chart"></div><p class="note">Cost and latency are run metadata, never score inputs. ≈ costs are list-price estimates; resumed runs may record only the final invocation.</p>`)}</div>
  ${section("Scenario × model", "A compact view of strengths and weaknesses. Scenario scores, 0–100; higher is better.", heatmap(models))}
  ${section(
    "Evaluated decision models",
    "Evaluation dates are not recorded, so this snapshot cannot establish a latest-to-oldest order.",
    table(
      ["Model", "Type", "EMBER Score", "Avoidable deaths", "API cost"],
      decisions.map((m) => [
        modelLink(m),
        kindBadge(m),
        fmt(m.score),
        count(m.totals.avoidable_casualties),
        costText(m),
      ]),
      "Evaluated decision models",
    ),
  )}
  <div class="grid2 explanatory"><section class="panel prose"><h2>What is EMBERBench?</h2><p>A deterministic benchmark for decision-making in dynamic fire evacuation. A policy routes people while fire spreads and shared corridors become congested.</p><h3>What does EMBER Score measure?</h3><p>Safety and execution quality under a fixed scoring definition. The score rewards evacuation and penalizes avoidable casualties, stranding and severe quality failures. It is not the percentage of people saved.</p><a href="methodology.html">Understand the scoring</a></section><section class="panel prose"><h2>What is a decision model?</h2><p>A model that receives the current state and chooses movement actions for each person. Reference agents use fixed, implemented strategies and provide comparison points.</p><h3>Why are these public-development results?</h3><p>These worlds are publicly available for development. Replay-authenticated scoring does not make a public evaluation a hidden-suite leaderboard or establish hidden-suite generalization.</p><p class="note">${esc([...new Set(models.map((m) => m.totals.trials))].join(", "))} trial per model in this snapshot. CIs do not measure between-trial variability for a one-trial model.</p></section></div>
  ${section("Reproducible by construction", "Every benchmark number here comes from the generated official report snapshot.", provenance(data))}`;
  const renderTop = () => {
    document.getElementById("top-chart").innerHTML = scoreBars(
      models.filter(
        (m) =>
          document.getElementById("top-kind").value === "all" ||
          m.kind === document.getElementById("top-kind").value,
      ),
      document.getElementById("highlight-model").value,
    );
  };
  const renderScatter = () => {
    const key = document.getElementById("analysis-metric").value;
    document.getElementById("scatter-chart").innerHTML = scatter(
      models,
      key,
      key === "quality" ? "safety" : "score",
      document.getElementById("highlight-model").value,
    );
  };
  document.getElementById("top-kind").addEventListener("change", renderTop);
  document
    .getElementById("analysis-metric")
    .addEventListener("change", renderScatter);
  document.getElementById("highlight-model").addEventListener("change", () => {
    renderTop();
    renderScatter();
  });
  renderTop();
  renderScatter();
}
main().catch(errorPage);
