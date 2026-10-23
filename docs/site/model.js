import {
  loadResults,
  shell,
  esc,
  fmt,
  count,
  ratio,
  duration,
  costText,
  modelLink,
  kindBadge,
  ciText,
  metricCard,
  section,
  table,
  provenance,
  COMPONENTS,
  rankOf,
  provider,
  scenarioName,
  errorPage,
} from "./common.js";
import { leaderboardOrder, finite } from "./analytics.js";
import { scoreBars, componentChart, heatmap } from "./charts.js";

function diagnosticRows(object) {
  return Object.entries(object ?? {}).flatMap(([key, v]) =>
    v && typeof v === "object"
      ? Object.entries(v).map(([sub, n]) => [
          esc(`${key} / ${sub}`.replaceAll("_", " ")),
          esc(n ?? "—"),
        ])
      : [
          [
            esc(key.replaceAll("_", " ")),
            esc(finite(v) ? fmt(v, Number.isInteger(v) ? 0 : 4) : (v ?? "—")),
          ],
        ],
  );
}
async function main() {
  const data = await loadResults();
  shell(data, "model");
  const id = new URLSearchParams(location.search).get("id"),
    m = data.models.find((m) => m.id === id);
  if (!m) {
    document.getElementById("main").innerHTML =
      `<header class="page-head"><h1>Choose a model profile</h1><p class="muted">${id ? "This model ID is not in the published data." : "Open a model from the leaderboard."}</p></header>${section(
        "Published entries",
        "",
        table(
          ["Model", "Type", "EMBER"],
          data.models.map((x) => [modelLink(x), kindBadge(x), fmt(x.score)]),
          "Available model profiles",
        ),
      )}`;
    return;
  }
  document.title = `${m.name} | EMBERBench`;
  const models = leaderboardOrder(data.models),
    t = m.totals,
    a = m.api,
    withinKind = models.filter((x) => x.kind === m.kind),
    worst = m.robustness;
  const anchors = [
    ["performance", "Performance overview"],
    ["components", "Score components"],
    ["scenarios", "Scenario performance"],
    ["worlds", "World by world"],
    ["people", "People outcomes"],
    ["robustness", "Robustness"],
    ["congestion", "Congestion"],
    ...(a ? [["api", "API efficiency"]] : []),
    ["provenance", "Provenance"],
  ];
  const hero = `<header class="profile-head"><div><a class="breadcrumb" href="leaderboard.html">Leaderboard / Model profile</a><h1>${esc(m.name)}</h1><p>${esc(provider(m))} ${kindBadge(m)}</p><p class="muted">Overall rank <b>${rankOf(models, m)} of ${models.length}</b> · Rank ${rankOf(withinKind, m)} of ${withinKind.length} ${esc(m.kind.toLowerCase())}s · ${count(t.trials)} trial</p><div class="toolbar"><label class="field">Browse models<select id="browse-model">${models.map((x) => `<option value="${esc(x.id)}" ${x.id === m.id ? "selected" : ""}>${esc(x.name)}</option>`).join("")}</select></label><a class="button" href="compare.html?models=${encodeURIComponent(m.id)}">Compare this model</a></div></div><div class="score-hero"><span>EMBER Score ↑</span><strong>${fmt(m.score)}</strong><small>95% CI ${ciText(m)}</small><small>Benchmark points / 100</small></div></header>`;
  const apiContent = a
    ? `<div class="metrics">${metricCard("API requests", count(a.requests), "Sent in the recorded invocation")}${metricCard("Input tokens", count(a.input_tokens), "API-reported")}${metricCard("Output tokens", count(a.output_tokens), "API-reported")}${metricCard("API cost", costText(m), a.cost_source ?? "Not reported")}${metricCard("Mean decision latency", duration(a.mean_act_latency_s), "Per recorded act call")}${metricCard("Recorded wall time", duration(a.wall_clock_s), "May be final resumed segment only")}</div><p class="note">API usage comes from run metadata, not scoring. Cached answers may be untimed. A resumed run can report only its last invocation, including provider waits. Mean decision latency is not measured streaming time to first token. ≈ means an estimate at published token prices.</p>${table(
        ["Field", "Recorded value"],
        [
          ["Resolved models", esc(a.resolved_models?.join(", ") || "—")],
          ["Cost source", esc(a.cost_source ?? "—")],
          ["Decision calls", count(a.decisions)],
          ["First recorded response", duration(a.first_response_s)],
          ["Median recorded response", duration(a.median_response_s)],
        ],
        "API measurement details",
      )}`
    : '<p class="muted">This reference agent uses implemented rules. API usage is inapplicable. Local compute cost and latency were not measured, so they are not assigned zero.</p>';
  const outcomes = table(
    ["Outcome", "People (summed across evaluated worlds/trials)"],
    [
      ["Evacuated", count(t.evacuated)],
      ["Avoidable casualties", count(t.avoidable_casualties)],
      ["Avoidable stranded", count(t.avoidable_stranded)],
      ["Individually escapable", count(t.escapable)],
      ["Unavoidable losses", count(t.unavoidable_losses)],
    ],
    "People outcomes",
  );
  document.getElementById("main").innerHTML =
    hero +
    `<div class="metrics">${metricCard("Safety ↑", ratio(m.safety), "0–1; safety caps the score")}${metricCard("Quality ↑", ratio(m.quality), "0–1; execution factors")}${metricCard("Evacuated ↑", count(t.evacuated), `of ${count(t.escapable)} individually escapable`)}${metricCard("Avoidable deaths ↓", count(t.avoidable_casualties), "Across evaluated worlds/trials")}${metricCard("Avoidable stranded ↓", count(t.avoidable_stranded), "Savable but active at horizon")}${metricCard("Worst scenario ↑", fmt(worst.worst_scenario), scenarioName(worst.worst_scenario_id ?? "—"))}</div><nav class="profile-nav" aria-label="Profile sections">${anchors.map(([id, label]) => `<a href="#${id}">${label}</a>`).join("")}</nav>` +
    `<div id="performance">${section("Performance overview", "Same public environment pack; order is descriptive, not statistical significance.", scoreBars(models, m.id))}</div>` +
    `<div id="components">${section(
      "Score components",
      "All seven factors are 0–1; higher is better. Aggregates are not a substitute for per-world scoring.",
      componentChart([m]) +
        table(
          [
            "Component",
            "Factor",
            "Defined episode trials",
            "Total episode trials",
          ],
          Object.entries(COMPONENTS).map(([k, label]) => [
            esc(label),
            ratio(m.components[k]?.value),
            count(m.components[k]?.defined),
            count(m.components[k]?.total),
          ]),
          "Component values and applicability",
        ),
    )}</div>` +
    `<div id="scenarios">${section(
      "Scenario performance",
      "Identify strong and weak scenarios without inventing capability categories.",
      heatmap([m]) +
        table(
          ["Scenario", "World scores", "EMBER ↑", "Safety ↑", "Worst world ↑"],
          m.scenarios.map((s) => [
            `<a href="scenarios.html?scenario=${encodeURIComponent(s.name)}">${esc(scenarioName(s.name))}</a>`,
            count(s.episodes),
            fmt(s.score),
            ratio(s.safety),
            fmt(s.worst),
          ]),
          "Per-scenario performance",
        ),
    )}</div>` +
    `<div id="worlds">${section(
      "World-by-world performance",
      "Official episode score records; seed and index identify the evaluated world.",
      table(
        [
          "Index",
          "Scenario",
          "Seed",
          "Status",
          "EMBER ↑",
          "Safety ↑",
          "Quality ↑",
          "Population",
          "Evacuated",
          "Casualties",
          "Stranded",
        ],
        m.episodes.map((e) => [
          count(e.index),
          esc(scenarioName(e.scenario_id)),
          esc(e.seed),
          esc(e.status),
          fmt(e.score),
          ratio(e.safety),
          ratio(e.quality),
          count(e.population),
          count(e.evacuated),
          count(e.casualties),
          count(e.stranded),
        ]),
        "Per-world outcomes and scores",
      ),
    )}</div>` +
    `<div class="grid2"><div id="people">${section("People outcomes", "Avoidable outcomes use evaluator-side individual escape references. Counts sum across worlds and trials.", outcomes)}</div><div id="robustness">${section("Robustness", "Reported tail behavior and failure counts. No between-trial inference from one trial.", table(["Diagnostic", "Value"], diagnosticRows(m.robustness), "Robustness diagnostics"))}</div></div>` +
    `<div id="congestion">${section("Congestion diagnostics", "The fire is frozen; actions change routes, queueing, and movement conflicts. These diagnostics never directly affect EMBER Score.", table(["Diagnostic", "Value"], diagnosticRows(m.diagnostics), "Congestion diagnostics"))}</div>` +
    `<div id="api">${section("API efficiency", a ? "Recorded operational measurements; absent values stay —." : "Reference agent: no remote API.", apiContent)}</div>` +
    `<div id="provenance">${section("Reproducibility / provenance", "Official score reports, generated data, and explicit uncertainty limits.", provenance(data, m) + `<p class="note">${count(t.trials)} trial for this model. Reported 95% CIs resample the evaluated world evidence and cannot measure unobserved between-trial variability with one trial. The current snapshot does not expose all report provenance fields; those are shown as — pending regeneration from the original reports.</p>`)}</div>`;
  document.getElementById("browse-model").addEventListener("change", (e) => {
    location.href = `model.html?id=${encodeURIComponent(e.target.value)}`;
  });
}
main().catch(errorPage);
