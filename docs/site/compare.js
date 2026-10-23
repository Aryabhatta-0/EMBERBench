import {
  loadResults,
  shell,
  esc,
  fmt,
  count,
  duration,
  costText,
  modelLink,
  kindBadge,
  ciText,
  ratio,
  COMPONENTS,
  metricCard,
  section,
  table,
  getSelection,
  setSelection,
  rankOf,
  scenarioName,
  errorPage,
  isDecision,
} from "./common.js";
import {
  selectedIds,
  pairedDifferences,
  leaderboardOrder,
  finite,
} from "./analytics.js";
import { componentChart, heatmap, differenceChart } from "./charts.js";

const fields = [
  ["Safety", (m) => ratio(m.safety)],
  ["Quality", (m) => ratio(m.quality)],
  ["Evacuated", (m) => count(m.totals.evacuated)],
  ["Avoidable casualties", (m) => count(m.totals.avoidable_casualties)],
  ["Avoidable stranded", (m) => count(m.totals.avoidable_stranded)],
  ["Worst scenario", (m) => fmt(m.robustness.worst_scenario)],
  ["Congestion block rate", (m) => ratio(m.diagnostics.congestion_block_rate)],
  ["Blocked moves", (m) => count(m.diagnostics.congestion_blocks)],
  ["API requests", (m) => count(m.api?.requests)],
  ["Input tokens", (m) => count(m.api?.input_tokens)],
  ["Output tokens", (m) => count(m.api?.output_tokens)],
  ["API cost", costText],
  ["Mean decision latency", (m) => duration(m.api?.mean_act_latency_s)],
  ["Trials", (m) => count(m.totals.trials)],
];
function nestedRows(object, prefix = "") {
  return Object.entries(object ?? {}).flatMap(([k, v]) =>
    v && typeof v === "object"
      ? nestedRows(v, `${prefix}${k} / `)
      : [[`${prefix}${k}`, v]],
  );
}
function worldScores(models) {
  const worlds = new Map();
  for (const m of models)
    for (const e of m.episodes) {
      const key = JSON.stringify([e.index, e.scenario_id, e.seed]);
      if (!worlds.has(key))
        worlds.set(key, {
          index: e.index,
          id: e.scenario_id,
          seed: e.seed,
          scores: new Map(),
        });
      const entry = worlds.get(key);
      const values = entry.scores.get(m.id) ?? [];
      if (finite(e.score)) values.push(e.score);
      entry.scores.set(m.id, values);
    }
  return [...worlds.values()]
    .sort((a, b) => a.index - b.index)
    .map((w) => [
      esc(scenarioName(w.id)),
      esc(w.seed),
      ...models.map((m) => {
        const vs = w.scores.get(m.id) ?? [];
        return vs.length ? fmt(vs.reduce((a, b) => a + b, 0) / vs.length) : "—";
      }),
    ]);
}
async function main() {
  const data = await loadResults();
  shell(data, "compare");
  const all = leaderboardOrder(data.models),
    params = new URLSearchParams(location.search);
  let ids = params.has("models")
    ? selectedIds(params.get("models").split(","), all)
    : getSelection(data);
  if (!params.has("models") && ids.length < 2)
    ids = all
      .filter(isDecision)
      .slice(0, 2)
      .map((m) => m.id);
  const render = () => {
    const models = ids.map((id) => all.find((m) => m.id === id));
    document.getElementById("main").innerHTML =
      `<header class="page-head"><div><p class="eyebrow">Side-by-side analysis</p><h1>Compare models</h1><p class="muted">Trace overall differences down to components, scenarios, and shared worlds.</p></div><a class="button" href="leaderboard.html">Select from leaderboard</a></header><section class="panel"><div class="toolbar compare-picker">${Array.from({ length: 4 }, (_, i) => `<label class="field">Model ${i + 1}${i > 1 ? " (optional)" : ""}<select data-model-slot="${i}"><option value="">Choose a model</option>${all.map((m) => `<option value="${esc(m.id)}" ${ids[i] === m.id ? "selected" : ""} ${ids.includes(m.id) && ids[i] !== m.id ? "disabled" : ""}>${esc(m.name)}</option>`).join("")}</select></label>`).join("")}</div><p class="note">Choose 2–4 entries. Comparison links preserve your selection. ${models.length} selected.</p><p class="note">Public-development comparison only. These are descriptive differences; no official paired inference or significance result is present in this snapshot.</p></section><div id="comparison-content"></div>`;
    for (const el of document.querySelectorAll("[data-model-slot]"))
      el.addEventListener("change", () => {
        const next = Array.from(document.querySelectorAll("[data-model-slot]"))
          .map((s) => s.value)
          .filter(Boolean);
        ids = selectedIds(next, all);
        setSelection(ids);
        history.replaceState(
          null,
          "",
          `?models=${encodeURIComponent(ids.join(","))}`,
        );
        const slot = el.dataset.modelSlot;
        render();
        document
          .querySelector(`[data-model-slot="${slot}"]`)
          ?.focus({ preventScroll: true });
      });
    const content = document.getElementById("comparison-content");
    if (models.length < 2) {
      content.innerHTML =
        '<div class="empty"><h2>Select at least two models</h2><p>Use the model selectors above or choose rows on the leaderboard.</p></div>';
      return;
    }
    const cards = `<div class="compare-grid">${models
      .map(
        (m, i) =>
          `<article class="compare-card series-card-${i}"><span class="eyebrow">${esc(m.kind)}</span><h2>${modelLink(m)}</h2><span class="muted">Overall rank ${rankOf(all, m)} / ${all.length}</span><strong class="metric-value">${fmt(m.score)}</strong><span class="metric-note">EMBER points · 95% CI ${ciText(m)}</span><dl class="metric-list">${fields
            .slice(0, 6)
            .map(
              ([label, get]) =>
                `<div class="metric-row"><dt>${label}</dt><dd>${esc(get(m))}</dd></div>`,
            )
            .join("")}</dl></article>`,
      )
      .join("")}</div>`;
    const metrics = table(
      ["Metric", ...models.map((m) => m.name)],
      fields.map(([label, get]) => [
        esc(label),
        ...models.map((m) => esc(get(m))),
      ]),
      "Side-by-side outcome and operational metrics",
    );
    const robustnessKeys = [
      ...new Set(
        models.flatMap((m) => nestedRows(m.robustness).map(([k]) => k)),
      ),
    ];
    const robustness = table(
      ["Robustness diagnostic", ...models.map((m) => m.name)],
      robustnessKeys.map((key) => [
        esc(key.replaceAll("_", " ")),
        ...models.map((m) => {
          const value = nestedRows(m.robustness).find(([k]) => k === key)?.[1];
          return esc(
            typeof value === "number"
              ? fmt(value, Number.isInteger(value) ? 0 : 2)
              : (value ?? "—"),
          );
        }),
      ]),
      "Reported robustness diagnostics",
    );
    const congestionKeys = [
      ...new Set(models.flatMap((m) => Object.keys(m.diagnostics))),
    ];
    const congestion = table(
      ["Congestion diagnostic", ...models.map((m) => m.name)],
      congestionKeys.map((key) => [
        esc(key.replaceAll("_", " ")),
        ...models.map((m) => {
          const value = m.diagnostics[key];
          return esc(
            finite(value) ? fmt(value, Number.isInteger(value) ? 0 : 4) : "—",
          );
        }),
      ]),
      "Reported congestion diagnostics",
    );
    let paired = "";
    if (models.length === 2) {
      const pair = pairedDifferences(models[0], models[1]);
      paired = section(
        "World-by-world paired differences",
        "Descriptive analysis from existing official world scores. No p-value or significance claim.",
        `<div class="metrics">${metricCard("Mean paired difference", `${pair.mean > 0 ? "+" : ""}${fmt(pair.mean)}`, "Unweighted points across matched worlds")}${metricCard("Matched worlds", count(pair.worlds.length), pair.complete ? "Complete paired coverage" : "Incomplete paired coverage")}</div>${differenceChart(pair, models[0], models[1])}<p class="note">Worlds are paired by scenario, seed, and scheduled index. Repeated trials, if present, are averaged within each world. Reported inputs are rounded by the site builder. Use the official compare_scores output for paired inference.</p>`,
      );
    } else
      paired = section(
        "World-by-world differences",
        "Choose exactly two models for a signed paired-difference chart.",
        `<p class="muted">The per-world table below supports all ${models.length} selected models.</p>`,
      );
    content.innerHTML =
      cards +
      section(
        "Performance and API usage",
        "Scores increase with performance; avoidable losses, block rate, API cost and latency are better when lower. Operational usage is diagnostic.",
        metrics,
      ) +
      section(
        "Seven score components",
        "Higher is better for every component factor. Values are family-balanced aggregates.",
        componentChart(models),
      ) +
      section(
        "Scenario comparison",
        "Same frozen worlds; scenario scores on a 0–100 scale.",
        heatmap(models),
      ) +
      paired +
      section(
        "Per-world scores",
        "Official episode scores; repeated trials averaged per matched world where applicable.",
        table(
          ["Scenario", "Seed", ...models.map((m) => m.name)],
          worldScores(models),
          "World-by-world scores",
        ),
      ) +
      section(
        "Robustness",
        "Reported tail performance, failure counts and worst-scenario summaries.",
        robustness,
      ) +
      section(
        "Congestion",
        "Action-dependent diagnostics. Congestion is not a score component.",
        congestion,
      ) +
      `<p class="note">95% intervals do not establish paired significance. ${esc([...new Set(models.map((m) => m.totals.trials))].join(", "))} trial per model in this selection. API cost may be a list-price estimate (≈); request/token/latency totals may cover only the final invocation of a resumed run. Missing API metrics are —.</p>`;
  };
  render();
}
main().catch(errorPage);
