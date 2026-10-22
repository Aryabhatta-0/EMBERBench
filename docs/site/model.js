import { dataTable, esc, fmt, hbarChart, installTooltip, loadResults, stackedBar, vbarChart, worldLabel } from "./common.js";

const COMPONENTS = {
  casualty: "Casualty avoidance",
  evacuation: "Evacuation",
  stranding: "Stranding avoidance",
  time: "Time efficiency",
  path: "Path efficiency",
  hazard: "Hazard exposure",
  replanning: "Replanning",
};

const tile = (label, value, note = "", hero = false) =>
  `<div class="tile${hero ? " hero" : ""}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>` +
  (note ? `<div class="note">${esc(note)}</div>` : "") + `</div>`;

const shortScenario = (id) => id.replace(/^v1-pub-/, "");

function render(data, m) {
  document.title = `${m.name} — EMBERBench V1`;
  const rank = data.models.indexOf(m) + 1;
  const t = m.totals;
  document.getElementById("name").textContent = m.name;
  document.getElementById("meta").innerHTML = `${esc(m.kind)} · model ID <code>${esc(m.id)}</code> · rank ${rank} of ${data.models.length}`;
  document.getElementById("label").textContent = data.label;

  // Headline tiles
  const ci = m.ci ? `95% CI ${fmt(m.ci[0])}–${fmt(m.ci[1])}` : "";
  document.getElementById("tiles").innerHTML = [
    tile("EMBER Score", fmt(m.score), `${ci} · out of 100`, true),
    tile("Rank", `${rank} / ${data.models.length}`),
    tile("Safety", fmt(m.safety, 3), "0–1"),
    tile("Quality", fmt(m.quality, 3), "0–1"),
    tile("Evacuated", `${t.evacuated}`, `of ${t.escapable} people who could escape`),
    tile("Avoidable deaths", `${t.avoidable_casualties}`),
    tile("Avoidable stranded", `${t.avoidable_stranded}`),
    tile("Worst scenario", fmt(m.robustness.worst_scenario), shortScenario(m.robustness.worst_scenario_id ?? "–")),
  ].join("");

  // People outcomes: one part-to-whole bar of the escapable people
  const css = getComputedStyle(document.documentElement);
  const parts = [
    { label: "Evacuated", value: t.evacuated, color: css.getPropertyValue("--series-1") },
    { label: "Avoidable deaths", value: t.avoidable_casualties, color: css.getPropertyValue("--series-2") },
    { label: "Avoidable stranded", value: t.avoidable_stranded, color: css.getPropertyValue("--series-3") },
  ];
  document.getElementById("people").innerHTML = stackedBar(parts, t.escapable);
  document.getElementById("people-legend").innerHTML = parts
    .map((p) => `<span style="--c:${p.color}">${esc(p.label)} <b>${p.value}</b> (${((p.value / t.escapable) * 100).toFixed(1)}%)</span>`)
    .join("");
  document.getElementById("people-note").textContent =
    `${t.escapable} people could escape` + (t.unavoidable_losses ? ` · ${t.unavoidable_losses} unavoidable losses` : "");

  // Score per scenario
  const scen = m.scenarios.map((g) => ({
    label: shortScenario(g.name),
    value: g.score,
    tip: `<b>${esc(g.name)}</b><br>Score ${fmt(g.score)} · safety ${fmt(g.safety, 3)}<br>${g.episodes} world(s), worst ${fmt(g.worst)}`,
  }));
  document.getElementById("scenarios").innerHTML =
    hbarChart(scen, { max: 100, labelWidth: 190 }) +
    dataTable(
      [{ label: "Scenario" }, { label: "Worlds", num: true }, { label: "Score", num: true }, { label: "Safety", num: true }, { label: "Worst world", num: true }],
      m.scenarios.map((g) => [g.name, g.episodes, fmt(g.score), fmt(g.safety, 3), fmt(g.worst)]),
    );

  // Score components
  const comps = Object.entries(COMPONENTS)
    .filter(([k]) => m.components[k])
    .map(([k, label]) => {
      const c = m.components[k];
      return { label, value: c.value, digits: 3, tip: `<b>${esc(label)}</b>: ${fmt(c.value, 3)}<br>defined in ${c.defined} of ${c.total} worlds` };
    });
  document.getElementById("components").innerHTML =
    hbarChart(comps, { max: 1, labelWidth: 150, digits: 3 }) +
    dataTable(
      [{ label: "Component" }, { label: "Value", num: true }, { label: "Defined in worlds", num: true }],
      Object.entries(COMPONENTS).filter(([k]) => m.components[k]).map(([k, label]) => [label, fmt(m.components[k].value, 3), `${m.components[k].defined} / ${m.components[k].total}`]),
    );

  // Score per world
  const counts = {};
  for (const e of m.episodes) counts[e.scenario_id] = (counts[e.scenario_id] ?? 0) + 1;
  const stochastic = new Set(Object.keys(counts).filter((k) => counts[k] > 1));
  const eps = m.episodes.map((e) => ({
    label: worldLabel(e, stochastic),
    value: e.score,
    tip: `<b>${esc(e.scenario_id)}</b> seed ${e.seed}<br>Score ${fmt(e.score)}<br>Evacuated ${e.evacuated} · deaths ${e.casualties} · stranded ${e.stranded} of ${e.population}`,
  }));
  document.getElementById("episodes").innerHTML =
    vbarChart(eps, { max: 100 }) +
    dataTable(
      [{ label: "Scenario" }, { label: "Seed", num: true }, { label: "Score", num: true }, { label: "Evacuated", num: true }, { label: "Deaths", num: true }, { label: "Stranded", num: true }, { label: "People", num: true }],
      m.episodes.map((e) => [e.scenario_id, e.seed, fmt(e.score), e.evacuated, e.casualties, e.stranded, e.population]),
    );

  // Robustness counts
  const r = m.robustness;
  const failed = Object.values(r.failed_episode_trials ?? {}).reduce((a, b) => a + b, 0);
  const severe = r.severe_episode_trials ?? {};
  const rob = [
    { label: "Zero-score worlds", value: r.zero_score_episode_trials },
    { label: "Catastrophic worlds", value: r.catastrophic_episode_trials },
    ...Object.entries(severe).map(([k, v]) => ({ label: `Severe: ${k}`, value: v })),
    { label: "Failed episodes", value: failed },
  ].map((row) => ({ ...row, display: String(row.value), tip: `<b>${esc(row.label)}</b>: ${row.value} of ${m.episodes.length} worlds` }));
  document.getElementById("robustness").innerHTML =
    hbarChart(rob, { max: m.episodes.length, labelWidth: 150, digits: 0, ticks: [0, 5, 10, 15] }) +
    `<p class="muted" style="margin-top:8px">10th-percentile world ${fmt(r.p10_episode)} · CVaR10 ${fmt(r.cvar10_episode)} · median scenario ${fmt(r.median_scenario)}</p>`;

  // Congestion diagnostics
  const d = m.diagnostics;
  document.getElementById("congestion").innerHTML = [
    tile("Blocked moves", `${(d.congestion_block_rate * 100).toFixed(1)}%`, `${d.congestion_blocks.toLocaleString()} of ${d.movement_attempts.toLocaleString()} attempts`),
    tile("Exit throughput", fmt(d.exit_throughput, 3), "evacuations per active exit step"),
    tile("Peak contested cells", `${d.peak_contested_cells}`),
    tile("Hotspot cells", `${d.hotspot_cells}`),
  ].join("");

  // API use (decision models only)
  if (m.api) {
    document.getElementById("api-card").hidden = false;
    const a = m.api;
    document.getElementById("api").innerHTML = [
      tile("Requests", a.requests?.toLocaleString() ?? "–"),
      tile("Input tokens", a.input_tokens?.toLocaleString() ?? "–"),
      tile("Cost", a.cost_usd ? `$${Number(a.cost_usd).toFixed(3)}` : "–"),
      tile("Mean step latency", a.mean_act_latency_s ? `${a.mean_act_latency_s} s` : "–"),
      tile("Answered by", "", (a.resolved_models ?? []).join(", ") || "–"),
    ].join("");
  }

  document.getElementById("footer").innerHTML =
    `Pack <code>${esc(data.pack_id)}</code> · digest <code>${esc(data.pack_digest.slice(0, 16))}…</code> · scoring version ${esc(data.scoring_version)}. ${esc(data.unit)}.`;
}

async function main() {
  installTooltip();
  const data = await loadResults();
  const id = new URLSearchParams(window.location.search).get("id");
  const m = data.models.find((x) => x.id === id);
  if (!m) throw new Error(`no model with ID ${id}`);
  render(data, m);
}

main().catch((error) => {
  document.getElementById("name").textContent = "Model not found";
  document.getElementById("meta").textContent = error.message;
});
