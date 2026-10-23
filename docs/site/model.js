import { barChart, count, dataTable, duration, esc, fmt, installTooltip, isDecision, loadResults, money, pct, scenarioName, stackedBar } from "./common.js";

const COMPONENTS = {
  casualty: "Casualty avoidance",
  evacuation: "Evacuation",
  stranding: "Stranding avoidance",
  time: "Time efficiency",
  path: "Path efficiency",
  hazard: "Hazard exposure",
  replanning: "Replanning",
};

const tile = (label, value, note = "", cls = "") =>
  `<div class="tile ${cls}"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>` +
  (note ? `<div class="note">${esc(note)}</div>` : "") + `</div>`;

const ordinal = (n) => `${n}${["th", "st", "nd", "rd"][n % 100 > 10 && n % 100 < 14 ? 0 : n % 10] ?? "th"}`;
const tokens = (m) => (m.api ? m.api.input_tokens + m.api.output_tokens : null);
const costOf = (m) => (m.api && m.api.cost_source !== "not reported" ? m.api.cost_usd : null);
const costText = (m) => (isDecision(m) ? money(costOf(m), m.api?.cost_source === "list price") : "Free");

// A comparison card: one bar per model, this model highlighted. lowerIsBetter sorts ascending.
function compareCard(title, sub, models, m, value, display, tipExtra, { lowerIsBetter = false, max = null, estimated = () => false, first = null } = {}) {
  const rows = models
    .filter((x) => value(x) !== null && value(x) !== undefined)
    .sort((a, b) => (lowerIsBetter ? value(a) - value(b) : value(b) - value(a)));
  const place = rows.findIndex((x) => x.id === m.id);
  const badge = place >= 0 ? `<span class="place">${place === 0 ? (first ?? (lowerIsBetter ? "Lowest" : "Best")) : ordinal(place + 1)} of ${rows.length}</span>` : "";
  const chart = barChart(
    rows.map((x) => ({
      id: x.id,
      label: x.name,
      value: value(x),
      display: display(x),
      highlight: x.id === m.id,
      kind: x.kind,
      estimated: estimated(x),
      tip: `<b>${esc(x.name)}</b><br>${esc(display(x))}${tipExtra ? `<br>${tipExtra(x)}` : ""}`,
    })),
    { max },
  );
  return `<section class="card"><div class="card-head"><div><h2>${esc(title)}</h2><p class="muted">${esc(sub)}</p></div>${badge}</div>${chart}</section>`;
}

function render(data, m) {
  document.title = `${m.name} — EMBERBench V1`;
  const t = m.totals;
  const decision = data.models.filter(isDecision);
  const rank = data.models.indexOf(m) + 1;
  const people = t.escapable;

  document.getElementById("name").textContent = m.name;
  document.getElementById("badges").innerHTML = [
    `<span class="badge ${isDecision(m) ? "ai" : ""}">${isDecision(m) ? "AI decision model" : "Reference program (rules, no AI)"}</span>`,
    `<span class="badge">${ordinal(rank)} of ${data.models.length} overall</span>`,
    isDecision(m) ? `<span class="badge">${ordinal(decision.indexOf(m) + 1)} of ${decision.length} AI decision models</span>` : "",
  ].join("");
  const lost = t.avoidable_casualties + t.avoidable_stranded;
  document.getElementById("verdict").innerHTML =
    `In ${m.episodes.length} simulated fires with ${people} people in danger, ${isDecision(m) ? "this model" : "this program"} got <b>${t.evacuated} people (${pct(t.evacuated, people)})</b> out safely. ` +
    (lost ? `<b>${t.avoidable_casualties}</b> people died and <b>${t.avoidable_stranded}</b> were trapped, though a better plan could have saved them.` : "Nobody was lost.");

  // Headline numbers
  const a = m.api;
  const estimated = a?.cost_source === "list price";
  document.getElementById("tiles").innerHTML = [
    tile("People saved", `${t.evacuated} of ${people}`, `${pct(t.evacuated, people)} reached an exit`, "key"),
    tile("EMBER Score", fmt(m.score), "out of 100 · higher is better", "key"),
    tile("Cost", costText(m), isDecision(m) ? (estimated ? "estimate from the published price" : "charged by the API for all 19 fires") : "runs without an AI API"),
    tile("Time to finish", isDecision(m) ? duration(a?.wall_clock_s) : "–", isDecision(m) ? "all 19 fires, start to end" : "not measured for programs"),
    tile("Response time", isDecision(m) ? duration(a?.mean_act_latency_s) : "–", isDecision(m) ? "average wait per decision = time to first token" : "not measured for programs"),
    tile("Tokens used", isDecision(m) ? count(tokens(m)) : "–", isDecision(m) ? `${count(a.input_tokens)} in · ${count(a.output_tokens)} out` : "no AI model"),
  ].join("");

  // People outcomes: one part-to-whole bar
  const css = getComputedStyle(document.documentElement);
  const parts = [
    { label: "Got out safely", value: t.evacuated, color: css.getPropertyValue("--series-1").trim() },
    { label: "Died (avoidable)", value: t.avoidable_casualties, color: css.getPropertyValue("--series-2").trim() },
    { label: "Trapped when time ran out (avoidable)", value: t.avoidable_stranded, color: css.getPropertyValue("--series-3").trim() },
  ];
  document.getElementById("people").innerHTML = stackedBar(parts, people);
  document.getElementById("people-legend").innerHTML = parts
    .map((p) => `<span style="--c:${p.color}">${esc(p.label)} <b>${p.value}</b> (${pct(p.value, people)})</span>`)
    .join("");
  document.getElementById("people-note").textContent =
    `${people} people in all fires together` + (t.unavoidable_losses ? ` · ${t.unavoidable_losses} could not be saved by any plan` : "");

  // Comparisons with the other models
  const cards = [
    compareCard("People saved", `Out of ${people}. Higher is better.`, data.models, m, (x) => x.totals.evacuated, (x) => `${x.totals.evacuated} (${pct(x.totals.evacuated, people)})`,
      (x) => `${x.totals.avoidable_casualties} avoidable deaths`, { max: people }),
    compareCard("EMBER Score", "0 to 100. Higher is better.", data.models, m, (x) => x.score, (x) => fmt(x.score), null, { max: 100 }),
  ];
  if (isDecision(m)) {
    cards.push(
      compareCard("Cost of the whole test", "US dollars for all 19 fires. Lower is better.", decision, m, costOf, costText,
        (x) => (x.api?.cost_source === "list price" ? "Estimate from the published price per token" : "Charged by the API"),
        { lowerIsBetter: true, estimated: (x) => x.api?.cost_source === "list price" }),
      compareCard("Time to finish", "All 19 fires, start to end. Lower is better.", decision, m, (x) => x.api?.wall_clock_s, (x) => duration(x.api?.wall_clock_s),
        (x) => `${count(x.api?.decisions)} decisions`, { lowerIsBetter: true, first: "Fastest" }),
      compareCard("Response time (time to first token)", "Average wait for one decision. Lower is better.", decision, m, (x) => x.api?.mean_act_latency_s,
        (x) => duration(x.api?.mean_act_latency_s), () => "These APIs send the whole answer at once, so the first token arrives with the answer", { lowerIsBetter: true, first: "Fastest" }),
      compareCard("Tokens used", "Text the model read and wrote, all 19 fires. Lower is better.", decision, m, tokens, (x) => count(tokens(x)),
        (x) => `${count(x.api?.input_tokens)} in · ${count(x.api?.output_tokens)} out`, { lowerIsBetter: true }),
    );
  } else {
    cards.push(`<section class="card"><h2>Cost and speed</h2><p class="muted">This is a reference program: fixed rules that run on a computer, with no AI API.
      It has no cost, no tokens and no API wait. It is here to show what simple and strong strategies achieve.</p></section>`);
  }
  document.getElementById("compare").innerHTML = cards.join("");

  // Fire by fire, against the best AI decision model in each scenario
  const best = {};
  for (const x of decision) for (const g of x.scenarios) if (g.score !== null && !(best[g.name]?.score >= g.score)) best[g.name] = { score: g.score, name: x.name };
  const saved = {};
  for (const e of m.episodes) {
    const s = (saved[e.scenario_id] ??= { evacuated: 0, population: 0 });
    s.evacuated += e.evacuated;
    s.population += e.population;
  }
  document.getElementById("scenarios").innerHTML =
    barChart(
      m.scenarios.map((g) => ({
        label: scenarioName(g.name),
        note: g.episodes > 1 ? `×${g.episodes}` : "",
        value: g.score,
        display: fmt(g.score, 1),
        mark: best[g.name]?.score,
        tip:
          `<b>${esc(scenarioName(g.name))}</b><br>EMBER Score ${fmt(g.score, 1)}` +
          `<br>Saved ${saved[g.name]?.evacuated ?? "–"} of ${saved[g.name]?.population ?? "–"} people` +
          (best[g.name] ? `<br>Best AI model here: ${esc(best[g.name].name)} (${fmt(best[g.name].score, 1)})` : ""),
      })),
      { max: 100, highlight: false },
    ) +
    dataTable(
      [{ label: "Fire" }, { label: "Versions", num: true }, { label: "EMBER Score", num: true }, { label: "People saved", num: true }, { label: "Best AI model", num: true }],
      m.scenarios.map((g) => [scenarioName(g.name), g.episodes, fmt(g.score, 1), `${saved[g.name]?.evacuated} / ${saved[g.name]?.population}`, fmt(best[g.name]?.score, 1)]),
    );

  // Technical details, as tables
  const r = m.robustness;
  const d = m.diagnostics;
  const failed = Object.values(r.failed_episode_trials ?? {}).reduce((s, v) => s + v, 0);
  let tech = dataTable(
    [{ label: "Score part" }, { label: "Value (0–1)", num: true }, { label: "Defined in worlds", num: true }],
    Object.entries(COMPONENTS).filter(([k]) => m.components[k]).map(([k, label]) => [label, fmt(m.components[k].value, 3), `${m.components[k].defined} / ${m.components[k].total}`]),
    { open: true, title: "Score parts" },
  );
  tech += dataTable(
    [{ label: "Item" }, { label: "Value", num: true }],
    [
      ["Safety", fmt(m.safety, 3)],
      ["Quality", fmt(m.quality, 3)],
      ["95% range of the score", m.ci ? `${fmt(m.ci[0])} – ${fmt(m.ci[1])}` : "–"],
      ["Worst scenario", `${fmt(r.worst_scenario)} (${scenarioName(r.worst_scenario_id ?? "")})`],
      ["Zero-score worlds", r.zero_score_episode_trials],
      ["Catastrophic worlds", r.catastrophic_episode_trials],
      ...Object.entries(r.severe_episode_trials ?? {}).map(([k, v]) => [`Severe: ${k}`, v]),
      ["Failed episodes", failed],
      ["10th-percentile world", fmt(r.p10_episode)],
      ["Blocked moves (crowding)", `${(d.congestion_block_rate * 100).toFixed(1)}% of ${d.movement_attempts.toLocaleString("en-US")}`],
      ["Exit throughput", fmt(d.exit_throughput, 3)],
      ["Peak contested cells", d.peak_contested_cells],
    ],
    { open: true, title: "Robustness and crowding" },
  );
  if (a) {
    tech += dataTable(
      [{ label: "Item" }, { label: "Value", num: true }],
      [
        ["Answered by", (a.resolved_models ?? []).join(", ") || "–"],
        ["HTTP requests", count(a.requests)],
        ["Decisions", count(a.decisions)],
        ["Input / output tokens", `${(a.input_tokens ?? 0).toLocaleString("en-US")} / ${(a.output_tokens ?? 0).toLocaleString("en-US")}`],
        ["Cost (USD)", `${a.cost_usd ?? "–"} (${a.cost_source})`],
        ["First / median response", `${duration(a.first_response_s)} / ${duration(a.median_response_s)}`],
      ],
      { open: true, title: "API use" },
    );
  }
  tech += dataTable(
    [{ label: "World" }, { label: "Seed", num: true }, { label: "Score", num: true }, { label: "Saved", num: true }, { label: "Died", num: true }, { label: "Trapped", num: true }, { label: "People", num: true }],
    m.episodes.map((e) => [scenarioName(e.scenario_id), e.seed, fmt(e.score, 1), e.evacuated, e.casualties, e.stranded, e.population]),
    { title: "Every world" },
  );
  document.getElementById("technical").innerHTML = `<p class="muted">Model ID <code>${esc(m.id)}</code></p>` + tech;

  document.getElementById("footer").innerHTML =
    `${esc(data.label)}. Pack <code>${esc(data.pack_id)}</code> · digest <code>${esc(data.pack_digest.slice(0, 16))}…</code> · scoring version ${esc(data.scoring_version)}. ${esc(data.unit)}.`;
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
  document.getElementById("verdict").textContent = error.message;
});
