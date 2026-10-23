import {
  esc,
  fmt,
  modelLink,
  scenarioName,
  table,
  COMPONENTS,
  ratio,
} from "./common.js";
import { finite, METRICS, paretoFrontier } from "./analytics.js";

export const legend = () =>
  `<div class="chart-legend"><span><i class="legend-dot decision"></i>Decision model</span><span><i class="legend-dot reference"></i>Reference agent (square)</span><span>Hover or focus marks for values. Select a name to open its profile.</span></div>`;
export function scoreBars(models, selected = "") {
  return `<div class="bar-chart">${models.map((m) => `<a class="bar-row ${m.kind === "Reference agent" ? "reference" : "decision"} ${m.id === selected ? "selected" : ""}" href="model.html?id=${encodeURIComponent(m.id)}" data-tip="${esc(`${m.name}: EMBER ${fmt(m.score)}; 95% CI ${m.ci?.map((v) => fmt(v)).join("–") ?? "—"}. ${m.kind}.`)}"><span class="bar-label">${esc(m.name)}</span><span class="bar-track"><span class="bar-fill" style="width:${finite(m.score) ? m.score : 0}%"></span>${m.ci?.every(finite) ? `<span class="ci-range" style="left:${m.ci[0]}%;width:${m.ci[1] - m.ci[0]}%"></span>` : ""}</span><strong class="bar-value">${fmt(m.score)}</strong></a>`).join("")}</div><div class="bar-axis"><span>0</span><span>25</span><span>50</span><span>75</span><span>100</span></div><p class="note">EMBER Score ↑ · 0–100 points. Whiskers show reported 95% CIs, where available.</p>`;
}
const chartValue = (v, key) =>
  key === "cost"
    ? `$${fmt(v, 4)}`
    : key === "latency"
      ? `${fmt(v)} s`
      : fmt(
          v,
          key === "safety" || key === "quality" || key === "congestion" ? 3 : 2,
        );
export function scatter(models, xKey, yKey = "score", selected = "") {
  const xMetric = METRICS[xKey],
    yMetric = METRICS[yKey];
  const rows = models.filter(
    (m) => finite(xMetric.get(m)) && finite(yMetric.get(m)),
  );
  if (!rows.length)
    return '<p class="empty">No models have both recorded metrics.</p>';
  const xMax = ["safety", "quality", "congestion"].includes(xKey)
    ? 1
    : Math.max(...rows.map(xMetric.get), 1e-6) * 1.12;
  const yMax = ["safety", "quality"].includes(yKey) ? 1 : 100;
  const X = (v) => 64 + (v / xMax) * 560,
    Y = (v) => 270 - (v / yMax) * 230;
  let body = "";
  for (let i = 0; i <= 4; i++) {
    const x = 64 + i * 140,
      y = 270 - i * 57.5;
    body += `<line class="chart-grid" x1="64" x2="624" y1="${y}" y2="${y}"/><text class="chart-axis" x="54" y="${y + 4}" text-anchor="end">${fmt((yMax * i) / 4, yMax === 1 ? 2 : 0)}</text><text class="chart-axis" x="${x}" y="291" text-anchor="middle">${xKey === "cost" ? "$" : ""}${fmt((xMax * i) / 4, xMax < 1 ? 3 : xMax < 10 ? 2 : 0)}</text>`;
  }
  if (
    xKey === "cost" &&
    rows.filter((m) => m.kind === "Decision model").length >= 3
  ) {
    const frontier = paretoFrontier(rows);
    body += `<polyline class="pareto-line" points="${frontier.map((m) => `${X(xMetric.get(m))},${Y(m.score)}`).join(" ")}"/>`;
  }
  body += rows
    .map((m) => {
      const x = X(xMetric.get(m)),
        y = Y(yMetric.get(m)),
        isRef = m.kind === "Reference agent",
        tip = `${m.name} · ${m.kind}\n${xMetric.label}: ${chartValue(xMetric.get(m), xKey)}\n${yMetric.label}: ${chartValue(yMetric.get(m), yKey)}${xKey === "cost" ? `\nCost source: ${m.api?.cost_source ?? "—"}` : ""}`;
      return `<a href="model.html?id=${encodeURIComponent(m.id)}" class="chart-point ${isRef ? "reference" : "decision"} ${m.id === selected ? "selected" : ""}" data-tip="${esc(tip)}" aria-label="${esc(tip)}">${m.id === selected ? `<circle class="selection-ring" cx="${x}" cy="${y}" r="11"/>` : ""}${isRef ? `<rect x="${x - 5}" y="${y - 5}" width="10" height="10"/>` : `<circle cx="${x}" cy="${y}" r="6"/>`}</a>`;
    })
    .join("");
  return `<div class="chart-scroll"><svg class="chart-svg" viewBox="0 0 680 340" role="group" aria-label="${esc(`${yMetric.label} versus ${xMetric.label}`)}">${body}<text class="chart-axis axis-title" x="344" y="328" text-anchor="middle">${esc(xMetric.label)} ${xMetric.direction === "higher" ? "↑ higher" : "↓ lower"} is better</text><text class="chart-axis axis-title" transform="translate(16 155) rotate(-90)" text-anchor="middle">${esc(yMetric.label)} ↑ higher is better</text></svg></div>${legend()}<p class="note">${rows.length} of ${models.length} entries have both metrics. ${xKey === "cost" ? "Line connects the descriptive score/cost Pareto frontier among decision models. Reported and estimated costs may differ in scope; this is not a price guarantee." : ""}</p><details><summary>View chart data</summary>${table(
    ["Model", xMetric.label, yMetric.label],
    rows.map((m) => [
      modelLink(m),
      esc(chartValue(xMetric.get(m), xKey)),
      esc(chartValue(yMetric.get(m), yKey)),
    ]),
    "Scatter plot values",
  )}</details>`;
}
export function heatmap(models, scenarios = null) {
  const ids = scenarios ?? [
    ...new Set(models.flatMap((m) => m.scenarios.map((s) => s.name))),
  ];
  return table(
    ["Model", ...ids.map(scenarioName)],
    models.map((m) => [
      modelLink(m),
      ...ids.map((id) => {
        const s = m.scenarios.find((s) => s.name === id);
        return finite(s?.score)
          ? `<span class="heat-cell" style="--heat:${s.score / 100}" tabindex="0" data-tip="${esc(`${m.name} · ${scenarioName(id)}: ${fmt(s.score)} points`)}">${fmt(s.score, 1)}</span>`
          : "—";
      }),
    ]),
    "Scenario × model EMBER Score. Higher is better.",
    "heatmap",
  );
}
export function componentChart(models) {
  return `<div class="component-group">${Object.entries(COMPONENTS)
    .map(
      ([key, label]) =>
        `<div><h3>${label}</h3><div class="group-bars">${models
          .map((m, i) => {
            const v = m.components[key]?.value;
            return `<div class="component-bar" tabindex="0" data-tip="${esc(`${m.name}: ${ratio(v)}; defined ${m.components[key]?.defined ?? "—"} of ${m.components[key]?.total ?? "—"} episode trials`)}"><span>${esc(m.name)}</span><span class="bar-track"><span class="bar-fill series-${i}" style="width:${finite(v) ? v * 100 : 0}%"></span></span><b>${ratio(v)}</b></div>`;
          })
          .join("")}</div></div>`,
    )
    .join(
      "",
    )}</div><p class="note">All seven component factors are 0–1; higher is better. Undefined components are shown as —, not zero.</p>`;
}
export function differenceChart(paired, a, b) {
  const max = Math.max(1, ...paired.worlds.map((w) => Math.abs(w.difference)));
  return `<p class="note">${esc(a.name)} minus ${esc(b.name)}. Positive favors the first model. This is an unweighted descriptive mean over matched world scores, not the family-balanced EMBER difference or an official paired inference.</p><div class="world-diff">${paired.worlds.map((w) => `<div class="diff-row" tabindex="0" data-tip="${esc(`${scenarioName(w.scenario_id)}, seed ${w.seed}: ${fmt(w.a)} − ${fmt(w.b)} = ${fmt(w.difference)} points`)}"><span>${esc(scenarioName(w.scenario_id))}<small>seed ${esc(w.seed)}</small></span><span class="diff-track"><i class="${w.difference >= 0 ? "positive" : "negative"}" style="left:${w.difference >= 0 ? 50 : 50 - (Math.abs(w.difference) / max) * 49}%;width:${(Math.abs(w.difference) / max) * 49}%"></i></span><b>${w.difference > 0 ? "+" : ""}${fmt(w.difference)}</b></div>`).join("")}</div>`;
}
