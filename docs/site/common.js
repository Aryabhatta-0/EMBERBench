// Shared helpers of the EMBERBench results site: data loading, plain-language formats, tooltips and charts.

// Reference programs in plain words. The model ID stays in the technical details.
const REFERENCE_NAMES = {
  "cooperative-forecast-whca-r1": "Team route planner with fire forecast",
  "forecast-shortest-path-r1": "Shortest path with fire forecast",
  "congestion-aware-v1": "Shortest path that avoids crowds",
  "fire-aware-shortest-path-v1": "Shortest path that avoids fire",
  "static-shortest-path-v1": "Shortest path, ignores fire",
  "nearest-exit-greedy-v1": "Walk to the nearest exit",
  "always-stay": "Nobody moves",
  "random-legal-v1": "Random moves",
};

export async function loadResults() {
  const data = (await import("./data/results.js")).default;
  for (const m of data.models) if (m.kind === "Reference agent" && REFERENCE_NAMES[m.id]) m.name = REFERENCE_NAMES[m.id];
  return data;
}

export const esc = (value) =>
  String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const fmt = (value, digits = 2) => (value === null || value === undefined ? "–" : Number(value).toFixed(digits));

export const isDecision = (m) => m.kind === "Decision model";

// ------------------------------------------------------------------ plain-language formats

export function money(usd, estimated = false) {
  if (usd === null || usd === undefined) return "–";
  const text = usd < 0.01 ? `$${usd.toFixed(4)}` : `$${usd.toFixed(2)}`;
  return estimated ? `≈ ${text}` : text;
}

export function duration(seconds) {
  if (seconds === null || seconds === undefined) return "–";
  if (seconds < 1) return `${Math.round(seconds * 1000)} ms`;
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 2 : 1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds - m * 60);
  return s ? `${m} min ${s} s` : `${m} min`;
}

export function count(n) {
  if (n === null || n === undefined) return "–";
  if (n >= 1e6) return `${(n / 1e6).toFixed(n >= 1e7 ? 0 : 1)} M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)} k`;
  return n.toLocaleString("en-US");
}

export const pct = (part, whole) => (whole ? `${((part / whole) * 100).toFixed(0)}%` : "–");

// The 10 scenarios in words a general reader understands.
export const SCENARIOS = {
  "v1-pub-01-single-exit-sparse": "One exit, few people",
  "v1-pub-02-single-exit-dense": "One exit, crowded room",
  "v1-pub-03-corridor-bottleneck": "Narrow corridor",
  "v1-pub-04-unequal-exits": "One near exit, one far exit",
  "v1-pub-05-central-ignition": "Fire starts in the middle",
  "v1-pub-06-threatened-exit": "Fire spreads toward an exit",
  "v1-pub-07-multiple-ignitions": "Several fires at once",
  "v1-pub-08-maze-deceptive-exit": "Maze with a misleading exit",
  "v1-pub-09-crowd-100-split-flow": "Crowd of 100, two routes",
  "v1-pub-10-route-invalidation": "Fire blocks the planned route",
};
export const scenarioName = (id) => SCENARIOS[id] ?? id.replace(/^v1-pub-\d+-/, "").replaceAll("-", " ");

// ------------------------------------------------------------------ tooltip

// One tooltip for the page. Any element with data-tip shows it on hover and focus.
export function installTooltip() {
  const tip = document.createElement("div");
  tip.id = "tip";
  tip.setAttribute("role", "tooltip");
  document.body.append(tip);
  const show = (target, x, y) => {
    tip.innerHTML = target.dataset.tip;
    tip.classList.add("on");
    const w = tip.offsetWidth;
    const h = tip.offsetHeight;
    tip.style.left = `${Math.min(x + 14, window.innerWidth - w - 8)}px`;
    tip.style.top = `${Math.max(y - h - 12, 8)}px`;
  };
  document.addEventListener("mousemove", (e) => {
    const target = e.target.closest?.("[data-tip]");
    if (target) show(target, e.clientX, e.clientY);
    else tip.classList.remove("on");
  });
  document.addEventListener("focusin", (e) => {
    const target = e.target.closest?.("[data-tip]");
    if (!target) return;
    const r = target.getBoundingClientRect();
    show(target, r.left + Math.min(r.width / 2, 240), r.top);
  });
  document.addEventListener("focusout", () => tip.classList.remove("on"));
}

// ------------------------------------------------------------------ charts

// Horizontal bars, one row per model or item. The highlighted row is in the series color; the others are
// recessive gray, so one model stands out against the rest.
// rows: [{id, label, value, display, tip, highlight, kind, estimated, note, mark}]; mark is a reference tick.
export function barChart(rows, { max = null, highlight = true, clickable = false } = {}) {
  const top = max ?? Math.max(...rows.map((r) => r.value ?? 0), 0) * 1.05;
  const body = rows
    .map((r) => {
      const w = top > 0 ? Math.max(0, Math.min(1, (r.value ?? 0) / top)) * 100 : 0;
      const cls = ["brow", r.highlight ? "hl" : "", !highlight || r.highlight ? "" : "dim", r.kind === "Reference agent" ? "ref" : "", clickable ? "click" : ""]
        .filter(Boolean)
        .join(" ");
      let bar = r.value === null || r.value === undefined ? "" : `<span class="bbar${r.estimated ? " est" : ""}" style="width:${w.toFixed(2)}%"></span>`;
      if (r.mark !== undefined && r.mark !== null && top > 0) {
        bar += `<span class="bmark" style="left:${(Math.max(0, Math.min(1, r.mark / top)) * 100).toFixed(2)}%"></span>`;
      }
      return `<div class="${cls}" ${r.id ? `data-id="${esc(r.id)}"` : ""} tabindex="0" data-tip="${esc(r.tip ?? `${r.label}: ${r.display}`)}">
        <span class="blabel">${esc(r.label)}${r.note ? ` <span class="bnote">${esc(r.note)}</span>` : ""}</span>
        <span class="btrack">${bar}</span>
        <span class="bval">${esc(r.display)}</span></div>`;
    })
    .join("");
  return `<div class="bchart">${body}</div>`;
}

// One stacked horizontal bar (part to whole). parts: [{label, value, color}]
export function stackedBar(parts, total) {
  const width = 1040;
  const h = 28;
  let svg = `<svg viewBox="0 0 ${width} ${h}" preserveAspectRatio="none" role="img" aria-label="People outcomes">`;
  let x = 0;
  parts.forEach((part, i) => {
    const w = (part.value / total) * width;
    if (w <= 0) return;
    const last = i === parts.length - 1 || parts.slice(i + 1).every((p) => p.value === 0);
    const segW = Math.max(0, w - (last ? 0 : 2)); // 2px surface gap between fills
    svg += `<g class="mark" tabindex="0" data-tip="${esc(`${part.label}: ${part.value} of ${total} people (${pct(part.value, total)})`)}">`;
    svg += `<rect x="${x}" y="0" width="${segW}" height="${h}" rx="${i === 0 || last ? 4 : 0}" fill="${part.color}"/>`;
    svg += `</g>`;
    x += w;
  });
  return svg + `</svg>`;
}

export function dataTable(headers, rows, { open = false, title = "Show table" } = {}) {
  const head = headers.map((h) => `<th${h.num ? ' class="num"' : ""}>${esc(h.label)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${r.map((c, i) => `<td${headers[i].num ? ' class="num"' : ""}>${esc(c)}</td>`).join("")}</tr>`)
    .join("");
  return `<details${open ? " open" : ""}><summary>${esc(title)}</summary><div class="table-wrap"><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></div></details>`;
}

export function openModel(id) {
  const url = `model.html?id=${encodeURIComponent(id)}`;
  const win = window.open(url, `ember-${id}`, "popup,width=1180,height=900");
  if (win) win.focus();
  else window.location.href = url; // pop-up blocked: open in this window
}
