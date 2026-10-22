// Shared helpers of the EMBERBench results site: data loading, tooltips and small SVG charts.

export async function loadResults() {
  return (await import("./data/results.js")).default;
}

export const esc = (value) =>
  String(value).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

export const fmt = (value, digits = 2) => (value === null || value === undefined ? "–" : Number(value).toFixed(digits));

export const NS = "http://www.w3.org/2000/svg";

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
    show(target, r.left + r.width / 2, r.top);
  });
  document.addEventListener("focusout", () => tip.classList.remove("on"));
}

// A rectangle with only its data end rounded (4px), anchored square on the baseline.
function endRoundedRect(x, y, w, h, horizontal) {
  const r = Math.min(4, horizontal ? h / 2 : w / 2, horizontal ? w : h);
  if (w <= 0 || h <= 0) return "";
  if (horizontal) {
    return `M${x},${y}h${w - r}a${r},${r} 0 0 1 ${r},${r}v${h - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z`;
  }
  return `M${x},${y + h}v-${h - r}a${r},${r} 0 0 1 ${r},-${r}h${w - 2 * r}a${r},${r} 0 0 1 ${r},${r}v${h - r}z`;
}

// Horizontal bars, one series. rows: [{label, value, tip, display}]
export function hbarChart(rows, { max = 100, labelWidth = 230, digits = 2, width = 480, ticks = null } = {}) {
  const barH = 18;
  const gap = 10;
  const valueW = 48;
  const plotW = width - labelWidth - valueW;
  const height = rows.length * (barH + gap) + 18;
  ticks ??= [0, 0.25, 0.5, 0.75, 1].map((t) => t * max);
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  for (const t of ticks) {
    const x = labelWidth + (t / max) * plotW;
    svg += `<line class="gridline" x1="${x}" x2="${x}" y1="0" y2="${height - 18}"/>`;
    svg += `<text x="${x}" y="${height - 4}" text-anchor="middle">${esc(fmt(t, max <= 1 ? 2 : 0))}</text>`;
  }
  rows.forEach((row, i) => {
    const y = i * (barH + gap);
    const v = row.value ?? 0;
    const w = Math.max(0, Math.min(1, v / max)) * plotW;
    svg += `<g class="mark" tabindex="0" data-tip="${esc(row.tip ?? `${row.label}: ${fmt(row.value, digits)}`)}">`;
    svg += `<rect class="hit" x="0" y="${y - gap / 2}" width="${width}" height="${barH + gap}"/>`;
    svg += `<text x="${labelWidth - 10}" y="${y + barH / 2 + 4}" text-anchor="end">${esc(row.label)}</text>`;
    svg += `<path class="bar" d="${endRoundedRect(labelWidth, y, w, barH, true)}"/>`;
    svg += `<text class="value" x="${labelWidth + w + 6}" y="${y + barH / 2 + 4}">${esc(row.display ?? fmt(row.value, digits))}</text>`;
    svg += `</g>`;
  });
  return svg + `</svg>`;
}

// Vertical bars, one series. rows: [{label, value, tip}]
export function vbarChart(rows, { max = 100, height = 240, width = 1040 } = {}) {
  const left = 34;
  const bottom = 34;
  const plotH = height - bottom - 8;
  const slot = (width - left) / rows.length;
  const barW = Math.max(6, slot - 4); // 2px surface gap on each side
  let svg = `<svg viewBox="0 0 ${width} ${height}" role="img">`;
  for (const t of [0, 25, 50, 75, 100].map((p) => (p / 100) * max)) {
    const y = 8 + plotH - (t / max) * plotH;
    svg += `<line class="gridline" x1="${left}" x2="${width}" y1="${y}" y2="${y}"/>`;
    svg += `<text x="${left - 6}" y="${y + 4}" text-anchor="end">${esc(fmt(t, 0))}</text>`;
  }
  rows.forEach((row, i) => {
    const x = left + i * slot + (slot - barW) / 2;
    const h = (Math.max(0, Math.min(max, row.value ?? 0)) / max) * plotH;
    const y = 8 + plotH - h;
    svg += `<g class="mark" tabindex="0" data-tip="${esc(row.tip)}">`;
    svg += `<rect class="hit" x="${left + i * slot}" y="0" width="${slot}" height="${height}"/>`;
    svg += `<path class="bar" d="${endRoundedRect(x, y, barW, h, false)}"/>`;
    svg += `<text x="${left + i * slot + slot / 2}" y="${height - bottom + 16}" text-anchor="middle">${esc(row.label)}</text>`;
    svg += `</g>`;
  });
  return svg + `</svg>`;
}

// One stacked horizontal bar (part to whole). parts: [{label, value, color}]
export function stackedBar(parts, total) {
  const width = 1040;
  const h = 28;
  let svg = `<svg viewBox="0 0 ${width} ${h}" role="img">`;
  let x = 0;
  parts.forEach((part, i) => {
    const w = (part.value / total) * width;
    if (w <= 0) return;
    const last = i === parts.length - 1 || parts.slice(i + 1).every((p) => p.value === 0);
    const segW = Math.max(0, w - (last ? 0 : 2)); // 2px surface gap between fills
    const pct = ((part.value / total) * 100).toFixed(1);
    svg += `<g class="mark" tabindex="0" data-tip="${esc(`${part.label}: ${part.value} of ${total} (${pct}%)`)}">`;
    svg += `<rect x="${x}" y="0" width="${segW}" height="${h}" rx="${i === 0 || last ? 4 : 0}" fill="${part.color}"/>`;
    svg += `</g>`;
    x += w;
  });
  return svg + `</svg>`;
}

export function dataTable(headers, rows) {
  const head = headers.map((h) => `<th${h.num ? ' class="num"' : ""}>${esc(h.label)}</th>`).join("");
  const body = rows
    .map((r) => `<tr>${r.map((c, i) => `<td${headers[i].num ? ' class="num"' : ""}>${esc(c)}</td>`).join("")}</tr>`)
    .join("");
  return `<details><summary>Show table</summary><table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table></details>`;
}

// Short world label: "v1-pub-10-route-invalidation" seed 3 -> "10·s3"; deterministic worlds -> "01".
export function worldLabel(episode, stochastic) {
  const n = episode.scenario_id.match(/^v1-pub-(\d+)/)?.[1] ?? episode.scenario_id;
  return stochastic.has(episode.scenario_id) ? `${n}·s${episode.seed}` : n;
}
