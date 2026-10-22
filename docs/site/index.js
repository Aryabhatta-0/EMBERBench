import { esc, fmt, installTooltip, loadResults } from "./common.js";

function scoreBar(m) {
  const x = (v) => Math.max(0, Math.min(100, v)) * 2; // viewBox width 200
  const w = x(m.score ?? 0);
  const r = Math.min(4, w);
  const bar = w > 0 ? `<path class="bar" d="M0,4h${w - r}a${r},${r} 0 0 1 ${r},${r}v${10 - 2 * r}a${r},${r} 0 0 1 -${r},${r}h-${w - r}z"/>` : "";
  let ci = "";
  if (m.ci && m.ci[1] - m.ci[0] > 0.05) {
    const [a, b] = [x(m.ci[0]), x(m.ci[1])];
    ci = `<path d="M${a},6v6M${a},9H${b}M${b},6v6" stroke="var(--text-primary)" stroke-width="1.5" fill="none"/>`;
  }
  return `<svg viewBox="0 0 200 18" preserveAspectRatio="none" aria-hidden="true">
    <rect class="track" x="0" y="4" width="200" height="10" rx="4"/>${bar}${ci}</svg>`;
}

function modelUrl(m) {
  return `model.html?id=${encodeURIComponent(m.id)}`;
}

function openModel(m) {
  const win = window.open(modelUrl(m), `ember-${m.id}`, "popup,width=1180,height=900");
  if (win) win.focus();
  else window.location.href = modelUrl(m); // pop-up blocked: open in this window
}

function render(data, kind) {
  const rows = document.getElementById("rows");
  const models = data.models.filter((m) => kind === "all" || m.kind === kind);
  rows.innerHTML = models
    .map((m, i) => {
      const t = m.totals;
      const ciText = m.ci ? `${fmt(m.ci[0])}–${fmt(m.ci[1])}` : "–";
      const tip = `<b>${esc(m.name)}</b><br>EMBER Score ${fmt(m.score)} (95% CI ${ciText})<br>Safety ${fmt(m.safety, 3)} · Quality ${fmt(m.quality, 3)}`;
      return `<tr class="row" data-index="${data.models.indexOf(m)}">
        <td class="rank">${i + 1}</td>
        <td><a class="model-link" href="${modelUrl(m)}" target="_blank" rel="noopener">${esc(m.name)}</a></td>
        <td><span class="kind ${m.kind === "Decision model" ? "decision" : ""}">${esc(m.kind)}</span></td>
        <td class="score-cell"><div class="score-wrap" data-tip="${esc(tip)}"><span class="score-val">${fmt(m.score)}</span>${scoreBar(m)}</div></td>
        <td class="num muted">${ciText}</td>
        <td class="num">${fmt(m.safety, 3)}</td>
        <td class="num">${fmt(m.quality, 3)}</td>
        <td class="num">${t.evacuated} / ${t.escapable}</td>
        <td class="num">${t.avoidable_casualties}</td>
      </tr>`;
    })
    .join("");
}

async function main() {
  installTooltip();
  const data = await loadResults();
  document.getElementById("label").textContent = data.label;
  document.getElementById("unit").textContent = data.unit;
  document.getElementById("footer").innerHTML =
    `Suite <code>${esc(data.suite_id)}</code> · pack <code>${esc(data.pack_id)}</code> · digest <code>${esc(data.pack_digest.slice(0, 16))}…</code> · scoring version ${esc(data.scoring_version)}. ` +
    `The suite is public, so these are development results, not hidden-suite scores.`;

  let kind = "all";
  render(data, kind);
  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      kind = chip.dataset.kind;
      for (const c of document.querySelectorAll(".chip")) c.setAttribute("aria-pressed", String(c === chip));
      render(data, kind);
    });
  }
  document.getElementById("rows").addEventListener("click", (e) => {
    const row = e.target.closest("tr.row");
    if (!row) return;
    e.preventDefault();
    openModel(data.models[Number(row.dataset.index)]);
  });
}

main().catch((error) => {
  document.getElementById("rows").innerHTML = `<tr><td colspan="9">Could not load the results: ${esc(error.message)}</td></tr>`;
});
