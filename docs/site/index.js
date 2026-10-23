import { barChart, duration, esc, fmt, installTooltip, isDecision, loadResults, money, openModel, pct } from "./common.js";

const tile = (label, value, note = "") =>
  `<div class="tile"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div>` +
  (note ? `<div class="note">${esc(note)}</div>` : "") + `</div>`;

function costText(m) {
  if (!isDecision(m)) return "free";
  return money(m.api?.cost_usd, m.api?.cost_source === "list price");
}

function renderRanking(data, kind) {
  const models = data.models.filter((m) => kind === "all" || m.kind === kind);
  document.getElementById("ranking").innerHTML = models
    .map((m) => {
      const rank = data.models.indexOf(m) + 1;
      const t = m.totals;
      const w = Math.max(0, Math.min(100, m.score ?? 0));
      const ci = m.ci && m.ci[1] - m.ci[0] > 0.05 ? `<br>95% range ${fmt(m.ci[0])} to ${fmt(m.ci[1])}` : "";
      const tip =
        `<b>${esc(m.name)}</b><br>EMBER Score ${fmt(m.score)} of 100${ci}<br>` +
        `Saved ${t.evacuated} of ${t.escapable} people · ${t.avoidable_casualties} avoidable deaths` +
        (isDecision(m) ? `<br>Cost ${esc(costText(m))} · finished in ${esc(duration(m.api?.wall_clock_s))}` : "");
      return `<a class="rrow ${isDecision(m) ? "" : "ref"}" href="model.html?id=${encodeURIComponent(m.id)}" data-id="${esc(m.id)}" data-tip="${esc(tip)}">
        <span class="rnum">${rank}</span>
        <span class="rname">${esc(m.name)}<span class="rkind">${isDecision(m) ? "AI decision model" : "Reference program"}</span></span>
        <span class="rscore"><span class="btrack"><span class="bbar" style="width:${w}%"></span></span><b>${fmt(m.score)}</b></span>
        <span class="num">${t.evacuated} <span class="muted">(${pct(t.evacuated, t.escapable)})</span></span>
        <span class="num">${esc(costText(m))}</span>
      </a>`;
    })
    .join("");
}

function render(data) {
  const decision = data.models.filter(isDecision);
  const best = decision[0];
  const people = data.models[0].totals.escapable;
  document.getElementById("facts").innerHTML = [
    tile("Models tested", `${data.models.length}`, `${decision.length} AI decision models, ${data.models.length - decision.length} reference programs`),
    tile("Simulated fires", `${data.models[0].episodes.length}`, "10 building layouts"),
    tile("People in danger", `${people}`, "in all fires together"),
    tile("Best AI model", best ? best.name : "–", best ? `EMBER Score ${fmt(best.score)} · saved ${best.totals.evacuated} of ${people}` : ""),
  ].join("");

  const byCost = [...decision].sort((a, b) => (a.api?.cost_usd ?? Infinity) - (b.api?.cost_usd ?? Infinity));
  document.getElementById("cost").innerHTML = barChart(
    byCost.map((m) => {
      const est = m.api?.cost_source === "list price";
      return {
        id: m.id,
        label: m.name,
        value: m.api?.cost_usd,
        display: costText(m),
        estimated: est,
        tip: `<b>${esc(m.name)}</b><br>${esc(costText(m))} for all 19 fires${est ? "<br>Estimate from the published price per token" : ""}<br>EMBER Score ${fmt(m.score)}`,
      };
    }),
    { highlight: false, clickable: true },
  );

  const byTime = [...decision].sort((a, b) => (a.api?.wall_clock_s ?? Infinity) - (b.api?.wall_clock_s ?? Infinity));
  document.getElementById("time").innerHTML = barChart(
    byTime.map((m) => ({
      id: m.id,
      label: m.name,
      value: m.api?.wall_clock_s,
      display: duration(m.api?.wall_clock_s),
      tip: `<b>${esc(m.name)}</b><br>${esc(duration(m.api?.wall_clock_s))} for all 19 fires<br>${esc(duration(m.api?.mean_act_latency_s))} per decision on average`,
    })),
    { highlight: false, clickable: true },
  );
}

async function main() {
  installTooltip();
  const data = await loadResults();
  document.getElementById("label").textContent = data.label;
  document.getElementById("footer").innerHTML =
    `Suite <code>${esc(data.suite_id)}</code> · pack <code>${esc(data.pack_id)}</code> · digest <code>${esc(data.pack_digest.slice(0, 16))}…</code> · scoring version ${esc(data.scoring_version)}. ` +
    `${esc(data.unit)}. The suite is public, so these are development results, not hidden-suite scores.`;

  render(data);
  let kind = "all";
  renderRanking(data, kind);
  for (const chip of document.querySelectorAll(".chip")) {
    chip.addEventListener("click", () => {
      kind = chip.dataset.kind;
      for (const c of document.querySelectorAll(".chip")) c.setAttribute("aria-pressed", String(c === chip));
      renderRanking(data, kind);
    });
  }
  document.addEventListener("click", (e) => {
    const row = e.target.closest("[data-id]");
    if (!row || e.ctrlKey || e.metaKey || e.shiftKey) return; // keep the browser's own new-tab clicks
    e.preventDefault();
    openModel(row.dataset.id);
  });
  document.addEventListener("keydown", (e) => {
    const row = e.target.closest?.(".brow.click");
    if (row && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      openModel(row.dataset.id);
    }
  });
}

main().catch((error) => {
  document.getElementById("ranking").innerHTML = `<p>Could not load the results: ${esc(error.message)}</p>`;
});
