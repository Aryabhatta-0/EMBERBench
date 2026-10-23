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
  getSelection,
  selectionBar,
  bindSelection,
  provider,
  rankOf,
  errorPage,
} from "./common.js";
import {
  METRICS,
  filterModels,
  sortModels,
  finite,
  leaderboardOrder,
} from "./analytics.js";

const columns = [
  ["score", "EMBER"],
  ["safety", "Safety"],
  ["quality", "Quality"],
  ["evacuated", "Evacuated"],
  ["casualties", "Avoidable deaths"],
  ["stranded", "Avoidable stranded"],
  ["blocked", "Blocked moves"],
  ["requests", "API requests"],
  ["input", "Input tokens"],
  ["cost", "API cost"],
  ["latency", "Mean decision latency"],
];
function value(m, key) {
  const v = METRICS[key].get(m);
  if (!finite(v)) return "—";
  if (key === "cost") return costText(m);
  if (key === "latency") return duration(v);
  if (key === "safety" || key === "quality") return fmt(v, 3);
  return key === "score" ? fmt(v) : count(v);
}
async function main() {
  const data = await loadResults();
  shell(data, "leaderboard");
  const params = new URLSearchParams(location.search);
  const state = {
    query: params.get("q") ?? "",
    kind: params.get("kind") ?? "all",
    creator: params.get("provider") ?? "all",
    sort: params.get("metric") ?? "score",
    direction: params.get("direction") ?? "desc",
  };
  if (!Object.hasOwn(METRICS, state.sort)) state.sort = "score";
  if (!["asc", "desc"].includes(state.direction)) state.direction = "desc";
  document.getElementById("main").innerHTML =
    `<header class="page-head"><div><p class="eyebrow">Results workspace</p><h1>Leaderboard</h1><p class="muted">A shared public pack. Distinct decision policies. Inspect the evidence behind each score.</p></div><a class="button" href="methodology.html">How to read these results</a></header><section class="panel leaderboard-panel"><div class="toolbar"><label class="field search-field">Search models<input id="search" type="search" placeholder="Model name or provider" value="${esc(state.query)}"></label><label class="field">Entry type<select id="kind"><option value="all">All entries</option><option>Decision model</option><option>Reference agent</option></select></label><label class="field">Provider<select id="provider"><option value="all">All providers</option>${[
      ...new Set(data.models.map(provider)),
    ]
      .sort()
      .map((p) => `<option>${esc(p)}</option>`)
      .join(
        "",
      )}</select></label><label class="field">Sort metric<select id="metric">${columns.map(([key, label]) => `<option value="${key}">${label} ${METRICS[key].direction === "higher" ? "↑" : "↓"}</option>`).join("")}</select></label><button class="button small" id="reset-filters">Reset filters</button></div><div class="section-head"><p id="result-count" class="muted" aria-live="polite"></p><p class="note">↑ Higher is better · ↓ Lower is better · — Unavailable</p></div><div id="selection"></div><p id="selection-status" class="sr-only" role="status"></p><div id="leaderboard"></div><p class="note">Rank follows the scoring specification: score rounded to 0.1 points, then Safety, avoidable losses per trial, worst scenario, and model ID. Custom sorts do not redefine rank. Sorting order is not statistical significance.</p><p class="note">Safety / Quality: 0–1 factors. Evacuated and avoidable losses sum across worlds and trials. ≈ marks list-price API cost estimates. API metadata may cover only the recorded invocation of a resumed run; missing API data is never ranked as zero.</p><p class="note">Reported 95% CIs use the score report’s resampling procedure. Each published model has ${esc([...new Set(data.models.map((m) => m.totals.trials))].join(", "))} trial in this snapshot; these CIs cannot capture unobserved between-trial variability.</p></section>`;
  for (const [id, key] of [
    ["kind", "kind"],
    ["provider", "creator"],
    ["metric", "sort"],
  ])
    document.getElementById(id).value = state[key];
  const render = () => {
    const oldScroll = document.querySelector(".leaderboard-scroll");
    const scroll = {
      left: oldScroll?.scrollLeft ?? 0,
      top: oldScroll?.scrollTop ?? 0,
    };
    const active = document.activeElement;
    const focusSort = active?.dataset?.sort;
    const focusModel = active?.hasAttribute("data-select-model")
      ? active.value
      : null;
    const filtered = filterModels(data.models, state),
      models =
        state.sort === "score" && state.direction === "desc"
          ? leaderboardOrder(filtered)
          : sortModels(filtered, state.sort, state.direction),
      ids = getSelection(data);
    document.getElementById("result-count").textContent =
      `${models.length} of ${data.models.length} entries · ${METRICS[state.sort].label} ${state.direction === "asc" ? "ascending" : "descending"}`;
    document.getElementById("selection").innerHTML = selectionBar(data);
    const th = (key, label) =>
      `<th scope="col" aria-sort="${state.sort === key ? (state.direction === "asc" ? "ascending" : "descending") : "none"}"><button data-sort="${key}" data-tip="${esc(METRICS[key].help)}">${esc(label)} <span>${METRICS[key].direction === "higher" ? "↑" : "↓"}</span></button></th>`;
    document.getElementById("leaderboard").innerHTML =
      `<div class="table-wrap leaderboard-scroll" role="region" aria-label="Sortable public development leaderboard" tabindex="0"><table class="leaderboard-table"><caption class="sr-only">Public-development results. Select 2 to 4 models for comparison. Sort metric headers to reorder.</caption><thead><tr><th scope="col">Compare</th><th scope="col">Rank</th><th scope="col">Model</th><th scope="col">Type</th>${columns.map(([key, label]) => th(key, label) + (key === "score" ? '<th scope="col"><span tabindex="0" data-tip="Reported 95% bootstrap confidence interval; one trial cannot measure between-trial variability.">95% CI</span></th>' : "")).join("")}</tr></thead><tbody>${models.map((m) => `<tr class="${ids.includes(m.id) ? "selected" : ""}"><td><input type="checkbox" data-select-model value="${esc(m.id)}" aria-label="Compare ${esc(m.name)}" ${ids.includes(m.id) ? "checked" : ""}></td><td class="num rank">${rankOf(data.models, m)}</td><th scope="row" class="model-cell">${modelLink(m)}<span class="model-provider">${esc(provider(m))}</span></th><td>${kindBadge(m)}</td>${columns.map(([key]) => `<td class="num ${key === "score" ? "score-cell" : ""}">${key === "score" ? `<span class="score-track"><span class="score-fill" style="width:${finite(m.score) ? m.score : 0}%"></span></span><b>${value(m, key)}</b>` : esc(value(m, key))}</td>${key === "score" ? `<td class="num ci-cell"><span>${ciText(m)}</span>${m.ci?.every(finite) ? `<span class="ci-track"><i class="ci-range" style="left:${m.ci[0]}%;width:${m.ci[1] - m.ci[0]}%"></i><i class="ci-dot" style="left:${m.score}%"></i></span>` : ""}</td>` : ""}`).join("")}</tr>`).join("") || `<tr><td colspan="16"><div class="empty">No matching models. Try another search or reset filters.</div></td></tr>`}</tbody></table></div>`;
    for (const b of document.querySelectorAll("[data-sort]"))
      b.addEventListener("click", () => {
        const key = b.dataset.sort;
        state.direction =
          state.sort === key
            ? state.direction === "desc"
              ? "asc"
              : "desc"
            : METRICS[key].direction === "higher"
              ? "desc"
              : "asc";
        state.sort = key;
        document.getElementById("metric").value = key;
        update();
      });
    bindSelection(data, render);
    const nextScroll = document.querySelector(".leaderboard-scroll");
    nextScroll.scrollLeft = scroll.left;
    nextScroll.scrollTop = scroll.top;
    if (focusSort)
      document
        .querySelector(`[data-sort="${CSS.escape(focusSort)}"]`)
        ?.focus({ preventScroll: true });
    if (focusModel)
      document
        .querySelector(`[data-select-model][value="${CSS.escape(focusModel)}"]`)
        ?.focus({ preventScroll: true });
  };
  const update = () => {
    const p = new URLSearchParams({
      q: state.query,
      kind: state.kind,
      provider: state.creator,
      metric: state.sort,
      direction: state.direction,
    });
    history.replaceState(null, "", `?${p}`);
    render();
  };
  document.getElementById("search").addEventListener("input", (e) => {
    state.query = e.target.value;
    update();
  });
  for (const [id, key] of [
    ["kind", "kind"],
    ["provider", "creator"],
    ["metric", "sort"],
  ])
    document.getElementById(id).addEventListener("change", (e) => {
      state[key] = e.target.value;
      if (key === "sort")
        state.direction =
          METRICS[state.sort].direction === "higher" ? "desc" : "asc";
      update();
    });
  document.getElementById("reset-filters").addEventListener("click", () => {
    state.query = "";
    state.kind = "all";
    state.creator = "all";
    state.sort = "score";
    state.direction = "desc";
    document.getElementById("search").value = "";
    document.getElementById("kind").value = "all";
    document.getElementById("provider").value = "all";
    document.getElementById("metric").value = "score";
    update();
  });
  render();
}
main().catch(errorPage);
