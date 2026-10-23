import {
  provider,
  rankOf,
  selectedIds,
  leaderboardOrder,
} from "./analytics.js";

export const esc = (v) =>
  String(v ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
export const fmt = (v, digits = 2) =>
  Number.isFinite(v) ? v.toFixed(digits) : "—";
export const count = (v) =>
  Number.isFinite(v) ? v.toLocaleString("en-US") : "—";
export const money = (v, estimated = false) =>
  Number.isFinite(v)
    ? `${estimated ? "≈ " : ""}$${v.toFixed(v < 0.01 ? 4 : 2)}`
    : "—";
export const duration = (v) =>
  Number.isFinite(v)
    ? v < 1
      ? `${Math.round(v * 1000)} ms`
      : `${fmt(v, 2)} s`
    : "—";
export const isDecision = (m) => m.kind === "Decision model";
export const ratio = (v) => (Number.isFinite(v) ? fmt(v, 3) : "—");
export const modelLink = (m) =>
  `<a class="model-name" href="model.html?id=${encodeURIComponent(m.id)}">${esc(m.name)}</a>`;
export const kindBadge = (m) =>
  `<span class="badge ${isDecision(m) ? "decision" : "reference"}">${esc(m.kind)}</span>`;
export const ciText = (m) =>
  m.ci?.every(Number.isFinite) ? `${fmt(m.ci[0])}–${fmt(m.ci[1])}` : "—";
export const costText = (m) =>
  money(m.api?.cost_usd, m.api?.cost_source === "list price");
export const scenarioName = (id) =>
  id
    .replace(/^v1-pub-\d+-/, "")
    .replaceAll("-", " ")
    .replace(/^./, (c) => c.toUpperCase());
export const COMPONENTS = {
  evacuation: "Evacuation",
  casualty: "Casualty penalty",
  stranding: "Stranding penalty",
  time: "Time efficiency",
  hazard: "Hazard avoidance",
  path: "Path efficiency",
  replanning: "Replanning",
};
export const REPO = "https://github.com/Aryabhatta-0/EMBERBench";

export async function loadResults() {
  const data = (await import("./data/results.js")).default;
  if (!data.models?.length || !data.pack_digest || !data.scoring_version)
    throw new Error("Results data is incomplete.");
  // New builds carry full-precision report ranks. Legacy snapshots predate
  // that field; their safe, unambiguous values use the scoring tie policy.
  if (!data.models.every((m) => Number.isInteger(m.rank)))
    leaderboardOrder(data.models).forEach((m, i) => {
      m.rank = i + 1;
    });
  return data;
}
function safeStorage(key, value) {
  try {
    if (value !== undefined) localStorage.setItem(key, value);
    else return localStorage.getItem(key);
  } catch {
    /* Storage is optional. */
  }
  return null;
}
export function getSelection(data) {
  return selectedIds(safeStorage("ember-compare"), data.models);
}
export function setSelection(ids) {
  safeStorage("ember-compare", JSON.stringify([...new Set(ids)].slice(0, 4)));
}
export function selectionBar(data) {
  const ids = getSelection(data);
  return `<div class="selection-bar" aria-live="polite"><span><b>${ids.length} of 4</b> selected for comparison</span><a class="button primary" href="compare.html?models=${encodeURIComponent(ids.join(","))}" ${ids.length < 2 ? 'aria-disabled="true" tabindex="-1"' : ""}>Compare models</a><button class="button small" id="clear-selection" type="button">Clear</button><span class="muted">Select 2–4. Ranks refer to this public pack.</span></div>`;
}
export function bindSelection(data, rerender = () => {}) {
  for (const input of document.querySelectorAll("[data-select-model]"))
    input.addEventListener("change", () => {
      let ids = getSelection(data);
      if (input.checked && !ids.includes(input.value)) {
        if (ids.length === 4) {
          input.checked = false;
          document.getElementById("selection-status").textContent =
            "Four models selected. Remove one to choose another.";
          return;
        }
        ids.push(input.value);
      } else ids = ids.filter((id) => id !== input.value);
      setSelection(ids);
      rerender();
    });
  document.getElementById("clear-selection")?.addEventListener("click", () => {
    setSelection([]);
    rerender();
  });
}
export function metricCard(label, value, note = "") {
  return `<div class="metric"><span class="metric-label">${esc(label)}</span><strong class="metric-value">${esc(value)}</strong><span class="metric-note">${esc(note)}</span></div>`;
}
export function section(title, note, body, extra = "") {
  return `<section class="panel ${extra}"><div class="section-head"><div><h2>${esc(title)}</h2>${note ? `<p class="muted">${esc(note)}</p>` : ""}</div></div>${body}</section>`;
}
export function table(headers, rows, caption = "", cls = "") {
  return `<div class="table-wrap" role="region" aria-label="${esc(caption || "Results table")}" tabindex="0"><table class="${cls}">${caption ? `<caption class="sr-only">${esc(caption)}</caption>` : ""}<thead><tr>${headers.map((h) => `<th scope="col">${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`;
}
export function provenance(data, m = null) {
  const fields = [
    ["EnvironmentPack", data.pack_id],
    ["Pack digest", data.pack_digest],
    ["Suite ID", data.suite_id],
    ["Scoring version", data.scoring_version],
    ["Constants digest", data.constants_digest],
    ["Evaluation manifest", data.evaluation_manifest_hash],
    ["Suite hash", data.suite_hash],
    ["Data generator", "tools/build_results_site.py"],
    ["Evaluation date", null],
  ];
  if (m)
    fields.push(
      ["Model ID", m.id],
      ["Trials", m.totals?.trials],
      ["Result digests", m.provenance?.result_digests?.join(", ")],
    );
  return `<dl class="provenance">${fields.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd><code>${esc(v ?? "—")}</code></dd></div>`).join("")}</dl><p class="note">Generated official report data. Missing fields are shown as —. No evaluation timestamp is recorded in this data snapshot. Scoring and benchmark results are unchanged.</p>`;
}
export function shell(data, active) {
  const pages = [
    ["overview", "index.html", "Overview"],
    ["leaderboard", "leaderboard.html", "Leaderboard"],
    ["compare", "compare.html", "Compare models"],
    ["model", "model.html", "Model details"],
    ["scenarios", "scenarios.html", "Scenarios"],
    ["methodology", "methodology.html", "Methodology"],
  ];
  const main = document.querySelector("main");
  main.classList.add("container");
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<a class="skip-link" href="#main">Skip to content</a><header class="site-header"><a href="index.html" class="brand" aria-label="EMBERBench overview"><span class="brand-mark" aria-hidden="true">E<span>↗</span></span><span>EMBER<span class="brand-light">Bench</span><small class="brand-sub">Decision intelligence / V1</small></span></a><button class="nav-toggle button" aria-expanded="false" aria-controls="site-nav">Menu</button><nav class="site-nav" id="site-nav" aria-label="Main navigation">${pages.map(([id, url, label]) => `<a href="${url}" class="${id === active ? "active" : ""}" ${id === active ? 'aria-current="page"' : ""}>${label}</a>`).join("")}</nav><div class="header-actions"><a class="repo-link" href="${REPO}">Repository ↗</a><button class="theme-toggle button small" type="button" aria-label="Switch color theme">Dark</button></div></header><div class="status-banner"><span class="status-dot" aria-hidden="true"></span><span class="status-copy">${esc(data.label)}</span><a href="methodology.html#public-development">Read the scope</a></div>`,
  );
  main.insertAdjacentHTML(
    "afterend",
    `<footer class="site-footer container"><div><a href="index.html" class="footer-brand">EMBERBench</a><p>Decision-making under dynamic fire evacuation.</p><p class="muted">Public development results. Generated from official score reports.</p></div><div><span>Pack <code>${esc(data.pack_digest.slice(0, 12))}…</code></span><span>Scoring ${esc(data.scoring_version)} · ${esc(data.suite_id)}</span><a href="methodology.html#provenance">Data provenance</a><a href="${REPO}/blob/main/docs/scoring.md">Scoring specification ↗</a></div></footer>`,
  );
  const themeButton = document.querySelector(".theme-toggle");
  const applyTheme = (theme) => {
    document.documentElement.dataset.theme = theme;
    themeButton.textContent = theme === "dark" ? "Light" : "Dark";
    themeButton.setAttribute(
      "aria-label",
      `Switch to ${theme === "dark" ? "light" : "dark"} theme`,
    );
  };
  applyTheme(
    safeStorage("ember-theme") ??
      (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"),
  );
  themeButton.addEventListener("click", () => {
    const theme =
      document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    safeStorage("ember-theme", theme);
    applyTheme(theme);
  });
  const navButton = document.querySelector(".nav-toggle");
  const updateOffset = () =>
    document.documentElement.style.setProperty(
      "--header-offset",
      `${Math.ceil(document.querySelector(".site-header").getBoundingClientRect().height + 14)}px`,
    );
  const toggleNav = (expanded) => {
    navButton.setAttribute("aria-expanded", String(expanded));
    document.getElementById("site-nav").classList.toggle("open", expanded);
    updateOffset();
  };
  navButton.addEventListener("click", () =>
    toggleNav(navButton.getAttribute("aria-expanded") !== "true"),
  );
  document
    .querySelector(".skip-link")
    .addEventListener("click", () => toggleNav(false));
  new ResizeObserver(updateOffset).observe(
    document.querySelector(".site-header"),
  );
  updateOffset();
  installTooltip();
}
export function installTooltip() {
  const tip = document.createElement("div");
  tip.id = "tip";
  tip.role = "tooltip";
  document.body.append(tip);
  let target = null;
  let hideTimer = null;
  const hide = () => {
    clearTimeout(hideTimer);
    target?.removeAttribute("aria-describedby");
    target = null;
    tip.classList.remove("on");
  };
  const scheduleHide = () => {
    clearTimeout(hideTimer);
    hideTimer = setTimeout(hide, 180);
  };
  const show = (el, x, y) => {
    clearTimeout(hideTimer);
    if (target !== el) target?.removeAttribute("aria-describedby");
    target = el;
    tip.textContent = el.dataset.tip;
    el.setAttribute("aria-describedby", "tip");
    tip.classList.add("on");
    tip.style.left = `${Math.max(8, Math.min(x + 12, innerWidth - tip.offsetWidth - 8))}px`;
    tip.style.top = `${Math.max(8, Math.min(y + 14, innerHeight - tip.offsetHeight - 8))}px`;
  };
  document.addEventListener("pointerover", (e) => {
    const el = e.target.closest?.("[data-tip]");
    if (e.target.closest?.("#tip")) {
      clearTimeout(hideTimer);
      return;
    }
    if (el) show(el, e.clientX, e.clientY);
    else scheduleHide();
  });
  document.addEventListener("focusin", (e) => {
    const el = e.target.closest?.("[data-tip]");
    if (el) {
      const r = el.getBoundingClientRect();
      show(el, r.left, r.bottom);
    } else hide();
  });
  document.addEventListener("focusout", hide);
  document.addEventListener("pointerout", (e) => {
    if (!e.relatedTarget?.closest?.("[data-tip], #tip")) scheduleHide();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") hide();
  });
  addEventListener("scroll", hide, true);
}
export function errorPage(error) {
  document.querySelector("main").innerHTML =
    `<div class="empty"><h1>Results unavailable</h1><p>${esc(error.message)}</p><a href="index.html">Return to overview</a></div>`;
}
export { provider, rankOf };
