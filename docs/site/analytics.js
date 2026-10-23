// Pure operations shared by the UI and its Node tests. Missing values stay missing.
export const finite = (v) => typeof v === "number" && Number.isFinite(v);
export function provider(m) {
  return m.kind === "Decision model"
    ? m.name.includes("/") && m.name.split("/")[0]
      ? m.name.split("/")[0]
      : "Unknown provider"
    : "Reference";
}
export const METRICS = {
  score: {
    label: "EMBER Score",
    direction: "higher",
    get: (m) => m.score,
    help: "0–100 benchmark-performance points. Higher is better; not percent evacuated.",
  },
  safety: {
    label: "Safety",
    direction: "higher",
    get: (m) => m.safety,
    help: "0–1 factor combining evacuation and casualty/stranding penalties, with a catastrophic gate.",
  },
  quality: {
    label: "Quality",
    direction: "higher",
    get: (m) => m.quality,
    help: "0–1 weighted geometric mean of defined time, hazard, path and replanning components.",
  },
  evacuated: {
    label: "Evacuated",
    direction: "higher",
    get: (m) => m.totals?.evacuated,
    help: "People reaching an exit, summed over evaluated worlds/trials.",
  },
  casualties: {
    label: "Avoidable deaths",
    direction: "lower",
    get: (m) => m.totals?.avoidable_casualties,
    help: "Casualties attributable to people individually escapable under the relaxed reference.",
  },
  stranded: {
    label: "Avoidable stranded",
    direction: "lower",
    get: (m) => m.totals?.avoidable_stranded,
    help: "Individually escapable people still active at the horizon; not a capability label.",
  },
  blocked: {
    label: "Blocked moves",
    direction: "lower",
    get: (m) => m.diagnostics?.congestion_blocks,
    help: "Movement attempts blocked by congestion. Diagnostic only; not a score component.",
  },
  congestion: {
    label: "Congestion block rate",
    direction: "lower",
    get: (m) => m.diagnostics?.congestion_block_rate,
    help: "Congestion blocks / movement attempts (0–1). Diagnostic; not part of EMBER Score.",
  },
  requests: {
    label: "API requests",
    direction: "lower",
    get: (m) => m.api?.requests,
    help: "Requests sent as recorded by the run. Missing for reference agents.",
  },
  input: {
    label: "Input tokens",
    direction: "lower",
    get: (m) => m.api?.input_tokens,
    help: "API-reported input tokens. Missing values are not zero.",
  },
  cost: {
    label: "API cost",
    direction: "lower",
    get: (m) => m.api?.cost_usd,
    help: "USD for the recorded run. ≈ denotes a list-price estimate; no compute cost is inferred for references.",
  },
  latency: {
    label: "Mean decision latency",
    direction: "lower",
    get: (m) => m.api?.mean_act_latency_s,
    help: "Mean seconds per act call, as recorded. Not run wall time or streaming time to first token.",
  },
};
export function sortModels(models, key = "score", direction = null) {
  const metric = METRICS[key] ?? METRICS.score;
  const sign =
    (direction ?? (metric.direction === "higher" ? "desc" : "asc")) === "asc"
      ? 1
      : -1;
  return [...models].sort((a, b) => {
    const av = metric.get(a),
      bv = metric.get(b);
    if (!finite(av) && !finite(bv)) return a.id.localeCompare(b.id);
    if (!finite(av)) return 1;
    if (!finite(bv)) return -1;
    return sign * (av - bv) || a.id.localeCompare(b.id);
  });
}
export function filterModels(
  models,
  { query = "", kind = "all", creator = "all" } = {},
) {
  const terms = query.trim().toLowerCase().split(/\s+/).filter(Boolean);
  return models.filter(
    (m) =>
      (kind === "all" || m.kind === kind) &&
      (creator === "all" || provider(m) === creator) &&
      terms.every((t) =>
        `${m.name} ${m.id} ${provider(m)}`.toLowerCase().includes(t),
      ),
  );
}
export function rankOf(models, m, key = "score") {
  if (key === "score")
    return leaderboardOrder(models).findIndex((x) => x.id === m.id) + 1;
  const metric = METRICS[key] ?? METRICS.score,
    v = metric.get(m);
  if (!finite(v)) return null;
  return (
    1 +
    models.filter(
      (x) =>
        finite(metric.get(x)) &&
        (metric.direction === "higher" ? metric.get(x) > v : metric.get(x) < v),
    ).length
  );
}
export function leaderboardOrder(models) {
  if (models.every((m) => Number.isInteger(m.rank)))
    return [...models].sort((a, b) => a.rank - b.rank);
  // Mirrors scoring.leaderboard_key using the precision in the generated snapshot.
  const roundEven = (v) => {
    const x = v * 10,
      n = Math.floor(x);
    return x - n === 0.5 ? (n % 2 ? n + 1 : n) : Math.round(x);
  };
  const key = (m) => [
    finite(m.score) ? -roundEven(m.score) : Infinity,
    finite(m.safety) ? -m.safety : Infinity,
    (m.totals?.avoidable_casualties ?? Infinity) / (m.totals?.trials || 1),
    (m.totals?.avoidable_stranded ?? Infinity) / (m.totals?.trials || 1),
    finite(m.robustness?.worst_scenario)
      ? -m.robustness.worst_scenario
      : Infinity,
  ];
  return [...models].sort((a, b) => {
    const ak = key(a),
      bk = key(b);
    for (let i = 0; i < ak.length; i++)
      if (ak[i] !== bk[i]) return ak[i] < bk[i] ? -1 : 1;
    return a.id.localeCompare(b.id);
  });
}
export function selectedIds(raw, models) {
  try {
    const ids = typeof raw === "string" ? JSON.parse(raw) : raw;
    return Array.isArray(ids)
      ? [
          ...new Set(
            ids.filter(
              (id) => typeof id === "string" && models.some((m) => m.id === id),
            ),
          ),
        ].slice(0, 4)
      : [];
  } catch {
    return [];
  }
}
export const worldKey = (e) =>
  `${e.scenario_id}\u0000${e.seed}\u0000${e.index}`;
export function pairedDifferences(a, b) {
  // Official reports are trial-major. Average all trials of each frozen world before
  // comparing; a partial or unknown trial must not silently become a complete pair.
  const averageWorlds = (model) => {
    const groups = new Map();
    for (const episode of model.episodes ?? []) {
      const key = worldKey(episode);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(episode);
    }
    const declared = model.trials ?? model.totals?.trials;
    const trials =
      Number.isInteger(declared) && declared > 0
        ? declared
        : Math.max(0, ...[...groups.values()].map((g) => g.length));
    return new Map(
      [...groups].map(([key, episodes]) => [
        key,
        {
          episode: episodes[0],
          score:
            episodes.length === trials && episodes.every((e) => finite(e.score))
              ? episodes.reduce((sum, e) => sum + e.score, 0) / trials
              : null,
        },
      ]),
    );
  };
  const left = averageWorlds(a),
    right = averageWorlds(b);
  const worlds = [...left].flatMap(([key, value]) => {
    const other = right.get(key),
      e = value.episode;
    return other && finite(value.score) && finite(other.score)
      ? [
          {
            index: e.index,
            scenario_id: e.scenario_id,
            seed: e.seed,
            a: value.score,
            b: other.score,
            difference: value.score - other.score,
          },
        ]
      : [];
  });
  return {
    worlds,
    mean: worlds.length
      ? worlds.reduce((s, w) => s + w.difference, 0) / worlds.length
      : null,
    complete:
      worlds.length > 0 &&
      worlds.length === left.size &&
      worlds.length === right.size,
  };
}
export function paretoFrontier(models) {
  const available = models.filter(
    (m) =>
      m.kind === "Decision model" && finite(m.api?.cost_usd) && finite(m.score),
  );
  return available
    .filter(
      (m) =>
        !available.some(
          (o) =>
            o.api.cost_usd <= m.api.cost_usd &&
            o.score >= m.score &&
            (o.api.cost_usd < m.api.cost_usd || o.score > m.score),
        ),
    )
    .sort((a, b) => a.api.cost_usd - b.api.cost_usd);
}
