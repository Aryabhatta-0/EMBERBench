import test from "node:test";
import assert from "node:assert/strict";
import { readFile, access } from "node:fs/promises";
import {
  finite,
  METRICS,
  provider,
  sortModels,
  filterModels,
  rankOf,
  leaderboardOrder,
  selectedIds,
  worldKey,
  pairedDifferences,
  paretoFrontier,
} from "../docs/site/analytics.js";
import {
  fmt,
  count,
  money,
  duration,
  ciText,
  esc,
  modelLink,
  selectionBar,
  provenance,
} from "../docs/site/common.js";
import {
  scoreBars,
  scatter,
  heatmap,
  componentChart,
  differenceChart,
} from "../docs/site/charts.js";
import data from "../docs/site/data/results.js";

const model = (id, extra = {}) => ({
  id,
  name: `provider/${id}`,
  kind: "Decision model",
  score: 50,
  safety: 0.5,
  quality: 0.8,
  totals: {
    trials: 1,
    evacuated: 2,
    avoidable_casualties: 0,
    avoidable_stranded: 0,
  },
  robustness: { worst_scenario: 25 },
  episodes: [],
  ...extra,
});
const ids = (models) => models.map((m) => m.id);
const episode = (score, index = 0, scenario_id = "scenario", seed = 0) => ({
  score,
  index,
  scenario_id,
  seed,
});

test("non-numeric and non-finite values remain unavailable", () => {
  for (const value of [null, undefined, NaN, Infinity, -Infinity, "0", false]) {
    assert.equal(finite(value), false);
    for (const format of [fmt, count, money, duration])
      assert.equal(format(value), "—");
  }
  assert.equal(finite(0), true);
  assert.equal(money(0), "$0.0000");
  assert.equal(money(0.1, true), "≈ $0.10");
  assert.equal(ciText({ ci: [null, null] }), "—");
  assert.equal(ciText({ ci: [1, 2] }), "1.00–2.00");
});

test("sorting puts unavailable API metrics last in both directions and preserves zero", () => {
  for (const [key, field] of [
    ["cost", "cost_usd"],
    ["latency", "mean_act_latency_s"],
    ["requests", "requests"],
    ["input", "input_tokens"],
  ]) {
    const models = [
      model("missing"),
      model("high", { api: { [field]: 10 } }),
      model("unknown", { api: { [field]: null } }),
      model("zero", { api: { [field]: 0 } }),
      model("bad", { api: { [field]: NaN } }),
    ];
    const original = [...models];
    assert.deepEqual(ids(sortModels(models, key, "asc")), [
      "zero",
      "high",
      "bad",
      "missing",
      "unknown",
    ]);
    assert.deepEqual(ids(sortModels(models, key, "desc")), [
      "high",
      "zero",
      "bad",
      "missing",
      "unknown",
    ]);
    assert.deepEqual(ids(sortModels(models, key)).slice(0, 2), [
      "zero",
      "high",
    ]);
    assert.deepEqual(models, original);
    assert.equal(rankOf(models, models[0], key), null);
  }
  assert.deepEqual(
    ids(sortModels([model("low", { score: 1 }), model("high", { score: 2 })])),
    ["high", "low"],
  );
});

test("search, entry type and provider filters combine without mutating input", () => {
  const models = [
    model("one", { name: "Maker/Alpha Large" }),
    model("two", { name: "Maker/Beta" }),
    model("three", { name: "Other/Alpha" }),
    model("rules", { name: "Shortest path", kind: "Reference agent" }),
  ];
  assert.equal(provider(models[0]), "Maker");
  assert.equal(provider(models[3]), "Reference");
  assert.equal(
    provider(model("unknown", { name: "unqualified-name" })),
    "Unknown provider",
  );
  assert.equal(
    provider(model("empty-prefix", { name: "/model" })),
    "Unknown provider",
  );
  assert.deepEqual(
    ids(
      filterModels(models, {
        query: " ALPHA   maker ",
        kind: "Decision model",
        creator: "Maker",
      }),
    ),
    ["one"],
  );
  assert.deepEqual(ids(filterModels(models, { query: "three" })), ["three"]);
  assert.deepEqual(ids(filterModels(models, { kind: "Reference agent" })), [
    "rules",
  ]);
  assert.deepEqual(ids(filterModels(models, { creator: "Reference" })), [
    "rules",
  ]);
  assert.deepEqual(filterModels(models, { creator: "Missing" }), []);
  assert.equal(models.length, 4);
});

test("rank follows the official rounded score and Safety tie policy", () => {
  const unsafe = model("raw-higher", { score: 80.04, safety: 0.6 });
  const safe = model("safe", { score: 80.01, safety: 0.9 });
  const leader = model("leader", { score: 80.16, safety: 0.1 });
  assert.deepEqual(ids(leaderboardOrder([unsafe, safe, leader])), [
    "leader",
    "safe",
    "raw-higher",
  ]);
  assert.equal(rankOf([unsafe, safe, leader], safe), 2);
  // Decimal half-even: 80.05 rounds to 80.0, and 80.15 rounds to 80.2.
  assert.deepEqual(
    ids(
      leaderboardOrder([
        model("first", { score: 80.05, safety: 0.1 }),
        model("second", { score: 80, safety: 0.9 }),
      ]),
    ),
    ["second", "first"],
  );
  assert.deepEqual(
    ids(
      leaderboardOrder([
        model("first", { score: 80.15, safety: 0.1 }),
        model("second", { score: 80.2, safety: 0.9 }),
      ]),
    ),
    ["second", "first"],
  );
});

test("rank tiebreaks use losses per trial, worst scenario, then ID", () => {
  const models = [
    model("casualties", {
      totals: { trials: 1, avoidable_casualties: 1, avoidable_stranded: 0 },
    }),
    model("stranded", {
      totals: { trials: 1, avoidable_casualties: 0, avoidable_stranded: 2 },
    }),
    model("two-trials", {
      totals: { trials: 2, avoidable_casualties: 0, avoidable_stranded: 2 },
    }),
    model("tail-low", { robustness: { worst_scenario: 1 } }),
    model("z-tail-high", { robustness: { worst_scenario: 40 } }),
    model("a-tail-high", { robustness: { worst_scenario: 40 } }),
  ];
  assert.deepEqual(ids(leaderboardOrder(models)), [
    "a-tail-high",
    "z-tail-high",
    "tail-low",
    "two-trials",
    "stranded",
    "casualties",
  ]);
  const equal = [
    model("a", { api: { cost_usd: 1 } }),
    model("b", { api: { cost_usd: 1 } }),
    model("c", { api: { cost_usd: 2 } }),
  ];
  assert.equal(rankOf(equal, equal[0], "cost"), 1);
  assert.equal(rankOf(equal, equal[1], "cost"), 1);
  assert.equal(rankOf(equal, equal[2], "cost"), 3);
});

test("selection accepts only existing unique string IDs, with a four-model limit", () => {
  const models = ["a", "b", "c", "d", "e"].map((id) => model(id));
  for (const raw of [undefined, null, "not JSON", "{}", 42])
    assert.deepEqual(selectedIds(raw, models), []);
  assert.deepEqual(
    selectedIds('["a","a","missing",1,null,"b","c","d","e"]', models),
    ["a", "b", "c", "d"],
  );
  assert.deepEqual(selectedIds(["b", "a"], models), ["b", "a"]);
  assert.deepEqual(selectedIds(["missing"], models), []);
});

test("comparison links require 2–4 valid selected models and survive blocked storage", () => {
  const models = ["a", "b", "c", "d", "e"].map((id) => model(id));
  const descriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    "localStorage",
  );
  try {
    for (const chosen of [
      [],
      ["a"],
      ["a", "b"],
      ["a", "b", "c", "d"],
      ["a", "b", "c", "d", "e"],
    ]) {
      Object.defineProperty(globalThis, "localStorage", {
        configurable: true,
        value: { getItem: () => JSON.stringify(chosen) },
      });
      const html = selectionBar({ models });
      assert.equal(html.includes('aria-disabled="true"'), chosen.length < 2);
      assert.ok(html.includes(`${Math.min(chosen.length, 4)} of 4`));
    }
    Object.defineProperty(globalThis, "localStorage", {
      configurable: true,
      get() {
        throw new Error("Storage blocked");
      },
    });
    assert.ok(selectionBar({ models }).includes("0 of 4"));
  } finally {
    if (descriptor)
      Object.defineProperty(globalThis, "localStorage", descriptor);
    else delete globalThis.localStorage;
  }
});

test("paired differences match full world identity and ignore record order", () => {
  const a = model("a", {
    episodes: [episode(10, 0, "one", 1), episode(40, 1, "two", 2)],
  });
  const b = model("b", {
    episodes: [episode(30, 1, "two", 2), episode(20, 0, "one", 1)],
  });
  const result = pairedDifferences(a, b);
  assert.equal(result.complete, true);
  assert.equal(result.mean, 0);
  assert.deepEqual(
    result.worlds.map((w) => w.difference),
    [-10, 10],
  );
  assert.notEqual(
    worldKey(episode(1, 0, "one", 1)),
    worldKey(episode(1, 0, "one", 2)),
  );
  for (const changed of [
    episode(20, 0, "other", 1),
    episode(20, 0, "one", 9),
    episode(20, 9, "one", 1),
  ]) {
    const incomplete = pairedDifferences(
      a,
      model("b", { episodes: [changed, episode(30, 1, "two", 2)] }),
    );
    assert.equal(incomplete.worlds.length, 1);
    assert.equal(incomplete.complete, false);
  }
});

test("paired differences preserve zero, omit unknown scores and report incomplete coverage", () => {
  const a = model("a", {
    episodes: [episode(0), episode(null, 1), episode(10, 2)],
  });
  const b = model("b", {
    episodes: [episode(0), episode(20, 1), episode(Infinity, 2)],
  });
  assert.deepEqual(pairedDifferences(a, b), {
    worlds: [
      { index: 0, scenario_id: "scenario", seed: 0, a: 0, b: 0, difference: 0 },
    ],
    mean: 0,
    complete: false,
  });
  assert.deepEqual(pairedDifferences(model("a"), model("b")), {
    worlds: [],
    mean: null,
    complete: false,
  });
  assert.equal(
    pairedDifferences(
      model("a", { episodes: [episode(null)] }),
      model("b", { episodes: [episode(1)] }),
    ).mean,
    null,
  );
});

test("paired differences average repeated trials before one comparison per world", () => {
  const a = model("a", {
    totals: { trials: 2 },
    episodes: [episode(10), episode(100, 1), episode(30), episode(80, 1)],
  });
  const b = model("b", {
    totals: { trials: 3 },
    episodes: [
      episode(10),
      episode(20, 1),
      episode(20),
      episode(40, 1),
      episode(30),
      episode(60, 1),
    ],
  });
  const result = pairedDifferences(a, b);
  assert.equal(result.complete, true);
  assert.equal(result.worlds.length, 2);
  assert.deepEqual(
    result.worlds.map((w) => [w.a, w.b, w.difference]),
    [
      [20, 20, 0],
      [90, 40, 50],
    ],
  );
  assert.equal(result.mean, 25);
  const partial = { ...a, episodes: a.episodes.slice(0, -1) };
  assert.equal(pairedDifferences(partial, b).complete, false);
  assert.equal(pairedDifferences(partial, b).worlds.length, 1);
  const unknown = {
    ...a,
    episodes: a.episodes.map((e, i) => (i === 2 ? { ...e, score: null } : e)),
  };
  assert.equal(pairedDifferences(unknown, b).worlds.length, 1);
  assert.equal(pairedDifferences(unknown, b).complete, false);
});

test("cost frontier uses available decision costs, preserves estimates and zero, excludes dominated models", () => {
  const models = [
    model("unknown", { score: 100 }),
    model("reference", {
      score: 100,
      kind: "Reference agent",
      api: { cost_usd: 0 },
    }),
    model("free", { score: 10, api: { cost_usd: 0, cost_source: "reported" } }),
    model("estimate", {
      score: 50,
      api: { cost_usd: 1, cost_source: "list price" },
    }),
    model("tie", { score: 50, api: { cost_usd: 1, cost_source: "reported" } }),
    model("dominated", { score: 40, api: { cost_usd: 2 } }),
    model("best", { score: 80, api: { cost_usd: 3 } }),
    model("bad", { score: Infinity, api: { cost_usd: 0 } }),
  ];
  assert.deepEqual(ids(paretoFrontier(models)), [
    "free",
    "estimate",
    "tie",
    "best",
  ]);
  assert.equal(paretoFrontier(models)[1].api.cost_source, "list price");
  assert.deepEqual(
    paretoFrontier([model("missing", { api: { cost_usd: null } })]),
    [],
  );
});

test("generated snapshot satisfies real page data assumptions without invented metadata", () => {
  assert.equal(
    data.label,
    "PUBLIC DEVELOPMENT EVALUATION — NOT A HIDDEN-SUITE LEADERBOARD",
  );
  assert.match(data.pack_digest, /^[a-f0-9]{64}$/);
  assert.ok(data.pack_id && data.suite_id && data.scoring_version);
  assert.equal(new Set(ids(data.models)).size, data.models.length);
  const expectedWorlds = new Set(data.models[0].episodes.map(worldKey));
  const scenarioIds = new Set(data.models[0].scenarios.map((s) => s.name));
  for (const m of data.models) {
    assert.ok(["Decision model", "Reference agent"].includes(m.kind));
    assert.ok(finite(m.score) && m.score >= 0 && m.score <= 100);
    assert.deepEqual(new Set(m.episodes.map(worldKey)), expectedWorlds);
    assert.deepEqual(new Set(m.scenarios.map((s) => s.name)), scenarioIds);
    for (const key of [
      "evacuation",
      "casualty",
      "stranding",
      "time",
      "hazard",
      "path",
      "replanning",
    ])
      assert.ok(m.components[key]);
    assert.equal(
      m.episodes.reduce((sum, e) => sum + e.evacuated, 0),
      m.totals.evacuated,
    );
    for (const e of m.episodes)
      assert.equal(e.evacuated + e.casualties + e.stranded, e.population);
    if (m.api)
      for (const key of [
        "cost_usd",
        "mean_act_latency_s",
        "input_tokens",
        "output_tokens",
        "requests",
      ])
        assert.ok(m.api[key] === null || finite(m.api[key]));
  }
  const emptyMetadata = {
    ...data,
    constants_digest: undefined,
    evaluation_manifest_hash: undefined,
    suite_hash: undefined,
  };
  assert.ok(provenance(emptyMetadata, data.models[0]).includes("—"));
  assert.ok(provenance(emptyMetadata).includes("No evaluation timestamp"));
});

test("real data comparison is symmetric and all models pair with themselves", () => {
  for (const m of data.models) {
    const pair = pairedDifferences(m, m);
    assert.equal(pair.complete, true);
    assert.equal(pair.mean, 0);
    assert.equal(pair.worlds.length, new Set(m.episodes.map(worldKey)).size);
  }
  const [a, b] = data.models,
    ab = pairedDifferences(a, b),
    ba = pairedDifferences(b, a);
  assert.ok(Math.abs(ab.mean + ba.mean) < 1e-10);
  for (let i = 0; i < ab.worlds.length; i++)
    assert.ok(
      Math.abs(ab.worlds[i].difference + ba.worlds[i].difference) < 1e-10,
    );
});

test("charts preserve accessible links, missing values and descriptive comparison labels", () => {
  const subset = data.models.slice(0, 3);
  for (const html of [
    scoreBars(subset),
    scatter(data.models, "cost"),
    heatmap(subset),
    componentChart(subset),
  ]) {
    assert.ok(html.length > 100);
    assert.doesNotMatch(html, /NaN|Infinity|undefined/);
  }
  assert.match(scatter(data.models, "cost"), /aria-label=/);
  assert.match(scatter(data.models, "cost"), /Cost source/);
  assert.match(
    differenceChart(
      pairedDifferences(subset[0], subset[1]),
      subset[0],
      subset[1],
    ),
    /not the family-balanced EMBER difference or an official paired inference/,
  );
  const unsafe = model('a&"<', { name: "<script>alert(1)</script>" });
  assert.doesNotMatch(modelLink(unsafe), /<script>/);
  assert.match(modelLink(unsafe), /a%26%22%3C/);
  assert.equal(esc("<>&\"'"), "&lt;&gt;&amp;&quot;&#39;");
});

test("all portal entry points use local modules, viewport metadata and an accessible main landmark", async () => {
  for (const page of [
    "index",
    "leaderboard",
    "compare",
    "model",
    "scenarios",
    "methodology",
  ]) {
    const url = new URL(`../docs/site/${page}.html`, import.meta.url),
      html = await readFile(url, "utf8");
    assert.match(html, /<html lang="en">/);
    assert.match(html, /name="viewport"/);
    assert.match(html, /<main id="main" tabindex="-1">/);
    assert.match(html, /type="module" src="[^" ]+\.js"/);
    assert.match(html, /<noscript\s*>/);
    for (const [, src] of html.matchAll(/(?:src|href)="([^"#]+)"/g))
      if (!/^https?:/.test(src)) await access(new URL(src, url));
  }
});

test("authoritative exported ranks survive display rounding and filtered subsets", () => {
  const lower = model("lower", { score: 88.15, safety: 0.9, rank: 2 });
  const higher = model("higher", { score: 88.15, safety: 0.1, rank: 1 });
  assert.deepEqual(ids(leaderboardOrder([lower, higher])), ["higher", "lower"]);
  assert.equal(rankOf([lower, higher], higher), 1);
});
