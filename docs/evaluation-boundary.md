# Public task and private official evaluation

EMBERBench-the-task is open. Its simulator, model interface, public scenarios,
development generator, baselines, CLI, metric definitions, scoring formula,
weights, gates, aggregation, statistics, replay verification and fairness rules
remain public.

The current website contains **public-suite development results**. Models may
train or tune on these environments. Reproducible does not mean contamination-resistant.
No official hidden-suite leaderboard or submission service is announced here.

## Official evaluator boundary

Official evaluation must require a separately maintained private policy and
frozen assets. Official family membership must not fall back to public family
thresholds or calibration rules. Private construction covers scenario families,
parameter distributions, mixture weights, candidate-pool and top-up procedures,
seeds, adversarial cases, out-of-distribution variants and future rotation schedules.
The public package does not ship an official policy, recipe catalog or composer.
Legacy composition entry points fail closed.

The private evaluator should support in-distribution, distribution-shift and
adversarial strata. Exact proportions and construction parameters stay private.
Public scoring still applies to authenticated trajectories under pinned versions.

## Historical disclosure

Earlier releases published construction recipes and research-holdout seeds.
They are exposed historical research. Removing current files does not erase Git
history, older packages, previews or third-party copies. They must not be reused
or advertised as a secret official distribution. A fresh private policy, assets,
calibration and independent evaluation are needed before an official claim.

The published scoring study's holdout is an **archived validation set / research
holdout**. Its public seed remains historical evidence, not a secret test seed.

## Publication and compatibility

Publish the scoring contract, reviewed aggregate results and approved opaque
commitments. Keep live-suite manifests and raw artifacts private where they embed
seeds. Existing fail-closed integrity checks remain; content checks additionally
reject known construction identifiers in public source and package artifacts.

Legacy version strings and an opaque retired-rule digest remain available only
for interpreting historical public artifacts. They do not certify a future
private evaluator. The simulator and scoring semantics are unchanged.

`ScoreReport.official` identifies use of the normative public scoring constants. It does not certify a private distribution or an official hidden-evaluation result.
