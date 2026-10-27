# Public development generator

`emberbench.generator` remains open for public training and development worlds.
Its family tags and calibration thresholds describe development behaviour, not
membership in an official hidden distribution.

Use [procedural generation](procedural-generation.md), the [CLI](cli.md),
[scenario schema](scenario-schema.md) and [simulation contract](simulation-contract.md).
Public users may generate unlimited training worlds within their own compute budget.

The previously published composition implementation, exact suite composition,
candidate-pool construction and targeted recipes are retired from this tree.
They are exposed historical research and must not be reused as a secret test.
Legacy composition entry points fail closed; there is no public fallback for
official construction. Their version labels and opaque digest are retained only
to interpret old public research records.

Official families, parameter distributions, mixture weights, membership thresholds,
seeds, adversarial templates, out-of-distribution variants and future rotation
schedules belong to the evaluator-private implementation. The official path must
not import public `families.py` or calibration thresholds as hidden membership.

See [Evaluation boundary](evaluation-boundary.md). Scoring remains fully public in
[scoring.md](scoring.md), including weights, gates, aggregation and failure handling.
