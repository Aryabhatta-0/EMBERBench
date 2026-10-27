"""Compose several frozen generated suites into one suite with neutral, position-only scenario IDs.

This is mechanism only: it holds no generator configuration, mixture or seed.
The public validation set (``build_public_validation.py``) and an evaluator's
external construction recipe both use it. Each part is a frozen generated suite
(``emberbench.freeze_generated_suite``), so every source scenario stays
auditable against its own freeze record; the composite renames each scenario to
``<prefix>-<position>`` in the given order and keeps its family labels.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from pathlib import Path

from emberbench import SCHEMA_VERSION, Suite, SuiteEntry, SuiteManifest, load_suite
from emberbench.generator import scenario_file_bytes
from emberbench.suite import MANIFEST_NAME, SeedRole, Visibility


def compose_suite(
    parts: Sequence[Suite],
    directory: str | Path,
    *,
    suite_id: str,
    id_prefix: str,
    order: Sequence[int] | None = None,
    suite_version: str = "1.0.0",
) -> tuple[Suite, list[tuple[int, str]]]:
    """Write the composite suite to an empty ``directory``; return it and ``[(part, source_id)]`` per new position.

    ``order`` is a permutation of the concatenated part scenarios (default: in
    part order). Use a secret permutation when the position must not reveal the part.
    """
    items = [(i, s, e) for i, part in enumerate(parts) for s, e in zip(part.scenarios, part.manifest.entries, strict=True)]
    positions = list(range(len(items))) if order is None else list(order)
    if sorted(positions) != list(range(len(items))):
        raise ValueError("order must be a permutation of the part scenarios")
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"{root}: exists and is not empty")
    root.mkdir(parents=True, exist_ok=True)
    width = max(4, len(str(len(items) - 1)))
    entries: list[SuiteEntry] = []
    sources: list[tuple[int, str]] = []
    for n, k in enumerate(positions):
        part, scenario, entry = items[k]
        renamed = dataclasses.replace(scenario, scenario_id=f"{id_prefix}-{n:0{width}d}")
        (root / f"{renamed.scenario_id}.json").write_bytes(scenario_file_bytes(renamed))
        entries.append(SuiteEntry.describe(renamed, families=entry.families, primary=entry.primary_family, purpose=entry.purpose))
        sources.append((part, scenario.scenario_id))
    manifest = SuiteManifest(suite_id, suite_version, Visibility.GENERATED, SeedRole.DEVELOPMENT, SCHEMA_VERSION, tuple(entries))
    (root / MANIFEST_NAME).write_bytes(manifest.to_file_text().encode("ascii"))
    return load_suite(root), sources
