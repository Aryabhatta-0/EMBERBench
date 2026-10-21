# Release checklist

Follow every step for every release. A release is an annotated git tag and,
optionally, a published wheel and sdist. The version policy is in
[versioning.md](versioning.md). Never publish from a branch, a dirty tree or a
local build that did not pass section 3.

Commands are for a POSIX shell (Linux or macOS). On Windows use
`.venv\Scripts\activate` instead of `. .venv/bin/activate`; the other
commands are the same.

## 1. Release blockers

Each of these must be resolved before a tag:

- [ ] **License.** No license is configured yet. Choosing one is a
      maintainer decision; no tool or pull request picks it. Until then
      `python tools/check_release.py ... --require-license` fails. Once a
      license is chosen, add:
      1. a `LICENSE` file with the license text at the repository root;
      2. in `pyproject.toml`, `[project]`: `license = "<SPDX expression>"`
         and `license-files = ["LICENSE"]`;
      3. the license name in this checklist and in the changelog entry.
- [ ] **Pending V1 pull requests.** Every pull request accepted for V1 is
      merged, `main` contains it, and this checklist runs on that `main`.
- [ ] **Package version decision.** The package version stays `0.1.0`
      unless the maintainers record another first release version
      ([versioning.md](versioning.md), section 1).

## 2. Clean checkout and pinned tools

```bash
git clone https://github.com/Aryabhatta-0/EMBERBench.git emberbench-release
cd emberbench-release
git checkout main
git status --porcelain
```

`git status --porcelain` prints nothing. Record the commit:
`git rev-parse HEAD`.

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -c tools/dev-constraints.txt pip
python -m pip install -c tools/dev-constraints.txt -e ".[dev]"
python tools/check_dev_constraints.py --installed
```

The last command reports `0 problem(s)`: every installed tool and every
dependency of a tool is the pinned version.

## 3. Release gates

All CI is green on the exact commit: the six `test` jobs (Ubuntu, Windows
and macOS on Python 3.12 and 3.13) and the `packaging` job. Then, locally:

```bash
python -m mypy --strict
python -m pytest -q
EMBERBENCH_RELEASE_SMOKE=1 python -m pytest -q tests/test_release.py
python tests/release_digests.py --check
python tools/check_release.py --contracts-only
python tools/check_public_assets.py --untracked
```

Also run the full opt-in mode of the public development evaluation pack
(`tests/test_evaluation_pack.py`, PR #31). It reproduces the golden index and
`REPORT.md`:

```bash
EMBERBENCH_FULL_PACK=1 python -m pytest -q tests/test_evaluation_pack.py
```

Then build, check and install the release archives with the pinned build
backend, offline:

```bash
python -m pip download -c tools/dev-constraints.txt --no-deps --only-binary=:all: --dest wheelhouse setuptools
SOURCE_DATE_EPOCH=$(git log -1 --format=%ct) python tools/check_release.py --dist dist --wheelhouse wheelhouse --require-license --require-clean
```

`tools/check_release.py` prints each step and fails closed with the reason.
All of these must hold:

- [ ] strict mypy reports no issues and the full test suite passes,
      including the opt-in release smoke;
- [ ] `tests/release_digests.py --check` prints `registry verified`;
- [ ] the leak guard reports **0 violations** for the repository, the wheel
      and the sdist. No held-out material (hidden suite, private evaluation
      manifest, private suite freeze record, selection keys, private seeds,
      submission metadata, evaluator workspaces, caches) is in the
      repository, the wheel or the sdist. Never widen the leak-guard
      allowlist to make a release pass;
- [ ] `tools/check_release.py` passes: the version contracts match the docs,
      two clean builds give byte-identical wheels, the wheel built from the
      sdist equals the wheel built from the tree, the wheel builds offline
      from the sdist with only the pinned build backend, the metadata and
      the license are present, the wheel holds only package code, the wheel
      installs into a clean virtual environment under a path with spaces
      and runs from there, the registry recomputes with the installed wheel,
      and the leak guard passes.

Install the checked wheel once more by hand and look at its version output:

```bash
python -m venv "release smoke"
"release smoke/bin/python" -m pip install --no-index --no-deps dist/emberbench-*.whl
"release smoke/bin/emberbench" version --json
"release smoke/bin/python" -I -m emberbench version --json
```

Both print the same JSON with the package version and benchmark revision of
this release.

## 4. Version and hash review

Compare with the previous release tag (`<previous-tag>`; skip the diffs for
the first release).

- [ ] **Benchmark-defining versions.** Each of these is pinned by
      provenance; a change to any of them needs a new `BENCHMARK_REVISION`
      ([versioning.md](versioning.md), section 2): `BENCHMARK_REVISION`,
      `SIMULATOR_VERSION`, `OBSERVATION_VERSION`, `ACTION_SCHEMA_VERSION`,
      `SCHEMA_VERSION`, `METRICS_VERSION`, `CROWD_DIAGNOSTICS_VERSION`,
      `STATISTICS_VERSION`, `GENERATOR_VERSION`, `GENERATOR_CONFIG_VERSION`,
      `GENERATION_MANIFEST_VERSION`, `EVALUATION_MANIFEST_VERSION`,
      `EVALUATION_RESULT_VERSION`, `SCORING_VERSION`,
      `ENVIRONMENT_PACK_VERSION`, `ENVIRONMENT_WORLD_VERSION`,
      `FAMILY_RULES_VERSION`, `CALIBRATION_PANEL_VERSION`,
      `DIFFICULTY_RULES_VERSION`, `ADMISSION_RULES_VERSION`,
      `COMPOSITION_VERSION`, `SELECTION_RULES_VERSION`,
      `RECIPE_CATALOG_VERSION`, `SUITE_FREEZE_RECORD_VERSION`.
- [ ] **Tooling versions.** A change needs at least a MINOR release:
      `CLI_VERSION`, `BASELINES_VERSION`, `INTEGRITY_PROTOCOL_VERSION`,
      `REPLAY_AUDIT_VERSION`, `ANIMATION_FORMAT_VERSION`. Suite labels:
      `GENERATED_SUITE_VERSION`, `COMPOSED_SUITE_VERSION`.
- [ ] Every changed version has a reason in the changelog, no version moved
      without an intended change, and no intended change is missing a bump:

```bash
git diff <previous-tag> -- src/emberbench docs/versioning.md
```

- [ ] **Frozen hashes reviewed.** Every changed registry entry has a bumped
      owner version and a changelog line. For a PATCH release the file has
      not changed:

```bash
git diff <previous-tag> -- tests/fixtures/release/digest_registry.json
```

- [ ] The package version in `src/emberbench/_version.py` (the only place)
      follows [versioning.md](versioning.md), section 1. A package version
      change also changes the registry entry `cli.version-json.stdout_sha256`
      and nothing else; regenerate it with
      `python tests/release_digests.py --write` and review the diff.
- [ ] The changelog ([changelog.md](changelog.md)) is updated: rename
      "Unreleased" to the version, add the release date, and list the
      package version, benchmark revision, changed component versions and
      changed hashes. `tools/check_release.py` fails on a changelog that
      does not name the current versions.
- [ ] Public docs are updated: README status, every document of a changed
      component, [cli.md](cli.md) for CLI changes. No document states a
      stale benchmark revision (`tools/check_release.py` checks this).

## 5. Tag and publish

Only after sections 1 to 4, on the checked commit:

```bash
git tag -a v<version> -m "EMBERBench <version> (benchmark revision <revision>)"
git push origin v<version>
```

- [ ] Publish exactly the wheel and sdist that `tools/check_release.py`
      wrote to `dist/` for that commit. Do not rebuild for publishing.
- [ ] Create the release notes from the changelog entry and attach
      `dist/*`.

## 6. Rollback and revision: a benchmark bug found after a release

There is no rollback in place. Never move or delete a published tag, and
never replace a published file. Users who need the previous behaviour pin
the previous release; the fix goes forward as a new release.

1. **Triage.** Decide if the bug changes any run record, metric, statistic,
   generated scenario, environment pack, world identity, score or
   provenance for some input.
   - It does not (crash, wrong message, packaging, docs): fix it and release
     a PATCH version. No pinned hash may change.
   - It does, or it can change a ranking or comparison: it is a benchmark
     bug. Continue below.
2. **Announce** the bug with the affected releases, components and
   benchmark revision. Mark the results of the affected revision as
   affected; do not delete them.
3. **Fix** it with a bump of the owning component version and of
   `BENCHMARK_REVISION`, updated pins and registry entries, and a changelog
   entry that names the bug. Release a new MAJOR version (MINOR while the
   package is below `1.0.0`).
4. **Re-evaluate** every model to compare on the new revision. Scores of
   different revisions are never compared or merged.
5. If the bug exposed held-out material, also follow
   [integrity.md](integrity.md), section 12: retire the batch, keep its
   provenance and mark its results as compromised.

To withdraw a broken package (not a benchmark bug), yank it on the package
index and release a fixed PATCH version.

## Reproducibility boundary

What `tools/check_release.py` tests, on the CI packaging job (Ubuntu,
Python 3.12, the pinned tools) and locally:

- `SOURCE_DATE_EPOCH` is frozen: the value of the environment variable, or
  the commit time of `HEAD` when it is unset. Every wheel member carries
  that time.
- Two builds from two clean copies of the same commit, with the same
  `SOURCE_DATE_EPOCH` and the same pinned tools, give **byte-identical
  wheels**.
- Their sdists have identical member names and member contents. The sdist
  archives are **not** byte-identical: the gzip header and the tar member
  times come from the build. Only the normalized comparison is required.
- The wheel built from the sdist, the wheel built from the source tree and
  the wheel built offline from the sdist in an isolated environment with
  only the pinned setuptools have identical members and contents.

The claim is exactly: **the same source commit, the same `SOURCE_DATE_EPOCH`
and the same pinned toolchain produce byte-identical wheels.** Not claimed:
byte-identical archives across operating systems, Python versions or tool
versions, or across different `SOURCE_DATE_EPOCH` values. The benchmark
artifacts themselves (records, manifests, results, statistics, scores,
packs, generated suites, provenance) are byte-identical on every supported
platform; the registry and the golden tests check that in all six CI jobs.

## Regression digest registry

`tests/fixtures/release/digest_registry.json` holds one digest per
release-visible contract. `SPEC` in `tests/release_digests.py` is its
source: each entry names its owner version, the test pin it mirrors and
what it covers. Every value is recomputed from public development assets
and test fixtures; no entry holds a path, a timestamp, a host value, a
private seed or held-out material.

- `python tests/release_digests.py --check` recomputes it and prints a
  registry/computed diff on drift.
- `tests/test_release.py` compares it with the test pins it mirrors and
  recomputes it in fresh interpreters under other hash seeds, locales, time
  zones, I/O encodings, a working directory with spaces and a CRLF copy of
  the public suite. Under the CRLF copy only the environment pack digest
  and the score report that names it change, because a pack pins exact file
  bytes by design ([versioning.md](versioning.md), section 4).
- `tools/check_release.py` recomputes it with the installed wheel.
- `python tests/release_digests.py --write` regenerates it. Use it only for
  a deliberate change with a bumped owner version and a changelog entry
  ([versioning.md](versioning.md), section 4).

## Development tool constraints

`pyproject.toml` declares bounded ranges for the development and build
tools. `tools/dev-constraints.txt` pins the exact version of every tool and
transitive dependency, and CI installs only those versions
(`PIP_CONSTRAINT`, `PIP_BUILD_CONSTRAINT`). A new upstream release therefore
cannot change CI until someone refreshes the pins in a pull request.
Runtime dependencies stay empty.

To refresh the pins:

1. If a tool needs a new major version, widen its range in
   `pyproject.toml` first and review its release notes.
2. Run `python tools/refresh_dev_constraints.py --write` (network access
   needed). It resolves the ranges for every CI platform and Python version
   and fails if they disagree.
3. Run the release gates in section 3. Fix new strict mypy findings in the
   same pull request.
4. Open a pull request that changes only the pins (and the ranges, if
   needed). CI fails on any unpinned or drifted installed tool or
   dependency of a tool (`tools/check_dev_constraints.py --installed`
   follows the declared tools' `Requires-Dist` under each platform's
   markers; packages a hosted runner preinstalls for itself are not
   checked); add a reported platform-only pin by hand.
