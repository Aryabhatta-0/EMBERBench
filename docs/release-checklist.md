# Release checklist

Follow every step for every release. A release is a git tag and, optionally,
a published wheel and sdist. The version policy is in
[versioning.md](versioning.md). Never publish from a branch, a dirty tree or a
local build.

## 1. Before the release

- [ ] Work from a clean `main`: `git status` is clean, `main` equals
      `origin/main`, and no open pull request that the release depends on is
      unmerged.
- [ ] All CI is green on the exact `main` commit to release: the six test
      jobs (Ubuntu, Windows and macOS on Python 3.12 and 3.13) and the
      `packaging` job.
- [ ] **Component and version review.** For every component version
      (table in [versioning.md](versioning.md), section 3), compare it with
      the previous release tag. Each change has a reason in the changelog.
      No component version moved without an intended change, and no intended
      change is missing a bump.
- [ ] **Benchmark revision review.** Decide if any change since the previous
      release is in the "mandatory" list of [versioning.md](versioning.md),
      section 2. If it is, `BENCHMARK_REVISION` is bumped and the package
      version is a new MAJOR version.
- [ ] **Frozen hashes reviewed.** Compare
      `tests/fixtures/release/digest_registry.json` with the previous release
      tag:
      `git diff <previous-tag> -- tests/fixtures/release/digest_registry.json`.
      Every changed entry has a bumped owner version and a changelog line. For
      a PATCH release the file has not changed.
- [ ] Set the package version in `src/emberbench/_version.py` (the only
      place) by the rules in [versioning.md](versioning.md), section 1.
- [ ] The changelog ([changelog.md](changelog.md)) is updated: rename
      "Unreleased" to the version, add the release date, and list the
      package version, benchmark revision, changed component versions and
      changed hashes.
- [ ] Public docs are updated: README status line, every document of a
      changed component, [cli.md](cli.md) for CLI changes.
- [ ] License metadata is set (`license` in `pyproject.toml` and a
      `LICENSE` file) by the maintainers. `tools/check_release.py` warns
      while it is missing; run it with `--require-license` for a release.

## 2. Release gates

Run each command from the repository root in a fresh virtual environment
with Python 3.12 or 3.13.

```bash
python -m pip install -c tools/dev-constraints.txt pip
```

```bash
python -m pip install -c tools/dev-constraints.txt -e ".[dev]"
```

```bash
python tools/check_dev_constraints.py --installed
```

```bash
python -m mypy --strict
```

```bash
python -m pytest -q
```

```bash
python tools/check_public_assets.py --untracked
```

```bash
python -m pip download -c tools/dev-constraints.txt --no-deps --only-binary=:all: --dest wheelhouse setuptools
```

```bash
python tools/check_release.py --dist dist --wheelhouse wheelhouse --require-license
```

All must pass:

- [ ] strict mypy reports no issues and the full test suite passes;
- [ ] the leak guard reports **0 violations** for the repository, the wheel
      and the sdist, and no held-out material (private seeds, scored
      manifests, evaluator workspaces, frozen held-out batches) is in the
      repository, the wheel or the sdist. Never widen the leak-guard
      allowlist to make a release pass;
- [ ] `tools/check_release.py` passes: two clean builds match, the wheel
      built from the sdist equals the wheel built from the tree, the wheel
      builds from the sdist with only the pinned build backend, the
      metadata is correct, and the wheel installs into a clean virtual
      environment and runs from there;
- [ ] the regression digest registry matches in fresh interpreters
      (`tests/test_release.py`).

## 3. Tag and publish

- [ ] Only after every check above: create an annotated tag `v<version>` on
      the checked `main` commit and push it.
- [ ] Publish exactly the wheel and sdist that `tools/check_release.py`
      wrote to `dist/` for that commit. Do not rebuild for publishing.
- [ ] Create the release notes from the changelog entry.

## 4. Rollback and revision: a benchmark bug found after a release

There is no rollback in place. Never move or delete a published tag, and
never replace a published file. Users who need the previous behaviour pin
the previous release; the fix goes forward as a new release.

1. **Triage.** Decide if the bug changes any run record, metric, statistic,
   generated scenario, provenance or score for some input.
   - It does not (crash, wrong message, packaging, docs): fix it and release
     a PATCH version. No pinned hash may change.
   - It does, or it can change a ranking or comparison: it is a benchmark
     bug. Continue below.
2. **Announce** the bug with the affected releases, components and
   benchmark revision. Mark the results of the affected revision as
   affected; do not delete them.
3. **Fix** it with a bump of the owning component version and of
   `BENCHMARK_REVISION`, updated pins and registry entries, and a changelog
   entry that names the bug. Release a new MAJOR version.
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

- Two builds from two clean copies of the same commit, with the same
  `SOURCE_DATE_EPOCH` and the same pinned tools, give **byte-identical
  wheels**.
- Their sdists have identical member names and member contents. The sdist
  archives are **not** byte-identical: the gzip header and the tar member
  times come from the build. Only the normalized comparison is required.
- The wheel built from the sdist, the wheel built from the source tree and
  the wheel built from the sdist in an isolated environment with only the
  pinned setuptools have identical members and contents.

What is **not** claimed: byte-identical archives across operating systems,
Python versions or tool versions, or across different `SOURCE_DATE_EPOCH`
values. The benchmark artifacts themselves (records, manifests, results,
statistics, generated suites, provenance) are byte-identical on every
supported platform; the registry and the golden tests check that in all six
CI jobs.

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
3. Run the release gates in section 2. Fix new strict mypy findings in the
   same pull request.
4. Open a pull request that changes only the pins (and the ranges, if
   needed). CI fails on any unpinned installed package
   (`tools/check_dev_constraints.py --installed`); add a reported
   platform-only pin by hand.
