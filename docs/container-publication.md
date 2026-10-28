# Publishing the runtime container

## Verified public runtime

The Linux/amd64 runtime is publicly available without authentication:

```text
ghcr.io/aryabhatta-0/emberbench@sha256:1142df88f0d68b5a3d34323481168c6d3253ef6cef4e7879e5de25f9b2b712d8
```

| Identity | Verified value |
| --- | --- |
| Image source commit | `9d9999b913bc304001e6cf41f6b34a970109f296` |
| Source tag | `ghcr.io/aryabhatta-0/emberbench:9d9999b913bc304001e6cf41f6b34a970109f296` |
| Registry manifest SHA-256 | `1142df88f0d68b5a3d34323481168c6d3253ef6cef4e7879e5de25f9b2b712d8` |
| Registry configuration SHA-256 | `f2ee14a8fb1416c9e08ec265ac73cafa69d9d2817e9b77ca54de7a4139cc9a15` |
| OCI source label | `https://github.com/Aryabhatta-0/EMBERBench` |
| OCI revision label | `9d9999b913bc304001e6cf41f6b34a970109f296` |
| Package / benchmark revision | `1.0.0rc1` / `3` |

The [source CI run](https://github.com/Aryabhatta-0/EMBERBench/actions/runs/40699433569)
passed all six OS/Python test jobs and packaging. The
[publishing workflow](https://github.com/Aryabhatta-0/EMBERBench/actions/runs/40735071235)
used a repository-scoped `GITHUB_TOKEN`, built only the runtime stage and
published the full source SHA tag after its audit and smoke test passed.

Independent verification read the actual GHCR manifest bytes and checked both
their SHA-256 and the registry's digest header. The configuration blob's hash,
source/revision labels, non-root user and CLI entrypoint also matched. This
digest was obtained from GHCR, not from the local candidate's OCI index.

After the package became public, Docker pulled by immutable digest with an
empty credential configuration. The pulled image's saved configuration bytes
matched the registry configuration digest. Its 41 source files and installed
package files matched the exact source checkout; the layer audit scanned all
10 layers and 92 application files and found no forbidden artifacts or
credential patterns. Scenario packs, private seeds, salts and backup keys are
not included in the runtime image.

With networking disabled, the pulled image passed `version --json`, validation
of all 10 public V1 scenarios, and a 10-episode `congestion-aware-v1` baseline
smoke on a frozen public-only pack. The smoke used 100 bootstrap resamples,
public development seed `0`, pack ID `container-public-smoke`, and pack digest
`f9afeaf5e2ab612e4df3f3826883ac6312c7d4da8084a01566ef1213889bf03b`.
No model API was called and no existing evaluation results were changed.

## Reproduce the public smoke

The following POSIX shell commands use the pinned runtime for pack creation
as well as execution, with no host Python installation. Choose fresh local
directory names if the example directories already exist:

```bash
git clone https://github.com/Aryabhatta-0/EMBERBench.git emberbench-container-smoke
cd emberbench-container-smoke
git checkout --detach 9d9999b913bc304001e6cf41f6b34a970109f296
IMAGE=ghcr.io/aryabhatta-0/emberbench@sha256:1142df88f0d68b5a3d34323481168c6d3253ef6cef4e7879e5de25f9b2b712d8
docker pull "$IMAGE"
docker run --rm --network none "$IMAGE" version --json
mkdir public-smoke-work public-smoke-output
docker run --rm -i --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/scenarios/public/v1,target=/public,readonly" \
  --mount "type=bind,source=$(pwd)/public-smoke-work,target=/work" \
  --entrypoint python "$IMAGE" - <<'PY'
from emberbench import load_suite, build_unique_world_manifest, freeze_environment_pack
suite = load_suite("/public")
manifest = build_unique_world_manifest(suite, [0], stochastic_worlds=1)
pack = freeze_environment_pack(suite, manifest, "/work/public-smoke-pack", pack_id="container-public-smoke")
print(pack.digest)
PY
docker run --rm --network none --user "$(id -u):$(id -g)" \
  --mount "type=bind,source=$(pwd)/public-smoke-work/public-smoke-pack,target=/suite,readonly" \
  --mount "type=bind,source=$(pwd)/public-smoke-output,target=/out" \
  "$IMAGE" run --suite /suite --baseline congestion-aware-v1 \
  --expected-digest f9afeaf5e2ab612e4df3f3826883ac6312c7d4da8084a01566ef1213889bf03b \
  --out /out --resamples 100 --json
```

For PowerShell, assign `$image = 'ghcr.io/aryabhatta-0/emberbench@sha256:1142df88f0d68b5a3d34323481168c6d3253ef6cef4e7879e5de25f9b2b712d8'`
and use `docker pull $image` and `docker run --rm --network none $image version --json`.
The writable parent mount allows atomic pack staging. On POSIX hosts,
`--user` maps writes to the current host user. Docker Desktop on Windows can
omit that option. Use absolute host paths for bind mounts. The Python code above can be passed
to the container's Python entrypoint using a PowerShell here-string and
`docker run -i` with the same mounts.

## Publish a future source commit

Use **Publish runtime image** (`.github/workflows/publish-image.yml`) on `main`,
with a full source commit SHA. This is a manual container publication, not a
final benchmark release or a model evaluation. It does not create a Git tag.

The workflow requires that the source commit is on `main` and has passed all
six OS/Python test jobs and the packaging job. It checks out that exact source,
runs the public-assets leak guard, builds only the Dockerfile's `runtime` stage,
and audits image layers and installed package source before registry login.
The source and revision OCI labels identify the repository and source commit.
No evaluator assets or credentials are supplied to the build.

Registry authentication uses the job's short-lived `GITHUB_TOKEN` with
`packages: write`; no personal access token is stored in the repository. The
full source SHA is the image tag. Existing tags are never overwritten.

After the workflow succeeds:

1. Read the actual manifest digest from GHCR, independently of the local build.
2. Confirm the registry manifest's configuration has the intended source and
   revision labels, and the installed package matches that source commit.
3. Set the linked container package's visibility to **public**, if permitted.
4. With an empty Docker credential configuration, pull by immutable digest and
   smoke-test using only public assets. Fail closed on any mismatch.
5. Record the full `ghcr.io/aryabhatta-0/emberbench@sha256:...` reference in the
   README, release documentation, changelog and AA brief. Open a documentation
   PR and wait for its CI before merging. The image source SHA stays distinct
   from this later documentation commit.

A successful push alone does not establish public access or a verified digest.
Never replace a registry digest with a local OCI index digest. Pulling by the
verified immutable digest fixes image bytes; rebuilding from source is not
claimed to reproduce identical container bytes. The pinned base and build tools,
source identity, public scoring contracts and runtime checks define the audit
boundary. The full benchmark release gates still apply before a final release.
