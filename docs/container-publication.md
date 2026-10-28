# Publishing the runtime container

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
