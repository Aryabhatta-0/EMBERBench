# EMBERBench runner image. The package has no runtime dependencies; the base image
# is pinned by digest and the build tools by tools/dev-constraints.txt.
#
#   docker build -t emberbench .
#   docker run --rm -v /path/to/environment-pack:/suite:ro -v "$PWD/out:/out" \
#       emberbench run --suite /suite --baseline congestion-aware-v1 --out /out
#
# The environment pack (public or private) is mounted read-only; pass the API key
# for `--model` with `-e EMBERBENCH_API_KEY`. `docker build --target test -t emberbench-test .`
# then `docker run --rm emberbench-test` runs the test suite in a clean environment. See docs/cli.md, "Docker".

FROM python:3.12.16-slim-bookworm@sha256:6dd8b6dab6b3adc7812bab69eca5dbdc1a5cc690ad28306d234ffb4ef8ac9753 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_CONSTRAINT=/opt/emberbench/tools/dev-constraints.txt

WORKDIR /opt/emberbench
COPY pyproject.toml README.md LICENSE ./
COPY tools/dev-constraints.txt tools/dev-constraints.txt
COPY src src
RUN python -m pip install pip && python -m pip install . \
    && useradd --create-home --uid 10001 ember && mkdir /out && chown ember /out

USER ember
WORKDIR /out
ENTRYPOINT ["emberbench"]
CMD ["--help"]

FROM runtime AS test
USER root
WORKDIR /opt/emberbench
COPY scenarios scenarios
COPY tests tests
COPY tools tools
COPY docs docs
COPY .github .github
RUN python -m pip install ".[dev]" && chown -R ember /opt/emberbench
USER ember
ENTRYPOINT []
CMD ["python", "-m", "pytest", "-q", "-p", "no:cacheprovider"]
