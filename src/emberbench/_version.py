"""The one authoritative package version (see docs/versioning.md).

``pyproject.toml`` reads it for the distribution metadata, and
``emberbench.__version__`` and ``emberbench version`` report it. It is the
version of the Python distribution only. It is not ``BENCHMARK_REVISION`` and
not a component contract version.
"""

__version__ = "0.1.0"
