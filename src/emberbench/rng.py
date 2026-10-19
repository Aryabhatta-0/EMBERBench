"""Simulation-owned, seeded pseudo-random number generation.

EMBERBench does not use the :mod:`random` module. Its algorithms have changed
between Python versions, and global state could leak between runs. Instead,
each simulator owns SplitMix64 streams. The algorithm is fully specified here,
its state is one 64-bit integer, and it gives the same output on every
platform and Python version.

Each subsystem (for example, fire) gets its own stream, derived from the run
seed and a stable label. Adding randomness to one subsystem later does not
shift the draws that another subsystem sees.
"""

import hashlib

MASK64 = (1 << 64) - 1
MAX_SEED = MASK64
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_TWO_POW_53_INV = 1.0 / (1 << 53)


def validate_seed(seed: object) -> int:
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if not 0 <= seed <= MAX_SEED:
        raise ValueError(f"seed must be in [0, 2**64 - 1], got {seed}")
    return seed


def derive_stream_seed(seed: int, label: str) -> int:
    """Derive an independent 64-bit stream seed from a run seed and a label."""
    validate_seed(seed)
    digest = hashlib.sha256(f"emberbench/v1/{seed}/{label}".encode("ascii")).digest()
    return int.from_bytes(digest[:8], "big")


class SplitMix64:
    """SplitMix64 generator (Steele, Lea & Flood, 2014)."""

    __slots__ = ("_state",)

    def __init__(self, state: int) -> None:
        self._state = validate_seed(state)

    @property
    def state(self) -> int:
        return self._state

    def next_u64(self) -> int:
        self._state = (self._state + _GOLDEN_GAMMA) & MASK64
        z = self._state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def random(self) -> float:
        """A float in ``[0, 1)`` built from the top 53 bits (exact, no rounding)."""
        return (self.next_u64() >> 11) * _TWO_POW_53_INV

    def bernoulli(self, p: float) -> bool:
        """Return ``True`` with probability ``p``.

        This always consumes exactly one draw, also for ``p`` of 0 or 1, so the
        stream position never depends on the probability value.
        """
        return self.random() < p
