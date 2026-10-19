import pytest

from emberbench.rng import SplitMix64, derive_stream_seed


def test_splitmix64_reference_vector() -> None:
    # Reference outputs of SplitMix64 seeded with 0 (Vigna's splitmix64.c).
    rng = SplitMix64(0)
    assert [rng.next_u64() for _ in range(3)] == [
        0xE220A8397B1DCDAF,
        0x6E789E6AA1B965F4,
        0x06C45D188009454F,
    ]


def test_same_seed_same_stream_and_state_resumes() -> None:
    a, b = SplitMix64(42), SplitMix64(42)
    assert [a.next_u64() for _ in range(100)] == [b.next_u64() for _ in range(100)]
    resumed = SplitMix64(a.state)
    assert [a.random() for _ in range(10)] == [resumed.random() for _ in range(10)]


def test_random_range_and_bernoulli_consumes_one_draw() -> None:
    rng = SplitMix64(7)
    values = [rng.random() for _ in range(1000)]
    assert all(0.0 <= v < 1.0 for v in values)
    a, b = SplitMix64(9), SplitMix64(9)
    assert a.bernoulli(0.0) is False
    assert b.bernoulli(1.0) is True
    assert a.state == b.state


def test_derived_streams_are_stable_and_label_separated() -> None:
    assert derive_stream_seed(1, "fire") == derive_stream_seed(1, "fire")
    assert derive_stream_seed(1, "fire") != derive_stream_seed(1, "movement")
    assert derive_stream_seed(1, "fire") != derive_stream_seed(2, "fire")
    # Pinned value. If this changes, every stochastic trajectory changes.
    assert derive_stream_seed(0, "fire") == int.from_bytes(
        __import__("hashlib").sha256(b"emberbench/v1/0/fire").digest()[:8], "big"
    )


@pytest.mark.parametrize("bad", [-1, 2**64, True, 1.5, "1"])
def test_bad_seeds_rejected(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        SplitMix64(bad)  # type: ignore[arg-type]
