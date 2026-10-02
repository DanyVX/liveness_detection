# ruff: noqa: S311  (seeded rng is deliberate in tests)
import itertools
import random
import threading

import pytest

from liveness.active.challenge import (
    Challenge,
    ChallengeKind,
    ChallengeStep,
    ConsumeStatus,
    NonceStore,
    generate_challenge,
)

ALL = list(ChallengeKind)


class FakeClock:
    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_default_challenge_is_unpredictable_and_well_formed() -> None:
    clock = FakeClock()
    a = generate_challenge(ALL, 4, clock=clock, ttl_s=20)
    b = generate_challenge(ALL, 4, clock=clock, ttl_s=20)
    assert a.nonce != b.nonce
    assert len(a.nonce) >= 32  # >= 128 bits of hex
    assert a.issued_at == 1000.0
    assert a.expires_at == 1020.0
    assert len(a.sequence) == 4
    seqs = {tuple(generate_challenge(ALL, 4).sequence) for _ in range(40)}
    assert len(seqs) > 10


def test_seeded_rng_is_deterministic_for_tests() -> None:
    a = generate_challenge(ALL, 5, rng=random.Random(3))
    b = generate_challenge(ALL, 5, rng=random.Random(3))
    assert a.sequence == b.sequence
    assert a.nonce == b.nonce
    assert len(a.nonce) == 64


def test_no_consecutive_repeats_when_choice_exists() -> None:
    for seed in range(30):
        seq = generate_challenge(ALL, 8, rng=random.Random(seed)).sequence
        assert all(x.kind is not y.kind for x, y in itertools.pairwise(seq))


def test_accessibility_restricted_kinds() -> None:
    allowed = {ChallengeKind.TURN_LEFT, ChallengeKind.TURN_RIGHT}
    for _ in range(20):
        ch = generate_challenge(allowed, 4)
        assert {s.kind for s in ch.sequence} <= allowed
    only = generate_challenge([ChallengeKind.OPEN_MOUTH], 3)
    assert [s.kind for s in only.sequence] == [ChallengeKind.OPEN_MOUTH] * 3


def test_blink_counts_within_bounds() -> None:
    for seed in range(20):
        ch = generate_challenge([ChallengeKind.BLINK_N], 2, rng=random.Random(seed), max_blinks=3)
        assert all(1 <= s.count <= 3 for s in ch.sequence)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"allowed_kinds": []},
        {"allowed_kinds": ALL, "length": 0},
        {"allowed_kinds": ALL, "length": 99},
        {"allowed_kinds": ALL, "ttl_s": 0},
        {"allowed_kinds": ALL, "max_blinks": 0},
    ],
)
def test_generate_rejects_bad_arguments(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        generate_challenge(**kwargs)  # type: ignore[arg-type]


def test_step_validation() -> None:
    with pytest.raises(ValueError):
        ChallengeStep(ChallengeKind.BLINK_N, 0)
    with pytest.raises(ValueError):
        ChallengeStep(ChallengeKind.TURN_LEFT, 2)


def test_nonce_one_time_use_and_replay_reason() -> None:
    clock = FakeClock()
    store = NonceStore(clock)
    ch = generate_challenge(ALL, clock=clock)
    store.register(ch)
    first = store.consume(ch.nonce)
    assert first.status is ConsumeStatus.OK
    assert first.challenge == ch
    second = store.consume(ch.nonce)
    assert second.status is ConsumeStatus.NONCE_REUSED
    assert second.challenge is None


def test_unknown_expired_and_purged() -> None:
    clock = FakeClock()
    store = NonceStore(clock, retention_s=100)
    assert store.consume("f" * 64).status is ConsumeStatus.NONCE_UNKNOWN
    ch = generate_challenge(ALL, clock=clock, ttl_s=10)
    store.register(ch)
    clock.now += 11
    assert store.consume(ch.nonce).status is ConsumeStatus.CHALLENGE_EXPIRED
    assert store.consume(ch.nonce).status is ConsumeStatus.NONCE_REUSED
    clock.now += 200  # past expiry + retention: forgotten, memory is released
    assert len(store) == 1
    assert store.consume(ch.nonce).status is ConsumeStatus.NONCE_UNKNOWN
    assert len(store) == 0


def test_expiry_boundary_is_inclusive_at_expires_at() -> None:
    clock = FakeClock()
    store = NonceStore(clock)
    ch = generate_challenge(ALL, clock=clock, ttl_s=10)
    store.register(ch)
    clock.now = ch.expires_at
    assert store.consume(ch.nonce).status is ConsumeStatus.OK


def test_register_validation_and_capacity() -> None:
    clock = FakeClock()
    store = NonceStore(clock, max_entries=2)
    with pytest.raises(ValueError, match="128 bits"):
        store.register(Challenge("abc", 0.0, 1.0, ()))
    a = generate_challenge(ALL, clock=clock)
    store.register(a)
    with pytest.raises(ValueError, match="already"):
        store.register(a)
    store.register(generate_challenge(ALL, clock=clock))
    with pytest.raises(RuntimeError, match="full"):
        store.register(generate_challenge(ALL, clock=clock))
    clock.now += 10_000
    store.purge()
    assert len(store) == 0


def test_concurrent_consume_succeeds_exactly_once() -> None:
    store = NonceStore()
    ch = generate_challenge(ALL)
    store.register(ch)
    results: list[ConsumeStatus] = []
    lock = threading.Lock()

    def worker() -> None:
        r = store.consume(ch.nonce).status
        with lock:
            results.append(r)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results.count(ConsumeStatus.OK) == 1
    assert results.count(ConsumeStatus.NONCE_REUSED) == 15
