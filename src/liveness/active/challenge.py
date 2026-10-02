"""Random challenge generation and one-time nonce bookkeeping (replay protection)."""

from __future__ import annotations

import heapq
import random
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

DEFAULT_TTL_S = 30.0
MAX_LENGTH = 8
NONCE_BYTES = 32
MIN_NONCE_HEX_CHARS = 32  # 128 bits


class ChallengeKind(StrEnum):
    BLINK_N = "blink_n"
    TURN_LEFT = "turn_left"  # the subject's own left
    TURN_RIGHT = "turn_right"
    OPEN_MOUTH = "open_mouth"


@dataclass(frozen=True)
class ChallengeStep:
    kind: ChallengeKind
    count: int = 1  # only meaningful for BLINK_N

    def __post_init__(self) -> None:
        if self.count < 1 or (self.kind is not ChallengeKind.BLINK_N and self.count != 1):
            raise ValueError(f"invalid count {self.count} for {self.kind}")


@dataclass(frozen=True)
class Challenge:
    nonce: str
    issued_at: float
    expires_at: float
    sequence: tuple[ChallengeStep, ...]


def generate_challenge(
    allowed_kinds: Iterable[ChallengeKind],
    length: int = 3,
    rng: random.Random | None = None,
    *,
    ttl_s: float = DEFAULT_TTL_S,
    max_blinks: int = 3,
    clock: Callable[[], float] = time.time,
) -> Challenge:
    """Draw a random step sequence from `allowed_kinds`.

    Default randomness is `secrets` (OS CSPRNG). Pass `rng` only in tests: a seeded
    `random.Random` makes the sequence and the nonce predictable. Consecutive identical kinds are
    avoided whenever more than one kind is allowed. A restricted `allowed_kinds` (accessibility)
    shrinks the challenge space; callers should compensate with a shorter ttl or more steps.
    """
    kinds = [k for k in ChallengeKind if k in set(allowed_kinds)]
    if not kinds:
        raise ValueError("allowed_kinds must not be empty")
    if not 1 <= length <= MAX_LENGTH:
        raise ValueError(f"length must be in 1..{MAX_LENGTH}")
    if ttl_s <= 0 or max_blinks < 1:
        raise ValueError("ttl_s and max_blinks must be positive")
    source = rng if rng is not None else secrets.SystemRandom()
    steps: list[ChallengeStep] = []
    for _ in range(length):
        options = [k for k in kinds if not steps or k is not steps[-1].kind] or kinds
        kind = source.choice(options)
        count = source.randint(1, max_blinks) if kind is ChallengeKind.BLINK_N else 1
        steps.append(ChallengeStep(kind, count))
    nonce = secrets.token_hex(NONCE_BYTES) if rng is None else f"{source.getrandbits(256):064x}"
    now = clock()
    return Challenge(nonce=nonce, issued_at=now, expires_at=now + ttl_s, sequence=tuple(steps))


class ConsumeStatus(StrEnum):
    OK = "OK"
    NONCE_UNKNOWN = "NONCE_UNKNOWN"
    NONCE_REUSED = "NONCE_REUSED"
    CHALLENGE_EXPIRED = "CHALLENGE_EXPIRED"


@dataclass(frozen=True)
class ConsumeResult:
    status: ConsumeStatus
    challenge: Challenge | None = None  # the server-side copy, only on OK


class NonceStore:
    """In-memory one-time nonce registry. Thread-safe; not shared across processes.

    Used and expired nonces are remembered for `retention_s` past expiry so a replay is reported
    as NONCE_REUSED / CHALLENGE_EXPIRED rather than NONCE_UNKNOWN; after that they are purged.
    """

    def __init__(
        self,
        clock: Callable[[], float] = time.time,
        retention_s: float = 300.0,
        max_entries: int = 100_000,
    ) -> None:
        self._clock = clock
        self._retention_s = retention_s
        self._max_entries = max_entries
        self._entries: dict[str, tuple[Challenge, bool]] = {}
        self._heap: list[tuple[float, str]] = []
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _purge_locked(self, now: float) -> None:
        while self._heap and self._heap[0][0] + self._retention_s < now:
            _, nonce = heapq.heappop(self._heap)
            self._entries.pop(nonce, None)

    def purge(self) -> None:
        with self._lock:
            self._purge_locked(self._clock())

    def register(self, challenge: Challenge) -> None:
        if len(challenge.nonce) < MIN_NONCE_HEX_CHARS:
            raise ValueError("nonce shorter than 128 bits")
        with self._lock:
            self._purge_locked(self._clock())
            if challenge.nonce in self._entries:
                raise ValueError("nonce already registered")
            if len(self._entries) >= self._max_entries:
                raise RuntimeError("nonce store full")
            self._entries[challenge.nonce] = (challenge, False)
            heapq.heappush(self._heap, (challenge.expires_at, challenge.nonce))

    def consume(self, nonce: str) -> ConsumeResult:
        with self._lock:
            now = self._clock()
            self._purge_locked(now)
            entry = self._entries.get(nonce)
            if entry is None:
                return ConsumeResult(ConsumeStatus.NONCE_UNKNOWN)
            challenge, used = entry
            if used:
                return ConsumeResult(ConsumeStatus.NONCE_REUSED)
            self._entries[nonce] = (challenge, True)
            if now > challenge.expires_at:
                return ConsumeResult(ConsumeStatus.CHALLENGE_EXPIRED)
            return ConsumeResult(ConsumeStatus.OK, challenge)
