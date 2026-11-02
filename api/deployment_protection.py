"""Public-demo generation controls with shared Redis/Valkey support.

Local development uses bounded process-local controls. Public-demo mode requires
a Redis-compatible shared store so limits remain effective across app instances.
"""

from __future__ import annotations

from hashlib import sha256
import os
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

try:
    import redis
except ImportError:  # pragma: no cover - dependency is declared by the project
    redis = None


class ProtectionStoreUnavailable(RuntimeError):
    """Shared public-demo controls could not reach their backing store."""


def env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _bounded_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return min(maximum, max(minimum, value))


def configured_origins() -> list[str]:
    """Parse exact CORS origins; paths, wildcards, and malformed origins fail closed."""
    raw = os.environ.get("THOUGHT2CODE_ALLOWED_ORIGINS", "http://localhost:5173")
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if not origins:
        raise RuntimeError("THOUGHT2CODE_ALLOWED_ORIGINS must contain an origin")
    for origin in origins:
        parsed = urlsplit(origin)
        try:
            _ = parsed.port
        except ValueError as error:
            raise RuntimeError(
                "THOUGHT2CODE_ALLOWED_ORIGINS contains an invalid origin"
            ) from error
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or any(character.isspace() for character in origin)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError("THOUGHT2CODE_ALLOWED_ORIGINS contains an invalid origin")
    return origins


class GenerationProtection:
    """Fixed-window per-client limiter and shared, leased generation slots."""

    _RATE_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
if count == 1 or redis.call('TTL', KEYS[1]) < 0 then
  redis.call('EXPIRE', KEYS[1], ARGV[1])
end
return {count, redis.call('TTL', KEYS[1])}
"""
    _ACQUIRE_SCRIPT = """
local t = redis.call('TIME')
local now = tonumber(t[1]) * 1000 + math.floor(tonumber(t[2]) / 1000)
redis.call('ZREMRANGEBYSCORE', KEYS[1], '-inf', now)
if redis.call('ZCARD', KEYS[1]) >= tonumber(ARGV[1]) then return 0 end
redis.call('ZADD', KEYS[1], now + tonumber(ARGV[2]), ARGV[3])
redis.call('PEXPIRE', KEYS[1], tonumber(ARGV[2]) * 2)
return 1
"""
    _RELEASE_SCRIPT = "return redis.call('ZREM', KEYS[1], ARGV[1])"

    def __init__(self, redis_url: str | None = None):
        self.public_demo = env_flag(
            "THOUGHT2CODE_PUBLIC_DEMO", env_flag("RENDER")
        )
        self.generation_enabled = env_flag("THOUGHT2CODE_GENERATION_ENABLED", True)
        self.client_limit = _bounded_int(
            "THOUGHT2CODE_CLIENT_RATE_LIMIT", 5 if self.public_demo else 100, 1, 100
        )
        self.window_seconds = _bounded_int(
            "THOUGHT2CODE_CLIENT_RATE_WINDOW_SECONDS", 3_600, 60, 86_400
        )
        self.max_concurrent = _bounded_int(
            "THOUGHT2CODE_MAX_CONCURRENT_GENERATIONS", 1, 1, 4
        )
        self.redis_url = redis_url if redis_url is not None else os.environ.get(
            "THOUGHT2CODE_REDIS_URL"
        )
        self.redis = None
        if self.redis_url and redis is not None:
            self.redis = redis.Redis.from_url(
                self.redis_url,
                socket_connect_timeout=1.0,
                socket_timeout=1.0,
                decode_responses=True,
            )
        self._lock = threading.Lock()
        self._client_windows: dict[str, tuple[float, int]] = {}
        self._local_slots = threading.BoundedSemaphore(self.max_concurrent)
        self._local_leases: set[str] = set()

    @property
    def shared_store_ready(self) -> bool:
        return self.redis is not None

    def request_allowed(self, client_identity: str) -> bool:
        client_identity = str(client_identity)[:128] or "unknown"
        if self.redis is not None:
            try:
                key = f"thought2code:rate:{sha256(client_identity.encode()).hexdigest()}"
                result = self.redis.eval(self._RATE_SCRIPT, 1, key, self.window_seconds)
                return int(result[0]) <= self.client_limit
            except Exception as error:
                raise ProtectionStoreUnavailable from error
        if self.public_demo:
            raise ProtectionStoreUnavailable

        now = time.monotonic()
        with self._lock:
            started, count = self._client_windows.get(client_identity, (now, 0))
            if now - started >= self.window_seconds:
                started, count = now, 0
            count += 1
            self._client_windows[client_identity] = (started, count)
            # Bound local memory if tests or local clients produce many identities.
            if len(self._client_windows) > 10_000:
                cutoff = now - self.window_seconds
                self._client_windows = {
                    key: value for key, value in self._client_windows.items()
                    if value[0] >= cutoff
                }
                if len(self._client_windows) > 10_000:
                    oldest = sorted(
                        self._client_windows,
                        key=lambda key: self._client_windows[key][0],
                    )[: len(self._client_windows) - 9_000]
                    for key in oldest:
                        self._client_windows.pop(key, None)
            return count <= self.client_limit

    def acquire_distributed_slot(self, *, lease_seconds: int = 1_800) -> str | None:
        token = uuid4().hex
        if self.redis is not None:
            try:
                acquired = self.redis.eval(
                    self._ACQUIRE_SCRIPT,
                    1,
                    "thought2code:generation-slots",
                    self.max_concurrent,
                    max(60, min(3_600, lease_seconds)) * 1_000,
                    token,
                )
            except Exception as error:
                raise ProtectionStoreUnavailable from error
            return token if int(acquired) == 1 else None
        if self.public_demo:
            raise ProtectionStoreUnavailable
        if not self._local_slots.acquire(blocking=False):
            return None
        with self._lock:
            self._local_leases.add(token)
        return token

    def release_distributed_slot(self, token: str) -> None:
        if self.redis is not None:
            try:
                self.redis.eval(
                    self._RELEASE_SCRIPT,
                    1,
                    "thought2code:generation-slots",
                    token,
                )
            except Exception as error:
                # The bounded lease expires automatically if release cannot reach Redis.
                raise ProtectionStoreUnavailable from error
            return
        with self._lock:
            if token in self._local_leases:
                self._local_leases.remove(token)
                self._local_slots.release()


protection = GenerationProtection()
