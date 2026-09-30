"""Per-user rate limits and daily LLM token budgets, stored in Redis.

Fixed-window counters (INCR + EXPIRE NX) are atomic per key, cheap, and good
enough for cost control. Keys are scoped by tenant *and* subject so one user
cannot exhaust another's allowance.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from config.settings import Settings

logger = logging.getLogger(__name__)


class QuotaExceeded(Exception):
    def __init__(self, kind: str, retry_after: int, message: str):
        super().__init__(message)
        self.kind = kind
        self.retry_after = max(1, int(retry_after))
        self.message = message


class QuotaBackendUnavailable(Exception):
    pass


@dataclass
class QuotaService:
    redis: Any
    settings: Settings

    def _enabled(self) -> bool:
        return self.settings.QUOTA_ENABLED and self.redis is not None

    async def _guard(self, coro):
        try:
            return await coro
        except (QuotaExceeded, QuotaBackendUnavailable):
            raise
        except Exception as exc:
            if self.settings.QUOTA_FAIL_OPEN:
                logger.warning("Quota backend unavailable; failing open", extra={"error": repr(exc)})
                return None
            raise QuotaBackendUnavailable(str(exc)) from exc

    async def _hit(self, key: str, limit: int, window: int, kind: str, label: str) -> None:
        if limit <= 0:
            return
        count = await self.redis.incr(key)
        if count == 1:
            await self.redis.expire(key, window)
        if count > limit:
            ttl = await self.redis.ttl(key)
            if ttl is None or ttl < 0:  # self-heal a key whose EXPIRE was lost
                await self.redis.expire(key, window)
                ttl = window
            raise QuotaExceeded(kind, ttl if ttl and ttl > 0 else window,
                                f"Rate limit exceeded: {limit} {label}.")

    @staticmethod
    def _day() -> str:
        return datetime.now(UTC).strftime("%Y%m%d")

    @staticmethod
    def _seconds_to_midnight() -> int:
        now = time.time()
        return int(86400 - (now % 86400))

    def _token_key(self, tenant: str, sub: str) -> str:
        return f"quota:tokens:{tenant}:{sub}:{self._day()}"

    # ── Public API ──────────────────────────────────────────────────────
    async def check_query(self, tenant: str, sub: str) -> None:
        if not self._enabled():
            return
        minute = int(time.time() // 60)
        await self._guard(self._hit(f"quota:q:{tenant}:{sub}:{minute}", self.settings.QUERY_RATE_PER_MINUTE,
                                    60, "query_rate", "queries per minute"))
        await self._guard(self._check_budget(tenant, sub))

    async def check_ingest(self, tenant: str, sub: str) -> None:
        if not self._enabled():
            return
        hour = int(time.time() // 3600)
        await self._guard(self._hit(f"quota:i:{tenant}:{sub}:{hour}", self.settings.INGEST_RATE_PER_HOUR,
                                    3600, "ingest_rate", "uploads per hour"))
        await self._guard(self._check_budget(tenant, sub))

    async def _check_budget(self, tenant: str, sub: str) -> None:
        budget = self.settings.DAILY_TOKEN_BUDGET
        if budget <= 0:
            return
        used = int(await self.redis.get(self._token_key(tenant, sub)) or 0)
        if used >= budget:
            raise QuotaExceeded("token_budget", self._seconds_to_midnight(),
                                f"Daily token budget of {budget:,} exhausted; resets at 00:00 UTC.")

    async def record_tokens(self, tenant: str, sub: str, tokens: int) -> None:
        if not self._enabled() or tokens <= 0:
            return

        async def _record():
            key = self._token_key(tenant, sub)
            await self.redis.incrby(key, int(tokens))
            await self.redis.expire(key, 2 * 86400)

        await self._guard(_record())


def record_tokens_sync(redis_client: Any, settings: Settings, tenant: str, sub: str, tokens: int) -> None:
    """Worker-side (synchronous) token accounting for ingestion LLM calls."""
    if not settings.QUOTA_ENABLED or tokens <= 0 or redis_client is None:
        return
    key = f"quota:tokens:{tenant}:{sub}:{datetime.now(UTC).strftime('%Y%m%d')}"
    try:
        redis_client.incrby(key, int(tokens))
        redis_client.expire(key, 2 * 86400)
    except Exception:
        logger.warning("Failed to record ingestion token usage", exc_info=True)
