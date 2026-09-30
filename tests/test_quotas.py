import asyncio

import pytest

from config.settings import get_settings
from src.quotas import QuotaBackendUnavailable, QuotaExceeded, QuotaService
from tests.fakes import FakeRedis


def _svc(redis=None, **overrides):
    s = get_settings().model_copy(update={"QUOTA_ENABLED": True, **overrides})
    return QuotaService(redis or FakeRedis(), s)


def run(coro):
    return asyncio.run(coro)


def test_query_rate_limit():
    svc = _svc(QUERY_RATE_PER_MINUTE=2, DAILY_TOKEN_BUDGET=0)
    run(svc.check_query("acme", "alice"))
    run(svc.check_query("acme", "alice"))
    with pytest.raises(QuotaExceeded) as exc:
        run(svc.check_query("acme", "alice"))
    assert exc.value.kind == "query_rate" and 1 <= exc.value.retry_after <= 60
    run(svc.check_query("acme", "bob"))  # other users unaffected


def test_token_budget():
    svc = _svc(QUERY_RATE_PER_MINUTE=0, DAILY_TOKEN_BUDGET=100)
    run(svc.check_query("acme", "alice"))
    run(svc.record_tokens("acme", "alice", 150))
    with pytest.raises(QuotaExceeded) as exc:
        run(svc.check_query("acme", "alice"))
    assert exc.value.kind == "token_budget"


def test_fail_open_and_fail_closed():
    run(_svc(FakeRedis(healthy=False), QUOTA_FAIL_OPEN=True).check_query("acme", "alice"))
    with pytest.raises(QuotaBackendUnavailable):
        run(_svc(FakeRedis(healthy=False), QUOTA_FAIL_OPEN=False).check_query("acme", "alice"))


def test_disabled():
    svc = _svc(QUOTA_ENABLED=False, QUERY_RATE_PER_MINUTE=1)
    for _ in range(5):
        run(svc.check_query("acme", "alice"))
