import pytest

from config.settings import get_settings
from src.tenancy import (
    TENANT_ALL,
    Principal,
    TenancyError,
    build_acl,
    namespace_for,
    principal_from_claims,
    query_filter,
)

S = get_settings()


def test_principal_from_claims():
    p = principal_from_claims({"sub": "alice", "tenant_id": "acme", "groups": ["finance", "bad group!"]},
                              {"rag:query"}, S)
    assert p.tenant == "acme" and p.groups == frozenset({"finance"}) and not p.is_admin
    assert p.acl_principals == sorted([TENANT_ALL, "user:alice", "group:finance"])


def test_missing_tenant_rejected_when_required():
    with pytest.raises(TenancyError):
        principal_from_claims({"sub": "alice"}, set(), S)


def test_default_tenant_when_not_required():
    s = S.model_copy(update={"REQUIRE_TENANT": False})
    assert principal_from_claims({"sub": "a"}, set(), s).tenant == s.DEFAULT_TENANT


@pytest.mark.parametrize("tenant", ["../x", "a b", "", "x" * 65])
def test_invalid_tenant_rejected(tenant):
    with pytest.raises(TenancyError):
        principal_from_claims({"sub": "a", "tenant_id": tenant}, set(), S)


def test_admin_scope():
    p = principal_from_claims({"sub": "a", "tenant_id": "acme"}, {S.JWT_ADMIN_SCOPE}, S)
    assert p.is_admin and query_filter(p) is None


def test_acl_defaults_to_whole_tenant():
    p = Principal(sub="alice", tenant="acme")
    assert build_acl(p, None, S) == sorted([TENANT_ALL, "user:alice"])


def test_acl_requires_membership_unless_admin():
    p = Principal(sub="alice", tenant="acme", groups=frozenset({"finance"}))
    assert build_acl(p, ["finance"], S) == ["group:finance", "user:alice"]
    with pytest.raises(TenancyError):
        build_acl(p, ["legal"], S)
    admin = Principal(sub="root", tenant="acme", is_admin=True)
    assert "group:legal" in build_acl(admin, ["legal"], S)


def test_visibility_rules():
    p = Principal(sub="bob", tenant="acme", groups=frozenset({"finance"}))
    assert p.can_see(["group:finance"])
    assert not p.can_see(["group:legal", "user:alice"])
    assert p.can_see(["group:legal"], owner="bob")
    assert query_filter(p) == {"acl": {"$in": p.acl_principals}}
    assert namespace_for("acme", S) == "tenant-acme"
