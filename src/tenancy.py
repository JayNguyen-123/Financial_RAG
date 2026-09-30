"""Tenant isolation and document-level access control (pure logic, no FastAPI).

Isolation is layered:
1. **Tenant** - every tenant has its own Pinecone namespace and storage prefix,
   so a query can never touch another tenant's vectors, even with a bad filter.
2. **ACL** - inside a tenant each vector carries an ``acl`` list of principals.
   Queries filter with ``{"acl": {"$in": <caller principals>}}``.

Principals:
    "tenant:all"   visible to everyone in the tenant (default)
    "group:<g>"    visible to members of group <g> (from the JWT groups claim)
    "user:<sub>"   the uploader (always added so owners keep access)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from config.settings import Settings

TENANT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]{0,63}$")
GROUP_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@\-]{0,127}$")
TENANT_ALL = "tenant:all"


class TenancyError(ValueError):
    """Raised for missing/invalid tenant or ACL input (maps to 401/403/400)."""


@dataclass(frozen=True)
class Principal:
    sub: str
    tenant: str
    groups: frozenset[str] = field(default_factory=frozenset)
    scopes: frozenset[str] = field(default_factory=frozenset)
    is_admin: bool = False

    @property
    def acl_principals(self) -> list[str]:
        return sorted({TENANT_ALL, f"user:{self.sub}", *(f"group:{g}" for g in self.groups)})

    def can_see(self, acl: list[str], owner: str | None = None) -> bool:
        if self.is_admin or owner == self.sub:
            return True
        return bool(set(acl) & set(self.acl_principals))

    def can_manage(self, owner: str | None) -> bool:
        return self.is_admin or owner == self.sub


def _as_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [v for v in re.split(r"[,\s]+", value) if v]
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value]
    return []


def principal_from_claims(claims: dict[str, Any], scopes: set[str], settings: Settings) -> Principal:
    sub = str(claims.get("sub") or "")
    if not sub:
        raise TenancyError("Token has no subject.")
    tenant = claims.get(settings.TENANT_CLAIM)
    if not tenant:
        if settings.REQUIRE_TENANT:
            raise TenancyError(f"Token is missing the '{settings.TENANT_CLAIM}' claim.")
        tenant = settings.DEFAULT_TENANT
    tenant = str(tenant)
    if not TENANT_RE.match(tenant):
        raise TenancyError("Invalid tenant identifier in token.")
    groups = frozenset(g for g in _as_str_list(claims.get(settings.GROUPS_CLAIM)) if GROUP_RE.match(g))
    return Principal(
        sub=sub,
        tenant=tenant,
        groups=groups,
        scopes=frozenset(scopes),
        is_admin=settings.JWT_ADMIN_SCOPE in scopes,
    )


def namespace_for(tenant: str, settings: Settings) -> str:
    return f"{settings.PINECONE_NAMESPACE_PREFIX}{tenant}"


def build_acl(principal: Principal, requested_groups: list[str] | None, settings: Settings) -> list[str]:
    """ACL for a newly ingested document.

    No groups -> visible to the whole tenant. With groups, the uploader must be a
    member of each (unless admin); otherwise a user could plant content into
    another group's retrieval results (a prompt-injection / poisoning vector).
    """
    groups = [g.strip() for g in (requested_groups or []) if g and g.strip()]
    if not groups:
        return sorted({TENANT_ALL, f"user:{principal.sub}"})
    if len(groups) > settings.MAX_ACL_GROUPS:
        raise TenancyError(f"At most {settings.MAX_ACL_GROUPS} groups may be granted.")
    bad = [g for g in groups if not GROUP_RE.match(g)]
    if bad:
        raise TenancyError(f"Invalid group name(s): {', '.join(bad[:3])}")
    if not principal.is_admin:
        foreign = sorted(set(groups) - set(principal.groups))
        if foreign:
            raise TenancyError(f"You are not a member of: {', '.join(foreign[:5])}")
    return sorted({f"user:{principal.sub}", *(f"group:{g}" for g in groups)})


def query_filter(principal: Principal) -> dict[str, Any] | None:
    """Pinecone metadata filter restricting results to documents the caller may see."""
    if principal.is_admin:
        return None
    return {"acl": {"$in": principal.acl_principals}}
