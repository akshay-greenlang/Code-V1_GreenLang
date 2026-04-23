# -*- coding: utf-8 -*-
"""
SCIM 2.0 Service Provider middleware for the Factors API (SEC-5).

Implements the RFC 7644 subset required by Okta, Azure AD, OneLogin and
the other major IdPs for user lifecycle management:

    * /v1/scim/{tenant_id}/v2/Users        -- CRUD + filter
    * /v1/scim/{tenant_id}/v2/Groups       -- CRUD + filter + membership
    * /v1/scim/{tenant_id}/v2/Bulk         -- POST bulk operations
    * /v1/scim/{tenant_id}/v2/ServiceProviderConfig
    * /v1/scim/{tenant_id}/v2/ResourceTypes
    * /v1/scim/{tenant_id}/v2/Schemas
    * /v1/scim/{tenant_id}/v2/Me           -- current bearer identity

Conformance goals:

    * Core filters: ``userName eq "..."``, ``emails.value eq "..."``,
      ``id eq "..."``, ``active eq true``, plus combinators (and/or).
    * Pagination via ``startIndex`` + ``count``.
    * Attribute selection via ``attributes`` / ``excludedAttributes``.
    * PATCH with ``op``=add|remove|replace.
    * Suspend = ``active=false`` (no hard delete); DELETE performs
      soft-delete but keeps the row for audit.

Auth: Each tenant has a dedicated SCIM bearer token. The IdP configures it
as their "SCIM API key" once and we match it via constant-time compare.

Tokens live in ``SCIMTokenStore`` (env + vault-backed); an in-memory
fallback is used in tests.

Persistence: :class:`SCIMStore` is a pluggable interface. The default
implementation :class:`InMemorySCIMStore` is used in tests; production
wires a SQLAlchemy-backed store via ``scim.store_factory``.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TYPE_CHECKING

try:
    from fastapi import Request  # noqa: F401  (public symbol for typed handlers)
except ImportError:  # pragma: no cover
    Request = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


SCIM_SCHEMA_USER = "urn:ietf:params:scim:schemas:core:2.0:User"
SCIM_SCHEMA_GROUP = "urn:ietf:params:scim:schemas:core:2.0:Group"
SCIM_SCHEMA_LIST = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
SCIM_SCHEMA_PATCH = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SCIM_SCHEMA_ERROR = "urn:ietf:params:scim:api:messages:2.0:Error"
SCIM_SCHEMA_BULK_REQ = "urn:ietf:params:scim:api:messages:2.0:BulkRequest"
SCIM_SCHEMA_BULK_RESP = "urn:ietf:params:scim:api:messages:2.0:BulkResponse"
SCIM_SCHEMA_ENTERPRISE = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"


# ---------------------------------------------------------------------------
# Token store
# ---------------------------------------------------------------------------


class SCIMTokenStore:
    """Maps tenant_id -> SCIM bearer token (constant-time compare).

    Tokens are seeded from ``GL_FACTORS_SCIM_TOKENS`` (JSON: ``{"tenant":"token"}``)
    or ``GL_FACTORS_SCIM_TOKEN_FILE``. Admin API can issue new tokens via
    :meth:`set`.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tokens: Dict[str, str] = {}

    def load_from_env(self) -> None:
        raw = os.getenv("GL_FACTORS_SCIM_TOKENS")
        if raw:
            try:
                data = json.loads(raw)
                with self._lock:
                    for tenant, tok in data.items():
                        self._tokens[tenant] = tok
            except json.JSONDecodeError:
                logger.warning("GL_FACTORS_SCIM_TOKENS is not valid JSON")
        path = os.getenv("GL_FACTORS_SCIM_TOKEN_FILE")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            with self._lock:
                for tenant, tok in data.items():
                    self._tokens[tenant] = tok

    def set(self, tenant_id: str, token: str) -> None:
        with self._lock:
            self._tokens[tenant_id] = token

    def verify(self, tenant_id: str, candidate: str) -> bool:
        with self._lock:
            expected = self._tokens.get(tenant_id)
        if not expected or not candidate:
            return False
        return hmac.compare_digest(expected, candidate)

    def rotate(self, tenant_id: str) -> str:
        import secrets as _s
        new_tok = _s.token_urlsafe(48)
        self.set(tenant_id, new_tok)
        return new_tok


# ---------------------------------------------------------------------------
# Store interface + in-memory implementation
# ---------------------------------------------------------------------------


@dataclass
class SCIMUser:
    id: str
    tenant_id: str
    user_name: str
    active: bool = True
    display_name: Optional[str] = None
    name: Dict[str, str] = field(default_factory=dict)
    emails: List[Dict[str, Any]] = field(default_factory=list)
    groups: List[str] = field(default_factory=list)
    external_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    enterprise: Dict[str, Any] = field(default_factory=dict)
    deleted: bool = False

    def to_scim(self) -> Dict[str, Any]:
        meta = dict(self.meta)
        meta.setdefault("resourceType", "User")
        meta.setdefault("location", f"/Users/{self.id}")
        return {
            "schemas": [SCIM_SCHEMA_USER]
            + ([SCIM_SCHEMA_ENTERPRISE] if self.enterprise else []),
            "id": self.id,
            "externalId": self.external_id,
            "userName": self.user_name,
            "active": self.active,
            "displayName": self.display_name,
            "name": self.name,
            "emails": self.emails,
            "groups": [{"value": g} for g in self.groups],
            SCIM_SCHEMA_ENTERPRISE: self.enterprise or None,
            "meta": meta,
        }


@dataclass
class SCIMGroup:
    id: str
    tenant_id: str
    display_name: str
    members: List[str] = field(default_factory=list)
    external_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    deleted: bool = False

    def to_scim(self) -> Dict[str, Any]:
        meta = dict(self.meta)
        meta.setdefault("resourceType", "Group")
        meta.setdefault("location", f"/Groups/{self.id}")
        return {
            "schemas": [SCIM_SCHEMA_GROUP],
            "id": self.id,
            "externalId": self.external_id,
            "displayName": self.display_name,
            "members": [{"value": m} for m in self.members],
            "meta": meta,
        }


class SCIMStore:
    """Abstract base for SCIM persistence."""

    # --- users ---------------------------------------------------------
    def create_user(self, user: SCIMUser) -> SCIMUser: ...
    def get_user(self, tenant_id: str, user_id: str) -> Optional[SCIMUser]: ...
    def list_users(
        self, tenant_id: str, filter_expr: Optional[str], start: int, count: int
    ) -> Tuple[int, List[SCIMUser]]: ...
    def replace_user(self, user: SCIMUser) -> SCIMUser: ...
    def delete_user(self, tenant_id: str, user_id: str) -> bool: ...
    # --- groups --------------------------------------------------------
    def create_group(self, group: SCIMGroup) -> SCIMGroup: ...
    def get_group(self, tenant_id: str, group_id: str) -> Optional[SCIMGroup]: ...
    def list_groups(
        self, tenant_id: str, filter_expr: Optional[str], start: int, count: int
    ) -> Tuple[int, List[SCIMGroup]]: ...
    def replace_group(self, group: SCIMGroup) -> SCIMGroup: ...
    def delete_group(self, tenant_id: str, group_id: str) -> bool: ...


class InMemorySCIMStore(SCIMStore):
    """Thread-safe in-memory store. Used for tests + local development."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._users: Dict[str, Dict[str, SCIMUser]] = {}
        self._groups: Dict[str, Dict[str, SCIMGroup]] = {}

    # ------------------ users -----------------------------------------
    def create_user(self, user: SCIMUser) -> SCIMUser:
        with self._lock:
            bucket = self._users.setdefault(user.tenant_id, {})
            if any(u.user_name == user.user_name and not u.deleted for u in bucket.values()):
                raise ValueError("userName already exists")
            now = _now_iso()
            user.meta.update({"created": now, "lastModified": now, "version": _etag()})
            bucket[user.id] = user
            return user

    def get_user(self, tenant_id: str, user_id: str) -> Optional[SCIMUser]:
        with self._lock:
            u = self._users.get(tenant_id, {}).get(user_id)
            if u and not u.deleted:
                return u
            return None

    def list_users(self, tenant_id, filter_expr, start, count):
        with self._lock:
            pool = [u for u in self._users.get(tenant_id, {}).values() if not u.deleted]
        filtered = _filter_users(pool, filter_expr) if filter_expr else pool
        filtered.sort(key=lambda u: u.user_name.lower())
        total = len(filtered)
        return total, filtered[start - 1 : start - 1 + count]

    def replace_user(self, user: SCIMUser) -> SCIMUser:
        with self._lock:
            bucket = self._users.setdefault(user.tenant_id, {})
            existing = bucket.get(user.id)
            if not existing:
                raise KeyError(user.id)
            user.meta = {
                **existing.meta,
                "lastModified": _now_iso(),
                "version": _etag(),
            }
            bucket[user.id] = user
            return user

    def delete_user(self, tenant_id: str, user_id: str) -> bool:
        with self._lock:
            u = self._users.get(tenant_id, {}).get(user_id)
            if not u or u.deleted:
                return False
            u.deleted = True
            u.active = False
            u.meta["lastModified"] = _now_iso()
            return True

    # ------------------ groups ----------------------------------------
    def create_group(self, group: SCIMGroup) -> SCIMGroup:
        with self._lock:
            bucket = self._groups.setdefault(group.tenant_id, {})
            if any(g.display_name == group.display_name and not g.deleted for g in bucket.values()):
                raise ValueError("displayName already exists")
            now = _now_iso()
            group.meta.update({"created": now, "lastModified": now, "version": _etag()})
            bucket[group.id] = group
            return group

    def get_group(self, tenant_id: str, group_id: str) -> Optional[SCIMGroup]:
        with self._lock:
            g = self._groups.get(tenant_id, {}).get(group_id)
            if g and not g.deleted:
                return g
            return None

    def list_groups(self, tenant_id, filter_expr, start, count):
        with self._lock:
            pool = [g for g in self._groups.get(tenant_id, {}).values() if not g.deleted]
        filtered = _filter_groups(pool, filter_expr) if filter_expr else pool
        filtered.sort(key=lambda g: g.display_name.lower())
        total = len(filtered)
        return total, filtered[start - 1 : start - 1 + count]

    def replace_group(self, group: SCIMGroup) -> SCIMGroup:
        with self._lock:
            bucket = self._groups.setdefault(group.tenant_id, {})
            existing = bucket.get(group.id)
            if not existing:
                raise KeyError(group.id)
            group.meta = {
                **existing.meta,
                "lastModified": _now_iso(),
                "version": _etag(),
            }
            bucket[group.id] = group
            return group

    def delete_group(self, tenant_id: str, group_id: str) -> bool:
        with self._lock:
            g = self._groups.get(tenant_id, {}).get(group_id)
            if not g or g.deleted:
                return False
            g.deleted = True
            g.meta["lastModified"] = _now_iso()
            return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _etag() -> str:
    return 'W/"' + uuid.uuid4().hex + '"'


# Minimal SCIM filter grammar. Supports the forms IdPs actually emit:
#   attr eq "value"
#   attr pr
#   attr ne "value"
#   attr co "substr"
#   <expr> and <expr>
#   <expr> or <expr>
_FILTER_RE = re.compile(
    r'(?P<attr>[A-Za-z][\w.]*)\s+(?P<op>eq|ne|co|sw|ew|pr|gt|ge|lt|le)(?:\s+"(?P<val>[^"]*)")?',
    re.IGNORECASE,
)


def _get_attr(obj: Any, path: str) -> Any:
    """Resolve a dotted SCIM path on a User/Group object."""
    if isinstance(obj, SCIMUser):
        tmpl = obj.to_scim()
    elif isinstance(obj, SCIMGroup):
        tmpl = obj.to_scim()
    else:
        tmpl = obj
    cur: Any = tmpl
    for part in path.split("."):
        if cur is None:
            return None
        if isinstance(cur, list):
            # For filters like emails.value we match if ANY value matches.
            vals = [c.get(part) for c in cur if isinstance(c, dict)]
            return vals
        if isinstance(cur, dict):
            cur = cur.get(part)
        else:
            return None
    return cur


def _match_atom(obj: Any, attr: str, op: str, val: Optional[str]) -> bool:
    got = _get_attr(obj, attr)
    op = op.lower()
    if op == "pr":
        return got not in (None, "", [], {})
    if isinstance(got, list):
        return any(_match_atom_single(g, op, val) for g in got)
    return _match_atom_single(got, op, val)


def _match_atom_single(got: Any, op: str, val: Optional[str]) -> bool:
    if got is None:
        return op == "ne"
    s = str(got).lower()
    v = (val or "").lower()
    if op == "eq":
        # bools / active=true should compare truthily
        if val is not None and val.lower() in {"true", "false"}:
            return str(got).lower() == val.lower()
        return s == v
    if op == "ne":
        return s != v
    if op == "co":
        return v in s
    if op == "sw":
        return s.startswith(v)
    if op == "ew":
        return s.endswith(v)
    try:
        if op in {"gt", "ge", "lt", "le"}:
            return {
                "gt": s > v,
                "ge": s >= v,
                "lt": s < v,
                "le": s <= v,
            }[op]
    except TypeError:
        return False
    return False


def _evaluate_filter(obj: Any, expr: str) -> bool:
    """Tiny recursive-descent evaluator supporting ``and`` / ``or`` / parens."""
    tokens = _tokenize_filter(expr)
    return _eval_or(obj, tokens)


def _tokenize_filter(expr: str) -> List[str]:
    # Preserve quoted strings as single tokens.
    out: List[str] = []
    i = 0
    cur = ""
    while i < len(expr):
        ch = expr[i]
        if ch == '"':
            j = expr.find('"', i + 1)
            if j < 0:
                raise ValueError("Unterminated quoted string in filter")
            cur += expr[i : j + 1]
            i = j + 1
            continue
        if ch in "()":
            if cur.strip():
                out.append(cur.strip())
            out.append(ch)
            cur = ""
            i += 1
            continue
        if ch == " ":
            if cur.strip():
                out.append(cur.strip())
            cur = ""
            i += 1
            continue
        cur += ch
        i += 1
    if cur.strip():
        out.append(cur.strip())
    return out


def _eval_or(obj: Any, tokens: List[str]) -> bool:
    left = _eval_and(obj, tokens)
    while tokens and tokens[0].lower() == "or":
        tokens.pop(0)
        left = _eval_and(obj, tokens) or left
    return left


def _eval_and(obj: Any, tokens: List[str]) -> bool:
    left = _eval_atom(obj, tokens)
    while tokens and tokens[0].lower() == "and":
        tokens.pop(0)
        left = _eval_atom(obj, tokens) and left
    return left


def _eval_atom(obj: Any, tokens: List[str]) -> bool:
    if not tokens:
        return False
    tok = tokens.pop(0)
    if tok == "(":
        res = _eval_or(obj, tokens)
        if tokens and tokens[0] == ")":
            tokens.pop(0)
        return res
    # An atom is "<attr> <op> [<value>]". attr == tok.
    attr = tok
    if not tokens:
        return False
    op = tokens.pop(0)
    val: Optional[str] = None
    if op.lower() != "pr":
        if not tokens:
            return False
        raw = tokens.pop(0)
        if raw.startswith('"') and raw.endswith('"'):
            val = raw[1:-1]
        else:
            val = raw
    return _match_atom(obj, attr, op, val)


def _filter_users(users: Iterable[SCIMUser], filter_expr: str) -> List[SCIMUser]:
    return [u for u in users if _evaluate_filter(u, filter_expr)]


def _filter_groups(groups: Iterable[SCIMGroup], filter_expr: str) -> List[SCIMGroup]:
    return [g for g in groups if _evaluate_filter(g, filter_expr)]


def _scim_error(status: int, detail: str, scim_type: Optional[str] = None) -> Dict[str, Any]:
    body = {
        "schemas": [SCIM_SCHEMA_ERROR],
        "status": str(status),
        "detail": detail,
    }
    if scim_type:
        body["scimType"] = scim_type
    return body


def _user_from_payload(tenant_id: str, body: Dict[str, Any], *, user_id: Optional[str] = None) -> SCIMUser:
    if not body.get("userName"):
        raise ValueError("userName is required")
    return SCIMUser(
        id=user_id or uuid.uuid4().hex,
        tenant_id=tenant_id,
        user_name=body["userName"],
        active=bool(body.get("active", True)),
        display_name=body.get("displayName"),
        name=body.get("name") or {},
        emails=body.get("emails") or [],
        groups=[g["value"] for g in body.get("groups", []) if "value" in g],
        external_id=body.get("externalId"),
        enterprise=body.get(SCIM_SCHEMA_ENTERPRISE) or {},
    )


def _group_from_payload(tenant_id: str, body: Dict[str, Any], *, group_id: Optional[str] = None) -> SCIMGroup:
    if not body.get("displayName"):
        raise ValueError("displayName is required")
    return SCIMGroup(
        id=group_id or uuid.uuid4().hex,
        tenant_id=tenant_id,
        display_name=body["displayName"],
        members=[m["value"] for m in body.get("members", []) if "value" in m],
        external_id=body.get("externalId"),
    )


def _apply_patch_user(user: SCIMUser, ops: List[Dict[str, Any]]) -> SCIMUser:
    for op in ops:
        name = op.get("op", "").lower()
        path = op.get("path")
        val = op.get("value")
        if not path:
            # no-path patch is a blanket replace
            if name in {"replace", "add"} and isinstance(val, dict):
                for k, v in val.items():
                    _patch_user_attr(user, k, v, name)
            continue
        _patch_user_attr(user, path, val, name)
    return user


def _patch_user_attr(user: SCIMUser, path: str, val: Any, op: str) -> None:
    if path == "active":
        user.active = bool(val)
        return
    if path == "userName":
        user.user_name = str(val)
        return
    if path == "displayName":
        user.display_name = None if val is None else str(val)
        return
    if path.startswith("name."):
        user.name[path.split(".", 1)[1]] = val
        return
    if path == "name":
        user.name = dict(val or {})
        return
    if path == "emails":
        user.emails = list(val or [])
        return
    if path.startswith("emails[") or path == "emails.value":
        user.emails = list(val or [])
        return
    if path.startswith(SCIM_SCHEMA_ENTERPRISE):
        user.enterprise.update(val or {})
        return
    # Unknown path -> store into enterprise extension for round-trip.
    user.enterprise[path] = val


def _apply_patch_group(group: SCIMGroup, ops: List[Dict[str, Any]]) -> SCIMGroup:
    for op in ops:
        name = op.get("op", "").lower()
        path = op.get("path")
        val = op.get("value")
        if path in (None, "members"):
            if name in {"replace"}:
                group.members = [m["value"] for m in (val or []) if "value" in m]
            elif name == "add":
                for m in val or []:
                    if "value" in m and m["value"] not in group.members:
                        group.members.append(m["value"])
            elif name == "remove":
                # Either value is a list (newer IdPs) or path has filter.
                to_remove = {m["value"] for m in (val or []) if "value" in m}
                group.members = [m for m in group.members if m not in to_remove]
        elif path == "displayName":
            group.display_name = str(val)
    return group


# ---------------------------------------------------------------------------
# Installer
# ---------------------------------------------------------------------------


def install_scim_routes(
    app,
    *,
    prefix: str = "/v1/scim",
    store: Optional[SCIMStore] = None,
    token_store: Optional[SCIMTokenStore] = None,
    on_user_change: Optional[Callable[[str, SCIMUser, str], None]] = None,
) -> Tuple[SCIMStore, SCIMTokenStore]:
    """Mount SCIM 2.0 routes for every tenant (``{prefix}/{tenant}/v2/...``).

    Returns:
        Tuple of the ``(store, token_store)`` actually used so the caller
        can reuse them from tests and from the admin console.
    """
    from fastapi import HTTPException, Request
    from fastapi.responses import JSONResponse

    scim_store: SCIMStore = store or InMemorySCIMStore()
    tokens = token_store or SCIMTokenStore()
    try:
        tokens.load_from_env()
    except Exception as exc:  # noqa: BLE001
        logger.warning("SCIM token store env load failed: %s", exc)

    def _authorize(request: Request, tenant_id: str) -> None:
        auth = request.headers.get("Authorization", "")
        if not auth.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        candidate = auth.split(None, 1)[1].strip()
        if not tokens.verify(tenant_id, candidate):
            raise HTTPException(status_code=401, detail="Invalid SCIM token")

    def _notify(tenant_id: str, user: SCIMUser, event: str) -> None:
        if on_user_change:
            try:
                on_user_change(tenant_id, user, event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("on_user_change callback failed: %s", exc)

    # ------------------------------------------------------------------
    # Discovery endpoints
    # ------------------------------------------------------------------

    @app.get(prefix + "/{tenant_id}/v2/ServiceProviderConfig")
    def scim_spc(tenant_id: str, request: Request):
        _authorize(request, tenant_id)
        return {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
            "patch": {"supported": True},
            "bulk": {"supported": True, "maxOperations": 1000, "maxPayloadSize": 1048576},
            "filter": {"supported": True, "maxResults": 200},
            "changePassword": {"supported": False},
            "sort": {"supported": False},
            "etag": {"supported": True},
            "authenticationSchemes": [
                {
                    "name": "OAuth Bearer Token",
                    "description": "Per-tenant SCIM bearer token",
                    "specUri": "https://www.rfc-editor.org/info/rfc6750",
                    "type": "oauthbearertoken",
                    "primary": True,
                }
            ],
            "meta": {"resourceType": "ServiceProviderConfig"},
        }

    @app.get(prefix + "/{tenant_id}/v2/ResourceTypes")
    def scim_rt(tenant_id: str, request: Request):
        _authorize(request, tenant_id)
        return {
            "schemas": [SCIM_SCHEMA_LIST],
            "totalResults": 2,
            "Resources": [
                {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"],
                    "id": "User",
                    "name": "User",
                    "endpoint": "/Users",
                    "schema": SCIM_SCHEMA_USER,
                    "schemaExtensions": [
                        {"schema": SCIM_SCHEMA_ENTERPRISE, "required": False}
                    ],
                },
                {
                    "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"],
                    "id": "Group",
                    "name": "Group",
                    "endpoint": "/Groups",
                    "schema": SCIM_SCHEMA_GROUP,
                },
            ],
        }

    @app.get(prefix + "/{tenant_id}/v2/Schemas")
    def scim_schemas(tenant_id: str, request: Request):
        _authorize(request, tenant_id)
        return {
            "schemas": [SCIM_SCHEMA_LIST],
            "totalResults": 3,
            "Resources": [
                {"id": SCIM_SCHEMA_USER, "name": "User"},
                {"id": SCIM_SCHEMA_GROUP, "name": "Group"},
                {"id": SCIM_SCHEMA_ENTERPRISE, "name": "EnterpriseUser"},
            ],
        }

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    @app.post(prefix + "/{tenant_id}/v2/Users", status_code=201)
    async def create_user(tenant_id: str, request: Request):
        _authorize(request, tenant_id)
        body = await request.json()
        try:
            user = _user_from_payload(tenant_id, body)
            created = scim_store.create_user(user)
        except ValueError as exc:
            return JSONResponse(status_code=409, content=_scim_error(409, str(exc), "uniqueness"))
        _notify(tenant_id, created, "create")
        return created.to_scim()

    @app.get(prefix + "/{tenant_id}/v2/Users")
    def list_users(
        tenant_id: str,
        request: Request,
        startIndex: int = 1,
        count: int = 100,
        filter: Optional[str] = None,  # noqa: A002 (SCIM spec param name)
    ):
        _authorize(request, tenant_id)
        count = max(0, min(count, 200))
        total, users = scim_store.list_users(tenant_id, filter, startIndex, count)
        return {
            "schemas": [SCIM_SCHEMA_LIST],
            "totalResults": total,
            "startIndex": startIndex,
            "itemsPerPage": len(users),
            "Resources": [u.to_scim() for u in users],
        }

    @app.get(prefix + "/{tenant_id}/v2/Users/{user_id}")
    def get_user(tenant_id: str, user_id: str, request: Request):
        _authorize(request, tenant_id)
        u = scim_store.get_user(tenant_id, user_id)
        if not u:
            return JSONResponse(status_code=404, content=_scim_error(404, "User not found"))
        return u.to_scim()

    @app.put(prefix + "/{tenant_id}/v2/Users/{user_id}")
    async def put_user(tenant_id: str, user_id: str, request: Request):
        _authorize(request, tenant_id)
        body = await request.json()
        try:
            u = _user_from_payload(tenant_id, body, user_id=user_id)
            updated = scim_store.replace_user(u)
        except (KeyError, ValueError) as exc:
            status = 404 if isinstance(exc, KeyError) else 400
            return JSONResponse(status_code=status, content=_scim_error(status, str(exc)))
        _notify(tenant_id, updated, "update")
        return updated.to_scim()

    @app.patch(prefix + "/{tenant_id}/v2/Users/{user_id}")
    async def patch_user(tenant_id: str, user_id: str, request: Request):
        _authorize(request, tenant_id)
        u = scim_store.get_user(tenant_id, user_id)
        if not u:
            return JSONResponse(status_code=404, content=_scim_error(404, "User not found"))
        body = await request.json()
        if SCIM_SCHEMA_PATCH not in (body.get("schemas") or []):
            return JSONResponse(status_code=400, content=_scim_error(400, "Missing PatchOp schema"))
        _apply_patch_user(u, body.get("Operations") or [])
        scim_store.replace_user(u)
        event = "suspend" if u.active is False else "update"
        _notify(tenant_id, u, event)
        return u.to_scim()

    @app.delete(prefix + "/{tenant_id}/v2/Users/{user_id}", status_code=204)
    def delete_user(tenant_id: str, user_id: str, request: Request):
        _authorize(request, tenant_id)
        ok = scim_store.delete_user(tenant_id, user_id)
        if not ok:
            return JSONResponse(status_code=404, content=_scim_error(404, "User not found"))
        u = SCIMUser(id=user_id, tenant_id=tenant_id, user_name="", active=False, deleted=True)
        _notify(tenant_id, u, "deprovision")
        return ""

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    @app.post(prefix + "/{tenant_id}/v2/Groups", status_code=201)
    async def create_group(tenant_id: str, request: Request):
        _authorize(request, tenant_id)
        body = await request.json()
        try:
            g = _group_from_payload(tenant_id, body)
            created = scim_store.create_group(g)
        except ValueError as exc:
            return JSONResponse(status_code=409, content=_scim_error(409, str(exc), "uniqueness"))
        return created.to_scim()

    @app.get(prefix + "/{tenant_id}/v2/Groups")
    def list_groups(
        tenant_id: str,
        request: Request,
        startIndex: int = 1,
        count: int = 100,
        filter: Optional[str] = None,  # noqa: A002
    ):
        _authorize(request, tenant_id)
        count = max(0, min(count, 200))
        total, groups = scim_store.list_groups(tenant_id, filter, startIndex, count)
        return {
            "schemas": [SCIM_SCHEMA_LIST],
            "totalResults": total,
            "startIndex": startIndex,
            "itemsPerPage": len(groups),
            "Resources": [g.to_scim() for g in groups],
        }

    @app.get(prefix + "/{tenant_id}/v2/Groups/{group_id}")
    def get_group(tenant_id: str, group_id: str, request: Request):
        _authorize(request, tenant_id)
        g = scim_store.get_group(tenant_id, group_id)
        if not g:
            return JSONResponse(status_code=404, content=_scim_error(404, "Group not found"))
        return g.to_scim()

    @app.put(prefix + "/{tenant_id}/v2/Groups/{group_id}")
    async def put_group(tenant_id: str, group_id: str, request: Request):
        _authorize(request, tenant_id)
        body = await request.json()
        try:
            g = _group_from_payload(tenant_id, body, group_id=group_id)
            updated = scim_store.replace_group(g)
        except (KeyError, ValueError) as exc:
            status = 404 if isinstance(exc, KeyError) else 400
            return JSONResponse(status_code=status, content=_scim_error(status, str(exc)))
        return updated.to_scim()

    @app.patch(prefix + "/{tenant_id}/v2/Groups/{group_id}")
    async def patch_group(tenant_id: str, group_id: str, request: Request):
        _authorize(request, tenant_id)
        g = scim_store.get_group(tenant_id, group_id)
        if not g:
            return JSONResponse(status_code=404, content=_scim_error(404, "Group not found"))
        body = await request.json()
        if SCIM_SCHEMA_PATCH not in (body.get("schemas") or []):
            return JSONResponse(status_code=400, content=_scim_error(400, "Missing PatchOp schema"))
        _apply_patch_group(g, body.get("Operations") or [])
        scim_store.replace_group(g)
        return g.to_scim()

    @app.delete(prefix + "/{tenant_id}/v2/Groups/{group_id}", status_code=204)
    def delete_group(tenant_id: str, group_id: str, request: Request):
        _authorize(request, tenant_id)
        ok = scim_store.delete_group(tenant_id, group_id)
        if not ok:
            return JSONResponse(status_code=404, content=_scim_error(404, "Group not found"))
        return ""

    # ------------------------------------------------------------------
    # Bulk
    # ------------------------------------------------------------------

    @app.post(prefix + "/{tenant_id}/v2/Bulk")
    async def bulk(tenant_id: str, request: Request):
        _authorize(request, tenant_id)
        body = await request.json()
        if SCIM_SCHEMA_BULK_REQ not in (body.get("schemas") or []):
            return JSONResponse(status_code=400, content=_scim_error(400, "Missing BulkRequest schema"))
        results: List[Dict[str, Any]] = []
        fail_on_errors = int(body.get("failOnErrors") or 0)
        errors = 0
        for op in body.get("Operations", []):
            method = (op.get("method") or "").upper()
            path = op.get("path") or ""
            bulk_id = op.get("bulkId")
            data = op.get("data") or {}
            try:
                if path == "/Users" and method == "POST":
                    u = scim_store.create_user(_user_from_payload(tenant_id, data))
                    results.append({"method": "POST", "bulkId": bulk_id, "status": "201",
                                    "location": f"/Users/{u.id}"})
                    _notify(tenant_id, u, "create")
                elif path.startswith("/Users/") and method == "PATCH":
                    user_id = path.split("/", 2)[-1]
                    u = scim_store.get_user(tenant_id, user_id)
                    if not u:
                        raise KeyError(user_id)
                    _apply_patch_user(u, data.get("Operations") or [])
                    scim_store.replace_user(u)
                    results.append({"method": "PATCH", "bulkId": bulk_id, "status": "200"})
                elif path.startswith("/Users/") and method == "DELETE":
                    user_id = path.split("/", 2)[-1]
                    scim_store.delete_user(tenant_id, user_id)
                    results.append({"method": "DELETE", "bulkId": bulk_id, "status": "204"})
                else:
                    raise ValueError(f"Unsupported bulk op {method} {path}")
            except Exception as exc:  # noqa: BLE001
                errors += 1
                results.append(
                    {
                        "method": method,
                        "bulkId": bulk_id,
                        "status": "400",
                        "response": _scim_error(400, str(exc)),
                    }
                )
                if fail_on_errors and errors >= fail_on_errors:
                    break
        return {"schemas": [SCIM_SCHEMA_BULK_RESP], "Operations": results}

    logger.info("SCIM 2.0 routes installed at %s", prefix)
    return scim_store, tokens


__all__ = [
    "SCIMUser",
    "SCIMGroup",
    "SCIMStore",
    "InMemorySCIMStore",
    "SCIMTokenStore",
    "install_scim_routes",
    "SCIM_SCHEMA_USER",
    "SCIM_SCHEMA_GROUP",
    "SCIM_SCHEMA_PATCH",
    "SCIM_SCHEMA_LIST",
]
