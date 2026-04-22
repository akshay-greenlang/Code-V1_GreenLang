# ADR 2026-04-22 — JWT secret env var unification

**Status**: Accepted.
**Date**: 2026-04-22.
**Decider**: Platform security lead + Factors engineering lead.
**Supersedes**: n/a.
**Superseded by**: n/a.

## Context

Two env var names were historically used for the JWT signing secret:

- `JWT_SECRET` — used by `greenlang/integration/api/dependencies.py`
  (`_validate_jwt_secret`, imported by many routes via `get_current_user`).
- `GL_JWT_SECRET` — used by the inline `get_current_user` in
  `greenlang/integration/api/main.py` line 216 and by the Factors
  middleware stack (`middleware/auth_metering.py`).

The result: operators deploying the Factors API had to set BOTH env vars
or risk an auth failure on a subset of routes. The deployment runbook
`docs/deployment/FACTORS-API-DEPLOY.md` called this out as gotcha #3
and the K8s `externalsecrets.yaml` template sets both names as a
belt-and-suspenders workaround.

This ADR formalizes the deprecation path.

## Decision

1. **`GL_JWT_SECRET` is canonical.** All new code reads this name. All
   platform products (Factors, CBAM, CSRD, Scope Engine) standardize on
   the `GL_` prefix that already governs `GL_FACTORS_SIGNING_SECRET`,
   `GL_FACTORS_TIER`, `GL_API_KEYS`, `GL_ENV`, `GL_FACTORS_ED25519_PRIVATE_KEY`.
2. **`JWT_SECRET` remains readable through v1.0.x** via a compatibility
   helper `get_jwt_secret()` in `dependencies.py`. When only the legacy
   name is set, a one-time deprecation warning is logged.
3. **v1.1.0 removes the legacy path.** The helper will stop reading
   `JWT_SECRET`; callers that still use the legacy name will fail loudly
   at import time in production (matching the existing insecure-secret
   behavior).

## Migration path

| Phase | Timeline | Action |
|---|---|---|
| **Now (v1.0.x)** | 2026-04-22 → v1.1.0 release | Operators MAY set `GL_JWT_SECRET` OR `JWT_SECRET`. The compat helper honours both, preferring `GL_JWT_SECRET`. Deprecation warning fires once per process when only the legacy name is set. |
| **Pre-v1.1.0** | 8-week window | `deployment/k8s/factors/base/externalsecrets.yaml` template updated to set ONLY `GL_JWT_SECRET`. Runbook gotcha #3 marked resolved. Every integration test + smoke test migrated. |
| **v1.1.0 release** | v1.1.0 cut | Helper stops reading `JWT_SECRET`. Legacy key removed from all templates. Release notes call out the breakage. |

## Consequences

**Positive**:
- One env var, one source of truth, no more dual-set templates.
- New platform products inherit the `GL_` prefix convention.
- Reduced attack surface: fewer places an operator can get the config
  half-right.

**Negative**:
- A short window where both names are read — introduces one extra env
  lookup per process start. Negligible.
- Any external tooling that injects `JWT_SECRET` needs to update by
  v1.1.0. Surfaced via the deprecation warning logged on import.

## Verification

- `tests/factors/security/test_jwt_secret_compat.py` asserts:
  1. `GL_JWT_SECRET` is preferred when both set.
  2. `JWT_SECRET` alone works + emits exactly one deprecation warning.
  3. Neither set → downstream `_validate_jwt_secret` raises in prod
     and warns in dev (existing behaviour preserved).
  4. Reading order is stable across process restarts.

## References

- `greenlang/integration/api/dependencies.py::get_jwt_secret`
- `greenlang/integration/api/main.py::get_current_user`
- `docs/deployment/FACTORS-API-DEPLOY.md` gotcha #3
- `deployment/k8s/factors/base/externalsecrets.yaml`
- `docs/security/FACTORS_API_HARDENING.md` §2 (AuthN/Z)
