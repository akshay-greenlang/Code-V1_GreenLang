-- =============================================================================
-- V504: GreenLang Factors v0.1 Phase 2 — API Keys + Entitlements + Releases
-- =============================================================================
-- Description: Phase 2 / WS7 migration that creates the factor catalog
--              access-control + release-manifest plane:
--                * `factors_v0_1.api_keys` — tenant-scoped API key hashes
--                  (separate from `public.api_keys` which is user-scoped;
--                  see "Linkage to SEC-001" note below).
--                * `factors_v0_1.entitlements` — DB-backed mirror of
--                  `config/entitlements/alpha_v0_1.yaml` (tenant -> source
--                  grant matrix).
--                * `factors_v0_1.release_manifests` — append-only record
--                  of every signed factor pack release (release_id,
--                  factor_urns array, schema_version, signature).
--
-- Linkage to SEC-001 (`public.api_keys`):
--   The SEC-001 / V009 service ships a `public.api_keys` table whose row
--   shape is **user-scoped** (user_id UUID, name, key_prefix, rate-limit
--   columns, allowed_ips, etc.). The factor catalog access-control plane
--   is **tenant-scoped** with source-URN-keyed entitlements; the column
--   set does not overlap meaningfully (no user_id, no rate-limits, no IP
--   ACL, no key_prefix), so we deliberately do NOT reuse
--   `public.api_keys`. A future v1.0 migration may unify the two via a
--   separate junction table once the tenant<->user model is finalized;
--   for v0.1 alpha they are two independent surfaces.
--
-- Authority:
--   - GreenLang Factors Phase 2 master plan §2.4 (table inventory rows
--     #12 api_keys, #13 entitlements, #14 release_manifests).
--   - CTO Phase 2 brief (TaskCreate #7 / WS7-T1).
--   - `config/entitlements/alpha_v0_1.yaml` (Phase 1 source-rights output).
--
-- Phase: 2 / TaskCreate #7 / WS7-T1
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-27
-- =============================================================================

SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- 1. api_keys — tenant-scoped factor catalog API keys.
--    `key_hash` stores a bcrypt or argon2id hash; plaintext keys are NEVER
--    stored. `scopes` is a flat array of strings ('factors:read',
--    'packs:download:tier_1', 'admin:publish', ...).
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.api_keys (
    pk_id          BIGSERIAL PRIMARY KEY,
    key_hash       TEXT NOT NULL UNIQUE,
    tenant         TEXT NOT NULL,
    scopes         TEXT[] NOT NULL,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_used_at   TIMESTAMPTZ,
    revoked_at     TIMESTAMPTZ,
    metadata       JSONB
);

-- Active-keys-by-tenant index (revoked rows excluded — partial index keeps
-- the working set small even after years of key rotation).
CREATE INDEX api_keys_tenant_idx
    ON factors_v0_1.api_keys (tenant)
    WHERE revoked_at IS NULL;

-- -----------------------------------------------------------------------------
-- 2. entitlements — tenant -> source URN grant matrix.
--    DB-backed mirror of `config/entitlements/alpha_v0_1.yaml`. The
--    SourceRightsService consults this table at query time.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.entitlements (
    pk_id          BIGSERIAL PRIMARY KEY,
    tenant         TEXT NOT NULL,
    source_urn     TEXT NOT NULL REFERENCES factors_v0_1.source(urn),
    granted_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at     TIMESTAMPTZ,
    terms_uri      TEXT,
    metadata       JSONB,
    UNIQUE (tenant, source_urn)
);

CREATE INDEX entitlements_tenant_idx
    ON factors_v0_1.entitlements (tenant);

CREATE INDEX entitlements_source_idx
    ON factors_v0_1.entitlements (source_urn);

-- -----------------------------------------------------------------------------
-- 3. release_manifests — append-only signed release records.
--    Each row is the canonical record of a published factor pack edition:
--    the release_id (URN-shaped), the array of factor URNs included, the
--    schema_version under which it was published, and the signature blob.
-- -----------------------------------------------------------------------------
CREATE TABLE factors_v0_1.release_manifests (
    pk_id          BIGSERIAL PRIMARY KEY,
    release_id     TEXT NOT NULL UNIQUE,
    factor_urns    TEXT[] NOT NULL,
    schema_version TEXT NOT NULL,
    signature      TEXT,
    released_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    released_by    TEXT NOT NULL,
    metadata       JSONB
);

CREATE INDEX release_manifests_released_at_idx
    ON factors_v0_1.release_manifests (released_at DESC);

-- =============================================================================
-- End V504
-- =============================================================================
