-- =============================================================================
-- V500: GreenLang Factors v0.1 Alpha — Canonical Schema (FORWARD)
-- =============================================================================
-- Description: First-class Postgres tables that mirror the FROZEN alpha
--              JSON Schema for Factor Records. Implements the registry
--              tables (source / methodology / geography / unit / factor_pack)
--              plus the central immutable `factor` table, an append-only
--              `factor_publish_log`, the URN pattern checks, the JSONB
--              provenance gates (`extraction`, `review`), and the
--              post-publish immutability trigger.
--
-- Authority:
--   - GreenLang_Climate_OS Final Product Definition Roadmap (CTO doc) §6.1
--   - Schema $id: https://schemas.greenlang.io/factors/factor_record_v0_1.schema.json
--   - Freeze date: 2026-04-25 (config/schemas/FACTOR_RECORD_V0_1_FREEZE.md)
--
-- Wave: B / TaskCreate #2 / WS1-T2
-- Postgres target: 16+
-- Author: GL-BackendDeveloper
-- Created: 2026-04-25
-- =============================================================================

-- -----------------------------------------------------------------------------
-- Schema setup
-- -----------------------------------------------------------------------------
CREATE SCHEMA IF NOT EXISTS factors_v0_1;
SET search_path TO factors_v0_1, public;

-- -----------------------------------------------------------------------------
-- Source registry — one row per upstream source (IPCC AR6, DEFRA 2025, etc.)
-- -----------------------------------------------------------------------------
CREATE TABLE source (
    pk_id              BIGSERIAL PRIMARY KEY,
    urn                TEXT NOT NULL UNIQUE
        CONSTRAINT source_urn_pattern CHECK (urn ~ '^urn:gl:source:[a-z0-9][a-z0-9-]*$'),
    source_id          TEXT NOT NULL UNIQUE, -- legacy id for back-compat
    name               TEXT NOT NULL,
    organization       TEXT NOT NULL,
    publication_country_iso  CHAR(2),
    primary_url        TEXT NOT NULL,
    licence            TEXT NOT NULL,
    license_class      TEXT NOT NULL CHECK (license_class IN (
        'public_us_government',
        'uk_open_government',
        'public_eu',
        'cc_by',
        'cc_by_sa',
        'registry_terms',
        'commercial_connector',
        'tenant_private'
    )),
    update_cadence     TEXT NOT NULL CHECK (update_cadence IN (
        'annual',
        'biannual',
        'quarterly',
        'monthly',
        'weekly',
        'daily',
        'hourly',
        'ad-hoc'
    )),
    source_owner       TEXT NOT NULL,
    parser_module      TEXT NOT NULL,
    parser_function    TEXT NOT NULL,
    parser_version     TEXT NOT NULL,
    source_version     TEXT NOT NULL,
    first_added_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    latest_ingestion_at TIMESTAMPTZ,
    legal_signoff_artifact TEXT,
    publication_url    TEXT NOT NULL,
    contact            JSONB,
    trust_tier         TEXT NOT NULL CHECK (trust_tier IN ('tier_1','tier_2','tier_3')),
    alpha_v0_1         BOOLEAN NOT NULL DEFAULT FALSE,
    provenance_completeness_score NUMERIC(3,2)
        CHECK (provenance_completeness_score IS NULL
               OR (provenance_completeness_score >= 0
                   AND provenance_completeness_score <= 1)),
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX source_alpha_v0_1_idx ON source (alpha_v0_1) WHERE alpha_v0_1 IS TRUE;

-- -----------------------------------------------------------------------------
-- Methodology registry
-- -----------------------------------------------------------------------------
CREATE TABLE methodology (
    pk_id              BIGSERIAL PRIMARY KEY,
    urn                TEXT NOT NULL UNIQUE
        CONSTRAINT methodology_urn_pattern CHECK (urn ~ '^urn:gl:methodology:[a-z0-9][a-z0-9-]*$'),
    name               TEXT NOT NULL,
    framework          TEXT NOT NULL,
    tier               TEXT,
    approach           TEXT CHECK (approach IS NULL OR approach IN (
        'activity-based',
        'spend-based',
        'supplier-specific',
        'hybrid',
        'measurement-based'
    )),
    boundary_template  TEXT,
    allocation_rules   TEXT,
    notes              TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Geography registry — countries, subregions, grid zones, etc.
-- -----------------------------------------------------------------------------
CREATE TABLE geography (
    pk_id              BIGSERIAL PRIMARY KEY,
    urn                TEXT NOT NULL UNIQUE
        CONSTRAINT geography_urn_pattern CHECK (urn ~ '^urn:gl:geo:(global|country|subregion|state_or_province|grid_zone|bidding_zone|balancing_authority):[a-zA-Z0-9._-]+$'),
    type               TEXT NOT NULL CHECK (type IN (
        'global',
        'country',
        'subregion',
        'state_or_province',
        'grid_zone',
        'bidding_zone',
        'balancing_authority'
    )),
    iso_code           CHAR(2),
    name               TEXT NOT NULL,
    parent_urn         TEXT REFERENCES geography(urn),
    centroid_lat       NUMERIC(10,7),
    centroid_lon       NUMERIC(10,7),
    tags               TEXT[],
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Unit registry — kgco2e/kwh, kgco2e/kg, kgco2e/tkm, etc.
-- -----------------------------------------------------------------------------
CREATE TABLE unit (
    pk_id              BIGSERIAL PRIMARY KEY,
    urn                TEXT NOT NULL UNIQUE
        CONSTRAINT unit_urn_pattern CHECK (urn ~ '^urn:gl:unit:[a-zA-Z0-9._/-]+$'),
    symbol             TEXT NOT NULL,
    dimension          TEXT NOT NULL,
    conversions        JSONB NOT NULL DEFAULT '{}'::jsonb,
    iso_reference      TEXT,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- -----------------------------------------------------------------------------
-- Factor pack — the addressable bundle each factor belongs to
-- -----------------------------------------------------------------------------
CREATE TABLE factor_pack (
    pk_id              BIGSERIAL PRIMARY KEY,
    urn                TEXT NOT NULL UNIQUE
        CONSTRAINT factor_pack_urn_pattern CHECK (urn ~ '^urn:gl:pack:[a-z0-9][a-z0-9-]*:[a-z0-9][a-z0-9._-]*:[A-Za-z0-9._-]+$'),
    name               TEXT NOT NULL,
    source_urn         TEXT NOT NULL REFERENCES source(urn),
    version            TEXT NOT NULL,
    release_notes      TEXT,
    tier               TEXT NOT NULL CHECK (tier IN ('community','commercial','private')),
    licence            TEXT NOT NULL,
    checksum           TEXT,
    valid_from         TIMESTAMPTZ NOT NULL,
    valid_until        TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX factor_pack_source_idx ON factor_pack (source_urn);

-- -----------------------------------------------------------------------------
-- Factor — central immutable table (post-publish enforced by trigger below)
-- -----------------------------------------------------------------------------
CREATE TABLE factor (
    pk_id              BIGSERIAL PRIMARY KEY,
    urn                TEXT NOT NULL UNIQUE
        CONSTRAINT factor_urn_pattern CHECK (urn ~ '^urn:gl:factor:[a-z0-9][a-z0-9-]*(:[A-Za-z0-9._-]+){2,4}:v[1-9][0-9]*$'),
    factor_id_alias    TEXT,
    source_urn         TEXT NOT NULL REFERENCES source(urn),
    factor_pack_urn    TEXT NOT NULL REFERENCES factor_pack(urn),
    name               TEXT NOT NULL CHECK (length(name) <= 200),
    description        TEXT NOT NULL CHECK (length(description) >= 30),
    category           TEXT NOT NULL CHECK (category IN (
        'scope1',
        'scope2_location_based',
        'scope2_market_based',
        'grid_intensity',
        'fuel',
        'refrigerant',
        'fugitive',
        'process',
        'cbam_default'
    )),
    value              NUMERIC(30,12) NOT NULL CHECK (value > 0),
    unit_urn           TEXT NOT NULL REFERENCES unit(urn),
    gwp_basis          TEXT NOT NULL CHECK (gwp_basis = 'ar6'),
    gwp_horizon        INTEGER NOT NULL CHECK (gwp_horizon IN (20,100,500)),
    geography_urn      TEXT NOT NULL REFERENCES geography(urn),
    vintage_start      DATE NOT NULL,
    vintage_end        DATE NOT NULL CHECK (vintage_end >= vintage_start),
    resolution         TEXT NOT NULL CHECK (resolution IN (
        'annual',
        'monthly',
        'hourly',
        'point-in-time'
    )),
    methodology_urn    TEXT NOT NULL REFERENCES methodology(urn),
    boundary           TEXT NOT NULL CHECK (length(boundary) >= 10),
    uncertainty        JSONB,
    licence            TEXT NOT NULL,
    licence_constraints JSONB,
    citations          JSONB NOT NULL CHECK (jsonb_array_length(citations) >= 1),
    tags               TEXT[],
    supersedes_urn     TEXT REFERENCES factor(urn),
    -- Provenance gate (extraction object — every field required by §19.1)
    extraction         JSONB NOT NULL,
    -- Review gate
    review             JSONB NOT NULL,
    -- Lifecycle
    published_at       TIMESTAMPTZ NOT NULL,
    deprecated_at      TIMESTAMPTZ,
    -- Mutation guard
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Provenance gate CHECKs (enforce all required sub-fields)
    CONSTRAINT factor_extraction_required_fields CHECK (
        extraction ? 'source_url'
        AND extraction ? 'source_record_id'
        AND extraction ? 'source_publication'
        AND extraction ? 'source_version'
        AND extraction ? 'raw_artifact_uri'
        AND extraction ? 'raw_artifact_sha256'
        AND extraction ? 'parser_id'
        AND extraction ? 'parser_version'
        AND extraction ? 'parser_commit'
        AND extraction ? 'row_ref'
        AND extraction ? 'ingested_at'
        AND extraction ? 'operator'
        AND (extraction->>'raw_artifact_sha256') ~ '^[a-f0-9]{64}$'
    ),
    CONSTRAINT factor_review_required_fields CHECK (
        review ? 'review_status'
        AND review ? 'reviewer'
        AND review ? 'reviewed_at'
        AND review->>'review_status' IN ('pending','approved','rejected')
    ),
    CONSTRAINT factor_review_approved_requires_approver CHECK (
        review->>'review_status' != 'approved'
        OR (review ? 'approved_by' AND review ? 'approved_at')
    ),
    CONSTRAINT factor_review_rejected_requires_reason CHECK (
        review->>'review_status' != 'rejected'
        OR review ? 'rejection_reason'
    )
);

-- Indexes for the factor table
CREATE INDEX factor_source_idx        ON factor (source_urn);
CREATE INDEX factor_pack_idx          ON factor (factor_pack_urn);
CREATE INDEX factor_geo_vintage_idx   ON factor (geography_urn, vintage_start, vintage_end);
CREATE INDEX factor_category_idx      ON factor (category);
CREATE INDEX factor_published_at_idx  ON factor (published_at);
CREATE INDEX factor_active_idx        ON factor ((review->>'review_status'))
    WHERE review->>'review_status' = 'approved';
CREATE INDEX factor_fts_idx           ON factor
    USING GIN (to_tsvector('english', name || ' ' || description));
CREATE INDEX factor_tags_idx          ON factor USING GIN (tags);
CREATE INDEX factor_alias_idx         ON factor (factor_id_alias)
    WHERE factor_id_alias IS NOT NULL;

-- -----------------------------------------------------------------------------
-- Immutability trigger: prevent UPDATE on key fields after publish
-- (Aligns with `published_at` immutability in JSON schema description.)
-- -----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION factor_immutable_trigger() RETURNS TRIGGER AS $$
BEGIN
    IF (OLD.urn IS NOT NULL AND OLD.published_at IS NOT NULL) THEN
        IF NEW.urn != OLD.urn THEN
            RAISE EXCEPTION 'factor.urn is immutable after publish';
        END IF;
        IF NEW.value != OLD.value THEN
            RAISE EXCEPTION 'factor.value is immutable after publish; supersede with a new urn';
        END IF;
        IF NEW.published_at != OLD.published_at THEN
            RAISE EXCEPTION 'factor.published_at is immutable';
        END IF;
        IF NEW.gwp_basis != OLD.gwp_basis THEN
            RAISE EXCEPTION 'factor.gwp_basis is immutable after publish';
        END IF;
        IF NEW.unit_urn != OLD.unit_urn THEN
            RAISE EXCEPTION 'factor.unit_urn is immutable after publish; supersede instead';
        END IF;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER factor_no_mutate_after_publish
BEFORE UPDATE ON factor
FOR EACH ROW EXECUTE FUNCTION factor_immutable_trigger();

-- -----------------------------------------------------------------------------
-- Factor publish log (append-only audit) — supports edition pinning lookups
-- -----------------------------------------------------------------------------
CREATE TABLE factor_publish_log (
    pk_id              BIGSERIAL PRIMARY KEY,
    factor_urn         TEXT NOT NULL,
    edition_id         TEXT NOT NULL,
    published_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    published_by       TEXT NOT NULL,
    diff_uri           TEXT,
    metadata           JSONB
);
CREATE INDEX factor_publish_log_factor_idx
    ON factor_publish_log (factor_urn, published_at DESC);
CREATE INDEX factor_publish_log_edition_idx
    ON factor_publish_log (edition_id);

-- =============================================================================
-- End V500
-- =============================================================================
