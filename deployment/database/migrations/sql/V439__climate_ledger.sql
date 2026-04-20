-- V439__climate_ledger.sql
-- Climate Ledger: append-only signed record table (v3 L2 System of Record).
-- Depends on: V001 (extensions).
--
-- The table mirrors the SQLite schema written by
-- greenlang.climate_ledger.ledger._SQLiteLedgerBackend.  Every row is an
-- immutable audit entry whose chain_hash is computed by the in-process
-- ProvenanceTracker.  Integrity checks run client-side on export.

CREATE TABLE IF NOT EXISTS climate_ledger_entries (
    id            BIGSERIAL   PRIMARY KEY,
    agent_name    TEXT        NOT NULL,
    entity_type   TEXT        NOT NULL,
    entity_id     TEXT        NOT NULL,
    operation     TEXT        NOT NULL,
    content_hash  CHAR(64)    NOT NULL,
    chain_hash    CHAR(64)    NOT NULL,
    metadata      JSONB       NOT NULL DEFAULT '{}'::jsonb,
    tenant_id     VARCHAR(64),
    recorded_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Fast per-entity chain reads (the hot path for verify()).
CREATE INDEX IF NOT EXISTS idx_cle_entity
    ON climate_ledger_entries (entity_type, entity_id, id);

-- Chain-hash lookup (forensics + cross-entity integrity audit).
CREATE INDEX IF NOT EXISTS idx_cle_chain_hash
    ON climate_ledger_entries (chain_hash);

-- Agent timeline (per-agent audit query).
CREATE INDEX IF NOT EXISTS idx_cle_agent_time
    ON climate_ledger_entries (agent_name, recorded_at DESC);

-- Tenant scoping for multi-tenant deployments.
CREATE INDEX IF NOT EXISTS idx_cle_tenant_time
    ON climate_ledger_entries (tenant_id, recorded_at DESC);

-- Enforce append-only semantics at the database level.  The trigger
-- aborts any UPDATE or DELETE so that even a compromised application
-- cannot mutate history without DBA intervention.
CREATE OR REPLACE FUNCTION climate_ledger_no_mutate() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'climate_ledger_entries is append-only (id=%)', OLD.id;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_climate_ledger_no_update ON climate_ledger_entries;
CREATE TRIGGER trg_climate_ledger_no_update
BEFORE UPDATE ON climate_ledger_entries
FOR EACH ROW EXECUTE FUNCTION climate_ledger_no_mutate();

DROP TRIGGER IF EXISTS trg_climate_ledger_no_delete ON climate_ledger_entries;
CREATE TRIGGER trg_climate_ledger_no_delete
BEFORE DELETE ON climate_ledger_entries
FOR EACH ROW EXECUTE FUNCTION climate_ledger_no_mutate();

COMMENT ON TABLE climate_ledger_entries IS
    'Climate Ledger: append-only signed provenance records. '
    'See greenlang.climate_ledger for the product API and docs/REPO_TOUR.md '
    'for the v3 layer map.';
