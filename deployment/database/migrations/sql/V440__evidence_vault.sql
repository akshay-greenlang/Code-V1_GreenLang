-- V440__evidence_vault.sql
-- Evidence Vault: append-only evidence records + raw-source attachments.
-- v3 L2 System of Record -- companion to V439 (Climate Ledger).
-- Depends on: V001 (extensions), V439 (climate_ledger_entries).

CREATE TABLE IF NOT EXISTS evidence_records (
    evidence_id    TEXT        PRIMARY KEY,
    vault_id       TEXT        NOT NULL,
    case_id        TEXT,
    evidence_type  TEXT        NOT NULL,
    source         TEXT        NOT NULL,
    data           JSONB       NOT NULL DEFAULT '{}'::jsonb,
    metadata       JSONB       NOT NULL DEFAULT '{}'::jsonb,
    content_hash   CHAR(64)    NOT NULL,
    tenant_id      VARCHAR(64),
    collected_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ev_vault_time
    ON evidence_records (vault_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_ev_case_time
    ON evidence_records (case_id, collected_at DESC);
CREATE INDEX IF NOT EXISTS idx_ev_type
    ON evidence_records (evidence_type);
CREATE INDEX IF NOT EXISTS idx_ev_hash
    ON evidence_records (content_hash);
CREATE INDEX IF NOT EXISTS idx_ev_tenant
    ON evidence_records (tenant_id, collected_at DESC);

CREATE TABLE IF NOT EXISTS evidence_attachments (
    id             BIGSERIAL   PRIMARY KEY,
    evidence_id    TEXT        NOT NULL REFERENCES evidence_records(evidence_id) ON DELETE RESTRICT,
    filename       TEXT        NOT NULL,
    content_hash   CHAR(64)    NOT NULL,
    content_bytes  BYTEA       NOT NULL,
    attached_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_att_evidence
    ON evidence_attachments (evidence_id, attached_at DESC);
CREATE INDEX IF NOT EXISTS idx_att_hash
    ON evidence_attachments (content_hash);

CREATE OR REPLACE FUNCTION evidence_vault_no_mutate() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'evidence_records is append-only (evidence_id=%)', OLD.evidence_id;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_evidence_no_update ON evidence_records;
CREATE TRIGGER trg_evidence_no_update
BEFORE UPDATE ON evidence_records
FOR EACH ROW EXECUTE FUNCTION evidence_vault_no_mutate();

DROP TRIGGER IF EXISTS trg_evidence_no_delete ON evidence_records;
CREATE TRIGGER trg_evidence_no_delete
BEFORE DELETE ON evidence_records
FOR EACH ROW EXECUTE FUNCTION evidence_vault_no_mutate();

COMMENT ON TABLE evidence_records IS
    'Evidence Vault: append-only evidence records for CBAM, CSRD, EUDR, '
    'ISO 14064, CDP. Bundles (ZIP) are generated client-side via '
    'greenlang.evidence_vault.EvidenceVault.bundle().';
COMMENT ON TABLE evidence_attachments IS
    'Raw-source attachments (parser logs, PDFs, XML) linked to an evidence record.';
