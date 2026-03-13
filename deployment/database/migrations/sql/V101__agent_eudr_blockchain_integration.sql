-- ============================================================================
-- V101: AGENT-EUDR-013 Blockchain Integration Agent
-- ============================================================================
-- Creates tables for blockchain anchoring records, Merkle tree management,
-- smart contract registry, contract event indexing, chain connections,
-- verification results, access grants, evidence packages, gas cost tracking,
-- batch jobs, and audit trails.
--
-- Tables: 12 (9 regular + 3 hypertables)
-- Hypertables: gl_eudr_bci_anchors, gl_eudr_bci_contract_events,
--              gl_eudr_bci_verification_results
-- Continuous Aggregates: 2 (hourly_anchor_stats + hourly_verification_stats)
-- Retention Policies: 3 (hypertables, 90 days operational window)
-- Indexes: 88
--
-- Dependencies: TimescaleDB extension (V002)
-- Author: GreenLang Platform Team
-- Date: March 2026
-- ============================================================================

BEGIN;

RAISE NOTICE 'V101: Creating AGENT-EUDR-013 Blockchain Integration tables...';

-- ============================================================================
-- 1. gl_eudr_bci_anchors — Blockchain anchoring records (hypertable)
-- ============================================================================
RAISE NOTICE 'V101 [1/12]: Creating gl_eudr_bci_anchors (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_anchors (
    id                      UUID            DEFAULT gen_random_uuid(),
    anchor_id               TEXT            UNIQUE NOT NULL,
    record_id               TEXT            NOT NULL,
    record_hash             TEXT            NOT NULL,
        -- SHA-256 hex digest of the anchored record
    event_type              TEXT            NOT NULL,
        -- 'dds_submission', 'geolocation_proof', 'shipment_transit',
        -- 'risk_assessment', 'compliance_decision', 'supply_chain_event'
    priority                TEXT            NOT NULL DEFAULT 'p1_standard',
        -- 'p0_critical', 'p1_standard', 'p2_batch'
    blockchain_network      TEXT            NOT NULL,
        -- 'polygon', 'ethereum', 'polygon_amoy', 'ethereum_sepolia'
    transaction_hash        TEXT,
    block_number            BIGINT,
    merkle_tree_id          UUID,
    merkle_leaf_index       INTEGER,
    merkle_proof            JSONB,
    status                  TEXT            NOT NULL DEFAULT 'pending',
        -- 'pending', 'submitted', 'confirmed', 'failed', 'retrying'
    gas_used                BIGINT,
    gas_price_wei           BIGINT,
    confirmation_count      INTEGER         DEFAULT 0,
    confirmed_at            TIMESTAMPTZ,
    error_message           TEXT,
    retry_count             INTEGER         DEFAULT 0,
    operator_id             UUID            NOT NULL,
    provenance_hash         TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_bci_anchors',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_anchor_id ON gl_eudr_bci_anchors (anchor_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_record_id ON gl_eudr_bci_anchors (record_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_record_hash ON gl_eudr_bci_anchors (record_hash, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_event_type ON gl_eudr_bci_anchors (event_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_priority ON gl_eudr_bci_anchors (priority, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_network ON gl_eudr_bci_anchors (blockchain_network, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_tx_hash ON gl_eudr_bci_anchors (transaction_hash, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_block ON gl_eudr_bci_anchors (block_number, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_merkle_tree ON gl_eudr_bci_anchors (merkle_tree_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_status ON gl_eudr_bci_anchors (status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_operator ON gl_eudr_bci_anchors (operator_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_confirmed ON gl_eudr_bci_anchors (confirmed_at, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_anc_proof ON gl_eudr_bci_anchors USING GIN (merkle_proof);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 2. gl_eudr_bci_merkle_trees — Merkle tree management
-- ============================================================================
RAISE NOTICE 'V101 [2/12]: Creating gl_eudr_bci_merkle_trees...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_merkle_trees (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tree_id                 TEXT            UNIQUE NOT NULL,
    root_hash               TEXT            NOT NULL,
    leaf_count              INTEGER         NOT NULL,
    tree_depth              INTEGER         NOT NULL,
    hash_algorithm          TEXT            NOT NULL DEFAULT 'sha256',
        -- 'sha256', 'keccak256'
    anchor_id               UUID,
    serialized_tree         JSONB,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_tree_id ON gl_eudr_bci_merkle_trees (tree_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_root_hash ON gl_eudr_bci_merkle_trees (root_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_anchor ON gl_eudr_bci_merkle_trees (anchor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_algorithm ON gl_eudr_bci_merkle_trees (hash_algorithm);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_leaf_count ON gl_eudr_bci_merkle_trees (leaf_count);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_created ON gl_eudr_bci_merkle_trees (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_mt_tree ON gl_eudr_bci_merkle_trees USING GIN (serialized_tree);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 3. gl_eudr_bci_merkle_leaves — Merkle tree leaf entries
-- ============================================================================
RAISE NOTICE 'V101 [3/12]: Creating gl_eudr_bci_merkle_leaves...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_merkle_leaves (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tree_id                 UUID            NOT NULL REFERENCES gl_eudr_bci_merkle_trees(id),
    leaf_index              INTEGER         NOT NULL,
    record_hash             TEXT            NOT NULL,
    record_id               TEXT            NOT NULL,
    record_type             TEXT            NOT NULL,
        -- 'dds', 'geolocation', 'shipment', 'risk_assessment', 'commodity'
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    UNIQUE(tree_id, leaf_index)
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ml_tree ON gl_eudr_bci_merkle_leaves (tree_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ml_record_hash ON gl_eudr_bci_merkle_leaves (record_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ml_record_id ON gl_eudr_bci_merkle_leaves (record_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ml_record_type ON gl_eudr_bci_merkle_leaves (record_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ml_tree_index ON gl_eudr_bci_merkle_leaves (tree_id, leaf_index);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ml_created ON gl_eudr_bci_merkle_leaves (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 4. gl_eudr_bci_smart_contracts — Smart contract registry
-- ============================================================================
RAISE NOTICE 'V101 [4/12]: Creating gl_eudr_bci_smart_contracts...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_smart_contracts (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    contract_id             TEXT            UNIQUE NOT NULL,
    contract_type           TEXT            NOT NULL,
        -- 'eudr_anchor', 'merkle_verifier', 'access_control', 'evidence_registry'
    blockchain_network      TEXT            NOT NULL,
    contract_address        TEXT,
    deployer_address        TEXT,
    deploy_tx_hash          TEXT,
    abi_json                JSONB           NOT NULL,
    bytecode_hash           TEXT,
    version                 TEXT            NOT NULL,
    status                  TEXT            NOT NULL DEFAULT 'deploying',
        -- 'deploying', 'deployed', 'paused', 'deprecated', 'failed'
    deployed_at             TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_contract_id ON gl_eudr_bci_smart_contracts (contract_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_type ON gl_eudr_bci_smart_contracts (contract_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_network ON gl_eudr_bci_smart_contracts (blockchain_network);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_address ON gl_eudr_bci_smart_contracts (contract_address);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_deployer ON gl_eudr_bci_smart_contracts (deployer_address);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_deploy_tx ON gl_eudr_bci_smart_contracts (deploy_tx_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_status ON gl_eudr_bci_smart_contracts (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_version ON gl_eudr_bci_smart_contracts (version);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_deployed ON gl_eudr_bci_smart_contracts (deployed_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_sc_abi ON gl_eudr_bci_smart_contracts USING GIN (abi_json);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 5. gl_eudr_bci_contract_events — Smart contract event indexing (hypertable)
-- ============================================================================
RAISE NOTICE 'V101 [5/12]: Creating gl_eudr_bci_contract_events (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_contract_events (
    id                      UUID            DEFAULT gen_random_uuid(),
    event_id                TEXT            UNIQUE NOT NULL,
    contract_id             UUID,
    event_type              TEXT            NOT NULL,
        -- 'RecordAnchored', 'MerkleRootStored', 'AccessGranted',
        -- 'AccessRevoked', 'ContractPaused', 'ContractUpgraded'
    transaction_hash        TEXT            NOT NULL,
    block_number            BIGINT          NOT NULL,
    log_index               INTEGER,
    event_data              JSONB,
    indexed_fields          JSONB,
    blockchain_network      TEXT            NOT NULL,
    event_timestamp         TIMESTAMPTZ     NOT NULL,
    indexed_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, event_timestamp)
);

SELECT create_hypertable(
    'gl_eudr_bci_contract_events',
    'event_timestamp',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_event_id ON gl_eudr_bci_contract_events (event_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_contract ON gl_eudr_bci_contract_events (contract_id, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_event_type ON gl_eudr_bci_contract_events (event_type, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_tx_hash ON gl_eudr_bci_contract_events (transaction_hash, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_block ON gl_eudr_bci_contract_events (block_number, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_network ON gl_eudr_bci_contract_events (blockchain_network, event_timestamp DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_event_data ON gl_eudr_bci_contract_events USING GIN (event_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ce_indexed_fields ON gl_eudr_bci_contract_events USING GIN (indexed_fields);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 6. gl_eudr_bci_chain_connections — Blockchain network connections
-- ============================================================================
RAISE NOTICE 'V101 [6/12]: Creating gl_eudr_bci_chain_connections...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_chain_connections (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id                TEXT            UNIQUE NOT NULL,
    blockchain_network      TEXT            NOT NULL,
        -- 'polygon', 'ethereum', 'polygon_amoy', 'ethereum_sepolia'
    rpc_endpoint            TEXT            NOT NULL,
    chain_id_numeric        INTEGER,
    is_primary              BOOLEAN         DEFAULT FALSE,
    status                  TEXT            NOT NULL DEFAULT 'disconnected',
        -- 'connected', 'disconnected', 'degraded', 'syncing'
    last_block_number       BIGINT,
    last_block_timestamp    TIMESTAMPTZ,
    connection_metadata     JSONB,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_chain_id ON gl_eudr_bci_chain_connections (chain_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_network ON gl_eudr_bci_chain_connections (blockchain_network);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_primary ON gl_eudr_bci_chain_connections (is_primary);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_status ON gl_eudr_bci_chain_connections (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_last_block ON gl_eudr_bci_chain_connections (last_block_number);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_chain_numeric ON gl_eudr_bci_chain_connections (chain_id_numeric);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_cc_metadata ON gl_eudr_bci_chain_connections USING GIN (connection_metadata);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 7. gl_eudr_bci_verification_results — On-chain verification results (hypertable)
-- ============================================================================
RAISE NOTICE 'V101 [7/12]: Creating gl_eudr_bci_verification_results (hypertable)...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_verification_results (
    id                      UUID            DEFAULT gen_random_uuid(),
    verification_id         TEXT            UNIQUE NOT NULL,
    record_id               TEXT            NOT NULL,
    record_hash             TEXT            NOT NULL,
    anchor_id               UUID,
    on_chain_hash           TEXT,
    blockchain_network      TEXT            NOT NULL,
    transaction_hash        TEXT,
    block_number            BIGINT,
    status                  TEXT            NOT NULL,
        -- 'verified', 'tampered', 'not_found', 'error', 'pending'
    merkle_proof_valid      BOOLEAN,
    temporal_valid          BOOLEAN,
    verified_at_block       BIGINT,
    processing_time_ms      FLOAT,
    operator_id             UUID            NOT NULL,
    provenance_hash         TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    PRIMARY KEY (id, created_at)
);

SELECT create_hypertable(
    'gl_eudr_bci_verification_results',
    'created_at',
    chunk_time_interval => INTERVAL '1 month',
    if_not_exists => TRUE
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_verification_id ON gl_eudr_bci_verification_results (verification_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_record_id ON gl_eudr_bci_verification_results (record_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_record_hash ON gl_eudr_bci_verification_results (record_hash, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_anchor ON gl_eudr_bci_verification_results (anchor_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_network ON gl_eudr_bci_verification_results (blockchain_network, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_tx_hash ON gl_eudr_bci_verification_results (transaction_hash, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_block ON gl_eudr_bci_verification_results (block_number, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_status ON gl_eudr_bci_verification_results (status, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_merkle_valid ON gl_eudr_bci_verification_results (merkle_proof_valid, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_temporal_valid ON gl_eudr_bci_verification_results (temporal_valid, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_vr_operator ON gl_eudr_bci_verification_results (operator_id, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 8. gl_eudr_bci_access_grants — On-chain access control grants
-- ============================================================================
RAISE NOTICE 'V101 [8/12]: Creating gl_eudr_bci_access_grants...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_access_grants (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    grant_id                TEXT            UNIQUE NOT NULL,
    record_id               TEXT,
    record_type             TEXT,
        -- 'dds', 'geolocation', 'shipment', 'commodity', 'all'
    grantor_id              UUID            NOT NULL,
    grantee_id              UUID            NOT NULL,
    grantee_type            TEXT            NOT NULL,
        -- 'operator', 'competent_authority', 'auditor', 'customs'
    access_level            TEXT            NOT NULL,
        -- 'read', 'verify', 'full', 'admin'
    blockchain_network      TEXT,
    on_chain_tx_hash        TEXT,
    status                  TEXT            NOT NULL DEFAULT 'active',
        -- 'active', 'revoked', 'expired', 'pending'
    expires_at              TIMESTAMPTZ,
    revoked_at              TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_grant_id ON gl_eudr_bci_access_grants (grant_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_record_id ON gl_eudr_bci_access_grants (record_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_record_type ON gl_eudr_bci_access_grants (record_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_grantor ON gl_eudr_bci_access_grants (grantor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_grantee ON gl_eudr_bci_access_grants (grantee_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_grantee_type ON gl_eudr_bci_access_grants (grantee_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_access_level ON gl_eudr_bci_access_grants (access_level);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_status ON gl_eudr_bci_access_grants (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_expires ON gl_eudr_bci_access_grants (expires_at);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ag_tx_hash ON gl_eudr_bci_access_grants (on_chain_tx_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 9. gl_eudr_bci_evidence_packages — Blockchain evidence packages
-- ============================================================================
RAISE NOTICE 'V101 [9/12]: Creating gl_eudr_bci_evidence_packages...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_evidence_packages (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    package_id              TEXT            UNIQUE NOT NULL,
    dds_id                  TEXT,
    operator_id             UUID            NOT NULL,
    format                  TEXT            NOT NULL,
        -- 'json', 'pdf', 'xml', 'zip'
    package_data            JSONB,
    anchor_references       JSONB,
    merkle_proofs           JSONB,
    verification_summary    JSONB,
    package_hash            TEXT,
    signature               TEXT,
    signed_at               TIMESTAMPTZ,
    retention_until         TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_package_id ON gl_eudr_bci_evidence_packages (package_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_dds ON gl_eudr_bci_evidence_packages (dds_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_operator ON gl_eudr_bci_evidence_packages (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_format ON gl_eudr_bci_evidence_packages (format);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_hash ON gl_eudr_bci_evidence_packages (package_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_retention ON gl_eudr_bci_evidence_packages (retention_until);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_created ON gl_eudr_bci_evidence_packages (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_data ON gl_eudr_bci_evidence_packages USING GIN (package_data);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_anchors ON gl_eudr_bci_evidence_packages USING GIN (anchor_references);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_ep_proofs ON gl_eudr_bci_evidence_packages USING GIN (merkle_proofs);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 10. gl_eudr_bci_gas_costs — Gas cost tracking per anchor
-- ============================================================================
RAISE NOTICE 'V101 [10/12]: Creating gl_eudr_bci_gas_costs...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_gas_costs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    anchor_id               UUID,
    blockchain_network      TEXT            NOT NULL,
    transaction_hash        TEXT,
    gas_used                BIGINT,
    gas_price_wei           BIGINT,
    total_cost_wei          BIGINT,
    total_cost_native       NUMERIC(28,18),
    native_token            TEXT,
        -- 'MATIC', 'ETH'
    usd_equivalent          NUMERIC(12,4),
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_anchor ON gl_eudr_bci_gas_costs (anchor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_network ON gl_eudr_bci_gas_costs (blockchain_network);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_tx_hash ON gl_eudr_bci_gas_costs (transaction_hash);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_gas_used ON gl_eudr_bci_gas_costs (gas_used);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_usd ON gl_eudr_bci_gas_costs (usd_equivalent);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_token ON gl_eudr_bci_gas_costs (native_token);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_gc_created ON gl_eudr_bci_gas_costs (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 11. gl_eudr_bci_batch_jobs — Batch anchoring job tracking
-- ============================================================================
RAISE NOTICE 'V101 [11/12]: Creating gl_eudr_bci_batch_jobs...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_batch_jobs (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id                  TEXT            UNIQUE NOT NULL,
    job_type                TEXT            NOT NULL,
        -- 'batch_anchor', 'batch_verify', 'merkle_build', 'evidence_generate'
    status                  TEXT            NOT NULL DEFAULT 'queued',
        -- 'queued', 'processing', 'completed', 'failed', 'cancelled'
    total_records           INTEGER         NOT NULL DEFAULT 0,
    processed_records       INTEGER         DEFAULT 0,
    failed_records          INTEGER         DEFAULT 0,
    merkle_tree_id          UUID            REFERENCES gl_eudr_bci_merkle_trees(id),
    anchor_id               UUID,
    error_details           JSONB,
    operator_id             UUID            NOT NULL,
    started_at              TIMESTAMPTZ,
    completed_at            TIMESTAMPTZ,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_job_id ON gl_eudr_bci_batch_jobs (job_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_job_type ON gl_eudr_bci_batch_jobs (job_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_status ON gl_eudr_bci_batch_jobs (status);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_merkle_tree ON gl_eudr_bci_batch_jobs (merkle_tree_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_anchor ON gl_eudr_bci_batch_jobs (anchor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_operator ON gl_eudr_bci_batch_jobs (operator_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_created ON gl_eudr_bci_batch_jobs (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_bj_errors ON gl_eudr_bci_batch_jobs USING GIN (error_details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- 12. gl_eudr_bci_audit_log — Immutable audit trail
-- ============================================================================
RAISE NOTICE 'V101 [12/12]: Creating gl_eudr_bci_audit_log...';

CREATE TABLE IF NOT EXISTS gl_eudr_bci_audit_log (
    id                      UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    log_id                  TEXT            UNIQUE NOT NULL,
    action                  TEXT            NOT NULL,
        -- 'anchored', 'verified', 'access_granted', 'access_revoked',
        -- 'contract_deployed', 'evidence_generated', 'batch_completed',
        -- 'chain_connected', 'chain_disconnected', 'merkle_built'
    entity_type             TEXT            NOT NULL,
        -- 'anchor', 'merkle_tree', 'merkle_leaf', 'smart_contract',
        -- 'contract_event', 'chain_connection', 'verification',
        -- 'access_grant', 'evidence_package', 'gas_cost', 'batch_job'
    entity_id               TEXT            NOT NULL,
    actor_id                UUID            NOT NULL,
    details                 JSONB,
    ip_address              INET,
    provenance_hash         TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_log_id ON gl_eudr_bci_audit_log (log_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_action ON gl_eudr_bci_audit_log (action);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_entity_type ON gl_eudr_bci_audit_log (entity_type);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_entity_id ON gl_eudr_bci_audit_log (entity_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_actor ON gl_eudr_bci_audit_log (actor_id);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_created ON gl_eudr_bci_audit_log (created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_details ON gl_eudr_bci_audit_log USING GIN (details);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_action_created ON gl_eudr_bci_audit_log (action, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;
DO $$ BEGIN
    CREATE INDEX idx_eudr_bci_al_entity_type_created ON gl_eudr_bci_audit_log (entity_type, created_at DESC);
EXCEPTION WHEN duplicate_table THEN NULL; END $$;


-- ============================================================================
-- Continuous Aggregates
-- ============================================================================
RAISE NOTICE 'V101: Creating continuous aggregates...';

-- 1. Hourly anchor statistics by network, event_type, and status
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_bci_hourly_anchor_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at)       AS bucket,
    blockchain_network,
    event_type,
    status,
    COUNT(*)                                AS anchor_count,
    COUNT(*) FILTER (WHERE status = 'confirmed')   AS confirmed_count,
    COUNT(*) FILTER (WHERE status = 'failed')      AS failed_count,
    COUNT(*) FILTER (WHERE status = 'pending')     AS pending_count,
    AVG(gas_used)                           AS avg_gas_used,
    SUM(gas_used)                           AS total_gas_used
FROM gl_eudr_bci_anchors
GROUP BY bucket, blockchain_network, event_type, status
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_bci_hourly_anchor_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);

-- 2. Hourly verification statistics by network and status
CREATE MATERIALIZED VIEW IF NOT EXISTS gl_eudr_bci_hourly_verification_stats
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', created_at)       AS bucket,
    blockchain_network,
    status,
    COUNT(*)                                AS verification_count,
    COUNT(*) FILTER (WHERE status = 'verified')    AS verified_count,
    COUNT(*) FILTER (WHERE status = 'tampered')    AS tampered_count,
    COUNT(*) FILTER (WHERE status = 'not_found')   AS not_found_count,
    COUNT(*) FILTER (WHERE merkle_proof_valid = TRUE)  AS merkle_valid_count,
    COUNT(*) FILTER (WHERE temporal_valid = TRUE)      AS temporal_valid_count,
    AVG(processing_time_ms)                 AS avg_processing_time_ms,
    MAX(processing_time_ms)                 AS max_processing_time_ms
FROM gl_eudr_bci_verification_results
GROUP BY bucket, blockchain_network, status
WITH NO DATA;

SELECT add_continuous_aggregate_policy('gl_eudr_bci_hourly_verification_stats',
    start_offset    => INTERVAL '3 days',
    end_offset      => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists   => TRUE
);


-- ============================================================================
-- Retention Policies (90 days operational window)
-- ============================================================================
RAISE NOTICE 'V101: Adding retention policies (90 days operational window)...';

SELECT add_retention_policy('gl_eudr_bci_anchors',
    INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_bci_contract_events',
    INTERVAL '90 days', if_not_exists => TRUE);
SELECT add_retention_policy('gl_eudr_bci_verification_results',
    INTERVAL '90 days', if_not_exists => TRUE);


-- ============================================================================
-- Comments
-- ============================================================================
RAISE NOTICE 'V101: Adding table comments...';

COMMENT ON TABLE gl_eudr_bci_anchors IS 'AGENT-EUDR-013: Blockchain anchoring records with SHA-256 record hashes, transaction tracking, Merkle tree linkage, and confirmation status (hypertable)';
COMMENT ON TABLE gl_eudr_bci_merkle_trees IS 'AGENT-EUDR-013: Merkle tree management with root hash, leaf count, depth, and serialized tree structure for batch anchoring';
COMMENT ON TABLE gl_eudr_bci_merkle_leaves IS 'AGENT-EUDR-013: Individual Merkle tree leaf entries linking record hashes to tree positions for proof generation';
COMMENT ON TABLE gl_eudr_bci_smart_contracts IS 'AGENT-EUDR-013: Smart contract registry tracking deployed EUDR anchor, verifier, and access control contracts across networks';
COMMENT ON TABLE gl_eudr_bci_contract_events IS 'AGENT-EUDR-013: Indexed smart contract events from on-chain RecordAnchored, MerkleRootStored, and AccessGranted emissions (hypertable)';
COMMENT ON TABLE gl_eudr_bci_chain_connections IS 'AGENT-EUDR-013: Blockchain network connection management with RPC endpoints, sync status, and health monitoring';
COMMENT ON TABLE gl_eudr_bci_verification_results IS 'AGENT-EUDR-013: On-chain verification results with Merkle proof validation, temporal checks, and tamper detection status (hypertable)';
COMMENT ON TABLE gl_eudr_bci_access_grants IS 'AGENT-EUDR-013: On-chain access control grants for operators, competent authorities, auditors, and customs with expiration tracking';
COMMENT ON TABLE gl_eudr_bci_evidence_packages IS 'AGENT-EUDR-013: Assembled blockchain evidence packages with anchor references, Merkle proofs, and verification summaries for EUDR DDS';
COMMENT ON TABLE gl_eudr_bci_gas_costs IS 'AGENT-EUDR-013: Gas cost tracking per anchor transaction with native token and USD equivalent calculations';
COMMENT ON TABLE gl_eudr_bci_batch_jobs IS 'AGENT-EUDR-013: Batch anchoring and verification job tracking with progress counters, Merkle tree linkage, and error details';
COMMENT ON TABLE gl_eudr_bci_audit_log IS 'AGENT-EUDR-013: Immutable audit trail for all blockchain integration operations with actor tracking, IP logging, and provenance hashing';

RAISE NOTICE 'V101: AGENT-EUDR-013 Blockchain Integration migration complete.';

COMMIT;
