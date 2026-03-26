-- =============================================================================
-- V339: PACK-042 Scope 3 Starter Pack - Double-Counting Prevention
-- =============================================================================
-- Pack:         PACK-042 (Scope 3 Starter Pack)
-- Migration:    004 of 010
-- Author:       GreenLang Platform Team
-- Date:         March 2026
--
-- Creates double-counting prevention and cross-category reconciliation tables.
-- Implements the GHG Protocol guidance on avoiding double-counting between
-- Scope 3 categories (e.g., Cat 1 vs Cat 4, Cat 3 vs Cat 1, Cat 6 vs Cat 7).
-- Stores predefined overlap rules, detection results, resolution decisions,
-- and a reconciliation summary showing the net effect on total Scope 3.
--
-- Tables (4):
--   1. ghg_accounting_scope3.overlap_rules
--   2. ghg_accounting_scope3.overlap_detections
--   3. ghg_accounting_scope3.overlap_resolutions
--   4. ghg_accounting_scope3.reconciliation_summary
--
-- Seed Data:
--   - 12 overlap rules with descriptions and detection methods
--
-- Also includes: indexes, RLS, comments.
-- Previous: V338__pack042_category_results.sql
-- =============================================================================

SET search_path TO ghg_accounting_scope3, public;

-- =============================================================================
-- Table 1: ghg_accounting_scope3.overlap_rules
-- =============================================================================
-- Predefined rules for detecting potential double-counting between Scope 3
-- categories. Each rule identifies a pair of categories that may overlap,
-- describes the nature of the overlap, and specifies the detection and
-- resolution methods. Based on GHG Protocol Technical Guidance.

CREATE TABLE ghg_accounting_scope3.overlap_rules (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    rule_id                     VARCHAR(20)     NOT NULL,
    -- Category pair
    category_a                  ghg_accounting_scope3.scope3_category_type NOT NULL,
    category_b                  ghg_accounting_scope3.scope3_category_type NOT NULL,
    -- Rule description
    rule_name                   VARCHAR(200)    NOT NULL,
    description                 TEXT            NOT NULL,
    overlap_type                VARCHAR(30)     NOT NULL DEFAULT 'BOUNDARY',
    -- Detection
    detection_method            VARCHAR(50)     NOT NULL,
    detection_description       TEXT,
    detection_query_hint        TEXT,
    -- Resolution
    resolution_method           VARCHAR(50)     NOT NULL,
    resolution_description      TEXT,
    default_priority_category   ghg_accounting_scope3.scope3_category_type,
    -- Severity
    risk_level                  VARCHAR(20)     NOT NULL DEFAULT 'MEDIUM',
    typical_overlap_pct         DECIMAL(5,2),
    -- Reference
    ghg_protocol_reference      VARCHAR(200),
    -- Metadata
    is_active                   BOOLEAN         NOT NULL DEFAULT true,
    notes                       TEXT,
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_or_overlap_type CHECK (
        overlap_type IN (
            'BOUNDARY', 'METHODOLOGY', 'DATA_SOURCE',
            'ALLOCATION', 'TEMPORAL', 'DEFINITIONAL'
        )
    ),
    CONSTRAINT chk_p042_or_detection CHECK (
        detection_method IN (
            'VENDOR_MATCH', 'SPEND_OVERLAP', 'ACTIVITY_MATCH',
            'EF_SCOPE_CHECK', 'BOUNDARY_ANALYSIS', 'RULE_BASED',
            'TRANSACTION_DEDUP', 'MANUAL_REVIEW'
        )
    ),
    CONSTRAINT chk_p042_or_resolution CHECK (
        resolution_method IN (
            'ALLOCATE_TO_PRIMARY', 'SUBTRACT_OVERLAP', 'SPLIT_PRO_RATA',
            'EXCLUDE_SECONDARY', 'APPLY_HIERARCHY', 'MANUAL_ADJUSTMENT',
            'FLAG_ONLY', 'ZERO_SECONDARY'
        )
    ),
    CONSTRAINT chk_p042_or_risk CHECK (
        risk_level IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')
    ),
    CONSTRAINT chk_p042_or_typical_pct CHECK (
        typical_overlap_pct IS NULL OR (typical_overlap_pct >= 0 AND typical_overlap_pct <= 100)
    ),
    CONSTRAINT chk_p042_or_diff_categories CHECK (
        category_a != category_b
    ),
    CONSTRAINT uq_p042_or_rule_id UNIQUE (rule_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_or_rule_id            ON ghg_accounting_scope3.overlap_rules(rule_id);
CREATE INDEX idx_p042_or_cat_a              ON ghg_accounting_scope3.overlap_rules(category_a);
CREATE INDEX idx_p042_or_cat_b              ON ghg_accounting_scope3.overlap_rules(category_b);
CREATE INDEX idx_p042_or_overlap_type       ON ghg_accounting_scope3.overlap_rules(overlap_type);
CREATE INDEX idx_p042_or_risk               ON ghg_accounting_scope3.overlap_rules(risk_level);
CREATE INDEX idx_p042_or_active             ON ghg_accounting_scope3.overlap_rules(is_active) WHERE is_active = true;

-- Composite: both categories for pair lookup
CREATE INDEX idx_p042_or_pair               ON ghg_accounting_scope3.overlap_rules(category_a, category_b);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_or_updated
    BEFORE UPDATE ON ghg_accounting_scope3.overlap_rules
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 2: ghg_accounting_scope3.overlap_detections
-- =============================================================================
-- Detection results from running overlap rules against an inventory. Each
-- record identifies a specific overlap instance between two categories,
-- quantifies the overlap amount, and records the confidence level of the
-- detection. Detections are reviewed and resolved in the resolutions table.

CREATE TABLE ghg_accounting_scope3.overlap_detections (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    rule_id                     UUID            NOT NULL REFERENCES ghg_accounting_scope3.overlap_rules(id) ON DELETE RESTRICT,
    -- Detection details
    detection_timestamp         TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    category_a_amount           DECIMAL(15,3)   NOT NULL,
    category_b_amount           DECIMAL(15,3)   NOT NULL,
    overlap_amount              DECIMAL(15,3)   NOT NULL,
    overlap_pct_of_total        DECIMAL(5,2),
    -- Confidence
    confidence                  DECIMAL(3,2)    NOT NULL DEFAULT 0.50,
    -- Evidence
    matching_records            INTEGER         DEFAULT 0,
    matching_vendors            TEXT[],
    matching_transactions       UUID[],
    evidence_detail             JSONB           DEFAULT '{}',
    -- Status
    status                      VARCHAR(30)     NOT NULL DEFAULT 'DETECTED',
    reviewed_by                 VARCHAR(255),
    reviewed_at                 TIMESTAMPTZ,
    -- Notes
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_od_cat_a CHECK (
        category_a_amount >= 0
    ),
    CONSTRAINT chk_p042_od_cat_b CHECK (
        category_b_amount >= 0
    ),
    CONSTRAINT chk_p042_od_overlap CHECK (
        overlap_amount >= 0
    ),
    CONSTRAINT chk_p042_od_overlap_pct CHECK (
        overlap_pct_of_total IS NULL OR (overlap_pct_of_total >= 0 AND overlap_pct_of_total <= 100)
    ),
    CONSTRAINT chk_p042_od_confidence CHECK (
        confidence >= 0 AND confidence <= 1
    ),
    CONSTRAINT chk_p042_od_records CHECK (
        matching_records IS NULL OR matching_records >= 0
    ),
    CONSTRAINT chk_p042_od_status CHECK (
        status IN (
            'DETECTED', 'REVIEWING', 'CONFIRMED', 'DISMISSED',
            'RESOLVED', 'FALSE_POSITIVE'
        )
    )
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_od_tenant             ON ghg_accounting_scope3.overlap_detections(tenant_id);
CREATE INDEX idx_p042_od_inventory          ON ghg_accounting_scope3.overlap_detections(inventory_id);
CREATE INDEX idx_p042_od_rule               ON ghg_accounting_scope3.overlap_detections(rule_id);
CREATE INDEX idx_p042_od_status             ON ghg_accounting_scope3.overlap_detections(status);
CREATE INDEX idx_p042_od_confidence         ON ghg_accounting_scope3.overlap_detections(confidence DESC);
CREATE INDEX idx_p042_od_overlap_amount     ON ghg_accounting_scope3.overlap_detections(overlap_amount DESC);
CREATE INDEX idx_p042_od_detected           ON ghg_accounting_scope3.overlap_detections(detection_timestamp DESC);
CREATE INDEX idx_p042_od_created            ON ghg_accounting_scope3.overlap_detections(created_at DESC);
CREATE INDEX idx_p042_od_evidence           ON ghg_accounting_scope3.overlap_detections USING GIN(evidence_detail);

-- Composite: inventory + unresolved for review queue
CREATE INDEX idx_p042_od_inv_unresolved     ON ghg_accounting_scope3.overlap_detections(inventory_id, overlap_amount DESC)
    WHERE status IN ('DETECTED', 'REVIEWING', 'CONFIRMED');

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_od_updated
    BEFORE UPDATE ON ghg_accounting_scope3.overlap_detections
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 3: ghg_accounting_scope3.overlap_resolutions
-- =============================================================================
-- Resolution decisions for detected overlaps. Records how each overlap was
-- resolved: which category was adjusted, the original and adjusted amounts,
-- and the justification. Maintains full audit trail for verification.

CREATE TABLE ghg_accounting_scope3.overlap_resolutions (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    detection_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.overlap_detections(id) ON DELETE CASCADE,
    -- Resolution details
    resolution_method           VARCHAR(50)     NOT NULL,
    adjusted_category           ghg_accounting_scope3.scope3_category_type NOT NULL,
    original_amount             DECIMAL(15,3)   NOT NULL,
    adjusted_amount             DECIMAL(15,3)   NOT NULL,
    adjustment_tco2e            DECIMAL(15,3)   GENERATED ALWAYS AS (original_amount - adjusted_amount) STORED,
    -- Justification
    justification               TEXT            NOT NULL,
    supporting_evidence         JSONB           DEFAULT '{}',
    -- Approval
    resolved_by                 VARCHAR(255)    NOT NULL,
    resolved_at                 TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    approved_by                 VARCHAR(255),
    approved_at                 TIMESTAMPTZ,
    -- Metadata
    notes                       TEXT,
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_ores_method CHECK (
        resolution_method IN (
            'ALLOCATE_TO_PRIMARY', 'SUBTRACT_OVERLAP', 'SPLIT_PRO_RATA',
            'EXCLUDE_SECONDARY', 'APPLY_HIERARCHY', 'MANUAL_ADJUSTMENT',
            'FLAG_ONLY', 'ZERO_SECONDARY'
        )
    ),
    CONSTRAINT chk_p042_ores_original CHECK (
        original_amount >= 0
    ),
    CONSTRAINT chk_p042_ores_adjusted CHECK (
        adjusted_amount >= 0
    ),
    CONSTRAINT chk_p042_ores_adjusted_le CHECK (
        adjusted_amount <= original_amount
    ),
    CONSTRAINT uq_p042_ores_detection UNIQUE (detection_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_ores_tenant           ON ghg_accounting_scope3.overlap_resolutions(tenant_id);
CREATE INDEX idx_p042_ores_detection        ON ghg_accounting_scope3.overlap_resolutions(detection_id);
CREATE INDEX idx_p042_ores_method           ON ghg_accounting_scope3.overlap_resolutions(resolution_method);
CREATE INDEX idx_p042_ores_category         ON ghg_accounting_scope3.overlap_resolutions(adjusted_category);
CREATE INDEX idx_p042_ores_resolved_at      ON ghg_accounting_scope3.overlap_resolutions(resolved_at DESC);
CREATE INDEX idx_p042_ores_created          ON ghg_accounting_scope3.overlap_resolutions(created_at DESC);
CREATE INDEX idx_p042_ores_evidence         ON ghg_accounting_scope3.overlap_resolutions USING GIN(supporting_evidence);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_ores_updated
    BEFORE UPDATE ON ghg_accounting_scope3.overlap_resolutions
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- =============================================================================
-- Table 4: ghg_accounting_scope3.reconciliation_summary
-- =============================================================================
-- Final reconciled Scope 3 totals after all double-counting adjustments.
-- Shows the before/after totals, total adjustment, and how many overlap
-- rules were triggered. Provides a single-row summary per inventory for
-- dashboard and reporting purposes.

CREATE TABLE ghg_accounting_scope3.reconciliation_summary (
    id                          UUID            PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id                   UUID            NOT NULL,
    inventory_id                UUID            NOT NULL REFERENCES ghg_accounting_scope3.scope3_inventories(id) ON DELETE CASCADE,
    -- Totals
    total_before_adjustment     DECIMAL(15,3)   NOT NULL,
    total_adjustment            DECIMAL(15,3)   NOT NULL DEFAULT 0,
    total_after_adjustment      DECIMAL(15,3)   NOT NULL,
    adjustment_pct              DECIMAL(5,2)    GENERATED ALWAYS AS (
        CASE WHEN total_before_adjustment > 0
            THEN ROUND(((total_adjustment / total_before_adjustment) * 100)::NUMERIC, 2)
            ELSE 0
        END
    ) STORED,
    -- Rule summary
    rules_evaluated             INTEGER         NOT NULL DEFAULT 0,
    rules_triggered             INTEGER         NOT NULL DEFAULT 0,
    detections_count            INTEGER         NOT NULL DEFAULT 0,
    resolutions_count           INTEGER         NOT NULL DEFAULT 0,
    false_positives_count       INTEGER         DEFAULT 0,
    -- Category adjustments (JSONB with per-category detail)
    category_adjustments        JSONB           DEFAULT '{}',
    -- Timing
    reconciliation_timestamp    TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    reconciliation_duration_ms  INTEGER,
    -- Metadata
    notes                       TEXT,
    metadata                    JSONB           DEFAULT '{}',
    provenance_hash             VARCHAR(64),
    created_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    -- Constraints
    CONSTRAINT chk_p042_rs_before CHECK (
        total_before_adjustment >= 0
    ),
    CONSTRAINT chk_p042_rs_adjustment CHECK (
        total_adjustment >= 0
    ),
    CONSTRAINT chk_p042_rs_after CHECK (
        total_after_adjustment >= 0
    ),
    CONSTRAINT chk_p042_rs_after_le_before CHECK (
        total_after_adjustment <= total_before_adjustment
    ),
    CONSTRAINT chk_p042_rs_rules_eval CHECK (
        rules_evaluated >= 0
    ),
    CONSTRAINT chk_p042_rs_rules_trig CHECK (
        rules_triggered >= 0 AND rules_triggered <= rules_evaluated
    ),
    CONSTRAINT chk_p042_rs_detections CHECK (
        detections_count >= 0
    ),
    CONSTRAINT chk_p042_rs_resolutions CHECK (
        resolutions_count >= 0 AND resolutions_count <= detections_count
    ),
    CONSTRAINT chk_p042_rs_false_pos CHECK (
        false_positives_count IS NULL OR false_positives_count >= 0
    ),
    CONSTRAINT chk_p042_rs_duration CHECK (
        reconciliation_duration_ms IS NULL OR reconciliation_duration_ms >= 0
    ),
    CONSTRAINT uq_p042_rs_inventory UNIQUE (inventory_id)
);

-- ---------------------------------------------------------------------------
-- Indexes
-- ---------------------------------------------------------------------------
CREATE INDEX idx_p042_rs_tenant             ON ghg_accounting_scope3.reconciliation_summary(tenant_id);
CREATE INDEX idx_p042_rs_inventory          ON ghg_accounting_scope3.reconciliation_summary(inventory_id);
CREATE INDEX idx_p042_rs_timestamp          ON ghg_accounting_scope3.reconciliation_summary(reconciliation_timestamp DESC);
CREATE INDEX idx_p042_rs_adjustment         ON ghg_accounting_scope3.reconciliation_summary(total_adjustment DESC);
CREATE INDEX idx_p042_rs_created            ON ghg_accounting_scope3.reconciliation_summary(created_at DESC);
CREATE INDEX idx_p042_rs_cat_adj            ON ghg_accounting_scope3.reconciliation_summary USING GIN(category_adjustments);

-- ---------------------------------------------------------------------------
-- Trigger
-- ---------------------------------------------------------------------------
CREATE TRIGGER trg_p042_rs_updated
    BEFORE UPDATE ON ghg_accounting_scope3.reconciliation_summary
    FOR EACH ROW EXECUTE FUNCTION ghg_accounting_scope3.fn_set_updated_at();

-- ---------------------------------------------------------------------------
-- Row-Level Security
-- ---------------------------------------------------------------------------
-- overlap_rules is reference data; no RLS needed
ALTER TABLE ghg_accounting_scope3.overlap_detections ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.overlap_resolutions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ghg_accounting_scope3.reconciliation_summary ENABLE ROW LEVEL SECURITY;

CREATE POLICY p042_od_tenant_isolation
    ON ghg_accounting_scope3.overlap_detections
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_od_service_bypass
    ON ghg_accounting_scope3.overlap_detections
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_ores_tenant_isolation
    ON ghg_accounting_scope3.overlap_resolutions
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_ores_service_bypass
    ON ghg_accounting_scope3.overlap_resolutions
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

CREATE POLICY p042_rs_tenant_isolation
    ON ghg_accounting_scope3.reconciliation_summary
    USING (tenant_id = current_setting('app.current_tenant')::UUID);
CREATE POLICY p042_rs_service_bypass
    ON ghg_accounting_scope3.reconciliation_summary
    TO greenlang_service
    USING (TRUE) WITH CHECK (TRUE);

-- ---------------------------------------------------------------------------
-- Grants
-- ---------------------------------------------------------------------------
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.overlap_rules TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.overlap_detections TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.overlap_resolutions TO PUBLIC;
GRANT SELECT, INSERT, UPDATE, DELETE ON ghg_accounting_scope3.reconciliation_summary TO PUBLIC;

-- =============================================================================
-- Seed Data: 12 Overlap Rules
-- =============================================================================
-- Predefined rules for the most common Scope 3 double-counting scenarios.
-- Source: GHG Protocol Corporate Value Chain Standard Technical Guidance.

INSERT INTO ghg_accounting_scope3.overlap_rules
    (rule_id, category_a, category_b, rule_name, description, overlap_type, detection_method, detection_description, resolution_method, resolution_description, default_priority_category, risk_level, typical_overlap_pct, ghg_protocol_reference)
VALUES
    ('OVR-001', 'CAT_1', 'CAT_4', 'Purchased Goods vs Upstream Transport',
     'Transport costs embedded in purchase price may be counted in both Cat 1 (spend-based) and Cat 4 (upstream T&D). When using spend-based method for Cat 1, transportation costs in purchase invoices overlap with separately calculated Cat 4.',
     'BOUNDARY', 'SPEND_OVERLAP', 'Check if Cat 1 spend includes freight/shipping line items that are also captured in Cat 4 transport records.',
     'SUBTRACT_OVERLAP', 'Subtract the freight component from Cat 1 spend before applying EEIO factors, or exclude matched transport records from Cat 4.',
     'CAT_4', 'HIGH', 5.00, 'GHG Protocol Scope 3 Standard, Chapter 5'),

    ('OVR-002', 'CAT_1', 'CAT_2', 'Purchased Goods vs Capital Goods',
     'Capital equipment purchases may be categorized as both Cat 1 and Cat 2. Items with useful life exceeding one year should be in Cat 2, not Cat 1.',
     'DEFINITIONAL', 'RULE_BASED', 'Flag transactions with GL accounts mapped to capital expenditure or with descriptions indicating durable goods (machinery, vehicles, IT equipment).',
     'APPLY_HIERARCHY', 'Reclassify capital items from Cat 1 to Cat 2 based on GL account or asset threshold rules.',
     'CAT_2', 'MEDIUM', 3.00, 'GHG Protocol Scope 3 Standard, Chapter 5'),

    ('OVR-003', 'CAT_3', 'CAT_1', 'Fuel & Energy vs Purchased Goods',
     'Fuel purchases may appear in both Cat 3 (fuel/energy not in Scope 1/2) and Cat 1 (purchased goods). Fuel consumed at owned facilities is Scope 1, but fuel-related upstream emissions belong in Cat 3.',
     'BOUNDARY', 'VENDOR_MATCH', 'Identify fuel/energy vendors in Cat 1 spend that should be allocated to Cat 3 (upstream emissions of purchased fuels).',
     'ALLOCATE_TO_PRIMARY', 'Allocate fuel-related upstream emissions to Cat 3 and remove from Cat 1 spend-based calculation.',
     'CAT_3', 'HIGH', 8.00, 'GHG Protocol Scope 3 Standard, Chapter 5'),

    ('OVR-004', 'CAT_4', 'CAT_9', 'Upstream vs Downstream Transport',
     'Transport for goods sold may be counted in both Cat 4 (upstream T&D) and Cat 9 (downstream T&D) if the company pays for shipping to customer. Inbound transport is Cat 4; outbound to customer is Cat 9.',
     'BOUNDARY', 'ACTIVITY_MATCH', 'Compare transport records: inbound shipments (supplier to company) belong in Cat 4; outbound (company to customer) belong in Cat 9.',
     'APPLY_HIERARCHY', 'Classify by direction: inbound = Cat 4, outbound = Cat 9. Remove duplicates where same shipment appears in both.',
     'CAT_4', 'MEDIUM', 2.00, 'GHG Protocol Scope 3 Standard, Chapter 5'),

    ('OVR-005', 'CAT_6', 'CAT_7', 'Business Travel vs Employee Commuting',
     'Travel between home and a temporary work location (e.g., conference, client site) could be classified as either Cat 6 (business travel) or Cat 7 (commuting). Definition boundary is key.',
     'DEFINITIONAL', 'RULE_BASED', 'Flag travel records where origin is employee home address and destination is not the regular office (potential Cat 6) vs regular office (Cat 7).',
     'APPLY_HIERARCHY', 'Apply company travel policy definition: travel to regular workplace = Cat 7, travel to other locations = Cat 6.',
     'CAT_6', 'LOW', 1.00, 'GHG Protocol Scope 3 Standard, Chapters 7/8'),

    ('OVR-006', 'CAT_1', 'CAT_5', 'Purchased Goods vs Waste in Operations',
     'Waste disposal costs in purchase data may appear in both Cat 1 (as purchased waste services) and Cat 5 (waste generated in operations). The emissions from waste treatment are Cat 5.',
     'BOUNDARY', 'VENDOR_MATCH', 'Identify waste management vendors (NAICS 562) in Cat 1 spend and cross-reference with Cat 5 waste records.',
     'ALLOCATE_TO_PRIMARY', 'Allocate waste service spend to Cat 5 and exclude from Cat 1 EEIO calculation.',
     'CAT_5', 'MEDIUM', 2.00, 'GHG Protocol Scope 3 Standard, Chapter 5'),

    ('OVR-007', 'CAT_8', 'CAT_13', 'Upstream vs Downstream Leased Assets',
     'Leased assets may be miscategorized between Cat 8 (upstream - assets leased by reporting company) and Cat 13 (downstream - assets leased to others). Direction of lease determines category.',
     'DEFINITIONAL', 'RULE_BASED', 'Check lease direction: company as lessee = Cat 8, company as lessor = Cat 13.',
     'APPLY_HIERARCHY', 'Reclassify based on lease agreement direction. Remove from incorrect category.',
     'CAT_8', 'LOW', 1.50, 'GHG Protocol Scope 3 Standard, Chapters 9/14'),

    ('OVR-008', 'CAT_1', 'CAT_6', 'Purchased Services vs Business Travel',
     'Consulting and professional services may include travel costs that overlap with Cat 6. When using spend-based for Cat 1, embedded travel costs in service invoices may duplicate Cat 6.',
     'BOUNDARY', 'SPEND_OVERLAP', 'Identify professional service invoices (NAICS 541) with travel/expense line items also reported in Cat 6 travel records.',
     'SUBTRACT_OVERLAP', 'Subtract travel reimbursement amounts from service invoices before Cat 1 EEIO calculation.',
     'CAT_6', 'LOW', 1.00, 'GHG Protocol Scope 3 Standard, Chapter 5'),

    ('OVR-009', 'CAT_11', 'CAT_12', 'Use of Sold Products vs End-of-Life',
     'Energy consumption during product use (Cat 11) and end-of-life treatment (Cat 12) should not double-count the product mass. Ensure product weight is not counted as both use-phase energy and end-of-life waste.',
     'BOUNDARY', 'ACTIVITY_MATCH', 'Compare product quantity used in Cat 11 energy calculations with product mass used in Cat 12 end-of-life calculations.',
     'FLAG_ONLY', 'Verify that Cat 11 and Cat 12 use consistent product quantity data. Flag if discrepancies exceed 5%.',
     NULL, 'LOW', 0.50, 'GHG Protocol Scope 3 Standard, Chapters 12/13'),

    ('OVR-010', 'CAT_3', 'CAT_4', 'Fuel & Energy vs Upstream Transport',
     'Fuel used in upstream transportation may be counted in both Cat 3 (upstream fuel emissions) and Cat 4 (upstream transport). Cat 3 covers well-to-tank emissions; Cat 4 covers tank-to-wheel.',
     'METHODOLOGY', 'EF_SCOPE_CHECK', 'Verify emission factor boundaries: Cat 3 should use WTT (well-to-tank) factors, Cat 4 should use TTW (tank-to-wheel). If Cat 4 uses WTW (well-to-wheel) factors, subtract WTT portion.',
     'SUBTRACT_OVERLAP', 'If Cat 4 uses well-to-wheel factors, subtract the well-to-tank component to avoid overlap with Cat 3.',
     'CAT_3', 'HIGH', 10.00, 'GHG Protocol Scope 3 Standard, Chapters 4/5'),

    ('OVR-011', 'CAT_14', 'CAT_1', 'Franchises vs Purchased Goods',
     'Franchise fees paid may be counted in both Cat 14 (franchise emissions) and Cat 1 (purchased services). Franchise-related procurement should be in Cat 14.',
     'DEFINITIONAL', 'VENDOR_MATCH', 'Identify franchise-related vendors in Cat 1 spend and cross-reference with Cat 14 franchise records.',
     'ALLOCATE_TO_PRIMARY', 'Allocate franchise-related spend from Cat 1 to Cat 14.',
     'CAT_14', 'LOW', 1.00, 'GHG Protocol Scope 3 Standard, Chapters 2/15'),

    ('OVR-012', 'CAT_15', 'CAT_1', 'Investments vs Purchased Services',
     'Financial service fees for investment management may appear in both Cat 15 (financed emissions) and Cat 1 (purchased financial services). The investment emissions (financed) are Cat 15; service fees are Cat 1.',
     'DEFINITIONAL', 'SPEND_OVERLAP', 'Distinguish between investment principal/returns (Cat 15) and investment management service fees (Cat 1).',
     'APPLY_HIERARCHY', 'Financed emissions based on investment value go to Cat 15. Management fees for financial services stay in Cat 1.',
     'CAT_15', 'MEDIUM', 2.00, 'GHG Protocol Scope 3 Standard, Chapters 2/16');

-- ---------------------------------------------------------------------------
-- Comments
-- ---------------------------------------------------------------------------
COMMENT ON TABLE ghg_accounting_scope3.overlap_rules IS
    'Predefined rules for detecting double-counting between Scope 3 categories based on GHG Protocol Technical Guidance.';
COMMENT ON TABLE ghg_accounting_scope3.overlap_detections IS
    'Detection results from running overlap rules against an inventory, quantifying overlap amounts between category pairs.';
COMMENT ON TABLE ghg_accounting_scope3.overlap_resolutions IS
    'Resolution decisions for detected overlaps with adjusted amounts, justification, and approval workflow.';
COMMENT ON TABLE ghg_accounting_scope3.reconciliation_summary IS
    'Final reconciled Scope 3 totals after all double-counting adjustments with before/after comparison.';

COMMENT ON COLUMN ghg_accounting_scope3.overlap_rules.rule_id IS 'Human-readable rule identifier (e.g., OVR-001).';
COMMENT ON COLUMN ghg_accounting_scope3.overlap_rules.overlap_type IS 'Nature of overlap: BOUNDARY, METHODOLOGY, DATA_SOURCE, ALLOCATION, TEMPORAL, DEFINITIONAL.';
COMMENT ON COLUMN ghg_accounting_scope3.overlap_rules.detection_method IS 'Automated detection approach: VENDOR_MATCH, SPEND_OVERLAP, ACTIVITY_MATCH, etc.';
COMMENT ON COLUMN ghg_accounting_scope3.overlap_rules.resolution_method IS 'Resolution approach: ALLOCATE_TO_PRIMARY, SUBTRACT_OVERLAP, SPLIT_PRO_RATA, etc.';

COMMENT ON COLUMN ghg_accounting_scope3.overlap_detections.overlap_amount IS 'Estimated overlap in tCO2e between the two categories.';
COMMENT ON COLUMN ghg_accounting_scope3.overlap_detections.confidence IS 'Confidence of overlap detection (0-1).';

COMMENT ON COLUMN ghg_accounting_scope3.overlap_resolutions.adjustment_tco2e IS 'Generated column: original_amount - adjusted_amount = tCO2e removed by this resolution.';
COMMENT ON COLUMN ghg_accounting_scope3.overlap_resolutions.justification IS 'Explanation of why this resolution approach was chosen (required for audit trail).';

COMMENT ON COLUMN ghg_accounting_scope3.reconciliation_summary.total_before_adjustment IS 'Sum of all category results before double-counting adjustments.';
COMMENT ON COLUMN ghg_accounting_scope3.reconciliation_summary.total_after_adjustment IS 'Reconciled total after removing all confirmed overlaps.';
COMMENT ON COLUMN ghg_accounting_scope3.reconciliation_summary.adjustment_pct IS 'Generated column: percentage reduction from double-counting adjustments.';
