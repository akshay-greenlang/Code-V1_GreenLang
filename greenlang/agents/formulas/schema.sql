-- ============================================================================
-- GreenLang Formula Versioning Database Schema
-- ============================================================================
--
-- This schema implements version-controlled formula management with:
-- - Complete audit trails
-- - Rollback capability
-- - A/B testing support
-- - Dependency tracking
-- - Execution logging
--
-- Database: SQLite (dev) / PostgreSQL (prod)
-- Version: 1.0.0
-- ============================================================================

-- ============================================================================
-- FORMULAS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS formulas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_code VARCHAR(50) UNIQUE NOT NULL,  -- "E1-1", "BOILER_EFF_001"
    formula_name VARCHAR(200) NOT NULL,
    category VARCHAR(50) NOT NULL,  -- "emissions", "efficiency", "cost", "energy"
    description TEXT,
    standard_reference VARCHAR(200),  -- "ESRS E1", "ASME PTC 4.1", "CBAM Regulation"

    -- Audit metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Indexes
    INDEX idx_formula_code (formula_code),
    INDEX idx_category (category)
);

-- ============================================================================
-- FORMULA_VERSIONS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS formula_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_id INTEGER NOT NULL,
    version_number INTEGER NOT NULL,

    -- Formula definition
    formula_expression TEXT NOT NULL,  -- "scope1 + scope2 + scope3"
    calculation_type VARCHAR(50) NOT NULL,  -- "sum", "database_lookup_and_multiply", "percentage"
    required_inputs TEXT NOT NULL,  -- JSON array: ["scope1", "scope2", "scope3"]
    optional_inputs TEXT,  -- JSON array: ["adjustment_factor"]
    output_unit VARCHAR(50),  -- "tCO2e", "kWh", "%", "EUR"
    output_type VARCHAR(50) DEFAULT 'numeric',  -- "numeric", "boolean", "string"

    -- Validation and constraints
    validation_rules TEXT,  -- JSON: {"min": 0, "max": 1000000, "required": true}
    deterministic BOOLEAN DEFAULT 1,  -- 1 = deterministic, 0 = non-deterministic
    zero_hallucination BOOLEAN DEFAULT 1,  -- 1 = zero-hallucination safe

    -- Version metadata
    version_status VARCHAR(20) DEFAULT 'draft',  -- draft, active, deprecated, archived
    effective_from DATE,
    effective_to DATE,

    -- Documentation
    change_notes TEXT,
    example_calculation TEXT,  -- Example showing inputs -> output

    -- A/B testing
    ab_test_group VARCHAR(50),  -- NULL, "control", "variant_a", "variant_b"
    ab_traffic_weight FLOAT DEFAULT 1.0,  -- 0.0 to 1.0 (for traffic splitting)

    -- Audit trail
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',

    -- Performance tracking
    avg_execution_time_ms FLOAT,
    execution_count INTEGER DEFAULT 0,

    FOREIGN KEY (formula_id) REFERENCES formulas(id) ON DELETE CASCADE,
    UNIQUE(formula_id, version_number),

    -- Indexes
    INDEX idx_formula_version (formula_id, version_number),
    INDEX idx_version_status (version_status),
    INDEX idx_effective_dates (effective_from, effective_to),
    INDEX idx_ab_test_group (ab_test_group)
);

-- ============================================================================
-- FORMULA_DEPENDENCIES TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS formula_dependencies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_version_id INTEGER NOT NULL,
    depends_on_formula_code VARCHAR(50) NOT NULL,
    depends_on_version_number INTEGER,  -- NULL = latest active version
    dependency_type VARCHAR(20) DEFAULT 'required',  -- required, optional

    FOREIGN KEY (formula_version_id) REFERENCES formula_versions(id) ON DELETE CASCADE,

    -- Indexes
    INDEX idx_dependency_lookup (formula_version_id),
    INDEX idx_reverse_dependency (depends_on_formula_code)
);

-- ============================================================================
-- FORMULA_EXECUTION_LOG TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS formula_execution_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    formula_version_id INTEGER NOT NULL,

    -- Execution context
    execution_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    agent_name VARCHAR(100),
    calculation_id VARCHAR(64),  -- Links to broader calculation context
    user_id VARCHAR(100),

    -- Input/Output hashing for provenance
    input_hash VARCHAR(64),  -- SHA-256 of canonical input JSON
    output_hash VARCHAR(64),  -- SHA-256 of output value
    input_data TEXT,  -- Full input JSON (for debugging)
    output_value TEXT,  -- Full output (for debugging)

    -- Performance metrics
    execution_time_ms FLOAT,

    -- Status
    execution_status VARCHAR(20) DEFAULT 'success',  -- success, error, timeout
    error_message TEXT,

    FOREIGN KEY (formula_version_id) REFERENCES formula_versions(id) ON DELETE CASCADE,

    -- Indexes
    INDEX idx_execution_timestamp (execution_timestamp),
    INDEX idx_calculation_id (calculation_id),
    INDEX idx_input_hash (input_hash),
    INDEX idx_execution_status (execution_status)
);

-- ============================================================================
-- FORMULA_AB_TESTS TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS formula_ab_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_name VARCHAR(100) UNIQUE NOT NULL,
    formula_code VARCHAR(50) NOT NULL,

    -- Test configuration
    control_version_id INTEGER NOT NULL,
    variant_version_id INTEGER NOT NULL,
    traffic_split FLOAT DEFAULT 0.5,  -- % of traffic to variant (0.0-1.0)

    -- Test status
    test_status VARCHAR(20) DEFAULT 'draft',  -- draft, running, completed, cancelled
    started_at TIMESTAMP,
    ended_at TIMESTAMP,

    -- Results tracking
    control_executions INTEGER DEFAULT 0,
    variant_executions INTEGER DEFAULT 0,
    control_avg_time_ms FLOAT,
    variant_avg_time_ms FLOAT,

    -- Decision
    winning_version_id INTEGER,
    decision_notes TEXT,

    -- Audit
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',

    FOREIGN KEY (control_version_id) REFERENCES formula_versions(id),
    FOREIGN KEY (variant_version_id) REFERENCES formula_versions(id),
    FOREIGN KEY (winning_version_id) REFERENCES formula_versions(id),

    -- Indexes
    INDEX idx_test_status (test_status),
    INDEX idx_formula_code_test (formula_code)
);

-- ============================================================================
-- FORMULA_MIGRATION_LOG TABLE
-- ============================================================================

CREATE TABLE IF NOT EXISTS formula_migration_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    migration_name VARCHAR(100) NOT NULL,
    source_type VARCHAR(50) NOT NULL,  -- "yaml", "python", "manual"
    source_file VARCHAR(500),
    formulas_migrated INTEGER DEFAULT 0,
    migration_status VARCHAR(20) DEFAULT 'pending',  -- pending, in_progress, completed, failed
    error_message TEXT,

    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(100) DEFAULT 'system',

    -- Indexes
    INDEX idx_migration_status (migration_status)
);

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Active formulas view
CREATE VIEW IF NOT EXISTS v_active_formulas AS
SELECT
    f.formula_code,
    f.formula_name,
    f.category,
    fv.version_number,
    fv.formula_expression,
    fv.calculation_type,
    fv.output_unit,
    fv.effective_from,
    fv.effective_to,
    fv.avg_execution_time_ms,
    fv.execution_count
FROM formulas f
JOIN formula_versions fv ON f.id = fv.formula_id
WHERE fv.version_status = 'active'
    AND (fv.effective_from IS NULL OR fv.effective_from <= DATE('now'))
    AND (fv.effective_to IS NULL OR fv.effective_to >= DATE('now'));

-- Formula dependency tree view
CREATE VIEW IF NOT EXISTS v_formula_dependencies AS
SELECT
    f.formula_code AS formula,
    fv.version_number,
    fd.depends_on_formula_code AS dependency,
    fd.dependency_type
FROM formulas f
JOIN formula_versions fv ON f.id = fv.formula_id
JOIN formula_dependencies fd ON fv.id = fd.formula_version_id
WHERE fv.version_status = 'active';

-- ============================================================================
-- TRIGGERS FOR AUDIT TRAIL
-- ============================================================================

-- Update formula updated_at timestamp
CREATE TRIGGER IF NOT EXISTS trg_formula_updated_at
AFTER UPDATE ON formulas
FOR EACH ROW
BEGIN
    UPDATE formulas SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- Track execution count and avg time
CREATE TRIGGER IF NOT EXISTS trg_formula_execution_stats
AFTER INSERT ON formula_execution_log
FOR EACH ROW
WHEN NEW.execution_status = 'success'
BEGIN
    UPDATE formula_versions
    SET
        execution_count = execution_count + 1,
        avg_execution_time_ms = (
            COALESCE(avg_execution_time_ms * execution_count, 0) + NEW.execution_time_ms
        ) / (execution_count + 1)
    WHERE id = NEW.formula_version_id;
END;

-- ============================================================================
-- INITIAL DATA SETUP
-- ============================================================================

-- Insert system user
INSERT OR IGNORE INTO formulas (id, formula_code, formula_name, category, description, created_by)
VALUES (0, '_SYSTEM', 'System Placeholder', 'system', 'Reserved for system use', 'system');

-- ============================================================================
-- SCHEMA VERSION TRACKING
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version VARCHAR(20) PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    description TEXT
);

INSERT OR REPLACE INTO schema_version (version, description)
VALUES ('1.0.0', 'Initial schema with formula versioning, dependencies, execution log, and A/B testing');

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Additional composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_formula_lookup
    ON formula_versions(formula_id, version_status, effective_from, effective_to);

CREATE INDEX IF NOT EXISTS idx_execution_performance
    ON formula_execution_log(formula_version_id, execution_timestamp);

CREATE INDEX IF NOT EXISTS idx_ab_test_lookup
    ON formula_ab_tests(formula_code, test_status);
