-- Test Database Schema for GreenLang Testing

-- Emissions table
CREATE TABLE IF NOT EXISTS emissions (
    id SERIAL PRIMARY KEY,
    emission_id VARCHAR(50) UNIQUE NOT NULL,
    scope INTEGER NOT NULL CHECK (scope IN (1, 2, 3)),
    category VARCHAR(100) NOT NULL,
    quantity DECIMAL(15, 2) NOT NULL,
    unit VARCHAR(50) NOT NULL,
    emission_factor DECIMAL(10, 4) NOT NULL,
    total_emissions DECIMAL(15, 2) NOT NULL,
    cost DECIMAL(15, 2),
    currency VARCHAR(10),
    date DATE NOT NULL,
    location VARCHAR(100),
    verified BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Suppliers table
CREATE TABLE IF NOT EXISTS suppliers (
    id SERIAL PRIMARY KEY,
    supplier_id VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    industry VARCHAR(100),
    location_country VARCHAR(100),
    location_city VARCHAR(100),
    location_region VARCHAR(100),
    sustainability_score DECIMAL(3, 1),
    emission_intensity DECIMAL(10, 4),
    annual_spend DECIMAL(15, 2),
    risk_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Supplier certifications table
CREATE TABLE IF NOT EXISTS supplier_certifications (
    id SERIAL PRIMARY KEY,
    supplier_id INTEGER REFERENCES suppliers(id) ON DELETE CASCADE,
    certification VARCHAR(100) NOT NULL,
    issued_date DATE,
    expiry_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reports table
CREATE TABLE IF NOT EXISTS reports (
    id SERIAL PRIMARY KEY,
    report_id VARCHAR(50) UNIQUE NOT NULL,
    title VARCHAR(200) NOT NULL,
    report_type VARCHAR(50) NOT NULL,
    reporting_period VARCHAR(50),
    total_emissions DECIMAL(15, 2),
    scope_1_emissions DECIMAL(15, 2),
    scope_2_emissions DECIMAL(15, 2),
    scope_3_emissions DECIMAL(15, 2),
    status VARCHAR(50) DEFAULT 'draft',
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    published_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Activities table (for detailed emission activities)
CREATE TABLE IF NOT EXISTS activities (
    id SERIAL PRIMARY KEY,
    activity_id VARCHAR(50) UNIQUE NOT NULL,
    emission_id INTEGER REFERENCES emissions(id) ON DELETE CASCADE,
    description TEXT,
    facility VARCHAR(200),
    department VARCHAR(100),
    responsible_person VARCHAR(200),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Reduction targets table
CREATE TABLE IF NOT EXISTS reduction_targets (
    id SERIAL PRIMARY KEY,
    target_id VARCHAR(50) UNIQUE NOT NULL,
    target_year INTEGER NOT NULL,
    baseline_year INTEGER NOT NULL,
    baseline_emissions DECIMAL(15, 2) NOT NULL,
    target_emissions DECIMAL(15, 2) NOT NULL,
    reduction_percentage DECIMAL(5, 2) NOT NULL,
    scope VARCHAR(20),
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Validation results table
CREATE TABLE IF NOT EXISTS validation_results (
    id SERIAL PRIMARY KEY,
    validation_id VARCHAR(50) UNIQUE NOT NULL,
    entity_type VARCHAR(50) NOT NULL,
    entity_id VARCHAR(50) NOT NULL,
    is_valid BOOLEAN NOT NULL,
    errors TEXT,
    warnings TEXT,
    validated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_emissions_scope ON emissions(scope);
CREATE INDEX IF NOT EXISTS idx_emissions_date ON emissions(date);
CREATE INDEX IF NOT EXISTS idx_emissions_category ON emissions(category);
CREATE INDEX IF NOT EXISTS idx_suppliers_risk ON suppliers(risk_level);
CREATE INDEX IF NOT EXISTS idx_suppliers_score ON suppliers(sustainability_score);
CREATE INDEX IF NOT EXISTS idx_reports_period ON reports(reporting_period);
CREATE INDEX IF NOT EXISTS idx_reports_status ON reports(status);

-- Sample data for testing
INSERT INTO suppliers (supplier_id, name, industry, location_country, location_city, location_region, sustainability_score, emission_intensity, annual_spend, risk_level)
VALUES
    ('sup_001', 'Steel Corp International', 'Manufacturing', 'USA', 'Pittsburgh', 'North America', 8.5, 2.5, 500000, 'low'),
    ('sup_002', 'Global Logistics Inc', 'Transportation', 'USA', 'Chicago', 'North America', 7.8, 0.12, 250000, 'medium'),
    ('sup_003', 'Clean Energy Solutions', 'Energy', 'Germany', 'Berlin', 'Europe', 9.2, 0.05, 800000, 'low');

INSERT INTO emissions (emission_id, scope, category, quantity, unit, emission_factor, total_emissions, cost, currency, date, location, verified)
VALUES
    ('em_001', 3, 'Purchased Goods and Services', 1000, 'kg', 2.5, 2500, 5000, 'USD', '2024-01-15', 'USA', TRUE),
    ('em_002', 3, 'Transportation and Distribution', 500, 'km', 0.12, 60, 1200, 'USD', '2024-01-20', 'USA', TRUE),
    ('em_003', 1, 'Stationary Combustion', 5000, 'cubic_meters', 2.0, 10000, 3000, 'USD', '2024-01-01', 'USA', TRUE),
    ('em_004', 2, 'Purchased Electricity', 10000, 'kWh', 0.4, 4000, 1500, 'USD', '2024-01-31', 'California', TRUE);

INSERT INTO supplier_certifications (supplier_id, certification, issued_date, expiry_date)
VALUES
    (1, 'ISO 14001', '2023-01-01', '2026-01-01'),
    (1, 'ISO 50001', '2023-06-01', '2026-06-01'),
    (2, 'SmartWay', '2023-03-01', '2026-03-01'),
    (3, 'B Corp', '2022-01-01', '2025-01-01');
