# PACK-024 Carbon Neutral Pack - Migration Index

## Quick Reference Guide

### Migration Structure
```
V138-V147: 10 migrations covering 40 tables in pack024_carbon_neutral schema
Total: 4,278 lines of SQL across 20 files (10 up + 10 down)
```

---

## Migration Overview Table

| Version | Name | Focus Area | Tables | Key Feature |
|---------|------|-----------|--------|-----------|
| V138 | Footprint Quantification | Emissions | 4 | Baseline, uncertainty, reconciliation |
| V139 | Management Plans | Strategy | 4 | Plans, pathways, actions, assignments |
| V140 | Credit Inventory | Portfolio | 4 | Holdings, transactions, validation, additionality |
| V141 | Portfolio Optimization | Analytics | 4 | Optimization, scenarios, recommendations, rebalancing |
| V142 | Registry Retirements | Execution | 4 | Retirement, statements, registry, certificates |
| V143 | Balance Reconciliation | Verification | 4 | Balance, reconciliation, achievement, trends |
| V144 | Claims Substantiation | Integrity | 4 | Claims, evidence, verification, disclosure |
| V145 | Verification Packages | Audit | 4 | Packages, documentation, findings, audit trail |
| V146 | Annual Cycles | Governance | 4 | Cycles, inventory, reviews, governance calendar |
| V147 | Permanence Assurance | Durability | 4 | Assessments, risk factors, monitoring, insurance |

---

## Table Index by Migration

### V138: Carbon Footprint Quantification Records
```
1. pack024_footprint_records (375 lines)
   - Emissions by scope, source, category with 18 indexes

2. pack024_footprint_components (15 indexes)
   - Component breakdown with contribution analysis

3. pack024_baseline_establishment (9 indexes)
   - Baseline year definition with revisions

4. pack024_uncertainty_assessment (9 indexes)
   - Uncertainty quantification with Monte Carlo support
```

### V139: Carbon Management Plans
```
1. pack024_management_plans (387 lines)
   - Strategic targets with board approval (16 indexes)

2. pack024_reduction_pathways (11 indexes)
   - Mechanism-specific strategies with ROI

3. pack024_management_actions (15 indexes)
   - Execution activities with resource allocation

4. pack024_action_assignments (11 indexes)
   - Responsibility matrix with accountability
```

### V140: Carbon Credit Inventory
```
1. pack024_credit_inventory (379 lines)
   - Holdings by standard with compliance (15 indexes)

2. pack024_credit_transactions (11 indexes)
   - Trading history with settlement tracking

3. pack024_credit_validation (10 indexes)
   - Compliance verification and authenticity

4. pack024_additionality_assessment (9 indexes)
   - Environmental integrity with scenario analysis
```

### V141: Portfolio Optimization
```
1. pack024_portfolio_optimization (389 lines)
   - Optimization runs with metrics (11 indexes)

2. pack024_optimization_scenarios (11 indexes)
   - Scenario-based variant analysis

3. pack024_optimization_recommendations (11 indexes)
   - Algorithmic recommendations with tracking

4. pack024_rebalancing_actions (11 indexes)
   - Rebalancing execution with approval
```

### V142: Registry Retirements
```
1. pack024_retirement_records (406 lines)
   - Retirement transactions (12 indexes)

2. pack024_retirement_statements (11 indexes)
   - Formal statements with assurance

3. pack024_registry_submissions (11 indexes)
   - Registry uploads with confirmation

4. pack024_retirement_certificates (12 indexes)
   - Digital certificates with blockchain
```

### V143: Neutralization Balance Reconciliation
```
1. pack024_neutralization_balance (428 lines)
   - Balance tracking and achievement (11 indexes)

2. pack024_balance_reconciliation (11 indexes)
   - Reconciliation between emissions/credits

3. pack024_net_zero_achievement (12 indexes)
   - Achievement status with verification

4. pack024_balance_trend_analysis (12 indexes)
   - Historical trends with forecasting
```

### V144: Claims Substantiation
```
1. pack024_claim_substantiation (433 lines)
   - Claims registry with approval (11 indexes)

2. pack024_claim_evidence (13 indexes)
   - Evidence documentation with validation

3. pack024_claim_verification (13 indexes)
   - Verification audits with assurance levels

4. pack024_claim_disclosure (12 indexes)
   - Public disclosure with communication
```

### V145: Verification Packages
```
1. pack024_verification_packages (399 lines)
   - Packages with readiness (12 indexes)

2. pack024_package_documentation (12 indexes)
   - Individual documents with acceptance

3. pack024_package_review_findings (11 indexes)
   - Findings with remediation tracking

4. pack024_package_audit_trail (8 indexes)
   - Comprehensive activity tracking
```

### V146: Annual Cycles
```
1. pack024_annual_cycles (416 lines)
   - Period definition with status (11 indexes)

2. pack024_annual_inventory_process (9 indexes)
   - Inventory phases with completion tracking

3. pack024_annual_review_schedule (10 indexes)
   - Review scheduling with findings

4. pack024_annual_governance_calendar (10 indexes)
   - Governance events and decisions
```

### V147: Permanence Assessments
```
1. pack024_permanence_assessments (454 lines)
   - Risk assessment with scenarios (13 indexes)

2. pack024_permanence_risk_factors (11 indexes)
   - Individual risk factors with mitigation

3. pack024_permanence_monitoring (11 indexes)
   - Monitoring activities with anomaly detection

4. pack024_permanence_insurance (12 indexes)
   - Insurance coverage and claims
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ UPSTREAM: Emissions & Credit Data Sources                       │
├─────────────────────────────────────────────────────────────────┤
│  AGENT-MRV Data   │   Carbon Credit Registries   │   Business   │
│  (V051-V081)      │   (VCS, Gold Standard, CDM)  │   Systems    │
└──────────┬─────────────────────────────────────────────┬────────┘
           │                                             │
           v                                             v
┌─────────────────────────────────────────────────────────────────┐
│ V138: FOOTPRINT QUANTIFICATION                                  │
│ - Emissions records with baseline & uncertainty                 │
│ - Reconciliation to MRV agents                                  │
└──────────┬─────────────────────────────────────────────┬────────┘
           │                                             │
           v                                             v
┌──────────────────┐      ┌─────────────────────────────────────┐
│ V139: MANAGEMENT │      │ V140: CREDIT INVENTORY              │
│ PLANS            │      │ - Holdings tracking                 │
│ - Strategies     │      │ - Multi-standard support            │
│ - Actions        │      │ - Additionality verification        │
└────────┬─────────┘      └──────────┬──────────────────────────┘
         │                           │
         v                           v
┌────────────────────────────────────────────────────────────────┐
│ V141: PORTFOLIO OPTIMIZATION                                   │
│ - Optimization algorithms & scenarios                          │
│ - Recommendations & rebalancing                               │
└────────┬─────────────────────────────────────────────┬────────┘
         │                                             │
         v                                             v
┌──────────────────────┐    ┌──────────────────────────────────┐
│ V142: RETIREMENT     │    │ V143: BALANCE RECONCILIATION    │
│ - Retirement records │    │ - Emissions-to-credits matching │
│ - Registry upload    │    │ - Achievement status            │
│ - Certificates       │    │ - Trend analysis & forecasting  │
└──────────┬───────────┘    └──────────┬─────────────────────┘
           │                           │
           v                           v
┌────────────────────────────────────────────────────────────────┐
│ V144: CLAIMS SUBSTANTIATION                                    │
│ - Claims documentation & evidence                              │
│ - Third-party verification                                    │
│ - Public disclosure planning                                  │
└────────┬─────────────────────────────────────────────┬────────┘
         │                                             │
         v                                             v
┌──────────────────────┐    ┌──────────────────────────────────┐
│ V145: VERIFICATION   │    │ V146: ANNUAL CYCLES            │
│ PACKAGES             │    │ - Annual governance             │
│ - Document bundles   │    │ - Inventory processes           │
│ - Audit preparation  │    │ - Review scheduling             │
└──────────┬───────────┘    └──────────┬─────────────────────┘
           │                           │
           v                           v
┌────────────────────────────────────────────────────────────────┐
│ V147: PERMANENCE ASSURANCE                                     │
│ - Risk assessment with scenario analysis                       │
│ - Monitoring activities & anomaly detection                    │
│ - Insurance & guarantee coverage                              │
└────────┬─────────────────────────────────────────────┬────────┘
         │                                             │
         v                                             v
┌─────────────────────────────────────────────────────────────────┐
│ DOWNSTREAM: Reporting & Compliance                             │
├─────────────────────────────────────────────────────────────────┤
│  GL-GHG-APP   │   GL-ISO14064-APP   │   Regulatory   │ External│
│  (Balance)    │   (Scope 3)         │   Filings      │ Audits  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Relationships

### Foreign Key Dependencies
```
V138 (Footprint)
  └─> V139 (Management Plans) - references baseline
      └─> V140 (Credit Inventory) - implements strategy
          └─> V141 (Optimization) - improves portfolio
              └─> V142 (Retirements) - executes optimization
                  └─> V143 (Balance) - verifies status
                      └─> V144 (Claims) - documents achievement
                          └─> V145 (Verification) - packages evidence
                              └─> V146 (Annual) - maintains annually
                                  └─> V147 (Permanence) - ensures durability
```

### Cross-Migration References
- V138.footprint_records ← referenced by V139, V143
- V139.management_plans ← referenced by V141
- V140.credit_inventory ← referenced by V141, V142, V143
- V141.optimization_run ← referenced by V142
- V142.retirement_records ← referenced by V143
- V143.balance ← referenced by V144
- V144.claims ← referenced by V145
- V145.packages ← referenced by V146
- V146.cycles ← referenced by V147

---

## Column Naming Conventions

### Standard Suffixes
- `_id`: Foreign key reference (e.g., `org_id`, `footprint_record_id`)
- `_date`: Date without time (e.g., `assessment_date`)
- `_at`: Timestamp with timezone (e.g., `created_at`, `updated_at`)
- `_status`: Status field (e.g., `cycle_status`, `claim_status`)
- `_pct`: Percentage value 0-100 (e.g., `completion_percentage`)
- `_score`: Numeric score (e.g., `risk_score`, `confidence_level`)
- `_usd`: Monetary value in USD (e.g., `cost_usd`, `price_per_unit_usd`)
- `_name`: Text identifier (e.g., `org_name`, `project_name`)

### Standard Prefixes
- `total_`: Aggregated value (e.g., `total_emissions_tco2e`)
- `annual_`: Per-year value (e.g., `annual_savings_mtco2e`)
- `target_`: Goal value (e.g., `target_emissions`, `target_year`)
- `actual_`: Realized value (e.g., `actual_completion_date`)
- `estimated_`: Forecast value (e.g., `estimated_weeks_to_ready`)
- `baseline_`: Reference value (e.g., `baseline_emissions`, `baseline_year`)

---

## Index Strategy

### B-Tree Indexes (Most Common)
Used for:
- Organization/tenant filtering (`idx_*_org`, `idx_*_tenant`)
- Status fields (`idx_*_status`, `idx_*_approval`)
- Date filtering (`idx_*_date`, `idx_*_deadline`)
- Numeric scoring (`idx_*_score`)
- Name/code lookups (`idx_*_name`, `idx_*_code`)

### GIN Indexes (Array/JSONB)
Used for:
- Array fields: `data_sources`, `stakeholders`, `dependencies`
- JSONB objects: `metadata`, `assumptions`, `resource_requirements`
- Enables `@>` (contains) and `?` (key exists) operators

### Composite Indexes
Used for:
- Common multi-column queries
- Foreign key + status patterns
- Date range + status patterns

---

## Testing Checklist

### Syntax Validation
- [ ] All SQL is valid syntax
- [ ] All keywords properly spelled
- [ ] No duplicate column names
- [ ] All data types valid PostgreSQL

### Constraint Validation
- [ ] CHECK constraints have valid expressions
- [ ] UNIQUE constraints on appropriate columns
- [ ] Foreign keys reference existing tables
- [ ] Primary keys on all tables
- [ ] NOT NULL where appropriate

### Index Validation
- [ ] All indexed columns exist
- [ ] GIN indexes on array/JSONB columns
- [ ] B-tree indexes on numeric/date/text columns
- [ ] Naming convention followed
- [ ] No duplicate indexes

### Migration Validity
- [ ] Up migrations create all objects
- [ ] Down migrations drop in reverse order
- [ ] Foreign key order respected
- [ ] Triggers created and reference functions
- [ ] Functions created before tables that use them

### Completeness
- [ ] All 10 migrations present (V138-V147)
- [ ] All up migrations present
- [ ] All down migrations present
- [ ] Total of 40 tables created
- [ ] Total of 450+ indexes created
- [ ] Total of 40 triggers created

---

## Performance Considerations

### Query Optimization Tips
1. **Filtering by Organization**
   ```sql
   SELECT * FROM pack024_footprint_records
   WHERE org_id = 'xxx' AND tenant_id = 'yyy'
   ORDER BY reporting_year DESC;
   ```
   Uses: `idx_pack024_fp_org`, `idx_pack024_fp_tenant`

2. **Status-Based Queries**
   ```sql
   SELECT * FROM pack024_management_plans
   WHERE approval_status = 'approved' AND plan_status = 'active';
   ```
   Uses: `idx_pack024_mgmt_approval`, `idx_pack024_mgmt_status`

3. **Date Range Queries**
   ```sql
   SELECT * FROM pack024_annual_cycles
   WHERE cycle_start_date >= '2025-01-01'
   AND cycle_end_date <= '2025-12-31';
   ```
   Uses: `idx_pack024_cycle_start_date`, `idx_pack024_cycle_end_date`

4. **JSONB Queries**
   ```sql
   SELECT * FROM pack024_footprint_records
   WHERE assumptions @> '{"normalization_applied": true}';
   ```
   Uses: `idx_pack024_fp_assumptions`

---

## Deployment Sequence

### Recommended Order (Forward)
1. V138 - Establishes base emissions records
2. V139 - Links to management strategy
3. V140 - Adds credit portfolio management
4. V141 - Enables optimization recommendations
5. V142 - Implements retirement execution
6. V143 - Verifies balance achievement
7. V144 - Documents claims
8. V145 - Organizes verification packages
9. V146 - Manages annual governance
10. V147 - Ensures permanence

### Rollback Sequence (Reverse)
Apply in reverse order (V147 down to V138)

---

## Support & Maintenance

### Monitoring Queries
```sql
-- Table sizes
SELECT
  schemaname, tablename,
  pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
WHERE schemaname = 'pack024_carbon_neutral'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Index usage
SELECT
  indexname, idx_scan, idx_tup_read, idx_tup_fetch
FROM pg_stat_user_indexes
WHERE schemaname = 'pack024_carbon_neutral'
ORDER BY idx_scan DESC;

-- Row counts
SELECT
  tablename, n_live_tup as row_count
FROM pg_stat_user_tables
WHERE schemaname = 'pack024_carbon_neutral'
ORDER BY n_live_tup DESC;
```

### Maintenance Tasks
- Analyze/vacuum tables after bulk operations
- Reindex if performance degrades
- Monitor index bloat and rebuild if needed
- Review slow queries and optimize with new indexes
- Monitor foreign key constraint performance

---

## Documentation Links

- **Full Details**: PACK-024-MIGRATIONS-SUMMARY.md
- **File Location**: `/deployment/database/migrations/sql/V{138-147}*.sql`
- **Schema Name**: `pack024_carbon_neutral`
- **Related PRD**: PRD-PACK-024-Carbon-Neutral-Pack.md

---

**Status**: Complete and Ready for Deployment
**Date**: March 2026
**Version**: 1.0
