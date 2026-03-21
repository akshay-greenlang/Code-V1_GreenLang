# PACK-027 Enterprise Net Zero Pack -- Architecture Document

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Design Principles](#design-principles)
3. [Three-Tier Architecture](#three-tier-architecture)
4. [Multi-Entity Consolidation Architecture](#multi-entity-consolidation-architecture)
5. [Engine Architecture Patterns](#engine-architecture-patterns)
6. [Workflow DAG Pipeline](#workflow-dag-pipeline)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Database Schema Overview](#database-schema-overview)
9. [ERP Integration Architecture](#erp-integration-architecture)
10. [Security Architecture](#security-architecture)
11. [Performance Architecture](#performance-architecture)
12. [Scalability Design](#scalability-design)
13. [Observability Architecture](#observability-architecture)
14. [Disaster Recovery](#disaster-recovery)

---

## System Overview

PACK-027 implements a horizontally scalable three-tier architecture designed for enterprise-grade GHG accounting at the scale of 100+ entities, 100,000+ suppliers, and 500,000+ employees. The system prioritizes financial-grade accuracy (+/-3%), deterministic reproducibility, and external assurance readiness.

```
+=============================================================================+
|                          PRESENTATION TIER                                   |
|                                                                             |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | Executive         |  | Analyst           |  | REST API          |        |
|  | Dashboard         |  | Workbench         |  | Endpoints         |        |
|  | (Board Reports)   |  | (Full Data)       |  | (JSON/gRPC)       |        |
|  +-------------------+  +-------------------+  +-------------------+        |
|                                                                             |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | Supplier Portal   |  | Auditor Portal    |  | Report Export     |        |
|  | (Data Collection) |  | (Read-Only)       |  | (PDF/XLSX/HTML)   |        |
|  +-------------------+  +-------------------+  +-------------------+        |
|                                                                             |
+=============================================================================+
|                          APPLICATION TIER                                     |
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  |              PACK ORCHESTRATOR (FOUND-001 DAG Engine)                  |  |
|  |                                                                       |  |
|  |  Workflow Layer:                                                       |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  | Compreh  | | SBTi     | | Annual   | | Scenario |                 |  |
|  |  | Baseline | | Submit   | | Inventor | | Analysis |                 |  |
|  |  | 6 phases | | 5 phases | | 5 phases | | 5 phases |                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  | Supply   | | Carbon   | | Multi-   | | External |                 |  |
|  |  | Chain    | | Pricing  | | Entity   | | Assrnce  |                 |  |
|  |  | 5 phases | | 4 phases | | 5 phases | | 5 phases |                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |                                                                       |  |
|  |  Engine Layer:                                                        |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  | Enterpr  | | SBTi     | | Scenario | | Carbon   |                 |  |
|  |  | Baseline | | Target   | | Model    | | Pricing  |                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  | Scope 4  | | Supply   | | Multi-   | | Financia |                 |  |
|  |  | Avoided  | | Chain    | | Entity   | | Integrat |                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |                                                                       |  |
|  |  Agent Bridge Layer:                                                  |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  |  | MRV      | | DATA     | | FOUND    | | DECARB-X |                 |  |
|  |  | Bridge   | | Bridge   | | Bridge   | | Bridge   |                 |  |
|  |  | 30 agent | | 20 agent | | 10 agent | | 21 agent |                 |  |
|  |  +----------+ +----------+ +----------+ +----------+                 |  |
|  +-----------------------------------------------------------------------+  |
|                                                                             |
|  +-----------------------------------------------------------------------+  |
|  |                    INTEGRATION LAYER                                   |  |
|  |  +--------+ +--------+ +--------+ +--------+ +--------+ +--------+  |  |
|  |  | SAP    | | Oracle | | Workday| | CDP    | | SBTi   | | Assrnc |  |  |
|  |  | Conn   | | Conn   | | Conn   | | Bridge | | Bridge | | Bridge |  |  |
|  |  +--------+ +--------+ +--------+ +--------+ +--------+ +--------+  |  |
|  |  +--------+ +--------+ +--------+ +--------+ +--------+ +--------+  |  |
|  |  | Multi- | | Carbon | | Supply | | Financ | | DQ     | | Setup  |  |  |
|  |  | Entity | | Market | | Portal | | System | | Guard  | | Wizard |  |  |
|  |  +--------+ +--------+ +--------+ +--------+ +--------+ +--------+  |  |
|  +-----------------------------------------------------------------------+  |
|                                                                             |
+=============================================================================+
|                           DATA TIER                                          |
|                                                                             |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | PostgreSQL 16     |  | Redis 7 Cluster   |  | S3 Object Store   |        |
|  | + TimescaleDB     |  |                   |  |                   |        |
|  | 15 pack tables    |  | EF cache          |  | Workpapers        |        |
|  | + platform tables |  | Session store     |  | Reports (PDF)     |        |
|  | RLS enabled       |  | Intermediate calc |  | Evidence files    |        |
|  +-------------------+  +-------------------+  +-------------------+        |
|                                                                             |
|  +-------------------+  +-------------------+  +-------------------+        |
|  | Emission Factor   |  | SDA Benchmarks    |  | Carbon Price      |        |
|  | Database          |  | (12 sectors)      |  | Scenarios         |        |
|  | DEFRA/EPA/IEA/    |  | IEA NZE aligned   |  | IEA/NGFS/WB      |        |
|  | ecoinvent/IPCC    |  |                   |  |                   |        |
|  +-------------------+  +-------------------+  +-------------------+        |
|                                                                             |
+=============================================================================+
```

---

## Design Principles

### 1. Zero-Hallucination Calculations

Every numeric calculation in PACK-027 uses deterministic formulas with published emission factors. No large language model or statistical estimation is used in any calculation path.

```python
# Example: Scope 1 stationary combustion (MRV-001)
# NO LLM, NO estimation, NO randomness
emissions_tco2e = (
    fuel_quantity_litres
    * fuel_density_kg_per_litre       # constant per fuel type
    * net_calorific_value_gj_per_kg   # IPCC/national factor
    * emission_factor_kgco2e_per_gj   # DEFRA/EPA/national factor
) / 1000

# All factors are versioned, sourced, and hashed
provenance = {
    "input_hash": sha256(fuel_quantity_litres),
    "emission_factors": {
        "source": "DEFRA 2024",
        "fuel_type": "diesel",
        "density": 0.8320,
        "ncv": 0.04306,
        "ef_co2": 74.10,
        "ef_ch4": 0.003,
        "ef_n2o": 0.006,
    },
    "calculation_version": "27.0.0",
    "output_hash": sha256(emissions_tco2e),
    "timestamp": "2026-03-19T10:00:00Z",
}
```

### 2. Financial-Grade Accuracy

PACK-027 targets +/-3% accuracy for the overall GHG inventory. This is achieved through:

- **5-level data quality hierarchy** per GHG Protocol (Level 1: +/-3%, Level 5: +/-40-60%)
- **Activity-based calculations** wherever data permits (not spend-based)
- **Supplier-specific emission factors** for top 50 suppliers by Scope 3 contribution
- **Cross-source reconciliation** (ERP vs. meter vs. invoice)
- **Automated outlier detection** and variance analysis

| DQ Level | Data Type | Accuracy | Application |
|----------|-----------|----------|-------------|
| 1 | Supplier-specific, verified | +/-3% | Top 50 suppliers (CDP/PACT data) |
| 2 | Supplier-specific, unverified | +/-5-10% | Next 200 suppliers (questionnaire) |
| 3 | Average data (physical) | +/-10-20% | Suppliers with quantity data |
| 4 | Spend-based (EEIO) | +/-20-40% | Tail spend, only when necessary |
| 5 | Proxy/extrapolation | +/-40-60% | Immaterial categories only |

### 3. Bit-Perfect Reproducibility

Given identical inputs, PACK-027 produces identical outputs, verifiable via SHA-256 hash comparison.

```python
# Reproducibility guarantee
result_1 = engine.calculate(input_data)
result_2 = engine.calculate(input_data)

assert result_1.provenance_hash == result_2.provenance_hash
# Always true: deterministic calculations with versioned factors
```

This is enforced by:
- Versioned emission factor databases (DEFRA 2024, EPA 2024, IEA 2024)
- Deterministic rounding (Python `decimal.Decimal` with `ROUND_HALF_UP`)
- Sorted key order in all dictionary operations
- Fixed random seeds for Monte Carlo (user-configurable for reproducible scenarios)

### 4. Separation of Concerns

```
Configuration --> Engine --> Workflow --> Template
     |               |           |           |
     |               |           |           v
     |               |           |       Output formats
     |               |           |       (MD/HTML/JSON/PDF)
     |               |           v
     |               |       Phase orchestration
     |               |       Error recovery
     |               v
     |           Calculation logic
     |           Model definitions
     v
  Pack config
  Preset loading
  Feature flags
```

Each layer has a single responsibility:
- **Configuration**: What to calculate (sector, entities, year, approach)
- **Engine**: How to calculate (formulas, factors, methodology)
- **Workflow**: When and in what order to calculate (phases, dependencies, retry)
- **Template**: How to present results (format, layout, regulatory mapping)

### 5. Enterprise Governance

The three-lines model for climate data governance:

| Line | Responsibility | PACK-027 Support |
|------|---------------|-----------------|
| First Line (Operations) | Entity-level data owners | Entity dashboards, completeness alerts, validation rules |
| Second Line (Compliance) | Central sustainability team | DQ scoring, outlier detection, reconciliation, analytical review |
| Third Line (Assurance) | Internal/external audit | Workpapers, sampling, control testing, management assertions |

### 6. Idempotent Operations

All engine calculations and workflow phases are idempotent. Running the same calculation twice with the same inputs produces the same result without side effects. This enables:

- Safe retry on failure
- Parallel execution across entities
- Cache invalidation without data corruption
- Audit trail consistency

---

## Multi-Entity Consolidation Architecture

### Entity Hierarchy Model

```
Corporate Group (Parent)
  |
  +-- Entity 1 (100% subsidiary) ------> Full consolidation (all approaches)
  |
  +-- Entity 2 (100% subsidiary) ------> Full consolidation (all approaches)
  |   |
  |   +-- Entity 2a (80% sub-sub) -----> Full (control) or 80% (equity share)
  |   |
  |   +-- Entity 2b (60% sub-sub) -----> Full (control) or 60% (equity share)
  |
  +-- Entity 3 (51% JV) ---------------> Depends on control assessment
  |                                       Financial control: 100%
  |                                       Operational control: 100% (if operator)
  |                                       Equity share: 51%
  |
  +-- Entity 4 (33% associate) ---------> Equity share only (33%)
  |                                       Not in control approaches
  |                                       Classified as Scope 3 Cat 15
  |
  +-- Entity 5 (acquired July 1) -------> Pro-rata from acquisition date
  |                                       6/12 of full-year emissions
  |
  +-- Entity 6 (divested April 1) ------> Pro-rata until divestiture
                                          3/12 of full-year emissions
```

### Consolidation Approaches

| Approach | Rule | Use Case | Total Emissions Impact |
|----------|------|----------|----------------------|
| Financial Control | 100% of entities where company directs financial and operating policies | IFRS reporting; aligns with financial statements | Highest (includes all consolidated subsidiaries) |
| Operational Control | 100% of entities where company has authority over operating policies | US GAAP preference; excludes JVs where not operator | Medium (may exclude JV emissions) |
| Equity Share | Proportional to equity ownership percentage | Most conservative; required by some regulators | Lowest (partial JV/associate shares) |

### Intercompany Elimination Flow

```
Step 1: Calculate each entity independently
  Entity A: Scope 1 = 10,000 tCO2e
  Entity B: Scope 1 = 8,000 tCO2e
  Entity C: Scope 2 = 5,000 tCO2e (buys electricity from Entity A)

Step 2: Identify intercompany transactions
  Entity A generates electricity --> Entity C purchases it
  Entity C's Scope 2 includes 1,000 tCO2e from Entity A's generation
  Entity A's Scope 1 already includes these generation emissions

Step 3: Eliminate double-counted emissions
  Remove 1,000 tCO2e from Entity C's Scope 2
  (Entity A's Scope 1 already covers the generation)

Step 4: Reconcile
  Sum of entities (before elimination): 23,000 tCO2e
  Intercompany eliminations: -1,000 tCO2e
  Consolidated total: 22,000 tCO2e

Step 5: Document
  Elimination entry logged with justification
  SHA-256 hash of elimination calculation
  Audit trail entry for assurance
```

### Base Year Recalculation Architecture

```
Trigger Event
    |
    v
Significance Assessment (>5% threshold?)
    |
    +-- NO --> No recalculation needed
    |          Document assessment in audit trail
    |
    +-- YES --> Recalculate base year
                |
                v
          Reconstruct base year as if current structure/methodology
          had been in place
                |
                v
          Adjust all interim years consistently
                |
                v
          Document: old base year, new base year, trigger, delta
                |
                v
          Report restated figures alongside original
                |
                v
          Update all trend analyses and target pathways
```

Recalculation triggers per GHG Protocol:
- Structural change (M&A, divestiture) exceeding 5% significance
- Methodology change (new emission factors, calculation approach)
- Discovery of significant error (>5% impact)
- Change in organizational boundary or consolidation approach
- Outsourcing/insourcing of emitting activities exceeding 5%

---

## Engine Architecture Patterns

### Common Engine Interface

Every engine in PACK-027 follows a consistent interface pattern:

```python
class BaseEngine:
    """Abstract base for all PACK-027 engines."""

    def __init__(self, config: PackConfig):
        self.config = config
        self.provenance_enabled = config.provenance_enabled
        self.provenance_algorithm = config.provenance_algorithm

    def calculate(self, input_data: BaseModel) -> BaseModel:
        """Execute the engine calculation."""
        # 1. Validate input
        validated_input = self._validate_input(input_data)

        # 2. Load emission factors
        factors = self._load_emission_factors()

        # 3. Execute deterministic calculation
        raw_result = self._execute(validated_input, factors)

        # 4. Compute data quality score
        dq_score = self._assess_data_quality(validated_input, raw_result)

        # 5. Generate provenance hash
        provenance = self._generate_provenance(
            input_data, factors, raw_result
        )

        # 6. Build typed result
        return self._build_result(raw_result, dq_score, provenance)

    def _validate_input(self, data: BaseModel) -> BaseModel:
        """Pydantic v2 validation with custom validators."""
        raise NotImplementedError

    def _load_emission_factors(self) -> Dict:
        """Load versioned emission factors from reference database."""
        raise NotImplementedError

    def _execute(self, data: BaseModel, factors: Dict) -> Dict:
        """Pure deterministic calculation. No side effects."""
        raise NotImplementedError

    def _assess_data_quality(self, data: BaseModel, result: Dict) -> float:
        """Score data quality per GHG Protocol 5-level hierarchy."""
        raise NotImplementedError

    def _generate_provenance(self, input_data, factors, result) -> Dict:
        """SHA-256 hash of inputs, factors, and outputs."""
        import hashlib, json
        content = json.dumps({
            "input": input_data.model_dump(),
            "factors": factors,
            "result": result,
        }, sort_keys=True, default=str)
        return {
            "hash": hashlib.sha256(content.encode()).hexdigest(),
            "algorithm": "sha256",
            "timestamp": datetime.utcnow().isoformat(),
            "engine_version": "27.0.0",
        }
```

### Engine-Specific Data Models

Each engine defines typed Pydantic v2 models for configuration, input, and output:

```
EnterpriseBaselineEngine:
    Config:  EnterpriseBaselineConfig
    Input:   EnterpriseBaselineInput (per-entity data packages)
    Output:  EnterpriseBaselineResult (totals, breakdown, DQ matrix)

SBTiTargetEngine:
    Config:  SBTiTargetConfig
    Input:   SBTiTargetInput (baseline result, sector, pathway)
    Output:  SBTiTargetResult (targets, 42 criteria, milestones)

ScenarioModelingEngine:
    Config:  ScenarioConfig
    Input:   ScenarioInput (baseline, portfolio, assumptions)
    Output:  ScenarioResult (trajectories, sensitivity, probability)

CarbonPricingEngine:
    Config:  CarbonPricingConfig
    Input:   CarbonPricingInput (baseline by BU, investments)
    Output:  CarbonPricingResult (carbon P&L, adjusted NPV, CBAM)

Scope4AvoidedEmissionsEngine:
    Config:  AvoidedEmissionsConfig
    Input:   AvoidedEmissionsInput (products, baselines, sales)
    Output:  AvoidedEmissionsResult (avoided by product, uncertainty)

SupplyChainMappingEngine:
    Config:  SupplyChainConfig
    Input:   SupplyChainInput (suppliers, spend, CDP scores)
    Output:  SupplyChainResult (tiers, hotspots, scorecards)

MultiEntityConsolidationEngine:
    Config:  ConsolidationConfig
    Input:   ConsolidationInput (entity results, hierarchy)
    Output:  ConsolidationResult (consolidated, eliminations)

FinancialIntegrationEngine:
    Config:  FinancialIntegrationConfig
    Input:   FinancialIntegrationInput (baseline, financials)
    Output:  FinancialIntegrationResult (carbon P&L, balance sheet)
```

### MRV Agent Bridge Architecture

The enterprise baseline engine orchestrates all 30 MRV agents via the MRV bridge:

```
Enterprise Baseline Engine
    |
    +-- MRV Bridge (mrv_bridge.py)
    |       |
    |       +-- Scope 1 Agents
    |       |     MRV-001: Stationary Combustion
    |       |     MRV-002: Refrigerants & F-Gas
    |       |     MRV-003: Mobile Combustion
    |       |     MRV-004: Process Emissions
    |       |     MRV-005: Fugitive Emissions
    |       |     MRV-006: Land Use Emissions
    |       |     MRV-007: Waste Treatment
    |       |     MRV-008: Agricultural Emissions
    |       |
    |       +-- Scope 2 Agents
    |       |     MRV-009: Scope 2 Location-Based
    |       |     MRV-010: Scope 2 Market-Based
    |       |     MRV-011: Steam/Heat Purchase
    |       |     MRV-012: Cooling Purchase
    |       |     MRV-013: Dual Reporting Reconciliation
    |       |
    |       +-- Scope 3 Agents
    |       |     MRV-014: Cat 1 Purchased Goods
    |       |     MRV-015: Cat 2 Capital Goods
    |       |     MRV-016: Cat 3 Fuel & Energy
    |       |     MRV-017: Cat 4 Upstream Transport
    |       |     MRV-018: Cat 5 Waste Generated
    |       |     MRV-019: Cat 6 Business Travel
    |       |     MRV-020: Cat 7 Employee Commuting
    |       |     MRV-021: Cat 8 Upstream Leased
    |       |     MRV-022: Cat 9 Downstream Transport
    |       |     MRV-023: Cat 10 Processing of Sold
    |       |     MRV-024: Cat 11 Use of Sold
    |       |     MRV-025: Cat 12 End-of-Life
    |       |     MRV-026: Cat 13 Downstream Leased
    |       |     MRV-027: Cat 14 Franchises
    |       |     MRV-028: Cat 15 Investments
    |       |
    |       +-- Cross-Cutting Agents
    |             MRV-029: Category Mapper
    |             MRV-030: Audit Trail & Lineage
    |
    +-- DATA Bridge (data_bridge.py)
    |       |
    |       +-- DATA-001 through DATA-020 (20 agents)
    |
    +-- FOUND Bridge (found_bridge.py)
    |       |
    |       +-- FOUND-001 through FOUND-010 (10 agents)
    |
    +-- DECARB Bridge (decarb_bridge.py)
            |
            +-- DECARB-X-001 through DECARB-X-021 (21 agents)
```

---

## Workflow DAG Pipeline

### Comprehensive Baseline Workflow (6 Phases)

```
Phase 1              Phase 2              Phase 3
ENTITY MAPPING  ---> DATA COLLECTION ---> QUALITY ASSURANCE
[Map org boundary]   [Extract from ERP]   [Profile, dedup,
 [Entity hierarchy]   [Manual uploads]     outlier, gap fill]
 [Control assess]    [per entity]          [DATA agents]

         |                                      |
         v                                      v

Phase 4              Phase 5              Phase 6
CALCULATION     ---> CONSOLIDATION   ---> REPORTING
[Run 30 MRV agents]  [Multi-entity]       [GHG inventory]
[Per entity]          [Intercompany]       [CDP/TCFD/CSRD]
[15 Scope 3 cats]     [Base year check]    [Executive dash]
```

### SBTi Submission Workflow (5 Phases)

```
Phase 1              Phase 2              Phase 3
BASELINE        ---> PATHWAY         ---> TARGET
VALIDATION           SELECTION            DEFINITION
[DQ meets SBTi]      [ACA vs SDA         [Near-term 5-10yr]
[Coverage check]      vs FLAG]            [Long-term 2050]
                     [Sector assess]      [Net-zero NZ-C]

         |                                      |
         v                                      v

Phase 4              Phase 5
CRITERIA        ---> SUBMISSION
VALIDATION           PACKAGE
[42 criteria]        [SBTi template]
[C1-C28 + NZ-C1      [Supporting docs]
 to NZ-C14]           [Export format]
```

### External Assurance Workflow (5 Phases)

```
Phase 1              Phase 2              Phase 3
SCOPE           ---> EVIDENCE        ---> WORKPAPER
DEFINITION           COLLECTION           GENERATION
[Limited vs          [Source data]         [15 workpapers]
 Reasonable]          [Methodology docs]   [Big 4 format]
[Boundary]            [Calculation traces] [WP-100 to
[Materiality]         [Control docs]        WP-1500]

         |                                      |
         v                                      v

Phase 4              Phase 5
CONTROL         ---> ASSURANCE
TESTING              PACKAGE
[Reconciliation]     [Management
[Analytical review]   assertion letter]
[Sample testing]     [Evidence index]
[60 items minimum]   [Findings register]
```

---

## Data Flow Diagrams

### Enterprise Data Ingestion Flow

```
External Data Sources                    GreenLang Data Tier
+--------------------+                 +--------------------+
|                    |                 |                    |
| SAP S/4HANA -------+-- OData/RFC -->| DATA-003 ERP       |
|   MM, FI, CO, SD,  |                |  Connector         |
|   PM, HCM, TM      |                |    |               |
|                    |                 |    v               |
| Oracle ERP Cloud ---+-- REST API -->| Data Transform     |
|   Procurement,      |                |   - Unit normalize |
|   Financial, SCM    |                |   - Currency conv  |
|                    |                 |   - Field mapping  |
| Workday HCM -------+-- REST API -->|    |               |
|   Headcount, Travel |                |    v               |
|                    |                 | Data Quality       |
| CDP Supply Chain ---+-- CDP API --->|   - DATA-010       |
|   Supplier scores   |                |     Profiler       |
|                    |                 |   - DATA-011       |
| Manual Upload ------+-- CSV/XLSX -->|     Dedup          |
|   Utility bills,    |                |   - DATA-013       |
|   meter readings    |                |     Outlier        |
+--------------------+                |   - DATA-014       |
                                      |     Gap Fill       |
                                      |    |               |
                                      |    v               |
                                      | PostgreSQL +       |
                                      | TimescaleDB        |
                                      | (15 pack tables)   |
                                      +--------------------+
```

### Calculation Flow

```
Carbon Data Store
    |
    v
Enterprise Baseline Engine
    |
    +-- Per Entity Loop (parallelizable)
    |       |
    |       +-- Scope 1: MRV-001 to MRV-008
    |       |     (8 agents, per-source detail)
    |       |
    |       +-- Scope 2: MRV-009 to MRV-013
    |       |     (dual reporting, reconciliation)
    |       |
    |       +-- Scope 3: MRV-014 to MRV-028
    |       |     (all 15 categories, per-method)
    |       |
    |       +-- MRV-029: Category Mapper
    |       |     (route data to correct agents)
    |       |
    |       +-- MRV-030: Audit Trail
    |             (SHA-256 per calculation)
    |
    v
Materiality Assessment
    |
    +-- Categories > 1%: full activity-based required
    +-- Categories 0.1-1%: average-data acceptable
    +-- Categories < 0.1%: may exclude with justification
    +-- Total exclusions < 5% of Scope 3
    |
    v
Data Quality Matrix
    |
    +-- Per-category DQ level (1-5)
    +-- Per-entity DQ level (1-5)
    +-- Weighted average DQ score
    +-- Confidence intervals
    |
    v
Multi-Entity Consolidation Engine
    |
    +-- Apply consolidation approach
    +-- Intercompany elimination
    +-- Base year recalculation check
    +-- Reconciliation
    |
    v
Consolidated Enterprise Baseline Result
    |
    +-- Total CO2e by scope
    +-- Per-entity breakdown
    +-- DQ matrix
    +-- Provenance hash chain
```

### Report Generation Flow

```
Consolidated Baseline Result
    |
    +----> GHG Inventory Report Template
    |        |
    |        +-- MD / HTML / JSON / XLSX
    |        +-- 20-40 page enterprise report
    |
    +----> SBTi Target Submission Template
    |        |
    |        +-- 42-criteria matrix
    |        +-- Pathway visualization
    |
    +----> CDP Climate Response Template
    |        |
    |        +-- C0-C15 modules auto-populated
    |        +-- A-list scoring optimization
    |
    +----> TCFD Report Template
    |        |
    |        +-- 4 pillars: Governance, Strategy,
    |        |   Risk Management, Metrics & Targets
    |
    +----> Executive Dashboard Template
    |        |
    |        +-- 15-20 KPIs, traffic lights
    |        +-- Board-ready single page
    |
    +----> Regulatory Filings Template
             |
             +-- SEC Climate Rule (S-X Article 14)
             +-- CSRD ESRS E1 (climate chapter)
             +-- California SB 253
             +-- ISO 14064-1 GHG statement
             +-- CDP questionnaire extract
```

---

## Database Schema Overview

### Pack-Specific Tables (15)

| Table | Purpose | Key Columns | Indexes |
|-------|---------|-------------|---------|
| `ent_corporate_profiles` | Enterprise profiles | `org_id`, `sector`, `consolidation_approach`, `revenue`, `employees` | PK on `org_id` |
| `ent_entity_hierarchy` | Entity tree structure | `entity_id`, `parent_id`, `ownership_pct`, `control_type`, `effective_date` | PK on `entity_id`, FK to parent |
| `ent_baselines` | GHG baseline records | `baseline_id`, `org_id`, `entity_id`, `reporting_year`, `scope`, `category`, `tco2e`, `dq_level` | Composite on (`org_id`, `reporting_year`, `entity_id`) |
| `ent_sbti_targets` | SBTi target records | `target_id`, `org_id`, `pathway`, `base_year`, `target_year`, `criteria_json` | PK on `target_id` |
| `ent_scenarios` | Scenario records | `scenario_id`, `org_id`, `scenario_type`, `monte_carlo_runs`, `results_json` | PK on `scenario_id` |
| `ent_carbon_pricing` | Carbon pricing config | `pricing_id`, `org_id`, `price_usd`, `escalation_pct`, `allocations_json` | PK on `pricing_id` |
| `ent_avoided_emissions` | Scope 4 records | `avoided_id`, `org_id`, `product_id`, `baseline_tco2e`, `product_tco2e`, `avoided_tco2e` | PK on `avoided_id` |
| `ent_supply_chain` | Supplier master | `supplier_id`, `org_id`, `tier`, `spend_usd`, `tco2e`, `cdp_score`, `sbti_status` | PK on `supplier_id`, idx on `tier` |
| `ent_consolidation` | Consolidation records | `consol_id`, `org_id`, `approach`, `eliminations_json`, `consolidated_tco2e` | PK on `consol_id` |
| `ent_financial_integration` | Carbon financials | `financial_id`, `org_id`, `bu_id`, `carbon_charge`, `adjusted_npv`, `cbam_exposure` | PK on `financial_id` |
| `ent_assurance` | Assurance records | `assurance_id`, `org_id`, `assurance_level`, `workpapers_json`, `findings_json` | PK on `assurance_id` |
| `ent_data_quality` | DQ scoring | `dq_id`, `org_id`, `entity_id`, `category`, `dq_level`, `improvement_plan` | Composite on (`org_id`, `entity_id`) |
| `ent_regulatory_filings` | Regulatory filings | `filing_id`, `org_id`, `framework`, `status`, `submission_date` | PK on `filing_id` |
| `ent_erp_connections` | ERP configs | `connection_id`, `org_id`, `erp_type`, `host`, `schedule`, `data_mapping_json` | PK on `connection_id` |
| `ent_audit_trail` | Audit log | `trail_id`, `org_id`, `action`, `user_id`, `timestamp`, `hash_chain` | PK on `trail_id`, idx on `timestamp` |

### Row-Level Security (RLS)

All pack tables enforce row-level security based on the user's organization and entity assignments:

```sql
-- Entity data owners can only see their assigned entities
CREATE POLICY entity_isolation ON ent_baselines
    USING (
        entity_id = ANY(
            SELECT entity_id FROM user_entity_assignments
            WHERE user_id = current_setting('app.user_id')::uuid
        )
        OR current_setting('app.role') IN ('cso', 'enterprise_admin', 'analyst', 'auditor')
    );
```

---

## ERP Integration Architecture

### Connector Pattern

```
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
| ERP System       |     | GreenLang        |     | Carbon Data      |
| (SAP/Oracle/     |<--->| ERP Connector    |---->| Store            |
|  Workday)        |     |                  |     | (PostgreSQL)     |
|                  |     | - Authentication |     |                  |
|                  |     | - Data Extract   |     |                  |
|                  |     | - Transform      |     |                  |
|                  |     | - Validate       |     |                  |
|                  |     | - Load           |     |                  |
+------------------+     +------------------+     +------------------+
                               |
                               v
                         +------------------+
                         | Reverse          |
                         | Integration      |
                         | (optional)       |
                         |                  |
                         | - GL carbon cost |
                         | - BU allocation  |
                         | - Carbon-adj P&L |
                         +------------------+
```

### Data Extraction Schedule

| Data Type | Default Frequency | Configurable | Typical Volume |
|-----------|------------------|--------------|----------------|
| Energy consumption | Daily | Yes | 1,000-10,000 records/day |
| Procurement spend | Weekly | Yes | 10,000-100,000 records/week |
| Fleet fuel cards | Daily | Yes | 500-5,000 records/day |
| Travel bookings | Weekly | Yes | 1,000-10,000 records/week |
| Employee headcount | Monthly | Yes | One snapshot/month |
| Waste manifests | Monthly | Yes | 100-1,000 records/month |
| Capital expenditure | Monthly | Yes | 50-500 records/month |
| Refrigerant logs | Quarterly | Yes | 10-100 records/quarter |

---

## Security Architecture

### Authentication Flow

```
User Request
    |
    v
API Gateway (Kong) --> JWT RS256 Validation
    |
    v
SSO Integration (SAML 2.0 / OIDC)
    |
    v
RBAC Evaluation (8 enterprise roles)
    |
    +-- Role-based: enterprise_admin / cso / sustainability_manager / etc.
    +-- Entity-based: entity_data_owner restricted to assigned entities
    +-- Data-based: RLS policies on all pack tables
    |
    v
Request Processing
    |
    v
Audit Trail Entry (SEC-005, immutable, hash-chained)
```

### Data Protection Layers

```
Layer 1: Network (TLS 1.3)
    |
Layer 2: API Gateway (Kong, rate limiting, WAF)
    |
Layer 3: Authentication (JWT RS256, SSO)
    |
Layer 4: Authorization (RBAC, 8 roles, RLS)
    |
Layer 5: Encryption at Rest (AES-256-GCM)
    |
Layer 6: Secrets Management (HashiCorp Vault)
    |
Layer 7: Audit Trail (immutable, cryptographic chain)
    |
Layer 8: Provenance (SHA-256 on all calculations)
```

---

## Performance Architecture

### Performance Targets

| Operation | Target | Architecture Support |
|-----------|--------|---------------------|
| Enterprise baseline (100 entities, 15 S3 cats) | <4 hours | Parallel entity processing, batch mode |
| Single entity baseline | <15 minutes | Direct engine execution |
| Multi-entity consolidation (100+ entities) | <30 minutes | Efficient tree traversal, cached factors |
| Monte Carlo (10,000 runs, 3 scenarios) | <30 minutes | Multi-worker parallelism (configurable) |
| SBTi 42-criteria validation | <10 minutes | In-memory validation, no I/O |
| CDP questionnaire auto-population | <30 minutes | Template rendering from cached results |
| Board climate report | <15 minutes | Pre-computed KPIs, incremental refresh |
| Supply chain heatmap (50,000 suppliers) | <60 minutes | Batch processing, materialized views |
| API response (p95) | <2 seconds | Redis caching, connection pooling |
| Batch throughput | 1,000 entity-years/hour | Horizontal pod scaling |

### Caching Strategy

```
Cache Layer 1: Redis (hot data)
    |
    +-- Emission factors (TTL: 24 hours)
    +-- SDA sector benchmarks (TTL: 24 hours)
    +-- Entity hierarchy (TTL: 1 hour)
    +-- Recent calculation results (TTL: 1 hour)
    +-- Session data (TTL: 30 minutes)
    |
    Target hit rate: 85%

Cache Layer 2: In-memory (per-engine)
    |
    +-- Loaded emission factor tables
    +-- GWP values (IPCC AR6)
    +-- Unit conversion constants
    +-- Validation rule sets
```

### Memory Management

| Component | Memory Ceiling | Burst Limit |
|-----------|---------------|-------------|
| Enterprise Baseline Engine | 4,096 MB | 8,192 MB |
| Scenario Modeling Engine | 4,096 MB | 16,384 MB (Monte Carlo) |
| Supply Chain Engine | 4,096 MB | 8,192 MB |
| Multi-Entity Consolidation | 4,096 MB | 8,192 MB |
| Other Engines | 2,048 MB | 4,096 MB |

---

## Scalability Design

### Horizontal Scaling

```
Kubernetes Cluster
    |
    +-- Pack-027 Deployment (replicas: 3, auto-scale to 10)
    |       |
    |       +-- Pod 1: API serving + lightweight calculations
    |       +-- Pod 2: API serving + lightweight calculations
    |       +-- Pod 3: API serving + lightweight calculations
    |
    +-- Pack-027 Workers (replicas: 2, auto-scale to 8)
    |       |
    |       +-- Worker 1: Batch baseline calculation
    |       +-- Worker 2: Monte Carlo simulation
    |
    +-- PostgreSQL (3-node HA cluster)
    |
    +-- Redis (3-node sentinel cluster)
```

### Batch Processing Architecture

For enterprise baselines with 100+ entities:

```
Batch Job Queue (Redis)
    |
    +-- Entity Batch 1 (entities 1-20)  --> Worker 1
    +-- Entity Batch 2 (entities 21-40) --> Worker 2
    +-- Entity Batch 3 (entities 41-60) --> Worker 1 (after batch 1)
    +-- ...
    |
    v
All entity results collected
    |
    v
Consolidation Engine (single-threaded, sequential)
    |
    v
Report Generation
```

---

## Observability Architecture

### Metrics (Prometheus)

| Metric | Type | Labels |
|--------|------|--------|
| `pack027_baseline_duration_seconds` | Histogram | `entity_id`, `scope` |
| `pack027_entities_processed_total` | Counter | `status` |
| `pack027_monte_carlo_runs_total` | Counter | `scenario` |
| `pack027_dq_score` | Gauge | `entity_id`, `category` |
| `pack027_cache_hit_ratio` | Gauge | `cache_layer` |
| `pack027_erp_extraction_duration` | Histogram | `erp_type`, `data_type` |
| `pack027_provenance_hashes_generated` | Counter | `engine` |

### Tracing (OpenTelemetry)

All workflow phases and engine calculations produce OpenTelemetry spans:

```
Workflow Span (comprehensive_baseline_workflow)
    |
    +-- Phase 1 Span (entity_mapping)
    |       +-- Entity registration spans
    |
    +-- Phase 2 Span (data_collection)
    |       +-- ERP extraction spans (per system)
    |       +-- Manual upload spans
    |
    +-- Phase 3 Span (quality_assurance)
    |       +-- DATA agent spans (profiler, dedup, outlier)
    |
    +-- Phase 4 Span (calculation)
    |       +-- Per-entity spans
    |           +-- Per-MRV-agent spans (30 agents)
    |
    +-- Phase 5 Span (consolidation)
    |       +-- Elimination spans
    |
    +-- Phase 6 Span (reporting)
            +-- Template rendering spans
```

### Alerting

| Alert | Condition | Severity |
|-------|-----------|----------|
| Baseline calculation failure | Any entity calculation fails | P1 |
| DQ score degradation | Weighted DQ > 3.0 | P2 |
| ERP extraction timeout | Extraction exceeds 4x normal duration | P2 |
| Monte Carlo OOM | Memory exceeds 90% of ceiling | P1 |
| Provenance hash mismatch | Re-calculation produces different hash | P1 (data integrity) |
| Stale data | Entity data older than configured freshness | P3 |

---

## Disaster Recovery

### Recovery Point Objective (RPO)

| Data Type | RPO | Mechanism |
|-----------|-----|-----------|
| Configuration data | 0 (real-time) | PostgreSQL synchronous replication |
| Calculation results | 1 hour | PostgreSQL WAL archiving |
| Audit trail | 0 (real-time) | Append-only, replicated |
| ERP extraction data | 24 hours | Re-extract from ERP |
| Report outputs | 24 hours | Regenerate from calculation results |
| Emission factor databases | N/A (immutable) | S3 versioned backup |

### Recovery Time Objective (RTO)

| Scenario | RTO | Procedure |
|----------|-----|-----------|
| Single pod failure | <30 seconds | K8s auto-restart |
| Database failover | <5 minutes | PostgreSQL automatic failover |
| Full cluster recovery | <2 hours | Terraform rebuild + data restore |
| Data corruption | <4 hours | Point-in-time recovery from WAL |

---

## Appendix: Technology Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| Models | Pydantic | v2 |
| Database | PostgreSQL + TimescaleDB | 16 |
| Cache | Redis | 7+ |
| Container | Docker | 24+ |
| Orchestration | Kubernetes (EKS) | 1.28+ |
| API Gateway | Kong | 3.4+ |
| Secrets | HashiCorp Vault | 1.15+ |
| Observability | Prometheus + Grafana + OpenTelemetry | Latest |
| CI/CD | GitHub Actions | Latest |
| Object Storage | S3 | N/A |
| IaC | Terraform | 1.6+ |
