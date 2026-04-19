# PACK-027 Enterprise Net Zero Pack -- Multi-Entity Guide

**Pack ID:** PACK-027-enterprise-net-zero
**Version:** 1.0.0
**Date:** 2026-03-19
**Author:** GreenLang Platform Engineering

---

## Table of Contents

1. [Introduction](#introduction)
2. [Entity Hierarchy Setup](#entity-hierarchy-setup)
3. [Consolidation Approaches](#consolidation-approaches)
4. [Entity Registration](#entity-registration)
5. [Intercompany Eliminations](#intercompany-eliminations)
6. [Reporting Currency and FX](#reporting-currency-and-fx)
7. [Minority Stakes and Associates](#minority-stakes-and-associates)
8. [Mergers and Acquisitions](#mergers-and-acquisitions)
9. [Divestitures](#divestitures)
10. [Base Year Recalculation](#base-year-recalculation)
11. [Entity-Level Reporting](#entity-level-reporting)
12. [Multi-Entity Rollup Workflow](#multi-entity-rollup-workflow)
13. [Data Collection by Entity](#data-collection-by-entity)
14. [Troubleshooting](#troubleshooting)

---

## Introduction

Large enterprises operate through complex corporate structures with 10 to 500+ subsidiaries, joint ventures, associates, and special-purpose vehicles across dozens of countries. The GHG Protocol Corporate Accounting and Reporting Standard (Chapter 3) requires organizations to define their organizational boundary using one of three consolidation approaches and apply it consistently.

PACK-027's Multi-Entity Consolidation Engine handles:

- Up to **500 entities** in a single corporate hierarchy
- **3 consolidation approaches**: financial control, operational control, equity share
- **Intercompany elimination** to prevent double-counting
- **Mid-year acquisitions and divestitures** with pro-rata allocation
- **Base year recalculation** triggers (5% significance threshold)
- **Multi-level hierarchies** (parent > subsidiary > sub-subsidiary)
- **Mixed ownership structures** (wholly-owned, majority, JV, associate)

---

## Entity Hierarchy Setup

### Defining Your Corporate Structure

```python
from integrations.multi_entity_orchestrator import MultiEntityOrchestrator

orchestrator = MultiEntityOrchestrator(config=config)

# Define the entity hierarchy
hierarchy = orchestrator.define_hierarchy(
    parent={
        "entity_id": "parent-001",
        "name": "GlobalMfg Corp (Parent)",
        "country": "US",
        "ownership_pct": 100,
        "control_type": "financial",
    },
    children=[
        {
            "entity_id": "sub-alpha",
            "name": "SubCo Alpha GmbH",
            "country": "DE",
            "ownership_pct": 100,
            "control_type": "financial",
            "parent_id": "parent-001",
        },
        {
            "entity_id": "sub-beta",
            "name": "SubCo Beta Ltd",
            "country": "GB",
            "ownership_pct": 100,
            "control_type": "financial",
            "parent_id": "parent-001",
            "children": [
                {
                    "entity_id": "sub-beta-1",
                    "name": "Beta Manufacturing Sp.z" ,
                    "country": "PL",
                    "ownership_pct": 80,
                    "control_type": "financial",
                    "parent_id": "sub-beta",
                },
            ],
        },
        {
            "entity_id": "jv-gamma",
            "name": "JV Gamma SA",
            "country": "FR",
            "ownership_pct": 51,
            "control_type": "financial",  # Financial control despite 51%
            "parent_id": "parent-001",
        },
        {
            "entity_id": "assoc-delta",
            "name": "Associate Delta Inc",
            "country": "JP",
            "ownership_pct": 33,
            "control_type": "none",  # No control, equity share only
            "parent_id": "parent-001",
        },
    ],
)

# Visualize hierarchy
orchestrator.print_hierarchy(hierarchy)
```

Output:
```
GlobalMfg Corp (Parent) [US, 100%, financial control]
  +-- SubCo Alpha GmbH [DE, 100%, financial control]
  +-- SubCo Beta Ltd [GB, 100%, financial control]
  |   +-- Beta Manufacturing Sp.z [PL, 80%, financial control]
  +-- JV Gamma SA [FR, 51%, financial control]
  +-- Associate Delta Inc [JP, 33%, no control -> equity share only]
```

### Entity Attributes

| Attribute | Description | Required |
|-----------|-------------|----------|
| `entity_id` | Unique identifier for the entity | Yes |
| `name` | Legal entity name | Yes |
| `country` | Country of incorporation (ISO 3166-1) | Yes |
| `ownership_pct` | Percentage ownership by parent (0-100) | Yes |
| `control_type` | `financial`, `operational`, or `none` | Yes |
| `parent_id` | ID of parent entity (null for group parent) | Yes |
| `effective_date` | Date ownership/control became effective | Yes |
| `end_date` | Date ownership/control ended (null if current) | No |
| `sector` | Entity sector (for SDA pathway if applicable) | No |
| `erp_system` | ERP system for this entity (SAP/Oracle/manual) | No |
| `data_owner` | User ID of entity data owner | No |
| `reporting_currency` | Local currency (ISO 4217) | No |

---

## Consolidation Approaches

### Choosing the Right Approach

| Approach | Rule | When to Use | Common for |
|----------|------|-------------|-----------|
| **Financial Control** | Include 100% of entities where you direct financial and operating policies | IFRS reporting; want alignment with financial statements | EU companies, IFRS reporters |
| **Operational Control** | Include 100% of entities where you have authority over operating policies | US GAAP preference; you operate the facilities | US companies, operators |
| **Equity Share** | Include emissions proportional to your equity ownership % | Most conservative; required by some regulators | Minority positions, JVs |

### Impact on Total Emissions

```python
from engines.multi_entity_consolidation_engine import MultiEntityConsolidationEngine

engine = MultiEntityConsolidationEngine(config=config)

# Compare all three approaches
comparison = engine.compare_approaches(
    entity_results=all_entity_results,
    hierarchy=hierarchy,
)

for approach in comparison.approaches:
    print(f"\n{approach.name}:")
    print(f"  Scope 1: {approach.scope1:,.0f} tCO2e")
    print(f"  Scope 2: {approach.scope2:,.0f} tCO2e")
    print(f"  Scope 3: {approach.scope3:,.0f} tCO2e")
    print(f"  Total: {approach.total:,.0f} tCO2e")
    print(f"  Entities included: {approach.entity_count}")
    print(f"  Delta vs. financial control: {approach.delta_pct:+.1f}%")
```

**Typical output:**
```
Financial Control:
  Scope 1: 85,000 tCO2e
  Scope 2: 35,000 tCO2e
  Scope 3: 250,000 tCO2e
  Total: 370,000 tCO2e
  Entities included: 4 (Alpha, Beta, Beta-1, Gamma)
  Delta vs. financial control: 0.0%

Operational Control:
  Scope 1: 70,000 tCO2e  (Gamma excluded if not operator)
  Scope 2: 28,000 tCO2e
  Scope 3: 235,000 tCO2e
  Total: 333,000 tCO2e
  Entities included: 3 (Alpha, Beta, Beta-1)
  Delta vs. financial control: -10.0%

Equity Share:
  Scope 1: 68,000 tCO2e  (Beta-1 at 80%, Gamma at 51%, Delta at 33%)
  Scope 2: 27,400 tCO2e
  Scope 3: 228,000 tCO2e
  Total: 323,400 tCO2e
  Entities included: 5 (all, at equity %)
  Delta vs. financial control: -12.6%
```

### Approach-Specific Rules

#### Financial Control

```python
# Include 100% if the parent has financial control
# Financial control = ability to direct financial and operating policies

for entity in hierarchy.entities:
    if entity.control_type == "financial":
        # Include 100% of emissions regardless of ownership %
        inclusion_pct = 100
    elif entity.control_type == "none":
        # Exclude from Scope 1+2; include in Scope 3 Cat 15
        inclusion_pct = 0  # (use equity share for Scope 3 Cat 15)
```

#### Operational Control

```python
# Include 100% if the parent operates the entity
for entity in hierarchy.entities:
    if entity.control_type == "operational":
        inclusion_pct = 100
    else:
        inclusion_pct = 0
```

#### Equity Share

```python
# Include proportional to ownership percentage
for entity in hierarchy.entities:
    inclusion_pct = entity.effective_ownership_pct
    # Note: effective ownership accounts for multi-level holdings
    # e.g., Beta-1: 100% (parent) * 100% (Beta) * 80% (Beta-1) = 80%
```

---

## Entity Registration

### Adding a New Entity

```python
# Register a new subsidiary
orchestrator.add_entity(
    entity_id="sub-epsilon",
    name="SubCo Epsilon Pty Ltd",
    country="AU",
    ownership_pct=100,
    control_type="financial",
    parent_id="parent-001",
    effective_date="2026-01-01",
    sector="manufacturing",
    erp_system="oracle",
    data_owner="user-au-001",
    reporting_currency="AUD",
)
```

### Modifying Ownership

```python
# Update ownership percentage (e.g., increasing stake in JV)
orchestrator.update_ownership(
    entity_id="jv-gamma",
    new_ownership_pct=60,  # Increased from 51%
    effective_date="2026-07-01",
    reason="Additional share purchase",
)

# This triggers base year recalculation assessment
```

### Deactivating an Entity

```python
# Mark entity as divested
orchestrator.deactivate_entity(
    entity_id="sub-alpha",
    end_date="2026-06-30",
    reason="Divestiture to AcquireCo",
    divestiture_price_usd=500_000_000,
)
```

---

## Intercompany Eliminations

### Why Eliminate Intercompany Emissions?

When Entity A's Scope 1 is also Entity B's Scope 3 Category 1 (purchased goods), including both would double-count emissions. Intercompany elimination removes this overlap.

### Common Intercompany Scenarios

| Scenario | Entity A (Seller) | Entity B (Buyer) | Elimination |
|----------|-------------------|-------------------|-------------|
| Internal electricity supply | Scope 1 (power generation) | Scope 2 (purchased electricity) | Remove from B's Scope 2 |
| Internal product supply | Scope 1 (manufacturing) | Scope 3 Cat 1 (purchased goods) | Remove from B's Scope 3 |
| Internal logistics | Scope 1 (fleet) | Scope 3 Cat 4 (transport) | Remove from B's Scope 3 |
| Shared services (data center) | Scope 2 (electricity) | Scope 3 Cat 1 (services) | Remove from B's Scope 3 |

### Registering Intercompany Transactions

```python
# Register intercompany transactions
engine.register_intercompany(
    transactions=[
        {
            "transaction_id": "IC-001",
            "from_entity": "jv-gamma",
            "to_entity": "sub-alpha",
            "type": "electricity_supply",
            "tco2e": 1_000,
            "scope_from": "scope1",
            "scope_to": "scope2",
            "description": "JV Gamma generates electricity consumed by SubCo Alpha",
            "value_usd": 500_000,
        },
        {
            "transaction_id": "IC-002",
            "from_entity": "sub-alpha",
            "to_entity": "sub-beta",
            "type": "product_supply",
            "tco2e": 2_500,
            "scope_from": "scope1",
            "scope_to": "scope3_cat1",
            "description": "SubCo Alpha manufactures components for SubCo Beta",
            "value_usd": 10_000_000,
        },
        {
            "transaction_id": "IC-003",
            "from_entity": "parent-001",
            "to_entity": "sub-beta",
            "type": "shared_services",
            "tco2e": 500,
            "scope_from": "scope2",
            "scope_to": "scope3_cat1",
            "description": "Parent provides shared IT services to SubCo Beta",
            "value_usd": 2_000_000,
        },
    ],
)

# Run consolidation with eliminations
result = engine.consolidate(
    entity_results=all_entity_results,
    hierarchy=hierarchy,
    approach="financial_control",
)

print(f"Sum before elimination: {result.sum_before_elimination:,.0f} tCO2e")
print(f"Total eliminations: {result.total_eliminations:,.0f} tCO2e")
print(f"Consolidated total: {result.consolidated_total:,.0f} tCO2e")

# Elimination details
for elim in result.elimination_entries:
    print(f"  {elim.transaction_id}: {elim.from_entity} -> {elim.to_entity}")
    print(f"    Eliminated: {elim.tco2e:,.0f} tCO2e from {elim.scope_to}")
```

---

## Reporting Currency and FX

### Currency Handling

PACK-027 handles multi-currency operations for spend-based Scope 3 calculations:

```python
# Configure currency settings
config.currency = "USD"  # Group reporting currency
config.fx_rates = {
    "EUR": 1.08,   # EUR/USD
    "GBP": 1.27,   # GBP/USD
    "JPY": 0.0067, # JPY/USD
    "AUD": 0.65,   # AUD/USD
    "PLN": 0.25,   # PLN/USD
    "CNY": 0.14,   # CNY/USD
}

# FX is applied to:
# - Spend-based Scope 3 calculations (local currency spend -> group currency -> EEIO factor)
# - Revenue intensity metrics (entity local revenue -> group revenue)
# - Carbon pricing allocation (local cost -> group reporting)
# - Financial integration (P&L in group currency)
```

### FX Rate Sources

| Source | Application | Update Frequency |
|--------|-------------|-----------------|
| Year-end spot rate | Balance sheet items | Annual |
| Average annual rate | P&L items (revenue, spend) | Annual (average of monthly) |
| Transaction date rate | Specific transactions | Per transaction |

---

## Minority Stakes and Associates

### Treatment by Consolidation Approach

| Entity Type | Financial Control | Operational Control | Equity Share |
|-------------|------------------|--------------------|----|
| 100% subsidiary | 100% Scope 1+2+3 | 100% Scope 1+2+3 | 100% Scope 1+2+3 |
| 80% subsidiary | 100% Scope 1+2+3 | 100% Scope 1+2+3 | 80% Scope 1+2+3 |
| 51% JV (with control) | 100% Scope 1+2+3 | 100% or 0% | 51% Scope 1+2+3 |
| 50/50 JV (no control) | 0% (Scope 3 Cat 15) | 0% (Scope 3 Cat 15) | 50% Scope 1+2+3 |
| 33% associate | 0% (Scope 3 Cat 15) | 0% (Scope 3 Cat 15) | 33% Scope 1+2+3 |
| 10% investment | 0% (Scope 3 Cat 15) | 0% (Scope 3 Cat 15) | 10% (Scope 3 Cat 15) |

### Associates as Scope 3 Category 15

When entities are excluded from Scope 1+2 consolidation, their emissions appear in Scope 3 Category 15 (Investments):

```python
# Automatic treatment for non-controlled entities
for entity in hierarchy.entities:
    if entity.control_type == "none" and config.consolidation_approach != "equity_share":
        # Entity excluded from Scope 1+2
        # Include in Scope 3 Cat 15 at equity share
        scope3_cat15_tco2e += entity.total_tco2e * entity.ownership_pct / 100
```

---

## Mergers and Acquisitions

### Mid-Year Acquisition

```python
# Register acquisition event
acquisition = orchestrator.register_acquisition(
    acquired_entity_id="acquired-co",
    acquired_entity_name="TargetCo Ltd",
    acquisition_date="2026-07-01",
    ownership_pct=100,
    control_type="financial",
    parent_id="parent-001",
    # Acquired entity may have sub-entities
    acquired_hierarchy=[
        {"entity_id": "target-sub-1", "name": "TargetCo Sub 1", "country": "DE"},
        {"entity_id": "target-sub-2", "name": "TargetCo Sub 2", "country": "FR"},
    ],
)

# Current year pro-rata calculation
# Acquired entity included from July 1 = 6/12 of full year
current_year_result = engine.consolidate_with_acquisition(
    pre_acquisition_results=existing_entity_results,
    acquired_entity_results=target_entity_results,
    acquisition_date="2026-07-01",
    pro_rata_method="monthly",  # 6/12 months
)

print(f"Pre-acquisition total: {current_year_result.pre_acquisition_tco2e:,.0f} tCO2e")
print(f"Acquired (pro-rata): {current_year_result.acquired_pro_rata_tco2e:,.0f} tCO2e")
print(f"Current year total: {current_year_result.total_tco2e:,.0f} tCO2e")
```

### Significance Assessment

```python
# Assess if base year recalculation is needed
significance = engine.assess_acquisition_significance(
    base_year_total=baseline.grand_total_tco2e,
    acquired_annual_total=target_annual_tco2e,
)

print(f"Acquired annual emissions: {significance.acquired_tco2e:,.0f} tCO2e")
print(f"Base year total: {significance.base_year_tco2e:,.0f} tCO2e")
print(f"Significance: {significance.significance_pct:.1f}%")
print(f"Exceeds 5% threshold: {significance.exceeds_threshold}")
print(f"Base year recalculation required: {significance.recalculation_required}")
```

---

## Divestitures

### Mid-Year Divestiture

```python
# Register divestiture event
divestiture = orchestrator.register_divestiture(
    entity_id="sub-alpha",
    divestiture_date="2026-04-01",
    reason="Strategic portfolio simplification",
)

# Current year pro-rata calculation
# Divested entity included only for January-March = 3/12
current_year_result = engine.consolidate_with_divestiture(
    full_year_results=all_entity_results,
    divested_entity_id="sub-alpha",
    divestiture_date="2026-04-01",
    pro_rata_method="monthly",
)

print(f"Divested entity (pro-rata 3/12): {current_year_result.divested_pro_rata_tco2e:,.0f} tCO2e")
```

---

## Base Year Recalculation

### Trigger Assessment

PACK-027 automatically assesses base year recalculation triggers during the annual inventory workflow:

```python
triggers = engine.assess_recalculation_triggers(
    base_year_result=baseline,
    current_year_events=[acquisition, divestiture, methodology_change],
)

for trigger in triggers.assessments:
    print(f"\nTrigger: {trigger.event_type}")
    print(f"  Description: {trigger.description}")
    print(f"  Impact: {trigger.impact_tco2e:,.0f} tCO2e")
    print(f"  Significance: {trigger.significance_pct:.1f}%")
    print(f"  Exceeds 5% threshold: {trigger.exceeds_threshold}")
    print(f"  Recalculation required: {trigger.recalculation_required}")
```

### Performing Recalculation

```python
if triggers.any_recalculation_required:
    recalculated = engine.recalculate_base_year(
        original_base_year=baseline,
        trigger_events=triggers.recalculation_events,
    )

    print(f"Original base year: {recalculated.original_total:,.0f} tCO2e")
    print(f"Recalculated base year: {recalculated.new_total:,.0f} tCO2e")
    print(f"Delta: {recalculated.delta_tco2e:,.0f} tCO2e ({recalculated.delta_pct:+.1f}%)")
    print(f"All interim years restated: {recalculated.interim_years_restated}")

    # Audit trail
    for entry in recalculated.audit_entries:
        print(f"  {entry.year}: {entry.old_total:,.0f} -> {entry.new_total:,.0f} tCO2e")
```

### GHG Protocol Recalculation Triggers

| Trigger | Description | Example |
|---------|-------------|---------|
| Structural change | Acquisition or divestiture | Acquiring a subsidiary with >5% of total emissions |
| Methodology change | New emission factors or calculation approach | Switching from DEFRA 2023 to DEFRA 2024 factors |
| Error correction | Discovery of significant error | Incorrect fuel data from ERP |
| Boundary change | Change in consolidation approach | Moving from operational to financial control |
| Outsourcing/insourcing | Transfer of emitting activities | Outsourcing manufacturing |

---

## Entity-Level Reporting

### Per-Entity Breakdown

```python
# Generate entity-level report
for entity in result.entities:
    print(f"\n{entity.name} ({entity.country}):")
    print(f"  Scope 1: {entity.scope1:,.0f} tCO2e")
    print(f"  Scope 2 (location): {entity.scope2_location:,.0f} tCO2e")
    print(f"  Scope 2 (market): {entity.scope2_market:,.0f} tCO2e")
    print(f"  Scope 3: {entity.scope3:,.0f} tCO2e")
    print(f"  Total: {entity.total:,.0f} tCO2e")
    print(f"  % of group: {entity.pct_of_group:.1f}%")
    print(f"  DQ score: {entity.dq_score:.1f}")
    print(f"  Data completeness: {entity.completeness_pct:.0f}%")
```

### Entity Comparison Dashboard

```python
from templates.executive_dashboard import ExecutiveDashboard

dashboard = ExecutiveDashboard()
entity_comparison = dashboard.render_entity_comparison(
    result=consolidated_result,
    sort_by="total_desc",
    top_n=20,
)
```

---

## Multi-Entity Rollup Workflow

### Running the Full Workflow

```python
from workflows.multi_entity_rollup_workflow import MultiEntityRollupWorkflow

workflow = MultiEntityRollupWorkflow(config=config)
result = workflow.execute()

# Phase 1: Entity Refresh
#   - Update entity hierarchy (new entities, ownership changes)
#   - Apply effective dates
#
# Phase 2: Data Validation
#   - Check per-entity data completeness
#   - Flag missing entities
#   - Escalate overdue submissions
#
# Phase 3: Entity Calculation
#   - Run baseline engine per entity (parallelized)
#   - 30 MRV agents per entity
#
# Phase 4: Elimination
#   - Apply intercompany eliminations
#   - Reconcile consolidated vs. sum of entities
#
# Phase 5: Consolidated Report
#   - Generate group-level report
#   - Per-entity breakdown
#   - Reconciliation appendix
```

---

## Data Collection by Entity

### Entity Data Collection Status

```python
status = orchestrator.get_collection_status()

for entity in status.entities:
    completeness = "COMPLETE" if entity.completeness >= 100 else "INCOMPLETE"
    print(f"  [{completeness}] {entity.name}: {entity.completeness:.0f}% complete")
    if entity.completeness < 100:
        for gap in entity.data_gaps:
            print(f"    Missing: {gap.data_type} ({gap.scope})")
```

### Data Collection Schedule

| Data Type | Deadline | Escalation |
|-----------|----------|-----------|
| Energy consumption | Reporting year + 30 days | Day 45: auto-email to data owner |
| Fuel consumption | Reporting year + 30 days | Day 45: auto-email to data owner |
| Procurement spend | Reporting year + 45 days | Day 60: escalate to sustainability manager |
| Travel and commuting | Reporting year + 30 days | Day 45: auto-email to data owner |
| Waste data | Reporting year + 30 days | Day 45: auto-email to data owner |
| Refrigerant logs | Reporting year + 30 days | Day 45: auto-email to data owner |
| Process emissions | Reporting year + 45 days | Day 60: escalate to sustainability manager |
| Financial data | Reporting year + 60 days | Day 75: escalate to CSO |

---

## Troubleshooting

### Entity Not Appearing in Consolidation

**Cause:** Entity effective date is after the reporting period.

**Resolution:**
```python
entity = orchestrator.get_entity("sub-epsilon")
print(f"Effective date: {entity.effective_date}")
print(f"Reporting year: {config.reporting_year}")
# Ensure effective_date <= reporting year end
```

### Ownership Chain Calculation Error

**Cause:** Multi-level ownership not resolved correctly.

**Resolution:**
```python
# Check effective ownership through the chain
chain = orchestrator.get_ownership_chain("sub-beta-1")
print(f"Direct ownership: {chain.direct_pct}%")
print(f"Effective ownership: {chain.effective_pct}%")
# Effective = 100% (parent) * 100% (Beta) * 80% (Beta-1) = 80%
```

### Consolidation Total Does Not Match Financial Statements

**Cause:** Different consolidation approach or entity boundary between GHG and financial reporting.

**Resolution:** Review entity-by-entity comparison:
```python
reconciliation = engine.reconcile_with_financial(
    ghg_entities=hierarchy.entities,
    financial_entities=financial_hierarchy.entities,
)

for diff in reconciliation.differences:
    print(f"  {diff.entity_name}: GHG={diff.in_ghg}, Financial={diff.in_financial}")
    print(f"    Reason: {diff.reason}")
```

### Intercompany Elimination Creates Negative Emissions

**Cause:** Elimination exceeds the receiving entity's reported emissions for that scope.

**Resolution:**
```python
# Cap elimination at the receiving entity's scope total
for elim in result.elimination_entries:
    if elim.tco2e > elim.receiving_entity_scope_total:
        print(f"  WARNING: Elimination {elim.transaction_id} ({elim.tco2e:,.0f} tCO2e)")
        print(f"           exceeds {elim.to_entity} {elim.scope_to} total")
        print(f"           ({elim.receiving_entity_scope_total:,.0f} tCO2e)")
        print(f"           Capping at entity total")
```
