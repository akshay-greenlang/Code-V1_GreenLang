# PACK-026 SME Net Zero - Field Name Mapping Reference
## Correct Pydantic Field Names for All 8 Engines

**Date**: 2026-03-18
**Purpose**: Reference guide for correct field names to fix all test fixtures

---

## Engine 1: SMEBaselineEngine

### Input Model: SMEBaselineInput

**REQUIRED FIELDS:**
- `entity_name`: str (min 1, max 300 chars)
- `reporting_year`: int (2015-2100)
- `sector`: SMESector enum
- `company_size`: CompanySize enum (default: "small")
- `headcount`: int (default: 10, range: 1-250)

**OPTIONAL FIELDS:**
- `revenue_usd`: Optional[Decimal] (>= 0)
- `region`: str (default: "GLOBAL_AVG")
- `data_tier`: DataTier enum (default: "bronze")
- `fuel_entries`: List[SMEFuelEntry] (default: [])
- `electricity_entries`: List[SMEElectricityEntry] (default: [])
- `refrigerant_entries`: List[SMERefrigerantEntry] (default: [])
- `vehicle_entries`: List[SMEVehicleEntry] (default: [])
- `spend_entries`: List[SMESpendEntry] (default: [])
- `total_annual_spend_usd`: Optional[Decimal]

**SMESector ENUM VALUES:**
- agriculture, manufacturing, construction, wholesale_retail
- transport_logistics, accommodation_food, information_technology
- financial_services, professional_services, healthcare, education
- arts_entertainment, other_services

**CompanySize ENUM VALUES:**
- micro, small, medium

**DataTier ENUM VALUES:**
- bronze, silver, gold

### Output Model: SMEBaselineResult

**Key Fields:**
- `scope1`: ScopeBreakdown
- `scope2`: ScopeBreakdown
- `scope3`: ScopeBreakdown
- `total_tco2e`: Decimal
- `accuracy_band`: AccuracyBand
- `intensity`: IntensityMetrics
- `data_quality`: DataQualityAssessment

---

## Engine 2: SimplifiedTargetEngine

### Input Model: TargetInput

**REQUIRED FIELDS:**
- `entity_name`: str
- `base_year`: int (>= 2018, <= 2025)
- `base_year_scope1_tco2e`: Decimal
- `base_year_scope2_tco2e`: Decimal
- `current_year`: int

**OPTIONAL FIELDS:**
- `base_year_scope3_tco2e`: Decimal (default: 0)
- `base_year_scope3_total_tco2e`: Decimal (default: 0)
- `current_scope1_tco2e`: Optional[Decimal]
- `current_scope2_tco2e`: Optional[Decimal]
- `current_scope3_tco2e`: Optional[Decimal]
- `scope3_categories_included`: List[ScopeInclusion]
- `commitment_type`: TargetCommitment (default: "sme_climate_hub")
- `custom_near_term_year`: Optional[int]

---

## Engine 3: QuickWinsEngine

### Input Model: QuickWinsInput

**REQUIRED FIELDS:**
- `entity_name`: str
- `headcount`: int (1-250)

**OPTIONAL FIELDS:**
- `sector`: SMESectorFilter (default: "all")
- `total_emissions_tco2e`: Decimal (default: 0)
- `scope1_tco2e`: Decimal (default: 0)
- `scope2_tco2e`: Decimal (default: 0)
- `scope3_tco2e`: Decimal (default: 0)
- `annual_budget_usd`: Optional[Decimal]
- `max_difficulty`: DifficultyLevel (default: "hard")
- `max_payback_years`: Decimal (default: 5.0)
- `top_n`: int (default: 10, range: 1-54)
- `exclude_categories`: List[ActionCategory] (default: [])

---

## Engine 4: Scope3EstimatorEngine

### Input Model: Scope3EstimatorInput

**REQUIRED FIELDS:**
- `entity_name`: str
- `reporting_year`: int (2015-2100)

**OPTIONAL FIELDS:**
- `industry`: IndustryType (default: "general")
- `spend_entries`: List[SpendEntry] (default: [])
- `commuting_estimate`: Optional[CommutingEstimateInput]
- `headcount`: int (default: 10, range: 1-250)
- `data_source_type`: DataSourceType (default: "manual_entry")
- `include_optional_categories`: bool (default: False)

**SpendEntry Fields:**
- `category`: Optional[Scope3Category]
- `amount`: Decimal (required)
- `currency`: SpendCurrency (default: "usd")
- `description`: str (default: "")
- `accounting_category`: Optional[str]
- `data_source`: DataSourceType (default: "manual_entry")
- `custom_factor`: Optional[Decimal]

---

## Engine 5: ActionPrioritizationEngine

### Input Model: PrioritizationInput

**REQUIRED FIELDS:**
- `entity_name`: str
- `actions`: List[ActionInput] (1-10 items)

**OPTIONAL FIELDS:**
- `total_emissions_tco2e`: Decimal (default: 0)
- `annual_budget_usd`: Optional[Decimal]
- `discount_rate`: Decimal (default: 0.08)
- `npv_horizon_years`: int (default: 5, range: 1-20)
- `carbon_price_usd_per_tco2e`: Optional[Decimal]

**ActionInput Fields:**
- `action_id`: str (auto-generated UUID)
- `name`: str (required)
- `capex_usd`: Decimal (required, >= 0)
- `annual_tco2e_reduction`: Decimal (required, >= 0)
- `description`: str (default: "")
- `scope`: ActionScope (default: "scope_1_2")
- `annual_opex_change_usd`: Decimal (default: 0)
- `annual_savings_usd`: Decimal (default: 0)
- `ease_of_implementation`: ActionEase (default: "moderate")
- `implementation_months`: int (default: 3, range: 1-36)
- `useful_life_years`: int (default: 10, range: 1-30)
- `grant_pct`: Decimal (default: 0, range: 0-100)
- `notes`: str (default: "", max: 500)

### Output Model: PrioritizationResult

**Key Access Patterns:**
- `result.actions`: List[PrioritizedAction]
- `result.roadmap`: RoadmapSummary
- `result.roadmap.total_npv_usd`: Decimal (NOT result.total_npv_eur!)
- `result.roadmap.total_capex_usd`: Decimal
- `result.roadmap.year_1_capex_usd`: Decimal

---

## Engine 6: CostBenefitEngine

### Input Model: CostBenefitInput

**REQUIRED FIELDS:**
- `entity_name`: str
- `items`: List[CostBenefitItem] (1-20 items)

**OPTIONAL FIELDS:**
- `discount_rate`: Decimal (default: 0.08)
- `analysis_horizon_years`: int (default: 10, range: 1-20)
- `inflation_rate`: Decimal (default: 0.02)

**CostBenefitItem Fields:**
- `name`: str (required)
- `capex_usd`: Decimal (required, >= 0)
- `item_id`: str (auto-generated UUID)
- `category`: CostCategory (default: "energy_efficiency")
- `annual_opex_savings_usd`: Decimal (default: 0)
- `annual_revenue_increase_usd`: Decimal (default: 0)
- `annual_tco2e_reduction`: Decimal (default: 0)
- `grant_pct`: Decimal (default: 0, range: 0-100)
- `grant_name`: str (default: "")
- `useful_life_years`: int (default: 10, range: 1-30)
- `implementation_months`: int (default: 3, range: 1-36)
- `maintenance_cost_annual_usd`: Decimal (default: 0)
- `carbon_price_usd_per_tco2e`: Decimal (default: 0)
- `residual_value_pct`: Decimal (default: 0, range: 0-100)
- `notes`: str (default: "", max: 500)

---

## Engine 7: GrantFinderEngine

### Input Model: GrantFinderInput

**REQUIRED FIELDS:**
- `entity_name`: str
- `country`: str (2-50 chars)

**OPTIONAL FIELDS:**
- `industry`: IndustryCode (default: "any")
- `company_size`: CompanySize (default: "small")
- `region_code`: Optional[str]
- `project_types`: List[ProjectType] (default: ["energy_efficiency"])
- `total_emissions_tco2e`: Decimal (default: 0)
- `scope1_pct`: Decimal (default: 33)
- `scope2_pct`: Decimal (default: 33)
- `scope3_pct`: Decimal (default: 34)
- `project_budget_usd`: Optional[Decimal]
- `top_n`: int (default: 5, range: 1-20)
- `include_tax_incentives`: bool (default: True)

---

## Engine 8: CertificationReadinessEngine

### Input Model: CertificationReadinessInput

**REQUIRED FIELDS:**
- `entity_name`: str

**OPTIONAL FIELDS:**
- `country`: str (default: "GB")
- `headcount`: int (default: 10, range: 1-250)
- `assessment_data`: DimensionInput (default_factory)
- `preferred_pathways`: List[CertificationPathway] (default: [])
- `current_certifications`: List[str] (default: [])
- `supply_chain_pressure`: bool (default: False)

**DimensionInput Fields:**
- `has_baseline`: bool (default: False)
- `baseline_data_tier`: str (default: "none")
- `baseline_scope_coverage`: str (default: "none")
- `has_targets`: bool (default: False)
- `target_type`: str (default: "none")
- `target_year`: Optional[int]
- `has_action_plan`: bool (default: False)
- `actions_identified`: int (default: 0)
- `actions_implemented`: int (default: 0)
- `has_board_oversight`: bool (default: False)
- `has_climate_policy`: bool (default: False)
- `has_sustainability_role`: bool (default: False)
- `has_public_disclosure`: bool (default: False)
- `has_reporting_process`: bool (default: False)
- `has_third_party_verification`: bool (default: False)
- `notes`: str (default: "")

---

## Common Field Name Mapping (OLD → NEW)

### Frequently Misused Fields:

| OLD (WRONG) | NEW (CORRECT) | Used In |
|-------------|---------------|---------|
| `sme_tier` | `company_size` | Baseline, Grant Finder, Cert Readiness |
| `employee_count` | `headcount` | All engines |
| `annual_revenue_eur` | `revenue_usd` | Baseline |
| `annual_spend_eur` | `annual_spend_usd` (or specific) | Baseline, Scope3 |
| `upfront_cost` | `capex_usd` | Action Prioritization, Cost Benefit |
| `upfront_cost_eur` | `capex_usd` | Action Prioritization, Cost Benefit |
| `reduction_tco2e` | `annual_tco2e_reduction` | All action/item models |
| `co2_reduction_tco2e` | `annual_tco2e_reduction` | All action/item models |
| `annual_savings_eur` | `annual_savings_usd` | Quick Wins, Actions |
| `method` | `data_tier` | Baseline |
| `method_used` | `data_tier` | Baseline |
| `accuracy_band_pct` | Use `accuracy_band.low_pct` and `.high_pct` | Baseline result |
| `food_beverage` | `accommodation_food` | SMESector enum |
| `technology` | `information_technology` | SMESector enum |
| `transport` | `transport_logistics` | SMESector enum |

### Currency Convention:
- All financial amounts use `_usd` suffix (not `_eur`)
- Examples: `capex_usd`, `annual_savings_usd`, `revenue_usd`

### Emission Fields:
- Always use `annual_tco2e_reduction` for emissions reduction
- Use `total_tco2e` for total emissions
- Use `scope1_tco2e`, `scope2_tco2e`, `scope3_tco2e` for scope breakdowns

---

## Result Access Patterns

### ActionPrioritizationResult:
```python
# WRONG:
result.total_npv_eur
result.total_reduction_tco2e

# CORRECT:
result.roadmap.total_npv_usd
result.roadmap.cumulative_reduction_tco2e
```

### SMEBaselineResult:
```python
# WRONG:
result.accuracy_band_pct

# CORRECT:
result.accuracy_band.lower_bound_tco2e
result.accuracy_band.upper_bound_tco2e
result.accuracy_band.confidence_pct
```

### QuickWinsResult:
```python
# CORRECT:
result.actions  # List[QuickWinAction]
result.summary.total_reduction_tco2e
result.summary.total_implementation_cost_usd
```

---

## Test Fixture Quick Reference

### Minimal Bronze Baseline:
```python
SMEBaselineInput(
    entity_name="Test Company",
    reporting_year=2025,
    sector="professional_services",  # SMESector enum
    company_size="small",            # CompanySize enum
    headcount=25,
    data_tier="bronze",              # DataTier enum
)
```

### Minimal Action Prioritization:
```python
ActionInput(
    name="LED Upgrade",
    capex_usd=Decimal("5000"),
    annual_tco2e_reduction=Decimal("2.5"),
    annual_savings_usd=Decimal("1200"),
)
```

### Minimal Grant Finder:
```python
GrantFinderInput(
    entity_name="Test Company",
    country="GB",
    company_size="small",
    industry="manufacturing",
    project_types=["energy_efficiency"],
)
```

---

**END OF FIELD MAPPING REFERENCE**
