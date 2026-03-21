# PACK-028 Sector Pathway Pack -- Integration Guide

**Pack ID:** PACK-028-sector-pathway
**Version:** 1.0.0
**Last Updated:** 2026-03-19

---

## Table of Contents

1. [Integration Architecture](#integration-architecture)
2. [SBTi SDA Integration](#sbti-sda-integration)
3. [IEA NZE Data Integration](#iea-nze-data-integration)
4. [IPCC AR6 Integration](#ipcc-ar6-integration)
5. [PACK-021 Integration](#pack-021-integration)
6. [MRV Agent Integration](#mrv-agent-integration)
7. [DATA Agent Integration](#data-agent-integration)
8. [Foundation Agent Integration](#foundation-agent-integration)
9. [Decarbonization Agent Integration](#decarbonization-agent-integration)
10. [Application Dependencies](#application-dependencies)
11. [API Authentication](#api-authentication)
12. [Webhook Configuration](#webhook-configuration)
13. [Data Flow Patterns](#data-flow-patterns)
14. [Integration Testing](#integration-testing)
15. [Troubleshooting Integrations](#troubleshooting-integrations)

---

## Integration Architecture

### Integration Overview

PACK-028 integrates with 10 external data sources and platform components through dedicated bridge modules.

```
+------------------------------------------------------------------+
|                      PACK-028 Core                                 |
|                                                                    |
|  +------------+  +------------+  +------------+  +------------+   |
|  | Sector     |  | Intensity  |  | Pathway    |  | Convergence|   |
|  | Classify   |  | Calculator |  | Generator  |  | Analyzer   |   |
|  +-----+------+  +-----+------+  +-----+------+  +-----+------+   |
|        |              |              |              |               |
|  +-----+--------------+--------------+--------------+------+       |
|  |                Integration Layer                        |       |
|  |  +----------+ +----------+ +----------+ +----------+   |       |
|  |  | SBTi SDA | | IEA NZE  | | IPCC AR6 | | PACK-021 |   |       |
|  |  | Bridge   | | Bridge   | | Bridge   | | Bridge   |   |       |
|  |  +----+-----+ +----+-----+ +----+-----+ +----+-----+   |       |
|  |       |           |           |           |             |       |
|  |  +----+-----+ +----+-----+ +----+-----+ +----------+   |       |
|  |  | MRV      | | DATA     | | Decarb   | | Health   |   |       |
|  |  | Bridge   | | Bridge   | | Bridge   | | Check    |   |       |
|  |  +----+-----+ +----+-----+ +----+-----+ +----------+   |       |
|  +-------+----------+----------+----------+-----------+    |       |
+----------+----------+----------+----------+-----------+----+       |
           |          |          |          |                         |
   +-------v---+ +---v------+ +-v--------+ +v---------+             |
   | SBTi SDA  | | IEA NZE  | | IPCC AR6 | | PACK-021 |             |
   | Tool V3.0 | | 2050 DB  | | Factors  | | Baseline |             |
   +-----------+ +----------+ +----------+ +----------+             |
                                                                     |
   +----------+ +----------+ +----------+                            |
   | 30 MRV   | | 20 DATA  | | 10 FOUND |                           |
   | Agents   | | Agents   | | Agents   |                           |
   +----------+ +----------+ +----------+                            |
```

### Integration Bridges Summary

| # | Bridge | File | Data Direction | Protocol |
|---|--------|------|---------------|----------|
| 1 | SBTi SDA Bridge | `sbti_sda_bridge.py` | Read-only | File-based + REST |
| 2 | IEA NZE Bridge | `iea_nze_bridge.py` | Read-only | File-based + REST |
| 3 | IPCC AR6 Bridge | `ipcc_ar6_bridge.py` | Read-only | File-based |
| 4 | PACK-021 Bridge | `pack021_bridge.py` | Bidirectional | Internal API |
| 5 | MRV Bridge | `mrv_bridge.py` | Request-Response | Internal API |
| 6 | DATA Bridge | `data_bridge.py` | Request-Response | Internal API |
| 7 | Decarb Bridge | `decarb_bridge.py` | Request-Response | Internal API |
| 8 | Health Check | `health_check.py` | Read-only | Multi-protocol |
| 9 | Setup Wizard | `setup_wizard.py` | Bidirectional | Internal |
| 10 | Pack Orchestrator | `pack_orchestrator.py` | Bidirectional | Internal |

---

## SBTi SDA Integration

### Overview

The SBTi SDA Bridge provides access to SBTi Sectoral Decarbonization Approach data including sector classification taxonomy, convergence factors, pathway benchmarks, and validation criteria.

### Data Sources

| Data Set | Description | Format | Update Frequency |
|----------|-------------|--------|-----------------|
| SDA Sector Taxonomy | 12 SDA sector definitions | JSON | Annual (with SBTi tool updates) |
| SDA Convergence Factors | Sector-specific intensity convergence factors | CSV | Annual |
| SDA Sector Pathways | Year-by-year intensity pathways per sector | CSV | Annual |
| SDA Validation Criteria | Criteria for SBTi target validation | JSON | Per SBTi Corporate Standard version |
| SDA Sector Averages | Global sector average intensities | CSV | Annual |

### Setup

```python
from integrations.sbti_sda_bridge import SBTiSDABridge

bridge = SBTiSDABridge(
    data_dir="/data/sbti_sda",         # Directory containing SBTi SDA data files
    version="v3.0",                     # SBTi SDA Tool version
    auto_update=True,                   # Auto-check for data updates
)

# Verify connection
status = bridge.verify()
print(f"SBTi SDA Bridge Status: {status.status}")
print(f"Data Version: {status.data_version}")
print(f"Sectors Loaded: {status.sectors_loaded}")
print(f"Convergence Factors: {status.convergence_factors_count}")
print(f"Last Updated: {status.last_updated}")
```

### Usage

```python
# Get sector pathway
pathway_data = bridge.get_sector_pathway(
    sector="steel",
    scenario="nze_15c",
    region="global",
)

# Get convergence factors
factors = bridge.get_convergence_factors(
    sector="cement",
    base_year=2023,
    target_year=2050,
)

# Get sector average intensity
average = bridge.get_sector_average(
    sector="power_generation",
    year=2023,
    region="global",
)

# Validate against SBTi criteria
validation = bridge.validate_pathway(
    sector="steel",
    base_year=2023,
    base_year_intensity=1.85,
    target_year=2030,
    target_intensity=1.25,
    scope1_2_coverage=0.98,
    scope3_coverage=0.70,
)
```

### Data File Structure

```
/data/sbti_sda/
  v3.0/
    sectors/
      sector_taxonomy.json       # Sector definitions
      sector_averages.csv        # Global sector averages
    pathways/
      power_generation.csv       # Power sector pathway
      steel.csv                  # Steel sector pathway
      cement.csv                 # Cement sector pathway
      aluminum.csv               # Aluminum sector pathway
      chemicals.csv              # Chemicals sector pathway
      pulp_paper.csv             # Pulp & paper sector pathway
      aviation.csv               # Aviation sector pathway
      shipping.csv               # Shipping sector pathway
      road_transport.csv         # Road transport sector pathway
      rail.csv                   # Rail sector pathway
      buildings_residential.csv  # Residential buildings pathway
      buildings_commercial.csv   # Commercial buildings pathway
    convergence/
      convergence_factors.csv    # Global convergence factors
      regional_adjustments.csv   # Regional pathway adjustments
    validation/
      criteria_v2.0.json         # SBTi Corporate Standard v2.0 criteria
      flag_criteria.json         # SBTi FLAG Guidance criteria
    checksums.sha256             # SHA-256 checksums for all files
```

### Configuration

```yaml
# In pack.yaml or preset YAML
sbti_sda:
  data_dir: "/data/sbti_sda"
  version: "v3.0"
  auto_update: true
  update_check_interval_hours: 24
  fallback_to_previous_version: true
  cache_ttl_seconds: 3600
  checksum_verification: true
```

---

## IEA NZE Data Integration

### Overview

The IEA NZE Bridge provides access to IEA Net Zero by 2050 sector pathway data, technology milestones, regional variants, and scenario comparisons.

### Data Sources

| Data Set | Description | Format | Update Frequency |
|----------|-------------|--------|-----------------|
| Sector Pathways | Year-by-year intensity trajectories for 15+ sectors | CSV | Annual (WEO release) |
| Technology Milestones | 400+ technology deployment milestones | JSON | Annual |
| Regional Pathways | OECD, emerging markets, global pathway variants | CSV | Annual |
| Scenario Data | NZE, APS, STEPS, WB2C, 2C scenario parameters | JSON | Annual |
| Cost Assumptions | Technology cost curves and learning rates | CSV | Annual |

### Setup

```python
from integrations.iea_nze_bridge import IEANZEBridge

bridge = IEANZEBridge(
    data_dir="/data/iea_nze_2050",
    version="2023_update",
    scenarios=["nze", "aps", "steps", "wb2c", "2c"],
)

# Verify connection
status = bridge.verify()
print(f"IEA NZE Bridge Status: {status.status}")
print(f"Data Version: {status.data_version}")
print(f"Sectors Covered: {status.sectors_covered}")
print(f"Milestones Loaded: {status.milestone_count}")
print(f"Scenarios Available: {status.scenarios_available}")
```

### Usage

```python
# Get sector pathway for specific scenario
pathway = bridge.get_sector_pathway(
    sector="steel",
    scenario="nze",
    region="global",
    start_year=2020,
    end_year=2050,
)

# Get technology milestones for sector
milestones = bridge.get_milestones(
    sector="steel",
    year_from=2025,
    year_to=2050,
)

# Get regional pathway variant
regional_pathway = bridge.get_regional_pathway(
    sector="power_generation",
    scenario="nze",
    region="oecd",
)

# Get scenario comparison
scenarios = bridge.compare_scenarios(
    sector="cement",
    scenarios=["nze", "aps", "steps"],
    milestones=[2025, 2030, 2035, 2040, 2045, 2050],
)

# Get technology cost curve
cost_curve = bridge.get_technology_cost(
    technology="solar_pv",
    region="global",
    start_year=2020,
    end_year=2050,
)
```

### Data File Structure

```
/data/iea_nze_2050/
  2023_update/
    sectors/
      power_generation/
        nze_pathway.csv
        aps_pathway.csv
        steps_pathway.csv
        technology_mix.csv
      steel/
        nze_pathway.csv
        aps_pathway.csv
        steps_pathway.csv
        technology_mix.csv
      cement/
        ...
    milestones/
      all_milestones.json
      power_milestones.json
      industry_milestones.json
      transport_milestones.json
      buildings_milestones.json
    regional/
      oecd_adjustments.csv
      emerging_markets_adjustments.csv
    costs/
      technology_costs.csv
      learning_rates.csv
    scenarios/
      scenario_definitions.json
      scenario_parameters.json
    checksums.sha256
```

---

## IPCC AR6 Integration

### Overview

The IPCC AR6 Bridge provides access to IPCC Sixth Assessment Report data including GWP-100 values, sector-specific emission factors, carbon budget data, and mitigation pathway scenarios.

### Setup

```python
from integrations.ipcc_ar6_bridge import IPCCAR6Bridge

bridge = IPCCAR6Bridge(
    data_dir="/data/ipcc_ar6",
)

# Verify connection
status = bridge.verify()
print(f"IPCC AR6 Bridge Status: {status.status}")
print(f"GWP Values Loaded: {status.gwp_count}")
print(f"Emission Factors: {status.emission_factor_count}")
```

### Usage

```python
# Get GWP-100 values (AR6)
gwp = bridge.get_gwp(
    gas="CH4",
    assessment_report="AR6",
    time_horizon=100,
)
print(f"CH4 GWP-100: {gwp.value}")  # 27.9 for AR6

# Get sector emission factors
factors = bridge.get_emission_factors(
    sector="steel",
    process="blast_furnace",
    region="global",
)

# Get carbon budget
budget = bridge.get_carbon_budget(
    temperature_target="1.5C",
    probability=0.50,
)
print(f"Remaining carbon budget: {budget.remaining_gtco2} GtCO2")

# Get SSP pathway
ssp = bridge.get_ssp_pathway(
    ssp_scenario="SSP1-1.9",
    variable="emissions_co2",
    start_year=2020,
    end_year=2100,
)
```

---

## PACK-021 Integration

### Overview

The PACK-021 Bridge enables bidirectional integration with the Net Zero Starter Pack, importing baseline emissions and target definitions while exporting sector pathway analysis results.

### Setup

```python
from integrations.pack021_bridge import Pack021Bridge

bridge = Pack021Bridge(
    base_url="http://localhost:8021",    # PACK-021 API endpoint
    api_key="internal_service_key",       # Internal service authentication
    sync_mode="pull",                     # "pull" or "bidirectional"
)

# Verify connection
status = bridge.verify()
print(f"PACK-021 Bridge Status: {status.status}")
print(f"PACK-021 Version: {status.pack_version}")
print(f"Baseline Available: {status.baseline_available}")
print(f"Targets Available: {status.targets_available}")
```

### Importing Baseline Data

```python
# Import baseline emissions from PACK-021
baseline = bridge.get_baseline(
    organization_id="org_abc123",
    reporting_year=2023,
)

print(f"Scope 1: {baseline.scope1_tco2e:,.0f} tCO2e")
print(f"Scope 2 (location): {baseline.scope2_location_tco2e:,.0f} tCO2e")
print(f"Scope 2 (market): {baseline.scope2_market_tco2e:,.0f} tCO2e")
print(f"Scope 3: {baseline.scope3_total_tco2e:,.0f} tCO2e")

# Import target definitions from PACK-021
targets = bridge.get_targets(
    organization_id="org_abc123",
)

print(f"Target Type: {targets.target_type}")
print(f"Base Year: {targets.base_year}")
print(f"Target Year: {targets.target_year}")
print(f"Reduction Target: {targets.reduction_pct:.1%}")
```

### Exporting Sector Pathway Data

```python
# Export sector pathway results back to PACK-021
bridge.export_sector_pathway(
    organization_id="org_abc123",
    pathway_id=pathway_result.pathway_id,
    sector="steel",
    intensity_pathway=pathway_result.annual_pathway,
    sbti_alignment=pathway_result.sbti_alignment,
)
```

### Standalone Mode (Without PACK-021)

If PACK-021 is not deployed, provide baseline data directly:

```python
# Configure standalone mode
config = PackConfig(
    pack021_integration=False,
    standalone_baseline={
        "scope1_tco2e": 7_500_000,
        "scope2_location_tco2e": 1_500_000,
        "scope2_market_tco2e": 1_200_000,
        "scope3_total_tco2e": 12_000_000,
    },
)
```

---

## MRV Agent Integration

### Overview

The MRV Bridge routes sector-specific emission calculations to the appropriate AGENT-MRV agents (30 agents covering Scope 1, 2, and 3).

### Sector-to-MRV Agent Routing

| Sector | Primary MRV Agents | Scope Focus |
|--------|-------------------|-------------|
| Power Generation | MRV-001 (Stationary Combustion), MRV-009 (Location-Based) | Scope 1, 2 |
| Steel | MRV-001, MRV-004 (Process Emissions), MRV-009, MRV-010 | Scope 1, 2 |
| Cement | MRV-001, MRV-004, MRV-009 | Scope 1, 2 |
| Aluminum | MRV-001, MRV-004, MRV-009, MRV-010 | Scope 1, 2 |
| Aviation | MRV-003 (Mobile Combustion), MRV-019 (Business Travel) | Scope 1, 3 |
| Shipping | MRV-003, MRV-017 (Upstream Transport) | Scope 1, 3 |
| Buildings | MRV-001, MRV-009, MRV-010, MRV-011 (Steam/Heat) | Scope 1, 2 |
| Agriculture | MRV-008 (Agricultural), MRV-006 (Land Use) | Scope 1 |

### Setup

```python
from integrations.mrv_bridge import MRVBridge

bridge = MRVBridge()

# Verify all 30 MRV agents are available
status = bridge.verify()
print(f"MRV Bridge Status: {status.status}")
print(f"Agents Available: {status.agents_available}/{status.agents_total}")

for agent in status.agents:
    icon = "OK" if agent.available else "MISSING"
    print(f"  [{icon}] {agent.agent_id}: {agent.name}")
```

### Usage

```python
# Route calculation to appropriate MRV agents based on sector
result = bridge.calculate_sector_emissions(
    sector="steel",
    activity_data={
        "stationary_combustion": {
            "natural_gas_kwh": 50_000_000,
            "coal_tonnes": 2_000_000,
        },
        "process_emissions": {
            "iron_ore_reduction_tonnes": 3_750_000,
            "lime_calcination_tonnes": 500_000,
        },
        "electricity": {
            "purchased_kwh": 200_000_000,
            "grid_region": "DE",
            "ppa_percentage": 15,
        },
    },
)

print(f"Scope 1: {result.scope1_tco2e:,.0f} tCO2e")
print(f"  Stationary Combustion (MRV-001): {result.mrv_001_tco2e:,.0f}")
print(f"  Process Emissions (MRV-004): {result.mrv_004_tco2e:,.0f}")
print(f"Scope 2 Location (MRV-009): {result.scope2_location_tco2e:,.0f} tCO2e")
print(f"Scope 2 Market (MRV-010): {result.scope2_market_tco2e:,.0f} tCO2e")
```

---

## DATA Agent Integration

### Overview

The DATA Bridge connects to 20 AGENT-DATA agents for sector-specific activity data intake, quality profiling, and validation.

### Key DATA Agents for Sector Pathway

| DATA Agent | Purpose in PACK-028 |
|-----------|-------------------|
| DATA-001 (PDF Extractor) | Extract production data from supplier invoices and reports |
| DATA-002 (Excel/CSV Normalizer) | Normalize activity data spreadsheets |
| DATA-003 (ERP Connector) | Pull production volumes and energy data from ERP |
| DATA-010 (Data Quality Profiler) | Assess quality of sector activity data |
| DATA-015 (Cross-Source Reconciliation) | Reconcile production data across multiple sources |
| DATA-018 (Data Lineage Tracker) | Track data provenance for audit trail |
| DATA-019 (Validation Rule Engine) | Validate sector-specific data constraints |

### Setup

```python
from integrations.data_bridge import DataBridge

bridge = DataBridge()

# Verify DATA agents
status = bridge.verify()
print(f"DATA Bridge Status: {status.status}")
print(f"Agents Available: {status.agents_available}/{status.agents_total}")
```

### Usage

```python
# Intake production data
result = bridge.intake_sector_data(
    sector="steel",
    data_sources=[
        {
            "source_type": "erp",
            "system": "SAP",
            "data_type": "production_volumes",
            "period": "2023-01-01/2023-12-31",
        },
        {
            "source_type": "spreadsheet",
            "file_path": "/data/uploads/steel_production_2023.xlsx",
            "data_type": "production_by_route",
        },
    ],
)

# Profile data quality
quality = bridge.profile_data_quality(
    data_set_id=result.data_set_id,
    sector="steel",
)

print(f"Data Quality Score: {quality.overall_score:.1f}/5.0")
print(f"Completeness: {quality.completeness:.1%}")
print(f"Accuracy: {quality.accuracy:.1%}")
print(f"Timeliness: {quality.timeliness}")
```

---

## Foundation Agent Integration

### Overview

PACK-028 leverages all 10 AGENT-FOUND foundation agents for core platform services.

| FOUND Agent | Role in PACK-028 |
|-------------|-----------------|
| FOUND-001 (Orchestrator) | DAG pipeline execution for workflows |
| FOUND-002 (Schema Compiler) | Validate sector pathway data schemas |
| FOUND-003 (Unit Normalizer) | Normalize sector-specific units (gCO2/kWh, tCO2e/t) |
| FOUND-004 (Assumptions Registry) | Register and track pathway assumptions |
| FOUND-005 (Citations) | Cite SBTi, IEA, IPCC reference sources |
| FOUND-006 (Access & Policy Guard) | Enforce RBAC for pathway data |
| FOUND-008 (Reproducibility) | Ensure deterministic pathway calculations |
| FOUND-009 (QA Test Harness) | Validate pathway calculation correctness |
| FOUND-010 (Observability) | Monitor engine performance and health |

---

## Decarbonization Agent Integration

### Overview

The Decarbonization Bridge connects sector pathway analysis with specific decarbonization action agents for technology-specific implementation planning.

### Setup

```python
from integrations.decarb_bridge import DecarbBridge

bridge = DecarbBridge()

# Get sector-specific decarbonization actions
actions = bridge.get_sector_actions(
    sector="steel",
    pathway=pathway_result,
    budget_usd=500_000_000,
    timeline_years=7,
)

for action in actions:
    print(f"  {action.name}: {action.reduction_tco2e:,.0f} tCO2e "
          f"(${action.cost_usd:,.0f}, {action.timeline_years} years)")
```

---

## Application Dependencies

### GL-SBTi-APP Integration

PACK-028 integrates with GL-SBTi-APP for advanced SBTi target validation and temperature scoring:

```python
# Temperature scoring via GL-SBTi-APP
from integrations.app_bridges import SBTiAppBridge

bridge = SBTiAppBridge()
temp_score = bridge.calculate_temperature_score(
    pathway=pathway_result,
    sector="steel",
)
print(f"Temperature Score: {temp_score.value:.1f}C")
print(f"Alignment: {temp_score.alignment}")
```

### GL-CDP-APP Integration

Sector pathway data feeds into CDP Climate Change questionnaire responses:

```python
from integrations.app_bridges import CDPAppBridge

bridge = CDPAppBridge()
cdp_section = bridge.populate_c4_targets(
    pathway=pathway_result,
    convergence=convergence_result,
)
```

### GL-TCFD-APP Integration

Scenario comparison data feeds into TCFD/ISSB S2 disclosures:

```python
from integrations.app_bridges import TCFDAppBridge

bridge = TCFDAppBridge()
tcfd_section = bridge.populate_strategy(
    scenario_comparison=scenario_result,
    technology_roadmap=roadmap_result,
)
```

---

## API Authentication

### JWT Authentication

All PACK-028 API endpoints require JWT Bearer token authentication.

```python
import requests

# Obtain token
response = requests.post(
    "https://api.greenlang.io/v1/auth/token",
    json={
        "username": "analyst@company.com",
        "password": "********",
    },
)
token = response.json()["access_token"]

# Use token in API calls
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json",
}

response = requests.post(
    "https://api.greenlang.io/v1/packs/028/engines/sector-classification/classify",
    headers=headers,
    json={"company_profile": {"nace_codes": ["C24.10"]}},
)
```

### Service-to-Service Authentication

Internal service calls between PACK-028 and other platform components use service account tokens:

```python
# Internal bridge authentication
bridge = Pack021Bridge(
    base_url="http://pack021-service:8021",
    api_key="svc_pack028_to_pack021_key",  # From Vault
)
```

### Permission Matrix

| Role | Engines | Workflows | Templates | Config | Admin |
|------|---------|-----------|-----------|--------|-------|
| `sector_pathway_admin` | Execute | Execute | Render | Read/Write | Full |
| `pathway_designer` | Execute | Execute | Render | Read | None |
| `sector_analyst` | Read | Read | Render | Read | None |
| `auditor` | Read | Read | Render | Read | None |

---

## Webhook Configuration

### Setting Up Webhooks

```python
from integrations.webhook_manager import WebhookManager

webhooks = WebhookManager()

# Register a webhook
webhook = webhooks.register(
    url="https://company.com/api/webhooks/pack028",
    events=[
        "pathway.generated",
        "convergence.alert",
        "milestone.missed",
        "benchmark.update",
    ],
    secret="whsec_your_secret_key",
    headers={
        "X-Custom-Header": "custom-value",
    },
)

print(f"Webhook ID: {webhook.id}")
print(f"Events: {webhook.events}")
```

### Webhook Payload Verification

```python
import hmac
import hashlib

def verify_webhook(payload: bytes, signature: str, secret: str) -> bool:
    """Verify webhook signature."""
    expected = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)
```

---

## Data Flow Patterns

### Pattern 1: SBTi Pathway Design Flow

```
PACK-021 Baseline --> PACK-028 Intensity Calculator
                          |
SBTi SDA Bridge -------> Pathway Generator
                          |
                     Convergence Analyzer
                          |
                     SBTi Validation Report
                          |
                     GL-SBTi-APP (submission)
```

### Pattern 2: Technology Roadmap Flow

```
PACK-028 Pathway Generator --> Technology Roadmap Engine
                                    |
IEA NZE Bridge (milestones) ------> Milestone Mapping
                                    |
Decarb Bridge (actions) ----------> CapEx Schedule
                                    |
                               Technology Roadmap Report
```

### Pattern 3: Progress Monitoring Flow

```
DATA Bridge (current data) --> Intensity Calculator (updated)
                                   |
PACK-021 Bridge (baseline) -----> Convergence Analyzer
                                   |
Sector Benchmark Engine <--------- Benchmark Data
                                   |
                              Progress Monitoring Report
                                   |
                              Webhook Notifications
```

---

## Integration Testing

### Running Integration Tests

```bash
# Run all integration tests
pytest tests/test_integrations.py -v

# Run specific bridge tests
pytest tests/test_sbti_sda_bridge.py -v
pytest tests/test_iea_nze_bridge.py -v
pytest tests/test_pack021_bridge.py -v
pytest tests/test_mrv_bridge.py -v

# Run with mock data (no external dependencies)
pytest tests/test_integrations.py -v --mock-bridges
```

### Integration Health Verification

```python
from integrations.health_check import HealthCheck

hc = HealthCheck()
result = hc.run(categories=["integrations_only"])

for category in result.categories:
    print(f"  [{category.status}] {category.name}: {category.score}/100")
    if category.score < 100:
        for issue in category.issues:
            print(f"         Issue: {issue.description}")
            print(f"         Fix: {issue.remediation}")
```

---

## Troubleshooting Integrations

### SBTi SDA Bridge Issues

**Problem:** `SBTiDataNotFoundError: Sector pathway data not found for 'steel'`

**Solution:**
```bash
# Verify SBTi data directory
ls $SECTOR_PATHWAY_SBTI_DATA_DIR/v3.0/pathways/
# Expected: steel.csv, cement.csv, etc.

# Verify data integrity
python -c "
from integrations.sbti_sda_bridge import SBTiSDABridge
bridge = SBTiSDABridge()
bridge.verify_checksums()
"
```

### IEA NZE Bridge Issues

**Problem:** `IEAMilestoneLoadError: Failed to load milestone data`

**Solution:**
```bash
# Verify IEA data files
ls $SECTOR_PATHWAY_IEA_DATA_DIR/2023_update/milestones/
# Expected: all_milestones.json

# Reload milestone data
python -c "
from integrations.iea_nze_bridge import IEANZEBridge
bridge = IEANZEBridge()
bridge.reload_milestones()
print(f'Milestones loaded: {bridge.milestone_count}')
"
```

### PACK-021 Bridge Issues

**Problem:** `Pack021ConnectionError: Unable to connect to PACK-021 service`

**Solution:**
```bash
# Verify PACK-021 service is running
curl http://localhost:8021/health

# Check environment variable
echo $SECTOR_PATHWAY_PACK021_BASE_URL

# Fall back to standalone mode
export SECTOR_PATHWAY_PACK021_ENABLED=false
```

### MRV Bridge Issues

**Problem:** `MRVAgentUnavailable: AGENT-MRV-004 (Process Emissions) not responding`

**Solution:**
```python
from integrations.mrv_bridge import MRVBridge

bridge = MRVBridge()
status = bridge.verify()

# Check individual agent availability
for agent in status.agents:
    if not agent.available:
        print(f"UNAVAILABLE: {agent.agent_id} - {agent.name}")
        print(f"  Last seen: {agent.last_seen}")
        print(f"  Error: {agent.error_message}")
```

---

**End of Integration Guide**
