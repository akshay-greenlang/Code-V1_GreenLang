# California SB 253 Climate Disclosure Agent - Comprehensive Implementation Plan

**Document ID:** GL-SB253-IMPL-PLAN-001
**Version:** 1.0.0
**Date:** 2025-12-04
**Author:** GL-SB253-PM (US State Climate Disclosure Platform)
**Status:** APPROVED FOR IMPLEMENTATION

---

## Executive Summary

This implementation plan outlines the architecture, code structure, and development tasks for the California SB 253 Climate Corporate Data Accountability Act compliance agent. The plan leverages the existing GL-VCCI-APP codebase (55% complete) and delivers a production-ready solution before the June 30, 2026 deadline.

### Key Deadlines
| Milestone | Date | Scope |
|-----------|------|-------|
| **Scope 1 & 2 Reporting** | June 30, 2026 | Direct + purchased energy emissions |
| **Scope 3 Reporting** | June 30, 2027 | Value chain emissions (15 categories) |
| **Limited Assurance** | 2026-2029 | Third-party verification |
| **Reasonable Assurance** | 2030+ | Enhanced verification |

### Target Metrics
- **Calculation Accuracy:** Scope 1: +/-1%, Scope 2: +/-2%, Scope 3: +/-5%
- **Golden Tests:** 300+ test scenarios
- **Audit Trail:** 100% provenance coverage
- **Third-Party Assurance:** Big 4 audit-ready

---

## 1. Agent Architecture Design

### 1.1 System Architecture Overview

```
+------------------------------------------------------------------+
|                    SB 253 DISCLOSURE PLATFORM                     |
+------------------------------------------------------------------+
|                                                                   |
|  +---------------------+    +---------------------+               |
|  | DataCollectionAgent |    | CalculationAgent    |               |
|  |---------------------|    |---------------------|               |
|  | - ERP Connectors    |    | - Scope 1 Engine    |               |
|  | - Utility APIs      |    | - Scope 2 Engine    |               |
|  | - Travel Systems    |--->| - Scope 3 Engine    |               |
|  | - Supplier Portal   |    | - Zero Hallucination|               |
|  +---------------------+    +----------+----------+               |
|                                        |                          |
|                                        v                          |
|  +---------------------+    +---------------------+               |
|  | AssuranceReadyAgent |<---| MultiStateFilingAgt |               |
|  |---------------------|    |---------------------|               |
|  | - SHA-256 Provenance|    | - CARB Portal       |               |
|  | - Audit Trails      |    | - Colorado CDPHE    |               |
|  | - DQI Scoring       |    | - Washington ECY    |               |
|  +----------+----------+    +---------------------+               |
|             |                                                     |
|             v                                                     |
|  +---------------------+                                          |
|  | ThirdPartyAssurance |                                          |
|  |---------------------|                                          |
|  | - ISAE 3410 Package |                                          |
|  | - Big 4 Support     |                                          |
|  | - Gap Analysis      |                                          |
|  +---------------------+                                          |
+------------------------------------------------------------------+
```

### 1.2 Directory Structure

```
greenlang/
├── agents/
│   └── sb253/
│       ├── __init__.py
│       ├── agent_spec.py              # AgentSpec v2 definition
│       ├── data_collection_agent.py   # Pipeline Agent 1
│       ├── calculation_agent.py       # Pipeline Agent 2
│       ├── assurance_ready_agent.py   # Pipeline Agent 3
│       ├── multi_state_filing_agent.py # Pipeline Agent 4
│       └── third_party_assurance_agent.py # Pipeline Agent 5
│
├── calculators/
│   └── sb253/
│       ├── __init__.py
│       ├── scope1/
│       │   ├── __init__.py
│       │   ├── stationary_combustion.py
│       │   ├── mobile_combustion.py
│       │   ├── fugitive_emissions.py
│       │   └── process_emissions.py
│       ├── scope2/
│       │   ├── __init__.py
│       │   ├── location_based.py
│       │   └── market_based.py
│       └── scope3/
│           ├── __init__.py
│           ├── category_01_purchased_goods.py
│           ├── category_02_capital_goods.py
│           ├── category_03_fuel_energy.py
│           ├── category_04_upstream_transport.py
│           ├── category_05_waste.py
│           ├── category_06_business_travel.py
│           ├── category_07_commuting.py
│           ├── category_08_upstream_leased.py
│           ├── category_09_downstream_transport.py
│           ├── category_10_processing.py
│           ├── category_11_use_of_sold.py
│           ├── category_12_end_of_life.py
│           ├── category_13_downstream_leased.py
│           ├── category_14_franchises.py
│           └── category_15_investments.py
│
├── emission_factors/
│   └── sb253/
│       ├── __init__.py
│       ├── california_grid_factors.py    # CAMX eGRID factors
│       ├── epa_ghg_factors.py            # EPA GHG EF Hub
│       ├── eeio_factors.py               # EPA EEIO for Scope 3
│       ├── glec_transport_factors.py     # GLEC Framework
│       ├── defra_travel_factors.py       # DEFRA 2024
│       └── refrigerant_gwp.py            # IPCC AR6 GWP values
│
├── schemas/
│   └── sb253/
│       ├── __init__.py
│       ├── company_profile.py            # Company input schema
│       ├── facility_data.py              # Facility schema
│       ├── fuel_consumption.py           # Scope 1 input
│       ├── electricity_usage.py          # Scope 2 input
│       ├── supply_chain_data.py          # Scope 3 input
│       ├── ghg_inventory.py              # Output schema
│       ├── sb253_report.py               # CARB report schema
│       └── assurance_package.py          # Verification schema
│
├── integrations/
│   └── sb253/
│       ├── __init__.py
│       ├── carb_portal.py                # CARB filing integration
│       ├── erp_connectors.py             # SAP/Oracle/Workday
│       ├── utility_apis.py               # PG&E/SCE/SDG&E
│       └── travel_systems.py             # Concur/SAP Travel
│
├── reports/
│   └── sb253/
│       ├── __init__.py
│       ├── carb_filing_generator.py      # CARB XML/JSON
│       ├── assurance_package_generator.py # ISAE 3410
│       └── disclosure_report.py          # Public report
│
└── provenance/
    └── sb253/
        ├── __init__.py
        ├── audit_trail.py                # SHA-256 tracking
        └── dqi_scorer.py                 # Data Quality Indicators
```

### 1.3 Agent Specification (pack.yaml)

```yaml
# File: greenlang/agents/sb253/pack.yaml
pack_id: gl-sb253-disclosure
name: "California SB 253 Climate Disclosure Agent"
version: "1.0.0"
description: |
  Automated GHG emissions calculation, verification, and reporting for
  California SB 253 (Climate Corporate Data Accountability Act) compliance.
  Calculates Scope 1, 2, and 3 emissions aligned with GHG Protocol standards.

author: GreenLang Framework Team
license: Proprietary
min_greenlang_version: "0.5.0"

regulatory_context:
  regulation: "California SB 253"
  jurisdiction: California
  effective_date: "2024-01-01"
  first_reporting: "2026-06-30"
  enforcement_agency: "California Air Resources Board (CARB)"

agents:
  - id: data_collection_agent
    name: "Data Collection Agent"
    type: data_ingestion
    priority: P0

  - id: calculation_agent
    name: "GHG Calculation Agent"
    type: calculator
    priority: P0

  - id: assurance_ready_agent
    name: "Assurance Ready Agent"
    type: verification
    priority: P0

  - id: multi_state_filing_agent
    name: "Multi-State Filing Agent"
    type: filing
    priority: P1

  - id: third_party_assurance_agent
    name: "Third-Party Assurance Agent"
    type: audit
    priority: P1

tools:
  - name: scope1_calculator
    type: calculator
    source: calculators/sb253/scope1

  - name: scope2_calculator
    type: calculator
    source: calculators/sb253/scope2

  - name: scope3_calculator
    type: calculator
    source: calculators/sb253/scope3

  - name: egrid_factor_lookup
    type: data_lookup
    source: emission_factors/sb253/california_grid_factors

  - name: eeio_factor_lookup
    type: data_lookup
    source: emission_factors/sb253/eeio_factors

  - name: carb_filing_generator
    type: report_generator
    source: reports/sb253/carb_filing_generator

  - name: assurance_package_generator
    type: report_generator
    source: reports/sb253/assurance_package_generator

  - name: provenance_tracker
    type: utility
    source: provenance/sb253/audit_trail

datasets:
  - id: epa_egrid_2023
    name: "EPA eGRID 2023"
    source: "https://www.epa.gov/egrid"

  - id: epa_ghg_ef_hub
    name: "EPA GHG Emission Factors Hub 2024"
    source: "https://www.epa.gov/climateleadership/ghg-emission-factors-hub"

  - id: ipcc_ar6_gwp
    name: "IPCC AR6 GWP Values"
    source: "https://www.ipcc.ch/assessment-report/ar6/"

evaluation:
  golden_tests: 300
  accuracy_thresholds:
    scope_1: 0.01
    scope_2: 0.02
    scope_3: 0.05
  benchmarks:
    latency_p95_seconds: 30
    cost_per_analysis_usd: 0.50
```

---

## 2. Input/Output Schema Definitions

### 2.1 Company Profile Input Schema

```python
# File: greenlang/schemas/sb253/company_profile.py
"""
SB 253 Company Profile Input Schema
Defines company information required for applicability determination and reporting.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field, validator
from datetime import date
import re


class Address(BaseModel):
    """Physical address schema."""
    street: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    zip: str = Field(..., pattern=r'^\d{5}(-\d{4})?$')
    country: str = Field(default="US")


class CompanyProfile(BaseModel):
    """
    Company profile for SB 253 reporting.

    Validates that company meets $1B revenue threshold for California.
    """
    company_name: str = Field(..., description="Legal entity name")
    ein: str = Field(
        ...,
        pattern=r'^\d{2}-\d{7}$',
        description="Employer Identification Number (XX-XXXXXXX)"
    )

    # Revenue validation
    total_revenue: float = Field(
        ...,
        ge=1_000_000_000,  # $1B minimum for SB 253
        description="Total annual revenue (USD) - must be $1B+ for SB 253"
    )
    california_revenue: float = Field(
        ...,
        ge=0,
        description="Annual revenue from California operations (USD)"
    )

    # Industry classification
    naics_code: str = Field(
        ...,
        pattern=r'^\d{6}$',
        description="6-digit NAICS code"
    )
    sic_code: Optional[str] = Field(
        None,
        pattern=r'^\d{4}$',
        description="4-digit SIC code (optional)"
    )

    # Organizational structure
    organizational_boundary: Literal[
        "equity_share",
        "operational_control",
        "financial_control"
    ] = Field(
        default="operational_control",
        description="GHG Protocol organizational boundary approach"
    )

    # Contact information
    headquarters_address: Address
    reporting_contact_email: str = Field(..., pattern=r'^[\w\.-]+@[\w\.-]+\.\w+$')
    reporting_contact_name: str

    # Reporting metadata
    reporting_year: int = Field(..., ge=2025, description="Calendar year for reporting")
    fiscal_year_end: date

    # Multi-state flags
    does_business_in_california: bool = Field(
        default=True,
        description="Confirms company does business in California"
    )
    colorado_applicable: bool = Field(
        default=False,
        description="Whether Colorado SB 23-016 also applies"
    )
    washington_applicable: bool = Field(
        default=False,
        description="Whether Washington HB 1589 also applies"
    )

    @validator('california_revenue')
    def california_revenue_less_than_total(cls, v, values):
        if 'total_revenue' in values and v > values['total_revenue']:
            raise ValueError('California revenue cannot exceed total revenue')
        return v

    @validator('does_business_in_california')
    def must_do_business_in_california(cls, v):
        if not v:
            raise ValueError('SB 253 only applies to companies doing business in California')
        return v

    class Config:
        schema_extra = {
            "example": {
                "company_name": "Acme Corporation",
                "ein": "12-3456789",
                "total_revenue": 2_500_000_000,
                "california_revenue": 800_000_000,
                "naics_code": "336111",
                "organizational_boundary": "operational_control",
                "headquarters_address": {
                    "street": "123 Main St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip": "94105",
                    "country": "US"
                },
                "reporting_contact_email": "sustainability@acme.com",
                "reporting_contact_name": "Jane Smith",
                "reporting_year": 2025,
                "fiscal_year_end": "2025-12-31",
                "does_business_in_california": True
            }
        }
```

### 2.2 Facility Data Input Schema

```python
# File: greenlang/schemas/sb253/facility_data.py
"""
SB 253 Facility Data Input Schema
Defines facility information including eGRID subregion mapping.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from enum import Enum


class EGridSubregion(str, Enum):
    """EPA eGRID Subregions."""
    CAMX = "CAMX"  # California (WECC California)
    AZNM = "AZNM"  # Southwest (WECC Southwest)
    NWPP = "NWPP"  # Northwest (WECC Northwest)
    RMPA = "RMPA"  # Rocky Mountain
    SRSO = "SRSO"  # SERC South
    ERCT = "ERCT"  # ERCOT (Texas)
    RFCW = "RFCW"  # RFC West
    NEWE = "NEWE"  # Northeast
    NYUP = "NYUP"  # New York Upstate
    NYCW = "NYCW"  # New York City/Westchester
    NYLI = "NYLI"  # New York Long Island
    RFCE = "RFCE"  # RFC East
    RFCM = "RFCM"  # RFC Michigan
    SRMW = "SRMW"  # SERC Midwest
    SRMV = "SRMV"  # SERC Mississippi Valley
    SRTV = "SRTV"  # SERC Tennessee Valley
    SRVC = "SRVC"  # SERC Virginia/Carolina
    SPNO = "SPNO"  # SPP North
    SPSO = "SPSO"  # SPP South
    MROE = "MROE"  # MRO East
    MROW = "MROW"  # MRO West
    FRCC = "FRCC"  # Florida
    HIOA = "HIOA"  # Hawaii Oahu
    HIMS = "HIMS"  # Hawaii Maui/So
    AKGD = "AKGD"  # Alaska Grid
    AKMS = "AKMS"  # Alaska Misc


class FacilityAddress(BaseModel):
    """Facility address with state for eGRID mapping."""
    street: str
    city: str
    state: str = Field(..., min_length=2, max_length=2)
    zip: str
    country: str = Field(default="US")


class FacilityData(BaseModel):
    """
    Individual facility data for SB 253 reporting.

    Each facility tracks location, eGRID subregion, and California presence.
    """
    facility_id: str = Field(..., description="Unique facility identifier")
    facility_name: str = Field(..., description="Facility name")

    # Location
    address: FacilityAddress
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)

    # Grid region for Scope 2 calculations
    egrid_subregion: EGridSubregion = Field(
        ...,
        description="EPA eGRID subregion for electricity emission factors"
    )

    # California presence flag
    california_facility: bool = Field(
        ...,
        description="Whether this facility is located in California"
    )

    # Operational data
    square_footage: Optional[float] = Field(None, ge=0)
    employee_count: Optional[int] = Field(None, ge=0)
    operating_hours_per_year: Optional[int] = Field(None, ge=0, le=8760)

    # Facility type
    facility_type: Literal[
        "office",
        "manufacturing",
        "warehouse",
        "retail",
        "data_center",
        "laboratory",
        "mixed_use",
        "other"
    ] = Field(default="office")

    class Config:
        schema_extra = {
            "example": {
                "facility_id": "FAC-001",
                "facility_name": "San Francisco Headquarters",
                "address": {
                    "street": "123 Market St",
                    "city": "San Francisco",
                    "state": "CA",
                    "zip": "94105",
                    "country": "US"
                },
                "egrid_subregion": "CAMX",
                "california_facility": True,
                "square_footage": 50000,
                "employee_count": 200,
                "facility_type": "office"
            }
        }
```

### 2.3 Fuel Consumption Input Schema (Scope 1)

```python
# File: greenlang/schemas/sb253/fuel_consumption.py
"""
SB 253 Fuel Consumption Input Schema
Defines Scope 1 fuel consumption data for stationary and mobile combustion.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import date
from enum import Enum


class FuelType(str, Enum):
    """Supported fuel types with EPA emission factors."""
    NATURAL_GAS = "natural_gas"
    DIESEL = "diesel"
    GASOLINE = "gasoline"
    PROPANE = "propane"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_4 = "fuel_oil_4"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL = "coal"
    LPG = "lpg"
    E10_GASOLINE = "e10_gasoline"
    E85_GASOLINE = "e85_gasoline"
    BIODIESEL_B20 = "biodiesel_b20"
    CARB_DIESEL = "carb_diesel"
    KEROSENE = "kerosene"


class FuelUnit(str, Enum):
    """Supported fuel consumption units."""
    THERMS = "therms"
    GALLONS = "gallons"
    LITERS = "liters"
    MMBTU = "MMBtu"
    KWH = "kWh"
    TONS = "tons"
    KG = "kg"
    MCF = "MCF"


class SourceCategory(str, Enum):
    """Scope 1 source categories."""
    STATIONARY_COMBUSTION = "stationary_combustion"
    MOBILE_COMBUSTION = "mobile_combustion"


class ReportingPeriod(BaseModel):
    """Time period for consumption data."""
    start_date: date
    end_date: date


class FuelConsumption(BaseModel):
    """
    Fuel consumption record for Scope 1 emissions calculation.

    Supports both stationary combustion (boilers, generators) and
    mobile combustion (fleet vehicles).
    """
    facility_id: str = Field(..., description="Link to facility record")
    fuel_type: FuelType = Field(..., description="Type of fuel consumed")
    quantity: float = Field(..., ge=0, description="Amount of fuel consumed")
    unit: FuelUnit = Field(..., description="Unit of measurement")

    source_category: SourceCategory = Field(
        ...,
        description="Stationary or mobile combustion"
    )

    # Optional detailed classification
    equipment_type: Optional[str] = Field(
        None,
        description="Equipment type (e.g., boiler, generator, forklift)"
    )

    # Reporting period
    reporting_period: ReportingPeriod

    # Data quality
    data_source: Optional[Literal[
        "meter_reading",
        "fuel_invoice",
        "fuel_card",
        "estimate",
        "other"
    ]] = Field(default="meter_reading")

    # Audit trail
    source_document_id: Optional[str] = Field(
        None,
        description="Reference to source document (invoice, meter log)"
    )

    class Config:
        schema_extra = {
            "example": {
                "facility_id": "FAC-001",
                "fuel_type": "natural_gas",
                "quantity": 50000,
                "unit": "therms",
                "source_category": "stationary_combustion",
                "equipment_type": "boiler",
                "reporting_period": {
                    "start_date": "2025-01-01",
                    "end_date": "2025-12-31"
                },
                "data_source": "meter_reading",
                "source_document_id": "UTB-2025-001234"
            }
        }
```

### 2.4 GHG Inventory Output Schema

```python
# File: greenlang/schemas/sb253/ghg_inventory.py
"""
SB 253 GHG Inventory Output Schema
Defines the complete GHG inventory structure for disclosure.
"""

from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class Scope3Category(BaseModel):
    """Individual Scope 3 category result."""
    category_number: int = Field(..., ge=1, le=15)
    category_name: str
    emissions_mt_co2e: float = Field(..., ge=0)
    calculation_method: Literal[
        "supplier_specific",
        "hybrid",
        "average_data",
        "spend_based"
    ]
    data_quality_score: float = Field(..., ge=1, le=5)
    materiality_assessment: Literal["material", "not_material", "excluded"]
    exclusion_reason: Optional[str] = None


class AuditTrailRecord(BaseModel):
    """Single audit trail entry for provenance tracking."""
    calculation_id: str
    timestamp: datetime
    scope: Literal["1", "2", "3"]
    category: Optional[str] = None
    input_hash: str = Field(..., description="SHA-256 hash of inputs")
    output_hash: str = Field(..., description="SHA-256 hash of outputs")
    emission_factor_source: str
    emission_factor_version: str
    calculation_formula: str


class Scope1Emissions(BaseModel):
    """Scope 1 direct emissions breakdown."""
    total_mt_co2e: float = Field(..., ge=0)
    stationary_combustion_mt_co2e: float = Field(default=0, ge=0)
    mobile_combustion_mt_co2e: float = Field(default=0, ge=0)
    fugitive_emissions_mt_co2e: float = Field(default=0, ge=0)
    process_emissions_mt_co2e: float = Field(default=0, ge=0)

    # Gas breakdown
    co2_mt: float = Field(default=0, ge=0)
    ch4_mt_co2e: float = Field(default=0, ge=0)
    n2o_mt_co2e: float = Field(default=0, ge=0)
    hfcs_mt_co2e: float = Field(default=0, ge=0)
    pfcs_mt_co2e: float = Field(default=0, ge=0)
    sf6_mt_co2e: float = Field(default=0, ge=0)


class Scope2Emissions(BaseModel):
    """Scope 2 indirect emissions (dual reporting required)."""
    location_based_mt_co2e: float = Field(..., ge=0)
    market_based_mt_co2e: float = Field(..., ge=0)

    # Breakdown by source
    purchased_electricity_location: float = Field(default=0, ge=0)
    purchased_electricity_market: float = Field(default=0, ge=0)
    purchased_steam_mt_co2e: float = Field(default=0, ge=0)
    purchased_heat_mt_co2e: float = Field(default=0, ge=0)
    purchased_cooling_mt_co2e: float = Field(default=0, ge=0)

    # Market instruments
    total_rec_kwh: float = Field(default=0, ge=0)
    total_ppa_kwh: float = Field(default=0, ge=0)


class Scope3Emissions(BaseModel):
    """Scope 3 value chain emissions (all 15 categories)."""
    total_mt_co2e: float = Field(..., ge=0)
    categories: Dict[str, Scope3Category] = Field(default_factory=dict)

    # Upstream totals
    upstream_total_mt_co2e: float = Field(default=0, ge=0)
    # Downstream totals
    downstream_total_mt_co2e: float = Field(default=0, ge=0)


class GHGInventory(BaseModel):
    """
    Complete GHG Inventory for SB 253 Disclosure.

    Aligns with GHG Protocol Corporate Standard and Scope 2/3 guidance.
    """
    # Company identification
    company_name: str
    ein: str
    reporting_year: int

    # Organizational boundary
    organizational_boundary: Literal[
        "equity_share",
        "operational_control",
        "financial_control"
    ]

    # Emissions by scope
    scope_1: Scope1Emissions
    scope_2: Scope2Emissions
    scope_3: Scope3Emissions

    # Totals
    total_emissions_mt_co2e: float = Field(..., ge=0)
    scope_1_2_total_mt_co2e: float = Field(..., ge=0)

    # Year-over-year comparison
    base_year: Optional[int] = None
    base_year_emissions_mt_co2e: Optional[float] = None
    yoy_change_percent: Optional[float] = None

    # Verification status
    verification_status: Literal[
        "unverified",
        "limited_assurance",
        "reasonable_assurance"
    ] = Field(default="unverified")

    # Complete audit trail
    audit_trail: List[AuditTrailRecord] = Field(default_factory=list)

    # Metadata
    calculation_timestamp: datetime
    greenlang_version: str
    methodology_version: str

    class Config:
        schema_extra = {
            "example": {
                "company_name": "Acme Corporation",
                "ein": "12-3456789",
                "reporting_year": 2025,
                "organizational_boundary": "operational_control",
                "scope_1": {
                    "total_mt_co2e": 5000,
                    "stationary_combustion_mt_co2e": 3500,
                    "mobile_combustion_mt_co2e": 1200,
                    "fugitive_emissions_mt_co2e": 300
                },
                "scope_2": {
                    "location_based_mt_co2e": 8000,
                    "market_based_mt_co2e": 4000
                },
                "scope_3": {
                    "total_mt_co2e": 120000,
                    "upstream_total_mt_co2e": 95000,
                    "downstream_total_mt_co2e": 25000
                },
                "total_emissions_mt_co2e": 133000,
                "scope_1_2_total_mt_co2e": 13000,
                "verification_status": "limited_assurance"
            }
        }
```

---

## 3. Calculation Module Specifications

### 3.1 Scope 1: Stationary Combustion Calculator

```python
# File: greenlang/calculators/sb253/scope1/stationary_combustion.py
"""
SB 253 Scope 1 Stationary Combustion Calculator

Calculates direct GHG emissions from stationary combustion sources:
- Boilers
- Furnaces
- Heaters
- Generators
- Other stationary equipment

Emission Factors: EPA GHG Emission Factors Hub 2024
GWP Values: IPCC AR6 (GWP-100)
"""

from typing import Dict, List, Optional, Tuple
from decimal import Decimal, ROUND_HALF_UP
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from ..base import BaseCalculator, CalculationResult, AuditRecord
from ...emission_factors.sb253.epa_ghg_factors import EPA_STATIONARY_FACTORS


@dataclass
class StationaryCombustionInput:
    """Input data for stationary combustion calculation."""
    facility_id: str
    fuel_type: str
    quantity: float
    unit: str
    reporting_period_start: str
    reporting_period_end: str
    equipment_type: Optional[str] = None
    source_document_id: Optional[str] = None


@dataclass
class EmissionFactorData:
    """Emission factor with metadata for audit trail."""
    factor_value: float
    factor_unit: str
    source: str
    source_uri: str
    version: str
    gwp_basis: str
    co2_factor: float
    ch4_factor: float
    n2o_factor: float


class StationaryCombustionCalculator(BaseCalculator):
    """
    Calculate Scope 1 emissions from stationary combustion.

    Formula:
        Emissions (kg CO2e) = Fuel Consumed (units) x Emission Factor (kg CO2e/unit)

    Accuracy Target: +/- 1%
    """

    CALCULATOR_ID = "sb253-scope1-stationary-v1"
    CALCULATOR_VERSION = "1.0.0"

    # EPA GHG Emission Factors Hub 2024 (kg CO2e per unit)
    EMISSION_FACTORS: Dict[str, EmissionFactorData] = {
        "natural_gas": EmissionFactorData(
            factor_value=5.30,
            factor_unit="kg CO2e/therm",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=5.27,
            ch4_factor=0.005,
            n2o_factor=0.0001
        ),
        "diesel": EmissionFactorData(
            factor_value=10.21,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=10.15,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "propane": EmissionFactorData(
            factor_value=5.72,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=5.68,
            ch4_factor=0.0003,
            n2o_factor=0.0001
        ),
        "fuel_oil_2": EmissionFactorData(
            factor_value=10.21,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=10.15,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "fuel_oil_6": EmissionFactorData(
            factor_value=11.27,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=11.21,
            ch4_factor=0.0004,
            n2o_factor=0.0001
        ),
        "kerosene": EmissionFactorData(
            factor_value=10.15,
            factor_unit="kg CO2e/gallon",
            source="EPA GHG Emission Factors Hub",
            source_uri="https://www.epa.gov/climateleadership/ghg-emission-factors-hub",
            version="2024",
            gwp_basis="IPCC AR6",
            co2_factor=10.10,
            ch4_factor=0.0003,
            n2o_factor=0.0001
        ),
    }

    # Unit conversion factors to standard units
    UNIT_CONVERSIONS: Dict[str, Dict[str, float]] = {
        "natural_gas": {
            "therms": 1.0,
            "kWh": 0.0341,  # 1 kWh = 0.0341 therms
            "MCF": 10.0,     # 1 MCF = 10 therms
            "MMBtu": 10.0,   # 1 MMBtu = 10 therms
        },
        "diesel": {
            "gallons": 1.0,
            "liters": 0.2642,  # 1 liter = 0.2642 gallons
        },
        "propane": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
        "fuel_oil_2": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
        "fuel_oil_6": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
        "kerosene": {
            "gallons": 1.0,
            "liters": 0.2642,
        },
    }

    def __init__(self):
        super().__init__(
            calculator_id=self.CALCULATOR_ID,
            version=self.CALCULATOR_VERSION
        )

    def calculate(
        self,
        inputs: List[StationaryCombustionInput]
    ) -> CalculationResult:
        """
        Calculate stationary combustion emissions for multiple fuel sources.

        Args:
            inputs: List of fuel consumption records

        Returns:
            CalculationResult with emissions and audit trail
        """
        total_emissions_kg = 0.0
        emissions_by_fuel = {}
        audit_records = []

        for input_data in inputs:
            # Validate fuel type
            if input_data.fuel_type not in self.EMISSION_FACTORS:
                raise ValueError(
                    f"Unsupported fuel type: {input_data.fuel_type}. "
                    f"Supported types: {list(self.EMISSION_FACTORS.keys())}"
                )

            # Get emission factor
            ef_data = self.EMISSION_FACTORS[input_data.fuel_type]

            # Convert units if necessary
            quantity_standardized = self._convert_units(
                input_data.quantity,
                input_data.unit,
                input_data.fuel_type
            )

            # Calculate emissions (deterministic - NO AI/estimation)
            emissions_kg = quantity_standardized * ef_data.factor_value

            # Calculate gas breakdown
            co2_kg = quantity_standardized * ef_data.co2_factor
            ch4_kg_co2e = quantity_standardized * ef_data.ch4_factor
            n2o_kg_co2e = quantity_standardized * ef_data.n2o_factor

            # Aggregate
            total_emissions_kg += emissions_kg
            if input_data.fuel_type not in emissions_by_fuel:
                emissions_by_fuel[input_data.fuel_type] = 0.0
            emissions_by_fuel[input_data.fuel_type] += emissions_kg

            # Create audit record
            audit_record = self._create_audit_record(
                input_data=input_data,
                ef_data=ef_data,
                quantity_standardized=quantity_standardized,
                emissions_kg=emissions_kg,
                co2_kg=co2_kg,
                ch4_kg_co2e=ch4_kg_co2e,
                n2o_kg_co2e=n2o_kg_co2e
            )
            audit_records.append(audit_record)

        # Convert to metric tons
        total_emissions_mt = total_emissions_kg / 1000.0

        return CalculationResult(
            success=True,
            scope="1",
            category="stationary_combustion",
            total_emissions_kg_co2e=total_emissions_kg,
            total_emissions_mt_co2e=total_emissions_mt,
            emissions_by_source=emissions_by_fuel,
            audit_records=audit_records,
            calculation_timestamp=datetime.utcnow().isoformat(),
            calculator_id=self.CALCULATOR_ID,
            calculator_version=self.CALCULATOR_VERSION
        )

    def _convert_units(
        self,
        quantity: float,
        unit: str,
        fuel_type: str
    ) -> float:
        """Convert input units to standard units for emission factor."""
        conversions = self.UNIT_CONVERSIONS.get(fuel_type, {})

        if unit not in conversions:
            raise ValueError(
                f"Unsupported unit '{unit}' for fuel type '{fuel_type}'. "
                f"Supported units: {list(conversions.keys())}"
            )

        conversion_factor = conversions[unit]
        return quantity * conversion_factor

    def _create_audit_record(
        self,
        input_data: StationaryCombustionInput,
        ef_data: EmissionFactorData,
        quantity_standardized: float,
        emissions_kg: float,
        co2_kg: float,
        ch4_kg_co2e: float,
        n2o_kg_co2e: float
    ) -> AuditRecord:
        """Create SHA-256 verified audit record."""

        # Create input hash
        input_dict = {
            "facility_id": input_data.facility_id,
            "fuel_type": input_data.fuel_type,
            "quantity": input_data.quantity,
            "unit": input_data.unit,
            "reporting_period_start": input_data.reporting_period_start,
            "reporting_period_end": input_data.reporting_period_end,
        }
        input_hash = hashlib.sha256(
            json.dumps(input_dict, sort_keys=True).encode()
        ).hexdigest()

        # Create output hash
        output_dict = {
            "emissions_kg_co2e": round(emissions_kg, 6),
            "co2_kg": round(co2_kg, 6),
            "ch4_kg_co2e": round(ch4_kg_co2e, 6),
            "n2o_kg_co2e": round(n2o_kg_co2e, 6),
        }
        output_hash = hashlib.sha256(
            json.dumps(output_dict, sort_keys=True).encode()
        ).hexdigest()

        return AuditRecord(
            calculation_id=f"{self.CALCULATOR_ID}-{input_hash[:12]}",
            timestamp=datetime.utcnow().isoformat(),
            scope="1",
            category="stationary_combustion",
            input_hash=input_hash,
            output_hash=output_hash,
            emission_factor_source=ef_data.source,
            emission_factor_version=ef_data.version,
            emission_factor_value=ef_data.factor_value,
            emission_factor_unit=ef_data.factor_unit,
            gwp_basis=ef_data.gwp_basis,
            calculation_formula=f"emissions = {quantity_standardized} x {ef_data.factor_value} = {emissions_kg} kg CO2e",
            inputs=input_dict,
            outputs=output_dict
        )
```

### 3.2 Scope 2: Location-Based Calculator

```python
# File: greenlang/calculators/sb253/scope2/location_based.py
"""
SB 253 Scope 2 Location-Based Electricity Calculator

Calculates indirect GHG emissions from purchased electricity using
grid average emission factors from EPA eGRID.

Emission Factors: EPA eGRID 2023
California Factor: CAMX = 0.254 kg CO2e/kWh
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from ..base import BaseCalculator, CalculationResult, AuditRecord


@dataclass
class ElectricityInput:
    """Input data for electricity consumption."""
    facility_id: str
    egrid_subregion: str
    quantity_kwh: float
    reporting_period_start: str
    reporting_period_end: str
    source_document_id: Optional[str] = None


class LocationBasedCalculator(BaseCalculator):
    """
    Calculate Scope 2 location-based emissions from purchased electricity.

    Formula:
        Emissions (kg CO2e) = Electricity (kWh) x Grid Factor (kg CO2e/kWh)

    Uses EPA eGRID 2023 subregional emission factors.
    CAMX (California) = 0.254 kg CO2e/kWh

    Accuracy Target: +/- 2%
    """

    CALCULATOR_ID = "sb253-scope2-location-v1"
    CALCULATOR_VERSION = "1.0.0"

    # EPA eGRID 2023 Subregional Emission Factors (kg CO2e/kWh)
    EGRID_FACTORS: Dict[str, Dict[str, any]] = {
        "CAMX": {
            "factor": 0.254,
            "name": "WECC California",
            "states": ["CA"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "AZNM": {
            "factor": 0.458,
            "name": "WECC Southwest",
            "states": ["AZ", "NM"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "NWPP": {
            "factor": 0.354,
            "name": "WECC Northwest",
            "states": ["WA", "OR", "ID", "MT", "WY", "NV", "UT", "CO"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "RMPA": {
            "factor": 0.684,
            "name": "WECC Rockies",
            "states": ["CO", "NE", "WY"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "ERCT": {
            "factor": 0.376,
            "name": "ERCOT All",
            "states": ["TX"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "FRCC": {
            "factor": 0.421,
            "name": "FRCC All",
            "states": ["FL"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "MROE": {
            "factor": 0.651,
            "name": "MRO East",
            "states": ["WI", "MI"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "MROW": {
            "factor": 0.583,
            "name": "MRO West",
            "states": ["MN", "IA", "ND", "SD", "NE", "MT"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "NEWE": {
            "factor": 0.183,
            "name": "NPCC New England",
            "states": ["CT", "MA", "ME", "NH", "RI", "VT"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "NYCW": {
            "factor": 0.260,
            "name": "NPCC NYC/Westchester",
            "states": ["NY"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "NYUP": {
            "factor": 0.160,
            "name": "NPCC Upstate NY",
            "states": ["NY"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "RFCE": {
            "factor": 0.355,
            "name": "RFC East",
            "states": ["PA", "NJ", "MD", "DE", "DC"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "RFCM": {
            "factor": 0.572,
            "name": "RFC Michigan",
            "states": ["MI"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "RFCW": {
            "factor": 0.568,
            "name": "RFC West",
            "states": ["OH", "IN", "KY", "WV"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SRMW": {
            "factor": 0.713,
            "name": "SERC Midwest",
            "states": ["MO", "IL", "AR"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SRMV": {
            "factor": 0.432,
            "name": "SERC Mississippi Valley",
            "states": ["LA", "MS", "AR"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SRSO": {
            "factor": 0.455,
            "name": "SERC South",
            "states": ["GA", "AL"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SRTV": {
            "factor": 0.478,
            "name": "SERC Tennessee Valley",
            "states": ["TN", "NC", "KY"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SRVC": {
            "factor": 0.379,
            "name": "SERC Virginia/Carolina",
            "states": ["VA", "NC", "SC"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SPNO": {
            "factor": 0.583,
            "name": "SPP North",
            "states": ["KS", "NE", "OK"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
        "SPSO": {
            "factor": 0.512,
            "name": "SPP South",
            "states": ["OK", "TX", "LA", "AR"],
            "source": "EPA eGRID 2023",
            "data_year": 2022,
        },
    }

    def __init__(self):
        super().__init__(
            calculator_id=self.CALCULATOR_ID,
            version=self.CALCULATOR_VERSION
        )

    def calculate(
        self,
        inputs: List[ElectricityInput]
    ) -> CalculationResult:
        """
        Calculate location-based Scope 2 emissions.

        Args:
            inputs: List of electricity consumption records

        Returns:
            CalculationResult with emissions and audit trail
        """
        total_emissions_kg = 0.0
        total_kwh = 0.0
        emissions_by_region = {}
        audit_records = []

        for input_data in inputs:
            # Validate eGRID subregion
            if input_data.egrid_subregion not in self.EGRID_FACTORS:
                raise ValueError(
                    f"Unknown eGRID subregion: {input_data.egrid_subregion}. "
                    f"Supported regions: {list(self.EGRID_FACTORS.keys())}"
                )

            # Get grid factor
            grid_data = self.EGRID_FACTORS[input_data.egrid_subregion]
            grid_factor = grid_data["factor"]

            # Calculate emissions (deterministic)
            emissions_kg = input_data.quantity_kwh * grid_factor

            # Aggregate
            total_emissions_kg += emissions_kg
            total_kwh += input_data.quantity_kwh

            region_key = input_data.egrid_subregion
            if region_key not in emissions_by_region:
                emissions_by_region[region_key] = {
                    "emissions_kg_co2e": 0.0,
                    "kwh": 0.0,
                    "factor": grid_factor,
                    "region_name": grid_data["name"]
                }
            emissions_by_region[region_key]["emissions_kg_co2e"] += emissions_kg
            emissions_by_region[region_key]["kwh"] += input_data.quantity_kwh

            # Create audit record
            audit_record = self._create_audit_record(
                input_data=input_data,
                grid_data=grid_data,
                emissions_kg=emissions_kg
            )
            audit_records.append(audit_record)

        # Convert to metric tons
        total_emissions_mt = total_emissions_kg / 1000.0

        return CalculationResult(
            success=True,
            scope="2",
            category="location_based",
            total_emissions_kg_co2e=total_emissions_kg,
            total_emissions_mt_co2e=total_emissions_mt,
            emissions_by_source=emissions_by_region,
            audit_records=audit_records,
            calculation_timestamp=datetime.utcnow().isoformat(),
            calculator_id=self.CALCULATOR_ID,
            calculator_version=self.CALCULATOR_VERSION,
            metadata={
                "total_kwh": total_kwh,
                "method": "location_based",
                "factor_source": "EPA eGRID 2023"
            }
        )

    def _create_audit_record(
        self,
        input_data: ElectricityInput,
        grid_data: Dict,
        emissions_kg: float
    ) -> AuditRecord:
        """Create audit record with provenance hash."""

        input_dict = {
            "facility_id": input_data.facility_id,
            "egrid_subregion": input_data.egrid_subregion,
            "quantity_kwh": input_data.quantity_kwh,
            "reporting_period_start": input_data.reporting_period_start,
            "reporting_period_end": input_data.reporting_period_end,
        }
        input_hash = hashlib.sha256(
            json.dumps(input_dict, sort_keys=True).encode()
        ).hexdigest()

        output_dict = {
            "emissions_kg_co2e": round(emissions_kg, 6),
        }
        output_hash = hashlib.sha256(
            json.dumps(output_dict, sort_keys=True).encode()
        ).hexdigest()

        return AuditRecord(
            calculation_id=f"{self.CALCULATOR_ID}-{input_hash[:12]}",
            timestamp=datetime.utcnow().isoformat(),
            scope="2",
            category="location_based",
            input_hash=input_hash,
            output_hash=output_hash,
            emission_factor_source=grid_data["source"],
            emission_factor_version="2023",
            emission_factor_value=grid_data["factor"],
            emission_factor_unit="kg CO2e/kWh",
            gwp_basis="IPCC AR6",
            calculation_formula=f"emissions = {input_data.quantity_kwh} kWh x {grid_data['factor']} = {emissions_kg} kg CO2e",
            inputs=input_dict,
            outputs=output_dict
        )
```

---

## 4. California-Specific Requirements

### 4.1 CARB Portal Integration

```python
# File: greenlang/integrations/sb253/carb_portal.py
"""
California Air Resources Board (CARB) Portal Integration

Handles SB 253 filing submission to the CARB reporting portal.
NOTE: CARB portal API specifications pending official release.
"""

from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import xml.etree.ElementTree as ET
import json

from ...schemas.sb253.ghg_inventory import GHGInventory
from ...schemas.sb253.company_profile import CompanyProfile


@dataclass
class CARBFilingResult:
    """Result of CARB filing submission."""
    success: bool
    filing_id: Optional[str]
    submission_timestamp: str
    confirmation_number: Optional[str]
    validation_errors: List[str]
    status: str  # "submitted", "pending_review", "accepted", "rejected"


class CARBPortalIntegration:
    """
    CARB SB 253 Portal Integration.

    Generates filing documents and submits to CARB portal.
    Portal API pending - currently generates XML/JSON for manual upload.
    """

    SCHEMA_VERSION = "1.0.0"
    FILING_TYPE = "SB253_CLIMATE_DISCLOSURE"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize CARB portal integration.

        Args:
            api_key: CARB API key (when available)
        """
        self.api_key = api_key
        self.portal_available = False  # Set True when CARB releases portal

    def generate_filing_xml(
        self,
        company: CompanyProfile,
        inventory: GHGInventory,
        reporting_year: int
    ) -> str:
        """
        Generate CARB SB 253 filing XML document.

        Args:
            company: Company profile data
            inventory: GHG inventory data
            reporting_year: Year being reported

        Returns:
            XML string for filing
        """
        root = ET.Element("SB253Filing")
        root.set("schemaVersion", self.SCHEMA_VERSION)
        root.set("filingType", self.FILING_TYPE)

        # Company information
        company_elem = ET.SubElement(root, "Company")
        ET.SubElement(company_elem, "Name").text = company.company_name
        ET.SubElement(company_elem, "EIN").text = company.ein
        ET.SubElement(company_elem, "TotalRevenue").text = str(company.total_revenue)
        ET.SubElement(company_elem, "CaliforniaRevenue").text = str(company.california_revenue)
        ET.SubElement(company_elem, "NAICSCode").text = company.naics_code

        # Reporting metadata
        reporting_elem = ET.SubElement(root, "ReportingPeriod")
        ET.SubElement(reporting_elem, "Year").text = str(reporting_year)
        ET.SubElement(reporting_elem, "OrganizationalBoundary").text = company.organizational_boundary

        # Scope 1 emissions
        scope1_elem = ET.SubElement(root, "Scope1Emissions")
        ET.SubElement(scope1_elem, "TotalMTCO2e").text = str(inventory.scope_1.total_mt_co2e)
        ET.SubElement(scope1_elem, "StationaryCombustion").text = str(inventory.scope_1.stationary_combustion_mt_co2e)
        ET.SubElement(scope1_elem, "MobileCombustion").text = str(inventory.scope_1.mobile_combustion_mt_co2e)
        ET.SubElement(scope1_elem, "FugitiveEmissions").text = str(inventory.scope_1.fugitive_emissions_mt_co2e)
        ET.SubElement(scope1_elem, "ProcessEmissions").text = str(inventory.scope_1.process_emissions_mt_co2e)

        # Scope 2 emissions (dual reporting)
        scope2_elem = ET.SubElement(root, "Scope2Emissions")
        ET.SubElement(scope2_elem, "LocationBasedMTCO2e").text = str(inventory.scope_2.location_based_mt_co2e)
        ET.SubElement(scope2_elem, "MarketBasedMTCO2e").text = str(inventory.scope_2.market_based_mt_co2e)

        # Scope 3 emissions
        scope3_elem = ET.SubElement(root, "Scope3Emissions")
        ET.SubElement(scope3_elem, "TotalMTCO2e").text = str(inventory.scope_3.total_mt_co2e)

        # Add each category
        for cat_key, cat_data in inventory.scope_3.categories.items():
            cat_elem = ET.SubElement(scope3_elem, "Category")
            cat_elem.set("number", str(cat_data.category_number))
            ET.SubElement(cat_elem, "Name").text = cat_data.category_name
            ET.SubElement(cat_elem, "EmissionsMTCO2e").text = str(cat_data.emissions_mt_co2e)
            ET.SubElement(cat_elem, "CalculationMethod").text = cat_data.calculation_method
            ET.SubElement(cat_elem, "DataQualityScore").text = str(cat_data.data_quality_score)

        # Total emissions
        totals_elem = ET.SubElement(root, "TotalEmissions")
        ET.SubElement(totals_elem, "TotalMTCO2e").text = str(inventory.total_emissions_mt_co2e)
        ET.SubElement(totals_elem, "Scope12TotalMTCO2e").text = str(inventory.scope_1_2_total_mt_co2e)

        # Verification status
        verification_elem = ET.SubElement(root, "Verification")
        ET.SubElement(verification_elem, "Status").text = inventory.verification_status

        # Convert to string
        return ET.tostring(root, encoding='unicode', method='xml')

    def generate_filing_json(
        self,
        company: CompanyProfile,
        inventory: GHGInventory,
        reporting_year: int
    ) -> Dict:
        """
        Generate CARB SB 253 filing JSON document.

        Args:
            company: Company profile data
            inventory: GHG inventory data
            reporting_year: Year being reported

        Returns:
            Dictionary for JSON filing
        """
        return {
            "schema_version": self.SCHEMA_VERSION,
            "filing_type": self.FILING_TYPE,
            "generated_at": datetime.utcnow().isoformat(),

            "company": {
                "name": company.company_name,
                "ein": company.ein,
                "total_revenue_usd": company.total_revenue,
                "california_revenue_usd": company.california_revenue,
                "naics_code": company.naics_code,
            },

            "reporting": {
                "year": reporting_year,
                "organizational_boundary": company.organizational_boundary,
            },

            "emissions": {
                "scope_1": {
                    "total_mt_co2e": inventory.scope_1.total_mt_co2e,
                    "stationary_combustion_mt_co2e": inventory.scope_1.stationary_combustion_mt_co2e,
                    "mobile_combustion_mt_co2e": inventory.scope_1.mobile_combustion_mt_co2e,
                    "fugitive_emissions_mt_co2e": inventory.scope_1.fugitive_emissions_mt_co2e,
                    "process_emissions_mt_co2e": inventory.scope_1.process_emissions_mt_co2e,
                },
                "scope_2": {
                    "location_based_mt_co2e": inventory.scope_2.location_based_mt_co2e,
                    "market_based_mt_co2e": inventory.scope_2.market_based_mt_co2e,
                },
                "scope_3": {
                    "total_mt_co2e": inventory.scope_3.total_mt_co2e,
                    "categories": {
                        k: {
                            "category_number": v.category_number,
                            "category_name": v.category_name,
                            "emissions_mt_co2e": v.emissions_mt_co2e,
                            "calculation_method": v.calculation_method,
                            "data_quality_score": v.data_quality_score,
                        }
                        for k, v in inventory.scope_3.categories.items()
                    }
                },
                "total_mt_co2e": inventory.total_emissions_mt_co2e,
                "scope_1_2_total_mt_co2e": inventory.scope_1_2_total_mt_co2e,
            },

            "verification": {
                "status": inventory.verification_status,
            }
        }
```

### 4.2 California Grid Emission Factors

```python
# File: greenlang/emission_factors/sb253/california_grid_factors.py
"""
California Grid Emission Factors for SB 253

Source: EPA eGRID 2023
California Subregion: CAMX (WECC California)
Factor: 0.254 kg CO2e/kWh

Also includes California utility-specific factors for market-based reporting.
"""

from typing import Dict
from dataclasses import dataclass
from datetime import datetime


@dataclass
class GridFactor:
    """Grid emission factor with metadata."""
    factor_kg_co2e_per_kwh: float
    source: str
    source_uri: str
    data_year: int
    last_updated: str
    states: list
    renewable_percentage: float
    notes: str


# California CAMX Grid Factor (EPA eGRID 2023)
CALIFORNIA_CAMX = GridFactor(
    factor_kg_co2e_per_kwh=0.254,
    source="EPA eGRID 2023",
    source_uri="https://www.epa.gov/egrid/download-data",
    data_year=2022,
    last_updated="2024-11-01",
    states=["CA"],
    renewable_percentage=44.5,  # Solar + Wind + Hydro
    notes="California has one of the cleanest grids in the US due to high renewable penetration (SB 100)"
)


# California Utility-Specific Factors (for market-based Scope 2)
CALIFORNIA_UTILITIES: Dict[str, GridFactor] = {
    "PGE": GridFactor(
        factor_kg_co2e_per_kwh=0.210,
        source="PG&E Power Content Label 2023",
        source_uri="https://www.pge.com/en/about/environment/what-we-are-doing/clean-energy-solutions/power-content-label.html",
        data_year=2023,
        last_updated="2024-06-01",
        states=["CA"],
        renewable_percentage=52,
        notes="Pacific Gas & Electric - Northern and Central California"
    ),
    "SCE": GridFactor(
        factor_kg_co2e_per_kwh=0.195,
        source="SCE Power Content Label 2023",
        source_uri="https://www.sce.com/about-us/environment/power-content-label",
        data_year=2023,
        last_updated="2024-06-01",
        states=["CA"],
        renewable_percentage=48,
        notes="Southern California Edison - Southern California (excluding LA)"
    ),
    "SDGE": GridFactor(
        factor_kg_co2e_per_kwh=0.235,
        source="SDG&E Power Content Label 2023",
        source_uri="https://www.sdge.com/regulatory-filing/power-content-label",
        data_year=2023,
        last_updated="2024-06-01",
        states=["CA"],
        renewable_percentage=45,
        notes="San Diego Gas & Electric - San Diego County"
    ),
    "LADWP": GridFactor(
        factor_kg_co2e_per_kwh=0.340,
        source="LADWP Power Content Label 2023",
        source_uri="https://www.ladwp.com/ladwp/faces/ladwp/aboutus/a-power",
        data_year=2023,
        last_updated="2024-06-01",
        states=["CA"],
        renewable_percentage=38,
        notes="Los Angeles Department of Water and Power - City of Los Angeles"
    ),
}


def get_california_grid_factor(
    utility: str = None,
    method: str = "location"
) -> float:
    """
    Get California grid emission factor.

    Args:
        utility: Specific utility (PGE, SCE, SDGE, LADWP) for market-based
        method: "location" for CAMX, "market" for utility-specific

    Returns:
        Emission factor in kg CO2e/kWh
    """
    if method == "location" or utility is None:
        return CALIFORNIA_CAMX.factor_kg_co2e_per_kwh

    if utility.upper() in CALIFORNIA_UTILITIES:
        return CALIFORNIA_UTILITIES[utility.upper()].factor_kg_co2e_per_kwh

    # Default to CAMX if utility not found
    return CALIFORNIA_CAMX.factor_kg_co2e_per_kwh


def get_factor_metadata(
    utility: str = None
) -> Dict:
    """Get emission factor metadata for audit trail."""
    if utility and utility.upper() in CALIFORNIA_UTILITIES:
        factor = CALIFORNIA_UTILITIES[utility.upper()]
    else:
        factor = CALIFORNIA_CAMX

    return {
        "factor_kg_co2e_per_kwh": factor.factor_kg_co2e_per_kwh,
        "source": factor.source,
        "source_uri": factor.source_uri,
        "data_year": factor.data_year,
        "last_updated": factor.last_updated,
        "renewable_percentage": factor.renewable_percentage,
        "notes": factor.notes,
    }
```

---

## 5. Implementation Tasks (Prioritized)

### 5.1 Phase 1: Foundation (Weeks 1-8) - CRITICAL PATH

| Task ID | Task Name | Priority | Duration | Dependencies | Assignee |
|---------|-----------|----------|----------|--------------|----------|
| **S1-001** | Stationary Combustion Calculator | P0 | 1 week | None | Backend |
| **S1-002** | Mobile Combustion Calculator | P0 | 1 week | S1-001 | Backend |
| **S1-003** | Fugitive Emissions Calculator | P0 | 1 week | None | Backend |
| **S1-004** | Process Emissions Calculator | P1 | 1 week | None | Backend |
| **S1-005** | Scope 1 Aggregator | P0 | 0.5 week | S1-001 to S1-004 | Backend |
| **S2-001** | Location-Based Electricity | P0 | 1 week | None | Backend |
| **S2-002** | Market-Based Electricity | P1 | 1 week | S2-001 | Backend |
| **S2-003** | Scope 2 Dual Reporting | P0 | 0.5 week | S2-001, S2-002 | Backend |
| **SCHEMA-001** | Input Schemas (Company, Facility) | P0 | 0.5 week | None | Backend |
| **SCHEMA-002** | Output Schemas (GHG Inventory) | P0 | 0.5 week | None | Backend |

### 5.2 Phase 2: Value Chain (Weeks 9-24)

| Task ID | Task Name | Priority | Duration | Dependencies |
|---------|-----------|----------|----------|--------------|
| **S3-01** | Category 1: Purchased Goods | P1 | 1.5 weeks | SCHEMA-002 |
| **S3-02** | Category 2: Capital Goods | P2 | 1 week | S3-01 |
| **S3-03** | Category 3: Fuel/Energy | P1 | 1 week | S1-005, S2-003 |
| **S3-04** | Category 4: Upstream Transport | P1 | 1.5 weeks | SCHEMA-002 |
| **S3-05** | Category 5: Waste | P1 | 1 week | SCHEMA-002 |
| **S3-06** | Category 6: Business Travel | P1 | 1.5 weeks | SCHEMA-002 |
| **S3-07** | Category 7: Employee Commuting | P1 | 1 week | SCHEMA-002 |
| **S3-08** | Category 8: Upstream Leased | P2 | 0.5 week | S2-003 |
| **S3-09** | Category 9: Downstream Transport | P2 | 0.5 week | S3-04 |
| **S3-10** | Category 10: Processing | P2 | 0.5 week | SCHEMA-002 |
| **S3-11** | Category 11: Use of Sold | P1 | 1 week | SCHEMA-002 |
| **S3-12** | Category 12: End-of-Life | P2 | 0.5 week | S3-05 |
| **S3-13** | Category 13: Downstream Leased | P2 | 0.5 week | S2-003 |
| **S3-14** | Category 14: Franchises | P2 | 0.5 week | SCHEMA-002 |
| **S3-15** | Category 15: Investments | P1 | 1 week | SCHEMA-002 |
| **S3-AGG** | Scope 3 Aggregator | P0 | 1 week | S3-01 to S3-15 |

### 5.3 Phase 3: Verification & Filing (Weeks 25-32)

| Task ID | Task Name | Priority | Duration | Dependencies |
|---------|-----------|----------|----------|--------------|
| **V-001** | Audit Trail Generator | P0 | 1 week | All S1/S2/S3 |
| **V-002** | Assurance Package Generator | P0 | 1.5 weeks | V-001 |
| **V-003** | DQI Scorer | P1 | 1 week | All S3 |
| **V-004** | CARB Portal Integration | P0 | 2 weeks | All schemas |
| **V-005** | Multi-State Filing (CO, WA) | P2 | 1 week | V-004 |
| **V-006** | Third-Party Verifier Support | P1 | 1 week | V-002 |

### 5.4 Phase 4: Testing & Launch (Weeks 33-36)

| Task ID | Task Name | Priority | Duration | Dependencies |
|---------|-----------|----------|----------|--------------|
| **TEST-001** | Scope 1 Golden Tests (60) | P0 | 1 week | S1-005 |
| **TEST-002** | Scope 2 Golden Tests (70) | P0 | 1 week | S2-003 |
| **TEST-003** | Scope 3 Golden Tests (120) | P0 | 2 weeks | S3-AGG |
| **TEST-004** | Verification Tests (50) | P0 | 1 week | V-006 |
| **BETA-001** | Pilot Company Onboarding | P0 | 1 week | All tests pass |
| **DEPLOY-001** | Production Deployment | P0 | 1 week | BETA-001 |

---

## 6. Testing Strategy

### 6.1 Golden Test Requirements

```python
# File: tests/golden/sb253/test_scope1_golden.py
"""
SB 253 Scope 1 Golden Tests

60 test scenarios covering:
- Stationary combustion (20 tests)
- Mobile combustion (15 tests)
- Fugitive emissions (10 tests)
- Process emissions (10 tests)
- Aggregation (5 tests)
"""

import pytest
from decimal import Decimal
from greenlang.calculators.sb253.scope1 import (
    StationaryCombustionCalculator,
    MobileCombustionCalculator,
    FugitiveEmissionsCalculator,
    ProcessEmissionsCalculator,
    Scope1Aggregator
)


class TestStationaryCombustionGolden:
    """Golden tests for stationary combustion."""

    @pytest.mark.golden
    def test_natural_gas_boiler_typical(self):
        """
        Golden Test S1-SC-001: Natural gas boiler - typical office building

        Input: 50,000 therms natural gas
        Expected: 265,000 kg CO2e (+/- 1%)
        Source: EPA GHG EF Hub 2024 (5.30 kg CO2e/therm)
        """
        calculator = StationaryCombustionCalculator()
        result = calculator.calculate([{
            "facility_id": "FAC-001",
            "fuel_type": "natural_gas",
            "quantity": 50000,
            "unit": "therms",
            "reporting_period_start": "2025-01-01",
            "reporting_period_end": "2025-12-31"
        }])

        expected = 265000  # 50,000 x 5.30
        tolerance = expected * 0.01  # 1%

        assert result.success
        assert abs(result.total_emissions_kg_co2e - expected) <= tolerance
        assert result.audit_records[0].emission_factor_value == 5.30

    @pytest.mark.golden
    def test_diesel_generator_backup(self):
        """
        Golden Test S1-SC-002: Diesel generator - backup power

        Input: 5,000 gallons diesel
        Expected: 51,050 kg CO2e (+/- 1%)
        Source: EPA GHG EF Hub 2024 (10.21 kg CO2e/gallon)
        """
        calculator = StationaryCombustionCalculator()
        result = calculator.calculate([{
            "facility_id": "FAC-001",
            "fuel_type": "diesel",
            "quantity": 5000,
            "unit": "gallons",
            "reporting_period_start": "2025-01-01",
            "reporting_period_end": "2025-12-31"
        }])

        expected = 51050  # 5,000 x 10.21
        tolerance = expected * 0.01

        assert result.success
        assert abs(result.total_emissions_kg_co2e - expected) <= tolerance

    @pytest.mark.golden
    def test_california_manufacturing_facility(self):
        """
        Golden Test S1-SC-010: California manufacturing - multi-fuel

        Input:
        - 200,000 therms natural gas
        - 10,000 gallons diesel
        - 2,000 gallons propane

        Expected: 1,173,440 kg CO2e (+/- 1%)
        """
        calculator = StationaryCombustionCalculator()
        result = calculator.calculate([
            {
                "facility_id": "FAC-CA-001",
                "fuel_type": "natural_gas",
                "quantity": 200000,
                "unit": "therms",
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31"
            },
            {
                "facility_id": "FAC-CA-001",
                "fuel_type": "diesel",
                "quantity": 10000,
                "unit": "gallons",
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31"
            },
            {
                "facility_id": "FAC-CA-001",
                "fuel_type": "propane",
                "quantity": 2000,
                "unit": "gallons",
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31"
            }
        ])

        expected = (200000 * 5.30) + (10000 * 10.21) + (2000 * 5.72)
        # = 1,060,000 + 102,100 + 11,440 = 1,173,540
        tolerance = expected * 0.01

        assert result.success
        assert abs(result.total_emissions_kg_co2e - expected) <= tolerance
        assert len(result.audit_records) == 3


class TestScope2LocationBasedGolden:
    """Golden tests for Scope 2 location-based calculations."""

    @pytest.mark.golden
    def test_california_office_camx(self):
        """
        Golden Test S2-LB-001: California office - CAMX grid

        Input: 1,000,000 kWh electricity
        Expected: 254,000 kg CO2e (+/- 2%)
        Source: EPA eGRID 2023 CAMX (0.254 kg CO2e/kWh)
        """
        from greenlang.calculators.sb253.scope2 import LocationBasedCalculator

        calculator = LocationBasedCalculator()
        result = calculator.calculate([{
            "facility_id": "FAC-CA-001",
            "egrid_subregion": "CAMX",
            "quantity_kwh": 1000000,
            "reporting_period_start": "2025-01-01",
            "reporting_period_end": "2025-12-31"
        }])

        expected = 254000  # 1,000,000 x 0.254
        tolerance = expected * 0.02  # 2%

        assert result.success
        assert abs(result.total_emissions_kg_co2e - expected) <= tolerance

    @pytest.mark.golden
    def test_multi_state_operations(self):
        """
        Golden Test S2-LB-015: Multi-state - CA, TX, NY facilities

        Input:
        - CA (CAMX): 2,000,000 kWh
        - TX (ERCT): 1,500,000 kWh
        - NY (NYUP): 800,000 kWh

        Expected: 1,200,000 kg CO2e (+/- 2%)
        """
        from greenlang.calculators.sb253.scope2 import LocationBasedCalculator

        calculator = LocationBasedCalculator()
        result = calculator.calculate([
            {
                "facility_id": "FAC-CA-001",
                "egrid_subregion": "CAMX",
                "quantity_kwh": 2000000,
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31"
            },
            {
                "facility_id": "FAC-TX-001",
                "egrid_subregion": "ERCT",
                "quantity_kwh": 1500000,
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31"
            },
            {
                "facility_id": "FAC-NY-001",
                "egrid_subregion": "NYUP",
                "quantity_kwh": 800000,
                "reporting_period_start": "2025-01-01",
                "reporting_period_end": "2025-12-31"
            }
        ])

        expected = (2000000 * 0.254) + (1500000 * 0.376) + (800000 * 0.160)
        # = 508,000 + 564,000 + 128,000 = 1,200,000
        tolerance = expected * 0.02

        assert result.success
        assert abs(result.total_emissions_kg_co2e - expected) <= tolerance
```

### 6.2 Test Coverage Requirements

| Component | Unit Tests | Integration Tests | Golden Tests | Total Coverage |
|-----------|------------|-------------------|--------------|----------------|
| Scope 1 Calculators | 40 | 10 | 60 | 95%+ |
| Scope 2 Calculators | 30 | 8 | 70 | 95%+ |
| Scope 3 Calculators | 60 | 15 | 120 | 90%+ |
| Verification | 20 | 10 | 50 | 90%+ |
| CARB Integration | 15 | 10 | 0 | 85%+ |
| **Total** | **165** | **53** | **300** | **90%+** |

---

## 7. Task Estimates Summary

### 7.1 Development Time by Phase

| Phase | Duration | Tasks | Engineers |
|-------|----------|-------|-----------|
| Phase 1: Foundation | 8 weeks | 10 | 2 Backend |
| Phase 2: Value Chain | 16 weeks | 17 | 2 Backend |
| Phase 3: Verification | 8 weeks | 6 | 1 Backend, 1 Integration |
| Phase 4: Testing & Launch | 4 weeks | 6 | 2 QA, 1 DevOps |
| **Total** | **36 weeks** | **39 core tasks** | **4-5 engineers** |

### 7.2 Critical Path

```
Week 1-4:   S1-001 -> S1-002 -> S1-005 (Scope 1 Foundation)
Week 5-8:   S2-001 -> S2-002 -> S2-003 (Scope 2 Foundation)
Week 9-20:  S3-01 through S3-15 parallel tracks
Week 21-24: S3-AGG + Integration testing
Week 25-28: V-001 -> V-002 (Assurance)
Week 29-32: V-004 (CARB Portal)
Week 33-36: Golden tests + Beta + Deploy
```

### 7.3 Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| CARB portal delayed | High | Manual filing fallback ready |
| Supplier data gaps | Medium | Spend-based calculation fallback |
| Scope 3 complexity | Medium | Prioritize material categories first |
| Big 4 assurance requirements | Medium | Early engagement with Deloitte/EY |

---

## 8. Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-12-04 | GL-SB253-PM | Initial comprehensive plan |

**Approvals Required:**

- [ ] Climate Science Team Lead
- [ ] Engineering Lead
- [ ] Product Manager
- [ ] Compliance Officer
- [ ] Program Director

---

**END OF IMPLEMENTATION PLAN**

**Total Implementation Effort:** 36 weeks, 4-5 engineers
**Deadline:** June 30, 2026 (Scope 1 & 2)
**Golden Tests:** 300+ scenarios
**Target Accuracy:** Scope 1: +/-1%, Scope 2: +/-2%, Scope 3: +/-5%
