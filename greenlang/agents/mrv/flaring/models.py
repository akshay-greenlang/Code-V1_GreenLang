# -*- coding: utf-8 -*-
"""
Flaring Agent Data Models - AGENT-MRV-006

Pydantic v2 data models for the Flaring Agent SDK covering Scope 1 GHG
emissions from flaring operations at industrial facilities. Includes:

- 8 flare types (elevated steam/air/unassisted, enclosed ground, MPGF,
  offshore, candlestick, low-pressure)
- 6 flaring event categories (routine, non-routine, emergency, maintenance,
  pilot/purge, well completion)
- 4 calculation methods (gas composition, default EF, engineering estimate,
  direct measurement)
- 15 gas composition components with heating values, molecular weights,
  and carbon counts
- 8 regulatory frameworks (GHG Protocol, ISO 14064, CSRD, EPA Subpart W,
  EU ETS MRR, EU Methane Reg, World Bank ZRF, OGMP 2.0)
- Monte Carlo uncertainty quantification models
- Combustion efficiency modeling with wind/tip velocity/assist adjustments
- Pilot and purge gas accounting
- OGMP 2.0 five-level reporting hierarchy

Enumerations (16):
    FlareType, FlaringEventCategory, CalculationMethod,
    EmissionFactorSource, GasComponent, EmissionGas, GWPSource,
    StandardCondition, AssistType, FlaringStatus, OGMPLevel,
    ComplianceFramework, CalculationStatus, DataQualityTier,
    SeverityLevel, ComplianceStatus

Data Models (16+):
    GasComposition, FlareSystemConfig, FlaringEventRecord,
    PilotPurgeConfig, CombustionEfficiencyParams, CalculationInput,
    CalculationResult, EmissionDetail, BatchCalculationRequest,
    BatchCalculationResponse, UncertaintyInput, UncertaintyResult,
    ComplianceCheckInput, ComplianceCheckResult, FlareSystemRegistration,
    FlaringStats, HealthResponse

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-006 Flaring Agent (GL-MRV-SCOPE1-006)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Service version string.
VERSION: str = "1.0.0"

#: Maximum number of calculations in a single batch request.
MAX_CALCULATIONS_PER_BATCH: int = 10_000

#: Maximum number of gas emission detail entries per calculation result.
MAX_GASES_PER_RESULT: int = 10

#: Maximum number of trace steps in a single calculation.
MAX_TRACE_STEPS: int = 200

#: Maximum number of gas composition components per analysis.
MAX_COMPOSITION_COMPONENTS: int = 20

#: CO2 molecular weight (g/mol).
CO2_MOLECULAR_WEIGHT: Decimal = Decimal("44.01")

#: Carbon atomic weight (g/mol).
CARBON_ATOMIC_WEIGHT: Decimal = Decimal("12.01")

#: Standard CO2, CH4, N2O GWP values by Assessment Report.
GWP_VALUES: Dict[str, Dict[str, float]] = {
    "AR4": {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0},
    "AR5": {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0},
    "AR6": {"CO2": 1.0, "CH4": 27.3, "N2O": 273.0},
    "AR6_20YR": {"CO2": 1.0, "CH4": 81.2, "N2O": 273.0},
}

#: Component Higher Heating Values in BTU per standard cubic foot (BTU/scf).
COMPONENT_HHV_BTU_SCF: Dict[str, Decimal] = {
    "CH4": Decimal("1012"),
    "C2H6": Decimal("1773"),
    "C3H8": Decimal("2524"),
    "N_C4H10": Decimal("3271"),
    "I_C4H10": Decimal("3254"),
    "C5H12": Decimal("4010"),
    "C6_PLUS": Decimal("4756"),
    "H2": Decimal("325"),
    "CO": Decimal("321"),
    "C2H4": Decimal("1614"),
    "C3H6": Decimal("2336"),
    # Non-combustible components have zero HHV
    "CO2": Decimal("0"),
    "N2": Decimal("0"),
    "H2S": Decimal("0"),
    "H2O": Decimal("0"),
}

#: Component molecular weights (g/mol).
COMPONENT_MOLECULAR_WEIGHTS: Dict[str, Decimal] = {
    "CH4": Decimal("16.04"),
    "C2H6": Decimal("30.07"),
    "C3H8": Decimal("44.10"),
    "N_C4H10": Decimal("58.12"),
    "I_C4H10": Decimal("58.12"),
    "C5H12": Decimal("72.15"),
    "C6_PLUS": Decimal("86.18"),
    "CO2": Decimal("44.01"),
    "N2": Decimal("28.01"),
    "H2S": Decimal("34.08"),
    "H2": Decimal("2.016"),
    "CO": Decimal("28.01"),
    "C2H4": Decimal("28.05"),
    "C3H6": Decimal("42.08"),
    "H2O": Decimal("18.015"),
}

#: Number of carbon atoms per molecule for each component.
COMPONENT_CARBON_COUNT: Dict[str, int] = {
    "CH4": 1,
    "C2H6": 2,
    "C3H8": 3,
    "N_C4H10": 4,
    "I_C4H10": 4,
    "C5H12": 5,
    "C6_PLUS": 6,
    "CO2": 1,
    "CO": 1,
    "C2H4": 2,
    "C3H6": 3,
    # Non-carbon-containing components
    "N2": 0,
    "H2S": 0,
    "H2": 0,
    "H2O": 0,
}

#: Default combustion efficiency by flare type (fraction, 0.0-1.0).
DEFAULT_COMBUSTION_EFFICIENCY: Dict[str, Decimal] = {
    "ELEVATED_STEAM_ASSISTED": Decimal("0.98"),
    "ELEVATED_AIR_ASSISTED": Decimal("0.98"),
    "ELEVATED_UNASSISTED": Decimal("0.98"),
    "ENCLOSED_GROUND": Decimal("0.99"),
    "MULTI_POINT_GROUND": Decimal("0.99"),
    "OFFSHORE_MARINE": Decimal("0.98"),
    "CANDLESTICK": Decimal("0.96"),
    "LOW_PRESSURE": Decimal("0.95"),
}


# =============================================================================
# Enumerations (16)
# =============================================================================


class FlareType(str, Enum):
    """Classification of flare system design and configuration.

    Flare type determines applicable default combustion efficiency,
    regulatory requirements, and emission factor adjustments.

    ELEVATED_STEAM_ASSISTED: High-pressure tip with steam injection for
        smokeless operation. Default CE 98%.
    ELEVATED_AIR_ASSISTED: Forced-draft air for combustion enhancement.
        Default CE 98%.
    ELEVATED_UNASSISTED: Simple pipe flare with no assist medium.
        Default CE 98%.
    ENCLOSED_GROUND: Multi-burner in refractory-lined enclosure.
        Default CE 99%.
    MULTI_POINT_GROUND: Multiple staged burners at ground level (MPGF).
        Default CE 99%.
    OFFSHORE_MARINE: Boom-mounted flare for offshore platforms.
        Default CE 98%.
    CANDLESTICK: Simple vertical pipe with no wind shielding.
        Default CE 96%.
    LOW_PRESSURE: Designed for low-flow, low-pressure waste gas.
        Default CE 95%.
    """

    ELEVATED_STEAM_ASSISTED = "elevated_steam_assisted"
    ELEVATED_AIR_ASSISTED = "elevated_air_assisted"
    ELEVATED_UNASSISTED = "elevated_unassisted"
    ENCLOSED_GROUND = "enclosed_ground"
    MULTI_POINT_GROUND = "multi_point_ground"
    OFFSHORE_MARINE = "offshore_marine"
    CANDLESTICK = "candlestick"
    LOW_PRESSURE = "low_pressure"


class FlaringEventCategory(str, Enum):
    """Classification of flaring event type for regulatory reporting.

    Event category determines reporting requirements under various
    frameworks (EPA Subpart W, EU Methane Regulation, World Bank ZRF).

    ROUTINE: Normal process flaring during steady-state operations.
    NON_ROUTINE: Planned but irregular events (well testing, tank flashing).
    EMERGENCY: Pressure relief, equipment failure, process upset.
    MAINTENANCE: Startup, shutdown, turnaround activities.
    PILOT_PURGE: Continuous pilot flame and purge gas consumption.
    WELL_COMPLETION: Upstream oil and gas flowback flaring.
    """

    ROUTINE = "routine"
    NON_ROUTINE = "non_routine"
    EMERGENCY = "emergency"
    MAINTENANCE = "maintenance"
    PILOT_PURGE = "pilot_purge"
    WELL_COMPLETION = "well_completion"


class CalculationMethod(str, Enum):
    """Calculation methodology for flaring emission quantification.

    GAS_COMPOSITION: Flow x gas_fraction x molecular_weight_ratio x CE.
        Highest accuracy using actual gas analysis.
    DEFAULT_EMISSION_FACTOR: Volume x default EF (kg CO2/scf or kg CO2/Nm3).
        Uses published average emission factors.
    ENGINEERING_ESTIMATE: Process mass balance, equipment design capacity.
        Engineering judgment when measurement unavailable.
    DIRECT_MEASUREMENT: Continuous flow and composition monitoring (CEMS,
        ultrasonic). Highest reliability tier.
    """

    GAS_COMPOSITION = "gas_composition"
    DEFAULT_EMISSION_FACTOR = "default_emission_factor"
    ENGINEERING_ESTIMATE = "engineering_estimate"
    DIRECT_MEASUREMENT = "direct_measurement"


class EmissionFactorSource(str, Enum):
    """Source authority for emission factor values.

    EPA: US Environmental Protection Agency (40 CFR Part 98 Subpart W).
    IPCC: IPCC 2006 Guidelines for National GHG Inventories.
    DEFRA: UK Department for Environment, Food and Rural Affairs.
    EU_ETS: European Union Emissions Trading System factors.
    API: American Petroleum Institute Compendium.
    CUSTOM: Organization-specific or facility-measured factors.
    """

    EPA = "EPA"
    IPCC = "IPCC"
    DEFRA = "DEFRA"
    EU_ETS = "EU_ETS"
    API = "API"
    CUSTOM = "CUSTOM"


class GasComponent(str, Enum):
    """Individual gas species in flare gas composition analysis.

    Covers all 15 components tracked for flaring emission calculations
    including hydrocarbons, inerts, sour gas, and olefins.
    """

    CH4 = "CH4"
    C2H6 = "C2H6"
    C3H8 = "C3H8"
    N_C4H10 = "N_C4H10"
    I_C4H10 = "I_C4H10"
    C5H12 = "C5H12"
    C6_PLUS = "C6_PLUS"
    CO2 = "CO2"
    N2 = "N2"
    H2S = "H2S"
    H2 = "H2"
    CO = "CO"
    C2H4 = "C2H4"
    C3H6 = "C3H6"
    H2O = "H2O"


class EmissionGas(str, Enum):
    """Greenhouse gases tracked in flaring emission calculations.

    CO2: Carbon dioxide - primary combustion product.
    CH4: Methane - uncombusted methane slip (1 - combustion_efficiency).
    N2O: Nitrous oxide - minor combustion by-product.
    BLACK_CARBON: Soot/black carbon - non-GHG but tracked for SLCP reporting.
    """

    CO2 = "CO2"
    CH4 = "CH4"
    N2O = "N2O"
    BLACK_CARBON = "BLACK_CARBON"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for GWP conversion factors.

    AR4: Fourth Assessment Report (2007). GWP-100: CH4=25, N2O=298.
    AR5: Fifth Assessment Report (2014). GWP-100: CH4=28, N2O=265.
    AR6: Sixth Assessment Report (2021). GWP-100: CH4=27.3, N2O=273.
    AR6_20YR: AR6 with 20-year time horizon. GWP-20: CH4=81.2, N2O=273.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"


class StandardCondition(str, Enum):
    """Standard temperature and pressure reference conditions.

    EPA_60F: EPA standard at 60 deg F (15.56 deg C) and 14.696 psia.
        Used in US regulatory reporting.
    ISO_15C: ISO standard at 15 deg C and 101.325 kPa.
        Used in international reporting.
    """

    EPA_60F = "EPA_60F"
    ISO_15C = "ISO_15C"


class AssistType(str, Enum):
    """Type of combustion assist medium for flare operation.

    STEAM: Steam injection for smokeless operation. Optimal ratio
        0.3-0.5 lb steam/lb gas.
    AIR: Forced-draft air for combustion enhancement. Uses
        stoichiometric plus excess air factor.
    NONE: No assist medium. Simple unassisted flare operation.
    """

    STEAM = "steam"
    AIR = "air"
    NONE = "none"


class FlaringStatus(str, Enum):
    """Operational status of a flare system.

    ACTIVE: Currently in service and capable of receiving gas.
    INACTIVE: Temporarily out of service (not decommissioned).
    DECOMMISSIONED: Permanently removed from service.
    UNDER_CONSTRUCTION: Not yet commissioned for operation.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"
    UNDER_CONSTRUCTION = "under_construction"


class OGMPLevel(str, Enum):
    """OGMP 2.0 reporting level for methane emission quantification.

    LEVEL_1: Source-level activity data (flare count, type).
    LEVEL_2: Generic emission factors from studies/guidelines.
    LEVEL_3: Site-specific emission factors from sampling/engineering.
    LEVEL_4: Site-level direct measurement (continuous monitoring).
    LEVEL_5: Reconciled with operator-level measurement.
    """

    LEVEL_1 = "LEVEL_1"
    LEVEL_2 = "LEVEL_2"
    LEVEL_3 = "LEVEL_3"
    LEVEL_4 = "LEVEL_4"
    LEVEL_5 = "LEVEL_5"


class ComplianceFramework(str, Enum):
    """Regulatory framework governing flaring emission reporting.

    GHG_PROTOCOL: WRI/WBCSD Corporate GHG Protocol (Chapter 5 - Flaring).
    ISO_14064: ISO 14064-1:2018 (Category 1 direct emissions).
    CSRD_ESRS: EU Corporate Sustainability Reporting Directive, ESRS E1.
    EPA_SUBPART_W: EPA 40 CFR Part 98 Subpart W (Sec. W.23 flare stacks).
    EU_ETS_MRR: EU ETS Monitoring and Reporting Regulation.
    EU_METHANE_REG: EU Methane Regulation 2024/1787 (Art. 14 flaring).
    WORLD_BANK_ZRF: World Bank Zero Routine Flaring by 2030 initiative.
    OGMP_2_0: Oil and Gas Methane Partnership 2.0 reporting framework.
    """

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS = "csrd_esrs"
    EPA_SUBPART_W = "epa_subpart_w"
    EU_ETS_MRR = "eu_ets_mrr"
    EU_METHANE_REG = "eu_methane_reg"
    WORLD_BANK_ZRF = "world_bank_zrf"
    OGMP_2_0 = "ogmp_2_0"


class CalculationStatus(str, Enum):
    """Status of a flaring emission calculation.

    PENDING: Calculation queued but not yet started.
    IN_PROGRESS: Calculation currently executing.
    COMPLETED: Calculation finished successfully.
    FAILED: Calculation terminated with an error.
    VALIDATED: Calculation completed and independently verified.
    """

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATED = "validated"


class DataQualityTier(str, Enum):
    """Data quality tier for input data classification.

    HIGH: Continuous measurement, certified lab analysis, CEMS data.
    MEDIUM: Periodic sampling, engineering calculations, vendor data.
    LOW: Estimated values, industry averages, extrapolation.
    DEFAULT: System defaults used when no site-specific data available.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    DEFAULT = "DEFAULT"


class SeverityLevel(str, Enum):
    """Severity level for compliance findings and validation messages.

    INFO: Informational observation, no action required.
    WARNING: Potential issue that should be reviewed.
    ERROR: Compliance gap that must be addressed.
    CRITICAL: Major regulatory non-compliance requiring immediate action.
    """

    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ComplianceStatus(str, Enum):
    """Overall compliance status for a regulatory framework check.

    COMPLIANT: All applicable requirements are met.
    NON_COMPLIANT: One or more requirements are not met.
    PARTIAL: Some requirements met, others outstanding.
    NOT_APPLICABLE: Framework requirements do not apply to this context.
    """

    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIAL = "partial"
    NOT_APPLICABLE = "not_applicable"


# =============================================================================
# Data Models (16+)
# =============================================================================


class GasComposition(BaseModel):
    """Gas composition analysis for flare gas stream.

    Represents a complete molar or volumetric gas analysis with component
    fractions that must sum to approximately 1.0 (within 0.02 tolerance).
    Used for gas composition method calculations and heating value
    determination.

    Attributes:
        composition_id: Unique identifier for this gas composition analysis.
        name: Human-readable label for the composition (e.g. 'Well A gas').
        components: Mapping of GasComponent enum value to mole fraction
            (0.0 to 1.0). All fractions must sum to approximately 1.0.
        analysis_date: Date when the gas analysis was performed.
        lab_reference: Laboratory analysis reference number.
        data_quality: Quality tier of the composition data.
        notes: Optional human-readable notes about the analysis.
    """

    model_config = ConfigDict(frozen=True)

    composition_id: str = Field(
        default_factory=lambda: f"comp_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this gas composition analysis",
    )
    name: str = Field(
        default="",
        max_length=255,
        description="Human-readable label for the composition",
    )
    components: Dict[str, Decimal] = Field(
        ...,
        description="Mapping of gas component to mole fraction (0.0-1.0)",
    )
    analysis_date: Optional[datetime] = Field(
        default=None,
        description="Date when the gas analysis was performed",
    )
    lab_reference: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Laboratory analysis reference number",
    )
    data_quality: DataQualityTier = Field(
        default=DataQualityTier.MEDIUM,
        description="Quality tier of the composition data",
    )
    notes: Optional[str] = Field(
        default=None,
        description="Optional notes about the analysis",
    )

    @field_validator("components")
    @classmethod
    def validate_component_fractions(
        cls, v: Dict[str, Decimal]
    ) -> Dict[str, Decimal]:
        """Validate that all component fractions are in [0, 1] and sum to ~1.0."""
        if not v:
            raise ValueError("components must not be empty")

        total = Decimal("0")
        for component_name, fraction in v.items():
            if fraction < Decimal("0") or fraction > Decimal("1"):
                raise ValueError(
                    f"Component '{component_name}' fraction must be in "
                    f"[0.0, 1.0], got {fraction}"
                )
            total += fraction

        tolerance = Decimal("0.02")
        if abs(total - Decimal("1")) > tolerance:
            raise ValueError(
                f"Component fractions must sum to approximately 1.0 "
                f"(within {tolerance}), got {total}"
            )
        return v


class FlareSystemConfig(BaseModel):
    """Configuration and specifications for a flare system.

    Describes the physical and operational characteristics of a flare
    stack or ground flare system used for emission calculations.

    Attributes:
        flare_id: Unique identifier for the flare system.
        name: Human-readable name or asset tag.
        flare_type: Classification of the flare design.
        facility_id: Parent facility identifier.
        capacity_mscfd: Maximum design capacity in thousand standard
            cubic feet per day (Mscf/d).
        tip_diameter_inches: Flare tip internal diameter in inches.
        height_meters: Flare stack height in meters above grade.
        assist_type: Type of combustion assist medium.
        num_pilots: Number of pilot burner tips.
        status: Current operational status.
        latitude: Geographic latitude of the flare location.
        longitude: Geographic longitude of the flare location.
        installation_date: Date the flare system was installed.
        last_inspection_date: Date of the most recent inspection.
        ogmp_level: OGMP 2.0 reporting level assigned.
    """

    model_config = ConfigDict(frozen=True)

    flare_id: str = Field(
        default_factory=lambda: f"flare_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for the flare system",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name or asset tag",
    )
    flare_type: FlareType = Field(
        ...,
        description="Classification of the flare design",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Parent facility identifier",
    )
    capacity_mscfd: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Maximum design capacity in Mscf/d",
    )
    tip_diameter_inches: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Flare tip internal diameter in inches",
    )
    height_meters: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Flare stack height in meters above grade",
    )
    assist_type: AssistType = Field(
        default=AssistType.NONE,
        description="Type of combustion assist medium",
    )
    num_pilots: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of pilot burner tips",
    )
    status: FlaringStatus = Field(
        default=FlaringStatus.ACTIVE,
        description="Current operational status",
    )
    latitude: Optional[Decimal] = Field(
        default=None,
        ge=-90,
        le=90,
        description="Geographic latitude of the flare location",
    )
    longitude: Optional[Decimal] = Field(
        default=None,
        ge=-180,
        le=180,
        description="Geographic longitude of the flare location",
    )
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Date the flare system was installed",
    )
    last_inspection_date: Optional[datetime] = Field(
        default=None,
        description="Date of the most recent inspection",
    )
    ogmp_level: OGMPLevel = Field(
        default=OGMPLevel.LEVEL_2,
        description="OGMP 2.0 reporting level assigned",
    )


class FlaringEventRecord(BaseModel):
    """Individual flaring event record for tracking and reporting.

    Captures a discrete flaring event including its category, timing,
    volume, composition, and operational context.

    Attributes:
        event_id: Unique identifier for this flaring event.
        flare_id: Identifier of the flare system used.
        category: Classification of the flaring event.
        start_time: UTC start time of the flaring event.
        end_time: UTC end time of the flaring event.
        duration_hours: Duration of the event in hours.
        volume_scf: Volume of gas flared in standard cubic feet.
        volume_nm3: Volume of gas flared in normal cubic meters.
        composition_id: Reference to the gas composition analysis used.
        flow_rate_scfh: Average flow rate in standard cubic feet per hour.
        cause: Description of the cause or reason for flaring.
        reported_by: Person or system that reported the event.
        data_quality: Quality tier of the event data.
        tenant_id: Tenant identifier for multi-tenancy isolation.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str = Field(
        default_factory=lambda: f"evt_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this flaring event",
    )
    flare_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the flare system used",
    )
    category: FlaringEventCategory = Field(
        ...,
        description="Classification of the flaring event",
    )
    start_time: datetime = Field(
        ...,
        description="UTC start time of the flaring event",
    )
    end_time: Optional[datetime] = Field(
        default=None,
        description="UTC end time of the flaring event",
    )
    duration_hours: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Duration of the event in hours",
    )
    volume_scf: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume of gas flared in standard cubic feet",
    )
    volume_nm3: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume of gas flared in normal cubic meters",
    )
    composition_id: Optional[str] = Field(
        default=None,
        description="Reference to the gas composition analysis used",
    )
    flow_rate_scfh: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Average flow rate in standard cubic feet per hour",
    )
    cause: Optional[str] = Field(
        default=None,
        max_length=1000,
        description="Description of the cause or reason for flaring",
    )
    reported_by: Optional[str] = Field(
        default=None,
        max_length=255,
        description="Person or system that reported the event",
    )
    data_quality: DataQualityTier = Field(
        default=DataQualityTier.MEDIUM,
        description="Quality tier of the event data",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation",
    )

    @field_validator("end_time")
    @classmethod
    def end_time_after_start(
        cls, v: Optional[datetime], info: Any
    ) -> Optional[datetime]:
        """Validate that end_time is after start_time when both set."""
        if v is not None and info.data.get("start_time") is not None:
            if v <= info.data["start_time"]:
                raise ValueError("end_time must be after start_time")
        return v


class PilotPurgeConfig(BaseModel):
    """Configuration for pilot and purge gas accounting.

    Tracks continuous pilot flame and purge gas consumption, which
    contribute to baseline flaring emissions even when no process
    gas is being flared.

    Attributes:
        pilot_flow_rate_mmbtu_hr: Pilot gas consumption rate per pilot
            tip in MMBTU per hour. Typical range 0.5-5.0.
        purge_flow_rate_scfh: Purge gas flow rate in standard cubic
            feet per hour.
        pilot_gas_type: Gas type used for pilot flame (natural_gas
            or composition_id reference).
        purge_gas_type: Gas type used for purge (nitrogen or natural_gas).
        num_pilots: Number of active pilot tips.
        pilot_composition_id: Optional gas composition ID for pilot gas.
        purge_composition_id: Optional gas composition ID for purge gas.
        is_purge_inert: Whether purge gas is inert (N2) with zero emissions.
    """

    model_config = ConfigDict(frozen=True)

    pilot_flow_rate_mmbtu_hr: Decimal = Field(
        default=Decimal("1.0"),
        ge=0,
        description="Pilot gas consumption rate per tip in MMBTU/hr",
    )
    purge_flow_rate_scfh: Decimal = Field(
        default=Decimal("100"),
        ge=0,
        description="Purge gas flow rate in standard cubic feet per hour",
    )
    pilot_gas_type: str = Field(
        default="natural_gas",
        description="Gas type used for pilot flame",
    )
    purge_gas_type: str = Field(
        default="nitrogen",
        description="Gas type used for purge (nitrogen or natural_gas)",
    )
    num_pilots: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of active pilot tips",
    )
    pilot_composition_id: Optional[str] = Field(
        default=None,
        description="Optional gas composition ID for pilot gas",
    )
    purge_composition_id: Optional[str] = Field(
        default=None,
        description="Optional gas composition ID for purge gas",
    )
    is_purge_inert: bool = Field(
        default=True,
        description="Whether purge gas is inert (N2) with zero emissions",
    )


class CombustionEfficiencyParams(BaseModel):
    """Parameters affecting combustion efficiency of a flare.

    Models the factors that influence the actual combustion efficiency
    of a flare system, including wind conditions, tip velocity, heating
    value, and assist medium ratios.

    Attributes:
        base_ce: Base combustion efficiency fraction (0.0-1.0).
        wind_speed_ms: Wind speed at flare tip in meters per second.
        tip_velocity_ms: Gas exit velocity at flare tip in meters per second.
        lhv_btu_scf: Lower heating value of flare gas in BTU per scf.
        steam_ratio: Steam-to-gas mass ratio (lb steam / lb gas).
            Optimal range 0.3-0.5.
        air_ratio: Air-to-gas volume ratio above stoichiometric.
        flare_type: Type of flare system for default CE selection.
    """

    model_config = ConfigDict(frozen=True)

    base_ce: Decimal = Field(
        default=Decimal("0.98"),
        ge=0,
        le=1,
        description="Base combustion efficiency fraction (0.0-1.0)",
    )
    wind_speed_ms: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Wind speed at flare tip in meters per second",
    )
    tip_velocity_ms: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Gas exit velocity at flare tip in meters per second",
    )
    lhv_btu_scf: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Lower heating value in BTU per scf",
    )
    steam_ratio: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Steam-to-gas mass ratio (lb steam / lb gas)",
    )
    air_ratio: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Air-to-gas volume ratio above stoichiometric",
    )
    flare_type: Optional[FlareType] = Field(
        default=None,
        description="Type of flare system for default CE selection",
    )


class CalculationInput(BaseModel):
    """Input data for a single flaring emission calculation.

    Contains all parameters needed to compute GHG emissions from a
    flaring event using any of the four supported calculation methods.

    Attributes:
        flare_id: Identifier of the flare system.
        method: Calculation methodology to apply.
        event_category: Classification of the flaring event.
        gas_volume_scf: Volume of gas flared in standard cubic feet.
        gas_volume_nm3: Volume of gas flared in normal cubic meters.
        gas_composition: Optional gas composition for composition method.
        composition_id: Reference to stored gas composition analysis.
        duration_hours: Duration of flaring in hours.
        standard_condition: Reference temperature and pressure standard.
        gwp_source: IPCC Assessment Report for GWP values.
        combustion_efficiency: Optional CE override (0.0-1.0).
        ce_params: Optional combustion efficiency modeling parameters.
        pilot_purge: Optional pilot and purge gas configuration.
        emission_factor_source: Source authority for default EFs.
        custom_emission_factor_co2: Custom CO2 emission factor override.
        custom_emission_factor_ch4: Custom CH4 emission factor override.
        flare_type: Flare type for default CE and factor selection.
        facility_id: Facility identifier for aggregation.
        tenant_id: Tenant identifier for multi-tenancy isolation.
        data_quality: Quality tier of the input data.
    """

    model_config = ConfigDict(frozen=True)

    flare_id: str = Field(
        ...,
        min_length=1,
        description="Identifier of the flare system",
    )
    method: CalculationMethod = Field(
        default=CalculationMethod.DEFAULT_EMISSION_FACTOR,
        description="Calculation methodology to apply",
    )
    event_category: FlaringEventCategory = Field(
        default=FlaringEventCategory.ROUTINE,
        description="Classification of the flaring event",
    )
    gas_volume_scf: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume of gas flared in standard cubic feet",
    )
    gas_volume_nm3: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume of gas flared in normal cubic meters",
    )
    gas_composition: Optional[GasComposition] = Field(
        default=None,
        description="Gas composition for composition method",
    )
    composition_id: Optional[str] = Field(
        default=None,
        description="Reference to stored gas composition analysis",
    )
    duration_hours: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Duration of flaring in hours",
    )
    standard_condition: StandardCondition = Field(
        default=StandardCondition.EPA_60F,
        description="Reference temperature and pressure standard",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    combustion_efficiency: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        description="Optional CE override (0.0-1.0)",
    )
    ce_params: Optional[CombustionEfficiencyParams] = Field(
        default=None,
        description="Optional combustion efficiency modeling parameters",
    )
    pilot_purge: Optional[PilotPurgeConfig] = Field(
        default=None,
        description="Optional pilot and purge gas configuration",
    )
    emission_factor_source: EmissionFactorSource = Field(
        default=EmissionFactorSource.EPA,
        description="Source authority for default emission factors",
    )
    custom_emission_factor_co2: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Custom CO2 emission factor override (kg CO2/scf)",
    )
    custom_emission_factor_ch4: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Custom CH4 emission factor override (kg CH4/scf)",
    )
    flare_type: Optional[FlareType] = Field(
        default=None,
        description="Flare type for default CE and factor selection",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Facility identifier for aggregation",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation",
    )
    data_quality: DataQualityTier = Field(
        default=DataQualityTier.MEDIUM,
        description="Quality tier of the input data",
    )

    @model_validator(mode="after")
    def validate_volume_present(self) -> CalculationInput:
        """Validate that at least one volume measurement is provided."""
        if self.gas_volume_scf is None and self.gas_volume_nm3 is None:
            raise ValueError(
                "At least one of gas_volume_scf or gas_volume_nm3 "
                "must be provided"
            )
        return self


class EmissionDetail(BaseModel):
    """Detailed emission result for a single greenhouse gas.

    Captures the calculated emissions for one gas species including
    the factor and methodology used for full traceability.

    Attributes:
        gas: Greenhouse gas species (CO2, CH4, N2O, BLACK_CARBON).
        kg: Emissions in kilograms of the specific gas.
        co2e_kg: Emissions in kilograms of CO2-equivalent.
        emission_factor_used: Numeric emission factor value applied.
        emission_factor_unit: Unit of the emission factor.
        source: Source authority for the emission factor.
        gwp_applied: GWP multiplier applied for CO2e conversion.
        methodology_note: Brief explanation of calculation approach.
    """

    model_config = ConfigDict(frozen=True)

    gas: EmissionGas = Field(
        ...,
        description="Greenhouse gas species",
    )
    kg: Decimal = Field(
        ...,
        ge=0,
        description="Emissions in kilograms of the specific gas",
    )
    co2e_kg: Decimal = Field(
        ...,
        ge=0,
        description="Emissions in kilograms of CO2-equivalent",
    )
    emission_factor_used: Optional[Decimal] = Field(
        default=None,
        description="Numeric emission factor value applied",
    )
    emission_factor_unit: Optional[str] = Field(
        default=None,
        description="Unit of the emission factor",
    )
    source: Optional[str] = Field(
        default=None,
        description="Source authority for the emission factor",
    )
    gwp_applied: Decimal = Field(
        default=Decimal("1"),
        gt=0,
        description="GWP multiplier applied for CO2e conversion",
    )
    methodology_note: Optional[str] = Field(
        default=None,
        description="Brief explanation of calculation approach",
    )


class CalculationResult(BaseModel):
    """Complete result of a single flaring emission calculation.

    Contains all calculated emissions by gas, total CO2e, the
    methodology parameters used, combustion efficiency applied, and
    a SHA-256 provenance hash for audit trail integrity.

    Attributes:
        calculation_id: Unique identifier for this calculation.
        flare_id: Flare system used for this calculation.
        co2_kg: Total CO2 emissions in kilograms.
        ch4_kg: Total CH4 emissions in kilograms (uncombusted slip).
        n2o_kg: Total N2O emissions in kilograms.
        co2e_kg: Total CO2-equivalent emissions in kilograms.
        co2e_tonnes: Total CO2-equivalent in metric tonnes.
        combustion_efficiency_used: Actual CE fraction applied.
        method: Calculation method used.
        event_category: Flaring event category.
        flare_type: Flare type used for calculations.
        gwp_source: GWP source applied.
        emission_details: Per-gas emission breakdown.
        heating_value_btu_scf: Calculated HHV of flare gas (BTU/scf).
        gas_volume_scf: Volume of gas flared (scf).
        pilot_emissions_co2e_kg: Pilot gas emissions (kg CO2e).
        purge_emissions_co2e_kg: Purge gas emissions (kg CO2e).
        provenance_hash: SHA-256 hash for audit trail integrity.
        calculation_trace: Ordered human-readable calculation steps.
        timestamp: UTC timestamp when calculation was performed.
        facility_id: Facility identifier.
        status: Calculation completion status.
        data_quality: Quality tier of input data.
        processing_time_ms: Calculation wall-clock time in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    calculation_id: str = Field(
        default_factory=lambda: f"calc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this calculation",
    )
    flare_id: str = Field(
        ...,
        description="Flare system used for this calculation",
    )
    co2_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total CO2 emissions in kilograms",
    )
    ch4_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total CH4 emissions in kilograms (uncombusted slip)",
    )
    n2o_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total N2O emissions in kilograms",
    )
    co2e_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total CO2-equivalent emissions in kilograms",
    )
    co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total CO2-equivalent in metric tonnes",
    )
    combustion_efficiency_used: Decimal = Field(
        default=Decimal("0.98"),
        ge=0,
        le=1,
        description="Actual CE fraction applied",
    )
    method: CalculationMethod = Field(
        ...,
        description="Calculation method used",
    )
    event_category: FlaringEventCategory = Field(
        default=FlaringEventCategory.ROUTINE,
        description="Flaring event category",
    )
    flare_type: Optional[FlareType] = Field(
        default=None,
        description="Flare type used for calculations",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source applied",
    )
    emission_details: List[EmissionDetail] = Field(
        default_factory=list,
        max_length=MAX_GASES_PER_RESULT,
        description="Per-gas emission breakdown",
    )
    heating_value_btu_scf: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Calculated HHV of flare gas (BTU/scf)",
    )
    gas_volume_scf: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Volume of gas flared (scf)",
    )
    pilot_emissions_co2e_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Pilot gas emissions (kg CO2e)",
    )
    purge_emissions_co2e_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Purge gas emissions (kg CO2e)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        max_length=MAX_TRACE_STEPS,
        description="Ordered human-readable calculation steps",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when calculation was performed",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Facility identifier",
    )
    status: CalculationStatus = Field(
        default=CalculationStatus.COMPLETED,
        description="Calculation completion status",
    )
    data_quality: DataQualityTier = Field(
        default=DataQualityTier.MEDIUM,
        description="Quality tier of input data",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Calculation wall-clock time in milliseconds",
    )


class BatchCalculationRequest(BaseModel):
    """Request model for batch flaring emission calculations.

    Groups multiple calculation inputs for processing as a single
    batch, sharing common parameters.

    Attributes:
        calculations: List of individual calculation inputs.
        gwp_source: IPCC Assessment Report for GWP values.
        standard_condition: Reference standard for all calculations.
        tenant_id: Tenant identifier for multi-tenancy isolation.
        batch_id: Optional batch identifier for tracking.
    """

    model_config = ConfigDict(frozen=True)

    calculations: List[CalculationInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual calculation inputs",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    standard_condition: StandardCondition = Field(
        default=StandardCondition.EPA_60F,
        description="Reference standard for all calculations",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation",
    )
    batch_id: Optional[str] = Field(
        default=None,
        description="Optional batch identifier for tracking",
    )


class BatchCalculationResponse(BaseModel):
    """Response model for a batch flaring emission calculation.

    Aggregates individual calculation results with batch-level totals
    and processing metadata.

    Attributes:
        success: Whether all calculations in the batch succeeded.
        batch_id: Batch identifier for tracking.
        results: List of individual calculation results.
        total_co2e_kg: Batch total CO2-equivalent in kilograms.
        total_co2e_tonnes: Batch total CO2-equivalent in metric tonnes.
        total_co2_kg: Batch total CO2 in kilograms.
        total_ch4_kg: Batch total CH4 in kilograms.
        total_n2o_kg: Batch total N2O in kilograms.
        emissions_by_flare_type: Emissions aggregated by flare type.
        emissions_by_event_category: Emissions aggregated by event category.
        calculation_count: Number of successful calculations.
        failed_count: Number of failed calculations.
        processing_time_ms: Total batch processing time in milliseconds.
        provenance_hash: SHA-256 hash covering the entire batch result.
        gwp_source: GWP source used for this batch.
    """

    model_config = ConfigDict(frozen=True)

    success: bool = Field(
        ...,
        description="Whether all calculations succeeded",
    )
    batch_id: str = Field(
        default_factory=lambda: f"batch_{uuid.uuid4().hex[:12]}",
        description="Batch identifier for tracking",
    )
    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    total_co2e_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Batch total CO2-equivalent in kilograms",
    )
    total_co2e_tonnes: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Batch total CO2-equivalent in metric tonnes",
    )
    total_co2_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Batch total CO2 in kilograms",
    )
    total_ch4_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Batch total CH4 in kilograms",
    )
    total_n2o_kg: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Batch total N2O in kilograms",
    )
    emissions_by_flare_type: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions aggregated by flare type (tCO2e)",
    )
    emissions_by_event_category: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Emissions aggregated by event category (tCO2e)",
    )
    calculation_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful calculations",
    )
    failed_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed calculations",
    )
    processing_time_ms: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Total batch processing time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash covering the entire batch result",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="GWP source used for this batch",
    )


class UncertaintyInput(BaseModel):
    """Input parameters for Monte Carlo uncertainty analysis.

    Specifies the calculation input and uncertainty parameters for
    a Monte Carlo simulation to quantify emission estimate uncertainty.

    Attributes:
        calculation_input: The base calculation input to analyze.
        iterations: Number of Monte Carlo iterations.
        confidence_levels: Confidence levels for interval computation.
        volume_uncertainty_pct: Relative uncertainty in gas volume (%).
        composition_uncertainty_pct: Relative uncertainty in composition (%).
        ce_uncertainty_pct: Relative uncertainty in combustion efficiency (%).
        seed: Random seed for reproducibility.
    """

    model_config = ConfigDict(frozen=True)

    calculation_input: CalculationInput = Field(
        ...,
        description="Base calculation input to analyze",
    )
    iterations: int = Field(
        default=5000,
        gt=0,
        le=1_000_000,
        description="Number of Monte Carlo iterations",
    )
    confidence_levels: List[float] = Field(
        default_factory=lambda: [90.0, 95.0, 99.0],
        description="Confidence levels for interval computation",
    )
    volume_uncertainty_pct: Decimal = Field(
        default=Decimal("5.0"),
        ge=0,
        le=100,
        description="Relative uncertainty in gas volume (%)",
    )
    composition_uncertainty_pct: Decimal = Field(
        default=Decimal("2.0"),
        ge=0,
        le=100,
        description="Relative uncertainty in composition (%)",
    )
    ce_uncertainty_pct: Decimal = Field(
        default=Decimal("2.0"),
        ge=0,
        le=100,
        description="Relative uncertainty in combustion efficiency (%)",
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility",
    )

    @field_validator("confidence_levels")
    @classmethod
    def validate_confidence_levels(
        cls, v: List[float]
    ) -> List[float]:
        """Validate all confidence levels are in (0, 100)."""
        for level in v:
            if not (0.0 < level < 100.0):
                raise ValueError(
                    f"Each confidence level must be in (0, 100), got {level}"
                )
        return v


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty quantification result.

    Provides statistical characterization of emission estimate
    uncertainty including mean, standard deviation, confidence
    intervals, and sensitivity analysis.

    Attributes:
        mean_co2e_kg: Mean CO2-equivalent emission estimate (kg).
        std_dev_kg: Standard deviation of the estimate (kg).
        coefficient_of_variation: CV = std_dev / mean (dimensionless).
        confidence_intervals: Confidence intervals keyed by level string.
        iterations: Number of Monte Carlo iterations performed.
        data_quality_score: Overall data quality indicator (1-5 scale).
        method: Calculation method used for uncertainty analysis.
        contributions: Parameter contribution to total variance.
        provenance_hash: SHA-256 hash for this uncertainty result.
    """

    model_config = ConfigDict(frozen=True)

    mean_co2e_kg: Decimal = Field(
        ...,
        ge=0,
        description="Mean CO2-equivalent emission estimate (kg)",
    )
    std_dev_kg: Decimal = Field(
        ...,
        ge=0,
        description="Standard deviation of the estimate (kg)",
    )
    coefficient_of_variation: Decimal = Field(
        ...,
        ge=0,
        description="CV = std_dev / mean (dimensionless)",
    )
    confidence_intervals: Dict[str, Tuple[Decimal, Decimal]] = Field(
        default_factory=dict,
        description="Confidence intervals keyed by level (e.g. '95' -> (lower, upper))",
    )
    iterations: int = Field(
        ...,
        gt=0,
        description="Number of Monte Carlo iterations performed",
    )
    data_quality_score: Optional[Decimal] = Field(
        default=None,
        ge=1,
        le=5,
        description="Data quality indicator (1-5 scale)",
    )
    method: CalculationMethod = Field(
        ...,
        description="Calculation method used for uncertainty analysis",
    )
    contributions: Dict[str, Decimal] = Field(
        default_factory=dict,
        description="Parameter contribution to total variance (name -> fraction)",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for this uncertainty result",
    )


class ComplianceCheckInput(BaseModel):
    """Input for a regulatory compliance check against a framework.

    Specifies the calculation result and framework to validate
    compliance against applicable regulatory requirements.

    Attributes:
        calculation_result: The calculation result to check.
        framework: Regulatory framework to check against.
        flare_system: Optional flare system config for context.
        event_record: Optional flaring event record for context.
        ogmp_level: OGMP 2.0 level to validate against.
        reporting_year: Calendar year for the compliance check.
        tenant_id: Tenant identifier for multi-tenancy isolation.
    """

    model_config = ConfigDict(frozen=True)

    calculation_result: CalculationResult = Field(
        ...,
        description="Calculation result to check",
    )
    framework: ComplianceFramework = Field(
        ...,
        description="Regulatory framework to check against",
    )
    flare_system: Optional[FlareSystemConfig] = Field(
        default=None,
        description="Optional flare system config for context",
    )
    event_record: Optional[FlaringEventRecord] = Field(
        default=None,
        description="Optional flaring event record for context",
    )
    ogmp_level: Optional[OGMPLevel] = Field(
        default=None,
        description="OGMP 2.0 level to validate against",
    )
    reporting_year: Optional[int] = Field(
        default=None,
        ge=2000,
        le=2100,
        description="Calendar year for the compliance check",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier for multi-tenancy isolation",
    )


class ComplianceCheckResult(BaseModel):
    """Result of a regulatory compliance check.

    Contains the overall compliance status, individual findings, and
    recommendations for the checked framework.

    Attributes:
        check_id: Unique identifier for this compliance check.
        framework: Regulatory framework checked.
        status: Overall compliance status.
        findings: List of individual compliance findings.
        recommendations: Actionable recommendations for gaps.
        requirements_checked: Total number of requirements evaluated.
        requirements_met: Number of requirements that passed.
        requirements_failed: Number of requirements that failed.
        provenance_hash: SHA-256 hash for this compliance result.
        timestamp: UTC timestamp when the check was performed.
    """

    model_config = ConfigDict(frozen=True)

    check_id: str = Field(
        default_factory=lambda: f"chk_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this compliance check",
    )
    framework: ComplianceFramework = Field(
        ...,
        description="Regulatory framework checked",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Overall compliance status",
    )
    findings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of individual compliance findings",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Actionable recommendations for gaps",
    )
    requirements_checked: int = Field(
        default=0,
        ge=0,
        description="Total number of requirements evaluated",
    )
    requirements_met: int = Field(
        default=0,
        ge=0,
        description="Number of requirements that passed",
    )
    requirements_failed: int = Field(
        default=0,
        ge=0,
        description="Number of requirements that failed",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for this compliance result",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the check was performed",
    )


class FlareSystemRegistration(BaseModel):
    """Request model for registering a new flare system.

    Contains all required and optional parameters for adding a flare
    system to the registry.

    Attributes:
        name: Human-readable name or asset tag.
        flare_type: Classification of the flare design.
        facility_id: Parent facility identifier.
        capacity_mscfd: Maximum design capacity in Mscf/d.
        tip_diameter_inches: Flare tip internal diameter in inches.
        height_meters: Flare stack height in meters above grade.
        assist_type: Type of combustion assist medium.
        num_pilots: Number of pilot burner tips.
        latitude: Geographic latitude.
        longitude: Geographic longitude.
        installation_date: Date the flare was installed.
        ogmp_level: OGMP 2.0 reporting level.
        pilot_purge_config: Pilot and purge gas configuration.
        tenant_id: Tenant identifier.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Human-readable name or asset tag",
    )
    flare_type: FlareType = Field(
        ...,
        description="Classification of the flare design",
    )
    facility_id: Optional[str] = Field(
        default=None,
        description="Parent facility identifier",
    )
    capacity_mscfd: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Maximum design capacity in Mscf/d",
    )
    tip_diameter_inches: Optional[Decimal] = Field(
        default=None,
        gt=0,
        description="Flare tip internal diameter in inches",
    )
    height_meters: Optional[Decimal] = Field(
        default=None,
        ge=0,
        description="Flare stack height in meters above grade",
    )
    assist_type: AssistType = Field(
        default=AssistType.NONE,
        description="Type of combustion assist medium",
    )
    num_pilots: int = Field(
        default=1,
        ge=0,
        le=10,
        description="Number of pilot burner tips",
    )
    latitude: Optional[Decimal] = Field(
        default=None,
        ge=-90,
        le=90,
        description="Geographic latitude",
    )
    longitude: Optional[Decimal] = Field(
        default=None,
        ge=-180,
        le=180,
        description="Geographic longitude",
    )
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Date the flare was installed",
    )
    ogmp_level: OGMPLevel = Field(
        default=OGMPLevel.LEVEL_2,
        description="OGMP 2.0 reporting level",
    )
    pilot_purge_config: Optional[PilotPurgeConfig] = Field(
        default=None,
        description="Pilot and purge gas configuration",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Tenant identifier",
    )


class FlaringStats(BaseModel):
    """Service-level statistics for the flaring agent.

    Provides aggregate operational metrics for monitoring and
    administration.

    Attributes:
        total_calculations: Total number of calculations performed.
        total_events_logged: Total flaring events recorded.
        total_flare_systems: Total registered flare systems.
        total_compositions: Total gas composition analyses.
        total_co2e_kg_calculated: Cumulative CO2e calculated (kg).
        routine_flaring_pct: Percentage of events that are routine.
        avg_combustion_efficiency: Average CE across all calculations.
        active_flare_systems: Number of active flare systems.
        calculation_methods_used: Count of calculations by method.
        uptime_seconds: Service uptime in seconds.
    """

    model_config = ConfigDict(frozen=True)

    total_calculations: int = Field(
        default=0,
        ge=0,
        description="Total calculations performed",
    )
    total_events_logged: int = Field(
        default=0,
        ge=0,
        description="Total flaring events recorded",
    )
    total_flare_systems: int = Field(
        default=0,
        ge=0,
        description="Total registered flare systems",
    )
    total_compositions: int = Field(
        default=0,
        ge=0,
        description="Total gas composition analyses",
    )
    total_co2e_kg_calculated: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Cumulative CO2e calculated (kg)",
    )
    routine_flaring_pct: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=100,
        description="Percentage of events that are routine",
    )
    avg_combustion_efficiency: Optional[Decimal] = Field(
        default=None,
        ge=0,
        le=1,
        description="Average CE across all calculations",
    )
    active_flare_systems: int = Field(
        default=0,
        ge=0,
        description="Number of active flare systems",
    )
    calculation_methods_used: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of calculations by method",
    )
    uptime_seconds: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Service uptime in seconds",
    )


class HealthResponse(BaseModel):
    """Health check response for the flaring agent service.

    Attributes:
        status: Overall service health status (healthy, degraded, unhealthy).
        version: Service version string.
        agent_id: Agent identifier.
        agent_name: Agent human-readable name.
        database_connected: Whether database is reachable.
        cache_connected: Whether cache is reachable.
        flare_systems_count: Number of registered flare systems.
        uptime_seconds: Service uptime in seconds.
        timestamp: UTC timestamp of the health check.
    """

    model_config = ConfigDict(frozen=True)

    status: str = Field(
        default="healthy",
        description="Overall service health (healthy, degraded, unhealthy)",
    )
    version: str = Field(
        default=VERSION,
        description="Service version string",
    )
    agent_id: str = Field(
        default="GL-MRV-SCOPE1-006",
        description="Agent identifier",
    )
    agent_name: str = Field(
        default="Flaring Agent",
        description="Agent human-readable name",
    )
    database_connected: bool = Field(
        default=False,
        description="Whether database is reachable",
    )
    cache_connected: bool = Field(
        default=False,
        description="Whether cache is reachable",
    )
    flare_systems_count: int = Field(
        default=0,
        ge=0,
        description="Number of registered flare systems",
    )
    uptime_seconds: Decimal = Field(
        default=Decimal("0"),
        ge=0,
        description="Service uptime in seconds",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp of the health check",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        """Normalize and validate health status."""
        normalised = v.strip().lower()
        valid_statuses = {"healthy", "degraded", "unhealthy"}
        if normalised not in valid_statuses:
            raise ValueError(
                f"status must be one of {sorted(valid_statuses)}, got '{v}'"
            )
        return normalised


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_COMPOSITION_COMPONENTS",
    "CO2_MOLECULAR_WEIGHT",
    "CARBON_ATOMIC_WEIGHT",
    "GWP_VALUES",
    "COMPONENT_HHV_BTU_SCF",
    "COMPONENT_MOLECULAR_WEIGHTS",
    "COMPONENT_CARBON_COUNT",
    "DEFAULT_COMBUSTION_EFFICIENCY",
    # Enums (16)
    "FlareType",
    "FlaringEventCategory",
    "CalculationMethod",
    "EmissionFactorSource",
    "GasComponent",
    "EmissionGas",
    "GWPSource",
    "StandardCondition",
    "AssistType",
    "FlaringStatus",
    "OGMPLevel",
    "ComplianceFramework",
    "CalculationStatus",
    "DataQualityTier",
    "SeverityLevel",
    "ComplianceStatus",
    # Data models (16+)
    "GasComposition",
    "FlareSystemConfig",
    "FlaringEventRecord",
    "PilotPurgeConfig",
    "CombustionEfficiencyParams",
    "CalculationInput",
    "CalculationResult",
    "EmissionDetail",
    "BatchCalculationRequest",
    "BatchCalculationResponse",
    "UncertaintyInput",
    "UncertaintyResult",
    "ComplianceCheckInput",
    "ComplianceCheckResult",
    "FlareSystemRegistration",
    "FlaringStats",
    "HealthResponse",
]
