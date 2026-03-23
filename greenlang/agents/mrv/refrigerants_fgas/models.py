# -*- coding: utf-8 -*-
"""
Refrigerants & F-Gas Agent Data Models - AGENT-MRV-002

Pydantic v2 data models for the Refrigerants & F-Gas Agent SDK covering
GHG Protocol Scope 1 refrigerant and fluorinated gas calculations including:
- HFC, HFO, PFC, SF6, NF3, HCFC, CFC, and natural refrigerants
- Equipment-based and mass-balance calculation methodologies
- Blend decomposition into constituent gases with weight-fraction GWP
- Equipment lifecycle tracking (installation, operating, end-of-life)
- Service event recording (recharge, repair, recovery, decommissioning)
- Leak rate estimation by equipment type and lifecycle stage
- Monte Carlo uncertainty quantification
- Multi-framework regulatory compliance (GHG Protocol, ISO 14064,
  CSRD, EPA, EU F-Gas, Kigali Amendment, UK F-Gas)
- HFC phase-down schedule tracking
- SHA-256 provenance chain for complete audit trails

Enumerations (15):
    - RefrigerantCategory, RefrigerantType, GWPSource, GWPTimeframe,
      CalculationMethod, EquipmentType, EquipmentStatus, ServiceEventType,
      LifecycleStage, CalculationStatus, ReportingPeriod,
      RegulatoryFramework, ComplianceStatus, PhaseDownSchedule, UnitType

Data Models (14):
    - GWPValue, BlendComponent, RefrigerantProperties, EquipmentProfile,
      ServiceEvent, LeakRateProfile, CalculationInput, MassBalanceData,
      GasEmission, CalculationResult, BatchCalculationRequest,
      BatchCalculationResponse, UncertaintyResult, ComplianceRecord

Author: GreenLang Platform Team
Date: February 2026
PRD: AGENT-MRV-002 Refrigerants & F-Gas (GL-MRV-SCOPE1-002)
Status: Production Ready
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


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

#: Maximum number of gas emission entries per calculation result.
MAX_GASES_PER_RESULT: int = 20

#: Maximum number of trace steps in a single calculation.
MAX_TRACE_STEPS: int = 200

#: Maximum number of blend components in a single refrigerant definition.
MAX_BLEND_COMPONENTS: int = 10

#: Maximum number of equipment profiles in a single calculation input.
MAX_EQUIPMENT_PROFILES: int = 1_000

#: Maximum number of service events in a single calculation input.
MAX_SERVICE_EVENTS: int = 10_000

#: Standard GWP-100yr values by Assessment Report for common F-gases.
#: Comprehensive values are maintained in the RefrigerantDatabaseEngine.
GWP_REFERENCE: Dict[str, Dict[str, float]] = {
    "AR4": {"CO2": 1.0, "CH4": 25.0, "N2O": 298.0, "SF6": 22800.0, "NF3": 17200.0},
    "AR5": {"CO2": 1.0, "CH4": 28.0, "N2O": 265.0, "SF6": 23500.0, "NF3": 16100.0},
    "AR6": {"CO2": 1.0, "CH4": 27.3, "N2O": 273.0, "SF6": 25200.0, "NF3": 17400.0},
}


# =============================================================================
# Enumerations (15)
# =============================================================================


class RefrigerantCategory(str, Enum):
    """Broad classification of refrigerant and fluorinated gas types.

    Used to group refrigerant types for reporting aggregation, regulatory
    compliance mapping, and to determine applicable phase-down schedules.

    HFC: Hydrofluorocarbons - primary regulated F-gases under Kigali.
    HFC_BLEND: Multi-component HFC mixtures (R-404A, R-410A, etc.).
    HFO: Hydrofluoroolefins - low-GWP HFC alternatives.
    PFC: Perfluorocarbons - high-GWP industrial gases.
    SF6: Sulphur hexafluoride - electrical switchgear insulation gas.
    NF3: Nitrogen trifluoride - semiconductor manufacturing.
    HCFC: Hydrochlorofluorocarbons - Montreal Protocol controlled.
    CFC: Chlorofluorocarbons - Montreal Protocol banned substances.
    NATURAL: Natural refrigerants (ammonia, CO2, propane, isobutane).
    OTHER: Other fluorinated gases not classified elsewhere.
    """

    HFC = "hfc"
    HFC_BLEND = "hfc_blend"
    HFO = "hfo"
    PFC = "pfc"
    SF6 = "sf6"
    NF3 = "nf3"
    HCFC = "hcfc"
    CFC = "cfc"
    NATURAL = "natural"
    OTHER = "other"


class RefrigerantType(str, Enum):
    """Specific refrigerant type identifiers for fluorinated gas tracking.

    Covers all major HFCs, HFC blends, HFOs, PFCs, SF6, NF3, HCFCs,
    CFCs, and natural refrigerants encountered in Scope 1 refrigerant
    emission inventories. Each type has associated GWP values by IPCC
    Assessment Report edition and physical properties.

    Naming follows ASHRAE Standard 34 refrigerant designation system
    with underscores replacing hyphens for Python identifier compatibility.
    """

    # -- HFCs (pure substances) ---------------------------------------------
    R_32 = "R_32"
    R_125 = "R_125"
    R_134A = "R_134A"
    R_143A = "R_143A"
    R_152A = "R_152A"
    R_227EA = "R_227EA"
    R_236FA = "R_236FA"
    R_245FA = "R_245FA"
    R_365MFC = "R_365MFC"
    R_23 = "R_23"
    R_41 = "R_41"

    # -- HFC Blends ---------------------------------------------------------
    R_404A = "R_404A"
    R_407A = "R_407A"
    R_407C = "R_407C"
    R_407F = "R_407F"
    R_410A = "R_410A"
    R_413A = "R_413A"
    R_417A = "R_417A"
    R_422D = "R_422D"
    R_427A = "R_427A"
    R_438A = "R_438A"
    R_448A = "R_448A"
    R_449A = "R_449A"
    R_452A = "R_452A"
    R_454B = "R_454B"
    R_507A = "R_507A"
    R_508B = "R_508B"

    # -- HFOs (Hydrofluoroolefins) ------------------------------------------
    R_1234YF = "R_1234YF"
    R_1234ZE = "R_1234ZE"
    R_1233ZD = "R_1233ZD"
    R_1336MZZ = "R_1336MZZ"

    # -- PFCs (Perfluorocarbons) --------------------------------------------
    CF4 = "CF4"
    C2F6 = "C2F6"
    C3F8 = "C3F8"
    C_C4F8 = "C_C4F8"
    C4F10 = "C4F10"
    C5F12 = "C5F12"
    C6F14 = "C6F14"

    # -- SF6 and NF3 --------------------------------------------------------
    SF6_GAS = "SF6_GAS"
    NF3_GAS = "NF3_GAS"
    SO2F2 = "SO2F2"

    # -- HCFCs (Hydrochlorofluorocarbons) -----------------------------------
    R_22 = "R_22"
    R_123 = "R_123"
    R_141B = "R_141B"
    R_142B = "R_142B"

    # -- CFCs (Chlorofluorocarbons) -----------------------------------------
    R_11 = "R_11"
    R_12 = "R_12"
    R_113 = "R_113"
    R_114 = "R_114"
    R_115 = "R_115"
    R_502 = "R_502"

    # -- Natural refrigerants -----------------------------------------------
    R_717 = "R_717"
    R_744 = "R_744"
    R_290 = "R_290"
    R_600A = "R_600A"

    # -- Custom / user-defined ----------------------------------------------
    CUSTOM = "CUSTOM"


class GWPSource(str, Enum):
    """IPCC Assessment Report edition used for GWP conversion factors.

    AR4: Fourth Assessment Report (2007).
    AR5: Fifth Assessment Report (2014).
    AR6: Sixth Assessment Report (2021).
    AR6_20YR: Sixth Assessment Report 20-year GWP values.
    CUSTOM: User-provided GWP values.
    """

    AR4 = "AR4"
    AR5 = "AR5"
    AR6 = "AR6"
    AR6_20YR = "AR6_20YR"
    CUSTOM = "CUSTOM"


class GWPTimeframe(str, Enum):
    """Time horizon for GWP integration.

    GWP_20YR: 20-year integration horizon (emphasizes short-lived gases).
    GWP_100YR: 100-year integration horizon (GHG Protocol default).
    """

    GWP_20YR = "GWP_20YR"
    GWP_100YR = "GWP_100YR"


class CalculationMethod(str, Enum):
    """Methodology for calculating refrigerant and F-gas emissions.

    EQUIPMENT_BASED: Bottom-up calculation using equipment charge, leak
        rates, and service event data per GHG Protocol Chapter 8.
    MASS_BALANCE: Material balance approach using purchases, inventory
        changes, and disposals per EPA 40 CFR Part 98.
    SCREENING: Simplified screening approach using average emission
        factors per equipment type (for initial assessments).
    DIRECT_MEASUREMENT: Direct measurement of losses via leak detection.
    TOP_DOWN: Top-down estimate from total refrigerant purchases.
    """

    EQUIPMENT_BASED = "EQUIPMENT_BASED"
    MASS_BALANCE = "MASS_BALANCE"
    SCREENING = "SCREENING"
    DIRECT_MEASUREMENT = "DIRECT_MEASUREMENT"
    TOP_DOWN = "TOP_DOWN"


class EquipmentType(str, Enum):
    """Classification of refrigerant-containing equipment.

    Equipment type determines applicable default leak rates, service
    frequencies, and regulatory requirements per IPCC and EPA guidance.
    """

    COMMERCIAL_REFRIGERATION_CENTRALIZED = "commercial_refrigeration_centralized"
    COMMERCIAL_REFRIGERATION_STANDALONE = "commercial_refrigeration_standalone"
    INDUSTRIAL_REFRIGERATION = "industrial_refrigeration"
    RESIDENTIAL_AC = "residential_ac"
    COMMERCIAL_AC = "commercial_ac"
    CHILLERS_CENTRIFUGAL = "chillers_centrifugal"
    CHILLERS_SCREW = "chillers_screw"
    HEAT_PUMPS = "heat_pumps"
    TRANSPORT_REFRIGERATION = "transport_refrigeration"
    SWITCHGEAR = "switchgear"
    SEMICONDUCTOR = "semiconductor"
    FIRE_SUPPRESSION = "fire_suppression"
    FOAM_BLOWING = "foam_blowing"
    AEROSOLS = "aerosols"
    SOLVENTS = "solvents"


class EquipmentStatus(str, Enum):
    """Operational status of refrigerant-containing equipment.

    ACTIVE: Equipment is in normal operation.
    INACTIVE: Equipment is not currently in use but not decommissioned.
    DECOMMISSIONED: Equipment has been permanently taken out of service.
    MAINTENANCE: Equipment is temporarily offline for service.
    """

    ACTIVE = "active"
    INACTIVE = "inactive"
    DECOMMISSIONED = "decommissioned"
    MAINTENANCE = "maintenance"


class ServiceEventType(str, Enum):
    """Type of service event performed on refrigerant-containing equipment.

    Service events are used in equipment-based calculation methodology
    to track refrigerant additions, removals, and conversions over time.

    INSTALLATION: Initial equipment charging with refrigerant.
    RECHARGE: Addition of refrigerant to replace losses.
    REPAIR: Maintenance involving refrigerant handling.
    RECOVERY: Controlled removal of refrigerant from equipment.
    LEAK_CHECK: Scheduled or triggered leak detection inspection.
    DECOMMISSIONING: Final refrigerant recovery at end of equipment life.
    CONVERSION: Refrigerant type change (retrofit or drop-in replacement).
    """

    INSTALLATION = "installation"
    RECHARGE = "recharge"
    REPAIR = "repair"
    RECOVERY = "recovery"
    LEAK_CHECK = "leak_check"
    DECOMMISSIONING = "decommissioning"
    CONVERSION = "conversion"


class LifecycleStage(str, Enum):
    """Lifecycle stage of refrigerant-containing equipment.

    Lifecycle stage affects the applicable leak rate: installation has
    initial charge losses, operating has annual leak rates, and
    end-of-life has recovery efficiency losses.

    INSTALLATION: Equipment being installed and initially charged.
    OPERATING: Equipment in normal operational use.
    END_OF_LIFE: Equipment being decommissioned with refrigerant recovery.
    """

    INSTALLATION = "installation"
    OPERATING = "operating"
    END_OF_LIFE = "end_of_life"


class CalculationStatus(str, Enum):
    """Status of a refrigerant emission calculation.

    PENDING: Calculation queued but not yet started.
    RUNNING: Calculation in progress.
    COMPLETED: Calculation finished successfully.
    FAILED: Calculation terminated with an error.
    CANCELLED: Calculation was cancelled before completion.
    """

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReportingPeriod(str, Enum):
    """Temporal granularity for emission reporting aggregation.

    MONTHLY: Calendar month aggregation.
    QUARTERLY: Calendar quarter (Q1-Q4) aggregation.
    ANNUAL: Full calendar or fiscal year aggregation.
    """

    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUAL = "annual"


class RegulatoryFramework(str, Enum):
    """Regulatory framework governing calculation methodology and reporting.

    GHG_PROTOCOL: WRI/WBCSD Corporate GHG Protocol Chapter 8.
    ISO_14064: ISO 14064-1 Organizational Level GHG Quantification.
    CSRD_ESRS_E1: EU Corporate Sustainability Reporting Directive,
        European Sustainability Reporting Standard E1 (Climate Change).
    EPA_40CFR98_DD: US EPA GHGRP Subpart DD (Electrical Equipment).
    EPA_40CFR98_OO: US EPA GHGRP Subpart OO (Substitutes for ODS).
    EPA_40CFR98_L: US EPA GHGRP Subpart L (Fluorinated Gas Production).
    EU_FGAS_2024_573: EU Regulation 2024/573 on fluorinated gases.
    KIGALI_AMENDMENT: Kigali Amendment to the Montreal Protocol.
    UK_FGAS: UK Fluorinated Greenhouse Gases Regulations.
    """

    GHG_PROTOCOL = "ghg_protocol"
    ISO_14064 = "iso_14064"
    CSRD_ESRS_E1 = "csrd_esrs_e1"
    EPA_40CFR98_DD = "epa_40cfr98_dd"
    EPA_40CFR98_OO = "epa_40cfr98_oo"
    EPA_40CFR98_L = "epa_40cfr98_l"
    EU_FGAS_2024_573 = "eu_fgas_2024_573"
    KIGALI_AMENDMENT = "kigali_amendment"
    UK_FGAS = "uk_fgas"


class ComplianceStatus(str, Enum):
    """Compliance status against a specific regulatory requirement.

    COMPLIANT: Fully meets the regulatory requirement.
    WARNING: Approaching non-compliance threshold.
    NON_COMPLIANT: Fails to meet the regulatory requirement.
    EXEMPTED: Exempted from the requirement under specific provisions.
    NOT_APPLICABLE: Requirement does not apply to this entity.
    """

    COMPLIANT = "compliant"
    WARNING = "warning"
    NON_COMPLIANT = "non_compliant"
    EXEMPTED = "exempted"
    NOT_APPLICABLE = "not_applicable"


class PhaseDownSchedule(str, Enum):
    """HFC phase-down schedule for quota and compliance tracking.

    EU_FGAS: EU F-Gas Regulation 2024/573 phase-down steps.
    KIGALI_A5: Kigali Amendment schedule for Article 5 parties
        (developing countries).
    KIGALI_NON_A5: Kigali Amendment schedule for non-Article 5 parties
        (developed countries).
    """

    EU_FGAS = "eu_fgas"
    KIGALI_A5 = "kigali_a5"
    KIGALI_NON_A5 = "kigali_non_a5"


class UnitType(str, Enum):
    """Physical units for refrigerant mass measurement.

    KG: Kilograms (SI base unit for mass).
    LB: Pounds (US customary).
    OZ: Ounces (US customary).
    GRAM: Grams (SI).
    TONNE: Metric tonnes (1000 kg).
    METRIC_TON: Alias for metric tonne.
    """

    KG = "kg"
    LB = "lb"
    OZ = "oz"
    GRAM = "gram"
    TONNE = "tonne"
    METRIC_TON = "metric_ton"


# =============================================================================
# Data Models (14)
# =============================================================================


class GWPValue(BaseModel):
    """Global Warming Potential value for a specific source and timeframe.

    Captures the GWP numeric value along with its provenance (which IPCC
    Assessment Report and time horizon) for full traceability.

    Attributes:
        gwp_source: IPCC Assessment Report edition providing this value.
        timeframe: GWP integration time horizon (20-year or 100-year).
        value: Numeric GWP value (dimensionless, relative to CO2=1).
        effective_date: Date from which this GWP value is applicable.
    """

    gwp_source: GWPSource = Field(
        ...,
        description="IPCC Assessment Report edition providing this value",
    )
    timeframe: GWPTimeframe = Field(
        default=GWPTimeframe.GWP_100YR,
        description="GWP integration time horizon (20yr or 100yr)",
    )
    value: float = Field(
        ...,
        ge=0,
        description="Numeric GWP value (dimensionless, CO2=1)",
    )
    effective_date: Optional[datetime] = Field(
        default=None,
        description="Date from which this GWP value is applicable",
    )


class BlendComponent(BaseModel):
    """A single constituent gas in a blended refrigerant mixture.

    Defines the weight fraction and individual GWP of one component
    in a multi-component refrigerant blend (e.g., R-410A = R-32/R-125).

    Attributes:
        refrigerant_type: The constituent refrigerant type.
        weight_fraction: Mass fraction of this component (0.0-1.0).
        gwp: GWP value for this component at the specified timeframe.
    """

    refrigerant_type: RefrigerantType = Field(
        ...,
        description="The constituent refrigerant type",
    )
    weight_fraction: float = Field(
        ...,
        gt=0,
        le=1.0,
        description="Mass fraction of this component (0.0-1.0)",
    )
    gwp: Optional[float] = Field(
        default=None,
        ge=0,
        description="GWP value for this component",
    )

    @field_validator("weight_fraction")
    @classmethod
    def validate_weight_fraction(cls, v: float) -> float:
        """Validate weight fraction is a valid proportion."""
        if v <= 0.0 or v > 1.0:
            raise ValueError(
                f"weight_fraction must be in (0.0, 1.0], got {v}"
            )
        return v


class RefrigerantProperties(BaseModel):
    """Physical and chemical properties of a refrigerant type.

    Defines GWP values, molecular properties, ozone depletion potential,
    atmospheric lifetime, and blend composition for a refrigerant.

    Attributes:
        refrigerant_type: Refrigerant type these properties describe.
        category: Broad classification (HFC, PFC, SF6, etc.).
        name: Common or trade name of the refrigerant.
        formula: Chemical formula (e.g., CH2F2, C2HF5).
        molecular_weight: Molecular weight in g/mol.
        boiling_point_c: Normal boiling point in degrees Celsius.
        odp: Ozone Depletion Potential (relative to CFC-11=1).
        atmospheric_lifetime_years: Atmospheric lifetime in years.
        gwp_values: GWP values keyed by source identifier string
            (e.g. "AR6_100yr" -> GWPValue).
        blend_components: For blended refrigerants, the list of
            constituent components and their weight fractions.
        is_blend: Whether this is a multi-component blend.
        is_regulated: Whether this refrigerant is regulated under any
            applicable F-gas regulation.
        phase_out_date: Scheduled phase-out date if applicable.
    """

    refrigerant_type: RefrigerantType = Field(
        ...,
        description="Refrigerant type these properties describe",
    )
    category: RefrigerantCategory = Field(
        ...,
        description="Broad classification (HFC, PFC, SF6, etc.)",
    )
    name: str = Field(
        ...,
        min_length=1,
        max_length=255,
        description="Common or trade name of the refrigerant",
    )
    formula: Optional[str] = Field(
        default=None,
        max_length=100,
        description="Chemical formula (e.g., CH2F2, C2HF5)",
    )
    molecular_weight: Optional[float] = Field(
        default=None,
        gt=0,
        description="Molecular weight in g/mol",
    )
    boiling_point_c: Optional[float] = Field(
        default=None,
        description="Normal boiling point in degrees Celsius",
    )
    odp: float = Field(
        default=0.0,
        ge=0,
        description="Ozone Depletion Potential (relative to CFC-11=1)",
    )
    atmospheric_lifetime_years: Optional[float] = Field(
        default=None,
        ge=0,
        description="Atmospheric lifetime in years",
    )
    gwp_values: Dict[str, GWPValue] = Field(
        default_factory=dict,
        description="GWP values keyed by source identifier (e.g. 'AR6_100yr')",
    )
    blend_components: Optional[List[BlendComponent]] = Field(
        default=None,
        max_length=MAX_BLEND_COMPONENTS,
        description="Constituent components for blended refrigerants",
    )
    is_blend: bool = Field(
        default=False,
        description="Whether this is a multi-component blend",
    )
    is_regulated: bool = Field(
        default=True,
        description="Whether this refrigerant is regulated under F-gas law",
    )
    phase_out_date: Optional[datetime] = Field(
        default=None,
        description="Scheduled phase-out date if applicable",
    )

    @field_validator("blend_components")
    @classmethod
    def validate_blend_weights(
        cls, v: Optional[List[BlendComponent]], info: Any
    ) -> Optional[List[BlendComponent]]:
        """Validate that blend component weight fractions sum to approximately 1.0."""
        if v is not None and len(v) > 0:
            total_weight = sum(c.weight_fraction for c in v)
            if abs(total_weight - 1.0) > 0.01:
                raise ValueError(
                    f"Blend component weight fractions must sum to ~1.0, "
                    f"got {total_weight:.4f}"
                )
        return v


class EquipmentProfile(BaseModel):
    """Operational profile for a refrigerant-containing equipment unit.

    Equipment profiles enable equipment-based calculations by incorporating
    refrigerant charge, equipment type, status, location, and optional
    custom leak rate overrides.

    Attributes:
        equipment_id: Unique identifier for this equipment unit.
        equipment_type: Classification of the equipment.
        refrigerant_type: Type of refrigerant charged in this equipment.
        charge_kg: Refrigerant charge amount in kilograms.
        equipment_count: Number of identical units (for fleet tracking).
        status: Current operational status.
        installation_date: Date when equipment was installed/commissioned.
        location: Physical location or site identifier.
        custom_leak_rate: Optional override leak rate (fraction per year).
    """

    equipment_id: str = Field(
        default_factory=lambda: f"eq_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this equipment unit",
    )
    equipment_type: EquipmentType = Field(
        ...,
        description="Classification of the equipment",
    )
    refrigerant_type: RefrigerantType = Field(
        ...,
        description="Type of refrigerant charged in this equipment",
    )
    charge_kg: float = Field(
        ...,
        gt=0,
        description="Refrigerant charge amount in kilograms",
    )
    equipment_count: int = Field(
        default=1,
        ge=1,
        description="Number of identical units (fleet tracking)",
    )
    status: EquipmentStatus = Field(
        default=EquipmentStatus.ACTIVE,
        description="Current operational status",
    )
    installation_date: Optional[datetime] = Field(
        default=None,
        description="Date when equipment was installed",
    )
    location: Optional[str] = Field(
        default=None,
        max_length=500,
        description="Physical location or site identifier",
    )
    custom_leak_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Optional override leak rate (fraction per year, 0.0-1.0)",
    )


class ServiceEvent(BaseModel):
    """A refrigerant service event on a specific equipment unit.

    Records additions, removals, and handling of refrigerant during
    equipment maintenance and lifecycle events. Used in equipment-based
    and mass-balance calculation methodologies.

    Attributes:
        event_id: Unique identifier for this service event.
        equipment_id: Equipment unit this event relates to.
        event_type: Type of service event performed.
        date: Date when the service event occurred.
        refrigerant_added_kg: Amount of refrigerant added (kg).
        refrigerant_recovered_kg: Amount of refrigerant recovered (kg).
        notes: Optional human-readable description or technician notes.
    """

    event_id: str = Field(
        default_factory=lambda: f"svc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this service event",
    )
    equipment_id: str = Field(
        ...,
        min_length=1,
        description="Equipment unit this event relates to",
    )
    event_type: ServiceEventType = Field(
        ...,
        description="Type of service event performed",
    )
    date: datetime = Field(
        ...,
        description="Date when the service event occurred",
    )
    refrigerant_added_kg: float = Field(
        default=0.0,
        ge=0,
        description="Amount of refrigerant added in kilograms",
    )
    refrigerant_recovered_kg: float = Field(
        default=0.0,
        ge=0,
        description="Amount of refrigerant recovered in kilograms",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional description or technician notes",
    )


class LeakRateProfile(BaseModel):
    """Effective leak rate profile for an equipment type and lifecycle stage.

    Combines base leak rate with adjustment factors for equipment age,
    climate zone, and LDAR (Leak Detection and Repair) program presence
    to produce an effective annual leak rate.

    Effective rate = base_rate * age_factor * climate_factor * ldar_factor

    Attributes:
        equipment_type: Type of equipment this leak rate applies to.
        lifecycle_stage: Equipment lifecycle stage.
        base_rate: Base annual leak rate as a fraction (0.0-1.0).
        age_factor: Multiplicative factor for equipment age (>=1.0 for
            older equipment, 1.0 for new).
        climate_factor: Multiplicative factor for climate zone (>=1.0
            for hot climates, 1.0 for temperate).
        ldar_factor: Multiplicative factor for LDAR program (<1.0
            when LDAR is in place, 1.0 otherwise).
        effective_rate: Computed effective annual leak rate.
    """

    equipment_type: EquipmentType = Field(
        ...,
        description="Type of equipment this leak rate applies to",
    )
    lifecycle_stage: LifecycleStage = Field(
        default=LifecycleStage.OPERATING,
        description="Equipment lifecycle stage",
    )
    base_rate: float = Field(
        ...,
        ge=0,
        le=1.0,
        description="Base annual leak rate (fraction, 0.0-1.0)",
    )
    age_factor: float = Field(
        default=1.0,
        gt=0,
        description="Age adjustment factor (>= 1.0 for older equipment)",
    )
    climate_factor: float = Field(
        default=1.0,
        gt=0,
        description="Climate zone adjustment factor (>= 1.0 for hot climates)",
    )
    ldar_factor: float = Field(
        default=1.0,
        gt=0,
        le=1.0,
        description="LDAR program adjustment factor (<= 1.0 with LDAR)",
    )
    effective_rate: float = Field(
        default=0.0,
        ge=0,
        le=1.0,
        description="Computed effective annual leak rate",
    )


class MassBalanceData(BaseModel):
    """Mass balance input data for a single refrigerant type.

    Implements the mass-balance equation per EPA 40 CFR Part 98 Subpart OO:
    Emissions = (Beginning Inventory + Purchases + Acquisitions)
              - (Ending Inventory + Sales + Divestitures + Capacity Change)

    Attributes:
        refrigerant_type: Type of refrigerant for this balance.
        beginning_inventory_kg: Refrigerant inventory at period start (kg).
        purchases_kg: Refrigerant purchased during the period (kg).
        sales_kg: Refrigerant sold during the period (kg).
        acquisitions_kg: Refrigerant acquired via equipment transfer (kg).
        divestitures_kg: Refrigerant transferred out via equipment (kg).
        ending_inventory_kg: Refrigerant inventory at period end (kg).
        capacity_change_kg: Net change in total equipment charge capacity
            due to new equipment or decommissioning (kg).
    """

    refrigerant_type: RefrigerantType = Field(
        ...,
        description="Type of refrigerant for this mass balance",
    )
    beginning_inventory_kg: float = Field(
        ...,
        ge=0,
        description="Refrigerant inventory at period start (kg)",
    )
    purchases_kg: float = Field(
        default=0.0,
        ge=0,
        description="Refrigerant purchased during the period (kg)",
    )
    sales_kg: float = Field(
        default=0.0,
        ge=0,
        description="Refrigerant sold during the period (kg)",
    )
    acquisitions_kg: float = Field(
        default=0.0,
        ge=0,
        description="Refrigerant acquired via equipment transfer (kg)",
    )
    divestitures_kg: float = Field(
        default=0.0,
        ge=0,
        description="Refrigerant transferred out via equipment (kg)",
    )
    ending_inventory_kg: float = Field(
        ...,
        ge=0,
        description="Refrigerant inventory at period end (kg)",
    )
    capacity_change_kg: float = Field(
        default=0.0,
        description="Net change in total equipment charge capacity (kg)",
    )


class CalculationInput(BaseModel):
    """Input data for a refrigerant and F-gas emission calculation.

    Supports multiple calculation methods: equipment-based (requires
    equipment_profiles), mass-balance (requires mass_balance_data),
    and screening (requires basic equipment type and charge data).

    Attributes:
        calculation_method: Methodology to use for this calculation.
        equipment_profiles: Equipment profiles for equipment-based method.
        mass_balance_data: Mass balance records for mass-balance method.
        screening_charge_kg: Total refrigerant charge for screening method.
        screening_equipment_type: Equipment type for screening method.
        screening_leak_rate: Override leak rate for screening method.
        gwp_source: IPCC Assessment Report for GWP values.
        gwp_timeframe: GWP time horizon to use.
        reporting_period: Temporal granularity for the calculation.
        organization_id: Organization identifier for aggregation.
        tenant_id: Multi-tenant isolation identifier.
        service_events: Service events to include in calculation.
    """

    calculation_method: CalculationMethod = Field(
        default=CalculationMethod.EQUIPMENT_BASED,
        description="Methodology to use for this calculation",
    )
    equipment_profiles: Optional[List[EquipmentProfile]] = Field(
        default=None,
        max_length=MAX_EQUIPMENT_PROFILES,
        description="Equipment profiles for equipment-based method",
    )
    mass_balance_data: Optional[List[MassBalanceData]] = Field(
        default=None,
        description="Mass balance records for mass-balance method",
    )
    screening_charge_kg: Optional[float] = Field(
        default=None,
        gt=0,
        description="Total refrigerant charge for screening method (kg)",
    )
    screening_equipment_type: Optional[EquipmentType] = Field(
        default=None,
        description="Equipment type for screening method",
    )
    screening_leak_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1.0,
        description="Override leak rate for screening method (fraction)",
    )
    gwp_source: GWPSource = Field(
        default=GWPSource.AR6,
        description="IPCC Assessment Report for GWP values",
    )
    gwp_timeframe: GWPTimeframe = Field(
        default=GWPTimeframe.GWP_100YR,
        description="GWP time horizon to use",
    )
    reporting_period: Optional[ReportingPeriod] = Field(
        default=None,
        description="Temporal granularity for the calculation",
    )
    organization_id: Optional[str] = Field(
        default=None,
        description="Organization identifier for aggregation",
    )
    tenant_id: Optional[str] = Field(
        default=None,
        description="Multi-tenant isolation identifier",
    )
    service_events: Optional[List[ServiceEvent]] = Field(
        default=None,
        max_length=MAX_SERVICE_EVENTS,
        description="Service events to include in calculation",
    )


class GasEmission(BaseModel):
    """Emission result for a single refrigerant gas from an emission event.

    Captures the calculated gas loss and CO2-equivalent emissions along
    with the GWP applied for full traceability.

    Attributes:
        refrigerant_type: Specific refrigerant gas type.
        gas_name: Human-readable gas name (e.g., "R-134a", "R-32").
        loss_kg: Gas loss in kilograms.
        gwp_applied: GWP multiplier used for CO2e conversion.
        gwp_source: Source of the GWP value used.
        emissions_kg_co2e: Emissions in kilograms CO2-equivalent.
        emissions_tco2e: Emissions in tonnes CO2-equivalent.
        is_blend_component: Whether this is a decomposed blend component.
    """

    refrigerant_type: RefrigerantType = Field(
        ...,
        description="Specific refrigerant gas type",
    )
    gas_name: str = Field(
        ...,
        min_length=1,
        description="Human-readable gas name (e.g., 'R-134a')",
    )
    loss_kg: float = Field(
        ...,
        ge=0,
        description="Gas loss in kilograms",
    )
    gwp_applied: float = Field(
        ...,
        ge=0,
        description="GWP multiplier used for CO2e conversion",
    )
    gwp_source: str = Field(
        ...,
        min_length=1,
        description="Source of the GWP value used",
    )
    emissions_kg_co2e: float = Field(
        ...,
        ge=0,
        description="Emissions in kilograms CO2-equivalent",
    )
    emissions_tco2e: float = Field(
        ...,
        ge=0,
        description="Emissions in tonnes CO2-equivalent",
    )
    is_blend_component: bool = Field(
        default=False,
        description="Whether this is a decomposed blend component",
    )


class CalculationResult(BaseModel):
    """Complete result of a single refrigerant and F-gas emission calculation.

    Contains all calculated emissions by gas, total CO2e, the methodology
    parameters used, blend decomposition details, uncertainty estimate,
    and a SHA-256 provenance hash for audit trail integrity.

    Attributes:
        calculation_id: Unique identifier for this calculation result.
        method: Calculation methodology used.
        results: Itemized emissions for each gas or blend component.
        total_loss_kg: Total refrigerant loss in kilograms.
        total_emissions_tco2e: Total CO2-equivalent emissions in tonnes.
        blend_decomposition: Whether blend decomposition was applied.
        uncertainty: Optional uncertainty estimate for this calculation.
        provenance_hash: SHA-256 hash for audit trail integrity.
        timestamp: UTC timestamp when the calculation was performed.
        calculation_trace: Ordered list of human-readable calculation steps.
    """

    calculation_id: str = Field(
        default_factory=lambda: f"calc_{uuid.uuid4().hex[:12]}",
        description="Unique identifier for this calculation result",
    )
    method: CalculationMethod = Field(
        ...,
        description="Calculation methodology used",
    )
    results: List[GasEmission] = Field(
        default_factory=list,
        max_length=MAX_GASES_PER_RESULT,
        description="Itemized emissions for each gas or blend component",
    )
    total_loss_kg: float = Field(
        default=0.0,
        ge=0,
        description="Total refrigerant loss in kilograms",
    )
    total_emissions_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Total CO2-equivalent emissions in metric tonnes",
    )
    blend_decomposition: bool = Field(
        default=False,
        description="Whether blend decomposition was applied",
    )
    uncertainty: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional uncertainty estimate for this calculation",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash for audit trail integrity",
    )
    timestamp: datetime = Field(
        default_factory=_utcnow,
        description="UTC timestamp when the calculation was performed",
    )
    calculation_trace: List[str] = Field(
        default_factory=list,
        max_length=MAX_TRACE_STEPS,
        description="Ordered list of human-readable calculation steps",
    )


class BatchCalculationRequest(BaseModel):
    """Request model for batch refrigerant and F-gas calculations.

    Groups multiple calculation inputs for processing as a single
    batch, optionally in parallel.

    Attributes:
        calculations: List of individual calculation inputs to process.
        parallel: Whether to process calculations in parallel.
    """

    calculations: List[CalculationInput] = Field(
        ...,
        min_length=1,
        max_length=MAX_CALCULATIONS_PER_BATCH,
        description="List of individual calculation inputs to process",
    )
    parallel: bool = Field(
        default=False,
        description="Whether to process calculations in parallel",
    )


class BatchCalculationResponse(BaseModel):
    """Response model for a batch refrigerant and F-gas calculation.

    Aggregates individual calculation results with batch-level totals
    and processing metadata.

    Attributes:
        results: List of individual calculation results.
        total_emissions_tco2e: Batch total CO2-equivalent in tonnes.
        success_count: Number of successful calculations.
        failure_count: Number of failed calculations.
        processing_time_ms: Total batch processing time in milliseconds.
        provenance_hash: SHA-256 hash covering the entire batch result.
    """

    results: List[CalculationResult] = Field(
        default_factory=list,
        description="List of individual calculation results",
    )
    total_emissions_tco2e: float = Field(
        default=0.0,
        ge=0,
        description="Batch total CO2-equivalent in metric tonnes",
    )
    success_count: int = Field(
        default=0,
        ge=0,
        description="Number of successful calculations",
    )
    failure_count: int = Field(
        default=0,
        ge=0,
        description="Number of failed calculations",
    )
    processing_time_ms: float = Field(
        default=0.0,
        ge=0,
        description="Total batch processing wall-clock time in milliseconds",
    )
    provenance_hash: str = Field(
        default="",
        description="SHA-256 hash covering the entire batch result",
    )


class UncertaintyResult(BaseModel):
    """Monte Carlo uncertainty quantification result for a refrigerant calculation.

    Provides statistical characterization of emission estimate uncertainty
    including mean, standard deviation, and confidence intervals at multiple
    levels.

    Attributes:
        method: Uncertainty quantification method used (monte_carlo,
            analytical, tier_default).
        mean: Mean emission estimate in tonnes CO2-equivalent.
        std_dev: Standard deviation of the estimate in tonnes CO2e.
        ci_90_lower: 90% confidence interval lower bound (tCO2e).
        ci_90_upper: 90% confidence interval upper bound (tCO2e).
        ci_95_lower: 95% confidence interval lower bound (tCO2e).
        ci_95_upper: 95% confidence interval upper bound (tCO2e).
        ci_99_lower: 99% confidence interval lower bound (tCO2e).
        ci_99_upper: 99% confidence interval upper bound (tCO2e).
        iterations: Number of Monte Carlo iterations performed.
        data_quality_score: Overall data quality indicator (1-5 scale).
    """

    method: str = Field(
        ...,
        min_length=1,
        description="Uncertainty quantification method used",
    )
    mean: float = Field(
        ...,
        ge=0,
        description="Mean emission estimate (tonnes CO2e)",
    )
    std_dev: float = Field(
        ...,
        ge=0,
        description="Standard deviation of the estimate (tonnes CO2e)",
    )
    ci_90_lower: float = Field(
        ...,
        ge=0,
        description="90% CI lower bound (tCO2e)",
    )
    ci_90_upper: float = Field(
        ...,
        ge=0,
        description="90% CI upper bound (tCO2e)",
    )
    ci_95_lower: float = Field(
        ...,
        ge=0,
        description="95% CI lower bound (tCO2e)",
    )
    ci_95_upper: float = Field(
        ...,
        ge=0,
        description="95% CI upper bound (tCO2e)",
    )
    ci_99_lower: float = Field(
        ...,
        ge=0,
        description="99% CI lower bound (tCO2e)",
    )
    ci_99_upper: float = Field(
        ...,
        ge=0,
        description="99% CI upper bound (tCO2e)",
    )
    iterations: int = Field(
        ...,
        gt=0,
        description="Number of Monte Carlo iterations performed",
    )
    data_quality_score: Optional[float] = Field(
        default=None,
        ge=1,
        le=5,
        description="Data quality indicator (1-5 scale, GHG Protocol guidance)",
    )


class ComplianceRecord(BaseModel):
    """Regulatory compliance assessment for refrigerant and F-gas emissions.

    Tracks compliance status against specific frameworks including
    quota usage tracking for phase-down schedules.

    Attributes:
        framework: Regulatory framework being assessed.
        status: Current compliance status.
        quota_co2e: Total allocated quota in tonnes CO2-equivalent
            (for phase-down compliance).
        usage_co2e: Current quota usage in tonnes CO2-equivalent.
        remaining_co2e: Remaining quota in tonnes CO2-equivalent.
        phase_down_target_pct: Target percentage of baseline for the
            current compliance period.
        notes: Optional compliance assessor notes.
    """

    framework: RegulatoryFramework = Field(
        ...,
        description="Regulatory framework being assessed",
    )
    status: ComplianceStatus = Field(
        ...,
        description="Current compliance status",
    )
    quota_co2e: Optional[float] = Field(
        default=None,
        ge=0,
        description="Total allocated quota (tCO2e, for phase-down compliance)",
    )
    usage_co2e: Optional[float] = Field(
        default=None,
        ge=0,
        description="Current quota usage (tCO2e)",
    )
    remaining_co2e: Optional[float] = Field(
        default=None,
        ge=0,
        description="Remaining quota (tCO2e)",
    )
    phase_down_target_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=100.0,
        description="Target percentage of baseline for current period",
    )
    notes: Optional[str] = Field(
        default=None,
        max_length=2000,
        description="Optional compliance assessor notes",
    )

    @field_validator("usage_co2e")
    @classmethod
    def usage_within_quota(
        cls, v: Optional[float], info: Any
    ) -> Optional[float]:
        """Warn-level validation: usage should not exceed quota."""
        # This is a soft validation; non-compliant usage is permitted
        # but tracked via ComplianceStatus.NON_COMPLIANT
        return v


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Constants
    "VERSION",
    "MAX_CALCULATIONS_PER_BATCH",
    "MAX_GASES_PER_RESULT",
    "MAX_TRACE_STEPS",
    "MAX_BLEND_COMPONENTS",
    "MAX_EQUIPMENT_PROFILES",
    "MAX_SERVICE_EVENTS",
    "GWP_REFERENCE",
    # Enums
    "RefrigerantCategory",
    "RefrigerantType",
    "GWPSource",
    "GWPTimeframe",
    "CalculationMethod",
    "EquipmentType",
    "EquipmentStatus",
    "ServiceEventType",
    "LifecycleStage",
    "CalculationStatus",
    "ReportingPeriod",
    "RegulatoryFramework",
    "ComplianceStatus",
    "PhaseDownSchedule",
    "UnitType",
    # Data models
    "GWPValue",
    "BlendComponent",
    "RefrigerantProperties",
    "EquipmentProfile",
    "ServiceEvent",
    "LeakRateProfile",
    "CalculationInput",
    "MassBalanceData",
    "GasEmission",
    "CalculationResult",
    "BatchCalculationRequest",
    "BatchCalculationResponse",
    "UncertaintyResult",
    "ComplianceRecord",
]
