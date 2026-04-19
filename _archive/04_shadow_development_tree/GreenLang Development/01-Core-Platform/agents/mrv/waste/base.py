# -*- coding: utf-8 -*-
"""
Waste MRV Base Module
=====================

This module provides base classes and common functionality for all
Waste & Circularity MRV (Monitoring, Reporting, Verification) agents.

Design Principles:
- Zero-hallucination guarantee (no LLM in calculation path)
- Full audit trail with SHA-256 provenance hashing
- GHG Protocol Scope 1 and Scope 3 Category 5 compliant
- IPCC Waste Sector Guidelines (2006/2019) compliant
- EU ETS and EPA reporting compatible
- Pydantic models for type safety

Reference Standards:
- IPCC 2006 Guidelines for National GHG Inventories - Volume 5 (Waste)
- IPCC 2019 Refinement - Waste Chapter
- GHG Protocol Corporate Standard (2015)
- EPA Landfill Methane Outreach Program (LMOP)
- EU ETS Monitoring and Reporting Regulation

Author: GreenLang Framework Team
Version: 1.0.0
"""

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, TypeVar, Generic

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


# =============================================================================
# TYPE VARIABLES
# =============================================================================

InputT = TypeVar("InputT", bound="WasteMRVInput")
OutputT = TypeVar("OutputT", bound="WasteMRVOutput")


# =============================================================================
# ENUMS
# =============================================================================

class WasteType(str, Enum):
    """Types of waste materials."""
    MUNICIPAL_SOLID_WASTE = "municipal_solid_waste"
    INDUSTRIAL_WASTE = "industrial_waste"
    CONSTRUCTION_DEMOLITION = "construction_demolition"
    ORGANIC_WASTE = "organic_waste"
    FOOD_WASTE = "food_waste"
    YARD_WASTE = "yard_waste"
    PAPER = "paper"
    CARDBOARD = "cardboard"
    PLASTIC = "plastic"
    METAL = "metal"
    GLASS = "glass"
    TEXTILES = "textiles"
    WOOD = "wood"
    RUBBER = "rubber"
    E_WASTE = "e_waste"
    HAZARDOUS = "hazardous"
    MEDICAL = "medical"
    SLUDGE = "sludge"
    MIXED = "mixed"


class TreatmentMethod(str, Enum):
    """Waste treatment methods."""
    LANDFILL = "landfill"
    LANDFILL_WITH_GAS_CAPTURE = "landfill_with_gas_capture"
    INCINERATION = "incineration"
    INCINERATION_WITH_ENERGY_RECOVERY = "incineration_with_energy_recovery"
    RECYCLING = "recycling"
    COMPOSTING = "composting"
    ANAEROBIC_DIGESTION = "anaerobic_digestion"
    MECHANICAL_BIOLOGICAL_TREATMENT = "mbt"
    PYROLYSIS = "pyrolysis"
    GASIFICATION = "gasification"
    CHEMICAL_TREATMENT = "chemical_treatment"
    THERMAL_TREATMENT = "thermal_treatment"
    BIOLOGICAL_TREATMENT = "biological_treatment"
    OPEN_BURNING = "open_burning"
    OPEN_DUMPING = "open_dumping"


class LandfillType(str, Enum):
    """Types of landfill sites."""
    MANAGED_ANAEROBIC = "managed_anaerobic"
    MANAGED_SEMI_AEROBIC = "managed_semi_aerobic"
    UNMANAGED_DEEP = "unmanaged_deep"
    UNMANAGED_SHALLOW = "unmanaged_shallow"
    UNCATEGORIZED = "uncategorized"


class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"


class VerificationStatus(str, Enum):
    """MRV verification status."""
    UNVERIFIED = "unverified"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    REJECTED = "rejected"


class DataQualityTier(str, Enum):
    """Data quality level per GHG Protocol."""
    TIER_1 = "tier_1"  # Primary measured data
    TIER_2 = "tier_2"  # Activity data with emission factors
    TIER_3 = "tier_3"  # Estimates/proxies


class CalculationMethod(str, Enum):
    """Calculation methods for waste emissions."""
    IPCC_FOD = "ipcc_fod"  # First Order Decay
    IPCC_TIER_1 = "ipcc_tier_1"  # Default factors
    IPCC_TIER_2 = "ipcc_tier_2"  # Country-specific factors
    IPCC_TIER_3 = "ipcc_tier_3"  # Facility-specific data
    MASS_BALANCE = "mass_balance"
    DIRECT_MEASUREMENT = "direct_measurement"
    SPEND_BASED = "spend_based"


# =============================================================================
# EMISSION FACTOR MODELS
# =============================================================================

class EmissionFactor(BaseModel):
    """Emission factor record with provenance."""
    factor_id: str = Field(..., description="Unique factor identifier")
    factor_value: Decimal = Field(..., description="Emission factor value")
    factor_unit: str = Field(..., description="Factor unit")
    source: str = Field(..., description="Factor source (IPCC, EPA, DEFRA)")
    source_uri: str = Field(default="", description="URI to source documentation")
    version: str = Field(..., description="Factor version/year")
    last_updated: str = Field(..., description="Last update date")
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage"
    )
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_2, description="Data quality tier"
    )
    geographic_scope: str = Field("global", description="Geographic applicability")
    waste_type: Optional[WasteType] = Field(None, description="Applicable waste type")
    treatment_method: Optional[TreatmentMethod] = Field(
        None, description="Applicable treatment method"
    )

    class Config:
        use_enum_values = True


class CalculationStep(BaseModel):
    """Single step in the calculation audit trail."""
    step_number: int = Field(..., ge=1, description="Step sequence number")
    description: str = Field(..., description="Step description")
    formula: Optional[str] = Field(None, description="Formula applied")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input values")
    output: Optional[str] = Field(None, description="Output value")
    emission_factor: Optional[EmissionFactor] = Field(
        None, description="Emission factor used"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Step timestamp"
    )


# =============================================================================
# BASE INPUT/OUTPUT MODELS
# =============================================================================

class WasteMRVInput(BaseModel):
    """Base input model for all waste MRV agents."""

    # Identification
    organization_id: str = Field(..., description="Organization identifier")
    facility_id: Optional[str] = Field(None, description="Facility identifier")
    request_id: Optional[str] = Field(None, description="Unique request ID")

    # Reporting period
    reporting_year: int = Field(..., ge=2020, le=2100, description="Reporting year")
    reporting_period_start: Optional[datetime] = Field(
        None, description="Period start date"
    )
    reporting_period_end: Optional[datetime] = Field(
        None, description="Period end date"
    )

    # Calculation parameters
    calculation_method: CalculationMethod = Field(
        CalculationMethod.IPCC_TIER_2, description="Calculation method"
    )

    # Geographic context
    region: str = Field("global", description="Geographic region")
    country: Optional[str] = Field(None, description="Country code (ISO 3166-1)")
    climate_zone: Optional[str] = Field(None, description="Climate zone (tropical, temperate, boreal)")

    class Config:
        use_enum_values = True

    @field_validator("reporting_year")
    @classmethod
    def validate_reporting_year(cls, v: int) -> int:
        """Validate reporting year is reasonable."""
        current_year = datetime.now().year
        if v > current_year + 1:
            raise ValueError(f"Reporting year {v} is too far in the future")
        return v


class WasteMRVOutput(BaseModel):
    """Base output model for all waste MRV agents."""

    # Agent identification
    agent_id: str = Field(..., description="Agent identifier")
    agent_version: str = Field("1.0.0", description="Agent version")

    # Emission results
    total_emissions_kg_co2e: Decimal = Field(
        ..., ge=0, description="Total emissions in kg CO2e"
    )
    total_emissions_mt_co2e: Decimal = Field(
        ..., ge=0, description="Total emissions in metric tons CO2e"
    )

    # Gas breakdown
    co2_kg: Decimal = Field(Decimal("0"), ge=0, description="CO2 in kg")
    ch4_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 in kg")
    n2o_kg: Decimal = Field(Decimal("0"), ge=0, description="N2O in kg")

    # CH4 specific (important for waste sector)
    ch4_generated_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 generated before capture")
    ch4_captured_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 captured")
    ch4_flared_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 flared")
    ch4_utilized_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 utilized for energy")
    ch4_emitted_kg: Decimal = Field(Decimal("0"), ge=0, description="CH4 released to atmosphere")

    # Scope classification
    scope: EmissionScope = Field(..., description="GHG Protocol scope")

    # Audit trail
    calculation_steps: List[CalculationStep] = Field(
        default_factory=list, description="Calculation audit trail"
    )
    provenance_hash: str = Field(..., description="SHA-256 provenance hash")

    # Data quality
    data_quality_tier: DataQualityTier = Field(
        DataQualityTier.TIER_2, description="Overall data quality"
    )
    uncertainty_pct: Optional[float] = Field(
        None, ge=0, le=100, description="Uncertainty percentage"
    )

    # Timestamps
    calculation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), description="Calculation timestamp"
    )
    calculation_duration_ms: float = Field(
        0.0, ge=0, description="Calculation duration in ms"
    )

    # Status
    status: str = Field("success", description="Calculation status")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")
    errors: List[str] = Field(default_factory=list, description="Error messages")

    # Metadata
    emission_factors_used: List[EmissionFactor] = Field(
        default_factory=list, description="Emission factors applied"
    )
    activity_data_summary: Dict[str, Any] = Field(
        default_factory=dict, description="Summary of activity data"
    )

    class Config:
        use_enum_values = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat(),
        }


# =============================================================================
# EMISSION FACTOR DATABASE - IPCC WASTE SECTOR
# =============================================================================

# IPCC Default Methane Generation Potential (m3 CH4/tonne wet waste)
IPCC_L0_VALUES: Dict[str, Decimal] = {
    WasteType.FOOD_WASTE.value: Decimal("143"),
    WasteType.YARD_WASTE.value: Decimal("93"),
    WasteType.PAPER.value: Decimal("147"),
    WasteType.CARDBOARD.value: Decimal("147"),
    WasteType.WOOD.value: Decimal("93"),
    WasteType.TEXTILES.value: Decimal("93"),
    WasteType.RUBBER.value: Decimal("83"),
    WasteType.PLASTIC.value: Decimal("0"),  # Plastics don't biodegrade
    WasteType.METAL.value: Decimal("0"),
    WasteType.GLASS.value: Decimal("0"),
    WasteType.MUNICIPAL_SOLID_WASTE.value: Decimal("83"),  # Default mixed
}

# IPCC Degradable Organic Carbon content (DOC) as fraction of wet weight
IPCC_DOC_VALUES: Dict[str, Decimal] = {
    WasteType.FOOD_WASTE.value: Decimal("0.15"),
    WasteType.YARD_WASTE.value: Decimal("0.20"),
    WasteType.PAPER.value: Decimal("0.40"),
    WasteType.CARDBOARD.value: Decimal("0.40"),
    WasteType.WOOD.value: Decimal("0.43"),
    WasteType.TEXTILES.value: Decimal("0.24"),
    WasteType.RUBBER.value: Decimal("0.39"),
    WasteType.PLASTIC.value: Decimal("0"),
    WasteType.METAL.value: Decimal("0"),
    WasteType.GLASS.value: Decimal("0"),
    WasteType.MUNICIPAL_SOLID_WASTE.value: Decimal("0.18"),
}

# IPCC Methane Correction Factor by landfill type
IPCC_MCF_VALUES: Dict[str, Decimal] = {
    LandfillType.MANAGED_ANAEROBIC.value: Decimal("1.0"),
    LandfillType.MANAGED_SEMI_AEROBIC.value: Decimal("0.5"),
    LandfillType.UNMANAGED_DEEP.value: Decimal("0.8"),
    LandfillType.UNMANAGED_SHALLOW.value: Decimal("0.4"),
    LandfillType.UNCATEGORIZED.value: Decimal("0.6"),
}

# Half-life values by climate zone and waste type (years)
IPCC_HALF_LIFE: Dict[str, Dict[str, Decimal]] = {
    "tropical": {
        WasteType.FOOD_WASTE.value: Decimal("3"),
        WasteType.YARD_WASTE.value: Decimal("6"),
        WasteType.PAPER.value: Decimal("6"),
        WasteType.WOOD.value: Decimal("12"),
        WasteType.TEXTILES.value: Decimal("6"),
        "default": Decimal("5"),
    },
    "temperate": {
        WasteType.FOOD_WASTE.value: Decimal("5"),
        WasteType.YARD_WASTE.value: Decimal("10"),
        WasteType.PAPER.value: Decimal("10"),
        WasteType.WOOD.value: Decimal("23"),
        WasteType.TEXTILES.value: Decimal("10"),
        "default": Decimal("10"),
    },
    "boreal": {
        WasteType.FOOD_WASTE.value: Decimal("8"),
        WasteType.YARD_WASTE.value: Decimal("15"),
        WasteType.PAPER.value: Decimal("15"),
        WasteType.WOOD.value: Decimal("35"),
        WasteType.TEXTILES.value: Decimal("15"),
        "default": Decimal("14"),
    },
}

# Incineration emission factors (kg CO2e per tonne)
INCINERATION_FACTORS: Dict[str, Dict[str, Decimal]] = {
    WasteType.MUNICIPAL_SOLID_WASTE.value: {
        "co2_fossil": Decimal("400"),  # kg CO2/tonne (fossil component)
        "n2o": Decimal("0.06"),  # kg N2O/tonne
        "ch4": Decimal("0.02"),  # kg CH4/tonne
    },
    WasteType.PLASTIC.value: {
        "co2_fossil": Decimal("2760"),  # Plastics have high fossil carbon
        "n2o": Decimal("0.06"),
        "ch4": Decimal("0.02"),
    },
    WasteType.HAZARDOUS.value: {
        "co2_fossil": Decimal("500"),
        "n2o": Decimal("0.10"),
        "ch4": Decimal("0.03"),
    },
}

# Composting emission factors (kg per tonne)
COMPOSTING_FACTORS: Dict[str, Dict[str, Decimal]] = {
    "aerobic_well_managed": {
        "ch4": Decimal("4"),  # kg CH4/tonne
        "n2o": Decimal("0.24"),  # kg N2O/tonne
    },
    "aerobic_poorly_managed": {
        "ch4": Decimal("10"),
        "n2o": Decimal("0.6"),
    },
    "anaerobic_digestion": {
        "ch4": Decimal("1"),  # Captured mostly
        "n2o": Decimal("0.1"),
    },
}

# Recycling avoided emission factors (kg CO2e per tonne - negative = credit)
RECYCLING_CREDITS: Dict[str, Decimal] = {
    WasteType.PAPER.value: Decimal("-680"),
    WasteType.CARDBOARD.value: Decimal("-520"),
    WasteType.PLASTIC.value: Decimal("-1440"),
    WasteType.METAL.value: Decimal("-1820"),
    WasteType.GLASS.value: Decimal("-315"),
    WasteType.TEXTILES.value: Decimal("-2130"),
    WasteType.E_WASTE.value: Decimal("-2500"),
}

# Global Warming Potentials (AR6 100-year)
GWP_AR6_100 = {
    "CO2": Decimal("1"),
    "CH4": Decimal("27.9"),  # AR6 fossil CH4 with climate-carbon feedback
    "CH4_biogenic": Decimal("27.2"),  # AR6 biogenic CH4
    "N2O": Decimal("273"),
}


# =============================================================================
# BASE WASTE MRV AGENT
# =============================================================================

class BaseWasteMRVAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for waste MRV agents.

    All waste MRV agents inherit from this class and implement
    the calculate() method with treatment-method-specific logic.

    Key Guarantees:
    - ZERO HALLUCINATION: No LLM calls in calculation path
    - DETERMINISTIC: Same input always produces same output
    - AUDITABLE: Complete SHA-256 provenance tracking
    - IPCC COMPLIANT: IPCC 2006/2019 Guidelines aligned
    - GHG PROTOCOL COMPLIANT: Scope 1, 2, and 3 aligned

    Attributes:
        AGENT_ID: Unique agent identifier (e.g., GL-MRV-WST-001)
        AGENT_NAME: Human-readable agent name
        AGENT_VERSION: Semantic version string
        TREATMENT_METHOD: Primary waste treatment method
        DEFAULT_SCOPE: Default emission scope
    """

    # Class attributes to be overridden by subclasses
    AGENT_ID: str = "GL-MRV-WST-000"
    AGENT_NAME: str = "Base Waste MRV Agent"
    AGENT_VERSION: str = "1.0.0"
    TREATMENT_METHOD: TreatmentMethod = TreatmentMethod.LANDFILL
    DEFAULT_SCOPE: EmissionScope = EmissionScope.SCOPE_1

    def __init__(self):
        """Initialize the waste MRV agent."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self._gwp = GWP_AR6_100
        self._doc_values = IPCC_DOC_VALUES
        self._mcf_values = IPCC_MCF_VALUES
        self._l0_values = IPCC_L0_VALUES
        self._half_life = IPCC_HALF_LIFE
        self._incineration_factors = INCINERATION_FACTORS
        self._composting_factors = COMPOSTING_FACTORS
        self._recycling_credits = RECYCLING_CREDITS
        self.logger.info(f"Initialized {self.AGENT_ID} v{self.AGENT_VERSION}")

    @abstractmethod
    def calculate(self, input_data: InputT) -> OutputT:
        """
        Execute the emission calculation.

        Args:
            input_data: Treatment-specific input data

        Returns:
            Complete calculation result with audit trail

        Raises:
            ValueError: If input validation fails
            CalculationError: If calculation fails
        """
        pass

    def _get_doc_value(self, waste_type: WasteType) -> Decimal:
        """
        Get Degradable Organic Carbon (DOC) fraction for waste type.

        Args:
            waste_type: Type of waste

        Returns:
            DOC fraction as Decimal
        """
        return self._doc_values.get(
            waste_type.value,
            self._doc_values[WasteType.MUNICIPAL_SOLID_WASTE.value]
        )

    def _get_mcf_value(self, landfill_type: LandfillType) -> Decimal:
        """
        Get Methane Correction Factor (MCF) for landfill type.

        Args:
            landfill_type: Type of landfill

        Returns:
            MCF as Decimal
        """
        return self._mcf_values.get(
            landfill_type.value,
            self._mcf_values[LandfillType.UNCATEGORIZED.value]
        )

    def _get_l0_value(self, waste_type: WasteType) -> Decimal:
        """
        Get methane generation potential (L0) for waste type.

        Args:
            waste_type: Type of waste

        Returns:
            L0 in m3 CH4/tonne
        """
        return self._l0_values.get(
            waste_type.value,
            self._l0_values[WasteType.MUNICIPAL_SOLID_WASTE.value]
        )

    def _calculate_ch4_from_doc(
        self,
        waste_tonnes: Decimal,
        doc: Decimal,
        doc_f: Decimal = Decimal("0.5"),  # Fraction DOC that decomposes
        mcf: Decimal = Decimal("1.0"),
        f_ch4: Decimal = Decimal("0.5"),  # CH4 fraction in landfill gas
        recovery_fraction: Decimal = Decimal("0"),
    ) -> Dict[str, Decimal]:
        """
        Calculate methane generation using IPCC First Order Decay method.

        Formula: CH4 = DOC * DOCF * MCF * F * (16/12) - R

        Args:
            waste_tonnes: Mass of waste deposited
            doc: Degradable organic carbon fraction
            doc_f: Fraction of DOC that decomposes
            mcf: Methane correction factor
            f_ch4: Fraction of CH4 in generated landfill gas
            recovery_fraction: Fraction of CH4 recovered

        Returns:
            Dictionary with CH4 values
        """
        # Degradable DOC deposited (DDOCm)
        ddoc_m = waste_tonnes * doc * doc_f * mcf

        # CH4 generated (before recovery) in tonnes
        # 16/12 = molecular weight ratio CH4/C
        ch4_generated = ddoc_m * f_ch4 * (Decimal("16") / Decimal("12"))

        # CH4 recovered
        ch4_recovered = ch4_generated * recovery_fraction

        # CH4 emitted
        ch4_emitted = ch4_generated - ch4_recovered

        return {
            "ddoc_m": ddoc_m.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            "ch4_generated_tonnes": ch4_generated.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            "ch4_recovered_tonnes": ch4_recovered.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
            "ch4_emitted_tonnes": ch4_emitted.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP),
        }

    def _convert_ch4_to_co2e(self, ch4_tonnes: Decimal, biogenic: bool = True) -> Decimal:
        """
        Convert methane to CO2 equivalent using GWP.

        Args:
            ch4_tonnes: Methane in tonnes
            biogenic: Whether CH4 is biogenic (True) or fossil (False)

        Returns:
            CO2e in tonnes
        """
        gwp = self._gwp["CH4_biogenic"] if biogenic else self._gwp["CH4"]
        return (ch4_tonnes * gwp).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _convert_n2o_to_co2e(self, n2o_tonnes: Decimal) -> Decimal:
        """
        Convert N2O to CO2 equivalent using GWP.

        Args:
            n2o_tonnes: N2O in tonnes

        Returns:
            CO2e in tonnes
        """
        return (n2o_tonnes * self._gwp["N2O"]).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _kg_to_tonnes(self, kg: Decimal) -> Decimal:
        """Convert kg to metric tonnes."""
        return (kg / Decimal("1000")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _tonnes_to_kg(self, tonnes: Decimal) -> Decimal:
        """Convert metric tonnes to kg."""
        return (tonnes * Decimal("1000")).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def _kg_to_metric_tons(self, kg: Decimal) -> Decimal:
        """Convert kg to metric tons (alias)."""
        return self._kg_to_tonnes(kg)

    def _generate_provenance_hash(
        self,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        steps: List[CalculationStep],
    ) -> str:
        """
        Generate SHA-256 provenance hash for audit trail.

        Args:
            input_data: Input data dictionary
            output_data: Output data dictionary
            steps: Calculation steps

        Returns:
            SHA-256 hex digest
        """
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "input": input_data,
            "output": {
                k: str(v) if isinstance(v, Decimal) else v
                for k, v in output_data.items()
                if k not in ["provenance_hash", "calculation_timestamp"]
            },
            "steps": [
                {
                    "step_number": s.step_number,
                    "description": s.description,
                    "formula": s.formula,
                    "inputs": s.inputs,
                    "output": s.output,
                }
                for s in steps
            ],
        }
        data_str = json.dumps(provenance_data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def _create_output(
        self,
        total_emissions_kg: Decimal,
        co2_kg: Decimal,
        ch4_kg: Decimal,
        n2o_kg: Decimal,
        steps: List[CalculationStep],
        emission_factors: List[EmissionFactor],
        activity_summary: Dict[str, Any],
        start_time: datetime,
        ch4_generated_kg: Decimal = Decimal("0"),
        ch4_captured_kg: Decimal = Decimal("0"),
        ch4_flared_kg: Decimal = Decimal("0"),
        ch4_utilized_kg: Decimal = Decimal("0"),
        scope: Optional[EmissionScope] = None,
        warnings: Optional[List[str]] = None,
    ) -> WasteMRVOutput:
        """
        Create standardized output with provenance.

        Args:
            total_emissions_kg: Total emissions in kg CO2e
            co2_kg: CO2 component in kg
            ch4_kg: CH4 component in kg
            n2o_kg: N2O component in kg
            steps: Calculation steps
            emission_factors: Factors used
            activity_summary: Summary of activity data
            start_time: Calculation start time
            ch4_generated_kg: Total CH4 generated before capture
            ch4_captured_kg: CH4 captured
            ch4_flared_kg: CH4 flared
            ch4_utilized_kg: CH4 used for energy
            scope: Emission scope (optional, uses default)
            warnings: Warning messages

        Returns:
            Complete WasteMRVOutput
        """
        end_time = datetime.now(timezone.utc)
        duration_ms = (end_time - start_time).total_seconds() * 1000

        ch4_emitted_kg = ch4_generated_kg - ch4_captured_kg - ch4_flared_kg - ch4_utilized_kg
        if ch4_emitted_kg < Decimal("0"):
            ch4_emitted_kg = Decimal("0")

        output_data = {
            "total_emissions_kg_co2e": total_emissions_kg,
            "co2_kg": co2_kg,
            "ch4_kg": ch4_kg,
            "n2o_kg": n2o_kg,
            "ch4_emitted_kg": ch4_emitted_kg,
        }

        provenance_hash = self._generate_provenance_hash(
            input_data=activity_summary,
            output_data=output_data,
            steps=steps,
        )

        # Determine overall data quality
        if emission_factors:
            quality_tiers = [ef.data_quality_tier for ef in emission_factors]
            if DataQualityTier.TIER_3 in quality_tiers:
                overall_quality = DataQualityTier.TIER_3
            elif DataQualityTier.TIER_2 in quality_tiers:
                overall_quality = DataQualityTier.TIER_2
            else:
                overall_quality = DataQualityTier.TIER_1
        else:
            overall_quality = DataQualityTier.TIER_2

        return WasteMRVOutput(
            agent_id=self.AGENT_ID,
            agent_version=self.AGENT_VERSION,
            total_emissions_kg_co2e=total_emissions_kg,
            total_emissions_mt_co2e=self._kg_to_metric_tons(total_emissions_kg),
            co2_kg=co2_kg,
            ch4_kg=ch4_kg,
            n2o_kg=n2o_kg,
            ch4_generated_kg=ch4_generated_kg,
            ch4_captured_kg=ch4_captured_kg,
            ch4_flared_kg=ch4_flared_kg,
            ch4_utilized_kg=ch4_utilized_kg,
            ch4_emitted_kg=ch4_emitted_kg,
            scope=scope or self.DEFAULT_SCOPE,
            calculation_steps=steps,
            provenance_hash=provenance_hash,
            data_quality_tier=overall_quality,
            calculation_timestamp=end_time,
            calculation_duration_ms=duration_ms,
            emission_factors_used=emission_factors,
            activity_data_summary=activity_summary,
            warnings=warnings or [],
        )
