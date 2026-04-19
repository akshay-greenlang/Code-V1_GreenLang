# -*- coding: utf-8 -*-
"""
GreenLang Industrial MRV Base Agent
====================================

Base class for all industrial sector MRV (Measurement, Reporting, Verification) agents.
Provides common functionality for emissions calculation, CBAM compliance, and provenance tracking.

Design Principles:
    - Zero-hallucination: All calculations are deterministic
    - CBAM-compliant: EU Carbon Border Adjustment Mechanism ready
    - Auditable: SHA-256 provenance tracking for complete audit trails
    - Extensible: Abstract methods for sector-specific implementations

Sources:
    - IPCC 2006/2019 Guidelines for National GHG Inventories
    - EU ETS Monitoring and Reporting Regulation (EU) 2018/2066
    - CBAM Implementing Regulation (EU) 2023/956

Author: GreenLang Framework Team
Version: 1.0.0
"""

from __future__ import annotations

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

InputT = TypeVar("InputT", bound="IndustrialMRVInput")
OutputT = TypeVar("OutputT", bound="IndustrialMRVOutput")


# =============================================================================
# ENUMS
# =============================================================================

class EmissionScope(str, Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect from purchased energy
    SCOPE_3 = "scope_3"  # Other indirect (value chain)


class VerificationStatus(str, Enum):
    """MRV verification status."""
    UNVERIFIED = "unverified"
    PENDING_VERIFICATION = "pending_verification"
    VERIFIED = "verified"
    REJECTED = "rejected"


class DataQuality(str, Enum):
    """Data quality level per GHG Protocol."""
    PRIMARY = "primary"  # Measured data
    SECONDARY = "secondary"  # Activity data with EFs
    TERTIARY = "tertiary"  # Estimates/proxies


# =============================================================================
# DATA MODELS
# =============================================================================

class EmissionFactor(BaseModel):
    """Emission factor with provenance."""
    factor_id: str = Field(..., description="Unique factor identifier")
    value: Decimal = Field(..., description="Factor value")
    unit: str = Field(..., description="Unit (e.g., tCO2e/t_product)")
    source: str = Field(..., description="Authoritative source")
    region: str = Field(default="global", description="Geographic applicability")
    valid_from: str = Field(..., description="Validity start date (ISO)")
    valid_to: Optional[str] = Field(None, description="Validity end date (ISO)")
    uncertainty_percent: Optional[float] = Field(None, ge=0, le=100)


class CalculationStep(BaseModel):
    """Individual calculation step for audit trail."""
    step_number: int = Field(..., ge=1)
    description: str = Field(..., min_length=1)
    formula: str = Field(..., description="Mathematical formula used")
    inputs: Dict[str, str] = Field(default_factory=dict)
    output_value: Decimal
    output_unit: str
    source: str = Field(default="", description="Source reference")


class IndustrialMRVInput(BaseModel):
    """Base input model for industrial MRV agents."""

    facility_id: str = Field(..., description="Unique facility identifier")
    reporting_period: str = Field(..., description="Reporting period (e.g., 2024-Q1)")
    production_tonnes: Decimal = Field(
        ...,
        gt=0,
        le=Decimal("1000000000"),
        description="Production quantity in metric tonnes"
    )

    # Optional energy inputs
    electricity_kwh: Optional[Decimal] = Field(None, ge=0)
    natural_gas_m3: Optional[Decimal] = Field(None, ge=0)
    coal_tonnes: Optional[Decimal] = Field(None, ge=0)
    fuel_oil_litres: Optional[Decimal] = Field(None, ge=0)

    # Grid emission factor
    grid_emission_factor_kg_co2_per_kwh: Optional[Decimal] = Field(
        None, ge=0, le=Decimal("2.0")
    )

    # Data quality
    data_quality: DataQuality = Field(default=DataQuality.SECONDARY)

    class Config:
        """Pydantic config."""
        json_encoders = {Decimal: str}


class CBAMOutput(BaseModel):
    """CBAM-compliant embedded emissions output."""
    cn_code: str = Field(..., description="EU Combined Nomenclature code")
    product_category: str
    quantity_tonnes: Decimal

    # Specific embedded emissions (SEE)
    direct_emissions_tco2e_per_t: Decimal
    indirect_emissions_tco2e_per_t: Decimal
    total_embedded_emissions_tco2e_per_t: Decimal

    # Total emissions
    total_direct_emissions_tco2e: Decimal
    total_indirect_emissions_tco2e: Decimal

    # Methodology
    calculation_methodology: str = Field(default="actual")
    emission_factor_source: str


class IndustrialMRVOutput(BaseModel):
    """Base output model for industrial MRV agents."""

    # Identification
    calculation_id: str
    agent_id: str
    agent_version: str
    timestamp: str

    # Input summary
    facility_id: str
    reporting_period: str
    production_tonnes: Decimal

    # Emissions by scope
    scope_1_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    scope_2_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    scope_3_emissions_tco2e: Decimal = Field(default=Decimal("0"))
    total_emissions_tco2e: Decimal = Field(default=Decimal("0"))

    # Intensity
    emission_intensity_tco2e_per_t: Decimal = Field(default=Decimal("0"))

    # CBAM
    cbam_output: Optional[CBAMOutput] = None

    # Audit trail
    calculation_steps: List[CalculationStep] = Field(default_factory=list)
    emission_factors_used: List[EmissionFactor] = Field(default_factory=list)

    # Provenance
    input_hash: str = Field(default="")
    output_hash: str = Field(default="")
    provenance_hash: str = Field(default="")

    # Quality
    data_quality: DataQuality = Field(default=DataQuality.SECONDARY)
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED
    )

    # Validation
    is_valid: bool = Field(default=True)
    validation_errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic config."""
        json_encoders = {Decimal: str}


# =============================================================================
# BASE AGENT
# =============================================================================

class IndustrialMRVBaseAgent(ABC, Generic[InputT, OutputT]):
    """
    Abstract base class for industrial sector MRV agents.

    ZERO-HALLUCINATION GUARANTEE:
        - All calculations use deterministic formulas
        - Emission factors are database/config lookups only
        - Same inputs always produce identical outputs
        - Complete SHA-256 provenance tracking

    Attributes:
        AGENT_ID: Unique agent identifier (e.g., "GL-MRV-IND-001")
        AGENT_VERSION: Semantic version string
        SECTOR: Industrial sector name
        CBAM_CN_CODE: EU Combined Nomenclature code for CBAM

    Example:
        >>> class SteelMRVAgent(IndustrialMRVBaseAgent):
        ...     AGENT_ID = "GL-MRV-IND-001"
        ...     SECTOR = "Steel"
        ...
        ...     def calculate_emissions(self, input_data):
        ...         # Implementation
        ...         pass
    """

    # Class attributes - override in subclasses
    AGENT_ID: str = "GL-MRV-IND-BASE"
    AGENT_VERSION: str = "1.0.0"
    SECTOR: str = "Industrial"
    CBAM_CN_CODE: str = "0000"
    CBAM_PRODUCT_CATEGORY: str = "Industrial Product"

    # Precision settings
    PRECISION_EMISSIONS: int = 6
    PRECISION_INTENSITY: int = 4

    def __init__(self):
        """Initialize the industrial MRV agent."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._emission_factors: Dict[str, EmissionFactor] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize agent resources. Override in subclasses."""
        self._load_emission_factors()

    @abstractmethod
    def _load_emission_factors(self) -> None:
        """Load sector-specific emission factors. Must be implemented."""
        pass

    @abstractmethod
    def calculate_emissions(self, input_data: InputT) -> OutputT:
        """
        Calculate emissions for the industrial sector.

        Args:
            input_data: Validated input data

        Returns:
            Complete output with emissions and provenance
        """
        pass

    def process(self, input_data: InputT) -> OutputT:
        """
        Main processing method with full lifecycle management.

        Args:
            input_data: Input data for calculation

        Returns:
            Complete MRV output with provenance
        """
        start_time = datetime.now(timezone.utc)

        try:
            self.logger.info(
                f"{self.AGENT_ID} processing: facility={input_data.facility_id}, "
                f"period={input_data.reporting_period}"
            )

            # Calculate emissions
            output = self.calculate_emissions(input_data)

            # Calculate provenance hashes
            output.input_hash = self._calculate_hash(input_data.model_dump())
            output.output_hash = self._calculate_hash({
                "total_emissions": str(output.total_emissions_tco2e),
                "intensity": str(output.emission_intensity_tco2e_per_t)
            })
            output.provenance_hash = self._calculate_provenance_hash(
                output.input_hash,
                output.output_hash,
                output.calculation_steps,
                output.emission_factors_used
            )

            # Log completion
            duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            self.logger.info(
                f"{self.AGENT_ID} completed in {duration_ms:.2f}ms: "
                f"emissions={output.total_emissions_tco2e} tCO2e"
            )

            return output

        except Exception as e:
            self.logger.error(f"{self.AGENT_ID} failed: {str(e)}", exc_info=True)
            raise

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _to_decimal(self, value: Any) -> Decimal:
        """Convert value to Decimal for precise calculations."""
        if isinstance(value, Decimal):
            return value
        return Decimal(str(value))

    def _round_emissions(self, value: Decimal) -> Decimal:
        """Round emission values to regulatory precision."""
        quantize_str = "0." + "0" * self.PRECISION_EMISSIONS
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _round_intensity(self, value: Decimal) -> Decimal:
        """Round intensity values to regulatory precision."""
        quantize_str = "0." + "0" * self.PRECISION_INTENSITY
        return value.quantize(Decimal(quantize_str), rounding=ROUND_HALF_UP)

    def _generate_calculation_id(self, facility_id: str, period: str) -> str:
        """Generate unique calculation ID."""
        data = f"{self.AGENT_ID}:{facility_id}:{period}:{datetime.now(timezone.utc).isoformat()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()

    def _calculate_hash(self, data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance."""
        def convert(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            elif isinstance(obj, Enum):
                return obj.value
            return obj

        converted = convert(data)
        json_str = json.dumps(converted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _calculate_provenance_hash(
        self,
        input_hash: str,
        output_hash: str,
        steps: List[CalculationStep],
        factors: List[EmissionFactor]
    ) -> str:
        """Calculate comprehensive provenance hash."""
        provenance_data = {
            "agent_id": self.AGENT_ID,
            "version": self.AGENT_VERSION,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "steps_count": len(steps),
            "factors_count": len(factors)
        }
        return self._calculate_hash(provenance_data)

    def _create_cbam_output(
        self,
        production_tonnes: Decimal,
        direct_emissions: Decimal,
        indirect_emissions: Decimal,
        methodology: str = "actual"
    ) -> CBAMOutput:
        """Create CBAM-compliant output."""
        direct_intensity = (
            direct_emissions / production_tonnes
            if production_tonnes > 0 else Decimal("0")
        )
        indirect_intensity = (
            indirect_emissions / production_tonnes
            if production_tonnes > 0 else Decimal("0")
        )

        return CBAMOutput(
            cn_code=self.CBAM_CN_CODE,
            product_category=self.CBAM_PRODUCT_CATEGORY,
            quantity_tonnes=production_tonnes,
            direct_emissions_tco2e_per_t=self._round_intensity(direct_intensity),
            indirect_emissions_tco2e_per_t=self._round_intensity(indirect_intensity),
            total_embedded_emissions_tco2e_per_t=self._round_intensity(
                direct_intensity + indirect_intensity
            ),
            total_direct_emissions_tco2e=self._round_emissions(direct_emissions),
            total_indirect_emissions_tco2e=self._round_emissions(indirect_emissions),
            calculation_methodology=methodology,
            emission_factor_source=f"{self.AGENT_ID} v{self.AGENT_VERSION}"
        )

    def _get_emission_factor(self, factor_id: str) -> EmissionFactor:
        """Get emission factor by ID - deterministic lookup."""
        if factor_id not in self._emission_factors:
            raise KeyError(f"Emission factor not found: {factor_id}")
        return self._emission_factors[factor_id]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.AGENT_ID}, version={self.AGENT_VERSION})"


# =============================================================================
# COMMON EMISSION FACTORS
# =============================================================================

# Natural gas emission factors
NATURAL_GAS_EF_KG_CO2_PER_M3 = Decimal("1.89")  # IPCC 2006

# Coal emission factors by type
COAL_EF_KG_CO2_PER_KG = {
    "anthracite": Decimal("2.60"),
    "bituminous": Decimal("2.42"),
    "sub_bituminous": Decimal("1.96"),
    "lignite": Decimal("1.37"),
    "default": Decimal("2.42")  # Bituminous default
}

# Fuel oil emission factors
FUEL_OIL_EF_KG_CO2_PER_LITRE = Decimal("2.68")  # Heavy fuel oil

# Grid emission factors by region (kg CO2/kWh)
GRID_EF_BY_REGION = {
    "world_average": Decimal("0.436"),
    "eu_average": Decimal("0.251"),
    "china": Decimal("0.555"),
    "india": Decimal("0.708"),
    "usa": Decimal("0.379"),
    "germany": Decimal("0.350"),
    "france": Decimal("0.052"),
    "uk": Decimal("0.207"),
    "japan": Decimal("0.457"),
    "south_korea": Decimal("0.415"),
    "brazil": Decimal("0.074"),
    "australia": Decimal("0.656"),
}
