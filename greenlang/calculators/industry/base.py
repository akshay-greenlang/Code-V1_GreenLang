# -*- coding: utf-8 -*-
"""
Industry Calculator Base Classes
================================

Base classes and data structures for all heavy industry emission calculators.
Provides CBAM-compatible output formats, SHA-256 provenance tracking, and
industry-standard emission factor management.

Sources:
- IPCC 2006/2019 Guidelines for National Greenhouse Gas Inventories
- IEA Energy Technology Perspectives
- World Steel Association CO2 Data Collection
- GCCA (Global Cement and Concrete Association) Guidelines
- International Aluminium Institute

Author: GreenLang Framework Team
Version: 1.0.0
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from abc import ABC, abstractmethod
from enum import Enum
import hashlib
import json
import uuid


class AllocationMethod(Enum):
    """Allocation methods for multi-product facilities."""
    MASS = "mass"
    ECONOMIC = "economic"
    ENERGY = "energy"
    PHYSICAL_CAUSATION = "physical_causation"


class EmissionScope(Enum):
    """GHG Protocol emission scopes."""
    SCOPE_1 = "scope_1"  # Direct emissions
    SCOPE_2 = "scope_2"  # Indirect from purchased energy
    SCOPE_3 = "scope_3"  # Other indirect


@dataclass
class EmissionFactor:
    """
    Emission factor with full provenance tracking.

    All factors are deterministic database lookups - NO LLM estimation.
    """
    factor_id: str
    value: Decimal
    unit: str  # e.g., "tCO2e/t_product"
    source: str  # e.g., "IPCC 2019", "World Steel Association"
    version: str  # e.g., "2023.1"
    region: str  # e.g., "global", "EU", "China"
    valid_from: str  # ISO date
    valid_to: Optional[str]  # ISO date or None for current
    uncertainty_percent: Optional[float]  # e.g., 5.0 for +/- 5%
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "factor_id": self.factor_id,
            "value": str(self.value),
            "unit": self.unit,
            "source": self.source,
            "version": self.version,
            "region": self.region,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "uncertainty_percent": self.uncertainty_percent,
            "notes": self.notes,
        }


@dataclass
class CalculationStep:
    """Individual calculation step with complete audit trail."""
    step_number: int
    description: str
    formula: str
    inputs: Dict[str, Any]
    output_value: Decimal
    output_unit: str
    emission_factor_used: Optional[EmissionFactor] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_number": self.step_number,
            "description": self.description,
            "formula": self.formula,
            "inputs": {k: str(v) if isinstance(v, Decimal) else v
                      for k, v in self.inputs.items()},
            "output_value": str(self.output_value),
            "output_unit": self.output_unit,
            "emission_factor": self.emission_factor_used.to_dict()
                              if self.emission_factor_used else None,
        }


@dataclass
class CBAMEmbeddedEmissions:
    """
    CBAM (Carbon Border Adjustment Mechanism) embedded emissions format.

    Compliant with EU Regulation 2023/956 Annex III requirements.
    """
    product_cn_code: str  # EU Combined Nomenclature code
    product_description: str

    # Specific embedded emissions (SEE)
    direct_emissions_t_co2e_per_t: Decimal  # Scope 1
    indirect_emissions_t_co2e_per_t: Decimal  # Scope 2
    total_embedded_emissions_t_co2e_per_t: Decimal

    # Production data
    production_quantity_tonnes: Decimal
    total_direct_emissions_t_co2e: Decimal
    total_indirect_emissions_t_co2e: Decimal

    # Methodology
    calculation_methodology: str  # e.g., "actual", "default", "transitional"
    emission_factor_source: str
    electricity_consumption_mwh_per_t: Optional[Decimal] = None
    grid_emission_factor_t_co2e_per_mwh: Optional[Decimal] = None

    # Precursors (for complex goods)
    precursors: List[Dict[str, Any]] = field(default_factory=list)

    # Verification
    verification_status: str = "unverified"
    verifier_accreditation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to CBAM-compatible dictionary format."""
        return {
            "cn_code": self.product_cn_code,
            "description": self.product_description,
            "specific_embedded_emissions": {
                "direct_tCO2e_per_t": str(self.direct_emissions_t_co2e_per_t),
                "indirect_tCO2e_per_t": str(self.indirect_emissions_t_co2e_per_t),
                "total_tCO2e_per_t": str(self.total_embedded_emissions_t_co2e_per_t),
            },
            "production": {
                "quantity_tonnes": str(self.production_quantity_tonnes),
                "total_direct_emissions_tCO2e": str(self.total_direct_emissions_t_co2e),
                "total_indirect_emissions_tCO2e": str(self.total_indirect_emissions_t_co2e),
            },
            "methodology": {
                "type": self.calculation_methodology,
                "emission_factor_source": self.emission_factor_source,
                "electricity_consumption_mwh_per_t": str(self.electricity_consumption_mwh_per_t)
                    if self.electricity_consumption_mwh_per_t else None,
                "grid_ef_tCO2e_per_mwh": str(self.grid_emission_factor_t_co2e_per_mwh)
                    if self.grid_emission_factor_t_co2e_per_mwh else None,
            },
            "precursors": self.precursors,
            "verification": {
                "status": self.verification_status,
                "verifier": self.verifier_accreditation,
            },
        }


@dataclass
class IndustryCalculationResult:
    """
    Result of industry emission calculation with complete provenance.

    Includes:
    - Total emissions by scope
    - CBAM-compatible embedded emissions
    - SHA-256 provenance hash
    - Complete calculation audit trail
    """
    success: bool
    calculation_id: str
    timestamp: str
    calculator_id: str
    calculator_version: str

    # Production details
    product_type: str
    production_route: str
    production_quantity_tonnes: Decimal

    # Emissions by scope
    scope_1_emissions_t_co2e: Decimal = Decimal("0")
    scope_2_emissions_t_co2e: Decimal = Decimal("0")
    scope_3_emissions_t_co2e: Decimal = Decimal("0")
    total_emissions_t_co2e: Decimal = Decimal("0")

    # Intensity metrics
    emission_intensity_t_co2e_per_t: Decimal = Decimal("0")

    # CBAM output
    cbam_embedded_emissions: Optional[CBAMEmbeddedEmissions] = None

    # Audit trail
    calculation_steps: List[CalculationStep] = field(default_factory=list)
    emission_factors_used: List[EmissionFactor] = field(default_factory=list)

    # Provenance
    input_hash: str = ""
    output_hash: str = ""
    provenance_hash: str = ""

    # Error handling
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization and API responses."""
        return {
            "success": self.success,
            "calculation_id": self.calculation_id,
            "timestamp": self.timestamp,
            "calculator": {
                "id": self.calculator_id,
                "version": self.calculator_version,
            },
            "production": {
                "product_type": self.product_type,
                "route": self.production_route,
                "quantity_tonnes": str(self.production_quantity_tonnes),
            },
            "emissions": {
                "scope_1_tCO2e": str(self.scope_1_emissions_t_co2e),
                "scope_2_tCO2e": str(self.scope_2_emissions_t_co2e),
                "scope_3_tCO2e": str(self.scope_3_emissions_t_co2e),
                "total_tCO2e": str(self.total_emissions_t_co2e),
                "intensity_tCO2e_per_t": str(self.emission_intensity_t_co2e_per_t),
            },
            "cbam": self.cbam_embedded_emissions.to_dict()
                   if self.cbam_embedded_emissions else None,
            "audit_trail": {
                "steps": [s.to_dict() for s in self.calculation_steps],
                "emission_factors": [ef.to_dict() for ef in self.emission_factors_used],
            },
            "provenance": {
                "input_hash": self.input_hash,
                "output_hash": self.output_hash,
                "provenance_hash": self.provenance_hash,
            },
            "error": self.error_message,
            "warnings": self.warnings,
        }


class IndustryBaseCalculator(ABC):
    """
    Abstract base class for heavy industry emission calculators.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations are deterministic
    - Emission factors are database lookups, not LLM outputs
    - Same inputs ALWAYS produce same outputs (bit-perfect)
    - Complete SHA-256 provenance tracking

    All derived calculators must implement:
    - calculate(): Main calculation method
    - _get_emission_factors(): Load factors for the specific industry
    - _get_cbam_cn_code(): Return appropriate CBAM product code
    """

    # Regulatory precision requirements (decimal places)
    PRECISION_EMISSIONS = 6  # tCO2e values
    PRECISION_INTENSITY = 4  # tCO2e/t values
    PRECISION_FACTORS = 6    # Emission factors

    def __init__(self, calculator_id: str, version: str):
        """
        Initialize industry calculator.

        Args:
            calculator_id: Unique calculator identifier
            version: Calculator version string
        """
        self.calculator_id = calculator_id
        self.version = version
        self._emission_factors: Dict[str, EmissionFactor] = {}
        self._load_emission_factors()

    @abstractmethod
    def calculate(
        self,
        production_quantity_tonnes: float,
        production_route: str,
        **kwargs
    ) -> IndustryCalculationResult:
        """
        Execute the emission calculation.

        Args:
            production_quantity_tonnes: Quantity produced in metric tonnes
            production_route: Production method/route
            **kwargs: Additional route-specific parameters

        Returns:
            IndustryCalculationResult with emissions and audit trail
        """
        pass

    @abstractmethod
    def _load_emission_factors(self) -> None:
        """Load emission factors for this industry sector."""
        pass

    @abstractmethod
    def _get_cbam_cn_code(self, product_type: str) -> str:
        """Get CBAM Combined Nomenclature code for product."""
        pass

    def _create_calculation_id(self) -> str:
        """Generate unique calculation ID."""
        return f"{self.calculator_id}-{uuid.uuid4().hex[:12]}"

    def _get_timestamp(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"

    def _to_decimal(self, value: Union[float, int, str, Decimal]) -> Decimal:
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

    def _create_hash(self, data: Dict[str, Any]) -> str:
        """
        Create SHA-256 hash for provenance tracking.

        Ensures deterministic hashing via sorted keys and string conversion.
        """
        def convert_decimals(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimals(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_decimals(i) for i in obj]
            return obj

        converted = convert_decimals(data)
        json_str = json.dumps(converted, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _create_provenance_hash(
        self,
        input_hash: str,
        output_hash: str,
        steps: List[CalculationStep],
        factors: List[EmissionFactor]
    ) -> str:
        """Create comprehensive provenance hash for complete audit trail."""
        provenance_data = {
            "calculator_id": self.calculator_id,
            "version": self.version,
            "input_hash": input_hash,
            "output_hash": output_hash,
            "steps": [s.to_dict() for s in steps],
            "factors": [f.to_dict() for f in factors],
        }
        return self._create_hash(provenance_data)

    def _validate_positive(self, value: float, field_name: str) -> None:
        """Validate that a value is non-negative."""
        if value < 0:
            raise ValueError(f"{field_name} must be non-negative, got {value}")

    def _get_emission_factor(self, factor_id: str) -> EmissionFactor:
        """
        Get emission factor by ID - DETERMINISTIC DATABASE LOOKUP.

        Raises:
            KeyError: If factor not found
        """
        if factor_id not in self._emission_factors:
            raise KeyError(f"Emission factor not found: {factor_id}")
        return self._emission_factors[factor_id]

    def _create_cbam_output(
        self,
        product_type: str,
        production_quantity: Decimal,
        direct_emissions: Decimal,
        indirect_emissions: Decimal,
        calculation_methodology: str = "actual",
        electricity_consumption_mwh_per_t: Optional[Decimal] = None,
        grid_ef: Optional[Decimal] = None,
        precursors: Optional[List[Dict]] = None,
    ) -> CBAMEmbeddedEmissions:
        """Create CBAM-compatible embedded emissions output."""
        direct_intensity = direct_emissions / production_quantity if production_quantity > 0 else Decimal("0")
        indirect_intensity = indirect_emissions / production_quantity if production_quantity > 0 else Decimal("0")
        total_intensity = direct_intensity + indirect_intensity

        return CBAMEmbeddedEmissions(
            product_cn_code=self._get_cbam_cn_code(product_type),
            product_description=product_type,
            direct_emissions_t_co2e_per_t=self._round_intensity(direct_intensity),
            indirect_emissions_t_co2e_per_t=self._round_intensity(indirect_intensity),
            total_embedded_emissions_t_co2e_per_t=self._round_intensity(total_intensity),
            production_quantity_tonnes=production_quantity,
            total_direct_emissions_t_co2e=self._round_emissions(direct_emissions),
            total_indirect_emissions_t_co2e=self._round_emissions(indirect_emissions),
            calculation_methodology=calculation_methodology,
            emission_factor_source=f"{self.calculator_id} v{self.version}",
            electricity_consumption_mwh_per_t=electricity_consumption_mwh_per_t,
            grid_emission_factor_t_co2e_per_mwh=grid_ef,
            precursors=precursors or [],
        )


# Grid emission factors for major steel/aluminum producing regions
# Source: IEA Emission Factors 2023
GRID_EMISSION_FACTORS = {
    # tCO2e per MWh
    "world_average": Decimal("0.436"),
    "eu_average": Decimal("0.251"),
    "china": Decimal("0.555"),
    "india": Decimal("0.708"),
    "usa": Decimal("0.379"),
    "brazil": Decimal("0.074"),
    "russia": Decimal("0.311"),
    "japan": Decimal("0.457"),
    "south_korea": Decimal("0.415"),
    "germany": Decimal("0.350"),
    "france": Decimal("0.052"),
    "uk": Decimal("0.207"),
    "canada": Decimal("0.120"),
    "australia": Decimal("0.656"),
    "south_africa": Decimal("0.928"),
    "mexico": Decimal("0.423"),
    "turkey": Decimal("0.439"),
    "indonesia": Decimal("0.761"),
    "vietnam": Decimal("0.513"),
    "poland": Decimal("0.635"),
    "italy": Decimal("0.315"),
    "spain": Decimal("0.187"),
    "netherlands": Decimal("0.328"),
    "belgium": Decimal("0.145"),
    "austria": Decimal("0.101"),
    "sweden": Decimal("0.013"),
    "norway": Decimal("0.008"),
    "iceland": Decimal("0.000"),  # 100% renewable
}
