"""
GL-001 ThermalCommand - EPA 40 CFR Compliance Mapping

Comprehensive mapping of EPA 40 CFR regulatory requirements to
ThermalCommand agent calculations with full traceability.

Regulatory References:
- EPA 40 CFR Part 60: Standards of Performance for New Stationary Sources
- EPA 40 CFR Part 75: Continuous Emission Monitoring
- EPA 40 CFR Part 98: Mandatory Greenhouse Gas Reporting

This module provides:
1. Regulatory requirement mappings to calculation methods
2. Emission factor databases with EPA citations
3. Compliance validation rules
4. Audit trail documentation generators
5. Reporting period aggregation methods

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import hashlib
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# EPA REGULATION ENUMERATIONS
# =============================================================================

class EPARegulation(Enum):
    """EPA regulatory programs with CFR citations."""

    # Standards of Performance for New Stationary Sources
    CFR_60_DA = "40 CFR 60 Subpart Da"  # Electric Utility Steam Generating Units
    CFR_60_DB = "40 CFR 60 Subpart Db"  # Industrial-Commercial Steam Generating Units
    CFR_60_DC = "40 CFR 60 Subpart Dc"  # Small Industrial Steam Generating Units

    # Continuous Emission Monitoring
    CFR_75 = "40 CFR Part 75"  # CEMS Requirements

    # Greenhouse Gas Reporting
    CFR_98_A = "40 CFR 98 Subpart A"   # General Provisions
    CFR_98_C = "40 CFR 98 Subpart C"   # Stationary Combustion
    CFR_98_D = "40 CFR 98 Subpart D"   # Electricity Generation

    # National Ambient Air Quality Standards
    CFR_50 = "40 CFR Part 50"  # NAAQS

    # Emission Guidelines
    CFR_60_CB = "40 CFR 60 Subpart Cb"  # Emission Guidelines for Large MWC


class PollutantType(Enum):
    """Regulated pollutants with EPA identifiers."""

    NOX = "NOx"           # Nitrogen Oxides
    SO2 = "SO2"           # Sulfur Dioxide
    CO = "CO"             # Carbon Monoxide
    PM = "PM"             # Particulate Matter
    PM10 = "PM10"         # PM 10 microns or less
    PM25 = "PM2.5"        # PM 2.5 microns or less
    CO2 = "CO2"           # Carbon Dioxide
    CH4 = "CH4"           # Methane
    N2O = "N2O"           # Nitrous Oxide
    VOC = "VOC"           # Volatile Organic Compounds
    HAP = "HAP"           # Hazardous Air Pollutants


class FuelCategory(Enum):
    """EPA fuel categories per 40 CFR 98 Table C-1."""

    NATURAL_GAS = "Natural Gas"
    DISTILLATE_FUEL_OIL = "Distillate Fuel Oil No. 1, 2, 4"
    RESIDUAL_FUEL_OIL = "Residual Fuel Oil No. 5, 6"
    COAL_BITUMINOUS = "Coal, Bituminous"
    COAL_SUBBITUMINOUS = "Coal, Sub-bituminous"
    COAL_LIGNITE = "Coal, Lignite"
    COAL_ANTHRACITE = "Coal, Anthracite"
    PETROLEUM_COKE = "Petroleum Coke"
    WOOD_BIOMASS = "Wood and Wood Residuals"
    LANDFILL_GAS = "Landfill Gas"


# =============================================================================
# EPA EMISSION FACTORS DATABASE
# =============================================================================

@dataclass
class EPAEmissionFactor:
    """
    EPA emission factor with full regulatory citation.

    All values are from EPA AP-42, 40 CFR Part 98, or
    other official EPA publications with traceable references.
    """

    pollutant: PollutantType
    fuel_category: FuelCategory
    value: Decimal
    unit: str
    source_document: str
    table_reference: str
    effective_date: str
    uncertainty_percent: Optional[Decimal] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pollutant": self.pollutant.value,
            "fuel_category": self.fuel_category.value,
            "value": str(self.value),
            "unit": self.unit,
            "source_document": self.source_document,
            "table_reference": self.table_reference,
            "effective_date": self.effective_date,
            "uncertainty_percent": str(self.uncertainty_percent) if self.uncertainty_percent else None,
            "notes": self.notes,
        }


# EPA 40 CFR Part 98 Table C-1: Default CO2 Emission Factors
EPA_CO2_FACTORS = {
    FuelCategory.NATURAL_GAS: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.NATURAL_GAS,
        value=Decimal("53.06"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("1.0"),
        notes="Higher heating value basis",
    ),
    FuelCategory.DISTILLATE_FUEL_OIL: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.DISTILLATE_FUEL_OIL,
        value=Decimal("73.96"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("1.0"),
    ),
    FuelCategory.RESIDUAL_FUEL_OIL: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.RESIDUAL_FUEL_OIL,
        value=Decimal("75.10"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("1.0"),
    ),
    FuelCategory.COAL_BITUMINOUS: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.COAL_BITUMINOUS,
        value=Decimal("93.28"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("2.0"),
    ),
    FuelCategory.COAL_SUBBITUMINOUS: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.COAL_SUBBITUMINOUS,
        value=Decimal("97.17"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("2.0"),
    ),
    FuelCategory.COAL_LIGNITE: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.COAL_LIGNITE,
        value=Decimal("97.72"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("2.5"),
    ),
    FuelCategory.WOOD_BIOMASS: EPAEmissionFactor(
        pollutant=PollutantType.CO2,
        fuel_category=FuelCategory.WOOD_BIOMASS,
        value=Decimal("93.80"),
        unit="kg CO2/MMBtu",
        source_document="40 CFR Part 98",
        table_reference="Table C-1",
        effective_date="2024-01-01",
        uncertainty_percent=Decimal("5.0"),
        notes="Biogenic CO2 - may be reported separately",
    ),
}

# EPA 40 CFR Part 98 Table C-2: Default CH4 and N2O Emission Factors
EPA_CH4_N2O_FACTORS = {
    FuelCategory.NATURAL_GAS: {
        PollutantType.CH4: EPAEmissionFactor(
            pollutant=PollutantType.CH4,
            fuel_category=FuelCategory.NATURAL_GAS,
            value=Decimal("0.001"),
            unit="kg CH4/MMBtu",
            source_document="40 CFR Part 98",
            table_reference="Table C-2",
            effective_date="2024-01-01",
        ),
        PollutantType.N2O: EPAEmissionFactor(
            pollutant=PollutantType.N2O,
            fuel_category=FuelCategory.NATURAL_GAS,
            value=Decimal("0.0001"),
            unit="kg N2O/MMBtu",
            source_document="40 CFR Part 98",
            table_reference="Table C-2",
            effective_date="2024-01-01",
        ),
    },
    FuelCategory.DISTILLATE_FUEL_OIL: {
        PollutantType.CH4: EPAEmissionFactor(
            pollutant=PollutantType.CH4,
            fuel_category=FuelCategory.DISTILLATE_FUEL_OIL,
            value=Decimal("0.003"),
            unit="kg CH4/MMBtu",
            source_document="40 CFR Part 98",
            table_reference="Table C-2",
            effective_date="2024-01-01",
        ),
        PollutantType.N2O: EPAEmissionFactor(
            pollutant=PollutantType.N2O,
            fuel_category=FuelCategory.DISTILLATE_FUEL_OIL,
            value=Decimal("0.0006"),
            unit="kg N2O/MMBtu",
            source_document="40 CFR Part 98",
            table_reference="Table C-2",
            effective_date="2024-01-01",
        ),
    },
    FuelCategory.COAL_BITUMINOUS: {
        PollutantType.CH4: EPAEmissionFactor(
            pollutant=PollutantType.CH4,
            fuel_category=FuelCategory.COAL_BITUMINOUS,
            value=Decimal("0.011"),
            unit="kg CH4/MMBtu",
            source_document="40 CFR Part 98",
            table_reference="Table C-2",
            effective_date="2024-01-01",
        ),
        PollutantType.N2O: EPAEmissionFactor(
            pollutant=PollutantType.N2O,
            fuel_category=FuelCategory.COAL_BITUMINOUS,
            value=Decimal("0.0016"),
            unit="kg N2O/MMBtu",
            source_document="40 CFR Part 98",
            table_reference="Table C-2",
            effective_date="2024-01-01",
        ),
    },
}

# Global Warming Potentials (100-year) per 40 CFR 98.A
# Note: EPA uses IPCC AR4 values for current reporting
EPA_GWP_VALUES = {
    PollutantType.CO2: Decimal("1"),
    PollutantType.CH4: Decimal("25"),    # AR4 value per 40 CFR 98.A
    PollutantType.N2O: Decimal("298"),   # AR4 value per 40 CFR 98.A
}


# =============================================================================
# EPA COMPLIANCE REQUIREMENTS
# =============================================================================

@dataclass
class ComplianceRequirement:
    """
    Single EPA compliance requirement mapped to calculation method.

    Provides traceability from regulatory requirement to
    the specific calculation method that satisfies it.
    """

    requirement_id: str
    regulation: EPARegulation
    cfr_section: str
    description: str
    pollutants: List[PollutantType]
    calculation_method: str
    reporting_frequency: str
    threshold_value: Optional[Decimal] = None
    threshold_unit: Optional[str] = None
    verification_procedure: Optional[str] = None
    record_retention_years: int = 5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requirement_id": self.requirement_id,
            "regulation": self.regulation.value,
            "cfr_section": self.cfr_section,
            "description": self.description,
            "pollutants": [p.value for p in self.pollutants],
            "calculation_method": self.calculation_method,
            "reporting_frequency": self.reporting_frequency,
            "threshold_value": str(self.threshold_value) if self.threshold_value else None,
            "threshold_unit": self.threshold_unit,
            "verification_procedure": self.verification_procedure,
            "record_retention_years": self.record_retention_years,
        }


# Comprehensive EPA compliance requirements mapping
EPA_COMPLIANCE_REQUIREMENTS = [
    # 40 CFR Part 98 - GHG Reporting Requirements
    ComplianceRequirement(
        requirement_id="GHG-001",
        regulation=EPARegulation.CFR_98_C,
        cfr_section="98.33(a)(1)",
        description="Calculate CO2 emissions from stationary fuel combustion using Tier 1 (fuel-based)",
        pollutants=[PollutantType.CO2],
        calculation_method="TIER_1_FUEL_ANALYSIS",
        reporting_frequency="Annual",
        threshold_value=Decimal("25000"),
        threshold_unit="metric tons CO2e",
        verification_procedure="Third-party verification for facilities >25,000 MTCO2e",
        record_retention_years=5,
    ),
    ComplianceRequirement(
        requirement_id="GHG-002",
        regulation=EPARegulation.CFR_98_C,
        cfr_section="98.33(a)(2)",
        description="Calculate CO2 emissions using Tier 2 (fuel sampling and analysis)",
        pollutants=[PollutantType.CO2],
        calculation_method="TIER_2_FUEL_SAMPLING",
        reporting_frequency="Annual",
        verification_procedure="Weekly fuel sampling per 98.34",
        record_retention_years=5,
    ),
    ComplianceRequirement(
        requirement_id="GHG-003",
        regulation=EPARegulation.CFR_98_C,
        cfr_section="98.33(a)(3)",
        description="Calculate CO2 emissions using Tier 3 (carbon content)",
        pollutants=[PollutantType.CO2],
        calculation_method="TIER_3_CARBON_CONTENT",
        reporting_frequency="Annual",
        verification_procedure="Monthly fuel sampling and analysis",
        record_retention_years=5,
    ),
    ComplianceRequirement(
        requirement_id="GHG-004",
        regulation=EPARegulation.CFR_98_C,
        cfr_section="98.33(a)(4)",
        description="Calculate CO2 emissions using Tier 4 (CEMS)",
        pollutants=[PollutantType.CO2],
        calculation_method="TIER_4_CEMS",
        reporting_frequency="Annual",
        verification_procedure="Continuous monitoring per 40 CFR Part 75",
        record_retention_years=5,
    ),
    ComplianceRequirement(
        requirement_id="GHG-005",
        regulation=EPARegulation.CFR_98_C,
        cfr_section="98.33(c)",
        description="Calculate CH4 emissions from stationary combustion",
        pollutants=[PollutantType.CH4],
        calculation_method="CH4_EMISSION_FACTOR",
        reporting_frequency="Annual",
        record_retention_years=5,
    ),
    ComplianceRequirement(
        requirement_id="GHG-006",
        regulation=EPARegulation.CFR_98_C,
        cfr_section="98.33(c)",
        description="Calculate N2O emissions from stationary combustion",
        pollutants=[PollutantType.N2O],
        calculation_method="N2O_EMISSION_FACTOR",
        reporting_frequency="Annual",
        record_retention_years=5,
    ),

    # 40 CFR Part 75 - CEMS Requirements
    ComplianceRequirement(
        requirement_id="CEMS-001",
        regulation=EPARegulation.CFR_75,
        cfr_section="75.10",
        description="Continuous monitoring of SO2 and NOx emission rates",
        pollutants=[PollutantType.SO2, PollutantType.NOX],
        calculation_method="CEMS_HOURLY_EMISSIONS",
        reporting_frequency="Hourly",
        verification_procedure="Daily calibration and QA/QC per Appendix B",
        record_retention_years=3,
    ),
    ComplianceRequirement(
        requirement_id="CEMS-002",
        regulation=EPARegulation.CFR_75,
        cfr_section="75.11",
        description="Continuous monitoring of CO2 emissions for Acid Rain Program",
        pollutants=[PollutantType.CO2],
        calculation_method="CEMS_CO2_MONITORING",
        reporting_frequency="Hourly",
        verification_procedure="RATA and bias adjustment per Appendix A",
        record_retention_years=3,
    ),

    # 40 CFR Part 60 - NSPS for Boilers
    ComplianceRequirement(
        requirement_id="NSPS-001",
        regulation=EPARegulation.CFR_60_DB,
        cfr_section="60.42b",
        description="SO2 emission limits for industrial steam generating units",
        pollutants=[PollutantType.SO2],
        calculation_method="SO2_MASS_BALANCE",
        reporting_frequency="Monthly",
        threshold_value=Decimal("0.50"),
        threshold_unit="lb/MMBtu",
        record_retention_years=2,
    ),
    ComplianceRequirement(
        requirement_id="NSPS-002",
        regulation=EPARegulation.CFR_60_DB,
        cfr_section="60.43b",
        description="PM emission limits for industrial steam generating units",
        pollutants=[PollutantType.PM],
        calculation_method="PM_STACK_TEST",
        reporting_frequency="Annual",
        threshold_value=Decimal("0.05"),
        threshold_unit="lb/MMBtu",
        record_retention_years=2,
    ),
    ComplianceRequirement(
        requirement_id="NSPS-003",
        regulation=EPARegulation.CFR_60_DB,
        cfr_section="60.44b",
        description="NOx emission limits for industrial steam generating units",
        pollutants=[PollutantType.NOX],
        calculation_method="NOX_CEMS_OR_STACK_TEST",
        reporting_frequency="Monthly",
        threshold_value=Decimal("0.30"),
        threshold_unit="lb/MMBtu",
        record_retention_years=2,
    ),
]


# =============================================================================
# EPA COMPLIANCE MAPPER
# =============================================================================

@dataclass
class ComplianceValidationResult:
    """Result of compliance validation check."""

    requirement_id: str
    is_compliant: bool
    measured_value: Decimal
    limit_value: Decimal
    unit: str
    margin_percent: Decimal
    calculation_method: str
    validation_timestamp: datetime
    provenance_hash: str
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "requirement_id": self.requirement_id,
            "is_compliant": self.is_compliant,
            "measured_value": str(self.measured_value),
            "limit_value": str(self.limit_value),
            "unit": self.unit,
            "margin_percent": str(self.margin_percent),
            "calculation_method": self.calculation_method,
            "validation_timestamp": self.validation_timestamp.isoformat(),
            "provenance_hash": self.provenance_hash,
            "notes": self.notes,
        }


class EPAComplianceMapper:
    """
    Maps EPA 40 CFR regulatory requirements to ThermalCommand calculations.

    Provides:
    - Requirement lookup by regulation, pollutant, or calculation type
    - Emission factor retrieval with full EPA citations
    - Compliance validation against emission limits
    - Audit documentation generation

    Example:
        >>> mapper = EPAComplianceMapper()
        >>> reqs = mapper.get_requirements_by_pollutant(PollutantType.CO2)
        >>> for req in reqs:
        ...     print(f"{req.requirement_id}: {req.description}")
    """

    VERSION = "1.0.0"

    def __init__(self) -> None:
        """Initialize EPA compliance mapper."""
        self._requirements = EPA_COMPLIANCE_REQUIREMENTS
        self._co2_factors = EPA_CO2_FACTORS
        self._ch4_n2o_factors = EPA_CH4_N2O_FACTORS
        self._gwp_values = EPA_GWP_VALUES

        logger.info(
            f"EPAComplianceMapper initialized with {len(self._requirements)} requirements"
        )

    def get_requirement(self, requirement_id: str) -> Optional[ComplianceRequirement]:
        """
        Get compliance requirement by ID.

        Args:
            requirement_id: Unique requirement identifier (e.g., "GHG-001")

        Returns:
            ComplianceRequirement or None if not found
        """
        for req in self._requirements:
            if req.requirement_id == requirement_id:
                return req
        return None

    def get_requirements_by_regulation(
        self,
        regulation: EPARegulation,
    ) -> List[ComplianceRequirement]:
        """
        Get all requirements for a specific EPA regulation.

        Args:
            regulation: EPA regulation enum value

        Returns:
            List of matching ComplianceRequirements
        """
        return [
            req for req in self._requirements
            if req.regulation == regulation
        ]

    def get_requirements_by_pollutant(
        self,
        pollutant: PollutantType,
    ) -> List[ComplianceRequirement]:
        """
        Get all requirements for a specific pollutant.

        Args:
            pollutant: Pollutant type enum value

        Returns:
            List of matching ComplianceRequirements
        """
        return [
            req for req in self._requirements
            if pollutant in req.pollutants
        ]

    def get_requirements_by_calculation_method(
        self,
        method: str,
    ) -> List[ComplianceRequirement]:
        """
        Get all requirements using a specific calculation method.

        Args:
            method: Calculation method name

        Returns:
            List of matching ComplianceRequirements
        """
        return [
            req for req in self._requirements
            if req.calculation_method == method
        ]

    def get_co2_emission_factor(
        self,
        fuel_category: FuelCategory,
    ) -> Optional[EPAEmissionFactor]:
        """
        Get EPA CO2 emission factor for fuel category.

        Args:
            fuel_category: Fuel category enum value

        Returns:
            EPAEmissionFactor with full citation or None
        """
        return self._co2_factors.get(fuel_category)

    def get_ch4_emission_factor(
        self,
        fuel_category: FuelCategory,
    ) -> Optional[EPAEmissionFactor]:
        """
        Get EPA CH4 emission factor for fuel category.

        Args:
            fuel_category: Fuel category enum value

        Returns:
            EPAEmissionFactor with full citation or None
        """
        factors = self._ch4_n2o_factors.get(fuel_category, {})
        return factors.get(PollutantType.CH4)

    def get_n2o_emission_factor(
        self,
        fuel_category: FuelCategory,
    ) -> Optional[EPAEmissionFactor]:
        """
        Get EPA N2O emission factor for fuel category.

        Args:
            fuel_category: Fuel category enum value

        Returns:
            EPAEmissionFactor with full citation or None
        """
        factors = self._ch4_n2o_factors.get(fuel_category, {})
        return factors.get(PollutantType.N2O)

    def get_gwp(self, pollutant: PollutantType) -> Decimal:
        """
        Get Global Warming Potential for pollutant per 40 CFR 98.A.

        Args:
            pollutant: Pollutant type enum value

        Returns:
            GWP value as Decimal
        """
        return self._gwp_values.get(pollutant, Decimal("1"))

    def validate_compliance(
        self,
        requirement_id: str,
        measured_value: Union[Decimal, float],
        calculation_method: Optional[str] = None,
    ) -> ComplianceValidationResult:
        """
        Validate measured value against EPA requirement limit.

        Args:
            requirement_id: Requirement to validate against
            measured_value: Measured or calculated value
            calculation_method: Method used (for verification)

        Returns:
            ComplianceValidationResult with compliance status

        Raises:
            ValueError: If requirement not found or has no threshold
        """
        req = self.get_requirement(requirement_id)
        if req is None:
            raise ValueError(f"Unknown requirement: {requirement_id}")

        if req.threshold_value is None:
            raise ValueError(f"Requirement {requirement_id} has no threshold value")

        measured_dec = Decimal(str(measured_value))
        limit = req.threshold_value

        is_compliant = measured_dec <= limit

        if limit != Decimal("0"):
            margin = ((limit - measured_dec) / limit) * Decimal("100")
        else:
            margin = Decimal("0")

        # Compute provenance hash
        provenance_data = {
            "requirement_id": requirement_id,
            "measured_value": str(measured_dec),
            "limit_value": str(limit),
            "is_compliant": is_compliant,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(provenance_data, sort_keys=True).encode()
        ).hexdigest()[:16]

        return ComplianceValidationResult(
            requirement_id=requirement_id,
            is_compliant=is_compliant,
            measured_value=measured_dec,
            limit_value=limit,
            unit=req.threshold_unit or "",
            margin_percent=margin.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP),
            calculation_method=calculation_method or req.calculation_method,
            validation_timestamp=datetime.now(timezone.utc),
            provenance_hash=provenance_hash,
        )

    def generate_compliance_report(
        self,
        facility_id: str,
        reporting_period: str,
        emissions_data: Dict[str, Decimal],
    ) -> Dict[str, Any]:
        """
        Generate EPA compliance report with all required documentation.

        Args:
            facility_id: Facility identifier
            reporting_period: Reporting period (e.g., "2024")
            emissions_data: Dictionary of pollutant to emission value

        Returns:
            Complete compliance report as dictionary
        """
        report_timestamp = datetime.now(timezone.utc)

        validations = []
        overall_compliant = True

        # Validate each applicable requirement
        for req in self._requirements:
            for pollutant in req.pollutants:
                pollutant_key = pollutant.value.lower()
                if pollutant_key in emissions_data and req.threshold_value:
                    try:
                        result = self.validate_compliance(
                            req.requirement_id,
                            emissions_data[pollutant_key],
                        )
                        validations.append(result.to_dict())
                        if not result.is_compliant:
                            overall_compliant = False
                    except ValueError as e:
                        logger.warning(f"Validation error: {e}")

        # Generate report hash for provenance
        report_data = {
            "facility_id": facility_id,
            "reporting_period": reporting_period,
            "emissions_data": {k: str(v) for k, v in emissions_data.items()},
            "validations": validations,
        }
        report_hash = hashlib.sha256(
            json.dumps(report_data, sort_keys=True).encode()
        ).hexdigest()

        return {
            "report_metadata": {
                "facility_id": facility_id,
                "reporting_period": reporting_period,
                "generated_at": report_timestamp.isoformat(),
                "report_hash": report_hash,
                "generator_version": self.VERSION,
            },
            "emissions_summary": {
                k: str(v) for k, v in emissions_data.items()
            },
            "compliance_validations": validations,
            "overall_compliant": overall_compliant,
            "applicable_regulations": [
                req.regulation.value for req in self._requirements
            ],
            "record_retention": {
                "minimum_years": 5,
                "requirement": "40 CFR 98.3(g)",
            },
        }


# =============================================================================
# CALCULATION METHOD MAPPING
# =============================================================================

@dataclass
class CalculationMethodMapping:
    """
    Maps EPA calculation tiers to ThermalCommand implementation methods.

    Provides traceability from regulatory calculation requirement
    to the specific code implementation that performs it.
    """

    method_id: str
    epa_tier: str
    cfr_reference: str
    description: str
    implementation_class: str
    implementation_method: str
    input_parameters: List[str]
    output_parameters: List[str]
    validation_rules: List[str]
    uncertainty_method: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method_id": self.method_id,
            "epa_tier": self.epa_tier,
            "cfr_reference": self.cfr_reference,
            "description": self.description,
            "implementation_class": self.implementation_class,
            "implementation_method": self.implementation_method,
            "input_parameters": self.input_parameters,
            "output_parameters": self.output_parameters,
            "validation_rules": self.validation_rules,
            "uncertainty_method": self.uncertainty_method,
        }


# Calculation method mappings for ThermalCommand
CALCULATION_METHOD_MAPPINGS = [
    CalculationMethodMapping(
        method_id="TIER_1_FUEL_ANALYSIS",
        epa_tier="Tier 1",
        cfr_reference="40 CFR 98.33(a)(1)",
        description="Calculate CO2 using default emission factors and fuel consumption",
        implementation_class="PrecisionEmissionsCalculator",
        implementation_method="calculate_ghg_emissions",
        input_parameters=["fuel_type", "heat_input_mmbtu"],
        output_parameters=["co2_kg", "ch4_kg", "n2o_kg", "co2e_kg"],
        validation_rules=[
            "fuel_type must be in EPA_EMISSION_FACTORS",
            "heat_input_mmbtu must be >= 0",
            "Result must include provenance hash",
        ],
        uncertainty_method="EPA default uncertainty per fuel type",
    ),
    CalculationMethodMapping(
        method_id="TIER_2_FUEL_SAMPLING",
        epa_tier="Tier 2",
        cfr_reference="40 CFR 98.33(a)(2)",
        description="Calculate CO2 using measured high heat value",
        implementation_class="PrecisionEmissionsCalculator",
        implementation_method="calculate_with_measured_hhv",
        input_parameters=["fuel_type", "fuel_quantity", "measured_hhv"],
        output_parameters=["co2_kg", "ch4_kg", "n2o_kg", "co2e_kg"],
        validation_rules=[
            "measured_hhv must have valid lab certification",
            "Sampling frequency per 40 CFR 98.34",
        ],
        uncertainty_method="Fuel-specific uncertainty from lab analysis",
    ),
    CalculationMethodMapping(
        method_id="CEMS_HOURLY_EMISSIONS",
        epa_tier="Tier 4 / CEMS",
        cfr_reference="40 CFR Part 75",
        description="Calculate emissions from continuous emission monitoring data",
        implementation_class="CEMSEmissionsCalculator",
        implementation_method="calculate_hourly_emissions",
        input_parameters=["concentration_ppm", "flow_rate_scfh", "temperature_f", "pressure_psia"],
        output_parameters=["mass_rate_lb_hr", "heat_rate_lb_mmbtu"],
        validation_rules=[
            "CEMS data must meet Part 75 QA requirements",
            "Data substitution per 40 CFR 75 Subpart D",
        ],
        uncertainty_method="CEMS RATA and calibration drift",
    ),
]


def get_calculation_method(method_id: str) -> Optional[CalculationMethodMapping]:
    """
    Get calculation method mapping by ID.

    Args:
        method_id: Method identifier

    Returns:
        CalculationMethodMapping or None
    """
    for method in CALCULATION_METHOD_MAPPINGS:
        if method.method_id == method_id:
            return method
    return None


def get_all_calculation_methods() -> List[CalculationMethodMapping]:
    """Get all calculation method mappings."""
    return CALCULATION_METHOD_MAPPINGS.copy()
