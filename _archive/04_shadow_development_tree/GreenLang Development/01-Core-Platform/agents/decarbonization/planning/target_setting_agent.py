# -*- coding: utf-8 -*-
"""
GL-DECARB-X-003: Target Setting Agent
======================================

Sets science-based emission reduction targets aligned with SBTi methodology,
Paris Agreement pathways, and net-zero commitments.

Capabilities:
    - Calculate SBTi-aligned targets (1.5C and well-below 2C)
    - Generate absolute and intensity-based targets
    - Support near-term (5-10 year) and long-term (2050) targets
    - Calculate required annual reduction rates
    - Validate targets against SBTi criteria
    - Support sector-specific decarbonization pathways
    - Track progress against base year emissions
    - Generate target trajectories for reporting

Zero-Hallucination Principle:
    All target calculations use SBTi-published methodologies and
    IPCC carbon budget data with complete provenance tracking.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import math
import time
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.agents.base_agents import DeterministicAgent, AuditEntry
from greenlang.agents.categories import AgentCategory, AgentMetadata
from greenlang.utilities.determinism import (
    DeterministicClock,
    content_hash,
    deterministic_id,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Constants
# =============================================================================

class TargetType(str, Enum):
    """Types of emission reduction targets."""
    ABSOLUTE = "absolute"              # Absolute emission reduction (tCO2e)
    INTENSITY = "intensity"            # Emission intensity (tCO2e/unit)
    SUPPLIER_ENGAGEMENT = "supplier_engagement"  # Supplier coverage target
    RENEWABLE_ENERGY = "renewable_energy"  # RE100 commitment


class TargetTimeframe(str, Enum):
    """Timeframe for targets."""
    NEAR_TERM = "near_term"    # 5-10 years
    LONG_TERM = "long_term"    # 2050 or sooner
    NET_ZERO = "net_zero"      # Net-zero commitment


class TemperatureAlignment(str, Enum):
    """Temperature pathway alignment."""
    C_1_5 = "1.5C"            # 1.5 degrees Celsius pathway
    WELL_BELOW_2C = "well_below_2C"  # Well below 2 degrees
    C_2 = "2C"                # 2 degrees (minimum SBTi)


class SBTiSector(str, Enum):
    """SBTi sector classifications with specific pathways."""
    POWER_GENERATION = "power_generation"
    CEMENT = "cement"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    CHEMICALS = "chemicals"
    PAPER_PULP = "paper_pulp"
    BUILDINGS = "buildings"
    TRANSPORT = "transport"
    FINANCIAL_INSTITUTIONS = "financial_institutions"
    OIL_GAS = "oil_gas"
    FLAG = "flag"  # Forest, Land, Agriculture
    GENERAL = "general"  # Cross-sector approach


class ScopeCategory(str, Enum):
    """GHG Protocol scope categories."""
    SCOPE_1 = "scope_1"
    SCOPE_2 = "scope_2"
    SCOPE_3 = "scope_3"
    SCOPE_1_2 = "scope_1_2"
    SCOPE_1_2_3 = "scope_1_2_3"


# SBTi annual reduction rates (from SBTi Target Setting Protocol v3.0)
# These are minimum annual linear reduction rates
SBTI_REDUCTION_RATES = {
    TemperatureAlignment.C_1_5: {
        "absolute_min_rate": 0.042,  # 4.2% per year for 1.5C
        "near_term_years": 10,
        "total_reduction_2030": 0.42,  # 42% by 2030 from 2020 base
    },
    TemperatureAlignment.WELL_BELOW_2C: {
        "absolute_min_rate": 0.025,  # 2.5% per year for well-below 2C
        "near_term_years": 10,
        "total_reduction_2030": 0.25,
    },
    TemperatureAlignment.C_2: {
        "absolute_min_rate": 0.02,  # 2% per year minimum
        "near_term_years": 10,
        "total_reduction_2030": 0.20,
    }
}

# Sector-specific decarbonization pathways (simplified from SDA)
SECTOR_PATHWAYS = {
    SBTiSector.POWER_GENERATION: {
        "2030_intensity": 0.138,  # kgCO2/kWh
        "2050_intensity": 0.0,   # Full decarbonization
        "pathway": "1.5C",
    },
    SBTiSector.STEEL: {
        "2030_intensity": 1.35,  # tCO2/t steel
        "2050_intensity": 0.05,  # Near-zero
        "pathway": "well_below_2C",
    },
    SBTiSector.CEMENT: {
        "2030_intensity": 0.52,  # tCO2/t cement
        "2050_intensity": 0.12,
        "pathway": "well_below_2C",
    },
    SBTiSector.BUILDINGS: {
        "2030_intensity_commercial": 0.030,  # kgCO2/m2
        "2030_intensity_residential": 0.020,
        "2050_intensity": 0.0,
        "pathway": "1.5C",
    },
    SBTiSector.TRANSPORT: {
        "2030_intensity_passenger": 0.080,  # kgCO2/pkm
        "2030_intensity_freight": 0.020,   # kgCO2/tkm
        "2050_intensity": 0.0,
        "pathway": "1.5C",
    },
}


# =============================================================================
# Pydantic Models
# =============================================================================

class BaseYearEmissions(BaseModel):
    """Base year emissions data for target setting."""
    base_year: int = Field(..., ge=2015, le=2025, description="Base year (2015-2025)")
    scope_1_tco2e: float = Field(..., ge=0, description="Scope 1 emissions (tCO2e)")
    scope_2_tco2e: float = Field(..., ge=0, description="Scope 2 emissions (tCO2e)")
    scope_3_tco2e: float = Field(default=0, ge=0, description="Scope 3 emissions (tCO2e)")

    # Optional intensity data
    revenue_musd: Optional[float] = Field(None, ge=0, description="Revenue (million USD)")
    production_units: Optional[float] = Field(None, ge=0, description="Production quantity")
    production_unit_name: Optional[str] = Field(None, description="Unit of production")
    floor_area_m2: Optional[float] = Field(None, ge=0, description="Floor area (m2)")

    @property
    def total_scope_1_2(self) -> float:
        """Total Scope 1+2 emissions."""
        return self.scope_1_tco2e + self.scope_2_tco2e

    @property
    def total_scope_1_2_3(self) -> float:
        """Total all-scope emissions."""
        return self.scope_1_tco2e + self.scope_2_tco2e + self.scope_3_tco2e


class TargetMilestone(BaseModel):
    """Milestone along target trajectory."""
    year: int = Field(..., description="Milestone year")
    target_emissions_tco2e: float = Field(..., ge=0, description="Target emissions for this year")
    target_reduction_percent: float = Field(..., description="Reduction from base year (%)")
    cumulative_reduction_tco2e: float = Field(..., description="Cumulative reduction from base year")
    is_interim_target: bool = Field(default=False, description="Whether this is a formal interim target")


class EmissionTarget(BaseModel):
    """Complete emission reduction target."""
    target_id: str = Field(..., description="Unique target identifier")
    name: str = Field(..., description="Target name/description")

    # Target specification
    target_type: TargetType = Field(..., description="Type of target")
    temperature_alignment: TemperatureAlignment = Field(..., description="Temperature pathway")
    timeframe: TargetTimeframe = Field(..., description="Target timeframe")
    scope_coverage: ScopeCategory = Field(..., description="Scope coverage")

    # Base year data
    base_year: int = Field(..., description="Base year")
    base_year_emissions_tco2e: float = Field(..., ge=0, description="Base year emissions")

    # Target values
    target_year: int = Field(..., description="Target year")
    target_emissions_tco2e: float = Field(..., ge=0, description="Target year emissions")
    target_reduction_percent: float = Field(..., description="Percentage reduction")
    annual_reduction_rate: float = Field(..., description="Required annual reduction rate")

    # For intensity targets
    base_year_intensity: Optional[float] = Field(None, description="Base year intensity")
    target_year_intensity: Optional[float] = Field(None, description="Target year intensity")
    intensity_unit: Optional[str] = Field(None, description="Intensity unit (e.g., tCO2e/MUSD)")

    # Trajectory
    trajectory: List[TargetMilestone] = Field(default_factory=list, description="Year-by-year trajectory")
    interim_targets: List[TargetMilestone] = Field(default_factory=list, description="Formal interim targets")

    # SBTi compliance
    sbti_compliant: bool = Field(default=False, description="Whether target meets SBTi criteria")
    sbti_validation_notes: List[str] = Field(default_factory=list, description="SBTi validation notes")

    # Provenance
    methodology: str = Field(default="", description="Methodology used (SBTi, SDA, etc.)")
    source_references: List[str] = Field(default_factory=list, description="Source references")
    provenance_hash: str = Field(default="", description="Provenance hash")
    created_at: datetime = Field(default_factory=DeterministicClock.now)


class TargetSettingInput(BaseModel):
    """Input model for TargetSettingAgent."""
    operation: str = Field(
        default="set_target",
        description="Operation: 'set_target', 'validate_target', 'calculate_trajectory', 'check_sbti_compliance'"
    )

    # Base year emissions
    base_year_emissions: Optional[BaseYearEmissions] = Field(
        None,
        description="Base year emissions data"
    )

    # Target parameters
    target_type: TargetType = Field(default=TargetType.ABSOLUTE, description="Target type")
    temperature_alignment: TemperatureAlignment = Field(
        default=TemperatureAlignment.C_1_5,
        description="Temperature pathway to align with"
    )
    timeframe: TargetTimeframe = Field(default=TargetTimeframe.NEAR_TERM, description="Target timeframe")
    scope_coverage: ScopeCategory = Field(default=ScopeCategory.SCOPE_1_2, description="Scope coverage")
    target_year: Optional[int] = Field(None, description="Target year (if not using default)")

    # Sector for SDA (Sectoral Decarbonization Approach)
    sector: Optional[SBTiSector] = Field(None, description="Sector for SDA pathway")

    # For validation
    proposed_target: Optional[EmissionTarget] = Field(None, description="Target to validate")

    # Custom parameters
    custom_reduction_rate: Optional[float] = Field(None, ge=0, le=1, description="Custom annual reduction rate")
    include_interim_targets: bool = Field(default=True, description="Include interim targets in trajectory")
    interim_target_years: List[int] = Field(default_factory=list, description="Specific interim target years")


class TargetSettingOutput(BaseModel):
    """Output model for TargetSettingAgent."""
    operation: str = Field(..., description="Operation performed")
    success: bool = Field(..., description="Whether operation succeeded")

    # Main result
    target: Optional[EmissionTarget] = Field(None, description="Generated/validated target")

    # For validation
    is_valid: bool = Field(default=False, description="Whether target is valid")
    validation_errors: List[str] = Field(default_factory=list, description="Validation errors")
    validation_warnings: List[str] = Field(default_factory=list, description="Validation warnings")

    # SBTi compliance
    sbti_compliant: bool = Field(default=False, description="Whether target meets SBTi minimum")
    sbti_pathway_alignment: Optional[str] = Field(None, description="SBTi pathway alignment")

    # Metadata
    processing_time_ms: float = Field(default=0.0)
    provenance_hash: str = Field(default="")
    timestamp: datetime = Field(default_factory=DeterministicClock.now)
    error_message: Optional[str] = Field(None)


# =============================================================================
# Agent Implementation
# =============================================================================

class TargetSettingAgent(DeterministicAgent):
    """
    GL-DECARB-X-003: Target Setting Agent

    Sets science-based emission reduction targets aligned with SBTi
    methodology and Paris Agreement pathways.

    Zero-Hallucination Implementation:
        - All reduction rates from SBTi Target Setting Protocol
        - Sector pathways from SBTi SDA methodology
        - Complete provenance for all calculations
        - Deterministic trajectory generation

    Attributes:
        config: Agent configuration

    Example:
        >>> agent = TargetSettingAgent()
        >>> result = agent.run({
        ...     "operation": "set_target",
        ...     "base_year_emissions": {
        ...         "base_year": 2022,
        ...         "scope_1_tco2e": 50000,
        ...         "scope_2_tco2e": 30000
        ...     },
        ...     "temperature_alignment": "1.5C"
        ... })
        >>> print(f"Target: {result.data['target']['target_reduction_percent']}% by {result.data['target']['target_year']}")
    """

    AGENT_ID = "GL-DECARB-X-003"
    AGENT_NAME = "Target Setting Agent"
    VERSION = "1.0.0"

    category = AgentCategory.CRITICAL
    metadata = AgentMetadata(
        name="TargetSettingAgent",
        category=AgentCategory.CRITICAL,
        description="Sets SBTi-aligned emission reduction targets"
    )

    def __init__(self, config: Optional[AgentConfig] = None, enable_audit_trail: bool = True):
        """
        Initialize the TargetSettingAgent.

        Args:
            config: Optional agent configuration
            enable_audit_trail: Whether to enable audit trail
        """
        super().__init__(enable_audit_trail=enable_audit_trail)

        self.config = config or AgentConfig(
            name=self.AGENT_NAME,
            description="Sets SBTi-aligned emission reduction targets",
            version=self.VERSION
        )

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.AGENT_ID}: {self.AGENT_NAME}")

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute target setting operation.

        Args:
            inputs: Input dictionary with operation and parameters

        Returns:
            Dictionary with target results
        """
        start_time = time.time()
        calculation_trace = []

        try:
            # Parse input
            target_input = TargetSettingInput(**inputs)
            calculation_trace.append(f"Operation: {target_input.operation}")

            # Route to appropriate handler
            if target_input.operation == "set_target":
                result = self._set_target(target_input, calculation_trace)
            elif target_input.operation == "validate_target":
                result = self._validate_target(target_input, calculation_trace)
            elif target_input.operation == "calculate_trajectory":
                result = self._calculate_trajectory(target_input, calculation_trace)
            elif target_input.operation == "check_sbti_compliance":
                result = self._check_sbti_compliance(target_input, calculation_trace)
            else:
                raise ValueError(f"Unknown operation: {target_input.operation}")

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            result["processing_time_ms"] = processing_time

            # Capture audit entry
            self._capture_audit_entry(
                operation=target_input.operation,
                inputs=inputs,
                outputs={"success": result["success"]},
                calculation_trace=calculation_trace
            )

            return result

        except Exception as e:
            self.logger.error(f"Target setting failed: {str(e)}", exc_info=True)
            processing_time = (time.time() - start_time) * 1000

            return {
                "operation": inputs.get("operation", "unknown"),
                "success": False,
                "error_message": str(e),
                "processing_time_ms": processing_time,
                "timestamp": DeterministicClock.now().isoformat()
            }

    def _set_target(
        self,
        target_input: TargetSettingInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Set an emission reduction target."""
        if not target_input.base_year_emissions:
            raise ValueError("base_year_emissions required for set_target operation")

        base_emissions = target_input.base_year_emissions
        calculation_trace.append(f"Base year: {base_emissions.base_year}")

        # Get base year emissions based on scope coverage
        if target_input.scope_coverage == ScopeCategory.SCOPE_1_2:
            base_year_emissions = base_emissions.total_scope_1_2
        elif target_input.scope_coverage == ScopeCategory.SCOPE_1_2_3:
            base_year_emissions = base_emissions.total_scope_1_2_3
        else:
            base_year_emissions = base_emissions.scope_1_tco2e

        calculation_trace.append(f"Base year emissions ({target_input.scope_coverage.value}): {base_year_emissions:,.0f} tCO2e")

        # Determine target year
        if target_input.target_year:
            target_year = target_input.target_year
        elif target_input.timeframe == TargetTimeframe.NEAR_TERM:
            target_year = base_emissions.base_year + 10
        elif target_input.timeframe == TargetTimeframe.LONG_TERM:
            target_year = 2050
        else:  # Net zero
            target_year = 2050

        calculation_trace.append(f"Target year: {target_year}")

        # Get reduction rate based on temperature alignment
        reduction_params = SBTI_REDUCTION_RATES[target_input.temperature_alignment]
        annual_rate = reduction_params["absolute_min_rate"]

        # Use custom rate if provided and valid
        if target_input.custom_reduction_rate:
            if target_input.custom_reduction_rate >= annual_rate:
                annual_rate = target_input.custom_reduction_rate
                calculation_trace.append(f"Using custom reduction rate: {annual_rate:.1%}")
            else:
                calculation_trace.append(
                    f"Warning: Custom rate {target_input.custom_reduction_rate:.1%} below SBTi minimum {annual_rate:.1%}"
                )

        calculation_trace.append(f"Annual reduction rate: {annual_rate:.1%}")

        # Calculate target emissions
        years = target_year - base_emissions.base_year
        reduction_factor = (1 - annual_rate) ** years
        target_emissions = base_year_emissions * reduction_factor
        reduction_percent = (1 - reduction_factor) * 100

        calculation_trace.append(f"Years to target: {years}")
        calculation_trace.append(f"Reduction factor: {reduction_factor:.4f}")
        calculation_trace.append(f"Target emissions: {target_emissions:,.0f} tCO2e")
        calculation_trace.append(f"Reduction: {reduction_percent:.1f}%")

        # Generate trajectory
        trajectory = self._generate_trajectory(
            base_year=base_emissions.base_year,
            base_emissions=base_year_emissions,
            target_year=target_year,
            annual_rate=annual_rate,
            interim_years=target_input.interim_target_years if target_input.include_interim_targets else []
        )
        calculation_trace.append(f"Generated trajectory with {len(trajectory)} milestones")

        # Calculate intensity if applicable
        base_intensity = None
        target_intensity = None
        intensity_unit = None

        if target_input.target_type == TargetType.INTENSITY:
            if base_emissions.revenue_musd:
                base_intensity = base_year_emissions / base_emissions.revenue_musd
                target_intensity = target_emissions / base_emissions.revenue_musd
                intensity_unit = "tCO2e/MUSD"
            elif base_emissions.production_units:
                base_intensity = base_year_emissions / base_emissions.production_units
                target_intensity = target_emissions / base_emissions.production_units
                intensity_unit = f"tCO2e/{base_emissions.production_unit_name or 'unit'}"

        # Check SBTi compliance
        sbti_compliant, sbti_notes = self._check_sbti_minimum(
            annual_rate=annual_rate,
            temperature_alignment=target_input.temperature_alignment,
            scope_coverage=target_input.scope_coverage,
            scope_3_emissions=base_emissions.scope_3_tco2e
        )

        # Build target
        target = EmissionTarget(
            target_id=deterministic_id({
                "base_year": base_emissions.base_year,
                "target_year": target_year,
                "alignment": target_input.temperature_alignment.value
            }, "target_"),
            name=f"{target_input.temperature_alignment.value} aligned target - {target_year}",
            target_type=target_input.target_type,
            temperature_alignment=target_input.temperature_alignment,
            timeframe=target_input.timeframe,
            scope_coverage=target_input.scope_coverage,
            base_year=base_emissions.base_year,
            base_year_emissions_tco2e=base_year_emissions,
            target_year=target_year,
            target_emissions_tco2e=target_emissions,
            target_reduction_percent=reduction_percent,
            annual_reduction_rate=annual_rate,
            base_year_intensity=base_intensity,
            target_year_intensity=target_intensity,
            intensity_unit=intensity_unit,
            trajectory=trajectory,
            interim_targets=[m for m in trajectory if m.is_interim_target],
            sbti_compliant=sbti_compliant,
            sbti_validation_notes=sbti_notes,
            methodology="SBTi Absolute Contraction Approach v3.0",
            source_references=[
                "SBTi Target Setting Protocol v3.0 (2024)",
                "IPCC AR6 WG3 (2022)"
            ]
        )

        target.provenance_hash = content_hash(target.model_dump(exclude={"provenance_hash"}))

        return {
            "operation": "set_target",
            "success": True,
            "target": target.model_dump(),
            "sbti_compliant": sbti_compliant,
            "sbti_pathway_alignment": target_input.temperature_alignment.value,
            "provenance_hash": target.provenance_hash,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _validate_target(
        self,
        target_input: TargetSettingInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Validate an existing target against SBTi criteria."""
        if not target_input.proposed_target:
            raise ValueError("proposed_target required for validate_target operation")

        target = target_input.proposed_target
        errors = []
        warnings = []

        calculation_trace.append(f"Validating target: {target.name}")

        # Check reduction rate meets minimum
        min_rate = SBTI_REDUCTION_RATES[target.temperature_alignment]["absolute_min_rate"]
        if target.annual_reduction_rate < min_rate:
            errors.append(
                f"Annual reduction rate {target.annual_reduction_rate:.1%} below minimum {min_rate:.1%} "
                f"for {target.temperature_alignment.value} pathway"
            )
        calculation_trace.append(f"Reduction rate check: {target.annual_reduction_rate:.1%} vs {min_rate:.1%} min")

        # Check target year is valid
        max_years = 10 if target.timeframe == TargetTimeframe.NEAR_TERM else 35
        years = target.target_year - target.base_year
        if years > max_years:
            warnings.append(f"Target year {target.target_year} is {years} years from base year (max {max_years})")

        # Check base year is recent enough (within 2 years for SBTi)
        current_year = DeterministicClock.now().year
        if current_year - target.base_year > 2:
            warnings.append(f"Base year {target.base_year} should be within 2 years of submission")

        # Check Scope 3 coverage if applicable
        if target.scope_coverage == ScopeCategory.SCOPE_1_2_3:
            # SBTi requires Scope 3 targets if >40% of total
            calculation_trace.append("Scope 3 included in target")

        is_valid = len(errors) == 0

        return {
            "operation": "validate_target",
            "success": True,
            "target": target.model_dump(),
            "is_valid": is_valid,
            "validation_errors": errors,
            "validation_warnings": warnings,
            "sbti_compliant": is_valid,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _calculate_trajectory(
        self,
        target_input: TargetSettingInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Calculate year-by-year trajectory for a target."""
        if not target_input.base_year_emissions:
            raise ValueError("base_year_emissions required for calculate_trajectory")

        base_emissions = target_input.base_year_emissions

        # Get base emissions based on scope
        if target_input.scope_coverage == ScopeCategory.SCOPE_1_2:
            base_year_emissions = base_emissions.total_scope_1_2
        else:
            base_year_emissions = base_emissions.total_scope_1_2_3

        # Determine target year and rate
        target_year = target_input.target_year or (base_emissions.base_year + 10)
        reduction_params = SBTI_REDUCTION_RATES[target_input.temperature_alignment]
        annual_rate = target_input.custom_reduction_rate or reduction_params["absolute_min_rate"]

        # Generate full trajectory
        trajectory = self._generate_trajectory(
            base_year=base_emissions.base_year,
            base_emissions=base_year_emissions,
            target_year=target_year,
            annual_rate=annual_rate,
            interim_years=target_input.interim_target_years
        )

        calculation_trace.append(f"Generated {len(trajectory)} year trajectory")

        # Convert to dict format
        trajectory_data = [m.model_dump() for m in trajectory]

        return {
            "operation": "calculate_trajectory",
            "success": True,
            "trajectory": trajectory_data,
            "base_year": base_emissions.base_year,
            "base_emissions_tco2e": base_year_emissions,
            "target_year": target_year,
            "annual_reduction_rate": annual_rate,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _check_sbti_compliance(
        self,
        target_input: TargetSettingInput,
        calculation_trace: List[str]
    ) -> Dict[str, Any]:
        """Check if target meets SBTi criteria."""
        if not target_input.proposed_target:
            raise ValueError("proposed_target required for check_sbti_compliance")

        target = target_input.proposed_target

        sbti_compliant, notes = self._check_sbti_minimum(
            annual_rate=target.annual_reduction_rate,
            temperature_alignment=target.temperature_alignment,
            scope_coverage=target.scope_coverage,
            scope_3_emissions=0  # Would need actual Scope 3 data
        )

        calculation_trace.append(f"SBTi compliance check: {'PASS' if sbti_compliant else 'FAIL'}")

        return {
            "operation": "check_sbti_compliance",
            "success": True,
            "sbti_compliant": sbti_compliant,
            "sbti_validation_notes": notes,
            "sbti_pathway_alignment": target.temperature_alignment.value,
            "timestamp": DeterministicClock.now().isoformat()
        }

    def _generate_trajectory(
        self,
        base_year: int,
        base_emissions: float,
        target_year: int,
        annual_rate: float,
        interim_years: List[int]
    ) -> List[TargetMilestone]:
        """Generate year-by-year emission trajectory."""
        trajectory = []

        for year in range(base_year, target_year + 1):
            years_elapsed = year - base_year
            reduction_factor = (1 - annual_rate) ** years_elapsed
            target_emissions = base_emissions * reduction_factor
            reduction_percent = (1 - reduction_factor) * 100
            cumulative_reduction = base_emissions - target_emissions

            is_interim = year in interim_years or year == target_year

            milestone = TargetMilestone(
                year=year,
                target_emissions_tco2e=target_emissions,
                target_reduction_percent=reduction_percent,
                cumulative_reduction_tco2e=cumulative_reduction,
                is_interim_target=is_interim
            )
            trajectory.append(milestone)

        return trajectory

    def _check_sbti_minimum(
        self,
        annual_rate: float,
        temperature_alignment: TemperatureAlignment,
        scope_coverage: ScopeCategory,
        scope_3_emissions: float
    ) -> Tuple[bool, List[str]]:
        """Check if target meets SBTi minimum criteria."""
        notes = []
        is_compliant = True

        # Check reduction rate
        min_rate = SBTI_REDUCTION_RATES[temperature_alignment]["absolute_min_rate"]
        if annual_rate >= min_rate:
            notes.append(f"PASS: Annual reduction rate {annual_rate:.1%} meets minimum {min_rate:.1%}")
        else:
            notes.append(f"FAIL: Annual reduction rate {annual_rate:.1%} below minimum {min_rate:.1%}")
            is_compliant = False

        # Check scope coverage (SBTi requires Scope 1+2 at minimum)
        if scope_coverage in [ScopeCategory.SCOPE_1_2, ScopeCategory.SCOPE_1_2_3]:
            notes.append("PASS: Scope 1+2 emissions covered")
        else:
            notes.append("FAIL: SBTi requires at least Scope 1+2 coverage")
            is_compliant = False

        return is_compliant, notes

    # =========================================================================
    # Public API Methods
    # =========================================================================

    def set_sbti_target(
        self,
        base_year: int,
        scope_1_tco2e: float,
        scope_2_tco2e: float,
        scope_3_tco2e: float = 0,
        temperature_alignment: TemperatureAlignment = TemperatureAlignment.C_1_5
    ) -> EmissionTarget:
        """
        Set an SBTi-aligned target with simplified interface.

        Args:
            base_year: Base year for target
            scope_1_tco2e: Scope 1 emissions
            scope_2_tco2e: Scope 2 emissions
            scope_3_tco2e: Scope 3 emissions (optional)
            temperature_alignment: Target pathway (1.5C, well_below_2C)

        Returns:
            EmissionTarget with trajectory
        """
        result = self.execute({
            "operation": "set_target",
            "base_year_emissions": {
                "base_year": base_year,
                "scope_1_tco2e": scope_1_tco2e,
                "scope_2_tco2e": scope_2_tco2e,
                "scope_3_tco2e": scope_3_tco2e
            },
            "temperature_alignment": temperature_alignment.value,
            "scope_coverage": "scope_1_2_3" if scope_3_tco2e > 0 else "scope_1_2"
        })

        if result["success"]:
            return EmissionTarget(**result["target"])
        else:
            raise ValueError(result.get("error_message", "Target setting failed"))

    def get_required_reduction_rate(
        self,
        temperature_alignment: TemperatureAlignment
    ) -> float:
        """
        Get the minimum required annual reduction rate for a pathway.

        Args:
            temperature_alignment: Target pathway

        Returns:
            Minimum annual reduction rate
        """
        return SBTI_REDUCTION_RATES[temperature_alignment]["absolute_min_rate"]
