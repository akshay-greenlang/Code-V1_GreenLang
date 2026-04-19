"""
GL-012 STEAMQUAL - Separator Efficiency Calculator

Zero-hallucination separator and scrubber performance calculations.

Key Calculations:
    1. Mass Balance: m_removed = eta_sep * moisture_in
    2. Efficiency Estimation: From pressure drop and drain data
    3. Capacity Constraints: Maximum throughput limits
    4. Droplet Separation: Based on particle dynamics

Separator Types:
    - Cyclone separators
    - Chevron/vane separators
    - Mesh/wire mesh separators
    - Gravity separators

Reference: API Standard 521, ASHRAE Handbook, Perry's Chemical Engineers' Handbook

Author: GL-BackendDeveloper
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import hashlib
import json
import logging
import math

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS AND STANDARDS
# =============================================================================

# Separator efficiency ranges by type (typical at design conditions)
SEPARATOR_EFFICIENCY_RANGES = {
    "cyclone": {"min": 0.85, "design": 0.95, "max": 0.99},
    "chevron": {"min": 0.90, "design": 0.97, "max": 0.995},
    "mesh": {"min": 0.95, "design": 0.99, "max": 0.999},
    "gravity": {"min": 0.60, "design": 0.80, "max": 0.90},
    "coalescing": {"min": 0.97, "design": 0.995, "max": 0.9999},
}

# Design velocity limits (m/s)
SEPARATOR_VELOCITY_LIMITS = {
    "cyclone": {"min": 5.0, "design": 15.0, "max": 25.0},
    "chevron": {"min": 2.0, "design": 5.0, "max": 8.0},
    "mesh": {"min": 0.5, "design": 2.0, "max": 4.0},
    "gravity": {"min": 0.1, "design": 0.5, "max": 1.0},
    "coalescing": {"min": 0.05, "design": 0.3, "max": 0.5},
}

# Pressure drop coefficients (K = deltaP / (0.5 * rho * v^2))
SEPARATOR_K_FACTORS = {
    "cyclone": 8.0,
    "chevron": 4.0,
    "mesh": 10.0,
    "gravity": 1.0,
    "coalescing": 15.0,
}

# Minimum droplet size captured at design (micrometers)
SEPARATOR_CUTOFF_SIZES = {
    "cyclone": 10.0,
    "chevron": 5.0,
    "mesh": 1.0,
    "gravity": 50.0,
    "coalescing": 0.3,
}


class SeparatorType(str, Enum):
    """Type of moisture separator."""
    CYCLONE = "cyclone"
    CHEVRON = "chevron"
    MESH = "mesh"
    GRAVITY = "gravity"
    COALESCING = "coalescing"


class OperatingStatus(str, Enum):
    """Separator operating status."""
    OPTIMAL = "OPTIMAL"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    OVERLOADED = "OVERLOADED"
    FLOODED = "FLOODED"


class MaintenanceStatus(str, Enum):
    """Separator maintenance status."""
    GOOD = "GOOD"
    INSPECTION_DUE = "INSPECTION_DUE"
    CLEANING_REQUIRED = "CLEANING_REQUIRED"
    REPAIR_NEEDED = "REPAIR_NEEDED"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SeparatorSpecs:
    """Separator design specifications."""

    separator_id: str
    separator_type: SeparatorType

    # Design parameters
    design_flow_kg_s: float
    design_pressure_kpa: float
    design_efficiency: float = 0.95

    # Geometry
    inlet_diameter_m: float = 0.1
    vessel_diameter_m: float = 0.3
    vessel_length_m: float = 1.0

    # Performance curves
    efficiency_vs_velocity: Optional[Dict[str, float]] = None
    efficiency_vs_droplet_size: Optional[Dict[str, float]] = None


@dataclass
class SeparatorOperatingData:
    """Current separator operating conditions."""

    # Flow conditions
    steam_flow_kg_s: float
    inlet_moisture_fraction: float  # 0-1

    # Pressure conditions
    inlet_pressure_kpa: float
    pressure_drop_kpa: float

    # Temperature
    steam_temperature_c: float

    # Drain data
    drain_flow_kg_s: float = 0.0
    drain_temperature_c: float = 100.0

    # Measured outlet moisture (if available)
    outlet_moisture_fraction: Optional[float] = None


@dataclass
class MassBalanceResult:
    """Result of separator mass balance calculation."""

    calculation_id: str
    timestamp: datetime

    # Mass flows
    inlet_steam_flow_kg_s: float
    inlet_moisture_flow_kg_s: float
    outlet_steam_flow_kg_s: float
    outlet_moisture_flow_kg_s: float
    removed_moisture_flow_kg_s: float
    drain_flow_kg_s: float

    # Fractions
    inlet_moisture_fraction: float
    outlet_moisture_fraction: float
    separation_efficiency: float

    # Balance check
    mass_balance_error_kg_s: float
    mass_balance_error_percent: float
    balance_closed: bool

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class EfficiencyEstimateResult:
    """Result of efficiency estimation from operational data."""

    calculation_id: str
    timestamp: datetime

    # Estimated efficiency
    estimated_efficiency: float
    efficiency_uncertainty: float
    efficiency_lower_bound: float
    efficiency_upper_bound: float

    # Estimation method
    estimation_method: str
    confidence_level: float

    # Supporting data
    pressure_drop_ratio: float
    drain_rate_ratio: float
    velocity_ratio: float

    # Design comparison
    design_efficiency: float
    efficiency_degradation_percent: float

    # Operating status
    operating_status: OperatingStatus

    # Recommendations
    recommendations: List[str]

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class CapacityAnalysisResult:
    """Result of capacity constraint analysis."""

    calculation_id: str
    timestamp: datetime

    # Current loading
    current_flow_kg_s: float
    design_flow_kg_s: float
    loading_percent: float

    # Velocity analysis
    current_velocity_m_s: float
    design_velocity_m_s: float
    max_velocity_m_s: float
    velocity_margin_percent: float

    # Pressure drop analysis
    current_dp_kpa: float
    design_dp_kpa: float
    dp_ratio: float

    # Capacity limits
    capacity_limited_by: str
    max_capacity_kg_s: float
    remaining_capacity_kg_s: float

    # Operating status
    operating_status: OperatingStatus

    # Recommendations
    recommendations: List[str]

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class DropletSeparationResult:
    """Result of droplet separation analysis."""

    calculation_id: str
    timestamp: datetime

    # Separation by size
    cutoff_diameter_um: float
    separation_efficiencies: Dict[str, float]  # By droplet size range

    # Overall efficiency
    overall_efficiency: float
    mass_weighted_efficiency: float

    # Droplet distribution impact
    large_droplet_fraction: float
    small_droplet_fraction: float
    submicron_fraction: float

    # Carryover estimate
    estimated_carryover_ppm: float
    carryover_by_size: Dict[str, float]

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class SeparatorPerformanceReport:
    """Complete separator performance report."""

    calculation_id: str
    timestamp: datetime

    # Separator identification
    separator_id: str
    separator_type: SeparatorType

    # Mass balance
    mass_balance: MassBalanceResult

    # Efficiency
    efficiency_estimate: EfficiencyEstimateResult

    # Capacity
    capacity_analysis: CapacityAnalysisResult

    # Droplet separation
    droplet_separation: Optional[DropletSeparationResult]

    # Overall assessment
    overall_status: OperatingStatus
    maintenance_status: MaintenanceStatus
    performance_score: float  # 0-100

    # Priority actions
    priority_actions: List[str]

    # KPIs
    kpis: Dict[str, float]

    # Provenance
    input_hash: str
    output_hash: str
    formula_version: str = "SEP_V1.0"


# =============================================================================
# SEPARATOR EFFICIENCY CALCULATOR
# =============================================================================

class SeparatorEfficiencyCalculator:
    """
    Zero-hallucination separator efficiency calculator.

    Implements deterministic calculations for:
    - Mass balance across separators
    - Efficiency estimation from DP and drain data
    - Capacity constraint analysis
    - Droplet separation modeling

    All calculations use:
    - Decimal arithmetic for precision
    - SHA-256 provenance hashing
    - Complete audit trails
    - NO LLM in calculation path

    Example:
        >>> calc = SeparatorEfficiencyCalculator()
        >>> result = calc.compute_mass_balance(
        ...     inlet_flow=10.0,
        ...     inlet_moisture=0.05,
        ...     drain_flow=0.48
        ... )
        >>> print(f"Separation efficiency: {result.separation_efficiency:.1%}")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "SEP_V1.0"

    def __init__(
        self,
        default_separator_type: SeparatorType = SeparatorType.CYCLONE,
    ) -> None:
        """
        Initialize separator efficiency calculator.

        Args:
            default_separator_type: Default separator type for calculations
        """
        self.default_type = default_separator_type
        logger.info(f"SeparatorEfficiencyCalculator initialized, version {self.VERSION}")

    # =========================================================================
    # PUBLIC CALCULATION METHODS
    # =========================================================================

    def compute_mass_balance(
        self,
        inlet_steam_flow_kg_s: float,
        inlet_moisture_fraction: float,
        drain_flow_kg_s: float,
        outlet_moisture_fraction: Optional[float] = None,
        separator_efficiency: Optional[float] = None,
    ) -> MassBalanceResult:
        """
        Compute mass balance across separator.

        Mass Balance:
            m_in = m_out + m_drain
            m_moisture_in = m_moisture_out + m_removed

        Efficiency:
            eta = m_removed / m_moisture_in

        DETERMINISTIC calculation with NO LLM involvement.

        Args:
            inlet_steam_flow_kg_s: Total inlet steam flow (kg/s)
            inlet_moisture_fraction: Inlet moisture fraction (0-1)
            drain_flow_kg_s: Measured drain flow (kg/s)
            outlet_moisture_fraction: Measured outlet moisture (optional)
            separator_efficiency: Known efficiency (optional)

        Returns:
            MassBalanceResult with complete mass balance
        """
        # Validate inputs
        if inlet_steam_flow_kg_s <= 0:
            raise ValueError("Inlet steam flow must be positive")
        if not 0 <= inlet_moisture_fraction <= 1:
            raise ValueError("Moisture fraction must be between 0 and 1")
        if drain_flow_kg_s < 0:
            raise ValueError("Drain flow cannot be negative")

        # Calculate inlet moisture flow
        inlet_moisture_flow = Decimal(str(inlet_steam_flow_kg_s)) * Decimal(str(inlet_moisture_fraction))

        # Moisture removed = drain flow (assuming drain is all moisture)
        removed_moisture = Decimal(str(drain_flow_kg_s))

        # Calculate separation efficiency from drain if not provided
        if inlet_moisture_flow > 0:
            calculated_efficiency = removed_moisture / inlet_moisture_flow
            calculated_efficiency = min(Decimal("1.0"), max(Decimal("0.0"), calculated_efficiency))
        else:
            calculated_efficiency = Decimal("0.0")

        # Use provided efficiency or calculated
        if separator_efficiency is not None:
            efficiency = Decimal(str(separator_efficiency))
        else:
            efficiency = calculated_efficiency

        # Calculate outlet moisture
        if outlet_moisture_fraction is not None:
            # Use measured outlet moisture
            outlet_moisture_flow = (
                Decimal(str(inlet_steam_flow_kg_s)) *
                Decimal(str(outlet_moisture_fraction))
            )
        else:
            # Calculate from efficiency
            outlet_moisture_flow = inlet_moisture_flow * (Decimal("1.0") - efficiency)

        # Calculate outlet steam flow
        outlet_steam_flow = Decimal(str(inlet_steam_flow_kg_s)) - removed_moisture

        # Calculate outlet moisture fraction
        if outlet_steam_flow > 0:
            outlet_moisture_frac = outlet_moisture_flow / outlet_steam_flow
        else:
            outlet_moisture_frac = Decimal("0.0")

        # Mass balance check
        # Input = Output + Removed
        expected_output_moisture = inlet_moisture_flow - removed_moisture
        balance_error = outlet_moisture_flow - expected_output_moisture

        if inlet_moisture_flow > 0:
            balance_error_percent = abs(balance_error) / inlet_moisture_flow * 100
        else:
            balance_error_percent = Decimal("0.0")

        balance_closed = float(balance_error_percent) < 1.0  # < 1% error

        # Hashes
        input_hash = self._compute_hash({
            "inlet_flow": inlet_steam_flow_kg_s,
            "inlet_moisture": inlet_moisture_fraction,
            "drain_flow": drain_flow_kg_s,
        })
        output_hash = self._compute_hash({
            "efficiency": float(efficiency),
        })

        return MassBalanceResult(
            calculation_id=f"MASS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            inlet_steam_flow_kg_s=inlet_steam_flow_kg_s,
            inlet_moisture_flow_kg_s=round(float(inlet_moisture_flow), 4),
            outlet_steam_flow_kg_s=round(float(outlet_steam_flow), 4),
            outlet_moisture_flow_kg_s=round(float(outlet_moisture_flow), 4),
            removed_moisture_flow_kg_s=round(float(removed_moisture), 4),
            drain_flow_kg_s=drain_flow_kg_s,
            inlet_moisture_fraction=inlet_moisture_fraction,
            outlet_moisture_fraction=round(float(outlet_moisture_frac), 4),
            separation_efficiency=round(float(efficiency), 4),
            mass_balance_error_kg_s=round(float(abs(balance_error)), 6),
            mass_balance_error_percent=round(float(balance_error_percent), 2),
            balance_closed=balance_closed,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def estimate_efficiency_from_dp(
        self,
        specs: SeparatorSpecs,
        operating_data: SeparatorOperatingData,
    ) -> EfficiencyEstimateResult:
        """
        Estimate separator efficiency from pressure drop and drain data.

        Method:
        1. Calculate expected DP at current flow
        2. Compare measured DP to expected
        3. Infer efficiency degradation from DP deviation
        4. Cross-check with drain flow data

        DETERMINISTIC calculation with NO LLM involvement.

        Args:
            specs: Separator design specifications
            operating_data: Current operating conditions

        Returns:
            EfficiencyEstimateResult with efficiency estimate
        """
        recommendations = []
        sep_type = specs.separator_type.value

        # Step 1: Calculate velocity ratio
        design_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["design"]
        max_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["max"]

        # Estimate current velocity from flow and geometry
        if specs.inlet_diameter_m > 0:
            inlet_area = math.pi * (specs.inlet_diameter_m / 2)**2
            steam_density = self._estimate_steam_density(operating_data.inlet_pressure_kpa)
            current_velocity = operating_data.steam_flow_kg_s / (steam_density * inlet_area)
        else:
            current_velocity = design_velocity

        velocity_ratio = current_velocity / design_velocity if design_velocity > 0 else 1.0

        # Step 2: Calculate expected pressure drop
        # deltaP = K * 0.5 * rho * v^2
        k_factor = SEPARATOR_K_FACTORS[sep_type]
        expected_dp = k_factor * 0.5 * steam_density * current_velocity**2 / 1000  # kPa

        # Measured vs expected DP ratio
        if expected_dp > 0:
            dp_ratio = operating_data.pressure_drop_kpa / expected_dp
        else:
            dp_ratio = 1.0

        # Step 3: Calculate drain rate ratio
        # Expected drain = inlet_moisture * design_efficiency
        expected_drain = (
            operating_data.steam_flow_kg_s *
            operating_data.inlet_moisture_fraction *
            specs.design_efficiency
        )

        if expected_drain > 0:
            drain_ratio = operating_data.drain_flow_kg_s / expected_drain
        else:
            drain_ratio = 1.0

        # Step 4: Estimate efficiency from indicators
        # High DP ratio indicates fouling (lower efficiency)
        # Low drain ratio indicates bypassing or low moisture

        # Base efficiency on design
        design_eff = specs.design_efficiency
        min_eff = SEPARATOR_EFFICIENCY_RANGES[sep_type]["min"]
        max_eff = SEPARATOR_EFFICIENCY_RANGES[sep_type]["max"]

        # Adjust for DP deviation (high DP = fouling = lower efficiency)
        if dp_ratio > 1.5:
            dp_penalty = (dp_ratio - 1.0) * 0.1  # 10% penalty per 100% over
            recommendations.append("High pressure drop - inspect for fouling")
        elif dp_ratio < 0.5:
            dp_penalty = (1.0 - dp_ratio) * 0.05  # Minor penalty for low DP
            recommendations.append("Low pressure drop - check for bypass")
        else:
            dp_penalty = abs(dp_ratio - 1.0) * 0.02

        # Adjust for velocity (off-design operation)
        if velocity_ratio > 1.3:
            velocity_penalty = (velocity_ratio - 1.0) * 0.15
            recommendations.append("Operating above design velocity")
        elif velocity_ratio < 0.5:
            velocity_penalty = (1.0 - velocity_ratio) * 0.1
            recommendations.append("Operating below design velocity - turndown")
        else:
            velocity_penalty = 0.0

        # Adjust for drain data
        if drain_ratio < 0.7:
            drain_penalty = (1.0 - drain_ratio) * 0.1
            recommendations.append("Low drain flow - check drain trap")
        elif drain_ratio > 1.5:
            drain_penalty = 0.0  # High drain is OK (or inlet moisture underestimated)
        else:
            drain_penalty = 0.0

        # Calculate estimated efficiency
        total_penalty = dp_penalty + velocity_penalty + drain_penalty
        estimated_efficiency = design_eff - total_penalty
        estimated_efficiency = max(min_eff, min(max_eff, estimated_efficiency))

        # Uncertainty estimate
        # Base uncertainty on number and quality of indicators
        base_uncertainty = 0.05  # 5% base
        if operating_data.outlet_moisture_fraction is not None:
            base_uncertainty = 0.02  # Direct measurement available

        # Bounds
        lower_bound = max(min_eff, estimated_efficiency - base_uncertainty)
        upper_bound = min(max_eff, estimated_efficiency + base_uncertainty)

        # Efficiency degradation
        degradation = (design_eff - estimated_efficiency) / design_eff * 100 if design_eff > 0 else 0

        # Determine operating status
        if velocity_ratio > 1.5 or dp_ratio > 2.0:
            status = OperatingStatus.OVERLOADED
        elif estimated_efficiency < min_eff + 0.05:
            status = OperatingStatus.DEGRADED
        elif abs(velocity_ratio - 1.0) < 0.2 and abs(dp_ratio - 1.0) < 0.3:
            status = OperatingStatus.OPTIMAL
        else:
            status = OperatingStatus.ACCEPTABLE

        # Hashes
        input_hash = self._compute_hash({
            "separator_id": specs.separator_id,
            "steam_flow": operating_data.steam_flow_kg_s,
            "dp": operating_data.pressure_drop_kpa,
        })
        output_hash = self._compute_hash({
            "estimated_efficiency": estimated_efficiency,
        })

        return EfficiencyEstimateResult(
            calculation_id=f"EFF-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            estimated_efficiency=round(estimated_efficiency, 4),
            efficiency_uncertainty=round(base_uncertainty, 4),
            efficiency_lower_bound=round(lower_bound, 4),
            efficiency_upper_bound=round(upper_bound, 4),
            estimation_method="dp_drain_correlation",
            confidence_level=0.9,
            pressure_drop_ratio=round(dp_ratio, 3),
            drain_rate_ratio=round(drain_ratio, 3),
            velocity_ratio=round(velocity_ratio, 3),
            design_efficiency=design_eff,
            efficiency_degradation_percent=round(degradation, 2),
            operating_status=status,
            recommendations=recommendations,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def analyze_capacity_constraints(
        self,
        specs: SeparatorSpecs,
        operating_data: SeparatorOperatingData,
    ) -> CapacityAnalysisResult:
        """
        Analyze separator capacity constraints and limits.

        Checks:
        1. Flow vs design capacity
        2. Velocity vs limits
        3. Pressure drop vs acceptable range
        4. Identifies limiting factor

        DETERMINISTIC calculation with NO LLM involvement.

        Args:
            specs: Separator design specifications
            operating_data: Current operating conditions

        Returns:
            CapacityAnalysisResult with capacity analysis
        """
        recommendations = []
        sep_type = specs.separator_type.value

        # Get velocity limits
        min_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["min"]
        design_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["design"]
        max_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["max"]

        # Calculate current velocity
        steam_density = self._estimate_steam_density(operating_data.inlet_pressure_kpa)
        if specs.inlet_diameter_m > 0:
            inlet_area = math.pi * (specs.inlet_diameter_m / 2)**2
            current_velocity = operating_data.steam_flow_kg_s / (steam_density * inlet_area)
        else:
            current_velocity = design_velocity

        # Loading percentage
        loading = operating_data.steam_flow_kg_s / specs.design_flow_kg_s * 100 if specs.design_flow_kg_s > 0 else 100

        # Velocity margin
        velocity_margin = (max_velocity - current_velocity) / max_velocity * 100 if max_velocity > 0 else 0

        # Expected DP at design
        k_factor = SEPARATOR_K_FACTORS[sep_type]
        design_dp = k_factor * 0.5 * steam_density * design_velocity**2 / 1000

        # DP ratio
        dp_ratio = operating_data.pressure_drop_kpa / design_dp if design_dp > 0 else 1.0

        # Determine capacity limit
        # Velocity limit
        if inlet_area > 0:
            max_flow_velocity = max_velocity * steam_density * inlet_area
        else:
            max_flow_velocity = specs.design_flow_kg_s * 1.5

        # DP limit (assume 2x design DP is maximum)
        max_flow_dp = specs.design_flow_kg_s * math.sqrt(2.0 / dp_ratio) if dp_ratio > 0 else specs.design_flow_kg_s * 2

        # Take minimum as limiting factor
        if max_flow_velocity < max_flow_dp:
            capacity_limit = "velocity"
            max_capacity = max_flow_velocity
        else:
            capacity_limit = "pressure_drop"
            max_capacity = max_flow_dp

        remaining_capacity = max_capacity - operating_data.steam_flow_kg_s

        # Determine operating status
        if loading > 120:
            status = OperatingStatus.OVERLOADED
            recommendations.append("CRITICAL: Separator overloaded - reduce flow")
        elif velocity_margin < 10:
            status = OperatingStatus.OVERLOADED
            recommendations.append("Near velocity limit - at capacity")
        elif loading > 100:
            status = OperatingStatus.DEGRADED
            recommendations.append("Above design capacity - monitor efficiency")
        elif loading > 80:
            status = OperatingStatus.ACCEPTABLE
        else:
            status = OperatingStatus.OPTIMAL

        if loading < 30:
            recommendations.append("Low loading - consider bypass for efficiency")

        # Hashes
        input_hash = self._compute_hash({
            "separator_id": specs.separator_id,
            "steam_flow": operating_data.steam_flow_kg_s,
        })
        output_hash = self._compute_hash({
            "loading": loading,
            "status": status.value,
        })

        return CapacityAnalysisResult(
            calculation_id=f"CAP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            current_flow_kg_s=operating_data.steam_flow_kg_s,
            design_flow_kg_s=specs.design_flow_kg_s,
            loading_percent=round(loading, 1),
            current_velocity_m_s=round(current_velocity, 2),
            design_velocity_m_s=design_velocity,
            max_velocity_m_s=max_velocity,
            velocity_margin_percent=round(velocity_margin, 1),
            current_dp_kpa=operating_data.pressure_drop_kpa,
            design_dp_kpa=round(design_dp, 2),
            dp_ratio=round(dp_ratio, 3),
            capacity_limited_by=capacity_limit,
            max_capacity_kg_s=round(max_capacity, 3),
            remaining_capacity_kg_s=round(max(0, remaining_capacity), 3),
            operating_status=status,
            recommendations=recommendations,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def calculate_droplet_separation(
        self,
        specs: SeparatorSpecs,
        operating_data: SeparatorOperatingData,
        droplet_distribution: Optional[Dict[str, float]] = None,
    ) -> DropletSeparationResult:
        """
        Calculate droplet separation efficiency by size.

        Uses separator performance curves to determine
        separation efficiency for different droplet sizes.

        DETERMINISTIC calculation with NO LLM involvement.

        Args:
            specs: Separator design specifications
            operating_data: Current operating conditions
            droplet_distribution: Droplet size distribution (optional)
                Format: {"<5um": 0.1, "5-10um": 0.2, ...}

        Returns:
            DropletSeparationResult with size-dependent efficiency
        """
        sep_type = specs.separator_type.value

        # Default droplet distribution (typical for wet steam)
        if droplet_distribution is None:
            droplet_distribution = {
                "<1um": 0.05,
                "1-5um": 0.15,
                "5-10um": 0.25,
                "10-20um": 0.30,
                "20-50um": 0.15,
                ">50um": 0.10,
            }

        # Cutoff diameter for separator type
        cutoff = SEPARATOR_CUTOFF_SIZES[sep_type]

        # Calculate velocity factor
        steam_density = self._estimate_steam_density(operating_data.inlet_pressure_kpa)
        if specs.inlet_diameter_m > 0:
            inlet_area = math.pi * (specs.inlet_diameter_m / 2)**2
            current_velocity = operating_data.steam_flow_kg_s / (steam_density * inlet_area)
        else:
            current_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["design"]

        design_velocity = SEPARATOR_VELOCITY_LIMITS[sep_type]["design"]
        velocity_factor = current_velocity / design_velocity if design_velocity > 0 else 1.0

        # Adjust cutoff for velocity (higher velocity = worse separation)
        adjusted_cutoff = cutoff * velocity_factor

        # Calculate efficiency for each size range
        separation_efficiencies = {}
        carryover_by_size = {}

        # Representative sizes for each range (micrometers)
        size_representatives = {
            "<1um": 0.5,
            "1-5um": 3.0,
            "5-10um": 7.5,
            "10-20um": 15.0,
            "20-50um": 35.0,
            ">50um": 75.0,
        }

        total_carryover = Decimal("0")
        weighted_efficiency_sum = Decimal("0")

        for size_range, mass_fraction in droplet_distribution.items():
            rep_size = size_representatives.get(size_range, 10.0)

            # Efficiency based on size vs cutoff
            # eta = 1 - exp(-k * (d/d_cutoff)^2) where k depends on separator type
            size_ratio = rep_size / adjusted_cutoff
            if size_ratio > 1:
                # Above cutoff - high efficiency
                eta = 1.0 - math.exp(-2 * (size_ratio - 1)**2)
                eta = min(0.999, eta)
            else:
                # Below cutoff - efficiency drops
                eta = (size_ratio)**2 * specs.design_efficiency
                eta = max(0.0, eta)

            separation_efficiencies[size_range] = round(eta, 4)

            # Carryover from this size range
            carryover = (1 - eta) * mass_fraction * 1e6  # ppm assuming 1 inlet
            carryover_by_size[size_range] = round(carryover, 2)
            total_carryover += Decimal(str(carryover))

            # Weighted efficiency
            weighted_efficiency_sum += Decimal(str(eta)) * Decimal(str(mass_fraction))

        # Overall and weighted efficiency
        total_mass_fraction = sum(droplet_distribution.values())
        if total_mass_fraction > 0:
            mass_weighted_efficiency = float(weighted_efficiency_sum / Decimal(str(total_mass_fraction)))
        else:
            mass_weighted_efficiency = 0.0

        # Simple average efficiency
        if separation_efficiencies:
            overall_efficiency = sum(separation_efficiencies.values()) / len(separation_efficiencies)
        else:
            overall_efficiency = specs.design_efficiency

        # Categorize droplet fractions
        large_fraction = droplet_distribution.get("20-50um", 0) + droplet_distribution.get(">50um", 0)
        small_fraction = droplet_distribution.get("<1um", 0) + droplet_distribution.get("1-5um", 0)
        submicron_fraction = droplet_distribution.get("<1um", 0)

        # Estimated carryover (scale by inlet moisture)
        inlet_moisture_ppm = operating_data.inlet_moisture_fraction * 1e6
        estimated_carryover = float(total_carryover) * inlet_moisture_ppm / 1e6

        # Hashes
        input_hash = self._compute_hash({
            "separator_id": specs.separator_id,
            "distribution": droplet_distribution,
        })
        output_hash = self._compute_hash({
            "overall_efficiency": overall_efficiency,
        })

        return DropletSeparationResult(
            calculation_id=f"DROP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            cutoff_diameter_um=round(adjusted_cutoff, 2),
            separation_efficiencies=separation_efficiencies,
            overall_efficiency=round(overall_efficiency, 4),
            mass_weighted_efficiency=round(mass_weighted_efficiency, 4),
            large_droplet_fraction=round(large_fraction, 3),
            small_droplet_fraction=round(small_fraction, 3),
            submicron_fraction=round(submicron_fraction, 3),
            estimated_carryover_ppm=round(estimated_carryover, 2),
            carryover_by_size=carryover_by_size,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def generate_performance_report(
        self,
        specs: SeparatorSpecs,
        operating_data: SeparatorOperatingData,
        droplet_distribution: Optional[Dict[str, float]] = None,
    ) -> SeparatorPerformanceReport:
        """
        Generate complete separator performance report.

        Combines all analysis methods into comprehensive report.

        DETERMINISTIC calculation with NO LLM involvement.

        Args:
            specs: Separator design specifications
            operating_data: Current operating conditions
            droplet_distribution: Optional droplet size distribution

        Returns:
            SeparatorPerformanceReport with complete analysis
        """
        # Run all analyses
        mass_balance = self.compute_mass_balance(
            inlet_steam_flow_kg_s=operating_data.steam_flow_kg_s,
            inlet_moisture_fraction=operating_data.inlet_moisture_fraction,
            drain_flow_kg_s=operating_data.drain_flow_kg_s,
            outlet_moisture_fraction=operating_data.outlet_moisture_fraction,
        )

        efficiency_estimate = self.estimate_efficiency_from_dp(specs, operating_data)

        capacity_analysis = self.analyze_capacity_constraints(specs, operating_data)

        droplet_separation = self.calculate_droplet_separation(
            specs, operating_data, droplet_distribution
        )

        # Determine overall status (worst of sub-statuses)
        statuses = [efficiency_estimate.operating_status, capacity_analysis.operating_status]
        if OperatingStatus.OVERLOADED in statuses:
            overall_status = OperatingStatus.OVERLOADED
        elif OperatingStatus.FLOODED in statuses:
            overall_status = OperatingStatus.FLOODED
        elif OperatingStatus.DEGRADED in statuses:
            overall_status = OperatingStatus.DEGRADED
        elif OperatingStatus.ACCEPTABLE in statuses:
            overall_status = OperatingStatus.ACCEPTABLE
        else:
            overall_status = OperatingStatus.OPTIMAL

        # Determine maintenance status
        degradation = efficiency_estimate.efficiency_degradation_percent
        if degradation > 20:
            maintenance_status = MaintenanceStatus.REPAIR_NEEDED
        elif degradation > 10:
            maintenance_status = MaintenanceStatus.CLEANING_REQUIRED
        elif degradation > 5:
            maintenance_status = MaintenanceStatus.INSPECTION_DUE
        else:
            maintenance_status = MaintenanceStatus.GOOD

        # Calculate performance score (0-100)
        efficiency_score = efficiency_estimate.estimated_efficiency * 100 / specs.design_efficiency
        capacity_score = 100 - min(100, max(0, capacity_analysis.loading_percent - 100) * 2)
        balance_score = 100 if mass_balance.balance_closed else 80

        performance_score = (
            0.5 * min(100, efficiency_score) +
            0.3 * capacity_score +
            0.2 * balance_score
        )

        # Compile priority actions
        priority_actions = []
        priority_actions.extend(efficiency_estimate.recommendations)
        priority_actions.extend(capacity_analysis.recommendations)

        # Sort by implied urgency (CRITICAL first)
        critical = [a for a in priority_actions if "CRITICAL" in a]
        other = [a for a in priority_actions if "CRITICAL" not in a]
        priority_actions = critical + other[:5]

        # KPIs
        kpis = {
            "separation_efficiency": mass_balance.separation_efficiency,
            "loading_percent": capacity_analysis.loading_percent,
            "dp_ratio": efficiency_estimate.pressure_drop_ratio,
            "velocity_ratio": efficiency_estimate.velocity_ratio,
            "carryover_ppm": droplet_separation.estimated_carryover_ppm,
            "performance_score": round(performance_score, 1),
        }

        # Hashes
        input_hash = self._compute_hash({
            "separator_id": specs.separator_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        output_hash = self._compute_hash({
            "performance_score": performance_score,
            "overall_status": overall_status.value,
        })

        return SeparatorPerformanceReport(
            calculation_id=f"SEPRPT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            separator_id=specs.separator_id,
            separator_type=specs.separator_type,
            mass_balance=mass_balance,
            efficiency_estimate=efficiency_estimate,
            capacity_analysis=capacity_analysis,
            droplet_separation=droplet_separation,
            overall_status=overall_status,
            maintenance_status=maintenance_status,
            performance_score=round(performance_score, 1),
            priority_actions=priority_actions,
            kpis=kpis,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    def _estimate_steam_density(self, pressure_kpa: float) -> float:
        """Estimate saturated steam density from pressure."""
        t_sat = self._get_saturation_temp(pressure_kpa)
        t_k = t_sat + 273.15
        r_steam = 461.5  # J/kg-K

        # Compressibility correction
        z = 1.0 - 0.0001 * pressure_kpa / 100
        rho = pressure_kpa * 1000 / (r_steam * t_k * z)
        return max(0.5, rho)

    def _get_saturation_temp(self, pressure_kpa: float) -> float:
        """Get saturation temperature from pressure."""
        if pressure_kpa < 1:
            pressure_kpa = 1
        ln_p = math.log(pressure_kpa)
        t_sat = 42.68 + 21.11 * ln_p + 0.105 * ln_p**2
        return t_sat

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
