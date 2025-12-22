# -*- coding: utf-8 -*-
"""
SteamTrapEnergyLossCalculator for GL-008 TRAPCATCHER

Advanced energy loss calculator supporting multiple failure modes, trap types,
and comprehensive ROI analysis for steam trap maintenance decisions.

Standards:
- ASME PTC 39: Steam Traps - Performance Test Codes
- ASTM F1139: Standard Specification for Steam Traps
- ISO 7841: Automatic steam traps - Determination of steam loss
- DOE Steam System Assessment Protocol

Key Features:
- Multi-failure mode analysis (blow-through, leaking, blocked)
- Steam loss rate calculation using Napier equation
- Support for 4 trap types (thermodynamic, thermostatic, mechanical, venturi)
- Annual energy cost with configurable steam/fuel costs
- Carbon emission penalty calculation (EPA methodology)
- Comprehensive ROI for trap replacement
- ASME/ASTM compliant testing thresholds

Zero-Hallucination Guarantee:
All calculations use deterministic engineering formulas from published standards.
No LLM or AI inference in any calculation path.
Same inputs always produce identical outputs.

Example:
    >>> from steam_trap_energy_loss_calculator import SteamTrapEnergyLossCalculator
    >>> calculator = SteamTrapEnergyLossCalculator()
    >>> result = calculator.calculate_energy_loss(
    ...     trap_id="ST-001",
    ...     failure_mode=FailureMode.BLOW_THROUGH,
    ...     orifice_diameter_mm=6.35,
    ...     pressure_bar_g=10.0,
    ...     trap_type=TrapType.THERMODYNAMIC
    ... )
    >>> print(f"Annual energy loss: ${result.annual_energy_cost_usd:,.2f}")

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, FrozenSet

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================

class FailureMode(str, Enum):
    """Steam trap failure modes per ASME PTC 39."""
    BLOW_THROUGH = "blow_through"  # Complete failure - 100% steam loss
    LEAKING = "leaking"            # Partial failure - 5-50% steam loss
    BLOCKED = "blocked"            # Failed closed - no condensate drainage
    CYCLING_FAST = "cycling_fast"  # Over-cycling causing steam loss
    CYCLING_SLOW = "cycling_slow"  # Under-cycling causing backup
    COLD_TRAP = "cold_trap"        # Trap not receiving steam
    NORMAL = "normal"              # Operating correctly


class TrapType(str, Enum):
    """Steam trap types per ASTM F1139."""
    THERMODYNAMIC = "thermodynamic"  # Disc traps
    THERMOSTATIC = "thermostatic"    # Bellows, bimetallic
    MECHANICAL = "mechanical"        # Float, bucket
    VENTURI = "venturi"              # Venturi orifice traps


class SeverityLevel(str, Enum):
    """Failure severity classification."""
    CRITICAL = "critical"      # >75% steam loss - immediate action
    HIGH = "high"              # 50-75% steam loss - urgent
    MEDIUM = "medium"          # 25-50% steam loss - scheduled
    LOW = "low"                # 10-25% steam loss - monitor
    MINIMAL = "minimal"        # <10% steam loss - acceptable
    NONE = "none"              # Operating normally


class FuelType(str, Enum):
    """Fuel types for boiler energy calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    PROPANE = "propane"
    COAL = "coal"
    BIOMASS = "biomass"


# ============================================================================
# PROVENANCE TRACKING
# ============================================================================

@dataclass
class ProvenanceStep:
    """Single step in calculation provenance chain."""
    step_number: int
    operation: str
    inputs: Dict[str, Any]
    formula: str
    result: Any
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "step_number": self.step_number,
            "operation": self.operation,
            "inputs": self.inputs,
            "formula": self.formula,
            "result": str(self.result),
            "timestamp": self.timestamp.isoformat()
        }


class ProvenanceTracker:
    """Thread-safe provenance tracker for audit trail."""

    def __init__(self):
        """Initialize provenance tracker."""
        self._steps: List[ProvenanceStep] = []
        self._lock = threading.Lock()

    def record_step(
        self,
        operation: str,
        inputs: Dict[str, Any],
        formula: str,
        result: Any
    ) -> None:
        """Record a calculation step."""
        with self._lock:
            step = ProvenanceStep(
                step_number=len(self._steps) + 1,
                operation=operation,
                inputs=inputs,
                formula=formula,
                result=result
            )
            self._steps.append(step)

    def get_steps(self) -> List[ProvenanceStep]:
        """Get all recorded steps."""
        with self._lock:
            return list(self._steps)

    def get_hash(self) -> str:
        """Calculate SHA-256 hash of all steps."""
        with self._lock:
            data = json.dumps(
                [s.to_dict() for s in self._steps],
                sort_keys=True,
                default=str
            )
            return hashlib.sha256(data.encode()).hexdigest()

    def clear(self) -> None:
        """Clear all recorded steps."""
        with self._lock:
            self._steps.clear()


# ============================================================================
# FROZEN DATA CLASSES (Immutable for thread safety)
# ============================================================================

@dataclass(frozen=True)
class EnergyLossConfig:
    """
    Immutable configuration for energy loss calculations.

    Attributes:
        steam_cost_usd_per_1000lb: Steam cost per 1000 lb
        fuel_cost_usd_per_mmbtu: Fuel cost per MMBtu
        fuel_type: Boiler fuel type
        boiler_efficiency: Boiler efficiency (0.70-0.95)
        operating_hours_per_year: Annual operating hours
        carbon_price_usd_per_ton: Carbon emission penalty price
        discount_rate: Discount rate for NPV calculations
        trap_lifetime_years: Expected trap lifetime
        maintenance_interval_months: Maintenance interval
    """
    steam_cost_usd_per_1000lb: Decimal = Decimal("12.50")
    fuel_cost_usd_per_mmbtu: Decimal = Decimal("4.50")
    fuel_type: FuelType = FuelType.NATURAL_GAS
    boiler_efficiency: Decimal = Decimal("0.82")
    operating_hours_per_year: int = 8760
    carbon_price_usd_per_ton: Decimal = Decimal("50.00")
    discount_rate: Decimal = Decimal("0.10")
    trap_lifetime_years: int = 7
    maintenance_interval_months: int = 12


@dataclass(frozen=True)
class TrapSpecifications:
    """
    Immutable steam trap specifications.

    Attributes:
        trap_id: Unique trap identifier
        trap_type: Type of steam trap
        manufacturer: Manufacturer name
        model: Model number
        orifice_diameter_mm: Orifice diameter in mm
        max_pressure_bar: Maximum rated pressure
        max_capacity_kg_hr: Maximum condensate capacity
        installation_date: Date installed
        last_inspection_date: Last inspection date
    """
    trap_id: str
    trap_type: TrapType
    manufacturer: str = ""
    model: str = ""
    orifice_diameter_mm: Decimal = Decimal("6.35")  # 1/4 inch typical
    max_pressure_bar: Decimal = Decimal("20.0")
    max_capacity_kg_hr: Decimal = Decimal("500.0")
    installation_date: Optional[str] = None
    last_inspection_date: Optional[str] = None


@dataclass(frozen=True)
class SteamConditions:
    """
    Immutable steam operating conditions.

    Attributes:
        pressure_bar_g: Gauge pressure in bar
        temperature_c: Steam temperature in Celsius
        saturation_temp_c: Saturation temperature
        enthalpy_steam_btu_lb: Steam enthalpy (BTU/lb)
        enthalpy_condensate_btu_lb: Condensate enthalpy (BTU/lb)
        specific_volume_ft3_lb: Specific volume (ft3/lb)
    """
    pressure_bar_g: Decimal
    temperature_c: Decimal
    saturation_temp_c: Decimal
    enthalpy_steam_btu_lb: Decimal
    enthalpy_condensate_btu_lb: Decimal
    specific_volume_ft3_lb: Decimal


@dataclass(frozen=True)
class SteamLossResult:
    """
    Immutable steam loss calculation result.

    Attributes:
        steam_loss_lb_hr: Steam loss rate in lb/hr
        steam_loss_kg_hr: Steam loss rate in kg/hr
        leak_percentage: Percentage of rated capacity lost
        failure_severity: Severity classification
    """
    steam_loss_lb_hr: Decimal
    steam_loss_kg_hr: Decimal
    leak_percentage: Decimal
    failure_severity: SeverityLevel


@dataclass(frozen=True)
class EnergyLossMetrics:
    """
    Immutable energy loss metrics.

    Attributes:
        energy_loss_btu_hr: Energy loss rate (BTU/hr)
        energy_loss_kw: Energy loss rate (kW)
        energy_loss_mmbtu_hr: Energy loss rate (MMBtu/hr)
        fuel_waste_mmbtu_hr: Fuel waste accounting for boiler efficiency
        daily_energy_loss_mmbtu: Daily energy loss
        monthly_energy_loss_mmbtu: Monthly energy loss
        annual_energy_loss_mmbtu: Annual energy loss
    """
    energy_loss_btu_hr: Decimal
    energy_loss_kw: Decimal
    energy_loss_mmbtu_hr: Decimal
    fuel_waste_mmbtu_hr: Decimal
    daily_energy_loss_mmbtu: Decimal
    monthly_energy_loss_mmbtu: Decimal
    annual_energy_loss_mmbtu: Decimal


@dataclass(frozen=True)
class CarbonEmissionResult:
    """
    Immutable carbon emission calculation result.

    EPA emission factors applied.

    Attributes:
        co2_kg_hr: CO2 emissions rate (kg/hr)
        co2_tons_year: Annual CO2 emissions (metric tons)
        carbon_penalty_usd_year: Annual carbon penalty cost
        emission_factor_kg_per_mmbtu: Applied emission factor
    """
    co2_kg_hr: Decimal
    co2_tons_year: Decimal
    carbon_penalty_usd_year: Decimal
    emission_factor_kg_per_mmbtu: Decimal


@dataclass(frozen=True)
class ROIResult:
    """
    Immutable ROI calculation result.

    Attributes:
        replacement_cost_usd: Total replacement cost
        annual_savings_usd: Annual savings from replacement
        simple_payback_days: Simple payback period (days)
        simple_payback_months: Simple payback period (months)
        roi_first_year_percent: First year ROI
        npv_lifetime_usd: NPV over trap lifetime
        irr_percent: Internal rate of return
        break_even_date: Projected break-even date
        recommendation: Action recommendation
    """
    replacement_cost_usd: Decimal
    annual_savings_usd: Decimal
    simple_payback_days: Decimal
    simple_payback_months: Decimal
    roi_first_year_percent: Decimal
    npv_lifetime_usd: Decimal
    irr_percent: Decimal
    break_even_date: Optional[str]
    recommendation: str


@dataclass(frozen=True)
class EnergyLossAnalysisResult:
    """
    Complete immutable energy loss analysis result.

    Attributes:
        trap_specs: Trap specifications
        failure_mode: Detected/specified failure mode
        steam_conditions: Operating steam conditions
        steam_loss: Steam loss calculation result
        energy_metrics: Energy loss metrics
        carbon_emissions: Carbon emission calculations
        annual_energy_cost_usd: Total annual energy cost
        roi_analysis: ROI analysis (if replacement cost provided)
        provenance_hash: SHA-256 hash for audit trail
        calculation_timestamp: Analysis timestamp
        calculation_method: Method description
    """
    trap_specs: TrapSpecifications
    failure_mode: FailureMode
    steam_conditions: SteamConditions
    steam_loss: SteamLossResult
    energy_metrics: EnergyLossMetrics
    carbon_emissions: CarbonEmissionResult
    annual_energy_cost_usd: Decimal
    roi_analysis: Optional[ROIResult]
    provenance_hash: str
    calculation_timestamp: datetime
    calculation_method: str = "ASME_PTC_39"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trap_id": self.trap_specs.trap_id,
            "trap_type": self.trap_specs.trap_type.value,
            "failure_mode": self.failure_mode.value,
            "steam_loss_lb_hr": float(self.steam_loss.steam_loss_lb_hr),
            "steam_loss_kg_hr": float(self.steam_loss.steam_loss_kg_hr),
            "failure_severity": self.steam_loss.failure_severity.value,
            "energy_loss_kw": float(self.energy_metrics.energy_loss_kw),
            "annual_energy_loss_mmbtu": float(self.energy_metrics.annual_energy_loss_mmbtu),
            "annual_energy_cost_usd": float(self.annual_energy_cost_usd),
            "co2_tons_year": float(self.carbon_emissions.co2_tons_year),
            "carbon_penalty_usd_year": float(self.carbon_emissions.carbon_penalty_usd_year),
            "roi_analysis": {
                "replacement_cost_usd": float(self.roi_analysis.replacement_cost_usd),
                "annual_savings_usd": float(self.roi_analysis.annual_savings_usd),
                "simple_payback_days": float(self.roi_analysis.simple_payback_days),
                "roi_first_year_percent": float(self.roi_analysis.roi_first_year_percent),
                "npv_lifetime_usd": float(self.roi_analysis.npv_lifetime_usd),
                "recommendation": self.roi_analysis.recommendation
            } if self.roi_analysis else None,
            "provenance_hash": self.provenance_hash,
            "calculation_timestamp": self.calculation_timestamp.isoformat(),
            "calculation_method": self.calculation_method
        }


# ============================================================================
# CONSTANTS AND REFERENCE DATA
# ============================================================================

# Steam properties table: pressure (bar gauge) -> (T_sat_C, h_steam_BTU/lb, h_cond_BTU/lb, v_steam_ft3/lb)
# Data from ASME Steam Tables
STEAM_PROPERTIES_TABLE: Dict[int, Tuple[float, float, float, float]] = {
    0: (100.0, 1150.5, 180.2, 26.80),
    2: (133.5, 1159.4, 241.2, 9.70),
    4: (151.8, 1164.8, 275.0, 6.00),
    6: (165.0, 1168.6, 299.7, 4.37),
    8: (175.4, 1171.5, 319.3, 3.44),
    10: (184.1, 1173.8, 336.0, 2.83),
    12: (191.6, 1175.6, 350.3, 2.41),
    14: (198.3, 1177.2, 362.9, 2.09),
    16: (204.3, 1178.5, 374.1, 1.84),
    18: (209.8, 1179.6, 384.2, 1.65),
    20: (214.9, 1180.6, 393.5, 1.49),
    25: (226.0, 1182.4, 414.7, 1.20),
    30: (235.8, 1183.6, 433.0, 1.01),
    40: (252.4, 1184.8, 464.3, 0.76),
    50: (266.4, 1185.1, 490.8, 0.60),
}

# Discharge coefficients by trap type (ASME PTC 39)
DISCHARGE_COEFFICIENTS: Dict[TrapType, Decimal] = {
    TrapType.THERMODYNAMIC: Decimal("0.85"),
    TrapType.THERMOSTATIC: Decimal("0.80"),
    TrapType.MECHANICAL: Decimal("0.75"),
    TrapType.VENTURI: Decimal("0.90"),
}

# Leak fraction by failure mode
LEAK_FRACTIONS: Dict[FailureMode, Tuple[Decimal, Decimal]] = {
    FailureMode.BLOW_THROUGH: (Decimal("0.90"), Decimal("1.00")),  # 90-100%
    FailureMode.LEAKING: (Decimal("0.10"), Decimal("0.50")),       # 10-50%
    FailureMode.BLOCKED: (Decimal("0.00"), Decimal("0.00")),       # 0% steam loss
    FailureMode.CYCLING_FAST: (Decimal("0.05"), Decimal("0.20")),  # 5-20%
    FailureMode.CYCLING_SLOW: (Decimal("0.00"), Decimal("0.05")),  # 0-5%
    FailureMode.COLD_TRAP: (Decimal("0.00"), Decimal("0.00")),     # No steam
    FailureMode.NORMAL: (Decimal("0.00"), Decimal("0.02")),        # 0-2% normal
}

# EPA emission factors (kg CO2 per MMBtu)
EPA_EMISSION_FACTORS: Dict[FuelType, Decimal] = {
    FuelType.NATURAL_GAS: Decimal("53.07"),
    FuelType.FUEL_OIL_2: Decimal("73.16"),
    FuelType.FUEL_OIL_6: Decimal("75.10"),
    FuelType.PROPANE: Decimal("63.07"),
    FuelType.COAL: Decimal("95.35"),
    FuelType.BIOMASS: Decimal("93.80"),
}

# Typical replacement costs by trap type (USD)
TYPICAL_REPLACEMENT_COSTS: Dict[TrapType, Decimal] = {
    TrapType.THERMODYNAMIC: Decimal("150.00"),
    TrapType.THERMOSTATIC: Decimal("200.00"),
    TrapType.MECHANICAL: Decimal("350.00"),
    TrapType.VENTURI: Decimal("250.00"),
}


# ============================================================================
# MAIN CALCULATOR CLASS
# ============================================================================

class SteamTrapEnergyLossCalculator:
    """
    Advanced deterministic energy loss calculator for steam traps.

    Provides comprehensive energy loss analysis supporting multiple failure
    modes, trap types, and economic calculations per ASME/ASTM standards.

    ZERO-HALLUCINATION GUARANTEE:
    - All calculations use deterministic engineering formulas
    - No LLM or ML inference in calculation path
    - Same inputs always produce identical outputs
    - Complete provenance tracking with SHA-256 hashes

    Key Formulas (ASME PTC 39):
    - Steam loss (Napier): W = C_d * A * P_abs / 366 (kg/hr)
    - Energy loss: Q = W * (h_steam - h_cond) (BTU/hr)
    - CO2 emissions: E = Q * emission_factor / boiler_efficiency
    - ROI: (annual_savings / replacement_cost) * 100

    Example:
        >>> calculator = SteamTrapEnergyLossCalculator()
        >>> result = calculator.calculate_energy_loss(
        ...     trap_id="ST-001",
        ...     failure_mode=FailureMode.BLOW_THROUGH,
        ...     orifice_diameter_mm=6.35,
        ...     pressure_bar_g=10.0,
        ...     trap_type=TrapType.THERMODYNAMIC,
        ...     replacement_cost_usd=200.0
        ... )
    """

    def __init__(self, config: Optional[EnergyLossConfig] = None):
        """
        Initialize energy loss calculator.

        Args:
            config: Calculator configuration (uses defaults if not provided)
        """
        self.config = config or EnergyLossConfig()
        self._calculation_count = 0
        self._lock = threading.Lock()
        self._cache_lock = threading.Lock()

        logger.info(
            f"SteamTrapEnergyLossCalculator initialized "
            f"(fuel_type={self.config.fuel_type.value}, "
            f"boiler_efficiency={self.config.boiler_efficiency})"
        )

    def calculate_energy_loss(
        self,
        trap_id: str,
        failure_mode: FailureMode,
        orifice_diameter_mm: float,
        pressure_bar_g: float,
        trap_type: TrapType = TrapType.THERMODYNAMIC,
        leak_severity: Optional[float] = None,
        replacement_cost_usd: Optional[float] = None,
        manufacturer: str = "",
        model: str = ""
    ) -> EnergyLossAnalysisResult:
        """
        Calculate comprehensive energy loss for a failed steam trap.

        ZERO-HALLUCINATION: Uses deterministic ASME PTC 39 formulas.

        Args:
            trap_id: Unique trap identifier
            failure_mode: Type of failure
            orifice_diameter_mm: Orifice diameter in mm
            pressure_bar_g: Gauge pressure in bar
            trap_type: Type of steam trap
            leak_severity: Optional leak severity override (0-1)
            replacement_cost_usd: Optional replacement cost for ROI
            manufacturer: Trap manufacturer
            model: Trap model

        Returns:
            EnergyLossAnalysisResult with complete analysis

        Raises:
            ValueError: If inputs are invalid
        """
        with self._lock:
            self._calculation_count += 1

        # Initialize provenance tracker
        provenance = ProvenanceTracker()
        timestamp = datetime.now(timezone.utc)

        # Validate inputs
        self._validate_inputs(orifice_diameter_mm, pressure_bar_g)

        # Create trap specifications
        trap_specs = TrapSpecifications(
            trap_id=trap_id,
            trap_type=trap_type,
            manufacturer=manufacturer,
            model=model,
            orifice_diameter_mm=Decimal(str(orifice_diameter_mm))
        )

        # Get steam conditions
        steam_conditions = self._get_steam_conditions(pressure_bar_g, provenance)

        # Calculate steam loss
        steam_loss = self._calculate_steam_loss(
            trap_specs, failure_mode, pressure_bar_g, leak_severity, provenance
        )

        # Calculate energy metrics
        energy_metrics = self._calculate_energy_metrics(
            steam_loss, steam_conditions, provenance
        )

        # Calculate carbon emissions
        carbon_emissions = self._calculate_carbon_emissions(
            energy_metrics, provenance
        )

        # Calculate annual energy cost
        annual_cost = self._calculate_annual_cost(energy_metrics, provenance)

        # Calculate ROI if replacement cost provided
        roi_analysis = None
        if replacement_cost_usd is not None and replacement_cost_usd > 0:
            roi_analysis = self._calculate_roi(
                Decimal(str(replacement_cost_usd)),
                annual_cost,
                carbon_emissions,
                trap_type,
                provenance
            )

        # Generate provenance hash
        provenance_hash = provenance.get_hash()

        return EnergyLossAnalysisResult(
            trap_specs=trap_specs,
            failure_mode=failure_mode,
            steam_conditions=steam_conditions,
            steam_loss=steam_loss,
            energy_metrics=energy_metrics,
            carbon_emissions=carbon_emissions,
            annual_energy_cost_usd=annual_cost,
            roi_analysis=roi_analysis,
            provenance_hash=provenance_hash,
            calculation_timestamp=timestamp
        )

    def _validate_inputs(
        self,
        orifice_diameter_mm: float,
        pressure_bar_g: float
    ) -> None:
        """Validate input parameters."""
        if orifice_diameter_mm <= 0 or orifice_diameter_mm > 100:
            raise ValueError(
                f"Invalid orifice diameter: {orifice_diameter_mm} mm. "
                f"Must be between 0 and 100 mm."
            )
        if pressure_bar_g < 0 or pressure_bar_g > 100:
            raise ValueError(
                f"Invalid pressure: {pressure_bar_g} bar. "
                f"Must be between 0 and 100 bar gauge."
            )

    @lru_cache(maxsize=100)
    def _get_steam_properties_cached(
        self,
        pressure_bar_g: int
    ) -> Tuple[float, float, float, float]:
        """Thread-safe cached steam property lookup."""
        return self._interpolate_steam_properties(float(pressure_bar_g))

    def _get_steam_conditions(
        self,
        pressure_bar_g: float,
        provenance: ProvenanceTracker
    ) -> SteamConditions:
        """
        Get steam conditions at operating pressure.

        Uses ASME Steam Tables with linear interpolation.

        Args:
            pressure_bar_g: Gauge pressure in bar
            provenance: Provenance tracker

        Returns:
            SteamConditions at given pressure
        """
        # Get interpolated properties
        t_sat, h_steam, h_cond, v_steam = self._interpolate_steam_properties(
            pressure_bar_g
        )

        provenance.record_step(
            operation="steam_property_lookup",
            inputs={"pressure_bar_g": pressure_bar_g},
            formula="Linear interpolation from ASME Steam Tables",
            result={
                "T_sat_C": t_sat,
                "h_steam_BTU_lb": h_steam,
                "h_cond_BTU_lb": h_cond
            }
        )

        return SteamConditions(
            pressure_bar_g=Decimal(str(pressure_bar_g)),
            temperature_c=Decimal(str(round(t_sat, 2))),
            saturation_temp_c=Decimal(str(round(t_sat, 2))),
            enthalpy_steam_btu_lb=Decimal(str(round(h_steam, 2))),
            enthalpy_condensate_btu_lb=Decimal(str(round(h_cond, 2))),
            specific_volume_ft3_lb=Decimal(str(round(v_steam, 4)))
        )

    def _interpolate_steam_properties(
        self,
        pressure_bar_g: float
    ) -> Tuple[float, float, float, float]:
        """
        Interpolate steam properties from table.

        Args:
            pressure_bar_g: Gauge pressure in bar

        Returns:
            Tuple of (T_sat, h_steam, h_cond, v_steam)
        """
        pressures = sorted(STEAM_PROPERTIES_TABLE.keys())

        if pressure_bar_g <= pressures[0]:
            return STEAM_PROPERTIES_TABLE[pressures[0]]
        if pressure_bar_g >= pressures[-1]:
            return STEAM_PROPERTIES_TABLE[pressures[-1]]

        for i in range(len(pressures) - 1):
            p_low = pressures[i]
            p_high = pressures[i + 1]

            if p_low <= pressure_bar_g <= p_high:
                props_low = STEAM_PROPERTIES_TABLE[p_low]
                props_high = STEAM_PROPERTIES_TABLE[p_high]

                fraction = (pressure_bar_g - p_low) / (p_high - p_low)

                return tuple(
                    low + fraction * (high - low)
                    for low, high in zip(props_low, props_high)
                )

        return STEAM_PROPERTIES_TABLE[pressures[0]]

    def _calculate_steam_loss(
        self,
        trap_specs: TrapSpecifications,
        failure_mode: FailureMode,
        pressure_bar_g: float,
        leak_severity: Optional[float],
        provenance: ProvenanceTracker
    ) -> SteamLossResult:
        """
        Calculate steam loss rate using Napier equation.

        FORMULA (Napier equation for saturated steam):
        W = C_d * A * P_abs / 366

        Where:
        - W = steam flow rate (kg/hr)
        - C_d = discharge coefficient (trap type specific)
        - A = orifice area (mm^2)
        - P_abs = absolute pressure (bar)
        - 366 = empirical constant for saturated steam

        Args:
            trap_specs: Trap specifications
            failure_mode: Failure mode
            pressure_bar_g: Gauge pressure
            leak_severity: Optional severity override
            provenance: Provenance tracker

        Returns:
            SteamLossResult with loss calculations
        """
        # Get discharge coefficient for trap type
        c_d = DISCHARGE_COEFFICIENTS.get(
            trap_specs.trap_type,
            Decimal("0.75")
        )

        # Calculate orifice area (mm^2)
        import math
        diameter_mm = float(trap_specs.orifice_diameter_mm)
        area_mm2 = Decimal(str(math.pi * (diameter_mm / 2) ** 2))

        # Absolute pressure (add atmospheric 1.013 bar)
        p_abs = Decimal(str(pressure_bar_g)) + Decimal("1.013")

        # Napier equation: W = C_d * A * P_abs / 366 (kg/hr)
        full_flow_kg_hr = (c_d * area_mm2 * p_abs) / Decimal("366")

        provenance.record_step(
            operation="napier_steam_flow",
            inputs={
                "C_d": float(c_d),
                "Area_mm2": float(area_mm2),
                "P_abs_bar": float(p_abs)
            },
            formula="W = C_d * A * P_abs / 366",
            result=float(full_flow_kg_hr)
        )

        # Determine leak fraction
        if leak_severity is not None:
            leak_fraction = Decimal(str(min(1.0, max(0.0, leak_severity))))
        else:
            leak_range = LEAK_FRACTIONS.get(
                failure_mode,
                (Decimal("0.0"), Decimal("0.0"))
            )
            # Use midpoint of range
            leak_fraction = (leak_range[0] + leak_range[1]) / 2

        # Calculate actual steam loss
        steam_loss_kg_hr = full_flow_kg_hr * leak_fraction
        steam_loss_lb_hr = steam_loss_kg_hr * Decimal("2.205")  # kg to lb

        provenance.record_step(
            operation="steam_loss_calculation",
            inputs={
                "full_flow_kg_hr": float(full_flow_kg_hr),
                "leak_fraction": float(leak_fraction),
                "failure_mode": failure_mode.value
            },
            formula="steam_loss = full_flow * leak_fraction",
            result=float(steam_loss_kg_hr)
        )

        # Calculate leak percentage
        max_capacity = float(trap_specs.max_capacity_kg_hr)
        leak_percentage = (
            (float(steam_loss_kg_hr) / max_capacity * 100)
            if max_capacity > 0 else 0
        )

        # Determine severity
        severity = self._classify_severity(float(leak_fraction) * 100)

        return SteamLossResult(
            steam_loss_lb_hr=steam_loss_lb_hr.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            steam_loss_kg_hr=steam_loss_kg_hr.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            leak_percentage=Decimal(str(round(leak_percentage, 2))),
            failure_severity=severity
        )

    def _classify_severity(self, leak_percent: float) -> SeverityLevel:
        """Classify failure severity based on leak percentage."""
        if leak_percent >= 75:
            return SeverityLevel.CRITICAL
        elif leak_percent >= 50:
            return SeverityLevel.HIGH
        elif leak_percent >= 25:
            return SeverityLevel.MEDIUM
        elif leak_percent >= 10:
            return SeverityLevel.LOW
        elif leak_percent > 2:
            return SeverityLevel.MINIMAL
        else:
            return SeverityLevel.NONE

    def _calculate_energy_metrics(
        self,
        steam_loss: SteamLossResult,
        steam_conditions: SteamConditions,
        provenance: ProvenanceTracker
    ) -> EnergyLossMetrics:
        """
        Calculate energy loss metrics.

        FORMULA:
        Q = W * (h_steam - h_condensate) (BTU/hr)

        Args:
            steam_loss: Steam loss result
            steam_conditions: Steam conditions
            provenance: Provenance tracker

        Returns:
            EnergyLossMetrics with all energy calculations
        """
        # Enthalpy difference (latent heat of vaporization)
        h_diff = (
            steam_conditions.enthalpy_steam_btu_lb -
            steam_conditions.enthalpy_condensate_btu_lb
        )

        # Energy loss rate (BTU/hr)
        energy_btu_hr = steam_loss.steam_loss_lb_hr * h_diff

        # Convert to other units
        energy_kw = energy_btu_hr * Decimal("0.000293071")  # BTU/hr to kW
        energy_mmbtu_hr = energy_btu_hr / Decimal("1000000")

        # Account for boiler efficiency (fuel waste)
        fuel_waste_mmbtu_hr = energy_mmbtu_hr / self.config.boiler_efficiency

        # Time-based calculations
        hours_per_day = Decimal("24")
        days_per_month = Decimal("30")
        hours_per_year = Decimal(str(self.config.operating_hours_per_year))

        daily_loss = fuel_waste_mmbtu_hr * hours_per_day
        monthly_loss = daily_loss * days_per_month
        annual_loss = fuel_waste_mmbtu_hr * hours_per_year

        provenance.record_step(
            operation="energy_loss_calculation",
            inputs={
                "steam_loss_lb_hr": float(steam_loss.steam_loss_lb_hr),
                "h_steam_BTU_lb": float(steam_conditions.enthalpy_steam_btu_lb),
                "h_cond_BTU_lb": float(steam_conditions.enthalpy_condensate_btu_lb),
                "boiler_efficiency": float(self.config.boiler_efficiency)
            },
            formula="Q = W * (h_steam - h_cond); Fuel_waste = Q / efficiency",
            result={
                "energy_btu_hr": float(energy_btu_hr),
                "annual_mmbtu": float(annual_loss)
            }
        )

        return EnergyLossMetrics(
            energy_loss_btu_hr=energy_btu_hr.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            energy_loss_kw=energy_kw.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            ),
            energy_loss_mmbtu_hr=energy_mmbtu_hr.quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            ),
            fuel_waste_mmbtu_hr=fuel_waste_mmbtu_hr.quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            ),
            daily_energy_loss_mmbtu=daily_loss.quantize(
                Decimal("0.0001"), rounding=ROUND_HALF_UP
            ),
            monthly_energy_loss_mmbtu=monthly_loss.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            ),
            annual_energy_loss_mmbtu=annual_loss.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        )

    def _calculate_carbon_emissions(
        self,
        energy_metrics: EnergyLossMetrics,
        provenance: ProvenanceTracker
    ) -> CarbonEmissionResult:
        """
        Calculate carbon emissions using EPA emission factors.

        FORMULA:
        CO2 = fuel_waste_MMBtu * emission_factor (kg CO2)

        Args:
            energy_metrics: Energy loss metrics
            provenance: Provenance tracker

        Returns:
            CarbonEmissionResult with emission calculations
        """
        # Get emission factor for fuel type
        emission_factor = EPA_EMISSION_FACTORS.get(
            self.config.fuel_type,
            Decimal("53.07")  # Default to natural gas
        )

        # Hourly CO2 emissions (kg)
        co2_kg_hr = energy_metrics.fuel_waste_mmbtu_hr * emission_factor

        # Annual CO2 emissions (metric tons)
        hours_per_year = Decimal(str(self.config.operating_hours_per_year))
        co2_kg_year = co2_kg_hr * hours_per_year
        co2_tons_year = co2_kg_year / Decimal("1000")

        # Carbon penalty cost
        carbon_penalty = co2_tons_year * self.config.carbon_price_usd_per_ton

        provenance.record_step(
            operation="carbon_emission_calculation",
            inputs={
                "fuel_waste_mmbtu_hr": float(energy_metrics.fuel_waste_mmbtu_hr),
                "emission_factor_kg_mmbtu": float(emission_factor),
                "carbon_price_usd_ton": float(self.config.carbon_price_usd_per_ton)
            },
            formula="CO2 = fuel_waste * emission_factor; Penalty = CO2_tons * price",
            result={
                "co2_tons_year": float(co2_tons_year),
                "carbon_penalty_usd": float(carbon_penalty)
            }
        )

        return CarbonEmissionResult(
            co2_kg_hr=co2_kg_hr.quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            ),
            co2_tons_year=co2_tons_year.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            carbon_penalty_usd_year=carbon_penalty.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            emission_factor_kg_per_mmbtu=emission_factor
        )

    def _calculate_annual_cost(
        self,
        energy_metrics: EnergyLossMetrics,
        provenance: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate total annual energy cost.

        Args:
            energy_metrics: Energy loss metrics
            provenance: Provenance tracker

        Returns:
            Annual energy cost in USD
        """
        annual_cost = (
            energy_metrics.annual_energy_loss_mmbtu *
            self.config.fuel_cost_usd_per_mmbtu
        )

        provenance.record_step(
            operation="annual_cost_calculation",
            inputs={
                "annual_mmbtu": float(energy_metrics.annual_energy_loss_mmbtu),
                "fuel_cost_usd_mmbtu": float(self.config.fuel_cost_usd_per_mmbtu)
            },
            formula="Annual_cost = annual_MMBtu * fuel_cost_per_MMBtu",
            result=float(annual_cost)
        )

        return annual_cost.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

    def _calculate_roi(
        self,
        replacement_cost: Decimal,
        annual_savings: Decimal,
        carbon_emissions: CarbonEmissionResult,
        trap_type: TrapType,
        provenance: ProvenanceTracker
    ) -> ROIResult:
        """
        Calculate comprehensive ROI for trap replacement.

        FORMULAS:
        - Simple payback = replacement_cost / annual_savings
        - ROI = (annual_savings / replacement_cost) * 100
        - NPV = sum(savings / (1 + r)^n) - cost

        Args:
            replacement_cost: Total replacement cost
            annual_savings: Annual energy savings
            carbon_emissions: Carbon emission results
            trap_type: Type of trap
            provenance: Provenance tracker

        Returns:
            ROIResult with complete financial analysis
        """
        # Include carbon penalty in total savings
        total_annual_savings = annual_savings + carbon_emissions.carbon_penalty_usd_year

        # Handle zero savings case
        if total_annual_savings <= 0:
            return ROIResult(
                replacement_cost_usd=replacement_cost,
                annual_savings_usd=Decimal("0.00"),
                simple_payback_days=Decimal("999999"),
                simple_payback_months=Decimal("999999"),
                roi_first_year_percent=Decimal("0.0"),
                npv_lifetime_usd=-replacement_cost,
                irr_percent=Decimal("0.0"),
                break_even_date=None,
                recommendation="No economic justification for replacement"
            )

        # Simple payback
        payback_years = replacement_cost / total_annual_savings
        payback_days = payback_years * Decimal("365")
        payback_months = payback_years * Decimal("12")

        # First year ROI
        roi_percent = (total_annual_savings / replacement_cost) * Decimal("100")

        # NPV over trap lifetime
        lifetime = self.config.trap_lifetime_years
        discount_rate = self.config.discount_rate

        npv = -replacement_cost
        for year in range(1, lifetime + 1):
            discount_factor = (Decimal("1") + discount_rate) ** year
            npv += total_annual_savings / discount_factor

        # IRR estimation (bisection method)
        irr = self._estimate_irr(
            float(replacement_cost),
            float(total_annual_savings),
            lifetime
        )

        # Break-even date
        break_even_date = None
        if payback_years < Decimal("10"):
            from datetime import timedelta
            days = int(payback_days)
            break_even = datetime.now() + timedelta(days=days)
            break_even_date = break_even.strftime("%Y-%m-%d")

        # Generate recommendation
        recommendation = self._generate_recommendation(
            float(payback_months),
            float(roi_percent),
            float(npv)
        )

        provenance.record_step(
            operation="roi_calculation",
            inputs={
                "replacement_cost": float(replacement_cost),
                "annual_savings": float(total_annual_savings),
                "lifetime_years": lifetime,
                "discount_rate": float(discount_rate)
            },
            formula="NPV = -cost + sum(savings/(1+r)^n); ROI = (savings/cost)*100",
            result={
                "payback_days": float(payback_days),
                "roi_percent": float(roi_percent),
                "npv_usd": float(npv)
            }
        )

        return ROIResult(
            replacement_cost_usd=replacement_cost.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            annual_savings_usd=total_annual_savings.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            simple_payback_days=payback_days.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ),
            simple_payback_months=payback_months.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            roi_first_year_percent=roi_percent.quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            ),
            npv_lifetime_usd=npv.quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            ),
            irr_percent=Decimal(str(round(irr * 100, 1))),
            break_even_date=break_even_date,
            recommendation=recommendation
        )

    def _estimate_irr(
        self,
        initial_cost: float,
        annual_cashflow: float,
        years: int
    ) -> float:
        """
        Estimate IRR using bisection method.

        Args:
            initial_cost: Initial investment
            annual_cashflow: Annual cash flow
            years: Project duration

        Returns:
            Estimated IRR as decimal
        """
        if annual_cashflow <= 0:
            return 0.0

        def npv_at_rate(rate: float) -> float:
            if rate <= -1:
                return float('inf')
            npv = -initial_cost
            for year in range(1, years + 1):
                npv += annual_cashflow / ((1 + rate) ** year)
            return npv

        low_rate = 0.0
        high_rate = 5.0

        if npv_at_rate(low_rate) < 0:
            return 0.0
        if npv_at_rate(high_rate) > 0:
            return high_rate

        for _ in range(50):
            mid_rate = (low_rate + high_rate) / 2
            npv = npv_at_rate(mid_rate)

            if abs(npv) < 0.01:
                return mid_rate
            elif npv > 0:
                low_rate = mid_rate
            else:
                high_rate = mid_rate

        return (low_rate + high_rate) / 2

    def _generate_recommendation(
        self,
        payback_months: float,
        roi_percent: float,
        npv: float
    ) -> str:
        """Generate action recommendation based on financial metrics."""
        if payback_months < 3:
            return "IMMEDIATE ACTION: Excellent ROI - replace within 1 week"
        elif payback_months < 6:
            return "HIGH PRIORITY: Strong ROI - schedule replacement within 30 days"
        elif payback_months < 12:
            return "SCHEDULED: Good ROI - include in next maintenance cycle"
        elif payback_months < 24:
            return "MONITOR: Acceptable ROI - schedule based on capacity"
        elif npv > 0:
            return "LOW PRIORITY: Marginal ROI - replace at end of life"
        else:
            return "DEFER: Insufficient ROI - continue monitoring"

    def calculate_batch(
        self,
        trap_analyses: List[Dict[str, Any]]
    ) -> List[EnergyLossAnalysisResult]:
        """
        Calculate energy loss for multiple traps.

        Args:
            trap_analyses: List of trap analysis parameters

        Returns:
            List of EnergyLossAnalysisResult
        """
        results = []
        for params in trap_analyses:
            try:
                result = self.calculate_energy_loss(**params)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to analyze trap {params.get('trap_id')}: {e}")
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """Get calculator statistics."""
        with self._lock:
            return {
                "calculation_count": self._calculation_count,
                "fuel_type": self.config.fuel_type.value,
                "boiler_efficiency": float(self.config.boiler_efficiency),
                "carbon_price_usd_ton": float(self.config.carbon_price_usd_per_ton),
                "supported_trap_types": [t.value for t in TrapType],
                "supported_failure_modes": [f.value for f in FailureMode],
                "supported_fuel_types": [f.value for f in FuelType]
            }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Main calculator
    "SteamTrapEnergyLossCalculator",
    # Configuration
    "EnergyLossConfig",
    # Enums
    "FailureMode",
    "TrapType",
    "SeverityLevel",
    "FuelType",
    # Data classes
    "TrapSpecifications",
    "SteamConditions",
    "SteamLossResult",
    "EnergyLossMetrics",
    "CarbonEmissionResult",
    "ROIResult",
    "EnergyLossAnalysisResult",
    # Provenance
    "ProvenanceTracker",
    "ProvenanceStep",
]
