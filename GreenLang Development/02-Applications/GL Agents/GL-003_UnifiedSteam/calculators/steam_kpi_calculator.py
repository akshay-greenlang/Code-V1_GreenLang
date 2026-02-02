"""
GL-003 UNIFIEDSTEAM - Steam KPI Calculator

Key Performance Indicators for steam system optimization and monitoring.

KPIs Implemented:
- Steam system efficiency
- Specific steam consumption
- Condensate return ratio
- Desuperheater performance
- Trap health index
- Energy intensity metrics

Reference: ISO 50001 Energy Management, ASME PTC 19.1, Industrial Best Practices

Author: GL-CalculatorEngineer
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

class KPIStatus(str, Enum):
    """KPI performance status."""
    EXCELLENT = "EXCELLENT"
    GOOD = "GOOD"
    ACCEPTABLE = "ACCEPTABLE"
    POOR = "POOR"
    CRITICAL = "CRITICAL"


class TrendDirection(str, Enum):
    """KPI trend direction."""
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DECLINING = "DECLINING"
    UNKNOWN = "UNKNOWN"


# Industry benchmark ranges for steam system KPIs
KPI_BENCHMARKS = {
    "steam_system_efficiency": {
        "excellent": (85, 100),
        "good": (80, 85),
        "acceptable": (70, 80),
        "poor": (60, 70),
        "critical": (0, 60),
    },
    "condensate_return_rate": {
        "excellent": (85, 100),
        "good": (70, 85),
        "acceptable": (50, 70),
        "poor": (30, 50),
        "critical": (0, 30),
    },
    "trap_health_rate": {
        "excellent": (95, 100),
        "good": (90, 95),
        "acceptable": (80, 90),
        "poor": (70, 80),
        "critical": (0, 70),
    },
    "desuperheater_temp_accuracy": {
        "excellent": (0, 2),    # Degrees C from target
        "good": (2, 5),
        "acceptable": (5, 10),
        "poor": (10, 20),
        "critical": (20, 100),
    },
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class KPIInput:
    """Input data for KPI calculations."""

    # Time period
    period_start: datetime
    period_end: datetime

    # Energy data
    fuel_input_kw: float
    useful_heat_kw: float

    # Steam production
    steam_produced_kg: float
    production_output_units: float
    production_unit_name: str = "tonnes"

    # Condensate data
    condensate_generated_kg: float
    condensate_returned_kg: float

    # Trap data
    total_traps: int
    healthy_traps: int
    trap_loss_rate_kg_s: float

    # Desuperheater data (optional)
    desup_target_temp_c: Optional[float] = None
    desup_actual_temp_c: Optional[float] = None
    desup_spray_flow_kg_s: Optional[float] = None


@dataclass
class PerformanceMetrics:
    """Desuperheater performance metrics."""

    calculation_id: str
    timestamp: datetime

    # Temperature control
    target_temperature_c: float
    actual_temperature_c: float
    temperature_deviation_c: float
    temperature_accuracy_percent: float

    # Control performance
    control_error_rms_c: float
    overshoot_percent: float
    settling_time_s: float

    # Spray performance
    spray_flow_kg_s: float
    spray_utilization_percent: float

    # Status
    performance_status: KPIStatus

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class TrapHealthKPI:
    """Steam trap health KPI."""

    calculation_id: str
    timestamp: datetime

    # Trap counts
    total_traps: int
    healthy_traps: int
    failed_traps: int
    marginal_traps: int

    # Health rates
    health_rate_percent: float
    failure_rate_percent: float

    # Weighted health (considers loss severity)
    weighted_health_score: float
    total_loss_rate_kg_s: float

    # Economic impact
    annual_loss_cost: float

    # Status
    health_status: KPIStatus
    trend: TrendDirection

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class EnergyKPI:
    """Energy-related KPI."""

    calculation_id: str
    timestamp: datetime

    # Efficiency
    steam_system_efficiency_percent: float
    efficiency_uncertainty_percent: float

    # Specific consumption
    specific_steam_consumption: float  # kg steam / unit production
    specific_energy_consumption: float  # kJ / unit production

    # Benchmarks
    benchmark_efficiency: float
    efficiency_gap_percent: float

    # Status
    efficiency_status: KPIStatus
    trend: TrendDirection

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class CondensateKPI:
    """Condensate recovery KPI."""

    calculation_id: str
    timestamp: datetime

    # Return rates
    return_rate_actual: float
    return_rate_potential: float
    return_rate_gap: float

    # Recovery metrics
    condensate_generated_kg_s: float
    condensate_returned_kg_s: float
    condensate_lost_kg_s: float

    # Heat recovery
    heat_recovered_kw: float
    potential_heat_recovery_kw: float
    recovery_efficiency_percent: float

    # Status
    recovery_status: KPIStatus
    trend: TrendDirection

    # Provenance
    input_hash: str
    output_hash: str


@dataclass
class KPIDashboard:
    """Complete KPI dashboard."""

    calculation_id: str
    timestamp: datetime
    period_description: str

    # Overall health score (0-100)
    overall_health_score: float
    overall_status: KPIStatus

    # Individual KPIs
    energy_kpi: EnergyKPI
    condensate_kpi: CondensateKPI
    trap_health_kpi: TrapHealthKPI
    desuperheater_kpi: Optional[PerformanceMetrics]

    # Summary metrics
    total_efficiency_percent: float
    total_losses_kw: float
    annual_savings_potential: float

    # Top improvement opportunities
    improvement_priorities: List[Dict[str, Any]]

    # Alerts
    active_alerts: List[str]

    # Provenance
    input_hash: str
    output_hash: str


# =============================================================================
# STEAM KPI CALCULATOR
# =============================================================================

class SteamKPICalculator:
    """
    Zero-hallucination steam system KPI calculator.

    Implements deterministic KPI calculations for:
    - Steam system efficiency
    - Specific steam consumption
    - Condensate return ratio
    - Desuperheater performance
    - Trap health index
    - Dashboard generation

    All calculations use:
    - Industry-standard formulas
    - SHA-256 provenance hashing
    - Complete audit trails
    - NO LLM in calculation path

    Example:
        >>> calc = SteamKPICalculator()
        >>> efficiency = calc.compute_steam_system_efficiency(
        ...     useful_heat=5000,  # kW
        ...     fuel_input=6000    # kW
        ... )
        >>> print(f"Efficiency: {efficiency:.1f}%")
    """

    VERSION = "1.0.0"
    FORMULA_VERSION = "KPI_V1.0"

    def __init__(
        self,
        steam_cost_per_kg: float = 0.03,
        operating_hours_per_year: int = 8000,
        benchmarks: Optional[Dict] = None,
    ) -> None:
        """
        Initialize KPI calculator.

        Args:
            steam_cost_per_kg: Cost of steam ($/kg)
            operating_hours_per_year: Annual operating hours
            benchmarks: Custom benchmark ranges
        """
        self.steam_cost = steam_cost_per_kg
        self.operating_hours = operating_hours_per_year
        self.benchmarks = benchmarks or KPI_BENCHMARKS

    def compute_steam_system_efficiency(
        self,
        useful_heat: float,
        fuel_input: float,
    ) -> Tuple[float, KPIStatus]:
        """
        Compute overall steam system efficiency.

        Formula:
            efficiency = useful_heat / fuel_input * 100

        DETERMINISTIC calculation.

        Args:
            useful_heat: Useful heat output (kW)
            fuel_input: Fuel heat input (kW)

        Returns:
            Tuple of (efficiency_percent, status)
        """
        if fuel_input <= 0:
            raise ValueError("Fuel input must be positive")

        efficiency = useful_heat / fuel_input * 100
        efficiency = min(100, max(0, efficiency))

        # Determine status
        status = self._classify_kpi(efficiency, "steam_system_efficiency")

        return round(efficiency, 2), status

    def compute_specific_steam_consumption(
        self,
        steam_flow: float,
        production_output: float,
    ) -> Tuple[float, str]:
        """
        Compute specific steam consumption.

        Formula:
            SSC = steam_flow / production_output

        DETERMINISTIC calculation.

        Args:
            steam_flow: Total steam consumption (kg or kg/s)
            production_output: Production output (units)

        Returns:
            Tuple of (specific_consumption, unit_string)
        """
        if production_output <= 0:
            raise ValueError("Production output must be positive")

        ssc = steam_flow / production_output

        return round(ssc, 3), "kg_steam/unit"

    def compute_condensate_return_kpi(
        self,
        return_rate_actual: float,
        return_rate_potential: float,
    ) -> Tuple[float, KPIStatus, float]:
        """
        Compute condensate return KPI.

        Formula:
            KPI = return_rate_actual / return_rate_potential * 100

        DETERMINISTIC calculation.

        Args:
            return_rate_actual: Actual return rate (0-1)
            return_rate_potential: Potential return rate (0-1)

        Returns:
            Tuple of (kpi_percent, status, gap_percent)
        """
        if return_rate_potential <= 0:
            raise ValueError("Potential return rate must be positive")

        if return_rate_actual < 0 or return_rate_actual > 1:
            raise ValueError("Actual return rate must be between 0 and 1")

        kpi = return_rate_actual / return_rate_potential * 100
        kpi = min(100, max(0, kpi))

        gap = (return_rate_potential - return_rate_actual) * 100

        # Classify based on actual return rate (as percentage)
        status = self._classify_kpi(return_rate_actual * 100, "condensate_return_rate")

        return round(kpi, 1), status, round(gap, 1)

    def compute_desuperheater_performance(
        self,
        actual_temp: float,
        target_temp: float,
        spray_flow: float,
        max_spray_flow: float = 10.0,
        historical_data: Optional[List[float]] = None,
    ) -> PerformanceMetrics:
        """
        Compute desuperheater temperature control performance.

        Metrics:
        - Temperature deviation from target
        - Control error RMS
        - Spray utilization

        DETERMINISTIC calculation.

        Args:
            actual_temp: Actual outlet temperature (C)
            target_temp: Target temperature (C)
            spray_flow: Current spray flow (kg/s)
            max_spray_flow: Maximum spray capacity (kg/s)
            historical_data: List of historical temp deviations (optional)

        Returns:
            PerformanceMetrics with control performance
        """
        # Temperature deviation
        deviation = actual_temp - target_temp

        # Accuracy percentage (100% = perfect, 0% = 10C off)
        accuracy = max(0, 100 - abs(deviation) * 10)

        # Calculate RMS error if historical data provided
        if historical_data and len(historical_data) > 0:
            rms_error = math.sqrt(sum(d**2 for d in historical_data) / len(historical_data))

            # Overshoot (max positive deviation)
            overshoot = max(0, max(historical_data)) if historical_data else 0

            # Estimate settling time (simplified)
            settling_time = len([d for d in historical_data if abs(d) > 2]) * 1.0
        else:
            rms_error = abs(deviation)
            overshoot = max(0, deviation)
            settling_time = 0.0

        # Spray utilization
        spray_utilization = spray_flow / max_spray_flow * 100 if max_spray_flow > 0 else 0

        # Determine status
        status = self._classify_kpi(abs(deviation), "desuperheater_temp_accuracy")

        # Compute hashes
        input_hash = self._compute_hash({
            "actual_temp": actual_temp,
            "target_temp": target_temp,
            "spray_flow": spray_flow,
        })

        output_hash = self._compute_hash({
            "deviation": deviation,
            "accuracy": accuracy,
        })

        return PerformanceMetrics(
            calculation_id=f"DESUPKPI-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            target_temperature_c=target_temp,
            actual_temperature_c=actual_temp,
            temperature_deviation_c=round(deviation, 2),
            temperature_accuracy_percent=round(accuracy, 1),
            control_error_rms_c=round(rms_error, 2),
            overshoot_percent=round(overshoot, 2),
            settling_time_s=round(settling_time, 1),
            spray_flow_kg_s=spray_flow,
            spray_utilization_percent=round(spray_utilization, 1),
            performance_status=status,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def compute_trap_health_kpi(
        self,
        healthy_traps: int,
        total_traps: int,
        weighted_loss_rate: float,
        failed_traps: int = 0,
        marginal_traps: int = 0,
        previous_health_rate: Optional[float] = None,
    ) -> TrapHealthKPI:
        """
        Compute steam trap health KPI.

        Metrics:
        - Simple health rate (healthy/total)
        - Weighted health score (considers loss severity)

        DETERMINISTIC calculation.

        Args:
            healthy_traps: Number of healthy traps
            total_traps: Total number of traps
            weighted_loss_rate: Total weighted steam loss (kg/s)
            failed_traps: Number of failed traps
            marginal_traps: Number of marginal traps
            previous_health_rate: Previous period health rate (for trend)

        Returns:
            TrapHealthKPI with health assessment
        """
        if total_traps <= 0:
            raise ValueError("Total traps must be positive")

        # Simple health rate
        health_rate = healthy_traps / total_traps * 100

        # Failure rate
        failure_rate = (total_traps - healthy_traps) / total_traps * 100

        # Weighted health score
        # Penalizes high loss rates more heavily
        # Score = health_rate - loss_penalty
        # Loss penalty = weighted_loss_rate * 100 (approx 1% per 0.01 kg/s)
        loss_penalty = weighted_loss_rate * 100
        weighted_score = max(0, health_rate - loss_penalty)

        # Annual loss cost
        annual_loss_cost = weighted_loss_rate * self.steam_cost * 3600 * self.operating_hours

        # Determine status
        status = self._classify_kpi(health_rate, "trap_health_rate")

        # Determine trend
        if previous_health_rate is not None:
            if health_rate > previous_health_rate + 2:
                trend = TrendDirection.IMPROVING
            elif health_rate < previous_health_rate - 2:
                trend = TrendDirection.DECLINING
            else:
                trend = TrendDirection.STABLE
        else:
            trend = TrendDirection.UNKNOWN

        # Compute hashes
        input_hash = self._compute_hash({
            "healthy_traps": healthy_traps,
            "total_traps": total_traps,
            "weighted_loss_rate": weighted_loss_rate,
        })

        output_hash = self._compute_hash({
            "health_rate": health_rate,
            "weighted_score": weighted_score,
        })

        return TrapHealthKPI(
            calculation_id=f"TRAPKPI-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            total_traps=total_traps,
            healthy_traps=healthy_traps,
            failed_traps=failed_traps,
            marginal_traps=marginal_traps,
            health_rate_percent=round(health_rate, 1),
            failure_rate_percent=round(failure_rate, 1),
            weighted_health_score=round(weighted_score, 1),
            total_loss_rate_kg_s=round(weighted_loss_rate, 4),
            annual_loss_cost=round(annual_loss_cost, 0),
            health_status=status,
            trend=trend,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def generate_kpi_dashboard(
        self,
        all_kpis: Dict[str, Any],
    ) -> KPIDashboard:
        """
        Generate comprehensive KPI dashboard.

        Aggregates all KPIs into a single dashboard with:
        - Overall health score
        - Individual KPI summaries
        - Improvement priorities
        - Active alerts

        DETERMINISTIC aggregation.

        Args:
            all_kpis: Dictionary containing individual KPI results

        Returns:
            KPIDashboard with complete overview
        """
        # Calculate energy KPI
        energy_kpi = self._calculate_energy_kpi(all_kpis)

        # Calculate condensate KPI
        condensate_kpi = self._calculate_condensate_kpi(all_kpis)

        # Get trap health KPI
        trap_kpi = all_kpis.get("trap_health_kpi")
        if trap_kpi is None:
            trap_kpi = TrapHealthKPI(
                calculation_id="N/A",
                timestamp=datetime.now(timezone.utc),
                total_traps=0,
                healthy_traps=0,
                failed_traps=0,
                marginal_traps=0,
                health_rate_percent=0,
                failure_rate_percent=0,
                weighted_health_score=0,
                total_loss_rate_kg_s=0,
                annual_loss_cost=0,
                health_status=KPIStatus.CRITICAL,
                trend=TrendDirection.UNKNOWN,
                input_hash="",
                output_hash="",
            )

        # Get desuperheater KPI (optional)
        desup_kpi = all_kpis.get("desuperheater_kpi")

        # Calculate overall health score (weighted average)
        scores = []
        weights = []

        # Energy efficiency contributes 40%
        scores.append(energy_kpi.steam_system_efficiency_percent)
        weights.append(0.40)

        # Condensate return contributes 25%
        scores.append(condensate_kpi.return_rate_actual * 100)
        weights.append(0.25)

        # Trap health contributes 25%
        scores.append(trap_kpi.health_rate_percent)
        weights.append(0.25)

        # Desuperheater contributes 10%
        if desup_kpi:
            scores.append(desup_kpi.temperature_accuracy_percent)
            weights.append(0.10)
        else:
            # Redistribute weight
            weights[0] += 0.05
            weights[1] += 0.025
            weights[2] += 0.025

        # Weighted average
        overall_score = sum(s * w for s, w in zip(scores, weights))

        # Determine overall status
        if overall_score >= 85:
            overall_status = KPIStatus.EXCELLENT
        elif overall_score >= 75:
            overall_status = KPIStatus.GOOD
        elif overall_score >= 60:
            overall_status = KPIStatus.ACCEPTABLE
        elif overall_score >= 45:
            overall_status = KPIStatus.POOR
        else:
            overall_status = KPIStatus.CRITICAL

        # Calculate total losses
        total_losses = (
            trap_kpi.total_loss_rate_kg_s * self.h_fg_ref +
            condensate_kpi.condensate_lost_kg_s * 400  # Approx sensible heat
        )

        # Calculate total efficiency
        total_efficiency = energy_kpi.steam_system_efficiency_percent

        # Estimate annual savings potential
        # Based on gap to best practice
        efficiency_gap = 90 - energy_kpi.steam_system_efficiency_percent
        condensate_gap = 0.85 - condensate_kpi.return_rate_actual
        trap_gap = 0.95 - trap_kpi.health_rate_percent / 100

        annual_savings = (
            max(0, efficiency_gap) * all_kpis.get("fuel_input_kw", 0) * self.operating_hours / 1000 * 5 +
            max(0, condensate_gap) * all_kpis.get("steam_flow_kg_s", 0) * self.steam_cost * 3600 * self.operating_hours +
            trap_kpi.annual_loss_cost
        )

        # Generate improvement priorities
        priorities = []

        if energy_kpi.efficiency_status in [KPIStatus.POOR, KPIStatus.CRITICAL]:
            priorities.append({
                "area": "Boiler Efficiency",
                "status": energy_kpi.efficiency_status.value,
                "gap": energy_kpi.efficiency_gap_percent,
                "savings_potential": efficiency_gap * all_kpis.get("fuel_input_kw", 0) * self.operating_hours / 1000 * 5,
                "recommended_action": "Optimize combustion, reduce excess air, improve heat recovery",
            })

        if condensate_kpi.recovery_status in [KPIStatus.POOR, KPIStatus.CRITICAL]:
            priorities.append({
                "area": "Condensate Recovery",
                "status": condensate_kpi.recovery_status.value,
                "gap": condensate_kpi.return_rate_gap * 100,
                "savings_potential": condensate_gap * all_kpis.get("steam_flow_kg_s", 0) * self.steam_cost * 3600 * self.operating_hours,
                "recommended_action": "Repair/install condensate recovery equipment",
            })

        if trap_kpi.health_status in [KPIStatus.POOR, KPIStatus.CRITICAL]:
            priorities.append({
                "area": "Steam Trap Health",
                "status": trap_kpi.health_status.value,
                "gap": 100 - trap_kpi.health_rate_percent,
                "savings_potential": trap_kpi.annual_loss_cost,
                "recommended_action": "Replace failed traps, implement monitoring program",
            })

        # Sort by savings potential
        priorities.sort(key=lambda x: x["savings_potential"], reverse=True)

        # Generate alerts
        alerts = []

        if energy_kpi.efficiency_status == KPIStatus.CRITICAL:
            alerts.append("CRITICAL: Steam system efficiency below 60%")

        if trap_kpi.health_status == KPIStatus.CRITICAL:
            alerts.append(f"CRITICAL: {trap_kpi.failed_traps} failed steam traps detected")

        if condensate_kpi.recovery_status == KPIStatus.CRITICAL:
            alerts.append("CRITICAL: Condensate return rate below 30%")

        if desup_kpi and desup_kpi.performance_status == KPIStatus.CRITICAL:
            alerts.append("CRITICAL: Desuperheater temperature deviation >20C")

        # Compute hashes
        input_hash = self._compute_hash({
            "kpi_keys": list(all_kpis.keys()),
        })

        output_hash = self._compute_hash({
            "overall_score": overall_score,
            "total_losses": total_losses,
        })

        return KPIDashboard(
            calculation_id=f"DASH-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            period_description=all_kpis.get("period_description", "Current Period"),
            overall_health_score=round(overall_score, 1),
            overall_status=overall_status,
            energy_kpi=energy_kpi,
            condensate_kpi=condensate_kpi,
            trap_health_kpi=trap_kpi,
            desuperheater_kpi=desup_kpi,
            total_efficiency_percent=round(total_efficiency, 1),
            total_losses_kw=round(total_losses, 1),
            annual_savings_potential=round(annual_savings, 0),
            improvement_priorities=priorities[:5],
            active_alerts=alerts,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    # =========================================================================
    # PRIVATE HELPER METHODS
    # =========================================================================

    # Reference enthalpy of vaporization (kJ/kg at ~1 MPa)
    h_fg_ref = 2015.0

    def _classify_kpi(
        self,
        value: float,
        kpi_name: str,
    ) -> KPIStatus:
        """Classify KPI value against benchmarks."""
        benchmarks = self.benchmarks.get(kpi_name, {})

        # For temperature accuracy, lower is better
        if "temp" in kpi_name.lower():
            if value <= benchmarks.get("excellent", (0, 2))[1]:
                return KPIStatus.EXCELLENT
            elif value <= benchmarks.get("good", (2, 5))[1]:
                return KPIStatus.GOOD
            elif value <= benchmarks.get("acceptable", (5, 10))[1]:
                return KPIStatus.ACCEPTABLE
            elif value <= benchmarks.get("poor", (10, 20))[1]:
                return KPIStatus.POOR
            else:
                return KPIStatus.CRITICAL
        else:
            # For efficiency/rate metrics, higher is better
            if value >= benchmarks.get("excellent", (85, 100))[0]:
                return KPIStatus.EXCELLENT
            elif value >= benchmarks.get("good", (80, 85))[0]:
                return KPIStatus.GOOD
            elif value >= benchmarks.get("acceptable", (70, 80))[0]:
                return KPIStatus.ACCEPTABLE
            elif value >= benchmarks.get("poor", (60, 70))[0]:
                return KPIStatus.POOR
            else:
                return KPIStatus.CRITICAL

    def _calculate_energy_kpi(self, all_kpis: Dict[str, Any]) -> EnergyKPI:
        """Calculate energy efficiency KPI from raw data."""
        useful_heat = all_kpis.get("useful_heat_kw", 0)
        fuel_input = all_kpis.get("fuel_input_kw", 1)  # Avoid div by zero

        efficiency, status = self.compute_steam_system_efficiency(useful_heat, fuel_input)

        # Calculate specific consumption
        steam_flow = all_kpis.get("steam_flow_kg_s", 0)
        production = all_kpis.get("production_output", 1)

        ssc, _ = self.compute_specific_steam_consumption(steam_flow * 3600, production)

        # Calculate specific energy
        sec = fuel_input / production if production > 0 else 0

        # Benchmark comparison
        benchmark_efficiency = 85.0  # Industry best practice
        efficiency_gap = benchmark_efficiency - efficiency

        # Trend (would need historical data)
        trend = TrendDirection.UNKNOWN

        # Compute hashes
        input_hash = self._compute_hash({
            "useful_heat_kw": useful_heat,
            "fuel_input_kw": fuel_input,
        })

        output_hash = self._compute_hash({
            "efficiency": efficiency,
        })

        return EnergyKPI(
            calculation_id=f"ENKPI-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            steam_system_efficiency_percent=efficiency,
            efficiency_uncertainty_percent=1.5,
            specific_steam_consumption=ssc,
            specific_energy_consumption=round(sec, 2),
            benchmark_efficiency=benchmark_efficiency,
            efficiency_gap_percent=round(efficiency_gap, 1),
            efficiency_status=status,
            trend=trend,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def _calculate_condensate_kpi(self, all_kpis: Dict[str, Any]) -> CondensateKPI:
        """Calculate condensate recovery KPI from raw data."""
        returned = all_kpis.get("condensate_returned_kg_s", 0)
        generated = all_kpis.get("condensate_generated_kg_s", 1)  # Avoid div by zero

        return_rate = returned / generated if generated > 0 else 0
        potential_rate = 0.85  # Industry best practice

        kpi, status, gap = self.compute_condensate_return_kpi(return_rate, potential_rate)

        lost = generated - returned

        # Heat recovery calculation
        # Condensate at ~100C (h=419), makeup at 15C (h=63)
        delta_h = 419 - 63
        heat_recovered = returned * delta_h
        potential_heat = generated * delta_h * potential_rate

        recovery_eff = heat_recovered / potential_heat * 100 if potential_heat > 0 else 0

        # Trend (would need historical data)
        trend = TrendDirection.UNKNOWN

        # Compute hashes
        input_hash = self._compute_hash({
            "returned": returned,
            "generated": generated,
        })

        output_hash = self._compute_hash({
            "return_rate": return_rate,
        })

        return CondensateKPI(
            calculation_id=f"CONDKPI-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}",
            timestamp=datetime.now(timezone.utc),
            return_rate_actual=round(return_rate, 3),
            return_rate_potential=potential_rate,
            return_rate_gap=round(potential_rate - return_rate, 3),
            condensate_generated_kg_s=generated,
            condensate_returned_kg_s=returned,
            condensate_lost_kg_s=round(lost, 4),
            heat_recovered_kw=round(heat_recovered, 1),
            potential_heat_recovery_kw=round(potential_heat, 1),
            recovery_efficiency_percent=round(recovery_eff, 1),
            recovery_status=status,
            trend=trend,
            input_hash=input_hash,
            output_hash=output_hash,
        )

    def _compute_hash(self, data: Dict[str, Any]) -> str:
        """Compute SHA-256 hash for provenance tracking."""
        json_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(json_str.encode()).hexdigest()[:16]
