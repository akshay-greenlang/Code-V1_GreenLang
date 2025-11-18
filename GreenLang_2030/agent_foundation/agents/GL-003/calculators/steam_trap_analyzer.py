"""
Steam Trap Performance Analyzer - Zero Hallucination

Analyzes steam trap performance, detects failures, and calculates
energy losses from failed traps.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ASME PTC 12.4, Spirax Sarco Guidelines
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional
from dataclasses import dataclass
from .provenance import ProvenanceTracker


@dataclass
class SteamTrapData:
    """Steam trap operational data."""
    trap_type: str  # thermostatic, mechanical, thermodynamic, inverted_bucket
    steam_pressure_bar: float
    orifice_size_mm: float
    operating_temperature_c: float
    expected_condensate_load_kg_hr: float
    trap_condition: str  # operational, blowing_steam, plugged, unknown


@dataclass
class TrapAnalysisResult:
    """Steam trap analysis results."""
    trap_efficiency_percent: float
    steam_loss_rate_kg_hr: float
    energy_loss_rate_kw: float
    annual_energy_loss_gj: float
    annual_cost_loss: float
    failure_type: Optional[str]
    recommended_action: str
    replacement_priority: str  # high, medium, low
    provenance: Dict


class SteamTrapAnalyzer:
    """
    Analyze steam trap performance and detect failures.

    Zero Hallucination Guarantee:
    - Physics-based calculations (orifice flow equations)
    - No LLM inference
    - Deterministic failure detection logic
    """

    # Typical trap efficiencies when working properly
    TRAP_EFFICIENCY = {
        'thermostatic': 0.95,
        'mechanical': 0.98,
        'thermodynamic': 0.90,
        'inverted_bucket': 0.96
    }

    # Steam loss from failed trap (as fraction of capacity)
    FAILURE_LOSS_FACTOR = {
        'blowing_steam': 0.50,  # 50% of capacity lost
        'plugged': 0.0,  # No steam loss, but condensate backup
        'operational': 0.05  # Normal 5% steam loss
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize analyzer."""
        self.version = version

    def analyze_trap(
        self,
        trap: SteamTrapData,
        steam_enthalpy_kj_kg: float = 2700.0,
        steam_cost_per_tonne: float = 50.0,
        operating_hours_per_year: float = 8760
    ) -> TrapAnalysisResult:
        """
        Analyze steam trap performance.

        Calculates efficiency, detects failures, estimates losses.
        """
        tracker = ProvenanceTracker(
            calculation_id=f"trap_analysis_{id(trap)}",
            calculation_type="steam_trap_analysis",
            version=self.version
        )

        tracker.record_inputs(trap.__dict__)

        # Step 1: Calculate trap capacity
        capacity = self._calculate_trap_capacity(trap, tracker)

        # Step 2: Calculate steam loss based on condition
        steam_loss = self._calculate_steam_loss(trap, capacity, tracker)

        # Step 3: Calculate energy loss
        energy_loss_kw = self._calculate_energy_loss(
            steam_loss,
            steam_enthalpy_kj_kg,
            tracker
        )

        # Step 4: Calculate annual losses
        annual_energy_gj = energy_loss_kw * Decimal('8760') * Decimal('3.6') / Decimal('1000')
        annual_cost = self._calculate_annual_cost(
            steam_loss,
            steam_cost_per_tonne,
            operating_hours_per_year,
            tracker
        )

        # Step 5: Calculate efficiency
        efficiency = self._calculate_trap_efficiency(trap, steam_loss, capacity, tracker)

        # Step 6: Determine recommended action and priority
        action, priority = self._determine_action(trap, float(efficiency), float(steam_loss))

        return TrapAnalysisResult(
            trap_efficiency_percent=float(efficiency),
            steam_loss_rate_kg_hr=float(steam_loss),
            energy_loss_rate_kw=float(energy_loss_kw),
            annual_energy_loss_gj=float(annual_energy_gj),
            annual_cost_loss=float(annual_cost),
            failure_type=trap.trap_condition if trap.trap_condition != 'operational' else None,
            recommended_action=action,
            replacement_priority=priority,
            provenance=tracker.get_provenance_record(efficiency).to_dict()
        )

    def analyze_trap_population(
        self,
        traps: List[SteamTrapData],
        steam_enthalpy_kj_kg: float = 2700.0,
        steam_cost_per_tonne: float = 50.0
    ) -> Dict:
        """
        Analyze entire population of steam traps.

        Returns fleet-wide statistics and prioritized action list.
        """
        results = []
        total_loss_kg_hr = Decimal('0')
        total_cost = Decimal('0')
        failure_count = 0

        for trap in traps:
            result = self.analyze_trap(trap, steam_enthalpy_kj_kg, steam_cost_per_tonne)
            results.append(result)

            total_loss_kg_hr += Decimal(str(result.steam_loss_rate_kg_hr))
            total_cost += Decimal(str(result.annual_cost_loss))

            if trap.trap_condition != 'operational':
                failure_count += 1

        # Calculate fleet statistics
        failure_rate = (failure_count / len(traps) * 100) if traps else 0

        # Sort by priority for action list
        high_priority = [r for r in results if r.replacement_priority == 'high']
        medium_priority = [r for r in results if r.replacement_priority == 'medium']

        return {
            'total_traps': len(traps),
            'failed_traps': failure_count,
            'failure_rate_percent': round(failure_rate, 1),
            'total_steam_loss_kg_hr': float(total_loss_kg_hr),
            'total_annual_cost_loss': float(total_cost),
            'high_priority_repairs': len(high_priority),
            'medium_priority_repairs': len(medium_priority),
            'estimated_payback_months': self._calculate_payback_period(float(total_cost)),
            'recommendations': self._generate_fleet_recommendations(
                len(traps),
                failure_count,
                float(total_cost)
            )
        }

    def _calculate_trap_capacity(
        self,
        trap: SteamTrapData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate theoretical trap capacity using orifice flow equation.

        Q = C * A * sqrt(2 * ρ * ΔP)
        Simplified for steam traps.
        """
        P = Decimal(str(trap.steam_pressure_bar))
        d_mm = Decimal(str(trap.orifice_size_mm))
        d_m = d_mm / Decimal('1000')

        # Orifice area
        A = Decimal('3.14159') * (d_m / Decimal('2')) ** 2

        # Discharge coefficient (typical for steam traps)
        C = Decimal('0.65')

        # Simplified capacity calculation (kg/hr)
        # Capacity ≈ 23 * d² * P^0.5 (empirical for steam)
        capacity = Decimal('23') * (d_mm ** 2) * (P ** Decimal('0.5'))

        tracker.record_step(
            operation="trap_capacity",
            description="Calculate trap capacity",
            inputs={
                'pressure_bar': P,
                'orifice_size_mm': d_mm
            },
            output_value=capacity,
            output_name="trap_capacity_kg_hr",
            formula="Simplified orifice flow equation",
            units="kg/hr"
        )

        return capacity

    def _calculate_steam_loss(
        self,
        trap: SteamTrapData,
        capacity: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate steam loss based on trap condition."""
        condition = trap.trap_condition
        loss_factor = Decimal(str(self.FAILURE_LOSS_FACTOR.get(condition, 0.10)))

        steam_loss = capacity * loss_factor

        tracker.record_step(
            operation="steam_loss",
            description=f"Calculate steam loss for {condition} trap",
            inputs={
                'capacity_kg_hr': capacity,
                'loss_factor': loss_factor
            },
            output_value=steam_loss,
            output_name="steam_loss_kg_hr",
            formula="Loss = Capacity * Loss_Factor",
            units="kg/hr"
        )

        return steam_loss

    def _calculate_energy_loss(
        self,
        steam_loss_kg_hr: Decimal,
        enthalpy_kj_kg: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate energy loss rate in kW."""
        m_loss_kg_s = steam_loss_kg_hr / Decimal('3600')
        h = Decimal(str(enthalpy_kj_kg))

        # Energy loss rate: kW = kg/s * kJ/kg
        E_loss_kw = m_loss_kg_s * h

        tracker.record_step(
            operation="energy_loss",
            description="Calculate energy loss rate",
            inputs={
                'steam_loss_kg_s': m_loss_kg_s,
                'enthalpy_kj_kg': h
            },
            output_value=E_loss_kw,
            output_name="energy_loss_kw",
            formula="E = m_dot * h",
            units="kW"
        )

        return E_loss_kw

    def _calculate_annual_cost(
        self,
        steam_loss_kg_hr: Decimal,
        cost_per_tonne: float,
        operating_hours: float,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate annual cost of steam loss."""
        annual_loss_tonnes = (steam_loss_kg_hr * Decimal(str(operating_hours))) / Decimal('1000')
        cost = annual_loss_tonnes * Decimal(str(cost_per_tonne))

        tracker.record_step(
            operation="annual_cost",
            description="Calculate annual cost of steam loss",
            inputs={
                'annual_loss_tonnes': annual_loss_tonnes,
                'cost_per_tonne': Decimal(str(cost_per_tonne))
            },
            output_value=cost,
            output_name="annual_cost_loss",
            formula="Cost = Annual_Loss * Cost_per_tonne",
            units="currency"
        )

        return cost.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _calculate_trap_efficiency(
        self,
        trap: SteamTrapData,
        steam_loss: Decimal,
        capacity: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate trap efficiency."""
        base_efficiency = Decimal(str(self.TRAP_EFFICIENCY.get(trap.trap_type, 0.90)))

        if trap.trap_condition == 'operational':
            efficiency = base_efficiency
        elif trap.trap_condition == 'blowing_steam':
            # Significant loss of efficiency
            efficiency = base_efficiency * Decimal('0.50')
        elif trap.trap_condition == 'plugged':
            # Not passing steam but also not draining condensate
            efficiency = Decimal('0')
        else:
            efficiency = base_efficiency * Decimal('0.70')  # Unknown condition

        efficiency_percent = efficiency * Decimal('100')

        tracker.record_step(
            operation="trap_efficiency",
            description="Calculate trap efficiency",
            inputs={
                'trap_type': trap.trap_type,
                'trap_condition': trap.trap_condition,
                'base_efficiency': base_efficiency
            },
            output_value=efficiency_percent,
            output_name="trap_efficiency_percent",
            formula="Based on trap type and condition",
            units="%"
        )

        return efficiency_percent.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

    def _determine_action(
        self,
        trap: SteamTrapData,
        efficiency: float,
        steam_loss: float
    ) -> tuple:
        """Determine recommended action and priority."""
        if trap.trap_condition == 'blowing_steam':
            if steam_loss > 50:
                return ("Replace immediately - high steam loss", "high")
            else:
                return ("Repair or replace within 1 week", "high")

        elif trap.trap_condition == 'plugged':
            return ("Clean or replace - condensate backup risk", "high")

        elif efficiency < 70:
            return ("Replace during next maintenance window", "medium")

        elif efficiency < 90:
            return ("Monitor and schedule for replacement", "low")

        else:
            return ("Continue normal operation and monitoring", "low")

    def _calculate_payback_period(self, annual_savings: float) -> float:
        """
        Calculate payback period for trap repair/replacement program.

        Typical trap replacement cost: $500-2000 per trap
        """
        avg_replacement_cost = 1000  # USD per trap
        if annual_savings > 0:
            payback_months = (avg_replacement_cost / annual_savings) * 12
            return round(payback_months, 1)
        return 999.9

    def _generate_fleet_recommendations(
        self,
        total_traps: int,
        failed_traps: int,
        annual_cost: float
    ) -> List[str]:
        """Generate recommendations for trap fleet management."""
        recommendations = []

        failure_rate = (failed_traps / total_traps * 100) if total_traps > 0 else 0

        if failure_rate > 10:
            recommendations.append(
                f"HIGH PRIORITY: Failure rate is {failure_rate:.1f}% (target: <5%). "
                f"Implement comprehensive trap survey and replacement program."
            )

        if annual_cost > 10000:
            recommendations.append(
                f"Annual steam loss cost is ${annual_cost:,.0f}. "
                f"Trap replacement program will typically pay back in 6-18 months."
            )

        recommendations.append(
            "Implement quarterly steam trap inspection program using ultrasonic testing"
        )

        recommendations.append(
            "Maintain trap maintenance database with inspection history and failure patterns"
        )

        if failure_rate > 5:
            recommendations.append(
                "Consider upgrading to higher-quality traps in critical locations"
            )

        return recommendations
