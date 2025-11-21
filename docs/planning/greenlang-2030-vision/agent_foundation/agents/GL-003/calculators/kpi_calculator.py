# -*- coding: utf-8 -*-
"""
Steam System KPI Calculator - Zero Hallucination

Calculates comprehensive Key Performance Indicators for steam system
monitoring and optimization.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: Best practice steam system metrics
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List
from dataclasses import dataclass
from .provenance import ProvenanceTracker
from greenlang.determinism import FinancialDecimal


@dataclass
class SystemMetrics:
    """Input metrics for KPI calculation."""
    # Energy metrics
    fuel_input_gj: float
    steam_output_tonnes: float
    steam_enthalpy_kj_kg: float

    # Distribution metrics
    steam_generated_tonnes: float
    steam_delivered_tonnes: float

    # Condensate metrics
    condensate_returned_tonnes: float

    # Losses
    distribution_loss_gj: float
    trap_losses_kg_hr: float

    # Operating parameters
    operating_hours: float
    number_of_steam_traps: int
    failed_traps: int


@dataclass
class KPIDashboard:
    """Comprehensive KPI dashboard."""
    # Overall system efficiency
    overall_system_efficiency_percent: float
    boiler_efficiency_percent: float
    distribution_efficiency_percent: float

    # Specific consumption
    specific_steam_consumption_gj_per_tonne: float

    # Losses
    total_losses_percent: float
    distribution_loss_percent: float
    trap_loss_percent: float

    # Condensate return
    condensate_return_rate_percent: float

    # Trap performance
    steam_trap_failure_rate_percent: float
    trap_performance_index: float

    # Energy savings opportunity
    total_savings_opportunity_gj: float
    total_savings_opportunity_percent: float

    # Cost metrics
    estimated_annual_savings: float

    # Benchmarking
    performance_rating: str  # excellent, good, fair, poor
    industry_comparison: Dict

    provenance: Dict


class KPICalculator:
    """
    Calculate comprehensive KPIs for steam system performance.

    Zero Hallucination Guarantee:
    - Pure mathematical calculations
    - Industry-standard formulas
    - No LLM inference
    """

    # Industry benchmarks
    BENCHMARKS = {
        'excellent': {
            'overall_efficiency': 85.0,
            'distribution_efficiency': 98.0,
            'condensate_return': 95.0,
            'trap_failure_rate': 3.0
        },
        'good': {
            'overall_efficiency': 80.0,
            'distribution_efficiency': 95.0,
            'condensate_return': 85.0,
            'trap_failure_rate': 7.0
        },
        'fair': {
            'overall_efficiency': 70.0,
            'distribution_efficiency': 90.0,
            'condensate_return': 70.0,
            'trap_failure_rate': 15.0
        },
        'poor': {
            'overall_efficiency': 60.0,
            'distribution_efficiency': 85.0,
            'condensate_return': 50.0,
            'trap_failure_rate': 25.0
        }
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize calculator."""
        self.version = version

    def calculate_kpis(
        self,
        metrics: SystemMetrics,
        energy_cost_per_gj: float = 20.0
    ) -> KPIDashboard:
        """
        Calculate comprehensive KPI dashboard.

        Returns all key performance indicators with benchmarking.
        """
        tracker = ProvenanceTracker(
            calculation_id=f"kpi_calc_{id(metrics)}",
            calculation_type="steam_system_kpis",
            version=self.version
        )

        tracker.record_inputs(metrics.__dict__)

        # Step 1: Calculate boiler efficiency
        boiler_eff = self._calculate_boiler_efficiency(metrics, tracker)

        # Step 2: Calculate distribution efficiency
        dist_eff = self._calculate_distribution_efficiency(metrics, tracker)

        # Step 3: Calculate overall system efficiency
        overall_eff = self._calculate_overall_efficiency(boiler_eff, dist_eff, tracker)

        # Step 4: Calculate specific steam consumption
        specific_consumption = self._calculate_specific_consumption(metrics, tracker)

        # Step 5: Calculate losses
        losses = self._calculate_losses(metrics, tracker)

        # Step 6: Calculate condensate return rate
        condensate_return = self._calculate_condensate_return_rate(metrics, tracker)

        # Step 7: Calculate trap performance
        trap_kpis = self._calculate_trap_kpis(metrics, tracker)

        # Step 8: Calculate savings opportunity
        savings = self._calculate_savings_opportunity(
            metrics,
            overall_eff,
            dist_eff,
            condensate_return,
            trap_kpis,
            tracker
        )

        # Step 9: Calculate annual cost savings
        annual_savings = savings['total_gj'] * Decimal(str(energy_cost_per_gj))

        # Step 10: Determine performance rating
        rating = self._determine_performance_rating(
            float(overall_eff),
            float(dist_eff),
            float(condensate_return),
            FinancialDecimal.from_string(trap_kpis['failure_rate'])
        )

        # Step 11: Industry comparison
        comparison = self._compare_to_industry(
            float(overall_eff),
            float(dist_eff),
            float(condensate_return),
            FinancialDecimal.from_string(trap_kpis['failure_rate'])
        )

        return KPIDashboard(
            overall_system_efficiency_percent=float(overall_eff),
            boiler_efficiency_percent=float(boiler_eff),
            distribution_efficiency_percent=float(dist_eff),
            specific_steam_consumption_gj_per_tonne=FinancialDecimal.from_string(specific_consumption),
            total_losses_percent=FinancialDecimal.from_string(losses['total']),
            distribution_loss_percent=float(losses['distribution']),
            trap_loss_percent=float(losses['traps']),
            condensate_return_rate_percent=FinancialDecimal.from_string(condensate_return),
            steam_trap_failure_rate_percent=FinancialDecimal.from_string(trap_kpis['failure_rate']),
            trap_performance_index=float(trap_kpis['performance_index']),
            total_savings_opportunity_gj=FinancialDecimal.from_string(savings['total_gj']),
            total_savings_opportunity_percent=FinancialDecimal.from_string(savings['percent']),
            estimated_annual_savings=float(annual_savings),
            performance_rating=rating,
            industry_comparison=comparison,
            provenance=tracker.get_provenance_record(overall_eff).to_dict()
        )

    def _calculate_boiler_efficiency(
        self,
        metrics: SystemMetrics,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate boiler efficiency.

        η_boiler = (Steam Output Energy / Fuel Input Energy) * 100
        """
        fuel_input = Decimal(str(metrics.fuel_input_gj))
        steam_output = Decimal(str(metrics.steam_output_tonnes))
        steam_enthalpy = Decimal(str(metrics.steam_enthalpy_kj_kg))

        # Steam energy output (GJ)
        steam_energy_gj = (steam_output * Decimal('1000') * steam_enthalpy) / Decimal('1000000')

        if fuel_input > Decimal('0'):
            efficiency = (steam_energy_gj / fuel_input) * Decimal('100')
        else:
            efficiency = Decimal('0')

        efficiency = efficiency.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="boiler_efficiency",
            description="Calculate boiler efficiency",
            inputs={
                'steam_energy_gj': steam_energy_gj,
                'fuel_input_gj': fuel_input
            },
            output_value=efficiency,
            output_name="boiler_efficiency_percent",
            formula="η = (Steam Energy / Fuel Energy) * 100",
            units="%"
        )

        return efficiency

    def _calculate_distribution_efficiency(
        self,
        metrics: SystemMetrics,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate distribution efficiency.

        η_dist = (Steam Delivered / Steam Generated) * 100
        """
        generated = Decimal(str(metrics.steam_generated_tonnes))
        delivered = Decimal(str(metrics.steam_delivered_tonnes))

        if generated > Decimal('0'):
            efficiency = (delivered / generated) * Decimal('100')
        else:
            efficiency = Decimal('0')

        efficiency = efficiency.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="distribution_efficiency",
            description="Calculate distribution efficiency",
            inputs={
                'steam_delivered_tonnes': delivered,
                'steam_generated_tonnes': generated
            },
            output_value=efficiency,
            output_name="distribution_efficiency_percent",
            formula="η = (Delivered / Generated) * 100",
            units="%"
        )

        return efficiency

    def _calculate_overall_efficiency(
        self,
        boiler_eff: Decimal,
        dist_eff: Decimal,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """
        Calculate overall system efficiency.

        η_overall = η_boiler * η_distribution / 100
        """
        overall = (boiler_eff * dist_eff) / Decimal('100')
        overall = overall.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="overall_efficiency",
            description="Calculate overall system efficiency",
            inputs={
                'boiler_efficiency': boiler_eff,
                'distribution_efficiency': dist_eff
            },
            output_value=overall,
            output_name="overall_efficiency_percent",
            formula="η_overall = η_boiler * η_dist / 100",
            units="%"
        )

        return overall

    def _calculate_specific_consumption(
        self,
        metrics: SystemMetrics,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate specific steam consumption (energy per tonne of steam)."""
        fuel = Decimal(str(metrics.fuel_input_gj))
        steam = Decimal(str(metrics.steam_output_tonnes))

        if steam > Decimal('0'):
            specific = fuel / steam
        else:
            specific = Decimal('0')

        tracker.record_step(
            operation="specific_consumption",
            description="Calculate specific steam consumption",
            inputs={
                'fuel_input_gj': fuel,
                'steam_output_tonnes': steam
            },
            output_value=specific,
            output_name="specific_consumption_gj_per_tonne",
            formula="Specific = Fuel / Steam",
            units="GJ/tonne"
        )

        return specific.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    def _calculate_losses(
        self,
        metrics: SystemMetrics,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate system losses as percentages."""
        fuel_input = Decimal(str(metrics.fuel_input_gj))

        # Distribution losses
        dist_loss = Decimal(str(metrics.distribution_loss_gj))
        if fuel_input > Decimal('0'):
            dist_loss_pct = (dist_loss / fuel_input) * Decimal('100')
        else:
            dist_loss_pct = Decimal('0')

        # Trap losses (convert kg/hr to GJ)
        trap_loss_kg_hr = Decimal(str(metrics.trap_losses_kg_hr))
        hours = Decimal(str(metrics.operating_hours))
        steam_enthalpy = Decimal(str(metrics.steam_enthalpy_kj_kg))

        trap_loss_gj = (trap_loss_kg_hr * hours * steam_enthalpy) / Decimal('1000000')
        if fuel_input > Decimal('0'):
            trap_loss_pct = (trap_loss_gj / fuel_input) * Decimal('100')
        else:
            trap_loss_pct = Decimal('0')

        # Total losses
        total_loss_pct = dist_loss_pct + trap_loss_pct

        tracker.record_step(
            operation="system_losses",
            description="Calculate system losses",
            inputs={
                'distribution_loss_gj': dist_loss,
                'trap_loss_gj': trap_loss_gj,
                'fuel_input_gj': fuel_input
            },
            output_value=total_loss_pct,
            output_name="total_losses_percent",
            formula="Total = Distribution + Trap losses",
            units="%"
        )

        return {
            'distribution': dist_loss_pct,
            'traps': trap_loss_pct,
            'total': total_loss_pct
        }

    def _calculate_condensate_return_rate(
        self,
        metrics: SystemMetrics,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate condensate return rate."""
        returned = Decimal(str(metrics.condensate_returned_tonnes))
        generated = Decimal(str(metrics.steam_generated_tonnes))

        if generated > Decimal('0'):
            return_rate = (returned / generated) * Decimal('100')
        else:
            return_rate = Decimal('0')

        return_rate = return_rate.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="condensate_return_rate",
            description="Calculate condensate return rate",
            inputs={
                'condensate_returned_tonnes': returned,
                'steam_generated_tonnes': generated
            },
            output_value=return_rate,
            output_name="condensate_return_rate_percent",
            formula="Return% = (Returned / Generated) * 100",
            units="%"
        )

        return return_rate

    def _calculate_trap_kpis(
        self,
        metrics: SystemMetrics,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate steam trap KPIs."""
        total_traps = Decimal(str(metrics.number_of_steam_traps))
        failed = Decimal(str(metrics.failed_traps))

        # Failure rate
        if total_traps > Decimal('0'):
            failure_rate = (failed / total_traps) * Decimal('100')
        else:
            failure_rate = Decimal('0')

        # Performance index (100 = perfect, 0 = all failed)
        performance_index = Decimal('100') - failure_rate

        tracker.record_step(
            operation="trap_kpis",
            description="Calculate trap performance KPIs",
            inputs={
                'total_traps': total_traps,
                'failed_traps': failed
            },
            output_value=failure_rate,
            output_name="trap_failure_rate_percent",
            formula="Failure% = (Failed / Total) * 100",
            units="%"
        )

        return {
            'failure_rate': failure_rate,
            'performance_index': performance_index
        }

    def _calculate_savings_opportunity(
        self,
        metrics: SystemMetrics,
        overall_eff: Decimal,
        dist_eff: Decimal,
        condensate_return: Decimal,
        trap_kpis: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate total energy savings opportunity."""
        fuel_input = Decimal(str(metrics.fuel_input_gj))
        savings_gj = Decimal('0')

        # Efficiency improvement opportunity (to 85% target)
        target_eff = Decimal('85')
        if overall_eff < target_eff:
            eff_improvement = (target_eff - overall_eff) / Decimal('100')
            savings_gj += fuel_input * eff_improvement

        # Distribution improvement (to 98% target)
        target_dist = Decimal('98')
        if dist_eff < target_dist:
            dist_improvement = (target_dist - dist_eff) / Decimal('100')
            savings_gj += fuel_input * dist_improvement * Decimal('0.5')

        # Condensate return improvement (to 95% target)
        target_condensate = Decimal('95')
        if condensate_return < target_condensate:
            cond_improvement = (target_condensate - condensate_return) / Decimal('100')
            savings_gj += fuel_input * cond_improvement * Decimal('0.05')

        # Trap repair opportunity
        failure_rate = trap_kpis['failure_rate']
        if failure_rate > Decimal('5'):
            trap_savings = fuel_input * (failure_rate / Decimal('100')) * Decimal('0.02')
            savings_gj += trap_savings

        # Savings as percentage of fuel input
        if fuel_input > Decimal('0'):
            savings_percent = (savings_gj / fuel_input) * Decimal('100')
        else:
            savings_percent = Decimal('0')

        tracker.record_step(
            operation="savings_opportunity",
            description="Calculate total energy savings opportunity",
            inputs={
                'fuel_input_gj': fuel_input,
                'efficiency_gap': target_eff - overall_eff,
                'distribution_gap': target_dist - dist_eff
            },
            output_value=savings_gj,
            output_name="savings_opportunity_gj",
            formula="Sum of improvement opportunities",
            units="GJ"
        )

        return {
            'total_gj': savings_gj.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP),
            'percent': savings_percent.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)
        }

    def _determine_performance_rating(
        self,
        overall_eff: float,
        dist_eff: float,
        condensate: float,
        trap_failure: float
    ) -> str:
        """Determine overall performance rating."""
        score = 0

        # Weight different metrics
        if overall_eff >= self.BENCHMARKS['excellent']['overall_efficiency']:
            score += 4
        elif overall_eff >= self.BENCHMARKS['good']['overall_efficiency']:
            score += 3
        elif overall_eff >= self.BENCHMARKS['fair']['overall_efficiency']:
            score += 2
        else:
            score += 1

        if dist_eff >= self.BENCHMARKS['excellent']['distribution_efficiency']:
            score += 4
        elif dist_eff >= self.BENCHMARKS['good']['distribution_efficiency']:
            score += 3
        elif dist_eff >= self.BENCHMARKS['fair']['distribution_efficiency']:
            score += 2
        else:
            score += 1

        if condensate >= self.BENCHMARKS['excellent']['condensate_return']:
            score += 2
        elif condensate >= self.BENCHMARKS['good']['condensate_return']:
            score += 1.5
        elif condensate >= self.BENCHMARKS['fair']['condensate_return']:
            score += 1
        else:
            score += 0.5

        # Average score
        avg_score = score / 10 * 4

        if avg_score >= 3.5:
            return 'excellent'
        elif avg_score >= 2.5:
            return 'good'
        elif avg_score >= 1.5:
            return 'fair'
        else:
            return 'poor'

    def _compare_to_industry(
        self,
        overall_eff: float,
        dist_eff: float,
        condensate: float,
        trap_failure: float
    ) -> Dict:
        """Compare metrics to industry benchmarks."""
        return {
            'overall_efficiency': {
                'your_value': overall_eff,
                'excellent': self.BENCHMARKS['excellent']['overall_efficiency'],
                'good': self.BENCHMARKS['good']['overall_efficiency'],
                'fair': self.BENCHMARKS['fair']['overall_efficiency']
            },
            'distribution_efficiency': {
                'your_value': dist_eff,
                'excellent': self.BENCHMARKS['excellent']['distribution_efficiency'],
                'good': self.BENCHMARKS['good']['distribution_efficiency'],
                'fair': self.BENCHMARKS['fair']['distribution_efficiency']
            },
            'condensate_return': {
                'your_value': condensate,
                'excellent': self.BENCHMARKS['excellent']['condensate_return'],
                'good': self.BENCHMARKS['good']['condensate_return'],
                'fair': self.BENCHMARKS['fair']['condensate_return']
            },
            'trap_failure_rate': {
                'your_value': trap_failure,
                'excellent': self.BENCHMARKS['excellent']['trap_failure_rate'],
                'good': self.BENCHMARKS['good']['trap_failure_rate'],
                'fair': self.BENCHMARKS['fair']['trap_failure_rate']
            }
        }
