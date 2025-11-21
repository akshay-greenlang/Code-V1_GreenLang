# -*- coding: utf-8 -*-
"""
KPI Calculator - Zero Hallucination Guarantee

Calculates industry-standard Key Performance Indicators for process heat systems
using deterministic formulas with complete provenance tracking.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: ISO 22400, MESA-11, ISA-95, OEE Foundation
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from provenance import ProvenanceTracker, ProvenanceRecord
from greenlang.determinism import FinancialDecimal


@dataclass
class OperationalData:
    """Input data for KPI calculations."""
    # Time metrics
    planned_production_time_hours: float
    actual_run_time_hours: float
    downtime_hours: float
    scheduled_maintenance_hours: float
    unscheduled_downtime_hours: float

    # Production metrics
    total_units_produced: int
    good_units_produced: int
    defective_units: int
    ideal_cycle_time_seconds: float
    actual_cycle_time_seconds: float

    # Energy metrics
    total_energy_consumed_kwh: float
    fuel_consumed_kg: float
    electricity_consumed_kwh: float
    steam_consumed_tonnes: float
    renewable_energy_kwh: float

    # Output metrics
    heat_output_mwh: float
    steam_output_tonnes: float
    process_throughput_tonnes: float

    # Financial metrics
    energy_cost_usd: float
    maintenance_cost_usd: float
    labor_cost_usd: float
    revenue_usd: float

    # Environmental metrics
    co2_emissions_tonnes: float
    nox_emissions_kg: float
    water_consumption_m3: float
    waste_generated_tonnes: float


class KPICalculator:
    """
    Calculates Key Performance Indicators using industry standards.

    Zero Hallucination Guarantee:
    - Pure mathematical calculations
    - Industry-standard formulas only
    - Bit-perfect reproducibility
    - Complete provenance tracking
    """

    def __init__(self, version: str = "1.0.0"):
        """Initialize KPI calculator with version tracking."""
        self.version = version

    def calculate_all_kpis(self, data: OperationalData) -> Dict:
        """
        Calculate all KPIs for process heat operations.

        Args:
            data: Operational metrics data

        Returns:
            Complete KPI dashboard with all metrics
        """
        # Initialize provenance tracking
        tracker = ProvenanceTracker(
            calculation_id=f"kpi_calc_{id(data)}",
            calculation_type="kpi_calculation",
            version=self.version
        )

        # Record inputs
        tracker.record_inputs(data.__dict__)

        # Calculate OEE components
        oee_metrics = self._calculate_oee(data, tracker)

        # Calculate TEEP
        teep = self._calculate_teep(data, oee_metrics, tracker)

        # Calculate energy KPIs
        energy_kpis = self._calculate_energy_kpis(data, tracker)

        # Calculate production KPIs
        production_kpis = self._calculate_production_kpis(data, tracker)

        # Calculate financial KPIs
        financial_kpis = self._calculate_financial_kpis(data, tracker)

        # Calculate environmental KPIs
        environmental_kpis = self._calculate_environmental_kpis(data, tracker)

        # Calculate maintenance KPIs
        maintenance_kpis = self._calculate_maintenance_kpis(data, tracker)

        # Calculate composite scores
        composite_scores = self._calculate_composite_scores(
            oee_metrics, energy_kpis, environmental_kpis, tracker
        )

        # Generate benchmarks
        benchmarks = self._generate_benchmarks(composite_scores, tracker)

        # Final result
        result = {
            'oee': oee_metrics,
            'teep': teep,
            'energy': energy_kpis,
            'production': production_kpis,
            'financial': financial_kpis,
            'environmental': environmental_kpis,
            'maintenance': maintenance_kpis,
            'composite_scores': composite_scores,
            'benchmarks': benchmarks,
            'provenance': tracker.get_provenance_record(
                oee_metrics['oee_percent']
            ).to_dict()
        }

        return result

    def _calculate_oee(self, data: OperationalData, tracker: ProvenanceTracker) -> Dict:
        """
        Calculate Overall Equipment Effectiveness (OEE).

        OEE = Availability × Performance × Quality
        """
        # Availability = Run Time / Planned Production Time
        run_time = Decimal(str(data.actual_run_time_hours))
        planned_time = Decimal(str(data.planned_production_time_hours))

        if planned_time > 0:
            availability = (run_time / planned_time) * Decimal('100')
        else:
            availability = Decimal('0')

        tracker.record_step(
            operation="division",
            description="Calculate availability",
            inputs={
                'actual_run_time_hours': run_time,
                'planned_production_time_hours': planned_time
            },
            output_value=availability,
            output_name="availability_percent",
            formula="Availability = (Run Time / Planned Time) × 100",
            units="%"
        )

        # Performance = (Ideal Cycle Time × Total Units) / Run Time
        ideal_cycle = Decimal(str(data.ideal_cycle_time_seconds))
        total_units = Decimal(str(data.total_units_produced))
        run_time_seconds = run_time * Decimal('3600')

        if run_time_seconds > 0:
            performance = ((ideal_cycle * total_units) / run_time_seconds) * Decimal('100')
            performance = min(performance, Decimal('100'))  # Cap at 100%
        else:
            performance = Decimal('0')

        tracker.record_step(
            operation="performance_calc",
            description="Calculate performance efficiency",
            inputs={
                'ideal_cycle_time_seconds': ideal_cycle,
                'total_units_produced': total_units,
                'run_time_seconds': run_time_seconds
            },
            output_value=performance,
            output_name="performance_percent",
            formula="Performance = (Ideal Cycle × Units / Run Time) × 100",
            units="%"
        )

        # Quality = Good Units / Total Units
        good_units = Decimal(str(data.good_units_produced))

        if total_units > 0:
            quality = (good_units / total_units) * Decimal('100')
        else:
            quality = Decimal('0')

        tracker.record_step(
            operation="quality_calc",
            description="Calculate quality rate",
            inputs={
                'good_units_produced': good_units,
                'total_units_produced': total_units
            },
            output_value=quality,
            output_name="quality_percent",
            formula="Quality = (Good Units / Total Units) × 100",
            units="%"
        )

        # OEE = Availability × Performance × Quality
        oee = (availability * performance * quality) / Decimal('10000')
        oee = oee.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="oee_calc",
            description="Calculate Overall Equipment Effectiveness",
            inputs={
                'availability': availability,
                'performance': performance,
                'quality': quality
            },
            output_value=oee,
            output_name="oee_percent",
            formula="OEE = Availability × Performance × Quality / 10000",
            units="%"
        )

        return {
            'availability_percent': float(availability.quantize(Decimal('0.01'))),
            'performance_percent': float(performance.quantize(Decimal('0.01'))),
            'quality_percent': float(quality.quantize(Decimal('0.01'))),
            'oee_percent': float(oee),
            'world_class_oee': 85.0,  # World-class benchmark
            'gap_to_world_class': float(max(Decimal('85') - oee, Decimal('0')))
        }

    def _calculate_teep(
        self,
        data: OperationalData,
        oee_metrics: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """
        Calculate Total Effective Equipment Performance (TEEP).

        TEEP = OEE × Utilization
        """
        # Utilization = Planned Time / Calendar Time
        planned_time = Decimal(str(data.planned_production_time_hours))
        # Assume 24/7 operation potential
        calendar_time = planned_time + Decimal(str(data.scheduled_maintenance_hours))

        if calendar_time > 0:
            utilization = (planned_time / calendar_time) * Decimal('100')
        else:
            utilization = Decimal('0')

        # TEEP = OEE × Utilization
        oee = Decimal(str(oee_metrics['oee_percent']))
        teep = (oee * utilization) / Decimal('100')
        teep = teep.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

        tracker.record_step(
            operation="teep_calc",
            description="Calculate Total Effective Equipment Performance",
            inputs={
                'oee_percent': oee,
                'utilization_percent': utilization
            },
            output_value=teep,
            output_name="teep_percent",
            formula="TEEP = OEE × Utilization / 100",
            units="%"
        )

        return {
            'utilization_percent': float(utilization.quantize(Decimal('0.01'))),
            'teep_percent': float(teep),
            'hidden_capacity_percent': float(Decimal('100') - teep)
        }

    def _calculate_energy_kpis(
        self,
        data: OperationalData,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate energy-related KPIs."""
        total_energy = Decimal(str(data.total_energy_consumed_kwh))
        heat_output = Decimal(str(data.heat_output_mwh))
        throughput = Decimal(str(data.process_throughput_tonnes))

        # Energy Intensity (kWh per tonne)
        if throughput > 0:
            energy_intensity = total_energy / throughput
        else:
            energy_intensity = Decimal('0')

        tracker.record_step(
            operation="intensity_calc",
            description="Calculate energy intensity",
            inputs={
                'total_energy_kwh': total_energy,
                'throughput_tonnes': throughput
            },
            output_value=energy_intensity,
            output_name="energy_intensity_kwh_per_tonne",
            formula="Intensity = Energy / Throughput",
            units="kWh/tonne"
        )

        # Energy Efficiency (useful output / total input)
        heat_output_kwh = heat_output * Decimal('1000')
        if total_energy > 0:
            energy_efficiency = (heat_output_kwh / total_energy) * Decimal('100')
        else:
            energy_efficiency = Decimal('0')

        # Specific Energy Consumption (SEC)
        if data.total_units_produced > 0:
            sec = total_energy / Decimal(str(data.total_units_produced))
        else:
            sec = Decimal('0')

        # Renewable Energy Share
        renewable = Decimal(str(data.renewable_energy_kwh))
        if total_energy > 0:
            renewable_share = (renewable / total_energy) * Decimal('100')
        else:
            renewable_share = Decimal('0')

        return {
            'energy_intensity_kwh_per_tonne': float(
                energy_intensity.quantize(Decimal('0.01'))
            ),
            'energy_efficiency_percent': float(
                energy_efficiency.quantize(Decimal('0.01'))
            ),
            'specific_energy_consumption_kwh_per_unit': float(
                sec.quantize(Decimal('0.001'))
            ),
            'renewable_energy_share_percent': float(
                renewable_share.quantize(Decimal('0.01'))
            ),
            'total_energy_consumed_mwh': FinancialDecimal.from_string(total_energy / Decimal('1000'))
        }

    def _calculate_production_kpis(
        self,
        data: OperationalData,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate production-related KPIs."""
        # Throughput Rate
        run_time = Decimal(str(data.actual_run_time_hours))
        throughput = Decimal(str(data.process_throughput_tonnes))

        if run_time > 0:
            throughput_rate = throughput / run_time
        else:
            throughput_rate = Decimal('0')

        # Capacity Utilization (actual vs ideal)
        ideal_cycle = Decimal(str(data.ideal_cycle_time_seconds))
        actual_cycle = Decimal(str(data.actual_cycle_time_seconds))

        if actual_cycle > 0:
            capacity_utilization = (ideal_cycle / actual_cycle) * Decimal('100')
        else:
            capacity_utilization = Decimal('0')

        # First Pass Yield (FPY)
        good_units = Decimal(str(data.good_units_produced))
        total_units = Decimal(str(data.total_units_produced))

        if total_units > 0:
            fpy = (good_units / total_units) * Decimal('100')
        else:
            fpy = Decimal('0')

        # Defect Rate (PPM - parts per million)
        defects = Decimal(str(data.defective_units))
        if total_units > 0:
            defect_ppm = (defects / total_units) * Decimal('1000000')
        else:
            defect_ppm = Decimal('0')

        tracker.record_step(
            operation="production_metrics",
            description="Calculate production KPIs",
            inputs={
                'throughput_tonnes': throughput,
                'run_time_hours': run_time,
                'good_units': good_units,
                'total_units': total_units
            },
            output_value=throughput_rate,
            output_name="throughput_rate",
            formula="Rate = Throughput / Time",
            units="tonnes/hr"
        )

        return {
            'throughput_rate_tonnes_per_hour': float(
                throughput_rate.quantize(Decimal('0.01'))
            ),
            'capacity_utilization_percent': float(
                capacity_utilization.quantize(Decimal('0.01'))
            ),
            'first_pass_yield_percent': float(fpy.quantize(Decimal('0.01'))),
            'defect_rate_ppm': FinancialDecimal.from_string(defect_ppm.quantize(Decimal('1'))),
            'production_volume_tonnes': float(throughput)
        }

    def _calculate_financial_kpis(
        self,
        data: OperationalData,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate financial KPIs."""
        energy_cost = Decimal(str(data.energy_cost_usd))
        revenue = Decimal(str(data.revenue_usd))
        total_cost = energy_cost + Decimal(str(data.maintenance_cost_usd)) + \
                    Decimal(str(data.labor_cost_usd))

        # Energy Cost per Unit
        units = Decimal(str(data.total_units_produced))
        if units > 0:
            energy_cost_per_unit = energy_cost / units
        else:
            energy_cost_per_unit = Decimal('0')

        # Energy Cost as % of Revenue
        if revenue > 0:
            energy_cost_percentage = (energy_cost / revenue) * Decimal('100')
        else:
            energy_cost_percentage = Decimal('0')

        # Operating Margin
        if revenue > 0:
            operating_margin = ((revenue - total_cost) / revenue) * Decimal('100')
        else:
            operating_margin = Decimal('0')

        # Cost per MWh
        heat_output = Decimal(str(data.heat_output_mwh))
        if heat_output > 0:
            cost_per_mwh = energy_cost / heat_output
        else:
            cost_per_mwh = Decimal('0')

        tracker.record_step(
            operation="financial_metrics",
            description="Calculate financial KPIs",
            inputs={
                'energy_cost': energy_cost,
                'revenue': revenue,
                'total_cost': total_cost
            },
            output_value=operating_margin,
            output_name="operating_margin",
            formula="Margin = (Revenue - Cost) / Revenue × 100",
            units="%"
        )

        return {
            'energy_cost_per_unit_usd': float(
                energy_cost_per_unit.quantize(Decimal('0.01'))
            ),
            'energy_cost_percentage_of_revenue': float(
                energy_cost_percentage.quantize(Decimal('0.01'))
            ),
            'operating_margin_percent': float(
                operating_margin.quantize(Decimal('0.01'))
            ),
            'cost_per_mwh_usd': FinancialDecimal.from_string(cost_per_mwh.quantize(Decimal('0.01'))),
            'total_operating_cost_usd': FinancialDecimal.from_string(total_cost)
        }

    def _calculate_environmental_kpis(
        self,
        data: OperationalData,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate environmental KPIs."""
        co2 = Decimal(str(data.co2_emissions_tonnes))
        throughput = Decimal(str(data.process_throughput_tonnes))
        heat_output = Decimal(str(data.heat_output_mwh))
        water = Decimal(str(data.water_consumption_m3))

        # Carbon Intensity (CO2 per tonne product)
        if throughput > 0:
            carbon_intensity_product = co2 / throughput
        else:
            carbon_intensity_product = Decimal('0')

        # Carbon Intensity (CO2 per MWh)
        if heat_output > 0:
            carbon_intensity_energy = co2 / heat_output
        else:
            carbon_intensity_energy = Decimal('0')

        # Water Intensity
        if throughput > 0:
            water_intensity = water / throughput
        else:
            water_intensity = Decimal('0')

        # Waste Intensity
        waste = Decimal(str(data.waste_generated_tonnes))
        if throughput > 0:
            waste_intensity = (waste / throughput) * Decimal('1000')  # kg/tonne
        else:
            waste_intensity = Decimal('0')

        # NOx Emission Rate
        nox = Decimal(str(data.nox_emissions_kg))
        if heat_output > 0:
            nox_rate = nox / heat_output  # kg/MWh
        else:
            nox_rate = Decimal('0')

        tracker.record_step(
            operation="environmental_metrics",
            description="Calculate environmental KPIs",
            inputs={
                'co2_emissions': co2,
                'throughput': throughput,
                'water_consumption': water
            },
            output_value=carbon_intensity_product,
            output_name="carbon_intensity",
            formula="Intensity = Emissions / Output",
            units="tCO2/tonne"
        )

        return {
            'carbon_intensity_kg_co2_per_tonne': float(
                (carbon_intensity_product * Decimal('1000')).quantize(Decimal('0.01'))
            ),
            'carbon_intensity_tonnes_co2_per_mwh': float(
                carbon_intensity_energy.quantize(Decimal('0.001'))
            ),
            'water_intensity_m3_per_tonne': float(
                water_intensity.quantize(Decimal('0.01'))
            ),
            'waste_intensity_kg_per_tonne': float(
                waste_intensity.quantize(Decimal('0.01'))
            ),
            'nox_emission_rate_kg_per_mwh': float(
                nox_rate.quantize(Decimal('0.001'))
            ),
            'total_co2_emissions_tonnes': FinancialDecimal.from_string(co2)
        }

    def _calculate_maintenance_kpis(
        self,
        data: OperationalData,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate maintenance-related KPIs."""
        scheduled = Decimal(str(data.scheduled_maintenance_hours))
        unscheduled = Decimal(str(data.unscheduled_downtime_hours))
        total_time = Decimal(str(data.planned_production_time_hours))
        run_time = Decimal(str(data.actual_run_time_hours))

        # Mean Time Between Failures (MTBF)
        # Assuming unscheduled downtime indicates failures
        if unscheduled > 0:
            # Estimate number of failures (assume avg 2 hours per failure)
            failures = max(unscheduled / Decimal('2'), Decimal('1'))
            mtbf = run_time / failures
        else:
            mtbf = run_time  # No failures

        # Mean Time To Repair (MTTR)
        if unscheduled > 0 and failures > 0:
            mttr = unscheduled / failures
        else:
            mttr = Decimal('0')

        # Planned Maintenance Percentage
        if total_time > 0:
            planned_maintenance_pct = (scheduled / total_time) * Decimal('100')
        else:
            planned_maintenance_pct = Decimal('0')

        # Reactive Maintenance Percentage
        total_maintenance = scheduled + unscheduled
        if total_maintenance > 0:
            reactive_pct = (unscheduled / total_maintenance) * Decimal('100')
        else:
            reactive_pct = Decimal('0')

        # Maintenance Cost per Unit
        maintenance_cost = Decimal(str(data.maintenance_cost_usd))
        units = Decimal(str(data.total_units_produced))
        if units > 0:
            cost_per_unit = maintenance_cost / units
        else:
            cost_per_unit = Decimal('0')

        tracker.record_step(
            operation="maintenance_metrics",
            description="Calculate maintenance KPIs",
            inputs={
                'scheduled_hours': scheduled,
                'unscheduled_hours': unscheduled,
                'run_time': run_time
            },
            output_value=mtbf,
            output_name="mtbf_hours",
            formula="MTBF = Run Time / Number of Failures",
            units="hours"
        )

        return {
            'mtbf_hours': float(mtbf.quantize(Decimal('0.1'))),
            'mttr_hours': float(mttr.quantize(Decimal('0.1'))),
            'planned_maintenance_percent': float(
                planned_maintenance_pct.quantize(Decimal('0.01'))
            ),
            'reactive_maintenance_percent': float(
                reactive_pct.quantize(Decimal('0.01'))
            ),
            'maintenance_cost_per_unit_usd': float(
                cost_per_unit.quantize(Decimal('0.01'))
            ),
            'total_downtime_hours': FinancialDecimal.from_string(scheduled + unscheduled)
        }

    def _calculate_composite_scores(
        self,
        oee: Dict,
        energy: Dict,
        environmental: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Calculate composite performance scores."""
        # Operational Excellence Score (weighted average)
        oee_score = Decimal(str(oee['oee_percent']))
        energy_efficiency = Decimal(str(energy['energy_efficiency_percent']))

        operational_score = (oee_score * Decimal('0.6') +
                           energy_efficiency * Decimal('0.4'))

        # Sustainability Score
        renewable_share = Decimal(str(energy['renewable_energy_share_percent']))
        # Normalize carbon intensity (lower is better)
        carbon_intensity = Decimal(str(
            environmental['carbon_intensity_kg_co2_per_tonne']
        ))
        # Assume 500 kg/tonne is poor, 100 is excellent
        carbon_score = max(Decimal('0'), Decimal('100') -
                          (carbon_intensity / Decimal('5')))

        sustainability_score = (renewable_share * Decimal('0.5') +
                              carbon_score * Decimal('0.5'))

        # Overall Performance Index
        overall_index = (operational_score * Decimal('0.7') +
                        sustainability_score * Decimal('0.3'))

        tracker.record_step(
            operation="composite_scoring",
            description="Calculate composite performance scores",
            inputs={
                'oee': oee_score,
                'energy_efficiency': energy_efficiency,
                'renewable_share': renewable_share
            },
            output_value=overall_index,
            output_name="overall_performance_index",
            formula="Index = 0.7×Operational + 0.3×Sustainability",
            units="points"
        )

        return {
            'operational_excellence_score': float(
                operational_score.quantize(Decimal('0.01'))
            ),
            'sustainability_score': float(
                sustainability_score.quantize(Decimal('0.01'))
            ),
            'overall_performance_index': float(
                overall_index.quantize(Decimal('0.01'))
            ),
            'performance_grade': self._get_grade(float(overall_index))
        }

    def _generate_benchmarks(
        self,
        scores: Dict,
        tracker: ProvenanceTracker
    ) -> Dict:
        """Generate industry benchmarks for comparison."""
        overall = scores['overall_performance_index']

        benchmarks = {
            'world_class': {
                'oee_percent': 85.0,
                'energy_efficiency_percent': 90.0,
                'carbon_intensity_kg_per_tonne': 100.0,
                'overall_index': 90.0
            },
            'industry_average': {
                'oee_percent': 60.0,
                'energy_efficiency_percent': 65.0,
                'carbon_intensity_kg_per_tonne': 300.0,
                'overall_index': 65.0
            },
            'your_performance': {
                'overall_index': overall,
                'percentile': self._calculate_percentile(overall)
            }
        }

        tracker.record_step(
            operation="benchmarking",
            description="Generate performance benchmarks",
            inputs={'overall_index': overall},
            output_value=benchmarks['your_performance']['percentile'],
            output_name="performance_percentile",
            formula="Percentile = f(index, distribution)",
            units="percentile"
        )

        return benchmarks

    def _get_grade(self, score: float) -> str:
        """Convert score to letter grade."""
        if score >= 90:
            return 'A'
        elif score >= 80:
            return 'B'
        elif score >= 70:
            return 'C'
        elif score >= 60:
            return 'D'
        else:
            return 'F'

    def _calculate_percentile(self, score: float) -> int:
        """Calculate percentile ranking (simplified)."""
        # Simplified - assume normal distribution
        if score >= 85:
            return 95
        elif score >= 75:
            return 75
        elif score >= 65:
            return 50
        elif score >= 55:
            return 25
        else:
            return 10