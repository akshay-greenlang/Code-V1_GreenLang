# -*- coding: utf-8 -*-
"""
Performance KPI Calculator for GL-007 FURNACEPULSE FurnacePerformanceMonitor

Implements deterministic calculation of furnace performance Key Performance Indicators
(KPIs) including Specific Fuel Consumption (SFC), Heat Rate, Availability, Utilization,
and Production Efficiency metrics with zero-hallucination guarantees.

Standards Compliance:
- ASME PTC 4.2: Performance Test Code on Industrial Furnaces
- ISO 50001: Energy Management Systems
- ISO 50006: Energy Baseline and Energy Performance Indicators
- EN 16231: Energy Efficiency Benchmarking Methodology

Author: GL-CalculatorEngineer
Agent: GL-007 FURNACEPULSE
Version: 1.0.0
"""

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta

from .provenance import ProvenanceTracker, ProvenanceRecord, CalculationCategory


class KPICategory(Enum):
    """Categories of furnace performance KPIs."""
    ENERGY_EFFICIENCY = "energy_efficiency"
    PRODUCTION_EFFICIENCY = "production_efficiency"
    AVAILABILITY = "availability"
    UTILIZATION = "utilization"
    ENVIRONMENTAL = "environmental"
    ECONOMIC = "economic"


class ProductUnit(Enum):
    """Units for product measurement."""
    TONNES = "tonnes"
    KILOGRAMS = "kilograms"
    PIECES = "pieces"
    SQUARE_METERS = "square_meters"
    CUBIC_METERS = "cubic_meters"


@dataclass
class FurnaceOperatingData:
    """
    Operating data for furnace KPI calculations.

    Attributes:
        furnace_id: Unique furnace identifier
        period_hours: Operating period duration (hours)
        fuel_consumption_kg: Total fuel consumed in period (kg)
        fuel_type: Type of fuel used
        fuel_lhv_mj_kg: Lower heating value of fuel (MJ/kg)
        electricity_consumption_kwh: Electrical energy consumed (kWh)
        production_output: Total production output in period
        production_unit: Unit of production measurement
        product_weight_kg: Average weight per unit (if pieces)
        operating_hours: Actual hours furnace was operating
        available_hours: Hours furnace was available for operation
        scheduled_hours: Total scheduled operating hours
        downtime_hours: Total unplanned downtime (hours)
        planned_maintenance_hours: Scheduled maintenance time (hours)
        startup_count: Number of cold/warm starts in period
        avg_furnace_temp_c: Average operating temperature (degC)
        design_capacity_kg_hr: Design throughput capacity (kg/hr)
        thermal_efficiency_percent: Measured thermal efficiency (%)
        co2_emissions_kg: Total CO2 emissions in period (kg)
        nox_emissions_kg: Total NOx emissions in period (kg)
        natural_gas_price_per_mj: Fuel cost (currency/MJ)
        electricity_price_per_kwh: Electricity cost (currency/kWh)
        product_value_per_unit: Product value (currency/unit)
    """
    furnace_id: str
    period_hours: float
    fuel_consumption_kg: float
    fuel_type: str = "natural_gas"
    fuel_lhv_mj_kg: float = 45.5
    electricity_consumption_kwh: float = 0.0
    production_output: float = 0.0
    production_unit: ProductUnit = ProductUnit.TONNES
    product_weight_kg: float = 1.0
    operating_hours: float = 0.0
    available_hours: float = 0.0
    scheduled_hours: float = 0.0
    downtime_hours: float = 0.0
    planned_maintenance_hours: float = 0.0
    startup_count: int = 0
    avg_furnace_temp_c: float = 1000.0
    design_capacity_kg_hr: float = 10000.0
    thermal_efficiency_percent: float = 80.0
    co2_emissions_kg: float = 0.0
    nox_emissions_kg: float = 0.0
    natural_gas_price_per_mj: float = 0.015
    electricity_price_per_kwh: float = 0.10
    product_value_per_unit: float = 100.0


@dataclass
class KPIResult:
    """
    Result container for a single KPI calculation.

    Attributes:
        kpi_name: Name of the KPI
        kpi_category: Category classification
        value: Calculated value
        unit: Engineering unit
        target: Target/benchmark value (if applicable)
        variance_percent: Variance from target (%)
        trend: Trend indicator ('improving', 'stable', 'declining')
        formula: Formula used for calculation
        standard_reference: Reference to applicable standard
    """
    kpi_name: str
    kpi_category: KPICategory
    value: Decimal
    unit: str
    target: Optional[Decimal] = None
    variance_percent: Optional[Decimal] = None
    trend: Optional[str] = None
    formula: Optional[str] = None
    standard_reference: Optional[str] = None


@dataclass
class PerformanceKPIResult:
    """
    Complete result of furnace performance KPI calculations.

    Contains all calculated KPIs with full provenance tracking.
    """
    # Energy Efficiency KPIs
    specific_fuel_consumption_mj_kg: Decimal
    specific_fuel_consumption_gj_t: Decimal
    heat_rate_mj_kg: Decimal
    energy_intensity_kwh_kg: Decimal
    thermal_efficiency_percent: Decimal

    # Production Efficiency KPIs
    production_rate_kg_hr: Decimal
    capacity_utilization_percent: Decimal
    yield_percent: Decimal
    oee_percent: Decimal  # Overall Equipment Effectiveness

    # Availability KPIs
    availability_percent: Decimal
    reliability_percent: Decimal
    mtbf_hours: Decimal  # Mean Time Between Failures
    mttr_hours: Decimal  # Mean Time To Repair

    # Utilization KPIs
    utilization_percent: Decimal
    operating_factor_percent: Decimal
    load_factor_percent: Decimal

    # Environmental KPIs
    co2_intensity_kg_t: Decimal
    nox_intensity_g_gj: Decimal
    specific_emissions_kg_co2_mj: Decimal

    # Economic KPIs
    energy_cost_per_unit: Decimal
    energy_cost_per_tonne: Decimal
    productivity_units_per_gj: Decimal

    # All KPIs as list
    kpi_list: List[KPIResult]

    # Provenance
    provenance: ProvenanceRecord


class PerformanceKPICalculator:
    """
    Deterministic Performance KPI Calculator for Industrial Furnaces.

    Calculates comprehensive furnace performance KPIs following ISO 50006
    methodology for Energy Performance Indicators (EnPIs). All calculations
    are deterministic and produce bit-perfect reproducible results.

    Zero-Hallucination Guarantees:
    - Pure mathematical calculations using Decimal arithmetic
    - No LLM inference or probabilistic methods
    - Complete provenance tracking with SHA-256 hashing
    - Formulas from published standards (ISO 50006, ASME PTC 4.2)

    KPI Categories:
    1. Energy Efficiency: SFC, Heat Rate, Energy Intensity
    2. Production Efficiency: Production Rate, Capacity Utilization, OEE
    3. Availability: Availability, Reliability, MTBF, MTTR
    4. Utilization: Operating Factor, Load Factor
    5. Environmental: CO2 Intensity, NOx Intensity
    6. Economic: Energy Cost per Unit

    Example:
        >>> calculator = PerformanceKPICalculator()
        >>> operating_data = FurnaceOperatingData(
        ...     furnace_id="FURNACE-001",
        ...     period_hours=720,  # 30 days
        ...     fuel_consumption_kg=50000,
        ...     production_output=1000,  # tonnes
        ...     operating_hours=650,
        ...     available_hours=700
        ... )
        >>> result = calculator.calculate(operating_data)
        >>> print(f"SFC: {result.specific_fuel_consumption_gj_t} GJ/t")
    """

    # Industry benchmark values (for variance calculation)
    BENCHMARK_VALUES = {
        "sfc_gj_t_reheat": Decimal("1.8"),
        "sfc_gj_t_heat_treatment": Decimal("2.5"),
        "sfc_gj_t_melting": Decimal("3.5"),
        "availability_percent": Decimal("95.0"),
        "utilization_percent": Decimal("85.0"),
        "oee_percent": Decimal("80.0"),
    }

    def __init__(self, version: str = "1.0.0"):
        """
        Initialize the Performance KPI Calculator.

        Args:
            version: Calculator version for provenance tracking
        """
        self.version = version

    def calculate(
        self,
        operating_data: FurnaceOperatingData,
        calculation_id: Optional[str] = None
    ) -> PerformanceKPIResult:
        """
        Calculate all furnace performance KPIs with complete provenance.

        Args:
            operating_data: FurnaceOperatingData with all required measurements
            calculation_id: Optional unique identifier for this calculation

        Returns:
            PerformanceKPIResult with all KPIs and provenance

        Raises:
            ValueError: If input data fails validation
        """
        # Validate inputs
        self._validate_inputs(operating_data)

        # Initialize provenance tracker
        calc_id = calculation_id or f"perf_kpi_{id(operating_data)}"
        tracker = ProvenanceTracker(
            calculation_id=calc_id,
            calculation_type=CalculationCategory.PERFORMANCE_KPI.value,
            version=self.version,
            standard_compliance=["ISO 50006", "ASME PTC 4.2", "EN 16231"]
        )

        # Record all inputs
        tracker.record_inputs(self._serialize_inputs(operating_data))

        # Calculate all KPI categories
        kpi_list: List[KPIResult] = []

        # 1. Energy Efficiency KPIs
        sfc_mj_kg, sfc_gj_t = self._calculate_specific_fuel_consumption(
            operating_data, tracker
        )
        kpi_list.append(KPIResult(
            kpi_name="Specific Fuel Consumption",
            kpi_category=KPICategory.ENERGY_EFFICIENCY,
            value=sfc_gj_t,
            unit="GJ/t",
            target=self.BENCHMARK_VALUES["sfc_gj_t_reheat"],
            formula="SFC = (Fuel Mass * LHV) / Production Output",
            standard_reference="ISO 50006 Section 6.3"
        ))

        heat_rate = self._calculate_heat_rate(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Heat Rate",
            kpi_category=KPICategory.ENERGY_EFFICIENCY,
            value=heat_rate,
            unit="MJ/kg",
            formula="HR = Total Energy Input / Useful Heat Output",
            standard_reference="ASME PTC 4.2"
        ))

        energy_intensity = self._calculate_energy_intensity(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Energy Intensity",
            kpi_category=KPICategory.ENERGY_EFFICIENCY,
            value=energy_intensity,
            unit="kWh/kg",
            formula="EI = Total Energy (kWh) / Production (kg)"
        ))

        thermal_eff = Decimal(str(operating_data.thermal_efficiency_percent))

        # 2. Production Efficiency KPIs
        prod_rate = self._calculate_production_rate(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Production Rate",
            kpi_category=KPICategory.PRODUCTION_EFFICIENCY,
            value=prod_rate,
            unit="kg/hr",
            formula="PR = Production Output / Operating Hours"
        ))

        capacity_util = self._calculate_capacity_utilization(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Capacity Utilization",
            kpi_category=KPICategory.PRODUCTION_EFFICIENCY,
            value=capacity_util,
            unit="%",
            target=self.BENCHMARK_VALUES["utilization_percent"],
            formula="CU = Actual Production Rate / Design Capacity * 100"
        ))

        yield_percent = self._calculate_yield(operating_data, tracker)

        oee = self._calculate_oee(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Overall Equipment Effectiveness",
            kpi_category=KPICategory.PRODUCTION_EFFICIENCY,
            value=oee,
            unit="%",
            target=self.BENCHMARK_VALUES["oee_percent"],
            formula="OEE = Availability * Performance * Quality",
            standard_reference="ISO 22400"
        ))

        # 3. Availability KPIs
        availability = self._calculate_availability(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Availability",
            kpi_category=KPICategory.AVAILABILITY,
            value=availability,
            unit="%",
            target=self.BENCHMARK_VALUES["availability_percent"],
            formula="A = (Scheduled Time - Downtime) / Scheduled Time * 100"
        ))

        reliability = self._calculate_reliability(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Reliability",
            kpi_category=KPICategory.AVAILABILITY,
            value=reliability,
            unit="%",
            formula="R = Operating Hours / (Operating Hours + Unplanned Downtime) * 100"
        ))

        mtbf = self._calculate_mtbf(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Mean Time Between Failures",
            kpi_category=KPICategory.AVAILABILITY,
            value=mtbf,
            unit="hours",
            formula="MTBF = Operating Hours / Number of Failures"
        ))

        mttr = self._calculate_mttr(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Mean Time To Repair",
            kpi_category=KPICategory.AVAILABILITY,
            value=mttr,
            unit="hours",
            formula="MTTR = Total Repair Time / Number of Repairs"
        ))

        # 4. Utilization KPIs
        utilization = self._calculate_utilization(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Utilization",
            kpi_category=KPICategory.UTILIZATION,
            value=utilization,
            unit="%",
            formula="U = Operating Hours / Period Hours * 100"
        ))

        operating_factor = self._calculate_operating_factor(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Operating Factor",
            kpi_category=KPICategory.UTILIZATION,
            value=operating_factor,
            unit="%",
            formula="OF = Operating Hours / Available Hours * 100"
        ))

        load_factor = self._calculate_load_factor(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Load Factor",
            kpi_category=KPICategory.UTILIZATION,
            value=load_factor,
            unit="%",
            formula="LF = Average Load / Design Capacity * 100"
        ))

        # 5. Environmental KPIs
        co2_intensity = self._calculate_co2_intensity(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="CO2 Intensity",
            kpi_category=KPICategory.ENVIRONMENTAL,
            value=co2_intensity,
            unit="kg CO2/t",
            formula="CI = Total CO2 Emissions / Production Output"
        ))

        nox_intensity = self._calculate_nox_intensity(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="NOx Intensity",
            kpi_category=KPICategory.ENVIRONMENTAL,
            value=nox_intensity,
            unit="g/GJ",
            formula="NI = NOx Emissions (g) / Heat Input (GJ)"
        ))

        specific_emissions = self._calculate_specific_emissions(operating_data, tracker)

        # 6. Economic KPIs
        energy_cost_unit = self._calculate_energy_cost_per_unit(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Energy Cost per Unit",
            kpi_category=KPICategory.ECONOMIC,
            value=energy_cost_unit,
            unit="currency/unit",
            formula="ECU = Total Energy Cost / Production Units"
        ))

        energy_cost_tonne = self._calculate_energy_cost_per_tonne(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Energy Cost per Tonne",
            kpi_category=KPICategory.ECONOMIC,
            value=energy_cost_tonne,
            unit="currency/t",
            formula="ECT = Total Energy Cost / Production (tonnes)"
        ))

        productivity = self._calculate_productivity(operating_data, tracker)
        kpi_list.append(KPIResult(
            kpi_name="Productivity",
            kpi_category=KPICategory.ECONOMIC,
            value=productivity,
            unit="units/GJ",
            formula="P = Production Units / Energy Consumed (GJ)"
        ))

        # Calculate variances for KPIs with targets
        for kpi in kpi_list:
            if kpi.target is not None and kpi.target > 0:
                if kpi.kpi_category in [KPICategory.ENERGY_EFFICIENCY, KPICategory.ENVIRONMENTAL]:
                    # Lower is better
                    variance = ((kpi.target - kpi.value) / kpi.target * Decimal("100"))
                else:
                    # Higher is better
                    variance = ((kpi.value - kpi.target) / kpi.target * Decimal("100"))
                kpi.variance_percent = variance.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        # Get provenance record
        provenance = tracker.get_provenance_record(final_result=sfc_gj_t)

        return PerformanceKPIResult(
            specific_fuel_consumption_mj_kg=sfc_mj_kg,
            specific_fuel_consumption_gj_t=sfc_gj_t,
            heat_rate_mj_kg=heat_rate,
            energy_intensity_kwh_kg=energy_intensity,
            thermal_efficiency_percent=thermal_eff,
            production_rate_kg_hr=prod_rate,
            capacity_utilization_percent=capacity_util,
            yield_percent=yield_percent,
            oee_percent=oee,
            availability_percent=availability,
            reliability_percent=reliability,
            mtbf_hours=mtbf,
            mttr_hours=mttr,
            utilization_percent=utilization,
            operating_factor_percent=operating_factor,
            load_factor_percent=load_factor,
            co2_intensity_kg_t=co2_intensity,
            nox_intensity_g_gj=nox_intensity,
            specific_emissions_kg_co2_mj=specific_emissions,
            energy_cost_per_unit=energy_cost_unit,
            energy_cost_per_tonne=energy_cost_tonne,
            productivity_units_per_gj=productivity,
            kpi_list=kpi_list,
            provenance=provenance
        )

    def calculate_specific_fuel_consumption(
        self,
        fuel_consumption_kg: float,
        fuel_lhv_mj_kg: float,
        production_tonnes: float,
        calculation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate Specific Fuel Consumption (SFC) as a standalone calculation.

        SFC is a key energy efficiency indicator defined as the energy input
        required per unit of production output.

        Formula:
            SFC (GJ/t) = (Fuel Mass * LHV) / Production Output

        Args:
            fuel_consumption_kg: Total fuel consumed (kg)
            fuel_lhv_mj_kg: Lower heating value of fuel (MJ/kg)
            production_tonnes: Production output (tonnes)
            calculation_id: Optional calculation identifier

        Returns:
            Dictionary with SFC values in multiple units and provenance
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"sfc_{id(self)}",
            calculation_type=CalculationCategory.SPECIFIC_FUEL_CONSUMPTION.value,
            version=self.version,
            standard_compliance=["ISO 50006"]
        )

        fuel_kg = Decimal(str(fuel_consumption_kg))
        lhv = Decimal(str(fuel_lhv_mj_kg))
        production_t = Decimal(str(production_tonnes))

        tracker.record_inputs({
            "fuel_consumption_kg": fuel_kg,
            "fuel_lhv_mj_kg": lhv,
            "production_tonnes": production_t
        })

        # Calculate energy input
        energy_input_mj = fuel_kg * lhv
        energy_input_gj = energy_input_mj / Decimal("1000")

        tracker.record_step(
            operation="multiply_divide",
            description="Calculate total energy input from fuel",
            inputs={"fuel_kg": fuel_kg, "lhv_mj_kg": lhv},
            output_value=energy_input_gj,
            output_name="energy_input_gj",
            formula="E = Fuel * LHV / 1000",
            units="GJ"
        )

        # Calculate SFC
        if production_t > 0:
            sfc_gj_t = (energy_input_gj / production_t).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            sfc_mj_kg = (sfc_gj_t * Decimal("1000") / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            sfc_gj_t = Decimal("0")
            sfc_mj_kg = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Specific Fuel Consumption",
            inputs={
                "energy_input_gj": energy_input_gj,
                "production_tonnes": production_t
            },
            output_value=sfc_gj_t,
            output_name="sfc_gj_t",
            formula="SFC = Energy Input / Production",
            units="GJ/t",
            standard_reference="ISO 50006 Section 6.3"
        )

        provenance = tracker.get_provenance_record(sfc_gj_t)

        return {
            "sfc_gj_t": float(sfc_gj_t),
            "sfc_mj_kg": float(sfc_mj_kg),
            "sfc_kwh_kg": float(sfc_mj_kg / Decimal("3.6")),
            "energy_input_gj": float(energy_input_gj),
            "benchmark_gj_t": float(self.BENCHMARK_VALUES["sfc_gj_t_reheat"]),
            "provenance_hash": provenance.provenance_hash
        }

    def calculate_heat_rate(
        self,
        total_energy_input_mj: float,
        useful_heat_output_mj: float,
        calculation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate Heat Rate for the furnace.

        Heat Rate is the ratio of energy input to useful heat output,
        essentially the inverse of thermal efficiency.

        Formula:
            HR = Total Energy Input / Useful Heat Output

        Args:
            total_energy_input_mj: Total energy input (MJ)
            useful_heat_output_mj: Useful heat delivered to product (MJ)
            calculation_id: Optional calculation identifier

        Returns:
            Dictionary with heat rate and thermal efficiency
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"heat_rate_{id(self)}",
            calculation_type=CalculationCategory.HEAT_RATE.value,
            version=self.version,
            standard_compliance=["ASME PTC 4.2"]
        )

        energy_in = Decimal(str(total_energy_input_mj))
        heat_out = Decimal(str(useful_heat_output_mj))

        tracker.record_inputs({
            "total_energy_input_mj": energy_in,
            "useful_heat_output_mj": heat_out
        })

        if heat_out > 0:
            heat_rate = (energy_in / heat_out).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            thermal_efficiency = (heat_out / energy_in * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            heat_rate = Decimal("0")
            thermal_efficiency = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Heat Rate",
            inputs={"energy_input_mj": energy_in, "heat_output_mj": heat_out},
            output_value=heat_rate,
            output_name="heat_rate",
            formula="HR = E_in / Q_useful",
            units="MJ/MJ",
            standard_reference="ASME PTC 4.2"
        )

        provenance = tracker.get_provenance_record(heat_rate)

        return {
            "heat_rate": float(heat_rate),
            "thermal_efficiency_percent": float(thermal_efficiency),
            "provenance_hash": provenance.provenance_hash
        }

    def calculate_availability_metrics(
        self,
        scheduled_hours: float,
        operating_hours: float,
        downtime_hours: float,
        planned_maintenance_hours: float,
        failure_count: int,
        calculation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate availability and reliability metrics.

        Calculates Availability, Reliability, MTBF, and MTTR based on
        operating time data.

        Formulas:
            Availability = (Scheduled - Downtime) / Scheduled * 100
            MTBF = Operating Hours / Number of Failures
            MTTR = Downtime / Number of Failures

        Args:
            scheduled_hours: Total scheduled operating hours
            operating_hours: Actual operating hours
            downtime_hours: Unplanned downtime hours
            planned_maintenance_hours: Scheduled maintenance hours
            failure_count: Number of failures/breakdowns
            calculation_id: Optional calculation identifier

        Returns:
            Dictionary with availability metrics
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"availability_{id(self)}",
            calculation_type=CalculationCategory.AVAILABILITY.value,
            version=self.version,
            standard_compliance=["ISO 22400"]
        )

        sched = Decimal(str(scheduled_hours))
        oper = Decimal(str(operating_hours))
        down = Decimal(str(downtime_hours))
        maint = Decimal(str(planned_maintenance_hours))
        failures = failure_count

        tracker.record_inputs({
            "scheduled_hours": sched,
            "operating_hours": oper,
            "downtime_hours": down,
            "planned_maintenance_hours": maint,
            "failure_count": failures
        })

        # Availability
        if sched > 0:
            available_time = sched - maint
            availability = ((available_time - down) / available_time * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            availability = Decimal("0")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Availability",
            inputs={"scheduled": sched, "downtime": down, "maintenance": maint},
            output_value=availability,
            output_name="availability_percent",
            formula="A = (Scheduled - Maintenance - Downtime) / (Scheduled - Maintenance) * 100",
            units="%"
        )

        # MTBF
        if failures > 0:
            mtbf = (oper / Decimal(str(failures))).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            mtbf = oper  # No failures, MTBF equals operating time

        tracker.record_step(
            operation="divide",
            description="Calculate Mean Time Between Failures",
            inputs={"operating_hours": oper, "failure_count": failures},
            output_value=mtbf,
            output_name="mtbf_hours",
            formula="MTBF = Operating Hours / Failures",
            units="hours"
        )

        # MTTR
        if failures > 0:
            mttr = (down / Decimal(str(failures))).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            mttr = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Mean Time To Repair",
            inputs={"downtime_hours": down, "failure_count": failures},
            output_value=mttr,
            output_name="mttr_hours",
            formula="MTTR = Downtime / Failures",
            units="hours"
        )

        # Reliability
        if oper + down > 0:
            reliability = (oper / (oper + down) * Decimal("100")).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            reliability = Decimal("0")

        provenance = tracker.get_provenance_record(availability)

        return {
            "availability_percent": float(availability),
            "reliability_percent": float(reliability),
            "mtbf_hours": float(mtbf),
            "mttr_hours": float(mttr),
            "benchmark_availability_percent": float(self.BENCHMARK_VALUES["availability_percent"]),
            "provenance_hash": provenance.provenance_hash
        }

    def calculate_oee(
        self,
        availability_percent: float,
        performance_percent: float,
        quality_percent: float,
        calculation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate Overall Equipment Effectiveness (OEE).

        OEE is a comprehensive metric combining Availability, Performance,
        and Quality factors per ISO 22400.

        Formula:
            OEE = Availability * Performance * Quality / 10000

        Args:
            availability_percent: Availability factor (%)
            performance_percent: Performance factor (%)
            quality_percent: Quality factor (%)
            calculation_id: Optional calculation identifier

        Returns:
            Dictionary with OEE and component factors
        """
        tracker = ProvenanceTracker(
            calculation_id=calculation_id or f"oee_{id(self)}",
            calculation_type="oee",
            version=self.version,
            standard_compliance=["ISO 22400"]
        )

        avail = Decimal(str(availability_percent))
        perf = Decimal(str(performance_percent))
        qual = Decimal(str(quality_percent))

        tracker.record_inputs({
            "availability_percent": avail,
            "performance_percent": perf,
            "quality_percent": qual
        })

        oee = (avail * perf * qual / Decimal("10000")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="multiply_divide",
            description="Calculate Overall Equipment Effectiveness",
            inputs={"availability": avail, "performance": perf, "quality": qual},
            output_value=oee,
            output_name="oee_percent",
            formula="OEE = A * P * Q / 10000",
            units="%",
            standard_reference="ISO 22400"
        )

        provenance = tracker.get_provenance_record(oee)

        return {
            "oee_percent": float(oee),
            "availability_percent": float(avail),
            "performance_percent": float(perf),
            "quality_percent": float(qual),
            "benchmark_oee_percent": float(self.BENCHMARK_VALUES["oee_percent"]),
            "provenance_hash": provenance.provenance_hash
        }

    # ========================================================================
    # PRIVATE CALCULATION METHODS
    # ========================================================================

    def _validate_inputs(self, data: FurnaceOperatingData) -> None:
        """Validate operating data inputs."""
        if data.period_hours <= 0:
            raise ValueError("Period hours must be positive")
        if data.fuel_consumption_kg < 0:
            raise ValueError("Fuel consumption cannot be negative")
        if data.operating_hours < 0:
            raise ValueError("Operating hours cannot be negative")
        if data.operating_hours > data.period_hours:
            raise ValueError("Operating hours cannot exceed period hours")

    def _serialize_inputs(self, data: FurnaceOperatingData) -> Dict[str, Any]:
        """Serialize input data for provenance tracking."""
        return {
            "furnace_id": data.furnace_id,
            "period_hours": data.period_hours,
            "fuel_consumption_kg": data.fuel_consumption_kg,
            "fuel_type": data.fuel_type,
            "fuel_lhv_mj_kg": data.fuel_lhv_mj_kg,
            "electricity_consumption_kwh": data.electricity_consumption_kwh,
            "production_output": data.production_output,
            "production_unit": data.production_unit.value,
            "operating_hours": data.operating_hours,
            "available_hours": data.available_hours,
            "scheduled_hours": data.scheduled_hours,
            "downtime_hours": data.downtime_hours,
            "design_capacity_kg_hr": data.design_capacity_kg_hr,
            "thermal_efficiency_percent": data.thermal_efficiency_percent
        }

    def _calculate_specific_fuel_consumption(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Tuple[Decimal, Decimal]:
        """Calculate SFC in MJ/kg and GJ/t."""
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))

        # Convert production to kg
        if data.production_unit == ProductUnit.TONNES:
            production_kg = Decimal(str(data.production_output)) * Decimal("1000")
        else:
            production_kg = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg))

        energy_input_mj = fuel_kg * lhv

        if production_kg > 0:
            sfc_mj_kg = (energy_input_mj / production_kg).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
            sfc_gj_t = (sfc_mj_kg * Decimal("1000") / Decimal("1000")).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            sfc_mj_kg = Decimal("0")
            sfc_gj_t = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Specific Fuel Consumption",
            inputs={
                "energy_input_mj": energy_input_mj,
                "production_kg": production_kg
            },
            output_value=sfc_gj_t,
            output_name="sfc_gj_t",
            formula="SFC = (Fuel * LHV) / Production",
            units="GJ/t",
            standard_reference="ISO 50006"
        )

        return sfc_mj_kg, sfc_gj_t

    def _calculate_heat_rate(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate heat rate (inverse of efficiency)."""
        eff = Decimal(str(data.thermal_efficiency_percent))

        if eff > 0:
            heat_rate = (Decimal("100") / eff).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            heat_rate = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Heat Rate from efficiency",
            inputs={"thermal_efficiency_percent": eff},
            output_value=heat_rate,
            output_name="heat_rate",
            formula="HR = 100 / Efficiency",
            units="MJ/MJ"
        )

        return heat_rate

    def _calculate_energy_intensity(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate energy intensity in kWh/kg."""
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))
        elec_kwh = Decimal(str(data.electricity_consumption_kwh))

        # Total energy in kWh (1 MJ = 0.2778 kWh)
        fuel_energy_kwh = fuel_kg * lhv * Decimal("0.2778")
        total_energy_kwh = fuel_energy_kwh + elec_kwh

        # Production in kg
        if data.production_unit == ProductUnit.TONNES:
            production_kg = Decimal(str(data.production_output)) * Decimal("1000")
        else:
            production_kg = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg))

        if production_kg > 0:
            energy_intensity = (total_energy_kwh / production_kg).quantize(
                Decimal("0.001"), rounding=ROUND_HALF_UP
            )
        else:
            energy_intensity = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Energy Intensity",
            inputs={
                "total_energy_kwh": total_energy_kwh,
                "production_kg": production_kg
            },
            output_value=energy_intensity,
            output_name="energy_intensity_kwh_kg",
            formula="EI = Total Energy / Production",
            units="kWh/kg"
        )

        return energy_intensity

    def _calculate_production_rate(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate average production rate."""
        oper_hours = Decimal(str(data.operating_hours))

        if data.production_unit == ProductUnit.TONNES:
            production_kg = Decimal(str(data.production_output)) * Decimal("1000")
        else:
            production_kg = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg))

        if oper_hours > 0:
            prod_rate = (production_kg / oper_hours).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            prod_rate = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Production Rate",
            inputs={"production_kg": production_kg, "operating_hours": oper_hours},
            output_value=prod_rate,
            output_name="production_rate_kg_hr",
            formula="PR = Production / Operating Hours",
            units="kg/hr"
        )

        return prod_rate

    def _calculate_capacity_utilization(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate capacity utilization percentage."""
        design_cap = Decimal(str(data.design_capacity_kg_hr))
        oper_hours = Decimal(str(data.operating_hours))

        if data.production_unit == ProductUnit.TONNES:
            production_kg = Decimal(str(data.production_output)) * Decimal("1000")
        else:
            production_kg = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg))

        if oper_hours > 0 and design_cap > 0:
            actual_rate = production_kg / oper_hours
            capacity_util = (actual_rate / design_cap * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            capacity_util = Decimal("0")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Capacity Utilization",
            inputs={
                "production_kg": production_kg,
                "operating_hours": oper_hours,
                "design_capacity_kg_hr": design_cap
            },
            output_value=capacity_util,
            output_name="capacity_utilization_percent",
            formula="CU = (Production / Hours) / Design Capacity * 100",
            units="%"
        )

        return capacity_util

    def _calculate_yield(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate yield (assume 98% for calculation purposes)."""
        # In real implementation, would use actual good/total production data
        yield_percent = Decimal("98.0")

        tracker.record_step(
            operation="constant",
            description="Apply standard yield assumption",
            inputs={"standard_yield": yield_percent},
            output_value=yield_percent,
            output_name="yield_percent",
            formula="Assumed standard yield",
            units="%"
        )

        return yield_percent

    def _calculate_oee(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate OEE = Availability * Performance * Quality."""
        availability = self._calculate_availability(data, tracker)
        performance = self._calculate_capacity_utilization(data, tracker)
        quality = Decimal("98.0")  # Assumed yield

        oee = (availability * performance * quality / Decimal("10000")).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        tracker.record_step(
            operation="multiply_divide",
            description="Calculate Overall Equipment Effectiveness",
            inputs={
                "availability": availability,
                "performance": performance,
                "quality": quality
            },
            output_value=oee,
            output_name="oee_percent",
            formula="OEE = A * P * Q / 10000",
            units="%",
            standard_reference="ISO 22400"
        )

        return oee

    def _calculate_availability(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate availability percentage."""
        sched = Decimal(str(data.scheduled_hours)) if data.scheduled_hours > 0 else Decimal(str(data.period_hours))
        down = Decimal(str(data.downtime_hours))
        maint = Decimal(str(data.planned_maintenance_hours))

        available_time = sched - maint
        if available_time > 0:
            availability = ((available_time - down) / available_time * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            availability = Decimal("0")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Availability",
            inputs={"scheduled": sched, "downtime": down, "maintenance": maint},
            output_value=availability,
            output_name="availability_percent",
            formula="A = (Scheduled - Maintenance - Downtime) / (Scheduled - Maintenance) * 100",
            units="%"
        )

        return availability

    def _calculate_reliability(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate reliability percentage."""
        oper = Decimal(str(data.operating_hours))
        down = Decimal(str(data.downtime_hours))

        if oper + down > 0:
            reliability = (oper / (oper + down) * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            reliability = Decimal("100")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Reliability",
            inputs={"operating_hours": oper, "downtime_hours": down},
            output_value=reliability,
            output_name="reliability_percent",
            formula="R = Operating / (Operating + Downtime) * 100",
            units="%"
        )

        return reliability

    def _calculate_mtbf(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate Mean Time Between Failures."""
        oper = Decimal(str(data.operating_hours))
        failures = data.startup_count  # Using startup count as proxy for failures

        if failures > 0:
            mtbf = (oper / Decimal(str(failures))).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            mtbf = oper

        tracker.record_step(
            operation="divide",
            description="Calculate Mean Time Between Failures",
            inputs={"operating_hours": oper, "failure_count": failures},
            output_value=mtbf,
            output_name="mtbf_hours",
            formula="MTBF = Operating Hours / Failures",
            units="hours"
        )

        return mtbf

    def _calculate_mttr(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate Mean Time To Repair."""
        down = Decimal(str(data.downtime_hours))
        failures = data.startup_count

        if failures > 0:
            mttr = (down / Decimal(str(failures))).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            mttr = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Mean Time To Repair",
            inputs={"downtime_hours": down, "failure_count": failures},
            output_value=mttr,
            output_name="mttr_hours",
            formula="MTTR = Downtime / Failures",
            units="hours"
        )

        return mttr

    def _calculate_utilization(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate utilization percentage."""
        oper = Decimal(str(data.operating_hours))
        period = Decimal(str(data.period_hours))

        if period > 0:
            utilization = (oper / period * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            utilization = Decimal("0")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Utilization",
            inputs={"operating_hours": oper, "period_hours": period},
            output_value=utilization,
            output_name="utilization_percent",
            formula="U = Operating / Period * 100",
            units="%"
        )

        return utilization

    def _calculate_operating_factor(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate operating factor percentage."""
        oper = Decimal(str(data.operating_hours))
        avail = Decimal(str(data.available_hours)) if data.available_hours > 0 else Decimal(str(data.period_hours))

        if avail > 0:
            operating_factor = (oper / avail * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            operating_factor = Decimal("0")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Operating Factor",
            inputs={"operating_hours": oper, "available_hours": avail},
            output_value=operating_factor,
            output_name="operating_factor_percent",
            formula="OF = Operating / Available * 100",
            units="%"
        )

        return operating_factor

    def _calculate_load_factor(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate load factor percentage."""
        design_cap = Decimal(str(data.design_capacity_kg_hr))
        oper_hours = Decimal(str(data.operating_hours))

        if data.production_unit == ProductUnit.TONNES:
            production_kg = Decimal(str(data.production_output)) * Decimal("1000")
        else:
            production_kg = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg))

        if oper_hours > 0 and design_cap > 0:
            avg_load = production_kg / oper_hours
            load_factor = (avg_load / design_cap * Decimal("100")).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            load_factor = Decimal("0")

        tracker.record_step(
            operation="divide_multiply",
            description="Calculate Load Factor",
            inputs={
                "production_kg": production_kg,
                "operating_hours": oper_hours,
                "design_capacity": design_cap
            },
            output_value=load_factor,
            output_name="load_factor_percent",
            formula="LF = Average Load / Design Capacity * 100",
            units="%"
        )

        return load_factor

    def _calculate_co2_intensity(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate CO2 intensity (kg CO2 per tonne of product)."""
        co2_kg = Decimal(str(data.co2_emissions_kg))

        if data.production_unit == ProductUnit.TONNES:
            production_t = Decimal(str(data.production_output))
        else:
            production_t = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg)) / Decimal("1000")

        if production_t > 0:
            co2_intensity = (co2_kg / production_t).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            # Calculate from fuel if CO2 not measured directly
            fuel_kg = Decimal(str(data.fuel_consumption_kg))
            lhv = Decimal(str(data.fuel_lhv_mj_kg))
            ef = Decimal("0.0561")  # Natural gas emission factor kg CO2/MJ
            co2_kg = fuel_kg * lhv * ef

            if production_t > 0:
                co2_intensity = (co2_kg / production_t).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
            else:
                co2_intensity = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate CO2 Intensity",
            inputs={"co2_emissions_kg": co2_kg, "production_tonnes": production_t},
            output_value=co2_intensity,
            output_name="co2_intensity_kg_t",
            formula="CI = CO2 Emissions / Production",
            units="kg CO2/t"
        )

        return co2_intensity

    def _calculate_nox_intensity(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate NOx intensity (g NOx per GJ of fuel)."""
        nox_kg = Decimal(str(data.nox_emissions_kg))
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))

        energy_gj = fuel_kg * lhv / Decimal("1000")

        if energy_gj > 0:
            nox_intensity = (nox_kg * Decimal("1000") / energy_gj).quantize(
                Decimal("0.1"), rounding=ROUND_HALF_UP
            )
        else:
            nox_intensity = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate NOx Intensity",
            inputs={"nox_emissions_kg": nox_kg, "energy_input_gj": energy_gj},
            output_value=nox_intensity,
            output_name="nox_intensity_g_gj",
            formula="NI = NOx (g) / Energy (GJ)",
            units="g/GJ"
        )

        return nox_intensity

    def _calculate_specific_emissions(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate specific CO2 emissions per MJ of energy input."""
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))

        energy_mj = fuel_kg * lhv

        # Use standard emission factor for natural gas
        ef = Decimal("0.0561")  # kg CO2/MJ

        tracker.record_step(
            operation="constant",
            description="Apply standard CO2 emission factor",
            inputs={"fuel_type": data.fuel_type, "emission_factor": ef},
            output_value=ef,
            output_name="specific_emissions_kg_co2_mj",
            formula="Standard emission factor for fuel type",
            units="kg CO2/MJ"
        )

        return ef

    def _calculate_energy_cost_per_unit(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate energy cost per production unit."""
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))
        fuel_price = Decimal(str(data.natural_gas_price_per_mj))
        elec_kwh = Decimal(str(data.electricity_consumption_kwh))
        elec_price = Decimal(str(data.electricity_price_per_kwh))
        production = Decimal(str(data.production_output))

        fuel_cost = fuel_kg * lhv * fuel_price
        elec_cost = elec_kwh * elec_price
        total_cost = fuel_cost + elec_cost

        if production > 0:
            cost_per_unit = (total_cost / production).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            cost_per_unit = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Energy Cost per Production Unit",
            inputs={
                "fuel_cost": fuel_cost,
                "electricity_cost": elec_cost,
                "production_units": production
            },
            output_value=cost_per_unit,
            output_name="energy_cost_per_unit",
            formula="ECU = (Fuel Cost + Electricity Cost) / Production",
            units="currency/unit"
        )

        return cost_per_unit

    def _calculate_energy_cost_per_tonne(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate energy cost per tonne of product."""
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))
        fuel_price = Decimal(str(data.natural_gas_price_per_mj))
        elec_kwh = Decimal(str(data.electricity_consumption_kwh))
        elec_price = Decimal(str(data.electricity_price_per_kwh))

        if data.production_unit == ProductUnit.TONNES:
            production_t = Decimal(str(data.production_output))
        else:
            production_t = Decimal(str(data.production_output)) * Decimal(str(data.product_weight_kg)) / Decimal("1000")

        fuel_cost = fuel_kg * lhv * fuel_price
        elec_cost = elec_kwh * elec_price
        total_cost = fuel_cost + elec_cost

        if production_t > 0:
            cost_per_tonne = (total_cost / production_t).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            cost_per_tonne = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Energy Cost per Tonne",
            inputs={
                "total_energy_cost": total_cost,
                "production_tonnes": production_t
            },
            output_value=cost_per_tonne,
            output_name="energy_cost_per_tonne",
            formula="ECT = Total Energy Cost / Production (t)",
            units="currency/t"
        )

        return cost_per_tonne

    def _calculate_productivity(
        self,
        data: FurnaceOperatingData,
        tracker: ProvenanceTracker
    ) -> Decimal:
        """Calculate productivity (units per GJ of energy)."""
        fuel_kg = Decimal(str(data.fuel_consumption_kg))
        lhv = Decimal(str(data.fuel_lhv_mj_kg))
        production = Decimal(str(data.production_output))

        energy_gj = fuel_kg * lhv / Decimal("1000")

        if energy_gj > 0:
            productivity = (production / energy_gj).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
        else:
            productivity = Decimal("0")

        tracker.record_step(
            operation="divide",
            description="Calculate Productivity",
            inputs={"production_units": production, "energy_gj": energy_gj},
            output_value=productivity,
            output_name="productivity_units_per_gj",
            formula="P = Production / Energy",
            units="units/GJ"
        )

        return productivity
