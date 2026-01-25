# -*- coding: utf-8 -*-
"""
Savings Calculator - Water, Energy, and Emissions Savings

This module calculates water savings, energy savings, and avoided emissions
from blowdown optimization with uncertainty quantification.

Author: GL-CalculatorEngineer
Version: 1.0.0
Standards: GHG Protocol, ISO 14064
Agent: GL-016_Waterguard

Zero Hallucination Guarantee:
- All calculations are deterministic
- Complete provenance tracking with SHA-256 hashes
- Uncertainty quantification based on sensor accuracy
- No LLM inference in calculation path
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import math

from .provenance import ProvenanceTracker, ProvenanceRecord, create_calculation_hash


class TimePeriod(Enum):
    """Supported time periods for savings calculations."""
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    YEAR = "year"


class FuelType(Enum):
    """Supported fuel types for emissions calculations."""
    NATURAL_GAS = "natural_gas"
    FUEL_OIL_2 = "fuel_oil_2"
    FUEL_OIL_6 = "fuel_oil_6"
    COAL_BITUMINOUS = "coal_bituminous"
    COAL_SUBBITUMINOUS = "coal_subbituminous"
    PROPANE = "propane"
    BIOMASS_WOOD = "biomass_wood"
    ELECTRICITY = "electricity"


@dataclass
class UncertaintyBand:
    """Uncertainty band for a calculated value."""
    lower_bound: Decimal
    central_value: Decimal
    upper_bound: Decimal
    confidence_level: Decimal = Decimal('0.95')  # 95% confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'lower_bound': float(self.lower_bound),
            'central_value': float(self.central_value),
            'upper_bound': float(self.upper_bound),
            'confidence_level': float(self.confidence_level)
        }


@dataclass
class SavingsResult:
    """Result of savings calculation with uncertainty bands."""
    value: Decimal
    unit: str
    uncertainty_bands: Optional[UncertaintyBand] = None
    time_period: str = "instantaneous"
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""
    baseline_value: Optional[Decimal] = None
    optimized_value: Optional[Decimal] = None
    savings_percent: Optional[Decimal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'value': float(self.value),
            'unit': self.unit,
            'uncertainty_bands': self.uncertainty_bands.to_dict() if self.uncertainty_bands else None,
            'time_period': self.time_period,
            'baseline_value': float(self.baseline_value) if self.baseline_value else None,
            'optimized_value': float(self.optimized_value) if self.optimized_value else None,
            'savings_percent': float(self.savings_percent) if self.savings_percent else None,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


@dataclass
class CumulativeSavings:
    """Cumulative savings over a time period."""
    water_savings_m3: Decimal
    energy_savings_gj: Decimal
    emissions_avoided_tco2e: Decimal
    cost_savings_usd: Decimal
    time_period_hours: Decimal
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None
    uncertainty_bands: Optional[Dict[str, UncertaintyBand]] = None
    provenance: Optional[ProvenanceRecord] = None
    calculation_hash: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'water_savings_m3': float(self.water_savings_m3),
            'energy_savings_gj': float(self.energy_savings_gj),
            'emissions_avoided_tco2e': float(self.emissions_avoided_tco2e),
            'cost_savings_usd': float(self.cost_savings_usd),
            'time_period_hours': float(self.time_period_hours),
            'start_timestamp': self.start_timestamp,
            'end_timestamp': self.end_timestamp,
            'uncertainty_bands': {k: v.to_dict() for k, v in self.uncertainty_bands.items()} if self.uncertainty_bands else None,
            'calculation_hash': self.calculation_hash,
            'provenance': self.provenance.to_dict() if self.provenance else None
        }


class EmissionFactorDatabase:
    """
    Emission factor database for common fuels.

    All factors are from authoritative sources:
    - EPA AP-42
    - IPCC Guidelines for National GHG Inventories
    - EIA CO2 Emission Coefficients
    """

    # Emission factors: kg CO2e per GJ of fuel energy
    # Source: IPCC 2006, EPA, EIA
    EMISSION_FACTORS = {
        FuelType.NATURAL_GAS: {
            'co2_kg_per_gj': Decimal('56.1'),
            'ch4_kg_per_gj': Decimal('0.001'),
            'n2o_kg_per_gj': Decimal('0.0001'),
            'source': 'IPCC 2006 Guidelines',
            'hhv_mj_per_unit': Decimal('38.3'),  # MJ/m3
            'unit': 'm3'
        },
        FuelType.FUEL_OIL_2: {
            'co2_kg_per_gj': Decimal('74.1'),
            'ch4_kg_per_gj': Decimal('0.003'),
            'n2o_kg_per_gj': Decimal('0.0006'),
            'source': 'EPA AP-42',
            'hhv_mj_per_unit': Decimal('38.68'),  # MJ/L
            'unit': 'L'
        },
        FuelType.FUEL_OIL_6: {
            'co2_kg_per_gj': Decimal('77.4'),
            'ch4_kg_per_gj': Decimal('0.003'),
            'n2o_kg_per_gj': Decimal('0.0006'),
            'source': 'EPA AP-42',
            'hhv_mj_per_unit': Decimal('42.5'),  # MJ/L
            'unit': 'L'
        },
        FuelType.COAL_BITUMINOUS: {
            'co2_kg_per_gj': Decimal('94.6'),
            'ch4_kg_per_gj': Decimal('0.01'),
            'n2o_kg_per_gj': Decimal('0.0015'),
            'source': 'IPCC 2006 Guidelines',
            'hhv_mj_per_unit': Decimal('26.2'),  # MJ/kg
            'unit': 'kg'
        },
        FuelType.COAL_SUBBITUMINOUS: {
            'co2_kg_per_gj': Decimal('96.1'),
            'ch4_kg_per_gj': Decimal('0.01'),
            'n2o_kg_per_gj': Decimal('0.0015'),
            'source': 'IPCC 2006 Guidelines',
            'hhv_mj_per_unit': Decimal('19.3'),  # MJ/kg
            'unit': 'kg'
        },
        FuelType.PROPANE: {
            'co2_kg_per_gj': Decimal('63.1'),
            'ch4_kg_per_gj': Decimal('0.001'),
            'n2o_kg_per_gj': Decimal('0.0001'),
            'source': 'EPA AP-42',
            'hhv_mj_per_unit': Decimal('50.35'),  # MJ/kg
            'unit': 'kg'
        },
        FuelType.BIOMASS_WOOD: {
            'co2_kg_per_gj': Decimal('0'),  # Biogenic - carbon neutral
            'ch4_kg_per_gj': Decimal('0.03'),
            'n2o_kg_per_gj': Decimal('0.004'),
            'source': 'IPCC 2006 Guidelines',
            'hhv_mj_per_unit': Decimal('15.0'),  # MJ/kg (dry)
            'unit': 'kg'
        },
        FuelType.ELECTRICITY: {
            'co2_kg_per_gj': Decimal('150'),  # Grid average (varies by region)
            'ch4_kg_per_gj': Decimal('0'),
            'n2o_kg_per_gj': Decimal('0'),
            'source': 'EIA Grid Average',
            'hhv_mj_per_unit': Decimal('3.6'),  # MJ/kWh
            'unit': 'kWh'
        }
    }

    # Global Warming Potentials (100-year, AR5)
    GWP = {
        'CO2': Decimal('1'),
        'CH4': Decimal('28'),
        'N2O': Decimal('265')
    }

    @classmethod
    def get_emission_factor(
        cls,
        fuel_type: FuelType,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Get total CO2e emission factor for fuel type.

        Returns: kg CO2e per GJ of fuel energy
        """
        factors = cls.EMISSION_FACTORS[fuel_type]

        co2 = factors['co2_kg_per_gj']
        ch4 = factors['ch4_kg_per_gj'] * cls.GWP['CH4']
        n2o = factors['n2o_kg_per_gj'] * cls.GWP['N2O']

        total = co2 + ch4 + n2o

        if tracker:
            tracker.record_step(
                operation="emission_factor_lookup",
                description=f"Get emission factor for {fuel_type.value}",
                inputs={
                    'fuel_type': fuel_type.value,
                    'co2_kg_per_gj': float(factors['co2_kg_per_gj']),
                    'ch4_kg_per_gj': float(factors['ch4_kg_per_gj']),
                    'n2o_kg_per_gj': float(factors['n2o_kg_per_gj'])
                },
                output_value=total,
                output_name="emission_factor_kg_co2e_per_gj",
                formula="EF = CO2 + CH4*GWP_CH4 + N2O*GWP_N2O",
                units="kg CO2e/GJ",
                source=factors['source']
            )

        return total.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)


class BaselineMethodology:
    """
    Baseline methodology for M&V calculations.

    Implements normalization for load and seasonality adjustments
    to ensure fair comparison between baseline and reporting periods.
    """

    @staticmethod
    def normalize_for_load(
        value: Decimal,
        baseline_load: Decimal,
        current_load: Decimal,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Normalize value for load differences.

        Formula: Normalized = Value * (Current_Load / Baseline_Load)

        Args:
            value: Value to normalize
            baseline_load: Baseline period load
            current_load: Current period load
            tracker: Optional provenance tracker

        Returns:
            Load-normalized value
        """
        if baseline_load == Decimal('0'):
            raise ValueError("Baseline load cannot be zero")

        load_ratio = current_load / baseline_load
        normalized = value * load_ratio

        if tracker:
            tracker.record_step(
                operation="load_normalization",
                description="Normalize value for load differences",
                inputs={
                    'original_value': value,
                    'baseline_load': baseline_load,
                    'current_load': current_load
                },
                output_value=normalized,
                output_name="normalized_value",
                formula="Normalized = Value * (Current_Load / Baseline_Load)",
                units="same as input"
            )

        return normalized.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)

    @staticmethod
    def normalize_for_seasonality(
        value: Decimal,
        baseline_season_factor: Decimal,
        current_season_factor: Decimal,
        tracker: Optional[ProvenanceTracker] = None
    ) -> Decimal:
        """
        Normalize value for seasonal differences.

        Args:
            value: Value to normalize
            baseline_season_factor: Seasonal factor for baseline period (0.8-1.2 typical)
            current_season_factor: Seasonal factor for current period
            tracker: Optional provenance tracker

        Returns:
            Seasonally-normalized value
        """
        if current_season_factor == Decimal('0'):
            raise ValueError("Current season factor cannot be zero")

        season_ratio = baseline_season_factor / current_season_factor
        normalized = value * season_ratio

        if tracker:
            tracker.record_step(
                operation="seasonal_normalization",
                description="Normalize value for seasonal differences",
                inputs={
                    'original_value': value,
                    'baseline_season_factor': baseline_season_factor,
                    'current_season_factor': current_season_factor
                },
                output_value=normalized,
                output_name="normalized_value",
                formula="Normalized = Value * (Baseline_SF / Current_SF)",
                units="same as input"
            )

        return normalized.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)


class SavingsCalculator:
    """
    Savings Calculator for Water, Energy, and Emissions.

    Calculates savings from blowdown optimization with:
    - Baseline methodology with normalization
    - Uncertainty quantification
    - Both instantaneous and cumulative tracking
    - Complete provenance

    Zero Hallucination Guarantee:
    - All calculations are deterministic
    - Complete provenance tracking with SHA-256 hashes
    - No LLM inference in calculation path
    """

    # Time period conversion factors (hours)
    TIME_FACTORS = {
        TimePeriod.HOUR: Decimal('1'),
        TimePeriod.DAY: Decimal('24'),
        TimePeriod.WEEK: Decimal('168'),
        TimePeriod.MONTH: Decimal('730'),  # Average month
        TimePeriod.YEAR: Decimal('8760')
    }

    def __init__(self, version: str = "1.0.0"):
        """Initialize savings calculator."""
        self.version = version
        self.emission_factors = EmissionFactorDatabase()
        self.baseline = BaselineMethodology()

        # Default sensor uncertainties
        self.sensor_uncertainties = {
            'flow_meter': Decimal('0.02'),      # 2% typical
            'tds_sensor': Decimal('0.03'),      # 3% typical
            'temperature': Decimal('0.01'),     # 1% typical
            'pressure': Decimal('0.005'),       # 0.5% typical
            'energy_meter': Decimal('0.02'),    # 2% typical
        }

    def calculate_water_savings(
        self,
        baseline_blowdown_kg_h: float,
        optimized_blowdown_kg_h: float,
        time_period: TimePeriod = TimePeriod.HOUR,
        include_uncertainty: bool = True
    ) -> SavingsResult:
        """
        Calculate water savings from blowdown optimization.

        Args:
            baseline_blowdown_kg_h: Baseline blowdown rate (kg/h)
            optimized_blowdown_kg_h: Optimized blowdown rate (kg/h)
            time_period: Time period for cumulative savings
            include_uncertainty: Include uncertainty quantification

        Returns:
            SavingsResult with water savings
        """
        tracker = ProvenanceTracker(
            calculation_id=f"water_savings_{time_period.value}",
            calculation_type="water_savings",
            version=self.version
        )

        tracker.record_inputs({
            'baseline_blowdown_kg_h': baseline_blowdown_kg_h,
            'optimized_blowdown_kg_h': optimized_blowdown_kg_h,
            'time_period': time_period.value
        })

        baseline = Decimal(str(baseline_blowdown_kg_h))
        optimized = Decimal(str(optimized_blowdown_kg_h))

        # Calculate instantaneous savings rate
        savings_rate = baseline - optimized

        if savings_rate < 0:
            raise ValueError(
                "Optimized blowdown rate exceeds baseline - no savings"
            )

        tracker.record_step(
            operation="instantaneous_savings",
            description="Calculate instantaneous water savings rate",
            inputs={'baseline': baseline, 'optimized': optimized},
            output_value=savings_rate,
            output_name="savings_rate_kg_h",
            formula="Savings = Baseline - Optimized",
            units="kg/h"
        )

        # Convert to time period
        time_factor = self.TIME_FACTORS[time_period]
        cumulative_savings_kg = savings_rate * time_factor

        # Convert to m3 (assuming water density = 1000 kg/m3)
        cumulative_savings_m3 = cumulative_savings_kg / Decimal('1000')

        tracker.record_step(
            operation="cumulative_savings",
            description=f"Calculate cumulative savings for {time_period.value}",
            inputs={
                'savings_rate_kg_h': savings_rate,
                'time_factor_h': time_factor
            },
            output_value=cumulative_savings_m3,
            output_name="savings_m3",
            formula="Savings_m3 = Savings_rate * Time_factor / 1000",
            units="m3"
        )

        # Calculate percentage savings
        savings_percent = Decimal('0')
        if baseline > 0:
            savings_percent = (savings_rate / baseline) * Decimal('100')

        tracker.record_step(
            operation="savings_percentage",
            description="Calculate percentage reduction",
            inputs={'savings_rate': savings_rate, 'baseline': baseline},
            output_value=savings_percent,
            output_name="savings_percent",
            formula="Percent = (Savings / Baseline) * 100",
            units="%"
        )

        # Calculate uncertainty if requested
        uncertainty_bands = None
        if include_uncertainty:
            uncertainty_bands = self._calculate_water_uncertainty(
                baseline, optimized, cumulative_savings_m3, tracker
            )

        provenance = tracker.get_provenance_record(cumulative_savings_m3)

        return SavingsResult(
            value=cumulative_savings_m3.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            unit="m3",
            uncertainty_bands=uncertainty_bands,
            time_period=time_period.value,
            baseline_value=baseline,
            optimized_value=optimized,
            savings_percent=savings_percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def calculate_energy_savings(
        self,
        baseline_energy_loss_kw: float,
        optimized_energy_loss_kw: float,
        time_period: TimePeriod = TimePeriod.HOUR,
        include_uncertainty: bool = True
    ) -> SavingsResult:
        """
        Calculate energy savings from blowdown optimization.

        Args:
            baseline_energy_loss_kw: Baseline energy loss rate (kW)
            optimized_energy_loss_kw: Optimized energy loss rate (kW)
            time_period: Time period for cumulative savings
            include_uncertainty: Include uncertainty quantification

        Returns:
            SavingsResult with energy savings in GJ
        """
        tracker = ProvenanceTracker(
            calculation_id=f"energy_savings_{time_period.value}",
            calculation_type="energy_savings",
            version=self.version
        )

        tracker.record_inputs({
            'baseline_energy_loss_kw': baseline_energy_loss_kw,
            'optimized_energy_loss_kw': optimized_energy_loss_kw,
            'time_period': time_period.value
        })

        baseline = Decimal(str(baseline_energy_loss_kw))
        optimized = Decimal(str(optimized_energy_loss_kw))

        # Calculate instantaneous savings rate
        savings_rate_kw = baseline - optimized

        if savings_rate_kw < 0:
            raise ValueError(
                "Optimized energy loss exceeds baseline - no savings"
            )

        tracker.record_step(
            operation="instantaneous_savings",
            description="Calculate instantaneous energy savings rate",
            inputs={'baseline': baseline, 'optimized': optimized},
            output_value=savings_rate_kw,
            output_name="savings_rate_kw",
            formula="Savings = Baseline - Optimized",
            units="kW"
        )

        # Convert to time period
        time_factor = self.TIME_FACTORS[time_period]
        cumulative_savings_kwh = savings_rate_kw * time_factor

        # Convert to GJ (1 kWh = 0.0036 GJ)
        cumulative_savings_gj = cumulative_savings_kwh * Decimal('0.0036')

        tracker.record_step(
            operation="cumulative_savings",
            description=f"Calculate cumulative savings for {time_period.value}",
            inputs={
                'savings_rate_kw': savings_rate_kw,
                'time_factor_h': time_factor
            },
            output_value=cumulative_savings_gj,
            output_name="savings_gj",
            formula="Savings_GJ = Savings_kW * Time_factor * 0.0036",
            units="GJ"
        )

        # Calculate percentage savings
        savings_percent = Decimal('0')
        if baseline > 0:
            savings_percent = (savings_rate_kw / baseline) * Decimal('100')

        # Calculate uncertainty if requested
        uncertainty_bands = None
        if include_uncertainty:
            uncertainty_bands = self._calculate_energy_uncertainty(
                baseline, optimized, cumulative_savings_gj, tracker
            )

        provenance = tracker.get_provenance_record(cumulative_savings_gj)

        return SavingsResult(
            value=cumulative_savings_gj.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            unit="GJ",
            uncertainty_bands=uncertainty_bands,
            time_period=time_period.value,
            baseline_value=baseline,
            optimized_value=optimized,
            savings_percent=savings_percent.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def calculate_emissions_avoided(
        self,
        energy_saved_gj: float,
        fuel_type: FuelType,
        boiler_efficiency: float = 0.85,
        time_period: TimePeriod = TimePeriod.HOUR,
        include_uncertainty: bool = True
    ) -> SavingsResult:
        """
        Calculate emissions avoided from energy savings.

        Args:
            energy_saved_gj: Energy saved (GJ)
            fuel_type: Type of fuel used in boiler
            boiler_efficiency: Boiler thermal efficiency (0-1)
            time_period: Time period for the savings
            include_uncertainty: Include uncertainty quantification

        Returns:
            SavingsResult with emissions avoided in tCO2e
        """
        tracker = ProvenanceTracker(
            calculation_id=f"emissions_avoided_{fuel_type.value}",
            calculation_type="emissions_avoided",
            version=self.version,
            standard="GHG Protocol"
        )

        tracker.record_inputs({
            'energy_saved_gj': energy_saved_gj,
            'fuel_type': fuel_type.value,
            'boiler_efficiency': boiler_efficiency,
            'time_period': time_period.value
        })

        energy = Decimal(str(energy_saved_gj))
        efficiency = Decimal(str(boiler_efficiency))

        # Calculate fuel energy saved (accounting for boiler efficiency)
        # Energy saved at load = Fuel energy * efficiency
        # So: Fuel energy saved = Energy saved at load / efficiency
        fuel_energy_saved_gj = energy / efficiency

        tracker.record_step(
            operation="fuel_energy_calculation",
            description="Calculate fuel energy saved accounting for boiler efficiency",
            inputs={
                'energy_saved_gj': energy,
                'boiler_efficiency': efficiency
            },
            output_value=fuel_energy_saved_gj,
            output_name="fuel_energy_saved_gj",
            formula="Fuel_Energy = Energy_Saved / Boiler_Efficiency",
            units="GJ"
        )

        # Get emission factor
        emission_factor = EmissionFactorDatabase.get_emission_factor(fuel_type, tracker)

        # Calculate emissions avoided (kg CO2e)
        emissions_kg = fuel_energy_saved_gj * emission_factor

        tracker.record_step(
            operation="emissions_calculation",
            description="Calculate emissions avoided",
            inputs={
                'fuel_energy_gj': fuel_energy_saved_gj,
                'emission_factor': emission_factor
            },
            output_value=emissions_kg,
            output_name="emissions_avoided_kg",
            formula="Emissions = Fuel_Energy * Emission_Factor",
            units="kg CO2e"
        )

        # Convert to tonnes
        emissions_tonnes = emissions_kg / Decimal('1000')

        tracker.record_step(
            operation="unit_conversion",
            description="Convert emissions to tonnes",
            inputs={'emissions_kg': emissions_kg},
            output_value=emissions_tonnes,
            output_name="emissions_avoided_tco2e",
            formula="Emissions_t = Emissions_kg / 1000",
            units="tCO2e"
        )

        # Calculate uncertainty if requested
        uncertainty_bands = None
        if include_uncertainty:
            uncertainty_bands = self._calculate_emissions_uncertainty(
                fuel_energy_saved_gj, emission_factor, emissions_tonnes, tracker
            )

        provenance = tracker.get_provenance_record(emissions_tonnes)

        return SavingsResult(
            value=emissions_tonnes.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            unit="tCO2e",
            uncertainty_bands=uncertainty_bands,
            time_period=time_period.value,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def calculate_cumulative_savings(
        self,
        water_savings_per_hour: SavingsResult,
        energy_savings_per_hour: SavingsResult,
        emissions_avoided_per_hour: SavingsResult,
        duration_hours: float,
        water_cost_per_m3: float = 2.0,
        fuel_cost_per_gj: float = 10.0,
        carbon_price_per_tco2e: float = 50.0
    ) -> CumulativeSavings:
        """
        Calculate cumulative savings over a duration.

        Args:
            water_savings_per_hour: Hourly water savings result
            energy_savings_per_hour: Hourly energy savings result
            emissions_avoided_per_hour: Hourly emissions avoided result
            duration_hours: Duration in hours
            water_cost_per_m3: Water cost per cubic meter
            fuel_cost_per_gj: Fuel cost per GJ
            carbon_price_per_tco2e: Carbon price per tonne CO2e

        Returns:
            CumulativeSavings with total savings and cost
        """
        tracker = ProvenanceTracker(
            calculation_id=f"cumulative_savings_{duration_hours}h",
            calculation_type="cumulative_savings",
            version=self.version
        )

        duration = Decimal(str(duration_hours))

        # Scale hourly savings to duration
        water_m3 = water_savings_per_hour.value * duration
        energy_gj = energy_savings_per_hour.value * duration
        emissions_tco2e = emissions_avoided_per_hour.value * duration

        tracker.record_step(
            operation="scale_water_savings",
            description="Scale water savings to duration",
            inputs={
                'hourly_savings': water_savings_per_hour.value,
                'duration_hours': duration
            },
            output_value=water_m3,
            output_name="total_water_m3",
            formula="Total = Hourly * Duration",
            units="m3"
        )

        tracker.record_step(
            operation="scale_energy_savings",
            description="Scale energy savings to duration",
            inputs={
                'hourly_savings': energy_savings_per_hour.value,
                'duration_hours': duration
            },
            output_value=energy_gj,
            output_name="total_energy_gj",
            formula="Total = Hourly * Duration",
            units="GJ"
        )

        tracker.record_step(
            operation="scale_emissions",
            description="Scale emissions avoided to duration",
            inputs={
                'hourly_savings': emissions_avoided_per_hour.value,
                'duration_hours': duration
            },
            output_value=emissions_tco2e,
            output_name="total_emissions_tco2e",
            formula="Total = Hourly * Duration",
            units="tCO2e"
        )

        # Calculate cost savings
        water_cost = Decimal(str(water_cost_per_m3))
        fuel_cost = Decimal(str(fuel_cost_per_gj))
        carbon_price = Decimal(str(carbon_price_per_tco2e))

        water_cost_savings = water_m3 * water_cost
        energy_cost_savings = energy_gj * fuel_cost
        carbon_cost_savings = emissions_tco2e * carbon_price
        total_cost_savings = water_cost_savings + energy_cost_savings + carbon_cost_savings

        tracker.record_step(
            operation="cost_savings",
            description="Calculate total cost savings",
            inputs={
                'water_savings_m3': water_m3,
                'energy_savings_gj': energy_gj,
                'emissions_tco2e': emissions_tco2e,
                'water_cost_per_m3': water_cost,
                'fuel_cost_per_gj': fuel_cost,
                'carbon_price_per_tco2e': carbon_price
            },
            output_value=total_cost_savings,
            output_name="total_cost_savings_usd",
            formula="Total = Water*WC + Energy*FC + Carbon*CP",
            units="USD"
        )

        # Combine uncertainty bands if available
        uncertainty_bands = None
        if water_savings_per_hour.uncertainty_bands:
            uncertainty_bands = {
                'water': UncertaintyBand(
                    lower_bound=water_savings_per_hour.uncertainty_bands.lower_bound * duration,
                    central_value=water_m3,
                    upper_bound=water_savings_per_hour.uncertainty_bands.upper_bound * duration
                ),
                'energy': UncertaintyBand(
                    lower_bound=energy_savings_per_hour.uncertainty_bands.lower_bound * duration if energy_savings_per_hour.uncertainty_bands else energy_gj,
                    central_value=energy_gj,
                    upper_bound=energy_savings_per_hour.uncertainty_bands.upper_bound * duration if energy_savings_per_hour.uncertainty_bands else energy_gj
                ),
                'emissions': UncertaintyBand(
                    lower_bound=emissions_avoided_per_hour.uncertainty_bands.lower_bound * duration if emissions_avoided_per_hour.uncertainty_bands else emissions_tco2e,
                    central_value=emissions_tco2e,
                    upper_bound=emissions_avoided_per_hour.uncertainty_bands.upper_bound * duration if emissions_avoided_per_hour.uncertainty_bands else emissions_tco2e
                )
            }

        provenance = tracker.get_provenance_record(total_cost_savings)

        return CumulativeSavings(
            water_savings_m3=water_m3.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            energy_savings_gj=energy_gj.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            emissions_avoided_tco2e=emissions_tco2e.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            cost_savings_usd=total_cost_savings.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP),
            time_period_hours=duration,
            uncertainty_bands=uncertainty_bands,
            provenance=provenance,
            calculation_hash=provenance.provenance_hash
        )

    def _calculate_water_uncertainty(
        self,
        baseline: Decimal,
        optimized: Decimal,
        savings: Decimal,
        tracker: ProvenanceTracker
    ) -> UncertaintyBand:
        """Calculate uncertainty for water savings."""
        # Use flow meter uncertainty
        u_flow = self.sensor_uncertainties['flow_meter']

        # Propagation: u_savings = sqrt(u_baseline^2 + u_optimized^2) * savings
        # Simplified: relative uncertainty = sqrt(2) * flow meter uncertainty
        u_rel = Decimal(str(math.sqrt(2))) * u_flow
        u_abs = savings * u_rel

        lower = savings - Decimal('1.96') * u_abs  # 95% CI
        upper = savings + Decimal('1.96') * u_abs

        tracker.record_step(
            operation="water_uncertainty",
            description="Calculate uncertainty for water savings",
            inputs={
                'savings': savings,
                'flow_meter_uncertainty': u_flow
            },
            output_value=u_abs,
            output_name="uncertainty_m3",
            formula="u = sqrt(2) * u_flow * savings",
            units="m3"
        )

        return UncertaintyBand(
            lower_bound=max(Decimal('0'), lower).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            central_value=savings.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            upper_bound=upper.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        )

    def _calculate_energy_uncertainty(
        self,
        baseline: Decimal,
        optimized: Decimal,
        savings: Decimal,
        tracker: ProvenanceTracker
    ) -> UncertaintyBand:
        """Calculate uncertainty for energy savings."""
        # Combined uncertainty from flow, temperature, pressure
        u_flow = self.sensor_uncertainties['flow_meter']
        u_temp = self.sensor_uncertainties['temperature']
        u_press = self.sensor_uncertainties['pressure']

        # RSS combination
        u_combined = Decimal(str(math.sqrt(
            float(u_flow ** 2 + u_temp ** 2 + u_press ** 2)
        )))

        u_rel = Decimal(str(math.sqrt(2))) * u_combined
        u_abs = savings * u_rel

        lower = savings - Decimal('1.96') * u_abs
        upper = savings + Decimal('1.96') * u_abs

        tracker.record_step(
            operation="energy_uncertainty",
            description="Calculate uncertainty for energy savings",
            inputs={
                'savings': savings,
                'combined_uncertainty': u_combined
            },
            output_value=u_abs,
            output_name="uncertainty_gj",
            formula="u = sqrt(2) * u_combined * savings",
            units="GJ"
        )

        return UncertaintyBand(
            lower_bound=max(Decimal('0'), lower).quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            central_value=savings.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP),
            upper_bound=upper.quantize(Decimal('0.001'), rounding=ROUND_HALF_UP)
        )

    def _calculate_emissions_uncertainty(
        self,
        fuel_energy: Decimal,
        emission_factor: Decimal,
        emissions: Decimal,
        tracker: ProvenanceTracker
    ) -> UncertaintyBand:
        """Calculate uncertainty for emissions avoided."""
        # Energy uncertainty (from earlier calculation)
        u_energy_rel = Decimal('0.05')  # 5% typical

        # Emission factor uncertainty (IPCC default)
        u_ef_rel = Decimal('0.05')  # 5% for well-characterized fuels

        # Combined uncertainty
        u_combined = Decimal(str(math.sqrt(
            float(u_energy_rel ** 2 + u_ef_rel ** 2)
        )))

        u_abs = emissions * u_combined

        lower = emissions - Decimal('1.96') * u_abs
        upper = emissions + Decimal('1.96') * u_abs

        tracker.record_step(
            operation="emissions_uncertainty",
            description="Calculate uncertainty for emissions avoided",
            inputs={
                'emissions': emissions,
                'energy_uncertainty': u_energy_rel,
                'ef_uncertainty': u_ef_rel
            },
            output_value=u_abs,
            output_name="uncertainty_tco2e",
            formula="u = sqrt(u_energy^2 + u_ef^2) * emissions",
            units="tCO2e"
        )

        return UncertaintyBand(
            lower_bound=max(Decimal('0'), lower).quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            central_value=emissions.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP),
            upper_bound=upper.quantize(Decimal('0.0001'), rounding=ROUND_HALF_UP)
        )


# Convenience functions

def calculate_water_savings(
    baseline_blowdown_kg_h: float,
    optimized_blowdown_kg_h: float,
    time_period: TimePeriod = TimePeriod.HOUR
) -> SavingsResult:
    """Calculate water savings - convenience function."""
    calculator = SavingsCalculator()
    return calculator.calculate_water_savings(
        baseline_blowdown_kg_h, optimized_blowdown_kg_h, time_period
    )


def calculate_energy_savings(
    baseline_energy_loss_kw: float,
    optimized_energy_loss_kw: float,
    time_period: TimePeriod = TimePeriod.HOUR
) -> SavingsResult:
    """Calculate energy savings - convenience function."""
    calculator = SavingsCalculator()
    return calculator.calculate_energy_savings(
        baseline_energy_loss_kw, optimized_energy_loss_kw, time_period
    )


def calculate_emissions_avoided(
    energy_saved_gj: float,
    fuel_type: FuelType,
    boiler_efficiency: float = 0.85
) -> SavingsResult:
    """Calculate emissions avoided - convenience function."""
    calculator = SavingsCalculator()
    return calculator.calculate_emissions_avoided(
        energy_saved_gj, fuel_type, boiler_efficiency
    )
