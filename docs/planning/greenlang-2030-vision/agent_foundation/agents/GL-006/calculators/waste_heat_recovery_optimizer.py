# -*- coding: utf-8 -*-
"""
GL-006 HEATRECLAIM - Waste Heat Recovery Optimizer
===================================================

Advanced calculator for waste heat recovery optimization including:
- Waste heat source characterization
- ORC/Kalina cycle feasibility analysis
- Heat pump integration analysis
- Thermal storage sizing
- Payback period calculation
- Carbon credit potential
- Cascading heat use optimization
- Heat quality (exergy) analysis

Standards: ISO 50001, ASME PTC 4, EPA CHP Guidelines

Zero-Hallucination Guarantee: All calculations use deterministic
engineering formulas with complete provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any

# ==============================================================================
# CONSTANTS
# ==============================================================================

# Reference temperature for exergy calculations (K)
T0_REFERENCE_K = 298.15  # 25°C

# Carnot efficiency reference
ABSOLUTE_ZERO_OFFSET = 273.15

# Heat pump COP correlation coefficients (empirical)
# COP = a - b * (T_lift / T_source)
HEAT_PUMP_COP_COEFFICIENTS = {
    "ammonia": {"a": 8.5, "b": 0.12},
    "r134a": {"a": 7.8, "b": 0.11},
    "co2": {"a": 6.5, "b": 0.10},
    "propane": {"a": 7.5, "b": 0.11},
}

# ORC working fluid properties
ORC_FLUIDS = {
    "r245fa": {"t_crit_c": 154.0, "efficiency_factor": 0.75, "min_temp_c": 80},
    "r134a": {"t_crit_c": 101.1, "efficiency_factor": 0.70, "min_temp_c": 60},
    "pentane": {"t_crit_c": 196.6, "efficiency_factor": 0.78, "min_temp_c": 100},
    "toluene": {"t_crit_c": 318.6, "efficiency_factor": 0.80, "min_temp_c": 150},
    "siloxane": {"t_crit_c": 245.5, "efficiency_factor": 0.76, "min_temp_c": 120},
}

# Kalina cycle efficiency factors by temperature range
KALINA_EFFICIENCY = {
    "low": {"min_c": 100, "max_c": 150, "eta": 0.12},
    "medium": {"min_c": 150, "max_c": 250, "eta": 0.18},
    "high": {"min_c": 250, "max_c": 400, "eta": 0.24},
}

# Carbon emission factors (kg CO2/kWh)
CARBON_EMISSION_FACTORS = {
    "natural_gas": Decimal("0.185"),
    "coal": Decimal("0.340"),
    "fuel_oil": Decimal("0.260"),
    "electricity_grid_avg": Decimal("0.400"),
    "biomass": Decimal("0.020"),  # Biogenic only
}

# Thermal storage media properties
THERMAL_STORAGE_MEDIA = {
    "water": {"cp_kj_kg_k": 4.18, "density_kg_m3": 1000, "max_temp_c": 95},
    "pressurized_water": {"cp_kj_kg_k": 4.5, "density_kg_m3": 850, "max_temp_c": 200},
    "thermal_oil": {"cp_kj_kg_k": 2.1, "density_kg_m3": 850, "max_temp_c": 350},
    "molten_salt": {"cp_kj_kg_k": 1.5, "density_kg_m3": 1800, "max_temp_c": 565},
    "concrete": {"cp_kj_kg_k": 0.88, "density_kg_m3": 2400, "max_temp_c": 400},
}


# ==============================================================================
# ENUMERATIONS
# ==============================================================================

class WasteHeatQuality(Enum):
    """Classification of waste heat quality by temperature."""
    HIGH = "high"           # > 400°C
    MEDIUM_HIGH = "medium_high"  # 200-400°C
    MEDIUM = "medium"       # 100-200°C
    LOW = "low"            # 50-100°C
    VERY_LOW = "very_low"   # < 50°C


class RecoveryTechnology(Enum):
    """Waste heat recovery technologies."""
    HEAT_EXCHANGER = "heat_exchanger"
    ORC = "organic_rankine_cycle"
    KALINA = "kalina_cycle"
    HEAT_PUMP = "heat_pump"
    ABSORPTION_CHILLER = "absorption_chiller"
    THERMOELECTRIC = "thermoelectric"
    THERMAL_STORAGE = "thermal_storage"


class CascadeLevel(Enum):
    """Cascade heat use priority levels."""
    POWER_GENERATION = 1
    PROCESS_HEATING = 2
    SPACE_HEATING = 3
    PREHEATING = 4
    COOLING = 5
    REJECT = 6


# ==============================================================================
# DATA CLASSES
# ==============================================================================

@dataclass(frozen=True)
class WasteHeatSource:
    """Characterization of a waste heat source."""
    source_id: str
    name: str
    temperature_c: float
    flow_rate_kg_s: float
    specific_heat_kj_kg_k: float
    min_return_temp_c: float
    availability_hours_year: float
    medium: str = "flue_gas"  # flue_gas, steam, water, oil
    contaminants: Optional[str] = None


@dataclass(frozen=True)
class WasteHeatCharacterization:
    """Complete waste heat source characterization."""
    source: WasteHeatSource
    quality: WasteHeatQuality
    heat_rate_kw: float
    exergy_rate_kw: float
    exergy_efficiency_pct: float
    carnot_efficiency_pct: float
    available_energy_mwh_year: float
    provenance_hash: str


@dataclass(frozen=True)
class ORCFeasibilityResult:
    """ORC cycle feasibility analysis results."""
    is_feasible: bool
    recommended_fluid: Optional[str]
    thermal_efficiency_pct: float
    electrical_output_kw: float
    annual_generation_mwh: float
    estimated_cost_usd: Decimal
    simple_payback_years: float
    npv_usd: Decimal
    irr_pct: float
    reasons: List[str]
    provenance_hash: str


@dataclass(frozen=True)
class KalinaFeasibilityResult:
    """Kalina cycle feasibility analysis results."""
    is_feasible: bool
    cycle_type: str
    thermal_efficiency_pct: float
    electrical_output_kw: float
    annual_generation_mwh: float
    ammonia_concentration_pct: float
    estimated_cost_usd: Decimal
    simple_payback_years: float
    reasons: List[str]
    provenance_hash: str


@dataclass(frozen=True)
class HeatPumpAnalysisResult:
    """Heat pump integration analysis results."""
    is_feasible: bool
    cop_heating: float
    cop_cooling: float
    temperature_lift_c: float
    heat_output_kw: float
    electrical_input_kw: float
    annual_savings_usd: Decimal
    carbon_savings_tonnes_year: float
    recommended_refrigerant: str
    simple_payback_years: float
    provenance_hash: str


@dataclass(frozen=True)
class ThermalStorageResult:
    """Thermal storage sizing results."""
    recommended_medium: str
    storage_capacity_kwh: float
    storage_volume_m3: float
    storage_mass_tonnes: float
    charge_rate_kw: float
    discharge_rate_kw: float
    round_trip_efficiency_pct: float
    estimated_cost_usd: Decimal
    annual_benefit_usd: Decimal
    simple_payback_years: float
    provenance_hash: str


@dataclass(frozen=True)
class CarbonCreditResult:
    """Carbon credit potential calculation."""
    annual_co2_avoided_tonnes: float
    carbon_credit_value_usd: Decimal
    carbon_price_usd_per_tonne: Decimal
    baseline_fuel: str
    calculation_methodology: str
    provenance_hash: str


@dataclass(frozen=True)
class CascadeOptimizationResult:
    """Cascading heat use optimization results."""
    cascade_levels: List[Dict[str, Any]]
    total_heat_recovered_kw: float
    recovery_efficiency_pct: float
    annual_savings_usd: Decimal
    carbon_savings_tonnes: float
    unrecovered_heat_kw: float
    provenance_hash: str


@dataclass(frozen=True)
class ExergyAnalysisResult:
    """Exergy (availability) analysis results."""
    exergy_input_kw: float
    exergy_output_kw: float
    exergy_destruction_kw: float
    exergy_efficiency_pct: float
    quality_factor: float
    second_law_efficiency_pct: float
    improvement_potential_kw: float
    provenance_hash: str


# ==============================================================================
# PROVENANCE TRACKER
# ==============================================================================

class ProvenanceTracker:
    """Thread-safe provenance tracking for audit trails."""

    def __init__(self):
        self._lock = threading.RLock()
        self._steps: List[Dict[str, Any]] = []
        self._timestamp = datetime.utcnow().isoformat()

    def add_step(self, step_name: str, formula: str, inputs: Dict, output: Any):
        """Record a calculation step."""
        with self._lock:
            self._steps.append({
                "step": step_name,
                "formula": formula,
                "inputs": inputs,
                "output": output,
                "timestamp": datetime.utcnow().isoformat()
            })

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of all calculation steps."""
        with self._lock:
            content = json.dumps(self._steps, sort_keys=True, default=str)
            return hashlib.sha256(content.encode()).hexdigest()

    def get_steps(self) -> List[Dict[str, Any]]:
        """Return copy of calculation steps."""
        with self._lock:
            return list(self._steps)


# ==============================================================================
# WASTE HEAT RECOVERY OPTIMIZER
# ==============================================================================

class WasteHeatRecoveryOptimizer:
    """
    Advanced waste heat recovery optimizer.

    Provides comprehensive analysis of waste heat recovery opportunities
    with zero-hallucination guarantee through deterministic calculations.
    """

    def __init__(
        self,
        reference_temp_k: float = T0_REFERENCE_K,
        electricity_price_usd_kwh: Decimal = Decimal("0.10"),
        natural_gas_price_usd_mmbtu: Decimal = Decimal("4.00"),
        carbon_price_usd_tonne: Decimal = Decimal("50.00"),
        discount_rate: float = 0.08,
        project_lifetime_years: int = 20
    ):
        self.reference_temp_k = reference_temp_k
        self.electricity_price = electricity_price_usd_kwh
        self.gas_price = natural_gas_price_usd_mmbtu
        self.carbon_price = carbon_price_usd_tonne
        self.discount_rate = discount_rate
        self.project_lifetime = project_lifetime_years
        self._cache_lock = threading.RLock()
        self._cache: Dict[str, Any] = {}

    def characterize_waste_heat(
        self,
        source: WasteHeatSource
    ) -> WasteHeatCharacterization:
        """
        Fully characterize a waste heat source.

        Calculates heat rate, exergy rate, quality classification,
        and annual energy availability.
        """
        tracker = ProvenanceTracker()

        # Calculate heat rate: Q = m * cp * (T_source - T_return)
        delta_t = source.temperature_c - source.min_return_temp_c
        heat_rate_kw = source.flow_rate_kg_s * source.specific_heat_kj_kg_k * delta_t

        tracker.add_step(
            "heat_rate_calculation",
            "Q = m_dot * cp * (T_hot - T_cold)",
            {
                "m_dot_kg_s": source.flow_rate_kg_s,
                "cp_kj_kg_k": source.specific_heat_kj_kg_k,
                "delta_T_C": delta_t
            },
            heat_rate_kw
        )

        # Calculate Carnot efficiency
        t_hot_k = source.temperature_c + ABSOLUTE_ZERO_OFFSET
        t_cold_k = source.min_return_temp_c + ABSOLUTE_ZERO_OFFSET
        carnot_eff = 1.0 - (t_cold_k / t_hot_k)

        tracker.add_step(
            "carnot_efficiency",
            "eta_carnot = 1 - T_cold/T_hot",
            {"T_hot_K": t_hot_k, "T_cold_K": t_cold_k},
            carnot_eff
        )

        # Calculate exergy rate: Ex = Q * (1 - T0/Tm)
        t_mean_k = (t_hot_k + t_cold_k) / 2.0
        exergy_rate_kw = heat_rate_kw * (1.0 - self.reference_temp_k / t_mean_k)

        tracker.add_step(
            "exergy_rate",
            "Ex = Q * (1 - T0/Tm)",
            {
                "Q_kW": heat_rate_kw,
                "T0_K": self.reference_temp_k,
                "Tm_K": t_mean_k
            },
            exergy_rate_kw
        )

        # Exergy efficiency
        exergy_eff = (exergy_rate_kw / heat_rate_kw * 100.0) if heat_rate_kw > 0 else 0.0

        # Classify quality
        quality = self._classify_heat_quality(source.temperature_c)

        # Annual energy availability
        available_energy_mwh = heat_rate_kw * source.availability_hours_year / 1000.0

        tracker.add_step(
            "annual_availability",
            "E_annual = Q * hours / 1000",
            {
                "Q_kW": heat_rate_kw,
                "hours_year": source.availability_hours_year
            },
            available_energy_mwh
        )

        return WasteHeatCharacterization(
            source=source,
            quality=quality,
            heat_rate_kw=round(heat_rate_kw, 2),
            exergy_rate_kw=round(exergy_rate_kw, 2),
            exergy_efficiency_pct=round(exergy_eff, 2),
            carnot_efficiency_pct=round(carnot_eff * 100, 2),
            available_energy_mwh_year=round(available_energy_mwh, 2),
            provenance_hash=tracker.compute_hash()
        )

    def analyze_orc_feasibility(
        self,
        characterization: WasteHeatCharacterization,
        orc_cost_usd_per_kw: Decimal = Decimal("2500")
    ) -> ORCFeasibilityResult:
        """
        Analyze ORC cycle feasibility for waste heat recovery.

        Evaluates multiple working fluids and selects optimal configuration.
        """
        tracker = ProvenanceTracker()
        reasons = []

        source_temp_c = characterization.source.temperature_c
        heat_rate_kw = characterization.heat_rate_kw

        # Find suitable ORC fluids
        suitable_fluids = []
        for fluid_name, props in ORC_FLUIDS.items():
            if source_temp_c >= props["min_temp_c"]:
                # Calculate ORC efficiency using simplified correlation
                # eta_ORC = eta_factor * eta_carnot * (1 - sqrt(T_cond/T_evap))
                t_evap_k = (source_temp_c - 20) + ABSOLUTE_ZERO_OFFSET
                t_cond_k = 35 + ABSOLUTE_ZERO_OFFSET  # Typical condensing temp

                eta_carnot = 1.0 - t_cond_k / t_evap_k
                eta_orc = props["efficiency_factor"] * eta_carnot

                suitable_fluids.append({
                    "fluid": fluid_name,
                    "efficiency": eta_orc,
                    "t_crit_c": props["t_crit_c"]
                })

        if not suitable_fluids:
            return ORCFeasibilityResult(
                is_feasible=False,
                recommended_fluid=None,
                thermal_efficiency_pct=0.0,
                electrical_output_kw=0.0,
                annual_generation_mwh=0.0,
                estimated_cost_usd=Decimal("0"),
                simple_payback_years=float("inf"),
                npv_usd=Decimal("0"),
                irr_pct=0.0,
                reasons=["Source temperature too low for ORC"],
                provenance_hash=tracker.compute_hash()
            )

        # Select best fluid by efficiency
        best_fluid = max(suitable_fluids, key=lambda x: x["efficiency"])
        eta_orc = best_fluid["efficiency"]

        tracker.add_step(
            "orc_efficiency",
            "eta_ORC = eta_factor * eta_carnot",
            {"fluid": best_fluid["fluid"], "eta_carnot": eta_orc / ORC_FLUIDS[best_fluid["fluid"]]["efficiency_factor"]},
            eta_orc
        )

        # Calculate electrical output
        electrical_output_kw = heat_rate_kw * eta_orc
        annual_generation_mwh = electrical_output_kw * characterization.source.availability_hours_year / 1000.0

        tracker.add_step(
            "electrical_output",
            "P_elec = Q * eta_ORC",
            {"Q_kW": heat_rate_kw, "eta_ORC": eta_orc},
            electrical_output_kw
        )

        # Economic analysis
        capital_cost = orc_cost_usd_per_kw * Decimal(str(electrical_output_kw))
        annual_revenue = self.electricity_price * Decimal(str(annual_generation_mwh * 1000))

        simple_payback = float(capital_cost / annual_revenue) if annual_revenue > 0 else float("inf")

        # NPV calculation
        npv = self._calculate_npv(capital_cost, annual_revenue, self.discount_rate, self.project_lifetime)

        # IRR calculation (simplified Newton-Raphson)
        irr = self._calculate_irr(capital_cost, annual_revenue, self.project_lifetime)

        tracker.add_step(
            "economics",
            "Payback = CAPEX / Annual_Revenue",
            {
                "CAPEX_USD": float(capital_cost),
                "Annual_Revenue_USD": float(annual_revenue)
            },
            simple_payback
        )

        is_feasible = (
            electrical_output_kw >= 50 and  # Minimum viable size
            simple_payback <= 7.0 and
            source_temp_c >= 80
        )

        if electrical_output_kw < 50:
            reasons.append("Electrical output below minimum viable size (50 kW)")
        if simple_payback > 7.0:
            reasons.append(f"Simple payback ({simple_payback:.1f} years) exceeds 7 year threshold")
        if is_feasible:
            reasons.append("ORC is technically and economically viable")

        return ORCFeasibilityResult(
            is_feasible=is_feasible,
            recommended_fluid=best_fluid["fluid"],
            thermal_efficiency_pct=round(eta_orc * 100, 2),
            electrical_output_kw=round(electrical_output_kw, 2),
            annual_generation_mwh=round(annual_generation_mwh, 2),
            estimated_cost_usd=capital_cost.quantize(Decimal("0.01")),
            simple_payback_years=round(simple_payback, 2),
            npv_usd=npv.quantize(Decimal("0.01")),
            irr_pct=round(irr * 100, 2),
            reasons=reasons,
            provenance_hash=tracker.compute_hash()
        )

    def analyze_kalina_feasibility(
        self,
        characterization: WasteHeatCharacterization,
        kalina_cost_usd_per_kw: Decimal = Decimal("3500")
    ) -> KalinaFeasibilityResult:
        """
        Analyze Kalina cycle feasibility for waste heat recovery.

        Kalina cycle uses ammonia-water mixture for improved efficiency
        at moderate temperature sources.
        """
        tracker = ProvenanceTracker()
        reasons = []

        source_temp_c = characterization.source.temperature_c
        heat_rate_kw = characterization.heat_rate_kw

        # Determine Kalina cycle type by temperature
        cycle_type = None
        eta_kalina = 0.0
        ammonia_conc = 0.0

        for level, params in KALINA_EFFICIENCY.items():
            if params["min_c"] <= source_temp_c < params["max_c"]:
                cycle_type = f"KCS-{level}"
                eta_kalina = params["eta"]
                # Ammonia concentration increases with temperature
                ammonia_conc = 70 + (source_temp_c - params["min_c"]) * 0.1
                break

        if cycle_type is None:
            if source_temp_c < 100:
                reasons.append("Source temperature too low for Kalina cycle (<100°C)")
            else:
                reasons.append("Source temperature too high for standard Kalina cycle (>400°C)")

            return KalinaFeasibilityResult(
                is_feasible=False,
                cycle_type="N/A",
                thermal_efficiency_pct=0.0,
                electrical_output_kw=0.0,
                annual_generation_mwh=0.0,
                ammonia_concentration_pct=0.0,
                estimated_cost_usd=Decimal("0"),
                simple_payback_years=float("inf"),
                reasons=reasons,
                provenance_hash=tracker.compute_hash()
            )

        tracker.add_step(
            "kalina_efficiency",
            "eta_Kalina based on temperature range",
            {"source_temp_C": source_temp_c, "cycle_type": cycle_type},
            eta_kalina
        )

        # Calculate outputs
        electrical_output_kw = heat_rate_kw * eta_kalina
        annual_generation_mwh = electrical_output_kw * characterization.source.availability_hours_year / 1000.0

        # Economics
        capital_cost = kalina_cost_usd_per_kw * Decimal(str(electrical_output_kw))
        annual_revenue = self.electricity_price * Decimal(str(annual_generation_mwh * 1000))
        simple_payback = float(capital_cost / annual_revenue) if annual_revenue > 0 else float("inf")

        is_feasible = (
            electrical_output_kw >= 100 and
            simple_payback <= 8.0
        )

        if is_feasible:
            reasons.append(f"Kalina {cycle_type} is viable for this application")
        else:
            if electrical_output_kw < 100:
                reasons.append("Output below minimum Kalina viable size (100 kW)")

        return KalinaFeasibilityResult(
            is_feasible=is_feasible,
            cycle_type=cycle_type,
            thermal_efficiency_pct=round(eta_kalina * 100, 2),
            electrical_output_kw=round(electrical_output_kw, 2),
            annual_generation_mwh=round(annual_generation_mwh, 2),
            ammonia_concentration_pct=round(ammonia_conc, 1),
            estimated_cost_usd=capital_cost.quantize(Decimal("0.01")),
            simple_payback_years=round(simple_payback, 2),
            reasons=reasons,
            provenance_hash=tracker.compute_hash()
        )

    def analyze_heat_pump_integration(
        self,
        characterization: WasteHeatCharacterization,
        target_temp_c: float,
        heat_demand_kw: float,
        refrigerant: str = "ammonia"
    ) -> HeatPumpAnalysisResult:
        """
        Analyze heat pump integration for waste heat upgrading.

        Calculates COP, economic benefits, and carbon savings.
        """
        tracker = ProvenanceTracker()

        source_temp_c = characterization.source.temperature_c
        temperature_lift = target_temp_c - source_temp_c

        # Get COP coefficients
        coeff = HEAT_PUMP_COP_COEFFICIENTS.get(refrigerant, HEAT_PUMP_COP_COEFFICIENTS["ammonia"])

        # Calculate COP: COP = a - b * (T_lift / T_source_K)
        t_source_k = source_temp_c + ABSOLUTE_ZERO_OFFSET
        cop_heating = coeff["a"] - coeff["b"] * (temperature_lift / t_source_k) * 100
        cop_heating = max(cop_heating, 1.5)  # Minimum practical COP

        # Cooling COP (EER)
        cop_cooling = cop_heating - 1.0

        tracker.add_step(
            "cop_calculation",
            "COP = a - b * (T_lift / T_source)",
            {
                "a": coeff["a"],
                "b": coeff["b"],
                "T_lift_C": temperature_lift,
                "T_source_K": t_source_k
            },
            cop_heating
        )

        # Size heat pump to meet demand or available heat
        available_heat_kw = characterization.heat_rate_kw
        heat_output_kw = min(heat_demand_kw, available_heat_kw * cop_heating / (cop_heating - 1))
        electrical_input_kw = heat_output_kw / cop_heating

        tracker.add_step(
            "heat_pump_sizing",
            "P_elec = Q_heat / COP",
            {"Q_heat_kW": heat_output_kw, "COP": cop_heating},
            electrical_input_kw
        )

        # Annual savings vs gas boiler
        hours_per_year = characterization.source.availability_hours_year
        heat_supplied_mwh = heat_output_kw * hours_per_year / 1000.0

        # Gas boiler alternative (90% efficiency)
        gas_consumption_mmbtu = heat_supplied_mwh * 3.412 / 0.90
        gas_cost = self.gas_price * Decimal(str(gas_consumption_mmbtu))

        # Heat pump electricity cost
        elec_consumption_mwh = electrical_input_kw * hours_per_year / 1000.0
        elec_cost = self.electricity_price * Decimal(str(elec_consumption_mwh * 1000))

        annual_savings = gas_cost - elec_cost

        # Carbon savings
        gas_emissions = float(CARBON_EMISSION_FACTORS["natural_gas"]) * heat_supplied_mwh * 1000 / 0.90
        elec_emissions = float(CARBON_EMISSION_FACTORS["electricity_grid_avg"]) * elec_consumption_mwh * 1000
        carbon_savings = (gas_emissions - elec_emissions) / 1000.0  # tonnes

        # Economics
        hp_cost_per_kw = Decimal("800")  # Typical industrial heat pump
        capital_cost = hp_cost_per_kw * Decimal(str(heat_output_kw))
        simple_payback = float(capital_cost / annual_savings) if annual_savings > 0 else float("inf")

        is_feasible = (
            temperature_lift <= 80 and  # Practical heat pump limit
            cop_heating >= 2.5 and
            simple_payback <= 5.0
        )

        return HeatPumpAnalysisResult(
            is_feasible=is_feasible,
            cop_heating=round(cop_heating, 2),
            cop_cooling=round(cop_cooling, 2),
            temperature_lift_c=round(temperature_lift, 1),
            heat_output_kw=round(heat_output_kw, 2),
            electrical_input_kw=round(electrical_input_kw, 2),
            annual_savings_usd=annual_savings.quantize(Decimal("0.01")),
            carbon_savings_tonnes_year=round(carbon_savings, 2),
            recommended_refrigerant=refrigerant,
            simple_payback_years=round(simple_payback, 2),
            provenance_hash=tracker.compute_hash()
        )

    def size_thermal_storage(
        self,
        characterization: WasteHeatCharacterization,
        storage_hours: float,
        delta_t_c: float = 30.0
    ) -> ThermalStorageResult:
        """
        Size thermal energy storage system.

        Calculates optimal storage medium, volume, and economics.
        """
        tracker = ProvenanceTracker()

        heat_rate_kw = characterization.heat_rate_kw
        storage_capacity_kwh = heat_rate_kw * storage_hours

        # Select storage medium based on temperature
        source_temp_c = characterization.source.temperature_c
        recommended_medium = self._select_storage_medium(source_temp_c)

        media_props = THERMAL_STORAGE_MEDIA[recommended_medium]

        tracker.add_step(
            "storage_capacity",
            "E_storage = Q * hours",
            {"Q_kW": heat_rate_kw, "hours": storage_hours},
            storage_capacity_kwh
        )

        # Calculate volume: V = E / (rho * cp * delta_T)
        # E in kJ = kWh * 3600
        energy_kj = storage_capacity_kwh * 3600
        storage_mass_kg = energy_kj / (media_props["cp_kj_kg_k"] * delta_t_c)
        storage_volume_m3 = storage_mass_kg / media_props["density_kg_m3"]

        tracker.add_step(
            "storage_volume",
            "V = E / (rho * cp * delta_T)",
            {
                "E_kJ": energy_kj,
                "rho_kg_m3": media_props["density_kg_m3"],
                "cp_kJ_kg_K": media_props["cp_kj_kg_k"],
                "delta_T_C": delta_t_c
            },
            storage_volume_m3
        )

        # Round-trip efficiency (typical values)
        round_trip_eff = 0.90 if recommended_medium in ["water", "pressurized_water"] else 0.85

        # Economics
        storage_cost_per_kwh = Decimal("50")  # Typical for sensible heat storage
        capital_cost = storage_cost_per_kwh * Decimal(str(storage_capacity_kwh))

        # Value of stored energy (peak vs off-peak arbitrage)
        peak_premium = Decimal("0.05")  # $/kWh premium during peak
        daily_cycles = 1.0
        annual_benefit = peak_premium * Decimal(str(storage_capacity_kwh * daily_cycles * 365 * round_trip_eff))

        simple_payback = float(capital_cost / annual_benefit) if annual_benefit > 0 else float("inf")

        return ThermalStorageResult(
            recommended_medium=recommended_medium,
            storage_capacity_kwh=round(storage_capacity_kwh, 2),
            storage_volume_m3=round(storage_volume_m3, 2),
            storage_mass_tonnes=round(storage_mass_kg / 1000, 2),
            charge_rate_kw=round(heat_rate_kw, 2),
            discharge_rate_kw=round(heat_rate_kw * round_trip_eff, 2),
            round_trip_efficiency_pct=round(round_trip_eff * 100, 1),
            estimated_cost_usd=capital_cost.quantize(Decimal("0.01")),
            annual_benefit_usd=annual_benefit.quantize(Decimal("0.01")),
            simple_payback_years=round(simple_payback, 2),
            provenance_hash=tracker.compute_hash()
        )

    def calculate_carbon_credits(
        self,
        characterization: WasteHeatCharacterization,
        recovered_heat_kw: float,
        baseline_fuel: str = "natural_gas"
    ) -> CarbonCreditResult:
        """
        Calculate carbon credit potential from waste heat recovery.

        Uses emissions avoided methodology comparing to baseline fuel.
        """
        tracker = ProvenanceTracker()

        # Annual heat recovered
        hours_per_year = characterization.source.availability_hours_year
        heat_recovered_mwh = recovered_heat_kw * hours_per_year / 1000.0

        # Baseline emissions (fuel that would otherwise be burned)
        emission_factor = CARBON_EMISSION_FACTORS.get(baseline_fuel, CARBON_EMISSION_FACTORS["natural_gas"])

        # Assume 85% boiler efficiency for baseline
        baseline_efficiency = Decimal("0.85")
        fuel_energy_mwh = Decimal(str(heat_recovered_mwh)) / baseline_efficiency

        co2_avoided_tonnes = float(fuel_energy_mwh * emission_factor)

        tracker.add_step(
            "carbon_calculation",
            "CO2_avoided = E_recovered / eta_boiler * EF",
            {
                "E_recovered_MWh": heat_recovered_mwh,
                "eta_boiler": float(baseline_efficiency),
                "EF_kg_CO2_kWh": float(emission_factor)
            },
            co2_avoided_tonnes
        )

        # Carbon credit value
        credit_value = self.carbon_price * Decimal(str(co2_avoided_tonnes))

        return CarbonCreditResult(
            annual_co2_avoided_tonnes=round(co2_avoided_tonnes, 2),
            carbon_credit_value_usd=credit_value.quantize(Decimal("0.01")),
            carbon_price_usd_per_tonne=self.carbon_price,
            baseline_fuel=baseline_fuel,
            calculation_methodology="Emissions Avoided - Fuel Substitution",
            provenance_hash=tracker.compute_hash()
        )

    def optimize_cascade_heat_use(
        self,
        characterization: WasteHeatCharacterization,
        demands: List[Dict[str, Any]]
    ) -> CascadeOptimizationResult:
        """
        Optimize cascading heat use across multiple temperature levels.

        Implements exergy-efficient cascade from high to low temperature uses.

        Args:
            characterization: Waste heat source characterization
            demands: List of heat demands with format:
                [{"level": CascadeLevel, "temp_required_c": float, "demand_kw": float}]
        """
        tracker = ProvenanceTracker()

        available_heat = characterization.heat_rate_kw
        available_temp = characterization.source.temperature_c

        # Sort demands by temperature (highest first) for optimal cascade
        sorted_demands = sorted(demands, key=lambda x: x["temp_required_c"], reverse=True)

        cascade_levels = []
        total_recovered = 0.0
        current_temp = available_temp
        remaining_heat = available_heat

        for demand in sorted_demands:
            required_temp = demand["temp_required_c"]
            required_heat = demand["demand_kw"]

            # Check if current temperature can meet demand (with 10°C approach)
            if current_temp >= required_temp + 10:
                # Calculate how much heat can be extracted
                # Heat extracted drops temperature: delta_T = Q / (m * cp)
                extractable = min(required_heat, remaining_heat)

                # Temperature drop
                delta_t = (extractable / available_heat) * (available_temp - characterization.source.min_return_temp_c)
                new_temp = current_temp - delta_t

                cascade_levels.append({
                    "level": demand.get("level", CascadeLevel.PROCESS_HEATING).value if isinstance(demand.get("level"), CascadeLevel) else demand.get("level", 2),
                    "name": demand.get("name", "Process Heat"),
                    "temp_in_c": round(current_temp, 1),
                    "temp_out_c": round(new_temp, 1),
                    "heat_supplied_kw": round(extractable, 2),
                    "demand_met_pct": round(extractable / required_heat * 100, 1)
                })

                total_recovered += extractable
                remaining_heat -= extractable
                current_temp = new_temp

                tracker.add_step(
                    f"cascade_level_{len(cascade_levels)}",
                    "Q_extracted, T_new = f(demand, available)",
                    {
                        "T_current": current_temp + delta_t,
                        "required_temp": required_temp,
                        "required_heat": required_heat
                    },
                    {"extracted": extractable, "new_temp": new_temp}
                )

        recovery_efficiency = (total_recovered / available_heat * 100) if available_heat > 0 else 0.0

        # Calculate annual savings
        hours_per_year = characterization.source.availability_hours_year
        heat_recovered_mwh = total_recovered * hours_per_year / 1000.0
        annual_savings = self.gas_price * Decimal(str(heat_recovered_mwh * 3.412 / 0.85))

        # Carbon savings
        carbon_savings = float(CARBON_EMISSION_FACTORS["natural_gas"]) * heat_recovered_mwh / 0.85

        return CascadeOptimizationResult(
            cascade_levels=cascade_levels,
            total_heat_recovered_kw=round(total_recovered, 2),
            recovery_efficiency_pct=round(recovery_efficiency, 2),
            annual_savings_usd=annual_savings.quantize(Decimal("0.01")),
            carbon_savings_tonnes=round(carbon_savings, 2),
            unrecovered_heat_kw=round(remaining_heat, 2),
            provenance_hash=tracker.compute_hash()
        )

    def analyze_exergy(
        self,
        characterization: WasteHeatCharacterization,
        recovery_efficiency: float = 0.80
    ) -> ExergyAnalysisResult:
        """
        Perform exergy (second law) analysis of waste heat recovery.

        Calculates exergy destruction and improvement potential.
        """
        tracker = ProvenanceTracker()

        exergy_input = characterization.exergy_rate_kw

        # Recovered exergy (accounting for recovery efficiency)
        exergy_output = exergy_input * recovery_efficiency

        # Exergy destruction
        exergy_destruction = exergy_input - exergy_output

        # Second law efficiency
        second_law_eff = (exergy_output / exergy_input * 100) if exergy_input > 0 else 0.0

        tracker.add_step(
            "exergy_balance",
            "Ex_destruction = Ex_in - Ex_out",
            {"Ex_in_kW": exergy_input, "Ex_out_kW": exergy_output},
            exergy_destruction
        )

        # Quality factor (exergy/energy ratio)
        quality_factor = (exergy_input / characterization.heat_rate_kw) if characterization.heat_rate_kw > 0 else 0.0

        # Improvement potential (theoretical max - actual)
        max_exergy_output = exergy_input * 0.95  # 95% theoretical max
        improvement_potential = max_exergy_output - exergy_output

        tracker.add_step(
            "improvement_potential",
            "Improvement = Ex_max - Ex_actual",
            {"Ex_max": max_exergy_output, "Ex_actual": exergy_output},
            improvement_potential
        )

        return ExergyAnalysisResult(
            exergy_input_kw=round(exergy_input, 2),
            exergy_output_kw=round(exergy_output, 2),
            exergy_destruction_kw=round(exergy_destruction, 2),
            exergy_efficiency_pct=round(recovery_efficiency * 100, 2),
            quality_factor=round(quality_factor, 3),
            second_law_efficiency_pct=round(second_law_eff, 2),
            improvement_potential_kw=round(improvement_potential, 2),
            provenance_hash=tracker.compute_hash()
        )

    # ==========================================================================
    # HELPER METHODS
    # ==========================================================================

    def _classify_heat_quality(self, temperature_c: float) -> WasteHeatQuality:
        """Classify waste heat quality by temperature."""
        if temperature_c > 400:
            return WasteHeatQuality.HIGH
        elif temperature_c > 200:
            return WasteHeatQuality.MEDIUM_HIGH
        elif temperature_c > 100:
            return WasteHeatQuality.MEDIUM
        elif temperature_c > 50:
            return WasteHeatQuality.LOW
        else:
            return WasteHeatQuality.VERY_LOW

    def _select_storage_medium(self, temperature_c: float) -> str:
        """Select appropriate thermal storage medium."""
        for medium, props in THERMAL_STORAGE_MEDIA.items():
            if temperature_c <= props["max_temp_c"]:
                return medium
        return "molten_salt"  # Highest temperature option

    def _calculate_npv(
        self,
        capital_cost: Decimal,
        annual_benefit: Decimal,
        discount_rate: float,
        years: int
    ) -> Decimal:
        """Calculate Net Present Value."""
        npv = -capital_cost
        for year in range(1, years + 1):
            npv += annual_benefit / Decimal(str((1 + discount_rate) ** year))
        return npv

    def _calculate_irr(
        self,
        capital_cost: Decimal,
        annual_benefit: Decimal,
        years: int,
        tolerance: float = 0.0001,
        max_iterations: int = 100
    ) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson."""
        if annual_benefit <= 0:
            return 0.0

        # Initial guess
        irr = float(annual_benefit / capital_cost)

        for _ in range(max_iterations):
            npv = float(-capital_cost)
            d_npv = 0.0

            for year in range(1, years + 1):
                discount = (1 + irr) ** year
                npv += float(annual_benefit) / discount
                d_npv -= year * float(annual_benefit) / (discount * (1 + irr))

            if abs(d_npv) < 1e-10:
                break

            new_irr = irr - npv / d_npv

            if abs(new_irr - irr) < tolerance:
                return new_irr

            irr = new_irr

        return irr


# ==============================================================================
# MODULE EXPORTS
# ==============================================================================

__all__ = [
    # Main class
    "WasteHeatRecoveryOptimizer",
    # Data classes
    "WasteHeatSource",
    "WasteHeatCharacterization",
    "ORCFeasibilityResult",
    "KalinaFeasibilityResult",
    "HeatPumpAnalysisResult",
    "ThermalStorageResult",
    "CarbonCreditResult",
    "CascadeOptimizationResult",
    "ExergyAnalysisResult",
    # Enums
    "WasteHeatQuality",
    "RecoveryTechnology",
    "CascadeLevel",
    # Constants
    "ORC_FLUIDS",
    "KALINA_EFFICIENCY",
    "THERMAL_STORAGE_MEDIA",
    "CARBON_EMISSION_FACTORS",
]
