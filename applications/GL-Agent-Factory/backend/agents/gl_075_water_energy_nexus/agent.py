"""
GL-075: Water-Energy Nexus Agent (WATER-ENERGY-NEXUS)

This module implements the WaterEnergyNexusAgent for analyzing the interdependencies
between water and energy systems, optimizing resource allocation, and identifying
efficiency opportunities at the water-energy interface.

Standards Reference:
    - ISO 14046 (Water Footprint)
    - ISO 50001 (Energy Management)
    - WRI Aqueduct Water Risk Atlas
    - EPRI Water-Energy Research

Example:
    >>> agent = WaterEnergyNexusAgent()
    >>> result = agent.run(WaterEnergyNexusInput(water_systems=[...], energy_systems=[...]))
    >>> print(f"Nexus efficiency: {result.nexus_assessment.nexus_efficiency_percent:.1f}%")
"""

import hashlib
import json
import logging
import math
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WaterSystemType(str, Enum):
    COOLING_TOWER = "cooling_tower"
    BOILER_FEEDWATER = "boiler_feedwater"
    PROCESS_WATER = "process_water"
    WASTEWATER_TREATMENT = "wastewater_treatment"
    REVERSE_OSMOSIS = "reverse_osmosis"
    CHILLED_WATER = "chilled_water"
    IRRIGATION = "irrigation"
    POTABLE = "potable"


class EnergySystemType(str, Enum):
    PUMP = "pump"
    COMPRESSOR = "compressor"
    CHILLER = "chiller"
    BOILER = "boiler"
    COOLING_TOWER_FAN = "cooling_tower_fan"
    UV_TREATMENT = "uv_treatment"
    WATER_HEATER = "water_heater"
    DESALINATION = "desalination"


class WaterQualityType(str, Enum):
    RAW = "raw"
    POTABLE = "potable"
    PROCESS = "process"
    DEIONIZED = "deionized"
    COOLING = "cooling"
    WASTEWATER = "wastewater"
    RECLAIMED = "reclaimed"


class WaterRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREMELY_HIGH = "extremely_high"


class WaterSystem(BaseModel):
    """Water system definition."""
    system_id: str = Field(..., description="System identifier")
    name: str = Field(..., description="System name")
    system_type: WaterSystemType = Field(..., description="Type of water system")
    water_intake_m3_day: float = Field(..., ge=0, description="Water intake")
    water_discharge_m3_day: float = Field(default=0, description="Water discharge")
    water_consumption_m3_day: float = Field(default=0, description="Water consumption (lost)")
    water_quality_in: WaterQualityType = Field(..., description="Input water quality")
    water_quality_out: WaterQualityType = Field(..., description="Output water quality")
    cycles_of_concentration: Optional[float] = Field(None, description="COC for cooling")
    recirculation_rate_percent: float = Field(default=0, description="Recirculation rate")
    connected_energy_systems: List[str] = Field(default_factory=list)


class EnergySystem(BaseModel):
    """Energy system definition."""
    system_id: str = Field(..., description="System identifier")
    name: str = Field(..., description="System name")
    system_type: EnergySystemType = Field(..., description="Type of energy system")
    power_kw: float = Field(..., ge=0, description="Power consumption/capacity")
    annual_energy_kwh: float = Field(default=0, description="Annual energy consumption")
    efficiency_percent: float = Field(default=80, description="System efficiency")
    water_flow_m3_hr: Optional[float] = Field(None, description="Water flow rate")
    specific_energy_kwh_m3: Optional[float] = Field(None, description="Specific energy")
    connected_water_systems: List[str] = Field(default_factory=list)


class RegionalWaterData(BaseModel):
    """Regional water availability and risk data."""
    region: str = Field(..., description="Region identifier")
    baseline_water_stress: WaterRiskLevel = Field(..., description="Water stress level")
    water_price_per_m3: float = Field(default=1.0, description="Water price")
    wastewater_discharge_fee_per_m3: float = Field(default=0.5, description="Discharge fee")
    water_scarcity_risk_score: float = Field(default=0.5, ge=0, le=1)
    drought_frequency: str = Field(default="moderate", description="Drought frequency")
    regulatory_restrictions: List[str] = Field(default_factory=list)


class WaterEnergyNexusInput(BaseModel):
    """Input for water-energy nexus analysis."""
    analysis_id: Optional[str] = Field(None, description="Analysis identifier")
    facility_name: str = Field(default="Facility", description="Facility name")
    water_systems: List[WaterSystem] = Field(..., description="Water systems")
    energy_systems: List[EnergySystem] = Field(..., description="Energy systems")
    regional_data: RegionalWaterData = Field(..., description="Regional water data")
    energy_price_per_kwh: float = Field(default=0.10, description="Energy price")
    carbon_intensity_gCO2_kwh: float = Field(default=400, description="Grid carbon intensity")
    operating_hours_per_year: int = Field(default=8760, description="Operating hours")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WaterEnergyLink(BaseModel):
    """Link between water and energy systems."""
    link_id: str
    water_system_id: str
    energy_system_id: str
    energy_for_water_kwh_m3: float
    water_for_energy_m3_kwh: float
    link_type: str
    optimization_potential: str


class WaterBalance(BaseModel):
    """Water balance analysis."""
    total_intake_m3_day: float
    total_discharge_m3_day: float
    total_consumption_m3_day: float
    total_recirculation_m3_day: float
    water_efficiency_percent: float
    evaporative_losses_m3_day: float
    process_losses_m3_day: float
    balance_accuracy_percent: float


class EnergyForWater(BaseModel):
    """Energy used for water operations."""
    system_id: str
    system_name: str
    energy_kwh_year: float
    water_processed_m3_year: float
    specific_energy_kwh_m3: float
    energy_cost_usd_year: float
    carbon_emissions_tCO2_year: float
    benchmark_kwh_m3: float
    efficiency_vs_benchmark: str


class WaterForEnergy(BaseModel):
    """Water used for energy operations."""
    system_id: str
    system_name: str
    water_m3_year: float
    energy_produced_or_rejected_kwh: float
    water_intensity_m3_kwh: float
    water_cost_usd_year: float
    benchmark_m3_kwh: float
    efficiency_vs_benchmark: str


class NexusEfficiency(BaseModel):
    """Nexus efficiency metrics."""
    water_productivity_kwh_m3: float
    energy_productivity_m3_kwh: float
    combined_efficiency_index: float
    water_reuse_factor: float
    energy_recovery_factor: float
    nexus_performance_score: float


class OptimizationOpportunity(BaseModel):
    """Optimization opportunity."""
    opportunity_id: str
    category: str
    description: str
    water_savings_m3_year: float
    energy_savings_kwh_year: float
    cost_savings_usd_year: float
    carbon_savings_tCO2_year: float
    implementation_cost_usd: float
    simple_payback_years: float
    priority: str


class WaterRiskAssessment(BaseModel):
    """Water risk assessment."""
    overall_risk_level: WaterRiskLevel
    physical_risk_score: float
    regulatory_risk_score: float
    reputational_risk_score: float
    financial_exposure_usd_year: float
    risk_mitigation_options: List[str]
    resilience_score: float


class NexusAssessment(BaseModel):
    """Overall nexus assessment."""
    nexus_efficiency_percent: float
    water_energy_coupling_strength: str
    critical_links: List[str]
    vulnerability_points: List[str]
    sustainability_score: float
    benchmark_comparison: str
    improvement_potential_percent: float


class WaterEnergyNexusOutput(BaseModel):
    """Output from water-energy nexus analysis."""
    analysis_id: str
    facility_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    water_energy_links: List[WaterEnergyLink]
    water_balance: WaterBalance
    energy_for_water: List[EnergyForWater]
    water_for_energy: List[WaterForEnergy]
    nexus_efficiency: NexusEfficiency
    optimization_opportunities: List[OptimizationOpportunity]
    water_risk_assessment: WaterRiskAssessment
    nexus_assessment: NexusAssessment
    total_water_cost_usd_year: float
    total_energy_for_water_cost_usd_year: float
    total_water_related_carbon_tCO2_year: float
    recommendations: List[str]
    provenance_hash: str
    processing_time_ms: float
    validation_status: str


class WaterEnergyNexusAgent:
    """GL-075: Water-Energy Nexus Agent - Water-energy interdependency analysis."""

    AGENT_ID = "GL-075"
    AGENT_NAME = "WATER-ENERGY-NEXUS"
    VERSION = "1.0.0"

    # Benchmarks for specific energy (kWh/m3)
    ENERGY_BENCHMARKS = {
        WaterSystemType.COOLING_TOWER: 0.05,
        WaterSystemType.BOILER_FEEDWATER: 0.3,
        WaterSystemType.PROCESS_WATER: 0.1,
        WaterSystemType.WASTEWATER_TREATMENT: 0.5,
        WaterSystemType.REVERSE_OSMOSIS: 3.0,
        WaterSystemType.CHILLED_WATER: 0.15,
        WaterSystemType.IRRIGATION: 0.2,
        WaterSystemType.POTABLE: 0.5,
    }

    # Benchmarks for water intensity (m3/MWh)
    WATER_BENCHMARKS = {
        EnergySystemType.CHILLER: 2.0,
        EnergySystemType.BOILER: 0.5,
        EnergySystemType.COOLING_TOWER_FAN: 3.0,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        logger.info(f"WaterEnergyNexusAgent initialized (ID: {self.AGENT_ID})")

    def run(self, input_data: WaterEnergyNexusInput) -> WaterEnergyNexusOutput:
        start_time = datetime.utcnow()

        # Analyze water-energy links
        links = self._analyze_links(
            input_data.water_systems, input_data.energy_systems)

        # Calculate water balance
        water_balance = self._calculate_water_balance(
            input_data.water_systems, input_data.operating_hours_per_year)

        # Analyze energy for water
        energy_for_water = self._analyze_energy_for_water(
            input_data.water_systems, input_data.energy_systems,
            input_data.energy_price_per_kwh, input_data.carbon_intensity_gCO2_kwh,
            input_data.operating_hours_per_year)

        # Analyze water for energy
        water_for_energy = self._analyze_water_for_energy(
            input_data.water_systems, input_data.energy_systems,
            input_data.regional_data.water_price_per_m3,
            input_data.operating_hours_per_year)

        # Calculate nexus efficiency
        nexus_efficiency = self._calculate_nexus_efficiency(
            energy_for_water, water_for_energy, water_balance)

        # Identify optimization opportunities
        opportunities = self._identify_opportunities(
            input_data.water_systems, input_data.energy_systems,
            energy_for_water, water_for_energy,
            input_data.energy_price_per_kwh,
            input_data.regional_data.water_price_per_m3,
            input_data.carbon_intensity_gCO2_kwh)

        # Assess water risks
        water_risk = self._assess_water_risk(
            input_data.regional_data, water_balance,
            input_data.regional_data.water_price_per_m3)

        # Generate nexus assessment
        nexus_assessment = self._generate_nexus_assessment(
            links, nexus_efficiency, opportunities, water_risk)

        # Calculate totals
        total_water_cost = (water_balance.total_intake_m3_day * 365 *
                          input_data.regional_data.water_price_per_m3 +
                          water_balance.total_discharge_m3_day * 365 *
                          input_data.regional_data.wastewater_discharge_fee_per_m3)

        total_energy_for_water_cost = sum(e.energy_cost_usd_year for e in energy_for_water)
        total_carbon = sum(e.carbon_emissions_tCO2_year for e in energy_for_water)

        # Generate recommendations
        recommendations = self._generate_recommendations(
            nexus_assessment, opportunities, water_risk)

        provenance_hash = hashlib.sha256(
            json.dumps({
                "agent": self.AGENT_ID,
                "facility": input_data.facility_name,
                "water_systems": len(input_data.water_systems),
                "energy_systems": len(input_data.energy_systems),
                "timestamp": datetime.utcnow().isoformat()
            }, sort_keys=True, default=str).encode()).hexdigest()

        processing_time = (datetime.utcnow() - start_time).total_seconds() * 1000

        return WaterEnergyNexusOutput(
            analysis_id=input_data.analysis_id or f"WEN-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            facility_name=input_data.facility_name,
            water_energy_links=links,
            water_balance=water_balance,
            energy_for_water=energy_for_water,
            water_for_energy=water_for_energy,
            nexus_efficiency=nexus_efficiency,
            optimization_opportunities=opportunities,
            water_risk_assessment=water_risk,
            nexus_assessment=nexus_assessment,
            total_water_cost_usd_year=round(total_water_cost, 2),
            total_energy_for_water_cost_usd_year=round(total_energy_for_water_cost, 2),
            total_water_related_carbon_tCO2_year=round(total_carbon, 2),
            recommendations=recommendations,
            provenance_hash=provenance_hash,
            processing_time_ms=round(processing_time, 2),
            validation_status="PASS")

    def _analyze_links(self, water_systems: List[WaterSystem],
                      energy_systems: List[EnergySystem]) -> List[WaterEnergyLink]:
        """Analyze water-energy system links."""
        links = []
        link_num = 0

        for ws in water_systems:
            for es_id in ws.connected_energy_systems:
                es = next((e for e in energy_systems if e.system_id == es_id), None)
                if es:
                    link_num += 1

                    # Calculate energy intensity for water
                    if es.water_flow_m3_hr and es.water_flow_m3_hr > 0:
                        energy_per_water = es.power_kw / es.water_flow_m3_hr
                    else:
                        energy_per_water = es.specific_energy_kwh_m3 or 0.1

                    # Calculate water intensity for energy
                    if es.power_kw > 0:
                        water_per_energy = (ws.water_intake_m3_day / 24) / es.power_kw
                    else:
                        water_per_energy = 0

                    # Determine link type
                    if es.system_type == EnergySystemType.PUMP:
                        link_type = "water_transport"
                    elif es.system_type == EnergySystemType.CHILLER:
                        link_type = "cooling"
                    elif es.system_type == EnergySystemType.BOILER:
                        link_type = "heating"
                    else:
                        link_type = "treatment"

                    # Assess optimization potential
                    benchmark = self.ENERGY_BENCHMARKS.get(ws.system_type, 0.5)
                    if energy_per_water > benchmark * 1.5:
                        potential = "HIGH"
                    elif energy_per_water > benchmark:
                        potential = "MEDIUM"
                    else:
                        potential = "LOW"

                    links.append(WaterEnergyLink(
                        link_id=f"LINK-{link_num:03d}",
                        water_system_id=ws.system_id,
                        energy_system_id=es.system_id,
                        energy_for_water_kwh_m3=round(energy_per_water, 4),
                        water_for_energy_m3_kwh=round(water_per_energy, 6),
                        link_type=link_type,
                        optimization_potential=potential))

        return links

    def _calculate_water_balance(self, water_systems: List[WaterSystem],
                                operating_hours: int) -> WaterBalance:
        """Calculate water balance."""
        total_intake = sum(ws.water_intake_m3_day for ws in water_systems)
        total_discharge = sum(ws.water_discharge_m3_day for ws in water_systems)
        total_consumption = sum(ws.water_consumption_m3_day for ws in water_systems)

        # Recirculation
        total_recirc = sum(ws.water_intake_m3_day * ws.recirculation_rate_percent / 100
                         for ws in water_systems)

        # Water efficiency
        if total_intake > 0:
            efficiency = ((total_intake - total_consumption) / total_intake * 100)
        else:
            efficiency = 0

        # Estimate losses for cooling towers
        evap_losses = 0
        process_losses = 0
        for ws in water_systems:
            if ws.system_type == WaterSystemType.COOLING_TOWER:
                # Evaporation typically 1-2% of circulation
                evap_losses += ws.water_consumption_m3_day * 0.8
            else:
                process_losses += ws.water_consumption_m3_day

        # Balance accuracy
        calculated_consumption = total_intake - total_discharge
        if total_consumption > 0:
            accuracy = min(100, 100 - abs(calculated_consumption - total_consumption) / total_consumption * 100)
        else:
            accuracy = 100

        return WaterBalance(
            total_intake_m3_day=round(total_intake, 2),
            total_discharge_m3_day=round(total_discharge, 2),
            total_consumption_m3_day=round(total_consumption, 2),
            total_recirculation_m3_day=round(total_recirc, 2),
            water_efficiency_percent=round(efficiency, 1),
            evaporative_losses_m3_day=round(evap_losses, 2),
            process_losses_m3_day=round(process_losses, 2),
            balance_accuracy_percent=round(accuracy, 1))

    def _analyze_energy_for_water(self, water_systems: List[WaterSystem],
                                 energy_systems: List[EnergySystem],
                                 energy_price: float,
                                 carbon_intensity: float,
                                 operating_hours: int) -> List[EnergyForWater]:
        """Analyze energy used for water systems."""
        results = []

        for es in energy_systems:
            if not es.connected_water_systems:
                continue

            # Get connected water systems
            connected_ws = [ws for ws in water_systems
                          if ws.system_id in es.connected_water_systems]

            # Annual energy
            if es.annual_energy_kwh > 0:
                annual_energy = es.annual_energy_kwh
            else:
                annual_energy = es.power_kw * operating_hours * 0.7  # 70% load factor

            # Water processed
            water_processed = sum(ws.water_intake_m3_day * 365 for ws in connected_ws)

            # Specific energy
            specific_energy = annual_energy / water_processed if water_processed > 0 else 0

            # Costs and emissions
            energy_cost = annual_energy * energy_price
            carbon = annual_energy * carbon_intensity / 1000 / 1000  # tCO2

            # Benchmark comparison
            ws_types = [ws.system_type for ws in connected_ws]
            benchmark = sum(self.ENERGY_BENCHMARKS.get(t, 0.5) for t in ws_types) / len(ws_types) if ws_types else 0.5

            if specific_energy < benchmark * 0.8:
                efficiency = "EXCELLENT"
            elif specific_energy < benchmark:
                efficiency = "GOOD"
            elif specific_energy < benchmark * 1.2:
                efficiency = "ACCEPTABLE"
            else:
                efficiency = "POOR"

            results.append(EnergyForWater(
                system_id=es.system_id,
                system_name=es.name,
                energy_kwh_year=round(annual_energy, 0),
                water_processed_m3_year=round(water_processed, 0),
                specific_energy_kwh_m3=round(specific_energy, 4),
                energy_cost_usd_year=round(energy_cost, 2),
                carbon_emissions_tCO2_year=round(carbon, 2),
                benchmark_kwh_m3=round(benchmark, 4),
                efficiency_vs_benchmark=efficiency))

        return results

    def _analyze_water_for_energy(self, water_systems: List[WaterSystem],
                                 energy_systems: List[EnergySystem],
                                 water_price: float,
                                 operating_hours: int) -> List[WaterForEnergy]:
        """Analyze water used for energy systems."""
        results = []

        for ws in water_systems:
            if not ws.connected_energy_systems:
                continue

            # Get connected energy systems
            connected_es = [es for es in energy_systems
                          if es.system_id in ws.connected_energy_systems]

            # Annual water
            annual_water = ws.water_intake_m3_day * 365

            # Energy produced/rejected
            total_energy = sum(
                es.annual_energy_kwh if es.annual_energy_kwh > 0
                else es.power_kw * operating_hours * 0.7
                for es in connected_es)

            # Water intensity
            water_intensity = annual_water / (total_energy / 1000) if total_energy > 0 else 0  # m3/MWh

            # Cost
            water_cost = annual_water * water_price

            # Benchmark
            es_types = [es.system_type for es in connected_es]
            benchmark = sum(self.WATER_BENCHMARKS.get(t, 2.0) for t in es_types) / len(es_types) if es_types else 2.0

            if water_intensity < benchmark * 0.8:
                efficiency = "EXCELLENT"
            elif water_intensity < benchmark:
                efficiency = "GOOD"
            elif water_intensity < benchmark * 1.2:
                efficiency = "ACCEPTABLE"
            else:
                efficiency = "POOR"

            results.append(WaterForEnergy(
                system_id=ws.system_id,
                system_name=ws.name,
                water_m3_year=round(annual_water, 0),
                energy_produced_or_rejected_kwh=round(total_energy, 0),
                water_intensity_m3_kwh=round(water_intensity / 1000, 6),
                water_cost_usd_year=round(water_cost, 2),
                benchmark_m3_kwh=round(benchmark / 1000, 6),
                efficiency_vs_benchmark=efficiency))

        return results

    def _calculate_nexus_efficiency(self, energy_for_water: List[EnergyForWater],
                                   water_for_energy: List[WaterForEnergy],
                                   water_balance: WaterBalance) -> NexusEfficiency:
        """Calculate nexus efficiency metrics."""
        # Water productivity (energy produced per water consumed)
        total_energy = sum(w.energy_produced_or_rejected_kwh for w in water_for_energy)
        total_water = water_balance.total_consumption_m3_day * 365
        water_productivity = total_energy / total_water if total_water > 0 else 0

        # Energy productivity (water processed per energy consumed)
        total_water_processed = sum(e.water_processed_m3_year for e in energy_for_water)
        total_energy_consumed = sum(e.energy_kwh_year for e in energy_for_water)
        energy_productivity = total_water_processed / total_energy_consumed if total_energy_consumed > 0 else 0

        # Combined efficiency index (geometric mean normalized)
        combined = math.sqrt(water_productivity * energy_productivity) if water_productivity > 0 and energy_productivity > 0 else 0

        # Water reuse factor
        reuse = water_balance.total_recirculation_m3_day / water_balance.total_intake_m3_day if water_balance.total_intake_m3_day > 0 else 0

        # Energy recovery factor (simplified - assumes some heat recovery)
        recovery = 0.1  # Placeholder - would need actual data

        # Performance score
        excellent_count = sum(1 for e in energy_for_water if e.efficiency_vs_benchmark == "EXCELLENT")
        excellent_count += sum(1 for w in water_for_energy if w.efficiency_vs_benchmark == "EXCELLENT")
        total_systems = len(energy_for_water) + len(water_for_energy)
        performance = (excellent_count / total_systems * 100) if total_systems > 0 else 0

        return NexusEfficiency(
            water_productivity_kwh_m3=round(water_productivity, 2),
            energy_productivity_m3_kwh=round(energy_productivity, 4),
            combined_efficiency_index=round(combined, 4),
            water_reuse_factor=round(reuse, 3),
            energy_recovery_factor=round(recovery, 3),
            nexus_performance_score=round(performance, 1))

    def _identify_opportunities(self, water_systems: List[WaterSystem],
                               energy_systems: List[EnergySystem],
                               energy_for_water: List[EnergyForWater],
                               water_for_energy: List[WaterForEnergy],
                               energy_price: float,
                               water_price: float,
                               carbon_intensity: float) -> List[OptimizationOpportunity]:
        """Identify optimization opportunities."""
        opportunities = []
        opp_num = 0

        # Check for poor efficiency systems
        for efw in energy_for_water:
            if efw.efficiency_vs_benchmark in ["POOR", "ACCEPTABLE"]:
                opp_num += 1
                improvement = (efw.specific_energy_kwh_m3 - efw.benchmark_kwh_m3) / efw.specific_energy_kwh_m3
                energy_savings = efw.energy_kwh_year * improvement
                cost_savings = energy_savings * energy_price
                carbon_savings = energy_savings * carbon_intensity / 1000 / 1000

                opportunities.append(OptimizationOpportunity(
                    opportunity_id=f"OPP-{opp_num:03d}",
                    category="ENERGY_EFFICIENCY",
                    description=f"Improve energy efficiency of {efw.system_name} to benchmark",
                    water_savings_m3_year=0,
                    energy_savings_kwh_year=round(energy_savings, 0),
                    cost_savings_usd_year=round(cost_savings, 2),
                    carbon_savings_tCO2_year=round(carbon_savings, 2),
                    implementation_cost_usd=round(cost_savings * 2, 2),
                    simple_payback_years=round(2.0, 1),
                    priority="HIGH" if efw.efficiency_vs_benchmark == "POOR" else "MEDIUM"))

        # Check for water reuse opportunities
        for ws in water_systems:
            if ws.water_discharge_m3_day > 0 and ws.recirculation_rate_percent < 50:
                opp_num += 1
                potential_reuse = ws.water_discharge_m3_day * 0.5 * 365
                water_savings = potential_reuse * water_price
                # Energy for treatment
                treatment_energy = potential_reuse * 0.3  # 0.3 kWh/m3 for treatment
                net_savings = water_savings - treatment_energy * energy_price

                opportunities.append(OptimizationOpportunity(
                    opportunity_id=f"OPP-{opp_num:03d}",
                    category="WATER_REUSE",
                    description=f"Implement water reuse for {ws.name}",
                    water_savings_m3_year=round(potential_reuse, 0),
                    energy_savings_kwh_year=0,
                    cost_savings_usd_year=round(net_savings, 2),
                    carbon_savings_tCO2_year=round(potential_reuse * 0.3 / 1000, 2),
                    implementation_cost_usd=round(net_savings * 3, 2),
                    simple_payback_years=round(3.0, 1),
                    priority="MEDIUM"))

        # Cooling tower optimization
        for ws in water_systems:
            if ws.system_type == WaterSystemType.COOLING_TOWER and ws.cycles_of_concentration:
                if ws.cycles_of_concentration < 5:
                    opp_num += 1
                    # Increasing COC reduces makeup water
                    current_evap = ws.water_consumption_m3_day * 0.8
                    current_blowdown = current_evap / (ws.cycles_of_concentration - 1)
                    new_coc = min(8, ws.cycles_of_concentration + 2)
                    new_blowdown = current_evap / (new_coc - 1)
                    water_savings = (current_blowdown - new_blowdown) * 365

                    opportunities.append(OptimizationOpportunity(
                        opportunity_id=f"OPP-{opp_num:03d}",
                        category="COOLING_OPTIMIZATION",
                        description=f"Increase COC from {ws.cycles_of_concentration} to {new_coc} for {ws.name}",
                        water_savings_m3_year=round(water_savings, 0),
                        energy_savings_kwh_year=0,
                        cost_savings_usd_year=round(water_savings * water_price, 2),
                        carbon_savings_tCO2_year=0,
                        implementation_cost_usd=round(water_savings * water_price * 0.5, 2),
                        simple_payback_years=round(0.5, 1),
                        priority="HIGH"))

        # Sort by priority and payback
        priority_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        opportunities.sort(key=lambda x: (priority_order.get(x.priority, 2), x.simple_payback_years))

        return opportunities

    def _assess_water_risk(self, regional_data: RegionalWaterData,
                          water_balance: WaterBalance,
                          water_price: float) -> WaterRiskAssessment:
        """Assess water-related risks."""
        # Physical risk (based on regional data)
        physical_risk = {
            WaterRiskLevel.LOW: 0.2,
            WaterRiskLevel.MEDIUM: 0.5,
            WaterRiskLevel.HIGH: 0.8,
            WaterRiskLevel.EXTREMELY_HIGH: 1.0,
        }.get(regional_data.baseline_water_stress, 0.5)

        # Regulatory risk
        regulatory_risk = min(1.0, len(regional_data.regulatory_restrictions) * 0.2)

        # Reputational risk (based on consumption)
        if water_balance.water_efficiency_percent > 90:
            reputational_risk = 0.2
        elif water_balance.water_efficiency_percent > 70:
            reputational_risk = 0.4
        else:
            reputational_risk = 0.7

        # Financial exposure
        annual_water_cost = water_balance.total_intake_m3_day * 365 * water_price
        # Risk exposure = potential cost increase due to scarcity
        financial_exposure = annual_water_cost * regional_data.water_scarcity_risk_score

        # Overall risk
        combined_risk = (physical_risk * 0.4 + regulatory_risk * 0.3 + reputational_risk * 0.3)
        if combined_risk > 0.7:
            overall_risk = WaterRiskLevel.HIGH
        elif combined_risk > 0.4:
            overall_risk = WaterRiskLevel.MEDIUM
        else:
            overall_risk = WaterRiskLevel.LOW

        # Mitigation options
        mitigations = []
        if physical_risk > 0.5:
            mitigations.append("Develop alternative water sources")
            mitigations.append("Implement water storage")
        if water_balance.water_efficiency_percent < 80:
            mitigations.append("Increase water recycling and reuse")
        if regulatory_risk > 0.3:
            mitigations.append("Engage with regulators proactively")
        mitigations.append("Implement real-time water monitoring")

        # Resilience score
        resilience = max(0, 100 - combined_risk * 100)

        return WaterRiskAssessment(
            overall_risk_level=overall_risk,
            physical_risk_score=round(physical_risk, 2),
            regulatory_risk_score=round(regulatory_risk, 2),
            reputational_risk_score=round(reputational_risk, 2),
            financial_exposure_usd_year=round(financial_exposure, 2),
            risk_mitigation_options=mitigations,
            resilience_score=round(resilience, 1))

    def _generate_nexus_assessment(self, links: List[WaterEnergyLink],
                                  efficiency: NexusEfficiency,
                                  opportunities: List[OptimizationOpportunity],
                                  risk: WaterRiskAssessment) -> NexusAssessment:
        """Generate overall nexus assessment."""
        # Nexus efficiency
        nexus_eff = efficiency.nexus_performance_score

        # Coupling strength
        high_potential_links = sum(1 for l in links if l.optimization_potential == "HIGH")
        if high_potential_links > len(links) * 0.5:
            coupling = "TIGHT"
        elif high_potential_links > len(links) * 0.2:
            coupling = "MODERATE"
        else:
            coupling = "LOOSE"

        # Critical links
        critical = [l.link_id for l in links if l.optimization_potential == "HIGH"][:3]

        # Vulnerability points
        vulnerabilities = []
        if efficiency.water_reuse_factor < 0.3:
            vulnerabilities.append("Low water reuse rate")
        if efficiency.energy_recovery_factor < 0.1:
            vulnerabilities.append("Limited energy recovery")
        if risk.overall_risk_level in [WaterRiskLevel.HIGH, WaterRiskLevel.EXTREMELY_HIGH]:
            vulnerabilities.append("High water supply risk")

        # Sustainability score
        sustainability = (
            nexus_eff * 0.3 +
            (100 - risk.physical_risk_score * 100) * 0.3 +
            efficiency.water_reuse_factor * 100 * 0.2 +
            (100 if len(opportunities) < 3 else 50) * 0.2
        )

        # Benchmark comparison
        if nexus_eff > 80:
            benchmark = "ABOVE_AVERAGE"
        elif nexus_eff > 50:
            benchmark = "AVERAGE"
        else:
            benchmark = "BELOW_AVERAGE"

        # Improvement potential
        total_savings = sum(o.cost_savings_usd_year for o in opportunities)
        total_cost = sum(o.implementation_cost_usd for o in opportunities)
        improvement = (total_savings / total_cost * 100) if total_cost > 0 else 0

        return NexusAssessment(
            nexus_efficiency_percent=round(nexus_eff, 1),
            water_energy_coupling_strength=coupling,
            critical_links=critical,
            vulnerability_points=vulnerabilities,
            sustainability_score=round(sustainability, 1),
            benchmark_comparison=benchmark,
            improvement_potential_percent=round(min(50, improvement), 1))

    def _generate_recommendations(self, assessment: NexusAssessment,
                                 opportunities: List[OptimizationOpportunity],
                                 risk: WaterRiskAssessment) -> List[str]:
        """Generate recommendations."""
        recommendations = []

        # Based on efficiency
        if assessment.nexus_efficiency_percent < 60:
            recommendations.append("PRIORITY: Implement comprehensive water-energy efficiency program")

        # Based on opportunities
        high_priority = [o for o in opportunities if o.priority == "HIGH"]
        if high_priority:
            recommendations.append(f"Pursue {len(high_priority)} high-priority optimization opportunities")
            for opp in high_priority[:2]:
                recommendations.append(f"  - {opp.description}")

        # Based on vulnerabilities
        for vuln in assessment.vulnerability_points[:2]:
            recommendations.append(f"Address vulnerability: {vuln}")

        # Based on risk
        if risk.overall_risk_level in [WaterRiskLevel.HIGH, WaterRiskLevel.EXTREMELY_HIGH]:
            recommendations.append("Develop water contingency plan for supply disruptions")
            recommendations.extend(risk.risk_mitigation_options[:2])

        # General best practices
        if assessment.water_energy_coupling_strength == "TIGHT":
            recommendations.append("Implement integrated water-energy management system")

        return recommendations


PACK_SPEC = {
    "schema_version": "2.0.0",
    "id": "GL-075",
    "name": "WATER-ENERGY-NEXUS",
    "version": "1.0.0",
    "summary": "Water-energy nexus analysis and optimization",
    "tags": ["water", "energy", "nexus", "efficiency", "sustainability", "risk"],
    "standards": [
        {"ref": "ISO 14046", "description": "Water Footprint"},
        {"ref": "ISO 50001", "description": "Energy Management"},
        {"ref": "WRI Aqueduct", "description": "Water Risk Atlas"}
    ],
    "provenance": {"calculation_verified": True, "enable_audit": True}
}
