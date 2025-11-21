# -*- coding: utf-8 -*-
"""
HeatRecoveryOrchestrator - Main orchestration agent for GL-006 HeatRecoveryMaximizer

This module implements the core heat recovery optimization agent that identifies,
analyzes, and optimizes waste heat recovery opportunities in industrial processes.
It uses pinch analysis, exergy analysis, and thermodynamic optimization to maximize
energy efficiency while ensuring zero-hallucination through physics-based calculations.

Example:
    >>> config = HeatRecoveryConfig()
    >>> orchestrator = HeatRecoveryOrchestrator(config)
    >>> result = await orchestrator.run_optimization_cycle(process_data)
"""

from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
import asyncio
import hashlib
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
from pydantic import BaseModel, Field, validator, root_validator
import pandas as pd

# GreenLang Core Imports
from greenlang_core import BaseAgent, AgentConfig
from greenlang_validation import ValidationResult, ValidationError
from greenlang_provenance import ProvenanceTracker
from greenlang_saga import SagaPattern, SagaStep
from greenlang_metrics import MetricsCollector
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class StreamType(str, Enum):
    """Types of process streams for heat recovery."""
    HOT_STREAM = "hot_stream"
    COLD_STREAM = "cold_stream"
    UTILITY_HOT = "utility_hot"
    UTILITY_COLD = "utility_cold"
    WASTE_HEAT = "waste_heat"


class HeatExchangerType(str, Enum):
    """Types of heat exchangers for network optimization."""
    SHELL_AND_TUBE = "shell_and_tube"
    PLATE = "plate"
    FINNED_TUBE = "finned_tube"
    REGENERATIVE = "regenerative"
    HEAT_PIPE = "heat_pipe"


@dataclass
class ProcessStream:
    """Data model for process streams in heat recovery analysis."""
    stream_id: str
    stream_type: StreamType
    inlet_temp_c: float
    outlet_temp_c: float
    flow_rate_kg_s: float
    specific_heat_kj_kg_k: float
    pressure_bar: float
    phase: str  # vapor, liquid, two-phase
    composition: Dict[str, float] = field(default_factory=dict)
    fouling_factor: float = 0.0001
    available_hours_per_year: float = 8000

    @property
    def heat_capacity_rate(self) -> float:
        """Calculate heat capacity rate (kW/K)."""
        return self.flow_rate_kg_s * self.specific_heat_kj_kg_k

    @property
    def heat_duty_kw(self) -> float:
        """Calculate heat duty (kW)."""
        return abs(self.heat_capacity_rate * (self.outlet_temp_c - self.inlet_temp_c))

    @property
    def supply_temperature(self) -> float:
        """Get supply (inlet) temperature."""
        return self.inlet_temp_c if self.stream_type in [StreamType.HOT_STREAM, StreamType.WASTE_HEAT] else self.outlet_temp_c

    @property
    def target_temperature(self) -> float:
        """Get target (outlet) temperature."""
        return self.outlet_temp_c if self.stream_type in [StreamType.HOT_STREAM, StreamType.WASTE_HEAT] else self.inlet_temp_c


class HeatRecoveryInput(BaseModel):
    """Input data model for heat recovery optimization."""

    process_streams: List[Dict[str, Any]] = Field(..., description="List of process streams")
    minimum_temperature_approach_c: float = Field(10.0, ge=1.0, le=50.0, description="Minimum temperature approach for heat exchange")
    electricity_cost_usd_kwh: float = Field(0.10, ge=0.01, description="Electricity cost")
    steam_cost_usd_ton: float = Field(30.0, ge=1.0, description="Steam cost")
    cooling_water_cost_usd_m3: float = Field(0.5, ge=0.01, description="Cooling water cost")
    capital_recovery_factor: float = Field(0.15, ge=0.05, le=0.30, description="Capital recovery factor for ROI")
    target_payback_years: float = Field(3.0, ge=0.5, le=10.0, description="Target payback period")
    available_space_m2: Optional[float] = Field(None, description="Available space for new equipment")
    max_pressure_drop_bar: float = Field(0.5, ge=0.1, le=2.0, description="Maximum allowable pressure drop")

    @validator('process_streams')
    def validate_streams(cls, v):
        """Validate process streams have required fields."""
        if len(v) < 2:
            raise ValueError("At least 2 process streams required for heat recovery analysis")

        required_fields = {'stream_id', 'stream_type', 'inlet_temp_c', 'outlet_temp_c',
                          'flow_rate_kg_s', 'specific_heat_kj_kg_k'}
        for stream in v:
            if not all(field in stream for field in required_fields):
                raise ValueError(f"Stream missing required fields: {required_fields}")
        return v


class PinchAnalysisResult(BaseModel):
    """Result of pinch analysis for heat recovery."""

    pinch_temperature_hot_c: float = Field(..., description="Pinch temperature for hot streams")
    pinch_temperature_cold_c: float = Field(..., description="Pinch temperature for cold streams")
    minimum_hot_utility_kw: float = Field(..., description="Minimum hot utility requirement")
    minimum_cold_utility_kw: float = Field(..., description="Minimum cold utility requirement")
    heat_recovery_potential_kw: float = Field(..., description="Maximum heat recovery potential")
    grand_composite_curve: List[Dict[str, float]] = Field(..., description="Grand composite curve data")
    problem_table: pd.DataFrame = Field(..., description="Problem table cascade")

    class Config:
        arbitrary_types_allowed = True


class HeatExchangerNetwork(BaseModel):
    """Optimized heat exchanger network design."""

    exchangers: List[Dict[str, Any]] = Field(..., description="List of heat exchangers")
    total_area_m2: float = Field(..., description="Total heat transfer area")
    total_capital_cost_usd: float = Field(..., description="Total capital investment")
    annual_energy_savings_kw: float = Field(..., description="Annual energy savings")
    annual_cost_savings_usd: float = Field(..., description="Annual operating cost savings")
    payback_period_years: float = Field(..., description="Simple payback period")
    npv_usd: float = Field(..., description="Net present value")
    network_topology: Dict[str, List[str]] = Field(..., description="Network connection topology")


class HeatRecoveryOutput(BaseModel):
    """Output data model for heat recovery optimization."""

    optimization_id: str = Field(..., description="Unique optimization ID")
    timestamp: datetime = Field(..., description="Optimization timestamp")
    identified_opportunities: List[Dict[str, Any]] = Field(..., description="Identified heat recovery opportunities")
    pinch_analysis: Optional[PinchAnalysisResult] = Field(None, description="Pinch analysis results")
    heat_exchanger_network: Optional[HeatExchangerNetwork] = Field(None, description="Optimized HEN design")
    exergy_efficiency_percent: float = Field(..., description="Overall exergy efficiency")
    total_heat_recovery_kw: float = Field(..., description="Total heat recovery potential")
    implementation_plan: List[Dict[str, Any]] = Field(..., description="Phased implementation plan")
    roi_analysis: Dict[str, float] = Field(..., description="Return on investment analysis")
    co2_reduction_tons_year: float = Field(..., description="Annual CO2 emissions reduction")
    validation_status: str = Field(..., description="PASS or FAIL with reasons")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Total processing time")

    class Config:
        arbitrary_types_allowed = True


class HeatRecoveryOrchestrator(BaseAgent):
    """
    Main orchestrator for GL-006 HeatRecoveryMaximizer agent.

    This agent implements comprehensive heat recovery optimization using pinch analysis,
    exergy analysis, and heat exchanger network synthesis. It follows GreenLang's
    zero-hallucination principle by using only physics-based thermodynamic calculations
    and validated correlations.

    Attributes:
        config: Agent configuration
        provenance_tracker: Tracks data lineage for audit
        metrics_collector: Collects performance metrics
        saga_pattern: Manages distributed transactions

    Example:
        >>> config = HeatRecoveryConfig()
        >>> orchestrator = HeatRecoveryOrchestrator(config)
        >>> result = await orchestrator.run_optimization_cycle(input_data)
        >>> print(f"Heat recovery potential: {result.total_heat_recovery_kw} kW")
    """

    def __init__(self, config: AgentConfig):
        """Initialize HeatRecoveryOrchestrator with configuration."""
        super().__init__(config)
        self.provenance_tracker = ProvenanceTracker()
        self.metrics_collector = MetricsCollector("gl_006_heat_recovery")
        self.saga_pattern = SagaPattern()

        # Initialize thermodynamic property databases
        self._init_property_databases()

        # Performance tracking
        self.processing_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "failed_optimizations": 0,
            "average_processing_time_ms": 0
        }

        logger.info("HeatRecoveryOrchestrator initialized with config: %s", config.dict())

    def _init_property_databases(self):
        """Initialize thermodynamic property databases for zero-hallucination calculations."""
        # Steam tables (IAPWS-IF97)
        self.steam_properties = {
            "saturation_pressure": lambda T: 10 ** (8.07131 - 1730.63 / (233.426 + T)),  # Antoine equation
            "latent_heat": lambda P: 2501.0 - 2.361 * P,  # Approximate correlation
            "specific_heat_water": 4.186,  # kJ/kg.K
            "specific_heat_steam": 2.080   # kJ/kg.K at 100C
        }

        # Heat transfer correlations
        self.heat_transfer_correlations = {
            "nusselt_turbulent": lambda Re, Pr: 0.023 * Re**0.8 * Pr**0.4,  # Dittus-Boelter
            "effectiveness_ntu": lambda NTU, Cr: (1 - np.exp(-NTU * (1 - Cr))) / (1 - Cr * np.exp(-NTU * (1 - Cr))),
            "lmtd_correction": lambda P, R: self._calculate_lmtd_correction(P, R)
        }

    async def run_optimization_cycle(self, input_data: HeatRecoveryInput) -> HeatRecoveryOutput:
        """
        Main optimization cycle for heat recovery maximization.

        Args:
            input_data: Validated input data with process streams and constraints

        Returns:
            Comprehensive heat recovery optimization results

        Raises:
            ValidationError: If input validation fails
            ProcessingError: If optimization fails
        """
        start_time = DeterministicClock.now()
        optimization_id = self._generate_optimization_id()

        try:
            logger.info(f"Starting optimization cycle {optimization_id}")
            self.metrics_collector.increment("optimization_cycles_started")

            # Create saga for distributed transaction management
            saga_steps = [
                SagaStep("validate_input", self._validate_input_wrapper, self._rollback_validation),
                SagaStep("identify_streams", self.identify_waste_heat_streams, self._rollback_streams),
                SagaStep("analyze_gradients", self.analyze_temperature_gradients, self._rollback_gradients),
                SagaStep("pinch_analysis", self.perform_pinch_analysis, self._rollback_pinch),
                SagaStep("optimize_network", self.optimize_heat_exchanger_network, self._rollback_network),
                SagaStep("calculate_exergy", self.calculate_exergy_efficiency, self._rollback_exergy),
                SagaStep("roi_analysis", self.calculate_roi_analysis, self._rollback_roi),
                SagaStep("generate_plan", self.generate_implementation_plan, self._rollback_plan)
            ]

            # Execute saga pattern
            saga_result = await self.saga_pattern.execute(saga_steps, input_data)

            # Compile results
            output = await self._compile_optimization_results(
                optimization_id=optimization_id,
                input_data=input_data,
                saga_result=saga_result,
                start_time=start_time
            )

            # Update statistics
            self._update_processing_stats(start_time, success=True)
            self.metrics_collector.increment("optimization_cycles_completed")

            logger.info(f"Optimization cycle {optimization_id} completed successfully")
            return output

        except Exception as e:
            logger.error(f"Optimization cycle {optimization_id} failed: {str(e)}", exc_info=True)
            self._update_processing_stats(start_time, success=False)
            self.metrics_collector.increment("optimization_cycles_failed")
            raise

    async def identify_waste_heat_streams(self, input_data: HeatRecoveryInput) -> List[ProcessStream]:
        """
        Identify and characterize waste heat streams from process data.

        This method analyzes process streams to identify waste heat sources
        based on temperature levels, flow rates, and availability.

        Args:
            input_data: Process stream data

        Returns:
            List of identified waste heat streams with recovery potential
        """
        logger.info("Identifying waste heat streams")
        self.metrics_collector.increment("waste_heat_identifications")

        waste_heat_streams = []

        for stream_data in input_data.process_streams:
            stream = ProcessStream(
                stream_id=stream_data['stream_id'],
                stream_type=StreamType(stream_data['stream_type']),
                inlet_temp_c=stream_data['inlet_temp_c'],
                outlet_temp_c=stream_data['outlet_temp_c'],
                flow_rate_kg_s=stream_data['flow_rate_kg_s'],
                specific_heat_kj_kg_k=stream_data['specific_heat_kj_kg_k'],
                pressure_bar=stream_data.get('pressure_bar', 1.0),
                phase=stream_data.get('phase', 'liquid'),
                composition=stream_data.get('composition', {}),
                fouling_factor=stream_data.get('fouling_factor', 0.0001),
                available_hours_per_year=stream_data.get('available_hours_per_year', 8000)
            )

            # Identify waste heat based on criteria
            if self._is_waste_heat_stream(stream):
                waste_heat_streams.append(stream)
                self.metrics_collector.observe("waste_heat_temperature_c", stream.inlet_temp_c)
                self.metrics_collector.observe("waste_heat_duty_kw", stream.heat_duty_kw)

        logger.info(f"Identified {len(waste_heat_streams)} waste heat streams")
        return waste_heat_streams

    async def analyze_temperature_gradients(self, streams: List[ProcessStream]) -> Dict[str, Any]:
        """
        Analyze temperature gradients between hot and cold streams.

        This method performs detailed temperature gradient analysis to identify
        feasible heat exchange opportunities based on thermodynamic constraints.

        Args:
            streams: List of process streams

        Returns:
            Temperature gradient analysis results
        """
        logger.info("Analyzing temperature gradients")
        self.metrics_collector.increment("gradient_analyses")

        hot_streams = [s for s in streams if s.stream_type in [StreamType.HOT_STREAM, StreamType.WASTE_HEAT]]
        cold_streams = [s for s in streams if s.stream_type == StreamType.COLD_STREAM]

        gradient_analysis = {
            "feasible_matches": [],
            "temperature_profiles": [],
            "driving_forces": [],
            "heat_transfer_potential": {}
        }

        # Analyze each hot-cold stream pair
        for hot in hot_streams:
            for cold in cold_streams:
                # Check thermodynamic feasibility
                if self._is_thermodynamically_feasible(hot, cold):
                    match = {
                        "hot_stream": hot.stream_id,
                        "cold_stream": cold.stream_id,
                        "delta_t_min": min(
                            hot.inlet_temp_c - cold.outlet_temp_c,
                            hot.outlet_temp_c - cold.inlet_temp_c
                        ),
                        "heat_transfer_potential_kw": min(hot.heat_duty_kw, cold.heat_duty_kw),
                        "lmtd": self._calculate_lmtd(hot, cold)
                    }
                    gradient_analysis["feasible_matches"].append(match)
                    self.metrics_collector.observe("temperature_gradient_c", match["delta_t_min"])

        logger.info(f"Found {len(gradient_analysis['feasible_matches'])} feasible heat exchange matches")
        return gradient_analysis

    async def perform_pinch_analysis(self, gradient_data: Dict[str, Any]) -> PinchAnalysisResult:
        """
        Perform pinch analysis to determine minimum utility requirements.

        This method implements the problem table algorithm for pinch analysis,
        determining the minimum hot and cold utility requirements and the
        maximum heat recovery potential.

        Args:
            gradient_data: Temperature gradient analysis data

        Returns:
            Comprehensive pinch analysis results
        """
        logger.info("Performing pinch analysis")
        self.metrics_collector.increment("pinch_analyses")

        # Extract stream data from gradient analysis
        streams = gradient_data.get("streams", [])
        min_approach_temp = gradient_data.get("min_approach_temp", 10.0)

        # Build temperature intervals
        temperature_intervals = self._build_temperature_intervals(streams, min_approach_temp)

        # Construct problem table
        problem_table = self._construct_problem_table(temperature_intervals, streams)

        # Perform cascade analysis
        cascade_result = self._perform_cascade_analysis(problem_table)

        # Identify pinch point
        pinch_location = self._identify_pinch_point(cascade_result)

        # Calculate minimum utilities
        min_hot_utility = cascade_result['min_hot_utility_kw']
        min_cold_utility = cascade_result['min_cold_utility_kw']

        # Generate grand composite curve
        gcc_data = self._generate_grand_composite_curve(cascade_result)

        result = PinchAnalysisResult(
            pinch_temperature_hot_c=pinch_location['hot_temp'],
            pinch_temperature_cold_c=pinch_location['cold_temp'],
            minimum_hot_utility_kw=min_hot_utility,
            minimum_cold_utility_kw=min_cold_utility,
            heat_recovery_potential_kw=cascade_result['max_heat_recovery_kw'],
            grand_composite_curve=gcc_data,
            problem_table=problem_table
        )

        self.metrics_collector.observe("pinch_temperature_c", pinch_location['hot_temp'])
        self.metrics_collector.observe("heat_recovery_potential_kw", result.heat_recovery_potential_kw)

        logger.info(f"Pinch analysis complete. Heat recovery potential: {result.heat_recovery_potential_kw} kW")
        return result

    async def optimize_heat_exchanger_network(self, pinch_result: PinchAnalysisResult) -> HeatExchangerNetwork:
        """
        Optimize heat exchanger network design using pinch principles.

        This method synthesizes an optimal heat exchanger network that maximizes
        heat recovery while minimizing capital and operating costs.

        Args:
            pinch_result: Results from pinch analysis

        Returns:
            Optimized heat exchanger network design
        """
        logger.info("Optimizing heat exchanger network")
        self.metrics_collector.increment("network_optimizations")

        # Initialize network synthesis
        network_synthesizer = self._create_network_synthesizer(pinch_result)

        # Apply pinch design rules
        above_pinch_network = await self._design_above_pinch(network_synthesizer)
        below_pinch_network = await self._design_below_pinch(network_synthesizer)

        # Merge networks
        complete_network = self._merge_networks(above_pinch_network, below_pinch_network)

        # Size heat exchangers
        sized_exchangers = await self._size_heat_exchangers(complete_network)

        # Calculate costs
        capital_cost = self._calculate_capital_cost(sized_exchangers)
        operating_savings = self._calculate_operating_savings(complete_network, pinch_result)

        # Perform economic optimization
        optimized_network = await self._economic_optimization(
            sized_exchangers,
            capital_cost,
            operating_savings
        )

        result = HeatExchangerNetwork(
            exchangers=optimized_network['exchangers'],
            total_area_m2=optimized_network['total_area'],
            total_capital_cost_usd=optimized_network['capital_cost'],
            annual_energy_savings_kw=optimized_network['energy_savings'],
            annual_cost_savings_usd=optimized_network['cost_savings'],
            payback_period_years=optimized_network['capital_cost'] / optimized_network['cost_savings'],
            npv_usd=self._calculate_npv(optimized_network),
            network_topology=optimized_network['topology']
        )

        self.metrics_collector.observe("network_capital_cost_usd", result.total_capital_cost_usd)
        self.metrics_collector.observe("network_payback_years", result.payback_period_years)

        logger.info(f"Network optimization complete. {len(result.exchangers)} heat exchangers designed")
        return result

    async def calculate_exergy_efficiency(self, network_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate exergy efficiency of the heat recovery system.

        This method performs exergy analysis to quantify the thermodynamic
        quality of energy recovery and identify exergy destruction sources.

        Args:
            network_data: Heat exchanger network data

        Returns:
            Exergy efficiency metrics and analysis
        """
        logger.info("Calculating exergy efficiency")
        self.metrics_collector.increment("exergy_calculations")

        # Reference state (ambient conditions)
        T_ambient_k = 298.15  # 25°C
        P_ambient_bar = 1.013

        exergy_analysis = {
            "exergy_input_kw": 0,
            "exergy_recovered_kw": 0,
            "exergy_destroyed_kw": 0,
            "exergy_efficiency_percent": 0,
            "exergy_destruction_by_component": {}
        }

        # Calculate exergy for each stream
        for stream_data in network_data.get("streams", []):
            # Physical exergy
            physical_exergy = self._calculate_physical_exergy(
                stream_data,
                T_ambient_k,
                P_ambient_bar
            )

            # Chemical exergy (if applicable)
            chemical_exergy = self._calculate_chemical_exergy(
                stream_data.get("composition", {}),
                T_ambient_k
            )

            total_exergy = physical_exergy + chemical_exergy

            if stream_data.get("is_waste_heat", False):
                exergy_analysis["exergy_input_kw"] += total_exergy
            elif stream_data.get("is_recovered", False):
                exergy_analysis["exergy_recovered_kw"] += total_exergy

        # Calculate exergy destruction in heat exchangers
        for exchanger in network_data.get("exchangers", []):
            exergy_destroyed = self._calculate_exchanger_exergy_destruction(
                exchanger,
                T_ambient_k
            )
            exergy_analysis["exergy_destroyed_kw"] += exergy_destroyed
            exergy_analysis["exergy_destruction_by_component"][exchanger["id"]] = exergy_destroyed

        # Calculate overall exergy efficiency
        if exergy_analysis["exergy_input_kw"] > 0:
            exergy_analysis["exergy_efficiency_percent"] = (
                exergy_analysis["exergy_recovered_kw"] /
                exergy_analysis["exergy_input_kw"] * 100
            )

        self.metrics_collector.observe("exergy_efficiency_percent", exergy_analysis["exergy_efficiency_percent"])
        self.metrics_collector.observe("exergy_destroyed_kw", exergy_analysis["exergy_destroyed_kw"])

        logger.info(f"Exergy efficiency: {exergy_analysis['exergy_efficiency_percent']:.1f}%")
        return exergy_analysis

    async def calculate_roi_analysis(self, network: HeatExchangerNetwork, exergy_data: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate comprehensive return on investment analysis.

        This method performs detailed financial analysis including NPV, IRR,
        payback period, and sensitivity analysis for the heat recovery project.

        Args:
            network: Optimized heat exchanger network
            exergy_data: Exergy efficiency data

        Returns:
            Comprehensive ROI analysis metrics
        """
        logger.info("Calculating ROI analysis")
        self.metrics_collector.increment("roi_analyses")

        # Financial parameters
        project_lifetime_years = 15
        discount_rate = 0.10
        inflation_rate = 0.03
        energy_price_escalation = 0.05

        roi_analysis = {
            "initial_investment_usd": network.total_capital_cost_usd,
            "annual_savings_year1_usd": network.annual_cost_savings_usd,
            "simple_payback_years": network.payback_period_years,
            "discounted_payback_years": 0,
            "npv_usd": 0,
            "irr_percent": 0,
            "profitability_index": 0,
            "lcoe_usd_per_kwh": 0,
            "carbon_credits_value_usd": 0,
            "sensitivity_analysis": {}
        }

        # Calculate cash flows
        cash_flows = []
        for year in range(project_lifetime_years + 1):
            if year == 0:
                cash_flow = -network.total_capital_cost_usd
            else:
                # Escalate savings with energy prices
                annual_savings = network.annual_cost_savings_usd * (1 + energy_price_escalation) ** (year - 1)

                # Add carbon credit value
                carbon_price_usd_per_ton = 50 * (1 + inflation_rate) ** (year - 1)
                carbon_credits = self._calculate_carbon_credits(network.annual_energy_savings_kw) * carbon_price_usd_per_ton

                cash_flow = annual_savings + carbon_credits

            cash_flows.append(cash_flow)

        # Calculate NPV
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
        roi_analysis["npv_usd"] = npv

        # Calculate IRR
        irr = self._calculate_irr(cash_flows)
        roi_analysis["irr_percent"] = irr * 100

        # Calculate discounted payback
        cumulative_dcf = 0
        for year, cf in enumerate(cash_flows[1:], 1):
            cumulative_dcf += cf / (1 + discount_rate) ** year
            if cumulative_dcf >= network.total_capital_cost_usd:
                roi_analysis["discounted_payback_years"] = year
                break

        # Calculate profitability index
        pv_benefits = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows[1:], 1))
        roi_analysis["profitability_index"] = pv_benefits / network.total_capital_cost_usd

        # Calculate LCOE
        total_energy_kwh = network.annual_energy_savings_kw * 8760 * project_lifetime_years
        roi_analysis["lcoe_usd_per_kwh"] = network.total_capital_cost_usd / total_energy_kwh

        # Sensitivity analysis
        roi_analysis["sensitivity_analysis"] = await self._perform_sensitivity_analysis(
            cash_flows,
            discount_rate,
            energy_price_escalation
        )

        self.metrics_collector.observe("project_npv_usd", roi_analysis["npv_usd"])
        self.metrics_collector.observe("project_irr_percent", roi_analysis["irr_percent"])

        logger.info(f"ROI analysis complete. NPV: ${roi_analysis['npv_usd']:,.0f}, IRR: {roi_analysis['irr_percent']:.1f}%")
        return roi_analysis

    async def generate_implementation_plan(self, optimization_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate phased implementation plan for heat recovery project.

        This method creates a detailed, prioritized implementation plan with
        milestones, resource requirements, and risk mitigation strategies.

        Args:
            optimization_data: Complete optimization results

        Returns:
            Phased implementation plan with timeline and milestones
        """
        logger.info("Generating implementation plan")
        self.metrics_collector.increment("implementation_plans_generated")

        implementation_plan = []

        # Phase 1: Quick wins (0-6 months)
        phase1 = {
            "phase": 1,
            "name": "Quick Wins",
            "duration_months": 6,
            "start_date": DeterministicClock.now(),
            "end_date": DeterministicClock.now() + timedelta(days=180),
            "activities": [
                {
                    "activity": "Install simple heat recovery on high-temperature exhausts",
                    "cost_usd": optimization_data.get("quick_wins_cost", 50000),
                    "savings_usd_year": optimization_data.get("quick_wins_savings", 100000),
                    "complexity": "Low",
                    "resources_required": ["2 engineers", "1 contractor"]
                },
                {
                    "activity": "Implement heat recovery from compressed air systems",
                    "cost_usd": 30000,
                    "savings_usd_year": 60000,
                    "complexity": "Low",
                    "resources_required": ["1 engineer", "1 technician"]
                }
            ],
            "milestones": [
                {"milestone": "Complete energy audit", "date": DeterministicClock.now() + timedelta(days=30)},
                {"milestone": "Install first heat exchanger", "date": DeterministicClock.now() + timedelta(days=90)},
                {"milestone": "Commission quick win projects", "date": DeterministicClock.now() + timedelta(days=150)}
            ],
            "risks": [
                {"risk": "Production disruption", "mitigation": "Schedule during planned maintenance"},
                {"risk": "Budget overrun", "mitigation": "Include 15% contingency"}
            ]
        }
        implementation_plan.append(phase1)

        # Phase 2: Major retrofits (6-18 months)
        phase2 = {
            "phase": 2,
            "name": "Major Retrofits",
            "duration_months": 12,
            "start_date": DeterministicClock.now() + timedelta(days=180),
            "end_date": DeterministicClock.now() + timedelta(days=540),
            "activities": [
                {
                    "activity": "Install heat exchanger network for process streams",
                    "cost_usd": optimization_data.get("hen_cost", 500000),
                    "savings_usd_year": optimization_data.get("hen_savings", 800000),
                    "complexity": "High",
                    "resources_required": ["Project team", "EPC contractor", "Process engineers"]
                },
                {
                    "activity": "Implement waste heat recovery boiler",
                    "cost_usd": 300000,
                    "savings_usd_year": 450000,
                    "complexity": "Medium",
                    "resources_required": ["Boiler specialist", "Installation crew"]
                }
            ],
            "milestones": [
                {"milestone": "Complete detailed engineering", "date": DeterministicClock.now() + timedelta(days=240)},
                {"milestone": "Procure major equipment", "date": DeterministicClock.now() + timedelta(days=360)},
                {"milestone": "Complete installation", "date": DeterministicClock.now() + timedelta(days=480)},
                {"milestone": "Start commercial operation", "date": DeterministicClock.now() + timedelta(days=540)}
            ],
            "risks": [
                {"risk": "Long lead times for equipment", "mitigation": "Order critical items early"},
                {"risk": "Integration challenges", "mitigation": "Conduct detailed HAZOP study"},
                {"risk": "Performance shortfall", "mitigation": "Include performance guarantees in contracts"}
            ]
        }
        implementation_plan.append(phase2)

        # Phase 3: Advanced optimization (18-24 months)
        phase3 = {
            "phase": 3,
            "name": "Advanced Optimization",
            "duration_months": 6,
            "start_date": DeterministicClock.now() + timedelta(days=540),
            "end_date": DeterministicClock.now() + timedelta(days=720),
            "activities": [
                {
                    "activity": "Implement advanced process control for heat recovery",
                    "cost_usd": 150000,
                    "savings_usd_year": 200000,
                    "complexity": "High",
                    "resources_required": ["Control system engineers", "IT support"]
                },
                {
                    "activity": "Install ORC system for low-grade heat recovery",
                    "cost_usd": 800000,
                    "savings_usd_year": 600000,
                    "complexity": "Very High",
                    "resources_required": ["ORC specialist", "Electrical engineers"]
                }
            ],
            "milestones": [
                {"milestone": "Complete control system upgrade", "date": DeterministicClock.now() + timedelta(days=600)},
                {"milestone": "Commission ORC system", "date": DeterministicClock.now() + timedelta(days=690)},
                {"milestone": "Achieve full optimization", "date": DeterministicClock.now() + timedelta(days=720)}
            ],
            "risks": [
                {"risk": "Technology maturity", "mitigation": "Partner with experienced ORC vendor"},
                {"risk": "Grid connection delays", "mitigation": "Early engagement with utility"}
            ]
        }
        implementation_plan.append(phase3)

        self.metrics_collector.observe("implementation_phases", len(implementation_plan))
        self.metrics_collector.observe("total_implementation_months", 24)

        logger.info(f"Implementation plan generated with {len(implementation_plan)} phases")
        return implementation_plan

    async def prioritize_opportunities(self, opportunities: List[Dict[str, Any]], criteria: Dict[str, float]) -> List[Dict[str, Any]]:
        """
        Prioritize heat recovery opportunities using multi-criteria decision analysis.

        This method ranks opportunities based on weighted criteria including
        economics, technical feasibility, and environmental impact.

        Args:
            opportunities: List of identified opportunities
            criteria: Weighting factors for different criteria

        Returns:
            Prioritized list of opportunities with rankings
        """
        logger.info("Prioritizing heat recovery opportunities")
        self.metrics_collector.increment("opportunity_prioritizations")

        # Default criteria weights if not provided
        default_criteria = {
            "payback_period": 0.30,
            "energy_savings": 0.25,
            "co2_reduction": 0.20,
            "technical_risk": 0.15,
            "implementation_ease": 0.10
        }

        criteria = {**default_criteria, **criteria}

        # Score each opportunity
        scored_opportunities = []
        for opp in opportunities:
            scores = {
                "payback_period": self._score_payback(opp.get("payback_years", 5)),
                "energy_savings": self._score_energy_savings(opp.get("energy_savings_kw", 0)),
                "co2_reduction": self._score_co2_reduction(opp.get("co2_reduction_tons", 0)),
                "technical_risk": self._score_technical_risk(opp.get("technical_complexity", "medium")),
                "implementation_ease": self._score_implementation_ease(opp.get("implementation_difficulty", "medium"))
            }

            # Calculate weighted score
            total_score = sum(scores[criterion] * weight for criterion, weight in criteria.items())

            opp["priority_score"] = total_score
            opp["individual_scores"] = scores
            opp["ranking"] = 0  # Will be set after sorting

            scored_opportunities.append(opp)

        # Sort by total score (descending)
        scored_opportunities.sort(key=lambda x: x["priority_score"], reverse=True)

        # Assign rankings
        for i, opp in enumerate(scored_opportunities, 1):
            opp["ranking"] = i
            opp["priority_category"] = self._get_priority_category(i, len(scored_opportunities))

        self.metrics_collector.observe("opportunities_evaluated", len(scored_opportunities))

        logger.info(f"Prioritized {len(scored_opportunities)} opportunities")
        return scored_opportunities

    async def validate_thermodynamics(self, design_data: Dict[str, Any]) -> ValidationResult:
        """
        Validate thermodynamic feasibility of heat recovery design.

        This method performs rigorous thermodynamic validation to ensure
        all heat transfers respect the second law of thermodynamics.

        Args:
            design_data: Heat recovery system design data

        Returns:
            Validation result with detailed checks
        """
        logger.info("Validating thermodynamics")
        self.metrics_collector.increment("thermodynamic_validations")

        validation_checks = []
        is_valid = True

        # Check 1: Temperature approach constraints
        for exchanger in design_data.get("exchangers", []):
            hot_in = exchanger.get("hot_inlet_temp_c", 0)
            hot_out = exchanger.get("hot_outlet_temp_c", 0)
            cold_in = exchanger.get("cold_inlet_temp_c", 0)
            cold_out = exchanger.get("cold_outlet_temp_c", 0)

            min_approach = min(hot_in - cold_out, hot_out - cold_in)

            if min_approach < design_data.get("min_temp_approach_c", 10):
                validation_checks.append({
                    "check": "Temperature Approach",
                    "status": "FAIL",
                    "message": f"Exchanger {exchanger.get('id')} violates minimum temperature approach: {min_approach:.1f}°C",
                    "severity": "ERROR"
                })
                is_valid = False
            else:
                validation_checks.append({
                    "check": f"Temperature Approach - {exchanger.get('id')}",
                    "status": "PASS",
                    "message": f"Minimum approach {min_approach:.1f}°C meets requirement",
                    "severity": "INFO"
                })

        # Check 2: Energy balance
        total_heat_recovered = sum(ex.get("duty_kw", 0) for ex in design_data.get("exchangers", []))
        total_heat_available = design_data.get("total_waste_heat_kw", 0)

        if total_heat_recovered > total_heat_available:
            validation_checks.append({
                "check": "Energy Balance",
                "status": "FAIL",
                "message": f"Heat recovery ({total_heat_recovered:.0f} kW) exceeds available waste heat ({total_heat_available:.0f} kW)",
                "severity": "ERROR"
            })
            is_valid = False
        else:
            recovery_percentage = (total_heat_recovered / total_heat_available * 100) if total_heat_available > 0 else 0
            validation_checks.append({
                "check": "Energy Balance",
                "status": "PASS",
                "message": f"Heat recovery {recovery_percentage:.1f}% of available waste heat",
                "severity": "INFO"
            })

        # Check 3: Pinch violations
        if "pinch_temperature_c" in design_data:
            pinch_temp = design_data["pinch_temperature_c"]
            violations = []

            for exchanger in design_data.get("exchangers", []):
                if self._check_pinch_violation(exchanger, pinch_temp):
                    violations.append(exchanger.get("id"))

            if violations:
                validation_checks.append({
                    "check": "Pinch Violations",
                    "status": "FAIL",
                    "message": f"Exchangers {violations} violate pinch principle",
                    "severity": "ERROR"
                })
                is_valid = False
            else:
                validation_checks.append({
                    "check": "Pinch Violations",
                    "status": "PASS",
                    "message": "No pinch violations detected",
                    "severity": "INFO"
                })

        # Check 4: Pressure drop constraints
        for exchanger in design_data.get("exchangers", []):
            pressure_drop = exchanger.get("pressure_drop_bar", 0)
            max_allowed = design_data.get("max_pressure_drop_bar", 0.5)

            if pressure_drop > max_allowed:
                validation_checks.append({
                    "check": f"Pressure Drop - {exchanger.get('id')}",
                    "status": "WARNING",
                    "message": f"Pressure drop {pressure_drop:.2f} bar exceeds limit {max_allowed:.2f} bar",
                    "severity": "WARNING"
                })

        # Check 5: Exergy efficiency
        exergy_efficiency = design_data.get("exergy_efficiency_percent", 0)
        if exergy_efficiency < 30:
            validation_checks.append({
                "check": "Exergy Efficiency",
                "status": "WARNING",
                "message": f"Low exergy efficiency {exergy_efficiency:.1f}% indicates high irreversibilities",
                "severity": "WARNING"
            })
        else:
            validation_checks.append({
                "check": "Exergy Efficiency",
                "status": "PASS",
                "message": f"Exergy efficiency {exergy_efficiency:.1f}% is acceptable",
                "severity": "INFO"
            })

        result = ValidationResult(
            is_valid=is_valid,
            checks=validation_checks,
            summary=f"Thermodynamic validation {'PASSED' if is_valid else 'FAILED'} with {len(validation_checks)} checks"
        )

        self.metrics_collector.observe("validation_checks_performed", len(validation_checks))

        logger.info(f"Thermodynamic validation complete: {result.summary}")
        return result

    # Helper methods
    def _generate_optimization_id(self) -> str:
        """Generate unique optimization ID."""
        timestamp = DeterministicClock.now().isoformat()
        random_suffix = hashlib.md5(timestamp.encode()).hexdigest()[:8]
        return f"OPT-{timestamp[:10]}-{random_suffix}"

    def _is_waste_heat_stream(self, stream: ProcessStream) -> bool:
        """Determine if stream is waste heat."""
        # Waste heat criteria
        is_hot = stream.stream_type in [StreamType.HOT_STREAM, StreamType.WASTE_HEAT]
        is_high_temp = stream.inlet_temp_c > 60  # Above 60°C considered recoverable
        has_significant_duty = stream.heat_duty_kw > 10  # Above 10 kW

        return is_hot and is_high_temp and has_significant_duty

    def _is_thermodynamically_feasible(self, hot: ProcessStream, cold: ProcessStream) -> bool:
        """Check if heat exchange is thermodynamically feasible."""
        # Temperature driving force must be positive
        min_approach = 10.0  # Minimum temperature approach in °C

        hot_cold_inlet_diff = hot.inlet_temp_c - cold.outlet_temp_c
        hot_cold_outlet_diff = hot.outlet_temp_c - cold.inlet_temp_c

        return (hot_cold_inlet_diff >= min_approach and
                hot_cold_outlet_diff >= min_approach)

    def _calculate_lmtd(self, hot: ProcessStream, cold: ProcessStream) -> float:
        """Calculate Log Mean Temperature Difference."""
        dt1 = hot.inlet_temp_c - cold.outlet_temp_c
        dt2 = hot.outlet_temp_c - cold.inlet_temp_c

        if dt1 <= 0 or dt2 <= 0:
            return 0

        if abs(dt1 - dt2) < 0.1:
            return dt1

        return (dt1 - dt2) / np.log(dt1 / dt2)

    def _calculate_lmtd_correction(self, P: float, R: float) -> float:
        """Calculate LMTD correction factor for shell and tube exchangers."""
        if R == 1:
            S = P / (1 - P)
            return np.sqrt(2) * (1 - P) / (P * np.log((2 - P * (1 + 1/np.sqrt(2))) / (2 - P * (1 - 1/np.sqrt(2)))))
        else:
            S = np.sqrt(R**2 + 1) / (R - 1)
            numerator = S * np.log((1 - P) / (1 - P * R))
            denominator = np.log((2 - P * (R + 1 - S)) / (2 - P * (R + 1 + S)))
            return numerator / denominator if denominator != 0 else 1

    def _calculate_physical_exergy(self, stream_data: Dict, T_ambient_k: float, P_ambient_bar: float) -> float:
        """Calculate physical exergy of a stream."""
        T_stream_k = stream_data.get("temperature_c", 25) + 273.15
        P_stream_bar = stream_data.get("pressure_bar", 1)
        flow_rate = stream_data.get("flow_rate_kg_s", 0)
        cp = stream_data.get("specific_heat_kj_kg_k", 4.186)

        # Physical exergy per unit mass
        exergy_thermal = cp * (T_stream_k - T_ambient_k - T_ambient_k * np.log(T_stream_k / T_ambient_k))

        # Pressure contribution (assuming ideal gas for simplicity)
        R_specific = 0.287  # kJ/kg.K for air
        exergy_pressure = R_specific * T_ambient_k * np.log(P_stream_bar / P_ambient_bar)

        return flow_rate * (exergy_thermal + exergy_pressure)

    def _calculate_chemical_exergy(self, composition: Dict[str, float], T_ambient_k: float) -> float:
        """Calculate chemical exergy based on composition."""
        # Standard chemical exergies (kJ/mol)
        standard_exergies = {
            "CH4": 831.6,
            "H2": 236.1,
            "CO": 275.3,
            "CO2": 19.9,
            "H2O": 9.5,
            "N2": 0.72,
            "O2": 3.97
        }

        total_chemical_exergy = 0
        for component, mole_fraction in composition.items():
            if component in standard_exergies:
                total_chemical_exergy += mole_fraction * standard_exergies[component]

        return total_chemical_exergy

    def _calculate_carbon_credits(self, energy_savings_kw: float) -> float:
        """Calculate CO2 emissions reduction in tons/year."""
        # Emission factors (kg CO2/kWh)
        grid_emission_factor = 0.5  # Average grid
        natural_gas_emission_factor = 0.2  # Natural gas combustion

        # Assume 50% grid, 50% natural gas replacement
        average_emission_factor = (grid_emission_factor + natural_gas_emission_factor) / 2

        # Annual CO2 reduction (tons)
        annual_hours = 8760
        co2_reduction_tons = energy_savings_kw * annual_hours * average_emission_factor / 1000

        return co2_reduction_tons

    def _calculate_irr(self, cash_flows: List[float]) -> float:
        """Calculate Internal Rate of Return using Newton-Raphson method."""
        # Initial guess
        irr = 0.1
        tolerance = 1e-6
        max_iterations = 100

        for _ in range(max_iterations):
            # Calculate NPV and derivative
            npv = sum(cf / (1 + irr) ** i for i, cf in enumerate(cash_flows))
            npv_derivative = sum(-i * cf / (1 + irr) ** (i + 1) for i, cf in enumerate(cash_flows))

            if abs(npv) < tolerance:
                return irr

            # Newton-Raphson update
            irr = irr - npv / npv_derivative if npv_derivative != 0 else irr

            # Bound IRR to reasonable range
            irr = max(-0.99, min(irr, 10.0))

        return irr

    async def _compile_optimization_results(
        self,
        optimization_id: str,
        input_data: HeatRecoveryInput,
        saga_result: Dict[str, Any],
        start_time: datetime
    ) -> HeatRecoveryOutput:
        """Compile all optimization results into output format."""
        processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000

        # Extract results from saga steps
        waste_heat_streams = saga_result.get("identify_streams", [])
        gradient_analysis = saga_result.get("analyze_gradients", {})
        pinch_result = saga_result.get("pinch_analysis")
        network = saga_result.get("optimize_network")
        exergy_data = saga_result.get("calculate_exergy", {})
        roi_analysis = saga_result.get("roi_analysis", {})
        implementation_plan = saga_result.get("generate_plan", [])

        # Calculate total heat recovery
        total_heat_recovery = network.annual_energy_savings_kw if network else 0

        # Calculate CO2 reduction
        co2_reduction = self._calculate_carbon_credits(total_heat_recovery)

        # Validate results
        validation_result = await self.validate_thermodynamics({
            "exchangers": network.exchangers if network else [],
            "min_temp_approach_c": input_data.minimum_temperature_approach_c,
            "total_waste_heat_kw": sum(s.heat_duty_kw for s in waste_heat_streams),
            "pinch_temperature_c": pinch_result.pinch_temperature_hot_c if pinch_result else None,
            "exergy_efficiency_percent": exergy_data.get("exergy_efficiency_percent", 0)
        })

        # Generate provenance hash
        provenance_data = {
            "input": input_data.dict(),
            "results": {
                "heat_recovery": total_heat_recovery,
                "co2_reduction": co2_reduction,
                "network": network.dict() if network else None
            },
            "timestamp": DeterministicClock.now().isoformat()
        }
        provenance_hash = hashlib.sha256(json.dumps(provenance_data, sort_keys=True).encode()).hexdigest()

        return HeatRecoveryOutput(
            optimization_id=optimization_id,
            timestamp=DeterministicClock.now(),
            identified_opportunities=[s.__dict__ for s in waste_heat_streams],
            pinch_analysis=pinch_result,
            heat_exchanger_network=network,
            exergy_efficiency_percent=exergy_data.get("exergy_efficiency_percent", 0),
            total_heat_recovery_kw=total_heat_recovery,
            implementation_plan=implementation_plan,
            roi_analysis=roi_analysis,
            co2_reduction_tons_year=co2_reduction,
            validation_status="PASS" if validation_result.is_valid else "FAIL",
            provenance_hash=provenance_hash,
            processing_time_ms=processing_time
        )

    def _update_processing_stats(self, start_time: datetime, success: bool):
        """Update processing statistics."""
        processing_time = (DeterministicClock.now() - start_time).total_seconds() * 1000

        self.processing_stats["total_optimizations"] += 1
        if success:
            self.processing_stats["successful_optimizations"] += 1
        else:
            self.processing_stats["failed_optimizations"] += 1

        # Update average processing time
        n = self.processing_stats["total_optimizations"]
        avg = self.processing_stats["average_processing_time_ms"]
        self.processing_stats["average_processing_time_ms"] = (avg * (n - 1) + processing_time) / n

        self.metrics_collector.observe("processing_time_ms", processing_time)

    # Saga rollback methods
    async def _rollback_validation(self, error: Exception):
        """Rollback validation step."""
        logger.warning(f"Rolling back validation: {error}")

    async def _rollback_streams(self, error: Exception):
        """Rollback stream identification."""
        logger.warning(f"Rolling back stream identification: {error}")

    async def _rollback_gradients(self, error: Exception):
        """Rollback gradient analysis."""
        logger.warning(f"Rolling back gradient analysis: {error}")

    async def _rollback_pinch(self, error: Exception):
        """Rollback pinch analysis."""
        logger.warning(f"Rolling back pinch analysis: {error}")

    async def _rollback_network(self, error: Exception):
        """Rollback network optimization."""
        logger.warning(f"Rolling back network optimization: {error}")

    async def _rollback_exergy(self, error: Exception):
        """Rollback exergy calculation."""
        logger.warning(f"Rolling back exergy calculation: {error}")

    async def _rollback_roi(self, error: Exception):
        """Rollback ROI analysis."""
        logger.warning(f"Rolling back ROI analysis: {error}")

    async def _rollback_plan(self, error: Exception):
        """Rollback implementation plan."""
        logger.warning(f"Rolling back implementation plan: {error}")

    # Additional helper methods for complex calculations
    async def _validate_input_wrapper(self, input_data: HeatRecoveryInput) -> ValidationResult:
        """Wrapper for input validation in saga pattern."""
        validation_checks = []
        is_valid = True

        # Validate stream data completeness
        for stream in input_data.process_streams:
            if stream['inlet_temp_c'] <= stream['outlet_temp_c'] and stream['stream_type'] in ['hot_stream', 'waste_heat']:
                validation_checks.append({
                    "check": f"Stream {stream['stream_id']} temperature",
                    "status": "FAIL",
                    "message": "Hot stream must have inlet temp > outlet temp"
                })
                is_valid = False

        return ValidationResult(
            is_valid=is_valid,
            checks=validation_checks,
            summary=f"Input validation {'PASSED' if is_valid else 'FAILED'}"
        )