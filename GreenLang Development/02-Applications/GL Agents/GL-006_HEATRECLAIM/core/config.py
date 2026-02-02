"""
GL-006 HEATRECLAIM - Configuration Module

Configuration settings, enums, and constants for the Heat Recovery
Maximizer agent. All configuration supports deterministic,
reproducible operation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import os


class StreamType(Enum):
    """Heat stream classification."""
    HOT = "hot"
    COLD = "cold"
    UTILITY_HOT = "utility_hot"
    UTILITY_COLD = "utility_cold"


class Phase(Enum):
    """Fluid phase."""
    LIQUID = "liquid"
    GAS = "gas"
    TWO_PHASE = "two_phase"
    SUPERCRITICAL = "supercritical"


class ExchangerType(Enum):
    """Heat exchanger types."""
    SHELL_AND_TUBE = "shell_and_tube"
    PLATE = "plate"
    PLATE_FIN = "plate_fin"
    SPIRAL = "spiral"
    AIR_COOLED = "air_cooled"
    DOUBLE_PIPE = "double_pipe"
    ECONOMIZER = "economizer"
    RECUPERATOR = "recuperator"


class FlowArrangement(Enum):
    """Heat exchanger flow arrangements."""
    COUNTER_CURRENT = "counter_current"
    CO_CURRENT = "co_current"
    CROSS_FLOW = "cross_flow"
    SHELL_PASS_1_TUBE_2 = "1-2"
    SHELL_PASS_2_TUBE_4 = "2-4"


class OptimizationMode(Enum):
    """Optimization modes."""
    GRASSROOTS = "grassroots"  # New plant design
    RETROFIT = "retrofit"      # Existing plant modification
    OPERATIONAL = "operational"  # Real-time re-optimization


class OptimizationObjective(Enum):
    """Optimization objectives."""
    MINIMIZE_UTILITY = "minimize_utility"
    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_EXERGY_LOSS = "minimize_exergy_loss"
    MINIMIZE_AREA = "minimize_area"
    MINIMIZE_EXCHANGERS = "minimize_exchangers"
    MULTI_OBJECTIVE = "multi_objective"


class SolverType(Enum):
    """Mathematical programming solvers."""
    PULP_CBC = "pulp_cbc"
    CVXPY = "cvxpy"
    PYMOO_NSGA2 = "pymoo_nsga2"
    PYMOO_NSGA3 = "pymoo_nsga3"
    SCIPY = "scipy"


class ExplainabilityMethod(Enum):
    """Explainability methods."""
    SHAP = "shap"
    LIME = "lime"
    CAUSAL = "causal"
    CONSTRAINT_BASED = "constraint_based"


# Physical Constants
REFERENCE_TEMPERATURE_K = 298.15  # 25°C
REFERENCE_PRESSURE_KPA = 101.325  # 1 atm
WATER_SPECIFIC_HEAT_KJ_KGK = 4.186
AIR_SPECIFIC_HEAT_KJ_KGK = 1.005
STEFAN_BOLTZMANN = 5.67e-8  # W/(m²·K⁴)


@dataclass
class ThermalConstraints:
    """Thermal safety and operability constraints."""

    # Minimum approach temperatures by stream class (°C)
    delta_t_min_default: float = 10.0
    delta_t_min_gas_gas: float = 20.0
    delta_t_min_gas_liquid: float = 15.0
    delta_t_min_liquid_liquid: float = 10.0
    delta_t_min_phase_change: float = 5.0

    # Temperature limits (°C)
    max_film_temperature: float = 400.0
    acid_dew_point: float = 120.0
    min_outlet_temperature: float = 80.0
    max_thermal_stress_rate: float = 5.0  # °C/min

    # Pressure drop limits (kPa)
    max_pressure_drop_liquid: float = 50.0
    max_pressure_drop_gas: float = 5.0

    # Fouling factors (m²·K/W)
    fouling_factor_clean_liquid: float = 0.0001
    fouling_factor_dirty_liquid: float = 0.0005
    fouling_factor_flue_gas: float = 0.002
    fouling_factor_cooling_water: float = 0.0002


@dataclass
class EconomicParameters:
    """Economic calculation parameters."""

    # Capital cost factors
    exchanger_cost_factor: float = 1000.0  # $/m²
    installation_factor: float = 1.5
    piping_factor: float = 0.2
    instrumentation_factor: float = 0.15

    # Economic parameters
    discount_rate: float = 0.10  # 10%
    project_lifetime_years: int = 20
    operating_hours_per_year: int = 8000

    # Utility costs ($/GJ)
    steam_cost_usd_gj: float = 15.0
    fuel_gas_cost_usd_gj: float = 8.0
    electricity_cost_usd_kwh: float = 0.10
    cooling_water_cost_usd_gj: float = 0.5
    chilled_water_cost_usd_gj: float = 5.0

    # Maintenance
    maintenance_cost_fraction: float = 0.03  # 3% of capex/year


@dataclass
class UncertaintyParameters:
    """Uncertainty quantification parameters."""

    # Flow rate uncertainty (%)
    flow_rate_uncertainty: float = 5.0

    # Temperature uncertainty (°C)
    temperature_uncertainty: float = 2.0

    # Cp uncertainty (%)
    cp_uncertainty: float = 3.0

    # Fouling uncertainty (%)
    fouling_uncertainty: float = 20.0

    # Price uncertainty (%)
    price_uncertainty: float = 15.0

    # Monte Carlo settings
    n_samples: int = 1000
    random_seed: int = 42
    confidence_level: float = 0.95


@dataclass
class HeatReclaimConfig:
    """
    Master configuration for GL-006 HEATRECLAIM agent.

    All settings required for deterministic, reproducible operation
    of the heat recovery optimization system.
    """

    # Agent identification
    agent_id: str = "GL-006"
    agent_name: str = "HEATRECLAIM"
    version: str = "1.0.0"

    # Operational mode
    mode: OptimizationMode = OptimizationMode.GRASSROOTS
    objective: OptimizationObjective = OptimizationObjective.MINIMIZE_COST

    # Reference conditions for exergy
    reference_temperature_K: float = REFERENCE_TEMPERATURE_K
    reference_pressure_kPa: float = REFERENCE_PRESSURE_KPA

    # Pinch analysis settings
    delta_t_min_C: float = 10.0
    allow_pinch_relaxation: bool = False
    pinch_relaxation_penalty: float = 1000.0  # $/kW violation

    # Optimization settings
    solver: SolverType = SolverType.PULP_CBC
    max_optimization_time_s: float = 300.0
    convergence_tolerance: float = 1e-6
    max_iterations: int = 10000

    # Pareto settings
    n_pareto_points: int = 50
    pareto_objectives: List[str] = field(default_factory=lambda: [
        "total_annual_cost",
        "heat_recovered",
        "exergy_destruction",
    ])

    # Constraints
    thermal_constraints: ThermalConstraints = field(default_factory=ThermalConstraints)
    economic_params: EconomicParameters = field(default_factory=EconomicParameters)
    uncertainty_params: UncertaintyParameters = field(default_factory=UncertaintyParameters)

    # Explainability
    explainability_methods: List[ExplainabilityMethod] = field(default_factory=lambda: [
        ExplainabilityMethod.SHAP,
        ExplainabilityMethod.CONSTRAINT_BASED,
    ])
    generate_causal_graph: bool = True

    # Audit and provenance
    enable_audit_logging: bool = True
    enable_provenance_tracking: bool = True
    deterministic_mode: bool = True
    fail_closed: bool = True

    # Integration
    kafka_brokers: str = field(default_factory=lambda: os.getenv("KAFKA_BROKERS", "localhost:9092"))
    opcua_endpoint: str = field(default_factory=lambda: os.getenv("OPCUA_ENDPOINT", "opc.tcp://localhost:4840"))
    graphql_port: int = 8080
    metrics_port: int = 9090

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "version": self.version,
            "mode": self.mode.value,
            "objective": self.objective.value,
            "delta_t_min_C": self.delta_t_min_C,
            "solver": self.solver.value,
            "deterministic_mode": self.deterministic_mode,
        }


# Default configuration instance
DEFAULT_CONFIG = HeatReclaimConfig()
