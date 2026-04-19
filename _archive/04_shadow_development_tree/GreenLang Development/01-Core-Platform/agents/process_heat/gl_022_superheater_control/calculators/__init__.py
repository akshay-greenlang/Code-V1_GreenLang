"""
GL-022 SUPERHEATER CONTROL - Calculators Module

This module provides ZERO-HALLUCINATION deterministic calculators for
superheater temperature control applications including:

- Thermodynamics: Steam property calculations using IAPWS-IF97 principles
- Spray Optimization: Spray water flow and mixing thermodynamics
- Temperature Control: PID, cascade, feedforward controllers
- Provenance: SHA-256 tracking and calculation audit trails

All calculations are deterministic, reproducible, and include complete
provenance tracking for regulatory compliance and audit purposes.

Standards Compliance:
- IAPWS-IF97: Industrial Formulation for Water and Steam Properties
- ISA-77.43: Fossil Fuel Power Plant Desuperheater Controls
- ASME PTC 4.2: Steam Generating Units
- EPRI Guidelines: Spray Desuperheater Operation

Key Formulas Implemented:
- Spray water required: m_spray = m_steam * (h_in - h_out) / (h_out - h_water)
- Temperature control: delta_T = Q / (m * Cp)
- Heat transfer: Q = U * A * LMTD
- Thermal stress: sigma = E * alpha * delta_T / (1 - nu)

Example:
    >>> from greenlang.agents.process_heat.gl_022_superheater_control.calculators import (
    ...     SteamThermodynamicsCalculator,
    ...     SprayFlowCalculator,
    ...     PIDController,
    ...     ProvenanceTracker,
    ... )
    >>>
    >>> # Steam property calculation
    >>> thermo = SteamThermodynamicsCalculator()
    >>> props = thermo.get_steam_properties(pressure_psig=600, temperature_f=900)
    >>> print(f"Enthalpy: {props.enthalpy_btu_lb:.1f} BTU/lb")
    >>>
    >>> # Spray flow calculation
    >>> spray_calc = SprayFlowCalculator()
    >>> result = spray_calc.calculate_spray_requirements(
    ...     steam_flow_lb_hr=100000,
    ...     inlet_temp_f=950.0,
    ...     target_temp_f=850.0,
    ...     spray_water_temp_f=250.0,
    ...     pressure_psig=600.0,
    ... )
    >>> print(f"Spray flow: {result.spray_flow_lb_hr:.0f} lb/hr")
    >>>
    >>> # PID control
    >>> pid = PIDController(kp=2.0, ki=0.1, kd=0.5)
    >>> output = pid.calculate(setpoint=850.0, process_value=860.0, dt=1.0)
    >>> print(f"Control output: {output.output:.1f}%")
    >>>
    >>> # Provenance tracking
    >>> tracker = ProvenanceTracker()
    >>> record = tracker.track_calculation(
    ...     calc_type="spray_flow",
    ...     inputs={"steam_flow": 100000},
    ...     outputs={"spray_flow": 5000},
    ...     formula="m_spray = m_steam * (h_in - h_out) / (h_out - h_water)",
    ... )
    >>> print(f"Provenance hash: {record.provenance_hash[:16]}...")
"""

# =============================================================================
# THERMODYNAMICS MODULE
# =============================================================================

from .thermodynamics import (
    # Constants
    IAPWSIF97Constants,
    ThermalStressConstants,
    # Data Classes
    SteamPropertiesResult,
    HeatTransferResult,
    EnthalpyBalanceResult,
    ThermalStressResult,
    # Calculators
    SteamThermodynamicsCalculator,
    SuperheaterHeatTransferCalculator,
    EnthalpyBalanceCalculator,
    ThermalStressCalculator,
    # Factory Functions
    create_thermodynamics_calculator,
    create_heat_transfer_calculator,
    create_enthalpy_balance_calculator,
    create_thermal_stress_calculator,
)

# =============================================================================
# SPRAY OPTIMIZATION MODULE
# =============================================================================

from .spray_optimization import (
    # Constants
    SprayDesuperheaterConstants,
    DropletEvaporationConstants,
    # Data Classes
    SprayRequirementsResult,
    MixingThermodynamicsResult,
    DropletEvaporationResult,
    SprayEfficiencyResult,
    WaterQualityImpactResult,
    # Calculators
    SprayFlowCalculator,
    MixingThermodynamicsCalculator,
    DropletEvaporationCalculator,
    SprayEfficiencyCalculator,
    WaterQualityImpactAnalyzer,
    # Factory Functions
    create_spray_flow_calculator,
    create_mixing_thermodynamics_calculator,
    create_droplet_evaporation_calculator,
    create_spray_efficiency_calculator,
    create_water_quality_impact_analyzer,
)

# =============================================================================
# TEMPERATURE CONTROL MODULE
# =============================================================================

from .temperature_control import (
    # Constants
    ControlSystemConstants,
    ThermalProtectionConstants,
    # Data Classes
    PIDControlResult,
    CascadeControlResult,
    FeedforwardResult,
    RateLimiterResult,
    ControllerState,
    # Controllers
    PIDController,
    CascadeController,
    FeedforwardController,
    RateLimiter,
    TemperatureRateLimiter,
    SimpleMPC,
    SuperheaterTemperatureController,
    # Factory Functions
    create_pid_controller,
    create_cascade_controller,
    create_feedforward_controller,
    create_superheater_temperature_controller,
)

# =============================================================================
# PROVENANCE MODULE
# =============================================================================

from .provenance import (
    # Constants
    ProvenanceConstants,
    # Data Classes
    CalculationInput,
    CalculationStep,
    ProvenanceRecord,
    AuditEntry,
    VerificationResult,
    # Trackers and Verifiers
    ProvenanceHashGenerator,
    CalculationAuditTrail,
    ProvenanceTracker,
    DeterministicVerifier,
    # Factory Functions
    create_provenance_tracker,
    create_audit_trail,
    create_hash_generator,
    create_deterministic_verifier,
    # Convenience Functions
    generate_provenance_hash,
    verify_provenance_hash,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # =========================================================================
    # THERMODYNAMICS
    # =========================================================================
    # Constants
    "IAPWSIF97Constants",
    "ThermalStressConstants",
    # Data Classes
    "SteamPropertiesResult",
    "HeatTransferResult",
    "EnthalpyBalanceResult",
    "ThermalStressResult",
    # Calculators
    "SteamThermodynamicsCalculator",
    "SuperheaterHeatTransferCalculator",
    "EnthalpyBalanceCalculator",
    "ThermalStressCalculator",
    # Factory Functions
    "create_thermodynamics_calculator",
    "create_heat_transfer_calculator",
    "create_enthalpy_balance_calculator",
    "create_thermal_stress_calculator",

    # =========================================================================
    # SPRAY OPTIMIZATION
    # =========================================================================
    # Constants
    "SprayDesuperheaterConstants",
    "DropletEvaporationConstants",
    # Data Classes
    "SprayRequirementsResult",
    "MixingThermodynamicsResult",
    "DropletEvaporationResult",
    "SprayEfficiencyResult",
    "WaterQualityImpactResult",
    # Calculators
    "SprayFlowCalculator",
    "MixingThermodynamicsCalculator",
    "DropletEvaporationCalculator",
    "SprayEfficiencyCalculator",
    "WaterQualityImpactAnalyzer",
    # Factory Functions
    "create_spray_flow_calculator",
    "create_mixing_thermodynamics_calculator",
    "create_droplet_evaporation_calculator",
    "create_spray_efficiency_calculator",
    "create_water_quality_impact_analyzer",

    # =========================================================================
    # TEMPERATURE CONTROL
    # =========================================================================
    # Constants
    "ControlSystemConstants",
    "ThermalProtectionConstants",
    # Data Classes
    "PIDControlResult",
    "CascadeControlResult",
    "FeedforwardResult",
    "RateLimiterResult",
    "ControllerState",
    # Controllers
    "PIDController",
    "CascadeController",
    "FeedforwardController",
    "RateLimiter",
    "TemperatureRateLimiter",
    "SimpleMPC",
    "SuperheaterTemperatureController",
    # Factory Functions
    "create_pid_controller",
    "create_cascade_controller",
    "create_feedforward_controller",
    "create_superheater_temperature_controller",

    # =========================================================================
    # PROVENANCE
    # =========================================================================
    # Constants
    "ProvenanceConstants",
    # Data Classes
    "CalculationInput",
    "CalculationStep",
    "ProvenanceRecord",
    "AuditEntry",
    "VerificationResult",
    # Trackers and Verifiers
    "ProvenanceHashGenerator",
    "CalculationAuditTrail",
    "ProvenanceTracker",
    "DeterministicVerifier",
    # Factory Functions
    "create_provenance_tracker",
    "create_audit_trail",
    "create_hash_generator",
    "create_deterministic_verifier",
    # Convenience Functions
    "generate_provenance_hash",
    "verify_provenance_hash",
]

# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__agent_id__ = "GL-022"
__agent_name__ = "SUPERHEATER CONTROL"

# =============================================================================
# MODULE DOCUMENTATION
# =============================================================================

MODULE_INFO = {
    "agent_id": __agent_id__,
    "agent_name": __agent_name__,
    "version": __version__,
    "description": "Zero-hallucination calculators for superheater temperature control",
    "modules": {
        "thermodynamics": {
            "description": "Steam property calculations using IAPWS-IF97",
            "key_classes": [
                "SteamThermodynamicsCalculator",
                "SuperheaterHeatTransferCalculator",
                "EnthalpyBalanceCalculator",
                "ThermalStressCalculator",
            ],
        },
        "spray_optimization": {
            "description": "Spray desuperheater optimization calculations",
            "key_classes": [
                "SprayFlowCalculator",
                "MixingThermodynamicsCalculator",
                "DropletEvaporationCalculator",
                "SprayEfficiencyCalculator",
                "WaterQualityImpactAnalyzer",
            ],
        },
        "temperature_control": {
            "description": "PID and advanced temperature control algorithms",
            "key_classes": [
                "PIDController",
                "CascadeController",
                "FeedforwardController",
                "SuperheaterTemperatureController",
            ],
        },
        "provenance": {
            "description": "SHA-256 provenance tracking and audit trails",
            "key_classes": [
                "ProvenanceTracker",
                "CalculationAuditTrail",
                "DeterministicVerifier",
            ],
        },
    },
    "key_formulas": {
        "spray_flow": "m_spray = m_steam * (h_in - h_out) / (h_out - h_water)",
        "heat_transfer": "Q = U * A * LMTD",
        "temperature_change": "delta_T = Q / (m * Cp)",
        "thermal_stress": "sigma = E * alpha * delta_T / (1 - nu)",
        "pid_control": "u = Kp*e + Ki*integral(e) + Kd*de/dt",
    },
    "standards": [
        "IAPWS-IF97 (Steam Properties)",
        "ISA-77.43 (Desuperheater Controls)",
        "ASME PTC 4.2 (Steam Generation)",
        "EPRI (Spray Desuperheater Guidelines)",
    ],
    "zero_hallucination_guarantee": True,
    "provenance_tracking": True,
    "deterministic": True,
}


def get_module_info() -> dict:
    """Get module information dictionary."""
    return MODULE_INFO
