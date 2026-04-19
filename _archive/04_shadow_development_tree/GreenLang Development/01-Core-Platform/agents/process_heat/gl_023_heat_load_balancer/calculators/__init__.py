"""
GL-023 HEATLOADBALANCER - Calculator Module

This module provides zero-hallucination calculation engines for heat load
balancing optimization. All calculators are deterministic with complete
provenance tracking for regulatory compliance.

Calculator Categories:
    - Efficiency Curves: Polynomial and piecewise efficiency calculations
    - Fuel Cost: Fuel cost, emissions cost, and multi-fuel optimization
    - Economic Dispatch: Load allocation using equal incremental cost
    - Provenance: SHA-256 hashing and audit trail generation

Key Features:
    - ZERO-HALLUCINATION: All calculations are deterministic (no ML/LLM)
    - Bit-perfect reproducibility (same input -> same output)
    - Complete provenance tracking with SHA-256 hashes
    - Standards compliant (ASME PTC 4.1, API 560, EPA Method 19)
    - Performance optimized (<5ms per calculation target)

Example:
    >>> from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators import (
    ...     PolynomialEfficiencyCalculator,
    ...     EconomicDispatchCalculator,
    ...     UnitConfiguration,
    ... )
    >>>
    >>> # Calculate efficiency at 80% load
    >>> eff_calc = PolynomialEfficiencyCalculator(
    ...     coefficients=[0.70, 0.40, -0.25, 0.05],
    ...     unit_id="BLR-001",
    ... )
    >>> result = eff_calc.calculate(load_fraction=0.8)
    >>> print(f"Efficiency: {result.value:.2f}% (hash: {result.calculation_hash[:16]}...)")
    >>>
    >>> # Economic dispatch across multiple units
    >>> units = [
    ...     UnitConfiguration(
    ...         unit_id="BLR-001",
    ...         min_capacity_mmbtu_hr=20,
    ...         max_capacity_mmbtu_hr=100,
    ...         fuel_price=3.50,
    ...     ),
    ...     UnitConfiguration(
    ...         unit_id="BLR-002",
    ...         min_capacity_mmbtu_hr=30,
    ...         max_capacity_mmbtu_hr=150,
    ...         fuel_price=3.25,
    ...     ),
    ... ]
    >>> dispatch = EconomicDispatchCalculator()
    >>> result = dispatch.dispatch(total_demand=200, units=units)
    >>> print(f"Total cost: ${result.total_cost_per_hr:.2f}/hr")

Standards Reference:
    - ASME PTC 4.1-2013: Steam Generator Performance Test Codes
    - API 560: Fired Heaters for General Refinery Service
    - EPA Method 19: Determination of Sulfur Dioxide Removal Efficiency
    - 40 CFR Part 98: GHG Emission Reporting
    - IEEE Economic Dispatch Standards

Author: GreenLang Process Heat Team
Version: 1.0.0
"""

# =============================================================================
# EFFICIENCY CURVES MODULE
# =============================================================================

from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators.efficiency_curves import (
    # Constants
    EfficiencyConstants,
    FuelProperties,

    # Data Models
    EfficiencyResult,
    FuelConsumptionResult,
    StackLossResult,
    CurveFitResult,

    # Calculators
    PolynomialEfficiencyCalculator,
    PiecewiseLinearEfficiency,
    EfficiencyCurveFitter,
    PartLoadEfficiencyCalculator,
    BatchEfficiencyCalculator,

    # Functions
    calculate_fuel_consumption,
    calculate_stack_losses,
)

# =============================================================================
# FUEL COST MODULE
# =============================================================================

from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators.fuel_cost import (
    # Constants
    FuelCostConstants,
    MaintenanceCostFactors,

    # Data Models
    FuelCostResult,
    EmissionsCostResult,
    TotalOperatingCostResult,
    MultiFuelOptimizationResult,
    SpotPriceData,

    # Calculators
    FuelCostCalculator,
    EmissionsCostCalculator,
    TotalOperatingCostCalculator,
    MultiFuelCostOptimizer,
    SpotPriceIntegration,
)

# =============================================================================
# ECONOMIC DISPATCH MODULE
# =============================================================================

from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators.economic_dispatch import (
    # Constants
    EconomicDispatchConstants,

    # Data Models
    UnitConfiguration,
    UnitDispatch,
    EconomicDispatchResult,
    IncrementalCostResult,
    ReserveMarginResult,
    LossFactorResult,

    # Calculators
    IncrementalCostCalculator,
    EqualIncrementalCostSolver,
    EconomicDispatchCalculator,
    LossFactorCalculator,
    ReserveMarginCalculator,
)

# =============================================================================
# PROVENANCE MODULE
# =============================================================================

from greenlang.agents.process_heat.gl_023_heat_load_balancer.calculators.provenance import (
    # Enums
    ProvenanceType,
    AuditEventType,
    ComplianceFramework,

    # Data Models
    ProvenanceRecord,
    AuditEvent,
    OptimizationAuditSummary,
    VerificationResult,

    # Classes
    ProvenanceHashGenerator,
    OptimizationAuditTrail,
    DeterministicVerifier,
    ComplianceRecordGenerator,
)

# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # =========================================================================
    # Efficiency Curves
    # =========================================================================
    # Constants
    "EfficiencyConstants",
    "FuelProperties",

    # Data Models
    "EfficiencyResult",
    "FuelConsumptionResult",
    "StackLossResult",
    "CurveFitResult",

    # Calculators
    "PolynomialEfficiencyCalculator",
    "PiecewiseLinearEfficiency",
    "EfficiencyCurveFitter",
    "PartLoadEfficiencyCalculator",
    "BatchEfficiencyCalculator",

    # Functions
    "calculate_fuel_consumption",
    "calculate_stack_losses",

    # =========================================================================
    # Fuel Cost
    # =========================================================================
    # Constants
    "FuelCostConstants",
    "MaintenanceCostFactors",

    # Data Models
    "FuelCostResult",
    "EmissionsCostResult",
    "TotalOperatingCostResult",
    "MultiFuelOptimizationResult",
    "SpotPriceData",

    # Calculators
    "FuelCostCalculator",
    "EmissionsCostCalculator",
    "TotalOperatingCostCalculator",
    "MultiFuelCostOptimizer",
    "SpotPriceIntegration",

    # =========================================================================
    # Economic Dispatch
    # =========================================================================
    # Constants
    "EconomicDispatchConstants",

    # Data Models
    "UnitConfiguration",
    "UnitDispatch",
    "EconomicDispatchResult",
    "IncrementalCostResult",
    "ReserveMarginResult",
    "LossFactorResult",

    # Calculators
    "IncrementalCostCalculator",
    "EqualIncrementalCostSolver",
    "EconomicDispatchCalculator",
    "LossFactorCalculator",
    "ReserveMarginCalculator",

    # =========================================================================
    # Provenance
    # =========================================================================
    # Enums
    "ProvenanceType",
    "AuditEventType",
    "ComplianceFramework",

    # Data Models
    "ProvenanceRecord",
    "AuditEvent",
    "OptimizationAuditSummary",
    "VerificationResult",

    # Classes
    "ProvenanceHashGenerator",
    "OptimizationAuditTrail",
    "DeterministicVerifier",
    "ComplianceRecordGenerator",
]


# =============================================================================
# VERSION INFO
# =============================================================================

__version__ = "1.0.0"
__author__ = "GreenLang Process Heat Team"
__standards__ = [
    "ASME PTC 4.1-2013",
    "API 560",
    "EPA Method 19",
    "40 CFR Part 98",
]


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _check_dependencies() -> None:
    """Check required dependencies are available."""
    required = ["pydantic"]
    missing = []

    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        import warnings
        warnings.warn(
            f"GL-023 HeatLoadBalancer calculators: Missing optional dependencies: {missing}. "
            "Some features may not be available."
        )


# Run dependency check on import
_check_dependencies()
