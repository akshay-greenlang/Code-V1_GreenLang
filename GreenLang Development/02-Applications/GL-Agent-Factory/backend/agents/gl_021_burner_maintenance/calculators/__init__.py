"""
Calculators for GL-021 BURNERSENTRY Agent

This module exports all calculator functions for burner maintenance prediction.
All calculations are deterministic, following reliability engineering standards
with zero-hallucination principles.

Calculator Modules:
    - weibull: Weibull distribution failure analysis
    - flame_quality: Combustion and flame quality assessment
    - health_score: Overall health scoring and degradation analysis
    - maintenance: Maintenance scheduling and replacement decisions
"""

from .weibull import (
    weibull_reliability,
    weibull_failure_rate,
    weibull_mean_life,
    remaining_useful_life,
    calculate_failure_probability,
    weibull_percentile_life,
)

from .flame_quality import (
    calculate_flame_quality_score,
    calculate_combustion_efficiency,
    detect_flame_anomalies,
    calculate_excess_air,
    calculate_adiabatic_flame_temp,
)

from .health_score import (
    calculate_overall_health,
    calculate_degradation_rate,
    determine_maintenance_priority,
    calculate_component_weight,
)

from .maintenance import (
    generate_maintenance_recommendations,
    calculate_next_maintenance_date,
    should_replace_burner,
    calculate_maintenance_cost_benefit,
)

__all__ = [
    # Weibull calculators
    "weibull_reliability",
    "weibull_failure_rate",
    "weibull_mean_life",
    "remaining_useful_life",
    "calculate_failure_probability",
    "weibull_percentile_life",
    # Flame quality calculators
    "calculate_flame_quality_score",
    "calculate_combustion_efficiency",
    "detect_flame_anomalies",
    "calculate_excess_air",
    "calculate_adiabatic_flame_temp",
    # Health score calculators
    "calculate_overall_health",
    "calculate_degradation_rate",
    "determine_maintenance_priority",
    "calculate_component_weight",
    # Maintenance calculators
    "generate_maintenance_recommendations",
    "calculate_next_maintenance_date",
    "should_replace_burner",
    "calculate_maintenance_cost_benefit",
]
