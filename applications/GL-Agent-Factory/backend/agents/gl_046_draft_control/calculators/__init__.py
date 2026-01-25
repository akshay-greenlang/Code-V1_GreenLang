"""
GL-046 Draft Control Calculators

Physics-based calculators for furnace draft, stack effect, and damper optimization.
All calculations follow NFPA 86 and API 560 standards.
"""

from .draft import (
    calculate_stack_effect,
    calculate_stack_effect_inwc,
    calculate_draft_loss,
    calculate_theoretical_draft,
    calculate_draft_velocity,
)
from .damper import (
    calculate_optimal_damper_position,
    calculate_damper_cv,
    calculate_pressure_drop_damper,
    calculate_damper_authority,
)
from .safety import (
    check_draft_safety,
    calculate_flue_gas_velocity,
    check_positive_pressure_risk,
    get_safety_interlock_status,
)

__all__ = [
    # Draft calculations
    "calculate_stack_effect",
    "calculate_stack_effect_inwc",
    "calculate_draft_loss",
    "calculate_theoretical_draft",
    "calculate_draft_velocity",
    # Damper calculations
    "calculate_optimal_damper_position",
    "calculate_damper_cv",
    "calculate_pressure_drop_damper",
    "calculate_damper_authority",
    # Safety calculations
    "check_draft_safety",
    "calculate_flue_gas_velocity",
    "check_positive_pressure_risk",
    "get_safety_interlock_status",
]
