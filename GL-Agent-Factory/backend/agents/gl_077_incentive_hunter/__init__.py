"""GL-077: Incentive Hunter Agent (INCENTIVEHUNTER)"""

from .agent import (
    IncentiveHunterAgent,
    IncentiveHunterInput,
    LocationInfo,
    EquipmentInfo,
    ProjectScope,
    UtilityProvider,
    IncentiveHunterOutput,
    AvailableIncentive,
    EligibilityStatus,
    ApplicationRequirement,
    ProvenanceRecord,
    IncentiveType,
    IncentiveCategory,
    EligibilityState,
    PACK_SPEC,
)

from .formulas import (
    calculate_incentive_value,
    calculate_payback_impact,
    calculate_stacking_limit,
    estimate_application_success,
)

__all__ = [
    "IncentiveHunterAgent",
    "IncentiveHunterInput",
    "LocationInfo",
    "EquipmentInfo",
    "ProjectScope",
    "UtilityProvider",
    "IncentiveHunterOutput",
    "AvailableIncentive",
    "EligibilityStatus",
    "ApplicationRequirement",
    "ProvenanceRecord",
    "IncentiveType",
    "IncentiveCategory",
    "EligibilityState",
    "PACK_SPEC",
    "calculate_incentive_value",
    "calculate_payback_impact",
    "calculate_stacking_limit",
    "estimate_application_success",
]

__version__ = "1.0.0"
__agent_id__ = "GL-077"
__agent_name__ = "INCENTIVEHUNTER"
