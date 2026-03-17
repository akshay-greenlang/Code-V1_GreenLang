"""
PACK-012 CSRD Financial Service Pack - Workflows module.

Provides eight workflow orchestrators for CSRD financial institution
compliance reporting.

Workflows:
    1. FinancedEmissionsWorkflow - PCAF financed emissions (5 phases)
    2. GARBTARWorkflow - EU Taxonomy GAR/BTAR (4 phases)
    3. InsuranceEmissionsWorkflow - Insurance-associated emissions (4 phases)
    4. ClimateStressTestWorkflow - Climate stress testing (5 phases)
    5. FSMaterialityWorkflow - FI double materiality (4 phases)
    6. TransitionPlanWorkflow - FI transition plan (4 phases)
    7. Pillar3ReportingWorkflow - EBA Pillar 3 ESG ITS (4 phases)
    8. RegulatoryIntegrationWorkflow - Cross-regulatory mapping (3 phases)
"""

from .financed_emissions_workflow import (
    FinancedEmissionsInput,
    FinancedEmissionsResult,
    FinancedEmissionsWorkflow,
)
from .gar_btar_workflow import (
    GARBTARInput,
    GARBTARResult,
    GARBTARWorkflow,
)
from .insurance_emissions_workflow import (
    InsuranceEmissionsInput,
    InsuranceEmissionsResult,
    InsuranceEmissionsWorkflow,
)
from .climate_stress_test_workflow import (
    ClimateStressTestInput,
    ClimateStressTestResult,
    ClimateStressTestWorkflow,
)
from .fs_materiality_workflow import (
    FSMaterialityInput,
    FSMaterialityResult,
    FSMaterialityWorkflow,
)
from .transition_plan_workflow import (
    TransitionPlanInput,
    TransitionPlanResult,
    TransitionPlanWorkflow,
)
from .pillar3_reporting_workflow import (
    Pillar3ReportingInput,
    Pillar3ReportingResult,
    Pillar3ReportingWorkflow,
)
from .regulatory_integration_workflow import (
    RegulatoryIntegrationInput,
    RegulatoryIntegrationResult,
    RegulatoryIntegrationWorkflow,
)

__all__ = [
    # Financed Emissions (5-phase)
    "FinancedEmissionsWorkflow",
    "FinancedEmissionsInput",
    "FinancedEmissionsResult",
    # GAR/BTAR (4-phase)
    "GARBTARWorkflow",
    "GARBTARInput",
    "GARBTARResult",
    # Insurance Emissions (4-phase)
    "InsuranceEmissionsWorkflow",
    "InsuranceEmissionsInput",
    "InsuranceEmissionsResult",
    # Climate Stress Test (5-phase)
    "ClimateStressTestWorkflow",
    "ClimateStressTestInput",
    "ClimateStressTestResult",
    # FS Materiality (4-phase)
    "FSMaterialityWorkflow",
    "FSMaterialityInput",
    "FSMaterialityResult",
    # Transition Plan (4-phase)
    "TransitionPlanWorkflow",
    "TransitionPlanInput",
    "TransitionPlanResult",
    # Pillar 3 Reporting (4-phase)
    "Pillar3ReportingWorkflow",
    "Pillar3ReportingInput",
    "Pillar3ReportingResult",
    # Regulatory Integration (3-phase)
    "RegulatoryIntegrationWorkflow",
    "RegulatoryIntegrationInput",
    "RegulatoryIntegrationResult",
]
