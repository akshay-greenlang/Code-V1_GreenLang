# -*- coding: utf-8 -*-
"""
PACK-032 Building Energy Assessment Pack - Workflow Orchestration
===================================================================

Building energy assessment workflow orchestrators for EPBD, EPC, LEED,
BREEAM, MEES, nZEB, and multi-framework compliance operations. Each
workflow coordinates GreenLang agents, data pipelines, calculation
engines, and validation systems into end-to-end building energy
assessment processes covering initial assessment, EPC generation,
retrofit planning, continuous monitoring, certification assessment,
tenant engagement, regulatory compliance, and nZEB readiness.

Workflows:
    - InitialBuildingAssessmentWorkflow: 5-phase comprehensive building
      energy assessment with building registration, data collection,
      envelope assessment, systems assessment, and report generation.

    - EPCGenerationWorkflow: 4-phase Energy Performance Certificate
      generation with building data validation, EN 15603 energy
      calculation, A-G rating assignment, and certificate lodgement.

    - RetrofitPlanningWorkflow: 4-phase building retrofit planning
      with baseline establishment, 60+ measure screening, NPV/IRR/
      payback cost-benefit analysis, and staged roadmap with MACC curve.

    - ContinuousBuildingMonitoringWorkflow: 4-phase ongoing performance
      monitoring with data ingestion, CUSUM deviation analysis, anomaly
      detection with BMS fault alerting, and YoY trend reporting.

    - CertificationAssessmentWorkflow: 4-phase green building
      certification assessment with scheme selection (LEED/BREEAM/
      Energy Star/NABERS), credit scoring, gap analysis, and
      prioritised action plan.

    - TenantEngagementWorkflow: 3-phase tenant energy engagement
      with tenant profiling, CIBSE TM46/BBP REEB benchmarking, and
      green lease compliance reporting.

    - RegulatoryComplianceWorkflow: 3-phase building energy regulatory
      compliance with EPBD/MEES/BPS obligation assessment, compliance
      gap checking, and penalty-avoidance action planning.

    - NZEBReadinessWorkflow: 4-phase Nearly Zero-Energy Building
      readiness assessment with current performance baseline, nZEB gap
      analysis by country, deep retrofit measure prioritisation, and
      staged roadmap with milestone verification.

Author: GreenLang Team
Version: 32.0.0
"""

# ---------------------------------------------------------------------------
# Initial Building Assessment Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.initial_building_assessment_workflow import (
    InitialBuildingAssessmentWorkflow,
    InitialBuildingAssessmentInput,
    InitialBuildingAssessmentResult,
    BuildingData,
    EnvelopeElement,
    UtilityBillRecord,
    BMSDataPoint,
    OccupancyRecord,
    HVACSystem,
    LightingSystem,
    DHWSystem,
    RenewableSystem,
    EnvelopeAssessmentResult,
    SystemsAssessmentResult,
    ImprovementRecommendation,
)

# ---------------------------------------------------------------------------
# EPC Generation Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.epc_generation_workflow import (
    EPCGenerationWorkflow,
    EPCGenerationInput,
    EPCGenerationResult,
    FabricInput,
    SystemsInput,
    GeometryInput,
    EnergyDemandBreakdown,
    RatingResult,
    EPCRecommendation,
    LodgementData,
    ValidationCheck,
)

# ---------------------------------------------------------------------------
# Retrofit Planning Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.retrofit_planning_workflow import (
    RetrofitPlanningWorkflow,
    RetrofitPlanningInput,
    RetrofitPlanningResult,
    BuildingBaseline,
    ScreenedMeasure,
    CostBenefitResult,
    RoadmapItem,
    MACCDataPoint,
)

# ---------------------------------------------------------------------------
# Continuous Building Monitoring Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.continuous_building_monitoring_workflow import (
    ContinuousBuildingMonitoringWorkflow,
    ContinuousBuildingMonitoringInput,
    ContinuousBuildingMonitoringResult,
    EnergyDataPoint,
    BMSReading,
    BaselineModel,
    CUSUMResult,
    MonitoringAlert,
    TrendMetric,
    PerformanceDashboard,
)

# ---------------------------------------------------------------------------
# Certification Assessment Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.certification_assessment_workflow import (
    CertificationAssessmentWorkflow,
    CertificationAssessmentInput,
    CertificationAssessmentResult,
    BuildingPerformanceData,
    CreditAssessment,
    GapItem,
    CertificationAction,
)

# ---------------------------------------------------------------------------
# Tenant Engagement Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.tenant_engagement_workflow import (
    TenantEngagementWorkflow,
    TenantEngagementInput,
    TenantEngagementResult,
    TenantSpace,
    TenantEnergyData,
    BuildingTotalData,
    PortfolioData,
    TenantBenchmark,
    GreenLeaseAssessment,
    TenantReport,
)

# ---------------------------------------------------------------------------
# Regulatory Compliance Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.regulatory_compliance_workflow import (
    RegulatoryComplianceWorkflow,
    RegulatoryComplianceInput,
    RegulatoryComplianceResult,
    RegulationObligation,
    ComplianceGap,
    ComplianceAction,
)

# ---------------------------------------------------------------------------
# nZEB Readiness Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_032_building_energy_assessment.workflows.nzeb_readiness_workflow import (
    NZEBReadinessWorkflow,
    NZEBReadinessInput,
    NZEBReadinessResult,
    CurrentPerformanceData,
    NZEBGap,
    NZEBMeasure,
    RoadmapMilestone,
)


def get_loaded_workflows() -> dict:
    """
    Return a mapping of workflow names to their classes.

    Useful for dynamic workflow dispatch in pipeline orchestrators
    and the GreenLang Agent Factory.

    Returns:
        Dictionary mapping workflow identifier strings to workflow classes.

    Example:
        >>> workflows = get_loaded_workflows()
        >>> wf_cls = workflows["initial_building_assessment"]
        >>> wf = wf_cls(config={})
        >>> result = await wf.execute(input_data)
    """
    return {
        "initial_building_assessment": InitialBuildingAssessmentWorkflow,
        "epc_generation": EPCGenerationWorkflow,
        "retrofit_planning": RetrofitPlanningWorkflow,
        "continuous_building_monitoring": ContinuousBuildingMonitoringWorkflow,
        "certification_assessment": CertificationAssessmentWorkflow,
        "tenant_engagement": TenantEngagementWorkflow,
        "regulatory_compliance": RegulatoryComplianceWorkflow,
        "nzeb_readiness": NZEBReadinessWorkflow,
    }


__all__ = [
    # --- Initial Building Assessment Workflow ---
    "InitialBuildingAssessmentWorkflow",
    "InitialBuildingAssessmentInput",
    "InitialBuildingAssessmentResult",
    "BuildingData",
    "EnvelopeElement",
    "UtilityBillRecord",
    "BMSDataPoint",
    "OccupancyRecord",
    "HVACSystem",
    "LightingSystem",
    "DHWSystem",
    "RenewableSystem",
    "EnvelopeAssessmentResult",
    "SystemsAssessmentResult",
    "ImprovementRecommendation",
    # --- EPC Generation Workflow ---
    "EPCGenerationWorkflow",
    "EPCGenerationInput",
    "EPCGenerationResult",
    "FabricInput",
    "SystemsInput",
    "GeometryInput",
    "EnergyDemandBreakdown",
    "RatingResult",
    "EPCRecommendation",
    "LodgementData",
    "ValidationCheck",
    # --- Retrofit Planning Workflow ---
    "RetrofitPlanningWorkflow",
    "RetrofitPlanningInput",
    "RetrofitPlanningResult",
    "BuildingBaseline",
    "ScreenedMeasure",
    "CostBenefitResult",
    "RoadmapItem",
    "MACCDataPoint",
    # --- Continuous Building Monitoring Workflow ---
    "ContinuousBuildingMonitoringWorkflow",
    "ContinuousBuildingMonitoringInput",
    "ContinuousBuildingMonitoringResult",
    "EnergyDataPoint",
    "BMSReading",
    "BaselineModel",
    "CUSUMResult",
    "MonitoringAlert",
    "TrendMetric",
    "PerformanceDashboard",
    # --- Certification Assessment Workflow ---
    "CertificationAssessmentWorkflow",
    "CertificationAssessmentInput",
    "CertificationAssessmentResult",
    "BuildingPerformanceData",
    "CreditAssessment",
    "GapItem",
    "CertificationAction",
    # --- Tenant Engagement Workflow ---
    "TenantEngagementWorkflow",
    "TenantEngagementInput",
    "TenantEngagementResult",
    "TenantSpace",
    "TenantEnergyData",
    "BuildingTotalData",
    "PortfolioData",
    "TenantBenchmark",
    "GreenLeaseAssessment",
    "TenantReport",
    # --- Regulatory Compliance Workflow ---
    "RegulatoryComplianceWorkflow",
    "RegulatoryComplianceInput",
    "RegulatoryComplianceResult",
    "RegulationObligation",
    "ComplianceGap",
    "ComplianceAction",
    # --- nZEB Readiness Workflow ---
    "NZEBReadinessWorkflow",
    "NZEBReadinessInput",
    "NZEBReadinessResult",
    "CurrentPerformanceData",
    "NZEBGap",
    "NZEBMeasure",
    "RoadmapMilestone",
    # --- Utility ---
    "get_loaded_workflows",
]
