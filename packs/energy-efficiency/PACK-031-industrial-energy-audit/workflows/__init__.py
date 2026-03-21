# -*- coding: utf-8 -*-
"""
PACK-031 Industrial Energy Audit Pack - Workflow Orchestration
================================================================

Industrial energy audit workflow orchestrators for EN 16247, ISO 50001,
IPMVP, and EED compliance operations. Each workflow coordinates GreenLang
agents, data pipelines, calculation engines, and validation systems into
end-to-end energy audit processes covering initial audits, continuous
monitoring, savings verification, compressed air, steam systems, waste
heat recovery, regulatory compliance, and ISO 50001 certification.

Workflows:
    - InitialEnergyAuditWorkflow: 5-phase comprehensive energy audit with
      facility registration, data collection, baseline establishment,
      EN 16247 compliant audit execution, and report generation.

    - ContinuousMonitoringWorkflow: 4-phase real-time monitoring with
      data ingestion, CUSUM deviation detection, alert generation, and
      EnPI trend analysis with seasonal adjustment.

    - EnergySavingsVerificationWorkflow: 4-phase IPMVP-compliant savings
      verification with baseline validation, implementation tracking,
      post-implementation measurement, and M&V reporting.

    - CompressedAirAuditWorkflow: 4-phase compressed air system audit
      with system mapping, ultrasonic leak survey, performance testing,
      and optimization recommendations.

    - SteamSystemAuditWorkflow: 4-phase steam system audit with boiler
      assessment, distribution survey, condensate analysis, and recovery
      optimization including CHP assessment.

    - WasteHeatRecoveryWorkflow: 4-phase waste heat recovery assessment
      with source identification, pinch analysis, technology selection,
      and ROI calculation with NPV analysis.

    - RegulatoryComplianceWorkflow: 3-phase EED compliance workflow with
      obligation checking, audit scheduling, and compliance reporting
      for national authority submissions.

    - ISO50001CertificationWorkflow: 4-phase ISO 50001:2018 certification
      support with EnMS gap analysis, energy policy development, EnPI
      tracking per ISO 50006, and management review preparation.

Author: GreenLang Team
Version: 31.0.0
"""

# ---------------------------------------------------------------------------
# Initial Energy Audit Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.initial_energy_audit_workflow import (
    InitialEnergyAuditWorkflow,
    InitialEnergyAuditInput,
    InitialEnergyAuditResult,
    FacilityData,
    EquipmentRecord,
    MeterRecord,
    EnergyConsumptionRecord,
    ProductionRecord,
    WeatherRecord,
    BaselineResult,
    AuditFinding,
    SavingsOpportunity,
)

# ---------------------------------------------------------------------------
# Continuous Monitoring Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.continuous_monitoring_workflow import (
    ContinuousMonitoringWorkflow,
    ContinuousMonitoringInput,
    ContinuousMonitoringResult,
    MeterReading,
    SCADADataPoint,
    BMSDataPoint,
    DeviationAlert,
    EnPITrend,
)

# ---------------------------------------------------------------------------
# Energy Savings Verification Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.energy_savings_verification_workflow import (
    EnergySavingsVerificationWorkflow,
    EnergySavingsVerificationInput,
    EnergySavingsVerificationResult,
    BaselinePeriodData,
    ECMImplementation,
    PostImplementationData,
    MVReport,
    IPMVPOption,
)

# ---------------------------------------------------------------------------
# Compressed Air Audit Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.compressed_air_audit_workflow import (
    CompressedAirAuditWorkflow,
    CompressedAirAuditInput,
    CompressedAirAuditResult,
    CompressorRecord,
    DryerRecord,
    ReceiverRecord,
    LeakRecord,
    CompressorPerformance,
    CompressedAirRecommendation,
)

# ---------------------------------------------------------------------------
# Steam System Audit Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.steam_system_audit_workflow import (
    SteamSystemAuditWorkflow,
    SteamSystemAuditInput,
    SteamSystemAuditResult,
    BoilerRecord,
    FlueGasAnalysis,
    SteamTrapRecord,
    InsulationRecord,
    CondensateRecord,
    SteamRecoveryOption,
)

# ---------------------------------------------------------------------------
# Waste Heat Recovery Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.waste_heat_recovery_workflow import (
    WasteHeatRecoveryWorkflow,
    WasteHeatRecoveryInput,
    WasteHeatRecoveryResult,
    HeatSource,
    HeatSink,
    PinchAnalysisResult,
    RecoveryTechnology,
    ROICalculation,
)

# ---------------------------------------------------------------------------
# Regulatory Compliance Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.regulatory_compliance_workflow import (
    RegulatoryComplianceWorkflow,
    RegulatoryComplianceInput,
    RegulatoryComplianceResult,
    FacilityObligationData,
    EEDObligationResult,
    AuditScheduleEntry,
    ComplianceReportData,
)

# ---------------------------------------------------------------------------
# ISO 50001 Certification Workflow
# ---------------------------------------------------------------------------
from packs.energy_efficiency.PACK_031_industrial_energy_audit.workflows.iso_50001_certification_workflow import (
    ISO50001CertificationWorkflow,
    ISO50001CertificationInput,
    ISO50001CertificationResult,
    GapAnalysisItem,
    EnergyPolicyDraft,
    EnPIRecord,
    ManagementReviewPackage,
)

__all__ = [
    # --- Initial Energy Audit Workflow ---
    "InitialEnergyAuditWorkflow",
    "InitialEnergyAuditInput",
    "InitialEnergyAuditResult",
    "FacilityData",
    "EquipmentRecord",
    "MeterRecord",
    "EnergyConsumptionRecord",
    "ProductionRecord",
    "WeatherRecord",
    "BaselineResult",
    "AuditFinding",
    "SavingsOpportunity",
    # --- Continuous Monitoring Workflow ---
    "ContinuousMonitoringWorkflow",
    "ContinuousMonitoringInput",
    "ContinuousMonitoringResult",
    "MeterReading",
    "SCADADataPoint",
    "BMSDataPoint",
    "DeviationAlert",
    "EnPITrend",
    # --- Energy Savings Verification Workflow ---
    "EnergySavingsVerificationWorkflow",
    "EnergySavingsVerificationInput",
    "EnergySavingsVerificationResult",
    "BaselinePeriodData",
    "ECMImplementation",
    "PostImplementationData",
    "MVReport",
    "IPMVPOption",
    # --- Compressed Air Audit Workflow ---
    "CompressedAirAuditWorkflow",
    "CompressedAirAuditInput",
    "CompressedAirAuditResult",
    "CompressorRecord",
    "DryerRecord",
    "ReceiverRecord",
    "LeakRecord",
    "CompressorPerformance",
    "CompressedAirRecommendation",
    # --- Steam System Audit Workflow ---
    "SteamSystemAuditWorkflow",
    "SteamSystemAuditInput",
    "SteamSystemAuditResult",
    "BoilerRecord",
    "FlueGasAnalysis",
    "SteamTrapRecord",
    "InsulationRecord",
    "CondensateRecord",
    "SteamRecoveryOption",
    # --- Waste Heat Recovery Workflow ---
    "WasteHeatRecoveryWorkflow",
    "WasteHeatRecoveryInput",
    "WasteHeatRecoveryResult",
    "HeatSource",
    "HeatSink",
    "PinchAnalysisResult",
    "RecoveryTechnology",
    "ROICalculation",
    # --- Regulatory Compliance Workflow ---
    "RegulatoryComplianceWorkflow",
    "RegulatoryComplianceInput",
    "RegulatoryComplianceResult",
    "FacilityObligationData",
    "EEDObligationResult",
    "AuditScheduleEntry",
    "ComplianceReportData",
    # --- ISO 50001 Certification Workflow ---
    "ISO50001CertificationWorkflow",
    "ISO50001CertificationInput",
    "ISO50001CertificationResult",
    "GapAnalysisItem",
    "EnergyPolicyDraft",
    "EnPIRecord",
    "ManagementReviewPackage",
]
