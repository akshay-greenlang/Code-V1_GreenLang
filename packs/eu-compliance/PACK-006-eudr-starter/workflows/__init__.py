# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - Workflow Orchestration
=====================================================

Pre-built workflow orchestrators for EU Deforestation Regulation (EUDR)
compliance operations. Each workflow coordinates GreenLang agents, calculation
engines, and validation pipelines into end-to-end EUDR processes aligned with
EU Regulation 2023/1115 and its implementing/delegated acts.

Workflows:
    - DDSGenerationWorkflow: 6-phase DDS generation (primary compliance flow)
    - SupplierOnboardingWorkflow: 4-phase supplier registration and profiling
    - QuarterlyComplianceReviewWorkflow: 3-phase periodic compliance review
    - DataQualityBaselineWorkflow: 3-phase data quality assessment
    - RiskReassessmentWorkflow: 3-phase periodic risk reassessment
    - BulkImportWorkflow: 3-phase bulk data import

Regulatory Context:
    The EUDR (EU Regulation 2023/1115) requires operators and traders placing
    relevant commodities (cattle, cocoa, coffee, oil palm, rubber, soya, wood)
    on the EU market to exercise due diligence ensuring products are
    deforestation-free and comply with country-of-origin legislation. Due
    Diligence Statements (DDS) must be submitted to the EU Information System
    prior to placing goods on the market.

    These workflows implement the operational processes needed for:
    - Supplier onboarding and data collection (Articles 9-10)
    - Geolocation validation (Article 9(1)(d))
    - Risk assessment and country benchmarking (Articles 10, 29)
    - DDS generation and submission (Articles 4, 9)
    - Ongoing monitoring and reassessment (Articles 11, 13)

Author: GreenLang Team
Version: 1.0.0
"""

from typing import Any, Dict, List, Optional, Type

from packs.eu_compliance.PACK_006_eudr_starter.workflows.dds_generation import (
    DDSGenerationWorkflow,
    DDSGenerationInput,
    DDSGenerationResult,
    DDSContent,
    SupplierData,
    PlotGeolocation,
    GeolocationPoint,
    CertificationRecord,
    RiskScore,
    PhaseResult as DDSPhaseResult,
    PhaseStatus,
    DDType,
    RiskLevel,
    EUDRCommodity,
    CertificationType,
    ReviewDecision,
    WorkflowContext,
)
from packs.eu_compliance.PACK_006_eudr_starter.workflows.supplier_onboarding import (
    SupplierOnboardingWorkflow,
    SupplierOnboardingInput,
    SupplierOnboardingResult,
    SupplierRecord,
    PlotData,
    DataSource,
    OnboardingStatus,
)
from packs.eu_compliance.PACK_006_eudr_starter.workflows.quarterly_compliance_review import (
    QuarterlyComplianceReviewWorkflow,
    QuarterlyReviewInput,
    QuarterlyReviewResult,
    CertificationStatus,
)
from packs.eu_compliance.PACK_006_eudr_starter.workflows.data_quality_baseline import (
    DataQualityBaselineWorkflow,
    DataQualityInput,
    DataQualityResult,
    QualityDimension,
    ValidationSeverity,
    ValidationCategory,
    RemediationPriority,
)
from packs.eu_compliance.PACK_006_eudr_starter.workflows.risk_reassessment import (
    RiskReassessmentWorkflow,
    RiskReassessmentInput,
    RiskReassessmentResult,
    RiskAlert,
    AlertSeverity,
    AlertType,
)
from packs.eu_compliance.PACK_006_eudr_starter.workflows.bulk_import import (
    BulkImportWorkflow,
    BulkImportInput,
    BulkImportResult,
    FileUpload,
    FileFormat,
    RecordType,
    ImportStatus,
)


# =============================================================================
# WORKFLOW REGISTRY
# =============================================================================


class WorkflowRegistry:
    """
    Central registry for all PACK-006 EUDR Starter Pack workflows.

    Provides discovery, instantiation, and metadata access for all
    registered workflows. Used by the GL-EUDR-APP to dynamically
    load and execute workflows based on user actions.

    Attributes:
        _workflows: Internal registry of workflow metadata.

    Example:
        >>> registry = WorkflowRegistry()
        >>> workflow_names = registry.list_workflows()
        >>> wf = registry.get_workflow("dds_generation")
        >>> result = await wf.run(input_data)
    """

    def __init__(self) -> None:
        """Initialize the WorkflowRegistry with all PACK-006 workflows."""
        self._workflows: Dict[str, Dict[str, Any]] = {
            "dds_generation": {
                "name": "DDS Generation",
                "description": (
                    "Six-phase Due Diligence Statement generation workflow. "
                    "Primary EUDR compliance workflow from supplier onboarding "
                    "through EU Information System submission."
                ),
                "class": DDSGenerationWorkflow,
                "input_class": DDSGenerationInput,
                "result_class": DDSGenerationResult,
                "phases": 6,
                "phase_names": [
                    "supplier_onboarding",
                    "geolocation_collection",
                    "document_collection",
                    "risk_assessment",
                    "dds_generation",
                    "review_and_submit",
                ],
                "typical_duration_days": 17,
                "regulatory_articles": ["4", "9", "10", "29"],
                "agent_dependencies": [
                    "EUDR-001", "EUDR-002", "EUDR-006", "EUDR-007",
                    "EUDR-008", "EUDR-012", "EUDR-016", "EUDR-017",
                    "EUDR-018", "EUDR-028", "EUDR-030", "EUDR-036",
                    "EUDR-037", "EUDR-038", "DATA-001", "DATA-002",
                    "FOUND-005",
                ],
                "version": "1.0.0",
            },
            "supplier_onboarding": {
                "name": "Supplier Onboarding",
                "description": (
                    "Four-phase supplier registration and profiling workflow. "
                    "Handles data intake, profiling, geolocation setup, and "
                    "initial risk scoring."
                ),
                "class": SupplierOnboardingWorkflow,
                "input_class": SupplierOnboardingInput,
                "result_class": SupplierOnboardingResult,
                "phases": 4,
                "phase_names": [
                    "data_intake",
                    "supplier_profiling",
                    "geolocation_setup",
                    "initial_risk_scoring",
                ],
                "typical_duration_days": 5,
                "regulatory_articles": ["9", "10", "12"],
                "agent_dependencies": [
                    "DATA-001", "DATA-002", "DATA-005", "DATA-010",
                    "EUDR-002", "EUDR-006", "EUDR-007", "EUDR-008",
                    "EUDR-016", "EUDR-017", "EUDR-018",
                ],
                "version": "1.0.0",
            },
            "quarterly_compliance_review": {
                "name": "Quarterly Compliance Review",
                "description": (
                    "Three-phase quarterly compliance review workflow. "
                    "Refreshes data, recalculates risks, and generates "
                    "compliance reporting."
                ),
                "class": QuarterlyComplianceReviewWorkflow,
                "input_class": QuarterlyReviewInput,
                "result_class": QuarterlyReviewResult,
                "phases": 3,
                "phase_names": [
                    "data_refresh",
                    "risk_recalculation",
                    "compliance_reporting",
                ],
                "typical_duration_days": 3,
                "regulatory_articles": ["11", "13", "29", "30"],
                "agent_dependencies": [
                    "DATA-001", "DATA-002", "EUDR-016", "EUDR-017",
                    "EUDR-018", "EUDR-030",
                ],
                "version": "1.0.0",
            },
            "data_quality_baseline": {
                "name": "Data Quality Baseline",
                "description": (
                    "Three-phase data quality assessment workflow. Profiles "
                    "data quality, applies 45 EUDR validation rules, and "
                    "generates a prioritized remediation plan."
                ),
                "class": DataQualityBaselineWorkflow,
                "input_class": DataQualityInput,
                "result_class": DataQualityResult,
                "phases": 3,
                "phase_names": [
                    "profiling",
                    "validation",
                    "remediation",
                ],
                "typical_duration_days": 2,
                "regulatory_articles": ["9", "10", "11"],
                "agent_dependencies": [
                    "DATA-010", "DATA-011", "DATA-019",
                ],
                "version": "1.0.0",
            },
            "risk_reassessment": {
                "name": "Risk Reassessment",
                "description": (
                    "Three-phase periodic risk reassessment workflow. Collects "
                    "updated data, recalculates scores, and generates alerts "
                    "for material risk changes."
                ),
                "class": RiskReassessmentWorkflow,
                "input_class": RiskReassessmentInput,
                "result_class": RiskReassessmentResult,
                "phases": 3,
                "phase_names": [
                    "data_collection",
                    "risk_recalculation",
                    "alert_generation",
                ],
                "typical_duration_days": 2,
                "regulatory_articles": ["11", "13", "29"],
                "agent_dependencies": [
                    "EUDR-016", "EUDR-017", "EUDR-018",
                ],
                "version": "1.0.0",
            },
            "bulk_import": {
                "name": "Bulk Import",
                "description": (
                    "Three-phase bulk data import workflow. Parses uploaded "
                    "files (CSV, Excel, JSON, GeoJSON), validates against "
                    "EUDR rules, and integrates into the compliance system."
                ),
                "class": BulkImportWorkflow,
                "input_class": BulkImportInput,
                "result_class": BulkImportResult,
                "phases": 3,
                "phase_names": [
                    "file_processing",
                    "validation_and_enrichment",
                    "integration",
                ],
                "typical_duration_days": 1,
                "regulatory_articles": ["9", "10"],
                "agent_dependencies": [
                    "DATA-010", "DATA-011", "DATA-019",
                ],
                "version": "1.0.0",
            },
        }

    def list_workflows(self) -> List[str]:
        """
        List all registered workflow identifiers.

        Returns:
            List of workflow identifier strings.
        """
        return list(self._workflows.keys())

    def get_workflow(
        self,
        workflow_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        Instantiate a workflow by its identifier.

        Args:
            workflow_id: Workflow identifier (e.g. 'dds_generation').
            config: Optional configuration to pass to the workflow.

        Returns:
            Instantiated workflow object.

        Raises:
            KeyError: If workflow_id is not registered.
        """
        if workflow_id not in self._workflows:
            raise KeyError(
                f"Unknown workflow '{workflow_id}'. "
                f"Available: {self.list_workflows()}"
            )

        workflow_class = self._workflows[workflow_id]["class"]
        return workflow_class(config=config)

    def get_metadata(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get metadata for a registered workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Dict with workflow metadata (name, description, phases, etc.).

        Raises:
            KeyError: If workflow_id is not registered.
        """
        if workflow_id not in self._workflows:
            raise KeyError(
                f"Unknown workflow '{workflow_id}'. "
                f"Available: {self.list_workflows()}"
            )

        meta = dict(self._workflows[workflow_id])
        # Remove class references for serialization safety
        meta.pop("class", None)
        meta.pop("input_class", None)
        meta.pop("result_class", None)
        return meta

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all registered workflows.

        Returns:
            Dict mapping workflow_id to metadata.
        """
        return {
            wf_id: self.get_metadata(wf_id)
            for wf_id in self._workflows
        }

    def get_input_class(self, workflow_id: str) -> Type:
        """
        Get the Pydantic input model class for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Pydantic BaseModel class for workflow input.
        """
        if workflow_id not in self._workflows:
            raise KeyError(f"Unknown workflow '{workflow_id}'")
        return self._workflows[workflow_id]["input_class"]

    def get_result_class(self, workflow_id: str) -> Type:
        """
        Get the Pydantic result model class for a workflow.

        Args:
            workflow_id: Workflow identifier.

        Returns:
            Pydantic BaseModel class for workflow result.
        """
        if workflow_id not in self._workflows:
            raise KeyError(f"Unknown workflow '{workflow_id}'")
        return self._workflows[workflow_id]["result_class"]

    @property
    def total_phases(self) -> int:
        """Total number of phases across all workflows."""
        return sum(w["phases"] for w in self._workflows.values())

    @property
    def total_agent_dependencies(self) -> int:
        """Total unique agent dependencies across all workflows."""
        all_deps: set = set()
        for w in self._workflows.values():
            all_deps.update(w.get("agent_dependencies", []))
        return len(all_deps)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Registry
    "WorkflowRegistry",
    # DDS Generation
    "DDSGenerationWorkflow",
    "DDSGenerationInput",
    "DDSGenerationResult",
    "DDSContent",
    "SupplierData",
    "PlotGeolocation",
    "GeolocationPoint",
    "CertificationRecord",
    "RiskScore",
    "WorkflowContext",
    # Supplier Onboarding
    "SupplierOnboardingWorkflow",
    "SupplierOnboardingInput",
    "SupplierOnboardingResult",
    "SupplierRecord",
    "PlotData",
    "DataSource",
    "OnboardingStatus",
    # Quarterly Compliance Review
    "QuarterlyComplianceReviewWorkflow",
    "QuarterlyReviewInput",
    "QuarterlyReviewResult",
    "CertificationStatus",
    # Data Quality Baseline
    "DataQualityBaselineWorkflow",
    "DataQualityInput",
    "DataQualityResult",
    "QualityDimension",
    "ValidationSeverity",
    "ValidationCategory",
    "RemediationPriority",
    # Risk Reassessment
    "RiskReassessmentWorkflow",
    "RiskReassessmentInput",
    "RiskReassessmentResult",
    "RiskAlert",
    "AlertSeverity",
    "AlertType",
    # Bulk Import
    "BulkImportWorkflow",
    "BulkImportInput",
    "BulkImportResult",
    "FileUpload",
    "FileFormat",
    "RecordType",
    "ImportStatus",
    # Shared Enums
    "PhaseStatus",
    "DDType",
    "RiskLevel",
    "EUDRCommodity",
    "CertificationType",
    "ReviewDecision",
]
