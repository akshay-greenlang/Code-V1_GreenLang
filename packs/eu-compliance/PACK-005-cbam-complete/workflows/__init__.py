# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Workflow Orchestration
====================================================

Production-grade workflow orchestrators for advanced CBAM (Carbon Border
Adjustment Mechanism) compliance operations. Each workflow coordinates
GreenLang agents, PACK-005 calculation engines, and validation pipelines
into end-to-end CBAM processes aligned with EU Regulation 2023/956 and
Implementing Regulation 2023/1773.

This module extends the foundation laid by PACK-004 (CBAM Readiness) with
enterprise-grade workflows for multi-entity groups, active certificate
trading, customs automation, cross-regulation synchronization, registry
submission, and audit preparation.

Workflows:
    - CertificateTradingWorkflow: 6-phase weekly certificate trading cycle
      covering price monitoring, obligation forecasting, purchase decisions
      (5 strategies), order execution, portfolio rebalancing, and reporting.

    - MultiEntityConsolidationWorkflow: 5-phase quarterly consolidation
      across group entities with per-entity calculation, group-level
      aggregation, cost allocation (5 methods), and consolidated reporting.

    - RegistrySubmissionWorkflow: 4-phase submission lifecycle with
      pre-validation, eIDAS-authenticated submission, status monitoring,
      and downstream confirmation with retry/backoff.

    - CrossRegulationSyncWorkflow: 4-phase sync triggered by CBAM data
      changes, mapping to CSRD ESRS E1, CDP C6/C7/C11, SBTi Scope 3,
      EU Taxonomy climate mitigation, EU ETS, and EUDR supply chains.

    - CustomsIntegrationWorkflow: 3-phase per-import workflow parsing
      SAD/CDS customs declarations, TARIC enrichment with anti-circumvention
      detection, and CBAM record linkage with preliminary emission estimates.

    - AuditPreparationWorkflow: 5-phase annual audit readiness cycle
      with completeness scanning, gap analysis, evidence assembly,
      quality review, and scored readiness assessment.

Shared Infrastructure:
    - WorkflowStatus / PhaseStatus enums for consistent state tracking
    - WorkflowContext for inter-phase state propagation
    - PhaseResult / WorkflowResult Pydantic models with provenance hashes
    - WorkflowRegistry for programmatic workflow discovery and instantiation
    - Checkpoint/resume support via phase-level status tracking

Regulatory Context:
    CBAM definitive period begins January 2026 with certificate
    purchase/surrender obligations. Annual declarations are due by
    May 31 each year. Certificates are priced at the EU ETS weekly
    weighted average auction price (1 certificate = 1 tCO2e). Quarterly
    holding compliance requires at least 50% of estimated annual
    obligation. Penalties are EUR 100/tCO2e (inflation-adjusted).

Author: GreenLang Team
Version: 1.0.0
"""

from packs.eu_compliance.PACK_005_cbam_complete.workflows.certificate_trading import (
    CertificateTradingWorkflow,
    CertificateTradingInput,
    CertificateTradingResult,
)
from packs.eu_compliance.PACK_005_cbam_complete.workflows.multi_entity_consolidation import (
    MultiEntityConsolidationWorkflow,
    MultiEntityConsolidationInput,
    MultiEntityConsolidationResult,
)
from packs.eu_compliance.PACK_005_cbam_complete.workflows.registry_submission import (
    RegistrySubmissionWorkflow,
    RegistrySubmissionInput,
    RegistrySubmissionResult,
)
from packs.eu_compliance.PACK_005_cbam_complete.workflows.cross_regulation_sync import (
    CrossRegulationSyncWorkflow,
    CrossRegulationSyncInput,
    CrossRegulationSyncResult,
)
from packs.eu_compliance.PACK_005_cbam_complete.workflows.customs_integration import (
    CustomsIntegrationWorkflow,
    CustomsIntegrationInput,
    CustomsIntegrationResult,
)
from packs.eu_compliance.PACK_005_cbam_complete.workflows.audit_preparation import (
    AuditPreparationWorkflow,
    AuditPreparationInput,
    AuditPreparationResult,
)

__version__ = "1.0.0"
__author__ = "GreenLang Team"


# =============================================================================
# WORKFLOW REGISTRY
# =============================================================================


class WorkflowRegistry:
    """
    Registry for PACK-005 CBAM Complete workflows.

    Provides programmatic discovery and instantiation of all available
    workflows. Each workflow is registered with its name, class, input
    model, result model, and a human-readable description.

    Attributes:
        _workflows: Internal mapping of workflow name to metadata dict.

    Example:
        >>> registry = WorkflowRegistry()
        >>> names = registry.list_workflows()
        >>> wf_cls = registry.get_workflow_class("certificate_trading")
        >>> instance = registry.create_workflow("certificate_trading")
        >>> assert isinstance(instance, CertificateTradingWorkflow)
    """

    _workflows = {
        "certificate_trading": {
            "class": CertificateTradingWorkflow,
            "input_model": CertificateTradingInput,
            "result_model": CertificateTradingResult,
            "description": (
                "6-phase weekly certificate trading cycle with price monitoring, "
                "obligation forecasting, purchase decisions (5 strategies), "
                "order execution, portfolio rebalancing, and reporting."
            ),
            "frequency": "weekly",
            "phases": 6,
        },
        "multi_entity_consolidation": {
            "class": MultiEntityConsolidationWorkflow,
            "input_model": MultiEntityConsolidationInput,
            "result_model": MultiEntityConsolidationResult,
            "description": (
                "5-phase quarterly consolidation across group entities with "
                "per-entity calculation, group aggregation, cost allocation, "
                "and consolidated reporting."
            ),
            "frequency": "quarterly",
            "phases": 5,
        },
        "registry_submission": {
            "class": RegistrySubmissionWorkflow,
            "input_model": RegistrySubmissionInput,
            "result_model": RegistrySubmissionResult,
            "description": (
                "4-phase submission lifecycle with pre-validation, eIDAS "
                "submission, status monitoring, and downstream confirmation."
            ),
            "frequency": "per_submission",
            "phases": 4,
        },
        "cross_regulation_sync": {
            "class": CrossRegulationSyncWorkflow,
            "input_model": CrossRegulationSyncInput,
            "result_model": CrossRegulationSyncResult,
            "description": (
                "4-phase sync triggered by CBAM data changes, mapping to "
                "CSRD, CDP, SBTi, EU Taxonomy, EU ETS, and EUDR."
            ),
            "frequency": "on_change",
            "phases": 4,
        },
        "customs_integration": {
            "class": CustomsIntegrationWorkflow,
            "input_model": CustomsIntegrationInput,
            "result_model": CustomsIntegrationResult,
            "description": (
                "3-phase per-import workflow parsing SAD/CDS declarations, "
                "TARIC enrichment with anti-circumvention detection, and "
                "CBAM record linkage."
            ),
            "frequency": "per_import",
            "phases": 3,
        },
        "audit_preparation": {
            "class": AuditPreparationWorkflow,
            "input_model": AuditPreparationInput,
            "result_model": AuditPreparationResult,
            "description": (
                "5-phase annual audit readiness cycle with completeness "
                "scanning, gap analysis, evidence assembly, quality review, "
                "and scored readiness assessment."
            ),
            "frequency": "annual",
            "phases": 5,
        },
    }

    def list_workflows(self) -> list:
        """
        List all registered workflow names.

        Returns:
            Sorted list of workflow name strings.
        """
        return sorted(self._workflows.keys())

    def get_workflow_metadata(self, name: str) -> dict:
        """
        Get full metadata for a workflow by name.

        Args:
            name: Workflow identifier (e.g., 'certificate_trading').

        Returns:
            Dict with class, input_model, result_model, description,
            frequency, and phases.

        Raises:
            KeyError: If workflow name is not registered.
        """
        if name not in self._workflows:
            raise KeyError(
                f"Unknown workflow '{name}'. "
                f"Available: {', '.join(self.list_workflows())}"
            )
        return self._workflows[name]

    def get_workflow_class(self, name: str) -> type:
        """
        Get the workflow class for a given name.

        Args:
            name: Workflow identifier.

        Returns:
            The workflow class (not an instance).

        Raises:
            KeyError: If workflow name is not registered.
        """
        return self.get_workflow_metadata(name)["class"]

    def create_workflow(self, name: str, **kwargs) -> object:
        """
        Instantiate a workflow by name.

        Args:
            name: Workflow identifier.
            **kwargs: Keyword arguments passed to the workflow constructor.

        Returns:
            An instance of the requested workflow class.

        Raises:
            KeyError: If workflow name is not registered.
        """
        cls = self.get_workflow_class(name)
        return cls(**kwargs)


__all__ = [
    # Certificate Trading
    "CertificateTradingWorkflow",
    "CertificateTradingInput",
    "CertificateTradingResult",
    # Multi-Entity Consolidation
    "MultiEntityConsolidationWorkflow",
    "MultiEntityConsolidationInput",
    "MultiEntityConsolidationResult",
    # Registry Submission
    "RegistrySubmissionWorkflow",
    "RegistrySubmissionInput",
    "RegistrySubmissionResult",
    # Cross-Regulation Sync
    "CrossRegulationSyncWorkflow",
    "CrossRegulationSyncInput",
    "CrossRegulationSyncResult",
    # Customs Integration
    "CustomsIntegrationWorkflow",
    "CustomsIntegrationInput",
    "CustomsIntegrationResult",
    # Audit Preparation
    "AuditPreparationWorkflow",
    "AuditPreparationInput",
    "AuditPreparationResult",
    # Registry
    "WorkflowRegistry",
]
