# -*- coding: utf-8 -*-
"""
GL-CBAM-APP Supplier Portal Module v1.1

Self-service portal for third-country installation operators to submit
embedded emissions data per EU CBAM Regulation 2023/956.

This module provides:
  - SupplierRegistryEngine: Supplier registration, EORI validation,
    installation management, verification lifecycle, importer linking.
  - EmissionsSubmissionEngine: Emissions data submission, validation,
    review workflow, amendment versioning, CN code checks.
  - SupplierDashboardService: Dashboard aggregation, quality scores,
    deadline tracking, emissions trends, importer summaries.
  - DataExchangeService: Importer-supplier data exchange, access requests,
    authorization management, bulk export, audit logging.

Thread Safety:
  All singleton engines use threading.RLock for safe concurrent access.
  Obtain a service instance via get_supplier_portal_service().

Usage:
    >>> from supplier_portal import get_supplier_portal_service
    >>> service = get_supplier_portal_service()
    >>> profile = service.registry.get_supplier("SUP-ABC123")

Version: 1.1.0
Author: GreenLang CBAM Team
"""

import logging
import threading
from typing import Optional

from supplier_portal.models import (
    # Enums
    SupplierStatus,
    VerificationStatus,
    InstallationType,
    CBAMSector,
    SubmissionStatus,
    EvidenceType,
    CalculationMethod,
    ExportFormat,
    AccessRequestStatus,
    # Constants
    CN_CODE_SECTORS,
    CBAM_PRODUCT_GROUPS,
    EORI_PATTERN,
    DEFAULT_MARKUP_SCHEDULE,
    MATERIALITY_THRESHOLD,
    # Models
    SupplierProfile,
    Installation,
    EmissionsDataSubmission,
    PrecursorEmission,
    VerificationRecord,
    SupportingDocument,
    DataQualityScore,
    SupplierSearchResult,
    ValidationIssue,
    AccessRequest,
    AccessEvent,
    Deadline,
    SupplierDashboard,
)
from supplier_portal.supplier_registry import SupplierRegistryEngine
from supplier_portal.emissions_submission import EmissionsSubmissionEngine
from supplier_portal.supplier_dashboard import SupplierDashboardService
from supplier_portal.data_exchange import DataExchangeService

logger = logging.getLogger(__name__)

__version__ = "1.1.0"
__author__ = "GreenLang CBAM Team"

__all__ = [
    # Version
    "__version__",
    # Enums
    "SupplierStatus",
    "VerificationStatus",
    "InstallationType",
    "CBAMSector",
    "SubmissionStatus",
    "EvidenceType",
    "CalculationMethod",
    "ExportFormat",
    "AccessRequestStatus",
    # Constants
    "CN_CODE_SECTORS",
    "CBAM_PRODUCT_GROUPS",
    "EORI_PATTERN",
    "DEFAULT_MARKUP_SCHEDULE",
    "MATERIALITY_THRESHOLD",
    # Models
    "SupplierProfile",
    "Installation",
    "EmissionsDataSubmission",
    "PrecursorEmission",
    "VerificationRecord",
    "SupportingDocument",
    "DataQualityScore",
    "SupplierSearchResult",
    "ValidationIssue",
    "AccessRequest",
    "AccessEvent",
    "Deadline",
    "SupplierDashboard",
    # Engines / Services
    "SupplierRegistryEngine",
    "EmissionsSubmissionEngine",
    "SupplierDashboardService",
    "DataExchangeService",
    # Facade
    "SupplierPortalService",
    "get_supplier_portal_service",
]


class SupplierPortalService:
    """
    Thread-safe facade providing unified access to all Supplier Portal engines.

    This is the primary entry point for interacting with the CBAM Supplier Portal.
    Exposes the four sub-services as attributes: registry, submissions,
    dashboard, and data_exchange.

    Attributes:
        registry: SupplierRegistryEngine for supplier/installation management.
        submissions: EmissionsSubmissionEngine for emissions data workflow.
        dashboard: SupplierDashboardService for analytics and deadlines.
        data_exchange: DataExchangeService for importer-supplier data sharing.
    """

    def __init__(self) -> None:
        """Initialize SupplierPortalService with all sub-engines."""
        self.registry = SupplierRegistryEngine()
        self.submissions = EmissionsSubmissionEngine(registry=self.registry)
        self.dashboard = SupplierDashboardService(
            registry=self.registry,
            submissions=self.submissions,
        )
        self.data_exchange = DataExchangeService(
            registry=self.registry,
            submissions=self.submissions,
        )
        logger.info("SupplierPortalService initialized (v%s)", __version__)


# ---------------------------------------------------------------------------
# Thread-safe singleton
# ---------------------------------------------------------------------------
_instance: Optional[SupplierPortalService] = None
_instance_lock = threading.Lock()


def get_supplier_portal_service() -> SupplierPortalService:
    """
    Return the singleton SupplierPortalService instance.

    Thread-safe: uses double-checked locking to ensure exactly one instance
    is created even under concurrent access.

    Returns:
        The global SupplierPortalService singleton.
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = SupplierPortalService()
                logger.info("SupplierPortalService singleton created")
    return _instance
