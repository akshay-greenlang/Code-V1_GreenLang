# -*- coding: utf-8 -*-
"""
Multi-Compliance Automation Module - SEC-010 Phase 5

Production-grade compliance automation for the GreenLang Climate OS platform.
Provides continuous compliance monitoring, automated evidence collection,
DSAR processing, and multi-framework support for ISO 27001, GDPR, PCI-DSS,
CCPA, and LGPD.

Public API:
    - ComplianceConfig: Configuration settings for compliance automation.
    - ComplianceFramework: Enum of supported compliance frameworks.
    - ComplianceStatus: Current compliance status for a framework.
    - ControlMapping: Mapping between framework controls and technical controls.
    - DSARRequest: Data Subject Access Request model.
    - DSARType: Enum of DSAR request types (Art. 15-22).
    - DSARStatus: Enum of DSAR processing statuses.
    - ConsentRecord: User consent tracking record.
    - DataRecord: Discovered user data record.
    - BaseComplianceFramework: Abstract base class for framework implementations.
    - ISO27001Mapper: ISO 27001:2022 control mapping and assessment.
    - DSARProcessor: GDPR DSAR request processing.
    - DataDiscovery: PII/data discovery across systems.
    - RetentionEnforcer: Data retention policy enforcement.
    - ConsentManager: Consent tracking and management.
    - CardDataMapper: PCI-DSS cardholder data flow mapping.
    - EncryptionChecker: PCI-DSS encryption verification.
    - ConsumerRightsProcessor: CCPA/LGPD consumer rights processing.
    - ComplianceMetrics: Prometheus metrics for compliance monitoring.
    - get_compliance_metrics: Factory function for metrics singleton.

Example:
    >>> from greenlang.infrastructure.compliance_automation import (
    ...     ComplianceFramework, DSARProcessor, ISO27001Mapper,
    ...     get_compliance_metrics,
    ... )
    >>> # Check GDPR DSAR processing
    >>> processor = DSARProcessor()
    >>> request = await processor.submit_request(dsar_data)
    >>>
    >>> # Check ISO 27001 compliance
    >>> mapper = ISO27001Mapper()
    >>> status = await mapper.assess_compliance()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
Status: Production Ready
"""

from __future__ import annotations

import logging

# Configuration
from greenlang.infrastructure.compliance_automation.config import (
    ComplianceConfig,
    EnvironmentProfile as ComplianceEnvironmentProfile,
    get_config as get_compliance_config,
    reset_config as reset_compliance_config,
)

# Models
from greenlang.infrastructure.compliance_automation.models import (
    ComplianceFramework,
    ComplianceStatus,
    ControlMapping,
    ControlStatus,
    DataCategory,
    DataRecord,
    DSARRequest,
    DSARStatus,
    DSARType,
    ConsentRecord,
    ConsentPurpose,
    RetentionPolicy,
    EvidenceSource,
    ComplianceReport,
    ComplianceGap,
)

# Base Framework
from greenlang.infrastructure.compliance_automation.base_framework import (
    BaseComplianceFramework,
)

# ISO 27001
from greenlang.infrastructure.compliance_automation.iso27001 import (
    ISO27001Mapper,
    ISO27001Evidence,
    ISO27001Reporter,
)

# GDPR
from greenlang.infrastructure.compliance_automation.gdpr import (
    DSARProcessor,
    DataDiscovery,
    RetentionEnforcer,
    ConsentManager,
)

# PCI-DSS
from greenlang.infrastructure.compliance_automation.pci_dss import (
    CardDataMapper,
    EncryptionChecker,
)

# CCPA
from greenlang.infrastructure.compliance_automation.ccpa import (
    ConsumerRightsProcessor,
)

# Metrics
from greenlang.infrastructure.compliance_automation.metrics import (
    ComplianceMetrics,
    get_compliance_metrics,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration
    "ComplianceConfig",
    "ComplianceEnvironmentProfile",
    "get_compliance_config",
    "reset_compliance_config",
    # Models
    "ComplianceFramework",
    "ComplianceStatus",
    "ControlMapping",
    "ControlStatus",
    "DataCategory",
    "DataRecord",
    "DSARRequest",
    "DSARStatus",
    "DSARType",
    "ConsentRecord",
    "ConsentPurpose",
    "RetentionPolicy",
    "EvidenceSource",
    "ComplianceReport",
    "ComplianceGap",
    # Base Framework
    "BaseComplianceFramework",
    # ISO 27001
    "ISO27001Mapper",
    "ISO27001Evidence",
    "ISO27001Reporter",
    # GDPR
    "DSARProcessor",
    "DataDiscovery",
    "RetentionEnforcer",
    "ConsentManager",
    # PCI-DSS
    "CardDataMapper",
    "EncryptionChecker",
    # CCPA
    "ConsumerRightsProcessor",
    # Metrics
    "ComplianceMetrics",
    "get_compliance_metrics",
]

__version__ = "1.0.0"
