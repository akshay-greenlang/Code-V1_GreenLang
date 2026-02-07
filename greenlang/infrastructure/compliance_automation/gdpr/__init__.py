# -*- coding: utf-8 -*-
"""
GDPR Compliance Module - SEC-010 Phase 5

Implementation of EU General Data Protection Regulation (GDPR) compliance
automation. Provides DSAR (Data Subject Access Request) processing, PII
discovery, data retention enforcement, and consent management.

GDPR Articles Covered:
- Article 15: Right of Access
- Article 16: Right to Rectification
- Article 17: Right to Erasure (Right to be Forgotten)
- Article 18: Right to Restriction of Processing
- Article 20: Right to Data Portability
- Article 21: Right to Object

Public API:
    - DSARProcessor: Process Data Subject Access Requests.
    - DataDiscovery: Discover PII across systems.
    - RetentionEnforcer: Enforce data retention policies.
    - ConsentManager: Manage user consent records.

Example:
    >>> from greenlang.infrastructure.compliance_automation.gdpr import (
    ...     DSARProcessor, DataDiscovery, ConsentManager,
    ... )
    >>> processor = DSARProcessor()
    >>> request = await processor.submit_request(dsar_data)
    >>> data = await processor.discover_data(user_id)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from greenlang.infrastructure.compliance_automation.gdpr.dsar_processor import (
    DSARProcessor,
)
from greenlang.infrastructure.compliance_automation.gdpr.data_discovery import (
    DataDiscovery,
)
from greenlang.infrastructure.compliance_automation.gdpr.retention_enforcer import (
    RetentionEnforcer,
)
from greenlang.infrastructure.compliance_automation.gdpr.consent_manager import (
    ConsentManager,
)

__all__ = [
    "DSARProcessor",
    "DataDiscovery",
    "RetentionEnforcer",
    "ConsentManager",
]
