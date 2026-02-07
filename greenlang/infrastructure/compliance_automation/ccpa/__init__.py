# -*- coding: utf-8 -*-
"""
CCPA/LGPD Compliance Module - SEC-010 Phase 5

Implementation of California Consumer Privacy Act (CCPA) and Brazil's Lei
Geral de Protecao de Dados (LGPD) compliance automation. Provides consumer
rights processing including right to know, right to delete, and opt-out
of sale.

CCPA Rights Covered:
- Right to Know (disclosure requests)
- Right to Delete
- Right to Opt-Out of Sale
- Right to Non-Discrimination

LGPD Rights Covered:
- Right of Confirmation
- Right of Access
- Right of Correction
- Right of Anonymization, Blocking, or Elimination
- Right of Portability
- Right to Information about Sharing

Public API:
    - ConsumerRightsProcessor: Process consumer rights requests.

Example:
    >>> from greenlang.infrastructure.compliance_automation.ccpa import (
    ...     ConsumerRightsProcessor,
    ... )
    >>> processor = ConsumerRightsProcessor()
    >>> result = await processor.process_access_request(request)

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from greenlang.infrastructure.compliance_automation.ccpa.consumer_rights import (
    ConsumerRightsProcessor,
)

__all__ = [
    "ConsumerRightsProcessor",
]
