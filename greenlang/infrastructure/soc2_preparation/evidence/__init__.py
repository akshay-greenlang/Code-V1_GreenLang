# -*- coding: utf-8 -*-
"""
Evidence Management Module - SEC-009 Phase 3

This module provides comprehensive evidence collection, packaging, versioning,
validation, and sampling capabilities for SOC 2 Type II audit preparation.

Components:
    - EvidenceCollector: Multi-source evidence collection with adapters
    - EvidencePackager: Create structured audit packages for auditors
    - EvidenceVersioner: Version control and history tracking for evidence
    - EvidenceValidator: Integrity, completeness, freshness validation
    - PopulationSampler: AICPA-compliant audit sampling

The evidence management system integrates with:
    - AWS CloudTrail for infrastructure audit logs
    - GitHub API for code changes and pull requests
    - PostgreSQL for security event data
    - Loki for log aggregation queries
    - Auth service for authentication events
    - Jira for change management tickets
    - Okta for SSO/identity events

Example:
    >>> from greenlang.infrastructure.soc2_preparation.evidence import (
    ...     EvidenceCollector,
    ...     EvidencePackager,
    ...     EvidenceValidator,
    ...     PopulationSampler,
    ... )
    >>> collector = EvidenceCollector(config)
    >>> evidence = await collector.collect_for_criterion("CC6.1", date_range)
    >>> packager = EvidencePackager(config)
    >>> package = await packager.create_package(["CC6.1", "CC6.2"], date_range)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from greenlang.infrastructure.soc2_preparation.evidence.collector import (
    EvidenceCollector,
)
from greenlang.infrastructure.soc2_preparation.evidence.packager import (
    EvidencePackager,
)
from greenlang.infrastructure.soc2_preparation.evidence.versioner import (
    EvidenceVersioner,
)
from greenlang.infrastructure.soc2_preparation.evidence.validator import (
    EvidenceValidator,
)
from greenlang.infrastructure.soc2_preparation.evidence.sampler import (
    PopulationSampler,
)

__all__ = [
    "EvidenceCollector",
    "EvidencePackager",
    "EvidenceVersioner",
    "EvidenceValidator",
    "PopulationSampler",
]
