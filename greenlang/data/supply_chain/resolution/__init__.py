"""
Entity Resolution Module.

Provides fuzzy matching and entity resolution capabilities for supplier
master data management, supporting:
- Fuzzy string matching for company names
- LEI (Legal Entity Identifier) integration
- DUNS number matching
- Address standardization and matching
- Confidence scoring for match quality
"""

from greenlang.supply_chain.resolution.entity_resolver import (
    EntityResolver,
    MatchResult,
    MatchConfidence,
    MatchStrategy,
    AddressNormalizer,
)

__all__ = [
    "EntityResolver",
    "MatchResult",
    "MatchConfidence",
    "MatchStrategy",
    "AddressNormalizer",
]
