"""
CBAM Importer Copilot - Provenance & Observability

Enterprise-grade provenance tracking for regulatory compliance.

Provides:
- SHA256 file hashing for input integrity
- Execution environment capture
- Dependency version tracking
- Audit trail generation
- Provenance validation

Version: 1.0.0
Author: GreenLang CBAM Team
"""

from .provenance_utils import (
    hash_file,
    get_environment_info,
    get_dependency_versions,
    create_provenance_record,
    validate_provenance,
    ProvenanceRecord
)

__version__ = "1.0.0"
__all__ = [
    "hash_file",
    "get_environment_info",
    "get_dependency_versions",
    "create_provenance_record",
    "validate_provenance",
    "ProvenanceRecord"
]
