"""
GreenLang Provenance Framework
================================

Enterprise-grade provenance tracking for regulatory compliance and reproducibility.

This framework provides:
- Cryptographic verification and signing
- SBOM validation
- Supply chain attestations
- Comprehensive audit trails
- Data lineage tracking
- Integrity verification

Meets regulatory requirements (e.g., EU CBAM).
"""

# Supply chain security (existing)
from .signing import (
    verify_pack_signature,
    sign_pack,
    sign_artifact,
    verify_artifact,
    verify_pack,
)

# Provenance tracking (new framework modules)
from .hashing import hash_file, hash_data, MerkleTree
from .environment import (
    get_environment_info,
    get_dependency_versions,
    get_system_info
)
from .records import ProvenanceRecord, ProvenanceContext
from .validation import validate_provenance, verify_integrity
from .reporting import generate_audit_report, generate_markdown_report
from .decorators import traced, track_provenance
from .tracker import (
    ProvenanceTracker,
    track_with_provenance,
    get_global_tracker,
    reset_global_tracker
)

__all__ = [
    # Supply chain security (existing)
    "verify_pack_signature",
    "sign_pack",
    "sign_artifact",
    "verify_artifact",
    "verify_pack",

    # Hashing
    "hash_file",
    "hash_data",
    "MerkleTree",

    # Environment
    "get_environment_info",
    "get_dependency_versions",
    "get_system_info",

    # Records
    "ProvenanceRecord",
    "ProvenanceContext",

    # Validation
    "validate_provenance",
    "verify_integrity",

    # Reporting
    "generate_audit_report",
    "generate_markdown_report",

    # Decorators
    "traced",
    "track_provenance",

    # Tracker
    "ProvenanceTracker",
    "track_with_provenance",
    "get_global_tracker",
    "reset_global_tracker",
]

__version__ = "1.0.0"
