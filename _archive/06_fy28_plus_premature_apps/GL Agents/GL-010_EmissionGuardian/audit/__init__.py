# -*- coding: utf-8 -*-
"""GL-010 EmissionsGuardian - Audit and Lineage Module."""

# Import chain_of_custody module directly (schemas.py has syntax issues)
from .chain_of_custody import (
    RegulatoryFramework,
    CustodyAction,
    CustodianType,
    EmissionDataType,
    ChainIntegrity,
    CustodyEntry,
    CustodyTransfer,
    DataLineageNode,
    ChainVerificationResult,
    ImmutableAuditLog,
    EPACompliancePackager,
    EUETSCompliancePackager,
    create_audit_log,
    create_epa_packager,
    create_eu_ets_packager,
    compute_sha256,
    generate_entry_id,
    get_utc_now,
)

__all__ = [
    "RegulatoryFramework",
    "CustodyAction",
    "CustodianType",
    "EmissionDataType",
    "ChainIntegrity",
    "CustodyEntry",
    "CustodyTransfer",
    "DataLineageNode",
    "ChainVerificationResult",
    "ImmutableAuditLog",
    "EPACompliancePackager",
    "EUETSCompliancePackager",
    "create_audit_log",
    "create_epa_packager",
    "create_eu_ets_packager",
    "compute_sha256",
    "generate_entry_id",
    "get_utc_now",
]

__version__ = "1.0.0"
__author__ = "GreenLang GL-010 EmissionsGuardian"
