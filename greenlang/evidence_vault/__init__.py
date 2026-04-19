# -*- coding: utf-8 -*-
"""
GreenLang Evidence Vault - v3 Product Module
==============================================

The Evidence Vault is GreenLang's L2 "System of Record" layer for regulatory
evidence.  It provides a unified API for collecting, packaging, verifying, and
exporting audit evidence required by frameworks such as CSRD, CBAM, EUDR,
ISO 14064, and CDP.

This is a **thin facade** that wraps existing infrastructure modules:

* ``greenlang.utilities.provenance`` -- hashing, records, validation, reporting
* ``greenlang.governance.compliance`` -- IED / EPA compliance checking
* ``greenlang.governance.policy`` -- OPA policy evaluation
* ``greenlang.utilities.provenance.signing`` -- artifact signing & verification

Quick-start::

    from greenlang.evidence_vault import EvidenceVault

    vault = EvidenceVault(vault_id="csrd-fy25")
    eid = vault.collect(
        evidence_type="emission_factor",
        source="scope1_agent",
        data={"co2e_tonnes": 1234.56},
    )
    ok, info = vault.verify(eid)
    package = vault.export()
"""
from __future__ import annotations

# -- Core vault class --------------------------------------------------------
from greenlang.evidence_vault.vault import EvidenceVault

# -- Compliance re-exports ---------------------------------------------------
from greenlang.evidence_vault.compliance import (
    IEDComplianceManager,
    ComplianceStatus,
    BATAEL,
    ComplianceAssessment,
)

# -- Reporting re-exports ----------------------------------------------------
from greenlang.evidence_vault.reporting import (
    generate_audit_report,
    generate_markdown_report,
    validate_provenance,
    verify_integrity,
)

__version__ = "0.1.0"

__all__ = [
    # Vault
    "EvidenceVault",
    # Compliance
    "IEDComplianceManager",
    "ComplianceStatus",
    "BATAEL",
    "ComplianceAssessment",
    # Reporting / Provenance
    "generate_audit_report",
    "generate_markdown_report",
    "validate_provenance",
    "verify_integrity",
]
