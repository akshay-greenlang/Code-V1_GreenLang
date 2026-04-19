# -*- coding: utf-8 -*-
"""
Evidence Vault - Reporting facade.

Thin re-export layer that surfaces provenance reporting and validation
utilities from ``greenlang.utilities.provenance`` under the Evidence Vault
product namespace.  No logic is duplicated here.

Example::

    from greenlang.evidence_vault.reporting import generate_audit_report
    report = generate_audit_report(provenance_record, format="markdown")
"""
from __future__ import annotations

from greenlang.utilities.provenance.reporting import (
    generate_audit_report,
    generate_markdown_report,
)
from greenlang.utilities.provenance.validation import (
    validate_provenance,
    verify_integrity,
)

__all__ = [
    "generate_audit_report",
    "generate_markdown_report",
    "validate_provenance",
    "verify_integrity",
]
