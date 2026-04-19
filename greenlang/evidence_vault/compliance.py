# -*- coding: utf-8 -*-
"""
Evidence Vault - Compliance facade.

Thin re-export layer that surfaces regulatory compliance classes from
the IED compliance module under the Evidence Vault product namespace.
No logic is duplicated here.

The upstream ``greenlang.governance.compliance`` package ``__init__.py``
contains a legacy alias import that may not resolve in all environments,
so this facade loads the source module directly via ``importlib`` to
avoid triggering that package init.

Example::

    from greenlang.evidence_vault.compliance import IEDComplianceManager
    manager = IEDComplianceManager(installation_id="INST-001")
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Direct-load the IED compliance module to bypass the broken
# greenlang.governance.compliance.__init__.py alias chain.
# ---------------------------------------------------------------------------

_IED_MODULE_NAME = "greenlang.governance.compliance.eu.ied_compliance"

if _IED_MODULE_NAME not in sys.modules:
    _ied_path = (
        Path(__file__).resolve().parent.parent
        / "governance" / "compliance" / "eu" / "ied_compliance.py"
    )
    _spec = importlib.util.spec_from_file_location(_IED_MODULE_NAME, _ied_path)
    if _spec is not None and _spec.loader is not None:
        _mod = importlib.util.module_from_spec(_spec)
        sys.modules[_IED_MODULE_NAME] = _mod
        _spec.loader.exec_module(_mod)
    else:
        logger.warning(
            "Could not load IED compliance module from %s", _ied_path
        )

_ied = sys.modules.get(_IED_MODULE_NAME)

if _ied is not None:
    IEDComplianceManager = _ied.IEDComplianceManager
    ComplianceStatus = _ied.ComplianceStatus
    BATAEL = _ied.BATAEL
    ComplianceAssessment = _ied.ComplianceAssessment
else:
    # Provide stub sentinels so that the rest of the vault can still load.
    IEDComplianceManager = None  # type: ignore[assignment,misc]
    ComplianceStatus = None  # type: ignore[assignment,misc]
    BATAEL = None  # type: ignore[assignment,misc]
    ComplianceAssessment = None  # type: ignore[assignment,misc]
    logger.warning(
        "IED compliance classes unavailable; "
        "greenlang.governance.compliance.eu.ied_compliance could not be loaded"
    )

__all__ = [
    "IEDComplianceManager",
    "ComplianceStatus",
    "BATAEL",
    "ComplianceAssessment",
]
