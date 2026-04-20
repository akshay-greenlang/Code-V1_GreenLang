# -*- coding: utf-8 -*-
"""GreenLang Comply — v3 L4 Compliance Cloud umbrella product.

Comply is the FY27 commercial wedge that bundles CSRD / ESRS, CBAM, SB 253,
TCFD, SBTi, ISO 14064, CDP, and EU Taxonomy on top of the substrate
(Factors + Climate Ledger + Evidence Vault + Entity Graph + Policy Graph +
Scope Engine).

Quick start::

    from greenlang.comply import ComplyOrchestrator, ComplianceRunRequest

    orchestrator = ComplyOrchestrator()
    result = orchestrator.run(request)

    for fr in result.framework_results:
        print(fr.regulation, fr.total_co2e_kg, fr.evidence_count)
"""
from __future__ import annotations

from greenlang.comply.models import (
    ComplianceRunRequest,
    ComplianceRunResult,
    EntityProfile,
    FrameworkResult,
)
from greenlang.comply.orchestrator import ComplyOrchestrator

__version__ = "0.1.0"

__all__ = [
    "ComplyOrchestrator",
    "ComplianceRunRequest",
    "ComplianceRunResult",
    "EntityProfile",
    "FrameworkResult",
]
