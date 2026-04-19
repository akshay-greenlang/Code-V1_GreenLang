# -*- coding: utf-8 -*-
"""
Compliance facade - v3 Policy Graph compliance engine registry
==============================================================

Re-exports key classes from :mod:`greenlang.governance.compliance` and provides
a thin :class:`ComplianceRegistry` for discovering available compliance engines.

Usage::

    from greenlang.policy_graph.compliance import ComplianceRegistry

    registry = ComplianceRegistry()
    for engine in registry.list_engines():
        print(engine["name"], engine["jurisdiction"], engine["regulation"])

    ied_cls = registry.get_engine("ied")
    manager = ied_cls(installation_id="INST-001")
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Type

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Re-exports from governance.compliance.eu
# ---------------------------------------------------------------------------
from greenlang.governance.compliance.eu.ied_compliance import (  # noqa: E402
    IEDComplianceManager,
    ComplianceStatus,
    BATAEL,
    ComplianceAssessment,
    IEDAnnexIActivity,
    MonitoringFrequency,
    PollutantCategory,
    BATConclusion,
    EmissionLimitValue,
    EmissionMeasurement,
    PermitCondition,
    DerogationRequest,
    AnnualReport,
)

# ---------------------------------------------------------------------------
# Re-exports from governance.compliance.epa
# ---------------------------------------------------------------------------
from greenlang.governance.compliance.epa.part60_nsps import (  # noqa: E402
    NSPSComplianceChecker,
    FFactorCalculator,
)
from greenlang.governance.compliance.epa.part98_ghg import (  # noqa: E402
    Part98Reporter,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Internal catalogue of known engines.  Each entry maps a short name to the
# concrete class, jurisdiction, and regulation string.
_ENGINE_CATALOGUE: Dict[str, Dict[str, Any]] = {
    "ied": {
        "cls": IEDComplianceManager,
        "jurisdiction": "EU",
        "regulation": "Industrial Emissions Directive 2010/75/EU",
    },
    "nsps": {
        "cls": NSPSComplianceChecker,
        "jurisdiction": "US",
        "regulation": "EPA 40 CFR Part 60 NSPS",
    },
    "part98": {
        "cls": Part98Reporter,
        "jurisdiction": "US",
        "regulation": "EPA 40 CFR Part 98 GHG Reporting",
    },
    "ffactor": {
        "cls": FFactorCalculator,
        "jurisdiction": "US",
        "regulation": "EPA Method 19 F-Factor Calculations",
    },
}


class ComplianceRegistry:
    """
    Registry of available compliance engines in the v3 Policy Graph.

    Provides a discovery mechanism so that callers can list and retrieve
    compliance engine classes without importing each one directly.

    Example::

        registry = ComplianceRegistry()
        engines = registry.list_engines()
        # [{"name": "ied", "jurisdiction": "EU", "regulation": "..."}, ...]

        ied_cls = registry.get_engine("ied")
        manager = ied_cls(installation_id="INST-001")
    """

    def list_engines(self) -> List[Dict[str, str]]:
        """
        List all registered compliance engines.

        Returns:
            A list of dicts with keys ``name``, ``jurisdiction``, and
            ``regulation``.
        """
        return [
            {
                "name": name,
                "jurisdiction": entry["jurisdiction"],
                "regulation": entry["regulation"],
            }
            for name, entry in _ENGINE_CATALOGUE.items()
        ]

    def get_engine(self, name: str) -> Type[Any]:
        """
        Retrieve the compliance engine class by short name.

        Args:
            name: Engine short name (e.g. ``"ied"``, ``"nsps"``,
                ``"part98"``).

        Returns:
            The engine class (not an instance).

        Raises:
            KeyError: If *name* is not registered.
        """
        entry = _ENGINE_CATALOGUE.get(name)
        if entry is None:
            available = ", ".join(sorted(_ENGINE_CATALOGUE))
            raise KeyError(
                f"Unknown compliance engine {name!r}. "
                f"Available: {available}"
            )
        return entry["cls"]


# ---------------------------------------------------------------------------
# Module exports
# ---------------------------------------------------------------------------

__all__ = [
    # Registry
    "ComplianceRegistry",
    # EU IED re-exports
    "IEDComplianceManager",
    "ComplianceStatus",
    "BATAEL",
    "ComplianceAssessment",
    "IEDAnnexIActivity",
    "MonitoringFrequency",
    "PollutantCategory",
    "BATConclusion",
    "EmissionLimitValue",
    "EmissionMeasurement",
    "PermitCondition",
    "DerogationRequest",
    "AnnualReport",
    # EPA re-exports
    "NSPSComplianceChecker",
    "FFactorCalculator",
    "Part98Reporter",
]
