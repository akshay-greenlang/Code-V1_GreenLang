# -*- coding: utf-8 -*-
"""DEPRECATED canonical_v2 compat shim.

The v2 canonical module (``greenlang/data/canonical_v2.py``) was an interim
Phase F1 extension layer. It is superseded by :mod:`canonical_v1`
(in this package), which is the frozen v1 of the factor record.

This module re-exports the old symbols with deprecation warnings so
downstream consumers can migrate incrementally. Every public call into
this module emits a :class:`DeprecationWarning` *once per process*.

Migration cheat-sheet::

    # Old
    from greenlang.data.canonical_v2 import MethodProfile, RedistributionClass
    # New
    from greenlang.factors.data.canonical_v1 import (
        METHOD_PROFILE_ENUM, REDISTRIBUTION_CLASS_ENUM,
    )

    # Old
    from greenlang.data.canonical_v2 import Jurisdiction, Explainability
    # New
    from greenlang.factors.data.canonical_v1 import JurisdictionV1, ExplainabilityV1
"""
from __future__ import annotations

import warnings
from typing import Any, Dict

from greenlang.factors.data import canonical_v1
from greenlang.factors.data.canonical_v1 import (
    ActivitySchemaV1,
    DEFAULT_GWP_SET,
    ExplainabilityV1,
    FACTOR_FAMILY_ENUM,
    FactorRecordV1,
    FORMULA_TYPE_ENUM,
    GWP_SET_ENUM,
    JurisdictionV1,
    LicensingV1,
    LineageV1,
    METHOD_PROFILE_ENUM,
    NumeratorV1,
    QualityV1,
    REDISTRIBUTION_CLASS_ENUM,
    RawRecordRefV1,
    STATUS_ENUM,
    from_legacy_dict,
    migrate_record,
    to_legacy_dict,
)

_WARNED: set[str] = set()


def _warn(symbol: str) -> None:
    if symbol in _WARNED:
        return
    _WARNED.add(symbol)
    warnings.warn(
        f"greenlang.factors.data.canonical_v2.{symbol} is deprecated; "
        "use greenlang.factors.data.canonical_v1 instead "
        "(see docs/specs/schema_v1_gap_report.md).",
        DeprecationWarning,
        stacklevel=2,
    )


# Re-export existing enum classes (legacy) with a deprecation shim.
def _legacy_enums() -> Dict[str, Any]:
    """Lazily import the legacy enums from greenlang.data.canonical_v2."""
    try:
        from greenlang.data import canonical_v2 as _legacy
    except ImportError:  # pragma: no cover
        return {}
    return {
        "FactorFamily": getattr(_legacy, "FactorFamily", None),
        "FormulaType": getattr(_legacy, "FormulaType", None),
        "SourceType": getattr(_legacy, "SourceType", None),
        "RedistributionClass": getattr(_legacy, "RedistributionClass", None),
        "VerificationStatus": getattr(_legacy, "VerificationStatus", None),
        "PrimaryDataFlag": getattr(_legacy, "PrimaryDataFlag", None),
        "UncertaintyDistribution": getattr(_legacy, "UncertaintyDistribution", None),
        "ElectricityBasis": getattr(_legacy, "ElectricityBasis", None),
        "MethodProfile": getattr(_legacy, "MethodProfile", None),
    }


def __getattr__(name: str) -> Any:
    """Lazy re-export from legacy + v1 modules with a single deprecation warn."""
    # v1 symbols (preferred passthrough, still emit warning to steer callers)
    v1_aliases = {
        "Jurisdiction": JurisdictionV1,
        "ActivitySchema": ActivitySchemaV1,
        "Explainability": ExplainabilityV1,
        "RawRecordRef": RawRecordRefV1,
    }
    if name in v1_aliases:
        _warn(name)
        return v1_aliases[name]

    legacy = _legacy_enums()
    if name in legacy and legacy[name] is not None:
        _warn(name)
        return legacy[name]

    raise AttributeError(f"module 'canonical_v2' has no attribute {name!r}")


def map_legacy_to_v1(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Map an old-shape legacy dict to a v1-shape dict.

    Thin wrapper around :func:`canonical_v1.migrate_record` kept here so
    callers using the old ``canonical_v2.map_*`` import path keep working.
    """
    _warn("map_legacy_to_v1")
    return migrate_record(payload)


__all__ = [
    # Re-exported v1 symbols (preferred)
    "FactorRecordV1",
    "JurisdictionV1",
    "ActivitySchemaV1",
    "NumeratorV1",
    "QualityV1",
    "LineageV1",
    "LicensingV1",
    "ExplainabilityV1",
    "RawRecordRefV1",
    "STATUS_ENUM",
    "REDISTRIBUTION_CLASS_ENUM",
    "GWP_SET_ENUM",
    "DEFAULT_GWP_SET",
    "METHOD_PROFILE_ENUM",
    "FACTOR_FAMILY_ENUM",
    "FORMULA_TYPE_ENUM",
    "from_legacy_dict",
    "to_legacy_dict",
    "migrate_record",
    "map_legacy_to_v1",
]
