# -*- coding: utf-8 -*-
"""GreenLang Factors typed schema mirrors.

This package holds the Pydantic v2 mirrors of the FROZEN JSON schemas that
live under ``config/schemas/``. The JSON schemas remain the source of
truth (CTO Phase 2 §2.1); the Pydantic mirrors give Python callers
IDE/mypy support and per-field validators that the JSON Schema cannot
fully express (cross-field rules, URN parsing, etc.).

Modules
-------
factor_record_v0_1
    Mirror of ``config/schemas/factor_record_v0_1.schema.json`` (FROZEN
    2026-04-25). Composed of nine field-group submodels that together
    flatten into the top-level :class:`FactorRecordV0_1` record.
"""

from greenlang.factors.schemas.factor_record_v0_1 import (
    ClimateBasisFields,
    ContextFields,
    Citation,
    ExtractionMetadata,
    FactorRecordV0_1,
    FrozenSchemaPath,
    IdentityFields,
    LicenceConstraints,
    LicenceFields,
    LifecycleFields,
    LineageFields,
    QualityFields,
    ReviewMetadata,
    TimeFields,
    UncertaintyDistribution,
    ValueUnitFields,
    model_to_jsonschema_diff,
)

__all__ = [
    "ClimateBasisFields",
    "ContextFields",
    "Citation",
    "ExtractionMetadata",
    "FactorRecordV0_1",
    "FrozenSchemaPath",
    "IdentityFields",
    "LicenceConstraints",
    "LicenceFields",
    "LifecycleFields",
    "LineageFields",
    "QualityFields",
    "ReviewMetadata",
    "TimeFields",
    "UncertaintyDistribution",
    "ValueUnitFields",
    "model_to_jsonschema_diff",
]
