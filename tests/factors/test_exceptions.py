# -*- coding: utf-8 -*-
"""Tests for greenlang.utilities.exceptions.factors."""

from __future__ import annotations

from greenlang.utilities.exceptions.base import GreenLangException
from greenlang.utilities.exceptions.factors import (
    FactorEditionError,
    FactorGovernanceError,
    FactorIngestionError,
    FactorLicenseError,
    FactorMatchingError,
    FactorValidationError,
    FactorsException,
    ParserError,
    QualityGateError,
    SourceRegistryError,
    WatchError,
)


def test_factors_exception_base():
    assert issubclass(FactorsException, GreenLangException)
    exc = FactorsException("test message")
    assert exc.message == "test message"


def test_error_prefix():
    exc = FactorsException("x")
    assert exc.ERROR_PREFIX == "GL_FACTORS"
    assert "GL_FACTORS" in exc.error_code


def test_context_propagation():
    ctx = {"key": "value", "count": 42}
    exc = FactorsException("ctx test", context=ctx)
    assert exc.context["key"] == "value"
    assert exc.context["count"] == 42


def test_to_dict_serialization():
    exc = FactorsException("ser test", agent_name="test_agent")
    d = exc.to_dict()
    assert d["error_type"] == "FactorsException"
    assert d["message"] == "ser test"
    assert d["agent_name"] == "test_agent"
    assert "error_code" in d
    assert "timestamp" in d


def test_specific_exceptions_inherit():
    classes = [
        FactorValidationError,
        FactorIngestionError,
        FactorGovernanceError,
        FactorEditionError,
        FactorLicenseError,
        FactorMatchingError,
        SourceRegistryError,
        ParserError,
        QualityGateError,
        WatchError,
    ]
    for cls in classes:
        assert issubclass(cls, FactorsException), f"{cls.__name__} does not inherit FactorsException"
        exc = cls("test")
        assert isinstance(exc, GreenLangException)


def test_factor_validation_error_details():
    exc = FactorValidationError(
        "schema check failed",
        factor_id="EF:US:diesel:2024:v1",
        gate="Q1_schema",
        violations=["vectors.CO2 is required", "factor_id prefix"],
    )
    assert exc.context["factor_id"] == "EF:US:diesel:2024:v1"
    assert exc.context["gate"] == "Q1_schema"
    assert len(exc.context["violations"]) == 2
