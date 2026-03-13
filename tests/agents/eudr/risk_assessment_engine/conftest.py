# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-028 Risk Assessment Engine test suite.

Provides common configuration, model instances, and helper objects
used across all test modules.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from greenlang.agents.eudr.risk_assessment_engine.config import (
    RiskAssessmentEngineConfig,
    reset_config,
)
from greenlang.agents.eudr.risk_assessment_engine.models import (
    Article10CriteriaResult,
    Article10Criterion,
    Article10CriterionEvaluation,
    CompositeRiskScore,
    CountryBenchmark,
    CountryBenchmarkLevel,
    CriterionResult,
    DimensionScore,
    EUDRCommodity,
    OverrideReason,
    RiskAssessmentOperation,
    RiskAssessmentStatus,
    RiskDimension,
    RiskFactorInput,
    RiskLevel,
    RiskOverride,
    RiskTrendPoint,
    SimplifiedDDEligibility,
    SourceAgent,
    TrendDirection,
)
from greenlang.agents.eudr.risk_assessment_engine.provenance import ProvenanceTracker


# ---------------------------------------------------------------------------
# Config fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def config() -> RiskAssessmentEngineConfig:
    """Create a RiskAssessmentEngineConfig with test defaults."""
    reset_config()
    cfg = RiskAssessmentEngineConfig()
    yield cfg
    reset_config()


# ---------------------------------------------------------------------------
# Provenance fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def tracker() -> ProvenanceTracker:
    """Return a fresh ProvenanceTracker instance."""
    return ProvenanceTracker()


# ---------------------------------------------------------------------------
# Model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_factor_inputs() -> List[RiskFactorInput]:
    """Create 8 RiskFactorInput objects covering all dimensions."""
    now = datetime.now(timezone.utc)
    inputs = [
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_016_COUNTRY,
            dimension=RiskDimension.COUNTRY,
            raw_score=Decimal("45"),
            confidence=Decimal("0.90"),
            metadata={"country_code": "BR"},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_018_COMMODITY,
            dimension=RiskDimension.COMMODITY,
            raw_score=Decimal("55"),
            confidence=Decimal("0.85"),
            metadata={"commodity": "cocoa"},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_017_SUPPLIER,
            dimension=RiskDimension.SUPPLIER,
            raw_score=Decimal("40"),
            confidence=Decimal("0.80"),
            metadata={"supplier_id": "SUP-001"},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_020_DEFORESTATION,
            dimension=RiskDimension.DEFORESTATION,
            raw_score=Decimal("60"),
            confidence=Decimal("0.82"),
            metadata={"country_code": "BR"},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_019_CORRUPTION,
            dimension=RiskDimension.CORRUPTION,
            raw_score=Decimal("62"),
            confidence=Decimal("0.88"),
            metadata={"country_code": "BR"},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_016_COUNTRY,
            dimension=RiskDimension.SUPPLY_CHAIN_COMPLEXITY,
            raw_score=Decimal("35"),
            confidence=Decimal("0.70"),
            metadata={"derived": True},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_016_COUNTRY,
            dimension=RiskDimension.MIXING_RISK,
            raw_score=Decimal("30"),
            confidence=Decimal("0.65"),
            metadata={"derived": True},
            timestamp=now,
        ),
        RiskFactorInput(
            source_agent=SourceAgent.EUDR_016_COUNTRY,
            dimension=RiskDimension.CIRCUMVENTION_RISK,
            raw_score=Decimal("25"),
            confidence=Decimal("0.60"),
            metadata={"derived": True},
            timestamp=now,
        ),
    ]
    return inputs


@pytest.fixture()
def sample_country_benchmarks() -> List[CountryBenchmark]:
    """Create CountryBenchmark objects with a mix of levels."""
    now = datetime.now(timezone.utc)
    return [
        CountryBenchmark(
            country_code="DE",
            benchmark_level=CountryBenchmarkLevel.LOW,
            effective_date=now,
            source="ec_default_2026",
        ),
        CountryBenchmark(
            country_code="BR",
            benchmark_level=CountryBenchmarkLevel.HIGH,
            effective_date=now,
            source="ec_default_2026",
        ),
        CountryBenchmark(
            country_code="IN",
            benchmark_level=CountryBenchmarkLevel.STANDARD,
            effective_date=now,
            source="ec_default_2026",
        ),
    ]


@pytest.fixture()
def sample_composite_score() -> CompositeRiskScore:
    """Create a sample CompositeRiskScore."""
    return CompositeRiskScore(
        overall_score=Decimal("48.50"),
        risk_level=RiskLevel.STANDARD,
        dimension_scores=[],
        total_weight=Decimal("1.00"),
        effective_confidence=Decimal("0.80"),
        country_benchmark_applied=False,
        benchmark_multiplier=None,
        provenance_hash="a" * 64,
    )


@pytest.fixture()
def sample_article10_result() -> Article10CriteriaResult:
    """Create a sample Article10CriteriaResult."""
    return Article10CriteriaResult(
        evaluations=[
            Article10CriterionEvaluation(
                criterion=Article10Criterion.PREVALENCE_OF_DEFORESTATION,
                result=CriterionResult.PASS,
                score=Decimal("30"),
            ),
            Article10CriterionEvaluation(
                criterion=Article10Criterion.SUPPLY_CHAIN_COMPLEXITY,
                result=CriterionResult.PASS,
                score=Decimal("25"),
            ),
        ],
        overall_concern_count=0,
        criteria_evaluated=2,
        criteria_passed=2,
        criteria_with_concerns=0,
    )


@pytest.fixture()
def sample_risk_assessment_operation() -> RiskAssessmentOperation:
    """Create a sample RiskAssessmentOperation."""
    return RiskAssessmentOperation(
        operation_id=f"OP-{uuid.uuid4().hex[:8]}",
        operator_id="OPERATOR-001",
        commodity=EUDRCommodity.COCOA,
        status=RiskAssessmentStatus.INITIATED,
    )
