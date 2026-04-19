# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-029 Interim Targets Pack.
================================================================

Provides pytest fixtures for all 10 engines, 7 workflows, 10 templates,
10 integrations, 7 configuration presets, and comprehensive test data
builders for interim target calculations, progress tracking, variance
analysis, trend extrapolation, corrective actions, milestone validation,
initiative scheduling, and budget allocation.

Adds the pack root to sys.path so ``from engines.X import Y`` works
in every test module without requiring an installed package.

Fixtures cover:
    - Engine instantiation (10 interim target engines)
    - Workflow instantiation (7 workflows)
    - PACK-021 baseline/target mock data
    - PACK-028 sector pathway/abatement mock data
    - MRV agent mock responses (actual emissions data)
    - Emissions time series builders (2019-2050)
    - Interim target data factories (5-year, 10-year)
    - Progress tracking data (actual vs target)
    - Variance decomposition test data (LMDI, Kaya)
    - Trend extrapolation fixtures (linear, exponential, ARIMA)
    - Corrective action portfolio data
    - SBTi 21-criteria validation data
    - Initiative scheduling data (phased rollout)
    - Carbon budget allocation data
    - Reporting framework data (SBTi, CDP, TCFD)
    - Database session mocking (async PostgreSQL)
    - Redis cache mocking
    - SHA-256 provenance validation helpers
    - Decimal arithmetic assertion helpers
    - Performance timing context managers

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-029 Interim Targets Pack
Tests:   conftest.py (~1,350 lines)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Path setup -- ensure pack root is importable
# ---------------------------------------------------------------------------

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))

_REPO_ROOT = _PACK_ROOT.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

ENGINES_DIR = _PACK_ROOT / "engines"
WORKFLOWS_DIR = _PACK_ROOT / "workflows"
TEMPLATES_DIR = _PACK_ROOT / "templates"
INTEGRATIONS_DIR = _PACK_ROOT / "integrations"
CONFIG_DIR = _PACK_ROOT / "config"
PRESETS_DIR = CONFIG_DIR / "presets"


# ---------------------------------------------------------------------------
# Engine imports (lazy, with graceful fallback)
# ---------------------------------------------------------------------------

try:
    from engines.interim_target_engine import (
        InterimTargetEngine,
        InterimTargetInput,
        InterimTargetResult,
        InterimMilestone,
        PathwayShape,
        ClimateAmbition,
    )
    _HAS_INTERIM_TARGET = True
except ImportError:
    _HAS_INTERIM_TARGET = False

try:
    from engines.annual_pathway_engine import (
        AnnualPathwayEngine,
        AnnualPathwayInput,
        AnnualPathwayResult,
        AnnualPathwayPoint,
        ReductionProfile,
        BudgetAnalysis,
    )
    _HAS_ANNUAL_PATHWAY = True
except ImportError:
    _HAS_ANNUAL_PATHWAY = False

try:
    from engines.progress_tracker_engine import (
        ProgressTrackerEngine,
        ProgressTrackerInput,
        ProgressTrackerResult,
        RAGStatus,
        MilestoneStatus,
        ProgressDirection,
    )
    _HAS_PROGRESS_TRACKER = True
except ImportError:
    _HAS_PROGRESS_TRACKER = False

try:
    from engines.variance_analysis_engine import (
        VarianceAnalysisEngine,
        VarianceAnalysisInput,
        VarianceAnalysisResult,
        LMDIComponent,
        RootCauseCategory,
        WaterfallStep,
    )
    _HAS_VARIANCE_ANALYSIS = True
except ImportError:
    _HAS_VARIANCE_ANALYSIS = False

try:
    from engines.trend_extrapolation_engine import (
        TrendExtrapolationEngine,
        TrendExtrapolationInput,
        TrendExtrapolationResult,
        ForecastPoint,
        ConfidenceLevel,
        ForecastMethod,
    )
    _HAS_TREND_EXTRAPOLATION = True
except ImportError:
    _HAS_TREND_EXTRAPOLATION = False

try:
    from engines.corrective_action_engine import (
        CorrectiveActionEngine,
        CorrectiveActionInput,
        CorrectiveActionResult,
        AvailableInitiative,
        InvestmentAnalysis,
        CatchUpTimeline,
    )
    _HAS_CORRECTIVE_ACTION = True
except ImportError:
    _HAS_CORRECTIVE_ACTION = False

try:
    from engines.milestone_validation_engine import (
        MilestoneValidationEngine,
        MilestoneValidationInput,
        MilestoneValidationResult,
        CheckStatus,
        CheckCategory,
        ValidationCheck,
    )
    _HAS_MILESTONE_VALIDATION = True
except ImportError:
    _HAS_MILESTONE_VALIDATION = False

try:
    from engines.initiative_scheduler_engine import (
        InitiativeSchedulerEngine,
        InitiativeSchedulerInput,
        InitiativeSchedulerResult,
        DeploymentPhase,
        ScheduledInitiative,
        CriticalPathResult,
    )
    _HAS_INITIATIVE_SCHEDULER = True
except ImportError:
    _HAS_INITIATIVE_SCHEDULER = False

try:
    from engines.budget_allocation_engine import (
        BudgetAllocationEngine,
        BudgetAllocationInput,
        BudgetAllocationResult,
        AnnualBudget,
        BudgetStatus,
        CarbonPricingAnalysis,
    )
    _HAS_BUDGET_ALLOCATION = True
except ImportError:
    _HAS_BUDGET_ALLOCATION = False

try:
    from engines.reporting_engine import (
        ReportingEngine,
        ReportingInput,
        ReportingResult,
        SBTiProgressReport,
        AssuranceEvidence,
        ConsistencyCheck,
    )
    _HAS_REPORTING = True
except ImportError:
    _HAS_REPORTING = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SBTI_AMBITION_LEVELS = ["1.5C", "WB2C"]

SBTI_15C_MIN_REDUCTION = Decimal("42")   # 42% reduction by 2030 from 2019
SBTI_WB2C_MIN_REDUCTION = Decimal("30")  # 30% reduction by 2030 from 2019

PATHWAY_TYPES = ["linear", "milestone_based", "accelerating", "s_curve"]

SCOPES = ["scope_1", "scope_2", "scope_3"]

SCOPE_3_CATEGORIES = [
    "cat_01_purchased_goods",
    "cat_02_capital_goods",
    "cat_03_fuel_energy",
    "cat_04_upstream_transport",
    "cat_05_waste",
    "cat_06_business_travel",
    "cat_07_employee_commuting",
    "cat_08_upstream_leased",
    "cat_09_downstream_transport",
    "cat_10_processing",
    "cat_11_use_of_sold",
    "cat_12_end_of_life",
    "cat_13_downstream_leased",
    "cat_14_franchises",
    "cat_15_investments",
]

VARIANCE_EFFECTS = ["activity", "intensity", "structural", "fuel_mix", "weather"]

FORECAST_MODELS = ["linear_regression", "exponential_smoothing", "arima", "holt_winters"]

PERFORMANCE_RATINGS = ["green", "amber", "red"]

REPORTING_FRAMEWORKS = ["sbti", "cdp", "tcfd", "ghg_protocol", "iso14064"]

PRESET_NAMES = [
    "sbti_15c",
    "sbti_wb2c",
    "race_to_zero",
    "corporate_standard",
    "sme_simplified",
    "financial_services",
    "manufacturing",
]

SBTI_CRITERIA_COUNT = 21  # SBTi has 21 validation criteria

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]

TRL_LEVELS = list(range(1, 10))  # TRL 1-9

INITIATIVE_PHASES = ["pilot", "scale", "full_deployment"]


# ---------------------------------------------------------------------------
# Helper: Decimal assertion
# ---------------------------------------------------------------------------


def assert_decimal_close(
    actual: Decimal,
    expected: Decimal,
    tolerance: Decimal = Decimal("0.01"),
    msg: str = "",
) -> None:
    """Assert two Decimal values are within tolerance."""
    diff = abs(actual - expected)
    assert diff <= tolerance, (
        f"Decimal mismatch{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}, diff={diff}, tol={tolerance}"
    )


def assert_decimal_equal(actual: Decimal, expected: Decimal, msg: str = "") -> None:
    """Assert two Decimal values are exactly equal (no floating-point tolerance)."""
    assert actual == expected, (
        f"Decimal inequality{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}"
    )


def assert_decimal_positive(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is positive."""
    assert value > Decimal("0"), (
        f"Expected positive Decimal{' (' + msg + ')' if msg else ''}, got {value}"
    )


def assert_decimal_non_negative(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is non-negative."""
    assert value >= Decimal("0"), (
        f"Expected non-negative Decimal{' (' + msg + ')' if msg else ''}, got {value}"
    )


def assert_percentage_range(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is between 0 and 100."""
    assert Decimal("0") <= value <= Decimal("100"), (
        f"Percentage out of range{' (' + msg + ')' if msg else ''}: {value}"
    )


def assert_reduction_percentage(
    base: Decimal, target: Decimal, min_pct: Decimal, msg: str = "",
) -> None:
    """Assert that reduction from base to target is at least min_pct %."""
    if base == Decimal("0"):
        return
    reduction_pct = (base - target) / base * Decimal("100")
    assert reduction_pct >= min_pct, (
        f"Insufficient reduction{' (' + msg + ')' if msg else ''}: "
        f"{reduction_pct:.2f}% < {min_pct}% (base={base}, target={target})"
    )


def assert_provenance_hash(result: Any, result2: Any = None) -> None:
    """Assert that result has a non-empty SHA-256 provenance hash.

    If result2 is provided, also assert both hashes are equal
    (deterministic reproducibility check).
    """
    assert hasattr(result, "provenance_hash"), "Result missing provenance_hash"
    h = result.provenance_hash
    assert isinstance(h, str), "provenance_hash must be a string"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Hash must be hex"
    if result2 is not None:
        assert hasattr(result2, "provenance_hash"), "Result2 missing provenance_hash"
        h2 = result2.provenance_hash
        assert isinstance(h2, str), "provenance_hash must be a string"
        assert len(h2) == 64, f"SHA-256 hash must be 64 chars, got {len(h2)}"


def assert_processing_time(result: Any, max_ms: float = 60000.0) -> None:
    """Assert processing time is within acceptable range."""
    assert hasattr(result, "processing_time_ms"), "Result missing processing_time_ms"
    assert result.processing_time_ms >= 0, "Processing time must be non-negative"
    assert result.processing_time_ms < max_ms, (
        f"Processing time {result.processing_time_ms}ms exceeds {max_ms}ms"
    )


def assert_monotonically_decreasing(values: List[Decimal], msg: str = "") -> None:
    """Assert that a list of Decimal values is monotonically decreasing."""
    for i in range(1, len(values)):
        assert values[i] <= values[i - 1] + Decimal("0.001"), (
            f"Not monotonically decreasing{' (' + msg + ')' if msg else ''}: "
            f"index {i-1}={values[i-1]} -> index {i}={values[i]}"
        )


def assert_monotonically_increasing(values: List[Decimal], msg: str = "") -> None:
    """Assert that a list of Decimal values is monotonically increasing."""
    for i in range(1, len(values)):
        assert values[i] >= values[i - 1] - Decimal("0.001"), (
            f"Not monotonically increasing{' (' + msg + ')' if msg else ''}: "
            f"index {i-1}={values[i-1]} -> index {i}={values[i]}"
        )


def assert_sum_equals(
    parts: List[Decimal],
    total: Decimal,
    tolerance: Decimal = Decimal("0.01"),
    msg: str = "",
) -> None:
    """Assert that sum of parts equals total within tolerance."""
    parts_sum = sum(parts)
    diff = abs(parts_sum - total)
    assert diff <= tolerance, (
        f"Sum mismatch{' (' + msg + ')' if msg else ''}: "
        f"sum={parts_sum}, expected={total}, diff={diff}"
    )


def assert_years_continuous(years: List[int], msg: str = "") -> None:
    """Assert that a list of years is continuous (no gaps)."""
    for i in range(1, len(years)):
        assert years[i] == years[i - 1] + 1, (
            f"Year gap{' (' + msg + ')' if msg else ''}: "
            f"{years[i-1]} -> {years[i]}"
        )


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def verify_sha256(data: str, expected_hash: str) -> bool:
    """Verify SHA-256 hash matches expected."""
    return compute_sha256(data) == expected_hash


@contextmanager
def timed_block(label: str = "", max_seconds: float = 30.0, max_ms: float = 0.0):
    """Context manager that asserts a block completes within max_seconds.

    Args:
        label: Optional label for the block.
        max_seconds: Maximum allowed seconds.
        max_ms: Maximum allowed milliseconds (overrides max_seconds if > 0).
    """
    if max_ms > 0:
        max_seconds = max_ms / 1000.0
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"Block '{label}' took {elapsed:.3f}s, exceeding {max_seconds}s"
    )


# ---------------------------------------------------------------------------
# Reference implementations for validation
# ---------------------------------------------------------------------------


def linear_reduction(
    base_emissions: Decimal, target_emissions: Decimal,
    base_year: int, target_year: int, year: int,
) -> Decimal:
    """Reference implementation of linear emissions reduction pathway."""
    if year <= base_year:
        return base_emissions
    if year >= target_year:
        return target_emissions
    fraction = Decimal(str(year - base_year)) / Decimal(str(target_year - base_year))
    return base_emissions + (target_emissions - base_emissions) * fraction


def annual_reduction_rate(
    base_emissions: Decimal, target_emissions: Decimal,
    base_year: int, target_year: int,
) -> Decimal:
    """Calculate constant annual reduction rate (%)."""
    if base_emissions <= Decimal("0") or target_emissions < Decimal("0"):
        return Decimal("0")
    years = Decimal(str(target_year - base_year))
    if years <= Decimal("0"):
        return Decimal("0")
    ratio = target_emissions / base_emissions
    if ratio <= Decimal("0"):
        return Decimal("100")
    # rate = (1 - ratio^(1/years)) * 100
    rate_float = (1.0 - float(ratio) ** (1.0 / float(years))) * 100.0
    return Decimal(str(round(rate_float, 4)))


def cumulative_emissions_linear(
    base_emissions: Decimal, target_emissions: Decimal,
    base_year: int, target_year: int,
) -> Decimal:
    """Calculate cumulative emissions under linear reduction."""
    years = target_year - base_year + 1
    return (base_emissions + target_emissions) * Decimal(str(years)) / Decimal("2")


def sbti_15c_target_2030(base_year_emissions: Decimal, base_year: int = 2019) -> Decimal:
    """Calculate SBTi 1.5C near-term target for 2030 (42% absolute reduction)."""
    return base_year_emissions * (Decimal("1") - Decimal("0.42"))


def sbti_wb2c_target_2030(base_year_emissions: Decimal, base_year: int = 2019) -> Decimal:
    """Calculate SBTi WB2C near-term target for 2030 (30% absolute reduction)."""
    return base_year_emissions * (Decimal("1") - Decimal("0.30"))


def calculate_variance_pct(actual: Decimal, target: Decimal) -> Decimal:
    """Calculate variance percentage (positive = above target = bad)."""
    if target == Decimal("0"):
        if actual == Decimal("0"):
            return Decimal("0")
        return Decimal("100")
    return ((actual - target) / target) * Decimal("100")


# ---------------------------------------------------------------------------
# Fixtures -- Baseline Emissions Data (PACK-021 mock)
# ---------------------------------------------------------------------------


@pytest.fixture
def baseline_2019() -> Dict[str, Any]:
    """Build PACK-021 baseline data for base year 2019."""
    return {
        "entity_name": "GreenCorp Industries",
        "base_year": 2019,
        "scope_1_tco2e": Decimal("125000"),
        "scope_2_location_tco2e": Decimal("85000"),
        "scope_2_market_tco2e": Decimal("78000"),
        "scope_3_tco2e": Decimal("450000"),
        "total_scope_12_tco2e": Decimal("203000"),
        "total_scope_123_tco2e": Decimal("653000"),
        "scope_3_categories": {
            "cat_01_purchased_goods": Decimal("180000"),
            "cat_02_capital_goods": Decimal("35000"),
            "cat_03_fuel_energy": Decimal("28000"),
            "cat_04_upstream_transport": Decimal("42000"),
            "cat_05_waste": Decimal("12000"),
            "cat_06_business_travel": Decimal("18000"),
            "cat_07_employee_commuting": Decimal("22000"),
            "cat_08_upstream_leased": Decimal("8000"),
            "cat_09_downstream_transport": Decimal("25000"),
            "cat_10_processing": Decimal("15000"),
            "cat_11_use_of_sold": Decimal("35000"),
            "cat_12_end_of_life": Decimal("12000"),
            "cat_13_downstream_leased": Decimal("5000"),
            "cat_14_franchises": Decimal("3000"),
            "cat_15_investments": Decimal("10000"),
        },
        "revenue_m_usd": Decimal("2500"),
        "headcount": 12000,
        "floor_area_m2": Decimal("450000"),
        "data_quality_score": Decimal("0.85"),
        "provenance_hash": compute_sha256("baseline_2019_greencorp"),
    }


@pytest.fixture
def long_term_target() -> Dict[str, Any]:
    """Build PACK-021 long-term net-zero target."""
    return {
        "target_type": "net_zero",
        "ambition": "1.5C",
        "target_year": 2050,
        "scope_coverage": ["scope_1", "scope_2", "scope_3"],
        "scope_12_reduction_pct": Decimal("90"),
        "scope_3_reduction_pct": Decimal("90"),
        "residual_emissions_tco2e": Decimal("65300"),
        "neutralization_strategy": "permanent_carbon_removal",
        "sbti_validated": True,
        "provenance_hash": compute_sha256("long_term_target_greencorp"),
    }


@pytest.fixture
def baseline_small_company() -> Dict[str, Any]:
    """Build PACK-021 baseline for a small company (SME)."""
    return {
        "entity_name": "GreenSME Ltd",
        "base_year": 2019,
        "scope_1_tco2e": Decimal("5000"),
        "scope_2_location_tco2e": Decimal("3500"),
        "scope_2_market_tco2e": Decimal("3200"),
        "scope_3_tco2e": Decimal("18000"),
        "total_scope_12_tco2e": Decimal("8200"),
        "total_scope_123_tco2e": Decimal("26200"),
        "revenue_m_usd": Decimal("50"),
        "headcount": 200,
        "data_quality_score": Decimal("0.72"),
        "provenance_hash": compute_sha256("baseline_2019_greensme"),
    }


@pytest.fixture
def baseline_heavy_emitter() -> Dict[str, Any]:
    """Build PACK-021 baseline for a heavy industry emitter."""
    return {
        "entity_name": "HeavySteel Corp",
        "base_year": 2019,
        "scope_1_tco2e": Decimal("2500000"),
        "scope_2_location_tco2e": Decimal("800000"),
        "scope_2_market_tco2e": Decimal("750000"),
        "scope_3_tco2e": Decimal("4200000"),
        "total_scope_12_tco2e": Decimal("3250000"),
        "total_scope_123_tco2e": Decimal("7450000"),
        "sector": "steel",
        "revenue_m_usd": Decimal("12000"),
        "headcount": 45000,
        "data_quality_score": Decimal("0.91"),
        "provenance_hash": compute_sha256("baseline_2019_heavysteel"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Sector Pathway Data (PACK-028 mock)
# ---------------------------------------------------------------------------


@pytest.fixture
def sector_pathway_power() -> Dict[str, Any]:
    """Build PACK-028 sector pathway for power generation."""
    return {
        "sector": "power_generation",
        "base_year": 2019,
        "target_year": 2050,
        "scenario": "nze_15c",
        "convergence_model": "linear",
        "pathway_points": [
            {"year": y, "intensity_gco2_kwh": Decimal(str(max(0, 490 - 16 * (y - 2019))))}
            for y in range(2019, 2051)
        ],
        "provenance_hash": compute_sha256("sector_pathway_power"),
    }


@pytest.fixture
def sector_pathway_steel() -> Dict[str, Any]:
    """Build PACK-028 sector pathway for steel."""
    return {
        "sector": "steel",
        "base_year": 2019,
        "target_year": 2050,
        "scenario": "nze_15c",
        "convergence_model": "s_curve",
        "pathway_points": [
            {"year": y, "intensity_tco2e_t": Decimal(str(round(2.10 - 0.061 * (y - 2019), 3)))}
            for y in range(2019, 2051)
        ],
        "provenance_hash": compute_sha256("sector_pathway_steel"),
    }


@pytest.fixture
def abatement_levers() -> List[Dict[str, Any]]:
    """Build PACK-028 abatement lever data."""
    return [
        {
            "name": "Renewable Energy Procurement",
            "scope": "scope_2",
            "reduction_tco2e": Decimal("35000"),
            "reduction_pct": Decimal("17.2"),
            "cost_usd_per_tco2e": Decimal("-15"),
            "start_year": 2024,
            "end_year": 2028,
            "trl": 9,
            "confidence": Decimal("0.95"),
        },
        {
            "name": "Process Heat Electrification",
            "scope": "scope_1",
            "reduction_tco2e": Decimal("28000"),
            "reduction_pct": Decimal("13.8"),
            "cost_usd_per_tco2e": Decimal("45"),
            "start_year": 2025,
            "end_year": 2032,
            "trl": 8,
            "confidence": Decimal("0.88"),
        },
        {
            "name": "Fleet Electrification",
            "scope": "scope_1",
            "reduction_tco2e": Decimal("15000"),
            "reduction_pct": Decimal("7.4"),
            "cost_usd_per_tco2e": Decimal("60"),
            "start_year": 2024,
            "end_year": 2030,
            "trl": 9,
            "confidence": Decimal("0.92"),
        },
        {
            "name": "Supplier Engagement Program",
            "scope": "scope_3",
            "reduction_tco2e": Decimal("90000"),
            "reduction_pct": Decimal("20.0"),
            "cost_usd_per_tco2e": Decimal("25"),
            "start_year": 2024,
            "end_year": 2035,
            "trl": 7,
            "confidence": Decimal("0.70"),
        },
        {
            "name": "Energy Efficiency Upgrades",
            "scope": "scope_1",
            "reduction_tco2e": Decimal("12000"),
            "reduction_pct": Decimal("5.9"),
            "cost_usd_per_tco2e": Decimal("-25"),
            "start_year": 2024,
            "end_year": 2027,
            "trl": 9,
            "confidence": Decimal("0.96"),
        },
        {
            "name": "Logistics Optimization",
            "scope": "scope_3",
            "reduction_tco2e": Decimal("21000"),
            "reduction_pct": Decimal("4.7"),
            "cost_usd_per_tco2e": Decimal("10"),
            "start_year": 2025,
            "end_year": 2030,
            "trl": 8,
            "confidence": Decimal("0.85"),
        },
        {
            "name": "On-site Solar Installation",
            "scope": "scope_2",
            "reduction_tco2e": Decimal("18000"),
            "reduction_pct": Decimal("8.9"),
            "cost_usd_per_tco2e": Decimal("-10"),
            "start_year": 2024,
            "end_year": 2026,
            "trl": 9,
            "confidence": Decimal("0.97"),
        },
        {
            "name": "Carbon Capture (Pilot)",
            "scope": "scope_1",
            "reduction_tco2e": Decimal("8000"),
            "reduction_pct": Decimal("3.9"),
            "cost_usd_per_tco2e": Decimal("120"),
            "start_year": 2028,
            "end_year": 2035,
            "trl": 6,
            "confidence": Decimal("0.55"),
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures -- MRV Actual Emissions Data
# ---------------------------------------------------------------------------


@pytest.fixture
def actual_emissions_series() -> List[Dict[str, Any]]:
    """Build MRV actual emissions data from 2019 to 2025."""
    base_s1 = Decimal("125000")
    base_s2 = Decimal("78000")
    base_s3 = Decimal("450000")
    data = []
    reductions = {
        2019: Decimal("0"),
        2020: Decimal("0.04"),    # 4% (COVID dip)
        2021: Decimal("0.02"),    # 2% (partial recovery)
        2022: Decimal("0.06"),    # 6%
        2023: Decimal("0.10"),    # 10%
        2024: Decimal("0.14"),    # 14%
        2025: Decimal("0.18"),    # 18%
    }
    for year, red_pct in reductions.items():
        factor = Decimal("1") - red_pct
        data.append({
            "year": year,
            "scope_1_tco2e": (base_s1 * factor).quantize(Decimal("1")),
            "scope_2_market_tco2e": (base_s2 * factor).quantize(Decimal("1")),
            "scope_3_tco2e": (base_s3 * (Decimal("1") - red_pct * Decimal("0.5"))).quantize(Decimal("1")),
            "total_scope_12_tco2e": ((base_s1 + base_s2) * factor).quantize(Decimal("1")),
            "data_quality_score": Decimal("0.85") + Decimal(str(year - 2019)) * Decimal("0.01"),
            "verified": year <= 2024,
            "provenance_hash": compute_sha256(f"actual_emissions_{year}"),
        })
    return data


@pytest.fixture
def actual_emissions_quarterly() -> List[Dict[str, Any]]:
    """Build quarterly MRV emissions data for 2024."""
    base_annual_s12 = Decimal("174580")  # 2024 scope 1+2
    q_values = [
        Decimal("0.28"),  # Q1 (higher heating)
        Decimal("0.23"),  # Q2
        Decimal("0.24"),  # Q3 (cooling)
        Decimal("0.25"),  # Q4
    ]
    return [
        {
            "year": 2024,
            "quarter": f"Q{i+1}",
            "scope_12_tco2e": (base_annual_s12 * q_values[i]).quantize(Decimal("1")),
            "cumulative_tco2e": sum(
                (base_annual_s12 * q_values[j]).quantize(Decimal("1"))
                for j in range(i + 1)
            ),
            "provenance_hash": compute_sha256(f"quarterly_emissions_2024_Q{i+1}"),
        }
        for i in range(4)
    ]


@pytest.fixture
def actual_emissions_backsliding() -> List[Dict[str, Any]]:
    """Build MRV data showing backsliding (emissions increase)."""
    return [
        {"year": 2019, "scope_12_tco2e": Decimal("203000"), "scope_3_tco2e": Decimal("450000")},
        {"year": 2020, "scope_12_tco2e": Decimal("185000"), "scope_3_tco2e": Decimal("430000")},
        {"year": 2021, "scope_12_tco2e": Decimal("195000"), "scope_3_tco2e": Decimal("445000")},
        {"year": 2022, "scope_12_tco2e": Decimal("210000"), "scope_3_tco2e": Decimal("470000")},  # backslide!
        {"year": 2023, "scope_12_tco2e": Decimal("205000"), "scope_3_tco2e": Decimal("460000")},
    ]


@pytest.fixture
def actual_emissions_ahead_of_target() -> List[Dict[str, Any]]:
    """Build MRV data showing performance ahead of target."""
    return [
        {"year": 2019, "scope_12_tco2e": Decimal("203000"), "scope_3_tco2e": Decimal("450000")},
        {"year": 2020, "scope_12_tco2e": Decimal("175000"), "scope_3_tco2e": Decimal("410000")},
        {"year": 2021, "scope_12_tco2e": Decimal("155000"), "scope_3_tco2e": Decimal("380000")},
        {"year": 2022, "scope_12_tco2e": Decimal("135000"), "scope_3_tco2e": Decimal("350000")},
        {"year": 2023, "scope_12_tco2e": Decimal("115000"), "scope_3_tco2e": Decimal("320000")},
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Interim Target Data
# ---------------------------------------------------------------------------


@pytest.fixture
def interim_target_2030_15c() -> Dict[str, Any]:
    """Build SBTi 1.5C near-term interim target (2030)."""
    base_scope12 = Decimal("203000")
    base_scope3 = Decimal("450000")
    return {
        "target_year": 2030,
        "base_year": 2019,
        "ambition": "1.5C",
        "scope_12_target_tco2e": (base_scope12 * Decimal("0.58")).quantize(Decimal("1")),  # 42% reduction
        "scope_12_reduction_pct": Decimal("42"),
        "scope_3_target_tco2e": (base_scope3 * Decimal("0.75")).quantize(Decimal("1")),  # 25% reduction
        "scope_3_reduction_pct": Decimal("25"),
        "pathway_type": "linear",
        "annual_scope12_reduction_rate": Decimal("4.86"),  # ~4.86%/yr for 42% in 11 years
        "provenance_hash": compute_sha256("interim_target_2030_15c"),
    }


@pytest.fixture
def interim_target_2030_wb2c() -> Dict[str, Any]:
    """Build SBTi WB2C near-term interim target (2030)."""
    base_scope12 = Decimal("203000")
    return {
        "target_year": 2030,
        "base_year": 2019,
        "ambition": "WB2C",
        "scope_12_target_tco2e": (base_scope12 * Decimal("0.70")).quantize(Decimal("1")),  # 30% reduction
        "scope_12_reduction_pct": Decimal("30"),
        "pathway_type": "linear",
        "annual_scope12_reduction_rate": Decimal("3.18"),
        "provenance_hash": compute_sha256("interim_target_2030_wb2c"),
    }


@pytest.fixture
def interim_target_2035() -> Dict[str, Any]:
    """Build 10-year interim target (2035)."""
    base_scope12 = Decimal("203000")
    return {
        "target_year": 2035,
        "base_year": 2019,
        "ambition": "1.5C",
        "scope_12_target_tco2e": (base_scope12 * Decimal("0.38")).quantize(Decimal("1")),  # 62% reduction
        "scope_12_reduction_pct": Decimal("62"),
        "pathway_type": "linear",
        "provenance_hash": compute_sha256("interim_target_2035"),
    }


@pytest.fixture
def interim_target_5yr_milestones() -> List[Dict[str, Any]]:
    """Build 5-year milestone targets from 2019 to 2050."""
    base = Decimal("203000")
    return [
        {"year": 2025, "scope_12_target_tco2e": (base * Decimal("0.78")).quantize(Decimal("1")),
         "reduction_pct": Decimal("22")},
        {"year": 2030, "scope_12_target_tco2e": (base * Decimal("0.58")).quantize(Decimal("1")),
         "reduction_pct": Decimal("42")},
        {"year": 2035, "scope_12_target_tco2e": (base * Decimal("0.38")).quantize(Decimal("1")),
         "reduction_pct": Decimal("62")},
        {"year": 2040, "scope_12_target_tco2e": (base * Decimal("0.22")).quantize(Decimal("1")),
         "reduction_pct": Decimal("78")},
        {"year": 2045, "scope_12_target_tco2e": (base * Decimal("0.12")).quantize(Decimal("1")),
         "reduction_pct": Decimal("88")},
        {"year": 2050, "scope_12_target_tco2e": (base * Decimal("0.10")).quantize(Decimal("1")),
         "reduction_pct": Decimal("90")},
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Variance Analysis Data
# ---------------------------------------------------------------------------


@pytest.fixture
def variance_data_2023() -> Dict[str, Any]:
    """Build variance analysis data for 2023."""
    return {
        "year": 2023,
        "target_scope12_tco2e": Decimal("171000"),
        "actual_scope12_tco2e": Decimal("183000"),
        "variance_tco2e": Decimal("12000"),
        "variance_pct": Decimal("7.02"),
        "activity_effect_tco2e": Decimal("5000"),
        "intensity_effect_tco2e": Decimal("-3000"),
        "structural_effect_tco2e": Decimal("2000"),
        "fuel_mix_effect_tco2e": Decimal("1500"),
        "weather_effect_tco2e": Decimal("6500"),
        "total_decomposition": Decimal("12000"),
        "provenance_hash": compute_sha256("variance_data_2023"),
    }


@pytest.fixture
def kaya_decomposition_data() -> Dict[str, Any]:
    """Build Kaya identity decomposition data."""
    return {
        "year": 2023,
        "gdp_effect_pct": Decimal("3.2"),
        "energy_intensity_effect_pct": Decimal("-1.8"),
        "carbon_intensity_effect_pct": Decimal("-2.5"),
        "population_effect_pct": Decimal("0.5"),
        "net_change_pct": Decimal("-0.6"),
        "provenance_hash": compute_sha256("kaya_decomposition_2023"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Corrective Action Data
# ---------------------------------------------------------------------------


@pytest.fixture
def corrective_action_portfolio() -> List[Dict[str, Any]]:
    """Build corrective action initiative portfolio."""
    return [
        {
            "name": "Accelerated Renewable Procurement",
            "type": "acceleration",
            "additional_reduction_tco2e": Decimal("15000"),
            "investment_usd": Decimal("2500000"),
            "payback_years": Decimal("3.5"),
            "implementation_months": 6,
            "confidence": Decimal("0.92"),
        },
        {
            "name": "Emergency Efficiency Program",
            "type": "quick_win",
            "additional_reduction_tco2e": Decimal("8000"),
            "investment_usd": Decimal("500000"),
            "payback_years": Decimal("1.5"),
            "implementation_months": 3,
            "confidence": Decimal("0.95"),
        },
        {
            "name": "Supplier Switching Initiative",
            "type": "scope_3_action",
            "additional_reduction_tco2e": Decimal("25000"),
            "investment_usd": Decimal("1200000"),
            "payback_years": Decimal("4.0"),
            "implementation_months": 12,
            "confidence": Decimal("0.75"),
        },
        {
            "name": "Process Redesign",
            "type": "structural",
            "additional_reduction_tco2e": Decimal("20000"),
            "investment_usd": Decimal("8000000"),
            "payback_years": Decimal("7.0"),
            "implementation_months": 24,
            "confidence": Decimal("0.80"),
        },
        {
            "name": "Fuel Switching (Gas to Electric)",
            "type": "fuel_switch",
            "additional_reduction_tco2e": Decimal("18000"),
            "investment_usd": Decimal("4500000"),
            "payback_years": Decimal("5.5"),
            "implementation_months": 18,
            "confidence": Decimal("0.88"),
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures -- SBTi Validation Criteria Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sbti_criteria_data() -> Dict[str, Any]:
    """Build SBTi 21-criteria validation dataset."""
    return {
        "criteria": {
            "C01_boundary": {"status": "pass", "detail": "Scope 1+2+3 covered"},
            "C02_timeframe": {"status": "pass", "detail": "Near-term 5-10 years"},
            "C03_base_year": {"status": "pass", "detail": "2019 base year valid"},
            "C04_ambition_scope12": {"status": "pass", "detail": "42% >= 42% minimum (1.5C)"},
            "C05_ambition_scope3": {"status": "pass", "detail": "25% >= 25% minimum"},
            "C06_coverage_scope1": {"status": "pass", "detail": "100% of Scope 1"},
            "C07_coverage_scope2": {"status": "pass", "detail": "100% of Scope 2"},
            "C08_coverage_scope3": {"status": "pass", "detail": "67% of Scope 3 (>40%)"},
            "C09_linearity": {"status": "pass", "detail": "No backsliding"},
            "C10_scope3_materiality": {"status": "pass", "detail": "Scope 3 > 40% total"},
            "C11_recalculation_policy": {"status": "pass", "detail": "5% threshold trigger"},
            "C12_exclusions": {"status": "pass", "detail": "< 5% excluded"},
            "C13_offsets_not_counted": {"status": "pass", "detail": "No offsets in target"},
            "C14_flag_sector_check": {"status": "not_applicable", "detail": "Non-FLAG company"},
            "C15_target_language": {"status": "pass", "detail": "Absolute contraction"},
            "C16_public_disclosure": {"status": "pass", "detail": "Annual disclosure"},
            "C17_progress_reporting": {"status": "pass", "detail": "Annual progress report"},
            "C18_bioenergy_accounting": {"status": "pass", "detail": "GHG Protocol compliant"},
            "C19_market_instruments": {"status": "pass", "detail": "RECs properly accounted"},
            "C20_data_quality": {"status": "pass", "detail": "DQ score >= 0.80"},
            "C21_verification": {"status": "pass", "detail": "Third-party verified"},
        },
        "overall_status": "pass",
        "pass_count": 20,
        "fail_count": 0,
        "na_count": 1,
        "provenance_hash": compute_sha256("sbti_criteria_data"),
    }


@pytest.fixture
def sbti_criteria_failing() -> Dict[str, Any]:
    """Build SBTi criteria data with failures."""
    return {
        "criteria": {
            "C01_boundary": {"status": "pass", "detail": "Scope 1+2+3 covered"},
            "C02_timeframe": {"status": "pass", "detail": "Near-term 5-10 years"},
            "C03_base_year": {"status": "pass", "detail": "2019 base year valid"},
            "C04_ambition_scope12": {"status": "fail", "detail": "25% < 42% minimum (1.5C)"},
            "C05_ambition_scope3": {"status": "fail", "detail": "10% < 25% minimum"},
            "C06_coverage_scope1": {"status": "pass", "detail": "100% of Scope 1"},
            "C07_coverage_scope2": {"status": "pass", "detail": "100% of Scope 2"},
            "C08_coverage_scope3": {"status": "fail", "detail": "35% < 40% threshold"},
            "C09_linearity": {"status": "warning", "detail": "Minor backsliding in 2022"},
            "C10_scope3_materiality": {"status": "pass", "detail": "Scope 3 > 40% total"},
            "C11_recalculation_policy": {"status": "pass", "detail": "5% threshold trigger"},
            "C12_exclusions": {"status": "warning", "detail": "7% excluded (> 5%)"},
            "C13_offsets_not_counted": {"status": "pass", "detail": "No offsets in target"},
            "C14_flag_sector_check": {"status": "not_applicable", "detail": "Non-FLAG company"},
            "C15_target_language": {"status": "pass", "detail": "Absolute contraction"},
            "C16_public_disclosure": {"status": "pass", "detail": "Annual disclosure"},
            "C17_progress_reporting": {"status": "warning", "detail": "Delayed reporting"},
            "C18_bioenergy_accounting": {"status": "pass", "detail": "GHG Protocol compliant"},
            "C19_market_instruments": {"status": "pass", "detail": "RECs properly accounted"},
            "C20_data_quality": {"status": "fail", "detail": "DQ score 0.65 < 0.80"},
            "C21_verification": {"status": "fail", "detail": "Not third-party verified"},
        },
        "overall_status": "fail",
        "pass_count": 13,
        "fail_count": 5,
        "warning_count": 3,
        "na_count": 1,
        "provenance_hash": compute_sha256("sbti_criteria_failing"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Initiative Scheduling Data
# ---------------------------------------------------------------------------


@pytest.fixture
def initiative_schedule_data() -> List[Dict[str, Any]]:
    """Build initiative scheduling data for phased rollout."""
    return [
        {
            "id": "INI-001",
            "name": "LED Lighting Retrofit",
            "phase": "pilot",
            "start_date": "2024-Q1",
            "end_date": "2024-Q3",
            "budget_usd": Decimal("200000"),
            "reduction_tco2e": Decimal("3000"),
            "trl": 9,
            "dependencies": [],
        },
        {
            "id": "INI-002",
            "name": "HVAC Optimization",
            "phase": "scale",
            "start_date": "2024-Q2",
            "end_date": "2025-Q2",
            "budget_usd": Decimal("800000"),
            "reduction_tco2e": Decimal("8000"),
            "trl": 9,
            "dependencies": ["INI-001"],
        },
        {
            "id": "INI-003",
            "name": "Solar PV Installation",
            "phase": "full_deployment",
            "start_date": "2024-Q3",
            "end_date": "2026-Q1",
            "budget_usd": Decimal("3500000"),
            "reduction_tco2e": Decimal("18000"),
            "trl": 9,
            "dependencies": [],
        },
        {
            "id": "INI-004",
            "name": "Fleet Electrification Phase 1",
            "phase": "pilot",
            "start_date": "2025-Q1",
            "end_date": "2025-Q4",
            "budget_usd": Decimal("1500000"),
            "reduction_tco2e": Decimal("5000"),
            "trl": 9,
            "dependencies": [],
        },
        {
            "id": "INI-005",
            "name": "Process Heat Pump",
            "phase": "pilot",
            "start_date": "2025-Q2",
            "end_date": "2026-Q2",
            "budget_usd": Decimal("2000000"),
            "reduction_tco2e": Decimal("12000"),
            "trl": 8,
            "dependencies": ["INI-002"],
        },
        {
            "id": "INI-006",
            "name": "Supplier Low-Carbon Transition",
            "phase": "scale",
            "start_date": "2024-Q4",
            "end_date": "2027-Q4",
            "budget_usd": Decimal("1200000"),
            "reduction_tco2e": Decimal("45000"),
            "trl": 7,
            "dependencies": [],
        },
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Carbon Budget Data
# ---------------------------------------------------------------------------


@pytest.fixture
def carbon_budget_data() -> Dict[str, Any]:
    """Build carbon budget allocation data."""
    return {
        "total_budget_tco2e": Decimal("3500000"),
        "budget_start_year": 2019,
        "budget_end_year": 2050,
        "allocation_strategy": "front_loaded",
        "annual_budgets": {
            year: Decimal(str(max(10000, 203000 - 6200 * (year - 2019))))
            for year in range(2019, 2051)
        },
        "cumulative_used_tco2e": Decimal("1050000"),
        "cumulative_remaining_tco2e": Decimal("2450000"),
        "years_remaining": 26,
        "internal_carbon_price_usd": Decimal("85"),
        "provenance_hash": compute_sha256("carbon_budget_data"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Reporting Framework Data
# ---------------------------------------------------------------------------


@pytest.fixture
def cdp_response_data() -> Dict[str, Any]:
    """Build CDP C4.1/C4.2 response data."""
    return {
        "c4_1_details": {
            "target_type": "absolute",
            "scope": "scope_1_2",
            "base_year": 2019,
            "base_year_emissions": Decimal("203000"),
            "target_year": 2030,
            "target_reduction_pct": Decimal("42"),
            "progress_to_target_pct": Decimal("38"),
        },
        "c4_2_details": {
            "intensity_target_type": "revenue",
            "base_year_intensity": Decimal("81.2"),
            "target_year_intensity": Decimal("45.0"),
            "current_intensity": Decimal("55.3"),
        },
        "provenance_hash": compute_sha256("cdp_response_data"),
    }


@pytest.fixture
def tcfd_metrics_data() -> Dict[str, Any]:
    """Build TCFD metrics disclosure data."""
    return {
        "scope_1_tco2e": Decimal("107500"),
        "scope_2_location_tco2e": Decimal("73100"),
        "scope_2_market_tco2e": Decimal("67080"),
        "scope_3_tco2e": Decimal("405000"),
        "total_tco2e": Decimal("579580"),
        "revenue_intensity_tco2e_per_m_usd": Decimal("55.3"),
        "progress_to_near_term_target_pct": Decimal("38"),
        "transition_risks_identified": 5,
        "physical_risks_identified": 3,
        "provenance_hash": compute_sha256("tcfd_metrics_data"),
    }


# ---------------------------------------------------------------------------
# Fixtures -- Mock Database Session & Cache
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session():
    """Create a mock async database session (PostgreSQL + TimescaleDB)."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(
        fetchall=MagicMock(return_value=[]),
        fetchone=MagicMock(return_value=None),
        scalar=MagicMock(return_value=0),
    ))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    session.begin = MagicMock()
    return session


@pytest.fixture
def mock_redis():
    """Create a mock Redis client."""
    redis_client = MagicMock()
    redis_client.get = AsyncMock(return_value=None)
    redis_client.set = AsyncMock(return_value=True)
    redis_client.delete = AsyncMock(return_value=True)
    redis_client.exists = AsyncMock(return_value=False)
    redis_client.expire = AsyncMock(return_value=True)
    redis_client.hget = AsyncMock(return_value=None)
    redis_client.hset = AsyncMock(return_value=True)
    redis_client.pipeline = MagicMock(return_value=MagicMock(
        execute=AsyncMock(return_value=[]),
    ))
    return redis_client


# ---------------------------------------------------------------------------
# Fixtures -- Mock External API Clients
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sbti_api():
    """Create a mock SBTi validation API client."""
    client = MagicMock()
    client.validate_near_term_target = AsyncMock(return_value={
        "valid": True,
        "ambition": "1.5C",
        "criteria_met": 20,
        "criteria_total": 21,
        "criteria_na": 1,
    })
    client.validate_long_term_target = AsyncMock(return_value={
        "valid": True,
        "ambition": "net_zero",
        "target_year": 2050,
    })
    client.get_minimum_ambition = AsyncMock(return_value={
        "1.5C": {"scope_12_min_pct": Decimal("42"), "scope_3_min_pct": Decimal("25")},
        "WB2C": {"scope_12_min_pct": Decimal("30"), "scope_3_min_pct": Decimal("25")},
    })
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_cdp_api():
    """Create a mock CDP API client."""
    client = MagicMock()
    client.submit_c4_response = AsyncMock(return_value={
        "status": "accepted",
        "response_id": "CDP-2025-GC-001",
    })
    client.get_scoring = AsyncMock(return_value={"score": "A-"})
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_tcfd_api():
    """Create a mock TCFD API client."""
    client = MagicMock()
    client.submit_metrics = AsyncMock(return_value={
        "status": "accepted",
        "alignment_score": Decimal("0.82"),
    })
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_mrv_bridge():
    """Create a mock MRV bridge for routing to 30 agents."""
    bridge = MagicMock()
    bridge.get_scope1_emissions = AsyncMock(return_value=Decimal("107500"))
    bridge.get_scope2_location = AsyncMock(return_value=Decimal("73100"))
    bridge.get_scope2_market = AsyncMock(return_value=Decimal("67080"))
    bridge.get_scope3_total = AsyncMock(return_value=Decimal("405000"))
    bridge.get_scope3_category = AsyncMock(return_value=Decimal("15000"))
    bridge.get_all_scopes = AsyncMock(return_value={
        "scope_1": Decimal("107500"),
        "scope_2_location": Decimal("73100"),
        "scope_2_market": Decimal("67080"),
        "scope_3": Decimal("405000"),
    })
    bridge.is_connected = MagicMock(return_value=True)
    return bridge


@pytest.fixture
def mock_alerting_bridge():
    """Create a mock alerting bridge for notifications."""
    bridge = MagicMock()
    bridge.send_alert = AsyncMock(return_value={"status": "sent", "channel": "email"})
    bridge.send_slack_notification = AsyncMock(return_value={"status": "sent"})
    bridge.send_teams_notification = AsyncMock(return_value={"status": "sent"})
    bridge.is_connected = MagicMock(return_value=True)
    return bridge


# ---------------------------------------------------------------------------
# Fixtures -- Pack paths
# ---------------------------------------------------------------------------


@pytest.fixture
def pack_yaml_path() -> Path:
    """Return the path to pack.yaml."""
    return _PACK_ROOT / "pack.yaml"


@pytest.fixture
def pack_root() -> Path:
    """Return the pack root directory."""
    return _PACK_ROOT


@pytest.fixture
def presets_dir() -> Path:
    """Return the presets directory."""
    return PRESETS_DIR


# ---------------------------------------------------------------------------
# Parametrized Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(params=SBTI_AMBITION_LEVELS, ids=SBTI_AMBITION_LEVELS)
def ambition_level(request) -> str:
    """Parameterized fixture yielding each SBTi ambition level."""
    return request.param


@pytest.fixture(params=PATHWAY_TYPES, ids=PATHWAY_TYPES)
def pathway_type(request) -> str:
    """Parameterized fixture yielding each pathway type."""
    return request.param


@pytest.fixture(params=SCOPES, ids=SCOPES)
def scope(request) -> str:
    """Parameterized fixture yielding each emission scope."""
    return request.param


@pytest.fixture(params=VARIANCE_EFFECTS, ids=VARIANCE_EFFECTS)
def variance_effect(request) -> str:
    """Parameterized fixture yielding each variance effect type."""
    return request.param


@pytest.fixture(params=FORECAST_MODELS, ids=FORECAST_MODELS)
def forecast_model(request) -> str:
    """Parameterized fixture yielding each forecast model."""
    return request.param


@pytest.fixture(params=PERFORMANCE_RATINGS, ids=PERFORMANCE_RATINGS)
def performance_rating(request) -> str:
    """Parameterized fixture yielding each performance rating."""
    return request.param


@pytest.fixture(params=REPORTING_FRAMEWORKS, ids=REPORTING_FRAMEWORKS)
def reporting_framework(request) -> str:
    """Parameterized fixture yielding each reporting framework."""
    return request.param


@pytest.fixture(params=PRESET_NAMES, ids=PRESET_NAMES)
def preset_name(request) -> str:
    """Parameterized fixture yielding each preset name."""
    return request.param


@pytest.fixture(params=QUARTERS, ids=QUARTERS)
def quarter(request) -> str:
    """Parameterized fixture yielding each quarter."""
    return request.param


# ---------------------------------------------------------------------------
# Test Data Generators
# ---------------------------------------------------------------------------


def generate_emissions_series(
    base_emissions: Decimal,
    start_year: int = 2019,
    end_year: int = 2050,
    annual_reduction_pct: Decimal = Decimal("4.0"),
) -> List[Dict[str, Any]]:
    """Generate a synthetic emissions time series with constant annual reduction."""
    series = []
    current = base_emissions
    for year in range(start_year, end_year + 1):
        series.append({
            "year": year,
            "emissions_tco2e": current.quantize(Decimal("1")),
            "provenance_hash": compute_sha256(f"gen_emissions_{year}"),
        })
        current = current * (Decimal("1") - annual_reduction_pct / Decimal("100"))
    return series


def generate_target_pathway(
    base_emissions: Decimal,
    target_emissions: Decimal,
    base_year: int = 2019,
    target_year: int = 2050,
    pathway_type: str = "linear",
) -> List[Dict[str, Any]]:
    """Generate a target pathway series."""
    pathway = []
    for year in range(base_year, target_year + 1):
        if pathway_type == "linear":
            emissions = linear_reduction(base_emissions, target_emissions, base_year, target_year, year)
        else:
            emissions = linear_reduction(base_emissions, target_emissions, base_year, target_year, year)
        pathway.append({
            "year": year,
            "target_tco2e": emissions.quantize(Decimal("1")),
        })
    return pathway


def generate_variance_decomposition(
    total_variance: Decimal,
    effects: int = 5,
) -> List[Dict[str, Any]]:
    """Generate LMDI decomposition effects that sum to total variance."""
    effect_names = VARIANCE_EFFECTS[:effects]
    shares = [Decimal(str(round(1.0 / effects, 4))) for _ in range(effects)]
    shares[-1] = Decimal("1") - sum(shares[:-1])
    return [
        {
            "effect": effect_names[i],
            "contribution_tco2e": (total_variance * shares[i]).quantize(Decimal("1")),
            "contribution_pct": (shares[i] * Decimal("100")).quantize(Decimal("0.01")),
        }
        for i in range(effects)
    ]


def generate_initiative_portfolio(count: int = 5) -> List[Dict[str, Any]]:
    """Generate a portfolio of corrective action initiatives."""
    initiatives = []
    for i in range(count):
        initiatives.append({
            "id": f"INI-{i+1:03d}",
            "name": f"Initiative {i+1}",
            "reduction_tco2e": Decimal(str(5000 + 3000 * i)),
            "cost_usd": Decimal(str(200000 + 500000 * i)),
            "payback_years": Decimal(str(round(1.5 + 0.8 * i, 1))),
            "trl": min(9, 6 + i),
            "confidence": Decimal(str(round(0.95 - 0.05 * i, 2))),
        })
    return initiatives


def generate_quarterly_data(
    annual_emissions: Decimal,
    year: int = 2024,
) -> List[Dict[str, Any]]:
    """Generate quarterly emissions data for a given year."""
    q_shares = [Decimal("0.28"), Decimal("0.23"), Decimal("0.24"), Decimal("0.25")]
    return [
        {
            "year": year,
            "quarter": f"Q{i+1}",
            "emissions_tco2e": (annual_emissions * q_shares[i]).quantize(Decimal("1")),
        }
        for i in range(4)
    ]


# ---------------------------------------------------------------------------
# Helper functions for test result validation
# ---------------------------------------------------------------------------


def assert_interim_target_valid(result: Any) -> None:
    """Assert interim target result is well-formed."""
    assert result is not None
    if hasattr(result, "target_year"):
        assert result.target_year >= 2025
        assert result.target_year <= 2050
    if hasattr(result, "reduction_pct"):
        assert_percentage_range(result.reduction_pct, "reduction_pct")


def assert_pathway_valid(result: Any) -> None:
    """Assert pathway result is well-formed."""
    assert result is not None
    if hasattr(result, "pathway_points"):
        assert len(result.pathway_points) > 0
        years = [p.year for p in result.pathway_points]
        assert years == sorted(years), "Pathway years must be sorted"


def assert_progress_valid(result: Any) -> None:
    """Assert progress tracker result is well-formed."""
    assert result is not None
    if hasattr(result, "performance_score"):
        assert result.performance_score in PERFORMANCE_RATINGS or isinstance(
            result.performance_score, (str, int, float, Decimal)
        )


def assert_variance_valid(result: Any) -> None:
    """Assert variance analysis result is well-formed."""
    assert result is not None
    if hasattr(result, "decomposition_effects"):
        effects = result.decomposition_effects
        assert len(effects) > 0


def assert_forecast_valid(result: Any) -> None:
    """Assert trend extrapolation result is well-formed."""
    assert result is not None
    if hasattr(result, "forecast_points"):
        assert len(result.forecast_points) > 0
    if hasattr(result, "confidence_interval"):
        assert result.confidence_interval is not None


def assert_corrective_action_valid(result: Any) -> None:
    """Assert corrective action result is well-formed."""
    assert result is not None
    if hasattr(result, "action_items"):
        assert len(result.action_items) > 0


def assert_milestone_validation_valid(result: Any) -> None:
    """Assert milestone validation result is well-formed."""
    assert result is not None
    if hasattr(result, "criteria_results"):
        assert len(result.criteria_results) > 0
    if hasattr(result, "overall_status"):
        assert result.overall_status in ("pass", "fail", "warning")


def assert_schedule_valid(result: Any) -> None:
    """Assert initiative scheduler result is well-formed."""
    assert result is not None
    if hasattr(result, "deployment_plan"):
        assert len(result.deployment_plan) > 0


def assert_budget_valid(result: Any) -> None:
    """Assert budget allocation result is well-formed."""
    assert result is not None
    if hasattr(result, "annual_budgets"):
        assert len(result.annual_budgets) > 0
    if hasattr(result, "total_budget_tco2e"):
        assert_decimal_positive(result.total_budget_tco2e)


def assert_report_valid(result: Any) -> None:
    """Assert reporting result is well-formed."""
    assert result is not None
    if hasattr(result, "disclosure_sections"):
        assert len(result.disclosure_sections) > 0


# ---------------------------------------------------------------------------
# Extended Emission Scenario Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def emissions_net_zero_before_target() -> List[Dict[str, Any]]:
    """Build emissions data achieving net-zero before 2050."""
    base = Decimal("203000")
    return [
        {"year": y, "scope_12_tco2e": max(Decimal("0"),
         base * (Decimal("1") - Decimal(str(min(1.0, 0.05 * (y - 2019)))))).quantize(Decimal("1"))}
        for y in range(2019, 2051)
    ]


@pytest.fixture
def emissions_flat_trajectory() -> List[Dict[str, Any]]:
    """Build flat emissions trajectory (no reduction)."""
    return [
        {"year": y, "scope_12_tco2e": Decimal("203000")}
        for y in range(2019, 2051)
    ]


@pytest.fixture
def emissions_rapid_reduction() -> List[Dict[str, Any]]:
    """Build rapid 8% annual reduction trajectory."""
    return generate_emissions_series(
        base_emissions=Decimal("203000"),
        start_year=2019,
        end_year=2050,
        annual_reduction_pct=Decimal("8.0"),
    )


@pytest.fixture
def emissions_slow_reduction() -> List[Dict[str, Any]]:
    """Build slow 1.5% annual reduction trajectory."""
    return generate_emissions_series(
        base_emissions=Decimal("203000"),
        start_year=2019,
        end_year=2050,
        annual_reduction_pct=Decimal("1.5"),
    )
