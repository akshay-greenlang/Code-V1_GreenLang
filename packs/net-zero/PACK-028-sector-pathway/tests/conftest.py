# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-028 Sector Pathway Pack.
=============================================================

Provides pytest fixtures for all 8 engines, 6 workflows, sample sector
data builders, database mock setup, mock external API helpers (SBTi, IEA),
and common test utilities tailored for sector-specific pathway analysis.

Adds the pack root to sys.path so ``from engines.X import Y`` works
in every test module without requiring an installed package.

Fixtures cover:
    - Engine instantiation (8 sector pathway engines)
    - Workflow instantiation (6 workflows)
    - Sector profile builders (power, steel, cement, aviation, etc.)
    - SBTi SDA reference data fixtures
    - IEA NZE 2050 reference data fixtures
    - Technology milestone fixtures
    - Intensity metric fixtures for 15+ sectors
    - Convergence model fixtures (linear, exponential, S-curve)
    - Abatement lever fixtures by sector
    - Database session mocking (PostgreSQL + TimescaleDB)
    - Redis cache mocking
    - SHA-256 provenance validation helpers
    - Decimal arithmetic assertion helpers
    - Performance timing context managers

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-028 Sector Pathway Pack
Tests:   conftest.py (~850 lines)
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
    from engines.sector_classification_engine import (
        SectorClassificationEngine,
        ClassificationInput,
        ClassificationResult,
        SectorCode,
        ClassificationSystem,
        SDAEligibility,
    )
    # Aliases for backward-compatible test references
    SectorClassInput = ClassificationInput
    SectorClassResult = ClassificationResult
    NACECode = SectorCode
    GICSCode = SectorCode
    ISICCode = SectorCode
    _HAS_CLASSIFICATION = True
except ImportError:
    _HAS_CLASSIFICATION = False

try:
    from engines.intensity_calculator_engine import (
        IntensityCalculatorEngine,
        IntensityInput,
        IntensityResult,
        IntensityMetricType,
        TrendDirection,
        DataQualityTier,
    )
    # Aliases
    IntensityMetric = IntensityMetricType
    IntensityTrend = TrendDirection
    DataQualityScore = DataQualityTier
    _HAS_INTENSITY = True
except ImportError:
    _HAS_INTENSITY = False

try:
    from engines.pathway_generator_engine import (
        PathwayGeneratorEngine,
        PathwayInput,
        PathwayResult,
        PathwayPoint,
        ConvergenceModel,
        ClimateScenario,
    )
    # Aliases
    ScenarioType = ClimateScenario
    _HAS_PATHWAY = True
except ImportError:
    _HAS_PATHWAY = False

try:
    from engines.convergence_analyzer_engine import (
        ConvergenceAnalyzerEngine,
        ConvergenceInput,
        ConvergenceResult,
        GapAnalysisPoint,
        CatchUpScenario,
        RiskLevel,
    )
    # Aliases
    GapAnalysis = GapAnalysisPoint
    AccelerationRequired = CatchUpScenario
    _HAS_CONVERGENCE = True
except ImportError:
    _HAS_CONVERGENCE = False

try:
    from engines.technology_roadmap_engine import (
        TechnologyRoadmapEngine,
        TechnologyRoadmapInput,
        TechnologyRoadmapResult,
        MilestoneTrackingResult,
        TechnologyAdoptionCurve,
        CapExPhase,
    )
    # Aliases
    TechRoadmapInput = TechnologyRoadmapInput
    TechRoadmapResult = TechnologyRoadmapResult
    TechnologyMilestone = MilestoneTrackingResult
    AdoptionCurve = TechnologyAdoptionCurve
    _HAS_TECH_ROADMAP = True
except ImportError:
    _HAS_TECH_ROADMAP = False

try:
    from engines.abatement_waterfall_engine import (
        AbatementWaterfallEngine,
        AbatementInput,
        AbatementResult,
        WaterfallLever,
        CostCurvePoint,
        ImplementationPhase,
    )
    # Aliases
    WaterfallInput = AbatementInput
    WaterfallResult = AbatementResult
    AbatementLever = WaterfallLever
    LeverSequence = ImplementationPhase
    _HAS_WATERFALL = True
except ImportError:
    _HAS_WATERFALL = False

try:
    from engines.sector_benchmark_engine import (
        SectorBenchmarkEngine,
        BenchmarkInput,
        BenchmarkResult,
        PercentileRanking,
        GapToLeader,
        IEAPathwayBenchmark,
    )
    # Aliases
    PeerComparison = PercentileRanking
    LeaderComparison = GapToLeader
    PathwayAlignment = IEAPathwayBenchmark
    _HAS_BENCHMARK = True
except ImportError:
    _HAS_BENCHMARK = False

try:
    from engines.scenario_comparison_engine import (
        ScenarioComparisonEngine,
        ComparisonInput,
        ComparisonResult,
        ScenarioPairDelta,
        ScenarioRiskReturn,
        OptimalPathwayRecommendation,
    )
    # Aliases
    ScenarioCompInput = ComparisonInput
    ScenarioCompResult = ComparisonResult
    ScenarioDelta = ScenarioPairDelta
    RiskReturnProfile = ScenarioRiskReturn
    OptimalPathway = OptimalPathwayRecommendation
    _HAS_SCENARIO_COMP = True
except ImportError:
    _HAS_SCENARIO_COMP = False


# ---------------------------------------------------------------------------
# Sector Constants
# ---------------------------------------------------------------------------

SDA_SECTORS = [
    "power_generation", "steel", "cement", "aluminum",
    "pulp_paper", "chemicals", "aviation", "shipping",
    "road_transport", "rail", "buildings_residential",
    "buildings_commercial",
]

EXTENDED_SECTORS = [
    "agriculture", "food_beverage", "oil_gas_upstream", "cross_sector",
]

ALL_SECTORS = SDA_SECTORS + EXTENDED_SECTORS

CONVERGENCE_MODELS = ["linear", "exponential", "s_curve", "stepped"]

SCENARIO_TYPES = ["nze_15c", "wb2c", "2c", "aps", "steps"]

INTENSITY_METRICS = {
    "power_generation": {"unit": "gCO2/kWh", "base_2020": Decimal("450"), "target_2050": Decimal("0")},
    "steel": {"unit": "tCO2e/tonne", "base_2020": Decimal("2.10"), "target_2050": Decimal("0.20")},
    "cement": {"unit": "tCO2e/tonne", "base_2020": Decimal("0.700"), "target_2050": Decimal("0.120")},
    "aluminum": {"unit": "tCO2e/tonne", "base_2020": Decimal("12.50"), "target_2050": Decimal("1.50")},
    "pulp_paper": {"unit": "tCO2e/tonne", "base_2020": Decimal("0.450"), "target_2050": Decimal("0.050")},
    "chemicals": {"unit": "tCO2e/tonne", "base_2020": Decimal("1.80"), "target_2050": Decimal("0.30")},
    "aviation": {"unit": "gCO2/pkm", "base_2020": Decimal("90"), "target_2050": Decimal("10")},
    "shipping": {"unit": "gCO2/tkm", "base_2020": Decimal("10.5"), "target_2050": Decimal("1.0")},
    "road_transport": {"unit": "gCO2/vkm", "base_2020": Decimal("170"), "target_2050": Decimal("0")},
    "rail": {"unit": "gCO2/pkm", "base_2020": Decimal("35"), "target_2050": Decimal("5")},
    "buildings_residential": {"unit": "kgCO2/m2/year", "base_2020": Decimal("25"), "target_2050": Decimal("2")},
    "buildings_commercial": {"unit": "kgCO2/m2/year", "base_2020": Decimal("45"), "target_2050": Decimal("3")},
    "agriculture": {"unit": "tCO2e/tonne_food", "base_2020": Decimal("0.85"), "target_2050": Decimal("0.25")},
    "food_beverage": {"unit": "tCO2e/tonne_product", "base_2020": Decimal("0.55"), "target_2050": Decimal("0.10")},
    "oil_gas_upstream": {"unit": "gCO2/MJ", "base_2020": Decimal("15.0"), "target_2050": Decimal("2.0")},
}

NACE_SECTOR_MAP = {
    "D35.11": "power_generation",
    "C24.10": "steel",
    "C23.51": "cement",
    "C24.42": "aluminum",
    "C17.11": "pulp_paper",
    "C20.11": "chemicals",
    "H51.10": "aviation",
    "H50.10": "shipping",
    "H49.10": "road_transport",
    "H49.20": "rail",
    "F41.20": "buildings_residential",
    "L68.20": "buildings_commercial",
    "A01.11": "agriculture",
    "C10.11": "food_beverage",
    "B06.10": "oil_gas_upstream",
}

IEA_MILESTONES = {
    "power_generation": [
        {"year": 2025, "milestone": "No new unabated coal plants"},
        {"year": 2030, "milestone": "Advanced economies phase out unabated coal"},
        {"year": 2035, "milestone": "All electricity generation is net-zero in OECD"},
        {"year": 2040, "milestone": "Global electricity is net-zero"},
        {"year": 2050, "milestone": "Almost all electricity from renewables/nuclear"},
    ],
    "steel": [
        {"year": 2025, "milestone": "Near-zero steel plants operational"},
        {"year": 2030, "milestone": "150 Mt near-zero steel production"},
        {"year": 2040, "milestone": "50% of steel production near-zero"},
        {"year": 2050, "milestone": "All primary steel near-zero"},
    ],
    "cement": [
        {"year": 2030, "milestone": "Clinker-to-cement ratio below 0.65"},
        {"year": 2035, "milestone": "All new kilns CCS-ready"},
        {"year": 2050, "milestone": "100% of cement plants with CCUS"},
    ],
    "aviation": [
        {"year": 2025, "milestone": "SAF 2% of aviation fuel"},
        {"year": 2030, "milestone": "SAF 10% of aviation fuel"},
        {"year": 2040, "milestone": "SAF 45% of aviation fuel"},
        {"year": 2050, "milestone": "SAF > 70% of aviation fuel, hydrogen for short-haul"},
    ],
}


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


def assert_decimal_positive(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is positive."""
    assert value > Decimal("0"), (
        f"Expected positive Decimal{' (' + msg + ')' if msg else ''}, got {value}"
    )


def assert_percentage_range(value: Decimal, msg: str = "") -> None:
    """Assert that a Decimal value is between 0 and 100."""
    assert Decimal("0") <= value <= Decimal("100"), (
        f"Percentage out of range{' (' + msg + ')' if msg else ''}: {value}"
    )


def assert_provenance_hash(result: Any) -> None:
    """Assert that result has a non-empty SHA-256 provenance hash."""
    assert hasattr(result, "provenance_hash"), "Result missing provenance_hash"
    h = result.provenance_hash
    assert isinstance(h, str), "provenance_hash must be a string"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Hash must be hex"


def assert_processing_time(result: Any, max_ms: float = 60000.0) -> None:
    """Assert processing time is within acceptable range."""
    assert hasattr(result, "processing_time_ms"), "Result missing processing_time_ms"
    assert result.processing_time_ms >= 0, "Processing time must be non-negative"
    assert result.processing_time_ms < max_ms, (
        f"Processing time {result.processing_time_ms}ms exceeds {max_ms}ms"
    )


def assert_intensity_accuracy(
    actual: Decimal,
    expected: Decimal,
    max_delta_pct: Decimal = Decimal("2.0"),
    msg: str = "",
) -> None:
    """Assert intensity values are within max_delta_pct of each other (default 2%)."""
    if expected == Decimal("0"):
        assert actual == Decimal("0"), f"Expected zero intensity, got {actual}"
        return
    delta_pct = abs((actual - expected) / expected * Decimal("100"))
    assert delta_pct <= max_delta_pct, (
        f"Intensity accuracy violation{' (' + msg + ')' if msg else ''}: "
        f"actual={actual}, expected={expected}, delta={delta_pct:.4f}%, max={max_delta_pct}%"
    )


def compute_sha256(data: str) -> str:
    """Compute SHA-256 hash of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


@contextmanager
def timed_block(label: str = "", max_seconds: float = 30.0):
    """Context manager that asserts a block completes within max_seconds."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"Block '{label}' took {elapsed:.3f}s, exceeding {max_seconds}s"
    )


def linear_convergence(base: Decimal, target: Decimal, base_year: int,
                        target_year: int, year: int) -> Decimal:
    """Reference implementation of linear convergence for test validation."""
    if year <= base_year:
        return base
    if year >= target_year:
        return target
    fraction = Decimal(str(year - base_year)) / Decimal(str(target_year - base_year))
    return base + (target - base) * fraction


def exponential_convergence(base: Decimal, target: Decimal, base_year: int,
                             target_year: int, year: int) -> Decimal:
    """Reference implementation of exponential convergence for test validation."""
    if year <= base_year:
        return base
    if year >= target_year:
        return target
    if base <= Decimal("0"):
        return target
    k = float(-1 * (target_year - base_year))
    t = float(year - base_year)
    ratio = Decimal(str(math.exp(-t / abs(k) * math.log(float(base / target))))) if target > 0 else Decimal("0")
    return base * ratio


def s_curve_convergence(base: Decimal, target: Decimal, base_year: int,
                         target_year: int, year: int,
                         inflection_year: Optional[int] = None,
                         k: float = 0.3) -> Decimal:
    """Reference implementation of S-curve convergence for test validation."""
    if year <= base_year:
        return base
    if year >= target_year:
        return target
    if inflection_year is None:
        inflection_year = (base_year + target_year) // 2
    t_inf = float(year - inflection_year)
    sigmoid = Decimal(str(1.0 / (1.0 + math.exp(-k * t_inf))))
    return base + (target - base) * sigmoid


# ---------------------------------------------------------------------------
# Fixtures -- Sector Profile Builders
# ---------------------------------------------------------------------------


@pytest.fixture
def power_sector_profile() -> Dict[str, Any]:
    """Build a power generation company profile."""
    return {
        "entity_name": "GreenPowerCo",
        "nace_code": "D35.11",
        "gics_code": "55101010",
        "isic_code": "3510",
        "sector": "power_generation",
        "country": "DE",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("450"),
        "base_year_activity": Decimal("50000000"),
        "base_year_emissions_tco2e": Decimal("22500000"),
        "intensity_unit": "gCO2/kWh",
        "activity_unit": "MWh",
        "generation_mix": {
            "coal": Decimal("30"), "gas": Decimal("25"),
            "nuclear": Decimal("15"), "wind": Decimal("15"),
            "solar": Decimal("10"), "hydro": Decimal("5"),
        },
        "renewable_capacity_mw": Decimal("2500"),
        "total_capacity_mw": Decimal("8000"),
    }


@pytest.fixture
def steel_sector_profile() -> Dict[str, Any]:
    """Build a steel company profile."""
    return {
        "entity_name": "GlobalSteel Corp",
        "nace_code": "C24.10",
        "gics_code": "15104010",
        "isic_code": "2410",
        "sector": "steel",
        "country": "DE",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("2.10"),
        "base_year_activity": Decimal("5000000"),
        "base_year_emissions_tco2e": Decimal("10500000"),
        "intensity_unit": "tCO2e/tonne",
        "activity_unit": "tonne_crude_steel",
        "production_route": {
            "bf_bof": Decimal("70"),
            "eaf_scrap": Decimal("25"),
            "dri": Decimal("5"),
        },
        "scrap_rate_pct": Decimal("25"),
    }


@pytest.fixture
def cement_sector_profile() -> Dict[str, Any]:
    """Build a cement company profile."""
    return {
        "entity_name": "CementWorks International",
        "nace_code": "C23.51",
        "gics_code": "15102010",
        "isic_code": "2394",
        "sector": "cement",
        "country": "IN",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("0.700"),
        "base_year_activity": Decimal("20000000"),
        "base_year_emissions_tco2e": Decimal("14000000"),
        "intensity_unit": "tCO2e/tonne",
        "activity_unit": "tonne_cement",
        "clinker_ratio": Decimal("0.75"),
        "alt_fuel_share_pct": Decimal("10"),
    }


@pytest.fixture
def aviation_sector_profile() -> Dict[str, Any]:
    """Build an aviation company profile."""
    return {
        "entity_name": "SkyTransit Airlines",
        "nace_code": "H51.10",
        "gics_code": "20302010",
        "isic_code": "5110",
        "sector": "aviation",
        "country": "GB",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("90"),
        "base_year_activity": Decimal("200000000000"),
        "base_year_emissions_tco2e": Decimal("18000000"),
        "intensity_unit": "gCO2/pkm",
        "activity_unit": "pkm",
        "fleet_size": 250,
        "avg_fleet_age_years": Decimal("8.5"),
        "saf_blend_pct": Decimal("1"),
        "load_factor_pct": Decimal("82"),
    }


@pytest.fixture
def buildings_sector_profile() -> Dict[str, Any]:
    """Build a commercial buildings company profile."""
    return {
        "entity_name": "RealEstate Holdings",
        "nace_code": "L68.20",
        "gics_code": "60101010",
        "isic_code": "6820",
        "sector": "buildings_commercial",
        "country": "US",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("45"),
        "base_year_activity": Decimal("5000000"),
        "base_year_emissions_tco2e": Decimal("225000"),
        "intensity_unit": "kgCO2/m2/year",
        "activity_unit": "m2",
        "building_types": {"office": Decimal("60"), "retail": Decimal("25"), "warehouse": Decimal("15")},
        "avg_building_age_years": Decimal("22"),
    }


@pytest.fixture
def aluminum_sector_profile() -> Dict[str, Any]:
    """Build an aluminum company profile."""
    return {
        "entity_name": "AluGlobal Corp",
        "nace_code": "C24.42",
        "gics_code": "15104020",
        "isic_code": "2420",
        "sector": "aluminum",
        "country": "NO",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("12.50"),
        "base_year_activity": Decimal("1000000"),
        "base_year_emissions_tco2e": Decimal("12500000"),
        "intensity_unit": "tCO2e/tonne",
        "activity_unit": "tonne_aluminum",
        "primary_pct": Decimal("70"),
        "secondary_pct": Decimal("30"),
    }


@pytest.fixture
def shipping_sector_profile() -> Dict[str, Any]:
    """Build a shipping company profile."""
    return {
        "entity_name": "OceanFreight Lines",
        "nace_code": "H50.10",
        "gics_code": "20303010",
        "isic_code": "5012",
        "sector": "shipping",
        "country": "SG",
        "base_year": 2020,
        "target_year": 2050,
        "base_year_intensity": Decimal("10.5"),
        "base_year_activity": Decimal("500000000000"),
        "base_year_emissions_tco2e": Decimal("5250000"),
        "intensity_unit": "gCO2/tkm",
        "activity_unit": "tkm",
        "fleet_dwt": Decimal("12000000"),
        "avg_speed_knots": Decimal("14"),
    }


@pytest.fixture
def multi_sector_company_profile() -> Dict[str, Any]:
    """Build a company operating across multiple sectors."""
    return {
        "entity_name": "DiversifiedIndustrial Corp",
        "nace_codes": ["C24.10", "C23.51", "D35.11"],
        "gics_code": "20105010",
        "country": "JP",
        "base_year": 2020,
        "target_year": 2050,
        "revenue_breakdown": {
            "steel": {"share_pct": Decimal("45"), "emissions_tco2e": Decimal("4500000")},
            "cement": {"share_pct": Decimal("30"), "emissions_tco2e": Decimal("2100000")},
            "power_generation": {"share_pct": Decimal("25"), "emissions_tco2e": Decimal("1800000")},
        },
        "total_emissions_tco2e": Decimal("8400000"),
        "primary_sector": "steel",
    }


@pytest.fixture(params=SDA_SECTORS, ids=SDA_SECTORS)
def sda_sector(request) -> str:
    """Parameterized fixture yielding each SDA sector."""
    return request.param


@pytest.fixture(params=ALL_SECTORS, ids=ALL_SECTORS)
def any_sector(request) -> str:
    """Parameterized fixture yielding each supported sector."""
    return request.param


@pytest.fixture(params=CONVERGENCE_MODELS, ids=CONVERGENCE_MODELS)
def convergence_model(request) -> str:
    """Parameterized fixture yielding each convergence model."""
    return request.param


@pytest.fixture(params=SCENARIO_TYPES, ids=SCENARIO_TYPES)
def scenario_type(request) -> str:
    """Parameterized fixture yielding each scenario type."""
    return request.param


# ---------------------------------------------------------------------------
# Fixtures -- SBTi SDA Reference Data
# ---------------------------------------------------------------------------


@pytest.fixture
def sbti_sda_reference_data() -> Dict[str, Any]:
    """Build SBTi SDA reference pathway data for test validation."""
    return {
        "power_generation": {
            "base_year_global_intensity": Decimal("490"),
            "2030_target_intensity": Decimal("138"),
            "2050_target_intensity": Decimal("0"),
            "annual_reduction_rate": Decimal("7.1"),
            "coverage_requirement_pct": Decimal("95"),
        },
        "steel": {
            "base_year_global_intensity": Decimal("1.89"),
            "2030_target_intensity": Decimal("1.48"),
            "2050_target_intensity": Decimal("0.20"),
            "annual_reduction_rate": Decimal("3.0"),
            "coverage_requirement_pct": Decimal("95"),
        },
        "cement": {
            "base_year_global_intensity": Decimal("0.611"),
            "2030_target_intensity": Decimal("0.503"),
            "2050_target_intensity": Decimal("0.120"),
            "annual_reduction_rate": Decimal("2.5"),
            "coverage_requirement_pct": Decimal("95"),
        },
        "aviation": {
            "base_year_global_intensity": Decimal("88"),
            "2030_target_intensity": Decimal("73"),
            "2050_target_intensity": Decimal("10"),
            "annual_reduction_rate": Decimal("2.8"),
            "coverage_requirement_pct": Decimal("95"),
        },
    }


@pytest.fixture
def iea_nze_reference_data() -> Dict[str, Any]:
    """Build IEA NZE 2050 reference pathway data for test validation."""
    return {
        "power_generation": {
            "nze_15c": {"2030": Decimal("140"), "2040": Decimal("30"), "2050": Decimal("0")},
            "aps": {"2030": Decimal("280"), "2040": Decimal("180"), "2050": Decimal("80")},
            "steps": {"2030": Decimal("360"), "2040": Decimal("300"), "2050": Decimal("250")},
        },
        "steel": {
            "nze_15c": {"2030": Decimal("1.50"), "2040": Decimal("0.80"), "2050": Decimal("0.20")},
            "aps": {"2030": Decimal("1.70"), "2040": Decimal("1.30"), "2050": Decimal("0.80")},
            "steps": {"2030": Decimal("1.85"), "2040": Decimal("1.70"), "2050": Decimal("1.50")},
        },
        "cement": {
            "nze_15c": {"2030": Decimal("0.50"), "2040": Decimal("0.28"), "2050": Decimal("0.12")},
            "aps": {"2030": Decimal("0.58"), "2040": Decimal("0.42"), "2050": Decimal("0.30")},
            "steps": {"2030": Decimal("0.63"), "2040": Decimal("0.55"), "2050": Decimal("0.48")},
        },
    }


# ---------------------------------------------------------------------------
# Fixtures -- Technology Data
# ---------------------------------------------------------------------------


@pytest.fixture
def technology_data() -> Dict[str, Any]:
    """Build sample technology data for roadmap testing."""
    return {
        "power_generation": {
            "technologies": [
                {"name": "Solar PV", "trl": 9, "capex_usd_per_kw": Decimal("800"),
                 "learning_rate_pct": Decimal("20"), "adoption_2020_pct": Decimal("10"),
                 "adoption_2050_pct": Decimal("40")},
                {"name": "Onshore Wind", "trl": 9, "capex_usd_per_kw": Decimal("1200"),
                 "learning_rate_pct": Decimal("15"), "adoption_2020_pct": Decimal("15"),
                 "adoption_2050_pct": Decimal("30")},
                {"name": "Battery Storage", "trl": 8, "capex_usd_per_kwh": Decimal("200"),
                 "learning_rate_pct": Decimal("18"), "adoption_2020_pct": Decimal("2"),
                 "adoption_2050_pct": Decimal("25")},
                {"name": "Green Hydrogen", "trl": 7, "capex_usd_per_kw": Decimal("2500"),
                 "learning_rate_pct": Decimal("12"), "adoption_2020_pct": Decimal("0"),
                 "adoption_2050_pct": Decimal("10")},
            ],
        },
        "steel": {
            "technologies": [
                {"name": "EAF with Scrap", "trl": 9, "capex_usd_per_tpa": Decimal("350"),
                 "learning_rate_pct": Decimal("8"), "adoption_2020_pct": Decimal("25"),
                 "adoption_2050_pct": Decimal("50")},
                {"name": "DRI with Green H2", "trl": 7, "capex_usd_per_tpa": Decimal("800"),
                 "learning_rate_pct": Decimal("15"), "adoption_2020_pct": Decimal("0"),
                 "adoption_2050_pct": Decimal("30")},
                {"name": "CCS for BF-BOF", "trl": 6, "capex_usd_per_tpa": Decimal("200"),
                 "learning_rate_pct": Decimal("10"), "adoption_2020_pct": Decimal("0"),
                 "adoption_2050_pct": Decimal("15")},
            ],
        },
    }


# ---------------------------------------------------------------------------
# Fixtures -- Abatement Levers
# ---------------------------------------------------------------------------


@pytest.fixture
def abatement_levers_power() -> List[Dict[str, Any]]:
    """Build abatement levers for power sector."""
    return [
        {"name": "Renewable Capacity Expansion", "reduction_pct": Decimal("35"),
         "cost_usd_per_tco2e": Decimal("-20"), "start_year": 2024, "end_year": 2035},
        {"name": "Coal Plant Phase-Out", "reduction_pct": Decimal("25"),
         "cost_usd_per_tco2e": Decimal("30"), "start_year": 2025, "end_year": 2035},
        {"name": "Battery Storage Deployment", "reduction_pct": Decimal("10"),
         "cost_usd_per_tco2e": Decimal("45"), "start_year": 2026, "end_year": 2040},
        {"name": "Gas Plant Efficiency", "reduction_pct": Decimal("8"),
         "cost_usd_per_tco2e": Decimal("15"), "start_year": 2024, "end_year": 2030},
        {"name": "Nuclear SMR Deployment", "reduction_pct": Decimal("12"),
         "cost_usd_per_tco2e": Decimal("80"), "start_year": 2030, "end_year": 2045},
        {"name": "CCS for Fossil Generation", "reduction_pct": Decimal("8"),
         "cost_usd_per_tco2e": Decimal("100"), "start_year": 2028, "end_year": 2045},
        {"name": "Smart Grid / Demand Response", "reduction_pct": Decimal("2"),
         "cost_usd_per_tco2e": Decimal("10"), "start_year": 2024, "end_year": 2035},
    ]


@pytest.fixture
def abatement_levers_steel() -> List[Dict[str, Any]]:
    """Build abatement levers for steel sector."""
    return [
        {"name": "BF Efficiency Improvements", "reduction_pct": Decimal("10"),
         "cost_usd_per_tco2e": Decimal("25"), "start_year": 2024, "end_year": 2030},
        {"name": "EAF Transition", "reduction_pct": Decimal("25"),
         "cost_usd_per_tco2e": Decimal("40"), "start_year": 2025, "end_year": 2040},
        {"name": "Green H2 DRI", "reduction_pct": Decimal("30"),
         "cost_usd_per_tco2e": Decimal("80"), "start_year": 2028, "end_year": 2045},
        {"name": "CCS for Integrated Plants", "reduction_pct": Decimal("15"),
         "cost_usd_per_tco2e": Decimal("90"), "start_year": 2030, "end_year": 2045},
        {"name": "Scrap Rate Increase", "reduction_pct": Decimal("12"),
         "cost_usd_per_tco2e": Decimal("15"), "start_year": 2024, "end_year": 2035},
        {"name": "Waste Heat Recovery", "reduction_pct": Decimal("8"),
         "cost_usd_per_tco2e": Decimal("20"), "start_year": 2024, "end_year": 2030},
    ]


# ---------------------------------------------------------------------------
# Fixtures -- Peer / Benchmark Data
# ---------------------------------------------------------------------------


@pytest.fixture
def peer_benchmark_data() -> Dict[str, Any]:
    """Build peer company benchmark data."""
    return {
        "power_generation": [
            {"name": "CompanyA", "intensity": Decimal("380"), "sbti_validated": True, "region": "EU"},
            {"name": "CompanyB", "intensity": Decimal("420"), "sbti_validated": True, "region": "EU"},
            {"name": "CompanyC", "intensity": Decimal("520"), "sbti_validated": False, "region": "APAC"},
            {"name": "CompanyD", "intensity": Decimal("280"), "sbti_validated": True, "region": "EU"},
            {"name": "CompanyE", "intensity": Decimal("610"), "sbti_validated": False, "region": "Americas"},
            {"name": "CompanyF", "intensity": Decimal("350"), "sbti_validated": True, "region": "EU"},
            {"name": "CompanyG", "intensity": Decimal("480"), "sbti_validated": False, "region": "APAC"},
            {"name": "CompanyH", "intensity": Decimal("190"), "sbti_validated": True, "region": "EU"},
            {"name": "CompanyI", "intensity": Decimal("550"), "sbti_validated": False, "region": "Americas"},
            {"name": "CompanyJ", "intensity": Decimal("310"), "sbti_validated": True, "region": "EU"},
        ],
        "steel": [
            {"name": "SteelA", "intensity": Decimal("1.85"), "sbti_validated": True, "region": "EU"},
            {"name": "SteelB", "intensity": Decimal("2.20"), "sbti_validated": False, "region": "APAC"},
            {"name": "SteelC", "intensity": Decimal("1.60"), "sbti_validated": True, "region": "EU"},
            {"name": "SteelD", "intensity": Decimal("2.50"), "sbti_validated": False, "region": "Americas"},
            {"name": "SteelE", "intensity": Decimal("1.95"), "sbti_validated": True, "region": "EU"},
        ],
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
# Fixtures -- Mock External API Clients
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_sbti_api():
    """Create a mock SBTi SDA API client."""
    client = MagicMock()
    client.get_sector_pathway = AsyncMock(return_value={
        "sector": "power_generation",
        "base_year": 2020,
        "target_year": 2050,
        "pathway_points": [
            {"year": 2020, "intensity": Decimal("490")},
            {"year": 2030, "intensity": Decimal("138")},
            {"year": 2050, "intensity": Decimal("0")},
        ],
    })
    client.validate_target = AsyncMock(return_value={
        "valid": True,
        "alignment": "1.5C",
        "criteria_met": ["coverage", "ambition", "timeframe"],
    })
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_iea_api():
    """Create a mock IEA NZE data API client."""
    client = MagicMock()
    client.get_sector_pathway = AsyncMock(return_value={
        "sector": "power_generation",
        "scenario": "nze_15c",
        "pathway_points": [
            {"year": 2020, "intensity": Decimal("490")},
            {"year": 2025, "intensity": Decimal("340")},
            {"year": 2030, "intensity": Decimal("140")},
            {"year": 2040, "intensity": Decimal("30")},
            {"year": 2050, "intensity": Decimal("0")},
        ],
    })
    client.get_milestones = AsyncMock(return_value=[
        {"year": 2025, "milestone": "No new unabated coal plants", "status": "on_track"},
        {"year": 2030, "milestone": "Advanced economies phase out unabated coal"},
    ])
    client.is_connected = MagicMock(return_value=True)
    return client


@pytest.fixture
def mock_ipcc_api():
    """Create a mock IPCC AR6 data API client."""
    client = MagicMock()
    client.get_emission_factors = AsyncMock(return_value={
        "gwp100_co2": Decimal("1"), "gwp100_ch4": Decimal("28"),
        "gwp100_n2o": Decimal("265"), "gwp100_sf6": Decimal("23500"),
    })
    client.get_sector_factors = AsyncMock(return_value={
        "sector": "power_generation",
        "factors": {"coal": Decimal("0.95"), "gas": Decimal("0.40")},
    })
    client.is_connected = MagicMock(return_value=True)
    return client


# ---------------------------------------------------------------------------
# Extended sector profile builders
# ---------------------------------------------------------------------------

@pytest.fixture(params=SDA_SECTORS)
def any_sda_sector_profile(request):
    """Parametrized fixture providing profiles for all 12 SDA sectors."""
    sector = request.param
    metrics = INTENSITY_METRICS.get(sector, {"base_2020": Decimal("1.0"), "target_2050": Decimal("0.1")})
    return {
        "sector": sector,
        "base_year": 2020,
        "base_intensity": metrics["base_2020"],
        "target_year": 2050,
        "target_intensity": metrics["target_2050"],
        "nace_codes": {
            "power_generation": "D35.11", "steel": "C24.10", "cement": "C23.51",
            "aluminum": "C24.42", "pulp_paper": "C17.11", "chemicals": "C20.11",
            "aviation": "H51.10", "shipping": "H50.10", "road_transport": "H49.10",
            "rail": "H49.20", "buildings_residential": "F41.10",
            "buildings_commercial": "F41.20",
        }.get(sector, "Z99.99"),
    }


@pytest.fixture
def multi_scenario_config():
    """Configuration for multi-scenario comparison testing."""
    return {
        "scenarios": SCENARIO_TYPES,
        "base_year": 2020,
        "target_year": 2050,
        "convergence_models": CONVERGENCE_MODELS,
    }


@pytest.fixture
def abatement_levers_steel():
    """Steel sector abatement levers fixture."""
    return [
        {"name": "Energy Efficiency", "max_reduction_pct": Decimal("15"),
         "cost_usd_per_tco2": Decimal("10"), "start_year": 2020, "end_year": 2035},
        {"name": "Scrap-based EAF", "max_reduction_pct": Decimal("25"),
         "cost_usd_per_tco2": Decimal("20"), "start_year": 2020, "end_year": 2040},
        {"name": "Hydrogen DRI", "max_reduction_pct": Decimal("35"),
         "cost_usd_per_tco2": Decimal("80"), "start_year": 2028, "end_year": 2050},
        {"name": "CCS", "max_reduction_pct": Decimal("20"),
         "cost_usd_per_tco2": Decimal("60"), "start_year": 2030, "end_year": 2050},
    ]


@pytest.fixture
def abatement_levers_cement():
    """Cement sector abatement levers fixture."""
    return [
        {"name": "Clinker Substitution", "max_reduction_pct": Decimal("20"),
         "cost_usd_per_tco2": Decimal("5"), "start_year": 2020, "end_year": 2040},
        {"name": "Alternative Fuels", "max_reduction_pct": Decimal("15"),
         "cost_usd_per_tco2": Decimal("12"), "start_year": 2020, "end_year": 2035},
        {"name": "Kiln Efficiency", "max_reduction_pct": Decimal("10"),
         "cost_usd_per_tco2": Decimal("8"), "start_year": 2020, "end_year": 2030},
        {"name": "Cement CCS", "max_reduction_pct": Decimal("40"),
         "cost_usd_per_tco2": Decimal("70"), "start_year": 2028, "end_year": 2050},
    ]


@pytest.fixture
def technology_data_power():
    """Power sector technology data fixture."""
    return [
        {"name": "Solar PV", "trl": 9, "adoption_2020_pct": Decimal("3"),
         "adoption_2050_pct": Decimal("45"), "learning_rate": Decimal("0.28")},
        {"name": "Onshore Wind", "trl": 9, "adoption_2020_pct": Decimal("6"),
         "adoption_2050_pct": Decimal("30"), "learning_rate": Decimal("0.15")},
        {"name": "Offshore Wind", "trl": 8, "adoption_2020_pct": Decimal("0.3"),
         "adoption_2050_pct": Decimal("15"), "learning_rate": Decimal("0.20")},
        {"name": "Nuclear SMR", "trl": 6, "adoption_2020_pct": Decimal("0"),
         "adoption_2050_pct": Decimal("5"), "learning_rate": Decimal("0.05")},
        {"name": "Battery Storage", "trl": 8, "adoption_2020_pct": Decimal("0.5"),
         "adoption_2050_pct": Decimal("20"), "learning_rate": Decimal("0.18")},
    ]


@pytest.fixture
def technology_data_steel():
    """Steel sector technology data fixture."""
    return [
        {"name": "DRI-EAF", "trl": 9, "adoption_2020_pct": Decimal("30"),
         "adoption_2050_pct": Decimal("65"), "learning_rate": Decimal("0.08")},
        {"name": "Hydrogen DRI", "trl": 7, "adoption_2020_pct": Decimal("0"),
         "adoption_2050_pct": Decimal("25"), "learning_rate": Decimal("0.12")},
        {"name": "Steel CCS", "trl": 6, "adoption_2020_pct": Decimal("0"),
         "adoption_2050_pct": Decimal("10"), "learning_rate": Decimal("0.10")},
    ]


# ---------------------------------------------------------------------------
# Helper functions for test result validation
# ---------------------------------------------------------------------------

def assert_pathway_monotonic(pathway_points, allow_plateau=True):
    """Assert that pathway intensity decreases (or stays flat) over time."""
    if not pathway_points:
        return
    for i in range(len(pathway_points) - 1):
        current = getattr(pathway_points[i], "intensity", None)
        next_val = getattr(pathway_points[i + 1], "intensity", None)
        if current is not None and next_val is not None:
            if allow_plateau:
                assert next_val <= current + Decimal("0.001"), \
                    f"Pathway not monotonic: {current} -> {next_val}"
            else:
                assert next_val < current, \
                    f"Pathway not strictly decreasing: {current} -> {next_val}"


def assert_waterfall_valid(result):
    """Assert waterfall result is well-formed."""
    assert result is not None
    assert hasattr(result, "lever_contributions")
    assert len(result.lever_contributions) > 0
    total_pct = sum(lc.reduction_pct for lc in result.lever_contributions)
    assert total_pct > Decimal("0")
    assert total_pct <= Decimal("110")  # Allow slight over-commitment


def assert_benchmark_valid(result):
    """Assert benchmark result is well-formed."""
    assert result is not None
    if hasattr(result, "peer_comparison"):
        assert result.peer_comparison is not None
        if hasattr(result.peer_comparison, "percentile"):
            assert Decimal("0") <= result.peer_comparison.percentile <= Decimal("100")


def assert_scenario_results_valid(result, expected_count=None):
    """Assert scenario comparison result is well-formed."""
    assert result is not None
    assert hasattr(result, "scenario_results")
    assert len(result.scenario_results) > 0
    if expected_count is not None:
        assert len(result.scenario_results) == expected_count
    for sr in result.scenario_results:
        assert hasattr(sr, "scenario")
        assert sr.scenario in SCENARIO_TYPES


def assert_roadmap_valid(result):
    """Assert technology roadmap result is well-formed."""
    assert result is not None
    if hasattr(result, "technologies") or hasattr(result, "adoption_curves"):
        tech_field = getattr(result, "technologies", None) or \
                     getattr(result, "adoption_curves", [])
        assert len(tech_field) > 0


def assert_convergence_valid(result):
    """Assert convergence analysis result is well-formed."""
    assert result is not None
    assert hasattr(result, "gap_analysis")
    assert hasattr(result, "risk_level")
    if hasattr(result.gap_analysis, "intensity_gap"):
        assert isinstance(result.gap_analysis.intensity_gap, Decimal)


def assert_classification_valid(result):
    """Assert classification result is well-formed."""
    assert result is not None
    if hasattr(result, "primary_sector"):
        assert isinstance(result.primary_sector, str)
        assert len(result.primary_sector) > 0


# ---------------------------------------------------------------------------
# Extended Peer Data Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def power_peer_data_10():
    """10 power generation peer companies for benchmarking."""
    return [
        {"name": f"PowerCo_{i}", "intensity": Decimal(str(0.3 + 0.05 * i)),
         "sbti_validated": i % 2 == 0, "region": ["EU", "US", "China", "Japan", "India"][i % 5],
         "reduction_rate": Decimal(str(1.5 + 0.5 * i))}
        for i in range(10)
    ]


@pytest.fixture
def steel_peer_data_10():
    """10 steel sector peer companies for benchmarking."""
    return [
        {"name": f"SteelCo_{i}", "intensity": Decimal(str(1.5 + 0.15 * i)),
         "sbti_validated": i % 3 == 0, "region": ["EU", "US", "China", "India", "Brazil"][i % 5],
         "reduction_rate": Decimal(str(1.0 + 0.3 * i))}
        for i in range(10)
    ]


@pytest.fixture
def cement_peer_data_10():
    """10 cement sector peer companies for benchmarking."""
    return [
        {"name": f"CementCo_{i}", "intensity": Decimal(str(0.45 + 0.04 * i)),
         "sbti_validated": i % 2 == 0, "region": ["EU", "US", "China", "India", "Turkey"][i % 5]}
        for i in range(10)
    ]


@pytest.fixture
def multi_sector_peer_data():
    """Peer data across multiple sectors."""
    sectors = ["power_generation", "steel", "cement", "aviation"]
    result = {}
    for sector in sectors:
        metrics = INTENSITY_METRICS.get(sector, {"base_2020": Decimal("1.0")})
        result[sector] = [
            {"name": f"{sector}_Peer_{i}",
             "intensity": metrics["base_2020"] * Decimal(str(0.5 + 0.1 * i)),
             "sbti_validated": i % 2 == 0, "region": "EU"}
            for i in range(5)
        ]
    return result


# ---------------------------------------------------------------------------
# Pathway Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pathway_points():
    """Sample pathway points for testing."""
    return [
        {"year": 2020, "intensity": Decimal("0.491")},
        {"year": 2025, "intensity": Decimal("0.380")},
        {"year": 2030, "intensity": Decimal("0.280")},
        {"year": 2035, "intensity": Decimal("0.190")},
        {"year": 2040, "intensity": Decimal("0.110")},
        {"year": 2045, "intensity": Decimal("0.050")},
        {"year": 2050, "intensity": Decimal("0.010")},
    ]


@pytest.fixture
def nze_pathway_power():
    """NZE 1.5C pathway for power generation."""
    return {
        "sector": "power_generation",
        "scenario": "nze_15c",
        "base_year": 2020,
        "target_year": 2050,
        "base_intensity": Decimal("0.491"),
        "target_intensity": Decimal("0.010"),
        "milestones": [
            {"year": 2025, "milestone": "No new unabated coal plants"},
            {"year": 2030, "milestone": "Coal phase-out OECD"},
            {"year": 2035, "milestone": "Net-zero electricity OECD"},
            {"year": 2040, "milestone": "Net-zero electricity global"},
        ],
    }


@pytest.fixture
def nze_pathway_steel():
    """NZE 1.5C pathway for steel."""
    return {
        "sector": "steel",
        "scenario": "nze_15c",
        "base_year": 2020,
        "target_year": 2050,
        "base_intensity": Decimal("2.10"),
        "target_intensity": Decimal("0.42"),
        "milestones": [
            {"year": 2025, "milestone": "Hydrogen DRI pilots operational"},
            {"year": 2030, "milestone": "Commercial hydrogen-based steelmaking"},
            {"year": 2040, "milestone": "50% near-zero steel production"},
        ],
    }


# ---------------------------------------------------------------------------
# Scenario Configuration Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def all_scenarios_config():
    """Configuration for all 5 climate scenarios."""
    return {
        "nze_15c": {
            "name": "Net Zero Emissions by 2050",
            "temperature_target": Decimal("1.5"),
            "ambition": "highest",
            "source": "IEA",
        },
        "wb2c": {
            "name": "Well Below 2C",
            "temperature_target": Decimal("1.7"),
            "ambition": "high",
            "source": "IPCC AR6",
        },
        "2c": {
            "name": "2 Degrees Celsius",
            "temperature_target": Decimal("2.0"),
            "ambition": "medium",
            "source": "IPCC AR6",
        },
        "aps": {
            "name": "Announced Pledges Scenario",
            "temperature_target": Decimal("2.5"),
            "ambition": "low",
            "source": "IEA",
        },
        "steps": {
            "name": "Stated Policies Scenario",
            "temperature_target": Decimal("3.0"),
            "ambition": "baseline",
            "source": "IEA",
        },
    }


# ---------------------------------------------------------------------------
# Test Data Generators
# ---------------------------------------------------------------------------

def generate_sector_intensity_series(sector, start_year=2020, end_year=2050, step=1):
    """Generate a time series of intensity values for a sector."""
    metrics = INTENSITY_METRICS.get(sector, {"base_2020": Decimal("1.0"), "target_2050": Decimal("0.1")})
    base = metrics["base_2020"]
    target = metrics["target_2050"]
    total_years = end_year - start_year
    series = []
    for year in range(start_year, end_year + 1, step):
        progress = Decimal(str((year - start_year) / total_years))
        intensity = base * (Decimal("1") - progress) + target * progress
        series.append({"year": year, "intensity": max(intensity, Decimal("0"))})
    return series


def generate_peer_set(sector, count=10, region="EU"):
    """Generate a peer company dataset for benchmarking."""
    metrics = INTENSITY_METRICS.get(sector, {"base_2020": Decimal("1.0")})
    return [
        {"name": f"{sector}_Peer_{i}",
         "intensity": metrics["base_2020"] * Decimal(str(0.3 + 0.07 * i)),
         "sbti_validated": i % 2 == 0,
         "region": region,
         "reduction_rate": Decimal(str(1 + 0.5 * i))}
        for i in range(count)
    ]


def generate_technology_set(sector, count=3):
    """Generate a technology dataset for roadmap testing."""
    return [
        {"name": f"Tech_{sector}_{i}", "trl": min(9, 5 + i),
         "adoption_2020_pct": Decimal(str(i * 2)),
         "adoption_2050_pct": Decimal(str(10 + i * 15)),
         "learning_rate": Decimal(str(0.05 + 0.03 * i))}
        for i in range(count)
    ]
