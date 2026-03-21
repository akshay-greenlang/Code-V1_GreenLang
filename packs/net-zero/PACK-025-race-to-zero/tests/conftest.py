# -*- coding: utf-8 -*-
"""
Shared test infrastructure for PACK-025 Race to Zero Pack.
=============================================================

Provides pytest fixtures for all 10 engines, 8 workflows, sample
data builders, database mock setup, and common test utilities.

Adds the pack root to sys.path so ``from engines.X import Y`` works
in every test module without requiring an installed package.

Fixtures cover:
    - Engine instantiation (10 engines)
    - Workflow instantiation (8 workflows)
    - Sample input builders for every engine
    - Database session mocking
    - Redis cache mocking
    - SHA-256 provenance validation helpers
    - Decimal arithmetic assertion helpers
    - Performance timing context managers

Author:  GreenLang Platform Team
Date:    March 2026
Pack:    PACK-025 Race to Zero Pack
Tests:   conftest.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Path setup -- ensure pack root is importable
# ---------------------------------------------------------------------------

_PACK_ROOT = Path(__file__).resolve().parents[1]
if str(_PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACK_ROOT))


# ---------------------------------------------------------------------------
# Engine imports
# ---------------------------------------------------------------------------

from engines.pledge_commitment_engine import (
    ActorType,
    CriterionStatus,
    EligibilityStatus,
    PartnerInitiative,
    PledgeCommitmentEngine,
    PledgeCommitmentInput,
    PledgeCriterionInput,
    PledgeQuality,
    PartnerAlignmentInput,
    CRITERION_IDS,
    CRITERION_WEIGHTS,
    CORE_CRITERIA,
)

from engines.starting_line_engine import (
    ComplianceStatus,
    StartingLineEngine,
    StartingLineInput,
    SubCriterionInput,
    SubCriterionStatus,
    SUB_CRITERIA,
    SUB_CRITERION_IDS,
    StartingLineCategory,
)

from engines.interim_target_engine import (
    InterimTargetEngine,
    InterimTargetInput,
    PathwayAlignment,
    ScopeTargetInput,
    TargetMethodology,
    TargetType,
    ComplianceLevel,
    IPCC_MIN_REDUCTION_PCT,
    R2Z_TARGET_REDUCTION_PCT,
    SBTI_1_5C_ANNUAL_RATE,
)

from engines.action_plan_engine import (
    ActionPlanEngine,
    ActionPlanInput,
    DecarbonizationActionInput,
    SectionInput,
    PlanSection,
    SectionRating,
    ActionCategory,
    PlanQuality,
    SECTION_IDS,
    SECTION_WEIGHTS,
)

from engines.progress_tracking_engine import ProgressTrackingEngine
from engines.sector_pathway_engine import SectorPathwayEngine
from engines.partnership_scoring_engine import PartnershipScoringEngine
from engines.campaign_reporting_engine import CampaignReportingEngine
from engines.credibility_assessment_engine import CredibilityAssessmentEngine
from engines.race_readiness_engine import RaceReadinessEngine


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


def assert_provenance_hash(result: Any) -> None:
    """Assert that result has a non-empty SHA-256 provenance hash."""
    assert hasattr(result, "provenance_hash"), "Result missing provenance_hash"
    h = result.provenance_hash
    assert isinstance(h, str), "provenance_hash must be a string"
    assert len(h) == 64, f"SHA-256 hash must be 64 chars, got {len(h)}"
    assert all(c in "0123456789abcdef" for c in h), "Hash must be hex"


def assert_processing_time(result: Any, max_ms: float = 30000.0) -> None:
    """Assert processing time is within acceptable range."""
    assert hasattr(result, "processing_time_ms"), "Result missing processing_time_ms"
    assert result.processing_time_ms >= 0, "Processing time must be non-negative"
    assert result.processing_time_ms < max_ms, (
        f"Processing time {result.processing_time_ms}ms exceeds {max_ms}ms"
    )


@contextmanager
def timed_block(label: str = "", max_seconds: float = 10.0):
    """Context manager that asserts a block completes within max_seconds."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    assert elapsed < max_seconds, (
        f"Block '{label}' took {elapsed:.3f}s, exceeding {max_seconds}s"
    )


# ---------------------------------------------------------------------------
# Fixtures -- Engine Instances
# ---------------------------------------------------------------------------


@pytest.fixture
def pledge_engine() -> PledgeCommitmentEngine:
    """Create a PledgeCommitmentEngine with default config."""
    return PledgeCommitmentEngine()


@pytest.fixture
def pledge_engine_custom() -> PledgeCommitmentEngine:
    """Create a PledgeCommitmentEngine with custom config."""
    return PledgeCommitmentEngine(config={
        "min_interim_reduction_pct": Decimal("45"),
        "max_net_zero_year": 2045,
    })


@pytest.fixture
def starting_line_engine() -> StartingLineEngine:
    """Create a StartingLineEngine with default config."""
    return StartingLineEngine()


@pytest.fixture
def interim_target_engine() -> InterimTargetEngine:
    """Create an InterimTargetEngine with default config."""
    return InterimTargetEngine()


@pytest.fixture
def action_plan_engine() -> ActionPlanEngine:
    """Create an ActionPlanEngine with default config."""
    return ActionPlanEngine()


@pytest.fixture
def progress_engine() -> ProgressTrackingEngine:
    """Create a ProgressTrackingEngine with default config."""
    return ProgressTrackingEngine()


@pytest.fixture
def sector_engine() -> SectorPathwayEngine:
    """Create a SectorPathwayEngine with default config."""
    return SectorPathwayEngine()


@pytest.fixture
def partnership_engine() -> PartnershipScoringEngine:
    """Create a PartnershipScoringEngine with default config."""
    return PartnershipScoringEngine()


@pytest.fixture
def campaign_engine() -> CampaignReportingEngine:
    """Create a CampaignReportingEngine with default config."""
    return CampaignReportingEngine()


@pytest.fixture
def credibility_engine() -> CredibilityAssessmentEngine:
    """Create a CredibilityAssessmentEngine with default config."""
    return CredibilityAssessmentEngine()


@pytest.fixture
def readiness_engine() -> RaceReadinessEngine:
    """Create a RaceReadinessEngine with default config."""
    return RaceReadinessEngine()


# ---------------------------------------------------------------------------
# Fixtures -- Sample Input Builders
# ---------------------------------------------------------------------------


@pytest.fixture
def strong_pledge_input() -> PledgeCommitmentInput:
    """Build a strong/eligible pledge commitment input (all criteria PASS)."""
    return PledgeCommitmentInput(
        entity_name="GreenCorp International",
        actor_type="corporate",
        sector="manufacturing",
        country="DE",
        employee_count=5000,
        net_zero_target_year=2050,
        interim_target_year=2030,
        interim_target_reduction_pct=Decimal("50"),
        baseline_year=2019,
        baseline_emissions_tco2e=Decimal("100000"),
        scope1_coverage_pct=Decimal("100"),
        scope2_coverage_pct=Decimal("100"),
        scope3_coverage_pct=Decimal("75"),
        commitment_statement="We commit to achieving net-zero GHG emissions by 2050.",
        board_approved=True,
        publicly_disclosed=True,
        action_plan_committed=True,
        action_plan_deadline_months=12,
        annual_reporting_committed=True,
        partners=[
            PartnerAlignmentInput(
                partner_id="sbti",
                membership_status="active",
                reporting_channel=True,
            ),
            PartnerAlignmentInput(
                partner_id="cdp",
                membership_status="active",
            ),
        ],
        include_recommendations=True,
    )


@pytest.fixture
def weak_pledge_input() -> PledgeCommitmentInput:
    """Build a weak/ineligible pledge commitment input."""
    return PledgeCommitmentInput(
        entity_name="SlowStart Ltd",
        actor_type="corporate",
        sector="retail",
        country="US",
        employee_count=200,
        net_zero_target_year=2055,
        interim_target_year=2030,
        interim_target_reduction_pct=Decimal("15"),
        baseline_year=2012,
        baseline_emissions_tco2e=Decimal("50000"),
        scope1_coverage_pct=Decimal("80"),
        scope2_coverage_pct=Decimal("85"),
        scope3_coverage_pct=Decimal("20"),
        commitment_statement="",
        board_approved=False,
        publicly_disclosed=False,
        action_plan_committed=False,
        annual_reporting_committed=False,
        partners=[],
        include_recommendations=True,
    )


@pytest.fixture
def compliant_starting_line_input() -> StartingLineInput:
    """Build a fully compliant Starting Line input."""
    return StartingLineInput(
        entity_name="GreenCorp International",
        actor_type="corporate",
        join_date="2025-01-15",
        net_zero_target_year=2050,
        interim_target_year=2030,
        interim_target_reduction_pct=Decimal("50"),
        baseline_year=2019,
        baseline_emissions_tco2e=Decimal("100000"),
        current_emissions_tco2e=Decimal("80000"),
        scope1_coverage_pct=Decimal("100"),
        scope2_coverage_pct=Decimal("100"),
        scope3_coverage_pct=Decimal("75"),
        action_plan_published=True,
        action_plan_date="2025-06-01",
        has_quantified_actions=True,
        has_timeline=True,
        has_resources=True,
        has_sector_alignment=True,
        immediate_actions_taken=True,
        emissions_reducing=True,
        investment_committed=True,
        governance_integrated=True,
        no_contradictory_actions=True,
        annual_reporting_done=True,
        emissions_disclosed=True,
        target_progress_reported=True,
        plan_updated_annually=True,
        methodology_transparent=True,
        include_remediation=True,
    )


@pytest.fixture
def non_compliant_starting_line_input() -> StartingLineInput:
    """Build a non-compliant Starting Line input."""
    return StartingLineInput(
        entity_name="SlowStart Ltd",
        actor_type="corporate",
        net_zero_target_year=2055,
        interim_target_year=2030,
        interim_target_reduction_pct=Decimal("10"),
        scope1_coverage_pct=Decimal("50"),
        scope2_coverage_pct=Decimal("50"),
        scope3_coverage_pct=Decimal("10"),
        action_plan_published=False,
        has_quantified_actions=False,
        has_timeline=False,
        has_resources=False,
        has_sector_alignment=False,
        immediate_actions_taken=False,
        emissions_reducing=False,
        investment_committed=False,
        governance_integrated=False,
        no_contradictory_actions=False,
        annual_reporting_done=False,
        emissions_disclosed=False,
        target_progress_reported=False,
        plan_updated_annually=False,
        methodology_transparent=False,
    )


@pytest.fixture
def aligned_interim_input() -> InterimTargetInput:
    """Build a 1.5C-aligned interim target input."""
    return InterimTargetInput(
        entity_name="GreenCorp International",
        actor_type="corporate",
        sector="manufacturing",
        baseline_year=2019,
        target_year=2030,
        total_baseline_emissions_tco2e=Decimal("100000"),
        total_target_emissions_tco2e=Decimal("50000"),
        target_type="absolute",
        target_reduction_pct=Decimal("50"),
        methodology="sbti_aca",
        scope1_coverage_pct=Decimal("100"),
        scope2_coverage_pct=Decimal("100"),
        scope3_coverage_pct=Decimal("75"),
        fair_share_considered=True,
        sbti_validated=True,
        include_pathway_comparison=True,
    )


@pytest.fixture
def misaligned_interim_input() -> InterimTargetInput:
    """Build a misaligned interim target input."""
    return InterimTargetInput(
        entity_name="SlowStart Ltd",
        actor_type="corporate",
        sector="retail",
        baseline_year=2019,
        target_year=2030,
        total_baseline_emissions_tco2e=Decimal("100000"),
        total_target_emissions_tco2e=Decimal("90000"),
        target_type="absolute",
        target_reduction_pct=Decimal("10"),
        methodology="none",
        scope1_coverage_pct=Decimal("80"),
        scope2_coverage_pct=Decimal("80"),
        scope3_coverage_pct=Decimal("20"),
        fair_share_considered=False,
        sbti_validated=False,
        include_pathway_comparison=True,
    )


@pytest.fixture
def complete_action_plan_input() -> ActionPlanInput:
    """Build a complete action plan input with 15 actions."""
    actions = []
    categories = list(ActionCategory)
    for i in range(15):
        cat = categories[i % len(categories)]
        actions.append(DecarbonizationActionInput(
            action_name=f"Action {i+1}: {cat.value.replace('_', ' ').title()}",
            category=cat.value,
            scope_impact=[1, 2] if i < 5 else [3] if i < 10 else [1, 2, 3],
            abatement_tco2e=Decimal(str(1000 + i * 500)),
            cost_total_usd=Decimal(str(50000 + i * 10000)),
            cost_per_tco2e_usd=Decimal(str(50 + i * 5)),
            start_year=2025,
            end_year=2028 + (i % 3),
            trl=9 - (i % 3),
            responsible_party=f"Team {chr(65 + i % 5)}",
            status="planned" if i < 10 else "in_progress",
            milestones=[f"M{j+1}: Phase {j+1}" for j in range(3)],
        ))

    sections = []
    for sec_id in SECTION_IDS:
        sections.append(SectionInput(
            section_id=sec_id,
            score=Decimal("9"),
            content_summary=f"Section {sec_id} fully documented.",
        ))

    return ActionPlanInput(
        entity_name="GreenCorp International",
        actor_type="corporate",
        sector="manufacturing",
        baseline_year=2019,
        baseline_emissions_tco2e=Decimal("100000"),
        target_year=2030,
        target_emissions_tco2e=Decimal("50000"),
        net_zero_year=2050,
        join_date="2025-01-15",
        plan_published=True,
        plan_publication_date="2025-06-01",
        actions=actions,
        sections=sections,
        scope1_emissions_tco2e=Decimal("30000"),
        scope2_emissions_tco2e=Decimal("20000"),
        scope3_emissions_tco2e=Decimal("50000"),
        total_budget_usd=Decimal("5000000"),
        fte_allocated=Decimal("10"),
        has_governance_structure=True,
        has_board_oversight=True,
        has_just_transition_plan=True,
        has_monitoring_kpis=True,
        include_prioritization=True,
    )


@pytest.fixture
def incomplete_action_plan_input() -> ActionPlanInput:
    """Build an incomplete action plan input with 3 actions."""
    actions = [
        DecarbonizationActionInput(
            action_name=f"Action {i+1}",
            category="energy_efficiency",
            scope_impact=[1],
            abatement_tco2e=Decimal("500"),
            cost_total_usd=Decimal("25000"),
        )
        for i in range(3)
    ]
    return ActionPlanInput(
        entity_name="SlowStart Ltd",
        actor_type="corporate",
        sector="general",
        baseline_year=2019,
        baseline_emissions_tco2e=Decimal("50000"),
        target_year=2030,
        target_emissions_tco2e=Decimal("40000"),
        actions=actions,
        sections=[],
        has_governance_structure=False,
        has_board_oversight=False,
        has_just_transition_plan=False,
        has_monitoring_kpis=False,
    )


# ---------------------------------------------------------------------------
# Fixtures -- Actor type parametrization
# ---------------------------------------------------------------------------


ACTOR_TYPES = [
    "corporate", "financial_institution", "city",
    "region", "sme", "university", "healthcare",
]

PARTNER_INITIATIVES = [p.value for p in PartnerInitiative]

PRESET_NAMES = [
    "corporate_commitment", "financial_institution",
    "city_municipality", "region_state", "sme_business",
    "high_emitter", "service_sector", "manufacturing_sector",
]


# ---------------------------------------------------------------------------
# Fixtures -- Mock database and cache
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()
    session.execute = AsyncMock(return_value=MagicMock(
        fetchall=MagicMock(return_value=[]),
        fetchone=MagicMock(return_value=None),
    ))
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
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
    return redis_client


# ---------------------------------------------------------------------------
# Fixtures -- pack.yaml path
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
    return _PACK_ROOT / "config" / "presets"
