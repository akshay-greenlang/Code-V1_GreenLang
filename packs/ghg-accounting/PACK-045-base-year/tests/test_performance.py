# -*- coding: utf-8 -*-
"""
Performance benchmarks for PACK-045.

Measures engine throughput, latency, and memory usage.
Target: ~10 tests.
"""

import time
import pytest
from decimal import Decimal
from pathlib import Path
import sys

PACK_ROOT = Path(__file__).resolve().parent.parent
if str(PACK_ROOT) not in sys.path:
    sys.path.insert(0, str(PACK_ROOT))

from engines.base_year_selection_engine import (
    BaseYearSelectionEngine, CandidateYear, SelectionConfig,
)
from engines.base_year_inventory_engine import (
    BaseYearInventoryEngine, SourceEmission, InventoryConfig, SourceCategory,
)
from engines.recalculation_policy_engine import RecalculationPolicyEngine
from engines.significance_assessment_engine import (
    SignificanceAssessmentEngine, TriggerInput, AssessmentPolicy,
    TriggerType as SigTriggerType,
)
from engines.target_tracking_engine import (
    TargetTrackingEngine, EmissionsTarget, YearlyActual,
    TargetType as TTTargetType, ScopeType as TTScopeType,
)
from config.pack_config import PackConfig, load_preset


class TestSelectionPerformance:
    def test_selection_under_50ms(self, selection_engine, candidate_years):
        """Selection engine should complete under 50ms."""
        start = time.perf_counter()
        for _ in range(10):
            selection_engine.evaluate_candidates(candidate_years)
        elapsed = (time.perf_counter() - start) * 1000 / 10
        assert elapsed < 50, f"Selection took {elapsed:.1f}ms (target <50ms)"

    def test_selection_20_candidates(self, selection_engine):
        """Evaluate 20 candidates under 100ms."""
        candidates = [
            CandidateYear(
                year=2000 + i,
                total_tco2e=Decimal(str(10000 + i * 100)),
                data_quality_score=Decimal(str(70 + i)),
                completeness_pct=Decimal(str(80 + i % 20)),
                methodology_tier=min(4, 1 + i % 4),
                is_verified=i % 2 == 0,
            )
            for i in range(20)
        ]
        start = time.perf_counter()
        result = selection_engine.evaluate_candidates(candidates)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 100, f"20-candidate selection took {elapsed:.1f}ms"
        assert result.recommended_year >= 2000


class TestInventoryPerformance:
    def test_inventory_100_sources_under_50ms(self, inventory_engine):
        """Establish inventory with 100 sources under 50ms."""
        sources = [
            SourceEmission(
                category=SourceCategory.STATIONARY_COMBUSTION,
                tco2e=Decimal(str(100 + i)),
                facility_id=f"FAC-{i:03d}",
            )
            for i in range(100)
        ]
        config = InventoryConfig(organization_id="PERF-001", base_year=2022)
        start = time.perf_counter()
        inv = inventory_engine.establish_inventory(sources, config)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50, f"100-source inventory took {elapsed:.1f}ms"
        assert inv.grand_total_tco2e > 0

    def test_inventory_1000_sources_under_200ms(self, inventory_engine):
        """Establish inventory with 1000 sources under 200ms."""
        categories = list(SourceCategory)
        sources = [
            SourceEmission(
                category=categories[i % len(categories)],
                tco2e=Decimal(str(50 + i)),
                facility_id=f"FAC-{i % 10:03d}",
            )
            for i in range(1000)
        ]
        config = InventoryConfig(organization_id="PERF-001", base_year=2022)
        start = time.perf_counter()
        inv = inventory_engine.establish_inventory(sources, config)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 200, f"1000-source inventory took {elapsed:.1f}ms"


class TestSignificancePerformance:
    def test_assess_10_triggers_under_20ms(self, significance_engine):
        """Assess 10 triggers under 20ms."""
        triggers = [
            TriggerInput(
                trigger_type=SigTriggerType.ACQUISITION,
                emissions_impact_tco2e=Decimal(str(1000 * (i + 1))),
            )
            for i in range(10)
        ]
        policy = AssessmentPolicy(
            individual_threshold_pct=Decimal("5.0"),
            cumulative_threshold_pct=Decimal("10.0"),
            base_year_total_tco2e=Decimal("100000"),
        )
        start = time.perf_counter()
        result = significance_engine.assess_significance(triggers, policy)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 20, f"10-trigger assessment took {elapsed:.1f}ms"


class TestTargetTrackingPerformance:
    def test_track_progress_under_20ms(self, target_engine):
        """Track progress with 10 years of data under 20ms."""
        target = EmissionsTarget(
            target_id="PERF-TGT",
            name="Perf Scope 1 Target",
            target_type=TTTargetType.ABSOLUTE,
            scopes=[TTScopeType.SCOPE_1],
            base_year=2019,
            base_year_tco2e=Decimal("100000"),
            target_year=2030,
            target_reduction_pct=Decimal("42.0"),
        )
        actuals = [
            YearlyActual(
                year=2019 + i,
                actual_tco2e=Decimal(str(100000 - 3000 * i)),
            )
            for i in range(10)
        ]
        start = time.perf_counter()
        result = target_engine.track_progress(target, actuals)
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 20, f"Target tracking took {elapsed:.1f}ms"


class TestConfigPerformance:
    def test_load_preset_under_50ms(self):
        """Load a preset under 50ms."""
        start = time.perf_counter()
        pc = load_preset("manufacturing")
        elapsed = (time.perf_counter() - start) * 1000
        assert elapsed < 50, f"Preset load took {elapsed:.1f}ms"
        assert pc.preset_name == "manufacturing"

    def test_config_hash_under_5ms(self):
        """Config hash computation under 5ms."""
        pc = PackConfig()
        start = time.perf_counter()
        for _ in range(100):
            pc.get_config_hash()
        elapsed = (time.perf_counter() - start) * 1000 / 100
        assert elapsed < 5, f"Config hash took {elapsed:.1f}ms"


class TestPolicyPerformance:
    def test_create_policy_under_10ms(self, policy_engine):
        """Create policy under 10ms."""
        from engines.recalculation_policy_engine import PolicyType
        start = time.perf_counter()
        for _ in range(10):
            policy_engine.create_policy(PolicyType.GHG_PROTOCOL_DEFAULT)
        elapsed = (time.perf_counter() - start) * 1000 / 10
        assert elapsed < 10, f"Policy creation took {elapsed:.1f}ms"
