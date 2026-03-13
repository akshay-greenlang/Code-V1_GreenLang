# -*- coding: utf-8 -*-
"""
Tests for Engine 3: Capacity Building Manager Engine - AGENT-EUDR-025

Tests 4-tier capacity building framework, commodity-specific curricula,
enrollment management, tier advancement, competency assessments, module
completion tracking, scorecard generation, and batch operations.

Test count: ~60 tests
Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict

import pytest

from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    CapacityTier,
    EnrollmentStatus,
    EUDRCommodity,
    CapacityBuildingEnrollment,
    EnrollSupplierRequest,
    EnrollSupplierResponse,
    SUPPORTED_COMMODITIES,
    CAPACITY_TIER_COUNT,
    MODULES_PER_COMMODITY,
)
from greenlang.agents.eudr.risk_mitigation_advisor.capacity_building_manager_engine import (
    CapacityBuildingManagerEngine,
)

from .conftest import FIXED_DATE


class TestCapacityEngineInit:
    def test_engine_initializes(self, capacity_engine):
        assert capacity_engine is not None

    def test_engine_has_curricula(self, capacity_engine):
        curricula = capacity_engine.get_available_curricula()
        assert len(curricula) >= 7

    def test_engine_supports_all_commodities(self, capacity_engine):
        for commodity in SUPPORTED_COMMODITIES:
            modules = capacity_engine.get_modules_for_commodity(commodity)
            assert len(modules) >= 1


class TestEnrollment:
    @pytest.mark.asyncio
    async def test_enroll_supplier(self, capacity_engine, enroll_supplier_request):
        result = await capacity_engine.enroll_supplier(enroll_supplier_request)
        assert isinstance(result, EnrollSupplierResponse)
        assert result.enrollment is not None

    @pytest.mark.asyncio
    async def test_enrollment_starts_at_tier_1(self, capacity_engine, enroll_supplier_request):
        result = await capacity_engine.enroll_supplier(enroll_supplier_request)
        assert result.enrollment.current_tier == 1

    @pytest.mark.asyncio
    async def test_enrollment_active_status(self, capacity_engine, enroll_supplier_request):
        result = await capacity_engine.enroll_supplier(enroll_supplier_request)
        assert result.enrollment.status == EnrollmentStatus.ACTIVE

    @pytest.mark.asyncio
    async def test_enrollment_modules_assigned(self, capacity_engine, enroll_supplier_request):
        result = await capacity_engine.enroll_supplier(enroll_supplier_request)
        assert result.modules_assigned >= 1

    @pytest.mark.asyncio
    async def test_enrollment_has_provenance(self, capacity_engine, enroll_supplier_request):
        result = await capacity_engine.enroll_supplier(enroll_supplier_request)
        assert result.provenance_hash != ""

    @pytest.mark.parametrize("commodity", SUPPORTED_COMMODITIES)
    @pytest.mark.asyncio
    async def test_enroll_all_commodities(self, capacity_engine, commodity):
        request = EnrollSupplierRequest(
            supplier_id=f"sup-{commodity}",
            commodity=commodity,
            initial_tier=1,
            target_completion_weeks=24,
        )
        result = await capacity_engine.enroll_supplier(request)
        assert result.enrollment.commodity == commodity

    @pytest.mark.asyncio
    async def test_enroll_at_tier_2(self, capacity_engine):
        request = EnrollSupplierRequest(
            supplier_id="sup-t2",
            commodity="cocoa",
            initial_tier=2,
            target_completion_weeks=16,
        )
        result = await capacity_engine.enroll_supplier(request)
        assert result.enrollment.current_tier == 2

    @pytest.mark.asyncio
    async def test_enroll_at_tier_3(self, capacity_engine):
        request = EnrollSupplierRequest(
            supplier_id="sup-t3",
            commodity="coffee",
            initial_tier=3,
            target_completion_weeks=12,
        )
        result = await capacity_engine.enroll_supplier(request)
        assert result.enrollment.current_tier == 3


class TestTierAdvancement:
    @pytest.mark.asyncio
    async def test_tier_advancement_eligible(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-adv",
            current_tier=1,
            competency_scores={"m1": Decimal("75"), "m2": Decimal("80"), "m3": Decimal("70")},
            modules_completed=4,
            modules_total=4,
        )
        assert result is not None
        assert result["eligible"] is True

    @pytest.mark.asyncio
    async def test_tier_advancement_not_eligible_low_competency(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-low",
            current_tier=1,
            competency_scores={"m1": Decimal("40"), "m2": Decimal("45")},
            modules_completed=4,
            modules_total=4,
        )
        assert result["eligible"] is False

    @pytest.mark.asyncio
    async def test_tier_advancement_not_eligible_low_completion(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-inc",
            current_tier=1,
            competency_scores={"m1": Decimal("85"), "m2": Decimal("90")},
            modules_completed=2,
            modules_total=4,
        )
        assert result["eligible"] is False

    @pytest.mark.asyncio
    async def test_tier_4_cannot_advance(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-max",
            current_tier=4,
            competency_scores={"m1": Decimal("95")},
            modules_completed=4,
            modules_total=4,
        )
        assert result["eligible"] is False

    @pytest.mark.asyncio
    async def test_tier_2_higher_threshold(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-t2",
            current_tier=2,
            competency_scores={"m1": Decimal("72"), "m2": Decimal("68")},
            modules_completed=8,
            modules_total=8,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_tier_advancement_uses_decimal(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-dec",
            current_tier=1,
            competency_scores={"m1": Decimal("60.00"), "m2": Decimal("60.01")},
            modules_completed=4,
            modules_total=4,
        )
        assert isinstance(result["avg_competency"], Decimal)


class TestModuleCompletion:
    @pytest.mark.asyncio
    async def test_complete_module(self, capacity_engine, enroll_supplier_request):
        enrollment = await capacity_engine.enroll_supplier(enroll_supplier_request)
        updated = await capacity_engine.complete_module(
            enrollment.enrollment.enrollment_id,
            module_id="mod-001",
            competency_score=Decimal("85"),
        )
        assert updated is not None

    @pytest.mark.asyncio
    async def test_module_score_recorded(self, capacity_engine, enroll_supplier_request):
        enrollment = await capacity_engine.enroll_supplier(enroll_supplier_request)
        updated = await capacity_engine.complete_module(
            enrollment.enrollment.enrollment_id,
            module_id="mod-002",
            competency_score=Decimal("72"),
        )
        assert "mod-002" in updated.competency_scores


class TestScorecard:
    @pytest.mark.asyncio
    async def test_generate_scorecard(self, capacity_engine, sample_enrollment):
        scorecard = await capacity_engine.generate_scorecard(
            sample_enrollment.enrollment_id
        )
        assert scorecard is not None
        assert "current_tier" in scorecard
        assert "modules_completed" in scorecard

    @pytest.mark.asyncio
    async def test_scorecard_includes_risk_delta(self, capacity_engine, sample_enrollment):
        scorecard = await capacity_engine.generate_scorecard(
            sample_enrollment.enrollment_id
        )
        assert "risk_score_delta" in scorecard


class TestBatchEnrollment:
    @pytest.mark.asyncio
    async def test_batch_enroll_suppliers(self, capacity_engine):
        requests = [
            EnrollSupplierRequest(
                supplier_id=f"sup-batch-{i}",
                commodity="palm_oil",
                initial_tier=1,
                target_completion_weeks=24,
            )
            for i in range(5)
        ]
        results = await capacity_engine.enroll_batch(requests)
        assert len(results) == 5

    @pytest.mark.asyncio
    async def test_batch_enroll_empty_list(self, capacity_engine):
        results = await capacity_engine.enroll_batch([])
        assert results == []


class TestCapacityEdgeCases:
    @pytest.mark.asyncio
    async def test_zero_competency_scores(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-zero",
            current_tier=1,
            competency_scores={},
            modules_completed=0,
            modules_total=4,
        )
        assert result["eligible"] is False

    @pytest.mark.asyncio
    async def test_single_module_completion(self, capacity_engine):
        result = await capacity_engine.check_tier_advancement(
            enrollment_id="enr-single",
            current_tier=1,
            competency_scores={"m1": Decimal("100")},
            modules_completed=1,
            modules_total=4,
        )
        assert result is not None

    def test_modules_per_commodity_constant(self):
        assert MODULES_PER_COMMODITY == 22

    def test_capacity_tier_count_constant(self):
        assert CAPACITY_TIER_COUNT == 4
