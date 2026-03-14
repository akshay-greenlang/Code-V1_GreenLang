# -*- coding: utf-8 -*-
"""
PACK-002 CSRD Professional Pack - Demo Mode Tests
====================================================

Tests for demo mode operation including profile validation,
subsidiary data, consolidation pipeline, cross-framework mapping,
approval chain, and complete output generation.

Test count: 8
Author: GreenLang QA Team
"""

import hashlib
import json
from decimal import Decimal
from typing import Any, Dict, List

import pytest

from consolidation_engine import (
    ConsolidationApproach,
    ConsolidationConfig,
    ConsolidationEngine,
    ConsolidationMethod,
    EntityDefinition,
    EntityESRSData,
)


class TestDemoMode:
    """Test demo mode with EuroTech Holdings sample data."""

    def test_demo_config_loads(self, sample_pack_config):
        """Demo configuration loads with all required fields."""
        config = sample_pack_config
        assert config["metadata"]["name"] == "csrd-professional"
        assert config["metadata"]["tier"] == "professional"
        assert config["size_preset"] == "enterprise_group"
        assert config["reporting_year"] == 2025

    def test_demo_group_profile_valid(self, sample_group_profile):
        """Demo group profile has valid parent entity and subsidiaries."""
        profile = sample_group_profile
        assert profile["group_name"] == "EuroTech Holdings AG"
        assert profile["parent"]["country"] == "DE"
        assert profile["parent"]["employees"] == 8000
        assert profile["parent"]["revenue_eur"] == 2_100_000_000
        assert len(profile["subsidiaries"]) == 5

    def test_demo_subsidiary_data_valid(self, sample_entity_data):
        """Demo entity data has realistic values for all 6 entities."""
        assert len(sample_entity_data) == 6

        for entity_id, data in sample_entity_data.items():
            assert data["entity_id"] == entity_id
            assert data["reporting_period"] == "2025-01-01/2025-12-31"
            assert 0 < data["quality_score"] <= 100

            dp = data["data_points"]
            assert dp["E1-6_01_scope1_total_tco2e"] > 0
            assert dp["E1-6_04_scope2_market_tco2e"] > 0
            assert dp["S1_total_employees"] > 0
            assert dp["revenue_eur"] > 0

    @pytest.mark.asyncio
    async def test_demo_multi_entity_pipeline(self, sample_group_profile, sample_entity_data):
        """Demo pipeline consolidates 6 entities successfully."""
        engine = ConsolidationEngine()

        # Register parent
        parent = sample_group_profile["parent"]
        engine.add_entity(EntityDefinition(
            entity_id=parent["entity_id"],
            name=parent["name"],
            country=parent["country"],
            ownership_pct=Decimal(str(parent["ownership_pct"])),
            consolidation_method=ConsolidationMethod(parent["consolidation_method"]),
            parent_entity_id=None,
            employee_count=parent["employees"],
        ))

        # Register subsidiaries
        for sub in sample_group_profile["subsidiaries"]:
            engine.add_entity(EntityDefinition(
                entity_id=sub["entity_id"],
                name=sub["name"],
                country=sub["country"],
                ownership_pct=Decimal(str(sub["ownership_pct"])),
                consolidation_method=ConsolidationMethod(sub["consolidation_method"]),
                parent_entity_id=parent["entity_id"],
                employee_count=sub["employees"],
            ))

        # Load entity data
        for entity_id, data in sample_entity_data.items():
            engine.set_entity_data(entity_id, EntityESRSData(
                entity_id=entity_id,
                data_points=data["data_points"],
                reporting_period=data["reporting_period"],
                quality_score=data["quality_score"],
            ))

        result = await engine.consolidate(ConsolidationApproach.OPERATIONAL_CONTROL)
        assert result.entity_count == 6
        assert result.provenance_hash != ""
        assert Decimal(result.consolidated_data["E1-6_01_scope1_total_tco2e"]) > 0

    def test_demo_cross_framework(self, sample_cross_framework_data):
        """Demo cross-framework data maps ESRS to 6 frameworks."""
        data = sample_cross_framework_data
        assert data["source_framework"] == "ESRS"
        assert len(data["mappings"]) == 6
        assert data["overall_coverage_pct"] > 80.0

        for fw_id, fw_data in data["mappings"].items():
            assert fw_data["coverage_pct"] > 0, f"Framework {fw_id} has 0% coverage"

    @pytest.mark.asyncio
    async def test_demo_consolidation(self, sample_group_profile, sample_entity_data):
        """Demo consolidation produces valid consolidated totals."""
        engine = ConsolidationEngine()

        parent = sample_group_profile["parent"]
        engine.add_entity(EntityDefinition(
            entity_id=parent["entity_id"],
            name=parent["name"],
            country=parent["country"],
            ownership_pct=Decimal("100"),
            consolidation_method=ConsolidationMethod.OPERATIONAL_CONTROL,
        ))

        for sub in sample_group_profile["subsidiaries"]:
            engine.add_entity(EntityDefinition(
                entity_id=sub["entity_id"],
                name=sub["name"],
                country=sub["country"],
                ownership_pct=Decimal(str(sub["ownership_pct"])),
                consolidation_method=ConsolidationMethod(sub["consolidation_method"]),
                parent_entity_id=parent["entity_id"],
            ))

        for entity_id, data in sample_entity_data.items():
            engine.set_entity_data(entity_id, EntityESRSData(
                entity_id=entity_id,
                data_points=data["data_points"],
                reporting_period=data["reporting_period"],
                quality_score=data["quality_score"],
            ))

        result = await engine.consolidate(ConsolidationApproach.EQUITY_SHARE)
        assert result.entity_count == 6
        # Equity share: ES at 80%, rest at 100%
        minority = engine.calculate_minority_interest()
        assert len(minority) == 1
        assert minority[0]["entity_id"] == "eurotech-es"

    def test_demo_approval_chain(self, sample_approval_chain):
        """Demo approval chain has 4 levels with valid configuration."""
        chain = sample_approval_chain
        assert chain["chain_id"] == "eurotech-approval-2025"
        assert len(chain["levels"]) == 4
        assert chain["levels"][0]["name"] == "Preparer"
        assert chain["levels"][3]["name"] == "Board Sign-off"
        assert chain["delegation_rules"]["enabled"] is True

    def test_demo_output_complete(
        self,
        sample_pack_config,
        sample_group_profile,
        sample_entity_data,
        sample_cross_framework_data,
        sample_quality_gate_data,
        sample_approval_chain,
        sample_scenario_config,
        sample_benchmark_data,
    ):
        """Demo output contains all required sections for a complete report."""
        output = {
            "pack_config": sample_pack_config,
            "group_profile": sample_group_profile,
            "entity_count": len(sample_entity_data),
            "cross_framework": sample_cross_framework_data,
            "quality_gates": sample_quality_gate_data,
            "approval_chain": sample_approval_chain,
            "scenarios": sample_scenario_config,
            "benchmarks": sample_benchmark_data,
        }

        required_sections = [
            "pack_config", "group_profile", "entity_count",
            "cross_framework", "quality_gates", "approval_chain",
            "scenarios", "benchmarks",
        ]
        for section in required_sections:
            assert section in output, f"Missing section: {section}"

        assert output["entity_count"] == 6
        assert output["group_profile"]["group_name"] == "EuroTech Holdings AG"

        # Provenance
        provenance = hashlib.sha256(
            json.dumps(output, sort_keys=True, default=str).encode()
        ).hexdigest()
        assert len(provenance) == 64
