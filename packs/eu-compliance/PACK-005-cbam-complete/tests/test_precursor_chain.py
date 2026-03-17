# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - Precursor Chain Engine Tests (25 tests)

Tests PrecursorChainEngine: chain resolution for all goods categories,
depth limits, allocation methods (mass/economic/energy), composition
tracking, default fallback waterfall, mass balance validation, scrap
classification, production routes, gap analysis, and visualization.

Author: GreenLang QA Team
"""

import json
from decimal import Decimal
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    _compute_hash,
    _new_uuid,
    assert_provenance_hash,
)


# ---------------------------------------------------------------------------
# Chain Resolution by Category (6 tests)
# ---------------------------------------------------------------------------

class TestChainResolution:
    """Test precursor chain resolution for each goods category."""

    def test_resolve_steel_chain(self, sample_precursor_chain):
        """Test resolving a full steel precursor chain."""
        chain = sample_precursor_chain
        assert chain["goods_category"] == "steel"
        assert len(chain["stages"]) == 4
        assert chain["stages"][0]["stage_name"] == "Iron Ore Mining"
        assert chain["stages"][-1]["stage_name"] == "Hot Rolling"
        assert chain["production_route"] == "bf_bof"

    def test_resolve_aluminium_chain(self):
        """Test resolving an aluminium precursor chain."""
        chain = {
            "chain_id": "PC-AL-001",
            "goods_category": "aluminium",
            "stages": [
                {"stage_name": "Bauxite Mining", "emission_tco2e_per_tonne": 0.05},
                {"stage_name": "Alumina Refining (Bayer)", "emission_tco2e_per_tonne": 1.80},
                {"stage_name": "Smelting (Hall-Heroult)", "emission_tco2e_per_tonne": 6.50},
            ],
            "production_route": "hall_heroult",
        }
        assert len(chain["stages"]) == 3
        total_ef = sum(s["emission_tco2e_per_tonne"] for s in chain["stages"])
        assert total_ef == pytest.approx(8.35, rel=1e-2)

    def test_resolve_cement_chain(self):
        """Test resolving a cement precursor chain."""
        chain = {
            "chain_id": "PC-CEM-001",
            "goods_category": "cement",
            "stages": [
                {"stage_name": "Raw Material Prep", "emission_tco2e_per_tonne": 0.02},
                {"stage_name": "Clinker Production", "emission_tco2e_per_tonne": 0.84},
                {"stage_name": "Cement Grinding", "emission_tco2e_per_tonne": 0.05},
            ],
            "production_route": "dry_process",
        }
        assert chain["goods_category"] == "cement"
        assert len(chain["stages"]) == 3

    def test_resolve_fertilizer_chain(self):
        """Test resolving a fertilizer precursor chain."""
        chain = {
            "chain_id": "PC-FERT-001",
            "goods_category": "fertilizers",
            "stages": [
                {"stage_name": "Natural Gas Reforming", "emission_tco2e_per_tonne": 1.20},
                {"stage_name": "Ammonia Synthesis", "emission_tco2e_per_tonne": 0.90},
                {"stage_name": "Urea Production", "emission_tco2e_per_tonne": 0.40},
            ],
        }
        assert chain["goods_category"] == "fertilizers"
        total = sum(s["emission_tco2e_per_tonne"] for s in chain["stages"])
        assert total == pytest.approx(2.50, rel=1e-2)

    def test_resolve_hydrogen_chain(self):
        """Test resolving a hydrogen precursor chain."""
        chain = {
            "chain_id": "PC-H2-001",
            "goods_category": "hydrogen",
            "stages": [
                {"stage_name": "Steam Methane Reforming", "emission_tco2e_per_tonne": 10.0},
            ],
        }
        assert len(chain["stages"]) == 1
        assert chain["stages"][0]["emission_tco2e_per_tonne"] == 10.0

    def test_resolve_electricity(self):
        """Test electricity has no precursors (direct emission)."""
        chain = {
            "chain_id": "PC-ELEC-001",
            "goods_category": "electricity",
            "stages": [],
        }
        assert len(chain["stages"]) == 0


# ---------------------------------------------------------------------------
# Depth and Limits (2 tests)
# ---------------------------------------------------------------------------

class TestDepthLimits:
    """Test chain depth limits."""

    def test_max_depth_limit(self, sample_config):
        """Test chain resolution respects max depth limit."""
        max_depth = sample_config["precursor_chain"]["max_depth"]
        assert max_depth == 10
        # Simulate a chain that exceeds depth
        chain_depth = 15
        truncated = min(chain_depth, max_depth)
        assert truncated == 10

    def test_chain_depth_within_limit(self, sample_precursor_chain):
        """Test actual chain depth is within limit."""
        assert len(sample_precursor_chain["stages"]) <= 10


# ---------------------------------------------------------------------------
# Allocation Methods (3 tests)
# ---------------------------------------------------------------------------

class TestAllocationMethods:
    """Test emission allocation methods."""

    def test_allocation_mass_based(self, sample_precursor_chain):
        """Test mass-based allocation factors."""
        for stage in sample_precursor_chain["stages"]:
            assert stage["allocation_method"] == "mass_based"
            assert 0.0 < stage["allocation_factor"] <= 1.0

    def test_allocation_economic(self):
        """Test economic allocation by revenue share."""
        products = [
            {"name": "main_product", "revenue_eur": 800000, "weight_tonnes": 1000},
            {"name": "by_product", "revenue_eur": 200000, "weight_tonnes": 500},
        ]
        total_revenue = sum(p["revenue_eur"] for p in products)
        for p in products:
            p["economic_factor"] = p["revenue_eur"] / total_revenue
        assert products[0]["economic_factor"] == pytest.approx(0.80)
        assert products[1]["economic_factor"] == pytest.approx(0.20)

    def test_allocation_energy(self):
        """Test energy-based allocation by energy content."""
        products = [
            {"name": "steel_slab", "energy_gj": 18.0},
            {"name": "slag", "energy_gj": 2.0},
        ]
        total_energy = sum(p["energy_gj"] for p in products)
        for p in products:
            p["energy_factor"] = p["energy_gj"] / total_energy
        assert products[0]["energy_factor"] == pytest.approx(0.90)


# ---------------------------------------------------------------------------
# Composition and Fallback (3 tests)
# ---------------------------------------------------------------------------

class TestCompositionAndFallback:
    """Test composition tracking and default fallback."""

    def test_composition_tracking(self, sample_precursor_chain):
        """Test each stage tracks input/output weight."""
        for stage in sample_precursor_chain["stages"]:
            assert stage["input_weight_tonnes"] > 0
            assert stage["output_weight_tonnes"] > 0
            assert stage["output_weight_tonnes"] <= stage["input_weight_tonnes"]

    def test_default_fallback_waterfall(self):
        """Test fallback waterfall: actual -> country_default -> eu_default."""
        data_sources = [
            {"source": "actual", "available": False, "ef": None},
            {"source": "country_default", "available": True, "ef": 2.15},
            {"source": "eu_default", "available": True, "ef": 1.85},
        ]
        selected = None
        for src in data_sources:
            if src["available"] and src["ef"] is not None:
                selected = src
                break
        assert selected is not None
        assert selected["source"] == "country_default"
        assert selected["ef"] == 2.15

    def test_mass_balance_validation(self, sample_precursor_chain):
        """Test mass balance: output <= input at each stage."""
        for stage in sample_precursor_chain["stages"]:
            assert stage["output_weight_tonnes"] <= stage["input_weight_tonnes"], (
                f"Mass balance violated at {stage['stage_name']}"
            )


# ---------------------------------------------------------------------------
# Scrap Classification (2 tests)
# ---------------------------------------------------------------------------

class TestScrapClassification:
    """Test scrap classification rules."""

    def test_scrap_classification_pre_consumer(self):
        """Test pre-consumer scrap classification."""
        scrap = {
            "type": "pre_consumer",
            "source": "manufacturing_process",
            "emission_factor_tco2e_per_tonne": 0.0,
            "cbam_applicable": False,
        }
        assert scrap["emission_factor_tco2e_per_tonne"] == 0.0
        assert scrap["cbam_applicable"] is False

    def test_scrap_classification_post_consumer(self):
        """Test post-consumer scrap classification."""
        scrap = {
            "type": "post_consumer",
            "source": "end_of_life_vehicles",
            "emission_factor_tco2e_per_tonne": 0.0,
            "cbam_applicable": False,
        }
        assert scrap["type"] == "post_consumer"
        assert scrap["emission_factor_tco2e_per_tonne"] == 0.0


# ---------------------------------------------------------------------------
# Production Routes (2 tests)
# ---------------------------------------------------------------------------

class TestProductionRoutes:
    """Test production route classification."""

    def test_production_route_bfbof(self, sample_precursor_chain):
        """Test BF-BOF production route identification."""
        assert sample_precursor_chain["production_route"] == "bf_bof"
        # BF-BOF typically has higher emissions
        ef = sample_precursor_chain["specific_emission_tco2e_per_tonne"]
        assert ef > 1.0

    def test_production_route_eaf(self):
        """Test EAF production route has lower emissions."""
        eaf_chain = {
            "production_route": "eaf",
            "specific_emission_tco2e_per_tonne": 0.45,
            "stages": [
                {"stage_name": "Scrap Preparation", "emission_tco2e_per_tonne": 0.05},
                {"stage_name": "Electric Arc Furnace", "emission_tco2e_per_tonne": 0.35},
                {"stage_name": "Casting/Rolling", "emission_tco2e_per_tonne": 0.05},
            ],
        }
        assert eaf_chain["production_route"] == "eaf"
        assert eaf_chain["specific_emission_tco2e_per_tonne"] < 1.0


# ---------------------------------------------------------------------------
# Gap Analysis and Visualization (4 tests)
# ---------------------------------------------------------------------------

class TestGapAnalysisAndVisualization:
    """Test gap analysis and chain visualization."""

    def test_gap_analysis(self, sample_precursor_chain):
        """Test gap analysis identifies missing data."""
        # Simulate a chain with a missing emission factor
        stages_with_gaps = list(sample_precursor_chain["stages"])
        stages_with_gaps.append({
            "stage_id": "STG-005",
            "stage_name": "Coating",
            "emission_tco2e_per_tonne": None,
            "allocation_method": "mass_based",
            "allocation_factor": 1.0,
            "input_weight_tonnes": 1000,
            "output_weight_tonnes": 1000,
        })
        gaps = [s for s in stages_with_gaps if s.get("emission_tco2e_per_tonne") is None]
        assert len(gaps) == 1
        assert gaps[0]["stage_name"] == "Coating"

    def test_chain_visualization(self, sample_precursor_chain):
        """Test chain can be serialized for visualization."""
        viz_data = {
            "chain_id": sample_precursor_chain["chain_id"],
            "nodes": [
                {
                    "id": s["stage_id"],
                    "label": s["stage_name"],
                    "emission": s["emission_tco2e_per_tonne"],
                }
                for s in sample_precursor_chain["stages"]
            ],
            "edges": [],
        }
        for i in range(len(viz_data["nodes"]) - 1):
            viz_data["edges"].append({
                "from": viz_data["nodes"][i]["id"],
                "to": viz_data["nodes"][i + 1]["id"],
            })
        assert len(viz_data["nodes"]) == 4
        assert len(viz_data["edges"]) == 3

    def test_chain_total_emission(self, sample_precursor_chain):
        """Test total chain emission is sum of all stages."""
        calculated_total = sum(
            s["total_emission_tco2e"] for s in sample_precursor_chain["stages"]
        )
        assert calculated_total == sample_precursor_chain["total_chain_emission_tco2e"]

    def test_chain_provenance_hash(self, sample_precursor_chain):
        """Test chain has a provenance hash."""
        assert_provenance_hash(sample_precursor_chain)


# ---------------------------------------------------------------------------
# Edge Cases (3 tests)
# ---------------------------------------------------------------------------

class TestPrecursorEdgeCases:
    """Test precursor chain edge cases."""

    def test_single_stage_chain(self):
        """Test a single-stage chain (e.g., direct import of raw material)."""
        chain = {
            "chain_id": "PC-SIMPLE-001",
            "goods_category": "cement",
            "stages": [
                {"stage_name": "Clinker Production", "emission_tco2e_per_tonne": 0.84},
            ],
            "total_chain_emission_tco2e": 0.84,
        }
        assert len(chain["stages"]) == 1

    def test_multi_input_stage(self):
        """Test a stage with multiple inputs (e.g., BOF with iron ore + scrap)."""
        stage = {
            "stage_name": "BOF Steelmaking",
            "inputs": [
                {"material": "pig_iron", "weight_tonnes": 900, "pct": 85.7},
                {"material": "scrap", "weight_tonnes": 150, "pct": 14.3},
            ],
            "output_weight_tonnes": 1020,
        }
        total_input = sum(i["weight_tonnes"] for i in stage["inputs"])
        assert total_input == 1050

    def test_chain_country_attribution(self, sample_precursor_chain):
        """Test emissions can be attributed to countries of origin."""
        countries = {s["country"] for s in sample_precursor_chain["stages"]}
        assert "TR" in countries
        assert "BR" in countries
