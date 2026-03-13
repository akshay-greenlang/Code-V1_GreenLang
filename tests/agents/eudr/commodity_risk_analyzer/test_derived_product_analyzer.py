# -*- coding: utf-8 -*-
"""
Unit tests for DerivedProductAnalyzer (AGENT-EUDR-018 Engine 2).

Tests derived product analysis including Annex I product mapping, processing
chain risk accumulation, transformation ratios, risk multipliers, traceability
loss calculation, mislabeling detection, batch analysis, provenance hashing,
and error handling for all 7 EUDR commodities and their derived products.

Author: GreenLang Platform Team
Date: March 2026
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Dict, List

import pytest

from greenlang.agents.eudr.commodity_risk_analyzer.derived_product_analyzer import (
    ANNEX_I_PRODUCT_MAP,
    DerivedProductAnalyzer,
    EUDR_PRIMARY_COMMODITIES,
    MISLABELING_INDICATORS,
    PROCESSING_STAGE_RISK,
)


# =========================================================================
# TestInit
# =========================================================================


class TestInit:
    """Tests for DerivedProductAnalyzer initialization."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Analyzer initializes with empty cache."""
        analyzer = DerivedProductAnalyzer()
        assert analyzer.cached_product_count == 0

    @pytest.mark.unit
    def test_custom_config(self):
        """Analyzer accepts optional config dictionary."""
        analyzer = DerivedProductAnalyzer(config={"test_key": "test_value"})
        assert analyzer.cached_product_count == 0

    @pytest.mark.unit
    def test_repr(self, derived_product_analyzer):
        """Repr contains class name and commodity count."""
        r = repr(derived_product_analyzer)
        assert "DerivedProductAnalyzer" in r
        assert "cached_products=0" in r


# =========================================================================
# TestAnalyzeDerivedProduct
# =========================================================================


class TestAnalyzeDerivedProduct:
    """Tests for analyze_derived_product method."""

    @pytest.mark.unit
    def test_analyze_returns_required_keys(self, derived_product_analyzer):
        """Analysis result contains all required top-level keys."""
        result = derived_product_analyzer.analyze_derived_product(
            product_id="test-product-001",
            source_commodity="cocoa",
            processing_stages=["fermentation", "drying", "roasting"],
        )
        required_keys = {
            "product_id",
            "source_commodity",
            "processing_stages",
            "stage_count",
            "transformation_risk",
            "risk_per_stage",
            "traceability_loss",
            "risk_multiplier",
            "annex_i_match",
            "provenance_hash",
            "processing_time_ms",
        }
        assert required_keys.issubset(set(result.keys()))

    @pytest.mark.unit
    def test_analyze_scores_in_range(self, derived_product_analyzer):
        """Transformation risk and traceability loss are within 0-100."""
        result = derived_product_analyzer.analyze_derived_product(
            product_id="test-002",
            source_commodity="oil_palm",
            processing_stages=["harvest", "milling", "refinery"],
        )
        assert Decimal("0") <= result["transformation_risk"] <= Decimal("100")
        assert Decimal("0") <= result["traceability_loss"] <= Decimal("100")

    @pytest.mark.unit
    def test_analyze_with_input_quantity(self, derived_product_analyzer):
        """Expected output quantity is calculated when input_quantity is given."""
        result = derived_product_analyzer.analyze_derived_product(
            product_id="test-003",
            source_commodity="cocoa",
            processing_stages=["fermentation", "drying", "roasting", "conching", "moulding"],
            input_quantity=Decimal("1000"),
        )
        # Annex I match should give a transformation ratio
        if result["annex_i_match"] is not None:
            assert result["expected_output_quantity"] is not None
            assert result["expected_output_quantity"] > Decimal("0")

    @pytest.mark.unit
    def test_analyze_caches_result(self, derived_product_analyzer):
        """Analysis result is cached by product_id."""
        derived_product_analyzer.analyze_derived_product(
            product_id="cache-test-001",
            source_commodity="soya",
            processing_stages=["cleaning", "crushing"],
        )
        assert derived_product_analyzer.cached_product_count >= 1

    @pytest.mark.unit
    def test_analyze_stage_count(self, derived_product_analyzer):
        """Stage count equals length of processing stages list."""
        stages = ["tapping", "cup_lump_collection", "processing_plant", "tire_factory"]
        result = derived_product_analyzer.analyze_derived_product(
            product_id="stage-count-001",
            source_commodity="rubber",
            processing_stages=stages,
        )
        assert result["stage_count"] == 4


# =========================================================================
# TestMapProcessingChain
# =========================================================================


class TestMapProcessingChain:
    """Tests for map_processing_chain method."""

    @pytest.mark.unit
    def test_map_chain_by_product_id(self, derived_product_analyzer):
        """Map chain by exact product_id succeeds."""
        result = derived_product_analyzer.map_processing_chain(
            source_commodity="cocoa",
            final_product="cocoa-chocolate-bars",
        )
        assert result["source_commodity"] == "cocoa"
        assert len(result["chain"]) > 0
        assert "cn_codes" in result

    @pytest.mark.unit
    def test_map_chain_by_product_name(self, derived_product_analyzer):
        """Map chain by product name (case-insensitive) succeeds."""
        result = derived_product_analyzer.map_processing_chain(
            source_commodity="wood",
            final_product="Charcoal",
        )
        assert result["source_commodity"] == "wood"
        assert result["total_stages"] > 0

    @pytest.mark.unit
    def test_map_chain_cumulative_risk(self, derived_product_analyzer):
        """Cumulative risk increases along the chain."""
        result = derived_product_analyzer.map_processing_chain(
            source_commodity="oil_palm",
            final_product="palm-crude-oil",
        )
        chain = result["chain"]
        for i in range(1, len(chain)):
            assert chain[i]["cumulative_risk"] >= chain[i - 1]["cumulative_risk"]

    @pytest.mark.unit
    def test_map_chain_unknown_product_raises(self, derived_product_analyzer):
        """Unknown product raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            derived_product_analyzer.map_processing_chain(
                source_commodity="cocoa",
                final_product="nonexistent-product",
            )

    @pytest.mark.unit
    def test_map_chain_transformation_ratio(self, derived_product_analyzer):
        """Transformation ratio is returned and positive."""
        result = derived_product_analyzer.map_processing_chain(
            source_commodity="soya",
            final_product="soya-meal",
        )
        assert result["transformation_ratio"] > Decimal("0")
        assert result["transformation_ratio"] <= Decimal("1")


# =========================================================================
# TestAnnexIMapping
# =========================================================================


class TestAnnexIMapping:
    """Parametrized tests for Annex I mapping across all 7 commodities."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "commodity",
        ["cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"],
    )
    def test_annex_i_mapping_per_commodity(self, derived_product_analyzer, commodity):
        """Each commodity has at least 1 Annex I derived product mapping."""
        products = derived_product_analyzer.get_annex_i_mapping(commodity)
        assert len(products) >= 1
        for product in products:
            assert "product_id" in product
            assert "product_name" in product
            assert "processing_stages" in product
            assert "transformation_ratio" in product
            assert product["transformation_ratio"] > Decimal("0")

    @pytest.mark.unit
    def test_annex_i_mapping_invalid_commodity(self, derived_product_analyzer):
        """Invalid commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            derived_product_analyzer.get_annex_i_mapping("banana")


# =========================================================================
# TestTransformationRisk
# =========================================================================


class TestTransformationRisk:
    """Tests for calculate_transformation_risk method."""

    @pytest.mark.unit
    def test_risk_increases_with_stages(self, derived_product_analyzer):
        """More stages accumulate more transformation risk."""
        short = derived_product_analyzer.calculate_transformation_risk(
            ["harvest", "milling"],
        )
        long = derived_product_analyzer.calculate_transformation_risk(
            ["harvest", "milling", "refinery", "transesterification"],
        )
        assert long > short

    @pytest.mark.unit
    def test_risk_clamped_to_100(self, derived_product_analyzer):
        """Very long chain is clamped to maximum of 100."""
        # Many high-risk stages
        stages = ["refinery", "transesterification", "slaughterhouse", "tannery"] * 10
        result = derived_product_analyzer.calculate_transformation_risk(stages)
        assert result <= Decimal("100.00")

    @pytest.mark.unit
    def test_empty_stages_raises(self, derived_product_analyzer):
        """Empty processing stages raise ValueError."""
        with pytest.raises(ValueError, match="must not be empty"):
            derived_product_analyzer.calculate_transformation_risk([])

    @pytest.mark.unit
    def test_unknown_stage_uses_default(self, derived_product_analyzer):
        """Unknown processing stage uses DEFAULT risk increment."""
        result = derived_product_analyzer.calculate_transformation_risk(
            ["unknown_stage"],
        )
        assert result == PROCESSING_STAGE_RISK["DEFAULT"]


# =========================================================================
# TestRiskMultiplier
# =========================================================================


class TestRiskMultiplier:
    """Tests for calculate_risk_multiplier method."""

    @pytest.mark.unit
    def test_multiplier_minimum_is_one(self, derived_product_analyzer):
        """Risk multiplier for zero-length chain is at least 1.0."""
        result = derived_product_analyzer.calculate_risk_multiplier(
            chain_length=0, processing_types=[],
        )
        assert result >= Decimal("1.00")

    @pytest.mark.unit
    def test_multiplier_increases_with_length(self, derived_product_analyzer):
        """Longer chains produce higher risk multipliers."""
        short = derived_product_analyzer.calculate_risk_multiplier(
            chain_length=2, processing_types=["drying", "roasting"],
        )
        long = derived_product_analyzer.calculate_risk_multiplier(
            chain_length=8, processing_types=[
                "drying", "roasting", "refinery", "milling",
                "blending", "pressing", "extraction", "carbonisation",
            ],
        )
        assert long > short

    @pytest.mark.unit
    def test_high_risk_stages_increase_multiplier(self, derived_product_analyzer):
        """High-risk stages (risk >= 4.0) add extra multiplier."""
        low_risk = derived_product_analyzer.calculate_risk_multiplier(
            chain_length=3, processing_types=["drying", "cleaning", "washing"],
        )
        high_risk = derived_product_analyzer.calculate_risk_multiplier(
            chain_length=3, processing_types=["refinery", "transesterification", "slaughterhouse"],
        )
        assert high_risk > low_risk

    @pytest.mark.unit
    def test_negative_chain_length_raises(self, derived_product_analyzer):
        """Negative chain length raises ValueError."""
        with pytest.raises(ValueError, match="chain_length must be >= 0"):
            derived_product_analyzer.calculate_risk_multiplier(
                chain_length=-1, processing_types=[],
            )


# =========================================================================
# TestTraceOrigin
# =========================================================================


class TestTraceOrigin:
    """Tests for trace_commodity_origin method."""

    @pytest.mark.unit
    def test_trace_known_product(self, derived_product_analyzer):
        """Tracing a known Annex I product_id returns full origin info."""
        result = derived_product_analyzer.trace_commodity_origin("cocoa-butter")
        assert result["source_commodity"] == "cocoa"
        assert "processing_chain" in result
        assert "forward_chain" in result
        assert result["traceability_assessment"] in ("FULL", "PARTIAL", "LIMITED")

    @pytest.mark.unit
    def test_trace_unknown_product_raises(self, derived_product_analyzer):
        """Tracing unknown product_id raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            derived_product_analyzer.trace_commodity_origin("nonexistent-id")

    @pytest.mark.unit
    def test_trace_empty_product_id_raises(self, derived_product_analyzer):
        """Empty product_id raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            derived_product_analyzer.trace_commodity_origin("")

    @pytest.mark.unit
    def test_trace_short_chain_is_full(self, derived_product_analyzer):
        """Short processing chain (<=3 stages) has FULL traceability."""
        result = derived_product_analyzer.trace_commodity_origin("cattle-tallow")
        # cattle-tallow has 2 stages: slaughterhouse, rendering_plant
        assert result["stage_count"] <= 3
        assert result["traceability_assessment"] == "FULL"


# =========================================================================
# TestMislabelingDetection
# =========================================================================


class TestMislabelingDetection:
    """Tests for detect_product_mislabeling method."""

    @pytest.mark.unit
    def test_no_discrepancies_clean_product(self, derived_product_analyzer):
        """Product with matching characteristics shows no mislabeling."""
        result = derived_product_analyzer.detect_product_mislabeling(
            declared_product={"commodity_type": "cocoa", "product_name": "cocoa butter"},
            actual_characteristics={},
        )
        assert result["is_mislabeled"] is False
        assert result["risk_level"] == "LOW"

    @pytest.mark.unit
    def test_commodity_mismatch_critical(self, derived_product_analyzer):
        """Commodity type mismatch is flagged as CRITICAL."""
        result = derived_product_analyzer.detect_product_mislabeling(
            declared_product={"commodity_type": "cocoa", "product_name": "cocoa"},
            actual_characteristics={"detected_commodity": "soya"},
        )
        assert result["is_mislabeled"] is True
        assert result["risk_level"] == "CRITICAL"
        assert len(result["discrepancies"]) >= 1

    @pytest.mark.unit
    def test_origin_mismatch_high(self, derived_product_analyzer):
        """Origin country mismatch is flagged as discrepancy."""
        result = derived_product_analyzer.detect_product_mislabeling(
            declared_product={
                "commodity_type": "coffee",
                "origin_country": "ET",
            },
            actual_characteristics={"detected_origin": "VN"},
        )
        assert result["is_mislabeled"] is True
        assert any(d["field"] == "origin_country" for d in result["discrepancies"])

    @pytest.mark.unit
    def test_invalid_certification_flagged(self, derived_product_analyzer):
        """Invalid certification is detected as discrepancy."""
        result = derived_product_analyzer.detect_product_mislabeling(
            declared_product={
                "commodity_type": "oil_palm",
                "certification": "RSPO",
            },
            actual_characteristics={"certification_valid": False},
        )
        assert result["is_mislabeled"] is True
        assert any(d["field"] == "certification" for d in result["discrepancies"])

    @pytest.mark.unit
    def test_missing_commodity_type_raises(self, derived_product_analyzer):
        """Missing commodity_type in declared_product raises ValueError."""
        with pytest.raises(ValueError, match="commodity_type"):
            derived_product_analyzer.detect_product_mislabeling(
                declared_product={"product_name": "test"},
                actual_characteristics={},
            )

    @pytest.mark.unit
    def test_known_indicators_returned(self, derived_product_analyzer):
        """Known fraud indicators for the commodity are included."""
        result = derived_product_analyzer.detect_product_mislabeling(
            declared_product={"commodity_type": "wood"},
            actual_characteristics={},
        )
        assert "known_indicators" in result
        assert len(result["known_indicators"]) > 0


# =========================================================================
# TestTraceabilityLoss
# =========================================================================


class TestTraceabilityLoss:
    """Tests for calculate_traceability_loss method."""

    @pytest.mark.unit
    def test_empty_chain_zero_loss(self, derived_product_analyzer):
        """Empty stage list produces zero traceability loss."""
        result = derived_product_analyzer.calculate_traceability_loss(
            {"stages": []},
        )
        assert result == Decimal("0.00")

    @pytest.mark.unit
    def test_mixing_stages_high_loss(self, derived_product_analyzer):
        """Mixing stages (milling, blending) cause higher traceability loss."""
        low_loss = derived_product_analyzer.calculate_traceability_loss(
            {"stages": ["drying", "washing"]},
        )
        high_loss = derived_product_analyzer.calculate_traceability_loss(
            {"stages": ["milling", "blending", "solvent_extraction"]},
        )
        assert high_loss > low_loss

    @pytest.mark.unit
    def test_loss_clamped_to_100(self, derived_product_analyzer):
        """Very long chain loss is clamped to 100."""
        stages = ["milling", "blending", "refinery", "rendering_plant"] * 10
        result = derived_product_analyzer.calculate_traceability_loss(
            {"stages": stages},
        )
        assert result <= Decimal("100.00")


# =========================================================================
# TestBatchAnalyze
# =========================================================================


class TestBatchAnalyze:
    """Tests for batch_analyze method."""

    @pytest.mark.unit
    def test_batch_multiple_products(self, derived_product_analyzer):
        """Batch analysis processes multiple products."""
        products = [
            {
                "product_id": "batch-001",
                "source_commodity": "cocoa",
                "processing_stages": ["fermentation", "drying"],
            },
            {
                "product_id": "batch-002",
                "source_commodity": "soya",
                "processing_stages": ["cleaning", "crushing"],
            },
        ]
        results = derived_product_analyzer.batch_analyze(products)
        assert len(results) == 2
        success_count = sum(1 for r in results if "error" not in r)
        assert success_count == 2

    @pytest.mark.unit
    def test_batch_handles_errors_gracefully(self, derived_product_analyzer):
        """Batch analysis includes error details for invalid entries."""
        products = [
            {
                "product_id": "good-001",
                "source_commodity": "cocoa",
                "processing_stages": ["fermentation"],
            },
            {
                "product_id": "bad-001",
                "source_commodity": "invalid_commodity",
                "processing_stages": ["drying"],
            },
        ]
        results = derived_product_analyzer.batch_analyze(products)
        assert len(results) == 2
        errors = [r for r in results if "error" in r]
        assert len(errors) == 1
        assert errors[0]["status"] == "FAILED"


# =========================================================================
# TestProvenance
# =========================================================================


class TestProvenance:
    """Tests for provenance hash generation on analyses."""

    @pytest.mark.unit
    def test_provenance_hash_64_chars(self, derived_product_analyzer):
        """Provenance hash is a valid 64-character SHA-256 hex string."""
        result = derived_product_analyzer.analyze_derived_product(
            product_id="prov-001",
            source_commodity="rubber",
            processing_stages=["tapping", "cup_lump_collection"],
        )
        assert len(result["provenance_hash"]) == 64
        int(result["provenance_hash"], 16)

    @pytest.mark.unit
    def test_map_chain_has_provenance(self, derived_product_analyzer):
        """Processing chain mapping includes provenance hash."""
        result = derived_product_analyzer.map_processing_chain(
            source_commodity="wood",
            final_product="wood-sawnwood",
        )
        assert len(result["provenance_hash"]) == 64

    @pytest.mark.unit
    def test_trace_origin_has_provenance(self, derived_product_analyzer):
        """Trace origin result includes provenance hash."""
        result = derived_product_analyzer.trace_commodity_origin("cocoa-paste")
        assert len(result["provenance_hash"]) == 64


# =========================================================================
# TestErrorHandling
# =========================================================================


class TestErrorHandling:
    """Tests for error handling and input validation."""

    @pytest.mark.unit
    def test_analyze_empty_product_id_raises(self, derived_product_analyzer):
        """Empty product_id raises ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty string"):
            derived_product_analyzer.analyze_derived_product(
                product_id="",
                source_commodity="cocoa",
                processing_stages=["drying"],
            )

    @pytest.mark.unit
    def test_analyze_invalid_commodity_raises(self, derived_product_analyzer):
        """Invalid source commodity raises ValueError."""
        with pytest.raises(ValueError, match="Invalid commodity_type"):
            derived_product_analyzer.analyze_derived_product(
                product_id="err-001",
                source_commodity="banana",
                processing_stages=["drying"],
            )

    @pytest.mark.unit
    def test_analyze_empty_stages_raises(self, derived_product_analyzer):
        """Empty processing stages raise ValueError."""
        with pytest.raises(ValueError, match="must be a non-empty list"):
            derived_product_analyzer.analyze_derived_product(
                product_id="err-002",
                source_commodity="cocoa",
                processing_stages=[],
            )

    @pytest.mark.unit
    def test_clear_cache(self, derived_product_analyzer):
        """clear_cache empties the product analysis cache."""
        derived_product_analyzer.analyze_derived_product(
            product_id="clear-test",
            source_commodity="wood",
            processing_stages=["felling", "sawing"],
        )
        assert derived_product_analyzer.cached_product_count >= 1
        derived_product_analyzer.clear_cache()
        assert derived_product_analyzer.cached_product_count == 0
