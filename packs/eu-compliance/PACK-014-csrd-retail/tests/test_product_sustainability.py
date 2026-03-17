# -*- coding: utf-8 -*-
"""
Unit tests for ProductSustainabilityEngine -- PACK-014 CSRD Retail Engine 4
=============================================================================

Tests ESPR Digital Product Passport completeness, ECGT green claims audit,
PEF environmental footprint scoring, textile microplastic assessment,
and repairability scoring.

Coverage target: 85%+
Total tests: ~40
"""

import importlib.util
import os
import sys

import pytest

# ---------------------------------------------------------------------------
# Dynamic module loading
# ---------------------------------------------------------------------------
ENGINES_DIR = os.path.join(os.path.dirname(__file__), "..", "engines")


def _load(name: str):
    path = os.path.join(ENGINES_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_m = _load("product_sustainability_engine")

ProductSustainabilityEngine = _m.ProductSustainabilityEngine
ProductSustainabilityInput = _m.ProductSustainabilityInput
ProductSustainabilityResult = _m.ProductSustainabilityResult
ProductData = _m.ProductData
DPPCategory = _m.DPPCategory
DPPData = _m.DPPData
DPPCompletenessResult = _m.DPPCompletenessResult
GreenClaim = _m.GreenClaim
GreenClaimType = _m.GreenClaimType
GreenClaimAuditResult = _m.GreenClaimAuditResult
ClaimVerificationStatus = _m.ClaimVerificationStatus
PEFImpactCategory = _m.PEFImpactCategory
PEFImpactData = _m.PEFImpactData
PEFNormalizedResult = _m.PEFNormalizedResult
FiberType = _m.FiberType
FiberComposition = _m.FiberComposition
MicroplasticAssessmentResult = _m.MicroplasticAssessmentResult
RepairabilityInput = _m.RepairabilityInput
RepairabilityGrade = _m.RepairabilityGrade
RepairabilityResult = _m.RepairabilityResult
DPP_MANDATORY_FIELDS = _m.DPP_MANDATORY_FIELDS
ECGT_PROHIBITED_CLAIMS = _m.ECGT_PROHIBITED_CLAIMS
PEF_NORMALIZATION_FACTORS = _m.PEF_NORMALIZATION_FACTORS
MICROPLASTIC_RELEASE_RATES = _m.MICROPLASTIC_RELEASE_RATES
REPAIRABILITY_CRITERIA = _m.REPAIRABILITY_CRITERIA
REPAIRABILITY_THRESHOLDS = _m.REPAIRABILITY_THRESHOLDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_input(**overrides) -> ProductSustainabilityInput:
    """Build a minimal valid ProductSustainabilityInput."""
    defaults = dict(
        organisation_id="ORG001",
        reporting_year=2026,
    )
    defaults.update(overrides)
    return ProductSustainabilityInput(**defaults)


def _make_dpp(**overrides) -> DPPData:
    """Build a minimal DPPData."""
    defaults = dict(
        product_id="PROD001",
        dpp_category=DPPCategory.TEXTILES,
    )
    defaults.update(overrides)
    return DPPData(**defaults)


def _make_product(**overrides) -> ProductData:
    """Build a minimal ProductData."""
    defaults = dict(
        product_id="PROD001",
        product_name="Test Product",
        category="textiles",
        weight_kg=0.5,
    )
    defaults.update(overrides)
    return ProductData(**defaults)


# ===================================================================
# TestInitialization
# ===================================================================


class TestInitialization:
    """Engine instantiation tests."""

    def test_default(self):
        engine = ProductSustainabilityEngine()
        assert engine is not None

    def test_config_dpp_fields(self):
        engine = ProductSustainabilityEngine()
        assert len(engine._dpp_fields) > 0

    def test_config_prohibited_claims(self):
        engine = ProductSustainabilityEngine()
        assert "carbon_neutral" in engine._prohibited_claims

    def test_config_pef_factors(self):
        engine = ProductSustainabilityEngine()
        assert PEFImpactCategory.CLIMATE_CHANGE in engine._pef_factors


# ===================================================================
# TestDPPCategories
# ===================================================================


class TestDPPCategories:
    """DPP category enum tests."""

    def test_all_5_defined(self):
        assert len(DPPCategory) == 5

    def test_textiles_25_fields(self):
        fields = DPP_MANDATORY_FIELDS[DPPCategory.TEXTILES]
        assert len(fields) == 25

    def test_electronics_30_fields(self):
        fields = DPP_MANDATORY_FIELDS[DPPCategory.ELECTRONICS]
        assert len(fields) == 30

    def test_furniture_20_fields(self):
        fields = DPP_MANDATORY_FIELDS[DPPCategory.FURNITURE]
        assert len(fields) == 20


# ===================================================================
# TestDPPCompleteness
# ===================================================================


class TestDPPCompleteness:
    """DPP completeness assessment tests."""

    def test_full_textile_dpp(self):
        engine = ProductSustainabilityEngine()
        # Create a fully-populated DPP with all fields provided
        fields_provided = {f: True for f in DPP_MANDATORY_FIELDS[DPPCategory.TEXTILES]}
        dpp = _make_dpp(
            dpp_category=DPPCategory.TEXTILES,
            recycled_content_pct=30.0,
            carbon_footprint_kg=5.0,
            water_footprint_litre=2000.0,
            has_digital_id=True,
            manufacturing_country="PT",
            recyclability_pct=80.0,
            repairability_score=7.0,
            durability_cycles=50,
            fiber_composition=[
                FiberComposition(fiber_type=FiberType.COTTON, percentage=100.0),
            ],
            fields_provided=fields_provided,
        )
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        assert len(result.dpp_results) == 1
        assert result.dpp_results[0].completeness_pct > 50.0

    def test_partial_dpp(self):
        engine = ProductSustainabilityEngine()
        dpp = _make_dpp(
            dpp_category=DPPCategory.TEXTILES,
            recycled_content_pct=20.0,
            has_digital_id=True,
        )
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        assert result.dpp_results[0].completeness_pct < 100.0

    def test_missing_fields_identified(self):
        engine = ProductSustainabilityEngine()
        dpp = _make_dpp(dpp_category=DPPCategory.TEXTILES)
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        assert len(result.dpp_results[0].missing_fields) > 0

    def test_zero_fields(self):
        engine = ProductSustainabilityEngine()
        dpp = _make_dpp(dpp_category=DPPCategory.TEXTILES)
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        assert result.dpp_results[0].total_mandatory_fields == 25

    def test_multiple_products(self):
        engine = ProductSustainabilityEngine()
        dpps = [
            _make_dpp(product_id="P1", dpp_category=DPPCategory.TEXTILES),
            _make_dpp(product_id="P2", dpp_category=DPPCategory.ELECTRONICS),
        ]
        inp = _make_input(dpp_data=dpps)
        result = engine.assess_products(inp)
        assert len(result.dpp_results) == 2


# ===================================================================
# TestGreenClaimsAudit
# ===================================================================


class TestGreenClaimsAudit:
    """Green claims ECGT audit tests."""

    def test_prohibited_carbon_neutral(self):
        engine = ProductSustainabilityEngine()
        claim = GreenClaim(
            claim_type=GreenClaimType.CARBON_NEUTRAL,
            claim_text="Our product is carbon neutral",
        )
        inp = _make_input(green_claims=[claim])
        result = engine.assess_products(inp)
        assert result.green_claims_audit[0].is_prohibited is True

    def test_prohibited_eco_friendly(self):
        engine = ProductSustainabilityEngine()
        claim = GreenClaim(
            claim_type=GreenClaimType.ECO_FRIENDLY,
            claim_text="Eco-friendly product",
        )
        inp = _make_input(green_claims=[claim])
        result = engine.assess_products(inp)
        assert result.green_claims_audit[0].is_prohibited is True

    def test_allowed_organic_certified(self):
        engine = ProductSustainabilityEngine()
        claim = GreenClaim(
            claim_type=GreenClaimType.ORGANIC,
            claim_text="Certified organic cotton",
            substantiation_evidence="GOTS certification #12345",
            third_party_verified=True,
            certification_scheme="GOTS",
        )
        inp = _make_input(green_claims=[claim])
        result = engine.assess_products(inp)
        audit = result.green_claims_audit[0]
        assert audit.is_prohibited is False
        assert audit.compliance_status == "compliant"

    def test_substantiation_check(self):
        engine = ProductSustainabilityEngine()
        claim = GreenClaim(
            claim_type=GreenClaimType.RECYCLABLE,
            claim_text="100% recyclable",
            substantiation_evidence="Tested per ISO 18604",
            third_party_verified=False,
        )
        inp = _make_input(green_claims=[claim])
        result = engine.assess_products(inp)
        audit = result.green_claims_audit[0]
        assert audit.has_substantiation is True
        assert audit.compliance_status == "requires_verification"

    def test_multiple_claims(self):
        engine = ProductSustainabilityEngine()
        claims = [
            GreenClaim(
                claim_type=GreenClaimType.CARBON_NEUTRAL,
                claim_text="Carbon neutral",
            ),
            GreenClaim(
                claim_type=GreenClaimType.ORGANIC,
                claim_text="Organic",
                substantiation_evidence="Cert #123",
                third_party_verified=True,
            ),
        ]
        inp = _make_input(green_claims=claims)
        result = engine.assess_products(inp)
        assert result.total_claims == 2
        assert result.prohibited_claims_count == 1

    def test_claim_enum_values(self):
        expected = {
            "carbon_neutral", "climate_positive", "eco_friendly",
            "sustainable", "recyclable", "biodegradable", "organic",
            "natural", "vegan", "fair_trade",
        }
        actual = {c.value for c in GreenClaimType}
        assert actual == expected


# ===================================================================
# TestPEFCalculation
# ===================================================================


class TestPEFCalculation:
    """PEF environmental footprint tests."""

    def test_climate_change_impact(self):
        engine = ProductSustainabilityEngine()
        impacts = [
            PEFImpactData(
                category=PEFImpactCategory.CLIMATE_CHANGE,
                characterization_value=50.0,
                unit="kg CO2 eq",
            ),
        ]
        inp = _make_input(pef_data={"PROD001": impacts})
        result = engine.assess_products(inp)
        assert len(result.pef_results) == 1
        pef = result.pef_results[0]
        assert "climate_change" in pef.impact_scores

    def test_normalized_score(self):
        engine = ProductSustainabilityEngine()
        impacts = [
            PEFImpactData(
                category=PEFImpactCategory.CLIMATE_CHANGE,
                characterization_value=810.0,  # 10% of normalization factor (8100)
            ),
        ]
        inp = _make_input(pef_data={"PROD001": impacts})
        result = engine.assess_products(inp)
        pef = result.pef_results[0]
        # Normalized = 810 / 8100 = 0.1
        scores = pef.impact_scores["climate_change"]
        assert scores["normalized"] == pytest.approx(0.1, rel=1e-2)

    def test_weighted_single_score(self):
        engine = ProductSustainabilityEngine()
        impacts = [
            PEFImpactData(
                category=PEFImpactCategory.CLIMATE_CHANGE,
                characterization_value=8100.0,
            ),
        ]
        inp = _make_input(pef_data={"PROD001": impacts})
        result = engine.assess_products(inp)
        pef = result.pef_results[0]
        # normalized = 1.0, weighted = 1.0 * 21.06/100 = 0.2106
        assert pef.total_weighted_score == pytest.approx(0.2106, rel=1e-2)

    def test_all_impact_categories(self):
        assert len(PEFImpactCategory) == 16


# ===================================================================
# TestMicroplastics
# ===================================================================


class TestMicroplastics:
    """Textile microplastic release tests."""

    def test_polyester_release(self):
        engine = ProductSustainabilityEngine()
        dpp = _make_dpp(
            dpp_category=DPPCategory.TEXTILES,
            fiber_composition=[
                FiberComposition(fiber_type=FiberType.POLYESTER, percentage=100.0),
            ],
        )
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        assert len(result.microplastic_assessments) == 1
        mp = result.microplastic_assessments[0]
        # 100% polyester: 124 mg/kg/wash
        assert mp.total_release_mg_per_wash == pytest.approx(124.0, rel=1e-2)

    def test_nylon_release(self):
        engine = ProductSustainabilityEngine()
        dpp = _make_dpp(
            dpp_category=DPPCategory.TEXTILES,
            fiber_composition=[
                FiberComposition(fiber_type=FiberType.NYLON, percentage=100.0),
            ],
        )
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        mp = result.microplastic_assessments[0]
        assert mp.total_release_mg_per_wash == pytest.approx(80.0, rel=1e-2)

    def test_cotton_zero(self):
        engine = ProductSustainabilityEngine()
        dpp = _make_dpp(
            dpp_category=DPPCategory.TEXTILES,
            fiber_composition=[
                FiberComposition(fiber_type=FiberType.COTTON, percentage=100.0),
            ],
        )
        inp = _make_input(dpp_data=[dpp])
        result = engine.assess_products(inp)
        mp = result.microplastic_assessments[0]
        # Cotton = 2 mg/kg/wash, negligible
        assert mp.total_release_mg_per_wash == pytest.approx(2.0, rel=1e-1)
        assert mp.exceeds_threshold is False

    def test_fiber_type_enum(self):
        expected = {
            "polyester", "nylon", "acrylic", "polypropylene",
            "cotton", "wool", "silk", "linen", "viscose", "lyocell",
            "recycled_polyester", "recycled_nylon",
        }
        actual = {f.value for f in FiberType}
        assert actual == expected


# ===================================================================
# TestRepairability
# ===================================================================


class TestRepairability:
    """Repairability scoring tests."""

    def test_grade_a(self):
        engine = ProductSustainabilityEngine()
        repair_input = RepairabilityInput(
            product_id="P1",
            documentation_score=9.0,
            disassembly_score=9.0,
            spare_parts_availability_score=9.0,
            spare_parts_price_score=9.0,
            product_specific_score=9.0,
        )
        inp = _make_input(repairability_data=[repair_input])
        result = engine.assess_products(inp)
        assert result.repairability_results[0].grade == "A"

    def test_grade_e(self):
        engine = ProductSustainabilityEngine()
        repair_input = RepairabilityInput(
            product_id="P1",
            documentation_score=1.0,
            disassembly_score=1.0,
            spare_parts_availability_score=1.0,
            spare_parts_price_score=1.0,
            product_specific_score=1.0,
        )
        inp = _make_input(repairability_data=[repair_input])
        result = engine.assess_products(inp)
        assert result.repairability_results[0].grade == "E"

    def test_weighted_score(self):
        engine = ProductSustainabilityEngine()
        repair_input = RepairabilityInput(
            product_id="P1",
            documentation_score=5.0,
            disassembly_score=5.0,
            spare_parts_availability_score=5.0,
            spare_parts_price_score=5.0,
            product_specific_score=5.0,
        )
        inp = _make_input(repairability_data=[repair_input])
        result = engine.assess_products(inp)
        # All scores are 5, weights sum to 100 -> weighted score = 5.0
        assert result.repairability_results[0].weighted_score == pytest.approx(5.0, rel=1e-2)

    def test_criteria_count(self):
        assert len(REPAIRABILITY_CRITERIA) == 5


# ===================================================================
# TestProductPortfolio
# ===================================================================


class TestProductPortfolio:
    """Product portfolio summary tests."""

    def test_multi_product_summary(self):
        engine = ProductSustainabilityEngine()
        products = [
            _make_product(product_id="P1", product_name="Shirt"),
            _make_product(product_id="P2", product_name="Pants"),
            _make_product(product_id="P3", product_name="Jacket"),
        ]
        inp = _make_input(products=products)
        result = engine.assess_products(inp)
        assert result.total_products == 3

    def test_dpp_coverage_pct(self):
        engine = ProductSustainabilityEngine()
        dpps = [
            _make_dpp(product_id="P1", dpp_category=DPPCategory.TEXTILES),
        ]
        inp = _make_input(dpp_data=dpps)
        result = engine.assess_products(inp)
        assert result.avg_dpp_completeness_pct >= 0.0

    def test_recommendations(self):
        engine = ProductSustainabilityEngine()
        claims = [
            GreenClaim(
                claim_type=GreenClaimType.CARBON_NEUTRAL,
                claim_text="Carbon neutral product",
            ),
        ]
        inp = _make_input(green_claims=claims)
        result = engine.assess_products(inp)
        assert len(result.recommendations) >= 1


# ===================================================================
# TestProvenance
# ===================================================================


class TestProvenance:
    """Provenance hash tests."""

    def test_hash(self):
        engine = ProductSustainabilityEngine()
        inp = _make_input(
            products=[_make_product()],
        )
        result = engine.assess_products(inp)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self):
        engine = ProductSustainabilityEngine()
        inp = _make_input(
            products=[_make_product()],
        )
        r1 = engine.assess_products(inp)
        r2 = engine.assess_products(inp)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input(self):
        engine = ProductSustainabilityEngine()
        inp1 = _make_input(
            organisation_id="ORG001",
            products=[_make_product(product_id="P1")],
        )
        inp2 = _make_input(
            organisation_id="ORG002",
            products=[_make_product(product_id="P2", product_name="Other Product")],
        )
        r1 = engine.assess_products(inp1)
        r2 = engine.assess_products(inp2)
        assert r1.provenance_hash != r2.provenance_hash


# ===================================================================
# TestEdgeCases
# ===================================================================


class TestEdgeCases:
    """Edge case and boundary tests."""

    def test_empty_products(self):
        engine = ProductSustainabilityEngine()
        inp = _make_input()
        result = engine.assess_products(inp)
        assert result.total_products == 0

    def test_single_product(self):
        engine = ProductSustainabilityEngine()
        inp = _make_input(products=[_make_product()])
        result = engine.assess_products(inp)
        assert result.total_products == 1

    def test_result_fields(self):
        engine = ProductSustainabilityEngine()
        inp = _make_input(products=[_make_product()])
        result = engine.assess_products(inp)
        assert isinstance(result, ProductSustainabilityResult)
        assert result.organisation_id == "ORG001"
        assert result.reporting_year == 2026
        assert result.engine_version == "1.0.0"
        assert result.processing_time_ms >= 0
