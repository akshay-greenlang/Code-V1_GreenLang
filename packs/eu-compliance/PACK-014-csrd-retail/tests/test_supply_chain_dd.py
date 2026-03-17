# -*- coding: utf-8 -*-
"""
Unit tests for SupplyChainDueDiligenceEngine (PACK-014, Engine 6)
==================================================================

Tests all methods of SupplyChainDueDiligenceEngine with 85%+ coverage.
Validates business logic, error handling, and edge cases.

Test count: ~41 tests
"""

import importlib.util
import os

import pytest

# ---------------------------------------------------------------------------
# Dynamic import via importlib
# ---------------------------------------------------------------------------

_ENGINE_PATH = os.path.join(
    os.path.dirname(__file__),
    os.pardir,
    "engines",
    "supply_chain_due_diligence_engine.py",
)
_ENGINE_PATH = os.path.normpath(_ENGINE_PATH)

_spec = importlib.util.spec_from_file_location("supply_chain_due_diligence_engine", _ENGINE_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

SupplyChainDueDiligenceEngine = _mod.SupplyChainDueDiligenceEngine
SupplierProfile = _mod.SupplierProfile
SupplyChainDDResult = _mod.SupplyChainDDResult
RiskAssessment = _mod.RiskAssessment
DueDiligenceRisk = _mod.DueDiligenceRisk
EUDRCommodity = _mod.EUDRCommodity
EUDRCommodityTrace = _mod.EUDRCommodityTrace
RemediationAction = _mod.RemediationAction
RemediationStatus = _mod.RemediationStatus
HumanRightsIssue = _mod.HumanRightsIssue
SupplierTier = _mod.SupplierTier
CSDDDApplicability = _mod.CSDDDApplicability
COUNTRY_RISK_SCORES = _mod.COUNTRY_RISK_SCORES
SECTOR_RISK_SCORES = _mod.SECTOR_RISK_SCORES
FORCED_LABOUR_INDICATORS = _mod.FORCED_LABOUR_INDICATORS
EUDR_HIGH_RISK_COUNTRIES = _mod.EUDR_HIGH_RISK_COUNTRIES
CSDDD_PHASE_THRESHOLDS = _mod.CSDDD_PHASE_THRESHOLDS
COMMODITY_RISK_SCORES = _mod.COMMODITY_RISK_SCORES


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine():
    """Create a SupplyChainDueDiligenceEngine instance."""
    return SupplyChainDueDiligenceEngine()


@pytest.fixture
def sample_suppliers():
    """Create a list of sample supplier profiles."""
    return [
        SupplierProfile(
            supplier_id="SUP-001",
            name="FreshPalm Ltd",
            country="ID",
            sector="palm_oil_production",
            tier=SupplierTier.TIER_2,
            commodities_supplied=["palm_oil"],
            incident_count=2,
        ),
        SupplierProfile(
            supplier_id="SUP-002",
            name="EuroPack GmbH",
            country="DE",
            sector="packaging",
            tier=SupplierTier.TIER_1,
            commodities_supplied=[],
            incident_count=0,
        ),
        SupplierProfile(
            supplier_id="SUP-003",
            name="TextileBD",
            country="BD",
            sector="garments_textiles",
            tier=SupplierTier.TIER_3,
            commodities_supplied=[],
            incident_count=3,
            forced_labour_indicators_present=["withholding_wages", "excessive_overtime"],
        ),
    ]


@pytest.fixture
def low_risk_supplier():
    """Create a low-risk supplier."""
    return SupplierProfile(
        supplier_id="SUP-LOW",
        name="SwedishWood AB",
        country="SE",
        sector="technology_services",
        tier=SupplierTier.TIER_1,
        commodities_supplied=[],
        incident_count=0,
        certifications=["FSC", "RSPO"],
    )


@pytest.fixture
def high_risk_supplier():
    """Create a high-risk supplier."""
    return SupplierProfile(
        supplier_id="SUP-HIGH",
        name="MyanmarMinerals",
        country="MM",
        sector="mining_minerals",
        tier=SupplierTier.TIER_3,
        commodities_supplied=[],
        incident_count=5,
        forced_labour_indicators_present=[
            "abuse_of_vulnerability",
            "restriction_of_movement",
            "withholding_wages",
        ],
    )


# ===========================================================================
# TestInitialization
# ===========================================================================


class TestInitialization:
    """Test engine initialisation."""

    def test_default_instantiation(self):
        """Engine can be created with no arguments."""
        engine = SupplyChainDueDiligenceEngine()
        assert engine is not None

    def test_engine_version(self):
        """Engine exposes a version string."""
        engine = SupplyChainDueDiligenceEngine()
        assert engine.engine_version == "1.0.0"

    def test_config_dict(self):
        """Engine accepts attribute changes."""
        engine = SupplyChainDueDiligenceEngine()
        engine.engine_version = "2.0.0"
        assert engine.engine_version == "2.0.0"

    def test_none_suppliers_raises(self, engine):
        """Passing empty supplier list raises ValueError."""
        with pytest.raises(ValueError, match="At least one"):
            engine.calculate([])


# ===========================================================================
# TestRiskLevels
# ===========================================================================


class TestRiskLevels:
    """Test DueDiligenceRisk enum."""

    def test_all_5_defined(self):
        """There must be exactly 5 risk levels."""
        assert len(DueDiligenceRisk) == 5

    def test_very_high(self):
        """VERY_HIGH value is correct."""
        assert DueDiligenceRisk.VERY_HIGH.value == "very_high"

    def test_negligible(self):
        """NEGLIGIBLE value is correct."""
        assert DueDiligenceRisk.NEGLIGIBLE.value == "negligible"

    def test_risk_enum_values(self):
        """All expected risk values present."""
        values = {r.value for r in DueDiligenceRisk}
        assert values == {"very_high", "high", "medium", "low", "negligible"}


# ===========================================================================
# TestCountryRisk
# ===========================================================================


class TestCountryRisk:
    """Test country risk scoring."""

    def test_high_risk_country(self):
        """Bangladesh has risk score of 4.2 (high risk)."""
        assert COUNTRY_RISK_SCORES["BD"] == pytest.approx(4.2, rel=1e-6)

    def test_low_risk_country(self):
        """Sweden has risk score of 1.0 (low risk)."""
        assert COUNTRY_RISK_SCORES["SE"] == pytest.approx(1.0, rel=1e-6)

    def test_50_countries_defined(self):
        """At least 50 countries have risk scores defined."""
        assert len(COUNTRY_RISK_SCORES) >= 50

    def test_score_range_1_5(self):
        """All country scores are between 1.0 and 5.0."""
        for country, score in COUNTRY_RISK_SCORES.items():
            assert 1.0 <= score <= 5.0, f"{country} score {score} out of range"

    def test_unknown_country(self, engine):
        """Unknown country defaults to 2.5 risk."""
        supplier = SupplierProfile(
            supplier_id="SUP-UNK",
            name="Unknown Corp",
            country="ZZ",
            sector="other",
            tier=SupplierTier.TIER_1,
        )
        result = engine.assess_single_supplier(supplier)
        assert result["country_risk"] == pytest.approx(2.5, rel=1e-2)


# ===========================================================================
# TestSectorRisk
# ===========================================================================


class TestSectorRisk:
    """Test sector risk scoring."""

    def test_garments_high(self):
        """Garments/textiles sector has risk score of 4.2."""
        assert SECTOR_RISK_SCORES["garments_textiles"] == pytest.approx(4.2, rel=1e-6)

    def test_food_processing_medium(self):
        """Food processing has risk score of 3.0."""
        assert SECTOR_RISK_SCORES["food_processing"] == pytest.approx(3.0, rel=1e-6)

    def test_electronics(self):
        """Electronics has risk score of 3.5."""
        assert SECTOR_RISK_SCORES["electronics"] == pytest.approx(3.5, rel=1e-6)

    def test_sector_count(self):
        """At least 20 sectors are defined."""
        assert len(SECTOR_RISK_SCORES) >= 20


# ===========================================================================
# TestCompositeRisk
# ===========================================================================


class TestCompositeRisk:
    """Test composite risk score calculation."""

    def test_weighted_calculation(self, engine, sample_suppliers):
        """Composite score is calculated for all suppliers."""
        result = engine.calculate(sample_suppliers)
        assert result.avg_composite_risk > 0.0

    def test_high_risk_supplier(self, engine, high_risk_supplier):
        """High-risk supplier should have composite >= 3.5 (very high risk)."""
        result = engine.assess_single_supplier(high_risk_supplier)
        assert result["composite_score"] >= 3.5

    def test_low_risk_supplier(self, engine, low_risk_supplier):
        """Low-risk supplier should have composite < 2.0."""
        result = engine.assess_single_supplier(low_risk_supplier)
        assert result["composite_score"] < 2.0

    def test_risk_distribution(self, engine, sample_suppliers):
        """Risk distribution sums to total suppliers."""
        result = engine.calculate(sample_suppliers)
        rd = result.risk_distribution
        total = rd.very_high + rd.high + rd.medium + rd.low + rd.negligible
        assert total == len(sample_suppliers)

    def test_all_tiers_counted(self, engine, sample_suppliers):
        """Risk by tier has entries for each represented tier."""
        result = engine.calculate(sample_suppliers)
        assert len(result.risk_by_tier) >= 1


# ===========================================================================
# TestEUDRCompliance
# ===========================================================================


class TestEUDRCompliance:
    """Test EUDR commodity compliance assessment."""

    def test_7_commodities_defined(self):
        """EUDRCommodity enum has exactly 7 members."""
        assert len(EUDRCommodity) == 7

    def test_palm_oil_trace(self, engine):
        """Palm oil trace with all documents is fully compliant."""
        traces = [
            EUDRCommodityTrace(
                commodity=EUDRCommodity.PALM_OIL,
                origin_country="ID",
                volume_tonnes=100.0,
                has_geolocation=True,
                has_deforestation_declaration=True,
                has_legality_proof=True,
            ),
        ]
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Palm Co",
                country="ID", sector="palm_oil_production",
                tier=SupplierTier.TIER_2, commodities_supplied=["palm_oil"],
            ),
        ]
        result = engine.calculate(suppliers, eudr_traces=traces)
        assert result.eudr_summary is not None
        assert result.eudr_summary.compliance_rate_pct == pytest.approx(100.0, rel=1e-2)

    def test_multi_commodity_product(self, engine):
        """Multiple EUDR commodities are tracked separately."""
        traces = [
            EUDRCommodityTrace(
                commodity=EUDRCommodity.PALM_OIL,
                origin_country="ID",
                volume_tonnes=50.0,
                has_geolocation=True,
                has_deforestation_declaration=True,
                has_legality_proof=True,
            ),
            EUDRCommodityTrace(
                commodity=EUDRCommodity.COCOA,
                origin_country="CI",
                volume_tonnes=30.0,
                has_geolocation=False,
                has_deforestation_declaration=True,
                has_legality_proof=False,
            ),
        ]
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Multi Co",
                country="ID", sector="food_processing",
                tier=SupplierTier.TIER_1,
            ),
        ]
        result = engine.calculate(suppliers, eudr_traces=traces)
        assert result.eudr_summary.total_commodities_traced == 2
        assert result.eudr_summary.compliance_rate_pct < 100.0

    def test_compliance_rate(self, engine):
        """Compliance rate is 0% when no documents available."""
        traces = [
            EUDRCommodityTrace(
                commodity=EUDRCommodity.SOY,
                origin_country="BR",
                volume_tonnes=200.0,
                has_geolocation=False,
                has_deforestation_declaration=False,
                has_legality_proof=False,
            ),
        ]
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Soy Co",
                country="BR", sector="agriculture",
                tier=SupplierTier.TIER_2,
            ),
        ]
        result = engine.calculate(suppliers, eudr_traces=traces)
        assert result.eudr_summary.compliance_rate_pct == pytest.approx(0.0, abs=1e-6)

    def test_high_risk_countries(self, engine):
        """High risk countries include Brazil and Indonesia."""
        assert "BR" in EUDR_HIGH_RISK_COUNTRIES
        assert "ID" in EUDR_HIGH_RISK_COUNTRIES


# ===========================================================================
# TestForcedLabour
# ===========================================================================


class TestForcedLabour:
    """Test forced labour screening."""

    def test_11_ilo_indicators(self):
        """FORCED_LABOUR_INDICATORS has exactly 11 entries."""
        assert len(FORCED_LABOUR_INDICATORS) == 11

    def test_flag_high_risk(self, engine, high_risk_supplier):
        """Supplier with FL indicators is flagged."""
        result = engine.screen_forced_labour(high_risk_supplier)
        assert result["forced_labour_risk_level"] == "CRITICAL"

    def test_no_flag_low_risk(self, engine, low_risk_supplier):
        """Low-risk supplier without indicators is not flagged critical."""
        result = engine.screen_forced_labour(low_risk_supplier)
        assert result["forced_labour_risk_level"] == "LOW"

    def test_screening_result(self, engine, high_risk_supplier):
        """Screening result contains provenance hash."""
        result = engine.screen_forced_labour(high_risk_supplier)
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# TestCSDDDApplicability
# ===========================================================================


class TestCSDDDApplicability:
    """Test CSDDD phase applicability determination."""

    def test_phase_1_large(self, engine):
        """Large company (6000 emp, 2B EUR) is Phase 1."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Big Co",
                country="DE", sector="retail_wholesale",
                tier=SupplierTier.TIER_1,
            ),
        ]
        result = engine.calculate(
            suppliers, employee_count=6000, turnover_eur=2_000_000_000.0,
        )
        assert result.csddd_applicability.in_scope is True
        assert result.csddd_applicability.phase == 1

    def test_phase_2_medium(self, engine):
        """Medium company (4000 emp, 1B EUR) is Phase 2."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Med Co",
                country="DE", sector="retail_wholesale",
                tier=SupplierTier.TIER_1,
            ),
        ]
        result = engine.calculate(
            suppliers, employee_count=4000, turnover_eur=1_000_000_000.0,
        )
        assert result.csddd_applicability.in_scope is True
        assert result.csddd_applicability.phase == 2

    def test_phase_3_threshold(self, engine):
        """Phase 3 threshold company (1500 emp, 500M EUR) is Phase 3."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Small-Med Co",
                country="DE", sector="retail_wholesale",
                tier=SupplierTier.TIER_1,
            ),
        ]
        result = engine.calculate(
            suppliers, employee_count=1500, turnover_eur=500_000_000.0,
        )
        assert result.csddd_applicability.in_scope is True
        assert result.csddd_applicability.phase == 3

    def test_not_applicable_sme(self, engine):
        """SME below all thresholds is not in scope."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="SME Co",
                country="DE", sector="retail_wholesale",
                tier=SupplierTier.TIER_1,
            ),
        ]
        result = engine.calculate(
            suppliers, employee_count=200, turnover_eur=50_000_000.0,
        )
        assert result.csddd_applicability.in_scope is False


# ===========================================================================
# TestRemediation
# ===========================================================================


class TestRemediation:
    """Test remediation action tracking."""

    def test_action_tracking(self, engine):
        """Remediation actions are summarised correctly."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="Supplier A",
                country="BD", sector="garments_textiles",
                tier=SupplierTier.TIER_1,
            ),
        ]
        actions = [
            RemediationAction(
                supplier_id="S1",
                issue=HumanRightsIssue.FORCED_LABOUR,
                action_plan="Audit and remediate",
                status=RemediationStatus.IN_PROGRESS,
            ),
            RemediationAction(
                supplier_id="S1",
                issue=HumanRightsIssue.LIVING_WAGE,
                action_plan="Wage increase programme",
                status=RemediationStatus.COMPLETED,
            ),
        ]
        result = engine.calculate(suppliers, remediation_actions=actions)
        assert result.remediation_summary is not None
        assert result.remediation_summary.total_actions == 2

    def test_status_enum(self):
        """RemediationStatus has all expected values."""
        values = {s.value for s in RemediationStatus}
        assert "not_started" in values
        assert "completed" in values
        assert "verified" in values
        assert "failed" in values

    def test_verification(self, engine):
        """Verified actions contribute to verification rate."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="X",
                country="DE", sector="other",
                tier=SupplierTier.TIER_1,
            ),
        ]
        actions = [
            RemediationAction(
                supplier_id="S1",
                issue=HumanRightsIssue.HEALTH_SAFETY,
                action_plan="Fix safety issues",
                status=RemediationStatus.VERIFIED,
            ),
        ]
        result = engine.calculate(suppliers, remediation_actions=actions)
        assert result.remediation_summary.verification_rate_pct == pytest.approx(100.0, rel=1e-2)


# ===========================================================================
# TestProvenance
# ===========================================================================


class TestProvenance:
    """Test SHA-256 provenance hashing."""

    def test_hash_length(self, engine, sample_suppliers):
        """Provenance hash is 64 hex characters."""
        result = engine.calculate(sample_suppliers)
        assert len(result.provenance_hash) == 64

    def test_deterministic(self, engine):
        """Provenance hash is a valid hex string derived from result data."""
        suppliers = [
            SupplierProfile(
                supplier_id="S1", name="TestCo",
                country="DE", sector="packaging",
                tier=SupplierTier.TIER_1,
            ),
        ]
        result = engine.calculate(suppliers)
        # Hash is valid hex
        assert all(c in "0123456789abcdef" for c in result.provenance_hash)
        # Recomputing: engine computes hash when provenance_hash is still ""
        from hashlib import sha256
        import json
        saved_hash = result.provenance_hash
        result.provenance_hash = ""
        serialized = json.dumps(result.model_dump(mode="json"), sort_keys=True, default=str)
        expected = sha256(serialized.encode("utf-8")).hexdigest()
        result.provenance_hash = saved_hash
        assert saved_hash == expected

    def test_different_input(self, engine):
        """Different inputs produce different hashes."""
        r1 = engine.calculate([
            SupplierProfile(
                supplier_id="S1", name="A",
                country="DE", sector="packaging",
                tier=SupplierTier.TIER_1,
            ),
        ])
        r2 = engine.calculate([
            SupplierProfile(
                supplier_id="S2", name="B",
                country="FR", sector="packaging",
                tier=SupplierTier.TIER_1,
            ),
        ])
        assert r1.provenance_hash != r2.provenance_hash


# ===========================================================================
# TestEdgeCases
# ===========================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_suppliers_raises(self, engine):
        """Empty supplier list raises ValueError."""
        with pytest.raises(ValueError):
            engine.calculate([])

    def test_single_supplier(self, engine, low_risk_supplier):
        """Engine works with a single supplier."""
        result = engine.calculate([low_risk_supplier])
        assert result.total_suppliers_assessed == 1

    def test_large_supply_chain(self, engine):
        """Engine handles a large number of suppliers."""
        suppliers = [
            SupplierProfile(
                supplier_id=f"SUP-{i}",
                name=f"Supplier {i}",
                country="DE",
                sector="packaging",
                tier=SupplierTier.TIER_1,
            )
            for i in range(200)
        ]
        result = engine.calculate(suppliers)
        assert result.total_suppliers_assessed == 200

    def test_result_fields(self, engine, sample_suppliers):
        """Result object contains all expected fields."""
        result = engine.calculate(sample_suppliers)
        assert hasattr(result, "total_suppliers_assessed")
        assert hasattr(result, "risk_distribution")
        assert hasattr(result, "avg_composite_risk")
        assert hasattr(result, "forced_labour_flags")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")
        assert result.processing_time_ms > 0.0
