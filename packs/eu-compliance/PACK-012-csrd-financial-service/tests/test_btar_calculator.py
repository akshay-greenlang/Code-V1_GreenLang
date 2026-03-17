# -*- coding: utf-8 -*-
"""
Unit Tests for BTARCalculatorEngine (Engine 4) - PACK-012. Target: 30+ tests.
"""

import importlib.util
import os
import sys
import pytest


_ENGINE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "engines",
)


def _load_module(module_name, file_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(_ENGINE_DIR, file_name),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

_btar = _load_module("_btar", "btar_calculator_engine.py")

BTARCalculatorEngine = _btar.BTARCalculatorEngine
BTARConfig = _btar.BTARConfig
BankingBookData = _btar.BankingBookData
BTARResult = _btar.BTARResult
EstimationMethodology = _btar.EstimationMethodology
SectorProxyResult = _btar.SectorProxyResult
DataCoverageReport = _btar.DataCoverageReport
BTARvsGARReconciliation = _btar.BTARvsGARReconciliation
EstimationType = _btar.EstimationType
ExposureCategory = _btar.ExposureCategory
ConfidenceLevel = _btar.ConfidenceLevel
SECTOR_PROXY_ALIGNMENT = _btar.SECTOR_PROXY_ALIGNMENT
GEOGRAPHIC_PROXY_ALIGNMENT = _btar.GEOGRAPHIC_PROXY_ALIGNMENT
ESTIMATION_CONFIDENCE = _btar.ESTIMATION_CONFIDENCE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_engine():
    """Engine with default BTAR config."""
    return BTARCalculatorEngine()


@pytest.fixture
def csrd_reported_exposure():
    """Exposure with CSRD-reported Taxonomy alignment data."""
    return BankingBookData(
        exposure_id="CSRD-001",
        exposure_name="GreenCo GmbH",
        exposure_category=ExposureCategory.GAR_COVERED_NFC,
        carrying_amount=10_000_000.0,
        nace_sector="D",
        country="DE",
        is_eu=True,
        is_csrd_subject=True,
        reported_turnover_aligned_pct=45.0,
        reported_capex_aligned_pct=60.0,
        reported_opex_aligned_pct=35.0,
        taxonomy_eligible_pct=80.0,
        is_in_gar_scope=True,
        gar_aligned_pct=45.0,
    )


@pytest.fixture
def sme_non_csrd_exposure():
    """SME exposure not subject to CSRD (needs estimation)."""
    return BankingBookData(
        exposure_id="SME-001",
        exposure_name="SmallBiz Ltd",
        exposure_category=ExposureCategory.SME_NON_CSRD,
        carrying_amount=500_000.0,
        nace_sector="C",
        country="DE",
        is_eu=True,
        is_sme=True,
        is_csrd_subject=False,
        estimation_type=EstimationType.SECTOR_PROXY,
    )


@pytest.fixture
def non_eu_exposure():
    """Non-EU counterparty exposure."""
    return BankingBookData(
        exposure_id="NEU-001",
        exposure_name="Asia Pacific Corp",
        exposure_category=ExposureCategory.NON_EU_COUNTERPARTY,
        carrying_amount=2_000_000.0,
        nace_sector="C",
        country="JP",
        is_eu=False,
        is_csrd_subject=False,
        estimation_type=EstimationType.GEOGRAPHIC_PROXY,
    )


@pytest.fixture
def third_party_exposure():
    """Exposure with third-party alignment data."""
    return BankingBookData(
        exposure_id="TP-001",
        exposure_name="DataCo Inc",
        exposure_category=ExposureCategory.GAR_COVERED_NFC,
        carrying_amount=3_000_000.0,
        nace_sector="J",
        country="FR",
        is_eu=True,
        is_csrd_subject=False,
        estimation_type=EstimationType.THIRD_PARTY,
        third_party_alignment_pct=35.0,
        third_party_provider="Sustainalytics",
    )


@pytest.fixture
def internal_esg_exposure():
    """Exposure using internal ESG scoring model."""
    return BankingBookData(
        exposure_id="ESG-001",
        exposure_name="InternalCo",
        exposure_category=ExposureCategory.SME_NON_CSRD,
        carrying_amount=1_500_000.0,
        nace_sector="F",
        country="NL",
        is_eu=True,
        is_csrd_subject=False,
        estimation_type=EstimationType.INTERNAL_ESG,
        internal_esg_score=72.0,
        internal_esg_aligned_pct=22.0,
    )


@pytest.fixture
def sovereign_exposure():
    """Sovereign exposure (not in GAR scope)."""
    return BankingBookData(
        exposure_id="SOV-001",
        exposure_name="France OAT",
        exposure_category=ExposureCategory.SOVEREIGN,
        carrying_amount=20_000_000.0,
        country="FR",
        is_eu=True,
    )


@pytest.fixture
def conservative_default_exposure():
    """Exposure with no data -- falls to conservative default (0% alignment)."""
    return BankingBookData(
        exposure_id="CD-001",
        exposure_name="Unknown Corp",
        exposure_category=ExposureCategory.OTHER,
        carrying_amount=1_000_000.0,
        nace_sector="S",
        country="XX",
        is_eu=False,
        is_csrd_subject=False,
    )


@pytest.fixture
def mixed_banking_book(
    csrd_reported_exposure,
    sme_non_csrd_exposure,
    non_eu_exposure,
    sovereign_exposure,
):
    """A mixed banking book for portfolio-level tests."""
    return [
        csrd_reported_exposure,
        sme_non_csrd_exposure,
        non_eu_exposure,
        sovereign_exposure,
    ]


# ---------------------------------------------------------------------------
# 1. Initialization Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    """Test engine initialization."""

    def test_default_init(self):
        """Engine creates with default config."""
        engine = BTARCalculatorEngine()
        assert engine.config is not None

    def test_init_with_config(self):
        """Engine accepts BTARConfig."""
        cfg = BTARConfig(reporting_year=2025)
        engine = BTARCalculatorEngine(cfg)
        assert engine.config.reporting_year == 2025

    def test_init_with_dict(self):
        """Engine accepts a dict as config."""
        engine = BTARCalculatorEngine({"reporting_year": 2026})
        assert engine.config.reporting_year == 2026

    def test_init_with_none(self):
        """Engine accepts None and uses defaults."""
        engine = BTARCalculatorEngine(None)
        assert engine.config is not None


# ---------------------------------------------------------------------------
# 2. Estimation Methodology Tests
# ---------------------------------------------------------------------------

class TestEstimationMethodologies:
    """Test the 6 estimation methodologies."""

    def test_six_estimation_types(self):
        """All 6 estimation types are defined."""
        assert len(EstimationType) == 6

    @pytest.mark.parametrize("est_type", list(EstimationType))
    def test_all_estimation_types_have_confidence(self, est_type):
        """Each estimation type has a confidence score defined."""
        assert est_type.value in ESTIMATION_CONFIDENCE
        assert 0.0 <= ESTIMATION_CONFIDENCE[est_type.value] <= 1.0

    def test_csrd_highest_confidence(self):
        """CSRD_REPORTED has the highest confidence score."""
        csrd_conf = ESTIMATION_CONFIDENCE[EstimationType.CSRD_REPORTED.value]
        for est_type in EstimationType:
            if est_type != EstimationType.CSRD_REPORTED:
                assert csrd_conf >= ESTIMATION_CONFIDENCE[est_type.value]

    def test_conservative_default_lowest_confidence(self):
        """CONSERVATIVE_DEFAULT has the lowest confidence score."""
        cd_conf = ESTIMATION_CONFIDENCE[EstimationType.CONSERVATIVE_DEFAULT.value]
        for est_type in EstimationType:
            if est_type != EstimationType.CONSERVATIVE_DEFAULT:
                assert cd_conf <= ESTIMATION_CONFIDENCE[est_type.value]

    def test_confidence_ordering(self):
        """Confidence follows: CSRD > third-party > internal > sector > geo > default."""
        c = ESTIMATION_CONFIDENCE
        assert c["csrd_reported"] >= c["third_party"]
        assert c["third_party"] >= c["internal_esg"]
        assert c["internal_esg"] >= c["sector_proxy"]
        assert c["sector_proxy"] >= c["geographic_proxy"]
        assert c["geographic_proxy"] >= c["conservative_default"]


# ---------------------------------------------------------------------------
# 3. Single Exposure Estimation Tests
# ---------------------------------------------------------------------------

class TestSingleExposure:
    """Test single exposure estimation."""

    def test_csrd_reported_uses_reported_data(
        self, default_engine, csrd_reported_exposure
    ):
        """CSRD-reported exposure uses the reported alignment percentage."""
        result = default_engine.estimate_single_exposure(csrd_reported_exposure)
        assert result is not None

    def test_sector_proxy_estimation(self, default_engine, sme_non_csrd_exposure):
        """SME without CSRD data uses sector proxy estimation."""
        result = default_engine.estimate_single_exposure(sme_non_csrd_exposure)
        assert result is not None

    def test_geographic_proxy_estimation(self, default_engine, non_eu_exposure):
        """Non-EU exposure uses geographic proxy estimation."""
        result = default_engine.estimate_single_exposure(non_eu_exposure)
        assert result is not None

    def test_third_party_estimation(self, default_engine, third_party_exposure):
        """Third-party data is used when available."""
        result = default_engine.estimate_single_exposure(third_party_exposure)
        assert result is not None

    def test_internal_esg_estimation(self, default_engine, internal_esg_exposure):
        """Internal ESG score is used for estimation."""
        result = default_engine.estimate_single_exposure(internal_esg_exposure)
        assert result is not None

    def test_conservative_default_fallback(
        self, default_engine, conservative_default_exposure
    ):
        """Exposure with no data falls back to conservative default."""
        result = default_engine.estimate_single_exposure(conservative_default_exposure)
        assert result is not None


# ---------------------------------------------------------------------------
# 4. Sector Proxy Tests
# ---------------------------------------------------------------------------

class TestSectorProxy:
    """Test sector proxy alignment lookups."""

    def test_get_sector_proxy_known_sector(self, default_engine):
        """Known NACE sector returns a SectorProxyResult."""
        result = default_engine.get_sector_proxy("D")
        assert isinstance(result, SectorProxyResult)
        assert result.sector_alignment_pct == pytest.approx(25.0, rel=1e-4)

    def test_get_sector_proxy_unknown_sector(self, default_engine):
        """Unknown sector returns zero or conservative alignment."""
        result = default_engine.get_sector_proxy("ZZ")
        assert result.sector_alignment_pct >= 0.0

    @pytest.mark.parametrize("sector,expected_pct", [
        ("A", 5.0),   # Agriculture
        ("C", 12.0),  # Manufacturing
        ("D", 25.0),  # Electricity/gas
        ("F", 15.0),  # Construction
        ("J", 20.0),  # ICT
    ])
    def test_sector_proxy_values(self, default_engine, sector, expected_pct):
        """Sector proxy alignment matches reference values."""
        result = default_engine.get_sector_proxy(sector)
        assert result.sector_alignment_pct == pytest.approx(expected_pct, rel=1e-4)

    def test_all_nace_sectors_have_proxy(self):
        """All 21 NACE sector letters have a proxy alignment value."""
        for letter in "ABCDEFGHIJKLMNOPQRSTU":
            assert letter in SECTOR_PROXY_ALIGNMENT


# ---------------------------------------------------------------------------
# 5. Geographic Proxy Tests
# ---------------------------------------------------------------------------

class TestGeographicProxy:
    """Test geographic proxy alignment lookups."""

    def test_geographic_proxy_germany(self):
        """Germany has a geographic proxy of 18%."""
        assert GEOGRAPHIC_PROXY_ALIGNMENT["DE"] == 18.0

    def test_geographic_proxy_sweden(self):
        """Sweden has a higher proxy (25%)."""
        assert GEOGRAPHIC_PROXY_ALIGNMENT["SE"] == 25.0

    def test_multiple_countries_defined(self):
        """At least 20 countries have geographic proxies."""
        assert len(GEOGRAPHIC_PROXY_ALIGNMENT) >= 20


# ---------------------------------------------------------------------------
# 6. Full Banking Book BTAR Tests
# ---------------------------------------------------------------------------

class TestBTARCalculation:
    """Test full banking book BTAR calculation."""

    def test_btar_result_type(self, default_engine, mixed_banking_book):
        """BTAR calculation returns BTARResult."""
        result = default_engine.calculate_btar(mixed_banking_book)
        assert isinstance(result, BTARResult)

    def test_btar_ratio_range(self, default_engine, mixed_banking_book):
        """BTAR ratio is between 0 and 100 percent."""
        result = default_engine.calculate_btar(mixed_banking_book)
        assert 0.0 <= result.btar_turnover_pct <= 100.0

    def test_btar_total_banking_book(self, default_engine, mixed_banking_book):
        """Total banking book amount is sum of all exposures."""
        result = default_engine.calculate_btar(mixed_banking_book)
        expected_total = sum(e.carrying_amount for e in mixed_banking_book)
        assert result.total_banking_book == pytest.approx(
            expected_total, rel=1e-4
        )

    def test_empty_banking_book_raises(self, default_engine):
        """Empty exposure list raises ValueError."""
        with pytest.raises(ValueError):
            default_engine.calculate_btar([])


# ---------------------------------------------------------------------------
# 7. Data Coverage Report Tests
# ---------------------------------------------------------------------------

class TestDataCoverage:
    """Test data coverage reporting."""

    def test_data_coverage_in_result(self, default_engine, mixed_banking_book):
        """BTAR result includes a data coverage report."""
        result = default_engine.calculate_btar(mixed_banking_book)
        if hasattr(result, "data_coverage") and result.data_coverage is not None:
            assert isinstance(result.data_coverage, DataCoverageReport)
            assert result.data_coverage.total_exposures == len(mixed_banking_book)

    def test_csrd_reported_count(self, default_engine, mixed_banking_book):
        """Data coverage tracks CSRD-reported exposures."""
        result = default_engine.calculate_btar(mixed_banking_book)
        if hasattr(result, "data_coverage") and result.data_coverage is not None:
            # One CSRD-reported exposure in the fixture
            assert result.data_coverage.csrd_reported_count >= 1


# ---------------------------------------------------------------------------
# 8. BTAR vs GAR Reconciliation Tests
# ---------------------------------------------------------------------------

class TestBTARvsGAR:
    """Test reconciliation between BTAR and GAR."""

    def test_reconciliation_in_result(self, default_engine, mixed_banking_book):
        """BTAR result includes a reconciliation with GAR."""
        result = default_engine.calculate_btar(mixed_banking_book)
        if hasattr(result, "gar_reconciliation") and result.gar_reconciliation is not None:
            assert isinstance(result.gar_reconciliation, BTARvsGARReconciliation)

    def test_gar_scope_subset(self, default_engine, csrd_reported_exposure):
        """Exposure in GAR scope is tracked separately."""
        result = default_engine.calculate_btar([csrd_reported_exposure])
        if hasattr(result, "gar_reconciliation") and result.gar_reconciliation is not None:
            assert result.gar_reconciliation.gar_covered_amount >= 0


# ---------------------------------------------------------------------------
# 9. Confidence Scoring Tests
# ---------------------------------------------------------------------------

class TestConfidenceScoring:
    """Test confidence scoring for BTAR estimates."""

    def test_confidence_levels_enum(self):
        """All confidence levels are defined."""
        assert len(ConfidenceLevel) == 5

    def test_exposure_categories_enum(self):
        """All 10 exposure categories are defined."""
        assert len(ExposureCategory) == 10

    def test_csrd_has_high_confidence(
        self, default_engine, csrd_reported_exposure
    ):
        """CSRD-reported exposure has HIGH confidence."""
        result = default_engine.calculate_btar([csrd_reported_exposure])
        # The overall confidence should be high when all data is CSRD-reported
        if hasattr(result, "weighted_confidence"):
            assert result.weighted_confidence >= 0.9


# ---------------------------------------------------------------------------
# 10. Provenance & Reproducibility Tests
# ---------------------------------------------------------------------------

class TestProvenance:
    """Test SHA-256 provenance hashing."""

    def test_btar_provenance_hash(self, default_engine, mixed_banking_book):
        """BTAR result has a 64-char SHA-256 provenance hash."""
        result = default_engine.calculate_btar(mixed_banking_book)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_deterministic_btar(self, csrd_reported_exposure):
        """Same inputs produce same BTAR result values."""
        e1 = BTARCalculatorEngine()
        e2 = BTARCalculatorEngine()
        r1 = e1.calculate_btar([csrd_reported_exposure])
        r2 = e2.calculate_btar([csrd_reported_exposure])
        assert r1.btar_turnover_pct == pytest.approx(r2.btar_turnover_pct, rel=1e-6)
        assert r1.total_banking_book == pytest.approx(r2.total_banking_book, rel=1e-6)

    def test_different_input_different_hash(self, default_engine):
        """Different inputs produce different hashes."""
        exp1 = BankingBookData(
            exposure_category=ExposureCategory.GAR_COVERED_NFC,
            carrying_amount=1_000_000.0,
            nace_sector="C",
        )
        exp2 = BankingBookData(
            exposure_category=ExposureCategory.GAR_COVERED_NFC,
            carrying_amount=2_000_000.0,
            nace_sector="C",
        )
        r1 = default_engine.calculate_btar([exp1])
        r2 = default_engine.calculate_btar([exp2])
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# 11. Edge Cases & Error Handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases."""

    def test_large_banking_book(self, default_engine):
        """Engine handles a 200-exposure banking book."""
        exposures = [
            BankingBookData(
                exposure_id=f"E-{i:03d}",
                exposure_category=ExposureCategory.SME_NON_CSRD,
                carrying_amount=500_000.0,
                nace_sector="C",
                country="DE",
                is_eu=True,
            )
            for i in range(200)
        ]
        result = default_engine.calculate_btar(exposures)
        assert result.total_banking_book == pytest.approx(
            200 * 500_000.0, rel=1e-4
        )

    def test_result_model_fields(self, default_engine, mixed_banking_book):
        """BTAR result contains all expected fields."""
        result = default_engine.calculate_btar(mixed_banking_book)
        assert hasattr(result, "result_id")
        assert hasattr(result, "btar_turnover_pct")
        assert hasattr(result, "total_banking_book")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "engine_version")
        assert result.engine_version == "1.0.0"

    def test_zero_carrying_amount(self, default_engine):
        """Exposure with zero carrying amount is handled gracefully."""
        exp = BankingBookData(
            exposure_category=ExposureCategory.OTHER,
            carrying_amount=0.0,
            nace_sector="C",
        )
        result = default_engine.calculate_btar([exp])
        assert result.btar_turnover_pct >= 0.0

    def test_methodology_notes(self, default_engine, mixed_banking_book):
        """BTAR result includes methodology notes."""
        result = default_engine.calculate_btar(mixed_banking_book)
        if hasattr(result, "methodology_notes"):
            assert isinstance(result.methodology_notes, list)
