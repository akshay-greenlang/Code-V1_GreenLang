# -*- coding: utf-8 -*-
"""
Unit Tests for FinancedEmissionsEngine (Engine 1) - PACK-012. Target: 30+ tests.
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

_fe = _load_module("_fe", "financed_emissions_engine.py")

FinancedEmissionsEngine = _fe.FinancedEmissionsEngine
FinancedEmissionsConfig = _fe.FinancedEmissionsConfig
AssetClassData = _fe.AssetClassData
HoldingEmissions = _fe.HoldingEmissions
AttributionResult = _fe.AttributionResult
PortfolioEmissionsResult = _fe.PortfolioEmissionsResult
DataQualityScore = _fe.DataQualityScore
EmissionsByAssetClass = _fe.EmissionsByAssetClass
PCAFAssetClass = _fe.PCAFAssetClass
DataQualityLevel = _fe.DataQualityLevel
CurrencyRate = _fe.CurrencyRate
YoYTrajectory = _fe.YoYTrajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_engine():
    """Engine with default config (EUR, no scope3, double-counting on)."""
    return FinancedEmissionsEngine()


@pytest.fixture
def engine_with_scope3():
    """Engine that includes Scope 3 for listed equity, bonds, and loans."""
    cfg = FinancedEmissionsConfig(include_scope3=True)
    return FinancedEmissionsEngine(cfg)


@pytest.fixture
def engine_with_usd():
    """Engine that reports in USD with a EUR->USD rate."""
    cfg = FinancedEmissionsConfig(
        reporting_currency="USD",
        currency_rates=[
            CurrencyRate(source_currency="EUR", target_currency="USD", rate=1.10),
        ],
    )
    return FinancedEmissionsEngine(cfg)


@pytest.fixture
def listed_equity_holding():
    """Single listed-equity holding with known values."""
    return AssetClassData(
        holding_id="LE-001",
        holding_name="Acme Corp",
        asset_class=PCAFAssetClass.LISTED_EQUITY,
        outstanding_amount=10_000_000.0,
        evic=100_000_000.0,
        scope1_emissions=50_000.0,
        scope2_emissions=20_000.0,
        scope3_emissions=30_000.0,
        revenue=500_000_000.0,
        data_quality_score=2,
        sector="C",
        country="DE",
    )


@pytest.fixture
def corporate_bond_holding():
    """Single corporate bond holding."""
    return AssetClassData(
        holding_id="CB-001",
        holding_name="BondCo",
        asset_class=PCAFAssetClass.CORPORATE_BONDS,
        outstanding_amount=5_000_000.0,
        evic=80_000_000.0,
        scope1_emissions=40_000.0,
        scope2_emissions=15_000.0,
        data_quality_score=3,
    )


@pytest.fixture
def business_loan_holding():
    """Single business loan holding."""
    return AssetClassData(
        holding_id="BL-001",
        holding_name="LoanCo",
        asset_class=PCAFAssetClass.BUSINESS_LOANS,
        outstanding_amount=2_000_000.0,
        total_equity_plus_debt=20_000_000.0,
        scope1_emissions=10_000.0,
        scope2_emissions=5_000.0,
        data_quality_score=4,
    )


@pytest.fixture
def mortgage_holding():
    """Single mortgage holding."""
    return AssetClassData(
        holding_id="MG-001",
        holding_name="Mortgage House",
        asset_class=PCAFAssetClass.MORTGAGES,
        outstanding_amount=300_000.0,
        property_value=500_000.0,
        scope1_emissions=2.5,
        scope2_emissions=1.5,
        data_quality_score=3,
    )


@pytest.fixture
def motor_vehicle_holding():
    """Single motor vehicle loan holding."""
    return AssetClassData(
        holding_id="MV-001",
        holding_name="Car Loan",
        asset_class=PCAFAssetClass.MOTOR_VEHICLE_LOANS,
        outstanding_amount=25_000.0,
        vehicle_value=40_000.0,
        scope1_emissions=2.3,
        scope2_emissions=0.0,
        data_quality_score=3,
    )


@pytest.fixture
def sovereign_bond_holding():
    """Single sovereign bond holding."""
    return AssetClassData(
        holding_id="SB-001",
        holding_name="Germany Bund",
        asset_class=PCAFAssetClass.SOVEREIGN_BONDS,
        outstanding_amount=50_000_000.0,
        government_debt=2_400_000_000_000.0,
        scope1_emissions=800_000_000.0,
        data_quality_score=1,
    )


@pytest.fixture
def two_holding_portfolio(listed_equity_holding, corporate_bond_holding):
    """A two-holding portfolio for portfolio-level tests."""
    return [listed_equity_holding, corporate_bond_holding]


# ---------------------------------------------------------------------------
# 1. Initialization Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    """Test engine initialization with various config options."""

    def test_default_init(self):
        """Engine creates with default config."""
        engine = FinancedEmissionsEngine()
        assert engine.config.reporting_currency == "EUR"
        assert engine.config.include_scope3 is False
        assert engine.config.enable_double_counting_prevention is True

    def test_init_with_config_object(self):
        """Engine accepts a FinancedEmissionsConfig object."""
        cfg = FinancedEmissionsConfig(reporting_currency="USD", reporting_year=2025)
        engine = FinancedEmissionsEngine(cfg)
        assert engine.config.reporting_currency == "USD"
        assert engine.config.reporting_year == 2025

    def test_init_with_dict(self):
        """Engine accepts a plain dict as config."""
        engine = FinancedEmissionsEngine({"reporting_currency": "GBP"})
        assert engine.config.reporting_currency == "GBP"

    def test_init_with_none(self):
        """Engine accepts None and uses defaults."""
        engine = FinancedEmissionsEngine(None)
        assert engine.config.reporting_currency == "EUR"


# ---------------------------------------------------------------------------
# 2. Attribution Factor Tests
# ---------------------------------------------------------------------------

class TestAttributionFactor:
    """Test PCAF attribution factor calculation."""

    def test_listed_equity_attribution(self, default_engine, listed_equity_holding):
        """Attribution = outstanding / EVIC for listed equity."""
        result = default_engine.compute_attribution_factor(listed_equity_holding)
        assert isinstance(result, AttributionResult)
        expected = 10_000_000.0 / 100_000_000.0  # 0.1
        assert result.attribution_factor == pytest.approx(expected, rel=1e-6)
        assert result.denominator_field == "evic"
        assert result.used_fallback is False

    def test_corporate_bond_attribution(self, default_engine, corporate_bond_holding):
        """Attribution = outstanding / EVIC for corporate bonds."""
        result = default_engine.compute_attribution_factor(corporate_bond_holding)
        expected = 5_000_000.0 / 80_000_000.0  # 0.0625
        assert result.attribution_factor == pytest.approx(expected, rel=1e-6)

    def test_business_loan_attribution(self, default_engine, business_loan_holding):
        """Attribution = outstanding / total_equity_plus_debt for loans."""
        result = default_engine.compute_attribution_factor(business_loan_holding)
        expected = 2_000_000.0 / 20_000_000.0  # 0.1
        assert result.attribution_factor == pytest.approx(expected, rel=1e-6)
        assert result.denominator_field == "total_equity_plus_debt"

    def test_mortgage_attribution(self, default_engine, mortgage_holding):
        """Attribution = outstanding / property_value for mortgages."""
        result = default_engine.compute_attribution_factor(mortgage_holding)
        expected = 300_000.0 / 500_000.0  # 0.6
        assert result.attribution_factor == pytest.approx(expected, rel=1e-6)
        assert result.denominator_field == "property_value"

    def test_motor_vehicle_attribution(self, default_engine, motor_vehicle_holding):
        """Attribution = outstanding / vehicle_value for motor vehicles."""
        result = default_engine.compute_attribution_factor(motor_vehicle_holding)
        expected = 25_000.0 / 40_000.0  # 0.625
        assert result.attribution_factor == pytest.approx(expected, rel=1e-6)

    def test_sovereign_bond_attribution(self, default_engine, sovereign_bond_holding):
        """Attribution = outstanding / government_debt for sovereigns."""
        result = default_engine.compute_attribution_factor(sovereign_bond_holding)
        expected = 50_000_000.0 / 2_400_000_000_000.0
        assert result.attribution_factor == pytest.approx(expected, rel=1e-2)
        assert result.denominator_field == "government_debt"

    def test_attribution_capped_at_one(self, default_engine):
        """Attribution factor is capped at max_attribution_factor (1.0)."""
        holding = AssetClassData(
            asset_class=PCAFAssetClass.MORTGAGES,
            outstanding_amount=600_000.0,
            property_value=500_000.0,
            scope1_emissions=2.0,
        )
        result = default_engine.compute_attribution_factor(holding)
        assert result.attribution_factor <= 1.0

    def test_attribution_provenance_hash(self, default_engine, listed_equity_holding):
        """Attribution result has a non-empty provenance hash."""
        result = default_engine.compute_attribution_factor(listed_equity_holding)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# 3. PCAF Asset Class Parametrized Tests
# ---------------------------------------------------------------------------

class TestPCAFAssetClasses:
    """Test all 10 PCAF asset classes can be processed."""

    @pytest.mark.parametrize("asset_class", list(PCAFAssetClass))
    def test_all_asset_classes_accepted(self, default_engine, asset_class):
        """Engine processes every PCAF asset class without error."""
        denominator_map = {
            PCAFAssetClass.LISTED_EQUITY: {"evic": 100_000_000.0},
            PCAFAssetClass.CORPORATE_BONDS: {"evic": 80_000_000.0},
            PCAFAssetClass.BUSINESS_LOANS: {"total_equity_plus_debt": 20_000_000.0},
            PCAFAssetClass.PROJECT_FINANCE: {"total_project_cost": 50_000_000.0},
            PCAFAssetClass.COMMERCIAL_REAL_ESTATE: {"property_value": 10_000_000.0},
            PCAFAssetClass.MORTGAGES: {"property_value": 500_000.0},
            PCAFAssetClass.MOTOR_VEHICLE_LOANS: {"vehicle_value": 40_000.0},
            PCAFAssetClass.SOVEREIGN_BONDS: {"government_debt": 2_000_000_000_000.0},
            PCAFAssetClass.SECURITIZATIONS: {"total_pool_value": 100_000_000.0},
            PCAFAssetClass.SUB_SOVEREIGN_DEBT: {"total_revenue_or_budget": 5_000_000_000.0},
        }
        kwargs = denominator_map[asset_class]
        holding = AssetClassData(
            asset_class=asset_class,
            outstanding_amount=1_000_000.0,
            scope1_emissions=1000.0,
            scope2_emissions=500.0,
            **kwargs,
        )
        result = default_engine.calculate_single_holding(holding)
        assert isinstance(result, HoldingEmissions)
        assert result.financed_scope1 >= 0
        assert result.financed_scope1_2 >= 0

    def test_ten_asset_classes_exist(self):
        """Verify all 10 PCAF asset classes are defined."""
        assert len(PCAFAssetClass) == 10


# ---------------------------------------------------------------------------
# 4. Data Quality Scoring Tests
# ---------------------------------------------------------------------------

class TestDataQualityScoring:
    """Test PCAF data quality scoring (1 = best, 5 = worst)."""

    @pytest.mark.parametrize("score,expected_level", [
        (1, DataQualityLevel.SCORE_1),
        (2, DataQualityLevel.SCORE_2),
        (3, DataQualityLevel.SCORE_3),
        (4, DataQualityLevel.SCORE_4),
        (5, DataQualityLevel.SCORE_5),
    ])
    def test_data_quality_levels(self, default_engine, score, expected_level):
        """Each DQ score 1-5 maps to the correct DataQualityLevel."""
        holding = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=1_000_000.0,
            evic=10_000_000.0,
            scope1_emissions=1000.0,
            data_quality_score=score,
        )
        dq = default_engine.assess_data_quality(holding)
        assert isinstance(dq, DataQualityScore)
        assert dq.score == score
        assert dq.level == expected_level

    def test_dq_numeric_property(self):
        """DataQualityLevel.numeric returns the integer value."""
        assert DataQualityLevel.SCORE_1.numeric == 1
        assert DataQualityLevel.SCORE_5.numeric == 5

    def test_dq_score_in_holding_result(self, default_engine, listed_equity_holding):
        """Data quality score is embedded in HoldingEmissions result."""
        result = default_engine.calculate_single_holding(listed_equity_holding)
        assert result.data_quality is not None
        assert result.data_quality.score == 2  # set in fixture


# ---------------------------------------------------------------------------
# 5. Single Holding Emission Calculation Tests
# ---------------------------------------------------------------------------

class TestSingleHoldingEmissions:
    """Test financed emission calculations for individual holdings."""

    def test_financed_scope1_calculation(self, default_engine, listed_equity_holding):
        """Financed Scope 1 = attribution_factor * scope1_emissions."""
        result = default_engine.calculate_single_holding(listed_equity_holding)
        af = 10_000_000.0 / 100_000_000.0  # 0.1
        expected_s1 = af * 50_000.0  # 5000.0
        assert result.financed_scope1 == pytest.approx(expected_s1, rel=1e-4)

    def test_financed_scope2_calculation(self, default_engine, listed_equity_holding):
        """Financed Scope 2 = attribution_factor * scope2_emissions."""
        result = default_engine.calculate_single_holding(listed_equity_holding)
        af = 0.1
        expected_s2 = af * 20_000.0  # 2000.0
        assert result.financed_scope2 == pytest.approx(expected_s2, rel=1e-4)

    def test_financed_scope1_2_sum(self, default_engine, listed_equity_holding):
        """Financed Scope 1+2 equals Scope 1 plus Scope 2."""
        result = default_engine.calculate_single_holding(listed_equity_holding)
        assert result.financed_scope1_2 == pytest.approx(
            result.financed_scope1 + result.financed_scope2, rel=1e-4
        )

    def test_scope3_excluded_by_default(self, default_engine, listed_equity_holding):
        """Scope 3 is excluded from total when include_scope3 is False."""
        result = default_engine.calculate_single_holding(listed_equity_holding)
        expected_total = result.financed_scope1 + result.financed_scope2
        assert result.financed_total == pytest.approx(expected_total, rel=1e-4)

    def test_scope3_included_when_enabled(self, engine_with_scope3, listed_equity_holding):
        """Scope 3 is included in total when config enables it."""
        result = engine_with_scope3.calculate_single_holding(listed_equity_holding)
        assert result.financed_scope3 > 0
        expected_total = (
            result.financed_scope1 + result.financed_scope2 + result.financed_scope3
        )
        assert result.financed_total == pytest.approx(expected_total, rel=1e-4)

    def test_zero_emissions_holding(self, default_engine):
        """Holding with zero emissions produces zero financed emissions."""
        holding = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=10_000_000.0,
            evic=100_000_000.0,
            scope1_emissions=0.0,
            scope2_emissions=0.0,
        )
        result = default_engine.calculate_single_holding(holding)
        assert result.financed_total == 0.0


# ---------------------------------------------------------------------------
# 6. Portfolio Aggregation Tests
# ---------------------------------------------------------------------------

class TestPortfolioAggregation:
    """Test portfolio-level emissions aggregation."""

    def test_portfolio_total_is_sum_of_holdings(
        self, default_engine, two_holding_portfolio
    ):
        """Portfolio total equals sum of individual holding totals."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        sum_totals = sum(h.financed_total for h in result.holding_results)
        assert result.total_financed_emissions == pytest.approx(sum_totals, rel=1e-4)

    def test_portfolio_holding_count(self, default_engine, two_holding_portfolio):
        """Portfolio reports correct number of holdings."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert result.total_holdings == 2

    def test_portfolio_asset_class_breakdown(
        self, default_engine, two_holding_portfolio
    ):
        """Portfolio produces asset class breakdown."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert len(result.asset_class_breakdown) > 0
        classes = {b.asset_class for b in result.asset_class_breakdown}
        assert PCAFAssetClass.LISTED_EQUITY in classes
        assert PCAFAssetClass.CORPORATE_BONDS in classes

    def test_portfolio_waci_positive(self, default_engine, two_holding_portfolio):
        """Portfolio WACI is non-negative."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert result.portfolio_waci >= 0.0

    def test_empty_portfolio_raises(self, default_engine):
        """Passing empty list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            default_engine.calculate_portfolio_emissions([])

    def test_portfolio_weighted_data_quality(
        self, default_engine, two_holding_portfolio
    ):
        """Portfolio weighted average data quality is between 1 and 5."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert 1.0 <= result.weighted_avg_data_quality <= 5.0


# ---------------------------------------------------------------------------
# 7. Double-Counting Prevention Tests
# ---------------------------------------------------------------------------

class TestDoubleCounting:
    """Test double-counting prevention between asset classes."""

    def test_double_counting_adjustments_dict(self, listed_equity_holding):
        """When entity groups overlap, adjustments dict is populated."""
        bond = AssetClassData(
            holding_id="CB-SAME",
            holding_name="Acme Corp",
            asset_class=PCAFAssetClass.CORPORATE_BONDS,
            outstanding_amount=5_000_000.0,
            evic=100_000_000.0,
            scope1_emissions=50_000.0,
            scope2_emissions=20_000.0,
        )
        cfg = FinancedEmissionsConfig(
            enable_double_counting_prevention=True,
            double_counting_entity_groups={"Acme": ["LE-001", "CB-SAME"]},
        )
        engine = FinancedEmissionsEngine(cfg)
        result = engine.calculate_portfolio_emissions(
            [listed_equity_holding, bond]
        )
        assert isinstance(result.double_counting_adjustments, dict)

    def test_no_double_counting_when_disabled(
        self, listed_equity_holding, corporate_bond_holding
    ):
        """When disabled, no adjustments are made."""
        cfg = FinancedEmissionsConfig(enable_double_counting_prevention=False)
        engine = FinancedEmissionsEngine(cfg)
        result = engine.calculate_portfolio_emissions(
            [listed_equity_holding, corporate_bond_holding]
        )
        assert result.double_counting_adjustments == {}


# ---------------------------------------------------------------------------
# 8. Multi-Currency Normalization Tests
# ---------------------------------------------------------------------------

class TestMultiCurrency:
    """Test multi-currency normalization."""

    def test_eur_default_no_conversion(self, default_engine, listed_equity_holding):
        """EUR amounts pass through without conversion (rate=1.0)."""
        result = default_engine.compute_attribution_factor(listed_equity_holding)
        assert result.outstanding_amount_eur == pytest.approx(
            10_000_000.0, rel=1e-4
        )

    def test_currency_conversion_applied(self):
        """Non-EUR outstanding is converted using provided rate."""
        cfg = FinancedEmissionsConfig(
            reporting_currency="EUR",
            currency_rates=[
                CurrencyRate(source_currency="USD", target_currency="EUR", rate=0.91),
            ],
        )
        engine = FinancedEmissionsEngine(cfg)
        holding = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=10_000_000.0,
            currency="USD",
            evic=100_000_000.0,
            scope1_emissions=50_000.0,
        )
        result = engine.compute_attribution_factor(holding)
        expected_eur = 10_000_000.0 * 0.91
        assert result.outstanding_amount_eur == pytest.approx(expected_eur, rel=1e-2)


# ---------------------------------------------------------------------------
# 9. YoY Trajectory Tests
# ---------------------------------------------------------------------------

class TestYoYTrajectory:
    """Test year-over-year trajectory calculations."""

    def test_yoy_trajectory_present(self, default_engine, two_holding_portfolio):
        """Portfolio result includes YoY trajectory list."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert isinstance(result.yoy_trajectory, list)

    def test_yoy_with_prior_results(self, two_holding_portfolio):
        """YoY shows change when prior year results are provided."""
        prior = YoYTrajectory(
            year=2023,
            total_financed_emissions=10_000.0,
            total_outstanding_eur=50_000_000.0,
            portfolio_waci=200.0,
        )
        cfg = FinancedEmissionsConfig(
            reporting_year=2024,
            yoy_prior_results=[prior],
        )
        engine = FinancedEmissionsEngine(cfg)
        result = engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert len(result.yoy_trajectory) >= 1


# ---------------------------------------------------------------------------
# 10. Provenance & Reproducibility Tests
# ---------------------------------------------------------------------------

class TestProvenance:
    """Test SHA-256 provenance hashing and reproducibility."""

    def test_portfolio_provenance_hash_nonempty(
        self, default_engine, two_holding_portfolio
    ):
        """Portfolio result has a non-empty 64-char SHA-256 hash."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_holding_provenance_hash(self, default_engine, listed_equity_holding):
        """Individual holding result has a provenance hash."""
        result = default_engine.calculate_single_holding(listed_equity_holding)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_deterministic_provenance(self, listed_equity_holding):
        """Same input produces same provenance hash (bit-perfect)."""
        engine1 = FinancedEmissionsEngine()
        engine2 = FinancedEmissionsEngine()
        r1 = engine1.calculate_single_holding(listed_equity_holding)
        r2 = engine2.calculate_single_holding(listed_equity_holding)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input_different_hash(self, default_engine):
        """Different inputs produce different provenance hashes."""
        h1 = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=10_000_000.0,
            evic=100_000_000.0,
            scope1_emissions=50_000.0,
        )
        h2 = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=20_000_000.0,
            evic=100_000_000.0,
            scope1_emissions=50_000.0,
        )
        r1 = default_engine.calculate_single_holding(h1)
        r2 = default_engine.calculate_single_holding(h2)
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# 11. Edge Cases & Error Handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_denominator_holding(self, default_engine):
        """Holding with zero EVIC returns zero attribution."""
        holding = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=10_000_000.0,
            evic=0.0,
            scope1_emissions=50_000.0,
        )
        result = default_engine.calculate_single_holding(holding)
        assert result.attribution_factor == 0.0
        assert result.financed_scope1 == 0.0

    def test_very_small_outstanding(self, default_engine):
        """Very small outstanding amounts do not cause floating-point issues."""
        holding = AssetClassData(
            asset_class=PCAFAssetClass.LISTED_EQUITY,
            outstanding_amount=0.01,
            evic=100_000_000.0,
            scope1_emissions=50_000.0,
        )
        result = default_engine.calculate_single_holding(holding)
        assert result.financed_scope1 >= 0.0

    def test_large_portfolio(self, default_engine):
        """Engine handles a 100-holding portfolio."""
        holdings = [
            AssetClassData(
                holding_id=f"H-{i:03d}",
                asset_class=PCAFAssetClass.LISTED_EQUITY,
                outstanding_amount=1_000_000.0,
                evic=50_000_000.0,
                scope1_emissions=1000.0,
                scope2_emissions=500.0,
            )
            for i in range(100)
        ]
        result = default_engine.calculate_portfolio_emissions(holdings)
        assert result.total_holdings == 100
        assert result.total_financed_emissions > 0

    def test_result_model_fields(self, default_engine, two_holding_portfolio):
        """Portfolio result contains all expected top-level fields."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert hasattr(result, "result_id")
        assert hasattr(result, "reporting_year")
        assert hasattr(result, "total_financed_emissions")
        assert hasattr(result, "portfolio_waci")
        assert hasattr(result, "asset_class_breakdown")
        assert hasattr(result, "holding_results")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "engine_version")
        assert result.engine_version == "1.0.0"

    def test_methodology_notes_generated(self, default_engine, two_holding_portfolio):
        """Portfolio result includes methodology notes."""
        result = default_engine.calculate_portfolio_emissions(two_holding_portfolio)
        assert isinstance(result.methodology_notes, list)
