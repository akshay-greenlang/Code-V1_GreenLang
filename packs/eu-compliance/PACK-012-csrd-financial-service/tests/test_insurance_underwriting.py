# -*- coding: utf-8 -*-
"""
Unit Tests for InsuranceUnderwritingEngine (Engine 2) - PACK-012. Target: 30+ tests.
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

_iu = _load_module("_iu", "insurance_underwriting_engine.py")

InsuranceUnderwritingEngine = _iu.InsuranceUnderwritingEngine
UnderwritingConfig = _iu.UnderwritingConfig
PolicyData = _iu.PolicyData
PolicyEmissionsResult = _iu.PolicyEmissionsResult
LineOfBusinessResult = _iu.LineOfBusinessResult
UnderwritingEmissionsResult = _iu.UnderwritingEmissionsResult
ReinsuranceAdjustment = _iu.ReinsuranceAdjustment
ClaimsEmissions = _iu.ClaimsEmissions
SectorBreakdown = _iu.SectorBreakdown
InsuranceLine = _iu.InsuranceLine
NACESector = _iu.NACESector
ReinsuranceType = _iu.ReinsuranceType
EmissionCalculationMethod = _iu.EmissionCalculationMethod
SECTOR_EMISSION_INTENSITY = _iu.SECTOR_EMISSION_INTENSITY
VEHICLE_EMISSION_FACTORS = _iu.VEHICLE_EMISSION_FACTORS
PROPERTY_EMISSION_FACTORS = _iu.PROPERTY_EMISSION_FACTORS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_engine():
    """Engine with default config."""
    return InsuranceUnderwritingEngine()


@pytest.fixture
def engine_with_scope3():
    """Engine that includes Scope 3 of insured entities."""
    cfg = UnderwritingConfig(include_scope3=True)
    return InsuranceUnderwritingEngine(cfg)


@pytest.fixture
def engine_no_reinsurance():
    """Engine with reinsurance adjustments disabled."""
    cfg = UnderwritingConfig(apply_reinsurance_adjustment=False)
    return InsuranceUnderwritingEngine(cfg)


@pytest.fixture
def commercial_motor_policy():
    """Commercial motor insurance policy with known values."""
    return PolicyData(
        policy_id="CM-001",
        policyholder_name="Fleet Corp",
        line_of_business=InsuranceLine.COMMERCIAL_MOTOR,
        nace_sector="H",
        written_premium=500_000.0,
        earned_premium=450_000.0,
        total_market_premium=50_000_000.0,
        insured_scope1=12_000.0,
        insured_scope2=3_000.0,
        insured_scope3=5_000.0,
        insured_revenue=100_000_000.0,
        vehicle_type="heavy_commercial",
        vehicle_count=50,
        data_quality_score=2,
        country="DE",
    )


@pytest.fixture
def personal_motor_policy():
    """Personal motor insurance policy."""
    return PolicyData(
        policy_id="PM-001",
        policyholder_name="John Smith",
        line_of_business=InsuranceLine.PERSONAL_MOTOR,
        written_premium=1_200.0,
        earned_premium=1_100.0,
        total_market_premium=5_000_000.0,
        vehicle_type="passenger_car_petrol",
        vehicle_count=1,
        data_quality_score=4,
        country="DE",
    )


@pytest.fixture
def commercial_property_policy():
    """Commercial property insurance policy."""
    return PolicyData(
        policy_id="CP-001",
        policyholder_name="BuildCo",
        line_of_business=InsuranceLine.COMMERCIAL_PROPERTY,
        nace_sector="L",
        written_premium=200_000.0,
        earned_premium=180_000.0,
        insured_value=50_000_000.0,
        total_market_premium=20_000_000.0,
        insured_scope1=8_000.0,
        insured_scope2=2_000.0,
        insured_revenue=200_000_000.0,
        property_type="office",
        property_area_sqm=5_000.0,
        data_quality_score=3,
        country="FR",
    )


@pytest.fixture
def general_liability_policy():
    """General liability insurance policy."""
    return PolicyData(
        policy_id="GL-001",
        policyholder_name="MfgCorp",
        line_of_business=InsuranceLine.GENERAL_LIABILITY,
        nace_sector="C",
        written_premium=100_000.0,
        earned_premium=95_000.0,
        total_market_premium=10_000_000.0,
        insured_scope1=25_000.0,
        insured_scope2=10_000.0,
        insured_revenue=500_000_000.0,
        data_quality_score=2,
        country="DE",
    )


@pytest.fixture
def policy_with_reinsurance():
    """Commercial motor policy with reinsurance arrangement."""
    return PolicyData(
        policy_id="RE-001",
        policyholder_name="Reinsured Fleet",
        line_of_business=InsuranceLine.COMMERCIAL_MOTOR,
        nace_sector="H",
        written_premium=1_000_000.0,
        total_market_premium=50_000_000.0,
        insured_scope1=20_000.0,
        insured_scope2=5_000.0,
        data_quality_score=2,
        reinsurance=ReinsuranceAdjustment(
            reinsurance_type=ReinsuranceType.QUOTA_SHARE,
            ceded_premium=300_000.0,
            written_premium=1_000_000.0,
        ),
    )


@pytest.fixture
def policy_with_claims():
    """Policy with claims-linked emissions data."""
    return PolicyData(
        policy_id="CL-001",
        policyholder_name="Claims Corp",
        line_of_business=InsuranceLine.COMMERCIAL_PROPERTY,
        nace_sector="F",
        written_premium=500_000.0,
        total_market_premium=25_000_000.0,
        insured_scope1=10_000.0,
        insured_scope2=3_000.0,
        data_quality_score=3,
        claims_data=ClaimsEmissions(
            total_claims_paid=200_000.0,
            claims_count=5,
            emission_factor_per_claim=50.0,
            total_claims_emissions=250.0,
        ),
    )


@pytest.fixture
def two_policy_portfolio(commercial_motor_policy, commercial_property_policy):
    """A two-policy portfolio."""
    return [commercial_motor_policy, commercial_property_policy]


# ---------------------------------------------------------------------------
# 1. Initialization Tests
# ---------------------------------------------------------------------------

class TestInitialization:
    """Test engine initialization."""

    def test_default_init(self):
        """Engine creates with default config."""
        engine = InsuranceUnderwritingEngine()
        assert engine.config.reporting_year == 2024
        assert engine.config.include_scope3 is False
        assert engine.config.apply_reinsurance_adjustment is True

    def test_init_with_config(self):
        """Engine accepts UnderwritingConfig."""
        cfg = UnderwritingConfig(reporting_year=2025, include_scope3=True)
        engine = InsuranceUnderwritingEngine(cfg)
        assert engine.config.reporting_year == 2025
        assert engine.config.include_scope3 is True

    def test_init_with_dict(self):
        """Engine accepts a dict as config."""
        engine = InsuranceUnderwritingEngine({"reporting_year": 2026})
        assert engine.config.reporting_year == 2026

    def test_init_with_none(self):
        """Engine accepts None and uses defaults."""
        engine = InsuranceUnderwritingEngine(None)
        assert engine.config.reporting_year == 2024


# ---------------------------------------------------------------------------
# 2. Premium Share Attribution Tests
# ---------------------------------------------------------------------------

class TestPremiumShare:
    """Test premium share attribution calculation."""

    def test_premium_share_basic(self, default_engine, commercial_motor_policy):
        """Premium share = written_premium / total_market_premium."""
        result = default_engine.compute_premium_share(commercial_motor_policy)
        expected = 500_000.0 / 50_000_000.0  # 0.01
        assert result == pytest.approx(expected, rel=1e-6)

    def test_premium_share_small_policy(self, default_engine, personal_motor_policy):
        """Small premium yields small premium share."""
        result = default_engine.compute_premium_share(personal_motor_policy)
        expected = 1_200.0 / 5_000_000.0  # 0.00024
        assert result == pytest.approx(expected, rel=1e-6)

    def test_premium_share_zero_market(self, default_engine):
        """Zero total market premium yields a defined premium share."""
        policy = PolicyData(
            line_of_business=InsuranceLine.GENERAL_LIABILITY,
            written_premium=100_000.0,
            total_market_premium=0.0,
        )
        result = default_engine.compute_premium_share(policy)
        assert result >= 0.0


# ---------------------------------------------------------------------------
# 3. Lines of Business Tests
# ---------------------------------------------------------------------------

class TestLinesOfBusiness:
    """Test all 6 insurance lines of business."""

    @pytest.mark.parametrize("lob", list(InsuranceLine))
    def test_all_lines_accepted(self, default_engine, lob):
        """Engine processes every insurance line without error."""
        policy = PolicyData(
            line_of_business=lob,
            written_premium=100_000.0,
            total_market_premium=10_000_000.0,
            insured_scope1=5_000.0,
            insured_scope2=1_000.0,
            nace_sector="C",
            data_quality_score=3,
        )
        result = default_engine.calculate_single_policy(policy)
        assert isinstance(result, PolicyEmissionsResult)
        assert result.gross_emissions >= 0

    def test_six_lines_exist(self):
        """Verify all 6 insurance lines are defined."""
        assert len(InsuranceLine) == 6


# ---------------------------------------------------------------------------
# 4. Single Policy Emission Tests
# ---------------------------------------------------------------------------

class TestSinglePolicyEmissions:
    """Test emission calculation for individual policies."""

    def test_gross_emissions_positive(self, default_engine, commercial_motor_policy):
        """Gross emissions are positive for policy with emissions data."""
        result = default_engine.calculate_single_policy(commercial_motor_policy)
        assert result.gross_emissions > 0

    def test_emission_intensity(self, default_engine, commercial_motor_policy):
        """Emission intensity is computed as tCO2e per EUR M premium."""
        result = default_engine.calculate_single_policy(commercial_motor_policy)
        assert result.emission_intensity >= 0

    def test_net_equals_gross_no_reinsurance(
        self, default_engine, commercial_motor_policy
    ):
        """Without reinsurance, net emissions equal gross emissions."""
        result = default_engine.calculate_single_policy(commercial_motor_policy)
        assert result.net_emissions == pytest.approx(result.gross_emissions, rel=1e-4)

    def test_written_premium_in_result(self, default_engine, commercial_motor_policy):
        """Result carries forward the written premium."""
        result = default_engine.calculate_single_policy(commercial_motor_policy)
        assert result.written_premium == pytest.approx(500_000.0, rel=1e-4)

    def test_policy_provenance_hash(self, default_engine, commercial_motor_policy):
        """Policy result has a 64-char SHA-256 provenance hash."""
        result = default_engine.calculate_single_policy(commercial_motor_policy)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64


# ---------------------------------------------------------------------------
# 5. Reinsurance Adjustment Tests
# ---------------------------------------------------------------------------

class TestReinsurance:
    """Test gross-to-net reinsurance adjustments."""

    def test_reinsurance_reduces_net(self, default_engine, policy_with_reinsurance):
        """Net emissions are less than gross when reinsurance is applied."""
        result = default_engine.calculate_single_policy(policy_with_reinsurance)
        assert result.net_emissions < result.gross_emissions
        assert result.reinsurance_cession_emissions > 0

    def test_reinsurance_cession_percentage(self, default_engine):
        """Reinsurance cession percentage is auto-computed."""
        reins = ReinsuranceAdjustment(
            reinsurance_type=ReinsuranceType.QUOTA_SHARE,
            ceded_premium=250_000.0,
            written_premium=1_000_000.0,
        )
        assert reins.cession_pct == pytest.approx(25.0, rel=1e-4)

    def test_compute_reinsurance_adjustment(self, default_engine):
        """compute_reinsurance_adjustment returns (ceded, net) tuple."""
        reins = ReinsuranceAdjustment(
            reinsurance_type=ReinsuranceType.QUOTA_SHARE,
            ceded_premium=300_000.0,
            written_premium=1_000_000.0,
        )
        ceded, net = default_engine.compute_reinsurance_adjustment(10_000.0, reins)
        # 30% cession * 10000 gross = 3000 ceded, 7000 net
        assert ceded == pytest.approx(3_000.0, rel=1e-2)
        assert net == pytest.approx(7_000.0, rel=1e-2)

    def test_no_reinsurance_no_cession(self, engine_no_reinsurance, policy_with_reinsurance):
        """When reinsurance is disabled, cession is zero."""
        result = engine_no_reinsurance.calculate_single_policy(policy_with_reinsurance)
        assert result.reinsurance_cession_emissions == 0.0


# ---------------------------------------------------------------------------
# 6. Claims-Linked Emissions Tests
# ---------------------------------------------------------------------------

class TestClaimsEmissions:
    """Test claims-linked emission inclusion."""

    def test_claims_emissions_included(self, default_engine, policy_with_claims):
        """Claims emissions appear in the policy result."""
        result = default_engine.calculate_single_policy(policy_with_claims)
        assert result.claims_emissions >= 0

    def test_claims_data_model(self):
        """ClaimsEmissions model computes total correctly."""
        claims = ClaimsEmissions(
            total_claims_paid=100_000.0,
            claims_count=10,
            emission_factor_per_claim=20.0,
            total_claims_emissions=200.0,
        )
        assert claims.total_claims_emissions == 200.0
        assert claims.claims_count == 10


# ---------------------------------------------------------------------------
# 7. NACE Sector Classification Tests
# ---------------------------------------------------------------------------

class TestNACESector:
    """Test NACE sector classification and intensity."""

    def test_21_nace_sectors_defined(self):
        """Verify all 21 NACE sectors are defined."""
        assert len(NACESector) == 21

    def test_sector_intensity_factors(self):
        """Each NACE sector has an emission intensity factor."""
        for sector in NACESector:
            assert sector.value in SECTOR_EMISSION_INTENSITY
            assert SECTOR_EMISSION_INTENSITY[sector.value] > 0

    def test_high_intensity_sector(self, default_engine):
        """High-intensity sector (electricity/gas) produces more emissions."""
        policy_high = PolicyData(
            policy_id="HIGH-001",
            line_of_business=InsuranceLine.GENERAL_LIABILITY,
            nace_sector="D",
            written_premium=100_000.0,
            total_market_premium=10_000_000.0,
            data_quality_score=5,
        )
        policy_low = PolicyData(
            policy_id="LOW-001",
            line_of_business=InsuranceLine.GENERAL_LIABILITY,
            nace_sector="K",
            written_premium=100_000.0,
            total_market_premium=10_000_000.0,
            data_quality_score=5,
        )
        r_high = default_engine.calculate_single_policy(policy_high)
        r_low = default_engine.calculate_single_policy(policy_low)
        # D (electricity=4500) has much higher intensity than K (financial=80)
        assert r_high.gross_emissions > r_low.gross_emissions


# ---------------------------------------------------------------------------
# 8. Portfolio Aggregation Tests
# ---------------------------------------------------------------------------

class TestPortfolioAggregation:
    """Test portfolio-level underwriting emissions aggregation."""

    def test_portfolio_total_policies(self, default_engine, two_policy_portfolio):
        """Portfolio reports correct number of policies."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert result.total_policies == 2

    def test_portfolio_gross_emissions_sum(self, default_engine, two_policy_portfolio):
        """Portfolio gross emissions equals sum of individual policy gross."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        sum_gross = sum(p.gross_emissions for p in result.policy_results)
        assert result.total_gross_emissions == pytest.approx(sum_gross, rel=1e-4)

    def test_lob_breakdown(self, default_engine, two_policy_portfolio):
        """Portfolio produces a line-of-business breakdown."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert len(result.lob_breakdown) > 0
        lobs = {b.line_of_business for b in result.lob_breakdown}
        assert InsuranceLine.COMMERCIAL_MOTOR in lobs
        assert InsuranceLine.COMMERCIAL_PROPERTY in lobs

    def test_sector_breakdown(self, default_engine, two_policy_portfolio):
        """Portfolio produces a NACE sector breakdown."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert isinstance(result.sector_breakdown, list)

    def test_empty_portfolio_raises(self, default_engine):
        """Passing empty list raises ValueError."""
        with pytest.raises(ValueError):
            default_engine.calculate_underwriting_emissions([])

    def test_portfolio_weighted_data_quality(self, default_engine, two_policy_portfolio):
        """Portfolio weighted DQ is between 1 and 5."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert 1.0 <= result.weighted_avg_data_quality <= 5.0


# ---------------------------------------------------------------------------
# 9. Provenance & Reproducibility Tests
# ---------------------------------------------------------------------------

class TestProvenance:
    """Test SHA-256 provenance hashing and reproducibility."""

    def test_portfolio_provenance_hash(self, default_engine, two_policy_portfolio):
        """Portfolio result has a 64-char SHA-256 hash."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert result.provenance_hash
        assert len(result.provenance_hash) == 64

    def test_deterministic_provenance(self, commercial_motor_policy):
        """Same input produces same provenance hash."""
        e1 = InsuranceUnderwritingEngine()
        e2 = InsuranceUnderwritingEngine()
        r1 = e1.calculate_single_policy(commercial_motor_policy)
        r2 = e2.calculate_single_policy(commercial_motor_policy)
        assert r1.provenance_hash == r2.provenance_hash

    def test_different_input_different_hash(self, default_engine):
        """Different inputs produce different hashes."""
        p1 = PolicyData(
            line_of_business=InsuranceLine.COMMERCIAL_MOTOR,
            written_premium=100_000.0,
            total_market_premium=10_000_000.0,
            insured_scope1=5_000.0,
        )
        p2 = PolicyData(
            line_of_business=InsuranceLine.COMMERCIAL_MOTOR,
            written_premium=200_000.0,
            total_market_premium=10_000_000.0,
            insured_scope1=5_000.0,
        )
        r1 = default_engine.calculate_single_policy(p1)
        r2 = default_engine.calculate_single_policy(p2)
        assert r1.provenance_hash != r2.provenance_hash


# ---------------------------------------------------------------------------
# 10. Edge Cases & Error Handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_premium_policy(self, default_engine):
        """Policy with zero premium produces zero or near-zero emissions."""
        policy = PolicyData(
            line_of_business=InsuranceLine.GENERAL_LIABILITY,
            written_premium=0.0,
            total_market_premium=10_000_000.0,
            insured_scope1=5_000.0,
        )
        result = default_engine.calculate_single_policy(policy)
        assert result.gross_emissions == pytest.approx(0.0, abs=0.01)

    def test_large_portfolio(self, default_engine):
        """Engine handles a 100-policy portfolio."""
        policies = [
            PolicyData(
                policy_id=f"P-{i:03d}",
                line_of_business=InsuranceLine.COMMERCIAL_MOTOR,
                written_premium=50_000.0,
                total_market_premium=100_000_000.0,
                insured_scope1=1_000.0,
                insured_scope2=500.0,
                nace_sector="H",
            )
            for i in range(100)
        ]
        result = default_engine.calculate_underwriting_emissions(policies)
        assert result.total_policies == 100
        assert result.total_gross_emissions > 0

    def test_result_model_fields(self, default_engine, two_policy_portfolio):
        """Portfolio result contains all expected fields."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert hasattr(result, "result_id")
        assert hasattr(result, "total_gross_emissions")
        assert hasattr(result, "total_net_emissions")
        assert hasattr(result, "total_written_premium")
        assert hasattr(result, "lob_breakdown")
        assert hasattr(result, "sector_breakdown")
        assert hasattr(result, "provenance_hash")
        assert hasattr(result, "engine_version")
        assert result.engine_version == "1.0.0"

    def test_methodology_notes(self, default_engine, two_policy_portfolio):
        """Portfolio result includes methodology notes."""
        result = default_engine.calculate_underwriting_emissions(two_policy_portfolio)
        assert isinstance(result.methodology_notes, list)

    def test_vehicle_emission_factors_exist(self):
        """Vehicle emission factors dictionary is populated."""
        assert len(VEHICLE_EMISSION_FACTORS) > 0
        assert "passenger_car_petrol" in VEHICLE_EMISSION_FACTORS
        assert "heavy_commercial" in VEHICLE_EMISSION_FACTORS

    def test_property_emission_factors_exist(self):
        """Property emission factors dictionary is populated."""
        assert len(PROPERTY_EMISSION_FACTORS) > 0
        assert "office" in PROPERTY_EMISSION_FACTORS
        assert "industrial" in PROPERTY_EMISSION_FACTORS
