# -*- coding: utf-8 -*-
"""
PACK-017 ESRS Full Coverage Pack - G1 Business Conduct Engine Tests
====================================================================

Unit tests for BusinessConductEngine (G1) covering policy assessment,
supplier management, corruption prevention, corruption incidents,
political influence, payment practices, full disclosure calculation,
completeness validation, and SHA-256 provenance.

ESRS G1: Business Conduct.

Target: 55+ tests.

Author:  GreenLang Platform Team (GL-TestEngineer)
Pack:    PACK-017 ESRS Full Coverage
Date:    March 2026
"""

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from .conftest import _load_engine, ENGINES_DIR


@pytest.fixture(scope="module")
def mod():
    return _load_engine("g1_business_conduct")


@pytest.fixture
def engine(mod):
    return mod.BusinessConductEngine()


@pytest.fixture
def comprehensive_policy(mod):
    return mod.BusinessConductPolicy(
        policy_name="Code of Conduct and Anti-Corruption Policy",
        covers_business_ethics=True,
        covers_anti_corruption=True,
        covers_whistleblower_protection=True,
        covers_supplier_standards=True,
        covers_political_engagement=True,
        approved_by_governance_body=True,
        training_types_associated=[
            mod.TrainingType.ANTI_CORRUPTION,
            mod.TrainingType.CODE_OF_CONDUCT,
        ],
        total_employees_trained=4500,
        total_employees_in_scope=5000,
    )


@pytest.fixture
def basic_policy(mod):
    return mod.BusinessConductPolicy(
        policy_name="Supplier Code of Conduct",
        covers_supplier_standards=True,
        covers_anti_corruption=False,
        approved_by_governance_body=False,
        total_employees_trained=0,
        total_employees_in_scope=5000,
    )


@pytest.fixture
def strategic_supplier(mod):
    return mod.SupplierRelationship(
        supplier_name="SteelCo International",
        category=mod.SupplierCategory.STRATEGIC,
        code_of_conduct_signed=True,
        last_audit_date=datetime(2025, 1, 15, tzinfo=timezone.utc),
        audit_passed=True,
        corruption_risk_level=mod.CorruptionRiskLevel.LOW,
        is_sme=False,
        country_code="DE",
    )


@pytest.fixture
def conditional_supplier(mod):
    return mod.SupplierRelationship(
        supplier_name="QuickParts Ltd",
        category=mod.SupplierCategory.CONDITIONAL,
        code_of_conduct_signed=False,
        corruption_risk_level=mod.CorruptionRiskLevel.HIGH,
        is_sme=True,
        country_code="NG",
    )


@pytest.fixture
def blocked_supplier(mod):
    return mod.SupplierRelationship(
        supplier_name="Blocked Corp",
        category=mod.SupplierCategory.BLOCKED,
        code_of_conduct_signed=False,
        corruption_risk_level=mod.CorruptionRiskLevel.VERY_HIGH,
        is_sme=False,
        country_code="XX",
    )


@pytest.fixture
def prevention_measure_training(mod):
    return mod.CorruptionPreventionMeasure(
        measure_type="training",
        description="Annual anti-corruption training for all staff",
        training_type=mod.TrainingType.ANTI_CORRUPTION,
        employees_covered=4000,
        total_employees=5000,
        covers_third_parties=False,
        is_active=True,
    )


@pytest.fixture
def prevention_measure_whistleblower(mod):
    return mod.CorruptionPreventionMeasure(
        measure_type="whistleblower",
        description="Anonymous whistleblower hotline",
        whistleblower_reports_received=12,
        whistleblower_reports_investigated=10,
        covers_third_parties=True,
        is_active=True,
    )


@pytest.fixture
def sample_incident(mod):
    return mod.CorruptionIncident(
        incident_type="bribery",
        date_confirmed=datetime(2025, 1, 20, tzinfo=timezone.utc),
        legal_proceedings_initiated=True,
        fine_amount_eur=Decimal("250000"),
        contracts_terminated=2,
    )


@pytest.fixture
def lobbying_activity(mod):
    return mod.PoliticalActivity(
        activity_type=mod.PoliticalActivityType.LOBBYING,
        description="EU regulatory engagement on sustainability reporting",
        amount_eur=Decimal("75000"),
        country_code="BE",
    )


@pytest.fixture
def donation_activity(mod):
    return mod.PoliticalActivity(
        activity_type=mod.PoliticalActivityType.POLITICAL_DONATIONS,
        description="Campaign contribution to local candidate",
        amount_eur=Decimal("5000"),
        country_code="DE",
    )


@pytest.fixture
def sme_payment(mod):
    return mod.PaymentPractice(
        supplier_id="SUP-001",
        is_sme_supplier=True,
        agreed_payment_days=30,
        actual_payment_days=42,
        invoice_amount_eur=Decimal("10000"),
    )


@pytest.fixture
def large_payment(mod):
    return mod.PaymentPractice(
        supplier_id="SUP-002",
        is_sme_supplier=False,
        agreed_payment_days=60,
        actual_payment_days=55,
        invoice_amount_eur=Decimal("250000"),
    )


# ===========================================================================
# Enum Tests
# ===========================================================================


class TestG1Enums:

    def test_corruption_risk_level_count(self, mod):
        assert len(mod.CorruptionRiskLevel) == 4

    def test_political_activity_type_count(self, mod):
        assert len(mod.PoliticalActivityType) == 5

    def test_supplier_category_count(self, mod):
        assert len(mod.SupplierCategory) == 5

    def test_payment_term_type_count(self, mod):
        assert len(mod.PaymentTermType) == 5

    def test_training_type_count(self, mod):
        assert len(mod.TrainingType) == 5


# ===========================================================================
# Constants Tests
# ===========================================================================


class TestG1Constants:

    def test_all_datapoints_count(self, mod):
        assert len(mod.ALL_G1_DATAPOINTS) == 40


# ===========================================================================
# Policy Assessment Tests (G1-1)
# ===========================================================================


class TestPolicyAssessment:

    def test_policy_count(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        assert result["policy_count"] == 1

    def test_covers_business_ethics(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        assert result["covers_business_ethics"] is True

    def test_covers_anti_corruption(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        assert result["covers_anti_corruption"] is True

    def test_covers_whistleblower(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        assert result["covers_whistleblower"] is True

    def test_training_coverage(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        pct = float(result["training_coverage_pct"])
        # 4500 / 5000 = 90%
        assert pct == pytest.approx(90.0, abs=1.0)

    def test_governance_body_oversight(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        pct = float(result["governance_body_oversight_pct"])
        assert pct == pytest.approx(100.0, abs=0.1)

    def test_empty_policies(self, engine):
        result = engine.assess_policies([])
        assert result["policy_count"] == 0

    def test_policy_provenance(self, engine, comprehensive_policy):
        result = engine.assess_policies([comprehensive_policy])
        assert len(result["provenance_hash"]) == 64

    def test_multiple_policies(self, engine, comprehensive_policy, basic_policy):
        result = engine.assess_policies([comprehensive_policy, basic_policy])
        assert result["policy_count"] == 2


# ===========================================================================
# Supplier Management Tests (G1-2)
# ===========================================================================


class TestSupplierManagement:

    def test_assessed_count(
        self, engine, strategic_supplier, conditional_supplier,
    ):
        result = engine.evaluate_supplier_management(
            [strategic_supplier, conditional_supplier]
        )
        assert result["assessed_count"] == 2

    def test_code_of_conduct_coverage(
        self, engine, strategic_supplier, conditional_supplier,
    ):
        result = engine.evaluate_supplier_management(
            [strategic_supplier, conditional_supplier]
        )
        assert result["code_of_conduct_signed_count"] == 1
        pct = float(result["code_of_conduct_coverage_pct"])
        assert pct == pytest.approx(50.0, abs=1.0)

    def test_blocked_count(
        self, engine, strategic_supplier, blocked_supplier,
    ):
        result = engine.evaluate_supplier_management(
            [strategic_supplier, blocked_supplier]
        )
        assert result["blocked_count"] == 1

    def test_sme_detection(
        self, engine, strategic_supplier, conditional_supplier,
    ):
        result = engine.evaluate_supplier_management(
            [strategic_supplier, conditional_supplier]
        )
        assert result["sme_count"] == 1

    def test_empty_suppliers(self, engine):
        result = engine.evaluate_supplier_management([])
        assert result["assessed_count"] == 0

    def test_supplier_provenance(self, engine, strategic_supplier):
        result = engine.evaluate_supplier_management([strategic_supplier])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Corruption Prevention Tests (G1-3)
# ===========================================================================


class TestCorruptionPrevention:

    def test_measure_count(
        self, engine, prevention_measure_training, prevention_measure_whistleblower,
    ):
        result = engine.assess_corruption_prevention(
            [prevention_measure_training, prevention_measure_whistleblower]
        )
        assert result["measure_count"] == 2

    def test_training_coverage(self, engine, prevention_measure_training):
        result = engine.assess_corruption_prevention([prevention_measure_training])
        # 4000 / 5000 = 80%
        pct = float(result["training_coverage_pct"])
        assert pct == pytest.approx(80.0, abs=1.0)

    def test_whistleblower_investigation_rate(
        self, engine, prevention_measure_whistleblower,
    ):
        result = engine.assess_corruption_prevention(
            [prevention_measure_whistleblower]
        )
        # 10 / 12 = 83.3%
        rate = float(result["investigation_rate_pct"])
        assert rate == pytest.approx(83.3, abs=1.0)

    def test_empty_measures(self, engine):
        result = engine.assess_corruption_prevention([])
        assert result["measure_count"] == 0

    def test_prevention_provenance(self, engine, prevention_measure_training):
        result = engine.assess_corruption_prevention([prevention_measure_training])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Corruption Incidents Tests (G1-4)
# ===========================================================================


class TestCorruptionIncidents:

    def test_incident_count(self, engine, sample_incident):
        result = engine.assess_corruption_incidents([sample_incident])
        assert result["incident_count"] == 1

    def test_legal_proceedings(self, engine, sample_incident):
        result = engine.assess_corruption_incidents([sample_incident])
        assert result["legal_proceedings_count"] == 1

    def test_total_fines(self, engine, sample_incident):
        result = engine.assess_corruption_incidents([sample_incident])
        total = Decimal(str(result["total_fines_eur"]))
        assert total == Decimal("250000")

    def test_contracts_terminated(self, engine, sample_incident):
        result = engine.assess_corruption_incidents([sample_incident])
        assert result["total_contracts_terminated"] == 2

    def test_empty_incidents(self, engine):
        result = engine.assess_corruption_incidents([])
        assert result["incident_count"] == 0

    def test_incident_provenance(self, engine, sample_incident):
        result = engine.assess_corruption_incidents([sample_incident])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Political Influence Tests (G1-5)
# ===========================================================================


class TestPoliticalInfluence:

    def test_activity_count(self, engine, lobbying_activity, donation_activity):
        result = engine.assess_political_influence(
            [lobbying_activity, donation_activity]
        )
        assert result["activity_count"] == 2

    def test_total_expenditure(self, engine, lobbying_activity, donation_activity):
        result = engine.assess_political_influence(
            [lobbying_activity, donation_activity]
        )
        total = Decimal(str(result["total_political_spend_eur"]))
        assert total == Decimal("80000")

    def test_by_type_breakdown(self, engine, lobbying_activity, donation_activity):
        result = engine.assess_political_influence(
            [lobbying_activity, donation_activity]
        )
        by_type = result["by_type"]
        assert by_type["lobbying"]["count"] >= 1

    def test_empty_activities(self, engine):
        result = engine.assess_political_influence([])
        assert result["activity_count"] == 0

    def test_political_provenance(self, engine, lobbying_activity):
        result = engine.assess_political_influence([lobbying_activity])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Payment Practices Tests (G1-6)
# ===========================================================================


class TestPaymentPractices:

    def test_payment_count(self, engine, sme_payment, large_payment):
        result = engine.calculate_payment_practices(
            [sme_payment, large_payment]
        )
        assert result["payment_count"] == 2

    def test_late_payment_detection(self, engine, sme_payment, large_payment):
        result = engine.calculate_payment_practices(
            [sme_payment, large_payment]
        )
        # sme_payment: 42 > 30 = late; large_payment: 55 < 60 = on time
        assert result["late_payment_count"] >= 1

    def test_sme_payment_tracking(self, engine, sme_payment, large_payment):
        result = engine.calculate_payment_practices(
            [sme_payment, large_payment]
        )
        assert result["sme_payment_count"] >= 1

    def test_empty_payments(self, engine):
        result = engine.calculate_payment_practices([])
        assert result["payment_count"] == 0

    def test_payment_provenance(self, engine, sme_payment):
        result = engine.calculate_payment_practices([sme_payment])
        assert len(result["provenance_hash"]) == 64


# ===========================================================================
# Full Disclosure Tests
# ===========================================================================


class TestG1Disclosure:

    def test_full_disclosure(
        self, engine, comprehensive_policy, strategic_supplier,
        prevention_measure_training, sample_incident,
        lobbying_activity, sme_payment,
    ):
        result = engine.calculate_g1_disclosure(
            policies=[comprehensive_policy],
            suppliers=[strategic_supplier],
            prevention_measures=[prevention_measure_training],
            incidents=[sample_incident],
            political_activities=[lobbying_activity],
            payments=[sme_payment],
        )
        assert result.total_policies > 0

    def test_disclosure_provenance(
        self, engine, comprehensive_policy, strategic_supplier,
        prevention_measure_training, sample_incident,
        lobbying_activity, sme_payment,
    ):
        result = engine.calculate_g1_disclosure(
            policies=[comprehensive_policy],
            suppliers=[strategic_supplier],
            prevention_measures=[prevention_measure_training],
            incidents=[sample_incident],
            political_activities=[lobbying_activity],
            payments=[sme_payment],
        )
        assert len(result.provenance_hash) == 64
        int(result.provenance_hash, 16)


# ===========================================================================
# Completeness Tests
# ===========================================================================


class TestG1Completeness:

    def test_completeness_structure(
        self, engine, comprehensive_policy, strategic_supplier,
        prevention_measure_training, sample_incident,
        lobbying_activity, sme_payment,
    ):
        result = engine.calculate_g1_disclosure(
            policies=[comprehensive_policy],
            suppliers=[strategic_supplier],
            prevention_measures=[prevention_measure_training],
            incidents=[sample_incident],
            political_activities=[lobbying_activity],
            payments=[sme_payment],
        )
        completeness = engine.validate_g1_completeness(result)
        assert completeness["total_datapoints"] == 40
        assert "per_dr_completeness" in completeness

    def test_partial_missing(self, engine, comprehensive_policy):
        result = engine.calculate_g1_disclosure(
            policies=[comprehensive_policy],
            suppliers=[],
            prevention_measures=[],
            incidents=[],
            political_activities=[],
            payments=[],
        )
        completeness = engine.validate_g1_completeness(result)
        assert len(completeness["missing_datapoints"]) > 0

    def test_completeness_provenance(
        self, engine, comprehensive_policy, strategic_supplier,
        prevention_measure_training, sample_incident,
        lobbying_activity, sme_payment,
    ):
        result = engine.calculate_g1_disclosure(
            policies=[comprehensive_policy],
            suppliers=[strategic_supplier],
            prevention_measures=[prevention_measure_training],
            incidents=[sample_incident],
            political_activities=[lobbying_activity],
            payments=[sme_payment],
        )
        completeness = engine.validate_g1_completeness(result)
        assert len(completeness["provenance_hash"]) == 64


# ===========================================================================
# Source Code Quality Tests
# ===========================================================================


class TestG1SourceQuality:

    def test_engine_has_docstring(self, mod):
        assert mod.BusinessConductEngine.__doc__ is not None

    def test_engine_source_has_sha256(self):
        source = (ENGINES_DIR / "g1_business_conduct_engine.py").read_text(
            encoding="utf-8"
        )
        assert "sha256" in source.lower() or "hashlib" in source

    def test_engine_source_has_decimal(self):
        source = (ENGINES_DIR / "g1_business_conduct_engine.py").read_text(
            encoding="utf-8"
        )
        assert "Decimal" in source

    def test_engine_source_has_basemodel(self):
        source = (ENGINES_DIR / "g1_business_conduct_engine.py").read_text(
            encoding="utf-8"
        )
        assert "BaseModel" in source

    @pytest.mark.parametrize(
        "dr", ["G1-1", "G1-2", "G1-3", "G1-4", "G1-5", "G1-6"]
    )
    def test_all_6_drs_referenced(self, dr):
        source = (ENGINES_DIR / "g1_business_conduct_engine.py").read_text(
            encoding="utf-8"
        )
        normalized = dr.replace("-", "_")
        assert dr in source or normalized in source, (
            f"G1 engine should reference {dr}"
        )
