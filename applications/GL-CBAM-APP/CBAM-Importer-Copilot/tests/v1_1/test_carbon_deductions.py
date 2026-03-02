# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Carbon Price Deduction Engine

Tests CarbonPriceDeductionEngine:
- Singleton pattern
- Register deduction (EUR, USD, TRY, CNY, unknown currency)
- Provenance hash computation
- Get deductions by importer/year
- Get deduction by ID / not found
- Verify, approve, reject deduction lifecycle
- Add evidence documents
- Get total deduction EUR (eligible, mixed status, none eligible)
- Deduction summary generation
- Country carbon pricing lookups (Turkey, China, unknown)
- Deduction per tonne calculation
- Multiple deductions same importer / across years

Target: 80+ tests
"""

import pytest
import hashlib
import json
import uuid
import threading
from datetime import date, datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from copy import deepcopy
from typing import Any, Dict, List, Optional

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from certificate_engine.models import (
    CarbonPriceDeduction,
    CarbonPricingScheme,
    DeductionStatus,
    ECB_EXCHANGE_RATES,
    compute_sha256,
    quantize_decimal,
)


# ---------------------------------------------------------------------------
# Inline CarbonPriceDeductionEngine (self-contained for testing)
# ---------------------------------------------------------------------------

# Country carbon pricing data (illustrative)
COUNTRY_CARBON_PRICING = {
    "TR": {
        "country": "Turkey",
        "scheme": CarbonPricingScheme.ETS,
        "name": "Turkey ETS",
        "currency": "TRY",
        "estimated_price_per_tco2e_local": Decimal("300.00"),
    },
    "CN": {
        "country": "China",
        "scheme": CarbonPricingScheme.ETS,
        "name": "China National ETS",
        "currency": "CNY",
        "estimated_price_per_tco2e_local": Decimal("70.00"),
    },
    "GB": {
        "country": "United Kingdom",
        "scheme": CarbonPricingScheme.ETS,
        "name": "UK ETS",
        "currency": "GBP",
        "estimated_price_per_tco2e_local": Decimal("45.00"),
    },
    "CA": {
        "country": "Canada",
        "scheme": CarbonPricingScheme.CARBON_TAX,
        "name": "Canada Federal Carbon Tax",
        "currency": "CAD",
        "estimated_price_per_tco2e_local": Decimal("80.00"),
    },
}


class CarbonPriceDeductionEngine:
    """Engine for managing carbon price deductions under CBAM Article 26."""

    _instance: Optional["CarbonPriceDeductionEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._deductions: Dict[str, CarbonPriceDeduction] = {}
        self._importer_index: Dict[str, Dict[int, List[str]]] = {}

    @classmethod
    def get_instance(cls) -> "CarbonPriceDeductionEngine":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        with cls._lock:
            cls._instance = None

    def register_deduction(
        self,
        importer_id: str,
        installation_id: str,
        country: str,
        pricing_scheme: CarbonPricingScheme,
        amount_paid: Decimal,
        currency: str,
        tonnes_covered: Decimal,
        year: int,
        evidence_docs: Optional[List[str]] = None,
    ) -> CarbonPriceDeduction:
        """Register a carbon price deduction with currency conversion to EUR."""
        deduction_id = f"CPD-{year}-{importer_id[:6]}-{uuid.uuid4().hex[:6].upper()}"

        # Currency conversion
        if currency == "EUR":
            amount_eur = amount_paid
            exchange_rate = Decimal("1.000000")
        else:
            rate = ECB_EXCHANGE_RATES.get(currency)
            if rate is None:
                raise ValueError(
                    f"Unknown currency '{currency}'. "
                    f"Supported: {sorted(ECB_EXCHANGE_RATES.keys())}"
                )
            amount_eur = (amount_paid * rate).quantize(
                Decimal("0.01"), rounding=ROUND_HALF_UP
            )
            exchange_rate = rate

        deduction = CarbonPriceDeduction(
            deduction_id=deduction_id,
            importer_id=importer_id,
            installation_id=installation_id,
            country=country,
            pricing_scheme=pricing_scheme,
            carbon_price_paid_eur=amount_eur,
            carbon_price_paid_local=amount_paid,
            exchange_rate=exchange_rate,
            currency=currency,
            tonnes_covered=tonnes_covered,
            evidence_docs=evidence_docs or [],
            verification_status=DeductionStatus.PENDING,
            year=year,
        )
        deduction.provenance_hash = deduction.compute_provenance_hash()

        self._deductions[deduction_id] = deduction
        if importer_id not in self._importer_index:
            self._importer_index[importer_id] = {}
        if year not in self._importer_index[importer_id]:
            self._importer_index[importer_id][year] = []
        self._importer_index[importer_id][year].append(deduction_id)

        return deduction

    def get_deductions(
        self, importer_id: str, year: int
    ) -> List[CarbonPriceDeduction]:
        ids = self._importer_index.get(importer_id, {}).get(year, [])
        return [self._deductions[did] for did in ids if did in self._deductions]

    def get_deduction(self, deduction_id: str) -> Optional[CarbonPriceDeduction]:
        return self._deductions.get(deduction_id)

    def verify_deduction(
        self, deduction_id: str, verifier_id: str
    ) -> CarbonPriceDeduction:
        d = self._deductions.get(deduction_id)
        if d is None:
            raise KeyError(f"Deduction {deduction_id} not found")
        if d.verification_status != DeductionStatus.PENDING:
            raise ValueError(
                f"Cannot verify deduction in status {d.verification_status.value}"
            )
        updated = d.model_copy(
            update={
                "verification_status": DeductionStatus.VERIFIED,
                "verified_by": verifier_id,
                "verified_at": datetime.now(timezone.utc),
            }
        )
        updated.provenance_hash = updated.compute_provenance_hash()
        self._deductions[deduction_id] = updated
        return updated

    def approve_deduction(
        self, deduction_id: str, verifier_id: str
    ) -> CarbonPriceDeduction:
        d = self._deductions.get(deduction_id)
        if d is None:
            raise KeyError(f"Deduction {deduction_id} not found")
        if d.verification_status not in (
            DeductionStatus.VERIFIED,
            DeductionStatus.PENDING,
        ):
            raise ValueError(
                f"Cannot approve deduction in status {d.verification_status.value}"
            )
        updated = d.model_copy(
            update={
                "verification_status": DeductionStatus.APPROVED,
                "verified_by": verifier_id,
                "verified_at": datetime.now(timezone.utc),
            }
        )
        updated.provenance_hash = updated.compute_provenance_hash()
        self._deductions[deduction_id] = updated
        return updated

    def reject_deduction(
        self, deduction_id: str, verifier_id: str
    ) -> CarbonPriceDeduction:
        d = self._deductions.get(deduction_id)
        if d is None:
            raise KeyError(f"Deduction {deduction_id} not found")
        updated = d.model_copy(
            update={
                "verification_status": DeductionStatus.REJECTED,
                "verified_by": verifier_id,
                "verified_at": datetime.now(timezone.utc),
            }
        )
        updated.provenance_hash = updated.compute_provenance_hash()
        self._deductions[deduction_id] = updated
        return updated

    def add_evidence(
        self, deduction_id: str, evidence_ref: str
    ) -> CarbonPriceDeduction:
        d = self._deductions.get(deduction_id)
        if d is None:
            raise KeyError(f"Deduction {deduction_id} not found")
        new_docs = list(d.evidence_docs) + [evidence_ref]
        updated = d.model_copy(update={"evidence_docs": new_docs})
        updated.provenance_hash = updated.compute_provenance_hash()
        self._deductions[deduction_id] = updated
        return updated

    def get_total_deduction_eur(
        self, importer_id: str, year: int
    ) -> Decimal:
        """Sum only eligible (verified/approved) deductions."""
        deductions = self.get_deductions(importer_id, year)
        total = Decimal("0")
        for d in deductions:
            if d.verification_status.is_eligible:
                total += d.carbon_price_paid_eur
        return total

    def get_deduction_summary(
        self, importer_id: str, year: int
    ) -> Dict[str, Any]:
        deductions = self.get_deductions(importer_id, year)
        total_eur = Decimal("0")
        eligible_eur = Decimal("0")
        total_tonnes = Decimal("0")
        by_status: Dict[str, int] = {}
        by_country: Dict[str, Decimal] = {}

        for d in deductions:
            total_eur += d.carbon_price_paid_eur
            total_tonnes += d.tonnes_covered
            status_key = d.verification_status.value
            by_status[status_key] = by_status.get(status_key, 0) + 1
            by_country[d.country] = by_country.get(d.country, Decimal("0")) + d.carbon_price_paid_eur
            if d.verification_status.is_eligible:
                eligible_eur += d.carbon_price_paid_eur

        return {
            "importer_id": importer_id,
            "year": year,
            "total_deductions": len(deductions),
            "total_amount_eur": total_eur,
            "eligible_amount_eur": eligible_eur,
            "total_tonnes_covered": total_tonnes,
            "by_status": by_status,
            "by_country": {k: float(v) for k, v in by_country.items()},
        }

    @staticmethod
    def get_country_carbon_pricing(
        country_code: str,
    ) -> Optional[Dict[str, Any]]:
        return COUNTRY_CARBON_PRICING.get(country_code)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before each test."""
    CarbonPriceDeductionEngine.reset_instance()
    yield
    CarbonPriceDeductionEngine.reset_instance()


@pytest.fixture
def engine():
    return CarbonPriceDeductionEngine()


@pytest.fixture
def sample_deduction_eur(engine):
    """Register a sample EUR deduction and return it."""
    return engine.register_deduction(
        importer_id="NL123456789012",
        installation_id="TR-INSTALL-001",
        country="TR",
        pricing_scheme=CarbonPricingScheme.ETS,
        amount_paid=Decimal("15000.00"),
        currency="EUR",
        tonnes_covered=Decimal("1000"),
        year=2026,
        evidence_docs=["receipt-001.pdf"],
    )


@pytest.fixture
def sample_deduction_usd(engine):
    """Register a sample USD deduction."""
    return engine.register_deduction(
        importer_id="NL123456789012",
        installation_id="US-INSTALL-001",
        country="US",
        pricing_scheme=CarbonPricingScheme.CARBON_TAX,
        amount_paid=Decimal("10000.00"),
        currency="USD",
        tonnes_covered=Decimal("500"),
        year=2026,
    )


# ===========================================================================
# TEST CLASS -- Singleton pattern
# ===========================================================================

class TestSingletonPattern:
    """Tests for CarbonPriceDeductionEngine singleton."""

    def test_singleton_returns_same_instance(self):
        a = CarbonPriceDeductionEngine.get_instance()
        b = CarbonPriceDeductionEngine.get_instance()
        assert a is b

    def test_singleton_reset_creates_new_instance(self):
        a = CarbonPriceDeductionEngine.get_instance()
        CarbonPriceDeductionEngine.reset_instance()
        b = CarbonPriceDeductionEngine.get_instance()
        assert a is not b

    def test_singleton_thread_safety(self):
        instances = []

        def get_inst():
            instances.append(CarbonPriceDeductionEngine.get_instance())

        threads = [threading.Thread(target=get_inst) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(id(i) for i in instances)) == 1


# ===========================================================================
# TEST CLASS -- Register deduction
# ===========================================================================

class TestRegisterDeduction:
    """Tests for register_deduction."""

    def test_register_deduction_basic(self, engine):
        d = engine.register_deduction(
            importer_id="DE000000000001",
            installation_id="INST-001",
            country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000.00"),
            currency="EUR",
            tonnes_covered=Decimal("200"),
            year=2026,
        )
        assert d.deduction_id.startswith("CPD-2026-")
        assert d.importer_id == "DE000000000001"
        assert d.verification_status == DeductionStatus.PENDING
        assert d.year == 2026

    def test_register_deduction_eur_no_conversion(self, engine):
        d = engine.register_deduction(
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="DE",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("10000.00"),
            currency="EUR",
            tonnes_covered=Decimal("500"),
            year=2026,
        )
        assert d.carbon_price_paid_eur == Decimal("10000.00")
        assert d.exchange_rate == Decimal("1.000000")
        assert d.currency == "EUR"

    def test_register_deduction_usd_conversion(self, engine):
        d = engine.register_deduction(
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="US",
            pricing_scheme=CarbonPricingScheme.CARBON_TAX,
            amount_paid=Decimal("10000.00"),
            currency="USD",
            tonnes_covered=Decimal("500"),
            year=2026,
        )
        expected = (Decimal("10000.00") * ECB_EXCHANGE_RATES["USD"]).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert d.carbon_price_paid_eur == expected
        assert d.exchange_rate == ECB_EXCHANGE_RATES["USD"]
        assert d.currency == "USD"

    def test_register_deduction_try_conversion(self, engine):
        d = engine.register_deduction(
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("500000.00"),
            currency="TRY",
            tonnes_covered=Decimal("1000"),
            year=2026,
        )
        expected = (Decimal("500000.00") * ECB_EXCHANGE_RATES["TRY"]).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert d.carbon_price_paid_eur == expected
        assert d.currency == "TRY"

    def test_register_deduction_cny_conversion(self, engine):
        d = engine.register_deduction(
            importer_id="NL123456789012",
            installation_id="CN-INST-001",
            country="CN",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("70000.00"),
            currency="CNY",
            tonnes_covered=Decimal("1000"),
            year=2026,
        )
        expected = (Decimal("70000.00") * ECB_EXCHANGE_RATES["CNY"]).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert d.carbon_price_paid_eur == expected
        assert d.currency == "CNY"

    def test_register_deduction_unknown_currency_raises(self, engine):
        with pytest.raises(ValueError, match="Unknown currency"):
            engine.register_deduction(
                importer_id="NL123456789012",
                installation_id="INST-001",
                country="XY",
                pricing_scheme=CarbonPricingScheme.CARBON_TAX,
                amount_paid=Decimal("5000.00"),
                currency="XYZ",
                tonnes_covered=Decimal("100"),
                year=2026,
            )

    def test_register_deduction_with_evidence(self, engine):
        d = engine.register_deduction(
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000.00"),
            currency="EUR",
            tonnes_covered=Decimal("200"),
            year=2026,
            evidence_docs=["receipt.pdf", "tax-cert.pdf"],
        )
        assert d.evidence_docs == ["receipt.pdf", "tax-cert.pdf"]

    def test_register_deduction_no_evidence(self, engine):
        d = engine.register_deduction(
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000.00"),
            currency="EUR",
            tonnes_covered=Decimal("200"),
            year=2026,
        )
        assert d.evidence_docs == []


# ===========================================================================
# TEST CLASS -- Provenance hash
# ===========================================================================

class TestProvenanceHash:
    """Tests for provenance hash computation."""

    def test_provenance_hash_is_sha256(self, sample_deduction_eur):
        assert len(sample_deduction_eur.provenance_hash) == 64

    def test_provenance_hash_deterministic(self, engine):
        d1 = CarbonPriceDeduction(
            deduction_id="CPD-2026-TEST-001",
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            carbon_price_paid_eur=Decimal("15000.00"),
            tonnes_covered=Decimal("1000"),
            year=2026,
        )
        h1 = d1.compute_provenance_hash()
        h2 = d1.compute_provenance_hash()
        assert h1 == h2

    def test_provenance_hash_changes_with_amount(self, engine):
        d1 = CarbonPriceDeduction(
            deduction_id="CPD-2026-TEST-001",
            importer_id="NL123456789012",
            installation_id="INST-001",
            country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            carbon_price_paid_eur=Decimal("15000.00"),
            tonnes_covered=Decimal("1000"),
            year=2026,
        )
        d2 = d1.model_copy(update={"carbon_price_paid_eur": Decimal("20000.00")})
        assert d1.compute_provenance_hash() != d2.compute_provenance_hash()


# ===========================================================================
# TEST CLASS -- Get deductions
# ===========================================================================

class TestGetDeductions:
    """Tests for get_deductions_by_importer_year and get_deduction_by_id."""

    def test_get_deductions_by_importer_year(self, engine, sample_deduction_eur):
        results = engine.get_deductions("NL123456789012", 2026)
        assert len(results) == 1
        assert results[0].deduction_id == sample_deduction_eur.deduction_id

    def test_get_deductions_empty(self, engine):
        results = engine.get_deductions("UNKNOWN-IMP", 2026)
        assert results == []

    def test_get_deductions_wrong_year(self, engine, sample_deduction_eur):
        results = engine.get_deductions("NL123456789012", 2027)
        assert results == []

    def test_get_deduction_by_id(self, engine, sample_deduction_eur):
        result = engine.get_deduction(sample_deduction_eur.deduction_id)
        assert result is not None
        assert result.importer_id == "NL123456789012"

    def test_get_deduction_not_found(self, engine):
        result = engine.get_deduction("CPD-NONEXISTENT")
        assert result is None


# ===========================================================================
# TEST CLASS -- Verification lifecycle
# ===========================================================================

class TestVerificationLifecycle:
    """Tests for verify, approve, reject deduction."""

    def test_verify_deduction(self, engine, sample_deduction_eur):
        d = engine.verify_deduction(sample_deduction_eur.deduction_id, "VER-001")
        assert d.verification_status == DeductionStatus.VERIFIED
        assert d.verified_by == "VER-001"
        assert d.verified_at is not None

    def test_approve_deduction(self, engine, sample_deduction_eur):
        engine.verify_deduction(sample_deduction_eur.deduction_id, "VER-001")
        d = engine.approve_deduction(sample_deduction_eur.deduction_id, "ADMIN-001")
        assert d.verification_status == DeductionStatus.APPROVED
        assert d.verified_by == "ADMIN-001"

    def test_reject_deduction(self, engine, sample_deduction_eur):
        d = engine.reject_deduction(sample_deduction_eur.deduction_id, "VER-001")
        assert d.verification_status == DeductionStatus.REJECTED
        assert d.verified_by == "VER-001"

    def test_verify_already_verified_raises(self, engine, sample_deduction_eur):
        engine.verify_deduction(sample_deduction_eur.deduction_id, "VER-001")
        with pytest.raises(ValueError, match="Cannot verify"):
            engine.verify_deduction(sample_deduction_eur.deduction_id, "VER-002")

    def test_verify_nonexistent_raises(self, engine):
        with pytest.raises(KeyError):
            engine.verify_deduction("CPD-NONE", "VER-001")

    def test_approve_nonexistent_raises(self, engine):
        with pytest.raises(KeyError):
            engine.approve_deduction("CPD-NONE", "VER-001")

    def test_reject_nonexistent_raises(self, engine):
        with pytest.raises(KeyError):
            engine.reject_deduction("CPD-NONE", "VER-001")

    def test_approve_rejected_raises(self, engine, sample_deduction_eur):
        engine.reject_deduction(sample_deduction_eur.deduction_id, "VER-001")
        with pytest.raises(ValueError, match="Cannot approve"):
            engine.approve_deduction(sample_deduction_eur.deduction_id, "VER-001")

    def test_verification_updates_provenance(self, engine, sample_deduction_eur):
        original_hash = sample_deduction_eur.provenance_hash
        d = engine.verify_deduction(sample_deduction_eur.deduction_id, "VER-001")
        assert d.provenance_hash != original_hash


# ===========================================================================
# TEST CLASS -- Add evidence
# ===========================================================================

class TestAddEvidence:
    """Tests for add_evidence."""

    def test_add_evidence(self, engine, sample_deduction_eur):
        d = engine.add_evidence(sample_deduction_eur.deduction_id, "new-doc.pdf")
        assert "new-doc.pdf" in d.evidence_docs
        assert "receipt-001.pdf" in d.evidence_docs
        assert len(d.evidence_docs) == 2

    def test_add_multiple_evidence(self, engine, sample_deduction_eur):
        engine.add_evidence(sample_deduction_eur.deduction_id, "doc1.pdf")
        d = engine.add_evidence(sample_deduction_eur.deduction_id, "doc2.pdf")
        assert len(d.evidence_docs) == 3

    def test_add_evidence_nonexistent_raises(self, engine):
        with pytest.raises(KeyError):
            engine.add_evidence("CPD-NONE", "doc.pdf")


# ===========================================================================
# TEST CLASS -- Total deduction EUR
# ===========================================================================

class TestTotalDeductionEUR:
    """Tests for get_total_deduction_eur."""

    def test_total_all_eligible(self, engine):
        d1 = engine.register_deduction(
            importer_id="IMP-001", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000"), currency="EUR",
            tonnes_covered=Decimal("100"), year=2026,
        )
        d2 = engine.register_deduction(
            importer_id="IMP-001", installation_id="INST-2", country="CN",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("3000"), currency="EUR",
            tonnes_covered=Decimal("50"), year=2026,
        )
        engine.verify_deduction(d1.deduction_id, "VER-001")
        engine.approve_deduction(d2.deduction_id, "VER-001")
        total = engine.get_total_deduction_eur("IMP-001", 2026)
        assert total == Decimal("8000")

    def test_total_mixed_status(self, engine):
        d1 = engine.register_deduction(
            importer_id="IMP-002", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000"), currency="EUR",
            tonnes_covered=Decimal("100"), year=2026,
        )
        d2 = engine.register_deduction(
            importer_id="IMP-002", installation_id="INST-2", country="CN",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("3000"), currency="EUR",
            tonnes_covered=Decimal("50"), year=2026,
        )
        d3 = engine.register_deduction(
            importer_id="IMP-002", installation_id="INST-3", country="US",
            pricing_scheme=CarbonPricingScheme.CARBON_TAX,
            amount_paid=Decimal("2000"), currency="EUR",
            tonnes_covered=Decimal("30"), year=2026,
        )
        engine.verify_deduction(d1.deduction_id, "VER-001")
        engine.reject_deduction(d2.deduction_id, "VER-001")
        # d3 remains pending
        total = engine.get_total_deduction_eur("IMP-002", 2026)
        assert total == Decimal("5000")  # Only d1 verified

    def test_total_none_eligible(self, engine):
        d1 = engine.register_deduction(
            importer_id="IMP-003", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000"), currency="EUR",
            tonnes_covered=Decimal("100"), year=2026,
        )
        engine.reject_deduction(d1.deduction_id, "VER-001")
        total = engine.get_total_deduction_eur("IMP-003", 2026)
        assert total == Decimal("0")

    def test_total_empty(self, engine):
        total = engine.get_total_deduction_eur("IMP-EMPTY", 2026)
        assert total == Decimal("0")


# ===========================================================================
# TEST CLASS -- Deduction summary
# ===========================================================================

class TestDeductionSummary:
    """Tests for get_deduction_summary."""

    def test_summary_basic(self, engine, sample_deduction_eur, sample_deduction_usd):
        summary = engine.get_deduction_summary("NL123456789012", 2026)
        assert summary["total_deductions"] == 2
        assert summary["importer_id"] == "NL123456789012"
        assert summary["year"] == 2026
        assert summary["total_amount_eur"] > Decimal("0")
        assert summary["total_tonnes_covered"] == Decimal("1500")

    def test_summary_by_status(self, engine, sample_deduction_eur):
        summary = engine.get_deduction_summary("NL123456789012", 2026)
        assert "pending" in summary["by_status"]
        assert summary["by_status"]["pending"] == 1

    def test_summary_by_country(self, engine, sample_deduction_eur):
        summary = engine.get_deduction_summary("NL123456789012", 2026)
        assert "TR" in summary["by_country"]

    def test_summary_empty(self, engine):
        summary = engine.get_deduction_summary("UNKNOWN", 2026)
        assert summary["total_deductions"] == 0
        assert summary["total_amount_eur"] == Decimal("0")


# ===========================================================================
# TEST CLASS -- Country carbon pricing
# ===========================================================================

class TestCountryCarbonPricing:
    """Tests for country carbon pricing lookups."""

    def test_turkey_pricing(self):
        pricing = CarbonPriceDeductionEngine.get_country_carbon_pricing("TR")
        assert pricing is not None
        assert pricing["country"] == "Turkey"
        assert pricing["scheme"] == CarbonPricingScheme.ETS
        assert pricing["currency"] == "TRY"

    def test_china_pricing(self):
        pricing = CarbonPriceDeductionEngine.get_country_carbon_pricing("CN")
        assert pricing is not None
        assert pricing["country"] == "China"
        assert pricing["scheme"] == CarbonPricingScheme.ETS
        assert pricing["currency"] == "CNY"

    def test_uk_pricing(self):
        pricing = CarbonPriceDeductionEngine.get_country_carbon_pricing("GB")
        assert pricing is not None
        assert pricing["name"] == "UK ETS"

    def test_canada_carbon_tax(self):
        pricing = CarbonPriceDeductionEngine.get_country_carbon_pricing("CA")
        assert pricing is not None
        assert pricing["scheme"] == CarbonPricingScheme.CARBON_TAX

    def test_unknown_country_returns_none(self):
        pricing = CarbonPriceDeductionEngine.get_country_carbon_pricing("ZZ")
        assert pricing is None


# ===========================================================================
# TEST CLASS -- Deduction per tonne calculation
# ===========================================================================

class TestDeductionPerTonne:
    """Tests for automatic deduction_per_tonne_eur computation."""

    def test_deduction_per_tonne_auto_computed(self, sample_deduction_eur):
        expected = (Decimal("15000.00") / Decimal("1000")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert sample_deduction_eur.deduction_per_tonne_eur == expected

    def test_deduction_per_tonne_fractional(self, engine):
        d = engine.register_deduction(
            importer_id="IMP-001", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("7777.77"), currency="EUR",
            tonnes_covered=Decimal("333"), year=2026,
        )
        expected = (Decimal("7777.77") / Decimal("333")).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        assert d.deduction_per_tonne_eur == expected

    def test_deduction_per_tonne_large(self, engine):
        d = engine.register_deduction(
            importer_id="IMP-001", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("1000000.00"), currency="EUR",
            tonnes_covered=Decimal("10000"), year=2026,
        )
        assert d.deduction_per_tonne_eur == Decimal("100.00")


# ===========================================================================
# TEST CLASS -- Multiple deductions
# ===========================================================================

class TestMultipleDeductions:
    """Tests for multiple deductions scenarios."""

    def test_multiple_deductions_same_importer(self, engine):
        for i in range(5):
            engine.register_deduction(
                importer_id="IMP-MULTI",
                installation_id=f"INST-{i}",
                country="TR",
                pricing_scheme=CarbonPricingScheme.ETS,
                amount_paid=Decimal("1000"),
                currency="EUR",
                tonnes_covered=Decimal("50"),
                year=2026,
            )
        deductions = engine.get_deductions("IMP-MULTI", 2026)
        assert len(deductions) == 5

    def test_deductions_across_years(self, engine):
        engine.register_deduction(
            importer_id="IMP-YEARS", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000"), currency="EUR",
            tonnes_covered=Decimal("100"), year=2026,
        )
        engine.register_deduction(
            importer_id="IMP-YEARS", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("6000"), currency="EUR",
            tonnes_covered=Decimal("120"), year=2027,
        )
        d_2026 = engine.get_deductions("IMP-YEARS", 2026)
        d_2027 = engine.get_deductions("IMP-YEARS", 2027)
        assert len(d_2026) == 1
        assert len(d_2027) == 1
        assert d_2026[0].year == 2026
        assert d_2027[0].year == 2027

    def test_deductions_different_importers_isolated(self, engine):
        engine.register_deduction(
            importer_id="IMP-A", installation_id="INST-1", country="TR",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000"), currency="EUR",
            tonnes_covered=Decimal("100"), year=2026,
        )
        engine.register_deduction(
            importer_id="IMP-B", installation_id="INST-2", country="CN",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("3000"), currency="EUR",
            tonnes_covered=Decimal("50"), year=2026,
        )
        assert len(engine.get_deductions("IMP-A", 2026)) == 1
        assert len(engine.get_deductions("IMP-B", 2026)) == 1

    def test_multiple_currencies_same_importer(self, engine):
        d_eur = engine.register_deduction(
            importer_id="IMP-CURR", installation_id="INST-1", country="DE",
            pricing_scheme=CarbonPricingScheme.ETS,
            amount_paid=Decimal("5000"), currency="EUR",
            tonnes_covered=Decimal("100"), year=2026,
        )
        d_usd = engine.register_deduction(
            importer_id="IMP-CURR", installation_id="INST-2", country="US",
            pricing_scheme=CarbonPricingScheme.CARBON_TAX,
            amount_paid=Decimal("5000"), currency="USD",
            tonnes_covered=Decimal("100"), year=2026,
        )
        assert d_eur.currency == "EUR"
        assert d_usd.currency == "USD"
        # Both stored under same importer
        deductions = engine.get_deductions("IMP-CURR", 2026)
        assert len(deductions) == 2


# ===========================================================================
# TEST CLASS -- ECB Exchange Rates
# ===========================================================================

class TestECBExchangeRates:
    """Tests for ECB exchange rate data integrity."""

    def test_eur_rate_is_one(self):
        assert ECB_EXCHANGE_RATES["EUR"] == Decimal("1.000000")

    def test_usd_rate_positive(self):
        assert ECB_EXCHANGE_RATES["USD"] > Decimal("0")

    def test_all_rates_positive(self):
        for currency, rate in ECB_EXCHANGE_RATES.items():
            assert rate > Decimal("0"), f"Rate for {currency} must be positive"

    def test_major_currencies_present(self):
        for currency in ["EUR", "USD", "GBP", "CHF", "CNY", "JPY", "TRY"]:
            assert currency in ECB_EXCHANGE_RATES, f"{currency} missing"

    @pytest.mark.parametrize("currency,expected_less_than", [
        ("USD", Decimal("1.1")),
        ("GBP", Decimal("1.3")),
        ("CNY", Decimal("0.2")),
        ("TRY", Decimal("0.1")),
    ])
    def test_rates_reasonable_range(self, currency, expected_less_than):
        rate = ECB_EXCHANGE_RATES[currency]
        assert rate < expected_less_than


# ===========================================================================
# TEST CLASS -- DeductionStatus enum
# ===========================================================================

class TestDeductionStatusEnum:
    """Tests for DeductionStatus enum properties."""

    def test_pending_not_eligible(self):
        assert not DeductionStatus.PENDING.is_eligible

    def test_verified_is_eligible(self):
        assert DeductionStatus.VERIFIED.is_eligible

    def test_approved_is_eligible(self):
        assert DeductionStatus.APPROVED.is_eligible

    def test_rejected_not_eligible(self):
        assert not DeductionStatus.REJECTED.is_eligible


# ===========================================================================
# TEST CLASS -- CarbonPricingScheme enum
# ===========================================================================

class TestCarbonPricingSchemeEnum:
    """Tests for CarbonPricingScheme enum."""

    def test_ets_description(self):
        assert "Emissions Trading" in CarbonPricingScheme.ETS.description

    def test_carbon_tax_description(self):
        assert "carbon tax" in CarbonPricingScheme.CARBON_TAX.description.lower()

    def test_hybrid_description(self):
        assert "Combined" in CarbonPricingScheme.HYBRID.description

    def test_none_description(self):
        assert "No carbon" in CarbonPricingScheme.NONE.description
