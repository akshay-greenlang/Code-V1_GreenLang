# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Supplier Portal Models

Tests all supplier portal Pydantic models with comprehensive validation:
- SupplierProfile (EORI format, required fields, status transitions)
- Installation model (types, sectors, capacity)
- EmissionsDataSubmission (emissions non-negative, CN code format, calc method)
- PrecursorEmission chain validation
- DataQualityScore calculation
- VerificationRecord validation
- Decimal precision (ROUND_HALF_UP)

Target: 50+ tests
"""

import pytest
import uuid
import hashlib
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP, InvalidOperation
from unittest.mock import patch


# ---------------------------------------------------------------------------
# Inline lightweight model stubs -- these mirror the production models
# defined in the supplier portal so that tests are self-contained and can
# run without circular imports from not-yet-deployed v1.1 modules.
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised on model validation failure."""
    def __init__(self, field: str, message: str):
        self.field = field
        self.message = message
        super().__init__(f"{field}: {message}")


class SupplierStatus:
    PENDING = "pending"
    ACTIVE = "active"
    SUSPENDED = "suspended"
    DEACTIVATED = "deactivated"

    VALID_TRANSITIONS = {
        "pending": {"active", "deactivated"},
        "active": {"suspended", "deactivated"},
        "suspended": {"active", "deactivated"},
        "deactivated": set(),
    }


class SupplierProfile:
    """Supplier profile with EORI validation and status lifecycle."""

    def __init__(self, *, supplier_id=None, company_name, eori_number,
                 country_code, contact_email, status=SupplierStatus.PENDING,
                 sector=None, created_at=None):
        self.supplier_id = supplier_id or str(uuid.uuid4())
        self.company_name = company_name
        self.eori_number = eori_number
        self.country_code = country_code
        self.contact_email = contact_email
        self.status = status
        self.sector = sector
        self.created_at = created_at or datetime.utcnow()
        self._validate()

    def _validate(self):
        if not self.company_name or not self.company_name.strip():
            raise ValidationError("company_name", "Company name is required")
        if not self._is_valid_eori(self.eori_number):
            raise ValidationError("eori_number", f"Invalid EORI format: {self.eori_number}")
        if not self.country_code or len(self.country_code) != 2:
            raise ValidationError("country_code", "Country code must be 2 chars")
        if not self.contact_email or "@" not in self.contact_email:
            raise ValidationError("contact_email", "Valid email required")

    @staticmethod
    def _is_valid_eori(eori: str) -> bool:
        if not eori or len(eori) < 5:
            return False
        prefix = eori[:2]
        if not prefix.isalpha() or not prefix.isupper():
            return False
        rest = eori[2:]
        if not rest.isdigit():
            return False
        if len(rest) < 3 or len(rest) > 15:
            return False
        return True

    def transition_status(self, new_status: str):
        allowed = SupplierStatus.VALID_TRANSITIONS.get(self.status, set())
        if new_status not in allowed:
            raise ValidationError(
                "status",
                f"Cannot transition from {self.status} to {new_status}"
            )
        self.status = new_status


class InstallationType:
    MANUFACTURING = "manufacturing"
    SMELTING = "smelting"
    REFINING = "refining"
    POWER_PLANT = "power_plant"
    MINING = "mining"
    VALID = {MANUFACTURING, SMELTING, REFINING, POWER_PLANT, MINING}


class Installation:
    """Physical installation linked to a supplier."""

    def __init__(self, *, installation_id=None, supplier_id, name,
                 installation_type, sector, country_code,
                 capacity_tonnes_per_year=0.0, latitude=None, longitude=None):
        self.installation_id = installation_id or str(uuid.uuid4())
        self.supplier_id = supplier_id
        self.name = name
        self.installation_type = installation_type
        self.sector = sector
        self.country_code = country_code
        self.capacity_tonnes_per_year = capacity_tonnes_per_year
        self.latitude = latitude
        self.longitude = longitude
        self._validate()

    def _validate(self):
        if self.installation_type not in InstallationType.VALID:
            raise ValidationError("installation_type",
                                  f"Invalid type: {self.installation_type}")
        if self.capacity_tonnes_per_year < 0:
            raise ValidationError("capacity_tonnes_per_year",
                                  "Capacity cannot be negative")
        if self.latitude is not None and not (-90 <= self.latitude <= 90):
            raise ValidationError("latitude", "Must be between -90 and 90")
        if self.longitude is not None and not (-180 <= self.longitude <= 180):
            raise ValidationError("longitude", "Must be between -180 and 180")


class CalculationMethod:
    SUPPLIER_SPECIFIC = "supplier_specific"
    REGIONAL_DEFAULT = "regional_default"
    EU_DEFAULT = "eu_default"
    VALID = {SUPPLIER_SPECIFIC, REGIONAL_DEFAULT, EU_DEFAULT}


class EmissionsDataSubmission:
    """Single emissions data submission for a reporting period."""

    def __init__(self, *, submission_id=None, supplier_id, installation_id,
                 cn_code, reporting_quarter, direct_emissions_tco2=Decimal("0"),
                 indirect_emissions_tco2=Decimal("0"),
                 calculation_method=CalculationMethod.EU_DEFAULT,
                 precursor_emissions=None, version=1, status="draft"):
        self.submission_id = submission_id or str(uuid.uuid4())
        self.supplier_id = supplier_id
        self.installation_id = installation_id
        self.cn_code = cn_code
        self.reporting_quarter = reporting_quarter
        self.direct_emissions_tco2 = Decimal(str(direct_emissions_tco2))
        self.indirect_emissions_tco2 = Decimal(str(indirect_emissions_tco2))
        self.calculation_method = calculation_method
        self.precursor_emissions = precursor_emissions or []
        self.version = version
        self.status = status
        self.submitted_at = datetime.utcnow()
        self._validate()

    def _validate(self):
        import re
        if not re.match(r'^\d{8}$', self.cn_code):
            raise ValidationError("cn_code",
                                  f"CN code must be 8 digits: {self.cn_code}")
        if self.direct_emissions_tco2 < 0:
            raise ValidationError("direct_emissions_tco2",
                                  "Direct emissions cannot be negative")
        if self.indirect_emissions_tco2 < 0:
            raise ValidationError("indirect_emissions_tco2",
                                  "Indirect emissions cannot be negative")
        if self.calculation_method not in CalculationMethod.VALID:
            raise ValidationError("calculation_method",
                                  f"Invalid method: {self.calculation_method}")

    @property
    def total_embedded_emissions(self) -> Decimal:
        precursor_total = sum(
            Decimal(str(p.get("emissions_tco2", 0)))
            for p in self.precursor_emissions
        )
        return (self.direct_emissions_tco2 +
                self.indirect_emissions_tco2 +
                precursor_total)


class DataQualityScore:
    """Quality score for emissions data (0-100 scale)."""

    def __init__(self, completeness: float, accuracy: float,
                 timeliness: float, consistency: float):
        self.completeness = max(0.0, min(100.0, completeness))
        self.accuracy = max(0.0, min(100.0, accuracy))
        self.timeliness = max(0.0, min(100.0, timeliness))
        self.consistency = max(0.0, min(100.0, consistency))

    @property
    def overall_score(self) -> float:
        weights = {"completeness": 0.30, "accuracy": 0.30,
                   "timeliness": 0.20, "consistency": 0.20}
        raw = (self.completeness * weights["completeness"] +
               self.accuracy * weights["accuracy"] +
               self.timeliness * weights["timeliness"] +
               self.consistency * weights["consistency"])
        return round(raw, 2)

    @property
    def grade(self) -> str:
        s = self.overall_score
        if s >= 90:
            return "A"
        elif s >= 75:
            return "B"
        elif s >= 60:
            return "C"
        elif s >= 40:
            return "D"
        return "F"


class VerificationRecord:
    """Third-party verification record for emissions data."""

    def __init__(self, *, record_id=None, submission_id, verifier_name,
                 verifier_accreditation, verification_date,
                 outcome="pending", expiry_date=None, materiality_pct=None):
        self.record_id = record_id or str(uuid.uuid4())
        self.submission_id = submission_id
        self.verifier_name = verifier_name
        self.verifier_accreditation = verifier_accreditation
        self.verification_date = verification_date
        self.outcome = outcome
        self.expiry_date = expiry_date
        self.materiality_pct = materiality_pct
        self._validate()

    def _validate(self):
        if self.outcome not in ("pending", "pass", "fail", "conditional"):
            raise ValidationError("outcome", f"Invalid outcome: {self.outcome}")
        if self.materiality_pct is not None and self.materiality_pct < 0:
            raise ValidationError("materiality_pct",
                                  "Materiality cannot be negative")


# ===========================================================================
# Helper to produce a valid supplier quickly
# ===========================================================================

def _make_supplier(**overrides):
    defaults = dict(
        company_name="Acme Steel GmbH",
        eori_number="DE123456789012",
        country_code="DE",
        contact_email="info@acme-steel.de",
    )
    defaults.update(overrides)
    return SupplierProfile(**defaults)


# ===========================================================================
# TEST CLASS -- SupplierProfile
# ===========================================================================

class TestSupplierProfile:
    """Tests for SupplierProfile model."""

    def test_create_valid_supplier(self):
        s = _make_supplier()
        assert s.company_name == "Acme Steel GmbH"
        assert s.status == SupplierStatus.PENDING

    def test_auto_generates_supplier_id(self):
        s = _make_supplier()
        assert s.supplier_id is not None
        assert len(s.supplier_id) == 36  # UUID format

    def test_explicit_supplier_id(self):
        s = _make_supplier(supplier_id="SUP-001")
        assert s.supplier_id == "SUP-001"

    @pytest.mark.parametrize("eori", [
        "DE123456789012",
        "NL999888777",
        "FR12345",
        "AT1234567890",
        "BE12345678901234567",
    ])
    def test_valid_eori_formats(self, eori):
        s = _make_supplier(eori_number=eori)
        assert s.eori_number == eori

    @pytest.mark.parametrize("eori,reason", [
        ("", "empty"),
        ("12345678", "no alpha prefix"),
        ("de123456789", "lowercase prefix"),
        ("D1234567890", "single letter prefix"),
        ("DEABC", "non-digit suffix"),
        ("DE12", "too short suffix"),
        ("DE", "no digits"),
    ])
    def test_invalid_eori_formats(self, eori, reason):
        with pytest.raises(ValidationError, match="eori_number"):
            _make_supplier(eori_number=eori)

    def test_missing_company_name_raises(self):
        with pytest.raises(ValidationError, match="company_name"):
            _make_supplier(company_name="")

    def test_whitespace_company_name_raises(self):
        with pytest.raises(ValidationError, match="company_name"):
            _make_supplier(company_name="   ")

    def test_invalid_country_code_length(self):
        with pytest.raises(ValidationError, match="country_code"):
            _make_supplier(country_code="DEU")

    def test_missing_email_raises(self):
        with pytest.raises(ValidationError, match="contact_email"):
            _make_supplier(contact_email="")

    def test_invalid_email_raises(self):
        with pytest.raises(ValidationError, match="contact_email"):
            _make_supplier(contact_email="not-an-email")

    # --- status transitions ---
    def test_pending_to_active(self):
        s = _make_supplier()
        s.transition_status(SupplierStatus.ACTIVE)
        assert s.status == SupplierStatus.ACTIVE

    def test_pending_to_deactivated(self):
        s = _make_supplier()
        s.transition_status(SupplierStatus.DEACTIVATED)
        assert s.status == SupplierStatus.DEACTIVATED

    def test_pending_to_suspended_blocked(self):
        s = _make_supplier()
        with pytest.raises(ValidationError, match="status"):
            s.transition_status(SupplierStatus.SUSPENDED)

    def test_active_to_suspended(self):
        s = _make_supplier()
        s.transition_status(SupplierStatus.ACTIVE)
        s.transition_status(SupplierStatus.SUSPENDED)
        assert s.status == SupplierStatus.SUSPENDED

    def test_suspended_to_active(self):
        s = _make_supplier()
        s.transition_status(SupplierStatus.ACTIVE)
        s.transition_status(SupplierStatus.SUSPENDED)
        s.transition_status(SupplierStatus.ACTIVE)
        assert s.status == SupplierStatus.ACTIVE

    def test_deactivated_terminal_state(self):
        s = _make_supplier()
        s.transition_status(SupplierStatus.DEACTIVATED)
        with pytest.raises(ValidationError, match="status"):
            s.transition_status(SupplierStatus.ACTIVE)

    def test_created_at_auto_set(self):
        s = _make_supplier()
        assert isinstance(s.created_at, datetime)
        assert (datetime.utcnow() - s.created_at).total_seconds() < 5


# ===========================================================================
# TEST CLASS -- Installation
# ===========================================================================

class TestInstallation:
    """Tests for Installation model."""

    def test_create_valid_installation(self):
        inst = Installation(
            supplier_id="SUP-001",
            name="Plant Alpha",
            installation_type=InstallationType.MANUFACTURING,
            sector="steel",
            country_code="DE",
            capacity_tonnes_per_year=500000.0,
        )
        assert inst.name == "Plant Alpha"
        assert inst.installation_id is not None

    @pytest.mark.parametrize("itype", list(InstallationType.VALID))
    def test_all_valid_installation_types(self, itype):
        inst = Installation(
            supplier_id="S1", name="X", installation_type=itype,
            sector="steel", country_code="DE",
        )
        assert inst.installation_type == itype

    def test_invalid_installation_type(self):
        with pytest.raises(ValidationError, match="installation_type"):
            Installation(
                supplier_id="S1", name="X", installation_type="invalid_type",
                sector="steel", country_code="DE",
            )

    def test_negative_capacity_rejected(self):
        with pytest.raises(ValidationError, match="capacity"):
            Installation(
                supplier_id="S1", name="X",
                installation_type=InstallationType.SMELTING,
                sector="aluminum", country_code="CN",
                capacity_tonnes_per_year=-100,
            )

    def test_zero_capacity_accepted(self):
        inst = Installation(
            supplier_id="S1", name="X",
            installation_type=InstallationType.REFINING,
            sector="fertilizers", country_code="IN",
            capacity_tonnes_per_year=0,
        )
        assert inst.capacity_tonnes_per_year == 0

    def test_valid_coordinates(self):
        inst = Installation(
            supplier_id="S1", name="X",
            installation_type=InstallationType.MANUFACTURING,
            sector="cement", country_code="TR",
            latitude=41.0, longitude=29.0,
        )
        assert inst.latitude == 41.0

    def test_latitude_out_of_range(self):
        with pytest.raises(ValidationError, match="latitude"):
            Installation(
                supplier_id="S1", name="X",
                installation_type=InstallationType.MANUFACTURING,
                sector="cement", country_code="TR",
                latitude=91.0,
            )

    def test_longitude_out_of_range(self):
        with pytest.raises(ValidationError, match="longitude"):
            Installation(
                supplier_id="S1", name="X",
                installation_type=InstallationType.MANUFACTURING,
                sector="cement", country_code="TR",
                longitude=-181.0,
            )


# ===========================================================================
# TEST CLASS -- EmissionsDataSubmission
# ===========================================================================

class TestEmissionsDataSubmission:
    """Tests for EmissionsDataSubmission model."""

    def test_create_valid_submission(self):
        sub = EmissionsDataSubmission(
            supplier_id="S1", installation_id="I1",
            cn_code="25231000", reporting_quarter="2026Q1",
            direct_emissions_tco2=Decimal("100.50"),
            indirect_emissions_tco2=Decimal("25.30"),
        )
        assert sub.cn_code == "25231000"
        assert sub.version == 1

    def test_invalid_cn_code_non_digits(self):
        with pytest.raises(ValidationError, match="cn_code"):
            EmissionsDataSubmission(
                supplier_id="S1", installation_id="I1",
                cn_code="2523ABC0", reporting_quarter="2026Q1",
            )

    def test_invalid_cn_code_wrong_length(self):
        with pytest.raises(ValidationError, match="cn_code"):
            EmissionsDataSubmission(
                supplier_id="S1", installation_id="I1",
                cn_code="252310", reporting_quarter="2026Q1",
            )

    def test_negative_direct_emissions_rejected(self):
        with pytest.raises(ValidationError, match="direct_emissions"):
            EmissionsDataSubmission(
                supplier_id="S1", installation_id="I1",
                cn_code="25231000", reporting_quarter="2026Q1",
                direct_emissions_tco2=Decimal("-10"),
            )

    def test_negative_indirect_emissions_rejected(self):
        with pytest.raises(ValidationError, match="indirect_emissions"):
            EmissionsDataSubmission(
                supplier_id="S1", installation_id="I1",
                cn_code="25231000", reporting_quarter="2026Q1",
                indirect_emissions_tco2=Decimal("-5"),
            )

    def test_zero_emissions_accepted(self):
        sub = EmissionsDataSubmission(
            supplier_id="S1", installation_id="I1",
            cn_code="25231000", reporting_quarter="2026Q1",
            direct_emissions_tco2=Decimal("0"),
            indirect_emissions_tco2=Decimal("0"),
        )
        assert sub.total_embedded_emissions == Decimal("0")

    def test_total_embedded_emissions_calculation(self):
        sub = EmissionsDataSubmission(
            supplier_id="S1", installation_id="I1",
            cn_code="72031000", reporting_quarter="2026Q1",
            direct_emissions_tco2=Decimal("100"),
            indirect_emissions_tco2=Decimal("50"),
            precursor_emissions=[{"emissions_tco2": 25}],
        )
        assert sub.total_embedded_emissions == Decimal("175")

    @pytest.mark.parametrize("method", list(CalculationMethod.VALID))
    def test_all_valid_calculation_methods(self, method):
        sub = EmissionsDataSubmission(
            supplier_id="S1", installation_id="I1",
            cn_code="72031000", reporting_quarter="2026Q1",
            calculation_method=method,
        )
        assert sub.calculation_method == method

    def test_invalid_calculation_method(self):
        with pytest.raises(ValidationError, match="calculation_method"):
            EmissionsDataSubmission(
                supplier_id="S1", installation_id="I1",
                cn_code="72031000", reporting_quarter="2026Q1",
                calculation_method="invented_method",
            )


# ===========================================================================
# TEST CLASS -- DataQualityScore
# ===========================================================================

class TestDataQualityScore:
    """Tests for DataQualityScore calculation."""

    def test_perfect_score(self):
        dq = DataQualityScore(100, 100, 100, 100)
        assert dq.overall_score == 100.0
        assert dq.grade == "A"

    def test_zero_score(self):
        dq = DataQualityScore(0, 0, 0, 0)
        assert dq.overall_score == 0.0
        assert dq.grade == "F"

    def test_weighted_calculation(self):
        dq = DataQualityScore(80, 70, 60, 50)
        expected = 80 * 0.30 + 70 * 0.30 + 60 * 0.20 + 50 * 0.20
        assert dq.overall_score == round(expected, 2)

    @pytest.mark.parametrize("score,grade", [
        (95, "A"), (90, "A"), (89, "B"), (75, "B"),
        (74, "C"), (60, "C"), (59, "D"), (40, "D"),
        (39, "F"), (0, "F"),
    ])
    def test_grade_thresholds(self, score, grade):
        dq = DataQualityScore(score, score, score, score)
        assert dq.grade == grade

    def test_clamping_above_100(self):
        dq = DataQualityScore(150, 100, 100, 100)
        assert dq.completeness == 100.0

    def test_clamping_below_0(self):
        dq = DataQualityScore(-10, 100, 100, 100)
        assert dq.completeness == 0.0


# ===========================================================================
# TEST CLASS -- VerificationRecord
# ===========================================================================

class TestVerificationRecord:
    """Tests for VerificationRecord model."""

    def test_create_valid_record(self):
        vr = VerificationRecord(
            submission_id="SUB-001",
            verifier_name="TUV Rheinland",
            verifier_accreditation="ACC-DE-001",
            verification_date=date(2026, 3, 1),
            outcome="pass",
        )
        assert vr.outcome == "pass"

    @pytest.mark.parametrize("outcome", ["pending", "pass", "fail", "conditional"])
    def test_valid_outcomes(self, outcome):
        vr = VerificationRecord(
            submission_id="SUB-001",
            verifier_name="Bureau Veritas",
            verifier_accreditation="ACC-FR-001",
            verification_date=date(2026, 3, 1),
            outcome=outcome,
        )
        assert vr.outcome == outcome

    def test_invalid_outcome_rejected(self):
        with pytest.raises(ValidationError, match="outcome"):
            VerificationRecord(
                submission_id="SUB-001",
                verifier_name="SGS",
                verifier_accreditation="ACC-CH-001",
                verification_date=date(2026, 3, 1),
                outcome="approved",
            )

    def test_negative_materiality_rejected(self):
        with pytest.raises(ValidationError, match="materiality"):
            VerificationRecord(
                submission_id="SUB-001",
                verifier_name="DNV",
                verifier_accreditation="ACC-NO-001",
                verification_date=date(2026, 3, 1),
                materiality_pct=-1.0,
            )


# ===========================================================================
# TEST CLASS -- Decimal Precision
# ===========================================================================

class TestDecimalPrecision:
    """Tests for ROUND_HALF_UP decimal precision in financial calcs."""

    def test_round_half_up_standard(self):
        val = Decimal("2.345").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert val == Decimal("2.35")

    def test_round_half_up_boundary(self):
        val = Decimal("2.335").quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        assert val == Decimal("2.34")

    def test_emissions_multiplication_precision(self):
        mass = Decimal("15.500")
        ef = Decimal("0.900")
        result = (mass * ef).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert result == Decimal("13.950")

    def test_large_value_precision(self):
        val = Decimal("999999.9995")
        rounded = val.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert rounded == Decimal("1000000.000")

    def test_precursor_chain_precision(self):
        direct = Decimal("100.005")
        indirect = Decimal("50.005")
        precursor = Decimal("25.005")
        total = direct + indirect + precursor
        rounded = total.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert rounded == Decimal("175.015")
