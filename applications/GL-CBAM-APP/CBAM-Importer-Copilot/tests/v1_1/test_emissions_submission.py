# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Emissions Data Submission

Tests emissions data submission:
- submit_emissions (valid, invalid CN code, negative emissions)
- validate_emissions_data (completeness, consistency)
- calculate_total_embedded_emissions (direct + indirect + precursors)
- amend_submission (version increment, within window, outside window)
- review_submission (accept, reject)
- calculate_data_quality_score (completeness, timeliness)
- Export formats (CSV, JSON, XML)

Target: 50+ tests
"""

import pytest
import json
import csv
import io
import uuid
from datetime import datetime, date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from copy import deepcopy
from xml.etree import ElementTree as ET


# ---------------------------------------------------------------------------
# Inline submission service for self-contained tests
# ---------------------------------------------------------------------------

class SubmissionError(Exception):
    pass


class AmendmentWindowError(SubmissionError):
    pass


class EmissionsSubmissionService:
    """Service for managing emissions data submissions."""

    AMENDMENT_WINDOW_DAYS = 60

    def __init__(self):
        self._submissions = {}
        self._reviews = {}

    def submit_emissions(self, *, supplier_id, installation_id, cn_code,
                         reporting_quarter, direct_emissions_tco2,
                         indirect_emissions_tco2,
                         precursor_emissions=None,
                         calculation_method="eu_default"):
        import re
        if not re.match(r'^\d{8}$', cn_code):
            raise SubmissionError(f"Invalid CN code format: {cn_code}")
        direct = Decimal(str(direct_emissions_tco2))
        indirect = Decimal(str(indirect_emissions_tco2))
        if direct < 0:
            raise SubmissionError("Direct emissions cannot be negative")
        if indirect < 0:
            raise SubmissionError("Indirect emissions cannot be negative")
        valid_methods = {"supplier_specific", "regional_default", "eu_default"}
        if calculation_method not in valid_methods:
            raise SubmissionError(f"Invalid calc method: {calculation_method}")

        sub_id = f"SUB-{uuid.uuid4().hex[:8].upper()}"
        submission = {
            "submission_id": sub_id,
            "supplier_id": supplier_id,
            "installation_id": installation_id,
            "cn_code": cn_code,
            "reporting_quarter": reporting_quarter,
            "direct_emissions_tco2": str(direct),
            "indirect_emissions_tco2": str(indirect),
            "precursor_emissions": precursor_emissions or [],
            "calculation_method": calculation_method,
            "version": 1,
            "status": "draft",
            "submitted_at": datetime.utcnow().isoformat(),
            "quarter_end_date": self._get_quarter_end(reporting_quarter).isoformat(),
        }
        self._submissions[sub_id] = submission
        return submission

    def _get_quarter_end(self, quarter: str) -> date:
        year = int(quarter[:4])
        q = int(quarter[5])
        ends = {1: date(year, 3, 31), 2: date(year, 6, 30),
                3: date(year, 9, 30), 4: date(year, 12, 31)}
        return ends[q]

    def get_submission(self, submission_id: str) -> dict:
        if submission_id not in self._submissions:
            raise SubmissionError(f"Submission not found: {submission_id}")
        return deepcopy(self._submissions[submission_id])

    def validate_emissions_data(self, submission_id: str) -> dict:
        sub = self.get_submission(submission_id)
        issues = []
        direct = Decimal(sub["direct_emissions_tco2"])
        indirect = Decimal(sub["indirect_emissions_tco2"])

        if direct == 0 and indirect == 0 and not sub["precursor_emissions"]:
            issues.append({"field": "emissions", "issue": "All emissions are zero"})
        if not sub["cn_code"]:
            issues.append({"field": "cn_code", "issue": "Missing CN code"})
        if not sub["reporting_quarter"]:
            issues.append({"field": "reporting_quarter", "issue": "Missing quarter"})

        total = self.calculate_total_embedded_emissions(submission_id)
        if total > Decimal("1000000"):
            issues.append({"field": "total", "issue": "Unusually high emissions"})

        return {
            "submission_id": submission_id,
            "is_valid": len(issues) == 0,
            "issues": issues,
            "completeness_score": self._calc_completeness(sub),
        }

    def _calc_completeness(self, sub: dict) -> float:
        required = ["supplier_id", "installation_id", "cn_code",
                     "reporting_quarter", "direct_emissions_tco2",
                     "indirect_emissions_tco2", "calculation_method"]
        filled = sum(1 for f in required if sub.get(f))
        return round(filled / len(required) * 100, 1)

    def calculate_total_embedded_emissions(self, submission_id: str) -> Decimal:
        sub = self.get_submission(submission_id)
        direct = Decimal(sub["direct_emissions_tco2"])
        indirect = Decimal(sub["indirect_emissions_tco2"])
        precursor_total = sum(
            Decimal(str(p.get("emissions_tco2", 0)))
            for p in sub.get("precursor_emissions", [])
        )
        return (direct + indirect + precursor_total).quantize(
            Decimal("0.001"), rounding=ROUND_HALF_UP
        )

    def amend_submission(self, submission_id: str, **changes) -> dict:
        sub = self._submissions.get(submission_id)
        if not sub:
            raise SubmissionError(f"Submission not found: {submission_id}")

        quarter_end = date.fromisoformat(sub["quarter_end_date"])
        deadline = quarter_end + timedelta(days=self.AMENDMENT_WINDOW_DAYS)
        if date.today() > deadline:
            raise AmendmentWindowError(
                f"Amendment window expired on {deadline.isoformat()}"
            )

        new_version = sub["version"] + 1
        amended = deepcopy(sub)
        amended.update(changes)
        amended["version"] = new_version
        amended["status"] = "amended"
        amended["amended_at"] = datetime.utcnow().isoformat()
        self._submissions[submission_id] = amended
        return amended

    def review_submission(self, submission_id: str, decision: str,
                          reviewer: str, comments: str = "") -> dict:
        if decision not in ("accepted", "rejected"):
            raise SubmissionError(f"Invalid decision: {decision}")
        sub = self._submissions.get(submission_id)
        if not sub:
            raise SubmissionError(f"Submission not found: {submission_id}")

        sub["status"] = decision
        review = {
            "submission_id": submission_id,
            "decision": decision,
            "reviewer": reviewer,
            "comments": comments,
            "reviewed_at": datetime.utcnow().isoformat(),
        }
        self._reviews[submission_id] = review
        return review

    def calculate_data_quality_score(self, submission_id: str) -> dict:
        sub = self.get_submission(submission_id)
        completeness = self._calc_completeness(sub)
        method_scores = {"supplier_specific": 100, "regional_default": 60,
                         "eu_default": 30}
        accuracy = method_scores.get(sub["calculation_method"], 0)
        submitted = datetime.fromisoformat(sub["submitted_at"])
        quarter_end = date.fromisoformat(sub["quarter_end_date"])
        days_after = (submitted.date() - quarter_end).days
        timeliness = max(0, 100 - days_after * 2) if days_after > 0 else 100.0
        consistency = 80.0
        overall = (completeness * 0.3 + accuracy * 0.3 +
                   timeliness * 0.2 + consistency * 0.2)
        return {
            "completeness": completeness,
            "accuracy": accuracy,
            "timeliness": round(timeliness, 1),
            "consistency": consistency,
            "overall": round(overall, 2),
        }

    def export_csv(self, submission_ids: list) -> str:
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "submission_id", "supplier_id", "cn_code",
            "reporting_quarter", "direct_emissions_tco2",
            "indirect_emissions_tco2", "calculation_method",
        ])
        writer.writeheader()
        for sid in submission_ids:
            sub = self.get_submission(sid)
            writer.writerow({k: sub[k] for k in writer.fieldnames})
        return output.getvalue()

    def export_json(self, submission_ids: list) -> str:
        data = [self.get_submission(sid) for sid in submission_ids]
        return json.dumps(data, indent=2)

    def export_xml(self, submission_ids: list) -> str:
        root = ET.Element("EmissionsSubmissions")
        for sid in submission_ids:
            sub = self.get_submission(sid)
            elem = ET.SubElement(root, "Submission")
            for k, v in sub.items():
                if isinstance(v, (list, dict)):
                    continue
                child = ET.SubElement(elem, k)
                child.text = str(v)
        return ET.tostring(root, encoding="unicode")


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def service():
    return EmissionsSubmissionService()


@pytest.fixture
def valid_submission(service):
    return service.submit_emissions(
        supplier_id="SUP-001",
        installation_id="INST-001",
        cn_code="25231000",
        reporting_quarter="2026Q1",
        direct_emissions_tco2=100.50,
        indirect_emissions_tco2=25.30,
    )


# ===========================================================================
# TEST CLASS -- submit_emissions
# ===========================================================================

class TestSubmitEmissions:
    """Tests for submit_emissions."""

    def test_valid_submission(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=200,
            indirect_emissions_tco2=50,
        )
        assert sub["submission_id"].startswith("SUB-")
        assert sub["status"] == "draft"
        assert sub["version"] == 1

    def test_invalid_cn_code(self, service):
        with pytest.raises(SubmissionError, match="CN code"):
            service.submit_emissions(
                supplier_id="S1", installation_id="I1", cn_code="ABC",
                reporting_quarter="2026Q1", direct_emissions_tco2=10,
                indirect_emissions_tco2=5,
            )

    def test_negative_direct_emissions(self, service):
        with pytest.raises(SubmissionError, match="negative"):
            service.submit_emissions(
                supplier_id="S1", installation_id="I1", cn_code="72031000",
                reporting_quarter="2026Q1", direct_emissions_tco2=-10,
                indirect_emissions_tco2=5,
            )

    def test_negative_indirect_emissions(self, service):
        with pytest.raises(SubmissionError, match="negative"):
            service.submit_emissions(
                supplier_id="S1", installation_id="I1", cn_code="72031000",
                reporting_quarter="2026Q1", direct_emissions_tco2=10,
                indirect_emissions_tco2=-5,
            )

    def test_zero_emissions_accepted(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=0,
            indirect_emissions_tco2=0,
        )
        assert sub is not None

    def test_with_precursor_emissions(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=100,
            indirect_emissions_tco2=50,
            precursor_emissions=[{"material": "coke", "emissions_tco2": 25}],
        )
        assert len(sub["precursor_emissions"]) == 1

    def test_invalid_calculation_method(self, service):
        with pytest.raises(SubmissionError, match="calc method"):
            service.submit_emissions(
                supplier_id="S1", installation_id="I1", cn_code="72031000",
                reporting_quarter="2026Q1", direct_emissions_tco2=10,
                indirect_emissions_tco2=5, calculation_method="magic",
            )

    @pytest.mark.parametrize("method", [
        "supplier_specific", "regional_default", "eu_default",
    ])
    def test_valid_calculation_methods(self, service, method):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=10,
            indirect_emissions_tco2=5, calculation_method=method,
        )
        assert sub["calculation_method"] == method


# ===========================================================================
# TEST CLASS -- validate_emissions_data
# ===========================================================================

class TestValidateEmissionsData:
    """Tests for validate_emissions_data."""

    def test_valid_data_passes(self, service, valid_submission):
        result = service.validate_emissions_data(valid_submission["submission_id"])
        assert result["is_valid"] is True

    def test_zero_emissions_flagged(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=0,
            indirect_emissions_tco2=0,
        )
        result = service.validate_emissions_data(sub["submission_id"])
        assert any("zero" in i["issue"].lower() for i in result["issues"])

    def test_completeness_score_full(self, service, valid_submission):
        result = service.validate_emissions_data(valid_submission["submission_id"])
        assert result["completeness_score"] == 100.0

    def test_nonexistent_submission_raises(self, service):
        with pytest.raises(SubmissionError):
            service.validate_emissions_data("NONEXISTENT")


# ===========================================================================
# TEST CLASS -- calculate_total_embedded_emissions
# ===========================================================================

class TestCalculateTotalEmbeddedEmissions:
    """Tests for calculate_total_embedded_emissions."""

    def test_direct_plus_indirect(self, service, valid_submission):
        total = service.calculate_total_embedded_emissions(
            valid_submission["submission_id"]
        )
        assert total == Decimal("125.800")

    def test_with_precursors(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=100,
            indirect_emissions_tco2=50,
            precursor_emissions=[
                {"material": "iron_ore", "emissions_tco2": 30},
                {"material": "coke", "emissions_tco2": 20},
            ],
        )
        total = service.calculate_total_embedded_emissions(sub["submission_id"])
        assert total == Decimal("200.000")

    def test_precision_three_decimals(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=33.3333,
            indirect_emissions_tco2=16.6667,
        )
        total = service.calculate_total_embedded_emissions(sub["submission_id"])
        assert total == Decimal("50.000")

    def test_zero_total(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=0,
            indirect_emissions_tco2=0,
        )
        total = service.calculate_total_embedded_emissions(sub["submission_id"])
        assert total == Decimal("0.000")


# ===========================================================================
# TEST CLASS -- amend_submission
# ===========================================================================

class TestAmendSubmission:
    """Tests for amend_submission."""

    def test_amend_increments_version(self, service, valid_submission):
        amended = service.amend_submission(
            valid_submission["submission_id"],
            direct_emissions_tco2="110.00",
        )
        assert amended["version"] == 2
        assert amended["status"] == "amended"

    def test_amend_updates_field(self, service, valid_submission):
        amended = service.amend_submission(
            valid_submission["submission_id"],
            indirect_emissions_tco2="30.00",
        )
        assert amended["indirect_emissions_tco2"] == "30.00"

    def test_amend_nonexistent_raises(self, service):
        with pytest.raises(SubmissionError):
            service.amend_submission("NONEXISTENT", direct_emissions_tco2="10")

    def test_amend_outside_window_raises(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2024Q1",
            direct_emissions_tco2=100, indirect_emissions_tco2=50,
        )
        with pytest.raises(AmendmentWindowError):
            service.amend_submission(sub["submission_id"],
                                    direct_emissions_tco2="110")

    def test_multiple_amendments_increment(self, service, valid_submission):
        sid = valid_submission["submission_id"]
        service.amend_submission(sid, direct_emissions_tco2="110")
        amended = service.amend_submission(sid, direct_emissions_tco2="120")
        assert amended["version"] == 3


# ===========================================================================
# TEST CLASS -- review_submission
# ===========================================================================

class TestReviewSubmission:
    """Tests for review_submission."""

    def test_accept_submission(self, service, valid_submission):
        review = service.review_submission(
            valid_submission["submission_id"], "accepted", "reviewer@eu.com",
        )
        assert review["decision"] == "accepted"

    def test_reject_submission(self, service, valid_submission):
        review = service.review_submission(
            valid_submission["submission_id"], "rejected", "reviewer@eu.com",
            comments="Missing supplier-specific data",
        )
        assert review["decision"] == "rejected"

    def test_invalid_decision_raises(self, service, valid_submission):
        with pytest.raises(SubmissionError, match="Invalid decision"):
            service.review_submission(
                valid_submission["submission_id"], "maybe", "r@eu.com",
            )

    def test_review_updates_status(self, service, valid_submission):
        sid = valid_submission["submission_id"]
        service.review_submission(sid, "accepted", "r@eu.com")
        sub = service.get_submission(sid)
        assert sub["status"] == "accepted"

    def test_review_nonexistent_raises(self, service):
        with pytest.raises(SubmissionError):
            service.review_submission("NONE", "accepted", "r@eu.com")


# ===========================================================================
# TEST CLASS -- calculate_data_quality_score
# ===========================================================================

class TestDataQualityScore:
    """Tests for calculate_data_quality_score."""

    def test_supplier_specific_high_accuracy(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=100,
            indirect_emissions_tco2=50,
            calculation_method="supplier_specific",
        )
        score = service.calculate_data_quality_score(sub["submission_id"])
        assert score["accuracy"] == 100

    def test_eu_default_low_accuracy(self, service):
        sub = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=100,
            indirect_emissions_tco2=50,
            calculation_method="eu_default",
        )
        score = service.calculate_data_quality_score(sub["submission_id"])
        assert score["accuracy"] == 30

    def test_overall_score_range(self, service, valid_submission):
        score = service.calculate_data_quality_score(
            valid_submission["submission_id"]
        )
        assert 0 <= score["overall"] <= 100


# ===========================================================================
# TEST CLASS -- Export formats
# ===========================================================================

class TestExportFormats:
    """Tests for CSV, JSON, XML export."""

    def test_export_csv(self, service, valid_submission):
        csv_str = service.export_csv([valid_submission["submission_id"]])
        assert "submission_id" in csv_str
        assert valid_submission["submission_id"] in csv_str

    def test_export_json(self, service, valid_submission):
        json_str = service.export_json([valid_submission["submission_id"]])
        data = json.loads(json_str)
        assert len(data) == 1
        assert data[0]["submission_id"] == valid_submission["submission_id"]

    def test_export_xml(self, service, valid_submission):
        xml_str = service.export_xml([valid_submission["submission_id"]])
        root = ET.fromstring(xml_str)
        assert root.tag == "EmissionsSubmissions"
        assert len(root.findall("Submission")) == 1

    def test_export_multiple(self, service):
        s1 = service.submit_emissions(
            supplier_id="S1", installation_id="I1", cn_code="72031000",
            reporting_quarter="2026Q1", direct_emissions_tco2=100,
            indirect_emissions_tco2=50,
        )
        s2 = service.submit_emissions(
            supplier_id="S2", installation_id="I2", cn_code="25231000",
            reporting_quarter="2026Q1", direct_emissions_tco2=80,
            indirect_emissions_tco2=20,
        )
        csv_str = service.export_csv([s1["submission_id"], s2["submission_id"]])
        lines = csv_str.strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
