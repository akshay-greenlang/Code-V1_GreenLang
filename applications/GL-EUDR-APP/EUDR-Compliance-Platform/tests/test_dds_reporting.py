"""
Unit tests for GL-EUDR-APP v1.0 DDS (Due Diligence Statement) Reporting Engine.

Tests DDS generation, validation, submission lifecycle, bulk operations,
download formats, amendments, annual summaries, and reference number
generation.

Test count target: 40+ tests
"""

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# DDS Reporting Engine (self-contained for testing)
# ---------------------------------------------------------------------------

DDS_STATUSES = {"draft", "review", "validated", "submitted", "accepted", "rejected", "amended"}
DDS_SECTIONS = [
    "operator_info", "product_description", "country_of_production",
    "geolocation_data", "risk_assessment", "risk_mitigation", "conclusion",
]


class DDSError(Exception):
    pass


class DDSNotFoundError(DDSError):
    pass


class DDSValidationError(DDSError):
    pass


class DDSReportingEngine:
    """Engine for DDS lifecycle management."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}
        self._sequences: Dict[str, int] = {}  # (country_iso3, year) -> counter

    def _next_reference(self, country_iso3: str, year: int) -> str:
        key = f"{country_iso3.upper()}_{year}"
        seq = self._sequences.get(key, 0) + 1
        self._sequences[key] = seq
        return f"EUDR-{country_iso3.upper()}-{year}-{seq:06d}"

    def generate_dds(self, supplier_id: str, commodity: str,
                     country_iso3: str, year: int,
                     plot_ids: Optional[List[str]] = None,
                     operator_info: Optional[Dict] = None,
                     product_description: Optional[Dict] = None,
                     country_of_production: Optional[Dict] = None,
                     geolocation_data: Optional[Dict] = None,
                     risk_assessment: Optional[Dict] = None,
                     risk_mitigation: Optional[Dict] = None,
                     conclusion: Optional[Dict] = None) -> Dict[str, Any]:
        """Generate a new DDS in draft status."""
        ref = self._next_reference(country_iso3, year)
        dds_id = f"dds_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc)

        dds = {
            "dds_id": dds_id,
            "reference_number": ref,
            "supplier_id": supplier_id,
            "commodity": commodity,
            "year": year,
            "country_iso3": country_iso3.upper(),
            "status": "draft",
            "operator_info": operator_info or {},
            "product_description": product_description or {},
            "country_of_production": country_of_production or {},
            "geolocation_data": geolocation_data or {},
            "risk_assessment": risk_assessment or {},
            "risk_mitigation": risk_mitigation or {},
            "conclusion": conclusion or {},
            "plot_ids": plot_ids or [],
            "document_ids": [],
            "overall_risk_score": None,
            "validation_result": None,
            "submission_date": None,
            "eu_reference": None,
            "eu_response": None,
            "amendment_of": None,
            "created_at": now,
            "updated_at": now,
        }
        self._store[dds_id] = dds
        return dds

    def get_dds(self, dds_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(dds_id)

    def list_dds(self, supplier_id: Optional[str] = None,
                 status: Optional[str] = None,
                 year: Optional[int] = None,
                 commodity: Optional[str] = None) -> List[Dict[str, Any]]:
        results = list(self._store.values())
        if supplier_id:
            results = [d for d in results if d["supplier_id"] == supplier_id]
        if status:
            results = [d for d in results if d["status"] == status]
        if year:
            results = [d for d in results if d["year"] == year]
        if commodity:
            results = [d for d in results if d["commodity"] == commodity]
        return results

    def validate_dds(self, dds_id: str) -> Dict[str, Any]:
        """Validate DDS completeness and correctness."""
        dds = self._store.get(dds_id)
        if not dds:
            raise DDSNotFoundError(f"DDS '{dds_id}' not found")

        section_results = {}
        for section in DDS_SECTIONS:
            data = dds.get(section, {})
            is_complete = bool(data) and len(data) > 0
            section_results[section] = {
                "complete": is_complete,
                "issues": [] if is_complete else [f"Section '{section}' is empty"],
            }

        all_valid = all(s["complete"] for s in section_results.values())
        failed_sections = [k for k, v in section_results.items() if not v["complete"]]

        if all_valid:
            dds["status"] = "validated"
        else:
            dds["status"] = "review"

        dds["validation_result"] = {
            "valid": all_valid,
            "sections": section_results,
            "failed_sections": failed_sections,
        }
        dds["updated_at"] = datetime.now(timezone.utc)

        return dds["validation_result"]

    def submit_dds(self, dds_id: str) -> Dict[str, Any]:
        """Submit a validated DDS to the EU system."""
        dds = self._store.get(dds_id)
        if not dds:
            raise DDSNotFoundError(f"DDS '{dds_id}' not found")
        if dds["status"] not in ("validated",):
            raise DDSValidationError(
                f"Cannot submit DDS in '{dds['status']}' status. Must be 'validated'."
            )

        dds["status"] = "submitted"
        dds["submission_date"] = datetime.now(timezone.utc)
        dds["eu_reference"] = f"EU-{uuid.uuid4().hex[:8].upper()}"
        dds["eu_response"] = {"status": "received", "message": "DDS received for processing."}
        dds["updated_at"] = datetime.now(timezone.utc)

        return {
            "dds_id": dds_id,
            "status": "submitted",
            "submission_date": dds["submission_date"],
            "eu_reference": dds["eu_reference"],
            "eu_response": dds["eu_response"],
        }

    def update_status(self, dds_id: str, new_status: str) -> Dict[str, Any]:
        """Update DDS status (for lifecycle transitions like accepted/rejected)."""
        dds = self._store.get(dds_id)
        if not dds:
            raise DDSNotFoundError(f"DDS '{dds_id}' not found")
        if new_status not in DDS_STATUSES:
            raise DDSValidationError(f"Invalid status '{new_status}'")
        dds["status"] = new_status
        dds["updated_at"] = datetime.now(timezone.utc)
        return dds

    def bulk_generate(self, requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate multiple DDS from a list of requests."""
        results = []
        errors = []
        for idx, req in enumerate(requests):
            try:
                dds = self.generate_dds(
                    supplier_id=req["supplier_id"],
                    commodity=req["commodity"],
                    country_iso3=req["country_iso3"],
                    year=req["year"],
                    plot_ids=req.get("plot_ids"),
                )
                results.append(dds)
            except Exception as e:
                errors.append({"index": idx, "error": str(e)})
        return {
            "total": len(requests),
            "created": len(results),
            "failed": len(errors),
            "dds_ids": [r["dds_id"] for r in results],
            "errors": errors,
        }

    def download_dds(self, dds_id: str, format: str = "json") -> Dict[str, Any]:
        """Download DDS in specified format."""
        dds = self._store.get(dds_id)
        if not dds:
            raise DDSNotFoundError(f"DDS '{dds_id}' not found")
        fmt = format.lower()
        if fmt not in ("json", "xml"):
            raise DDSValidationError(f"Invalid format '{format}'. Supported: json, xml")
        if fmt == "json":
            return {"format": "json", "data": dds}
        else:
            # Simulated XML conversion
            return {"format": "xml", "data": f"<dds id='{dds_id}'></dds>"}

    def amend_dds(self, dds_id: str) -> Dict[str, Any]:
        """Create an amendment of an existing DDS."""
        original = self._store.get(dds_id)
        if not original:
            raise DDSNotFoundError(f"DDS '{dds_id}' not found")
        if original["status"] not in ("submitted", "accepted"):
            raise DDSValidationError(
                f"Can only amend submitted/accepted DDS, got '{original['status']}'"
            )

        amendment = self.generate_dds(
            supplier_id=original["supplier_id"],
            commodity=original["commodity"],
            country_iso3=original["country_iso3"],
            year=original["year"],
            plot_ids=original.get("plot_ids"),
            operator_info=original.get("operator_info"),
            product_description=original.get("product_description"),
            country_of_production=original.get("country_of_production"),
            geolocation_data=original.get("geolocation_data"),
            risk_assessment=original.get("risk_assessment"),
            risk_mitigation=original.get("risk_mitigation"),
            conclusion=original.get("conclusion"),
        )
        amendment["amendment_of"] = dds_id
        original["status"] = "amended"
        original["updated_at"] = datetime.now(timezone.utc)

        return amendment

    def annual_summary(self, year: int) -> Dict[str, Any]:
        """Get annual DDS summary statistics."""
        year_dds = [d for d in self._store.values() if d["year"] == year]
        by_status = {}
        by_commodity = {}
        by_country = {}
        for d in year_dds:
            by_status[d["status"]] = by_status.get(d["status"], 0) + 1
            by_commodity[d["commodity"]] = by_commodity.get(d["commodity"], 0) + 1
            by_country[d["country_iso3"]] = by_country.get(d["country_iso3"], 0) + 1
        return {
            "year": year,
            "total": len(year_dds),
            "by_status": by_status,
            "by_commodity": by_commodity,
            "by_country": by_country,
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return DDSReportingEngine()


@pytest.fixture
def sample_dds(engine):
    return engine.generate_dds(
        supplier_id="sup_test123",
        commodity="soya",
        country_iso3="BRA",
        year=2026,
        plot_ids=["plot_a", "plot_b"],
        operator_info={"name": "Test Corp", "eori": "DE123"},
        product_description={"hs_code": "1201.90"},
        country_of_production={"iso3": "BRA"},
        geolocation_data={"plots": [{"lat": -12.0, "lon": -55.0}]},
        risk_assessment={"score": 0.3, "level": "standard"},
        risk_mitigation={"measures": ["satellite monitoring"]},
        conclusion={"compliant": True},
    )


@pytest.fixture
def validated_dds(engine, sample_dds):
    engine.validate_dds(sample_dds["dds_id"])
    return sample_dds


@pytest.fixture
def submitted_dds(engine, validated_dds):
    engine.submit_dds(validated_dds["dds_id"])
    return validated_dds


# ---------------------------------------------------------------------------
# TestGenerateDDS
# ---------------------------------------------------------------------------

class TestGenerateDDS:

    def test_creates_with_all_7_sections(self, engine, sample_dds):
        for section in DDS_SECTIONS:
            assert section in sample_dds
            assert isinstance(sample_dds[section], dict)

    def test_reference_number_format(self, engine, sample_dds):
        ref = sample_dds["reference_number"]
        assert re.match(r"^EUDR-BRA-2026-\d{6}$", ref)

    def test_initial_status_is_draft(self, engine, sample_dds):
        assert sample_dds["status"] == "draft"

    def test_links_to_supplier_and_plots(self, engine, sample_dds):
        assert sample_dds["supplier_id"] == "sup_test123"
        assert sample_dds["plot_ids"] == ["plot_a", "plot_b"]

    def test_generates_unique_ids(self, engine):
        d1 = engine.generate_dds("s1", "wood", "BRA", 2026)
        d2 = engine.generate_dds("s2", "wood", "BRA", 2026)
        assert d1["dds_id"] != d2["dds_id"]
        assert d1["reference_number"] != d2["reference_number"]


# ---------------------------------------------------------------------------
# TestValidateDDS
# ---------------------------------------------------------------------------

class TestValidateDDS:

    def test_complete_dds_passes(self, engine, sample_dds):
        result = engine.validate_dds(sample_dds["dds_id"])
        assert result["valid"] is True
        assert len(result["failed_sections"]) == 0

    def test_missing_geolocation_fails(self, engine):
        dds = engine.generate_dds("s1", "wood", "BRA", 2026,
                                  operator_info={"name": "X"},
                                  product_description={"hs": "44"},
                                  country_of_production={"iso3": "BRA"},
                                  risk_assessment={"score": 0.1},
                                  risk_mitigation={"m": "none"},
                                  conclusion={"ok": True})
        result = engine.validate_dds(dds["dds_id"])
        assert result["valid"] is False
        assert "geolocation_data" in result["failed_sections"]

    def test_missing_risk_assessment_fails(self, engine):
        dds = engine.generate_dds("s1", "wood", "BRA", 2026,
                                  operator_info={"name": "X"},
                                  product_description={"hs": "44"},
                                  country_of_production={"iso3": "BRA"},
                                  geolocation_data={"lat": -12},
                                  risk_mitigation={"m": "none"},
                                  conclusion={"ok": True})
        result = engine.validate_dds(dds["dds_id"])
        assert result["valid"] is False
        assert "risk_assessment" in result["failed_sections"]

    def test_returns_per_section_results(self, engine, sample_dds):
        result = engine.validate_dds(sample_dds["dds_id"])
        for section in DDS_SECTIONS:
            assert section in result["sections"]
            assert "complete" in result["sections"][section]

    def test_nonexistent_raises(self, engine):
        with pytest.raises(DDSNotFoundError):
            engine.validate_dds("dds_nonexistent")


# ---------------------------------------------------------------------------
# TestSubmitDDS
# ---------------------------------------------------------------------------

class TestSubmitDDS:

    def test_validated_dds_can_submit(self, engine, validated_dds):
        result = engine.submit_dds(validated_dds["dds_id"])
        assert result["status"] == "submitted"

    def test_draft_cannot_submit(self, engine):
        dds = engine.generate_dds("s1", "wood", "BRA", 2026)
        with pytest.raises(DDSValidationError, match="Cannot submit"):
            engine.submit_dds(dds["dds_id"])

    def test_sets_submitted_status(self, engine, validated_dds):
        engine.submit_dds(validated_dds["dds_id"])
        dds = engine.get_dds(validated_dds["dds_id"])
        assert dds["status"] == "submitted"

    def test_records_submission_date(self, engine, validated_dds):
        result = engine.submit_dds(validated_dds["dds_id"])
        assert result["submission_date"] is not None

    def test_simulates_eu_response(self, engine, validated_dds):
        result = engine.submit_dds(validated_dds["dds_id"])
        assert result["eu_reference"] is not None
        assert result["eu_response"]["status"] == "received"

    def test_nonexistent_raises(self, engine):
        with pytest.raises(DDSNotFoundError):
            engine.submit_dds("dds_nonexistent")

    def test_double_submit_raises(self, engine, validated_dds):
        engine.submit_dds(validated_dds["dds_id"])
        with pytest.raises(DDSValidationError, match="Cannot submit"):
            engine.submit_dds(validated_dds["dds_id"])


# ---------------------------------------------------------------------------
# TestDDSLifecycle
# ---------------------------------------------------------------------------

class TestDDSLifecycle:

    def test_draft_to_accepted(self, engine, sample_dds):
        dds_id = sample_dds["dds_id"]
        # draft -> review (via validate with incomplete -> sets to review if fails)
        engine.update_status(dds_id, "review")
        assert engine.get_dds(dds_id)["status"] == "review"
        # review -> validated
        engine.update_status(dds_id, "validated")
        # validated -> submitted
        engine.submit_dds(dds_id)
        assert engine.get_dds(dds_id)["status"] == "submitted"
        # submitted -> accepted
        engine.update_status(dds_id, "accepted")
        assert engine.get_dds(dds_id)["status"] == "accepted"

    def test_draft_to_rejected(self, engine, sample_dds):
        dds_id = sample_dds["dds_id"]
        engine.update_status(dds_id, "review")
        engine.update_status(dds_id, "validated")
        engine.submit_dds(dds_id)
        engine.update_status(dds_id, "rejected")
        assert engine.get_dds(dds_id)["status"] == "rejected"

    def test_amendment_workflow(self, engine, submitted_dds):
        original_id = submitted_dds["dds_id"]
        amendment = engine.amend_dds(original_id)
        assert amendment["amendment_of"] == original_id
        assert engine.get_dds(original_id)["status"] == "amended"
        assert amendment["status"] == "draft"


# ---------------------------------------------------------------------------
# TestBulkGenerate
# ---------------------------------------------------------------------------

class TestBulkGenerate:

    def test_multiple_suppliers(self, engine):
        requests = [
            {"supplier_id": "s1", "commodity": "wood", "country_iso3": "BRA", "year": 2026},
            {"supplier_id": "s2", "commodity": "soya", "country_iso3": "IDN", "year": 2026},
        ]
        result = engine.bulk_generate(requests)
        assert result["created"] == 2
        assert result["failed"] == 0

    def test_handles_partial_failures(self, engine):
        requests = [
            {"supplier_id": "s1", "commodity": "wood", "country_iso3": "BRA", "year": 2026},
            {"commodity": "soya"},  # missing supplier_id will cause KeyError
        ]
        result = engine.bulk_generate(requests)
        assert result["created"] == 1
        assert result["failed"] == 1

    def test_returns_counts(self, engine):
        requests = [
            {"supplier_id": f"s{i}", "commodity": "wood", "country_iso3": "BRA", "year": 2026}
            for i in range(5)
        ]
        result = engine.bulk_generate(requests)
        assert result["total"] == 5
        assert result["created"] == 5


# ---------------------------------------------------------------------------
# TestDownloadDDS
# ---------------------------------------------------------------------------

class TestDownloadDDS:

    def test_json_format(self, engine, sample_dds):
        result = engine.download_dds(sample_dds["dds_id"], "json")
        assert result["format"] == "json"
        assert result["data"]["dds_id"] == sample_dds["dds_id"]

    def test_xml_format(self, engine, sample_dds):
        result = engine.download_dds(sample_dds["dds_id"], "xml")
        assert result["format"] == "xml"
        assert sample_dds["dds_id"] in result["data"]

    def test_invalid_format_raises(self, engine, sample_dds):
        with pytest.raises(DDSValidationError, match="Invalid format"):
            engine.download_dds(sample_dds["dds_id"], "csv")

    def test_nonexistent_raises(self, engine):
        with pytest.raises(DDSNotFoundError):
            engine.download_dds("dds_nonexistent")


# ---------------------------------------------------------------------------
# TestAmendDDS
# ---------------------------------------------------------------------------

class TestAmendDDS:

    def test_creates_amendment_with_new_reference(self, engine, submitted_dds):
        amendment = engine.amend_dds(submitted_dds["dds_id"])
        assert amendment["reference_number"] != submitted_dds["reference_number"]

    def test_links_to_original(self, engine, submitted_dds):
        amendment = engine.amend_dds(submitted_dds["dds_id"])
        assert amendment["amendment_of"] == submitted_dds["dds_id"]

    def test_only_submitted_accepted_can_amend(self, engine):
        dds = engine.generate_dds("s1", "wood", "BRA", 2026)
        with pytest.raises(DDSValidationError, match="Can only amend"):
            engine.amend_dds(dds["dds_id"])

    def test_accepted_can_amend(self, engine, submitted_dds):
        engine.update_status(submitted_dds["dds_id"], "accepted")
        amendment = engine.amend_dds(submitted_dds["dds_id"])
        assert amendment["amendment_of"] == submitted_dds["dds_id"]


# ---------------------------------------------------------------------------
# TestAnnualSummary
# ---------------------------------------------------------------------------

class TestAnnualSummary:

    def test_counts_by_status(self, engine):
        engine.generate_dds("s1", "wood", "BRA", 2026)
        dds2 = engine.generate_dds("s2", "soya", "IDN", 2026)
        engine.validate_dds(dds2["dds_id"])  # sets it to validated or review
        summary = engine.annual_summary(2026)
        assert summary["total"] == 2

    def test_counts_by_commodity(self, engine):
        engine.generate_dds("s1", "wood", "BRA", 2026)
        engine.generate_dds("s2", "soya", "BRA", 2026)
        engine.generate_dds("s3", "wood", "IDN", 2026)
        summary = engine.annual_summary(2026)
        assert summary["by_commodity"]["wood"] == 2
        assert summary["by_commodity"]["soya"] == 1

    def test_counts_by_country(self, engine):
        engine.generate_dds("s1", "wood", "BRA", 2026)
        engine.generate_dds("s2", "soya", "BRA", 2026)
        engine.generate_dds("s3", "cocoa", "IDN", 2026)
        summary = engine.annual_summary(2026)
        assert summary["by_country"]["BRA"] == 2
        assert summary["by_country"]["IDN"] == 1

    def test_empty_year(self, engine):
        summary = engine.annual_summary(2030)
        assert summary["total"] == 0
        assert summary["by_status"] == {}


# ---------------------------------------------------------------------------
# TestReferenceNumber
# ---------------------------------------------------------------------------

class TestReferenceNumber:

    def test_format_eudr_iso3_year_seq(self, engine):
        dds = engine.generate_dds("s1", "wood", "BRA", 2026)
        assert re.match(r"^EUDR-BRA-2026-\d{6}$", dds["reference_number"])

    def test_sequential_numbering(self, engine):
        d1 = engine.generate_dds("s1", "wood", "BRA", 2026)
        d2 = engine.generate_dds("s2", "soya", "BRA", 2026)
        seq1 = int(d1["reference_number"].split("-")[-1])
        seq2 = int(d2["reference_number"].split("-")[-1])
        assert seq2 == seq1 + 1

    def test_resets_per_country_per_year(self, engine):
        d_bra = engine.generate_dds("s1", "wood", "BRA", 2026)
        d_idn = engine.generate_dds("s2", "wood", "IDN", 2026)
        seq_bra = int(d_bra["reference_number"].split("-")[-1])
        seq_idn = int(d_idn["reference_number"].split("-")[-1])
        assert seq_bra == 1
        assert seq_idn == 1

    def test_different_years_reset(self, engine):
        d_2026 = engine.generate_dds("s1", "wood", "BRA", 2026)
        d_2027 = engine.generate_dds("s1", "wood", "BRA", 2027)
        assert "2026" in d_2026["reference_number"]
        assert "2027" in d_2027["reference_number"]
        seq_2026 = int(d_2026["reference_number"].split("-")[-1])
        seq_2027 = int(d_2027["reference_number"].split("-")[-1])
        assert seq_2026 == 1
        assert seq_2027 == 1

    def test_six_digit_zero_padded(self, engine):
        dds = engine.generate_dds("s1", "wood", "BRA", 2026)
        seq_part = dds["reference_number"].split("-")[-1]
        assert len(seq_part) == 6
        assert seq_part == "000001"
