"""
Unit tests for GL-EUDR-APP v1.0 Supplier Intake Engine.

Tests supplier CRUD operations, search, bulk import, compliance
status tracking, and ERP data normalization.

Test count target: 30+ tests
"""

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Supplier Intake Engine (self-contained for testing)
# ---------------------------------------------------------------------------

EUDR_COMMODITIES = {"cattle", "cocoa", "coffee", "oil_palm", "rubber", "soya", "wood"}
VALID_COMPLIANCE = {"compliant", "pending", "non_compliant", "under_review"}
VALID_RISK_LEVELS = {"low", "standard", "high", "critical"}


class SupplierIntakeError(Exception):
    pass


class SupplierNotFoundError(SupplierIntakeError):
    pass


class SupplierValidationError(SupplierIntakeError):
    pass


class SupplierIntakeEngine:
    """Engine for supplier lifecycle management in EUDR compliance."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def create_supplier(self, name: str, country_iso3: str,
                        commodities: List[str],
                        tax_id: Optional[str] = None,
                        **kwargs) -> Dict[str, Any]:
        if not name or not name.strip():
            raise SupplierValidationError("name is required")
        if not country_iso3 or len(country_iso3) != 3 or not country_iso3.isalpha():
            raise SupplierValidationError("country_iso3 must be a 3-letter ISO code")
        if not commodities:
            raise SupplierValidationError("At least one commodity is required")
        for c in commodities:
            if c.lower() not in EUDR_COMMODITIES:
                raise SupplierValidationError(f"Invalid commodity '{c}'")

        now = datetime.now(timezone.utc)
        supplier_id = f"sup_{uuid.uuid4().hex[:12]}"
        record = {
            "supplier_id": supplier_id,
            "name": name.strip(),
            "country_iso3": country_iso3.upper(),
            "tax_id": tax_id,
            "commodities": [c.lower() for c in commodities],
            "compliance_status": "pending",
            "risk_level": "standard",
            "overall_risk_score": 0.0,
            "erp_source": kwargs.get("erp_source"),
            "created_at": now,
            "updated_at": now,
        }
        self._store[supplier_id] = record
        return record

    def update_supplier(self, supplier_id: str, **updates) -> Dict[str, Any]:
        record = self._store.get(supplier_id)
        if not record:
            raise SupplierNotFoundError(f"Supplier '{supplier_id}' not found")
        if "commodities" in updates:
            for c in updates["commodities"]:
                if c.lower() not in EUDR_COMMODITIES:
                    raise SupplierValidationError(f"Invalid commodity '{c}'")
            updates["commodities"] = [c.lower() for c in updates["commodities"]]
        if "country_iso3" in updates:
            updates["country_iso3"] = updates["country_iso3"].upper()
        for key, val in updates.items():
            record[key] = val
        record["updated_at"] = datetime.now(timezone.utc)
        return record

    def get_supplier(self, supplier_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(supplier_id)

    def list_suppliers(self, country: Optional[str] = None,
                       commodity: Optional[str] = None,
                       risk_level: Optional[str] = None,
                       page: int = 1, limit: int = 20) -> Dict[str, Any]:
        results = list(self._store.values())
        if country:
            results = [s for s in results if s["country_iso3"] == country.upper()]
        if commodity:
            results = [s for s in results if commodity.lower() in s["commodities"]]
        if risk_level:
            results = [s for s in results if s["risk_level"] == risk_level.lower()]
        total = len(results)
        start = (page - 1) * limit
        return {
            "items": results[start:start + limit],
            "total": total,
            "page": page,
            "limit": limit,
        }

    def search_suppliers(self, query: str) -> List[Dict[str, Any]]:
        q = query.lower()
        results = []
        for s in self._store.values():
            if q in s["name"].lower() or (s.get("tax_id") and q in s["tax_id"].lower()):
                results.append(s)
        return results

    def bulk_import(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not records:
            return {"total": 0, "created": 0, "failed": 0, "ids": [], "errors": []}
        created_ids = []
        errors = []
        seen_tax_ids = set()
        for idx, rec in enumerate(records):
            try:
                tax_id = rec.get("tax_id")
                if tax_id and tax_id in seen_tax_ids:
                    raise SupplierValidationError(f"Duplicate tax_id '{tax_id}'")
                if tax_id:
                    seen_tax_ids.add(tax_id)
                result = self.create_supplier(
                    name=rec.get("name", ""),
                    country_iso3=rec.get("country_iso3", ""),
                    commodities=rec.get("commodities", []),
                    tax_id=tax_id,
                )
                created_ids.append(result["supplier_id"])
            except Exception as e:
                errors.append({"index": idx, "error": str(e)})
        return {
            "total": len(records),
            "created": len(created_ids),
            "failed": len(errors),
            "ids": created_ids,
            "errors": errors,
        }

    def get_compliance_status(self, supplier_id: str) -> Dict[str, Any]:
        record = self._store.get(supplier_id)
        if not record:
            raise SupplierNotFoundError(f"Supplier '{supplier_id}' not found")
        issues = []
        if record["compliance_status"] == "pending":
            issues.append("No Due Diligence Statement submitted")
        if record["compliance_status"] == "non_compliant":
            issues.append("Missing required documents")
        return {
            "supplier_id": supplier_id,
            "compliance_status": record["compliance_status"],
            "issues": issues,
        }

    def normalize_erp_data(self, data: Dict[str, Any], erp_format: str) -> Dict[str, Any]:
        """Normalize supplier data from various ERP formats."""
        erp_format = erp_format.lower()
        if erp_format == "sap":
            return {
                "name": data.get("LIFNR_NAME", data.get("NAME1", "")),
                "country_iso3": data.get("LAND1", ""),
                "tax_id": data.get("STCEG", ""),
                "commodities": data.get("MATKL_LIST", []),
            }
        elif erp_format == "oracle":
            return {
                "name": data.get("vendor_name", ""),
                "country_iso3": data.get("country", ""),
                "tax_id": data.get("tax_registration_number", ""),
                "commodities": data.get("category_list", []),
            }
        elif erp_format == "csv":
            return {
                "name": data.get("supplier_name", data.get("name", "")),
                "country_iso3": data.get("country_code", data.get("country", "")),
                "tax_id": data.get("tax_id", ""),
                "commodities": data.get("commodities", "").split(",") if isinstance(
                    data.get("commodities"), str) else data.get("commodities", []),
            }
        else:
            raise SupplierValidationError(f"Unknown ERP format '{erp_format}'")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    return SupplierIntakeEngine()


@pytest.fixture
def sample_supplier(engine):
    return engine.create_supplier(
        name="Amazon Soya Ltd.",
        country_iso3="BRA",
        commodities=["soya", "cattle"],
        tax_id="BR-TAX-12345",
    )


# ---------------------------------------------------------------------------
# TestCreateSupplier
# ---------------------------------------------------------------------------

class TestCreateSupplier:

    def test_valid_creation(self, engine):
        s = engine.create_supplier("Test Co.", "BRA", ["wood"])
        assert s["name"] == "Test Co."
        assert s["country_iso3"] == "BRA"
        assert s["compliance_status"] == "pending"

    def test_missing_name_raises(self, engine):
        with pytest.raises(SupplierValidationError, match="name is required"):
            engine.create_supplier("", "BRA", ["wood"])

    def test_missing_commodities_raises(self, engine):
        with pytest.raises(SupplierValidationError, match="At least one commodity"):
            engine.create_supplier("Test", "BRA", [])

    def test_invalid_commodity_raises(self, engine):
        with pytest.raises(SupplierValidationError, match="Invalid commodity"):
            engine.create_supplier("Test", "BRA", ["bananas"])

    def test_country_code_format(self, engine):
        with pytest.raises(SupplierValidationError, match="3-letter"):
            engine.create_supplier("Test", "BR", ["wood"])

    def test_country_code_non_alpha(self, engine):
        with pytest.raises(SupplierValidationError, match="3-letter"):
            engine.create_supplier("Test", "12A", ["wood"])

    def test_commodities_lowercased(self, engine):
        s = engine.create_supplier("Test", "BRA", ["COCOA", "Coffee"])
        assert s["commodities"] == ["cocoa", "coffee"]


# ---------------------------------------------------------------------------
# TestUpdateSupplier
# ---------------------------------------------------------------------------

class TestUpdateSupplier:

    def test_partial_update(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        updated = engine.update_supplier(sid, name="New Name")
        assert updated["name"] == "New Name"
        assert updated["commodities"] == sample_supplier["commodities"]

    def test_update_nonexistent_raises(self, engine):
        with pytest.raises(SupplierNotFoundError):
            engine.update_supplier("sup_nonexistent", name="X")

    def test_commodity_change(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        updated = engine.update_supplier(sid, commodities=["wood", "rubber"])
        assert updated["commodities"] == ["wood", "rubber"]

    def test_update_invalid_commodity_raises(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        with pytest.raises(SupplierValidationError):
            engine.update_supplier(sid, commodities=["bananas"])

    def test_update_timestamps(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        old_updated = sample_supplier["updated_at"]
        updated = engine.update_supplier(sid, name="TS Test")
        assert updated["updated_at"] >= old_updated


# ---------------------------------------------------------------------------
# TestGetSupplier
# ---------------------------------------------------------------------------

class TestGetSupplier:

    def test_get_by_id(self, engine, sample_supplier):
        result = engine.get_supplier(sample_supplier["supplier_id"])
        assert result is not None
        assert result["name"] == "Amazon Soya Ltd."

    def test_get_nonexistent_returns_none(self, engine):
        assert engine.get_supplier("sup_nonexistent") is None


# ---------------------------------------------------------------------------
# TestListSuppliers
# ---------------------------------------------------------------------------

class TestListSuppliers:

    def test_list_all(self, engine):
        engine.create_supplier("A", "BRA", ["wood"])
        engine.create_supplier("B", "IDN", ["oil_palm"])
        result = engine.list_suppliers()
        assert result["total"] == 2

    def test_filter_by_country(self, engine):
        engine.create_supplier("A", "BRA", ["wood"])
        engine.create_supplier("B", "IDN", ["oil_palm"])
        result = engine.list_suppliers(country="BRA")
        assert result["total"] == 1
        assert result["items"][0]["country_iso3"] == "BRA"

    def test_filter_by_commodity(self, engine):
        engine.create_supplier("A", "BRA", ["wood", "soya"])
        engine.create_supplier("B", "IDN", ["oil_palm"])
        result = engine.list_suppliers(commodity="soya")
        assert result["total"] == 1

    def test_filter_by_risk_level(self, engine):
        s = engine.create_supplier("A", "BRA", ["wood"])
        engine.update_supplier(s["supplier_id"], risk_level="high")
        engine.create_supplier("B", "IDN", ["oil_palm"])
        result = engine.list_suppliers(risk_level="high")
        assert result["total"] == 1

    def test_pagination(self, engine):
        for i in range(25):
            engine.create_supplier(f"Supplier_{i}", "BRA", ["wood"])
        page1 = engine.list_suppliers(page=1, limit=10)
        assert len(page1["items"]) == 10
        assert page1["total"] == 25
        page3 = engine.list_suppliers(page=3, limit=10)
        assert len(page3["items"]) == 5


# ---------------------------------------------------------------------------
# TestSearchSuppliers
# ---------------------------------------------------------------------------

class TestSearchSuppliers:

    def test_search_by_name(self, engine, sample_supplier):
        results = engine.search_suppliers("Amazon")
        assert len(results) == 1

    def test_search_by_tax_id(self, engine, sample_supplier):
        results = engine.search_suppliers("BR-TAX")
        assert len(results) == 1

    def test_search_case_insensitive(self, engine, sample_supplier):
        results = engine.search_suppliers("amazon")
        assert len(results) == 1

    def test_search_no_match(self, engine, sample_supplier):
        results = engine.search_suppliers("nonexistent_company")
        assert len(results) == 0


# ---------------------------------------------------------------------------
# TestBulkImport
# ---------------------------------------------------------------------------

class TestBulkImport:

    def test_valid_records(self, engine):
        records = [
            {"name": "A", "country_iso3": "BRA", "commodities": ["wood"]},
            {"name": "B", "country_iso3": "IDN", "commodities": ["oil_palm"]},
        ]
        result = engine.bulk_import(records)
        assert result["created"] == 2
        assert result["failed"] == 0
        assert len(result["ids"]) == 2

    def test_mixed_valid_invalid(self, engine):
        records = [
            {"name": "Valid", "country_iso3": "BRA", "commodities": ["wood"]},
            {"name": "", "country_iso3": "BRA", "commodities": ["wood"]},  # invalid: empty name
            {"name": "Also Valid", "country_iso3": "DEU", "commodities": ["cocoa"]},
        ]
        result = engine.bulk_import(records)
        assert result["created"] == 2
        assert result["failed"] == 1

    def test_duplicate_tax_id_detection(self, engine):
        records = [
            {"name": "A", "country_iso3": "BRA", "commodities": ["wood"], "tax_id": "TX001"},
            {"name": "B", "country_iso3": "BRA", "commodities": ["wood"], "tax_id": "TX001"},
        ]
        result = engine.bulk_import(records)
        assert result["created"] == 1
        assert result["failed"] == 1
        assert "Duplicate" in result["errors"][0]["error"]

    def test_empty_input(self, engine):
        result = engine.bulk_import([])
        assert result["total"] == 0
        assert result["created"] == 0


# ---------------------------------------------------------------------------
# TestComplianceStatus
# ---------------------------------------------------------------------------

class TestComplianceStatus:

    def test_compliant_supplier(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        engine.update_supplier(sid, compliance_status="compliant")
        status = engine.get_compliance_status(sid)
        assert status["compliance_status"] == "compliant"
        assert len(status["issues"]) == 0

    def test_non_compliant_missing_docs(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        engine.update_supplier(sid, compliance_status="non_compliant")
        status = engine.get_compliance_status(sid)
        assert "Missing required documents" in status["issues"]

    def test_pending_shows_issues(self, engine, sample_supplier):
        sid = sample_supplier["supplier_id"]
        status = engine.get_compliance_status(sid)
        assert status["compliance_status"] == "pending"
        assert "No Due Diligence Statement" in status["issues"][0]

    def test_nonexistent_raises(self, engine):
        with pytest.raises(SupplierNotFoundError):
            engine.get_compliance_status("sup_nonexistent")


# ---------------------------------------------------------------------------
# TestERPNormalization
# ---------------------------------------------------------------------------

class TestERPNormalization:

    def test_sap_format(self, engine):
        data = {"LIFNR_NAME": "SAP Supplier", "LAND1": "DEU", "STCEG": "DE123",
                "MATKL_LIST": ["wood"]}
        result = engine.normalize_erp_data(data, "SAP")
        assert result["name"] == "SAP Supplier"
        assert result["country_iso3"] == "DEU"

    def test_oracle_format(self, engine):
        data = {"vendor_name": "Oracle Vendor", "country": "USA",
                "tax_registration_number": "US-999", "category_list": ["cocoa"]}
        result = engine.normalize_erp_data(data, "Oracle")
        assert result["name"] == "Oracle Vendor"
        assert result["commodities"] == ["cocoa"]

    def test_csv_format(self, engine):
        data = {"supplier_name": "CSV Corp", "country_code": "GBR",
                "tax_id": "GB-111", "commodities": "wood,rubber"}
        result = engine.normalize_erp_data(data, "csv")
        assert result["name"] == "CSV Corp"
        assert result["commodities"] == ["wood", "rubber"]

    def test_unknown_format_raises(self, engine):
        with pytest.raises(SupplierValidationError, match="Unknown ERP format"):
            engine.normalize_erp_data({}, "dynamics365")
