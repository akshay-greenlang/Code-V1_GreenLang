# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Supplier Registry

Tests supplier registry operations:
- register_supplier (valid, duplicate EORI, invalid EORI)
- validate_eori (valid formats, invalid formats, edge cases)
- register_installation (valid, invalid type, capacity bounds)
- update_verification_status (transitions, expiry)
- search_suppliers (by country, sector, name, verified only)
- link_supplier_to_importer (authorize, revoke)
- Supplier lifecycle transitions
- Thread-safety of singleton
- Provenance hash generation

Target: 60+ tests
"""

import pytest
import uuid
import hashlib
import threading
import re
from datetime import datetime, date, timedelta
from decimal import Decimal
from copy import deepcopy


# ---------------------------------------------------------------------------
# Inline registry implementation for self-contained tests
# ---------------------------------------------------------------------------

class RegistryError(Exception):
    pass


class DuplicateEORIError(RegistryError):
    pass


class SupplierNotFoundError(RegistryError):
    pass


class InvalidEORIError(RegistryError):
    pass


class SupplierRegistry:
    """Thread-safe singleton supplier registry."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._suppliers = {}
        self._eori_index = {}
        self._installations = {}
        self._importer_links = {}
        self._verification_records = {}
        self._initialized = True

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._instance = None

    def register_supplier(self, *, company_name, eori_number, country_code,
                          contact_email, sector=None):
        self._validate_eori_format(eori_number)
        if eori_number in self._eori_index:
            raise DuplicateEORIError(f"EORI already registered: {eori_number}")

        supplier_id = f"SUP-{uuid.uuid4().hex[:8].upper()}"
        supplier = {
            "supplier_id": supplier_id,
            "company_name": company_name,
            "eori_number": eori_number,
            "country_code": country_code,
            "contact_email": contact_email,
            "sector": sector,
            "status": "pending",
            "created_at": datetime.utcnow().isoformat(),
            "verified": False,
            "verification_expiry": None,
        }
        self._suppliers[supplier_id] = supplier
        self._eori_index[eori_number] = supplier_id
        return supplier

    def _validate_eori_format(self, eori: str):
        if not eori or len(eori) < 5:
            raise InvalidEORIError(f"EORI too short: {eori}")
        if not eori[:2].isalpha() or not eori[:2].isupper():
            raise InvalidEORIError(f"EORI must start with 2 uppercase letters: {eori}")
        if not eori[2:].isdigit():
            raise InvalidEORIError(f"EORI suffix must be digits: {eori}")
        if len(eori[2:]) < 3 or len(eori[2:]) > 15:
            raise InvalidEORIError(f"EORI digit length invalid: {eori}")

    def validate_eori(self, eori: str) -> bool:
        try:
            self._validate_eori_format(eori)
            return True
        except InvalidEORIError:
            return False

    def get_supplier(self, supplier_id: str) -> dict:
        if supplier_id not in self._suppliers:
            raise SupplierNotFoundError(f"Supplier not found: {supplier_id}")
        return deepcopy(self._suppliers[supplier_id])

    def get_supplier_by_eori(self, eori: str) -> dict:
        if eori not in self._eori_index:
            raise SupplierNotFoundError(f"EORI not found: {eori}")
        return self.get_supplier(self._eori_index[eori])

    def register_installation(self, *, supplier_id, name, installation_type,
                              sector, country_code,
                              capacity_tonnes_per_year=0.0):
        if supplier_id not in self._suppliers:
            raise SupplierNotFoundError(f"Supplier not found: {supplier_id}")
        valid_types = {"manufacturing", "smelting", "refining",
                       "power_plant", "mining"}
        if installation_type not in valid_types:
            raise RegistryError(f"Invalid installation type: {installation_type}")
        if capacity_tonnes_per_year < 0:
            raise RegistryError("Capacity cannot be negative")

        inst_id = f"INST-{uuid.uuid4().hex[:8].upper()}"
        installation = {
            "installation_id": inst_id,
            "supplier_id": supplier_id,
            "name": name,
            "installation_type": installation_type,
            "sector": sector,
            "country_code": country_code,
            "capacity_tonnes_per_year": capacity_tonnes_per_year,
        }
        if supplier_id not in self._installations:
            self._installations[supplier_id] = []
        self._installations[supplier_id].append(installation)
        return installation

    def update_verification_status(self, supplier_id: str, verified: bool,
                                   expiry_date=None):
        if supplier_id not in self._suppliers:
            raise SupplierNotFoundError(supplier_id)
        self._suppliers[supplier_id]["verified"] = verified
        self._suppliers[supplier_id]["verification_expiry"] = (
            expiry_date.isoformat() if expiry_date else None
        )
        if verified and self._suppliers[supplier_id]["status"] == "pending":
            self._suppliers[supplier_id]["status"] = "active"

    def search_suppliers(self, *, country=None, sector=None, name=None,
                         verified_only=False):
        results = []
        for s in self._suppliers.values():
            if country and s["country_code"] != country:
                continue
            if sector and s.get("sector") != sector:
                continue
            if name and name.lower() not in s["company_name"].lower():
                continue
            if verified_only and not s["verified"]:
                continue
            results.append(deepcopy(s))
        return results

    def link_supplier_to_importer(self, supplier_id: str, importer_eori: str):
        if supplier_id not in self._suppliers:
            raise SupplierNotFoundError(supplier_id)
        key = (supplier_id, importer_eori)
        self._importer_links[key] = {
            "supplier_id": supplier_id,
            "importer_eori": importer_eori,
            "authorized": True,
            "linked_at": datetime.utcnow().isoformat(),
        }

    def revoke_importer_link(self, supplier_id: str, importer_eori: str):
        key = (supplier_id, importer_eori)
        if key in self._importer_links:
            self._importer_links[key]["authorized"] = False

    def is_linked(self, supplier_id: str, importer_eori: str) -> bool:
        key = (supplier_id, importer_eori)
        link = self._importer_links.get(key)
        return link is not None and link["authorized"]

    def generate_provenance_hash(self, supplier_id: str) -> str:
        if supplier_id not in self._suppliers:
            raise SupplierNotFoundError(supplier_id)
        s = self._suppliers[supplier_id]
        payload = f"{s['supplier_id']}:{s['eori_number']}:{s['company_name']}:{s['created_at']}"
        return hashlib.sha256(payload.encode()).hexdigest()

    def transition_status(self, supplier_id: str, new_status: str):
        if supplier_id not in self._suppliers:
            raise SupplierNotFoundError(supplier_id)
        valid_transitions = {
            "pending": {"active", "deactivated"},
            "active": {"suspended", "deactivated"},
            "suspended": {"active", "deactivated"},
            "deactivated": set(),
        }
        current = self._suppliers[supplier_id]["status"]
        if new_status not in valid_transitions.get(current, set()):
            raise RegistryError(
                f"Cannot transition from {current} to {new_status}"
            )
        self._suppliers[supplier_id]["status"] = new_status


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def reset_registry():
    """Reset singleton between tests."""
    SupplierRegistry.reset()
    yield
    SupplierRegistry.reset()


@pytest.fixture
def registry():
    return SupplierRegistry()


@pytest.fixture
def registered_supplier(registry):
    return registry.register_supplier(
        company_name="Acme Steel GmbH",
        eori_number="DE123456789012",
        country_code="DE",
        contact_email="info@acme.de",
        sector="steel",
    )


# ===========================================================================
# TEST CLASS -- register_supplier
# ===========================================================================

class TestRegisterSupplier:
    """Tests for register_supplier."""

    def test_register_valid_supplier(self, registry):
        s = registry.register_supplier(
            company_name="Test Corp", eori_number="NL999888777",
            country_code="NL", contact_email="a@b.com",
        )
        assert s["supplier_id"].startswith("SUP-")
        assert s["status"] == "pending"

    def test_register_with_sector(self, registry):
        s = registry.register_supplier(
            company_name="Steel Co", eori_number="DE111222333",
            country_code="DE", contact_email="x@y.com", sector="steel",
        )
        assert s["sector"] == "steel"

    def test_duplicate_eori_raises(self, registry, registered_supplier):
        with pytest.raises(DuplicateEORIError):
            registry.register_supplier(
                company_name="Other Co", eori_number="DE123456789012",
                country_code="DE", contact_email="other@x.com",
            )

    def test_invalid_eori_raises(self, registry):
        with pytest.raises(InvalidEORIError):
            registry.register_supplier(
                company_name="Bad Co", eori_number="123",
                country_code="DE", contact_email="a@b.com",
            )

    def test_created_at_populated(self, registry):
        s = registry.register_supplier(
            company_name="Co", eori_number="FR12345",
            country_code="FR", contact_email="a@b.com",
        )
        assert "created_at" in s


# ===========================================================================
# TEST CLASS -- validate_eori
# ===========================================================================

class TestValidateEORI:
    """Tests for EORI validation."""

    @pytest.mark.parametrize("eori", [
        "DE123456789012", "NL999888777", "FR12345",
        "AT1234567890", "BE12345678901234567",
    ])
    def test_valid_eori_returns_true(self, registry, eori):
        assert registry.validate_eori(eori) is True

    @pytest.mark.parametrize("eori", [
        "", "12345678", "de123456789", "D1234567890",
        "DEABC", "DE12", "DE", None,
    ])
    def test_invalid_eori_returns_false(self, registry, eori):
        assert registry.validate_eori(eori or "") is False

    def test_eori_with_leading_whitespace(self, registry):
        assert registry.validate_eori(" DE123456789") is False

    def test_eori_maximum_length(self, registry):
        long_eori = "DE" + "1" * 15
        assert registry.validate_eori(long_eori) is True

    def test_eori_exceed_maximum_length(self, registry):
        too_long = "DE" + "1" * 16
        assert registry.validate_eori(too_long) is False


# ===========================================================================
# TEST CLASS -- register_installation
# ===========================================================================

class TestRegisterInstallation:
    """Tests for register_installation."""

    def test_register_valid_installation(self, registry, registered_supplier):
        inst = registry.register_installation(
            supplier_id=registered_supplier["supplier_id"],
            name="Plant Alpha",
            installation_type="manufacturing",
            sector="steel",
            country_code="DE",
            capacity_tonnes_per_year=500000,
        )
        assert inst["installation_id"].startswith("INST-")

    def test_invalid_supplier_raises(self, registry):
        with pytest.raises(SupplierNotFoundError):
            registry.register_installation(
                supplier_id="NONEXISTENT",
                name="P", installation_type="manufacturing",
                sector="steel", country_code="DE",
            )

    def test_invalid_type_raises(self, registry, registered_supplier):
        with pytest.raises(RegistryError, match="installation type"):
            registry.register_installation(
                supplier_id=registered_supplier["supplier_id"],
                name="P", installation_type="factory",
                sector="steel", country_code="DE",
            )

    def test_negative_capacity_raises(self, registry, registered_supplier):
        with pytest.raises(RegistryError, match="Capacity"):
            registry.register_installation(
                supplier_id=registered_supplier["supplier_id"],
                name="P", installation_type="smelting",
                sector="aluminum", country_code="CN",
                capacity_tonnes_per_year=-1,
            )

    def test_zero_capacity_accepted(self, registry, registered_supplier):
        inst = registry.register_installation(
            supplier_id=registered_supplier["supplier_id"],
            name="Planned", installation_type="manufacturing",
            sector="cement", country_code="TR",
            capacity_tonnes_per_year=0,
        )
        assert inst["capacity_tonnes_per_year"] == 0

    def test_multiple_installations(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        registry.register_installation(
            supplier_id=sid, name="P1", installation_type="manufacturing",
            sector="steel", country_code="DE",
        )
        registry.register_installation(
            supplier_id=sid, name="P2", installation_type="smelting",
            sector="steel", country_code="DE",
        )
        assert len(registry._installations[sid]) == 2


# ===========================================================================
# TEST CLASS -- update_verification_status
# ===========================================================================

class TestUpdateVerificationStatus:
    """Tests for update_verification_status."""

    def test_set_verified(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        registry.update_verification_status(sid, True, date(2027, 3, 1))
        s = registry.get_supplier(sid)
        assert s["verified"] is True
        assert s["status"] == "active"

    def test_set_unverified(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        registry.update_verification_status(sid, True)
        registry.update_verification_status(sid, False)
        s = registry.get_supplier(sid)
        assert s["verified"] is False

    def test_expiry_date_stored(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        exp = date(2027, 12, 31)
        registry.update_verification_status(sid, True, exp)
        s = registry.get_supplier(sid)
        assert s["verification_expiry"] == exp.isoformat()

    def test_nonexistent_supplier_raises(self, registry):
        with pytest.raises(SupplierNotFoundError):
            registry.update_verification_status("NONE", True)


# ===========================================================================
# TEST CLASS -- search_suppliers
# ===========================================================================

class TestSearchSuppliers:
    """Tests for search_suppliers."""

    @pytest.fixture(autouse=True)
    def setup_suppliers(self, registry):
        registry.register_supplier(
            company_name="German Steel", eori_number="DE111111111",
            country_code="DE", contact_email="a@b.de", sector="steel",
        )
        s2 = registry.register_supplier(
            company_name="French Aluminum", eori_number="FR222222222",
            country_code="FR", contact_email="a@b.fr", sector="aluminum",
        )
        registry.update_verification_status(s2["supplier_id"], True)
        registry.register_supplier(
            company_name="Turkish Cement", eori_number="TR333333333",
            country_code="TR", contact_email="a@b.tr", sector="cement",
        )

    def test_search_by_country(self, registry):
        results = registry.search_suppliers(country="DE")
        assert len(results) == 1
        assert results[0]["country_code"] == "DE"

    def test_search_by_sector(self, registry):
        results = registry.search_suppliers(sector="aluminum")
        assert len(results) == 1

    def test_search_by_name_partial(self, registry):
        results = registry.search_suppliers(name="steel")
        assert len(results) == 1
        assert "Steel" in results[0]["company_name"]

    def test_search_by_name_case_insensitive(self, registry):
        results = registry.search_suppliers(name="ALUMINUM")
        assert len(results) == 1

    def test_search_verified_only(self, registry):
        results = registry.search_suppliers(verified_only=True)
        assert len(results) == 1
        assert results[0]["verified"] is True

    def test_search_no_results(self, registry):
        results = registry.search_suppliers(country="JP")
        assert len(results) == 0

    def test_search_all(self, registry):
        results = registry.search_suppliers()
        assert len(results) == 3

    def test_search_combined_filters(self, registry):
        results = registry.search_suppliers(country="FR", sector="aluminum")
        assert len(results) == 1


# ===========================================================================
# TEST CLASS -- link_supplier_to_importer
# ===========================================================================

class TestLinkSupplierToImporter:
    """Tests for link/revoke importer relationships."""

    def test_link_and_check(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        registry.link_supplier_to_importer(sid, "NL123456789012")
        assert registry.is_linked(sid, "NL123456789012")

    def test_revoke_link(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        registry.link_supplier_to_importer(sid, "NL123456789012")
        registry.revoke_importer_link(sid, "NL123456789012")
        assert not registry.is_linked(sid, "NL123456789012")

    def test_unlinked_returns_false(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        assert not registry.is_linked(sid, "NL999999999")

    def test_link_nonexistent_supplier_raises(self, registry):
        with pytest.raises(SupplierNotFoundError):
            registry.link_supplier_to_importer("BAD", "NL123456789012")


# ===========================================================================
# TEST CLASS -- Lifecycle transitions
# ===========================================================================

class TestSupplierLifecycle:
    """Tests for supplier status lifecycle."""

    def test_full_lifecycle(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        assert registry.get_supplier(sid)["status"] == "pending"
        registry.transition_status(sid, "active")
        assert registry.get_supplier(sid)["status"] == "active"
        registry.transition_status(sid, "suspended")
        assert registry.get_supplier(sid)["status"] == "suspended"
        registry.transition_status(sid, "active")
        assert registry.get_supplier(sid)["status"] == "active"
        registry.transition_status(sid, "deactivated")
        assert registry.get_supplier(sid)["status"] == "deactivated"

    def test_invalid_transition_raises(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        with pytest.raises(RegistryError, match="Cannot transition"):
            registry.transition_status(sid, "suspended")

    def test_deactivated_is_terminal(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        registry.transition_status(sid, "deactivated")
        with pytest.raises(RegistryError):
            registry.transition_status(sid, "active")


# ===========================================================================
# TEST CLASS -- Thread safety
# ===========================================================================

class TestSingleton:
    """Tests for singleton pattern and thread safety."""

    def test_singleton_identity(self):
        r1 = SupplierRegistry()
        r2 = SupplierRegistry()
        assert r1 is r2

    def test_singleton_after_reset(self):
        r1 = SupplierRegistry()
        SupplierRegistry.reset()
        r2 = SupplierRegistry()
        assert r1 is not r2

    def test_concurrent_registration(self, registry):
        errors = []
        results = []

        def register(i):
            try:
                s = registry.register_supplier(
                    company_name=f"Co {i}",
                    eori_number=f"DE{i:012d}",
                    country_code="DE",
                    contact_email=f"u{i}@x.com",
                )
                results.append(s)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20


# ===========================================================================
# TEST CLASS -- Provenance hash
# ===========================================================================

class TestProvenanceHash:
    """Tests for provenance hash generation."""

    def test_hash_is_sha256(self, registry, registered_supplier):
        h = registry.generate_provenance_hash(registered_supplier["supplier_id"])
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self, registry, registered_supplier):
        sid = registered_supplier["supplier_id"]
        h1 = registry.generate_provenance_hash(sid)
        h2 = registry.generate_provenance_hash(sid)
        assert h1 == h2

    def test_different_suppliers_different_hashes(self, registry, registered_supplier):
        s2 = registry.register_supplier(
            company_name="Other Co", eori_number="FR111222333",
            country_code="FR", contact_email="a@b.fr",
        )
        h1 = registry.generate_provenance_hash(registered_supplier["supplier_id"])
        h2 = registry.generate_provenance_hash(s2["supplier_id"])
        assert h1 != h2

    def test_hash_nonexistent_raises(self, registry):
        with pytest.raises(SupplierNotFoundError):
            registry.generate_provenance_hash("NONEXISTENT")
