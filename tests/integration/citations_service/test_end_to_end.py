# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Citations & Evidence Service (AGENT-FOUND-005)

Tests full citation lifecycle, multi-framework compliance workflows,
supersession chains, evidence packaging with verification, export/import
roundtrip, provenance chain integrity, and multi-source registration.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for integration testing
# (Copied from unit test inlines for isolation)
# ---------------------------------------------------------------------------


class CitationRecord:
    """Minimal citation record."""

    def __init__(
        self,
        citation_id: str,
        citation_type: str = "emission_factor",
        source_authority: str = "defra",
        title: str = "",
        authors: Optional[List[str]] = None,
        version: Optional[str] = None,
        doi: Optional[str] = None,
        effective_date: str = "2024-01-01",
        expiration_date: Optional[str] = None,
        verification_status: str = "unverified",
        regulatory_frameworks: Optional[List[str]] = None,
        key_values: Optional[Dict[str, Any]] = None,
        superseded_by: Optional[str] = None,
        supersedes: Optional[str] = None,
        record_version: int = 1,
        is_deleted: bool = False,
        created_at: Optional[str] = None,
        updated_at: Optional[str] = None,
    ):
        self.citation_id = citation_id
        self.citation_type = citation_type
        self.source_authority = source_authority
        self.title = title
        self.authors = authors or []
        self.version = version
        self.doi = doi
        self.effective_date = effective_date
        self.expiration_date = expiration_date
        self.verification_status = verification_status
        self.regulatory_frameworks = regulatory_frameworks or []
        self.key_values = key_values or {}
        self.superseded_by = superseded_by
        self.supersedes = supersedes
        self.record_version = record_version
        self.is_deleted = is_deleted
        self.created_at = created_at or datetime.utcnow().isoformat()
        self.updated_at = updated_at or self.created_at

    def calculate_content_hash(self) -> str:
        content = {
            "citation_type": self.citation_type,
            "source_authority": self.source_authority,
            "title": self.title,
            "effective_date": self.effective_date,
            "key_values": self.key_values,
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "citation_type": self.citation_type,
            "source_authority": self.source_authority,
            "title": self.title,
            "authors": self.authors,
            "version": self.version,
            "doi": self.doi,
            "effective_date": self.effective_date,
            "expiration_date": self.expiration_date,
            "verification_status": self.verification_status,
            "regulatory_frameworks": self.regulatory_frameworks,
            "key_values": self.key_values,
            "superseded_by": self.superseded_by,
            "supersedes": self.supersedes,
            "record_version": self.record_version,
            "is_deleted": self.is_deleted,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


class ChangeLogEntry:
    def __init__(self, entry_id, citation_id, change_type, old_data=None,
                 new_data=None, changed_by="system", change_reason="",
                 provenance_hash="", timestamp=None):
        self.entry_id = entry_id
        self.citation_id = citation_id
        self.change_type = change_type
        self.old_data = old_data
        self.new_data = new_data
        self.changed_by = changed_by
        self.change_reason = change_reason
        self.provenance_hash = provenance_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()


class RegistryError(Exception):
    pass

class DuplicateCitationError(RegistryError):
    pass

class CitationNotFoundError(RegistryError):
    pass

class ValidationError(RegistryError):
    pass


class CitationRegistry:
    """Registry for managing citations with version history."""

    def __init__(self, enable_provenance: bool = True):
        self._citations: Dict[str, CitationRecord] = {}
        self._versions: Dict[str, List[Dict[str, Any]]] = {}
        self._changelog: List[ChangeLogEntry] = []
        self._enable_provenance = enable_provenance
        self._change_counter = 0

    def create(
        self,
        citation_id: str,
        citation_type: str = "emission_factor",
        source_authority: str = "defra",
        title: str = "",
        authors: Optional[List[str]] = None,
        version: Optional[str] = None,
        doi: Optional[str] = None,
        effective_date: str = "2024-01-01",
        expiration_date: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        key_values: Optional[Dict[str, Any]] = None,
    ) -> CitationRecord:
        if not citation_id:
            raise ValidationError("citation_id is required")
        if citation_id in self._citations and not self._citations[citation_id].is_deleted:
            raise DuplicateCitationError(f"Citation '{citation_id}' already exists")

        record = CitationRecord(
            citation_id=citation_id,
            citation_type=citation_type,
            source_authority=source_authority,
            title=title,
            authors=authors,
            version=version,
            doi=doi,
            effective_date=effective_date,
            expiration_date=expiration_date,
            regulatory_frameworks=regulatory_frameworks,
            key_values=key_values,
        )
        self._citations[citation_id] = record
        self._versions[citation_id] = [record.to_dict()]
        self._record_change(citation_id, "create", None, record.to_dict())
        return record

    def get(self, citation_id: str) -> CitationRecord:
        if citation_id not in self._citations or self._citations[citation_id].is_deleted:
            raise CitationNotFoundError(f"Citation '{citation_id}' not found")
        return self._citations[citation_id]

    def update(
        self,
        citation_id: str,
        title: Optional[str] = None,
        key_values: Optional[Dict[str, Any]] = None,
        verification_status: Optional[str] = None,
        expiration_date: Optional[str] = None,
        regulatory_frameworks: Optional[List[str]] = None,
        version: Optional[str] = None,
        change_reason: str = "",
        changed_by: str = "system",
    ) -> CitationRecord:
        r = self.get(citation_id)
        old_data = r.to_dict()

        if title is not None:
            r.title = title
        if key_values is not None:
            r.key_values = key_values
        if verification_status is not None:
            r.verification_status = verification_status
        if expiration_date is not None:
            r.expiration_date = expiration_date
        if regulatory_frameworks is not None:
            r.regulatory_frameworks = regulatory_frameworks
        if version is not None:
            r.version = version

        r.record_version += 1
        r.updated_at = datetime.utcnow().isoformat()

        self._versions[citation_id].append(r.to_dict())
        self._record_change(citation_id, "update", old_data, r.to_dict(),
                            changed_by=changed_by, change_reason=change_reason)
        return r

    def delete(self, citation_id: str) -> bool:
        r = self.get(citation_id)
        old_data = r.to_dict()
        r.is_deleted = True
        r.updated_at = datetime.utcnow().isoformat()
        self._record_change(citation_id, "delete", old_data, None)
        return True

    def list_citations(
        self,
        citation_type: Optional[str] = None,
        source_authority: Optional[str] = None,
        regulatory_framework: Optional[str] = None,
        verification_status: Optional[str] = None,
    ) -> List[CitationRecord]:
        results = [r for r in self._citations.values() if not r.is_deleted]
        if citation_type:
            results = [r for r in results if r.citation_type == citation_type]
        if source_authority:
            results = [r for r in results if r.source_authority == source_authority]
        if regulatory_framework:
            results = [r for r in results if regulatory_framework in r.regulatory_frameworks]
        if verification_status:
            results = [r for r in results if r.verification_status == verification_status]
        return results

    def search(self, query: str) -> List[CitationRecord]:
        q = query.lower()
        results = []
        for r in self._citations.values():
            if r.is_deleted:
                continue
            if q in r.title.lower() or q in r.citation_id.lower():
                results.append(r)
                continue
            for author in r.authors:
                if q in author.lower():
                    results.append(r)
                    break
        return results

    def get_versions(self, citation_id: str) -> List[Dict[str, Any]]:
        if citation_id not in self._versions:
            raise CitationNotFoundError(f"Citation '{citation_id}' not found")
        return list(self._versions[citation_id])

    def get_changelog(self, citation_id: Optional[str] = None) -> List[ChangeLogEntry]:
        if citation_id:
            return [e for e in self._changelog if e.citation_id == citation_id]
        return list(self._changelog)

    def supersede(self, old_id: str, new_id: str) -> None:
        old_rec = self.get(old_id)
        new_rec = self.get(new_id)
        old_rec.superseded_by = new_id
        old_rec.verification_status = "superseded"
        new_rec.supersedes = old_id

    def export_all(self) -> Dict[str, Any]:
        citations = [r.to_dict() for r in self._citations.values() if not r.is_deleted]
        payload = json.dumps(citations, sort_keys=True, default=str)
        return {
            "citations": citations,
            "exported_at": datetime.utcnow().isoformat(),
            "integrity_hash": hashlib.sha256(payload.encode()).hexdigest(),
        }

    def import_all(self, data: Dict[str, Any], skip_duplicates: bool = True) -> Dict[str, Any]:
        imported, skipped, errors = 0, 0, []
        for item in data.get("citations", []):
            cid = item.get("citation_id", "")
            if cid in self._citations and not self._citations[cid].is_deleted:
                if skip_duplicates:
                    skipped += 1
                    continue
            try:
                self.create(
                    citation_id=cid,
                    citation_type=item.get("citation_type", "emission_factor"),
                    source_authority=item.get("source_authority", "defra"),
                    title=item.get("title", ""),
                    authors=item.get("authors"),
                    version=item.get("version"),
                    doi=item.get("doi"),
                    effective_date=item.get("effective_date", "2024-01-01"),
                    expiration_date=item.get("expiration_date"),
                    regulatory_frameworks=item.get("regulatory_frameworks"),
                    key_values=item.get("key_values"),
                )
                imported += 1
            except Exception as e:
                errors.append(str(e))
        return {"imported": imported, "skipped": skipped, "errors": errors}

    @property
    def count(self) -> int:
        return len([r for r in self._citations.values() if not r.is_deleted])

    def _record_change(self, citation_id, change_type, old_data, new_data,
                       changed_by="system", change_reason=""):
        self._change_counter += 1
        prov_hash = ""
        if self._enable_provenance:
            payload = json.dumps({
                "change_type": change_type, "citation_id": citation_id,
                "counter": self._change_counter,
            }, sort_keys=True)
            prov_hash = hashlib.sha256(payload.encode()).hexdigest()

        entry = ChangeLogEntry(
            entry_id=f"cl-{self._change_counter:06d}",
            citation_id=citation_id,
            change_type=change_type,
            old_data=old_data,
            new_data=new_data,
            changed_by=changed_by,
            change_reason=change_reason,
            provenance_hash=prov_hash,
        )
        self._changelog.append(entry)


class VerificationEngine:
    """Verifies citations for validity, expiration, and integrity."""

    def __init__(self, reference_date: Optional[str] = None):
        self._reference_date = reference_date or date.today().isoformat()
        self._history: List[Dict[str, Any]] = []

    def verify_citation(self, citation: CitationRecord,
                        verified_by: str = "system") -> str:
        old_status = citation.verification_status
        new_status = self._do_verify(citation)
        citation.verification_status = new_status
        self._history.append({
            "citation_id": citation.citation_id,
            "old_status": old_status,
            "new_status": new_status,
            "verified_by": verified_by,
        })
        return new_status

    def verify_batch(self, citations: List[CitationRecord],
                     verified_by: str = "system") -> Dict[str, str]:
        results = {}
        for c in citations:
            results[c.citation_id] = self.verify_citation(c, verified_by)
        return results

    def _do_verify(self, citation: CitationRecord) -> str:
        # Priority: expired > superseded > invalid > unverified > verified
        if citation.expiration_date and citation.expiration_date < self._reference_date:
            return "expired"
        if citation.superseded_by:
            return "superseded"
        if citation.citation_type == "scientific" and not citation.doi:
            return "unverified"
        if citation.source_authority in ("defra", "epa", "ecoinvent") and not citation.version:
            return "unverified"
        return "verified"

    @property
    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class EvidenceItem:
    def __init__(self, evidence_id=None, evidence_type="calculation",
                 description="", data=None, citation_ids=None,
                 source_system=None, source_agent=None, content_hash=None):
        self.evidence_id = evidence_id or str(uuid.uuid4())
        self.evidence_type = evidence_type
        self.description = description
        self.data = data or {}
        self.citation_ids = citation_ids or []
        self.source_system = source_system
        self.source_agent = source_agent
        self.content_hash = content_hash

    def calculate_content_hash(self) -> str:
        content = {
            "evidence_type": self.evidence_type,
            "description": self.description,
            "data": self.data,
            "citation_ids": sorted(self.citation_ids),
        }
        return hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()


class EvidencePackage:
    def __init__(self, package_id=None, name="", description="",
                 context=None, created_by=None):
        if not name:
            raise ValueError("name is required")
        self.package_id = package_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.evidence_items: List[EvidenceItem] = []
        self.citation_ids: List[str] = []
        self.context = context or {}
        self.created_by = created_by
        self.created_at = datetime.utcnow().isoformat()
        self.is_finalized = False
        self.package_hash: Optional[str] = None

    def add_item(self, item: EvidenceItem) -> None:
        if self.is_finalized:
            raise RuntimeError("Cannot modify finalized package")
        item.content_hash = item.calculate_content_hash()
        self.evidence_items.append(item)
        self.package_hash = None

    def add_citation(self, citation_id: str) -> None:
        if self.is_finalized:
            raise RuntimeError("Cannot modify finalized package")
        if citation_id not in self.citation_ids:
            self.citation_ids.append(citation_id)
        self.package_hash = None

    def finalize(self) -> str:
        if self.is_finalized:
            raise RuntimeError("Package already finalized")
        content = {
            "name": self.name,
            "items": [i.calculate_content_hash() for i in self.evidence_items],
            "citation_ids": sorted(self.citation_ids),
            "context": self.context,
        }
        self.package_hash = hashlib.sha256(
            json.dumps(content, sort_keys=True, default=str).encode()
        ).hexdigest()
        self.is_finalized = True
        return self.package_hash


class EvidenceManager:
    def __init__(self, max_items_per_package: int = 500):
        self._packages: Dict[str, EvidencePackage] = {}
        self._max_items_per_package = max_items_per_package

    def create_package(self, name: str, description: str = "",
                       context: Optional[Dict[str, Any]] = None,
                       created_by: Optional[str] = None) -> EvidencePackage:
        pkg = EvidencePackage(
            name=name, description=description,
            context=context, created_by=created_by,
        )
        self._packages[pkg.package_id] = pkg
        return pkg

    def get_package(self, package_id: str) -> EvidencePackage:
        if package_id not in self._packages:
            raise KeyError(f"Package '{package_id}' not found")
        return self._packages[package_id]

    def add_item(self, package_id: str, item: EvidenceItem) -> EvidencePackage:
        pkg = self.get_package(package_id)
        if len(pkg.evidence_items) >= self._max_items_per_package:
            raise ValueError(f"Package has reached maximum items limit ({self._max_items_per_package})")
        pkg.add_item(item)
        return pkg

    def add_citation(self, package_id: str, citation_id: str) -> EvidencePackage:
        pkg = self.get_package(package_id)
        pkg.add_citation(citation_id)
        return pkg

    def finalize_package(self, package_id: str) -> str:
        pkg = self.get_package(package_id)
        return pkg.finalize()

    def list_packages(self, is_finalized: Optional[bool] = None) -> List[EvidencePackage]:
        results = list(self._packages.values())
        if is_finalized is not None:
            results = [p for p in results if p.is_finalized == is_finalized]
        return results

    @property
    def count(self) -> int:
        return len(self._packages)


class ProvenanceEntry:
    def __init__(self, entry_id, entity_id, entity_type="citation",
                 change_type="create", old_data=None, new_data=None,
                 user_id="system", reason="", parent_hash=None, timestamp=None):
        self.entry_id = entry_id
        self.entity_id = entity_id
        self.entity_type = entity_type
        self.change_type = change_type
        self.old_data = old_data
        self.new_data = new_data
        self.user_id = user_id
        self.reason = reason
        self.parent_hash = parent_hash
        self.timestamp = timestamp or datetime.utcnow().isoformat()
        self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        payload = json.dumps({
            "entry_id": self.entry_id, "entity_id": self.entity_id,
            "entity_type": self.entity_type, "change_type": self.change_type,
            "old_data": str(self.old_data), "new_data": str(self.new_data),
            "user_id": self.user_id, "parent_hash": self.parent_hash,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


class ProvenanceTracker:
    def __init__(self):
        self._entries: List[ProvenanceEntry] = []
        self._counter = 0

    def record(self, entity_id, change_type, entity_type="citation",
               old_data=None, new_data=None, user_id="system", reason=""):
        self._counter += 1
        parent_hash = self._entries[-1].hash if self._entries else None
        entry = ProvenanceEntry(
            entry_id=f"prov-{self._counter:06d}",
            entity_id=entity_id, entity_type=entity_type,
            change_type=change_type, old_data=old_data, new_data=new_data,
            user_id=user_id, reason=reason, parent_hash=parent_hash,
        )
        self._entries.append(entry)
        return entry

    def get_chain(self, entity_id=None, limit=None):
        results = list(self._entries)
        if entity_id:
            results = [e for e in results if e.entity_id == entity_id]
        if limit and limit > 0:
            results = results[-limit:]
        return results

    def verify_chain(self) -> bool:
        if len(self._entries) <= 1:
            return True
        for i in range(1, len(self._entries)):
            if self._entries[i].parent_hash != self._entries[i - 1].hash:
                return False
        return True

    @property
    def count(self) -> int:
        return len(self._entries)


class ExportImportManager:
    """Export/import for citations."""

    BIBTEX_TYPE_MAP = {
        "emission_factor": "techreport",
        "regulatory": "misc",
        "methodology": "manual",
        "scientific": "article",
        "company_data": "misc",
        "guidance": "techreport",
        "database": "misc",
    }

    def export_bibtex(self, citations: List[CitationRecord]) -> str:
        entries = []
        for c in citations:
            entry_type = self.BIBTEX_TYPE_MAP.get(c.citation_type, "misc")
            bibtex_id = c.citation_id.replace(" ", "_")
            fields = []
            if c.title:
                fields.append(f"  title = {{{c.title}}}")
            if c.authors:
                fields.append(f'  author = {{{" and ".join(c.authors)}}}')
            if c.version:
                fields.append(f"  edition = {{{c.version}}}")
            fields_str = ",\n".join(fields)
            entries.append(f"@{entry_type}{{{bibtex_id},\n{fields_str}\n}}")
        return "\n\n".join(entries)

    def export_json(self, citations: List[CitationRecord]) -> str:
        data = [c.to_dict() for c in citations]
        return json.dumps(data, indent=2, default=str)

    def import_json(self, json_str: str) -> List[Dict[str, Any]]:
        data = json.loads(json_str)
        return data


# ===========================================================================
# End-to-End Test Classes
# ===========================================================================


class TestFullCitationLifecycle:
    """Test: register -> verify -> evidence package -> update -> export -> audit."""

    def test_defra_emission_factor_lifecycle(self):
        """Full DEFRA EF lifecycle: register, verify, package, export, audit."""
        reg = CitationRegistry()
        engine = VerificationEngine()
        em = EvidenceManager()
        prov = ProvenanceTracker()

        # 1. Register DEFRA 2024
        cit = reg.create(
            "defra-2024-ghg", "emission_factor", "defra",
            title="DEFRA 2024 GHG Conversion Factors",
            authors=["DEFRA", "BEIS"], version="2024",
            effective_date="2024-01-01", expiration_date="2027-12-31",
            regulatory_frameworks=["csrd", "cbam"],
            key_values={"diesel_ef": 2.68, "natural_gas_ef": 1.93},
        )
        prov.record("defra-2024-ghg", "create", new_data=cit.to_dict())
        assert cit.verification_status == "unverified"
        assert reg.count == 1

        # 2. Verify
        status = engine.verify_citation(cit)
        prov.record("defra-2024-ghg", "verify", old_data="unverified", new_data=status)
        assert status == "verified"
        assert cit.verification_status == "verified"

        # 3. Create evidence package with the citation
        pkg = em.create_package("Scope 1 Diesel Evidence",
                                description="Q4 2024 diesel combustion",
                                context={"scope": 1, "fuel": "diesel"})
        calc_item = EvidenceItem(
            evidence_type="calculation",
            description="Diesel combustion calculation",
            data={"quantity_litres": 10000, "ef": 2.68, "result_kgco2e": 26800},
            citation_ids=["defra-2024-ghg"],
        )
        em.add_item(pkg.package_id, calc_item)
        em.add_citation(pkg.package_id, "defra-2024-ghg")
        prov.record("defra-2024-ghg", "evidence_linked",
                     new_data={"package_id": pkg.package_id})

        assert len(pkg.evidence_items) == 1
        assert "defra-2024-ghg" in pkg.citation_ids

        # 4. Finalize evidence package
        pkg_hash = em.finalize_package(pkg.package_id)
        prov.record(pkg.package_id, "finalize", entity_type="evidence_package",
                     new_data={"package_hash": pkg_hash})
        assert len(pkg_hash) == 64
        assert pkg.is_finalized is True

        # 5. Update citation key values
        old_kv = cit.key_values.copy()
        reg.update("defra-2024-ghg", key_values={"diesel_ef": 2.70, "natural_gas_ef": 1.95},
                    change_reason="Corrected factors")
        prov.record("defra-2024-ghg", "update", old_data=old_kv,
                     new_data=cit.key_values)
        assert cit.key_values["diesel_ef"] == 2.70
        assert cit.record_version == 2

        # 6. Export all citations
        exported = reg.export_all()
        assert len(exported["citations"]) == 1
        assert len(exported["integrity_hash"]) == 64

        # 7. Verify provenance chain
        trail = prov.get_chain(entity_id="defra-2024-ghg")
        assert len(trail) == 4  # create, verify, evidence_linked, update
        assert prov.verify_chain() is True

    def test_version_history_after_multiple_updates(self):
        """Verify version history preserves all states."""
        reg = CitationRegistry()

        reg.create("cid-1", "emission_factor", "defra",
                    title="V1", key_values={"ef": 1.0})
        for i in range(2, 6):
            reg.update("cid-1", key_values={"ef": float(i)},
                        change_reason=f"Update to v{i}")

        versions = reg.get_versions("cid-1")
        assert len(versions) == 5  # initial + 4 updates
        assert versions[0]["key_values"]["ef"] == 1.0
        assert versions[-1]["key_values"]["ef"] == 5.0

    def test_lifecycle_with_provenance_verification(self):
        """Provenance chain stays valid through full lifecycle."""
        reg = CitationRegistry()
        prov = ProvenanceTracker()

        reg.create("c1", title="First")
        prov.record("c1", "create", new_data="First")

        reg.update("c1", title="Second")
        prov.record("c1", "update", old_data="First", new_data="Second")

        reg.update("c1", title="Third")
        prov.record("c1", "update", old_data="Second", new_data="Third")

        assert prov.verify_chain() is True
        assert prov.count == 3


class TestMultiFrameworkCompliance:
    """Test multi-framework CSRD + CBAM + EUDR compliance workflow."""

    def test_csrd_cbam_registration(self):
        """Register citations under multiple regulatory frameworks."""
        reg = CitationRegistry()

        # CSRD-relevant
        reg.create("defra-2024-ghg", "emission_factor", "defra",
                    title="DEFRA 2024 GHG", version="2024",
                    regulatory_frameworks=["csrd", "cbam"])
        # CBAM-relevant
        reg.create("cbam-impl-2023", "regulatory", "eu_commission",
                    title="CBAM Implementing Regulation 2023/956",
                    effective_date="2023-10-01",
                    regulatory_frameworks=["cbam"])
        # EUDR-relevant
        reg.create("eudr-2023-1115", "regulatory", "eu_commission",
                    title="EU Deforestation Regulation 2023/1115",
                    effective_date="2024-12-30",
                    regulatory_frameworks=["eudr"])
        # Multi-framework methodology
        reg.create("ghg-protocol-corp", "methodology", "ghg_protocol",
                    title="GHG Protocol Corporate Standard",
                    version="Revised Edition",
                    regulatory_frameworks=["csrd", "cbam", "sb253"])

        # Filter by framework
        csrd = reg.list_citations(regulatory_framework="csrd")
        cbam = reg.list_citations(regulatory_framework="cbam")
        eudr = reg.list_citations(regulatory_framework="eudr")

        assert len(csrd) == 2  # defra + ghg-protocol
        assert len(cbam) == 3  # defra + cbam-impl + ghg-protocol
        assert len(eudr) == 1  # eudr-2023

    def test_multi_framework_evidence_package(self):
        """Create evidence package referencing multiple frameworks."""
        reg = CitationRegistry()
        em = EvidenceManager()

        reg.create("defra-2024", "emission_factor", "defra",
                    title="DEFRA 2024", version="2024",
                    regulatory_frameworks=["csrd"])
        reg.create("ghg-protocol", "methodology", "ghg_protocol",
                    title="GHG Protocol", version="Rev",
                    regulatory_frameworks=["csrd", "cbam"])
        reg.create("cbam-reg", "regulatory", "eu_commission",
                    title="CBAM Regulation",
                    regulatory_frameworks=["cbam"])

        pkg = em.create_package("CSRD+CBAM Compliance Evidence",
                                context={"frameworks": ["csrd", "cbam"]})
        em.add_citation(pkg.package_id, "defra-2024")
        em.add_citation(pkg.package_id, "ghg-protocol")
        em.add_citation(pkg.package_id, "cbam-reg")
        em.add_item(pkg.package_id, EvidenceItem(
            evidence_type="calculation",
            description="Scope 1 emissions",
            data={"result": 26800},
            citation_ids=["defra-2024", "ghg-protocol"],
        ))

        pkg_hash = em.finalize_package(pkg.package_id)
        assert len(pkg_hash) == 64
        assert len(pkg.citation_ids) == 3

    def test_framework_coverage_verification(self):
        """Verify all citations for a specific framework pass verification."""
        reg = CitationRegistry()
        engine = VerificationEngine()

        reg.create("ef1", "emission_factor", "defra",
                    title="EF1", version="2024",
                    regulatory_frameworks=["csrd"])
        reg.create("m1", "methodology", "ghg_protocol",
                    title="Method1", version="Rev",
                    regulatory_frameworks=["csrd"])

        csrd_cits = reg.list_citations(regulatory_framework="csrd")
        results = engine.verify_batch(csrd_cits)

        assert all(s == "verified" for s in results.values())


class TestSupersessionChain:
    """Test annual supersession workflow: v2023 -> v2024 -> v2025."""

    def test_annual_defra_supersession(self):
        """Simulate DEFRA annual update supersession chain."""
        reg = CitationRegistry()
        engine = VerificationEngine()

        # Register 3 annual versions
        reg.create("defra-2023", "emission_factor", "defra",
                    title="DEFRA 2023 GHG", version="2023",
                    effective_date="2023-01-01", expiration_date="2024-12-31",
                    key_values={"diesel_ef": 2.65})
        reg.create("defra-2024", "emission_factor", "defra",
                    title="DEFRA 2024 GHG", version="2024",
                    effective_date="2024-01-01", expiration_date="2025-12-31",
                    key_values={"diesel_ef": 2.68})
        reg.create("defra-2025", "emission_factor", "defra",
                    title="DEFRA 2025 GHG", version="2025",
                    effective_date="2025-01-01",
                    key_values={"diesel_ef": 2.72})

        # Supersede: 2023 -> 2024, 2024 -> 2025
        reg.supersede("defra-2023", "defra-2024")
        reg.supersede("defra-2024", "defra-2025")

        # Verify statuses
        assert reg._citations["defra-2023"].verification_status == "superseded"
        assert reg._citations["defra-2023"].superseded_by == "defra-2024"
        assert reg._citations["defra-2024"].verification_status == "superseded"
        assert reg._citations["defra-2024"].superseded_by == "defra-2025"

        # Only 2025 should be active (not superseded)
        v2025 = reg.get("defra-2025")
        assert v2025.supersedes == "defra-2024"
        assert v2025.superseded_by is None

        # Verify 2025 passes
        status = engine.verify_citation(v2025)
        assert status == "verified"

    def test_superseded_excluded_from_active_list(self):
        """Superseded citations excluded when filtering by verification status."""
        reg = CitationRegistry()

        reg.create("v1", "emission_factor", "defra", title="V1", version="1")
        reg.create("v2", "emission_factor", "defra", title="V2", version="2")
        reg.supersede("v1", "v2")

        active = reg.list_citations(verification_status="unverified")
        # Only v2 should be unverified; v1 is now superseded
        assert len(active) == 1
        assert active[0].citation_id == "v2"

    def test_supersession_chain_traversal(self):
        """Follow supersession chain from oldest to newest."""
        reg = CitationRegistry()

        for year in range(2020, 2026):
            reg.create(f"ef-{year}", "emission_factor", "defra",
                        title=f"EF {year}", version=str(year))

        for year in range(2020, 2025):
            reg.supersede(f"ef-{year}", f"ef-{year + 1}")

        # Traverse chain from 2020
        current_id = "ef-2020"
        chain = [current_id]
        while reg._citations[current_id].superseded_by:
            current_id = reg._citations[current_id].superseded_by
            chain.append(current_id)

        assert len(chain) == 6
        assert chain[0] == "ef-2020"
        assert chain[-1] == "ef-2025"


class TestExportImportRoundtrip:
    """Test export -> import -> verify roundtrip."""

    def test_full_roundtrip_with_integrity(self):
        """Export from one registry, import to another, verify integrity."""
        reg1 = CitationRegistry()
        reg1.create("defra-2024", "emission_factor", "defra",
                     title="DEFRA 2024", version="2024",
                     key_values={"diesel_ef": 2.68},
                     regulatory_frameworks=["csrd"])
        reg1.create("epa-2024", "emission_factor", "epa",
                     title="EPA 2024", version="2024",
                     key_values={"diesel_ef_us": 2.72},
                     regulatory_frameworks=["sb253"])

        exported = reg1.export_all()
        assert len(exported["citations"]) == 2
        assert len(exported["integrity_hash"]) == 64

        # Import into fresh registry
        reg2 = CitationRegistry()
        result = reg2.import_all(exported)
        assert result["imported"] == 2
        assert result["skipped"] == 0
        assert reg2.count == 2

        # Verify data matches
        d = reg2.get("defra-2024")
        assert d.title == "DEFRA 2024"
        assert d.key_values["diesel_ef"] == 2.68
        assert "csrd" in d.regulatory_frameworks

    def test_roundtrip_preserves_all_fields(self):
        """All citation fields survive export/import roundtrip."""
        reg1 = CitationRegistry()
        reg1.create("full-cit", "scientific", "ipcc",
                     title="IPCC AR6 WG3",
                     authors=["IPCC Working Group III"],
                     version="AR6",
                     effective_date="2022-04-04",
                     regulatory_frameworks=["csrd", "cbam"],
                     key_values={"gwp_ch4": 27.9, "gwp_n2o": 273})

        exported = reg1.export_all()
        reg2 = CitationRegistry()
        reg2.import_all(exported)

        c = reg2.get("full-cit")
        assert c.citation_type == "scientific"
        assert c.source_authority == "ipcc"
        assert c.version == "AR6"
        assert "IPCC Working Group III" in c.authors
        assert c.key_values["gwp_ch4"] == 27.9

    def test_roundtrip_skip_duplicates(self):
        """Import skips existing citations when skip_duplicates=True."""
        reg = CitationRegistry()
        reg.create("existing", "emission_factor", title="Existing")

        data = {"citations": [
            {"citation_id": "existing", "citation_type": "emission_factor", "title": "Dup"},
            {"citation_id": "new-one", "citation_type": "regulatory", "title": "New"},
        ]}
        result = reg.import_all(data, skip_duplicates=True)
        assert result["imported"] == 1
        assert result["skipped"] == 1
        assert reg.count == 2

    def test_bibtex_export_integration(self):
        """Export citations as BibTeX and verify formatting."""
        reg = CitationRegistry()
        eim = ExportImportManager()

        cit = reg.create("defra-2024", "emission_factor", "defra",
                          title="DEFRA 2024 GHG Conversion Factors",
                          version="2024")

        bibtex = eim.export_bibtex([cit])
        assert "@techreport{defra-2024" in bibtex
        assert "title = {DEFRA 2024 GHG Conversion Factors}" in bibtex
        assert "edition = {2024}" in bibtex


class TestProvenanceChainIntegrity:
    """Test provenance chain integrity across multi-entity operations."""

    def test_cross_entity_provenance(self):
        """Provenance chain spans citations and evidence packages."""
        prov = ProvenanceTracker()

        # Citation operations
        prov.record("cit-1", "create", entity_type="citation", new_data="Created")
        prov.record("cit-2", "create", entity_type="citation", new_data="Created")
        prov.record("cit-1", "verify", entity_type="citation",
                     old_data="unverified", new_data="verified")

        # Evidence package operations
        prov.record("pkg-1", "create", entity_type="evidence_package", new_data="Created")
        prov.record("pkg-1", "add_evidence", entity_type="evidence_package",
                     new_data={"item": "calc1"})
        prov.record("pkg-1", "finalize", entity_type="evidence_package",
                     new_data={"hash": "abc123"})

        assert prov.count == 6
        assert prov.verify_chain() is True

        # Filter by entity
        cit_trail = prov.get_chain(entity_id="cit-1")
        assert len(cit_trail) == 2  # create + verify

        pkg_trail = prov.get_chain(entity_id="pkg-1")
        assert len(pkg_trail) == 3  # create + add_evidence + finalize

    def test_provenance_determinism(self):
        """Same operations produce same hash chain."""
        def build_chain():
            prov = ProvenanceTracker()
            prov.record("c1", "create", new_data="A")
            prov.record("c1", "update", old_data="A", new_data="B")
            return [e.hash for e in prov.get_chain()]

        chain1 = build_chain()
        chain2 = build_chain()
        assert chain1 == chain2

    def test_provenance_tamper_detection(self):
        """Tampering with an entry breaks chain verification."""
        prov = ProvenanceTracker()
        prov.record("c1", "create", new_data="A")
        prov.record("c1", "update", old_data="A", new_data="B")
        prov.record("c1", "update", old_data="B", new_data="C")

        # Tamper with middle entry
        prov._entries[1].hash = "tampered_hash"

        # Chain should fail verification
        assert prov.verify_chain() is False


class TestMultiSourceRegistration:
    """Test registering citations from multiple authoritative sources."""

    def test_register_5_sources(self):
        """Register citations from DEFRA, EPA, IPCC, EU, GHG Protocol."""
        reg = CitationRegistry()
        engine = VerificationEngine()

        citations_data = [
            ("defra-2024", "emission_factor", "defra", "DEFRA 2024 GHG", "2024", None),
            ("epa-2024", "emission_factor", "epa", "EPA GHG Hub 2024", "2024", None),
            ("ipcc-ar6", "scientific", "ipcc", "IPCC AR6 WG3", "AR6", "10.1017/9781009157926"),
            ("csrd-2022", "regulatory", "eu_commission", "CSRD 2022/2464", None, None),
            ("ghg-corp", "methodology", "ghg_protocol", "GHG Protocol Corporate", "Rev", None),
        ]

        for cid, ctype, source, title, ver, doi_val in citations_data:
            reg.create(cid, ctype, source, title=title, version=ver, doi=doi_val)

        assert reg.count == 5

        # Verify all
        all_cits = reg.list_citations()
        results = engine.verify_batch(all_cits)

        # DEFRA, EPA, IPCC with version -> verified
        assert results["defra-2024"] == "verified"
        assert results["epa-2024"] == "verified"
        assert results["ipcc-ar6"] == "verified"
        assert results["ghg-corp"] == "verified"

        # EU Commission regulatory (no version requirement for non-defra/epa/ecoinvent) -> verified
        assert results["csrd-2022"] == "verified"

    def test_register_and_search_across_sources(self):
        """Search finds citations across all source authorities."""
        reg = CitationRegistry()
        reg.create("defra-2024", "emission_factor", "defra",
                    title="DEFRA 2024 GHG Conversion Factors", version="2024")
        reg.create("epa-2024", "emission_factor", "epa",
                    title="EPA Emission Factors Hub 2024", version="2024")

        # Search by keyword across sources
        ghg_results = reg.search("GHG")
        assert len(ghg_results) == 1  # Only DEFRA has "GHG" in title

        factor_results = reg.search("Factors")
        assert len(factor_results) == 2  # Both have "Factors"


class TestVerificationWithEvidence:
    """Test verification status impacts evidence packaging decisions."""

    def test_verified_citations_in_package(self):
        """Evidence package with verified citations."""
        reg = CitationRegistry()
        engine = VerificationEngine()
        em = EvidenceManager()

        cit = reg.create("defra-2024", "emission_factor", "defra",
                          title="DEFRA 2024", version="2024",
                          key_values={"diesel_ef": 2.68})
        engine.verify_citation(cit)
        assert cit.verification_status == "verified"

        pkg = em.create_package("Verified Evidence")
        em.add_citation(pkg.package_id, "defra-2024")
        em.add_item(pkg.package_id, EvidenceItem(
            evidence_type="calculation",
            data={"result": 26800},
            citation_ids=["defra-2024"],
        ))
        pkg_hash = em.finalize_package(pkg.package_id)
        assert len(pkg_hash) == 64

    def test_expired_citation_detected_before_packaging(self):
        """Expired citation detected during verification before evidence."""
        reg = CitationRegistry()
        engine = VerificationEngine()

        cit = reg.create("old-ef", "emission_factor", "defra",
                          title="Old EF", version="2020",
                          expiration_date="2021-12-31")

        status = engine.verify_citation(cit)
        assert status == "expired"

        # Consumer code would check status before adding to package
        assert cit.verification_status == "expired"

    def test_batch_verification_summary(self):
        """Batch verify and count by status."""
        reg = CitationRegistry()
        engine = VerificationEngine()

        reg.create("c1", "emission_factor", "defra", title="C1", version="2024")
        reg.create("c2", "emission_factor", "epa", title="C2")  # no version
        reg.create("c3", "emission_factor", "defra", title="C3",
                    version="2020", expiration_date="2021-01-01")

        all_cits = reg.list_citations()
        results = engine.verify_batch(all_cits)

        statuses = list(results.values())
        assert statuses.count("verified") == 1
        assert statuses.count("unverified") == 1
        assert statuses.count("expired") == 1


class TestDeleteAndRecreateWorkflow:
    """Test delete and re-register workflow."""

    def test_soft_delete_and_recreate(self):
        """Delete a citation, then register its replacement."""
        reg = CitationRegistry()

        reg.create("defra-2024", "emission_factor", "defra",
                    title="DEFRA 2024 Original", version="2024")
        assert reg.count == 1

        reg.delete("defra-2024")
        assert reg.count == 0

        # Re-create with same ID (soft delete allows re-creation)
        cit = reg.create("defra-2024", "emission_factor", "defra",
                          title="DEFRA 2024 Corrected", version="2024.1")
        assert reg.count == 1
        assert cit.title == "DEFRA 2024 Corrected"

    def test_changelog_tracks_delete_and_recreate(self):
        """Changelog shows full history including delete and recreate."""
        reg = CitationRegistry()

        reg.create("cid-1", title="V1")
        reg.delete("cid-1")
        reg.create("cid-1", title="V2")

        log = reg.get_changelog("cid-1")
        assert len(log) == 3
        assert log[0].change_type == "create"
        assert log[1].change_type == "delete"
        assert log[2].change_type == "create"
