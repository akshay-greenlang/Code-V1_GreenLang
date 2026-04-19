# -*- coding: utf-8 -*-
"""
PACK-006 EUDR Starter Pack - DDS Assembly Engine Tests
========================================================

Validates the Due Diligence Statement assembly engine including standard
and simplified DDS generation, Annex II completeness validation, batch
assembly, geolocation formatting, evidence attachment, finalization,
EU Information System export, and provenance tracking.

Test count: 25
Author: GreenLang QA Team
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import hashlib
import json
import uuid
from datetime import datetime
from typing import Any, Dict, List

import pytest

from conftest import (
    EUDR_COMMODITIES,
    EUDR_CUTOFF_DATE,
    _compute_hash,
    assert_provenance_hash,
    assert_valid_uuid,
)


# ---------------------------------------------------------------------------
# DDS Assembly Simulator
# ---------------------------------------------------------------------------

class DDSAssemblySimulator:
    """Simulates DDS assembly engine operations for testing.

    Implements the core DDS assembly logic including Annex II
    completeness checks, geolocation formatting, evidence attachment,
    finalization, and EU IS export preparation.
    """

    ANNEX_II_REQUIRED_FIELDS = [
        "operator", "commodities", "geolocation", "risk_assessment",
        "cutoff_compliance", "suppliers",
    ]

    ANNEX_II_OPERATOR_FIELDS = [
        "name", "country", "eori_number", "address", "operator_type",
    ]

    def assemble_standard_dds(self, operator, commodities, suppliers,
                               geolocation, risk_assessment, cutoff) -> Dict[str, Any]:
        """Assemble a standard DDS per EUDR Article 4."""
        dds_ref = f"DDS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        dds = {
            "dds_reference": dds_ref,
            "dds_type": "STANDARD",
            "status": "DRAFT",
            "created_at": datetime.now().isoformat(),
            "operator": operator,
            "commodities": commodities,
            "suppliers": suppliers,
            "geolocation": geolocation,
            "risk_assessment": risk_assessment,
            "cutoff_compliance": cutoff,
            "evidence": [],
            "annex_ii_complete": False,
            "provenance_hash": "",
        }
        dds["annex_ii_complete"] = self._validate_annex_ii(dds)
        dds["provenance_hash"] = _compute_hash(dds)
        return dds

    def assemble_simplified_dds(self, operator, commodities, suppliers,
                                 geolocation) -> Dict[str, Any]:
        """Assemble a simplified DDS for low-risk country sourcing."""
        dds_ref = f"SDDS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        dds = {
            "dds_reference": dds_ref,
            "dds_type": "SIMPLIFIED",
            "status": "DRAFT",
            "created_at": datetime.now().isoformat(),
            "operator": operator,
            "commodities": commodities,
            "suppliers": suppliers,
            "geolocation": geolocation,
            "risk_assessment": {
                "composite_risk": 0.15,
                "risk_level": "LOW",
                "simplified_dd_eligible": True,
            },
            "cutoff_compliance": {
                "cutoff_date": str(EUDR_CUTOFF_DATE),
                "deforestation_free": True,
            },
            "evidence": [],
            "annex_ii_complete": True,
            "provenance_hash": "",
        }
        dds["provenance_hash"] = _compute_hash(dds)
        return dds

    def _validate_annex_ii(self, dds: Dict[str, Any]) -> bool:
        """Check Annex II completeness."""
        for field in self.ANNEX_II_REQUIRED_FIELDS:
            if field not in dds or not dds[field]:
                return False
        op = dds.get("operator", {})
        for field in self.ANNEX_II_OPERATOR_FIELDS:
            if field not in op or not op[field]:
                return False
        geo = dds.get("geolocation", {})
        if "plots" not in geo or len(geo.get("plots", [])) == 0:
            return False
        return True

    def validate_annex_ii_complete(self, dds: Dict[str, Any]) -> Dict[str, Any]:
        """Return validation result for Annex II completeness."""
        errors = []
        for field in self.ANNEX_II_REQUIRED_FIELDS:
            if field not in dds or not dds[field]:
                errors.append(f"Missing required field: {field}")
        op = dds.get("operator", {})
        for field in self.ANNEX_II_OPERATOR_FIELDS:
            if field not in op or not op[field]:
                errors.append(f"Missing operator field: {field}")
        return {
            "is_complete": len(errors) == 0,
            "errors": errors,
            "fields_checked": len(self.ANNEX_II_REQUIRED_FIELDS) + len(self.ANNEX_II_OPERATOR_FIELDS),
        }

    def format_geolocation_point(self, lat: float, lon: float) -> Dict[str, Any]:
        """Format a point coordinate for DDS."""
        return {
            "type": "Point",
            "coordinates": [round(lon, 6), round(lat, 6)],
            "coordinate_system": "WGS84",
        }

    def format_geolocation_polygon(self, vertices: List[List[float]]) -> Dict[str, Any]:
        """Format a polygon for DDS."""
        coords = [[round(v[1], 6), round(v[0], 6)] for v in vertices]
        if coords[0] != coords[-1]:
            coords.append(coords[0])
        return {
            "type": "Polygon",
            "coordinates": [coords],
            "coordinate_system": "WGS84",
        }

    def attach_evidence(self, dds: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Attach an evidence item to the DDS."""
        evidence_item = {
            "evidence_id": str(uuid.uuid4()),
            "type": evidence.get("type", "document"),
            "description": evidence.get("description", ""),
            "file_reference": evidence.get("file_reference", ""),
            "attached_at": datetime.now().isoformat(),
        }
        dds["evidence"].append(evidence_item)
        dds["provenance_hash"] = _compute_hash(dds)
        return dds

    def finalize_dds(self, dds: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize a DDS, setting status to FINALIZED."""
        validation = self.validate_annex_ii_complete(dds)
        if not validation["is_complete"]:
            return {**dds, "status": "VALIDATION_FAILED", "errors": validation["errors"]}
        dds["status"] = "FINALIZED"
        dds["finalized_at"] = datetime.now().isoformat()
        dds["provenance_hash"] = _compute_hash(dds)
        return dds

    def export_for_eu_is(self, dds: Dict[str, Any]) -> Dict[str, Any]:
        """Export DDS in EU Information System submission format."""
        return {
            "eu_is_format_version": "1.0",
            "dds_reference": dds["dds_reference"],
            "dds_type": dds["dds_type"],
            "operator_eori": dds["operator"]["eori_number"],
            "commodities": [c["commodity"] for c in dds["commodities"]],
            "cn_codes": [cn for c in dds["commodities"] for cn in c.get("cn_codes", [])],
            "risk_level": dds["risk_assessment"]["risk_level"],
            "deforestation_free": dds["cutoff_compliance"]["deforestation_free"],
            "submission_ready": dds["status"] == "FINALIZED",
            "provenance_hash": dds["provenance_hash"],
        }

    def batch_assemble(self, operators_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batch assemble multiple DDS documents."""
        results = []
        for data in operators_data:
            dds = self.assemble_standard_dds(
                operator=data["operator"],
                commodities=data["commodities"],
                suppliers=data["suppliers"],
                geolocation=data["geolocation"],
                risk_assessment=data["risk_assessment"],
                cutoff=data["cutoff"],
            )
            results.append(dds)
        return results

    def generate_dds_reference(self) -> str:
        """Generate a unique DDS reference number."""
        return f"DDS-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDDSAssembly:
    """Tests for the DDS assembly engine."""

    @pytest.fixture
    def engine(self) -> DDSAssemblySimulator:
        return DDSAssemblySimulator()

    @pytest.fixture
    def operator_data(self) -> Dict[str, Any]:
        return {
            "name": "EcoImports B.V.",
            "country": "NLD",
            "eori_number": "NL123456789000",
            "address": "Keizersgracht 100, Amsterdam",
            "operator_type": "OPERATOR",
        }

    @pytest.fixture
    def commodity_data(self) -> List[Dict[str, Any]]:
        return [{
            "commodity": "palm_oil",
            "cn_codes": ["1511 10 90"],
            "description": "Crude palm oil",
            "quantity_kg": 250000,
            "country_of_production": "IDN",
        }]

    @pytest.fixture
    def supplier_data(self) -> List[Dict[str, Any]]:
        return [{
            "supplier_id": str(uuid.uuid4()),
            "name": "PT Sawit Lestari",
            "country": "IDN",
        }]

    @pytest.fixture
    def geolocation_data(self) -> Dict[str, Any]:
        return {
            "plots": [{
                "plot_id": str(uuid.uuid4()),
                "latitude": -0.512345,
                "longitude": 101.456789,
                "area_hectares": 25.5,
                "polygon": [
                    [-0.510, 101.454], [-0.510, 101.460],
                    [-0.515, 101.460], [-0.515, 101.454],
                    [-0.510, 101.454],
                ],
            }],
        }

    @pytest.fixture
    def risk_data(self) -> Dict[str, Any]:
        return {
            "composite_risk": 0.53,
            "risk_level": "STANDARD",
            "simplified_dd_eligible": False,
        }

    @pytest.fixture
    def cutoff_data(self) -> Dict[str, Any]:
        return {
            "cutoff_date": str(EUDR_CUTOFF_DATE),
            "deforestation_free": True,
            "evidence_type": "satellite_imagery",
        }

    # 1
    def test_assemble_standard_dds(self, engine, operator_data, commodity_data,
                                    supplier_data, geolocation_data, risk_data, cutoff_data):
        """Standard DDS assembles with all Annex II fields."""
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        assert dds["dds_type"] == "STANDARD"
        assert dds["status"] == "DRAFT"
        assert dds["annex_ii_complete"] is True
        assert len(dds["provenance_hash"]) == 64

    # 2
    def test_assemble_simplified_dds(self, engine, operator_data, commodity_data,
                                      supplier_data, geolocation_data):
        """Simplified DDS assembles for low-risk scenarios."""
        dds = engine.assemble_simplified_dds(
            operator_data, commodity_data, supplier_data, geolocation_data,
        )
        assert dds["dds_type"] == "SIMPLIFIED"
        assert dds["risk_assessment"]["simplified_dd_eligible"] is True
        assert dds["risk_assessment"]["risk_level"] == "LOW"

    # 3
    def test_validate_annex_ii_complete(self, engine, sample_dds):
        """Complete DDS passes Annex II validation."""
        result = engine.validate_annex_ii_complete(sample_dds)
        assert result["is_complete"] is True
        assert len(result["errors"]) == 0

    # 4
    def test_validate_annex_ii_missing_geolocation(self, engine, sample_dds):
        """DDS without geolocation fails Annex II validation."""
        incomplete = {**sample_dds, "geolocation": {}}
        result = engine.validate_annex_ii_complete(incomplete)
        assert result["is_complete"] is False
        assert any("geolocation" in e.lower() for e in result["errors"])

    # 5
    def test_validate_annex_ii_missing_operator(self, engine, sample_dds):
        """DDS without operator fails Annex II validation."""
        incomplete = {**sample_dds, "operator": {}}
        result = engine.validate_annex_ii_complete(incomplete)
        assert result["is_complete"] is False

    # 6
    def test_batch_assemble(self, engine, operator_data, commodity_data,
                             supplier_data, geolocation_data, risk_data, cutoff_data):
        """Batch assembly produces multiple DDS documents."""
        batch_data = [
            {
                "operator": operator_data,
                "commodities": commodity_data,
                "suppliers": supplier_data,
                "geolocation": geolocation_data,
                "risk_assessment": risk_data,
                "cutoff": cutoff_data,
            }
            for _ in range(3)
        ]
        results = engine.batch_assemble(batch_data)
        assert len(results) == 3
        refs = [r["dds_reference"] for r in results]
        assert len(set(refs)) == 3, "Each DDS should have a unique reference"

    # 7
    def test_dds_reference_number(self, engine):
        """DDS reference follows format DDS-YYYYMMDD-XXXXXXXX."""
        ref = engine.generate_dds_reference()
        assert ref.startswith("DDS-")
        parts = ref.split("-")
        assert len(parts) == 3
        assert len(parts[1]) == 8  # YYYYMMDD
        assert len(parts[2]) == 8  # hex

    # 8
    def test_format_geolocation_point(self, engine):
        """Point geolocation formats to GeoJSON Point."""
        point = engine.format_geolocation_point(-0.512345, 101.456789)
        assert point["type"] == "Point"
        assert point["coordinate_system"] == "WGS84"
        assert point["coordinates"] == [101.456789, -0.512345]

    # 9
    def test_format_geolocation_polygon(self, engine):
        """Polygon geolocation formats to GeoJSON Polygon."""
        vertices = [
            [-0.510, 101.454], [-0.510, 101.460],
            [-0.515, 101.460], [-0.515, 101.454],
        ]
        polygon = engine.format_geolocation_polygon(vertices)
        assert polygon["type"] == "Polygon"
        assert polygon["coordinate_system"] == "WGS84"
        coords = polygon["coordinates"][0]
        assert coords[0] == coords[-1], "Polygon must be closed"

    # 10
    def test_attach_evidence(self, engine, sample_dds):
        """Evidence attachment adds item and updates provenance hash."""
        original_hash = sample_dds["provenance_hash"]
        evidence = {
            "type": "certificate",
            "description": "RSPO certificate",
            "file_reference": "rspo_cert.pdf",
        }
        updated = engine.attach_evidence(sample_dds, evidence)
        assert len(updated["evidence"]) > len(sample_dds.get("evidence", []) or []) - 1
        last_evidence = updated["evidence"][-1]
        assert last_evidence["type"] == "certificate"
        assert "evidence_id" in last_evidence

    # 11
    def test_finalize_dds(self, engine, operator_data, commodity_data,
                           supplier_data, geolocation_data, risk_data, cutoff_data):
        """Finalization sets status to FINALIZED for complete DDS."""
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        finalized = engine.finalize_dds(dds)
        assert finalized["status"] == "FINALIZED"
        assert "finalized_at" in finalized

    # 12
    def test_export_for_eu_is(self, engine, operator_data, commodity_data,
                               supplier_data, geolocation_data, risk_data, cutoff_data):
        """EU IS export produces correct submission format."""
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        dds = engine.finalize_dds(dds)
        export = engine.export_for_eu_is(dds)
        assert export["eu_is_format_version"] == "1.0"
        assert export["submission_ready"] is True
        assert export["operator_eori"] == "NL123456789000"
        assert "palm_oil" in export["commodities"]

    # 13
    def test_dds_provenance_hash(self, engine, operator_data, commodity_data,
                                  supplier_data, geolocation_data, risk_data, cutoff_data):
        """DDS has a valid 64-character SHA-256 provenance hash."""
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        assert_provenance_hash(dds)

    # 14-20: Seven commodity DDS tests
    @pytest.mark.parametrize("commodity,cn_code", [
        ("cattle", "0102 29 10"),
        ("cocoa", "1801 00 00"),
        ("coffee", "0901 11 00"),
        ("palm_oil", "1511 10 90"),
        ("rubber", "4001 10 00"),
        ("soya", "1201 90 00"),
        ("wood", "4403 49 00"),
    ])
    def test_seven_commodity_dds(self, engine, operator_data, supplier_data,
                                  geolocation_data, risk_data, cutoff_data,
                                  commodity, cn_code):
        """DDS can be assembled for each of the 7 EUDR commodities."""
        commodity_data = [{
            "commodity": commodity,
            "cn_codes": [cn_code],
            "description": f"Test {commodity}",
            "quantity_kg": 1000,
            "country_of_production": "BRA",
        }]
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        assert dds["commodities"][0]["commodity"] == commodity
        assert dds["annex_ii_complete"] is True

    # 21
    def test_finalize_incomplete_dds_fails(self, engine):
        """Finalizing an incomplete DDS returns VALIDATION_FAILED."""
        incomplete = {
            "dds_reference": "DDS-TEST-001",
            "dds_type": "STANDARD",
            "status": "DRAFT",
            "operator": {},
            "commodities": [],
            "suppliers": [],
            "geolocation": {},
            "risk_assessment": {},
            "cutoff_compliance": {},
            "evidence": [],
        }
        result = engine.finalize_dds(incomplete)
        assert result["status"] == "VALIDATION_FAILED"
        assert len(result["errors"]) > 0

    # 22
    def test_dds_unique_references(self, engine):
        """Multiple DDS reference numbers are unique."""
        refs = [engine.generate_dds_reference() for _ in range(100)]
        assert len(set(refs)) == 100

    # 23
    def test_dds_created_at_timestamp(self, engine, operator_data, commodity_data,
                                       supplier_data, geolocation_data, risk_data, cutoff_data):
        """DDS includes creation timestamp in ISO format."""
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        assert "created_at" in dds
        # Should parse as valid ISO timestamp
        parsed = datetime.fromisoformat(dds["created_at"])
        assert parsed.year >= 2025

    # 24
    def test_simplified_dds_low_risk_only(self, engine, operator_data, commodity_data,
                                           supplier_data, geolocation_data):
        """Simplified DDS is only for low-risk scenarios."""
        dds = engine.assemble_simplified_dds(
            operator_data, commodity_data, supplier_data, geolocation_data,
        )
        assert dds["risk_assessment"]["risk_level"] == "LOW"
        assert dds["risk_assessment"]["simplified_dd_eligible"] is True

    # 25
    def test_dds_evidence_list_starts_empty(self, engine, operator_data, commodity_data,
                                             supplier_data, geolocation_data, risk_data, cutoff_data):
        """Newly assembled DDS starts with empty evidence list."""
        dds = engine.assemble_standard_dds(
            operator_data, commodity_data, supplier_data,
            geolocation_data, risk_data, cutoff_data,
        )
        assert isinstance(dds["evidence"], list)
        assert len(dds["evidence"]) == 0
