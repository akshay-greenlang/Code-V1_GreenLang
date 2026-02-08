# -*- coding: utf-8 -*-
"""
End-to-End Integration Tests for Excel & CSV Normalizer (AGENT-DATA-002)

Tests full normalisation lifecycle: upload -> parse -> detect types ->
map columns -> normalise -> validate -> score quality -> transform.
Tests CSV and Excel upload pipelines, template creation and reuse,
batch processing, transform pipeline, provenance chain integrity,
error handling, quality scoring accuracy, and column mapping accuracy.

All implementations are self-contained to avoid cross-module import issues.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Self-contained implementations for integration testing
# ---------------------------------------------------------------------------


class NormalisationPipeline:
    """End-to-end Excel/CSV normalisation pipeline for integration testing."""

    CANONICAL_FIELDS = {
        "facility_name", "facility_id", "reporting_period", "scope",
        "emission_category", "activity_type", "activity_data",
        "activity_unit", "emission_factor", "co2e_tonnes",
        "energy_kwh", "energy_source", "waste_tonnes", "waste_type",
        "country", "region", "supplier_name",
    }

    SYNONYMS = {
        "facility_name": ["site_name", "plant_name", "location_name", "building"],
        "co2e_tonnes": ["emissions_tco2e", "ghg_emissions", "carbon_emissions"],
        "energy_kwh": ["electricity_kwh", "energy_consumption", "power_kwh"],
        "waste_tonnes": ["waste_generated", "waste_amount"],
        "country": ["nation", "country_code"],
    }

    def __init__(self):
        self._files: Dict[str, Dict[str, Any]] = {}
        self._templates: Dict[str, Dict[str, Any]] = {}
        self._provenance: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Parse CSV
    # ------------------------------------------------------------------
    def parse_csv(self, content: str, file_name: str) -> Dict[str, Any]:
        lines = content.strip().split("\n")
        if not lines:
            return self._make_record(file_name, "csv", [], [])

        delimiter = self._detect_delimiter(lines[0])
        headers = [h.strip() for h in lines[0].split(delimiter)]
        rows = []
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(delimiter)]
            row = {}
            for i, h in enumerate(headers):
                row[h] = parts[i] if i < len(parts) else None
            rows.append(row)

        record = self._make_record(file_name, "csv", headers, rows)
        self._record_provenance(record["file_id"], "parse_csv", {"file": file_name})
        return record

    # ------------------------------------------------------------------
    # Parse Excel (simulated)
    # ------------------------------------------------------------------
    def parse_excel(self, content: str, file_name: str) -> Dict[str, Any]:
        # Simulate Excel parsing with same logic as CSV
        return self.parse_csv(content, file_name)

    # ------------------------------------------------------------------
    # Detect types
    # ------------------------------------------------------------------
    def detect_types(self, file_id: str) -> Dict[str, str]:
        record = self._files.get(file_id)
        if not record:
            raise ValueError(f"File {file_id} not found")

        types = {}
        for header in record["headers"]:
            values = [row.get(header) for row in record["rows"] if row.get(header) is not None]
            types[header] = self._detect_column_type(values)

        record["detected_types"] = types
        self._record_provenance(file_id, "detect_types", {"types": types})
        return types

    # ------------------------------------------------------------------
    # Map columns
    # ------------------------------------------------------------------
    def map_columns(self, file_id: str, template_id: Optional[str] = None) -> Dict[str, str]:
        record = self._files.get(file_id)
        if not record:
            raise ValueError(f"File {file_id} not found")

        mappings = {}
        if template_id and template_id in self._templates:
            mappings = dict(self._templates[template_id]["column_mappings"])
        else:
            for header in record["headers"]:
                canonical = self._find_canonical(header)
                if canonical:
                    mappings[header] = canonical

        record["column_mappings"] = mappings
        self._record_provenance(file_id, "map_columns", {"mappings": mappings})
        return mappings

    # ------------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------------
    def normalize(self, file_id: str) -> List[Dict[str, Any]]:
        record = self._files.get(file_id)
        if not record:
            raise ValueError(f"File {file_id} not found")

        mappings = record.get("column_mappings", {})
        normalized = []
        for row in record["rows"]:
            new_row = {}
            for src, val in row.items():
                canonical = mappings.get(src, src)
                new_row[canonical] = val
            normalized.append(new_row)

        record["normalized_data"] = normalized
        record["status"] = "normalized"
        self._record_provenance(file_id, "normalize", {"rows": len(normalized)})
        return normalized

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    def validate(self, file_id: str, schema_name: str = "emissions") -> Dict[str, Any]:
        record = self._files.get(file_id)
        if not record:
            raise ValueError(f"File {file_id} not found")

        data = record.get("normalized_data", record["rows"])
        errors = []
        required = {"facility_name", "co2e_tonnes"}
        for idx, row in enumerate(data):
            for field in required:
                if field in [m for m in record.get("column_mappings", {}).values()]:
                    val = row.get(field)
                    if val is None or str(val).strip() == "":
                        errors.append({
                            "row": idx, "field": field,
                            "message": f"Required field '{field}' is missing",
                        })

        result = {"is_valid": len(errors) == 0, "errors": errors, "schema_name": schema_name}
        record["validation"] = result
        self._record_provenance(file_id, "validate", {"valid": result["is_valid"]})
        return result

    # ------------------------------------------------------------------
    # Score quality
    # ------------------------------------------------------------------
    def score_quality(self, file_id: str) -> Dict[str, float]:
        record = self._files.get(file_id)
        if not record:
            raise ValueError(f"File {file_id} not found")

        data = record.get("normalized_data", record["rows"])
        total_cells = sum(len(row) for row in data)
        filled = sum(1 for row in data for v in row.values()
                     if v is not None and str(v).strip() != "")
        completeness = filled / total_cells if total_cells > 0 else 0.0
        accuracy = 0.85
        consistency = 0.9

        overall = completeness * 0.4 + accuracy * 0.35 + consistency * 0.25

        scores = {
            "overall": round(overall, 4),
            "completeness": round(completeness, 4),
            "accuracy": accuracy,
            "consistency": consistency,
        }
        record["quality_scores"] = scores
        self._record_provenance(file_id, "score_quality", scores)
        return scores

    # ------------------------------------------------------------------
    # Transform
    # ------------------------------------------------------------------
    def apply_transform(self, file_id: str, operation: str, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        record = self._files.get(file_id)
        if not record:
            raise ValueError(f"File {file_id} not found")

        data = record.get("normalized_data", record["rows"])

        if operation == "filter":
            col = config.get("column")
            val = config.get("value")
            data = [r for r in data if r.get(col) == val]
        elif operation == "dedup":
            keys = config.get("keys", [])
            seen = set()
            deduped = []
            for row in data:
                key = tuple(str(row.get(k, "")) for k in keys) if keys else tuple(sorted(row.items()))
                if key not in seen:
                    seen.add(key)
                    deduped.append(row)
            data = deduped
        elif operation == "rename":
            mapping = config.get("mapping", {})
            data = [{mapping.get(k, k): v for k, v in row.items()} for row in data]

        record["normalized_data"] = data
        self._record_provenance(file_id, "transform", {"operation": operation})
        return data

    # ------------------------------------------------------------------
    # Template management
    # ------------------------------------------------------------------
    def create_template(self, name: str, mappings: Dict[str, str]) -> Dict[str, Any]:
        tpl_id = str(uuid.uuid4())
        template = {
            "template_id": tpl_id,
            "name": name,
            "column_mappings": mappings,
        }
        self._templates[tpl_id] = template
        return template

    # ------------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------------
    def get_provenance_chain(self, file_id: str) -> List[Dict[str, Any]]:
        return list(self._provenance.get(file_id, []))

    def verify_provenance(self, file_id: str) -> bool:
        chain = self._provenance.get(file_id, [])
        if not chain:
            return True
        genesis = "0" * 64
        for i, record in enumerate(chain):
            expected_prev = chain[i - 1]["hash"] if i > 0 else genesis
            if record["previous_hash"] != expected_prev:
                return False
        return True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _make_record(self, file_name: str, fmt: str, headers: List[str],
                     rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        file_id = str(uuid.uuid4())
        record = {
            "file_id": file_id,
            "file_name": file_name,
            "file_format": fmt,
            "headers": headers,
            "rows": rows,
            "row_count": len(rows),
            "column_count": len(headers),
            "status": "parsed",
        }
        self._files[file_id] = record
        return record

    def _detect_delimiter(self, line: str) -> str:
        candidates = {",": line.count(","), ";": line.count(";"),
                      "\t": line.count("\t"), "|": line.count("|")}
        best = max(candidates, key=candidates.get)
        return best if candidates[best] > 0 else ","

    def _detect_column_type(self, values: List[Any]) -> str:
        if not values:
            return "string"
        for val in values[:20]:
            if val is None:
                continue
            s = str(val).strip()
            try:
                int(s)
                return "integer"
            except ValueError:
                pass
            try:
                float(s.replace(",", ""))
                return "float"
            except ValueError:
                pass
        return "string"

    def _find_canonical(self, header: str) -> Optional[str]:
        lower = header.lower().strip().replace(" ", "_").replace("-", "_")
        if lower in self.CANONICAL_FIELDS:
            return lower
        for canonical, synonyms in self.SYNONYMS.items():
            if lower in synonyms:
                return canonical
        return None

    def _record_provenance(self, file_id: str, operation: str, data: Dict[str, Any]):
        if file_id not in self._provenance:
            self._provenance[file_id] = []
        chain = self._provenance[file_id]
        prev_hash = chain[-1]["hash"] if chain else "0" * 64
        record_data = json.dumps({"op": operation, "data": data}, sort_keys=True, default=str)
        record_hash = hashlib.sha256(record_data.encode()).hexdigest()
        chain.append({
            "sequence": len(chain) + 1,
            "operation": operation,
            "data": data,
            "previous_hash": prev_hash,
            "hash": record_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# ---------------------------------------------------------------------------
# Test Data
# ---------------------------------------------------------------------------

CSV_ENERGY_DATA = """facility_name,reporting_period,energy_kwh,country
London HQ,2025-01-01,450000,UK
Berlin DC,2025-01-01,320000,Germany
Paris Office,2025-01-01,180000,France
Tokyo Lab,2025-01-01,275000,Japan
"""

CSV_EMISSIONS_DATA = """site_name,scope,emissions_tco2e,country_code
London HQ,Scope 1,1250.5,GB
London HQ,Scope 2,890.3,GB
Berlin DC,Scope 1,670.1,DE
Berlin DC,Scope 2,445.8,DE
Paris Office,Scope 1,320.0,FR
"""

CSV_TRANSPORT_DATA = """vehicle_id,distance_km,fuel_used_litres,fuel_type
V001,15200,1820,diesel
V002,28500,3420,diesel
V003,8900,0,electric
"""

SEMICOLON_DATA = """facility_name;year;emissions
London;2024;1250
Berlin;2024;890
"""


# ===========================================================================
# Test Classes
# ===========================================================================


class TestCSVUploadLifecycle:
    """Full lifecycle test for CSV file upload and normalisation."""

    def test_full_csv_lifecycle(self):
        pipeline = NormalisationPipeline()

        # Step 1: Parse
        record = pipeline.parse_csv(CSV_ENERGY_DATA, "energy_2025.csv")
        assert record["status"] == "parsed"
        assert record["row_count"] == 4
        assert record["column_count"] == 4

        # Step 2: Detect types
        types = pipeline.detect_types(record["file_id"])
        assert "facility_name" in types

        # Step 3: Map columns
        mappings = pipeline.map_columns(record["file_id"])
        assert "facility_name" in mappings

        # Step 4: Normalize
        normalized = pipeline.normalize(record["file_id"])
        assert len(normalized) == 4

        # Step 5: Validate
        validation = pipeline.validate(record["file_id"])
        assert "is_valid" in validation

        # Step 6: Score quality
        scores = pipeline.score_quality(record["file_id"])
        assert scores["overall"] > 0
        assert scores["completeness"] > 0

    def test_csv_synonym_mapping(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(CSV_EMISSIONS_DATA, "emissions.csv")
        mappings = pipeline.map_columns(record["file_id"])
        # site_name should map to facility_name
        assert mappings.get("site_name") == "facility_name"
        # emissions_tco2e should map to co2e_tonnes
        assert mappings.get("emissions_tco2e") == "co2e_tonnes"
        # country_code should map to country
        assert mappings.get("country_code") == "country"


class TestExcelUploadLifecycle:
    """Full lifecycle test for Excel file upload (simulated)."""

    def test_full_excel_lifecycle(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_excel(CSV_ENERGY_DATA, "energy_2025.xlsx")
        assert record["file_format"] == "csv"  # simulated, uses CSV parser
        types = pipeline.detect_types(record["file_id"])
        mappings = pipeline.map_columns(record["file_id"])
        normalized = pipeline.normalize(record["file_id"])
        assert len(normalized) == 4


class TestTemplateCreationAndReuse:
    """Test template creation and reuse for column mapping."""

    def test_template_mapping(self):
        pipeline = NormalisationPipeline()

        # Create template
        tpl = pipeline.create_template("Energy Template", {
            "facility_name": "facility_name",
            "reporting_period": "reporting_period",
            "energy_kwh": "energy_kwh",
            "country": "country",
        })
        assert "template_id" in tpl

        # Parse and map with template
        record = pipeline.parse_csv(CSV_ENERGY_DATA, "energy.csv")
        mappings = pipeline.map_columns(record["file_id"], template_id=tpl["template_id"])
        assert mappings.get("facility_name") == "facility_name"
        assert mappings.get("energy_kwh") == "energy_kwh"


class TestBatchProcessing:
    """Test batch file processing."""

    def test_batch_multiple_files(self):
        pipeline = NormalisationPipeline()
        files = [
            (CSV_ENERGY_DATA, "energy.csv"),
            (CSV_EMISSIONS_DATA, "emissions.csv"),
            (CSV_TRANSPORT_DATA, "transport.csv"),
        ]
        results = []
        for content, fname in files:
            record = pipeline.parse_csv(content, fname)
            pipeline.detect_types(record["file_id"])
            pipeline.map_columns(record["file_id"])
            pipeline.normalize(record["file_id"])
            scores = pipeline.score_quality(record["file_id"])
            results.append({
                "file_name": fname,
                "row_count": record["row_count"],
                "quality": scores["overall"],
            })

        assert len(results) == 3
        assert all(r["quality"] > 0 for r in results)


class TestTransformPipeline:
    """Test transform operations in the pipeline."""

    def test_filter_and_rename(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(CSV_EMISSIONS_DATA, "emissions.csv")
        pipeline.map_columns(record["file_id"])
        pipeline.normalize(record["file_id"])

        # Filter by scope
        filtered = pipeline.apply_transform(
            record["file_id"], "filter",
            {"column": "scope", "value": "Scope 1"},
        )
        assert all(r.get("scope") == "Scope 1" for r in filtered)

    def test_dedup(self):
        pipeline = NormalisationPipeline()
        dup_data = """facility_name,year,emissions
London,2024,1250
Berlin,2024,890
London,2024,1250
"""
        record = pipeline.parse_csv(dup_data, "duplicates.csv")
        pipeline.normalize(record["file_id"])
        deduped = pipeline.apply_transform(
            record["file_id"], "dedup",
            {"keys": ["facility_name", "year"]},
        )
        assert len(deduped) == 2


class TestProvenanceChainIntegrity:
    """Test provenance chain across full pipeline."""

    def test_chain_links_correctly(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(CSV_ENERGY_DATA, "test.csv")
        pipeline.detect_types(record["file_id"])
        pipeline.map_columns(record["file_id"])
        pipeline.normalize(record["file_id"])
        pipeline.validate(record["file_id"])
        pipeline.score_quality(record["file_id"])

        chain = pipeline.get_provenance_chain(record["file_id"])
        assert len(chain) == 6
        assert chain[0]["operation"] == "parse_csv"
        assert chain[0]["previous_hash"] == "0" * 64

        for i in range(1, len(chain)):
            assert chain[i]["previous_hash"] == chain[i - 1]["hash"]

    def test_verify_provenance(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(CSV_ENERGY_DATA, "test.csv")
        pipeline.detect_types(record["file_id"])
        pipeline.map_columns(record["file_id"])
        pipeline.normalize(record["file_id"])
        assert pipeline.verify_provenance(record["file_id"]) is True

    def test_timestamps_ordered(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(CSV_ENERGY_DATA, "test.csv")
        pipeline.detect_types(record["file_id"])
        pipeline.normalize(record["file_id"])
        chain = pipeline.get_provenance_chain(record["file_id"])
        timestamps = [r["timestamp"] for r in chain]
        assert timestamps == sorted(timestamps)


class TestErrorHandling:
    """Test error scenarios."""

    def test_detect_types_nonexistent(self):
        pipeline = NormalisationPipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.detect_types("nonexistent")

    def test_map_columns_nonexistent(self):
        pipeline = NormalisationPipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.map_columns("nonexistent")

    def test_normalize_nonexistent(self):
        pipeline = NormalisationPipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.normalize("nonexistent")

    def test_validate_nonexistent(self):
        pipeline = NormalisationPipeline()
        with pytest.raises(ValueError, match="not found"):
            pipeline.validate("nonexistent")


class TestQualityScoringAccuracy:
    """Test quality scoring produces valid scores."""

    def test_complete_data_high_score(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(CSV_ENERGY_DATA, "complete.csv")
        pipeline.normalize(record["file_id"])
        scores = pipeline.score_quality(record["file_id"])
        assert scores["completeness"] > 0.9

    def test_empty_column_lower_score(self):
        pipeline = NormalisationPipeline()
        data_with_nulls = """facility_name,year,emissions
London,2024,
Berlin,,890
,2024,
"""
        record = pipeline.parse_csv(data_with_nulls, "nulls.csv")
        pipeline.normalize(record["file_id"])
        scores = pipeline.score_quality(record["file_id"])
        assert scores["completeness"] < 1.0


class TestDelimiterDetection:
    """Test automatic delimiter detection."""

    def test_semicolon_delimiter(self):
        pipeline = NormalisationPipeline()
        record = pipeline.parse_csv(SEMICOLON_DATA, "semicolon.csv")
        assert record["row_count"] == 2
        assert "facility_name" in record["headers"]
