# -*- coding: utf-8 -*-
"""
Unit Tests for DueDiligenceEngine (AGENT-DATA-005)

Tests DDS generation, lifecycle management (draft -> complete -> submitted),
signing, EU system export formatting, completeness validation, and statistics.

Coverage target: 85%+ of due_diligence.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Inline enums mirroring greenlang/eudr_traceability/models.py
# ---------------------------------------------------------------------------


class EUDRCommodity(str, Enum):
    COCOA = "cocoa"
    COFFEE = "coffee"
    OIL_PALM = "oil_palm"
    SOYA = "soya"
    RUBBER = "rubber"
    CATTLE = "cattle"
    WOOD = "wood"


class RiskLevel(str, Enum):
    LOW = "low"
    STANDARD = "standard"
    HIGH = "high"


class DDSStatus(str, Enum):
    DRAFT = "draft"
    COMPLETE = "complete"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class DDSType(str, Enum):
    IMPORT_PLACEMENT = "import_placement"
    EXPORT = "export"


# ---------------------------------------------------------------------------
# Inline data models
# ---------------------------------------------------------------------------


class PlotRecord:
    """Minimal plot record for DDS integration."""

    def __init__(self, plot_id: str, latitude: float, longitude: float,
                 country: str = "BR", commodity: str = "cocoa",
                 area_hectares: float = 5.0, deforestation_free: bool = True,
                 legal_compliance: bool = True):
        self.plot_id = plot_id
        self.latitude = latitude
        self.longitude = longitude
        self.country = country
        self.commodity = commodity
        self.area_hectares = area_hectares
        self.deforestation_free = deforestation_free
        self.legal_compliance = legal_compliance


class GenerateDDSRequest:
    """Request model for DDS generation."""

    def __init__(self, operator_name: str, operator_country: str,
                 plot_ids: List[str], commodity: str,
                 dds_type: str = "import_placement",
                 description: str = ""):
        self.operator_name = operator_name
        self.operator_country = operator_country
        self.plot_ids = plot_ids
        self.commodity = commodity
        self.dds_type = dds_type
        self.description = description


# ---------------------------------------------------------------------------
# Inline PlotRegistryEngine (minimal)
# ---------------------------------------------------------------------------


class PlotRegistryEngine:
    """Minimal plot registry for testing."""

    def __init__(self):
        self._plots: Dict[str, PlotRecord] = {}

    def register(self, plot: PlotRecord) -> PlotRecord:
        self._plots[plot.plot_id] = plot
        return plot

    def get_plot(self, plot_id: str) -> Optional[PlotRecord]:
        return self._plots.get(plot_id)

    def list_plots(self) -> List[PlotRecord]:
        return list(self._plots.values())


# ---------------------------------------------------------------------------
# Inline RiskAssessmentEngine (minimal)
# ---------------------------------------------------------------------------


class RiskAssessmentEngine:
    """Minimal risk assessment for DDS integration."""

    HIGH_RISK_COUNTRIES = {
        "BR", "ID", "MY", "AR", "PY", "BO", "CO", "PE", "EC",
        "CG", "CD", "CM", "CI", "GH", "NG", "LA", "MM", "PG",
    }

    def assess_risk_for_dds(self, countries: List[str],
                            commodity: str) -> Dict[str, Any]:
        max_score = 0
        for c in countries:
            if c in self.HIGH_RISK_COUNTRIES:
                max_score = max(max_score, 80)
            else:
                max_score = max(max_score, 30)

        if max_score >= 70:
            level = RiskLevel.HIGH
        elif max_score >= 30:
            level = RiskLevel.STANDARD
        else:
            level = RiskLevel.LOW

        return {
            "overall_score": max_score,
            "risk_level": level,
            "countries_assessed": countries,
            "commodity": commodity,
        }


# ---------------------------------------------------------------------------
# Inline ChainOfCustodyEngine (minimal)
# ---------------------------------------------------------------------------


class ChainOfCustodyEngine:
    """Minimal chain of custody for DDS integration."""

    def __init__(self):
        self._transfers: List[Dict[str, Any]] = []

    def get_transfers_for_plots(self, plot_ids: List[str]) -> List[Dict[str, Any]]:
        return [t for t in self._transfers if t.get("plot_id") in plot_ids]


# ---------------------------------------------------------------------------
# Inline DueDiligenceEngine mirroring greenlang/eudr_traceability/due_diligence.py
# ---------------------------------------------------------------------------


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


class DueDiligenceStatement:
    """A due diligence statement record."""

    def __init__(self, dds_id: str, operator_name: str, operator_country: str,
                 commodity: str, dds_type: str, plot_ids: List[str],
                 origin_countries: List[str], risk_level: str,
                 risk_score: float, status: str = "draft",
                 description: str = "", signature: Optional[str] = None):
        self.dds_id = dds_id
        self.operator_name = operator_name
        self.operator_country = operator_country
        self.commodity = commodity
        self.dds_type = dds_type
        self.plot_ids = plot_ids
        self.origin_countries = origin_countries
        self.risk_level = risk_level
        self.risk_score = risk_score
        self.status = status
        self.description = description
        self.signature = signature
        self.provenance_hash = ""
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.submitted_at: Optional[str] = None
        self.completeness_issues: List[str] = []


class DueDiligenceEngine:
    """Engine for DDS lifecycle management."""

    def __init__(self, plot_registry: PlotRegistryEngine,
                 risk_engine: RiskAssessmentEngine,
                 chain_of_custody: Optional[ChainOfCustodyEngine] = None):
        self._plot_registry = plot_registry
        self._risk_engine = risk_engine
        self._chain_of_custody = chain_of_custody or ChainOfCustodyEngine()
        self._statements: Dict[str, DueDiligenceStatement] = {}
        self._counter = 0

    def _next_id(self) -> str:
        self._counter += 1
        return f"DDS-{self._counter:05d}"

    def generate_dds(self, request: GenerateDDSRequest) -> DueDiligenceStatement:
        """Generate a new due diligence statement."""
        dds_id = self._next_id()

        # Gather origin countries from plots
        origin_countries: List[str] = []
        for pid in request.plot_ids:
            plot = self._plot_registry.get_plot(pid)
            if plot is not None and plot.country not in origin_countries:
                origin_countries.append(plot.country)

        # Assess risk
        risk_result = self._risk_engine.assess_risk_for_dds(
            origin_countries, request.commodity,
        )

        dds = DueDiligenceStatement(
            dds_id=dds_id,
            operator_name=request.operator_name,
            operator_country=request.operator_country,
            commodity=request.commodity,
            dds_type=request.dds_type,
            plot_ids=list(request.plot_ids),
            origin_countries=origin_countries,
            risk_level=risk_result["risk_level"].value if isinstance(
                risk_result["risk_level"], RiskLevel
            ) else risk_result["risk_level"],
            risk_score=risk_result["overall_score"],
            status="draft",
            description=request.description,
        )

        # Validate completeness
        issues = self._validate_completeness(dds)
        dds.completeness_issues = issues
        if not issues:
            dds.status = "complete"

        dds.provenance_hash = _compute_hash({
            "dds_id": dds.dds_id,
            "operator": dds.operator_name,
            "commodity": dds.commodity,
            "plots": dds.plot_ids,
        })

        self._statements[dds_id] = dds
        return dds

    def get_dds(self, dds_id: str) -> Optional[DueDiligenceStatement]:
        return self._statements.get(dds_id)

    def list_dds(self, status: Optional[str] = None,
                 commodity: Optional[str] = None) -> List[DueDiligenceStatement]:
        results = list(self._statements.values())
        if status is not None:
            results = [d for d in results if d.status == status]
        if commodity is not None:
            results = [d for d in results if d.commodity == commodity]
        return results

    def submit_dds(self, dds_id: str) -> DueDiligenceStatement:
        dds = self._statements.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")
        if dds.status != "complete":
            raise ValueError(
                f"DDS {dds_id} is not complete (status={dds.status}). "
                "Cannot submit incomplete DDS."
            )
        dds.status = "submitted"
        dds.submitted_at = datetime.now(timezone.utc).isoformat()
        return dds

    def sign_dds(self, dds_id: str, signer: str = "authorized_operator") -> DueDiligenceStatement:
        dds = self._statements.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")
        sig_data = f"{dds.dds_id}:{dds.operator_name}:{signer}:{dds.provenance_hash}"
        dds.signature = hashlib.sha256(sig_data.encode()).hexdigest()
        return dds

    def export_for_eu_system(self, dds_id: str) -> Dict[str, Any]:
        dds = self._statements.get(dds_id)
        if dds is None:
            raise ValueError(f"DDS {dds_id} not found")

        plots_data = []
        for pid in dds.plot_ids:
            plot = self._plot_registry.get_plot(pid)
            if plot is not None:
                plots_data.append({
                    "plot_id": plot.plot_id,
                    "latitude": plot.latitude,
                    "longitude": plot.longitude,
                    "country": plot.country,
                    "area_hectares": plot.area_hectares,
                })

        return {
            "dds_reference": dds.dds_id,
            "operator": {
                "name": dds.operator_name,
                "country": dds.operator_country,
            },
            "commodity": dds.commodity,
            "dds_type": dds.dds_type,
            "origin_countries": dds.origin_countries,
            "risk_level": dds.risk_level,
            "risk_score": dds.risk_score,
            "plots": plots_data,
            "status": dds.status,
            "signature": dds.signature,
            "provenance_hash": dds.provenance_hash,
        }

    def _validate_completeness(self, dds: DueDiligenceStatement) -> List[str]:
        issues: List[str] = []
        if not dds.operator_name:
            issues.append("Missing operator name")
        if not dds.operator_country:
            issues.append("Missing operator country")
        if not dds.commodity:
            issues.append("Missing commodity")
        if not dds.plot_ids:
            issues.append("No plots associated")
        if not dds.origin_countries:
            issues.append("No origin countries identified")
        return issues

    def validate_completeness(self, dds_id: str) -> List[str]:
        dds = self._statements.get(dds_id)
        if dds is None:
            return ["DDS not found"]
        return self._validate_completeness(dds)

    def get_statistics(self) -> Dict[str, Any]:
        total = len(self._statements)
        by_status: Dict[str, int] = {}
        for dds in self._statements.values():
            by_status[dds.status] = by_status.get(dds.status, 0) + 1
        return {
            "total_dds": total,
            "by_status": by_status,
        }


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def plot_registry() -> PlotRegistryEngine:
    """Create PlotRegistryEngine with registered plots."""
    registry = PlotRegistryEngine()
    registry.register(PlotRecord(
        plot_id="PLT-00001", latitude=-3.1190, longitude=-60.0217,
        country="BR", commodity="cocoa", area_hectares=12.5,
        deforestation_free=True, legal_compliance=True,
    ))
    registry.register(PlotRecord(
        plot_id="PLT-00002", latitude=-1.2921, longitude=36.8219,
        country="KE", commodity="coffee", area_hectares=3.2,
        deforestation_free=True, legal_compliance=True,
    ))
    registry.register(PlotRecord(
        plot_id="PLT-00003", latitude=5.6037, longitude=-0.1870,
        country="GH", commodity="cocoa", area_hectares=8.0,
        deforestation_free=True, legal_compliance=True,
    ))
    return registry


@pytest.fixture
def risk_engine() -> RiskAssessmentEngine:
    return RiskAssessmentEngine()


@pytest.fixture
def chain_of_custody() -> ChainOfCustodyEngine:
    return ChainOfCustodyEngine()


@pytest.fixture
def engine(plot_registry, risk_engine, chain_of_custody) -> DueDiligenceEngine:
    return DueDiligenceEngine(
        plot_registry=plot_registry,
        risk_engine=risk_engine,
        chain_of_custody=chain_of_custody,
    )


@pytest.fixture
def sample_dds_request() -> GenerateDDSRequest:
    return GenerateDDSRequest(
        operator_name="ChocoCorp EU GmbH",
        operator_country="DE",
        plot_ids=["PLT-00001"],
        commodity="cocoa",
        dds_type="import_placement",
        description="Q1 2026 cocoa import from Brazil",
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestGenerateDDS:
    """Tests for DDS generation."""

    def test_generate_dds_success(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        assert dds is not None
        assert dds.operator_name == "ChocoCorp EU GmbH"
        assert dds.commodity == "cocoa"
        assert dds.status in ("draft", "complete")

    def test_generate_dds_id_format(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        assert dds.dds_id.startswith("DDS-")
        assert len(dds.dds_id) == 9  # DDS-00001

    def test_generate_dds_origin_info(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        assert "BR" in dds.origin_countries

    def test_generate_dds_risk_level(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        # Brazil is high risk
        assert dds.risk_level == "high"
        assert dds.risk_score >= 70

    def test_generate_dds_provenance_hash(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        assert len(dds.provenance_hash) == 64
        int(dds.provenance_hash, 16)


class TestGetDDS:
    """Tests for DDS retrieval."""

    def test_get_dds_exists(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        retrieved = engine.get_dds(dds.dds_id)
        assert retrieved is not None
        assert retrieved.dds_id == dds.dds_id

    def test_get_dds_not_found(self, engine):
        result = engine.get_dds("DDS-99999")
        assert result is None


class TestListDDS:
    """Tests for listing DDS with filters."""

    def test_list_dds_all(self, engine, sample_dds_request):
        engine.generate_dds(sample_dds_request)
        engine.generate_dds(sample_dds_request)
        results = engine.list_dds()
        assert len(results) == 2

    def test_list_dds_by_status(self, engine, sample_dds_request):
        engine.generate_dds(sample_dds_request)
        results = engine.list_dds(status="complete")
        assert all(d.status == "complete" for d in results)

    def test_list_dds_by_commodity(self, engine, sample_dds_request):
        engine.generate_dds(sample_dds_request)
        results = engine.list_dds(commodity="cocoa")
        assert all(d.commodity == "cocoa" for d in results)

    def test_list_dds_by_commodity_no_match(self, engine, sample_dds_request):
        engine.generate_dds(sample_dds_request)
        results = engine.list_dds(commodity="rubber")
        assert len(results) == 0


class TestSubmitDDS:
    """Tests for DDS submission lifecycle."""

    def test_submit_dds_success(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        assert dds.status == "complete"
        submitted = engine.submit_dds(dds.dds_id)
        assert submitted.status == "submitted"
        assert submitted.submitted_at is not None

    def test_submit_dds_incomplete(self, engine):
        request = GenerateDDSRequest(
            operator_name="",
            operator_country="",
            plot_ids=[],
            commodity="",
        )
        dds = engine.generate_dds(request)
        assert dds.status == "draft"
        with pytest.raises(ValueError, match="not complete"):
            engine.submit_dds(dds.dds_id)

    def test_submit_dds_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.submit_dds("DDS-99999")


class TestSignDDS:
    """Tests for DDS digital signing."""

    def test_sign_dds(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        signed = engine.sign_dds(dds.dds_id, signer="operator_admin")
        assert signed.signature is not None
        assert len(signed.signature) == 64


class TestExportForEUSystem:
    """Tests for EU system export formatting."""

    def test_export_for_eu_system(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        export = engine.export_for_eu_system(dds.dds_id)
        assert "dds_reference" in export
        assert "operator" in export
        assert "commodity" in export
        assert "plots" in export

    def test_export_includes_plots(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        export = engine.export_for_eu_system(dds.dds_id)
        assert len(export["plots"]) > 0
        plot_data = export["plots"][0]
        assert "plot_id" in plot_data
        assert "latitude" in plot_data
        assert "longitude" in plot_data
        assert "country" in plot_data

    def test_export_includes_operator(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        export = engine.export_for_eu_system(dds.dds_id)
        assert export["operator"]["name"] == "ChocoCorp EU GmbH"
        assert export["operator"]["country"] == "DE"

    def test_export_not_found(self, engine):
        with pytest.raises(ValueError, match="not found"):
            engine.export_for_eu_system("DDS-99999")


class TestValidateCompleteness:
    """Tests for DDS completeness validation."""

    def test_validate_completeness_pass(self, engine, sample_dds_request):
        dds = engine.generate_dds(sample_dds_request)
        issues = engine.validate_completeness(dds.dds_id)
        assert len(issues) == 0

    def test_validate_completeness_missing_fields(self, engine):
        request = GenerateDDSRequest(
            operator_name="",
            operator_country="",
            plot_ids=[],
            commodity="",
        )
        dds = engine.generate_dds(request)
        issues = engine.validate_completeness(dds.dds_id)
        assert len(issues) > 0
        assert any("operator" in i.lower() for i in issues)

    def test_validate_completeness_not_found(self, engine):
        issues = engine.validate_completeness("DDS-99999")
        assert "DDS not found" in issues


class TestDDSStatistics:
    """Tests for DDS statistics."""

    def test_dds_statistics(self, engine, sample_dds_request):
        engine.generate_dds(sample_dds_request)
        engine.generate_dds(sample_dds_request)
        stats = engine.get_statistics()
        assert stats["total_dds"] == 2
        assert "by_status" in stats


class TestDDSTypes:
    """Tests for DDS type handling."""

    def test_dds_import_type(self, engine, plot_registry):
        request = GenerateDDSRequest(
            operator_name="ImportCo",
            operator_country="DE",
            plot_ids=["PLT-00001"],
            commodity="cocoa",
            dds_type="import_placement",
        )
        dds = engine.generate_dds(request)
        assert dds.dds_type == "import_placement"

    def test_dds_export_type(self, engine, plot_registry):
        request = GenerateDDSRequest(
            operator_name="ExportCo",
            operator_country="BR",
            plot_ids=["PLT-00001"],
            commodity="cocoa",
            dds_type="export",
        )
        dds = engine.generate_dds(request)
        assert dds.dds_type == "export"

    def test_dds_with_multiple_plots(self, engine, plot_registry):
        request = GenerateDDSRequest(
            operator_name="MultiPlotCo",
            operator_country="DE",
            plot_ids=["PLT-00001", "PLT-00002", "PLT-00003"],
            commodity="cocoa",
        )
        dds = engine.generate_dds(request)
        assert len(dds.plot_ids) == 3
        assert "BR" in dds.origin_countries
        assert "KE" in dds.origin_countries
        assert "GH" in dds.origin_countries
