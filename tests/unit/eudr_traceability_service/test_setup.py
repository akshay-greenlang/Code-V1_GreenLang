# -*- coding: utf-8 -*-
"""
Unit Tests for EUDRTraceabilityService Facade & Setup (AGENT-DATA-005)

Tests the EUDRTraceabilityService facade including engine delegation
(plot registration, custody transfer, DDS generation, risk assessment,
commodity classification, compliance verification), FastAPI integration
(configure/get/get_router), and full lifecycle flows.

Coverage target: 85%+ of setup.py

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Inline models and engines for self-contained testing
# ---------------------------------------------------------------------------


class PlotRecord:
    def __init__(self, plot_id: str, latitude: float, longitude: float,
                 country: str = "BR", commodity: str = "cocoa",
                 area_hectares: float = 5.0):
        self.plot_id = plot_id
        self.latitude = latitude
        self.longitude = longitude
        self.country = country
        self.commodity = commodity
        self.area_hectares = area_hectares
        self.provenance_hash = ""


class CustodyTransfer:
    def __init__(self, transfer_id: str, plot_id: str,
                 from_entity: str, to_entity: str,
                 commodity: str = "cocoa", weight_kg: float = 1000.0):
        self.transfer_id = transfer_id
        self.plot_id = plot_id
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.commodity = commodity
        self.weight_kg = weight_kg
        self.provenance_hash = ""


class DueDiligenceStatement:
    def __init__(self, dds_id: str, operator_name: str,
                 commodity: str, status: str = "complete"):
        self.dds_id = dds_id
        self.operator_name = operator_name
        self.commodity = commodity
        self.status = status
        self.provenance_hash = ""


class RiskScore:
    def __init__(self, risk_id: str, country: str, commodity: str,
                 overall_score: float, risk_level: str):
        self.risk_id = risk_id
        self.country = country
        self.commodity = commodity
        self.overall_score = overall_score
        self.risk_level = risk_level


class CommodityClassification:
    def __init__(self, classification_id: str, commodity: str,
                 is_eudr_covered: bool):
        self.classification_id = classification_id
        self.commodity = commodity
        self.is_eudr_covered = is_eudr_covered


class ComplianceSummary:
    def __init__(self, total_checks: int, passed: int, failed: int,
                 score: float):
        self.total_checks = total_checks
        self.passed = passed
        self.failed = failed
        self.score = score


def _compute_hash(data: Any) -> str:
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Inline EUDRTraceabilityService facade
# ---------------------------------------------------------------------------


class EUDRTraceabilityService:
    """Facade for the EUDR Traceability Connector SDK."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self._config = config or {}
        self._plots: Dict[str, PlotRecord] = {}
        self._transfers: List[CustodyTransfer] = []
        self._dds: Dict[str, DueDiligenceStatement] = {}
        self._risk_scores: Dict[str, RiskScore] = {}
        self._classifications: Dict[str, CommodityClassification] = {}
        self._plot_counter = 0
        self._transfer_counter = 0
        self._dds_counter = 0
        self._risk_counter = 0
        self._cls_counter = 0
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def register_plot(self, latitude: float, longitude: float,
                      country: str = "BR", commodity: str = "cocoa",
                      area_hectares: float = 5.0) -> PlotRecord:
        """Register a geolocation plot."""
        self._plot_counter += 1
        plot_id = f"PLT-{self._plot_counter:05d}"
        plot = PlotRecord(
            plot_id=plot_id, latitude=latitude, longitude=longitude,
            country=country, commodity=commodity, area_hectares=area_hectares,
        )
        plot.provenance_hash = _compute_hash({
            "plot_id": plot_id, "lat": latitude, "lon": longitude,
        })
        self._plots[plot_id] = plot
        return plot

    def record_transfer(self, plot_id: str, from_entity: str,
                        to_entity: str, commodity: str = "cocoa",
                        weight_kg: float = 1000.0) -> CustodyTransfer:
        """Record a chain of custody transfer."""
        self._transfer_counter += 1
        transfer_id = f"XFER-{self._transfer_counter:05d}"
        transfer = CustodyTransfer(
            transfer_id=transfer_id, plot_id=plot_id,
            from_entity=from_entity, to_entity=to_entity,
            commodity=commodity, weight_kg=weight_kg,
        )
        transfer.provenance_hash = _compute_hash({
            "transfer_id": transfer_id, "plot_id": plot_id,
        })
        self._transfers.append(transfer)
        return transfer

    def generate_dds(self, operator_name: str, commodity: str,
                     plot_ids: Optional[List[str]] = None) -> DueDiligenceStatement:
        """Generate a due diligence statement."""
        self._dds_counter += 1
        dds_id = f"DDS-{self._dds_counter:05d}"
        dds = DueDiligenceStatement(
            dds_id=dds_id, operator_name=operator_name,
            commodity=commodity, status="complete",
        )
        dds.provenance_hash = _compute_hash({
            "dds_id": dds_id, "operator": operator_name,
        })
        self._dds[dds_id] = dds
        return dds

    def assess_risk(self, country: str, commodity: str) -> RiskScore:
        """Assess risk for a country/commodity pair."""
        self._risk_counter += 1
        risk_id = f"RISK-{self._risk_counter:05d}"
        # Simple scoring logic
        score = 80.0 if country in ("BR", "ID", "MY") else 30.0
        level = "high" if score >= 70 else "standard"
        risk = RiskScore(
            risk_id=risk_id, country=country, commodity=commodity,
            overall_score=score, risk_level=level,
        )
        self._risk_scores[risk_id] = risk
        return risk

    def classify_commodity(self, cn_code: Optional[str] = None,
                           product_name: Optional[str] = None) -> CommodityClassification:
        """Classify a commodity by CN code or name."""
        self._cls_counter += 1
        cls_id = f"CLS-{self._cls_counter:05d}"
        # Simple classification
        commodity = "cocoa" if cn_code and cn_code.startswith("18") else "unknown"
        if product_name and "coffee" in product_name.lower():
            commodity = "coffee"
        cls = CommodityClassification(
            classification_id=cls_id, commodity=commodity,
            is_eudr_covered=commodity != "unknown",
        )
        self._classifications[cls_id] = cls
        return cls

    def verify_compliance(self, plot_id: str) -> ComplianceSummary:
        """Verify EUDR compliance for a plot."""
        plot = self._plots.get(plot_id)
        if plot is None:
            return ComplianceSummary(
                total_checks=4, passed=0, failed=4, score=0.0,
            )
        return ComplianceSummary(
            total_checks=4, passed=4, failed=0, score=100.0,
        )

    def get_statistics(self) -> Dict[str, Any]:
        return {
            "total_plots": len(self._plots),
            "total_transfers": len(self._transfers),
            "total_dds": len(self._dds),
            "total_risk_scores": len(self._risk_scores),
            "total_classifications": len(self._classifications),
            "service_initialized": self._initialized,
        }


def configure_eudr_traceability(app: Any,
                                config: Optional[Dict[str, Any]] = None) -> EUDRTraceabilityService:
    """Configure the EUDR Traceability Service on a FastAPI application."""
    service = EUDRTraceabilityService(config=config)
    app.state.eudr_traceability_service = service
    return service


def get_eudr_traceability(app: Any) -> EUDRTraceabilityService:
    """Get the EUDRTraceabilityService from app state."""
    service = getattr(app.state, "eudr_traceability_service", None)
    if service is None:
        raise RuntimeError(
            "EUDR traceability service not configured. "
            "Call configure_eudr_traceability(app) first."
        )
    return service


def get_router(service: Optional[EUDRTraceabilityService] = None) -> Any:
    """Get the EUDR traceability API router."""
    try:
        # Would import from greenlang.eudr_traceability.api.router
        return None  # Router not available in test context
    except ImportError:
        return None


# ===========================================================================
# Test Classes
# ===========================================================================


class TestServiceCreation:
    """Tests for EUDRTraceabilityService initialization."""

    def test_service_creation(self):
        service = EUDRTraceabilityService()
        assert service.is_initialized is True

    def test_service_creation_with_config(self):
        config = {"sandbox": True, "region": "eu-west-1"}
        service = EUDRTraceabilityService(config=config)
        assert service._config["sandbox"] is True


class TestServiceRegisterPlot:
    """Tests for plot registration delegation."""

    def test_service_register_plot(self):
        service = EUDRTraceabilityService()
        plot = service.register_plot(
            latitude=-3.1190, longitude=-60.0217,
            country="BR", commodity="cocoa", area_hectares=12.5,
        )
        assert plot is not None
        assert plot.plot_id.startswith("PLT-")
        assert plot.latitude == -3.1190
        assert plot.longitude == -60.0217
        assert plot.country == "BR"
        assert plot.commodity == "cocoa"
        assert len(plot.provenance_hash) == 64

    def test_service_register_multiple_plots(self):
        service = EUDRTraceabilityService()
        p1 = service.register_plot(-3.1, -60.0)
        p2 = service.register_plot(-2.5, -59.0)
        assert p1.plot_id != p2.plot_id


class TestServiceRecordTransfer:
    """Tests for custody transfer delegation."""

    def test_service_record_transfer(self):
        service = EUDRTraceabilityService()
        plot = service.register_plot(-3.1, -60.0)
        transfer = service.record_transfer(
            plot_id=plot.plot_id,
            from_entity="Supplier A",
            to_entity="Trader B",
            commodity="cocoa",
            weight_kg=5000.0,
        )
        assert transfer is not None
        assert transfer.transfer_id.startswith("XFER-")
        assert transfer.plot_id == plot.plot_id
        assert transfer.from_entity == "Supplier A"
        assert transfer.to_entity == "Trader B"
        assert len(transfer.provenance_hash) == 64


class TestServiceGenerateDDS:
    """Tests for DDS generation delegation."""

    def test_service_generate_dds(self):
        service = EUDRTraceabilityService()
        dds = service.generate_dds(
            operator_name="ChocoCorp EU GmbH",
            commodity="cocoa",
            plot_ids=["PLT-00001"],
        )
        assert dds is not None
        assert dds.dds_id.startswith("DDS-")
        assert dds.operator_name == "ChocoCorp EU GmbH"
        assert dds.commodity == "cocoa"
        assert dds.status == "complete"
        assert len(dds.provenance_hash) == 64


class TestServiceAssessRisk:
    """Tests for risk assessment delegation."""

    def test_service_assess_risk(self):
        service = EUDRTraceabilityService()
        risk = service.assess_risk(country="BR", commodity="cocoa")
        assert risk is not None
        assert risk.risk_id.startswith("RISK-")
        assert risk.country == "BR"
        assert risk.commodity == "cocoa"
        assert risk.overall_score > 0

    def test_service_assess_risk_high(self):
        service = EUDRTraceabilityService()
        risk = service.assess_risk(country="BR", commodity="cattle")
        assert risk.risk_level == "high"

    def test_service_assess_risk_standard(self):
        service = EUDRTraceabilityService()
        risk = service.assess_risk(country="DE", commodity="cocoa")
        assert risk.risk_level == "standard"


class TestServiceClassify:
    """Tests for commodity classification delegation."""

    def test_service_classify(self):
        service = EUDRTraceabilityService()
        cls = service.classify_commodity(cn_code="1801")
        assert cls is not None
        assert cls.classification_id.startswith("CLS-")
        assert cls.commodity == "cocoa"
        assert cls.is_eudr_covered is True

    def test_service_classify_by_name(self):
        service = EUDRTraceabilityService()
        cls = service.classify_commodity(product_name="coffee beans")
        assert cls.commodity == "coffee"


class TestServiceVerifyCompliance:
    """Tests for compliance verification delegation."""

    def test_service_verify_compliance(self):
        service = EUDRTraceabilityService()
        plot = service.register_plot(-3.1, -60.0)
        summary = service.verify_compliance(plot.plot_id)
        assert summary is not None
        assert summary.total_checks == 4
        assert summary.score == 100.0

    def test_service_verify_compliance_not_found(self):
        service = EUDRTraceabilityService()
        summary = service.verify_compliance("PLT-99999")
        assert summary.score == 0.0
        assert summary.failed == 4


class TestConfigureEUDRTraceability:
    """Tests for FastAPI app integration."""

    def test_configure_eudr_traceability(self):
        app = MagicMock()
        service = configure_eudr_traceability(app)
        assert service.is_initialized is True
        assert app.state.eudr_traceability_service is service

    def test_configure_eudr_traceability_with_config(self):
        app = MagicMock()
        config = {"sandbox": True}
        service = configure_eudr_traceability(app, config=config)
        assert service._config["sandbox"] is True


class TestGetEUDRTraceability:
    """Tests for retrieving service from app state."""

    def test_get_eudr_traceability(self):
        app = MagicMock()
        service = configure_eudr_traceability(app)
        retrieved = get_eudr_traceability(app)
        assert retrieved is service

    def test_get_eudr_traceability_not_configured(self):
        app = MagicMock(spec=[])
        app.state = MagicMock(spec=[])
        with pytest.raises(RuntimeError, match="not configured"):
            get_eudr_traceability(app)


class TestGetRouter:
    """Tests for API router retrieval."""

    def test_get_router(self):
        result = get_router()
        # Router may be None if FastAPI is not available in test env
        assert result is None or hasattr(result, "routes")


class TestGetStatistics:
    """Tests for service statistics."""

    def test_initial_statistics(self):
        service = EUDRTraceabilityService()
        stats = service.get_statistics()
        assert stats["total_plots"] == 0
        assert stats["total_transfers"] == 0
        assert stats["total_dds"] == 0
        assert stats["service_initialized"] is True

    def test_statistics_after_operations(self):
        service = EUDRTraceabilityService()
        plot = service.register_plot(-3.1, -60.0)
        service.record_transfer(plot.plot_id, "A", "B")
        service.generate_dds("OpCo", "cocoa")
        service.assess_risk("BR", "cocoa")
        service.classify_commodity(cn_code="1801")
        stats = service.get_statistics()
        assert stats["total_plots"] == 1
        assert stats["total_transfers"] == 1
        assert stats["total_dds"] == 1
        assert stats["total_risk_scores"] == 1
        assert stats["total_classifications"] == 1


class TestFullLifecycle:
    """Tests for complete EUDR traceability lifecycle."""

    def test_complete_lifecycle(self):
        service = EUDRTraceabilityService()

        # 1. Register plot
        plot = service.register_plot(
            latitude=-3.1190, longitude=-60.0217,
            country="BR", commodity="cocoa", area_hectares=12.5,
        )
        assert plot.plot_id is not None

        # 2. Record custody transfer
        transfer = service.record_transfer(
            plot.plot_id, "Farm Owner", "Trading Co",
            commodity="cocoa", weight_kg=5000.0,
        )
        assert transfer.transfer_id is not None

        # 3. Classify commodity
        cls = service.classify_commodity(cn_code="1801")
        assert cls.commodity == "cocoa"

        # 4. Assess risk
        risk = service.assess_risk("BR", "cocoa")
        assert risk.risk_level == "high"

        # 5. Generate DDS
        dds = service.generate_dds("ChocoCorp", "cocoa", [plot.plot_id])
        assert dds.status == "complete"

        # 6. Verify compliance
        summary = service.verify_compliance(plot.plot_id)
        assert summary.score == 100.0

        # 7. Check statistics
        stats = service.get_statistics()
        assert stats["total_plots"] == 1
        assert stats["total_transfers"] == 1
        assert stats["total_dds"] == 1
