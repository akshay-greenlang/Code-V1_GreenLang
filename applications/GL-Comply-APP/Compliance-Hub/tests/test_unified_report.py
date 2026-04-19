# -*- coding: utf-8 -*-
"""Unified report agent tests."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from agents.orchestrator_agent import ComplianceOrchestrator
from agents.unified_report_agent import UnifiedReportAgent
from schemas.models import ComplianceRequest, EntitySnapshot, FrameworkEnum
from greenlang.schemas.enums import ReportFormat


def _produce_report() -> object:
    req = ComplianceRequest(
        entity=EntitySnapshot(
            entity_id="acme-de",
            legal_name="ACME GmbH",
            jurisdiction="DE",
            revenue_eur=75_000_000,
            employees=400,
        ),
        reporting_period_start=datetime(2025, 1, 1, tzinfo=timezone.utc),
        reporting_period_end=datetime(2025, 12, 31, tzinfo=timezone.utc),
        frameworks=[FrameworkEnum.GHG_PROTOCOL, FrameworkEnum.CSRD],
        data_sources={
            "activities": [
                {
                    "activity_id": "a1",
                    "activity_type": "stationary_combustion",
                    "fuel_type": "diesel",
                    "quantity": "1000",
                    "unit": "gallons",
                    "year": 2024,
                }
            ]
        },
    )
    return asyncio.run(ComplianceOrchestrator().run(req))


def test_generate_json(tmp_path: Path) -> None:
    report = _produce_report()
    agent = UnifiedReportAgent()
    out = agent.generate(report, ReportFormat.JSON, tmp_path)
    assert out["format"] == "json"
    assert Path(out["path"]).exists()
    data = json.loads(Path(out["path"]).read_text(encoding="utf-8"))
    assert data["overall_status"] == report.overall_status.value


def test_generate_xbrl_lite(tmp_path: Path) -> None:
    report = _produce_report()
    agent = UnifiedReportAgent()
    out = agent.generate(report, ReportFormat.XML, tmp_path)
    content = Path(out["path"]).read_text(encoding="utf-8")
    assert content.startswith("<?xml")
    assert "<ComplianceReport>" in content
    assert "<Framework name='ghg'>" in content or "<Framework name='csrd'>" in content


def test_generate_pdf_fallback_or_real(tmp_path: Path) -> None:
    report = _produce_report()
    out = UnifiedReportAgent().generate(report, ReportFormat.PDF, tmp_path)
    # Either real PDF (reportlab installed) or text fallback
    assert out["format"] == "pdf"
    assert out["size_bytes"] > 0
    assert Path(out["path"]).exists()
