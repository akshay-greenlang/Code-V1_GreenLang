# -*- coding: utf-8 -*-
"""
Pytest Fixtures for Supplier Questionnaire Processor Service Unit Tests (AGENT-DATA-008)
========================================================================================

Provides shared fixtures for testing the supplier questionnaire config, models,
provenance tracker, metrics, engines, setup facade, and API router.

All tests are self-contained with no external dependencies.

Author: GreenLang Platform Team
Date: February 2026
"""

from __future__ import annotations

import os
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Environment cleanup fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_supplier_quest_env(monkeypatch):
    """Remove any GL_SUPPLIER_QUEST_ env vars between tests and reset config singleton."""
    prefix = "GL_SUPPLIER_QUEST_"
    for key in list(os.environ.keys()):
        if key.startswith(prefix):
            monkeypatch.delenv(key, raising=False)

    # Reset the config singleton so each test starts fresh
    from greenlang.supplier_questionnaire.config import reset_config
    reset_config()


# ---------------------------------------------------------------------------
# Sample Supplier Data Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_suppliers() -> List[Dict[str, Any]]:
    """Realistic supplier data (10 suppliers across sectors and regions)."""
    return [
        {"supplier_id": "SUP-001", "supplier_name": "EcoSteel GmbH", "email": "contact@ecosteel.de", "country": "DE", "sector": "metals"},
        {"supplier_id": "SUP-002", "supplier_name": "GreenLogistics AG", "email": "info@greenlog.nl", "country": "NL", "sector": "transportation"},
        {"supplier_id": "SUP-003", "supplier_name": "CleanEnergy Corp", "email": "sales@cleanenergy.com", "country": "US", "sector": "energy"},
        {"supplier_id": "SUP-004", "supplier_name": "SustainPack Ltd", "email": "procure@sustainpack.co.uk", "country": "GB", "sector": "packaging"},
        {"supplier_id": "SUP-005", "supplier_name": "BioChemicals SA", "email": "admin@biochem.fr", "country": "FR", "sector": "chemicals"},
        {"supplier_id": "SUP-006", "supplier_name": "CircularWaste Inc", "email": "ops@circularwaste.com", "country": "US", "sector": "waste_management"},
        {"supplier_id": "SUP-007", "supplier_name": "CloudIT Services", "email": "hello@cloudit.in", "country": "IN", "sector": "it_services"},
        {"supplier_id": "SUP-008", "supplier_name": "FacilityMgmt Co", "email": "fm@facilitymgmt.com", "country": "US", "sector": "facilities"},
        {"supplier_id": "SUP-009", "supplier_name": "RenewableParts Pty", "email": "parts@renewableparts.com.au", "country": "AU", "sector": "raw_materials"},
        {"supplier_id": "SUP-010", "supplier_name": "BusinessTravel GmbH", "email": "travel@biztravel.de", "country": "DE", "sector": "travel"},
    ]


# ---------------------------------------------------------------------------
# Sample Template Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_template() -> Dict[str, Any]:
    """A simple questionnaire template dict with one section and two questions."""
    return {
        "name": "Annual Carbon Disclosure 2025",
        "description": "Standard carbon disclosure questionnaire for Tier-1 suppliers",
        "framework": "cdp_climate",
        "language": "en",
        "tags": ["carbon", "scope1", "scope2", "annual"],
        "sections": [
            {
                "section_name": "Energy Consumption",
                "template_id": "tpl-dummy",
                "description": "Questions about direct energy usage",
                "order": 0,
                "questions": [
                    {
                        "section_id": "sec-dummy",
                        "question_text": "Total electricity consumed (MWh)?",
                        "question_type": "numeric",
                        "required": True,
                        "score_weight": 2.0,
                        "order": 0,
                    },
                    {
                        "section_id": "sec-dummy",
                        "question_text": "Do you use renewable energy?",
                        "question_type": "boolean",
                        "required": True,
                        "score_weight": 1.0,
                        "order": 1,
                    },
                ],
            },
        ],
    }


# ---------------------------------------------------------------------------
# Sample Response Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_response() -> Dict[str, Any]:
    """A simple questionnaire response dict with two answers."""
    return {
        "distribution_id": "dist-001",
        "template_id": "tpl-001",
        "supplier_id": "SUP-001",
        "supplier_name": "EcoSteel GmbH",
        "answers": [
            {
                "response_id": "resp-001",
                "question_id": "q-001",
                "answer_numeric": 4500.0,
                "confidence_score": 90.0,
            },
            {
                "response_id": "resp-001",
                "question_id": "q-002",
                "answer_value": "true",
                "confidence_score": 95.0,
            },
        ],
    }


# ---------------------------------------------------------------------------
# Sample Distribution Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_distribution() -> Dict[str, Any]:
    """A simple distribution dict for a single supplier."""
    return {
        "template_id": "tpl-001",
        "supplier_id": "SUP-001",
        "supplier_name": "EcoSteel GmbH",
        "supplier_email": "contact@ecosteel.de",
        "channel": "email",
        "campaign_id": "camp-2025-Q2",
    }


# ---------------------------------------------------------------------------
# Mock Prometheus Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_prometheus():
    """Mock prometheus_client for metrics testing."""
    mock_counter = MagicMock()
    mock_counter.labels.return_value = mock_counter
    mock_histogram = MagicMock()
    mock_histogram.labels.return_value = mock_histogram
    mock_gauge = MagicMock()
    mock_gauge.labels.return_value = mock_gauge

    mock_prom = MagicMock()
    mock_prom.Counter.return_value = mock_counter
    mock_prom.Histogram.return_value = mock_histogram
    mock_prom.Gauge.return_value = mock_gauge
    return mock_prom
