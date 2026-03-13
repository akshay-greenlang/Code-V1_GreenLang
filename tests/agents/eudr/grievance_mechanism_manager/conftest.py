# -*- coding: utf-8 -*-
"""
Shared fixtures for AGENT-EUDR-032 Grievance Mechanism Manager tests.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, List

from greenlang.agents.eudr.grievance_mechanism_manager.config import (
    GrievanceMechanismManagerConfig,
    reset_config,
)
from greenlang.agents.eudr.grievance_mechanism_manager.models import (
    AnalysisMethod,
    AuditAction,
    AuditEntry,
    CausalChainStep,
    CollectiveDemand,
    CollectiveGrievanceRecord,
    CollectiveStatus,
    GrievanceAnalyticsRecord,
    ImplementationStatus,
    MediationRecord,
    MediationSession,
    MediationStage,
    MediatorType,
    NegotiationStatus,
    PatternType,
    RemediationAction,
    RemediationRecord,
    RemediationType,
    RegulatoryReport,
    RegulatoryReportType,
    ReportSection,
    RiskLevel,
    RiskScope,
    RiskScoreRecord,
    ScoreFactor,
    SettlementStatus,
    TrendDirection,
)
from greenlang.agents.eudr.grievance_mechanism_manager.provenance import (
    GENESIS_HASH,
    ProvenanceTracker,
)


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before/after each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def sample_config() -> GrievanceMechanismManagerConfig:
    return GrievanceMechanismManagerConfig()


@pytest.fixture
def provenance_tracker() -> ProvenanceTracker:
    return ProvenanceTracker()


@pytest.fixture
def sample_grievances() -> List[Dict]:
    """Sample grievance data list for testing."""
    return [
        {
            "grievance_id": "g-001", "operator_id": "OP-001",
            "category": "environmental", "severity": "high",
            "status": "resolved", "description": "Water pollution from factory site",
            "complainant_stakeholder_id": "stk-001",
        },
        {
            "grievance_id": "g-002", "operator_id": "OP-001",
            "category": "environmental", "severity": "medium",
            "status": "under_investigation", "description": "Soil contamination near plantation",
            "complainant_stakeholder_id": "stk-002",
        },
        {
            "grievance_id": "g-003", "operator_id": "OP-001",
            "category": "human_rights", "severity": "critical",
            "status": "submitted", "description": "Indigenous rights violation and forced displacement",
            "complainant_stakeholder_id": "stk-003",
        },
        {
            "grievance_id": "g-004", "operator_id": "OP-001",
            "category": "labor", "severity": "high",
            "status": "resolved", "description": "Unsafe working conditions at processing plant",
            "complainant_stakeholder_id": "stk-004",
        },
        {
            "grievance_id": "g-005", "operator_id": "OP-001",
            "category": "environmental", "severity": "high",
            "status": "appealed", "description": "Deforestation observed near protected area",
            "complainant_stakeholder_id": "stk-005",
        },
    ]


@pytest.fixture
def sample_analytics_record() -> GrievanceAnalyticsRecord:
    return GrievanceAnalyticsRecord(
        analytics_id="ana-001",
        operator_id="OP-001",
        grievance_ids=["g-001", "g-002"],
        pattern_type=PatternType.RECURRING,
        pattern_description="Recurring environmental issues",
        affected_stakeholder_count=5,
        severity_distribution={"high": 3, "medium": 2},
        category_distribution={"environmental": 4, "labor": 1},
        trend_direction=TrendDirection.WORSENING,
        trend_confidence=Decimal("75"),
    )


@pytest.fixture
def sample_root_cause_record() -> Dict:
    return {
        "description": "Water pollution from factory site",
        "category": "environmental",
        "severity": "high",
    }


@pytest.fixture
def sample_mediation_parties() -> List[Dict]:
    return [
        {"role": "complainant", "name": "Community A", "id": "stk-001"},
        {"role": "respondent", "name": "Operator Corp", "id": "OP-001"},
        {"role": "mediator", "name": "External Mediator", "id": "med-001"},
    ]


@pytest.fixture
def sample_remediation_actions() -> List[Dict]:
    return [
        {"action": "Install water treatment system", "status": "pending", "responsible_party": "Engineering"},
        {"action": "Compensate affected community", "status": "pending", "responsible_party": "CSR Team"},
    ]


@pytest.fixture
def sample_collective_demands() -> List[Dict]:
    return [
        {"demand": "Clean water supply restoration", "priority": "critical", "negotiable": False},
        {"demand": "Environmental monitoring station", "priority": "high", "negotiable": True},
        {"demand": "Community health assessment", "priority": "medium", "negotiable": True},
    ]
