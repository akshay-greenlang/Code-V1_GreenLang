# -*- coding: utf-8 -*-
"""
GreenLang Threat Modeling System - SEC-010 Phase 2

Production-grade threat modeling system for the GreenLang Climate OS platform.
Provides systematic security design review using STRIDE methodology, attack
surface mapping, risk scoring, and control mapping for regulatory compliance.

Public API:
    - STRIDEEngine: STRIDE threat identification and analysis engine.
    - RiskScorer: Risk scoring based on likelihood, impact, and CVSS.
    - AttackSurfaceMapper: Attack surface discovery and exposure scoring.
    - ControlMapper: Map threats to security controls from SEC-001 to SEC-009.
    - DataFlowValidator: Validate data flow diagrams for completeness.
    - ThreatModel: Core threat model definition.
    - Threat: Individual threat instance.
    - Component: System component in the threat model.
    - DataFlow: Data flow between components.
    - TrustBoundary: Trust boundary definition.
    - Mitigation: Threat mitigation control.
    - ThreatCategory: STRIDE category enumeration.

Example:
    >>> from greenlang.infrastructure.threat_modeling import (
    ...     STRIDEEngine, ThreatModel, Component, ComponentType, ThreatCategory,
    ... )
    >>> engine = STRIDEEngine()
    >>> component = Component(
    ...     name="API Gateway",
    ...     component_type=ComponentType.API,
    ...     trust_level=2,
    ... )
    >>> threats = engine.analyze_component(component)
    >>> for threat in threats:
    ...     print(f"{threat.category.value}: {threat.title}")

Author: GreenLang Security Team
Date: February 2026
"""

from __future__ import annotations

import logging

from greenlang.infrastructure.threat_modeling.models import (
    Component,
    ComponentType,
    DataFlow,
    Mitigation,
    MitigationStatus,
    Threat,
    ThreatCategory,
    ThreatModel,
    ThreatModelStatus,
    ThreatStatus,
    TrustBoundary,
)
from greenlang.infrastructure.threat_modeling.stride_engine import STRIDEEngine
from greenlang.infrastructure.threat_modeling.attack_surface import AttackSurfaceMapper
from greenlang.infrastructure.threat_modeling.dfd_validator import DataFlowValidator
from greenlang.infrastructure.threat_modeling.risk_scorer import RiskScorer
from greenlang.infrastructure.threat_modeling.control_mapper import ControlMapper
from greenlang.infrastructure.threat_modeling.config import (
    ThreatModelingConfig,
    get_config,
    reset_config,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Configuration
    "ThreatModelingConfig",
    "get_config",
    "reset_config",
    # Core Models
    "ThreatModel",
    "ThreatModelStatus",
    "Component",
    "ComponentType",
    "DataFlow",
    "TrustBoundary",
    "Threat",
    "ThreatCategory",
    "ThreatStatus",
    "Mitigation",
    "MitigationStatus",
    # Engines
    "STRIDEEngine",
    "RiskScorer",
    "AttackSurfaceMapper",
    "DataFlowValidator",
    "ControlMapper",
]
