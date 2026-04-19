# -*- coding: utf-8 -*-
"""
GL-017 CONDENSYNC - Audit Module

Comprehensive audit package for condenser optimization providing:
- Provenance tracking with SHA-256 hashing
- Step-by-step calculation logging
- Equation version tracking
- Immutable audit records
- Evidence generation for recommendations
- Compliance reporting

Zero-Hallucination Guarantee:
All calculations traceable to physics equations.
Complete provenance chain with SHA-256 hashes.
Immutable audit records for regulatory compliance.

Author: GL-BackendDeveloper
Date: December 2025
Version: 1.0.0
"""

from .provenance_tracker import (
    ProvenanceTracker,
    CalculationContext,
    AuditRecord,
    CalculationStep,
    ProvenanceChain,
    RegisteredEquation,
    CalculationType,
    CalculationStatus,
    AuditLevel,
    EquationStatus,
    EQUATION_REGISTRY,
)

from .evidence_generator import (
    EvidenceGenerator,
    EvidenceItem,
    EvidenceChain,
    RecommendationEvidence,
    ComplianceReport,
    DataSource,
    EvidenceType,
    EvidenceStrength,
    RecommendationType,
    ComplianceFramework,
    PARAMETER_THRESHOLDS,
    PHYSICS_EQUATIONS,
    COMPLIANCE_MAPPINGS,
)


# Agent identification
AGENT_ID = "GL-017"
AGENT_NAME = "Condensync"
MODULE_VERSION = "1.0.0"


__all__ = [
    # Provenance Tracker
    "ProvenanceTracker",
    "CalculationContext",
    "AuditRecord",
    "CalculationStep",
    "ProvenanceChain",
    "RegisteredEquation",
    "CalculationType",
    "CalculationStatus",
    "AuditLevel",
    "EquationStatus",
    "EQUATION_REGISTRY",

    # Evidence Generator
    "EvidenceGenerator",
    "EvidenceItem",
    "EvidenceChain",
    "RecommendationEvidence",
    "ComplianceReport",
    "DataSource",
    "EvidenceType",
    "EvidenceStrength",
    "RecommendationType",
    "ComplianceFramework",
    "PARAMETER_THRESHOLDS",
    "PHYSICS_EQUATIONS",
    "COMPLIANCE_MAPPINGS",

    # Module info
    "AGENT_ID",
    "AGENT_NAME",
    "MODULE_VERSION",
]
