# -*- coding: utf-8 -*-
"""
GreenLang Security Scanning Module - SEC-007

Production-grade security scanning orchestration for GreenLang Climate OS.
Provides unified scanning across SAST, SCA, secrets, container, IaC, and
DAST tools with deduplication, SARIF output, and vulnerability management.

This module implements SEC-007 from the GreenLang security program:
- TR-001: Security Scanning Orchestrator
- TR-002: DAST Integration (OWASP ZAP)
- TR-003: Advanced SAST (Semgrep + CodeQL)
- TR-004: Container Image Signing (Cosign/Sigstore)
- TR-005: SBOM Signing and Attestation

Sub-modules:
    config         - Scanner and orchestrator configuration
    models         - Data models for findings, results, reports
    orchestrator   - Main scan orchestration logic
    deduplication  - CVE and fingerprint-based deduplication
    sarif_generator - SARIF 2.1.0 output generation
    scanners/      - Individual scanner implementations
    sbom_signing   - SBOM signing with Cosign
    supply_chain   - Supply chain verification

Quick start:
    >>> from greenlang.infrastructure.security_scanning import (
    ...     ScanOrchestrator,
    ...     ScanOrchestratorConfig,
    ... )
    >>> config = ScanOrchestratorConfig.from_environment()
    >>> orchestrator = ScanOrchestrator(config)
    >>> report = await orchestrator.scan("/path/to/code")
    >>> print(f"Found {report.get_total_finding_count()} issues")

Security Compliance: SOC 2 CC7.1, ISO 27001 A.14.2.5, GDPR Art. 32, SLSA Level 2

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
Version: 1.0.0
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Module version
__version__ = "1.0.0"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

from greenlang.infrastructure.security_scanning.config import (
    ScannerConfig,
    ScanOrchestratorConfig,
    ScannerType,
    Severity,
    SLAPriority,
    SEVERITY_SLA_MAP,
    SLA_DAYS_MAP,
)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

from greenlang.infrastructure.security_scanning.models import (
    # Status enums
    FindingStatus,
    ScanStatus,
    # Location models
    FileLocation,
    ContainerLocation,
    # Vulnerability info
    VulnerabilityInfo,
    RemediationInfo,
    # Core models
    ScanFinding,
    ScanResult,
    ScanReport,
)

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

from greenlang.infrastructure.security_scanning.deduplication import (
    DeduplicationEngine,
    DeduplicationResult,
    normalize_severity,
    cvss_to_severity,
    calculate_fingerprint,
    group_findings_by_cve,
    get_unique_cves,
)

# ---------------------------------------------------------------------------
# SARIF Generator
# ---------------------------------------------------------------------------

from greenlang.infrastructure.security_scanning.sarif_generator import (
    SARIFGenerator,
    merge_sarif_reports,
    validate_sarif,
)

# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

from greenlang.infrastructure.security_scanning.orchestrator import (
    ScanOrchestrator,
    get_orchestrator,
    reset_orchestrator,
)

# ---------------------------------------------------------------------------
# Scanner Exceptions
# ---------------------------------------------------------------------------

from greenlang.infrastructure.security_scanning.scanners.base import (
    ScannerError,
    ScannerExecutionError,
    ScannerTimeoutError,
    ScannerNotFoundError,
    ScannerParseError,
)

# ---------------------------------------------------------------------------
# Vulnerability Management (Phase 5)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.security_scanning.vulnerability_service import (
        VulnerabilityService,
        VulnerabilityServiceConfig,
        VulnerabilitySeverity,
        VulnerabilityStatus,
        ExceptionType,
        ScanFinding as VulnScanFinding,
        Vulnerability,
        VulnerabilityFilter,
        SLAReport,
        RiskAcceptance,
        get_vulnerability_service,
        configure_vulnerability_service,
    )
    VULNERABILITY_SERVICE_AVAILABLE = True
except ImportError:
    VulnerabilityService = None  # type: ignore
    VulnerabilityServiceConfig = None  # type: ignore
    VulnerabilitySeverity = None  # type: ignore
    VulnerabilityStatus = None  # type: ignore
    ExceptionType = None  # type: ignore
    VulnScanFinding = None  # type: ignore
    Vulnerability = None  # type: ignore
    VulnerabilityFilter = None  # type: ignore
    SLAReport = None  # type: ignore
    RiskAcceptance = None  # type: ignore
    get_vulnerability_service = None  # type: ignore
    configure_vulnerability_service = None  # type: ignore
    VULNERABILITY_SERVICE_AVAILABLE = False

# ---------------------------------------------------------------------------
# Risk Scoring (Phase 5)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.security_scanning.risk_scoring import (
        RiskScorer,
        RiskScorerConfig,
        RiskScoreResult,
        AssetCriticality,
        CVSSVector,
        EPSSData,
        KEVEntry,
        get_risk_scorer,
        configure_risk_scorer,
    )
    RISK_SCORING_AVAILABLE = True
except ImportError:
    RiskScorer = None  # type: ignore
    RiskScorerConfig = None  # type: ignore
    RiskScoreResult = None  # type: ignore
    AssetCriticality = None  # type: ignore
    CVSSVector = None  # type: ignore
    EPSSData = None  # type: ignore
    KEVEntry = None  # type: ignore
    get_risk_scorer = None  # type: ignore
    configure_risk_scorer = None  # type: ignore
    RISK_SCORING_AVAILABLE = False

# ---------------------------------------------------------------------------
# Metrics (Phase 6)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.security_scanning.metrics import (
        SecurityMetrics,
        get_security_metrics,
    )
    METRICS_AVAILABLE = True
except ImportError:
    SecurityMetrics = None  # type: ignore
    get_security_metrics = None  # type: ignore
    METRICS_AVAILABLE = False

# ---------------------------------------------------------------------------
# PII Detection (Phase 7)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.security_scanning.pii_scanner import (
        PIIScanner,
        PIIPattern,
        PIIFinding,
        ScanResult as PIIScanResult,
        DataClassification,
        PIIType,
        DetectionMethod,
        get_pii_scanner,
    )
    PII_SCANNER_AVAILABLE = True
except ImportError:
    PIIScanner = None  # type: ignore
    PIIPattern = None  # type: ignore
    PIIFinding = None  # type: ignore
    PIIScanResult = None  # type: ignore
    DataClassification = None  # type: ignore
    PIIType = None  # type: ignore
    DetectionMethod = None  # type: ignore
    get_pii_scanner = None  # type: ignore
    PII_SCANNER_AVAILABLE = False

try:
    from greenlang.infrastructure.security_scanning.pii_ml import (
        PresidioPIIScanner,
        HybridPIIScanner,
        PIIEntity,
        AnalysisResult,
        PRESIDIO_AVAILABLE,
        get_presidio_scanner,
    )
    PII_ML_AVAILABLE = True
except ImportError:
    PresidioPIIScanner = None  # type: ignore
    HybridPIIScanner = None  # type: ignore
    PIIEntity = None  # type: ignore
    AnalysisResult = None  # type: ignore
    PRESIDIO_AVAILABLE = False
    get_presidio_scanner = None  # type: ignore
    PII_ML_AVAILABLE = False

try:
    from greenlang.infrastructure.security_scanning.pii_alerts import (
        PIIAlertRouter,
        PIIAlert,
        RoutingRule,
        AlertSeverity,
        AlertTeam,
        AlertStatus,
        REMEDIATION_TEMPLATES,
        get_pii_alert_router,
        configure_pii_alert_router,
    )
    PII_ALERTS_AVAILABLE = True
except ImportError:
    PIIAlertRouter = None  # type: ignore
    PIIAlert = None  # type: ignore
    RoutingRule = None  # type: ignore
    AlertSeverity = None  # type: ignore
    AlertTeam = None  # type: ignore
    AlertStatus = None  # type: ignore
    REMEDIATION_TEMPLATES = {}  # type: ignore
    get_pii_alert_router = None  # type: ignore
    configure_pii_alert_router = None  # type: ignore
    PII_ALERTS_AVAILABLE = False

# ---------------------------------------------------------------------------
# API Routes (Phase 5-6)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.security_scanning.api import (
        security_router,
        vulnerabilities_router,
        scans_router,
        dashboard_router,
        FASTAPI_AVAILABLE,
    )
except ImportError:
    security_router = None  # type: ignore
    vulnerabilities_router = None  # type: ignore
    scans_router = None  # type: ignore
    dashboard_router = None  # type: ignore
    FASTAPI_AVAILABLE = False

# ---------------------------------------------------------------------------
# SBOM and Supply Chain (existing exports)
# ---------------------------------------------------------------------------

try:
    from greenlang.infrastructure.security_scanning.sbom_signing import (
        SBOMSigner,
        SBOMSigningConfig,
        SBOMSigningResult,
    )
    SBOM_SIGNING_AVAILABLE = True
except ImportError:
    SBOMSigner = None  # type: ignore
    SBOMSigningConfig = None  # type: ignore
    SBOMSigningResult = None  # type: ignore
    SBOM_SIGNING_AVAILABLE = False

try:
    from greenlang.infrastructure.security_scanning.supply_chain import (
        SupplyChainVerifier,
        SupplyChainConfig,
        VerificationResult,
        SignatureInfo,
        SBOMInfo,
        ProvenanceInfo,
    )
    SUPPLY_CHAIN_AVAILABLE = True
except ImportError:
    SupplyChainVerifier = None  # type: ignore
    SupplyChainConfig = None  # type: ignore
    VerificationResult = None  # type: ignore
    SignatureInfo = None  # type: ignore
    SBOMInfo = None  # type: ignore
    ProvenanceInfo = None  # type: ignore
    SUPPLY_CHAIN_AVAILABLE = False

# ---------------------------------------------------------------------------
# Scanner Availability Flags
# ---------------------------------------------------------------------------

SAST_AVAILABLE = True
SCA_AVAILABLE = True
SECRETS_AVAILABLE = True
CONTAINER_AVAILABLE = True
IAC_AVAILABLE = True
DAST_AVAILABLE = True

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    # Version
    "__version__",
    # Configuration
    "ScannerConfig",
    "ScanOrchestratorConfig",
    "ScannerType",
    "Severity",
    "SLAPriority",
    "SEVERITY_SLA_MAP",
    "SLA_DAYS_MAP",
    # Status enums
    "FindingStatus",
    "ScanStatus",
    # Location models
    "FileLocation",
    "ContainerLocation",
    # Vulnerability info
    "VulnerabilityInfo",
    "RemediationInfo",
    # Core models
    "ScanFinding",
    "ScanResult",
    "ScanReport",
    # Deduplication
    "DeduplicationEngine",
    "DeduplicationResult",
    "normalize_severity",
    "cvss_to_severity",
    "calculate_fingerprint",
    "group_findings_by_cve",
    "get_unique_cves",
    # SARIF
    "SARIFGenerator",
    "merge_sarif_reports",
    "validate_sarif",
    # Orchestrator
    "ScanOrchestrator",
    "get_orchestrator",
    "reset_orchestrator",
    # Exceptions
    "ScannerError",
    "ScannerExecutionError",
    "ScannerTimeoutError",
    "ScannerNotFoundError",
    "ScannerParseError",
    # SBOM Signing (existing)
    "SBOMSigner",
    "SBOMSigningConfig",
    "SBOMSigningResult",
    # Supply Chain Verification (existing)
    "SupplyChainVerifier",
    "SupplyChainConfig",
    "VerificationResult",
    "SignatureInfo",
    "SBOMInfo",
    "ProvenanceInfo",
    # Availability flags
    "SAST_AVAILABLE",
    "SCA_AVAILABLE",
    "SECRETS_AVAILABLE",
    "CONTAINER_AVAILABLE",
    "IAC_AVAILABLE",
    "DAST_AVAILABLE",
    "SBOM_SIGNING_AVAILABLE",
    "SUPPLY_CHAIN_AVAILABLE",
]

logger.debug("Security scanning module loaded (version %s)", __version__)
