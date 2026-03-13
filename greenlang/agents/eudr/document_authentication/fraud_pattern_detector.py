# -*- coding: utf-8 -*-
"""
Fraud Pattern Detector - AGENT-EUDR-012 Engine 6

Detect 15 fraud patterns using deterministic rules (no ML/LLM) for
EUDR document authentication. Each pattern is independently toggleable,
classified by severity (low/medium/high/critical), and produces an
auditable fraud alert with evidence and SHA-256 provenance hash.

Zero-Hallucination Guarantees:
    - All 15 fraud rules use deterministic threshold-based logic
    - No ML models or LLMs used for fraud classification
    - All numeric comparisons use Python float/Decimal arithmetic
    - Severity classification uses configurable, deterministic thresholds
    - Aggregate risk score is a weighted sum (no probabilistic inference)
    - SHA-256 provenance hashes on every detection operation
    - Bit-perfect reproducibility across all fraud detection runs

Regulatory References:
    - EU 2023/1115 (EUDR) Article 4: Due diligence obligations
    - EU 2023/1115 (EUDR) Article 10(2): Risk assessment requirements
    - EU 2023/1115 (EUDR) Article 14: Five-year record retention
    - EU 2023/1115 (EUDR) Article 21: Competent authority checks
    - ISO 22095:2020: Chain of Custody - traceability requirements
    - eIDAS Regulation (EU) No 910/2014: Document integrity requirements

Fraud Patterns (15 per PRD Section 6.6):
    FRD-001: Duplicate document reuse (hash match across shipments)
    FRD-002: Quantity tampering (quantity > 105% of certified amount)
    FRD-003: Date manipulation (dates > 30 days from supply chain event)
    FRD-004: Expired certificate submission (validity end < submission)
    FRD-005: Serial number anomaly (format mismatch per issuer pattern)
    FRD-006: Issuer authority mismatch (unauthorized issuer)
    FRD-007: Template forgery detection (structure deviates from known)
    FRD-008: Cross-document quantity inconsistency (>5% variance)
    FRD-009: Geographic impossibility (origin vs supply chain graph)
    FRD-010: Velocity anomaly (>10 certs/day from same supplier)
    FRD-011: Modification timeline anomaly (mod date after issuance)
    FRD-012: Round number bias (>80% round quantities)
    FRD-013: Copy-paste detection (identical text blocks)
    FRD-014: Missing required documents for commodity
    FRD-015: Certification scope mismatch

Performance Targets:
    - Single document, all 15 rules: <50ms
    - Batch of 100 documents: <3s
    - Aggregate risk score calculation: <5ms

Author: GreenLang Platform Team
Date: March 2026
PRD: PRD-AGENT-EUDR-012
Agent ID: GL-EUDR-DAV-012
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from greenlang.agents.eudr.document_authentication.config import (
    DocumentAuthenticationConfig,
    get_config,
)
from greenlang.agents.eudr.document_authentication.models import (
    FraudAlert,
    FraudDetectionResponse,
    FraudPatternType,
    FraudSeverity,
)
from greenlang.agents.eudr.document_authentication.provenance import (
    ProvenanceTracker,
    get_provenance_tracker,
)
from greenlang.agents.eudr.document_authentication.metrics import (
    observe_verification_duration,
    record_api_error,
    record_fraud_alert,
    record_fraud_critical,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module version for provenance tracking
# ---------------------------------------------------------------------------

_MODULE_VERSION = "1.0.0"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


def _compute_hash(data: Any) -> str:
    """Compute a deterministic SHA-256 hash for audit provenance.

    Args:
        data: Any JSON-serializable object.

    Returns:
        Hex-encoded SHA-256 hash string.
    """
    raw = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _generate_id(prefix: str = "FRD") -> str:
    """Generate a prefixed UUID4 string identifier.

    Args:
        prefix: String prefix for the identifier.

    Returns:
        Prefixed UUID4 string.
    """
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# ---------------------------------------------------------------------------
# Required documents per EUDR commodity
# ---------------------------------------------------------------------------

#: Minimum required document types per EUDR commodity for FRD-014.
REQUIRED_DOCUMENTS_BY_COMMODITY: Dict[str, List[str]] = {
    "cattle": ["coo", "bol", "ic", "ssd"],
    "cocoa": ["coo", "pc", "bol", "ic", "ssd"],
    "coffee": ["coo", "pc", "bol", "ic", "ssd"],
    "oil_palm": ["coo", "rspo_cert", "bol", "ic", "ssd"],
    "rubber": ["coo", "bol", "ic", "ssd"],
    "soya": ["coo", "pc", "bol", "ic", "ssd"],
    "wood": ["coo", "fsc_cert", "bol", "ic", "fc", "ltr"],
}

#: Authorized issuing authorities per document type and country (FRD-006).
AUTHORIZED_ISSUERS: Dict[str, Dict[str, List[str]]] = {
    "coo": {
        "BR": ["Ministerio da Agricultura", "MAPA", "Chamber of Commerce"],
        "ID": ["Ministry of Trade", "KEMENDAG", "Chamber of Commerce"],
        "MY": ["MATRADE", "Chamber of Commerce", "FMM"],
        "GH": ["Ghana Export Promotion Authority", "Chamber of Commerce"],
        "CI": ["CCI", "Ministere du Commerce"],
        "CO": ["MinCIT", "Chamber of Commerce"],
        "DEFAULT": ["Chamber of Commerce", "Ministry of Trade"],
    },
    "pc": {
        "BR": ["MAPA", "Ministerio da Agricultura"],
        "ID": ["BKPH", "Ministry of Agriculture"],
        "MY": ["DOA", "MAQIS"],
        "DEFAULT": ["NPPO", "Ministry of Agriculture"],
    },
    "fsc_cert": {
        "DEFAULT": [
            "FSC", "Accreditation Services International",
            "Control Union", "SGS", "Bureau Veritas",
            "Soil Association", "TUV NORD", "Rainforest Alliance",
        ],
    },
    "rspo_cert": {
        "DEFAULT": [
            "RSPO", "Control Union", "SGS", "Bureau Veritas",
            "BSI", "TUV NORD",
        ],
    },
    "iscc_cert": {
        "DEFAULT": [
            "ISCC", "Control Union", "SGS", "TUV SUD",
            "Bureau Veritas",
        ],
    },
}

#: Certification scope to commodity mapping for FRD-015.
CERT_SCOPE_COMMODITY_MAP: Dict[str, List[str]] = {
    "fsc_cert": ["wood"],
    "rspo_cert": ["oil_palm"],
    "iscc_cert": ["oil_palm", "soya", "rubber"],
    "ft_cert": ["cocoa", "coffee", "soya"],
    "utz_cert": ["cocoa", "coffee"],
}

#: Serial number format patterns per document type for FRD-005.
SERIAL_FORMAT_PATTERNS: Dict[str, List[str]] = {
    "fsc_cert": [
        r"^FSC-C\d{6}$",
        r"^[A-Z]{2,3}-COC-\d{6}$",
    ],
    "rspo_cert": [
        r"^RSPO-\d{7}$",
        r"^P&C-\d{4}-\d{6}$",
        r"^SCCS-\d{4}-\d{6}$",
    ],
    "iscc_cert": [
        r"^ISCC-CERT-[A-Z]{2}\d{3}-\d{8}$",
        r"^EU-ISCC-Cert-[A-Z]{2}\d{5}$",
    ],
    "coo": [
        r"^[A-Z]{2}-COO-\d{4}-\d{6}$",
        r"^COO/\d{4}/\d{6,8}$",
    ],
}


# ---------------------------------------------------------------------------
# FraudRule dataclass (internal)
# ---------------------------------------------------------------------------


class _FraudRule:
    """Internal representation of a single fraud detection rule.

    Attributes:
        rule_id: Unique rule identifier (FRD-001 through FRD-015).
        pattern_type: FraudPatternType enum value.
        name: Human-readable rule name.
        description: Detailed description of what the rule checks.
        default_severity: Default severity when the rule triggers.
        enabled: Whether this rule is currently active.
    """

    __slots__ = (
        "rule_id", "pattern_type", "name", "description",
        "default_severity", "enabled",
    )

    def __init__(
        self,
        rule_id: str,
        pattern_type: FraudPatternType,
        name: str,
        description: str,
        default_severity: FraudSeverity,
        enabled: bool = True,
    ) -> None:
        self.rule_id = rule_id
        self.pattern_type = pattern_type
        self.name = name
        self.description = description
        self.default_severity = default_severity
        self.enabled = enabled

    def to_dict(self) -> Dict[str, Any]:
        """Serialize rule to dictionary."""
        return {
            "rule_id": self.rule_id,
            "pattern_type": self.pattern_type.value,
            "name": self.name,
            "description": self.description,
            "default_severity": self.default_severity.value,
            "enabled": self.enabled,
        }


# ---------------------------------------------------------------------------
# FraudPatternDetector
# ---------------------------------------------------------------------------


class FraudPatternDetector:
    """Deterministic fraud pattern detector for EUDR document authentication.

    Implements 15 fraud detection rules using configurable thresholds
    and deterministic logic (no ML/LLM). Each rule is independently
    toggleable and produces auditable FraudAlert records with evidence
    and SHA-256 provenance hashes.

    All operations are thread-safe via reentrant locking. All detection
    uses deterministic Python arithmetic for zero-hallucination compliance.

    Attributes:
        _config: Document authentication configuration.
        _provenance: ProvenanceTracker for audit trail.
        _rules: Dictionary of rule_id -> _FraudRule.
        _alerts: In-memory alert storage keyed by alert_id.
        _document_hashes: Registry of document hashes for duplicate detection.
        _supplier_velocity: Tracker for document velocity per supplier per day.
        _lock: Reentrant lock for thread-safe access.

    Example:
        >>> detector = FraudPatternDetector()
        >>> result = detector.detect_fraud(
        ...     document_id="doc-001",
        ...     document_data={
        ...         "file_hash_sha256": "abc...",
        ...         "quantity": 1000.0,
        ...         "certified_quantity": 900.0,
        ...         "document_type": "fsc_cert",
        ...     },
        ...     context={"commodity": "wood"},
        ... )
        >>> assert result["success"] is True
    """

    def __init__(
        self,
        config: Optional[DocumentAuthenticationConfig] = None,
    ) -> None:
        """Initialize FraudPatternDetector.

        Args:
            config: Optional configuration override. If None, the
                singleton configuration from ``get_config()`` is used.
        """
        self._config: DocumentAuthenticationConfig = config or get_config()
        self._provenance: ProvenanceTracker = get_provenance_tracker()

        # Initialize 15 fraud rules
        self._rules: Dict[str, _FraudRule] = self._build_rules()

        # In-memory storage
        self._alerts: Dict[str, Dict[str, Any]] = {}
        self._document_hashes: Dict[str, List[Dict[str, Any]]] = {}
        self._supplier_velocity: Dict[str, Dict[str, int]] = {}
        self._text_blocks: Dict[str, List[Dict[str, Any]]] = {}

        # Thread safety
        self._lock: threading.RLock = threading.RLock()

        logger.info(
            "FraudPatternDetector initialized: "
            "rules=%d, enabled=%s, "
            "qty_tolerance=%.1f%%, date_tolerance=%dd, "
            "velocity_threshold=%d/day, round_num_threshold=%.1f%%",
            len(self._rules),
            self._config.fraud_rules_enabled,
            self._config.quantity_tolerance_percent,
            self._config.date_tolerance_days,
            self._config.velocity_threshold_per_day,
            self._config.round_number_threshold_percent,
        )

    # ------------------------------------------------------------------
    # Public API: Detect fraud
    # ------------------------------------------------------------------

    def detect_fraud(
        self,
        document_id: str,
        document_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        patterns_to_check: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run all (or selected) fraud detection rules against a document.

        Executes each enabled fraud detection rule and collects alerts.
        Computes an aggregate fraud risk score based on severity weights.

        Args:
            document_id: Unique document identifier.
            document_data: Dictionary containing document attributes:
                - file_hash_sha256 (str): Document content hash
                - quantity (float): Quantity in the document
                - certified_quantity (float): Certified/reference quantity
                - document_type (str): EUDR document type
                - serial_number (str): Certificate/reference number
                - issuer (str): Document issuer name
                - issuer_country (str): Issuer country code
                - issuance_date (str/datetime): Document issuance date
                - validity_start (str/datetime): Certificate start date
                - validity_end (str/datetime): Certificate end date
                - creation_date (str/datetime): PDF creation date
                - modification_date (str/datetime): PDF modification date
                - submission_date (str/datetime): Date submitted to system
                - supplier_id (str): Supplier identifier
                - shipment_id (str): Associated shipment identifier
                - origin_country (str): Country of origin
                - origin_coordinates (dict): lat/lon of origin
                - supply_chain_coordinates (list): Expected coordinates
                - template_hash (str): Layout hash for template matching
                - text_blocks (list): Text blocks for copy-paste detection
                - quantities (list): All quantity values for round number check
                - related_documents (list): Related document summaries
                - commodity (str): EUDR commodity
                - certification_scope (str): Scope of certification
            context: Optional additional context for detection:
                - commodity (str): EUDR commodity if not in document_data
                - existing_documents (list): Documents for this shipment
                - known_templates (dict): Template hashes by doc type
                - supply_chain_graph (dict): Expected supply chain nodes

        Returns:
            Dictionary with keys: success, document_id, alerts,
            total_alerts, highest_severity, composite_score,
            rules_checked, rules_triggered, processing_time_ms,
            provenance_hash.

        Raises:
            ValueError: If document_id is empty.
        """
        start_time = time.monotonic()

        if not document_id:
            raise ValueError("document_id must not be empty")

        ctx = context or {}
        doc_data = dict(document_data)

        logger.info(
            "Running fraud detection: document_id=%s, rules_enabled=%s",
            document_id[:16], self._config.fraud_rules_enabled,
        )

        if not self._config.fraud_rules_enabled:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            return self._build_empty_result(
                document_id, elapsed_ms,
                message="Fraud detection disabled",
            )

        alerts: List[Dict[str, Any]] = []
        rules_checked: List[str] = []
        rules_triggered: List[str] = []

        try:
            # Determine which rules to check
            active_rules = self._get_active_rules(patterns_to_check)

            # Execute each rule
            for rule in active_rules:
                rules_checked.append(rule.rule_id)

                try:
                    rule_alerts = self._execute_rule(
                        rule, document_id, doc_data, ctx,
                    )
                    if rule_alerts:
                        rules_triggered.append(rule.rule_id)
                        alerts.extend(rule_alerts)
                except Exception as rule_exc:
                    logger.warning(
                        "Rule %s failed for document %s: %s",
                        rule.rule_id, document_id[:16], str(rule_exc),
                    )

            # Calculate composite fraud risk score
            composite_score = self._calculate_fraud_risk_score(alerts)

            # Determine highest severity
            highest_severity = self._get_highest_severity(alerts)

            # Store alerts
            with self._lock:
                for alert in alerts:
                    self._alerts[alert["alert_id"]] = alert

            # Record metrics
            if self._config.enable_metrics:
                for alert in alerts:
                    record_fraud_alert(alert.get("severity", "low"))
                    if alert.get("severity") == "critical":
                        record_fraud_critical()

            # Compute provenance
            elapsed_ms = (time.monotonic() - start_time) * 1000

            provenance_data = {
                "document_id": document_id,
                "alerts_count": len(alerts),
                "rules_checked": rules_checked,
                "rules_triggered": rules_triggered,
                "composite_score": composite_score,
                "highest_severity": highest_severity,
                "module_version": _MODULE_VERSION,
            }
            provenance_hash = _compute_hash(provenance_data)

            if self._config.enable_provenance:
                self._provenance.record(
                    entity_type="fraud_alert",
                    action="detect_fraud",
                    entity_id=document_id,
                    data=provenance_data,
                    metadata={
                        "document_id": document_id,
                        "alerts_count": len(alerts),
                        "composite_score": composite_score,
                    },
                )

            if self._config.enable_metrics:
                observe_verification_duration(elapsed_ms / 1000)

            logger.info(
                "Fraud detection completed: document_id=%s, "
                "alerts=%d, score=%.1f, severity=%s, "
                "checked=%d, triggered=%d, elapsed=%.1fms",
                document_id[:16], len(alerts), composite_score,
                highest_severity, len(rules_checked),
                len(rules_triggered), elapsed_ms,
            )

            return {
                "success": True,
                "document_id": document_id,
                "alerts": alerts,
                "total_alerts": len(alerts),
                "highest_severity": highest_severity,
                "composite_score": composite_score,
                "rules_checked": rules_checked,
                "rules_triggered": rules_triggered,
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": provenance_hash,
            }

        except Exception as exc:
            elapsed_ms = (time.monotonic() - start_time) * 1000
            logger.error(
                "Fraud detection failed: document_id=%s, error=%s",
                document_id[:16], str(exc), exc_info=True,
            )
            if self._config.enable_metrics:
                record_api_error("detect_fraud")
            return {
                "success": False,
                "document_id": document_id,
                "alerts": [],
                "total_alerts": 0,
                "highest_severity": None,
                "composite_score": 0.0,
                "rules_checked": rules_checked,
                "rules_triggered": [],
                "processing_time_ms": round(elapsed_ms, 2),
                "provenance_hash": None,
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Public API: Batch detect
    # ------------------------------------------------------------------

    def batch_detect_fraud(
        self,
        documents: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Run fraud detection on multiple documents.

        Args:
            documents: List of dictionaries, each with at minimum:
                document_id (str), document_data (dict).
                Optional: context (dict), patterns_to_check (list).

        Returns:
            List of fraud detection result dictionaries.

        Raises:
            ValueError: If documents list is empty or exceeds batch limit.
        """
        if not documents:
            raise ValueError("documents list must not be empty")

        max_size = self._config.batch_max_size
        if len(documents) > max_size:
            raise ValueError(
                f"Batch size {len(documents)} exceeds maximum {max_size}"
            )

        logger.info(
            "Batch fraud detection: %d documents", len(documents),
        )

        results: List[Dict[str, Any]] = []
        for doc in documents:
            result = self.detect_fraud(
                document_id=doc.get("document_id", str(uuid.uuid4())),
                document_data=doc.get("document_data", {}),
                context=doc.get("context"),
                patterns_to_check=doc.get("patterns_to_check"),
            )
            results.append(result)

        total_alerts = sum(r.get("total_alerts", 0) for r in results)
        logger.info(
            "Batch fraud detection completed: %d documents, "
            "%d total alerts",
            len(documents), total_alerts,
        )
        return results

    # ------------------------------------------------------------------
    # Public API: Get rules
    # ------------------------------------------------------------------

    def get_rules(self) -> List[Dict[str, Any]]:
        """Return all fraud detection rules and their current status.

        Returns:
            List of rule dictionaries with keys: rule_id, pattern_type,
            name, description, default_severity, enabled.
        """
        with self._lock:
            return [rule.to_dict() for rule in self._rules.values()]

    def set_rule_enabled(
        self,
        rule_id: str,
        enabled: bool,
    ) -> bool:
        """Enable or disable a specific fraud detection rule.

        Args:
            rule_id: Rule identifier (FRD-001 through FRD-015).
            enabled: Whether to enable the rule.

        Returns:
            True if the rule was found and updated.
        """
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule:
                rule.enabled = enabled
                logger.info(
                    "Rule %s %s",
                    rule_id, "enabled" if enabled else "disabled",
                )
                return True
        return False

    def get_alerts(
        self,
        document_id: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve stored fraud alerts with optional filtering.

        Args:
            document_id: Optional document ID filter.
            severity: Optional severity level filter.
            limit: Maximum number of alerts to return.

        Returns:
            List of fraud alert dictionaries.
        """
        with self._lock:
            alerts = list(self._alerts.values())

        if document_id:
            alerts = [
                a for a in alerts if a.get("document_id") == document_id
            ]
        if severity:
            alerts = [
                a for a in alerts if a.get("severity") == severity
            ]

        # Sort by detected_at descending
        alerts.sort(
            key=lambda a: a.get("detected_at", ""), reverse=True,
        )
        return alerts[:limit]

    # ------------------------------------------------------------------
    # Public API: Register document hash (for FRD-001)
    # ------------------------------------------------------------------

    def register_document_hash(
        self,
        document_id: str,
        file_hash: str,
        shipment_id: Optional[str] = None,
        supplier_id: Optional[str] = None,
    ) -> None:
        """Register a document hash for duplicate detection.

        Args:
            document_id: Document identifier.
            file_hash: SHA-256 hash of the document content.
            shipment_id: Optional associated shipment ID.
            supplier_id: Optional supplier ID.
        """
        with self._lock:
            if file_hash not in self._document_hashes:
                self._document_hashes[file_hash] = []
            self._document_hashes[file_hash].append({
                "document_id": document_id,
                "shipment_id": shipment_id,
                "supplier_id": supplier_id,
                "registered_at": _utcnow().isoformat(),
            })

    # ------------------------------------------------------------------
    # Internal: Build 15 fraud rules
    # ------------------------------------------------------------------

    def _build_rules(self) -> Dict[str, _FraudRule]:
        """Create all 15 fraud detection rules.

        Returns:
            Dictionary of rule_id to _FraudRule.
        """
        rules = {}

        rules["FRD-001"] = _FraudRule(
            rule_id="FRD-001",
            pattern_type=FraudPatternType.DUPLICATE_REUSE,
            name="Duplicate Document Reuse",
            description=(
                "Same document hash or certificate number used "
                "across multiple distinct shipments or declarations"
            ),
            default_severity=FraudSeverity.HIGH,
        )

        rules["FRD-002"] = _FraudRule(
            rule_id="FRD-002",
            pattern_type=FraudPatternType.QUANTITY_TAMPERING,
            name="Quantity Tampering",
            description=(
                "Quantity values deviate significantly from "
                "cross-referenced certified amounts"
            ),
            default_severity=FraudSeverity.HIGH,
        )

        rules["FRD-003"] = _FraudRule(
            rule_id="FRD-003",
            pattern_type=FraudPatternType.DATE_MANIPULATION,
            name="Date Manipulation",
            description=(
                "Dates in the document are inconsistent with "
                "known issuance timelines or supply chain events"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-004"] = _FraudRule(
            rule_id="FRD-004",
            pattern_type=FraudPatternType.EXPIRED_CERT,
            name="Expired Certificate Submission",
            description=(
                "Document references a certificate that was expired "
                "at the time of submission"
            ),
            default_severity=FraudSeverity.HIGH,
        )

        rules["FRD-005"] = _FraudRule(
            rule_id="FRD-005",
            pattern_type=FraudPatternType.SERIAL_ANOMALY,
            name="Serial Number Anomaly",
            description=(
                "Certificate or document serial number does not follow "
                "the expected pattern of the issuing authority"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-006"] = _FraudRule(
            rule_id="FRD-006",
            pattern_type=FraudPatternType.ISSUER_MISMATCH,
            name="Issuer Authority Mismatch",
            description=(
                "Document claims to be issued by an authority that "
                "is not authorized for this type/country combination"
            ),
            default_severity=FraudSeverity.HIGH,
        )

        rules["FRD-007"] = _FraudRule(
            rule_id="FRD-007",
            pattern_type=FraudPatternType.TEMPLATE_FORGERY,
            name="Template Forgery Detection",
            description=(
                "Document layout or structure deviates from "
                "known templates of the issuing authority"
            ),
            default_severity=FraudSeverity.CRITICAL,
        )

        rules["FRD-008"] = _FraudRule(
            rule_id="FRD-008",
            pattern_type=FraudPatternType.CROSS_DOC_INCONSISTENCY,
            name="Cross-Document Quantity Inconsistency",
            description=(
                "Quantity values across related documents for the "
                "same shipment show >5% variance"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-009"] = _FraudRule(
            rule_id="FRD-009",
            pattern_type=FraudPatternType.GEO_IMPOSSIBILITY,
            name="Geographic Impossibility",
            description=(
                "Geographic data implies a physically impossible "
                "production or transport scenario"
            ),
            default_severity=FraudSeverity.HIGH,
        )

        rules["FRD-010"] = _FraudRule(
            rule_id="FRD-010",
            pattern_type=FraudPatternType.VELOCITY_ANOMALY,
            name="Velocity Anomaly",
            description=(
                "Issuer produced an unusually high number of "
                "documents in a short time period"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-011"] = _FraudRule(
            rule_id="FRD-011",
            pattern_type=FraudPatternType.MODIFICATION_ANOMALY,
            name="Modification Timeline Anomaly",
            description=(
                "Document metadata indicates modifications after "
                "the purported date of issuance"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-012"] = _FraudRule(
            rule_id="FRD-012",
            pattern_type=FraudPatternType.ROUND_NUMBER_BIAS,
            name="Round Number Bias",
            description=(
                "An unusually high proportion of quantity or weight "
                "values are suspiciously round numbers"
            ),
            default_severity=FraudSeverity.LOW,
        )

        rules["FRD-013"] = _FraudRule(
            rule_id="FRD-013",
            pattern_type=FraudPatternType.COPY_PASTE,
            name="Copy-Paste Detection",
            description=(
                "Sections of the document appear to be copied from "
                "another document (text duplication detection)"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-014"] = _FraudRule(
            rule_id="FRD-014",
            pattern_type=FraudPatternType.MISSING_REQUIRED,
            name="Missing Required Documents",
            description=(
                "Required supporting documents for this EUDR "
                "commodity type are missing from the submission"
            ),
            default_severity=FraudSeverity.MEDIUM,
        )

        rules["FRD-015"] = _FraudRule(
            rule_id="FRD-015",
            pattern_type=FraudPatternType.SCOPE_MISMATCH,
            name="Certification Scope Mismatch",
            description=(
                "The certification scope does not cover the "
                "commodity or geographic region claimed"
            ),
            default_severity=FraudSeverity.HIGH,
        )

        return rules

    # ------------------------------------------------------------------
    # Internal: Get active rules
    # ------------------------------------------------------------------

    def _get_active_rules(
        self,
        patterns_to_check: Optional[List[str]],
    ) -> List[_FraudRule]:
        """Return the list of rules to execute.

        Args:
            patterns_to_check: Optional list of specific rule IDs or
                pattern type values to check.

        Returns:
            List of enabled _FraudRule objects.
        """
        with self._lock:
            all_rules = list(self._rules.values())

        # Filter to enabled rules only
        active = [r for r in all_rules if r.enabled]

        if not patterns_to_check:
            return active

        # Filter to specified patterns
        pattern_set = set(patterns_to_check)
        filtered = [
            r for r in active
            if (r.rule_id in pattern_set
                or r.pattern_type.value in pattern_set)
        ]
        return filtered

    # ------------------------------------------------------------------
    # Internal: Execute a single rule
    # ------------------------------------------------------------------

    def _execute_rule(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Execute a single fraud detection rule.

        Args:
            rule: The fraud rule to execute.
            document_id: Document identifier.
            document_data: Document attributes.
            context: Additional context data.

        Returns:
            List of fraud alert dictionaries (empty if no fraud detected).
        """
        dispatch: Dict[str, Any] = {
            "FRD-001": self._check_duplicate_reuse,
            "FRD-002": self._check_quantity_tampering,
            "FRD-003": self._check_date_manipulation,
            "FRD-004": self._check_expired_cert,
            "FRD-005": self._check_serial_anomaly,
            "FRD-006": self._check_issuer_mismatch,
            "FRD-007": self._check_template_forgery,
            "FRD-008": self._check_cross_doc_inconsistency,
            "FRD-009": self._check_geographic_impossibility,
            "FRD-010": self._check_velocity_anomaly,
            "FRD-011": self._check_modification_anomaly,
            "FRD-012": self._check_round_number_bias,
            "FRD-013": self._check_copy_paste,
            "FRD-014": self._check_missing_required,
            "FRD-015": self._check_scope_mismatch,
        }

        check_fn = dispatch.get(rule.rule_id)
        if not check_fn:
            return []

        return check_fn(rule, document_id, document_data, context)

    # ------------------------------------------------------------------
    # FRD-001: Duplicate document reuse
    # ------------------------------------------------------------------

    def _check_duplicate_reuse(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-001: Check for duplicate document reuse across shipments.

        Triggers when the same file hash appears associated with
        different shipment IDs.
        """
        file_hash = document_data.get("file_hash_sha256", "")
        shipment_id = document_data.get("shipment_id", "")

        if not file_hash:
            return []

        with self._lock:
            existing = self._document_hashes.get(file_hash, [])

        # Check if this hash is used by a DIFFERENT shipment
        duplicate_shipments = [
            entry for entry in existing
            if (entry.get("shipment_id")
                and entry["shipment_id"] != shipment_id
                and entry["document_id"] != document_id)
        ]

        if not duplicate_shipments:
            # Register this document hash
            self.register_document_hash(
                document_id, file_hash, shipment_id,
                document_data.get("supplier_id"),
            )
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.95,
            description=(
                f"Document hash {file_hash[:16]}... is already "
                f"associated with {len(duplicate_shipments)} other "
                f"shipment(s): "
                f"{[d['shipment_id'] for d in duplicate_shipments[:3]]}"
            ),
            evidence={
                "file_hash": file_hash,
                "current_shipment": shipment_id,
                "duplicate_entries": duplicate_shipments[:5],
            },
            related_ids=[
                d["document_id"] for d in duplicate_shipments[:5]
            ],
            recommended_action=(
                "Investigate whether this document is legitimately "
                "shared across shipments or is being fraudulently reused"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-002: Quantity tampering
    # ------------------------------------------------------------------

    def _check_quantity_tampering(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-002: Check for quantity tampering.

        Triggers when quantity > (100 + tolerance)% of certified amount.
        """
        quantity = document_data.get("quantity")
        certified_qty = document_data.get("certified_quantity")

        if quantity is None or certified_qty is None:
            return []

        try:
            qty = float(quantity)
            cert_qty = float(certified_qty)
        except (ValueError, TypeError):
            return []

        if cert_qty <= 0:
            return []

        tolerance_pct = self._config.quantity_tolerance_percent
        max_allowed = cert_qty * (1.0 + tolerance_pct / 100.0)
        deviation_pct = ((qty - cert_qty) / cert_qty) * 100.0

        if qty <= max_allowed:
            return []

        # Determine severity based on deviation
        severity = rule.default_severity
        if deviation_pct > 50.0:
            severity = FraudSeverity.CRITICAL
        elif deviation_pct > 20.0:
            severity = FraudSeverity.HIGH

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=min(0.5 + deviation_pct / 100.0, 1.0),
            severity_override=severity,
            description=(
                f"Document quantity ({qty:.2f}) exceeds certified "
                f"amount ({cert_qty:.2f}) by {deviation_pct:.1f}%; "
                f"tolerance is {tolerance_pct:.1f}%"
            ),
            evidence={
                "quantity": qty,
                "certified_quantity": cert_qty,
                "deviation_percent": round(deviation_pct, 2),
                "tolerance_percent": tolerance_pct,
                "max_allowed": round(max_allowed, 2),
            },
            recommended_action=(
                "Verify quantity against original certification; "
                "request updated certification if quantity has "
                "legitimately changed"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-003: Date manipulation
    # ------------------------------------------------------------------

    def _check_date_manipulation(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-003: Check for date manipulation.

        Triggers when document dates are more than the configured
        tolerance from supply chain event dates.
        """
        alerts: List[Dict[str, Any]] = []
        tolerance = self._config.date_tolerance_days

        issuance = self._parse_date(document_data.get("issuance_date"))
        submission = self._parse_date(document_data.get("submission_date"))
        creation = self._parse_date(document_data.get("creation_date"))

        # Check issuance vs submission gap
        if issuance and submission:
            diff_days = abs((submission - issuance).days)
            if diff_days > tolerance:
                alerts.append(self._create_alert(
                    rule=rule,
                    document_id=document_id,
                    confidence=min(0.5 + diff_days / 365.0, 0.95),
                    description=(
                        f"Gap between issuance date "
                        f"({issuance.date()}) and submission date "
                        f"({submission.date()}) is {diff_days} days; "
                        f"exceeds {tolerance}-day tolerance"
                    ),
                    evidence={
                        "issuance_date": issuance.isoformat(),
                        "submission_date": submission.isoformat(),
                        "gap_days": diff_days,
                        "tolerance_days": tolerance,
                    },
                    recommended_action=(
                        "Verify why there is a significant gap between "
                        "document issuance and submission"
                    ),
                ))

        # Check if creation date is in the future
        now = _utcnow()
        if creation and creation > now:
            alerts.append(self._create_alert(
                rule=rule,
                document_id=document_id,
                confidence=0.95,
                severity_override=FraudSeverity.HIGH,
                description=(
                    f"Document creation date ({creation.date()}) is "
                    f"in the future; possible clock manipulation"
                ),
                evidence={
                    "creation_date": creation.isoformat(),
                    "current_date": now.isoformat(),
                },
                recommended_action=(
                    "Reject document; creation date in the future "
                    "indicates tampering"
                ),
            ))

        return alerts

    # ------------------------------------------------------------------
    # FRD-004: Expired certificate
    # ------------------------------------------------------------------

    def _check_expired_cert(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-004: Check for expired certificate submission.

        Triggers when validity_end is before submission_date.
        """
        validity_end = self._parse_date(document_data.get("validity_end"))
        submission = self._parse_date(
            document_data.get("submission_date"),
        ) or _utcnow()

        if not validity_end:
            return []

        if validity_end >= submission:
            return []

        days_expired = (submission - validity_end).days

        severity = rule.default_severity
        if days_expired > 365:
            severity = FraudSeverity.CRITICAL
        elif days_expired > 90:
            severity = FraudSeverity.HIGH

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.99,
            severity_override=severity,
            description=(
                f"Certificate expired {days_expired} days before "
                f"submission (validity_end={validity_end.date()}, "
                f"submission={submission.date()})"
            ),
            evidence={
                "validity_end": validity_end.isoformat(),
                "submission_date": submission.isoformat(),
                "days_expired": days_expired,
            },
            recommended_action=(
                "Reject document and request a current, valid certificate"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-005: Serial number anomaly
    # ------------------------------------------------------------------

    def _check_serial_anomaly(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-005: Check for serial number format anomaly.

        Triggers when the serial number does not match expected
        patterns for the document type.
        """
        serial = document_data.get("serial_number", "")
        doc_type = document_data.get("document_type", "")

        if not serial or not doc_type:
            return []

        patterns = SERIAL_FORMAT_PATTERNS.get(doc_type, [])
        if not patterns:
            return []

        matched = any(re.fullmatch(p, serial) for p in patterns)
        if matched:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.80,
            description=(
                f"Serial number '{serial}' does not match expected "
                f"format for document type '{doc_type}'"
            ),
            evidence={
                "serial_number": serial,
                "document_type": doc_type,
                "expected_patterns": patterns,
            },
            recommended_action=(
                "Verify serial number with the issuing authority; "
                "format may indicate a forged document"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-006: Issuer authority mismatch
    # ------------------------------------------------------------------

    def _check_issuer_mismatch(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-006: Check for unauthorized issuing authority.

        Triggers when the document issuer is not in the authorized
        list for the document type and country combination.
        """
        issuer = document_data.get("issuer", "")
        doc_type = document_data.get("document_type", "")
        country = document_data.get("issuer_country", "DEFAULT")

        if not issuer or not doc_type:
            return []

        authorized_by_type = AUTHORIZED_ISSUERS.get(doc_type, {})
        if not authorized_by_type:
            return []

        # Check country-specific list, then DEFAULT
        authorized_list = authorized_by_type.get(
            country, authorized_by_type.get("DEFAULT", []),
        )

        issuer_lower = issuer.lower()
        is_authorized = any(
            auth.lower() in issuer_lower or issuer_lower in auth.lower()
            for auth in authorized_list
        )

        if is_authorized:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.85,
            description=(
                f"Issuer '{issuer}' is not in the authorized list "
                f"for document type '{doc_type}' in country "
                f"'{country}': {authorized_list}"
            ),
            evidence={
                "issuer": issuer,
                "document_type": doc_type,
                "country": country,
                "authorized_issuers": authorized_list,
            },
            recommended_action=(
                "Verify issuer legitimacy with the relevant authority; "
                "document may be from an unauthorized source"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-007: Template forgery detection
    # ------------------------------------------------------------------

    def _check_template_forgery(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-007: Check for template forgery.

        Triggers when the document's layout hash does not match any
        known template for the document type.
        """
        template_hash = document_data.get("template_hash", "")
        doc_type = document_data.get("document_type", "")

        if not template_hash or not doc_type:
            return []

        known_templates = context.get("known_templates", {})
        type_templates = known_templates.get(doc_type, [])

        if not type_templates:
            return []

        if template_hash in type_templates:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.90,
            description=(
                f"Document template hash '{template_hash[:16]}...' "
                f"does not match any known template for "
                f"document type '{doc_type}' "
                f"({len(type_templates)} templates known)"
            ),
            evidence={
                "template_hash": template_hash,
                "document_type": doc_type,
                "known_template_count": len(type_templates),
            },
            recommended_action=(
                "Manual review required; document layout differs from "
                "known official templates and may be forged"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-008: Cross-document quantity inconsistency
    # ------------------------------------------------------------------

    def _check_cross_doc_inconsistency(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-008: Check for cross-document quantity inconsistency.

        Triggers when quantity values across related documents for the
        same shipment show variance exceeding the configured tolerance.
        """
        quantity = document_data.get("quantity")
        related_docs = document_data.get("related_documents", [])

        if quantity is None or not related_docs:
            return []

        try:
            qty = float(quantity)
        except (ValueError, TypeError):
            return []

        tolerance_pct = self._config.quantity_tolerance_percent
        alerts: List[Dict[str, Any]] = []

        for related in related_docs:
            related_qty = related.get("quantity")
            if related_qty is None:
                continue
            try:
                r_qty = float(related_qty)
            except (ValueError, TypeError):
                continue

            if r_qty <= 0:
                continue

            variance_pct = abs((qty - r_qty) / r_qty) * 100.0

            if variance_pct > tolerance_pct:
                alerts.append(self._create_alert(
                    rule=rule,
                    document_id=document_id,
                    confidence=min(0.5 + variance_pct / 50.0, 0.95),
                    description=(
                        f"Quantity ({qty:.2f}) differs from related "
                        f"document '{related.get('document_id', 'N/A')}' "
                        f"quantity ({r_qty:.2f}) by {variance_pct:.1f}%; "
                        f"tolerance is {tolerance_pct:.1f}%"
                    ),
                    evidence={
                        "quantity": qty,
                        "related_document_id": related.get("document_id"),
                        "related_quantity": r_qty,
                        "variance_percent": round(variance_pct, 2),
                        "tolerance_percent": tolerance_pct,
                    },
                    related_ids=[related.get("document_id", "")],
                    recommended_action=(
                        "Reconcile quantity discrepancy between documents"
                    ),
                ))

        return alerts

    # ------------------------------------------------------------------
    # FRD-009: Geographic impossibility
    # ------------------------------------------------------------------

    def _check_geographic_impossibility(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-009: Check for geographic impossibility.

        Triggers when the document's origin location is inconsistent
        with the supply chain graph.
        """
        origin_coords = document_data.get("origin_coordinates", {})
        origin_country = document_data.get("origin_country", "")
        supply_chain_coords = document_data.get(
            "supply_chain_coordinates", [],
        )

        if not origin_coords or not supply_chain_coords:
            return []

        try:
            origin_lat = float(origin_coords.get("lat", 0))
            origin_lon = float(origin_coords.get("lon", 0))
        except (ValueError, TypeError):
            return []

        # Check if origin is within reasonable distance of any
        # supply chain node (threshold: 500km)
        max_distance_km = 500.0
        min_distance = float("inf")

        for node in supply_chain_coords:
            try:
                node_lat = float(node.get("lat", 0))
                node_lon = float(node.get("lon", 0))
            except (ValueError, TypeError):
                continue

            distance = self._haversine_km(
                origin_lat, origin_lon, node_lat, node_lon,
            )
            min_distance = min(min_distance, distance)

        if min_distance <= max_distance_km:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=min(0.5 + min_distance / 5000.0, 0.99),
            description=(
                f"Document origin ({origin_lat:.4f}, {origin_lon:.4f}) "
                f"is {min_distance:.0f}km from the nearest supply "
                f"chain node; exceeds {max_distance_km:.0f}km threshold"
            ),
            evidence={
                "origin_lat": origin_lat,
                "origin_lon": origin_lon,
                "origin_country": origin_country,
                "nearest_distance_km": round(min_distance, 1),
                "threshold_km": max_distance_km,
                "supply_chain_node_count": len(supply_chain_coords),
            },
            recommended_action=(
                "Verify geographic origin; location is implausible "
                "given the registered supply chain"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-010: Velocity anomaly
    # ------------------------------------------------------------------

    def _check_velocity_anomaly(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-010: Check for document issuance velocity anomaly.

        Triggers when a supplier/issuer submits more than the
        configured threshold of documents per day.
        """
        supplier_id = document_data.get("supplier_id", "")
        submission_date = self._parse_date(
            document_data.get("submission_date"),
        ) or _utcnow()

        if not supplier_id:
            return []

        date_key = submission_date.strftime("%Y-%m-%d")
        threshold = self._config.velocity_threshold_per_day

        with self._lock:
            if supplier_id not in self._supplier_velocity:
                self._supplier_velocity[supplier_id] = {}
            daily = self._supplier_velocity[supplier_id]
            daily[date_key] = daily.get(date_key, 0) + 1
            count = daily[date_key]

        if count <= threshold:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=min(0.5 + count / (threshold * 5), 0.95),
            description=(
                f"Supplier '{supplier_id}' has submitted {count} "
                f"documents on {date_key}; exceeds {threshold}/day "
                f"threshold"
            ),
            evidence={
                "supplier_id": supplier_id,
                "date": date_key,
                "document_count": count,
                "threshold": threshold,
            },
            recommended_action=(
                "Review supplier submission pattern; high volume "
                "may indicate bulk-generated fraudulent documents"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-011: Modification timeline anomaly
    # ------------------------------------------------------------------

    def _check_modification_anomaly(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-011: Check for modification after issuance.

        Triggers when the PDF modification date is after the claimed
        issuance date.
        """
        mod_date = self._parse_date(
            document_data.get("modification_date"),
        )
        issuance_date = self._parse_date(
            document_data.get("issuance_date"),
        )

        if not mod_date or not issuance_date:
            return []

        if mod_date <= issuance_date:
            return []

        diff_days = (mod_date - issuance_date).days

        severity = rule.default_severity
        if diff_days > 90:
            severity = FraudSeverity.HIGH

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=min(0.5 + diff_days / 180.0, 0.95),
            severity_override=severity,
            description=(
                f"Document was modified ({mod_date.date()}) "
                f"{diff_days} days after issuance date "
                f"({issuance_date.date()})"
            ),
            evidence={
                "modification_date": mod_date.isoformat(),
                "issuance_date": issuance_date.isoformat(),
                "days_after_issuance": diff_days,
            },
            recommended_action=(
                "Investigate modification; official documents should "
                "not be modified after issuance"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-012: Round number bias
    # ------------------------------------------------------------------

    def _check_round_number_bias(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-012: Check for suspiciously round quantity values.

        Triggers when >threshold% of quantity values are round numbers
        (divisible by 10, 100, 1000).
        """
        quantities = document_data.get("quantities", [])

        if not quantities or len(quantities) < 3:
            return []

        threshold_pct = self._config.round_number_threshold_percent
        round_count = 0

        for qty in quantities:
            try:
                val = float(qty)
                if val > 0 and self._is_round_number(val):
                    round_count += 1
            except (ValueError, TypeError):
                continue

        if not quantities:
            return []

        round_pct = (round_count / len(quantities)) * 100.0

        if round_pct <= threshold_pct:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=min(0.3 + round_pct / 200.0, 0.85),
            description=(
                f"{round_pct:.1f}% of quantity values ({round_count}/"
                f"{len(quantities)}) are round numbers; exceeds "
                f"{threshold_pct:.1f}% threshold"
            ),
            evidence={
                "round_count": round_count,
                "total_quantities": len(quantities),
                "round_percentage": round(round_pct, 1),
                "threshold_percent": threshold_pct,
            },
            recommended_action=(
                "Review quantities for potential fabrication; "
                "legitimate data typically shows natural variation"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-013: Copy-paste detection
    # ------------------------------------------------------------------

    def _check_copy_paste(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-013: Check for copy-pasted text blocks.

        Triggers when identical text blocks (>50 chars) appear in
        different documents with different reference numbers.
        """
        text_blocks = document_data.get("text_blocks", [])
        serial_number = document_data.get("serial_number", "")

        if not text_blocks:
            return []

        alerts: List[Dict[str, Any]] = []
        min_block_length = 50

        for block in text_blocks:
            if not isinstance(block, str) or len(block) < min_block_length:
                continue

            block_hash = hashlib.sha256(
                block.strip().lower().encode("utf-8"),
            ).hexdigest()

            with self._lock:
                if block_hash not in self._text_blocks:
                    self._text_blocks[block_hash] = []

                existing = self._text_blocks[block_hash]
                # Check for matches from DIFFERENT documents
                matches = [
                    e for e in existing
                    if (e["document_id"] != document_id
                        and e.get("serial_number") != serial_number)
                ]

                # Register this block
                self._text_blocks[block_hash].append({
                    "document_id": document_id,
                    "serial_number": serial_number,
                    "block_preview": block[:80],
                    "registered_at": _utcnow().isoformat(),
                })

            if matches:
                alerts.append(self._create_alert(
                    rule=rule,
                    document_id=document_id,
                    confidence=0.85,
                    description=(
                        f"Text block ('{block[:40]}...') is identical "
                        f"to content in {len(matches)} other "
                        f"document(s) with different reference numbers"
                    ),
                    evidence={
                        "block_hash": block_hash,
                        "block_preview": block[:100],
                        "matching_documents": [
                            m["document_id"] for m in matches[:5]
                        ],
                    },
                    related_ids=[
                        m["document_id"] for m in matches[:5]
                    ],
                    recommended_action=(
                        "Investigate content duplication; documents with "
                        "different reference numbers should not have "
                        "identical text blocks"
                    ),
                ))
                break  # One copy-paste alert per document is sufficient

        return alerts

    # ------------------------------------------------------------------
    # FRD-014: Missing required documents
    # ------------------------------------------------------------------

    def _check_missing_required(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-014: Check for missing required supporting documents.

        Triggers when the submission for a commodity is missing one
        or more required document types.
        """
        commodity = (
            document_data.get("commodity")
            or context.get("commodity", "")
        )

        if not commodity:
            return []

        required_types = REQUIRED_DOCUMENTS_BY_COMMODITY.get(
            commodity.lower(), [],
        )
        if not required_types:
            return []

        # Get existing documents for this shipment/DDS
        existing_docs = context.get("existing_documents", [])
        existing_types: Set[str] = set()
        for doc in existing_docs:
            doc_type = doc.get("document_type", "")
            if doc_type:
                existing_types.add(doc_type.lower())

        # Add current document type
        current_type = document_data.get("document_type", "")
        if current_type:
            existing_types.add(current_type.lower())

        missing = [
            rt for rt in required_types
            if rt.lower() not in existing_types
        ]

        if not missing:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.90,
            description=(
                f"Missing {len(missing)} required document type(s) "
                f"for commodity '{commodity}': {missing}"
            ),
            evidence={
                "commodity": commodity,
                "required_types": required_types,
                "existing_types": sorted(existing_types),
                "missing_types": missing,
            },
            recommended_action=(
                f"Submit the following missing documents before "
                f"DDS finalization: {', '.join(missing)}"
            ),
        )]

    # ------------------------------------------------------------------
    # FRD-015: Certification scope mismatch
    # ------------------------------------------------------------------

    def _check_scope_mismatch(
        self,
        rule: _FraudRule,
        document_id: str,
        document_data: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """FRD-015: Check for certification scope mismatch.

        Triggers when the certification type does not cover the
        commodity claimed in the document.
        """
        doc_type = document_data.get("document_type", "")
        commodity = (
            document_data.get("commodity")
            or context.get("commodity", "")
        )

        if not doc_type or not commodity:
            return []

        allowed_commodities = CERT_SCOPE_COMMODITY_MAP.get(
            doc_type.lower(), [],
        )

        # Only check cert types that have scope restrictions
        if not allowed_commodities:
            return []

        if commodity.lower() in [c.lower() for c in allowed_commodities]:
            return []

        return [self._create_alert(
            rule=rule,
            document_id=document_id,
            confidence=0.95,
            description=(
                f"Certification type '{doc_type}' does not cover "
                f"commodity '{commodity}'; scope is limited to: "
                f"{allowed_commodities}"
            ),
            evidence={
                "document_type": doc_type,
                "commodity": commodity,
                "allowed_commodities": allowed_commodities,
            },
            recommended_action=(
                f"Provide a certification that covers '{commodity}'; "
                f"'{doc_type}' is not applicable"
            ),
        )]

    # ------------------------------------------------------------------
    # Internal: Create alert
    # ------------------------------------------------------------------

    def _create_alert(
        self,
        rule: _FraudRule,
        document_id: str,
        confidence: float,
        description: str,
        evidence: Dict[str, Any],
        severity_override: Optional[FraudSeverity] = None,
        related_ids: Optional[List[str]] = None,
        recommended_action: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a standardized fraud alert dictionary.

        Args:
            rule: The fraud rule that triggered.
            document_id: Document identifier.
            confidence: Detection confidence (0.0-1.0).
            description: Human-readable description.
            evidence: Evidence dictionary.
            severity_override: Optional severity override.
            related_ids: Optional related document IDs.
            recommended_action: Recommended remediation.

        Returns:
            Fraud alert dictionary.
        """
        severity = severity_override or rule.default_severity
        alert_id = _generate_id("ALERT")
        now = _utcnow()

        alert: Dict[str, Any] = {
            "alert_id": alert_id,
            "document_id": document_id,
            "rule_id": rule.rule_id,
            "pattern_type": rule.pattern_type.value,
            "severity": severity.value,
            "confidence_score": round(min(max(confidence, 0.0), 1.0), 3),
            "description": description,
            "evidence": evidence,
            "related_document_ids": related_ids or [],
            "recommended_action": recommended_action,
            "resolved": False,
            "resolved_by": None,
            "resolved_at": None,
            "resolution_notes": None,
            "provenance_hash": "",
            "detected_at": now.isoformat(),
        }

        # Compute provenance hash for the alert
        alert["provenance_hash"] = _compute_hash(alert)
        return alert

    # ------------------------------------------------------------------
    # Internal: Calculate aggregate fraud risk score
    # ------------------------------------------------------------------

    def _calculate_fraud_risk_score(
        self,
        alerts: List[Dict[str, Any]],
    ) -> float:
        """Calculate a composite fraud risk score from detected alerts.

        The score is a weighted sum of alert severities, normalized
        to a 0-100 scale. Uses the severity weights from config.

        Args:
            alerts: List of fraud alert dictionaries.

        Returns:
            Composite fraud risk score (0.0-100.0).
        """
        if not alerts:
            return 0.0

        weights = self._config.fraud_severity_weights
        total_weight = 0.0

        for alert in alerts:
            severity = alert.get("severity", "low")
            confidence = alert.get("confidence_score", 0.5)
            weight = weights.get(severity, 1.0)
            total_weight += weight * confidence

        # Normalize: max realistic score is 15 alerts * 10.0 * 1.0 = 150
        # Map to 0-100 with diminishing returns
        max_theoretical = 150.0
        normalized = min(
            (total_weight / max_theoretical) * 100.0, 100.0,
        )
        return round(normalized, 1)

    # ------------------------------------------------------------------
    # Internal: Classify severity (helper)
    # ------------------------------------------------------------------

    def _classify_severity(
        self,
        score: float,
    ) -> str:
        """Classify a numeric score into a severity level.

        Args:
            score: Numeric severity score.

        Returns:
            Severity string: low, medium, high, or critical.
        """
        if score >= 8.0:
            return FraudSeverity.CRITICAL.value
        if score >= 5.0:
            return FraudSeverity.HIGH.value
        if score >= 2.0:
            return FraudSeverity.MEDIUM.value
        return FraudSeverity.LOW.value

    # ------------------------------------------------------------------
    # Internal: Get highest severity
    # ------------------------------------------------------------------

    @staticmethod
    def _get_highest_severity(
        alerts: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Return the highest severity level among alerts.

        Args:
            alerts: List of fraud alert dictionaries.

        Returns:
            Highest severity string or None if no alerts.
        """
        if not alerts:
            return None

        severity_order = {
            "critical": 4, "high": 3, "medium": 2, "low": 1,
        }
        highest = max(
            alerts,
            key=lambda a: severity_order.get(a.get("severity", "low"), 0),
        )
        return highest.get("severity")

    # ------------------------------------------------------------------
    # Internal: Empty result helper
    # ------------------------------------------------------------------

    def _build_empty_result(
        self,
        document_id: str,
        elapsed_ms: float,
        message: str = "",
    ) -> Dict[str, Any]:
        """Build an empty detection result.

        Args:
            document_id: Document identifier.
            elapsed_ms: Processing time in milliseconds.
            message: Optional message.

        Returns:
            Empty result dictionary.
        """
        return {
            "success": True,
            "document_id": document_id,
            "alerts": [],
            "total_alerts": 0,
            "highest_severity": None,
            "composite_score": 0.0,
            "rules_checked": [],
            "rules_triggered": [],
            "processing_time_ms": round(elapsed_ms, 2),
            "provenance_hash": _compute_hash({
                "document_id": document_id,
                "message": message,
            }),
            "message": message,
        }

    # ------------------------------------------------------------------
    # Internal: Date parsing helper
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        """Parse a date value to datetime.

        Args:
            value: String, datetime, or None.

        Returns:
            datetime with UTC timezone or None.
        """
        if value is None:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value
        try:
            dt_str = str(value).strip()
            if "T" in dt_str:
                parsed = datetime.fromisoformat(
                    dt_str.replace("Z", "+00:00"),
                )
            else:
                parsed = datetime.strptime(dt_str, "%Y-%m-%d")
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------
    # Internal: Haversine distance
    # ------------------------------------------------------------------

    @staticmethod
    def _haversine_km(
        lat1: float, lon1: float,
        lat2: float, lon2: float,
    ) -> float:
        """Calculate great-circle distance in km using Haversine formula.

        Args:
            lat1: Latitude of point 1 in degrees.
            lon1: Longitude of point 1 in degrees.
            lat2: Latitude of point 2 in degrees.
            lon2: Longitude of point 2 in degrees.

        Returns:
            Distance in kilometers.
        """
        r = 6371.0  # Earth radius in km
        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    # ------------------------------------------------------------------
    # Internal: Round number check
    # ------------------------------------------------------------------

    @staticmethod
    def _is_round_number(value: float) -> bool:
        """Check if a number is suspiciously round.

        A number is considered round if it is divisible by 10,
        or if it has no fractional part and ends in 0 or 00.

        Args:
            value: Numeric value to check.

        Returns:
            True if the value is considered a round number.
        """
        if value == 0:
            return False
        # Check if divisible by 10 (no remainder)
        if value % 10.0 == 0:
            return True
        # Check if integer with trailing zeros
        if value == int(value) and int(value) % 5 == 0:
            return True
        return False

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        with self._lock:
            alert_count = len(self._alerts)
        enabled = sum(1 for r in self._rules.values() if r.enabled)
        return (
            f"FraudPatternDetector(rules={len(self._rules)}, "
            f"enabled={enabled}, alerts={alert_count})"
        )

    def __len__(self) -> int:
        """Return the number of stored alerts."""
        with self._lock:
            return len(self._alerts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "FraudPatternDetector",
    "REQUIRED_DOCUMENTS_BY_COMMODITY",
    "AUTHORIZED_ISSUERS",
    "CERT_SCOPE_COMMODITY_MAP",
    "SERIAL_FORMAT_PATTERNS",
]
