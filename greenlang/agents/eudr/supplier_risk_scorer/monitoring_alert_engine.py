# -*- coding: utf-8 -*-
"""
Monitoring Alert Engine - AGENT-EUDR-017 Engine 3

Continuous supplier risk monitoring and alert generation with configurable
frequency, multi-severity alerting, sanction screening, watchlist management,
automated re-assessment scheduling, risk heat map generation, and portfolio
risk aggregation.

Monitoring and Alert Capabilities:
    - Configurable monitoring frequency per supplier (daily, weekly,
      monthly, quarterly)
    - Alert types: RISK_THRESHOLD, CERTIFICATION_EXPIRY, DOCUMENT_MISSING,
      DD_OVERDUE, SANCTION_HIT, BEHAVIOR_CHANGE
    - Alert severity: INFO, WARNING, HIGH, CRITICAL
    - Risk threshold breach detection with configurable thresholds
    - Change detection in supplier behavior/characteristics
    - Sanction list screening (EU sanctions, OFAC, UN)
    - Watchlist management (add/remove/query flagged suppliers)
    - Automated re-assessment scheduling and triggers
    - Risk heat map generation for supplier portfolio
    - Portfolio risk aggregation (concentration by country, commodity,
      certification status)
    - Alert acknowledgment and resolution tracking

Zero-Hallucination: All monitoring checks and alert generation are
    deterministic rule-based operations. No LLM calls in the alerting
    path. Sanction screening uses exact string matching against lists.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-017 Supplier Risk Scorer (GL-EUDR-SRS-017)
Status: Production Ready
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import get_config
from .metrics import record_alert_generation
from .models import (
    AlertSeverity,
    AlertType,
    MonitoringConfig,
    MonitoringFrequency,
    RiskLevel,
    SupplierAlert,
)
from .provenance import get_tracker

from greenlang.schemas import utcnow

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Alert severity thresholds (risk score ranges).
_ALERT_SEVERITY_THRESHOLDS: Dict[AlertSeverity, Tuple[float, float]] = {
    AlertSeverity.INFO: (0.0, 30.0),
    AlertSeverity.WARNING: (30.0, 50.0),
    AlertSeverity.HIGH: (50.0, 75.0),
    AlertSeverity.CRITICAL: (75.0, 100.0),
}

#: Certification expiry warning buffer (days).
_CERT_EXPIRY_WARNING_DAYS: int = 90

#: Due diligence overdue threshold (days).
_DD_OVERDUE_THRESHOLD_DAYS: int = 30

#: Behavior change detection thresholds.
_BEHAVIOR_CHANGE_RISK_THRESHOLD: float = 15.0  # Risk change >15 points
_BEHAVIOR_CHANGE_VOLUME_THRESHOLD: float = 30.0  # Volume change >30%

#: Watchlist maximum size.
_WATCHLIST_MAX_SIZE: int = 10000

#: Portfolio risk concentration threshold (HHI).
_PORTFOLIO_CONCENTRATION_THRESHOLD: int = 1800

#: EU sanction list (simplified for demo).
_EU_SANCTIONS: Set[str] = {
    "COMPANY_SANCTIONED_1",
    "COMPANY_SANCTIONED_2",
    "SUPPLIER_SANCTIONED_BR_123",
}

#: OFAC sanction list (simplified for demo).
_OFAC_SANCTIONS: Set[str] = {
    "COMPANY_OFAC_1",
    "SUPPLIER_OFAC_ID_456",
}

#: UN sanction list (simplified for demo).
_UN_SANCTIONS: Set[str] = {
    "COMPANY_UN_1",
}

#: Monitoring frequency to re-assessment interval mapping (days).
_FREQUENCY_TO_DAYS: Dict[MonitoringFrequency, int] = {
    MonitoringFrequency.DAILY: 1,
    MonitoringFrequency.WEEKLY: 7,
    MonitoringFrequency.BIWEEKLY: 14,
    MonitoringFrequency.MONTHLY: 30,
    MonitoringFrequency.QUARTERLY: 90,
}

# ---------------------------------------------------------------------------
# MonitoringAlertEngine
# ---------------------------------------------------------------------------

class MonitoringAlertEngine:
    """Continuous supplier risk monitoring and alert generation.

    Monitors supplier risk profiles on a configurable schedule, detects
    threshold breaches, certification expiries, document gaps, due
    diligence delays, sanction hits, and behavior changes. Generates
    alerts with severity levels, manages watchlists, schedules
    re-assessments, and provides portfolio-level risk aggregation.

    All checks are deterministic rule-based operations. No LLM calls
    in the alerting path (zero-hallucination).

    Attributes:
        _configs: In-memory monitoring config store keyed by supplier_id.
        _alerts: In-memory alert store keyed by alert_id.
        _watchlist: Set of flagged supplier IDs.
        _lock: Threading lock for thread-safe access.
        _last_assessment_time: Dict tracking last assessment time per supplier.

    Example:
        >>> engine = MonitoringAlertEngine()
        >>> config = engine.configure_monitoring(
        ...     supplier_id="SUP-BR-12345",
        ...     frequency=MonitoringFrequency.WEEKLY,
        ...     alert_thresholds={"RISK_THRESHOLD": 70.0},
        ... )
        >>> alerts = engine.check_alerts(supplier_id="SUP-BR-12345", current_risk=75.0)
        >>> print(len(alerts), alerts[0].alert_type)
        1 RISK_THRESHOLD
    """

    def __init__(self) -> None:
        """Initialize MonitoringAlertEngine."""
        self._configs: Dict[str, MonitoringConfig] = {}
        self._alerts: Dict[str, SupplierAlert] = {}
        self._watchlist: Set[str] = set()
        self._lock = threading.Lock()
        self._last_assessment_time: Dict[str, datetime] = {}
        logger.info("MonitoringAlertEngine initialized")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def configure_monitoring(
        self,
        supplier_id: str,
        frequency: MonitoringFrequency = MonitoringFrequency.MONTHLY,
        alert_thresholds: Optional[Dict[str, float]] = None,
        enabled_alert_types: Optional[List[AlertType]] = None,
    ) -> MonitoringConfig:
        """Configure monitoring for a supplier.

        Args:
            supplier_id: Unique supplier identifier.
            frequency: Monitoring frequency (daily, weekly, monthly, quarterly).
            alert_thresholds: Dict mapping alert_type -> threshold value.
                Example: {"RISK_THRESHOLD": 70.0, "CERT_EXPIRY": 90}
            enabled_alert_types: List of alert types to enable.
                Default: all alert types enabled.

        Returns:
            MonitoringConfig with configuration details.
        """
        start_time = time.perf_counter()

        # Default alert thresholds
        if alert_thresholds is None:
            cfg = get_config()
            alert_thresholds = {
                "RISK_THRESHOLD": float(cfg.high_risk_threshold),
                "CERT_EXPIRY": float(cfg.cert_expiry_buffer_days or _CERT_EXPIRY_WARNING_DAYS),
                "DD_OVERDUE": float(cfg.dd_overdue_limit_days or _DD_OVERDUE_THRESHOLD_DAYS),
            }

        # Default enabled alert types (all)
        if enabled_alert_types is None:
            enabled_alert_types = list(AlertType)

        # Create config
        config_id = str(uuid.uuid4())
        config = MonitoringConfig(
            config_id=config_id,
            supplier_id=supplier_id,
            frequency=frequency,
            alert_thresholds=alert_thresholds,
            enabled_alert_types=[t.value for t in enabled_alert_types],
            last_check_at=None,
            next_check_at=utcnow() + timedelta(days=_FREQUENCY_TO_DAYS[frequency]),
            is_active=True,
            created_at=utcnow(),
        )

        # Store config
        with self._lock:
            self._configs[supplier_id] = config

        # Record provenance
        provenance = get_tracker()
        provenance.record(
            entity_type="monitoring",
            entity_id=config_id,
            action="configure",
            details={
                "supplier_id": supplier_id,
                "frequency": frequency.value,
                "alert_thresholds": alert_thresholds,
            },
        )

        duration = time.perf_counter() - start_time
        logger.info(
            f"Monitoring configured for supplier {supplier_id}: "
            f"frequency={frequency.value}, duration={duration:.3f}s"
        )

        return config

    def check_alerts(
        self,
        supplier_id: str,
        current_risk: float,
        certifications: Optional[List[Dict[str, Any]]] = None,
        documents: Optional[List[Dict[str, Any]]] = None,
        dd_status: Optional[Dict[str, Any]] = None,
        historical_risk: Optional[float] = None,
    ) -> List[SupplierAlert]:
        """Check for alert conditions and generate alerts.

        Args:
            supplier_id: Unique supplier identifier.
            current_risk: Current risk score (0-100).
            certifications: List of certification dicts with keys:
                - scheme, expiry_date, status.
            documents: List of document dicts with keys:
                - document_type, status, uploaded_at.
            dd_status: Due diligence status dict with keys:
                - last_assessment_date, status, overdue.
            historical_risk: Historical risk score for behavior change detection.

        Returns:
            List of SupplierAlert objects generated.
        """
        start_time = time.perf_counter()
        alerts: List[SupplierAlert] = []

        # Get monitoring config
        config = self._configs.get(supplier_id)
        if not config or not config.is_active:
            return alerts

        alert_thresholds = config.alert_thresholds
        enabled_types = [AlertType(t) for t in config.enabled_alert_types]

        # Check 1: Risk threshold breach
        if AlertType.RISK_THRESHOLD in enabled_types:
            threshold = alert_thresholds.get("RISK_THRESHOLD", 70.0)
            if current_risk >= threshold:
                severity = self._determine_severity(current_risk)
                alert = self.generate_alert(
                    supplier_id=supplier_id,
                    alert_type=AlertType.RISK_THRESHOLD,
                    severity=severity,
                    message=f"Supplier risk score {current_risk:.1f} exceeds threshold {threshold:.1f}",
                    metadata={"current_risk": current_risk, "threshold": threshold},
                )
                alerts.append(alert)

        # Check 2: Certification expiry
        if AlertType.CERTIFICATION_EXPIRY in enabled_types and certifications:
            expiry_buffer = alert_thresholds.get("CERT_EXPIRY", _CERT_EXPIRY_WARNING_DAYS)
            expiry_date_threshold = utcnow() + timedelta(days=expiry_buffer)

            for cert in certifications:
                expiry_date_str = cert.get("expiry_date")
                if expiry_date_str:
                    expiry_date = datetime.fromisoformat(expiry_date_str)
                    if expiry_date <= expiry_date_threshold:
                        days_until_expiry = (expiry_date - utcnow()).days
                        severity = (
                            AlertSeverity.CRITICAL if days_until_expiry <= 30
                            else AlertSeverity.HIGH if days_until_expiry <= 60
                            else AlertSeverity.WARNING
                        )
                        alert = self.generate_alert(
                            supplier_id=supplier_id,
                            alert_type=AlertType.CERTIFICATION_EXPIRY,
                            severity=severity,
                            message=f"Certification {cert.get('scheme')} expires in {days_until_expiry} days",
                            metadata={
                                "scheme": cert.get("scheme"),
                                "expiry_date": expiry_date_str,
                                "days_until_expiry": days_until_expiry,
                            },
                        )
                        alerts.append(alert)

        # Check 3: Missing documents
        if AlertType.DOCUMENT_MISSING in enabled_types and documents is not None:
            required_docs = ["geolocation", "dds_reference", "compliance_declaration"]
            provided_docs = {doc.get("document_type") for doc in documents}
            missing_docs = [d for d in required_docs if d not in provided_docs]

            if missing_docs:
                severity = (
                    AlertSeverity.HIGH if len(missing_docs) >= 2
                    else AlertSeverity.WARNING
                )
                alert = self.generate_alert(
                    supplier_id=supplier_id,
                    alert_type=AlertType.DOCUMENT_MISSING,
                    severity=severity,
                    message=f"Missing required documents: {', '.join(missing_docs)}",
                    metadata={"missing_documents": missing_docs},
                )
                alerts.append(alert)

        # Check 4: Due diligence overdue
        if AlertType.DD_OVERDUE in enabled_types and dd_status:
            last_assessment_str = dd_status.get("last_assessment_date")
            if last_assessment_str:
                last_assessment = datetime.fromisoformat(last_assessment_str)
                days_since_assessment = (utcnow() - last_assessment).days
                overdue_threshold = alert_thresholds.get("DD_OVERDUE", _DD_OVERDUE_THRESHOLD_DAYS)

                if days_since_assessment > overdue_threshold:
                    severity = (
                        AlertSeverity.CRITICAL if days_since_assessment > overdue_threshold * 2
                        else AlertSeverity.HIGH
                    )
                    alert = self.generate_alert(
                        supplier_id=supplier_id,
                        alert_type=AlertType.DD_OVERDUE,
                        severity=severity,
                        message=f"Due diligence overdue by {days_since_assessment - overdue_threshold} days",
                        metadata={
                            "last_assessment_date": last_assessment_str,
                            "days_overdue": days_since_assessment - overdue_threshold,
                        },
                    )
                    alerts.append(alert)

        # Check 5: Sanction hit
        if AlertType.SANCTION_HIT in enabled_types:
            sanction_result = self.screen_sanctions(supplier_id)
            if sanction_result["is_sanctioned"]:
                alert = self.generate_alert(
                    supplier_id=supplier_id,
                    alert_type=AlertType.SANCTION_HIT,
                    severity=AlertSeverity.CRITICAL,
                    message=f"Supplier found on sanction list: {', '.join(sanction_result['sanction_lists'])}",
                    metadata=sanction_result,
                )
                alerts.append(alert)

        # Check 6: Behavior change
        if AlertType.BEHAVIOR_CHANGE in enabled_types and historical_risk is not None:
            risk_change = abs(current_risk - historical_risk)
            if risk_change >= _BEHAVIOR_CHANGE_RISK_THRESHOLD:
                severity = (
                    AlertSeverity.HIGH if risk_change >= 25.0
                    else AlertSeverity.WARNING
                )
                alert = self.generate_alert(
                    supplier_id=supplier_id,
                    alert_type=AlertType.BEHAVIOR_CHANGE,
                    severity=severity,
                    message=f"Supplier risk changed by {risk_change:.1f} points",
                    metadata={
                        "previous_risk": historical_risk,
                        "current_risk": current_risk,
                        "risk_change": risk_change,
                    },
                )
                alerts.append(alert)

        # Update last check time
        with self._lock:
            if supplier_id in self._configs:
                self._configs[supplier_id].last_check_at = utcnow()
                # Schedule next check
                frequency_days = _FREQUENCY_TO_DAYS[config.frequency]
                self._configs[supplier_id].next_check_at = utcnow() + timedelta(days=frequency_days)

        duration = time.perf_counter() - start_time
        logger.info(
            f"Alert check completed for supplier {supplier_id}: "
            f"{len(alerts)} alerts generated, duration={duration:.3f}s"
        )

        return alerts

    def generate_alert(
        self,
        supplier_id: str,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SupplierAlert:
        """Generate a supplier alert.

        Args:
            supplier_id: Unique supplier identifier.
            alert_type: Type of alert.
            severity: Alert severity level.
            message: Human-readable alert message.
            metadata: Optional metadata dict with alert details.

        Returns:
            SupplierAlert object.
        """
        alert_id = str(uuid.uuid4())
        alert = SupplierAlert(
            alert_id=alert_id,
            supplier_id=supplier_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            metadata=metadata or {},
            is_acknowledged=False,
            is_resolved=False,
            acknowledged_at=None,
            resolved_at=None,
            created_at=utcnow(),
        )

        # Store alert
        with self._lock:
            self._alerts[alert_id] = alert

        # Record provenance
        provenance = get_tracker()
        provenance.record(
            entity_type="alert",
            entity_id=alert_id,
            action="generate",
            details={
                "supplier_id": supplier_id,
                "alert_type": alert_type.value,
                "severity": severity.value,
                "message": message,
            },
        )

        # Record metrics
        record_alert_generation(alert_type.value, severity.value)

        logger.info(
            f"Alert generated: {alert_id} for supplier {supplier_id}, "
            f"type={alert_type.value}, severity={severity.value}"
        )

        return alert

    def screen_sanctions(
        self,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Screen supplier against sanction lists.

        Checks EU sanctions, OFAC, and UN sanction lists for exact
        string matches.

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            Dict with screening results:
                - is_sanctioned: True if found on any list.
                - sanction_lists: List of sanction lists where found.
                - details: Additional details.
        """
        sanction_lists = []

        # Check EU sanctions
        if supplier_id in _EU_SANCTIONS:
            sanction_lists.append("EU")

        # Check OFAC
        if supplier_id in _OFAC_SANCTIONS:
            sanction_lists.append("OFAC")

        # Check UN
        if supplier_id in _UN_SANCTIONS:
            sanction_lists.append("UN")

        is_sanctioned = len(sanction_lists) > 0

        return {
            "is_sanctioned": is_sanctioned,
            "sanction_lists": sanction_lists,
            "details": f"Supplier {supplier_id} found on: {', '.join(sanction_lists)}" if is_sanctioned else "No sanctions found",
        }

    def manage_watchlist(
        self,
        action: str,
        supplier_id: str,
    ) -> Dict[str, Any]:
        """Manage watchlist (add/remove/query flagged suppliers).

        Args:
            action: Action to perform (add, remove, query).
            supplier_id: Unique supplier identifier.

        Returns:
            Dict with operation result:
                - action: Action performed.
                - supplier_id: Supplier identifier.
                - is_on_watchlist: True if supplier is on watchlist.
                - watchlist_size: Current watchlist size.
        """
        with self._lock:
            if action == "add":
                if len(self._watchlist) >= _WATCHLIST_MAX_SIZE:
                    logger.warning(f"Watchlist at maximum size {_WATCHLIST_MAX_SIZE}")
                    return {
                        "action": "add",
                        "supplier_id": supplier_id,
                        "is_on_watchlist": supplier_id in self._watchlist,
                        "watchlist_size": len(self._watchlist),
                        "error": "Watchlist at maximum size",
                    }
                self._watchlist.add(supplier_id)
                logger.info(f"Added supplier {supplier_id} to watchlist")

            elif action == "remove":
                self._watchlist.discard(supplier_id)
                logger.info(f"Removed supplier {supplier_id} from watchlist")

            elif action == "query":
                pass  # Just return status

            is_on_watchlist = supplier_id in self._watchlist
            watchlist_size = len(self._watchlist)

        return {
            "action": action,
            "supplier_id": supplier_id,
            "is_on_watchlist": is_on_watchlist,
            "watchlist_size": watchlist_size,
        }

    def schedule_reassessment(
        self,
        supplier_id: str,
        trigger: str,
        reassessment_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Schedule automated re-assessment for a supplier.

        Args:
            supplier_id: Unique supplier identifier.
            trigger: Trigger reason (alert, threshold_breach, periodic, manual).
            reassessment_date: Scheduled reassessment date. If None, uses
                monitoring frequency to determine next date.

        Returns:
            Dict with scheduling result:
                - supplier_id: Supplier identifier.
                - trigger: Trigger reason.
                - scheduled_date: Scheduled reassessment date.
                - days_until_reassessment: Days until reassessment.
        """
        # Get monitoring config to determine frequency
        config = self._configs.get(supplier_id)
        if config and reassessment_date is None:
            frequency_days = _FREQUENCY_TO_DAYS[config.frequency]
            reassessment_date = utcnow() + timedelta(days=frequency_days)
        elif reassessment_date is None:
            # Default: 30 days
            reassessment_date = utcnow() + timedelta(days=30)

        days_until = (reassessment_date - utcnow()).days

        # Update last assessment time
        with self._lock:
            self._last_assessment_time[supplier_id] = utcnow()

        # Record provenance
        provenance = get_tracker()
        provenance.record(
            entity_type="monitoring",
            entity_id=supplier_id,
            action="schedule_reassessment",
            details={
                "supplier_id": supplier_id,
                "trigger": trigger,
                "scheduled_date": reassessment_date.isoformat(),
                "days_until": days_until,
            },
        )

        logger.info(
            f"Re-assessment scheduled for supplier {supplier_id}: "
            f"trigger={trigger}, date={reassessment_date.date()}"
        )

        return {
            "supplier_id": supplier_id,
            "trigger": trigger,
            "scheduled_date": reassessment_date.isoformat(),
            "days_until_reassessment": days_until,
        }

    def generate_heatmap(
        self,
        supplier_risks: Dict[str, float],
    ) -> Dict[str, Any]:
        """Generate risk heat map for supplier portfolio.

        Args:
            supplier_risks: Dict mapping supplier_id -> risk_score (0-100).

        Returns:
            Dict with heat map data:
                - risk_distribution: Count of suppliers by risk level.
                - high_risk_suppliers: List of high/critical risk supplier IDs.
                - low_risk_suppliers: List of low risk supplier IDs.
                - average_risk: Portfolio average risk score.
                - risk_percentiles: 25th, 50th, 75th, 90th percentiles.
        """
        if not supplier_risks:
            return {
                "risk_distribution": {},
                "high_risk_suppliers": [],
                "low_risk_suppliers": [],
                "average_risk": 0.0,
                "risk_percentiles": {},
            }

        # Calculate risk distribution
        risk_distribution = {
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
        high_risk_suppliers = []
        low_risk_suppliers = []

        cfg = get_config()
        for supplier_id, risk_score in supplier_risks.items():
            if risk_score >= cfg.critical_risk_threshold:
                risk_distribution["critical"] += 1
                high_risk_suppliers.append(supplier_id)
            elif risk_score >= cfg.high_risk_threshold:
                risk_distribution["high"] += 1
                high_risk_suppliers.append(supplier_id)
            elif risk_score >= cfg.medium_risk_threshold:
                risk_distribution["medium"] += 1
            else:
                risk_distribution["low"] += 1
                low_risk_suppliers.append(supplier_id)

        # Calculate average risk
        average_risk = sum(supplier_risks.values()) / len(supplier_risks)

        # Calculate percentiles
        sorted_risks = sorted(supplier_risks.values())
        n = len(sorted_risks)
        risk_percentiles = {
            "p25": sorted_risks[int(n * 0.25)] if n > 0 else 0.0,
            "p50": sorted_risks[int(n * 0.50)] if n > 0 else 0.0,
            "p75": sorted_risks[int(n * 0.75)] if n > 0 else 0.0,
            "p90": sorted_risks[int(n * 0.90)] if n > 0 else 0.0,
        }

        return {
            "risk_distribution": risk_distribution,
            "high_risk_suppliers": high_risk_suppliers,
            "low_risk_suppliers": low_risk_suppliers,
            "average_risk": average_risk,
            "risk_percentiles": risk_percentiles,
        }

    def aggregate_portfolio_risk(
        self,
        supplier_data: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Aggregate portfolio-level risk metrics.

        Args:
            supplier_data: List of supplier dicts with keys:
                - supplier_id, risk_score, country, commodity, cert_status.

        Returns:
            Dict with portfolio risk aggregation:
                - total_suppliers: Total supplier count.
                - average_risk: Portfolio average risk score.
                - country_concentration: Risk concentration by country (HHI).
                - commodity_concentration: Risk concentration by commodity (HHI).
                - uncertified_percent: Percentage of uncertified suppliers.
                - high_risk_count: Count of high/critical risk suppliers.
        """
        if not supplier_data:
            return {
                "total_suppliers": 0,
                "average_risk": 0.0,
                "country_concentration": 0.0,
                "commodity_concentration": 0.0,
                "uncertified_percent": 0.0,
                "high_risk_count": 0,
            }

        total_suppliers = len(supplier_data)
        average_risk = sum(s.get("risk_score", 0.0) for s in supplier_data) / total_suppliers

        # Calculate country concentration (HHI)
        country_counts: Dict[str, int] = defaultdict(int)
        for supplier in supplier_data:
            country = supplier.get("country", "UNKNOWN")
            country_counts[country] += 1

        country_hhi = sum(
            (count / total_suppliers) ** 2 for count in country_counts.values()
        ) * 10000

        # Calculate commodity concentration (HHI)
        commodity_counts: Dict[str, int] = defaultdict(int)
        for supplier in supplier_data:
            commodity = supplier.get("commodity", "UNKNOWN")
            commodity_counts[commodity] += 1

        commodity_hhi = sum(
            (count / total_suppliers) ** 2 for count in commodity_counts.values()
        ) * 10000

        # Calculate uncertified percentage
        uncertified_count = sum(
            1 for s in supplier_data if not s.get("cert_status", False)
        )
        uncertified_percent = uncertified_count / total_suppliers * 100

        # Count high-risk suppliers
        cfg = get_config()
        high_risk_count = sum(
            1 for s in supplier_data
            if s.get("risk_score", 0.0) >= cfg.high_risk_threshold
        )

        return {
            "total_suppliers": total_suppliers,
            "average_risk": average_risk,
            "country_concentration": country_hhi,
            "commodity_concentration": commodity_hhi,
            "uncertified_percent": uncertified_percent,
            "high_risk_count": high_risk_count,
        }

    def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> Optional[SupplierAlert]:
        """Acknowledge an alert.

        Args:
            alert_id: Unique alert identifier.
            acknowledged_by: User/system identifier that acknowledged.

        Returns:
            Updated SupplierAlert if found, else None.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.is_acknowledged = True
                alert.acknowledged_at = utcnow()
                alert.metadata["acknowledged_by"] = acknowledged_by

        if alert:
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")

        return alert

    def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        resolution_notes: Optional[str] = None,
    ) -> Optional[SupplierAlert]:
        """Resolve an alert.

        Args:
            alert_id: Unique alert identifier.
            resolved_by: User/system identifier that resolved.
            resolution_notes: Optional resolution notes.

        Returns:
            Updated SupplierAlert if found, else None.
        """
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert:
                alert.is_resolved = True
                alert.resolved_at = utcnow()
                alert.metadata["resolved_by"] = resolved_by
                if resolution_notes:
                    alert.metadata["resolution_notes"] = resolution_notes

        if alert:
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")

        return alert

    def get_alert_summary(
        self,
        supplier_id: Optional[str] = None,
        severity: Optional[AlertSeverity] = None,
        is_resolved: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Get alert summary with optional filters.

        Args:
            supplier_id: Filter by supplier ID.
            severity: Filter by severity level.
            is_resolved: Filter by resolution status.

        Returns:
            Dict with alert summary:
                - total_alerts: Total alert count.
                - by_severity: Count by severity level.
                - by_type: Count by alert type.
                - unresolved_count: Count of unresolved alerts.
                - alerts: List of alert dicts (filtered).
        """
        with self._lock:
            alerts = list(self._alerts.values())

        # Apply filters
        if supplier_id:
            alerts = [a for a in alerts if a.supplier_id == supplier_id]
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if is_resolved is not None:
            alerts = [a for a in alerts if a.is_resolved == is_resolved]

        # Calculate summary metrics
        total_alerts = len(alerts)
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        unresolved_count = 0

        for alert in alerts:
            by_severity[alert.severity.value] += 1
            by_type[alert.alert_type.value] += 1
            if not alert.is_resolved:
                unresolved_count += 1

        return {
            "total_alerts": total_alerts,
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "unresolved_count": unresolved_count,
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "supplier_id": a.supplier_id,
                    "alert_type": a.alert_type.value,
                    "severity": a.severity.value,
                    "message": a.message,
                    "is_acknowledged": a.is_acknowledged,
                    "is_resolved": a.is_resolved,
                    "created_at": a.created_at.isoformat(),
                }
                for a in alerts
            ],
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _determine_severity(self, risk_score: float) -> AlertSeverity:
        """Determine alert severity based on risk score.

        Args:
            risk_score: Risk score (0-100).

        Returns:
            AlertSeverity enum value.
        """
        for severity, (min_risk, max_risk) in _ALERT_SEVERITY_THRESHOLDS.items():
            if min_risk <= risk_score < max_risk:
                return severity
        return AlertSeverity.CRITICAL  # Default for scores >= 75

    def get_config(self, supplier_id: str) -> Optional[MonitoringConfig]:
        """Retrieve monitoring config by supplier ID.

        Args:
            supplier_id: Unique supplier identifier.

        Returns:
            MonitoringConfig if found, else None.
        """
        with self._lock:
            return self._configs.get(supplier_id)

    def list_configs(
        self,
        is_active: Optional[bool] = None,
    ) -> List[MonitoringConfig]:
        """List monitoring configs with optional filters.

        Args:
            is_active: Filter by active status.

        Returns:
            List of MonitoringConfig objects.
        """
        with self._lock:
            configs = list(self._configs.values())

        if is_active is not None:
            configs = [c for c in configs if c.is_active == is_active]

        return configs
