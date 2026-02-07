# -*- coding: utf-8 -*-
"""
Incident Correlator - SEC-010

Correlates related alerts into incidents using time-window based grouping,
similarity scoring, and deduplication. Reduces alert fatigue by combining
multiple related alerts into a single incident.

Example:
    >>> from greenlang.infrastructure.incident_response.correlator import (
    ...     IncidentCorrelator,
    ... )
    >>> correlator = IncidentCorrelator(config)
    >>> incidents = await correlator.correlate(alerts)

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Alert,
    AlertSource,
    EscalationLevel,
    Incident,
    IncidentStatus,
    IncidentType,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Incident Type Mapping
# ---------------------------------------------------------------------------

ALERT_TYPE_TO_INCIDENT_TYPE: Dict[str, IncidentType] = {
    # Credential-related
    "credential_leak": IncidentType.CREDENTIAL_COMPROMISE,
    "password_spray": IncidentType.CREDENTIAL_COMPROMISE,
    "credential_stuffing": IncidentType.CREDENTIAL_COMPROMISE,
    # DDoS
    "high_traffic": IncidentType.DDOS_ATTACK,
    "rate_limit_exceeded": IncidentType.DDOS_ATTACK,
    "ddos_detected": IncidentType.DDOS_ATTACK,
    # Unauthorized access
    "unauthorized_access": IncidentType.UNAUTHORIZED_ACCESS,
    "forbidden_access": IncidentType.UNAUTHORIZED_ACCESS,
    "access_denied": IncidentType.UNAUTHORIZED_ACCESS,
    # Brute force
    "failed_login": IncidentType.BRUTE_FORCE,
    "brute_force": IncidentType.BRUTE_FORCE,
    "authentication_failure": IncidentType.BRUTE_FORCE,
    # Injection
    "sql_injection": IncidentType.SQL_INJECTION,
    "sqli_detected": IncidentType.SQL_INJECTION,
    # XSS
    "xss_detected": IncidentType.XSS_ATTACK,
    "cross_site_scripting": IncidentType.XSS_ATTACK,
    # Malware
    "malware_detected": IncidentType.MALWARE,
    "trojan_detected": IncidentType.MALWARE,
    "ransomware_detected": IncidentType.RANSOMWARE,
    # Session
    "session_hijack": IncidentType.SESSION_HIJACK,
    "suspicious_session": IncidentType.SESSION_HIJACK,
    # Privilege
    "privilege_escalation": IncidentType.PRIVILEGE_ESCALATION,
    "root_access": IncidentType.PRIVILEGE_ESCALATION,
    # Data
    "data_exfiltration": IncidentType.DATA_EXFILTRATION,
    "data_leak": IncidentType.DATA_EXFILTRATION,
    "data_breach": IncidentType.DATA_BREACH,
    # API
    "api_abuse": IncidentType.API_ABUSE,
    "api_anomaly": IncidentType.API_ABUSE,
    # Insider
    "insider_threat": IncidentType.INSIDER_THREAT,
    "suspicious_user": IncidentType.INSIDER_THREAT,
    # Availability
    "service_down": IncidentType.AVAILABILITY,
    "high_error_rate": IncidentType.AVAILABILITY,
    "latency_spike": IncidentType.AVAILABILITY,
    # Compliance
    "compliance_violation": IncidentType.COMPLIANCE_VIOLATION,
    "policy_violation": IncidentType.COMPLIANCE_VIOLATION,
    # Config
    "config_drift": IncidentType.CONFIGURATION_DRIFT,
    "security_group_change": IncidentType.CONFIGURATION_DRIFT,
}


def _infer_incident_type(alert: Alert) -> IncidentType:
    """Infer incident type from alert.

    Args:
        alert: Alert to analyze.

    Returns:
        Inferred incident type.
    """
    alert_type_lower = alert.alert_type.lower()

    # Check direct mapping
    for pattern, incident_type in ALERT_TYPE_TO_INCIDENT_TYPE.items():
        if pattern in alert_type_lower:
            return incident_type

    # Check message content
    message_lower = alert.message.lower()
    for pattern, incident_type in ALERT_TYPE_TO_INCIDENT_TYPE.items():
        if pattern.replace("_", " ") in message_lower:
            return incident_type

    # GuardDuty-specific mappings
    if alert.source == AlertSource.GUARDDUTY:
        if "recon" in alert_type_lower:
            return IncidentType.UNAUTHORIZED_ACCESS
        if "trojan" in alert_type_lower:
            return IncidentType.MALWARE
        if "backdoor" in alert_type_lower:
            return IncidentType.MALWARE
        if "cryptocurrency" in alert_type_lower:
            return IncidentType.MALWARE
        if "pentest" in alert_type_lower:
            return IncidentType.UNAUTHORIZED_ACCESS

    return IncidentType.UNKNOWN


# ---------------------------------------------------------------------------
# Incident Correlator
# ---------------------------------------------------------------------------


class IncidentCorrelator:
    """Correlates related alerts into incidents.

    Uses time-window based grouping, similarity scoring, and type/source
    matching to combine related alerts into incidents. Helps reduce alert
    fatigue and provides a single view of related issues.

    Attributes:
        config: Incident response configuration.
        incident_counter: Counter for generating incident numbers.

    Example:
        >>> correlator = IncidentCorrelator(config)
        >>> incidents = await correlator.correlate(alerts)
        >>> merged = correlator.merge_incidents(incidents)
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
    ) -> None:
        """Initialize the incident correlator.

        Args:
            config: Incident response configuration.
        """
        self.config = config or get_config()
        self._incident_counter = 0
        self._existing_incidents: Dict[str, Incident] = {}

        logger.info(
            "IncidentCorrelator initialized (correlation_window=%ds)",
            self.config.correlation_window_seconds,
        )

    def _generate_incident_number(self) -> str:
        """Generate a unique incident number.

        Returns:
            Incident number string (e.g., INC-2026-0001).
        """
        self._incident_counter += 1
        year = datetime.now(timezone.utc).year
        return f"{self.config.incident_number_prefix}-{year}-{self._incident_counter:04d}"

    def calculate_similarity(self, a1: Alert, a2: Alert) -> float:
        """Calculate similarity score between two alerts.

        Considers time proximity, source, type, severity, and label overlap
        to determine how related two alerts are.

        Args:
            a1: First alert.
            a2: Second alert.

        Returns:
            Similarity score (0.0-1.0).
        """
        score = 0.0

        # Time proximity (max 0.3)
        time1 = a1.starts_at or a1.received_at
        time2 = a2.starts_at or a2.received_at
        time_diff = abs((time1 - time2).total_seconds())
        window = self.config.correlation_window_seconds

        if time_diff <= window:
            score += 0.3 * (1.0 - (time_diff / window))

        # Same source (0.15)
        if a1.source == a2.source:
            score += 0.15

        # Same alert type (0.25)
        if a1.alert_type.lower() == a2.alert_type.lower():
            score += 0.25
        elif _similar_alert_types(a1.alert_type, a2.alert_type):
            score += 0.15

        # Same severity (0.1)
        if a1.severity == a2.severity:
            score += 0.1

        # Label overlap (max 0.2)
        if a1.labels and a2.labels:
            common_keys = set(a1.labels.keys()) & set(a2.labels.keys())
            if common_keys:
                matching_values = sum(
                    1 for k in common_keys if a1.labels[k] == a2.labels[k]
                )
                overlap_ratio = matching_values / len(common_keys)
                score += 0.2 * overlap_ratio

        return min(1.0, score)

    async def correlate(self, alerts: List[Alert]) -> List[Incident]:
        """Correlate alerts into incidents.

        Groups related alerts based on similarity scoring and time windows,
        then creates incidents from the groups.

        Args:
            alerts: List of alerts to correlate.

        Returns:
            List of correlated incidents.
        """
        if not alerts:
            return []

        logger.info("Correlating %d alerts", len(alerts))

        # Sort alerts by timestamp
        sorted_alerts = sorted(
            alerts,
            key=lambda a: a.starts_at or a.received_at,
        )

        # Group alerts using union-find approach
        groups = self._group_alerts(sorted_alerts)

        # Create incidents from groups
        incidents: List[Incident] = []
        for alert_group in groups:
            incident = self._create_incident_from_alerts(alert_group)
            incidents.append(incident)

        logger.info(
            "Created %d incidents from %d alerts",
            len(incidents),
            len(alerts),
        )

        return incidents

    def _group_alerts(self, alerts: List[Alert]) -> List[List[Alert]]:
        """Group related alerts using similarity scoring.

        Args:
            alerts: Sorted list of alerts.

        Returns:
            List of alert groups.
        """
        if not alerts:
            return []

        # Union-find data structure
        parent: Dict[str, str] = {}
        alert_by_id: Dict[str, Alert] = {}

        def find(x: str) -> str:
            if x not in parent:
                parent[x] = x
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x: str, y: str) -> None:
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        # Build alert lookup
        for alert in alerts:
            alert_id = str(alert.id)
            alert_by_id[alert_id] = alert
            parent[alert_id] = alert_id

        # Find similar alerts and union them
        similarity_threshold = 0.5
        window = timedelta(seconds=self.config.correlation_window_seconds)

        for i, alert1 in enumerate(alerts):
            for j in range(i + 1, len(alerts)):
                alert2 = alerts[j]

                # Skip if too far apart in time
                time1 = alert1.starts_at or alert1.received_at
                time2 = alert2.starts_at or alert2.received_at
                if time2 - time1 > window:
                    break

                # Check similarity
                similarity = self.calculate_similarity(alert1, alert2)
                if similarity >= similarity_threshold:
                    union(str(alert1.id), str(alert2.id))

        # Collect groups
        groups_dict: Dict[str, List[Alert]] = defaultdict(list)
        for alert_id, alert in alert_by_id.items():
            root = find(alert_id)
            groups_dict[root].append(alert)

        return list(groups_dict.values())

    def _create_incident_from_alerts(self, alerts: List[Alert]) -> Incident:
        """Create an incident from a group of correlated alerts.

        Args:
            alerts: Group of related alerts.

        Returns:
            New Incident object.
        """
        if not alerts:
            raise ValueError("Cannot create incident from empty alert list")

        # Sort by severity (highest first) then time (earliest first)
        sorted_alerts = sorted(
            alerts,
            key=lambda a: (
                -list(EscalationLevel).index(a.severity),
                a.starts_at or a.received_at,
            ),
        )

        primary_alert = sorted_alerts[0]

        # Determine overall severity (highest among alerts)
        severity = max(alerts, key=lambda a: list(EscalationLevel).index(a.severity)).severity

        # Determine incident type
        incident_type = _infer_incident_type(primary_alert)

        # Generate title
        if len(alerts) == 1:
            title = primary_alert.message[:200]
        else:
            title = f"{incident_type.value.replace('_', ' ').title()}: {len(alerts)} related alerts"

        # Generate description
        description_parts = [
            f"Correlated incident from {len(alerts)} alerts.",
            f"Primary alert: {primary_alert.message}",
            f"Sources: {', '.join(set(a.source.value for a in alerts))}",
        ]
        description = "\n".join(description_parts)

        # Collect affected systems from labels
        affected_systems: Set[str] = set()
        for alert in alerts:
            if "instance" in alert.labels:
                affected_systems.add(alert.labels["instance"])
            if "service" in alert.labels:
                affected_systems.add(alert.labels["service"])
            if "host" in alert.labels:
                affected_systems.add(alert.labels["host"])

        # Earliest detection time
        detected_at = min(
            a.starts_at or a.received_at for a in alerts
        )

        # Create incident
        incident = Incident(
            id=uuid4(),
            incident_number=self._generate_incident_number(),
            title=title,
            description=description,
            severity=severity,
            status=IncidentStatus.DETECTED,
            incident_type=incident_type,
            source=primary_alert.source,
            detected_at=detected_at,
            related_alerts=[a.id for a in alerts],
            affected_systems=list(affected_systems),
            tags=list(set(primary_alert.labels.get("alertname", "").split(","))),
            metadata={
                "alert_count": len(alerts),
                "correlation_method": "similarity",
            },
        )

        # Update alert references
        for alert in alerts:
            alert.incident_id = incident.id

        # Calculate provenance hash
        incident.provenance_hash = incident.calculate_provenance_hash()

        return incident

    def merge_incidents(self, incidents: List[Incident]) -> List[Incident]:
        """Merge duplicate or highly similar incidents.

        Args:
            incidents: List of incidents to potentially merge.

        Returns:
            Deduplicated list of incidents.
        """
        if len(incidents) <= 1:
            return incidents

        logger.debug("Checking %d incidents for merging", len(incidents))

        # Group by incident type and severity
        groups: Dict[Tuple[IncidentType, EscalationLevel], List[Incident]] = defaultdict(list)
        for incident in incidents:
            key = (incident.incident_type, incident.severity)
            groups[key].append(incident)

        merged: List[Incident] = []
        window = timedelta(seconds=self.config.deduplication_window_seconds)

        for (inc_type, severity), group in groups.items():
            if len(group) == 1:
                merged.extend(group)
                continue

            # Sort by detection time
            sorted_group = sorted(group, key=lambda i: i.detected_at)

            # Merge incidents within time window
            current_batch: List[Incident] = [sorted_group[0]]

            for incident in sorted_group[1:]:
                last_in_batch = current_batch[-1]
                if incident.detected_at - last_in_batch.detected_at <= window:
                    current_batch.append(incident)
                else:
                    # Process current batch
                    merged.append(self._merge_incident_batch(current_batch))
                    current_batch = [incident]

            # Process final batch
            merged.append(self._merge_incident_batch(current_batch))

        logger.info("Merged %d incidents into %d", len(incidents), len(merged))

        return merged

    def _merge_incident_batch(self, incidents: List[Incident]) -> Incident:
        """Merge a batch of similar incidents into one.

        Args:
            incidents: Batch of incidents to merge.

        Returns:
            Merged incident.
        """
        if len(incidents) == 1:
            return incidents[0]

        # Use earliest incident as base
        base = min(incidents, key=lambda i: i.detected_at)

        # Aggregate data from all incidents
        all_alerts: List[Any] = []
        all_systems: Set[str] = set()
        all_tags: Set[str] = set()

        for incident in incidents:
            all_alerts.extend(incident.related_alerts)
            all_systems.update(incident.affected_systems)
            all_tags.update(incident.tags)

        # Update base incident
        base.related_alerts = list(set(all_alerts))
        base.affected_systems = list(all_systems)
        base.tags = list(all_tags)
        base.title = f"{base.incident_type.value.replace('_', ' ').title()}: {len(all_alerts)} alerts ({len(incidents)} merged)"
        base.metadata["merged_incidents"] = len(incidents)
        base.metadata["alert_count"] = len(all_alerts)

        # Recalculate provenance hash
        base.provenance_hash = base.calculate_provenance_hash()

        return base

    def add_existing_incident(self, incident: Incident) -> None:
        """Register an existing incident for correlation.

        Args:
            incident: Existing incident to consider.
        """
        key = self._incident_key(incident)
        self._existing_incidents[key] = incident

    def _incident_key(self, incident: Incident) -> str:
        """Generate a key for incident lookup.

        Args:
            incident: Incident to generate key for.

        Returns:
            Key string.
        """
        return f"{incident.incident_type.value}:{incident.severity.value}"

    def find_matching_incident(self, alert: Alert) -> Optional[Incident]:
        """Find an existing incident that matches an alert.

        Args:
            alert: Alert to match.

        Returns:
            Matching incident or None.
        """
        incident_type = _infer_incident_type(alert)
        key = f"{incident_type.value}:{alert.severity.value}"

        if key not in self._existing_incidents:
            return None

        incident = self._existing_incidents[key]

        # Check if within deduplication window
        window = timedelta(seconds=self.config.deduplication_window_seconds)
        alert_time = alert.starts_at or alert.received_at

        if alert_time - incident.detected_at <= window:
            return incident

        return None


def _similar_alert_types(type1: str, type2: str) -> bool:
    """Check if two alert types are similar.

    Args:
        type1: First alert type.
        type2: Second alert type.

    Returns:
        True if types are similar.
    """
    # Normalize types
    t1 = type1.lower().replace("_", "").replace("-", "")
    t2 = type2.lower().replace("_", "").replace("-", "")

    # Check for common prefix
    min_len = min(len(t1), len(t2))
    if min_len >= 5:
        if t1[:5] == t2[:5]:
            return True

    # Check for substring relationship
    if t1 in t2 or t2 in t1:
        return True

    return False


# ---------------------------------------------------------------------------
# Global Correlator Instance
# ---------------------------------------------------------------------------

_global_correlator: Optional[IncidentCorrelator] = None


def get_correlator(
    config: Optional[IncidentResponseConfig] = None,
) -> IncidentCorrelator:
    """Get or create the global incident correlator.

    Args:
        config: Optional configuration override.

    Returns:
        The global IncidentCorrelator instance.
    """
    global _global_correlator

    if _global_correlator is None:
        _global_correlator = IncidentCorrelator(config)

    return _global_correlator


def reset_correlator() -> None:
    """Reset the global correlator."""
    global _global_correlator
    _global_correlator = None


__all__ = [
    "IncidentCorrelator",
    "get_correlator",
    "reset_correlator",
    "ALERT_TYPE_TO_INCIDENT_TYPE",
]
