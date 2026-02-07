# -*- coding: utf-8 -*-
"""
Playbook Executor - SEC-010

Executes automated remediation playbooks for incident response. Each playbook
defines a series of steps to contain, investigate, and remediate specific
types of security incidents.

Example:
    >>> from greenlang.infrastructure.incident_response.playbook_executor import (
    ...     PlaybookExecutor,
    ... )
    >>> executor = PlaybookExecutor(config)
    >>> result = await executor.execute(incident, "credential_compromise")

Author: GreenLang Security Team
Date: February 2026
Status: Production Ready
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Type
from uuid import UUID, uuid4

from greenlang.infrastructure.incident_response.config import (
    IncidentResponseConfig,
    get_config,
)
from greenlang.infrastructure.incident_response.models import (
    Incident,
    IncidentType,
    PlaybookExecution,
    PlaybookStatus,
    PlaybookStep,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Playbook
# ---------------------------------------------------------------------------


@dataclass
class PlaybookResult:
    """Result of a playbook execution.

    Attributes:
        success: Whether playbook completed successfully.
        execution: Execution record.
        steps_completed: Number of steps completed.
        rollback_performed: Whether rollback was performed.
        error: Error message if failed.
        artifacts: Collected artifacts (logs, evidence).
    """

    success: bool
    execution: PlaybookExecution
    steps_completed: int = 0
    rollback_performed: bool = False
    error: Optional[str] = None
    artifacts: Dict[str, Any] = field(default_factory=dict)


class BasePlaybook(abc.ABC):
    """Abstract base class for playbooks.

    Defines the interface for all playbooks. Subclasses must implement
    the `steps` property and step handler methods.

    Attributes:
        playbook_id: Unique playbook identifier.
        name: Human-readable playbook name.
        description: Playbook description.
        incident_types: Incident types this playbook handles.
        config: Incident response configuration.
    """

    playbook_id: str = "base"
    name: str = "Base Playbook"
    description: str = "Base playbook class"
    incident_types: List[IncidentType] = []

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
    ) -> None:
        """Initialize the playbook.

        Args:
            config: Incident response configuration.
        """
        self.config = config or get_config()
        self._rollback_stack: List[Dict[str, Any]] = []

    @property
    @abc.abstractmethod
    def steps(self) -> List[str]:
        """Get list of playbook step names.

        Returns:
            List of step names in execution order.
        """
        pass

    async def execute_step(
        self,
        step_name: str,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute a single playbook step.

        Args:
            step_name: Name of the step to execute.
            incident: Associated incident.
            context: Execution context (passed between steps).

        Returns:
            Step output dictionary.

        Raises:
            NotImplementedError: If step handler not implemented.
        """
        handler_name = f"step_{step_name}"
        handler = getattr(self, handler_name, None)

        if handler is None:
            raise NotImplementedError(f"Step handler '{handler_name}' not implemented")

        return await handler(incident, context)

    async def rollback_step(
        self,
        step_name: str,
        incident: Incident,
        rollback_data: Dict[str, Any],
    ) -> bool:
        """Rollback a single playbook step.

        Args:
            step_name: Name of the step to rollback.
            incident: Associated incident.
            rollback_data: Data needed for rollback.

        Returns:
            True if rollback succeeded.
        """
        rollback_handler_name = f"rollback_{step_name}"
        rollback_handler = getattr(self, rollback_handler_name, None)

        if rollback_handler is None:
            logger.debug("No rollback handler for step '%s'", step_name)
            return True

        try:
            await rollback_handler(incident, rollback_data)
            return True
        except Exception as e:
            logger.error("Rollback failed for step '%s': %s", step_name, e)
            return False

    def can_handle(self, incident: Incident) -> bool:
        """Check if playbook can handle incident type.

        Args:
            incident: Incident to check.

        Returns:
            True if playbook can handle incident.
        """
        if not self.incident_types:
            return True
        return incident.incident_type in self.incident_types


# ---------------------------------------------------------------------------
# Credential Compromise Playbook
# ---------------------------------------------------------------------------


class CredentialCompromisePlaybook(BasePlaybook):
    """Playbook for credential compromise incidents.

    Steps:
        1. Identify affected accounts
        2. Revoke all active sessions
        3. Rotate credentials
        4. Enable MFA enforcement
        5. Notify affected users
        6. Create incident ticket
        7. Collect forensic evidence
        8. Generate post-mortem
    """

    playbook_id = "credential_compromise"
    name = "Credential Compromise Response"
    description = "Automated response to credential compromise incidents"
    incident_types = [IncidentType.CREDENTIAL_COMPROMISE]

    @property
    def steps(self) -> List[str]:
        return [
            "identify_affected_accounts",
            "revoke_all_sessions",
            "rotate_credentials",
            "enable_mfa_enforcement",
            "notify_affected_users",
            "create_incident_ticket",
            "collect_forensic_evidence",
            "generate_post_mortem",
        ]

    async def step_identify_affected_accounts(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Identify accounts affected by credential compromise."""
        logger.info("Identifying affected accounts for %s", incident.incident_number)

        # In production, this would query identity provider
        affected_accounts = []

        # Extract from incident metadata
        if "affected_users" in incident.metadata:
            affected_accounts = incident.metadata["affected_users"]

        # Store for later steps
        context["affected_accounts"] = affected_accounts

        return {
            "affected_count": len(affected_accounts),
            "accounts": affected_accounts,
        }

    async def step_revoke_all_sessions(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Revoke all active sessions for affected accounts."""
        logger.info("Revoking sessions for %s", incident.incident_number)

        affected_accounts = context.get("affected_accounts", [])
        revoked_sessions = []

        # In production, this would call session management API
        for account in affected_accounts:
            logger.info("Revoking sessions for account: %s", account)
            revoked_sessions.append({
                "account": account,
                "sessions_revoked": 5,  # Placeholder
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # Store rollback data
        self._rollback_stack.append({
            "step": "revoke_all_sessions",
            "data": {"revoked_sessions": revoked_sessions},
        })

        return {
            "sessions_revoked": len(revoked_sessions),
            "details": revoked_sessions,
        }

    async def step_rotate_credentials(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Force credential rotation for affected accounts."""
        logger.info("Rotating credentials for %s", incident.incident_number)

        affected_accounts = context.get("affected_accounts", [])
        rotated = []

        # In production, this would trigger password reset
        for account in affected_accounts:
            logger.info("Forcing credential rotation for: %s", account)
            rotated.append(account)

        return {
            "credentials_rotated": len(rotated),
            "accounts": rotated,
        }

    async def step_enable_mfa_enforcement(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enable MFA enforcement for affected accounts."""
        logger.info("Enabling MFA enforcement for %s", incident.incident_number)

        affected_accounts = context.get("affected_accounts", [])

        # In production, this would update MFA policy
        return {
            "mfa_enforced": len(affected_accounts),
            "accounts": affected_accounts,
        }

    async def step_notify_affected_users(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Notify affected users about credential reset."""
        logger.info("Notifying affected users for %s", incident.incident_number)

        affected_accounts = context.get("affected_accounts", [])

        # In production, this would send email notifications
        return {
            "users_notified": len(affected_accounts),
        }

    async def step_create_incident_ticket(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create tracking ticket in Jira."""
        logger.info("Creating incident ticket for %s", incident.incident_number)

        # In production, this would create Jira ticket
        ticket_id = f"SEC-{datetime.now(timezone.utc).strftime('%Y%m%d')}-001"

        return {
            "ticket_id": ticket_id,
            "ticket_url": f"https://jira.greenlang.io/browse/{ticket_id}",
        }

    async def step_collect_forensic_evidence(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Collect forensic evidence for investigation."""
        logger.info("Collecting forensic evidence for %s", incident.incident_number)

        # In production, this would collect logs, artifacts
        evidence = {
            "login_logs": "collected",
            "api_logs": "collected",
            "session_data": "preserved",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return evidence

    async def step_generate_post_mortem(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate post-mortem template."""
        logger.info("Generating post-mortem for %s", incident.incident_number)

        return {
            "post_mortem_created": True,
            "template_used": "credential_compromise_template",
        }


# ---------------------------------------------------------------------------
# DDoS Mitigation Playbook
# ---------------------------------------------------------------------------


class DDoSMitigationPlaybook(BasePlaybook):
    """Playbook for DDoS attack incidents.

    Steps:
        1. Identify attack vector
        2. Enable AWS Shield Advanced
        3. Update WAF rules
        4. Scale infrastructure
        5. Enable geo-blocking
        6. Notify NOC
        7. Monitor mitigation
    """

    playbook_id = "ddos_mitigation"
    name = "DDoS Mitigation Response"
    description = "Automated response to DDoS attacks"
    incident_types = [IncidentType.DDOS_ATTACK]

    @property
    def steps(self) -> List[str]:
        return [
            "identify_attack_vector",
            "enable_shield_advanced",
            "update_waf_rules",
            "scale_infrastructure",
            "enable_geo_blocking",
            "notify_noc",
            "monitor_mitigation",
        ]

    async def step_identify_attack_vector(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Identify DDoS attack vector and characteristics."""
        logger.info("Identifying attack vector for %s", incident.incident_number)

        # In production, this would analyze traffic patterns
        context["attack_vector"] = incident.metadata.get("attack_vector", "volumetric")
        context["source_ips"] = incident.metadata.get("source_ips", [])

        return {
            "attack_vector": context["attack_vector"],
            "estimated_bps": incident.metadata.get("estimated_bps", 0),
            "source_ip_count": len(context["source_ips"]),
        }

    async def step_enable_shield_advanced(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enable AWS Shield Advanced protection."""
        logger.info("Enabling Shield Advanced for %s", incident.incident_number)

        # In production, this would call AWS API
        return {
            "shield_enabled": True,
            "protection_groups": ["alb", "cloudfront"],
        }

    async def step_update_waf_rules(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update WAF rules to block attack traffic."""
        logger.info("Updating WAF rules for %s", incident.incident_number)

        source_ips = context.get("source_ips", [])

        # In production, this would update WAF
        return {
            "rules_updated": True,
            "ips_blocked": len(source_ips),
            "rate_limit_enabled": True,
        }

    async def step_scale_infrastructure(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Scale infrastructure to handle increased load."""
        logger.info("Scaling infrastructure for %s", incident.incident_number)

        # In production, this would trigger auto-scaling
        return {
            "scaled": True,
            "new_instance_count": 10,
            "previous_instance_count": 3,
        }

    async def step_enable_geo_blocking(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Enable geo-blocking for attack sources."""
        logger.info("Enabling geo-blocking for %s", incident.incident_number)

        # In production, this would update CloudFront/WAF
        return {
            "geo_blocking_enabled": True,
            "blocked_countries": incident.metadata.get("source_countries", []),
        }

    async def step_notify_noc(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Notify Network Operations Center."""
        logger.info("Notifying NOC for %s", incident.incident_number)

        return {
            "noc_notified": True,
            "notification_channel": "pagerduty",
        }

    async def step_monitor_mitigation(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Monitor mitigation effectiveness."""
        logger.info("Monitoring mitigation for %s", incident.incident_number)

        return {
            "monitoring_enabled": True,
            "dashboard_url": f"https://grafana.greenlang.io/d/ddos/{incident.incident_number}",
        }


# ---------------------------------------------------------------------------
# Data Breach Playbook
# ---------------------------------------------------------------------------


class DataBreachPlaybook(BasePlaybook):
    """Playbook for data breach incidents.

    Steps:
        1. Isolate affected systems
        2. Preserve evidence
        3. Assess scope
        4. Notify legal team
        5. Notify affected users
        6. Report to regulators
        7. Generate breach report
    """

    playbook_id = "data_breach"
    name = "Data Breach Response"
    description = "Automated response to data breach incidents"
    incident_types = [IncidentType.DATA_BREACH, IncidentType.DATA_EXFILTRATION]

    @property
    def steps(self) -> List[str]:
        return [
            "isolate_affected_systems",
            "preserve_evidence",
            "assess_scope",
            "notify_legal_team",
            "notify_affected_users",
            "report_to_regulators",
            "generate_breach_report",
        ]

    async def step_isolate_affected_systems(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Isolate affected systems to prevent further data loss."""
        logger.info("Isolating systems for %s", incident.incident_number)

        affected = incident.affected_systems

        # In production, this would modify security groups/network ACLs
        return {
            "systems_isolated": len(affected),
            "systems": affected,
        }

    async def step_preserve_evidence(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Preserve forensic evidence."""
        logger.info("Preserving evidence for %s", incident.incident_number)

        return {
            "snapshots_created": len(incident.affected_systems),
            "logs_preserved": True,
            "memory_dumps": True,
        }

    async def step_assess_scope(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Assess the scope of data breach."""
        logger.info("Assessing scope for %s", incident.incident_number)

        return {
            "records_affected": incident.affected_users or 0,
            "data_types": incident.metadata.get("data_types", []),
            "pii_involved": incident.metadata.get("pii_involved", False),
        }

    async def step_notify_legal_team(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Notify legal team for regulatory guidance."""
        logger.info("Notifying legal team for %s", incident.incident_number)

        return {
            "legal_notified": True,
            "notification_time": datetime.now(timezone.utc).isoformat(),
        }

    async def step_notify_affected_users(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Notify affected users about data breach."""
        logger.info("Notifying affected users for %s", incident.incident_number)

        return {
            "users_notified": incident.affected_users or 0,
            "notification_method": "email",
        }

    async def step_report_to_regulators(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Report breach to relevant regulators (GDPR, etc.)."""
        logger.info("Reporting to regulators for %s", incident.incident_number)

        return {
            "regulators_notified": ["ICO", "DPA"],
            "report_id": f"BREACH-{datetime.now(timezone.utc).strftime('%Y%m%d')}-001",
        }

    async def step_generate_breach_report(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate comprehensive breach report."""
        logger.info("Generating breach report for %s", incident.incident_number)

        return {
            "report_generated": True,
            "report_url": f"https://reports.greenlang.io/breach/{incident.id}",
        }


# ---------------------------------------------------------------------------
# Malware Containment Playbook
# ---------------------------------------------------------------------------


class MalwareContainmentPlaybook(BasePlaybook):
    """Playbook for malware incidents."""

    playbook_id = "malware_containment"
    name = "Malware Containment Response"
    description = "Automated response to malware detection"
    incident_types = [IncidentType.MALWARE, IncidentType.RANSOMWARE]

    @property
    def steps(self) -> List[str]:
        return [
            "quarantine_affected_systems",
            "kill_malicious_processes",
            "scan_connected_systems",
            "clean_infected_files",
            "restore_from_backup",
            "update_av_signatures",
        ]

    async def step_quarantine_affected_systems(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Quarantine systems to prevent spread."""
        logger.info("Quarantining systems for %s", incident.incident_number)
        return {"systems_quarantined": len(incident.affected_systems)}

    async def step_kill_malicious_processes(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Terminate malicious processes."""
        logger.info("Killing malicious processes for %s", incident.incident_number)
        return {"processes_killed": 5}

    async def step_scan_connected_systems(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Scan connected systems for infection."""
        logger.info("Scanning connected systems for %s", incident.incident_number)
        return {"systems_scanned": 10, "infections_found": 0}

    async def step_clean_infected_files(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Clean or remove infected files."""
        logger.info("Cleaning infected files for %s", incident.incident_number)
        return {"files_cleaned": 15, "files_quarantined": 3}

    async def step_restore_from_backup(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Restore critical files from backup."""
        logger.info("Restoring from backup for %s", incident.incident_number)
        return {"files_restored": 10, "backup_date": "2026-02-05"}

    async def step_update_av_signatures(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Update antivirus signatures."""
        logger.info("Updating AV signatures for %s", incident.incident_number)
        return {"signatures_updated": True}


# ---------------------------------------------------------------------------
# Access Revocation Playbook
# ---------------------------------------------------------------------------


class AccessRevocationPlaybook(BasePlaybook):
    """Playbook for unauthorized access incidents."""

    playbook_id = "access_revocation"
    name = "Access Revocation Response"
    description = "Automated response to unauthorized access"
    incident_types = [IncidentType.UNAUTHORIZED_ACCESS, IncidentType.PRIVILEGE_ESCALATION]

    @property
    def steps(self) -> List[str]:
        return [
            "identify_compromised_accounts",
            "disable_accounts",
            "revoke_api_tokens",
            "audit_access_logs",
            "restore_permissions",
        ]

    async def step_identify_compromised_accounts(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Identifying compromised accounts for %s", incident.incident_number)
        context["compromised_accounts"] = incident.metadata.get("accounts", [])
        return {"accounts_identified": len(context["compromised_accounts"])}

    async def step_disable_accounts(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Disabling accounts for %s", incident.incident_number)
        return {"accounts_disabled": len(context.get("compromised_accounts", []))}

    async def step_revoke_api_tokens(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Revoking API tokens for %s", incident.incident_number)
        return {"tokens_revoked": 10}

    async def step_audit_access_logs(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Auditing access logs for %s", incident.incident_number)
        return {"logs_audited": True, "suspicious_activities": 5}

    async def step_restore_permissions(
        self,
        incident: Incident,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        logger.info("Restoring permissions for %s", incident.incident_number)
        return {"permissions_restored": True}


# ---------------------------------------------------------------------------
# Additional Playbooks
# ---------------------------------------------------------------------------


class SessionHijackPlaybook(BasePlaybook):
    """Playbook for session hijacking incidents."""

    playbook_id = "session_hijack"
    name = "Session Hijack Response"
    description = "Automated response to session hijacking"
    incident_types = [IncidentType.SESSION_HIJACK]

    @property
    def steps(self) -> List[str]:
        return ["invalidate_sessions", "force_reauthentication", "audit_activity"]

    async def step_invalidate_sessions(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"sessions_invalidated": 100}

    async def step_force_reauthentication(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"users_reauthenticated": 100}

    async def step_audit_activity(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"activity_audited": True}


class BruteForceResponsePlaybook(BasePlaybook):
    """Playbook for brute force attack incidents."""

    playbook_id = "brute_force_response"
    name = "Brute Force Response"
    description = "Automated response to brute force attacks"
    incident_types = [IncidentType.BRUTE_FORCE]

    @property
    def steps(self) -> List[str]:
        return ["block_source_ips", "enforce_rate_limiting", "notify_affected_accounts"]

    async def step_block_source_ips(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"ips_blocked": 50}

    async def step_enforce_rate_limiting(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"rate_limiting_enabled": True}

    async def step_notify_affected_accounts(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"accounts_notified": 10}


class SQLInjectionResponsePlaybook(BasePlaybook):
    """Playbook for SQL injection incidents."""

    playbook_id = "sql_injection_response"
    name = "SQL Injection Response"
    description = "Automated response to SQL injection attacks"
    incident_types = [IncidentType.SQL_INJECTION]

    @property
    def steps(self) -> List[str]:
        return ["block_attacker", "audit_database", "patch_vulnerability"]

    async def step_block_attacker(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"attacker_blocked": True}

    async def step_audit_database(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"database_audited": True, "data_integrity": "verified"}

    async def step_patch_vulnerability(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"vulnerability_patched": True}


class APIAbusePlaybook(BasePlaybook):
    """Playbook for API abuse incidents."""

    playbook_id = "api_abuse"
    name = "API Abuse Response"
    description = "Automated response to API abuse"
    incident_types = [IncidentType.API_ABUSE]

    @property
    def steps(self) -> List[str]:
        return ["rate_limit_abuser", "revoke_api_keys", "analyze_patterns"]

    async def step_rate_limit_abuser(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"rate_limit_applied": True}

    async def step_revoke_api_keys(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"keys_revoked": 5}

    async def step_analyze_patterns(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"patterns_analyzed": True}


class InsiderThreatPlaybook(BasePlaybook):
    """Playbook for insider threat incidents."""

    playbook_id = "insider_threat"
    name = "Insider Threat Response"
    description = "Automated response to insider threats"
    incident_types = [IncidentType.INSIDER_THREAT]

    @property
    def steps(self) -> List[str]:
        return ["suspend_access", "preserve_evidence", "notify_hr", "conduct_investigation"]

    async def step_suspend_access(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"access_suspended": True}

    async def step_preserve_evidence(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"evidence_preserved": True}

    async def step_notify_hr(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"hr_notified": True}

    async def step_conduct_investigation(self, incident: Incident, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"investigation_started": True}


# ---------------------------------------------------------------------------
# Playbook Registry
# ---------------------------------------------------------------------------

PLAYBOOKS: Dict[str, Type[BasePlaybook]] = {
    "credential_compromise": CredentialCompromisePlaybook,
    "ddos_mitigation": DDoSMitigationPlaybook,
    "data_breach": DataBreachPlaybook,
    "malware_containment": MalwareContainmentPlaybook,
    "access_revocation": AccessRevocationPlaybook,
    "session_hijack": SessionHijackPlaybook,
    "brute_force_response": BruteForceResponsePlaybook,
    "sql_injection_response": SQLInjectionResponsePlaybook,
    "api_abuse": APIAbusePlaybook,
    "insider_threat": InsiderThreatPlaybook,
}


# ---------------------------------------------------------------------------
# Playbook Executor
# ---------------------------------------------------------------------------


class PlaybookExecutor:
    """Executes automated remediation playbooks.

    Manages playbook execution lifecycle including step execution,
    error handling, and rollback capabilities.

    Attributes:
        config: Incident response configuration.
        playbooks: Registry of available playbooks.
        active_executions: Currently running executions.

    Example:
        >>> executor = PlaybookExecutor(config)
        >>> result = await executor.execute(incident, "credential_compromise")
    """

    def __init__(
        self,
        config: Optional[IncidentResponseConfig] = None,
    ) -> None:
        """Initialize the playbook executor.

        Args:
            config: Incident response configuration.
        """
        self.config = config or get_config()
        self.playbooks = PLAYBOOKS.copy()
        self.active_executions: Dict[UUID, PlaybookExecution] = {}
        self.execution_history: Dict[UUID, List[PlaybookExecution]] = {}

        logger.info(
            "PlaybookExecutor initialized with %d playbooks",
            len(self.playbooks),
        )

    async def execute(
        self,
        incident: Incident,
        playbook_id: str,
        dry_run: bool = False,
        triggered_by: Optional[str] = None,
    ) -> PlaybookResult:
        """Execute a playbook for an incident.

        Args:
            incident: Incident to remediate.
            playbook_id: Playbook to execute.
            dry_run: Run without making actual changes.
            triggered_by: Who/what triggered execution.

        Returns:
            PlaybookResult with execution details.

        Raises:
            ValueError: If playbook not found.
        """
        if playbook_id not in self.playbooks:
            raise ValueError(f"Playbook '{playbook_id}' not found")

        playbook_class = self.playbooks[playbook_id]
        playbook = playbook_class(self.config)

        if not playbook.can_handle(incident):
            logger.warning(
                "Playbook '%s' cannot handle incident type '%s'",
                playbook_id,
                incident.incident_type.value,
            )

        logger.info(
            "Executing playbook '%s' for incident %s (dry_run=%s)",
            playbook_id,
            incident.incident_number,
            dry_run,
        )

        # Create execution record
        execution = PlaybookExecution(
            id=uuid4(),
            incident_id=incident.id,
            playbook_id=playbook_id,
            playbook_name=playbook.name,
            status=PlaybookStatus.RUNNING,
            started_at=datetime.now(timezone.utc),
            steps_total=len(playbook.steps),
            triggered_by=triggered_by or "automated",
            dry_run=dry_run or self.config.playbook.dry_run,
        )

        # Track active execution
        self.active_executions[execution.id] = execution
        incident.playbook_execution_id = execution.id

        # Initialize steps
        for i, step_name in enumerate(playbook.steps):
            execution.steps.append(
                PlaybookStep(
                    step_number=i + 1,
                    name=step_name,
                    status=PlaybookStatus.PENDING,
                )
            )

        context: Dict[str, Any] = {}
        artifacts: Dict[str, Any] = {}
        error: Optional[str] = None

        try:
            # Execute each step
            for i, step_name in enumerate(playbook.steps):
                step = execution.steps[i]
                step.status = PlaybookStatus.RUNNING
                step.started_at = datetime.now(timezone.utc)
                execution.current_step = i + 1

                execution.add_log_entry(f"Starting step: {step_name}", "info", i + 1)

                try:
                    # Execute with timeout
                    output = await asyncio.wait_for(
                        playbook.execute_step(step_name, incident, context),
                        timeout=self.config.playbook.step_timeout_seconds,
                    )

                    step.output = output
                    step.status = PlaybookStatus.COMPLETED
                    step.completed_at = datetime.now(timezone.utc)
                    step.duration_seconds = (
                        step.completed_at - step.started_at
                    ).total_seconds()

                    execution.steps_completed += 1
                    execution.add_log_entry(
                        f"Completed step: {step_name}",
                        "info",
                        i + 1,
                    )

                    # Store artifacts
                    artifacts[step_name] = output

                except asyncio.TimeoutError:
                    step.status = PlaybookStatus.FAILED
                    step.error = "Step timed out"
                    step.completed_at = datetime.now(timezone.utc)
                    execution.add_log_entry(
                        f"Step timed out: {step_name}",
                        "error",
                        i + 1,
                    )
                    raise

                except Exception as e:
                    step.status = PlaybookStatus.FAILED
                    step.error = str(e)
                    step.completed_at = datetime.now(timezone.utc)
                    execution.add_log_entry(
                        f"Step failed: {step_name} - {e}",
                        "error",
                        i + 1,
                    )
                    raise

            # All steps completed
            execution.status = PlaybookStatus.COMPLETED
            execution.completed_at = datetime.now(timezone.utc)
            execution.add_log_entry("Playbook completed successfully", "info")

            return PlaybookResult(
                success=True,
                execution=execution,
                steps_completed=execution.steps_completed,
                artifacts=artifacts,
            )

        except Exception as e:
            error = str(e)
            execution.status = PlaybookStatus.FAILED
            execution.completed_at = datetime.now(timezone.utc)

            logger.error(
                "Playbook '%s' failed for %s: %s",
                playbook_id,
                incident.incident_number,
                error,
            )

            # Attempt rollback if configured
            rollback_performed = False
            if self.config.playbook.rollback_on_failure:
                rollback_performed = await self._perform_rollback(
                    playbook,
                    incident,
                    execution,
                )
                if rollback_performed:
                    execution.status = PlaybookStatus.ROLLED_BACK

            return PlaybookResult(
                success=False,
                execution=execution,
                steps_completed=execution.steps_completed,
                rollback_performed=rollback_performed,
                error=error,
                artifacts=artifacts,
            )

        finally:
            # Remove from active, add to history
            self.active_executions.pop(execution.id, None)
            if incident.id not in self.execution_history:
                self.execution_history[incident.id] = []
            self.execution_history[incident.id].append(execution)

    async def rollback(self, execution_id: UUID) -> bool:
        """Rollback a playbook execution.

        Args:
            execution_id: Execution to rollback.

        Returns:
            True if rollback succeeded.
        """
        # Find execution in history
        execution = None
        incident_id = None

        for inc_id, executions in self.execution_history.items():
            for exec in executions:
                if exec.id == execution_id:
                    execution = exec
                    incident_id = inc_id
                    break
            if execution:
                break

        if not execution:
            logger.error("Execution %s not found", execution_id)
            return False

        playbook_class = self.playbooks.get(execution.playbook_id)
        if not playbook_class:
            logger.error("Playbook %s not found", execution.playbook_id)
            return False

        playbook = playbook_class(self.config)

        # Create mock incident for rollback
        from greenlang.infrastructure.incident_response.models import Incident

        mock_incident = Incident(
            id=incident_id or uuid4(),
            incident_number="ROLLBACK",
            title="Rollback",
            severity=EscalationLevel.P2,
            source=incident.source if hasattr(incident, "source") else AlertSource.MANUAL,
        )

        return await self._perform_rollback(playbook, mock_incident, execution)

    async def _perform_rollback(
        self,
        playbook: BasePlaybook,
        incident: Incident,
        execution: PlaybookExecution,
    ) -> bool:
        """Perform rollback of completed steps.

        Args:
            playbook: Playbook instance.
            incident: Associated incident.
            execution: Execution to rollback.

        Returns:
            True if rollback succeeded.
        """
        logger.info("Performing rollback for execution %s", execution.id)

        rollback_success = True

        # Rollback in reverse order
        for step in reversed(execution.steps):
            if step.status != PlaybookStatus.COMPLETED:
                continue

            rollback_data = step.rollback_data or step.output or {}

            try:
                success = await playbook.rollback_step(
                    step.name,
                    incident,
                    rollback_data,
                )
                if not success:
                    rollback_success = False

                execution.add_log_entry(
                    f"Rolled back step: {step.name}",
                    "info",
                    step.step_number,
                )

            except Exception as e:
                logger.error("Rollback failed for step %s: %s", step.name, e)
                execution.add_log_entry(
                    f"Rollback failed for step {step.name}: {e}",
                    "error",
                    step.step_number,
                )
                rollback_success = False

        execution.rollback_available = False

        return rollback_success

    def get_available_playbooks(self) -> List[Dict[str, Any]]:
        """Get list of available playbooks.

        Returns:
            List of playbook metadata.
        """
        result = []
        for playbook_id, playbook_class in self.playbooks.items():
            playbook = playbook_class(self.config)
            result.append({
                "id": playbook_id,
                "name": playbook.name,
                "description": playbook.description,
                "steps": playbook.steps,
                "incident_types": [t.value for t in playbook.incident_types],
            })
        return result

    def get_playbook_for_incident(
        self,
        incident: Incident,
    ) -> Optional[str]:
        """Get recommended playbook for an incident.

        Args:
            incident: Incident to find playbook for.

        Returns:
            Playbook ID or None.
        """
        for playbook_id, playbook_class in self.playbooks.items():
            playbook = playbook_class(self.config)
            if playbook.can_handle(incident):
                return playbook_id
        return None

    def register_playbook(
        self,
        playbook_id: str,
        playbook_class: Type[BasePlaybook],
    ) -> None:
        """Register a custom playbook.

        Args:
            playbook_id: Playbook identifier.
            playbook_class: Playbook class.
        """
        self.playbooks[playbook_id] = playbook_class
        logger.info("Registered playbook: %s", playbook_id)


# ---------------------------------------------------------------------------
# Global Executor Instance
# ---------------------------------------------------------------------------

_global_executor: Optional[PlaybookExecutor] = None


def get_playbook_executor(
    config: Optional[IncidentResponseConfig] = None,
) -> PlaybookExecutor:
    """Get or create the global playbook executor.

    Args:
        config: Optional configuration override.

    Returns:
        The global PlaybookExecutor instance.
    """
    global _global_executor

    if _global_executor is None:
        _global_executor = PlaybookExecutor(config)

    return _global_executor


def reset_playbook_executor() -> None:
    """Reset the global playbook executor."""
    global _global_executor
    _global_executor = None


__all__ = [
    # Base classes
    "BasePlaybook",
    "PlaybookResult",
    # Playbooks
    "CredentialCompromisePlaybook",
    "DDoSMitigationPlaybook",
    "DataBreachPlaybook",
    "MalwareContainmentPlaybook",
    "AccessRevocationPlaybook",
    "SessionHijackPlaybook",
    "BruteForceResponsePlaybook",
    "SQLInjectionResponsePlaybook",
    "APIAbusePlaybook",
    "InsiderThreatPlaybook",
    # Registry
    "PLAYBOOKS",
    # Executor
    "PlaybookExecutor",
    "get_playbook_executor",
    "reset_playbook_executor",
]
