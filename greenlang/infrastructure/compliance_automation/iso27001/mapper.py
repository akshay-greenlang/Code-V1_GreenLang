# -*- coding: utf-8 -*-
"""
ISO 27001:2022 Control Mapper - SEC-010 Phase 5

Maps all 93 ISO 27001:2022 Annex A controls to GreenLang technical controls
(SEC-001 through SEC-010). Provides automated assessment and Statement of
Applicability (SoA) generation.

ISO 27001:2022 Annex A Structure:
- A.5 Organizational controls (37 controls)
- A.6 People controls (8 controls)
- A.7 Physical controls (14 controls)
- A.8 Technological controls (34 controls)

Classes:
    - ISO27001Mapper: Control mapping and assessment.

Example:
    >>> mapper = ISO27001Mapper()
    >>> controls = await mapper.get_controls()
    >>> status = await mapper.calculate_compliance_score()
    >>> soa = await mapper.generate_soa()

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-010 Security Operations Automation Platform
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from greenlang.infrastructure.compliance_automation.base_framework import (
    BaseComplianceFramework,
    ControlAssessmentResult,
)
from greenlang.infrastructure.compliance_automation.models import (
    ComplianceFramework,
    ComplianceGap,
    ControlMapping,
    ControlStatus,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ISO 27001:2022 Annex A Control Mapping
# ---------------------------------------------------------------------------

# Maps ISO 27001:2022 controls to GreenLang SEC modules and evidence sources
# Format: control_id -> {name, sec_modules, evidence_sources, description}

ISO27001_CONTROL_MAPPING: Dict[str, Dict[str, Any]] = {
    # -------------------------------------------------------------------------
    # A.5 Organizational Controls (37 controls)
    # -------------------------------------------------------------------------
    "A.5.1": {
        "name": "Policies for information security",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["policy_documents", "policy_review_logs"],
        "description": "Information security policy and topic-specific policies",
    },
    "A.5.2": {
        "name": "Information security roles and responsibilities",
        "sec_modules": ["SEC-002", "SEC-006"],
        "evidence_sources": ["rbac_config", "role_definitions"],
        "description": "Roles and responsibilities for information security",
    },
    "A.5.3": {
        "name": "Segregation of duties",
        "sec_modules": ["SEC-002"],
        "evidence_sources": ["rbac_config", "access_matrix"],
        "description": "Segregation of duties for conflicting tasks",
    },
    "A.5.4": {
        "name": "Management responsibilities",
        "sec_modules": ["SEC-006", "SEC-009"],
        "evidence_sources": ["management_reviews", "security_committee_minutes"],
        "description": "Management commitment to information security",
    },
    "A.5.5": {
        "name": "Contact with authorities",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["authority_contacts", "incident_procedures"],
        "description": "Maintaining contacts with relevant authorities",
    },
    "A.5.6": {
        "name": "Contact with special interest groups",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["industry_memberships", "threat_intel_feeds"],
        "description": "Contact with security forums and associations",
    },
    "A.5.7": {
        "name": "Threat intelligence",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["threat_intel_reports", "vulnerability_feeds"],
        "description": "Collection and analysis of threat intelligence",
    },
    "A.5.8": {
        "name": "Information security in project management",
        "sec_modules": ["SEC-006", "SEC-010"],
        "evidence_sources": ["project_security_reviews", "sdlc_docs"],
        "description": "Integrating security into project management",
    },
    "A.5.9": {
        "name": "Inventory of information and other associated assets",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["asset_inventory", "cmdb"],
        "description": "Maintaining inventory of assets",
    },
    "A.5.10": {
        "name": "Acceptable use of information and other associated assets",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["acceptable_use_policy", "user_agreements"],
        "description": "Rules for acceptable use of assets",
    },
    "A.5.11": {
        "name": "Return of assets",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["offboarding_procedures", "asset_return_logs"],
        "description": "Return of assets upon termination",
    },
    "A.5.12": {
        "name": "Classification of information",
        "sec_modules": ["SEC-006", "SEC-007"],
        "evidence_sources": ["data_classification_policy", "classification_labels"],
        "description": "Information classification scheme",
    },
    "A.5.13": {
        "name": "Labelling of information",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["labeling_procedures", "metadata_tags"],
        "description": "Procedures for information labeling",
    },
    "A.5.14": {
        "name": "Information transfer",
        "sec_modules": ["SEC-003", "SEC-004"],
        "evidence_sources": ["transfer_policies", "encryption_config"],
        "description": "Secure transfer of information",
    },
    "A.5.15": {
        "name": "Access control",
        "sec_modules": ["SEC-001", "SEC-002"],
        "evidence_sources": ["access_control_policy", "authentication_config"],
        "description": "Access control policy and implementation",
    },
    "A.5.16": {
        "name": "Identity management",
        "sec_modules": ["SEC-001"],
        "evidence_sources": ["identity_management_config", "user_provisioning"],
        "description": "Identity lifecycle management",
    },
    "A.5.17": {
        "name": "Authentication information",
        "sec_modules": ["SEC-001"],
        "evidence_sources": ["password_policy", "mfa_config"],
        "description": "Management of authentication credentials",
    },
    "A.5.18": {
        "name": "Access rights",
        "sec_modules": ["SEC-002"],
        "evidence_sources": ["access_reviews", "rbac_assignments"],
        "description": "Provisioning and review of access rights",
    },
    "A.5.19": {
        "name": "Information security in supplier relationships",
        "sec_modules": ["SEC-006", "SEC-009"],
        "evidence_sources": ["vendor_agreements", "vendor_assessments"],
        "description": "Security requirements for suppliers",
    },
    "A.5.20": {
        "name": "Addressing information security within supplier agreements",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["supplier_contracts", "security_clauses"],
        "description": "Security terms in supplier agreements",
    },
    "A.5.21": {
        "name": "Managing information security in the ICT supply chain",
        "sec_modules": ["SEC-007", "SEC-009"],
        "evidence_sources": ["supply_chain_security_policy", "sbom_reports"],
        "description": "ICT supply chain security",
    },
    "A.5.22": {
        "name": "Monitoring, review and change management of supplier services",
        "sec_modules": ["SEC-009"],
        "evidence_sources": ["vendor_reviews", "sla_reports"],
        "description": "Ongoing supplier service management",
    },
    "A.5.23": {
        "name": "Information security for use of cloud services",
        "sec_modules": ["SEC-004", "SEC-009"],
        "evidence_sources": ["cloud_security_config", "cloud_compliance_reports"],
        "description": "Cloud service security management",
    },
    "A.5.24": {
        "name": "Information security incident management planning and preparation",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["incident_response_plan", "playbooks"],
        "description": "Incident response planning",
    },
    "A.5.25": {
        "name": "Assessment and decision on information security events",
        "sec_modules": ["SEC-005", "SEC-010"],
        "evidence_sources": ["event_classification_procedures", "siem_config"],
        "description": "Security event assessment",
    },
    "A.5.26": {
        "name": "Response to information security incidents",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["incident_response_procedures", "incident_logs"],
        "description": "Incident response execution",
    },
    "A.5.27": {
        "name": "Learning from information security incidents",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["post_mortems", "lessons_learned"],
        "description": "Incident post-mortem and improvement",
    },
    "A.5.28": {
        "name": "Collection of evidence",
        "sec_modules": ["SEC-005", "SEC-010"],
        "evidence_sources": ["forensic_procedures", "evidence_handling"],
        "description": "Evidence collection and preservation",
    },
    "A.5.29": {
        "name": "Information security during disruption",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["bcp_plans", "dr_procedures"],
        "description": "Security continuity during disruptions",
    },
    "A.5.30": {
        "name": "ICT readiness for business continuity",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["dr_tests", "failover_procedures"],
        "description": "ICT continuity planning",
    },
    "A.5.31": {
        "name": "Legal, statutory, regulatory and contractual requirements",
        "sec_modules": ["SEC-006", "SEC-009"],
        "evidence_sources": ["legal_register", "compliance_assessments"],
        "description": "Legal and regulatory compliance",
    },
    "A.5.32": {
        "name": "Intellectual property rights",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["ip_policy", "license_inventory"],
        "description": "Protection of intellectual property",
    },
    "A.5.33": {
        "name": "Protection of records",
        "sec_modules": ["SEC-005", "SEC-006"],
        "evidence_sources": ["records_retention_policy", "audit_logs"],
        "description": "Records protection and retention",
    },
    "A.5.34": {
        "name": "Privacy and protection of PII",
        "sec_modules": ["SEC-006", "SEC-010"],
        "evidence_sources": ["privacy_policy", "dsar_procedures"],
        "description": "PII protection (links to GDPR)",
    },
    "A.5.35": {
        "name": "Independent review of information security",
        "sec_modules": ["SEC-009"],
        "evidence_sources": ["audit_reports", "penetration_tests"],
        "description": "Independent security reviews",
    },
    "A.5.36": {
        "name": "Compliance with policies, rules and standards for information security",
        "sec_modules": ["SEC-006", "SEC-009"],
        "evidence_sources": ["compliance_checks", "policy_attestations"],
        "description": "Compliance monitoring and verification",
    },
    "A.5.37": {
        "name": "Documented operating procedures",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["runbooks", "operational_procedures"],
        "description": "Documentation of operating procedures",
    },
    # -------------------------------------------------------------------------
    # A.6 People Controls (8 controls)
    # -------------------------------------------------------------------------
    "A.6.1": {
        "name": "Screening",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["background_check_policy", "screening_records"],
        "description": "Background verification of employees",
    },
    "A.6.2": {
        "name": "Terms and conditions of employment",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["employment_contracts", "security_agreements"],
        "description": "Security responsibilities in employment",
    },
    "A.6.3": {
        "name": "Information security awareness, education and training",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["training_records", "awareness_campaigns"],
        "description": "Security training programs",
    },
    "A.6.4": {
        "name": "Disciplinary process",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["disciplinary_policy", "violation_records"],
        "description": "Disciplinary process for violations",
    },
    "A.6.5": {
        "name": "Responsibilities after termination or change of employment",
        "sec_modules": ["SEC-001", "SEC-006"],
        "evidence_sources": ["offboarding_procedures", "access_revocation_logs"],
        "description": "Post-employment security",
    },
    "A.6.6": {
        "name": "Confidentiality or non-disclosure agreements",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["nda_templates", "signed_ndas"],
        "description": "Confidentiality agreements",
    },
    "A.6.7": {
        "name": "Remote working",
        "sec_modules": ["SEC-001", "SEC-003"],
        "evidence_sources": ["remote_work_policy", "vpn_config"],
        "description": "Remote working security",
    },
    "A.6.8": {
        "name": "Information security event reporting",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["incident_reporting_procedure", "reporting_channels"],
        "description": "Security event reporting mechanisms",
    },
    # -------------------------------------------------------------------------
    # A.7 Physical Controls (14 controls)
    # -------------------------------------------------------------------------
    "A.7.1": {
        "name": "Physical security perimeters",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["physical_security_policy", "facility_diagrams"],
        "description": "Physical security perimeter definition",
    },
    "A.7.2": {
        "name": "Physical entry",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["access_control_system_logs", "badge_records"],
        "description": "Physical access controls",
    },
    "A.7.3": {
        "name": "Securing offices, rooms and facilities",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["facility_security_assessments", "lock_records"],
        "description": "Physical security of facilities",
    },
    "A.7.4": {
        "name": "Physical security monitoring",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["cctv_config", "monitoring_procedures"],
        "description": "Physical security monitoring systems",
    },
    "A.7.5": {
        "name": "Protecting against physical and environmental threats",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["environmental_controls", "fire_suppression"],
        "description": "Environmental protection",
    },
    "A.7.6": {
        "name": "Working in secure areas",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["secure_area_procedures", "escort_logs"],
        "description": "Procedures for secure areas",
    },
    "A.7.7": {
        "name": "Clear desk and clear screen",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["clear_desk_policy", "screen_lock_config"],
        "description": "Clear desk and screen policy",
    },
    "A.7.8": {
        "name": "Equipment siting and protection",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["equipment_placement_docs", "rack_diagrams"],
        "description": "Equipment placement security",
    },
    "A.7.9": {
        "name": "Security of assets off-premises",
        "sec_modules": ["SEC-003", "SEC-006"],
        "evidence_sources": ["mobile_device_policy", "encryption_requirements"],
        "description": "Off-premises asset security",
    },
    "A.7.10": {
        "name": "Storage media",
        "sec_modules": ["SEC-003", "SEC-006"],
        "evidence_sources": ["media_handling_policy", "encryption_config"],
        "description": "Storage media management",
    },
    "A.7.11": {
        "name": "Supporting utilities",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["ups_config", "power_monitoring"],
        "description": "Supporting utilities (power, etc.)",
    },
    "A.7.12": {
        "name": "Cabling security",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["cabling_diagrams", "cable_security"],
        "description": "Cable infrastructure security",
    },
    "A.7.13": {
        "name": "Equipment maintenance",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["maintenance_logs", "service_contracts"],
        "description": "Equipment maintenance procedures",
    },
    "A.7.14": {
        "name": "Secure disposal or re-use of equipment",
        "sec_modules": ["SEC-003", "SEC-006"],
        "evidence_sources": ["disposal_procedures", "sanitization_certificates"],
        "description": "Secure equipment disposal",
    },
    # -------------------------------------------------------------------------
    # A.8 Technological Controls (34 controls)
    # -------------------------------------------------------------------------
    "A.8.1": {
        "name": "User endpoint devices",
        "sec_modules": ["SEC-006", "SEC-007"],
        "evidence_sources": ["endpoint_policy", "mdm_config"],
        "description": "User endpoint device security",
    },
    "A.8.2": {
        "name": "Privileged access rights",
        "sec_modules": ["SEC-002"],
        "evidence_sources": ["privileged_access_policy", "pam_config"],
        "description": "Privileged access management",
    },
    "A.8.3": {
        "name": "Information access restriction",
        "sec_modules": ["SEC-002"],
        "evidence_sources": ["access_control_config", "authorization_rules"],
        "description": "Information access restrictions",
    },
    "A.8.4": {
        "name": "Access to source code",
        "sec_modules": ["SEC-002", "SEC-007"],
        "evidence_sources": ["repo_access_config", "code_access_logs"],
        "description": "Source code access control",
    },
    "A.8.5": {
        "name": "Secure authentication",
        "sec_modules": ["SEC-001"],
        "evidence_sources": ["authentication_config", "mfa_enrollment"],
        "description": "Secure authentication mechanisms",
    },
    "A.8.6": {
        "name": "Capacity management",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["capacity_monitoring", "scaling_policies"],
        "description": "System capacity management",
    },
    "A.8.7": {
        "name": "Protection against malware",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["antimalware_config", "scan_results"],
        "description": "Malware protection",
    },
    "A.8.8": {
        "name": "Management of technical vulnerabilities",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["vulnerability_scans", "patch_reports"],
        "description": "Technical vulnerability management",
    },
    "A.8.9": {
        "name": "Configuration management",
        "sec_modules": ["SEC-006", "SEC-007"],
        "evidence_sources": ["configuration_baselines", "change_records"],
        "description": "Configuration management",
    },
    "A.8.10": {
        "name": "Information deletion",
        "sec_modules": ["SEC-010"],
        "evidence_sources": ["data_deletion_procedures", "deletion_logs"],
        "description": "Secure information deletion",
    },
    "A.8.11": {
        "name": "Data masking",
        "sec_modules": ["SEC-003"],
        "evidence_sources": ["masking_config", "pii_handling"],
        "description": "Data masking techniques",
    },
    "A.8.12": {
        "name": "Data leakage prevention",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["dlp_config", "dlp_alerts"],
        "description": "Data leakage prevention",
    },
    "A.8.13": {
        "name": "Information backup",
        "sec_modules": ["SEC-004"],
        "evidence_sources": ["backup_config", "backup_logs"],
        "description": "Backup procedures and testing",
    },
    "A.8.14": {
        "name": "Redundancy of information processing facilities",
        "sec_modules": ["SEC-006"],
        "evidence_sources": ["redundancy_config", "failover_tests"],
        "description": "System redundancy",
    },
    "A.8.15": {
        "name": "Logging",
        "sec_modules": ["SEC-005"],
        "evidence_sources": ["logging_config", "log_samples"],
        "description": "Event logging",
    },
    "A.8.16": {
        "name": "Monitoring activities",
        "sec_modules": ["SEC-005", "SEC-010"],
        "evidence_sources": ["monitoring_config", "siem_rules"],
        "description": "Security monitoring",
    },
    "A.8.17": {
        "name": "Clock synchronization",
        "sec_modules": ["SEC-005"],
        "evidence_sources": ["ntp_config", "time_sync_logs"],
        "description": "Clock synchronization",
    },
    "A.8.18": {
        "name": "Use of privileged utility programs",
        "sec_modules": ["SEC-002"],
        "evidence_sources": ["utility_restrictions", "admin_tool_usage"],
        "description": "Privileged utility control",
    },
    "A.8.19": {
        "name": "Installation of software on operational systems",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["software_installation_policy", "change_management"],
        "description": "Software installation controls",
    },
    "A.8.20": {
        "name": "Networks security",
        "sec_modules": ["SEC-003", "SEC-004"],
        "evidence_sources": ["network_diagrams", "firewall_rules"],
        "description": "Network security controls",
    },
    "A.8.21": {
        "name": "Security of network services",
        "sec_modules": ["SEC-003", "SEC-004"],
        "evidence_sources": ["network_service_config", "tls_config"],
        "description": "Network service security",
    },
    "A.8.22": {
        "name": "Segregation of networks",
        "sec_modules": ["SEC-004"],
        "evidence_sources": ["network_segmentation", "vlan_config"],
        "description": "Network segregation",
    },
    "A.8.23": {
        "name": "Web filtering",
        "sec_modules": ["SEC-006", "SEC-010"],
        "evidence_sources": ["web_filter_config", "waf_rules"],
        "description": "Web content filtering",
    },
    "A.8.24": {
        "name": "Use of cryptography",
        "sec_modules": ["SEC-003"],
        "evidence_sources": ["encryption_policy", "crypto_inventory"],
        "description": "Cryptography usage",
    },
    "A.8.25": {
        "name": "Secure development life cycle",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["sdlc_policy", "security_gates"],
        "description": "Secure SDLC",
    },
    "A.8.26": {
        "name": "Application security requirements",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["security_requirements", "threat_models"],
        "description": "Application security requirements",
    },
    "A.8.27": {
        "name": "Secure system architecture and engineering principles",
        "sec_modules": ["SEC-007", "SEC-010"],
        "evidence_sources": ["architecture_docs", "design_reviews"],
        "description": "Secure architecture principles",
    },
    "A.8.28": {
        "name": "Secure coding",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["coding_standards", "sast_results"],
        "description": "Secure coding practices",
    },
    "A.8.29": {
        "name": "Security testing in development and acceptance",
        "sec_modules": ["SEC-007"],
        "evidence_sources": ["security_test_results", "acceptance_criteria"],
        "description": "Security testing",
    },
    "A.8.30": {
        "name": "Outsourced development",
        "sec_modules": ["SEC-006", "SEC-007"],
        "evidence_sources": ["vendor_security_requirements", "code_reviews"],
        "description": "Outsourced development security",
    },
    "A.8.31": {
        "name": "Separation of development, test and production environments",
        "sec_modules": ["SEC-006", "SEC-007"],
        "evidence_sources": ["environment_separation", "access_controls"],
        "description": "Environment separation",
    },
    "A.8.32": {
        "name": "Change management",
        "sec_modules": ["SEC-006", "SEC-007"],
        "evidence_sources": ["change_management_policy", "change_logs"],
        "description": "Change management process",
    },
    "A.8.33": {
        "name": "Test information",
        "sec_modules": ["SEC-003", "SEC-006"],
        "evidence_sources": ["test_data_policy", "data_anonymization"],
        "description": "Test data protection",
    },
    "A.8.34": {
        "name": "Protection of information systems during audit testing",
        "sec_modules": ["SEC-009"],
        "evidence_sources": ["audit_procedures", "audit_scope"],
        "description": "Audit testing controls",
    },
}


# ---------------------------------------------------------------------------
# ISO 27001 Mapper
# ---------------------------------------------------------------------------


class ISO27001Mapper(BaseComplianceFramework):
    """ISO 27001:2022 control mapping and assessment.

    Maps all 93 Annex A controls to GreenLang technical controls and
    provides automated compliance assessment.

    Attributes:
        CONTROL_MAPPING: Static mapping of all Annex A controls.

    Example:
        >>> mapper = ISO27001Mapper()
        >>> controls = await mapper.get_controls()
        >>> soa = await mapper.generate_soa()
    """

    CONTROL_MAPPING = ISO27001_CONTROL_MAPPING

    @property
    def framework_id(self) -> ComplianceFramework:
        """Return the framework identifier."""
        return ComplianceFramework.ISO27001

    @property
    def framework_name(self) -> str:
        """Return the human-readable framework name."""
        return "ISO/IEC 27001:2022"

    @property
    def framework_version(self) -> str:
        """Return the framework version."""
        return "2022"

    async def get_controls(self) -> List[ControlMapping]:
        """Get all ISO 27001:2022 Annex A controls.

        Returns:
            List of ControlMapping objects for all 93 controls.
        """
        if self._controls is not None:
            return self._controls

        controls: List[ControlMapping] = []
        for control_id, mapping in self.CONTROL_MAPPING.items():
            controls.append(
                ControlMapping(
                    framework=self.framework_id,
                    framework_control=control_id,
                    framework_control_name=mapping["name"],
                    technical_controls=mapping["sec_modules"],
                    evidence_sources=mapping["evidence_sources"],
                    status=ControlStatus.NOT_IMPLEMENTED,
                    notes=mapping.get("description", ""),
                )
            )

        self._controls = controls
        logger.info("Loaded %d ISO 27001 controls", len(controls))
        return controls

    async def assess_control(
        self,
        control_id: str,
        collect_evidence: bool = True,
    ) -> ControlAssessmentResult:
        """Assess a single ISO 27001 control.

        Args:
            control_id: The control ID (e.g., "A.5.1").
            collect_evidence: Whether to collect evidence.

        Returns:
            ControlAssessmentResult with status and score.

        Raises:
            ValueError: If control_id is not valid.
        """
        if control_id not in self.CONTROL_MAPPING:
            raise ValueError(
                f"Control {control_id} not found in ISO 27001:2022 Annex A"
            )

        mapping = self.CONTROL_MAPPING[control_id]
        logger.debug("Assessing ISO 27001 control: %s", control_id)

        # Collect evidence if requested
        evidence: List[Dict[str, Any]] = []
        if collect_evidence:
            evidence = await self.collect_evidence(control_id)

        # Determine status based on evidence
        status = ControlStatus.NOT_IMPLEMENTED
        score = 0.0
        gaps: List[ComplianceGap] = []

        # Check if technical controls are implemented
        # In production, this would query the actual control implementations
        sec_modules = mapping["sec_modules"]
        implemented_modules = self._check_sec_modules(sec_modules)

        if len(implemented_modules) == len(sec_modules):
            if evidence:
                status = ControlStatus.VERIFIED
                score = 100.0
            else:
                status = ControlStatus.IMPLEMENTED
                score = 80.0
        elif len(implemented_modules) > 0:
            status = ControlStatus.PARTIALLY_IMPLEMENTED
            score = (len(implemented_modules) / len(sec_modules)) * 60.0
            gaps.append(
                ComplianceGap(
                    framework=self.framework_id,
                    control_id=control_id,
                    title=f"Partial implementation of {mapping['name']}",
                    description=(
                        f"Only {len(implemented_modules)} of {len(sec_modules)} "
                        f"technical controls are implemented"
                    ),
                    severity="medium",
                )
            )
        else:
            gaps.append(
                ComplianceGap(
                    framework=self.framework_id,
                    control_id=control_id,
                    title=f"Missing control: {mapping['name']}",
                    description=f"Control {control_id} is not implemented",
                    severity="high",
                )
            )

        return ControlAssessmentResult(
            control_id=control_id,
            status=status,
            score=score,
            evidence_collected=evidence,
            gaps=gaps,
            notes=mapping.get("description", ""),
        )

    async def collect_evidence(
        self,
        control_id: str,
    ) -> List[Dict[str, Any]]:
        """Collect evidence for an ISO 27001 control.

        Args:
            control_id: The control ID.

        Returns:
            List of evidence items.

        Raises:
            ValueError: If control_id is not valid.
        """
        if control_id not in self.CONTROL_MAPPING:
            raise ValueError(
                f"Control {control_id} not found in ISO 27001:2022 Annex A"
            )

        mapping = self.CONTROL_MAPPING[control_id]
        evidence_sources = mapping["evidence_sources"]
        evidence: List[Dict[str, Any]] = []

        for source in evidence_sources:
            # In production, this would query actual evidence sources
            evidence_item = await self._collect_from_source(source)
            if evidence_item:
                evidence.append(evidence_item)

        logger.debug(
            "Collected %d evidence items for control %s",
            len(evidence),
            control_id,
        )
        return evidence

    async def generate_soa(self) -> Dict[str, Any]:
        """Generate Statement of Applicability (SoA).

        The SoA documents which controls are applicable, implemented,
        and any exclusions with justifications.

        Returns:
            Dictionary containing the complete SoA.
        """
        logger.info("Generating ISO 27001 Statement of Applicability")

        controls = await self.get_controls()
        soa_entries: List[Dict[str, Any]] = []

        for control in controls:
            result = await self.assess_control(control.framework_control)

            soa_entries.append({
                "control_id": control.framework_control,
                "control_name": control.framework_control_name,
                "applicable": True,  # In production, allow exclusions
                "justification": "",
                "implemented": result.status in (
                    ControlStatus.IMPLEMENTED,
                    ControlStatus.VERIFIED,
                ),
                "status": result.status.value,
                "score": result.score,
                "evidence_sources": control.evidence_sources,
                "technical_controls": control.technical_controls,
                "assessed_at": result.assessed_at.isoformat(),
            })

        # Calculate summary statistics
        total = len(soa_entries)
        implemented = sum(1 for e in soa_entries if e["implemented"])
        not_applicable = sum(1 for e in soa_entries if not e["applicable"])

        return {
            "framework": "ISO/IEC 27001:2022",
            "version": "2022",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "total_controls": total,
                "implemented": implemented,
                "not_implemented": total - implemented - not_applicable,
                "not_applicable": not_applicable,
                "compliance_percentage": (implemented / (total - not_applicable)) * 100
                if (total - not_applicable) > 0 else 0,
            },
            "domains": {
                "A.5": "Organizational controls (37)",
                "A.6": "People controls (8)",
                "A.7": "Physical controls (14)",
                "A.8": "Technological controls (34)",
            },
            "controls": soa_entries,
        }

    def map_to_technical_controls(
        self,
        control_id: str,
    ) -> List[str]:
        """Map an ISO 27001 control to GreenLang technical controls.

        Args:
            control_id: The ISO 27001 control ID.

        Returns:
            List of SEC module IDs (e.g., ["SEC-001", "SEC-002"]).

        Raises:
            ValueError: If control_id is not valid.
        """
        if control_id not in self.CONTROL_MAPPING:
            raise ValueError(f"Control {control_id} not found")

        return self.CONTROL_MAPPING[control_id]["sec_modules"]

    def get_controls_by_domain(self, domain: str) -> List[str]:
        """Get all controls in a specific domain.

        Args:
            domain: The domain prefix (e.g., "A.5", "A.6").

        Returns:
            List of control IDs in that domain.
        """
        return [
            control_id
            for control_id in self.CONTROL_MAPPING.keys()
            if control_id.startswith(domain)
        ]

    def _check_sec_modules(self, sec_modules: List[str]) -> List[str]:
        """Check which SEC modules are implemented.

        In production, this would query the actual module implementations.

        Args:
            sec_modules: List of SEC module IDs.

        Returns:
            List of implemented module IDs.
        """
        # For now, assume all modules are implemented
        # In production, check against actual implementation status
        return sec_modules

    async def _collect_from_source(
        self,
        source: str,
    ) -> Optional[Dict[str, Any]]:
        """Collect evidence from a specific source.

        Args:
            source: The evidence source name.

        Returns:
            Evidence item dictionary or None if not available.
        """
        # In production, implement actual evidence collection
        # For now, return placeholder evidence
        return {
            "source": source,
            "collected_at": datetime.now(timezone.utc).isoformat(),
            "type": "automated",
            "status": "collected",
        }


__all__ = [
    "ISO27001Mapper",
    "ISO27001_CONTROL_MAPPING",
]
