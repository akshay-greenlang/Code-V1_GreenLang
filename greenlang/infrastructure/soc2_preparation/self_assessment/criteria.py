# -*- coding: utf-8 -*-
"""
SOC 2 Trust Service Criteria Definitions - SEC-009

Complete definitions of all 48 SOC 2 Trust Service Criteria (TSC) as defined
by AICPA. Each criterion includes the identifier, description, category,
control points, and evidence requirements for self-assessment.

Trust Service Categories:
    - Security (Common Criteria CC1-CC9): Required for all SOC 2 reports
    - Availability (A1): System availability commitments
    - Confidentiality (C1): Protection of confidential information
    - Processing Integrity (PI1): Accuracy and completeness of processing
    - Privacy (P1-P8): Protection of personal information

This module provides:
    - TSC_CRITERIA: Complete dictionary of all 48 criteria
    - CATEGORY_WEIGHTS: Category weights for score calculation
    - Helper functions for criteria lookup and filtering

Example:
    >>> from greenlang.infrastructure.soc2_preparation.self_assessment.criteria import (
    ...     TSC_CRITERIA, get_criteria_by_category, get_criterion
    ... )
    >>> cc61 = get_criterion("CC6.1")
    >>> cc61["description"]
    'The entity implements logical access security software...'
    >>> security_criteria = get_criteria_by_category("security")
    >>> len(security_criteria)
    33

Author: GreenLang Security Team
Date: February 2026
PRD: SEC-009 SOC 2 Type II Preparation
Reference: AICPA 2017 Trust Services Criteria
"""

from __future__ import annotations

from typing import Dict, List, Optional, TypedDict


class CriterionDefinition(TypedDict):
    """Type definition for a SOC 2 criterion."""

    id: str
    description: str
    category: str
    subcategory: str
    control_points: List[str]
    evidence_requirements: List[str]
    common_controls: List[str]
    risk_level: str


# ---------------------------------------------------------------------------
# Category Weights for Score Calculation
# ---------------------------------------------------------------------------

CATEGORY_WEIGHTS: Dict[str, float] = {
    "security": 1.0,  # Security is required and weighted highest
    "availability": 0.8,
    "confidentiality": 0.8,
    "processing_integrity": 0.7,
    "privacy": 0.9,
}

# ---------------------------------------------------------------------------
# Trust Service Criteria - Security (Common Criteria)
# ---------------------------------------------------------------------------

_CC1_CONTROL_ENVIRONMENT: Dict[str, CriterionDefinition] = {
    "CC1.1": {
        "id": "CC1.1",
        "description": "The entity demonstrates a commitment to integrity and ethical values.",
        "category": "security",
        "subcategory": "control_environment",
        "control_points": [
            "Code of conduct or ethics policy exists and is communicated",
            "Management sets tone at the top regarding integrity",
            "Violations of code of conduct are addressed",
            "Background checks performed on employees",
        ],
        "evidence_requirements": [
            "Code of conduct/ethics policy document",
            "Employee acknowledgment records",
            "Disciplinary action documentation (sanitized)",
            "Background check policy and procedures",
            "Training completion records",
        ],
        "common_controls": [
            "Code of conduct",
            "Ethics hotline",
            "Background checks",
            "Annual ethics training",
        ],
        "risk_level": "high",
    },
    "CC1.2": {
        "id": "CC1.2",
        "description": "The board of directors demonstrates independence from management and exercises oversight.",
        "category": "security",
        "subcategory": "control_environment",
        "control_points": [
            "Board or oversight body is independent from management",
            "Board reviews security and compliance matters",
            "Audit committee or equivalent exists",
            "Board receives regular security reports",
        ],
        "evidence_requirements": [
            "Board charter or bylaws",
            "Board meeting minutes (redacted)",
            "Security/compliance committee charter",
            "Board member independence documentation",
            "Security reports presented to board",
        ],
        "common_controls": [
            "Board oversight",
            "Audit committee",
            "Regular reporting",
        ],
        "risk_level": "medium",
    },
    "CC1.3": {
        "id": "CC1.3",
        "description": "Management establishes structures, reporting lines, and authorities and responsibilities.",
        "category": "security",
        "subcategory": "control_environment",
        "control_points": [
            "Organizational structure is defined and communicated",
            "Reporting lines are clear",
            "Roles and responsibilities are documented",
            "Security responsibilities are assigned",
        ],
        "evidence_requirements": [
            "Organizational chart",
            "Role descriptions/job descriptions",
            "RACI matrix for security functions",
            "Security team structure documentation",
            "Authority and responsibility documentation",
        ],
        "common_controls": [
            "Org chart",
            "Job descriptions",
            "RACI matrix",
        ],
        "risk_level": "medium",
    },
    "CC1.4": {
        "id": "CC1.4",
        "description": "The entity demonstrates a commitment to attract, develop, and retain competent individuals.",
        "category": "security",
        "subcategory": "control_environment",
        "control_points": [
            "Competency requirements are defined for positions",
            "Training and development programs exist",
            "Performance evaluations are conducted",
            "Security awareness training is provided",
        ],
        "evidence_requirements": [
            "Job requirements and qualifications",
            "Training program documentation",
            "Security awareness training records",
            "Performance evaluation process documentation",
            "Certification requirements (where applicable)",
        ],
        "common_controls": [
            "Security awareness training",
            "Performance management",
            "Competency requirements",
        ],
        "risk_level": "medium",
    },
    "CC1.5": {
        "id": "CC1.5",
        "description": "The entity holds individuals accountable for their internal control responsibilities.",
        "category": "security",
        "subcategory": "control_environment",
        "control_points": [
            "Performance measures include security responsibilities",
            "Accountability is enforced through performance management",
            "Disciplinary actions for policy violations exist",
            "Incentives align with security objectives",
        ],
        "evidence_requirements": [
            "Performance evaluation criteria including security",
            "Disciplinary policy and procedures",
            "Examples of accountability enforcement (sanitized)",
            "Incentive/compensation structure documentation",
        ],
        "common_controls": [
            "Performance management",
            "Disciplinary procedures",
            "Accountability framework",
        ],
        "risk_level": "medium",
    },
}

_CC2_COMMUNICATION: Dict[str, CriterionDefinition] = {
    "CC2.1": {
        "id": "CC2.1",
        "description": "The entity obtains or generates and uses relevant, quality information to support the functioning of internal control.",
        "category": "security",
        "subcategory": "communication",
        "control_points": [
            "Information requirements are identified",
            "Information is captured from internal and external sources",
            "Data quality processes exist",
            "Information is processed and maintained for decision-making",
        ],
        "evidence_requirements": [
            "Information management policy",
            "Data quality procedures",
            "Security metrics and reporting",
            "Information source documentation",
            "Data validation procedures",
        ],
        "common_controls": [
            "Security metrics",
            "Data quality controls",
            "Information management",
        ],
        "risk_level": "medium",
    },
    "CC2.2": {
        "id": "CC2.2",
        "description": "The entity internally communicates information necessary to support the functioning of internal control.",
        "category": "security",
        "subcategory": "communication",
        "control_points": [
            "Communication policies exist",
            "Security policies are communicated to employees",
            "Changes to policies are communicated",
            "Communication channels exist for security matters",
        ],
        "evidence_requirements": [
            "Internal communication policy",
            "Policy communication records",
            "Security newsletter or bulletins",
            "Communication of policy changes",
            "Employee acknowledgment records",
        ],
        "common_controls": [
            "Policy communication",
            "Security bulletins",
            "Internal announcements",
        ],
        "risk_level": "medium",
    },
    "CC2.3": {
        "id": "CC2.3",
        "description": "The entity communicates with external parties regarding matters affecting the functioning of internal control.",
        "category": "security",
        "subcategory": "communication",
        "control_points": [
            "External communication channels exist",
            "Security commitments are communicated to customers",
            "Regulatory communications are managed",
            "Vendor communications regarding security exist",
        ],
        "evidence_requirements": [
            "External communication policy",
            "Customer security documentation (SOC report, security page)",
            "Regulatory correspondence records",
            "Vendor security communication records",
            "Security incident notification procedures",
        ],
        "common_controls": [
            "SOC 2 report distribution",
            "Security page/trust center",
            "Vendor security questionnaires",
        ],
        "risk_level": "medium",
    },
}

_CC3_RISK_ASSESSMENT: Dict[str, CriterionDefinition] = {
    "CC3.1": {
        "id": "CC3.1",
        "description": "The entity specifies objectives with sufficient clarity to enable the identification and assessment of risks relating to objectives.",
        "category": "security",
        "subcategory": "risk_assessment",
        "control_points": [
            "Security objectives are defined",
            "Objectives are measurable",
            "Objectives align with business strategy",
            "Objectives cover relevant security domains",
        ],
        "evidence_requirements": [
            "Security objectives documentation",
            "Security strategy or roadmap",
            "Business alignment documentation",
            "Security program charter",
            "KPIs and metrics for objectives",
        ],
        "common_controls": [
            "Security strategy",
            "Security objectives",
            "KPI tracking",
        ],
        "risk_level": "high",
    },
    "CC3.2": {
        "id": "CC3.2",
        "description": "The entity identifies risks to the achievement of its objectives and analyzes risks as a basis for determining how the risks should be managed.",
        "category": "security",
        "subcategory": "risk_assessment",
        "control_points": [
            "Risk identification process exists",
            "Risk register is maintained",
            "Risks are analyzed for likelihood and impact",
            "Risk owners are assigned",
        ],
        "evidence_requirements": [
            "Risk assessment methodology",
            "Risk register",
            "Risk analysis documentation",
            "Risk owner assignments",
            "Risk assessment reports",
        ],
        "common_controls": [
            "Risk register",
            "Risk assessments",
            "Risk analysis",
        ],
        "risk_level": "high",
    },
    "CC3.3": {
        "id": "CC3.3",
        "description": "The entity considers the potential for fraud in assessing risks to the achievement of objectives.",
        "category": "security",
        "subcategory": "risk_assessment",
        "control_points": [
            "Fraud risk assessment is performed",
            "Fraud risks are identified and documented",
            "Anti-fraud controls are implemented",
            "Fraud awareness training exists",
        ],
        "evidence_requirements": [
            "Fraud risk assessment documentation",
            "Anti-fraud policy",
            "Fraud detection controls documentation",
            "Fraud awareness training records",
            "Fraud risk register entries",
        ],
        "common_controls": [
            "Fraud risk assessment",
            "Anti-fraud controls",
            "Fraud awareness training",
        ],
        "risk_level": "high",
    },
    "CC3.4": {
        "id": "CC3.4",
        "description": "The entity identifies and assesses changes that could significantly impact the system of internal control.",
        "category": "security",
        "subcategory": "risk_assessment",
        "control_points": [
            "Change management process includes risk assessment",
            "Significant changes are evaluated for control impact",
            "External changes (regulatory, technology) are monitored",
            "Change impact assessments are performed",
        ],
        "evidence_requirements": [
            "Change management policy",
            "Change risk assessment procedures",
            "Examples of change impact assessments",
            "External change monitoring documentation",
            "Regulatory change tracking",
        ],
        "common_controls": [
            "Change management",
            "Impact assessments",
            "Regulatory monitoring",
        ],
        "risk_level": "medium",
    },
}

_CC4_MONITORING: Dict[str, CriterionDefinition] = {
    "CC4.1": {
        "id": "CC4.1",
        "description": "The entity selects, develops, and performs ongoing and/or separate evaluations to ascertain whether the components of internal control are present and functioning.",
        "category": "security",
        "subcategory": "monitoring",
        "control_points": [
            "Ongoing monitoring activities exist",
            "Separate evaluations (audits) are performed",
            "Monitoring covers all control components",
            "Monitoring frequency is appropriate",
        ],
        "evidence_requirements": [
            "Monitoring program documentation",
            "Continuous monitoring tools and dashboards",
            "Internal audit reports",
            "Self-assessment results",
            "Security metrics and KPIs",
        ],
        "common_controls": [
            "Security monitoring",
            "Internal audits",
            "Self-assessments",
        ],
        "risk_level": "high",
    },
    "CC4.2": {
        "id": "CC4.2",
        "description": "The entity evaluates and communicates internal control deficiencies in a timely manner to those parties responsible for taking corrective action.",
        "category": "security",
        "subcategory": "monitoring",
        "control_points": [
            "Deficiency identification process exists",
            "Deficiencies are communicated to appropriate parties",
            "Corrective action tracking exists",
            "Deficiency remediation is monitored",
        ],
        "evidence_requirements": [
            "Deficiency reporting procedures",
            "Deficiency tracking system/register",
            "Corrective action plans",
            "Management reporting on deficiencies",
            "Remediation progress reports",
        ],
        "common_controls": [
            "Issue tracking",
            "Corrective action plans",
            "Management reporting",
        ],
        "risk_level": "high",
    },
}

_CC5_CONTROL_ACTIVITIES: Dict[str, CriterionDefinition] = {
    "CC5.1": {
        "id": "CC5.1",
        "description": "The entity selects and develops control activities that contribute to the mitigation of risks to the achievement of objectives to acceptable levels.",
        "category": "security",
        "subcategory": "control_activities",
        "control_points": [
            "Control activities are designed based on risk assessment",
            "Controls are documented",
            "Controls address identified risks",
            "Control design is reviewed periodically",
        ],
        "evidence_requirements": [
            "Control inventory/framework documentation",
            "Risk-to-control mapping",
            "Control design documentation",
            "Control review records",
            "Policy and procedure documentation",
        ],
        "common_controls": [
            "Control framework",
            "Risk-control mapping",
            "Control documentation",
        ],
        "risk_level": "high",
    },
    "CC5.2": {
        "id": "CC5.2",
        "description": "The entity also selects and develops general control activities over technology to support the achievement of objectives.",
        "category": "security",
        "subcategory": "control_activities",
        "control_points": [
            "IT general controls are defined",
            "Technology controls support business objectives",
            "IT controls cover access, change, operations",
            "IT controls are documented and tested",
        ],
        "evidence_requirements": [
            "IT general controls documentation",
            "Technology control framework",
            "IT control testing results",
            "Access control documentation",
            "Change management procedures",
        ],
        "common_controls": [
            "IT general controls",
            "Access management",
            "Change management",
        ],
        "risk_level": "high",
    },
    "CC5.3": {
        "id": "CC5.3",
        "description": "The entity deploys control activities through policies that establish what is expected and procedures that put policies into action.",
        "category": "security",
        "subcategory": "control_activities",
        "control_points": [
            "Security policies are established",
            "Procedures implement policy requirements",
            "Policies are approved and communicated",
            "Policy compliance is monitored",
        ],
        "evidence_requirements": [
            "Security policy library",
            "Procedure documentation",
            "Policy approval records",
            "Policy communication records",
            "Policy compliance monitoring results",
        ],
        "common_controls": [
            "Security policies",
            "Operating procedures",
            "Policy management",
        ],
        "risk_level": "medium",
    },
}

_CC6_LOGICAL_AND_PHYSICAL_ACCESS: Dict[str, CriterionDefinition] = {
    "CC6.1": {
        "id": "CC6.1",
        "description": "The entity implements logical access security software, infrastructure, and architectures over protected information assets to protect them from security events.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Logical access controls are implemented",
            "Access control software/tools are deployed",
            "Security architecture is defined",
            "Protected assets are identified",
        ],
        "evidence_requirements": [
            "Access control policy",
            "Security architecture documentation",
            "Access control tool inventory",
            "Asset classification documentation",
            "Network security architecture",
        ],
        "common_controls": [
            "Access control system",
            "Network security",
            "Identity management",
        ],
        "risk_level": "critical",
    },
    "CC6.2": {
        "id": "CC6.2",
        "description": "Prior to issuing system credentials and granting system access, the entity registers and authorizes new internal and external users.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "User registration process exists",
            "Access authorization is required before provisioning",
            "User identity is verified",
            "Access is based on business need",
        ],
        "evidence_requirements": [
            "User provisioning procedures",
            "Access request and approval records",
            "Identity verification procedures",
            "User onboarding documentation",
            "Access authorization workflow",
        ],
        "common_controls": [
            "User provisioning",
            "Access requests",
            "Identity verification",
        ],
        "risk_level": "high",
    },
    "CC6.3": {
        "id": "CC6.3",
        "description": "The entity authorizes, modifies, or removes access to data, software, functions, and other protected information assets based on roles, responsibilities, or the system design.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Role-based access control (RBAC) is implemented",
            "Access modifications follow approval process",
            "Access removal/deprovisioning is timely",
            "Access changes are logged",
        ],
        "evidence_requirements": [
            "RBAC documentation",
            "Access modification procedures",
            "Termination/deprovisioning procedures",
            "Access change logs",
            "Role definitions and assignments",
        ],
        "common_controls": [
            "RBAC",
            "Access modification",
            "Deprovisioning",
        ],
        "risk_level": "high",
    },
    "CC6.4": {
        "id": "CC6.4",
        "description": "The entity restricts physical access to facilities and protected information assets to authorized personnel.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Physical access controls exist for facilities",
            "Data center access is restricted",
            "Visitor access is controlled",
            "Physical access logs are maintained",
        ],
        "evidence_requirements": [
            "Physical security policy",
            "Physical access control procedures",
            "Data center security documentation",
            "Visitor management procedures",
            "Physical access logs",
        ],
        "common_controls": [
            "Badge access",
            "Data center security",
            "Visitor management",
        ],
        "risk_level": "high",
    },
    "CC6.5": {
        "id": "CC6.5",
        "description": "The entity discontinues logical and physical protections over physical assets only after the ability to read or recover data and software has been diminished.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Asset disposal procedures exist",
            "Data destruction is verified",
            "Media sanitization is performed",
            "Asset disposal is documented",
        ],
        "evidence_requirements": [
            "Asset disposal policy",
            "Data destruction procedures",
            "Media sanitization records",
            "Certificates of destruction",
            "Asset disposal logs",
        ],
        "common_controls": [
            "Media sanitization",
            "Asset disposal",
            "Data destruction",
        ],
        "risk_level": "medium",
    },
    "CC6.6": {
        "id": "CC6.6",
        "description": "The entity implements logical access security measures to protect against threats from sources outside its system boundaries.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Perimeter security controls exist",
            "Firewall and network security are implemented",
            "Intrusion detection/prevention exists",
            "External threat protection is implemented",
        ],
        "evidence_requirements": [
            "Network security architecture",
            "Firewall configuration and rules",
            "IDS/IPS documentation",
            "Perimeter security documentation",
            "Threat intelligence integration",
        ],
        "common_controls": [
            "Firewalls",
            "IDS/IPS",
            "WAF",
            "DDoS protection",
        ],
        "risk_level": "critical",
    },
    "CC6.7": {
        "id": "CC6.7",
        "description": "The entity restricts the transmission, movement, and removal of information to authorized internal and external users and processes.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Data transfer controls exist",
            "Encryption is used for data in transit",
            "Data loss prevention controls exist",
            "External data sharing is controlled",
        ],
        "evidence_requirements": [
            "Data transfer policy",
            "Encryption standards and configuration",
            "DLP policy and tool configuration",
            "External data sharing procedures",
            "Secure file transfer documentation",
        ],
        "common_controls": [
            "TLS/encryption",
            "DLP",
            "Secure file transfer",
        ],
        "risk_level": "high",
    },
    "CC6.8": {
        "id": "CC6.8",
        "description": "The entity implements controls to prevent or detect and act upon the introduction of unauthorized or malicious software.",
        "category": "security",
        "subcategory": "access_controls",
        "control_points": [
            "Malware protection is implemented",
            "Anti-malware software is deployed",
            "Software installation is controlled",
            "Malware detection and response exists",
        ],
        "evidence_requirements": [
            "Anti-malware policy",
            "Endpoint protection documentation",
            "Software installation controls",
            "Malware detection logs and alerts",
            "Incident response for malware",
        ],
        "common_controls": [
            "Endpoint protection",
            "Anti-malware",
            "Application control",
        ],
        "risk_level": "high",
    },
}

_CC7_SYSTEM_OPERATIONS: Dict[str, CriterionDefinition] = {
    "CC7.1": {
        "id": "CC7.1",
        "description": "To meet its objectives, the entity uses detection and monitoring procedures to identify changes to configurations that result in the introduction of new vulnerabilities.",
        "category": "security",
        "subcategory": "system_operations",
        "control_points": [
            "Configuration monitoring exists",
            "Vulnerability scanning is performed",
            "Configuration drift detection exists",
            "Security monitoring is implemented",
        ],
        "evidence_requirements": [
            "Configuration management policy",
            "Vulnerability scanning reports",
            "Configuration monitoring documentation",
            "Security monitoring tools and dashboards",
            "Drift detection alerts",
        ],
        "common_controls": [
            "Vulnerability scanning",
            "Configuration monitoring",
            "Security monitoring",
        ],
        "risk_level": "high",
    },
    "CC7.2": {
        "id": "CC7.2",
        "description": "The entity monitors system components and the operation of those components for anomalies that are indicative of malicious acts, natural disasters, and errors.",
        "category": "security",
        "subcategory": "system_operations",
        "control_points": [
            "System monitoring is implemented",
            "Anomaly detection exists",
            "Alerting is configured",
            "Monitoring covers critical systems",
        ],
        "evidence_requirements": [
            "System monitoring documentation",
            "Anomaly detection configuration",
            "Alert definitions and thresholds",
            "SIEM or log management documentation",
            "Monitoring coverage map",
        ],
        "common_controls": [
            "SIEM",
            "Log management",
            "Anomaly detection",
        ],
        "risk_level": "high",
    },
    "CC7.3": {
        "id": "CC7.3",
        "description": "The entity evaluates security events to determine whether they could or have resulted in a failure of the entity to meet its objectives.",
        "category": "security",
        "subcategory": "system_operations",
        "control_points": [
            "Security event evaluation process exists",
            "Event triage procedures exist",
            "Impact assessment is performed",
            "Event escalation procedures exist",
        ],
        "evidence_requirements": [
            "Security event management procedures",
            "Event triage documentation",
            "Impact assessment procedures",
            "Escalation procedures",
            "Sample event evaluations",
        ],
        "common_controls": [
            "Event management",
            "Triage procedures",
            "Escalation",
        ],
        "risk_level": "high",
    },
    "CC7.4": {
        "id": "CC7.4",
        "description": "The entity responds to identified security incidents by executing a defined incident response program.",
        "category": "security",
        "subcategory": "system_operations",
        "control_points": [
            "Incident response plan exists",
            "Incident response team is defined",
            "Incident response procedures are documented",
            "Incident response is tested",
        ],
        "evidence_requirements": [
            "Incident response plan",
            "Incident response team roster",
            "Incident response procedures",
            "Incident response test results (tabletop exercises)",
            "Incident response metrics",
        ],
        "common_controls": [
            "Incident response plan",
            "IR team",
            "IR procedures",
        ],
        "risk_level": "critical",
    },
    "CC7.5": {
        "id": "CC7.5",
        "description": "The entity identifies, develops, and implements activities to recover from identified security incidents.",
        "category": "security",
        "subcategory": "system_operations",
        "control_points": [
            "Recovery procedures exist",
            "Post-incident recovery is documented",
            "Lessons learned are captured",
            "Recovery testing is performed",
        ],
        "evidence_requirements": [
            "Recovery procedures documentation",
            "Post-incident reports",
            "Lessons learned documentation",
            "Recovery test results",
            "Business continuity/DR documentation",
        ],
        "common_controls": [
            "Recovery procedures",
            "Post-incident review",
            "Lessons learned",
        ],
        "risk_level": "high",
    },
}

_CC8_CHANGE_MANAGEMENT: Dict[str, CriterionDefinition] = {
    "CC8.1": {
        "id": "CC8.1",
        "description": "The entity authorizes, designs, develops or acquires, configures, documents, tests, approves, and implements changes to infrastructure, data, software, and procedures.",
        "category": "security",
        "subcategory": "change_management",
        "control_points": [
            "Change management process exists",
            "Changes require authorization",
            "Changes are tested before deployment",
            "Emergency change procedures exist",
        ],
        "evidence_requirements": [
            "Change management policy",
            "Change management procedures",
            "Change approval records",
            "Change testing documentation",
            "Emergency change procedures",
            "Change logs",
        ],
        "common_controls": [
            "Change management",
            "Change approval",
            "Change testing",
        ],
        "risk_level": "high",
    },
}

_CC9_RISK_MITIGATION: Dict[str, CriterionDefinition] = {
    "CC9.1": {
        "id": "CC9.1",
        "description": "The entity identifies, selects, and develops risk mitigation activities for risks arising from potential business disruptions.",
        "category": "security",
        "subcategory": "risk_mitigation",
        "control_points": [
            "Business continuity planning exists",
            "Disaster recovery procedures exist",
            "Risk mitigation strategies are documented",
            "BC/DR is tested",
        ],
        "evidence_requirements": [
            "Business continuity plan",
            "Disaster recovery plan",
            "Risk mitigation documentation",
            "BC/DR test results",
            "Recovery time objectives (RTO/RPO)",
        ],
        "common_controls": [
            "Business continuity",
            "Disaster recovery",
            "BC/DR testing",
        ],
        "risk_level": "high",
    },
    "CC9.2": {
        "id": "CC9.2",
        "description": "The entity assesses and manages risks associated with vendors and business partners.",
        "category": "security",
        "subcategory": "risk_mitigation",
        "control_points": [
            "Vendor risk management program exists",
            "Vendor assessments are performed",
            "Vendor contracts include security requirements",
            "Ongoing vendor monitoring exists",
        ],
        "evidence_requirements": [
            "Vendor management policy",
            "Vendor risk assessment procedures",
            "Vendor assessment reports",
            "Vendor contract security requirements",
            "Vendor monitoring documentation",
        ],
        "common_controls": [
            "Vendor management",
            "Vendor assessments",
            "Contract security requirements",
        ],
        "risk_level": "high",
    },
}

# ---------------------------------------------------------------------------
# Trust Service Criteria - Availability
# ---------------------------------------------------------------------------

_A1_AVAILABILITY: Dict[str, CriterionDefinition] = {
    "A1.1": {
        "id": "A1.1",
        "description": "The entity maintains, monitors, and evaluates current processing capacity and use of system components to manage capacity demand and to enable the implementation of additional capacity.",
        "category": "availability",
        "subcategory": "capacity_management",
        "control_points": [
            "Capacity planning process exists",
            "Capacity monitoring is implemented",
            "Capacity thresholds and alerts exist",
            "Capacity is scaled as needed",
        ],
        "evidence_requirements": [
            "Capacity management procedures",
            "Capacity monitoring dashboards",
            "Capacity planning documentation",
            "Auto-scaling configuration",
            "Capacity alerts and thresholds",
        ],
        "common_controls": [
            "Capacity monitoring",
            "Auto-scaling",
            "Capacity planning",
        ],
        "risk_level": "high",
    },
    "A1.2": {
        "id": "A1.2",
        "description": "The entity authorizes, designs, develops or acquires, implements, operates, approves, maintains, and monitors environmental protections, software, data backup processes, and recovery infrastructure.",
        "category": "availability",
        "subcategory": "system_recovery",
        "control_points": [
            "Environmental controls exist",
            "Backup processes are implemented",
            "Recovery infrastructure exists",
            "Backups are tested",
        ],
        "evidence_requirements": [
            "Environmental controls documentation",
            "Backup policy and procedures",
            "Backup configuration and schedules",
            "Backup test/restore results",
            "Recovery infrastructure documentation",
        ],
        "common_controls": [
            "Data backups",
            "Recovery infrastructure",
            "Environmental controls",
        ],
        "risk_level": "high",
    },
    "A1.3": {
        "id": "A1.3",
        "description": "The entity tests recovery plan procedures supporting system recovery to meet its objectives.",
        "category": "availability",
        "subcategory": "system_recovery",
        "control_points": [
            "Recovery testing is performed",
            "Recovery tests are documented",
            "Test results are reviewed",
            "Issues identified are remediated",
        ],
        "evidence_requirements": [
            "Recovery test plan",
            "Recovery test results",
            "Recovery test schedule",
            "Issue remediation from tests",
            "RTO/RPO achievement documentation",
        ],
        "common_controls": [
            "DR testing",
            "Recovery validation",
            "Test documentation",
        ],
        "risk_level": "high",
    },
}

# ---------------------------------------------------------------------------
# Trust Service Criteria - Confidentiality
# ---------------------------------------------------------------------------

_C1_CONFIDENTIALITY: Dict[str, CriterionDefinition] = {
    "C1.1": {
        "id": "C1.1",
        "description": "The entity identifies and maintains confidential information to meet the entity's objectives related to confidentiality.",
        "category": "confidentiality",
        "subcategory": "data_classification",
        "control_points": [
            "Confidential information is identified",
            "Data classification exists",
            "Confidential data inventory is maintained",
            "Classification is applied consistently",
        ],
        "evidence_requirements": [
            "Data classification policy",
            "Data classification scheme",
            "Confidential data inventory",
            "Classification procedures",
            "Data handling guidelines",
        ],
        "common_controls": [
            "Data classification",
            "Data inventory",
            "Labeling",
        ],
        "risk_level": "high",
    },
    "C1.2": {
        "id": "C1.2",
        "description": "The entity disposes of confidential information to meet the entity's objectives related to confidentiality.",
        "category": "confidentiality",
        "subcategory": "data_disposal",
        "control_points": [
            "Data retention policy exists",
            "Data disposal procedures exist",
            "Disposal is verified",
            "Disposal is documented",
        ],
        "evidence_requirements": [
            "Data retention policy",
            "Data disposal procedures",
            "Disposal verification records",
            "Certificates of destruction",
            "Retention schedule",
        ],
        "common_controls": [
            "Data retention",
            "Data disposal",
            "Destruction verification",
        ],
        "risk_level": "medium",
    },
}

# ---------------------------------------------------------------------------
# Trust Service Criteria - Processing Integrity
# ---------------------------------------------------------------------------

_PI1_PROCESSING_INTEGRITY: Dict[str, CriterionDefinition] = {
    "PI1.1": {
        "id": "PI1.1",
        "description": "The entity obtains or generates, uses, and communicates relevant, quality information regarding the objectives related to processing.",
        "category": "processing_integrity",
        "subcategory": "data_quality",
        "control_points": [
            "Input validation exists",
            "Data quality controls exist",
            "Processing accuracy is verified",
            "Output validation exists",
        ],
        "evidence_requirements": [
            "Data quality policy",
            "Input validation documentation",
            "Processing controls documentation",
            "Output validation procedures",
            "Data quality metrics",
        ],
        "common_controls": [
            "Input validation",
            "Processing controls",
            "Data quality",
        ],
        "risk_level": "high",
    },
    "PI1.2": {
        "id": "PI1.2",
        "description": "The entity implements policies and procedures over system processing to result in products, services, and reporting to meet the entity's objectives.",
        "category": "processing_integrity",
        "subcategory": "processing_controls",
        "control_points": [
            "Processing policies exist",
            "Processing procedures are documented",
            "Processing is monitored",
            "Processing errors are identified and corrected",
        ],
        "evidence_requirements": [
            "Processing policies",
            "Processing procedures",
            "Processing monitoring documentation",
            "Error handling procedures",
            "Processing reconciliation records",
        ],
        "common_controls": [
            "Processing procedures",
            "Error handling",
            "Reconciliation",
        ],
        "risk_level": "high",
    },
}

# ---------------------------------------------------------------------------
# Trust Service Criteria - Privacy
# ---------------------------------------------------------------------------

_PRIVACY_CRITERIA: Dict[str, CriterionDefinition] = {
    "P1.0": {
        "id": "P1.0",
        "description": "The entity provides notice to data subjects about its privacy practices.",
        "category": "privacy",
        "subcategory": "notice",
        "control_points": [
            "Privacy notice exists",
            "Notice is accessible to data subjects",
            "Notice describes data practices",
            "Notice is updated as needed",
        ],
        "evidence_requirements": [
            "Privacy policy/notice",
            "Website privacy statement",
            "Notice accessibility documentation",
            "Notice update history",
        ],
        "common_controls": [
            "Privacy notice",
            "Privacy policy",
        ],
        "risk_level": "high",
    },
    "P2.0": {
        "id": "P2.0",
        "description": "The entity communicates choices available to data subjects.",
        "category": "privacy",
        "subcategory": "choice_consent",
        "control_points": [
            "Choice mechanisms exist",
            "Consent is obtained where required",
            "Choices are documented",
            "Choice changes are processed",
        ],
        "evidence_requirements": [
            "Consent management procedures",
            "Consent records",
            "Choice/preference center documentation",
            "Opt-out mechanisms",
        ],
        "common_controls": [
            "Consent management",
            "Preference center",
            "Opt-out",
        ],
        "risk_level": "high",
    },
    "P3.0": {
        "id": "P3.0",
        "description": "The entity collects personal information only for the purposes identified in the notice.",
        "category": "privacy",
        "subcategory": "collection",
        "control_points": [
            "Collection is limited to stated purposes",
            "Collection methods are documented",
            "Data minimization is practiced",
            "Collection is lawful",
        ],
        "evidence_requirements": [
            "Data collection policy",
            "Purpose limitation documentation",
            "Data minimization practices",
            "Collection methods documentation",
        ],
        "common_controls": [
            "Purpose limitation",
            "Data minimization",
            "Collection controls",
        ],
        "risk_level": "high",
    },
    "P4.0": {
        "id": "P4.0",
        "description": "The entity limits the use of personal information to the purposes identified in the notice.",
        "category": "privacy",
        "subcategory": "use_retention",
        "control_points": [
            "Use is limited to stated purposes",
            "Secondary use requires consent",
            "Use is monitored",
            "Use violations are addressed",
        ],
        "evidence_requirements": [
            "Use limitation policy",
            "Secondary use procedures",
            "Use monitoring documentation",
            "Violation handling procedures",
        ],
        "common_controls": [
            "Use limitation",
            "Purpose enforcement",
        ],
        "risk_level": "high",
    },
    "P5.0": {
        "id": "P5.0",
        "description": "The entity provides data subjects with access to their personal information for review and correction.",
        "category": "privacy",
        "subcategory": "access",
        "control_points": [
            "Access request process exists",
            "Data subjects can review their data",
            "Correction mechanisms exist",
            "Access requests are timely processed",
        ],
        "evidence_requirements": [
            "Data subject access request procedures",
            "DSAR fulfillment documentation",
            "Access portal/mechanism documentation",
            "Response time metrics",
        ],
        "common_controls": [
            "DSAR process",
            "Access portal",
            "Correction mechanism",
        ],
        "risk_level": "high",
    },
    "P6.0": {
        "id": "P6.0",
        "description": "The entity discloses personal information to third parties only for the purposes identified in the notice.",
        "category": "privacy",
        "subcategory": "disclosure",
        "control_points": [
            "Disclosures are limited to stated purposes",
            "Third party agreements exist",
            "Disclosures are documented",
            "Third party use is monitored",
        ],
        "evidence_requirements": [
            "Disclosure policy",
            "Third party agreements/DPAs",
            "Disclosure records",
            "Third party monitoring documentation",
        ],
        "common_controls": [
            "Disclosure controls",
            "DPAs",
            "Third party management",
        ],
        "risk_level": "high",
    },
    "P7.0": {
        "id": "P7.0",
        "description": "The entity protects personal information against unauthorized access.",
        "category": "privacy",
        "subcategory": "security_for_privacy",
        "control_points": [
            "Personal information is protected",
            "Access controls exist for personal data",
            "Encryption is used where appropriate",
            "Security measures are documented",
        ],
        "evidence_requirements": [
            "Personal data protection policy",
            "Access controls for personal data",
            "Encryption documentation",
            "Security measures documentation",
        ],
        "common_controls": [
            "Access controls",
            "Encryption",
            "Security measures",
        ],
        "risk_level": "critical",
    },
    "P8.0": {
        "id": "P8.0",
        "description": "The entity monitors compliance with its privacy policies and procedures.",
        "category": "privacy",
        "subcategory": "quality",
        "control_points": [
            "Privacy compliance monitoring exists",
            "Privacy assessments are performed",
            "Privacy incidents are tracked",
            "Remediation is implemented",
        ],
        "evidence_requirements": [
            "Privacy compliance monitoring program",
            "Privacy impact assessments",
            "Privacy incident records",
            "Remediation tracking",
        ],
        "common_controls": [
            "Privacy monitoring",
            "PIAs",
            "Incident tracking",
        ],
        "risk_level": "high",
    },
}

# ---------------------------------------------------------------------------
# Complete TSC Criteria Dictionary
# ---------------------------------------------------------------------------

TSC_CRITERIA: Dict[str, CriterionDefinition] = {
    # Security - Control Environment (CC1)
    **_CC1_CONTROL_ENVIRONMENT,
    # Security - Communication (CC2)
    **_CC2_COMMUNICATION,
    # Security - Risk Assessment (CC3)
    **_CC3_RISK_ASSESSMENT,
    # Security - Monitoring (CC4)
    **_CC4_MONITORING,
    # Security - Control Activities (CC5)
    **_CC5_CONTROL_ACTIVITIES,
    # Security - Logical and Physical Access (CC6)
    **_CC6_LOGICAL_AND_PHYSICAL_ACCESS,
    # Security - System Operations (CC7)
    **_CC7_SYSTEM_OPERATIONS,
    # Security - Change Management (CC8)
    **_CC8_CHANGE_MANAGEMENT,
    # Security - Risk Mitigation (CC9)
    **_CC9_RISK_MITIGATION,
    # Availability (A1)
    **_A1_AVAILABILITY,
    # Confidentiality (C1)
    **_C1_CONFIDENTIALITY,
    # Processing Integrity (PI1)
    **_PI1_PROCESSING_INTEGRITY,
    # Privacy (P1-P8)
    **_PRIVACY_CRITERIA,
}


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def get_criterion(criterion_id: str) -> Optional[CriterionDefinition]:
    """Get a specific criterion by ID.

    Args:
        criterion_id: The criterion identifier (e.g., "CC6.1", "A1.2").

    Returns:
        The criterion definition or None if not found.

    Example:
        >>> criterion = get_criterion("CC6.1")
        >>> criterion["category"]
        'security'
    """
    return TSC_CRITERIA.get(criterion_id.upper())


def get_criteria_by_category(category: str) -> Dict[str, CriterionDefinition]:
    """Get all criteria for a specific category.

    Args:
        category: The category name (security, availability, confidentiality,
            processing_integrity, privacy).

    Returns:
        Dictionary of criteria for the specified category.

    Example:
        >>> security = get_criteria_by_category("security")
        >>> len(security)
        33
    """
    return {
        k: v for k, v in TSC_CRITERIA.items()
        if v["category"] == category.lower()
    }


def get_criteria_by_subcategory(subcategory: str) -> Dict[str, CriterionDefinition]:
    """Get all criteria for a specific subcategory.

    Args:
        subcategory: The subcategory name (e.g., "access_controls",
            "change_management").

    Returns:
        Dictionary of criteria for the specified subcategory.
    """
    return {
        k: v for k, v in TSC_CRITERIA.items()
        if v["subcategory"] == subcategory.lower()
    }


def get_criteria_by_risk_level(risk_level: str) -> Dict[str, CriterionDefinition]:
    """Get all criteria at a specific risk level.

    Args:
        risk_level: The risk level (critical, high, medium, low).

    Returns:
        Dictionary of criteria at the specified risk level.
    """
    return {
        k: v for k, v in TSC_CRITERIA.items()
        if v["risk_level"] == risk_level.lower()
    }


def get_all_criterion_ids() -> List[str]:
    """Get a sorted list of all criterion IDs.

    Returns:
        Sorted list of criterion IDs.
    """
    return sorted(TSC_CRITERIA.keys())


def get_category_criteria_count() -> Dict[str, int]:
    """Get the count of criteria per category.

    Returns:
        Dictionary mapping category names to criterion counts.
    """
    counts: Dict[str, int] = {}
    for criterion in TSC_CRITERIA.values():
        category = criterion["category"]
        counts[category] = counts.get(category, 0) + 1
    return counts


def get_security_criteria() -> Dict[str, CriterionDefinition]:
    """Get all Security (Common Criteria) criteria.

    Convenience function for the required SOC 2 category.

    Returns:
        Dictionary of all CC criteria (33 total).
    """
    return get_criteria_by_category("security")


__all__ = [
    "TSC_CRITERIA",
    "CATEGORY_WEIGHTS",
    "CriterionDefinition",
    "get_criterion",
    "get_criteria_by_category",
    "get_criteria_by_subcategory",
    "get_criteria_by_risk_level",
    "get_all_criterion_ids",
    "get_category_criteria_count",
    "get_security_criteria",
]
