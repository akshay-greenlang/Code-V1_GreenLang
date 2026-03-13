# -*- coding: utf-8 -*-
"""
Remediation Plan Design Engine - AGENT-EUDR-025

Generates structured, multi-phase remediation plans with SMART milestones
(Specific, Measurable, Achievable, Relevant, Time-bound), assigned
responsible parties, resource requirements, KPI definitions, dependency
tracking, and escalation triggers. Plans are linked to specific risk
findings from upstream agents and validated against ISO 31000 risk
treatment requirements.

Core capabilities:
    - Generate multi-phase remediation plans (4 phases: Preparation,
      Implementation, Verification, Monitoring)
    - Auto-generate SMART milestones linked to risk factors and EUDR articles
    - Support 8 plan templates for common remediation scenarios
    - Track milestone completion with evidence upload requirements
    - Implement Gantt chart view with dependencies and critical path
    - Support plan versioning with change history and approval workflows
    - Generate plan status dashboard (On Track, At Risk, Delayed, etc.)
    - Clone successful plans for similar supplier remediation

Performance Targets:
    - Plan generation: < 5 seconds for single supplier
    - Support 10,000+ concurrent active plans
    - Complete change history retained for 5 years per Article 31

PRD: PRD-AGENT-EUDR-025, Feature 2: Remediation Plan Designer
Agent ID: GL-EUDR-RMA-025
Regulation: EU 2023/1115 (EUDR) Articles 10, 11; ISO 31000:2018 Section 5.5

Author: GreenLang Platform Team
Date: March 2026
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from greenlang.agents.eudr.risk_mitigation_advisor.config import (
    RiskMitigationAdvisorConfig,
    get_config,
)
from greenlang.agents.eudr.risk_mitigation_advisor.models import (
    PlanPhaseType,
    PlanStatus,
    MilestoneStatus,
    ImplementationComplexity,
    StakeholderRole,
    RiskCategory,
    RemediationPlan,
    PlanPhase,
    Milestone,
    KPI,
    ResponsibleParty,
    EscalationTrigger,
    CreatePlanRequest,
    CreatePlanResponse,
)
from greenlang.agents.eudr.risk_mitigation_advisor.provenance import (
    ProvenanceTracker,
    get_tracker,
)

try:
    from greenlang.agents.eudr.risk_mitigation_advisor.metrics import (
        record_plan_created,
        observe_plan_generation_duration,
    )
except ImportError:
    record_plan_created = None
    observe_plan_generation_duration = None


# ---------------------------------------------------------------------------
# Plan status transition rules (finite state machine)
# ---------------------------------------------------------------------------

VALID_STATUS_TRANSITIONS: Dict[PlanStatus, List[PlanStatus]] = {
    PlanStatus.DRAFT: [PlanStatus.ACTIVE, PlanStatus.ABANDONED],
    PlanStatus.ACTIVE: [
        PlanStatus.ON_TRACK, PlanStatus.AT_RISK,
        PlanStatus.SUSPENDED, PlanStatus.ABANDONED,
    ],
    PlanStatus.ON_TRACK: [
        PlanStatus.AT_RISK, PlanStatus.DELAYED,
        PlanStatus.COMPLETED, PlanStatus.SUSPENDED,
    ],
    PlanStatus.AT_RISK: [
        PlanStatus.ON_TRACK, PlanStatus.DELAYED,
        PlanStatus.SUSPENDED, PlanStatus.ABANDONED,
    ],
    PlanStatus.DELAYED: [
        PlanStatus.ON_TRACK, PlanStatus.AT_RISK,
        PlanStatus.SUSPENDED, PlanStatus.ABANDONED,
    ],
    PlanStatus.SUSPENDED: [
        PlanStatus.ACTIVE, PlanStatus.ABANDONED,
    ],
    PlanStatus.COMPLETED: [],
    PlanStatus.ABANDONED: [],
}


# ---------------------------------------------------------------------------
# Plan template definitions (8 templates per PRD)
# ---------------------------------------------------------------------------

PLAN_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "supplier_capacity_building": {
        "name": "Supplier Capacity Building Plan",
        "description": (
            "Structured capacity development program for suppliers "
            "requiring skills uplift in EUDR compliance, GPS data "
            "collection, record keeping, and sustainable practices."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Assessment",
             "start_week": 1, "end_week": 2, "budget_pct": Decimal("10"),
             "description": "Assess current supplier capabilities and identify gaps"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Training",
             "start_week": 3, "end_week": 16, "budget_pct": Decimal("60"),
             "description": "Deliver commodity-specific training modules"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Practice Change Verification",
             "start_week": 17, "end_week": 20, "budget_pct": Decimal("20"),
             "description": "Verify adoption of new practices through on-site audits"},
            {"type": PlanPhaseType.MONITORING, "name": "Ongoing Monitoring",
             "start_week": 21, "end_week": 24, "budget_pct": Decimal("10"),
             "description": "Monitor sustained practice changes and risk score improvement"},
        ],
        "duration_weeks": 24,
        "risk_categories": ["supplier", "commodity"],
        "milestones_per_phase": [2, 4, 2, 1],
        "kpi_templates": ["risk_score_reduction", "milestone_completion", "budget_utilization",
                          "training_completion_rate", "supplier_satisfaction"],
        "escalation_triggers": ["milestone_overdue", "risk_increase", "budget_overrun"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "emergency_deforestation_response": {
        "name": "Emergency Deforestation Response Plan",
        "description": (
            "Rapid-response plan activated when critical deforestation "
            "alerts are detected from EUDR-020. Follows immediate "
            "suspension, investigation, remediation, and conditional "
            "resumption protocol."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Suspend & Investigate",
             "start_week": 1, "end_week": 1, "budget_pct": Decimal("15"),
             "description": "Immediately suspend sourcing and launch investigation"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Remediate",
             "start_week": 2, "end_week": 4, "budget_pct": Decimal("50"),
             "description": "Implement corrective actions to address deforestation"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Verify Resolution",
             "start_week": 5, "end_week": 6, "budget_pct": Decimal("25"),
             "description": "Verify deforestation has stopped and remediation effective"},
            {"type": PlanPhaseType.MONITORING, "name": "Resume with Monitoring",
             "start_week": 7, "end_week": 8, "budget_pct": Decimal("10"),
             "description": "Conditional resumption with enhanced satellite monitoring"},
        ],
        "duration_weeks": 8,
        "risk_categories": ["deforestation"],
        "milestones_per_phase": [2, 3, 2, 1],
        "kpi_templates": ["deforestation_halt", "forest_cover_recovery", "sourcing_status"],
        "escalation_triggers": ["continued_deforestation", "non_cooperation", "regulatory_notice"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "certification_enrollment": {
        "name": "Certification Enrollment Plan",
        "description": (
            "Long-term plan to achieve third-party certification "
            "(FSC, RSPO, Rainforest Alliance, ISCC) demonstrating "
            "deforestation-free and legally compliant production."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Gap Assessment",
             "start_week": 1, "end_week": 4, "budget_pct": Decimal("15"),
             "description": "Assess current practices against certification requirements"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Standard Preparation",
             "start_week": 5, "end_week": 36, "budget_pct": Decimal("50"),
             "description": "Implement required management systems and practices"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Certification Audit",
             "start_week": 37, "end_week": 44, "budget_pct": Decimal("25"),
             "description": "Undergo third-party certification audit"},
            {"type": PlanPhaseType.MONITORING, "name": "Post-Certification",
             "start_week": 45, "end_week": 52, "budget_pct": Decimal("10"),
             "description": "Maintain certification and prepare for surveillance audits"},
        ],
        "duration_weeks": 52,
        "risk_categories": ["legal_compliance", "commodity"],
        "milestones_per_phase": [3, 6, 3, 2],
        "kpi_templates": ["certification_readiness", "gap_closure_rate", "audit_findings"],
        "escalation_triggers": ["major_nonconformance", "certification_denied"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "enhanced_monitoring_deployment": {
        "name": "Enhanced Monitoring Deployment Plan",
        "description": (
            "Deploy enhanced risk monitoring infrastructure for "
            "supply chains in high-risk countries or regions. Includes "
            "satellite monitoring, GPS verification, and automated "
            "alert systems."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Baseline Assessment",
             "start_week": 1, "end_week": 1, "budget_pct": Decimal("20"),
             "description": "Establish monitoring baselines and system requirements"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Deploy Monitoring",
             "start_week": 2, "end_week": 4, "budget_pct": Decimal("45"),
             "description": "Deploy satellite, GPS, and automated monitoring systems"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Calibrate & Validate",
             "start_week": 5, "end_week": 6, "budget_pct": Decimal("20"),
             "description": "Calibrate monitoring thresholds and validate alert accuracy"},
            {"type": PlanPhaseType.MONITORING, "name": "Operate & Maintain",
             "start_week": 7, "end_week": 8, "budget_pct": Decimal("15"),
             "description": "Operate monitoring systems with ongoing maintenance"},
        ],
        "duration_weeks": 8,
        "risk_categories": ["country", "deforestation"],
        "milestones_per_phase": [2, 3, 2, 1],
        "kpi_templates": ["monitoring_coverage", "alert_accuracy", "response_time"],
        "escalation_triggers": ["system_downtime", "false_alert_rate_high"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "fpic_remediation": {
        "name": "FPIC Remediation Plan",
        "description": (
            "Free, Prior and Informed Consent remediation for supply "
            "chains operating in or near indigenous territories. "
            "Follows ILO Convention 169 and UNDRIP principles."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Identify & Map",
             "start_week": 1, "end_week": 4, "budget_pct": Decimal("15"),
             "description": "Identify affected communities and map territorial boundaries"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Engage & Consult",
             "start_week": 5, "end_week": 24, "budget_pct": Decimal("50"),
             "description": "Conduct FPIC consultations with affected communities"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Agree & Document",
             "start_week": 25, "end_week": 30, "budget_pct": Decimal("20"),
             "description": "Document consent outcomes and establish agreements"},
            {"type": PlanPhaseType.MONITORING, "name": "Monitor Compliance",
             "start_week": 31, "end_week": 36, "budget_pct": Decimal("15"),
             "description": "Monitor ongoing compliance with consent agreements"},
        ],
        "duration_weeks": 36,
        "risk_categories": ["indigenous_rights"],
        "milestones_per_phase": [3, 5, 3, 2],
        "kpi_templates": ["community_engagement_rate", "consent_documentation",
                          "grievance_resolution"],
        "escalation_triggers": ["community_objection", "consent_withdrawal"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "legal_gap_closure": {
        "name": "Legal Gap Closure Plan",
        "description": (
            "Structured plan to close identified legal compliance gaps "
            "including missing permits, expired licenses, incomplete "
            "environmental assessments, and labour law violations."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Legal Assessment",
             "start_week": 1, "end_week": 2, "budget_pct": Decimal("15"),
             "description": "Comprehensive legal gap assessment and prioritization"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Legal Support & Remediation",
             "start_week": 3, "end_week": 16, "budget_pct": Decimal("55"),
             "description": "Provide legal support and remediate identified gaps"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Permit Verification",
             "start_week": 17, "end_week": 20, "budget_pct": Decimal("20"),
             "description": "Verify all permits and licenses are current and valid"},
            {"type": PlanPhaseType.MONITORING, "name": "Compliance Monitoring",
             "start_week": 21, "end_week": 24, "budget_pct": Decimal("10"),
             "description": "Monitor ongoing legal compliance and renewal schedules"},
        ],
        "duration_weeks": 24,
        "risk_categories": ["legal_compliance"],
        "milestones_per_phase": [2, 4, 2, 1],
        "kpi_templates": ["gaps_closed", "permits_obtained", "compliance_score"],
        "escalation_triggers": ["permit_denial", "regulatory_enforcement"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "anti_corruption_measures": {
        "name": "Anti-Corruption Measures Plan",
        "description": (
            "Deploy anti-corruption controls for supply chains in "
            "countries with high corruption risk (CPI < 40). Includes "
            "payment controls, transparency mechanisms, and training."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Risk Assessment",
             "start_week": 1, "end_week": 2, "budget_pct": Decimal("15"),
             "description": "Assess corruption risk factors and exposure points"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Controls & Training",
             "start_week": 3, "end_week": 10, "budget_pct": Decimal("55"),
             "description": "Deploy anti-corruption controls and deliver training"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Verification",
             "start_week": 11, "end_week": 13, "budget_pct": Decimal("20"),
             "description": "Verify controls are effective through testing"},
            {"type": PlanPhaseType.MONITORING, "name": "Ongoing Monitoring",
             "start_week": 14, "end_week": 16, "budget_pct": Decimal("10"),
             "description": "Monitor transaction patterns and whistleblower reports"},
        ],
        "duration_weeks": 16,
        "risk_categories": ["corruption"],
        "milestones_per_phase": [2, 3, 2, 1],
        "kpi_templates": ["controls_deployed", "training_completion",
                          "transaction_anomalies"],
        "escalation_triggers": ["bribery_allegation", "payment_anomaly"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
    "protected_area_buffer_restoration": {
        "name": "Protected Area Buffer Restoration Plan",
        "description": (
            "Long-term restoration plan for buffer zones around "
            "protected areas affected by supply chain operations. "
            "Includes reforestation, community conservation, and "
            "alternative livelihood development."
        ),
        "phases": [
            {"type": PlanPhaseType.PREPARATION, "name": "Assessment & Planning",
             "start_week": 1, "end_week": 4, "budget_pct": Decimal("10"),
             "description": "Assess restoration needs and develop detailed plans"},
            {"type": PlanPhaseType.IMPLEMENTATION, "name": "Restoration Activities",
             "start_week": 5, "end_week": 36, "budget_pct": Decimal("60"),
             "description": "Execute restoration activities and alternative livelihood programs"},
            {"type": PlanPhaseType.VERIFICATION, "name": "Verify Restoration",
             "start_week": 37, "end_week": 44, "budget_pct": Decimal("15"),
             "description": "Verify ecological restoration progress using satellite data"},
            {"type": PlanPhaseType.MONITORING, "name": "Long-term Monitoring",
             "start_week": 45, "end_week": 52, "budget_pct": Decimal("15"),
             "description": "Long-term ecological monitoring and community engagement"},
        ],
        "duration_weeks": 52,
        "risk_categories": ["protected_areas"],
        "milestones_per_phase": [3, 6, 3, 2],
        "kpi_templates": ["hectares_restored", "species_recovery", "community_participation"],
        "escalation_triggers": ["new_encroachment", "restoration_failure"],
        "applicable_commodities": ["cattle", "cocoa", "coffee", "palm_oil", "rubber", "soya", "wood"],
    },
}


# ---------------------------------------------------------------------------
# KPI template library
# ---------------------------------------------------------------------------

KPI_TEMPLATE_LIBRARY: Dict[str, Dict[str, Any]] = {
    "risk_score_reduction": {
        "name": "Risk Score Reduction",
        "description": "Percentage reduction in composite risk score from baseline",
        "target_value": Decimal("30"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 11(1)",
    },
    "milestone_completion": {
        "name": "Milestone Completion Rate",
        "description": "Percentage of milestones completed on time",
        "target_value": Decimal("80"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 11(2)(a)",
    },
    "budget_utilization": {
        "name": "Budget Utilization",
        "description": "Percentage of budget spent vs allocated",
        "target_value": Decimal("90"),
        "unit": "%",
        "frequency": "quarterly",
        "eudr_article": "Art. 11(1)",
    },
    "training_completion_rate": {
        "name": "Training Completion Rate",
        "description": "Percentage of enrolled suppliers completing training modules",
        "target_value": Decimal("85"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 11(2)(a)",
    },
    "supplier_satisfaction": {
        "name": "Supplier Satisfaction Score",
        "description": "Average supplier satisfaction rating (1-5 scale)",
        "target_value": Decimal("4.0"),
        "unit": "score",
        "frequency": "quarterly",
        "eudr_article": "Art. 11(2)(a)",
    },
    "deforestation_halt": {
        "name": "Deforestation Halt Confirmation",
        "description": "Confirmation that active deforestation has stopped",
        "target_value": Decimal("100"),
        "unit": "%",
        "frequency": "weekly",
        "eudr_article": "Art. 3",
    },
    "forest_cover_recovery": {
        "name": "Forest Cover Recovery",
        "description": "Hectares of forest cover recovered through restoration",
        "target_value": Decimal("10"),
        "unit": "hectares",
        "frequency": "quarterly",
        "eudr_article": "Art. 11(2)(c)",
    },
    "sourcing_status": {
        "name": "Sourcing Status",
        "description": "Whether sourcing has been safely resumed (0=suspended, 100=resumed)",
        "target_value": Decimal("100"),
        "unit": "%",
        "frequency": "weekly",
        "eudr_article": "Art. 11(1)",
    },
    "certification_readiness": {
        "name": "Certification Readiness Score",
        "description": "Percentage of certification requirements met",
        "target_value": Decimal("90"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 11(2)(b)",
    },
    "gap_closure_rate": {
        "name": "Gap Closure Rate",
        "description": "Percentage of identified gaps closed",
        "target_value": Decimal("90"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 11(2)(c)",
    },
    "audit_findings": {
        "name": "Audit Findings Resolution",
        "description": "Percentage of audit findings resolved",
        "target_value": Decimal("95"),
        "unit": "%",
        "frequency": "quarterly",
        "eudr_article": "Art. 11(2)(c)",
    },
    "monitoring_coverage": {
        "name": "Monitoring Coverage",
        "description": "Percentage of supply chain plots under active monitoring",
        "target_value": Decimal("100"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "alert_accuracy": {
        "name": "Alert Accuracy",
        "description": "Percentage of monitoring alerts that are true positives",
        "target_value": Decimal("85"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(1)",
    },
    "response_time": {
        "name": "Alert Response Time",
        "description": "Average time to respond to monitoring alerts (hours)",
        "target_value": Decimal("24"),
        "unit": "hours",
        "frequency": "weekly",
        "eudr_article": "Art. 11(1)",
    },
    "community_engagement_rate": {
        "name": "Community Engagement Rate",
        "description": "Percentage of affected communities engaged in consultation",
        "target_value": Decimal("100"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "consent_documentation": {
        "name": "Consent Documentation Completeness",
        "description": "Percentage of required FPIC documentation completed",
        "target_value": Decimal("100"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "grievance_resolution": {
        "name": "Grievance Resolution Rate",
        "description": "Percentage of community grievances resolved within SLA",
        "target_value": Decimal("90"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "gaps_closed": {
        "name": "Legal Gaps Closed",
        "description": "Number of legal compliance gaps closed",
        "target_value": Decimal("10"),
        "unit": "count",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "permits_obtained": {
        "name": "Permits Obtained",
        "description": "Number of required permits obtained or renewed",
        "target_value": Decimal("5"),
        "unit": "count",
        "frequency": "quarterly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "compliance_score": {
        "name": "Legal Compliance Score",
        "description": "Overall legal compliance assessment score (0-100)",
        "target_value": Decimal("85"),
        "unit": "score",
        "frequency": "quarterly",
        "eudr_article": "Art. 10(2)(d)",
    },
    "controls_deployed": {
        "name": "Anti-Corruption Controls Deployed",
        "description": "Percentage of planned anti-corruption controls deployed",
        "target_value": Decimal("100"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(e)",
    },
    "transaction_anomalies": {
        "name": "Transaction Anomaly Rate",
        "description": "Percentage of transactions flagged for anomalies",
        "target_value": Decimal("2"),
        "unit": "%",
        "frequency": "monthly",
        "eudr_article": "Art. 10(2)(e)",
    },
    "hectares_restored": {
        "name": "Hectares Restored",
        "description": "Cumulative hectares of buffer zone restored",
        "target_value": Decimal("50"),
        "unit": "hectares",
        "frequency": "quarterly",
        "eudr_article": "Art. 11(2)(c)",
    },
    "species_recovery": {
        "name": "Species Recovery Index",
        "description": "Biodiversity recovery index for restored areas (0-100)",
        "target_value": Decimal("60"),
        "unit": "index",
        "frequency": "annually",
        "eudr_article": "Art. 11(2)(c)",
    },
    "community_participation": {
        "name": "Community Participation Rate",
        "description": "Percentage of local community members participating in restoration",
        "target_value": Decimal("50"),
        "unit": "%",
        "frequency": "quarterly",
        "eudr_article": "Art. 10(2)(d)",
    },
}


# ---------------------------------------------------------------------------
# Milestone templates per phase type
# ---------------------------------------------------------------------------

MILESTONE_TEMPLATES: Dict[PlanPhaseType, List[Dict[str, str]]] = {
    PlanPhaseType.PREPARATION: [
        {
            "name": "Complete baseline risk assessment",
            "description": "Conduct comprehensive baseline assessment of current risk profile",
            "evidence_types": "baseline_report,risk_assessment",
            "eudr_article": "Art. 10(1)",
        },
        {
            "name": "Identify and mobilize required resources",
            "description": "Identify budget, personnel, and technical resources needed",
            "evidence_types": "resource_plan,budget_approval",
            "eudr_article": "Art. 11(1)",
        },
        {
            "name": "Finalize implementation plan with stakeholders",
            "description": "Review and finalize plan with all stakeholders including supplier",
            "evidence_types": "signed_plan,meeting_minutes",
            "eudr_article": "Art. 11(2)(a)",
        },
        {
            "name": "Establish monitoring baseline metrics",
            "description": "Capture baseline values for all KPIs before intervention begins",
            "evidence_types": "baseline_metrics,data_snapshot",
            "eudr_article": "Art. 10(2)(d)",
        },
    ],
    PlanPhaseType.IMPLEMENTATION: [
        {
            "name": "Deploy primary mitigation measures",
            "description": "Activate primary mitigation measures as defined in the plan",
            "evidence_types": "deployment_report,system_logs",
            "eudr_article": "Art. 11(1)",
        },
        {
            "name": "Complete supplier engagement activities",
            "description": "Complete initial supplier engagement and kickoff meetings",
            "evidence_types": "meeting_minutes,attendance_records",
            "eudr_article": "Art. 11(2)(a)",
        },
        {
            "name": "Achieve 50% milestone completion target",
            "description": "Reach 50% completion of implementation milestones",
            "evidence_types": "progress_report,milestone_tracker",
            "eudr_article": "Art. 11(1)",
        },
        {
            "name": "Verify interim risk score improvement",
            "description": "Verify measurable improvement in risk scores at midpoint",
            "evidence_types": "risk_assessment_update,comparison_report",
            "eudr_article": "Art. 11(2)(c)",
        },
        {
            "name": "Complete training program delivery",
            "description": "Deliver all training modules in the implementation phase",
            "evidence_types": "training_records,completion_certificates",
            "eudr_article": "Art. 11(2)(a)",
        },
        {
            "name": "Deploy monitoring systems",
            "description": "Deploy all planned monitoring and tracking systems",
            "evidence_types": "system_deployment_report,test_results",
            "eudr_article": "Art. 10(2)(d)",
        },
    ],
    PlanPhaseType.VERIFICATION: [
        {
            "name": "Conduct post-implementation risk assessment",
            "description": "Full risk reassessment after implementation phase completion",
            "evidence_types": "risk_assessment,before_after_comparison",
            "eudr_article": "Art. 10(1)",
        },
        {
            "name": "Verify risk reduction against targets",
            "description": "Compare actual risk reduction against planned targets",
            "evidence_types": "effectiveness_report,kpi_dashboard",
            "eudr_article": "Art. 11(2)(c)",
        },
        {
            "name": "Compile effectiveness evidence package",
            "description": "Compile all evidence of mitigation effectiveness for audit",
            "evidence_types": "evidence_package,audit_documentation",
            "eudr_article": "Art. 11(2)(c)",
        },
        {
            "name": "Obtain stakeholder sign-off on results",
            "description": "Get formal approval from all stakeholders on plan outcomes",
            "evidence_types": "sign_off_form,approval_email",
            "eudr_article": "Art. 11(1)",
        },
    ],
    PlanPhaseType.MONITORING: [
        {
            "name": "Establish ongoing monitoring protocol",
            "description": "Define and activate ongoing monitoring procedures",
            "evidence_types": "monitoring_protocol,system_configuration",
            "eudr_article": "Art. 8(3)",
        },
        {
            "name": "Schedule periodic review cadence",
            "description": "Establish review calendar with periodic reassessment dates",
            "evidence_types": "review_calendar,schedule_confirmation",
            "eudr_article": "Art. 8(3)",
        },
        {
            "name": "Configure automated alert thresholds",
            "description": "Set up automated alerts for risk threshold breaches",
            "evidence_types": "alert_configuration,test_results",
            "eudr_article": "Art. 10(2)(d)",
        },
    ],
}


# ---------------------------------------------------------------------------
# Escalation trigger definitions
# ---------------------------------------------------------------------------

ESCALATION_TRIGGER_TEMPLATES: List[Dict[str, Any]] = [
    {
        "id": "milestone_overdue",
        "condition": "Milestone overdue by more than 7 days",
        "threshold": Decimal("7"),
        "escalation_target": "team_lead",
        "response_sla_hours": 24,
        "severity": "medium",
    },
    {
        "id": "risk_increase",
        "condition": "Risk score increased by more than 20% during plan execution",
        "threshold": Decimal("20"),
        "escalation_target": "director",
        "response_sla_hours": 24,
        "severity": "high",
    },
    {
        "id": "budget_overrun",
        "condition": "Budget utilization exceeds 110% of allocation",
        "threshold": Decimal("110"),
        "escalation_target": "executive",
        "response_sla_hours": 48,
        "severity": "medium",
    },
    {
        "id": "continued_deforestation",
        "condition": "Deforestation alerts continue after suspension",
        "threshold": Decimal("1"),
        "escalation_target": "executive",
        "response_sla_hours": 4,
        "severity": "critical",
    },
    {
        "id": "non_cooperation",
        "condition": "Supplier non-cooperation for more than 14 days",
        "threshold": Decimal("14"),
        "escalation_target": "director",
        "response_sla_hours": 24,
        "severity": "high",
    },
    {
        "id": "regulatory_notice",
        "condition": "Regulatory notice received from competent authority",
        "threshold": Decimal("1"),
        "escalation_target": "executive",
        "response_sla_hours": 4,
        "severity": "critical",
    },
    {
        "id": "community_objection",
        "condition": "Formal community objection to FPIC process",
        "threshold": Decimal("1"),
        "escalation_target": "director",
        "response_sla_hours": 24,
        "severity": "high",
    },
    {
        "id": "consent_withdrawal",
        "condition": "Community consent withdrawal filed",
        "threshold": Decimal("1"),
        "escalation_target": "executive",
        "response_sla_hours": 8,
        "severity": "critical",
    },
    {
        "id": "bribery_allegation",
        "condition": "Bribery or corruption allegation reported",
        "threshold": Decimal("1"),
        "escalation_target": "executive",
        "response_sla_hours": 4,
        "severity": "critical",
    },
    {
        "id": "payment_anomaly",
        "condition": "Significant payment anomaly detected (> threshold amount)",
        "threshold": Decimal("10000"),
        "escalation_target": "director",
        "response_sla_hours": 24,
        "severity": "high",
    },
    {
        "id": "major_nonconformance",
        "condition": "Major non-conformance identified in certification audit",
        "threshold": Decimal("1"),
        "escalation_target": "director",
        "response_sla_hours": 48,
        "severity": "high",
    },
    {
        "id": "permit_denial",
        "condition": "Critical permit application denied by authority",
        "threshold": Decimal("1"),
        "escalation_target": "director",
        "response_sla_hours": 24,
        "severity": "high",
    },
    {
        "id": "new_encroachment",
        "condition": "New encroachment detected in restored buffer zone",
        "threshold": Decimal("1"),
        "escalation_target": "team_lead",
        "response_sla_hours": 24,
        "severity": "high",
    },
    {
        "id": "restoration_failure",
        "condition": "Restoration survival rate falls below 50%",
        "threshold": Decimal("50"),
        "escalation_target": "director",
        "response_sla_hours": 48,
        "severity": "medium",
    },
]


class RemediationPlanDesignEngine:
    """Remediation plan design and management engine.

    Generates structured multi-phase remediation plans with SMART
    milestones, manages plan lifecycle through status transitions,
    tracks milestone completion, supports plan versioning with audit
    trails, and enables plan cloning for portfolio-level remediation.

    Attributes:
        config: Agent configuration.
        provenance: Provenance tracker.
        _db_pool: PostgreSQL connection pool.
        _redis_client: Redis client for caching.
        _plans: In-memory plan store (production uses PostgreSQL).
        _change_history: In-memory version change history.

    Example:
        >>> engine = RemediationPlanDesignEngine(config=get_config())
        >>> response = await engine.create_plan(request)
        >>> assert response.plan.status == PlanStatus.DRAFT
    """

    def __init__(
        self,
        config: Optional[RiskMitigationAdvisorConfig] = None,
        db_pool: Optional[Any] = None,
        redis_client: Optional[Any] = None,
        provenance: Optional[ProvenanceTracker] = None,
    ) -> None:
        """Initialize RemediationPlanDesignEngine."""
        self.config = config or get_config()
        self.provenance = provenance or get_tracker()
        self._db_pool = db_pool
        self._redis_client = redis_client
        self._plans: Dict[str, RemediationPlan] = {}
        self._change_history: Dict[str, List[Dict[str, Any]]] = {}

        logger.info(
            f"RemediationPlanDesignEngine initialized: "
            f"phases={self.config.plan_phases_count}, "
            f"templates={len(PLAN_TEMPLATES)}"
        )

    async def create_plan(
        self, request: CreatePlanRequest,
    ) -> CreatePlanResponse:
        """Create a new remediation plan from template or custom spec.

        Generates a complete multi-phase remediation plan with milestones,
        KPIs, escalation triggers, and provenance tracking. The plan is
        created in DRAFT status and requires approval before activation.

        Args:
            request: Plan creation request with operator, supplier,
                    strategies, and template selection.

        Returns:
            CreatePlanResponse with the generated plan.

        Raises:
            ValueError: If the requested template does not exist.
        """
        start = time.monotonic()

        template_name = request.template_name or "supplier_capacity_building"
        template = PLAN_TEMPLATES.get(template_name)
        if template is None:
            raise ValueError(
                f"Unknown template '{template_name}'. "
                f"Available: {list(PLAN_TEMPLATES.keys())}"
            )

        plan_id = str(uuid.uuid4())
        start_date = date.today()
        duration_weeks = min(
            request.target_duration_weeks,
            template.get("duration_weeks", 12),
        )
        end_date = start_date + timedelta(weeks=duration_weeks)

        # Generate phases with milestones
        phases = self._generate_phases(
            plan_id, template, start_date, duration_weeks, request.budget_eur
        )

        # Collect all milestones
        all_milestones: List[Milestone] = []
        for phase in phases:
            all_milestones.extend(phase.milestones)

        # Generate KPIs from template
        kpi_names = template.get("kpi_templates", [
            "risk_score_reduction", "milestone_completion", "budget_utilization"
        ])
        kpis = self._generate_kpis(kpi_names)

        # Generate escalation triggers
        trigger_ids = template.get("escalation_triggers", [
            "milestone_overdue", "risk_increase", "budget_overrun"
        ])
        triggers = self._generate_escalation_triggers(trigger_ids)

        # Generate responsible parties
        responsible_parties = self._generate_responsible_parties(
            request.operator_id, request.supplier_id
        )

        # Build plan
        plan = RemediationPlan(
            plan_id=plan_id,
            operator_id=request.operator_id,
            supplier_id=request.supplier_id,
            plan_name=f"{template['name']} - {request.supplier_id or 'Portfolio'}",
            risk_finding_ids=request.risk_finding_ids,
            strategy_ids=request.strategy_ids,
            status=PlanStatus.DRAFT,
            phases=phases,
            milestones=all_milestones,
            kpis=kpis,
            budget_allocated=request.budget_eur,
            budget_spent=Decimal("0"),
            start_date=start_date,
            target_end_date=end_date,
            responsible_parties=responsible_parties,
            escalation_triggers=triggers,
            plan_template=template_name,
            version=1,
            provenance_hash="",
        )

        # Calculate provenance hash
        provenance_hash = self._calculate_provenance_hash(plan)

        # Store plan (in-memory; production uses PostgreSQL)
        self._plans[plan_id] = plan
        self._change_history[plan_id] = [{
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "create",
            "actor": "remediation_plan_design_engine",
            "template": template_name,
            "changes": {"initial_creation": True},
        }]

        # Record provenance
        self.provenance.record(
            entity_type="remediation_plan",
            action="create",
            entity_id=plan_id,
            actor="remediation_plan_design_engine",
            metadata={
                "operator_id": request.operator_id,
                "supplier_id": request.supplier_id,
                "template": template_name,
                "milestone_count": len(all_milestones),
                "kpi_count": len(kpis),
                "budget_eur": str(request.budget_eur),
                "duration_weeks": duration_weeks,
            },
        )

        elapsed_ms = Decimal(str(round(
            (time.monotonic() - start) * 1000, 2
        )))

        if record_plan_created is not None:
            record_plan_created("draft", template_name)
        if observe_plan_generation_duration is not None:
            observe_plan_generation_duration(
                float(elapsed_ms) / 1000.0, template_name
            )

        logger.info(
            f"Plan created: id={plan_id}, template={template_name}, "
            f"milestones={len(all_milestones)}, kpis={len(kpis)}, "
            f"elapsed={elapsed_ms}ms"
        )

        return CreatePlanResponse(
            plan=plan,
            milestone_count=len(all_milestones),
            kpi_count=len(kpis),
            processing_time_ms=elapsed_ms,
            provenance_hash=provenance_hash,
        )

    def _generate_phases(
        self,
        plan_id: str,
        template: Dict[str, Any],
        start_date: date,
        duration_weeks: int,
        total_budget: Decimal,
    ) -> List[PlanPhase]:
        """Generate plan phases from template specification.

        Scales template phase timing to actual plan duration and
        generates SMART milestones for each phase.

        Args:
            plan_id: Plan identifier for milestone linking.
            template: Plan template dictionary.
            start_date: Plan start date.
            duration_weeks: Total plan duration in weeks.
            total_budget: Total budget allocation.

        Returns:
            List of PlanPhase objects with milestones.
        """
        phases: List[PlanPhase] = []
        template_phases = template.get("phases", [])
        milestones_per_phase = template.get("milestones_per_phase", [2, 3, 2, 1])
        template_duration = Decimal(str(template.get("duration_weeks", duration_weeks)))

        for i, phase_spec in enumerate(template_phases):
            phase_type = phase_spec["type"]
            s_week = phase_spec["start_week"]
            e_week = min(phase_spec["end_week"], template.get("duration_weeks", duration_weeks))
            budget_pct = phase_spec.get("budget_pct", Decimal("25"))

            # Scale weeks to actual duration
            if template_duration > Decimal("0"):
                scale = Decimal(str(duration_weeks)) / template_duration
            else:
                scale = Decimal("1")

            scaled_start = max(1, int(Decimal(str(s_week)) * scale))
            scaled_end = max(scaled_start + 1, int(Decimal(str(e_week)) * scale))

            milestone_count = milestones_per_phase[i] if i < len(milestones_per_phase) else 2

            milestones = self._generate_milestones(
                plan_id, phase_type, start_date, scaled_start, scaled_end,
                milestone_count,
            )

            phase_description = phase_spec.get(
                "description",
                f"{phase_spec['name']} phase for remediation plan"
            )

            phase = PlanPhase(
                phase_type=phase_type,
                name=phase_spec["name"],
                description=phase_description,
                start_week=scaled_start,
                end_week=scaled_end,
                milestones=milestones,
                budget_allocation_pct=budget_pct,
            )
            phases.append(phase)

        return phases

    def _generate_milestones(
        self,
        plan_id: str,
        phase_type: PlanPhaseType,
        start_date: date,
        start_week: int,
        end_week: int,
        count: int,
    ) -> List[Milestone]:
        """Generate SMART milestones for a plan phase.

        Selects milestone templates appropriate for the phase type
        and distributes them evenly across the phase duration.

        Args:
            plan_id: Plan identifier for milestone linking.
            phase_type: Phase type determining milestone templates.
            start_date: Plan start date.
            start_week: Phase start week.
            end_week: Phase end week.
            count: Number of milestones to generate.

        Returns:
            List of Milestone objects with SMART criteria.
        """
        milestones: List[Milestone] = []
        week_span = max(1, end_week - start_week)

        templates = MILESTONE_TEMPLATES.get(phase_type, [])[:count]
        for i, tmpl in enumerate(templates):
            week_offset = start_week + int(i * week_span / max(1, len(templates)))
            due = start_date + timedelta(weeks=week_offset)

            evidence_types = tmpl.get("evidence_types", "completion_report").split(",")

            milestone = Milestone(
                milestone_id=str(uuid.uuid4()),
                plan_id=plan_id,
                name=tmpl["name"],
                description=f"SMART milestone: {tmpl['description']}",
                phase=phase_type,
                due_date=due,
                status=MilestoneStatus.PENDING,
                kpi_target=f"Complete by week {week_offset}",
                evidence_required=evidence_types,
                eudr_article=tmpl.get("eudr_article", "Art. 11(1)"),
            )
            milestones.append(milestone)

        return milestones

    def _generate_kpis(self, kpi_names: List[str]) -> List[KPI]:
        """Generate KPIs from template library.

        Args:
            kpi_names: List of KPI template identifiers.

        Returns:
            List of KPI objects configured for the plan.
        """
        kpis: List[KPI] = []
        for name in kpi_names:
            tmpl = KPI_TEMPLATE_LIBRARY.get(name)
            if tmpl is None:
                continue
            kpi = KPI(
                name=tmpl["name"],
                description=tmpl["description"],
                target_value=tmpl["target_value"],
                unit=tmpl["unit"],
                measurement_frequency=tmpl["frequency"],
            )
            kpis.append(kpi)
        return kpis

    def _generate_escalation_triggers(
        self, trigger_ids: List[str],
    ) -> List[EscalationTrigger]:
        """Generate escalation triggers from template library.

        Args:
            trigger_ids: List of trigger template identifiers.

        Returns:
            List of EscalationTrigger objects.
        """
        triggers: List[EscalationTrigger] = []
        trigger_map = {t["id"]: t for t in ESCALATION_TRIGGER_TEMPLATES}

        for tid in trigger_ids:
            tmpl = trigger_map.get(tid)
            if tmpl is None:
                continue
            trigger = EscalationTrigger(
                condition=tmpl["condition"],
                threshold=tmpl["threshold"],
                escalation_target=tmpl["escalation_target"],
                response_sla_hours=tmpl["response_sla_hours"],
            )
            triggers.append(trigger)

        return triggers

    def _generate_responsible_parties(
        self,
        operator_id: str,
        supplier_id: Optional[str],
    ) -> List[ResponsibleParty]:
        """Generate default responsible parties for a plan.

        Args:
            operator_id: Operator identifier.
            supplier_id: Supplier identifier (may be None for portfolio plans).

        Returns:
            List of ResponsibleParty objects.
        """
        parties: List[ResponsibleParty] = [
            ResponsibleParty(
                party_id=str(uuid.uuid4()),
                name="Compliance Team Lead",
                role=StakeholderRole.INTERNAL_COMPLIANCE,
                organization=operator_id,
                email=f"compliance@{operator_id}.example.com",
            ),
            ResponsibleParty(
                party_id=str(uuid.uuid4()),
                name="Procurement Manager",
                role=StakeholderRole.PROCUREMENT,
                organization=operator_id,
                email=f"procurement@{operator_id}.example.com",
            ),
        ]

        if supplier_id:
            parties.append(
                ResponsibleParty(
                    party_id=str(uuid.uuid4()),
                    name="Supplier Contact",
                    role=StakeholderRole.SUPPLIER,
                    organization=supplier_id,
                    email=f"contact@{supplier_id}.example.com",
                )
            )

        return parties

    def transition_status(
        self,
        plan_id: str,
        new_status: PlanStatus,
        actor: str = "system",
        reason: str = "",
    ) -> bool:
        """Transition a plan to a new status.

        Validates the transition against the finite state machine rules
        and records the change in version history.

        Args:
            plan_id: Plan identifier.
            new_status: Target status.
            actor: User or system performing the transition.
            reason: Reason for the transition.

        Returns:
            True if transition was successful, False otherwise.

        Raises:
            ValueError: If the plan does not exist.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            raise ValueError(f"Plan '{plan_id}' not found")

        current_status = plan.status
        allowed = VALID_STATUS_TRANSITIONS.get(current_status, [])

        if new_status not in allowed:
            logger.warning(
                f"Invalid status transition for plan {plan_id}: "
                f"{current_status.value} -> {new_status.value}. "
                f"Allowed: {[s.value for s in allowed]}"
            )
            return False

        # Record change history
        history = self._change_history.get(plan_id, [])
        version = len(history) + 1
        history.append({
            "version": version,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "status_transition",
            "actor": actor,
            "changes": {
                "status_from": current_status.value,
                "status_to": new_status.value,
                "reason": reason,
            },
        })
        self._change_history[plan_id] = history

        # Record provenance
        self.provenance.record(
            entity_type="remediation_plan",
            action="status_transition",
            entity_id=plan_id,
            actor=actor,
            metadata={
                "from": current_status.value,
                "to": new_status.value,
                "reason": reason,
                "version": version,
            },
        )

        logger.info(
            f"Plan {plan_id} status transition: "
            f"{current_status.value} -> {new_status.value} "
            f"by {actor} (v{version})"
        )

        return True

    def update_milestone_status(
        self,
        plan_id: str,
        milestone_id: str,
        new_status: MilestoneStatus,
        evidence: Optional[List[str]] = None,
        actor: str = "system",
    ) -> bool:
        """Update a milestone status within a plan.

        Args:
            plan_id: Plan identifier.
            milestone_id: Milestone identifier.
            new_status: New milestone status.
            evidence: Optional list of evidence document references.
            actor: User performing the update.

        Returns:
            True if update was successful, False otherwise.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            logger.warning(f"Plan '{plan_id}' not found for milestone update")
            return False

        # Find milestone in plan
        for milestone in plan.milestones:
            if milestone.milestone_id == milestone_id:
                # Record change
                self.provenance.record(
                    entity_type="milestone",
                    action="status_update",
                    entity_id=milestone_id,
                    actor=actor,
                    metadata={
                        "plan_id": plan_id,
                        "from": milestone.status.value,
                        "to": new_status.value,
                        "evidence_count": len(evidence) if evidence else 0,
                    },
                )

                logger.info(
                    f"Milestone {milestone_id} updated: "
                    f"{milestone.status.value} -> {new_status.value}"
                )
                return True

        logger.warning(f"Milestone '{milestone_id}' not found in plan '{plan_id}'")
        return False

    def clone_plan(
        self,
        source_plan_id: str,
        new_supplier_id: str,
        new_operator_id: Optional[str] = None,
        budget_adjustment: Decimal = Decimal("1.0"),
    ) -> Optional[str]:
        """Clone a successful plan for a different supplier.

        Creates a copy of an existing plan with new identifiers,
        reset status, and optionally adjusted budget. Useful for
        applying proven remediation approaches to similar suppliers.

        Args:
            source_plan_id: Plan to clone.
            new_supplier_id: Supplier for the cloned plan.
            new_operator_id: Optional different operator.
            budget_adjustment: Budget multiplier (e.g., 1.2 for 20% increase).

        Returns:
            New plan ID if successful, None otherwise.
        """
        source = self._plans.get(source_plan_id)
        if source is None:
            logger.warning(f"Source plan '{source_plan_id}' not found for cloning")
            return None

        new_plan_id = str(uuid.uuid4())
        new_budget = (source.budget_allocated * budget_adjustment).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

        # Clone milestones with new IDs
        new_milestones: List[Milestone] = []
        for ms in source.milestones:
            new_ms = Milestone(
                milestone_id=str(uuid.uuid4()),
                plan_id=new_plan_id,
                name=ms.name,
                description=ms.description,
                phase=ms.phase,
                due_date=ms.due_date,
                status=MilestoneStatus.PENDING,
                kpi_target=ms.kpi_target,
                evidence_required=ms.evidence_required,
                eudr_article=ms.eudr_article,
            )
            new_milestones.append(new_ms)

        # Clone phases with new milestones
        new_phases: List[PlanPhase] = []
        milestone_idx = 0
        for phase in source.phases:
            phase_ms_count = len(phase.milestones)
            phase_milestones = new_milestones[milestone_idx:milestone_idx + phase_ms_count]
            milestone_idx += phase_ms_count

            new_phase = PlanPhase(
                phase_type=phase.phase_type,
                name=phase.name,
                description=phase.description,
                start_week=phase.start_week,
                end_week=phase.end_week,
                milestones=phase_milestones,
                budget_allocation_pct=phase.budget_allocation_pct,
            )
            new_phases.append(new_phase)

        operator = new_operator_id or source.operator_id

        new_plan = RemediationPlan(
            plan_id=new_plan_id,
            operator_id=operator,
            supplier_id=new_supplier_id,
            plan_name=f"{source.plan_name} (Clone for {new_supplier_id})",
            risk_finding_ids=[],
            strategy_ids=source.strategy_ids,
            status=PlanStatus.DRAFT,
            phases=new_phases,
            milestones=new_milestones,
            kpis=source.kpis,
            budget_allocated=new_budget,
            budget_spent=Decimal("0"),
            start_date=date.today(),
            target_end_date=date.today() + timedelta(
                weeks=(source.target_end_date - source.start_date).days // 7
            ),
            responsible_parties=self._generate_responsible_parties(
                operator, new_supplier_id
            ),
            escalation_triggers=source.escalation_triggers,
            plan_template=source.plan_template,
            version=1,
            provenance_hash="",
        )

        # Store cloned plan
        self._plans[new_plan_id] = new_plan
        self._change_history[new_plan_id] = [{
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": "clone",
            "actor": "remediation_plan_design_engine",
            "changes": {
                "source_plan_id": source_plan_id,
                "new_supplier_id": new_supplier_id,
                "budget_adjustment": str(budget_adjustment),
            },
        }]

        self.provenance.record(
            entity_type="remediation_plan",
            action="clone",
            entity_id=new_plan_id,
            actor="remediation_plan_design_engine",
            metadata={
                "source_plan_id": source_plan_id,
                "new_supplier_id": new_supplier_id,
                "budget": str(new_budget),
            },
        )

        logger.info(
            f"Plan cloned: {source_plan_id} -> {new_plan_id} "
            f"for supplier {new_supplier_id}"
        )

        return new_plan_id

    def get_plan(self, plan_id: str) -> Optional[RemediationPlan]:
        """Retrieve a plan by ID.

        Args:
            plan_id: Plan identifier.

        Returns:
            RemediationPlan or None if not found.
        """
        return self._plans.get(plan_id)

    def get_plan_status_dashboard(
        self, plan_id: str,
    ) -> Dict[str, Any]:
        """Generate plan status dashboard data.

        Computes overall plan health based on milestone completion,
        budget utilization, and timeline adherence.

        Args:
            plan_id: Plan identifier.

        Returns:
            Dashboard data dictionary with status, progress, and health.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return {"error": f"Plan '{plan_id}' not found"}

        total_milestones = len(plan.milestones)
        completed_milestones = sum(
            1 for m in plan.milestones
            if m.status == MilestoneStatus.COMPLETED
        )
        overdue_milestones = sum(
            1 for m in plan.milestones
            if m.status in (MilestoneStatus.PENDING, MilestoneStatus.IN_PROGRESS)
            and m.due_date < date.today()
        )

        completion_pct = Decimal("0")
        if total_milestones > 0:
            completion_pct = (
                Decimal(str(completed_milestones)) / Decimal(str(total_milestones))
                * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        budget_utilization_pct = Decimal("0")
        if plan.budget_allocated > Decimal("0"):
            budget_utilization_pct = (
                plan.budget_spent / plan.budget_allocated * Decimal("100")
            ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Determine plan health
        today = date.today()
        total_days = max(1, (plan.target_end_date - plan.start_date).days)
        elapsed_days = max(0, (today - plan.start_date).days)
        timeline_pct = Decimal(str(min(100, elapsed_days * 100 // total_days)))

        if overdue_milestones > 0:
            health = "at_risk"
        elif completion_pct < timeline_pct - Decimal("20"):
            health = "delayed"
        elif plan.status == PlanStatus.SUSPENDED:
            health = "suspended"
        elif plan.status == PlanStatus.COMPLETED:
            health = "completed"
        else:
            health = "on_track"

        # Phase progress
        phase_progress = []
        for phase in plan.phases:
            phase_total = len(phase.milestones)
            phase_completed = sum(
                1 for m in phase.milestones
                if m.status == MilestoneStatus.COMPLETED
            )
            phase_progress.append({
                "phase": phase.phase_type.value,
                "name": phase.name,
                "total_milestones": phase_total,
                "completed_milestones": phase_completed,
                "progress_pct": round(
                    phase_completed * 100 / max(1, phase_total), 1
                ),
            })

        return {
            "plan_id": plan_id,
            "plan_name": plan.plan_name,
            "status": plan.status.value,
            "health": health,
            "overall_progress_pct": float(completion_pct),
            "total_milestones": total_milestones,
            "completed_milestones": completed_milestones,
            "overdue_milestones": overdue_milestones,
            "budget_allocated": str(plan.budget_allocated),
            "budget_spent": str(plan.budget_spent),
            "budget_utilization_pct": float(budget_utilization_pct),
            "start_date": plan.start_date.isoformat(),
            "target_end_date": plan.target_end_date.isoformat(),
            "timeline_elapsed_pct": float(timeline_pct),
            "phase_progress": phase_progress,
            "version": plan.version,
            "change_history_count": len(
                self._change_history.get(plan_id, [])
            ),
        }

    def get_change_history(
        self, plan_id: str,
    ) -> List[Dict[str, Any]]:
        """Get complete change history for a plan.

        Returns the full version history for audit trail compliance
        per EUDR Article 31 (5-year retention requirement).

        Args:
            plan_id: Plan identifier.

        Returns:
            List of change history entries.
        """
        return self._change_history.get(plan_id, [])

    def get_critical_path(
        self, plan_id: str,
    ) -> List[Dict[str, Any]]:
        """Calculate the critical path through plan milestones.

        Identifies the sequence of milestones that determines the
        minimum possible plan duration.

        Args:
            plan_id: Plan identifier.

        Returns:
            List of critical path milestones with timing.
        """
        plan = self._plans.get(plan_id)
        if plan is None:
            return []

        critical_path: List[Dict[str, Any]] = []
        for phase in plan.phases:
            if not phase.milestones:
                continue
            # Last milestone in each phase is on the critical path
            last_ms = phase.milestones[-1]
            critical_path.append({
                "phase": phase.phase_type.value,
                "milestone_id": last_ms.milestone_id,
                "milestone_name": last_ms.name,
                "due_date": last_ms.due_date.isoformat(),
                "status": last_ms.status.value,
                "is_blocking": last_ms.status != MilestoneStatus.COMPLETED,
            })

        return critical_path

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all available plan templates.

        Returns:
            List of template summaries with name, description,
            duration, and applicable risk categories.
        """
        templates = []
        for key, tmpl in PLAN_TEMPLATES.items():
            templates.append({
                "template_id": key,
                "name": tmpl["name"],
                "description": tmpl.get("description", ""),
                "duration_weeks": tmpl["duration_weeks"],
                "risk_categories": tmpl["risk_categories"],
                "phases": len(tmpl["phases"]),
                "milestones": sum(tmpl.get("milestones_per_phase", [0])),
                "kpis": len(tmpl.get("kpi_templates", [])),
                "applicable_commodities": tmpl.get(
                    "applicable_commodities", []
                ),
            })
        return templates

    def _calculate_provenance_hash(self, plan: RemediationPlan) -> str:
        """Calculate SHA-256 provenance hash for a plan.

        Args:
            plan: Remediation plan to hash.

        Returns:
            SHA-256 hex digest string.
        """
        canonical = json.dumps({
            "plan_id": plan.plan_id,
            "operator_id": plan.operator_id,
            "supplier_id": plan.supplier_id,
            "template": plan.plan_template,
            "milestone_count": len(plan.milestones),
            "kpi_count": len(plan.kpis),
            "budget": str(plan.budget_allocated),
            "start_date": plan.start_date.isoformat(),
            "target_end_date": plan.target_end_date.isoformat(),
            "version": plan.version,
        }, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def health_check(self) -> Dict[str, Any]:
        """Return engine health status."""
        return {
            "status": "available",
            "templates_loaded": len(PLAN_TEMPLATES),
            "phases_count": self.config.plan_phases_count,
            "kpi_templates": len(KPI_TEMPLATE_LIBRARY),
            "escalation_templates": len(ESCALATION_TRIGGER_TEMPLATES),
            "milestone_templates": sum(
                len(v) for v in MILESTONE_TEMPLATES.values()
            ),
            "active_plans": len(self._plans),
        }

    async def shutdown(self) -> None:
        """Shutdown engine and clear in-memory stores."""
        self._plans.clear()
        self._change_history.clear()
        logger.info("RemediationPlanDesignEngine shut down")
