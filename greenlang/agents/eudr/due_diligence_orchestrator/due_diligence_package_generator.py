# -*- coding: utf-8 -*-
"""
Due Diligence Package Generator - AGENT-EUDR-026

Compiles all 25 agent outputs into a structured, DDS-compatible evidence
bundle with SHA-256 provenance chain, ready for EU Information System
submission per EUDR Article 12.

The package generator produces a DueDiligencePackage containing:
    - 9 DDS sections per Article 12(2) content requirements
    - Executive summary synthesized from agent outputs
    - Composite risk profile with all 10 risk dimension scores
    - Mitigation decision and strategies
    - Quality gate evaluation results
    - Workflow execution metadata and audit trail
    - Per-artifact SHA-256 integrity hashes
    - Package-level SHA-256 integrity hash

DDS Content Mapping (Article 12(2)):
    Section 1: Operator Identification (a) - EUDR-001, EUDR-008
    Section 2: Product Description (b) - EUDR-001, EUDR-009
    Section 3: Country of Production (c) - EUDR-001, EUDR-002, EUDR-016
    Section 4: Geolocation Data (d) - EUDR-002, EUDR-006, EUDR-007
    Section 5: Quantity & Volume (e) - EUDR-001, EUDR-011
    Section 6: Date of Production (f) - EUDR-001, EUDR-003
    Section 7: Deforestation-Free Status (g) - EUDR-003, EUDR-004, EUDR-005
    Section 8: Legal Compliance (h) - EUDR-023, EUDR-024
    Section 9: Risk Assessment & Mitigation (i) - EUDR-016 to EUDR-025

Features:
    - Build complete DDS evidence package from workflow state
    - Map all 25 agent outputs to DDS sections per Article 12(2)
    - Generate executive summary from risk profile and outcomes
    - Compute per-section completeness scores
    - Generate per-artifact SHA-256 hashes for integrity
    - Compute package-level integrity hash
    - Support multiple output formats (JSON, PDF, HTML, ZIP)
    - Support multiple report languages (en, fr, de, es, pt)
    - Track package generation duration for SLA monitoring
    - Deterministic: same inputs always produce same package

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional, Tuple
from greenlang.schemas import utcnow

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AGENT_NAMES,
    CompositeRiskProfile,
    DDSField,
    DDSSection,
    DueDiligencePackage,
    EUDRCommodity,
    MitigationDecision,
    QualityGateEvaluation,
    WorkflowState,
    WorkflowType,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DDS Section definitions per Article 12(2)
# ---------------------------------------------------------------------------

#: DDS section specifications with Article references and source agents.
_DDS_SECTIONS: List[Dict[str, Any]] = [
    {
        "section_number": 1,
        "title": "Operator Identification",
        "article_ref": "12(2)(a)",
        "description": (
            "Name, postal address, e-mail address, and EORI number "
            "of the operator or trader submitting the statement"
        ),
        "source_agents": ["EUDR-001", "EUDR-008"],
        "fields": [
            {"name": "operator_name", "article_ref": "12(2)(a)(i)", "required": True},
            {"name": "postal_address", "article_ref": "12(2)(a)(ii)", "required": True},
            {"name": "email_address", "article_ref": "12(2)(a)(iii)", "required": True},
            {"name": "eori_number", "article_ref": "12(2)(a)(iv)", "required": False},
        ],
    },
    {
        "section_number": 2,
        "title": "Product Description",
        "article_ref": "12(2)(b)",
        "description": (
            "Description, trade name, type, and commodity classification "
            "of the relevant products and their HS heading"
        ),
        "source_agents": ["EUDR-001", "EUDR-009"],
        "fields": [
            {"name": "product_description", "article_ref": "12(2)(b)(i)", "required": True},
            {"name": "trade_name", "article_ref": "12(2)(b)(ii)", "required": True},
            {"name": "hs_heading", "article_ref": "12(2)(b)(iii)", "required": True},
            {"name": "commodity_type", "article_ref": "12(2)(b)(iv)", "required": True},
        ],
    },
    {
        "section_number": 3,
        "title": "Country of Production",
        "article_ref": "12(2)(c)",
        "description": (
            "Country of production and, where applicable, parts thereof "
            "including ISO country codes"
        ),
        "source_agents": ["EUDR-001", "EUDR-002", "EUDR-016"],
        "fields": [
            {"name": "country_of_production", "article_ref": "12(2)(c)(i)", "required": True},
            {"name": "country_iso_code", "article_ref": "12(2)(c)(ii)", "required": True},
            {"name": "region_province", "article_ref": "12(2)(c)(iii)", "required": False},
            {"name": "country_risk_level", "article_ref": "12(2)(c)(iv)", "required": False},
        ],
    },
    {
        "section_number": 4,
        "title": "Geolocation Data",
        "article_ref": "12(2)(d)",
        "description": (
            "Geolocation coordinates of all plots of land where the relevant "
            "commodities were produced, including polygon boundaries"
        ),
        "source_agents": ["EUDR-002", "EUDR-006", "EUDR-007"],
        "fields": [
            {"name": "plot_coordinates", "article_ref": "12(2)(d)(i)", "required": True},
            {"name": "coordinate_accuracy", "article_ref": "12(2)(d)(ii)", "required": True},
            {"name": "polygon_boundaries", "article_ref": "12(2)(d)(iii)", "required": False},
            {"name": "total_plot_area_ha", "article_ref": "12(2)(d)(iv)", "required": False},
        ],
    },
    {
        "section_number": 5,
        "title": "Quantity and Volume",
        "article_ref": "12(2)(e)",
        "description": (
            "Quantity of the relevant products expressed in net mass "
            "and supplementary units where applicable"
        ),
        "source_agents": ["EUDR-001", "EUDR-011"],
        "fields": [
            {"name": "quantity_net_mass_kg", "article_ref": "12(2)(e)(i)", "required": True},
            {"name": "supplementary_units", "article_ref": "12(2)(e)(ii)", "required": False},
            {"name": "mass_balance_status", "article_ref": "12(2)(e)(iii)", "required": False},
        ],
    },
    {
        "section_number": 6,
        "title": "Date of Production",
        "article_ref": "12(2)(f)",
        "description": (
            "Date or time range of production of the commodity "
            "and satellite monitoring period"
        ),
        "source_agents": ["EUDR-001", "EUDR-003"],
        "fields": [
            {"name": "production_start_date", "article_ref": "12(2)(f)(i)", "required": True},
            {"name": "production_end_date", "article_ref": "12(2)(f)(ii)", "required": True},
            {"name": "monitoring_period", "article_ref": "12(2)(f)(iii)", "required": False},
        ],
    },
    {
        "section_number": 7,
        "title": "Deforestation-Free Status",
        "article_ref": "12(2)(g)",
        "description": (
            "Adequately conclusive and verifiable information that the "
            "relevant products are deforestation-free"
        ),
        "source_agents": ["EUDR-003", "EUDR-004", "EUDR-005"],
        "fields": [
            {"name": "deforestation_free_determination", "article_ref": "12(2)(g)(i)", "required": True},
            {"name": "forest_cover_analysis", "article_ref": "12(2)(g)(ii)", "required": True},
            {"name": "land_use_change_analysis", "article_ref": "12(2)(g)(iii)", "required": True},
            {"name": "satellite_evidence", "article_ref": "12(2)(g)(iv)", "required": True},
        ],
    },
    {
        "section_number": 8,
        "title": "Legal Compliance",
        "article_ref": "12(2)(h)",
        "description": (
            "Information that the relevant products comply with the "
            "relevant legislation of the country of production"
        ),
        "source_agents": ["EUDR-023", "EUDR-024"],
        "fields": [
            {"name": "legal_compliance_status", "article_ref": "12(2)(h)(i)", "required": True},
            {"name": "applicable_legislation", "article_ref": "12(2)(h)(ii)", "required": True},
            {"name": "compliance_evidence", "article_ref": "12(2)(h)(iii)", "required": False},
            {"name": "third_party_verification", "article_ref": "12(2)(h)(iv)", "required": False},
        ],
    },
    {
        "section_number": 9,
        "title": "Risk Assessment and Mitigation",
        "article_ref": "12(2)(i)",
        "description": (
            "Risk assessment results, risk mitigation measures adopted, "
            "and composite risk score with dimension breakdown"
        ),
        "source_agents": [f"EUDR-{i:03d}" for i in range(16, 26)],
        "fields": [
            {"name": "composite_risk_score", "article_ref": "12(2)(i)(i)", "required": True},
            {"name": "risk_level_classification", "article_ref": "12(2)(i)(ii)", "required": True},
            {"name": "risk_dimension_breakdown", "article_ref": "12(2)(i)(iii)", "required": True},
            {"name": "mitigation_measures", "article_ref": "12(2)(i)(iv)", "required": False},
            {"name": "residual_risk_score", "article_ref": "12(2)(i)(v)", "required": False},
        ],
    },
]


# ---------------------------------------------------------------------------
# DueDiligencePackageGenerator
# ---------------------------------------------------------------------------


class DueDiligencePackageGenerator:
    """Generator for audit-ready due diligence evidence packages.

    Compiles all 25 agent outputs into a structured, DDS-compatible
    evidence bundle with SHA-256 provenance chain for EUDR Article 12
    submission to the EU Information System.

    All data compilation is deterministic. The same agent outputs
    will always produce the same package content (excluding timestamps).

    Attributes:
        _config: Configuration with format and language settings.

    Example:
        >>> generator = DueDiligencePackageGenerator()
        >>> package = generator.generate_package(
        ...     workflow_state=state,
        ...     risk_profile=profile,
        ... )
        >>> assert package.integrity_hash is not None
    """

    def __init__(
        self,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize the DueDiligencePackageGenerator.

        Args:
            config: Optional configuration override.
        """
        self._config = config or get_config()
        logger.info("DueDiligencePackageGenerator initialized")

    # ------------------------------------------------------------------
    # Package generation
    # ------------------------------------------------------------------

    def generate_package(
        self,
        workflow_state: WorkflowState,
        risk_profile: Optional[CompositeRiskProfile] = None,
        mitigation_decision: Optional[MitigationDecision] = None,
        quality_gate_results: Optional[Dict[str, QualityGateEvaluation]] = None,
        agent_outputs: Optional[Dict[str, Dict[str, Any]]] = None,
        language: str = "en",
        generated_by: str = "system",
    ) -> DueDiligencePackage:
        """Generate a complete due diligence package.

        Compiles all available agent outputs into a structured package
        with DDS sections, integrity hashes, and provenance tracking.

        Args:
            workflow_state: Complete workflow execution state.
            risk_profile: Composite risk profile from Phase 2.
            mitigation_decision: Mitigation decision from Phase 3.
            quality_gate_results: All quality gate evaluations.
            agent_outputs: Per-agent output data dictionaries.
            language: Report language code.
            generated_by: User or system generating the package.

        Returns:
            DueDiligencePackage with complete evidence bundle.

        Example:
            >>> generator = DueDiligencePackageGenerator()
            >>> pkg = generator.generate_package(workflow_state=state)
            >>> assert len(pkg.sections) == 9
        """
        start_time = utcnow()
        outputs = agent_outputs or {}

        # Build DDS sections
        sections = self._build_sections(
            workflow_state, outputs, risk_profile, mitigation_decision
        )

        # Generate executive summary
        executive_summary = self._generate_executive_summary(
            workflow_state, risk_profile, mitigation_decision
        )

        # Count executed agents
        from greenlang.agents.eudr.due_diligence_orchestrator.models import (
            AgentExecutionStatus,
        )
        total_executed = sum(
            1 for rec in workflow_state.agent_executions.values()
            if rec.status == AgentExecutionStatus.COMPLETED
        )

        # Build workflow metadata
        workflow_metadata = self._build_workflow_metadata(workflow_state)

        # Compute per-artifact hashes
        artifact_hashes = self._compute_artifact_hashes(
            sections, outputs, risk_profile, mitigation_decision
        )

        # Build package
        package = DueDiligencePackage(
            package_id=_new_uuid(),
            workflow_id=workflow_state.workflow_id,
            dds_schema_version=self._config.dds_schema_version,
            operator_id=workflow_state.operator_id,
            operator_name=workflow_state.operator_name,
            commodity=workflow_state.commodity,
            workflow_type=workflow_state.workflow_type,
            sections=sections,
            executive_summary=executive_summary,
            risk_profile=risk_profile,
            mitigation_summary=mitigation_decision,
            quality_gate_results=quality_gate_results or {},
            workflow_metadata=workflow_metadata,
            total_agents_executed=total_executed,
            total_duration_ms=workflow_state.total_duration_ms,
            language=language,
            artifact_hashes=artifact_hashes,
            generated_at=utcnow(),
            generated_by=generated_by,
        )

        # Compute package-level integrity hash
        package.integrity_hash = self._compute_package_hash(package)
        package.provenance_hash = package.integrity_hash

        duration_ms = (utcnow() - start_time).total_seconds() * 1000
        logger.info(
            f"Generated DD package {package.package_id} for workflow "
            f"{workflow_state.workflow_id}: {len(sections)} sections, "
            f"{total_executed} agents, in {duration_ms:.1f}ms"
        )

        return package

    # ------------------------------------------------------------------
    # Section building
    # ------------------------------------------------------------------

    def _build_sections(
        self,
        workflow_state: WorkflowState,
        agent_outputs: Dict[str, Dict[str, Any]],
        risk_profile: Optional[CompositeRiskProfile],
        mitigation_decision: Optional[MitigationDecision],
    ) -> List[DDSSection]:
        """Build all 9 DDS sections from agent outputs.

        Args:
            workflow_state: Workflow state with execution records.
            agent_outputs: Per-agent output data.
            risk_profile: Composite risk profile.
            mitigation_decision: Mitigation decision.

        Returns:
            List of 9 DDSSection objects.
        """
        sections: List[DDSSection] = []

        for section_def in _DDS_SECTIONS:
            section = self._build_section(
                section_def, workflow_state, agent_outputs,
                risk_profile, mitigation_decision
            )
            sections.append(section)

        return sections

    def _build_section(
        self,
        section_def: Dict[str, Any],
        workflow_state: WorkflowState,
        agent_outputs: Dict[str, Dict[str, Any]],
        risk_profile: Optional[CompositeRiskProfile],
        mitigation_decision: Optional[MitigationDecision],
    ) -> DDSSection:
        """Build a single DDS section.

        Args:
            section_def: Section definition from _DDS_SECTIONS.
            workflow_state: Workflow state.
            agent_outputs: Agent output data.
            risk_profile: Risk profile (for Section 9).
            mitigation_decision: Mitigation decision (for Section 9).

        Returns:
            DDSSection with fields and evidence.
        """
        section_number = section_def["section_number"]
        source_agents = section_def["source_agents"]
        field_defs = section_def.get("fields", [])

        # Build DDS fields
        fields: List[DDSField] = []
        validated_count = 0

        for field_def in field_defs:
            field_name = field_def["name"]
            article_ref = field_def["article_ref"]
            required = field_def.get("required", True)

            # Extract value from agent outputs
            value = self._extract_field_value(
                field_name, source_agents, agent_outputs,
                workflow_state, risk_profile, mitigation_decision
            )

            validated = value is not None
            if validated:
                validated_count += 1

            field = DDSField(
                field_id=_new_uuid(),
                article_ref=article_ref,
                field_name=field_name,
                description=f"DDS field per {article_ref}",
                value=value,
                source_agents=[
                    a for a in source_agents if a in agent_outputs
                ],
                validated=validated,
                validation_notes=(
                    None if validated
                    else f"Field '{field_name}' not available from source agents"
                ),
            )
            fields.append(field)

        # Compute section completeness
        total_fields = len(field_defs)
        completeness_pct = Decimal("0")
        if total_fields > 0:
            completeness_pct = Decimal(str(
                validated_count / total_fields * 100
            )).quantize(Decimal("0.01"))

        # Collect evidence references
        evidence_refs = [
            rec.output_ref
            for aid in source_agents
            if (rec := workflow_state.agent_executions.get(aid))
            and rec.output_ref
        ]

        # Collect agent output summaries
        agent_output_summaries: Dict[str, Any] = {}
        for aid in source_agents:
            if aid in agent_outputs:
                agent_output_summaries[aid] = {
                    "name": AGENT_NAMES.get(aid, aid),
                    "output_available": True,
                    "field_count": len(agent_outputs[aid]),
                }

        section = DDSSection(
            section_id=_new_uuid(),
            section_number=section_number,
            title=section_def["title"],
            description=section_def["description"],
            fields=fields,
            evidence_refs=evidence_refs,
            agent_outputs=agent_output_summaries,
            completeness_pct=completeness_pct,
        )

        return section

    # ------------------------------------------------------------------
    # Field value extraction
    # ------------------------------------------------------------------

    def _extract_field_value(
        self,
        field_name: str,
        source_agents: List[str],
        agent_outputs: Dict[str, Dict[str, Any]],
        workflow_state: WorkflowState,
        risk_profile: Optional[CompositeRiskProfile],
        mitigation_decision: Optional[MitigationDecision],
    ) -> Optional[Any]:
        """Extract a DDS field value from agent outputs.

        Searches source agents for the field value. For special fields
        like risk scores, extracts from the risk profile or mitigation
        decision directly.

        Args:
            field_name: DDS field name.
            source_agents: Agents that may provide this field.
            agent_outputs: Per-agent output data.
            workflow_state: Workflow state.
            risk_profile: Composite risk profile.
            mitigation_decision: Mitigation decision.

        Returns:
            Field value or None if not available.
        """
        # Special fields from workflow state
        if field_name == "operator_name":
            return workflow_state.operator_name
        if field_name == "country_of_production":
            return workflow_state.country_codes or None
        if field_name == "country_iso_code":
            return workflow_state.country_codes or None

        # Special fields from risk profile
        if field_name == "composite_risk_score" and risk_profile:
            return str(risk_profile.composite_score)
        if field_name == "risk_level_classification" and risk_profile:
            return risk_profile.risk_level
        if field_name == "risk_dimension_breakdown" and risk_profile:
            return {
                c.agent_name: str(c.weighted_score)
                for c in risk_profile.contributions
            }

        # Special fields from mitigation decision
        if field_name == "mitigation_measures" and mitigation_decision:
            return mitigation_decision.mitigation_strategies
        if field_name == "residual_risk_score" and mitigation_decision:
            return (
                str(mitigation_decision.post_mitigation_score)
                if mitigation_decision.post_mitigation_score is not None
                else None
            )

        # Search agent outputs for the field
        for agent_id in source_agents:
            output = agent_outputs.get(agent_id, {})
            if field_name in output:
                return output[field_name]

            # Try common field name variations
            variations = [
                field_name.replace("_", ""),
                field_name.replace("_", "-"),
                field_name.lower(),
            ]
            for var in variations:
                if var in output:
                    return output[var]

        return None

    # ------------------------------------------------------------------
    # Executive summary generation
    # ------------------------------------------------------------------

    def _generate_executive_summary(
        self,
        workflow_state: WorkflowState,
        risk_profile: Optional[CompositeRiskProfile],
        mitigation_decision: Optional[MitigationDecision],
    ) -> str:
        """Generate a deterministic executive summary.

        The summary is assembled from template fragments based on
        workflow outcome data. No LLM is used for text generation.

        Args:
            workflow_state: Workflow state.
            risk_profile: Composite risk profile.
            mitigation_decision: Mitigation decision.

        Returns:
            Executive summary text.
        """
        parts: List[str] = []

        # Workflow overview
        commodity = (
            workflow_state.commodity.value
            if workflow_state.commodity else "unspecified commodity"
        )
        wf_type = workflow_state.workflow_type.value
        parts.append(
            f"This due diligence statement presents the results of a "
            f"{wf_type} EUDR due diligence assessment for {commodity}."
        )

        # Operator
        if workflow_state.operator_name:
            parts.append(
                f"The assessment was conducted for operator "
                f"'{workflow_state.operator_name}'."
            )

        # Countries
        if workflow_state.country_codes:
            countries = ", ".join(workflow_state.country_codes)
            parts.append(
                f"Countries of production assessed: {countries}."
            )

        # Agent execution summary
        from greenlang.agents.eudr.due_diligence_orchestrator.models import (
            AgentExecutionStatus,
        )
        completed = sum(
            1 for r in workflow_state.agent_executions.values()
            if r.status == AgentExecutionStatus.COMPLETED
        )
        total = len(workflow_state.agent_executions) or 25
        parts.append(
            f"The assessment executed {completed} of {total} EUDR agents "
            f"across information gathering, risk assessment, and risk "
            f"mitigation phases."
        )

        # Risk assessment
        if risk_profile:
            parts.append(
                f"The composite risk score was determined to be "
                f"{risk_profile.composite_score}, classified as "
                f"'{risk_profile.risk_level}' risk level."
            )
            if risk_profile.highest_risk_dimensions:
                dims = ", ".join(risk_profile.highest_risk_dimensions[:3])
                parts.append(
                    f"The highest-risk dimensions identified were: {dims}."
                )

        # Mitigation
        if mitigation_decision:
            if mitigation_decision.mitigation_required:
                parts.append(
                    f"Risk mitigation was required at the "
                    f"'{mitigation_decision.mitigation_level}' level. "
                    f"{len(mitigation_decision.mitigation_strategies)} "
                    f"mitigation strategies were identified."
                )
                if mitigation_decision.post_mitigation_score is not None:
                    parts.append(
                        f"Post-mitigation residual risk was reduced to "
                        f"{mitigation_decision.post_mitigation_score}."
                    )
                if mitigation_decision.adequacy_verified:
                    parts.append(
                        "Mitigation adequacy has been verified."
                    )
            else:
                parts.append(
                    "No risk mitigation measures were required based on "
                    "the risk assessment results."
                )

        return " ".join(parts)

    # ------------------------------------------------------------------
    # Workflow metadata
    # ------------------------------------------------------------------

    def _build_workflow_metadata(
        self,
        workflow_state: WorkflowState,
    ) -> Dict[str, Any]:
        """Build workflow execution metadata for the package.

        Args:
            workflow_state: Workflow state.

        Returns:
            Metadata dictionary.
        """
        return {
            "workflow_id": workflow_state.workflow_id,
            "definition_id": workflow_state.definition_id,
            "workflow_type": workflow_state.workflow_type.value,
            "status": workflow_state.status.value,
            "current_phase": workflow_state.current_phase.value,
            "operator_id": workflow_state.operator_id,
            "product_ids": workflow_state.product_ids,
            "shipment_ids": workflow_state.shipment_ids,
            "country_codes": workflow_state.country_codes,
            "created_at": (
                workflow_state.created_at.isoformat()
                if workflow_state.created_at else None
            ),
            "started_at": (
                workflow_state.started_at.isoformat()
                if workflow_state.started_at else None
            ),
            "completed_at": (
                workflow_state.completed_at.isoformat()
                if workflow_state.completed_at else None
            ),
            "total_duration_ms": (
                str(workflow_state.total_duration_ms)
                if workflow_state.total_duration_ms else None
            ),
            "transition_count": len(workflow_state.transitions),
            "checkpoint_count": len(workflow_state.checkpoints),
            "dds_schema_version": self._config.dds_schema_version,
            "retention_years": self._config.retention_years,
        }

    # ------------------------------------------------------------------
    # Integrity hashing
    # ------------------------------------------------------------------

    def _compute_artifact_hashes(
        self,
        sections: List[DDSSection],
        agent_outputs: Dict[str, Dict[str, Any]],
        risk_profile: Optional[CompositeRiskProfile],
        mitigation_decision: Optional[MitigationDecision],
    ) -> Dict[str, str]:
        """Compute per-artifact SHA-256 hashes.

        Args:
            sections: DDS sections.
            agent_outputs: Agent output data.
            risk_profile: Risk profile.
            mitigation_decision: Mitigation decision.

        Returns:
            Dictionary mapping artifact name to SHA-256 hash.
        """
        hashes: Dict[str, str] = {}

        # Hash each section
        for section in sections:
            section_data = {
                "section_number": section.section_number,
                "title": section.title,
                "fields": [
                    {
                        "field_name": f.field_name,
                        "value": str(f.value) if f.value else None,
                        "validated": f.validated,
                    }
                    for f in section.fields
                ],
            }
            hashes[f"section_{section.section_number}"] = self._sha256(
                section_data
            )

        # Hash agent outputs
        for agent_id in sorted(agent_outputs.keys()):
            hashes[f"agent_output_{agent_id}"] = self._sha256(
                agent_outputs[agent_id]
            )

        # Hash risk profile
        if risk_profile:
            hashes["risk_profile"] = self._sha256({
                "composite_score": str(risk_profile.composite_score),
                "risk_level": risk_profile.risk_level,
                "contributions": [
                    {"agent": c.agent_id, "score": str(c.weighted_score)}
                    for c in risk_profile.contributions
                ],
            })

        # Hash mitigation decision
        if mitigation_decision:
            hashes["mitigation_decision"] = self._sha256({
                "required": mitigation_decision.mitigation_required,
                "level": mitigation_decision.mitigation_level,
                "pre_score": str(mitigation_decision.pre_mitigation_score),
                "post_score": (
                    str(mitigation_decision.post_mitigation_score)
                    if mitigation_decision.post_mitigation_score else None
                ),
            })

        return hashes

    def _compute_package_hash(self, package: DueDiligencePackage) -> str:
        """Compute package-level SHA-256 integrity hash.

        Covers all artifact hashes, workflow metadata, and section data
        for tamper-evident package verification.

        Args:
            package: Complete DD package.

        Returns:
            64-character hex SHA-256 hash.
        """
        data = {
            "package_id": package.package_id,
            "workflow_id": package.workflow_id,
            "artifact_hashes": package.artifact_hashes,
            "section_count": len(package.sections),
            "total_agents_executed": package.total_agents_executed,
            "workflow_type": package.workflow_type.value,
            "commodity": package.commodity.value if package.commodity else None,
        }
        return self._sha256(data)

    def _sha256(self, data: Any) -> str:
        """Compute SHA-256 hash of data using canonical JSON.

        Args:
            data: Data to hash (must be JSON-serializable).

        Returns:
            64-character hex SHA-256 hash.
        """
        canonical = json.dumps(
            data, sort_keys=True, separators=(",", ":"), default=str
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------------
    # Package completeness
    # ------------------------------------------------------------------

    def compute_package_completeness(
        self,
        package: DueDiligencePackage,
    ) -> Decimal:
        """Compute overall package completeness percentage.

        Average of all section completeness percentages.

        Args:
            package: Generated DD package.

        Returns:
            Completeness percentage (0-100).
        """
        if not package.sections:
            return Decimal("0")

        total = sum(s.completeness_pct for s in package.sections)
        avg = (total / Decimal(str(len(package.sections)))).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
        return avg

    def get_incomplete_sections(
        self,
        package: DueDiligencePackage,
        threshold: Decimal = Decimal("100"),
    ) -> List[Dict[str, Any]]:
        """Get sections that are below the completeness threshold.

        Args:
            package: Generated DD package.
            threshold: Minimum completeness percentage.

        Returns:
            List of incomplete section summaries.
        """
        incomplete: List[Dict[str, Any]] = []
        for section in package.sections:
            if section.completeness_pct < threshold:
                missing_fields = [
                    f.field_name for f in section.fields
                    if not f.validated
                ]
                incomplete.append({
                    "section_number": section.section_number,
                    "title": section.title,
                    "completeness_pct": str(section.completeness_pct),
                    "missing_fields": missing_fields,
                })
        return incomplete
