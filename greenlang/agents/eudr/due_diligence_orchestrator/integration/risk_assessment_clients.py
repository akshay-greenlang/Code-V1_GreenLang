# -*- coding: utf-8 -*-
"""
Risk Assessment Integration Clients - AGENT-EUDR-026

Typed wrapper clients for all 10 Phase 2 EUDR agents (EUDR-016 through
EUDR-025) responsible for multi-dimensional risk assessment covering
country risk, supplier risk, commodity risk, corruption, deforestation
alerts, indigenous rights, protected areas, legal compliance,
third-party audits, and risk mitigation advisory.

Each client class encapsulates the agent-specific input preparation,
output validation, risk score extraction, and error handling for its
upstream agent, while delegating the actual HTTP call to the shared
AgentClient.

Clients:
    - CountryRiskClient            (EUDR-016)
    - SupplierRiskClient           (EUDR-017)
    - CommodityRiskClient          (EUDR-018)
    - CorruptionIndexClient        (EUDR-019)
    - DeforestationAlertClient     (EUDR-020)
    - IndigenousRightsClient       (EUDR-021)
    - ProtectedAreaClient          (EUDR-022)
    - LegalComplianceClient        (EUDR-023)
    - ThirdPartyAuditClient        (EUDR-024)
    - RiskMitigationClient         (EUDR-025)

Zero-Hallucination:
    These clients only transform and relay agent-produced data; no
    numeric computation or LLM generation occurs here. Risk score
    extraction returns the raw agent-computed values without
    modification.

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-026 Due Diligence Orchestrator (GL-EUDR-DDO-026)
Status: Production Ready
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any, Dict, List, Optional

from greenlang.agents.eudr.due_diligence_orchestrator.config import (
    DueDiligenceOrchestratorConfig,
    get_config,
)
from greenlang.agents.eudr.due_diligence_orchestrator.integration.agent_client import (
    AgentCallResult,
    AgentClient,
)
from greenlang.agents.eudr.due_diligence_orchestrator.models import (
    AGENT_NAMES,
    _new_uuid,
    _utcnow,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base Phase 2 Client
# ---------------------------------------------------------------------------


class _BasePhase2Client:
    """Base class for all Phase 2 risk assessment agent clients.

    Provides shared infrastructure for risk input preparation, score
    extraction, and health checking. Every Phase 2 agent returns a
    risk score in the range [0, 100] with an associated confidence.

    Attributes:
        _agent_id: EUDR agent identifier (e.g., "EUDR-016").
        _agent_name: Human-readable agent name.
        _risk_dimension: Name of the risk dimension this agent covers.
        _client: Shared AgentClient instance.
        _config: Orchestrator configuration.
    """

    def __init__(
        self,
        agent_id: str,
        risk_dimension: str,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize base Phase 2 client.

        Args:
            agent_id: EUDR agent identifier.
            risk_dimension: Name of the risk dimension.
            client: Optional shared AgentClient instance.
            config: Optional configuration override.
        """
        self._agent_id = agent_id
        self._agent_name = AGENT_NAMES.get(agent_id, agent_id)
        self._risk_dimension = risk_dimension
        self._config = config or get_config()
        self._client = client or AgentClient(self._config)

    def call(
        self,
        input_data: Dict[str, Any],
        timeout_s: Optional[int] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> AgentCallResult:
        """Call the agent with the given input data.

        Args:
            input_data: Agent-specific input payload.
            timeout_s: Optional timeout override in seconds.
            headers: Optional additional HTTP headers.

        Returns:
            AgentCallResult with response data or error details.
        """
        logger.info(
            f"Calling {self._agent_name} ({self._agent_id}) "
            f"for risk dimension '{self._risk_dimension}'"
        )
        return self._client.call_agent(
            self._agent_id,
            input_data,
            timeout_s=timeout_s,
            headers=headers,
        )

    def extract_risk_score(
        self, result: AgentCallResult
    ) -> Optional[Decimal]:
        """Extract the raw risk score from an agent result.

        Args:
            result: AgentCallResult from the risk agent.

        Returns:
            Risk score as Decimal in [0, 100], or None if unavailable.
        """
        if not result.success:
            return None

        raw = result.output_data.get("risk_score")
        if raw is None:
            raw = result.output_data.get("score")
        if raw is None:
            return None

        try:
            return Decimal(str(raw))
        except Exception:
            logger.warning(
                f"Cannot parse risk score from {self._agent_id}: {raw}"
            )
            return None

    def extract_confidence(
        self, result: AgentCallResult
    ) -> Optional[Decimal]:
        """Extract the confidence score from an agent result.

        Args:
            result: AgentCallResult from the risk agent.

        Returns:
            Confidence as Decimal in [0, 1], or None if unavailable.
        """
        if not result.success:
            return None

        raw = result.output_data.get("confidence")
        if raw is None:
            raw = result.output_data.get("confidence_score")
        if raw is None:
            return None

        try:
            return Decimal(str(raw))
        except Exception:
            return None

    def is_healthy(self) -> bool:
        """Check if the agent endpoint is healthy.

        Returns:
            True if the agent responds to health check.
        """
        return self._client.check_agent_health(self._agent_id)

    @property
    def agent_id(self) -> str:
        """Return the EUDR agent identifier."""
        return self._agent_id

    @property
    def agent_name(self) -> str:
        """Return the human-readable agent name."""
        return self._agent_name

    @property
    def risk_dimension(self) -> str:
        """Return the risk dimension name."""
        return self._risk_dimension


# ---------------------------------------------------------------------------
# EUDR-016: Country Risk Evaluator
# ---------------------------------------------------------------------------


class CountryRiskClient(_BasePhase2Client):
    """Client for EUDR-016 Country Risk Evaluator.

    Evaluates country-level risk based on governance indicators,
    deforestation rates, corruption indices, and EUDR country
    benchmarking data.

    Example:
        >>> client = CountryRiskClient()
        >>> result = client.evaluate_country_risk(
        ...     countries=["BR", "ID", "GH"],
        ...     commodity="cocoa"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-016 client."""
        super().__init__("EUDR-016", "country", client, config)

    def evaluate_country_risk(
        self,
        countries: List[str],
        commodity: str,
        include_sub_national: bool = True,
    ) -> AgentCallResult:
        """Evaluate country risk for specified countries.

        Args:
            countries: List of ISO 3166-1 alpha-2 country codes.
            commodity: EUDR commodity type.
            include_sub_national: Include sub-national risk variation.

        Returns:
            AgentCallResult with country risk scores.
        """
        return self.call({
            "countries": countries,
            "commodity": commodity,
            "include_sub_national": include_sub_national,
        })

    def extract_country_scores(
        self, result: AgentCallResult
    ) -> Dict[str, Decimal]:
        """Extract per-country risk scores from result.

        Args:
            result: Successful AgentCallResult from EUDR-016.

        Returns:
            Dictionary mapping country code to risk score.
        """
        if not result.success:
            return {}

        scores: Dict[str, Decimal] = {}
        country_data = result.output_data.get("country_scores", {})
        for code, score in country_data.items():
            try:
                scores[code] = Decimal(str(score))
            except Exception:
                pass
        return scores


# ---------------------------------------------------------------------------
# EUDR-017: Supplier Risk Scorer
# ---------------------------------------------------------------------------


class SupplierRiskClient(_BasePhase2Client):
    """Client for EUDR-017 Supplier Risk Scorer.

    Scores supplier risk based on compliance history, certification
    status, location risk, and supply chain position.

    Example:
        >>> client = SupplierRiskClient()
        >>> result = client.score_supplier_risk(
        ...     supplier_ids=["SUP-001", "SUP-002"],
        ...     commodity="palm_oil"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-017 client."""
        super().__init__("EUDR-017", "supplier", client, config)

    def score_supplier_risk(
        self,
        supplier_ids: List[str],
        commodity: str,
        include_history: bool = True,
    ) -> AgentCallResult:
        """Score risk for specified suppliers.

        Args:
            supplier_ids: List of supplier identifiers.
            commodity: EUDR commodity type.
            include_history: Include historical compliance data.

        Returns:
            AgentCallResult with supplier risk scores.
        """
        return self.call({
            "supplier_ids": supplier_ids,
            "commodity": commodity,
            "include_history": include_history,
        })


# ---------------------------------------------------------------------------
# EUDR-018: Commodity Risk Analyzer
# ---------------------------------------------------------------------------


class CommodityRiskClient(_BasePhase2Client):
    """Client for EUDR-018 Commodity Risk Analyzer.

    Analyzes commodity-specific risk factors including market dynamics,
    deforestation linkage, and supply chain complexity.

    Example:
        >>> client = CommodityRiskClient()
        >>> result = client.analyze_commodity_risk(
        ...     commodity="soya",
        ...     countries=["BR"],
        ...     volume_tonnes=10000
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-018 client."""
        super().__init__("EUDR-018", "commodity", client, config)

    def analyze_commodity_risk(
        self,
        commodity: str,
        countries: List[str],
        volume_tonnes: Optional[float] = None,
    ) -> AgentCallResult:
        """Analyze commodity risk.

        Args:
            commodity: EUDR commodity type.
            countries: Origin country codes.
            volume_tonnes: Optional trade volume in tonnes.

        Returns:
            AgentCallResult with commodity risk analysis.
        """
        input_data: Dict[str, Any] = {
            "commodity": commodity,
            "countries": countries,
        }
        if volume_tonnes is not None:
            input_data["volume_tonnes"] = volume_tonnes

        return self.call(input_data)


# ---------------------------------------------------------------------------
# EUDR-019: Corruption Index Monitor
# ---------------------------------------------------------------------------


class CorruptionIndexClient(_BasePhase2Client):
    """Client for EUDR-019 Corruption Index Monitor.

    Monitors corruption risk based on Transparency International CPI,
    World Bank governance indicators, and other integrity indices.

    Example:
        >>> client = CorruptionIndexClient()
        >>> result = client.assess_corruption_risk(
        ...     countries=["BR", "ID"],
        ...     year=2025
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-019 client."""
        super().__init__("EUDR-019", "corruption", client, config)

    def assess_corruption_risk(
        self,
        countries: List[str],
        year: Optional[int] = None,
    ) -> AgentCallResult:
        """Assess corruption risk for specified countries.

        Args:
            countries: List of ISO 3166-1 alpha-2 country codes.
            year: Assessment year (default: current year).

        Returns:
            AgentCallResult with corruption risk scores.
        """
        input_data: Dict[str, Any] = {"countries": countries}
        if year is not None:
            input_data["year"] = year

        return self.call(input_data)


# ---------------------------------------------------------------------------
# EUDR-020: Deforestation Alert System
# ---------------------------------------------------------------------------


class DeforestationAlertClient(_BasePhase2Client):
    """Client for EUDR-020 Deforestation Alert System.

    Monitors and evaluates deforestation alerts from satellite-based
    alert systems (GFW, GLAD, JRC-TMF, etc.).

    Example:
        >>> client = DeforestationAlertClient()
        >>> result = client.check_deforestation_alerts(
        ...     plot_ids=["PLOT-001"],
        ...     cutoff_date="2020-12-31"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-020 client."""
        super().__init__("EUDR-020", "deforestation", client, config)

    def check_deforestation_alerts(
        self,
        plot_ids: List[str],
        cutoff_date: str = "2020-12-31",
        alert_sources: Optional[List[str]] = None,
    ) -> AgentCallResult:
        """Check deforestation alerts for production plots.

        Args:
            plot_ids: Production plot identifiers.
            cutoff_date: EUDR cutoff date.
            alert_sources: Optional list of alert data sources.

        Returns:
            AgentCallResult with deforestation alert assessment.
        """
        input_data: Dict[str, Any] = {
            "plot_ids": plot_ids,
            "cutoff_date": cutoff_date,
        }
        if alert_sources:
            input_data["alert_sources"] = alert_sources

        return self.call(input_data)

    def has_active_alerts(self, result: AgentCallResult) -> bool:
        """Check if there are active deforestation alerts.

        Args:
            result: AgentCallResult from EUDR-020.

        Returns:
            True if active deforestation alerts exist.
        """
        if not result.success:
            return False

        alerts = result.output_data.get("active_alerts", [])
        return len(alerts) > 0


# ---------------------------------------------------------------------------
# EUDR-021: Indigenous Rights Checker
# ---------------------------------------------------------------------------


class IndigenousRightsClient(_BasePhase2Client):
    """Client for EUDR-021 Indigenous Rights Checker.

    Assesses risk related to indigenous peoples' rights, FPIC
    (Free, Prior and Informed Consent), and customary land rights.

    Example:
        >>> client = IndigenousRightsClient()
        >>> result = client.assess_indigenous_rights(
        ...     plot_ids=["PLOT-001"],
        ...     countries=["BR"]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-021 client."""
        super().__init__("EUDR-021", "indigenous", client, config)

    def assess_indigenous_rights(
        self,
        plot_ids: List[str],
        countries: List[str],
        check_fpic: bool = True,
    ) -> AgentCallResult:
        """Assess indigenous rights risk for production areas.

        Args:
            plot_ids: Production plot identifiers.
            countries: Country codes of production plots.
            check_fpic: Whether to check FPIC compliance.

        Returns:
            AgentCallResult with indigenous rights assessment.
        """
        return self.call({
            "plot_ids": plot_ids,
            "countries": countries,
            "check_fpic": check_fpic,
        })


# ---------------------------------------------------------------------------
# EUDR-022: Protected Area Validator
# ---------------------------------------------------------------------------


class ProtectedAreaClient(_BasePhase2Client):
    """Client for EUDR-022 Protected Area Validator.

    Validates that production plots do not overlap with protected
    areas, national parks, or other conservation zones.

    Example:
        >>> client = ProtectedAreaClient()
        >>> result = client.validate_protected_areas(
        ...     plot_boundaries=[...],
        ...     countries=["BR"]
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-022 client."""
        super().__init__("EUDR-022", "protected_area", client, config)

    def validate_protected_areas(
        self,
        plot_boundaries: List[Dict[str, Any]],
        countries: List[str],
        buffer_km: float = 0.0,
    ) -> AgentCallResult:
        """Validate plots against protected area boundaries.

        Args:
            plot_boundaries: List of plot boundary GeoJSON objects.
            countries: Country codes for reference data lookup.
            buffer_km: Buffer zone around protected areas in km.

        Returns:
            AgentCallResult with protected area validation.
        """
        return self.call({
            "plot_boundaries": plot_boundaries,
            "countries": countries,
            "buffer_km": buffer_km,
        })

    def has_overlaps(self, result: AgentCallResult) -> bool:
        """Check if any plots overlap with protected areas.

        Args:
            result: AgentCallResult from EUDR-022.

        Returns:
            True if any overlaps detected.
        """
        if not result.success:
            return False

        overlaps = result.output_data.get("overlapping_plots", [])
        return len(overlaps) > 0


# ---------------------------------------------------------------------------
# EUDR-023: Legal Compliance Verifier
# ---------------------------------------------------------------------------


class LegalComplianceClient(_BasePhase2Client):
    """Client for EUDR-023 Legal Compliance Verifier.

    Verifies compliance with relevant legislation in the country
    of production including land tenure, environmental permits,
    labour rights, and trade regulations.

    Example:
        >>> client = LegalComplianceClient()
        >>> result = client.verify_legal_compliance(
        ...     countries=["BR"],
        ...     commodity="soya",
        ...     operator_id="OP-001"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-023 client."""
        super().__init__("EUDR-023", "legal", client, config)

    def verify_legal_compliance(
        self,
        countries: List[str],
        commodity: str,
        operator_id: Optional[str] = None,
    ) -> AgentCallResult:
        """Verify legal compliance in production countries.

        Args:
            countries: ISO 3166-1 alpha-2 country codes.
            commodity: EUDR commodity type.
            operator_id: Optional operator identifier.

        Returns:
            AgentCallResult with legal compliance assessment.
        """
        input_data: Dict[str, Any] = {
            "countries": countries,
            "commodity": commodity,
        }
        if operator_id:
            input_data["operator_id"] = operator_id

        return self.call(input_data)

    def extract_applicable_legislation(
        self, result: AgentCallResult
    ) -> List[Dict[str, Any]]:
        """Extract applicable legislation list from result.

        Args:
            result: Successful AgentCallResult from EUDR-023.

        Returns:
            List of applicable legislation records.
        """
        if not result.success:
            return []
        return result.output_data.get("applicable_legislation", [])


# ---------------------------------------------------------------------------
# EUDR-024: Third-Party Audit Manager
# ---------------------------------------------------------------------------


class ThirdPartyAuditClient(_BasePhase2Client):
    """Client for EUDR-024 Third-Party Audit Manager.

    Manages third-party audit requirements, tracks audit status,
    and evaluates audit findings for risk assessment.

    Example:
        >>> client = ThirdPartyAuditClient()
        >>> result = client.evaluate_audits(
        ...     supplier_ids=["SUP-001"],
        ...     commodity="wood"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-024 client."""
        super().__init__("EUDR-024", "audit", client, config)

    def evaluate_audits(
        self,
        supplier_ids: List[str],
        commodity: str,
        certification_schemes: Optional[List[str]] = None,
    ) -> AgentCallResult:
        """Evaluate third-party audits for suppliers.

        Args:
            supplier_ids: Supplier identifiers.
            commodity: EUDR commodity type.
            certification_schemes: Optional certification schemes to check.

        Returns:
            AgentCallResult with audit evaluation results.
        """
        input_data: Dict[str, Any] = {
            "supplier_ids": supplier_ids,
            "commodity": commodity,
        }
        if certification_schemes:
            input_data["certification_schemes"] = certification_schemes

        return self.call(input_data)


# ---------------------------------------------------------------------------
# EUDR-025: Risk Mitigation Advisor
# ---------------------------------------------------------------------------


class RiskMitigationClient(_BasePhase2Client):
    """Client for EUDR-025 Risk Mitigation Advisor.

    Recommends and tracks risk mitigation measures based on
    identified risks, computes residual risk after mitigation,
    and evaluates mitigation adequacy per Article 11.

    This agent participates in both Phase 2 (readiness assessment)
    and Phase 3 (mitigation execution).

    Example:
        >>> client = RiskMitigationClient()
        >>> result = client.recommend_mitigations(
        ...     risk_profile={"composite_score": 62.5, ...},
        ...     commodity="palm_oil"
        ... )
    """

    def __init__(
        self,
        client: Optional[AgentClient] = None,
        config: Optional[DueDiligenceOrchestratorConfig] = None,
    ) -> None:
        """Initialize EUDR-025 client."""
        super().__init__("EUDR-025", "mitigation", client, config)

    def recommend_mitigations(
        self,
        risk_profile: Dict[str, Any],
        commodity: str,
        target_residual_risk: float = 15.0,
    ) -> AgentCallResult:
        """Get mitigation recommendations for identified risks.

        Args:
            risk_profile: Composite risk profile from Phase 2.
            commodity: EUDR commodity type.
            target_residual_risk: Target residual risk score.

        Returns:
            AgentCallResult with mitigation recommendations.
        """
        return self.call({
            "risk_profile": risk_profile,
            "commodity": commodity,
            "target_residual_risk": target_residual_risk,
        })

    def evaluate_mitigation_adequacy(
        self,
        applied_mitigations: List[Dict[str, Any]],
        risk_profile: Dict[str, Any],
    ) -> AgentCallResult:
        """Evaluate adequacy of applied mitigation measures.

        Args:
            applied_mitigations: List of applied mitigation records.
            risk_profile: Original risk profile.

        Returns:
            AgentCallResult with adequacy evaluation.
        """
        return self.call({
            "mode": "adequacy_evaluation",
            "applied_mitigations": applied_mitigations,
            "risk_profile": risk_profile,
        })

    def extract_residual_risk(
        self, result: AgentCallResult
    ) -> Optional[Decimal]:
        """Extract residual risk score from mitigation result.

        Args:
            result: AgentCallResult from EUDR-025.

        Returns:
            Residual risk score as Decimal, or None.
        """
        if not result.success:
            return None

        raw = result.output_data.get("residual_risk_score")
        if raw is None:
            raw = result.output_data.get("residual_risk")
        if raw is None:
            return None

        try:
            return Decimal(str(raw))
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Map of agent_id to client class for dynamic instantiation.
PHASE2_CLIENT_REGISTRY: Dict[str, type] = {
    "EUDR-016": CountryRiskClient,
    "EUDR-017": SupplierRiskClient,
    "EUDR-018": CommodityRiskClient,
    "EUDR-019": CorruptionIndexClient,
    "EUDR-020": DeforestationAlertClient,
    "EUDR-021": IndigenousRightsClient,
    "EUDR-022": ProtectedAreaClient,
    "EUDR-023": LegalComplianceClient,
    "EUDR-024": ThirdPartyAuditClient,
    "EUDR-025": RiskMitigationClient,
}


def get_phase2_client(
    agent_id: str,
    shared_client: Optional[AgentClient] = None,
    config: Optional[DueDiligenceOrchestratorConfig] = None,
) -> _BasePhase2Client:
    """Factory function to get a Phase 2 client by agent ID.

    Args:
        agent_id: EUDR agent identifier (EUDR-016 through EUDR-025).
        shared_client: Optional shared AgentClient instance.
        config: Optional configuration override.

    Returns:
        Initialized Phase 2 client instance.

    Raises:
        ValueError: If agent_id is not a Phase 2 agent.
    """
    client_cls = PHASE2_CLIENT_REGISTRY.get(agent_id)
    if client_cls is None:
        raise ValueError(
            f"Agent {agent_id} is not a Phase 2 agent. "
            f"Valid: {sorted(PHASE2_CLIENT_REGISTRY.keys())}"
        )
    return client_cls(client=shared_client, config=config)


def get_all_phase2_clients(
    shared_client: Optional[AgentClient] = None,
    config: Optional[DueDiligenceOrchestratorConfig] = None,
) -> Dict[str, _BasePhase2Client]:
    """Create all 10 Phase 2 clients with a shared AgentClient.

    Args:
        shared_client: Optional shared AgentClient instance.
        config: Optional configuration override.

    Returns:
        Dictionary mapping agent_id to client instance.
    """
    _config = config or get_config()
    _client = shared_client or AgentClient(_config)
    return {
        agent_id: get_phase2_client(agent_id, _client, _config)
        for agent_id in sorted(PHASE2_CLIENT_REGISTRY.keys())
    }
