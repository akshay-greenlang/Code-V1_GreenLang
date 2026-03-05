"""
CDP Climate Change Disclosure Platform -- Service Facade

This module provides the ``CDPPlatform`` class, a unified facade that composes
all 13 service engines and manages their shared state.  It is the single entry
point for the API layer and external integrations.

Features:
  - Creates and wires all 13 service instances with shared in-memory stores
  - Provides convenience methods that orchestrate multi-engine workflows
  - Manages application lifecycle (initialization, configuration, health check)
  - Computes dashboard metrics from live questionnaire data
  - Runs the full disclosure pipeline (populate -> score -> gap -> benchmark)

Engines wired (13 total):
  1. QuestionnaireEngine     -- CDP questionnaire (13 modules, 200+ questions)
  2. ResponseManager         -- Response lifecycle with versioning and review
  3. ScoringSimulator        -- 17-category CDP scoring engine
  4. DataConnector           -- MRV agent integration (30 agents)
  5. GapAnalysisEngine       -- Gap identification and recommendations
  6. BenchmarkingEngine      -- Sector peer comparison
  7. SupplyChainModule       -- Supplier engagement management
  8. TransitionPlanEngine    -- 1.5C transition plan builder
  9. VerificationTracker     -- Third-party verification management
  10. HistoricalTracker      -- Year-over-year score progression
  11. ReportGenerator        -- Multi-format report generation

Shared stores:
  - questionnaire_store: QuestionnaireEngine owns questionnaires
  - response_store: ResponseManager owns responses
  - question_store: QuestionnaireEngine owns questions

Example:
    >>> from services.setup import CDPPlatform
    >>> platform = CDPPlatform()
    >>> org = platform.create_organization("Acme Corp", "20", "US")
    >>> q = platform.create_questionnaire(org.id, 2026)
    >>> result = platform.run_full_pipeline(q.id, org.id)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from .config import (
    CDPAppConfig,
    MODULE_DEFINITIONS,
    ReportFormat,
    ResponseStatus,
    ScoringLevel,
    SCORING_CATEGORY_WEIGHTS,
    SCORING_LEVEL_THRESHOLDS,
)
from .models import (
    CDPOrganization,
    CreateOrganizationRequest,
    DashboardAlert,
    DashboardMetrics,
    GapSeverity,
    Questionnaire,
    ScoringBand,
    _new_id,
    _now,
)

from .questionnaire_engine import QuestionnaireEngine
from .response_manager import ResponseManager
from .scoring_simulator import ScoringSimulator
from .data_connector import DataConnector
from .gap_analysis_engine import GapAnalysisEngine
from .benchmarking_engine import BenchmarkingEngine
from .supply_chain_module import SupplyChainModule
from .transition_plan_engine import TransitionPlanEngine
from .verification_tracker import VerificationTracker
from .historical_tracker import HistoricalTracker
from .report_generator import ReportGenerator

logger = logging.getLogger(__name__)


class CDPPlatform:
    """
    Unified facade composing all CDP Climate Change Disclosure Platform engines.

    Holds shared in-memory stores and wires every engine to the same data
    references so that changes propagate immediately.  Provides orchestrated
    workflows that span multiple engines (e.g. full scoring pipeline).

    Attributes:
        config: Application configuration.
        questionnaire: CDP questionnaire engine (13 modules, 200+ questions).
        response: Response lifecycle manager.
        scoring: 17-category scoring simulator.
        data_connector: MRV agent data integration.
        gap_analysis: Gap identification engine.
        benchmarking: Sector peer benchmarking.
        supply_chain: Supplier engagement module.
        transition: 1.5C transition plan engine.
        verification: Third-party verification tracker.
        historical: Year-over-year score tracker.
        reporter: Multi-format report generator.

    Example:
        >>> platform = CDPPlatform()
        >>> info = platform.get_platform_info()
        >>> print(info["engine_count"])
        11
    """

    def __init__(self, config: Optional[CDPAppConfig] = None) -> None:
        """
        Initialize the CDP Platform with all 11 engines.

        Args:
            config: Optional configuration override.
        """
        self.config = config or CDPAppConfig()

        # Shared stores
        self._organizations: Dict[str, CDPOrganization] = {}
        self._questionnaires: Dict[str, Questionnaire] = {}
        self._org_questionnaires: Dict[str, List[str]] = {}  # org_id -> [qid]

        # Initialize engines in dependency order
        # Layer 0: No dependencies
        self.questionnaire = QuestionnaireEngine(self.config)
        self.supply_chain = SupplyChainModule(self.config)
        self.transition = TransitionPlanEngine(self.config)
        self.verification = VerificationTracker(self.config)
        self.historical = HistoricalTracker(self.config)
        self.reporter = ReportGenerator(self.config)

        # Layer 1: Depends on questionnaire engine
        self.response = ResponseManager(self.config, self.questionnaire)

        # Layer 2: Depends on questionnaire + response
        self.scoring = ScoringSimulator(self.config, self.questionnaire, self.response)
        self.data_connector = DataConnector(self.config, self.questionnaire, self.response)

        # Layer 3: Depends on questionnaire + response + scoring
        self.gap_analysis = GapAnalysisEngine(
            self.config, self.questionnaire, self.response, self.scoring,
        )
        self.benchmarking = BenchmarkingEngine(self.config, self.scoring)

        logger.info(
            "CDPPlatform v%s initialized with all %d engines",
            self.config.version, 11,
        )

    # ------------------------------------------------------------------
    # Organization Management
    # ------------------------------------------------------------------

    def create_organization(
        self,
        name: str,
        gics_sector: str,
        country: str,
        **kwargs: Any,
    ) -> CDPOrganization:
        """
        Create a new CDP organization.

        Args:
            name: Legal entity name.
            gics_sector: GICS sector code (e.g. "20").
            country: ISO 3166 country code.
            **kwargs: Additional fields (description, contact_person, etc.).

        Returns:
            Created CDPOrganization.
        """
        is_fs = gics_sector == "40"
        org = CDPOrganization(
            name=name,
            gics_sector=gics_sector,
            country=country,
            is_financial_services=is_fs,
            **kwargs,
        )
        self._organizations[org.id] = org
        logger.info(
            "Created organization '%s' (sector=%s, country=%s, id=%s)",
            name, gics_sector, country, org.id,
        )
        return org

    def get_organization(self, org_id: str) -> Optional[CDPOrganization]:
        """Get an organization by ID."""
        return self._organizations.get(org_id)

    def list_organizations(self) -> List[CDPOrganization]:
        """List all registered organizations."""
        return list(self._organizations.values())

    # ------------------------------------------------------------------
    # Questionnaire Management
    # ------------------------------------------------------------------

    def create_questionnaire(
        self,
        org_id: str,
        year: int = 2026,
    ) -> Questionnaire:
        """
        Create a new CDP questionnaire for an organization-year.

        Initializes the questionnaire with all applicable modules and
        questions based on the organization's sector.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            Created Questionnaire.
        """
        org = self._organizations.get(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")

        # Create questionnaire
        q = self.questionnaire.create_questionnaire(org_id, year, org.gics_sector)

        # Store in shared registry
        self._questionnaires[q.id] = q
        if org_id not in self._org_questionnaires:
            self._org_questionnaires[org_id] = []
        self._org_questionnaires[org_id].append(q.id)

        # Set deadline
        q.submission_deadline = date(
            year,
            self.config.submission_deadline_month,
            self.config.submission_deadline_day,
        )

        logger.info(
            "Created questionnaire %s for org %s, year %d",
            q.id, org_id, year,
        )
        return q

    def get_questionnaire(self, questionnaire_id: str) -> Optional[Questionnaire]:
        """Get a questionnaire by ID."""
        return self._questionnaires.get(questionnaire_id)

    def list_questionnaires(self, org_id: str) -> List[Questionnaire]:
        """List questionnaires for an organization."""
        qids = self._org_questionnaires.get(org_id, [])
        return [self._questionnaires[qid] for qid in qids if qid in self._questionnaires]

    # ------------------------------------------------------------------
    # Full Disclosure Pipeline
    # ------------------------------------------------------------------

    def run_full_pipeline(
        self,
        questionnaire_id: str,
        org_id: str,
        target_level: ScoringLevel = ScoringLevel.A,
    ) -> Dict[str, Any]:
        """
        Run the full CDP disclosure pipeline.

        Pipeline steps:
          1. Auto-populate responses from MRV agents
          2. Calculate current scoring (17 categories)
          3. Run gap analysis against target level
          4. Generate sector benchmark
          5. Check verification status
          6. Validate submission readiness

        Args:
            questionnaire_id: Questionnaire ID.
            org_id: Organization ID.
            target_level: Target scoring level for gap analysis.

        Returns:
            Pipeline result with scoring, gaps, benchmark, verification.
        """
        org = self._organizations.get(org_id)
        if not org:
            raise ValueError(f"Organization {org_id} not found")

        questionnaire = self._questionnaires.get(questionnaire_id)
        if not questionnaire:
            raise ValueError(f"Questionnaire {questionnaire_id} not found")

        start_time = _now()

        # Step 1: Auto-populate from MRV agents
        auto_pop_result = self.data_connector.auto_populate(
            questionnaire_id, questionnaire.year, org_id=org_id,
        )

        # Step 2: Calculate scoring
        responses = self.response.get_all_responses(questionnaire_id)
        scoring_result = self.scoring.calculate_score(
            questionnaire_id, org_id=org_id, year=questionnaire.year,
        )

        # Step 3: Gap analysis
        gap_result = self.gap_analysis.analyze(
            questionnaire_id, target_level=target_level,
        )

        # Step 4: Benchmarking
        benchmark_result = self.benchmarking.benchmark_organization(
            questionnaire_id, sector=org.gics_sector,
        )

        # Step 5: Verification status
        verification_status = self.verification.get_verification_status(
            questionnaire_id, org_id,
        )

        # Step 6: Submission validation
        validation = self.reporter.validate_submission(
            questionnaire_id, org_id, responses, questionnaire.year,
        )

        end_time = _now()
        processing_ms = (end_time - start_time).total_seconds() * 1000

        # Extract summary values for logging (handle both dict and model)
        score_pct = (
            scoring_result.overall_score_pct
            if hasattr(scoring_result, "overall_score_pct")
            else 0.0
        )
        gap_count = (
            gap_result.total_gaps
            if hasattr(gap_result, "total_gaps")
            else 0
        )

        logger.info(
            "Full pipeline complete for org %s, questionnaire %s: "
            "score=%.1f%%, gaps=%d, valid=%s, %.0fms",
            org_id, questionnaire_id[:8],
            score_pct, gap_count,
            validation.get("valid", False),
            processing_ms,
        )

        return {
            "questionnaire_id": questionnaire_id,
            "org_id": org_id,
            "auto_population": auto_pop_result,
            "scoring": scoring_result,
            "gap_analysis": gap_result,
            "benchmarking": benchmark_result,
            "verification": verification_status,
            "validation": validation,
            "processing_ms": round(processing_ms, 1),
        }

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def get_dashboard(
        self,
        org_id: str,
        year: int = 2026,
    ) -> DashboardMetrics:
        """
        Generate dashboard metrics for an organization-year.

        Aggregates scoring, completion, gap, and verification data
        into a single dashboard payload for frontend consumption.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            DashboardMetrics for the dashboard.
        """
        org = self._organizations.get(org_id)
        if not org:
            return DashboardMetrics(org_id=org_id, year=year)

        # Find current questionnaire
        questionnaire = None
        for qid in self._org_questionnaires.get(org_id, []):
            q = self._questionnaires.get(qid)
            if q and q.year == year:
                questionnaire = q
                break

        if not questionnaire:
            return DashboardMetrics(org_id=org_id, year=year)

        # Get responses
        responses = self.response.get_all_responses(questionnaire.id)
        total_questions = questionnaire.total_questions
        answered = sum(
            1 for r in responses
            if r.content or r.table_data or r.numeric_value or r.selected_options
        )
        approved = sum(1 for r in responses if r.status == ResponseStatus.APPROVED)

        completion_pct = 0.0
        if total_questions > 0:
            completion_pct = round(answered / total_questions * 100, 1)

        # Module progress
        module_progress: Dict[str, float] = {}
        module_counts: Dict[str, Dict[str, int]] = {}
        for resp in responses:
            mod = resp.module_code.value
            if mod not in module_counts:
                module_counts[mod] = {"total": 0, "answered": 0}
            module_counts[mod]["total"] += 1
            if resp.content or resp.table_data or resp.numeric_value or resp.selected_options:
                module_counts[mod]["answered"] += 1

        for mod, counts in module_counts.items():
            if counts["total"] > 0:
                module_progress[mod] = round(counts["answered"] / counts["total"] * 100, 1)

        # Scoring (if responses exist)
        predicted_score = 0.0
        predicted_level = ScoringLevel.D_MINUS
        predicted_band = ScoringBand.DISCLOSURE
        category_scores: Dict[str, float] = {}
        a_requirements_met = 0

        if answered > 0:
            scoring_result_obj = self.scoring.calculate_score(
                questionnaire.id, org_id=org_id, year=year,
            )
            predicted_score = scoring_result_obj.overall_score_pct
            predicted_level = scoring_result_obj.overall_level
            predicted_band = scoring_result_obj.overall_band

            for cat in scoring_result_obj.category_scores:
                category_scores[cat.category_id] = cat.score_pct

            a_requirements_met = sum(1 for ar in scoring_result_obj.a_requirements if ar.met)

        # Gap summary
        gap_summary: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        if answered > 0:
            gap_result_obj = self.gap_analysis.analyze(questionnaire.id)
            gap_summary = gap_result_obj.gaps_by_severity

        # Days until deadline
        days_until_deadline = None
        if questionnaire.submission_deadline:
            delta = questionnaire.submission_deadline - date.today()
            days_until_deadline = max(0, delta.days)

        # Previous year score
        previous_year_score = None
        score_delta = None
        prev_record = self.historical.get_year_score(org_id, year - 1)
        if prev_record:
            previous_year_score = prev_record.overall_score
            score_delta = round(predicted_score - previous_year_score, 1)

        # Alerts
        alerts = self._generate_dashboard_alerts(
            completion_pct, predicted_score, days_until_deadline,
            a_requirements_met, gap_summary,
        )

        return DashboardMetrics(
            org_id=org_id,
            year=year,
            predicted_score=predicted_score,
            predicted_level=predicted_level,
            predicted_band=predicted_band,
            completion_pct=completion_pct,
            answered_questions=answered,
            total_questions=total_questions,
            approved_questions=approved,
            module_progress=module_progress,
            gap_summary=gap_summary,
            days_until_deadline=days_until_deadline,
            a_requirements_met=a_requirements_met,
            a_requirements_total=5,
            previous_year_score=previous_year_score,
            score_delta=score_delta,
            category_scores=category_scores,
            alerts=alerts,
        )

    # ------------------------------------------------------------------
    # Health Check / Platform Info
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return platform health status."""
        return {
            "status": "healthy",
            "version": self.config.version,
            "standard": "CDP Climate Change Questionnaire 2025/2026",
            "engines": {
                "questionnaire_engine": "ok",
                "response_manager": "ok",
                "scoring_simulator": "ok",
                "data_connector": "ok",
                "gap_analysis_engine": "ok",
                "benchmarking_engine": "ok",
                "supply_chain_module": "ok",
                "transition_plan_engine": "ok",
                "verification_tracker": "ok",
                "historical_tracker": "ok",
                "report_generator": "ok",
            },
            "engine_count": 11,
            "modules": 14,
            "scoring_categories": 17,
            "mrv_agents_integrated": 30,
            "organization_count": len(self._organizations),
            "questionnaire_count": len(self._questionnaires),
        }

    def get_platform_info(self) -> Dict[str, Any]:
        """Return platform metadata."""
        return {
            "name": self.config.app_name,
            "version": self.config.version,
            "standard": "CDP Climate Change Questionnaire 2025/2026",
            "engine_count": 11,
            "modules": 14,
            "module_list": [
                {"code": code, "name": defn["name"]}
                for code, defn in MODULE_DEFINITIONS.items()
            ],
            "scoring_categories": 17,
            "scoring_category_list": [
                {"id": cat_id, "name": cat_data["name"]}
                for cat_id, cat_data in SCORING_CATEGORY_WEIGHTS.items()
            ],
            "scoring_levels": [level.value for level in ScoringLevel],
            "mrv_agents_integrated": 30,
            "a_level_requirements": 5,
            "supported_report_formats": [f.value for f in ReportFormat],
            "aligned_frameworks": [
                "IFRS S2", "ESRS E1", "TCFD", "GRI 305",
                "SBTi", "GHG Protocol", "ISO 14064-1",
            ],
        }

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _generate_dashboard_alerts(
        self,
        completion_pct: float,
        predicted_score: float,
        days_until_deadline: Optional[int],
        a_requirements_met: int,
        gap_summary: Dict[str, int],
    ) -> List[DashboardAlert]:
        """Generate contextual alerts for the dashboard."""
        alerts: List[DashboardAlert] = []

        # Deadline alerts
        if days_until_deadline is not None:
            if days_until_deadline <= 0:
                alerts.append(DashboardAlert(
                    severity=GapSeverity.CRITICAL,
                    title="Submission Deadline Passed",
                    message="The CDP submission deadline has passed. Submit as soon as possible.",
                ))
            elif days_until_deadline <= 7:
                alerts.append(DashboardAlert(
                    severity=GapSeverity.HIGH,
                    title="Deadline Approaching",
                    message=f"Only {days_until_deadline} day(s) remaining until the CDP submission deadline.",
                ))
            elif days_until_deadline <= 30:
                alerts.append(DashboardAlert(
                    severity=GapSeverity.MEDIUM,
                    title="Deadline in 30 Days",
                    message=f"{days_until_deadline} days until the CDP submission deadline.",
                ))

        # Completion alerts
        if completion_pct < 25:
            alerts.append(DashboardAlert(
                severity=GapSeverity.HIGH,
                title="Low Completion",
                message=f"Only {completion_pct:.0f}% of the questionnaire is complete.",
            ))
        elif completion_pct < 50:
            alerts.append(DashboardAlert(
                severity=GapSeverity.MEDIUM,
                title="Partial Completion",
                message=f"Questionnaire is {completion_pct:.0f}% complete. Continue answering questions.",
            ))

        # Critical gaps
        critical_gaps = gap_summary.get("critical", 0)
        if critical_gaps > 0:
            alerts.append(DashboardAlert(
                severity=GapSeverity.CRITICAL,
                title="Critical Gaps Detected",
                message=f"{critical_gaps} critical gap(s) identified that significantly impact your score.",
            ))

        # A-level eligibility
        if predicted_score >= 70 and a_requirements_met < 5:
            remaining = 5 - a_requirements_met
            alerts.append(DashboardAlert(
                severity=GapSeverity.MEDIUM,
                title="A-Level Requirements",
                message=f"Score qualifies for A-/A but {remaining} A-level requirement(s) still unmet.",
            ))

        return alerts
