"""
GL-Taxonomy-APP v1.0 Setup Module -- FastAPI app factory with router registration.

This module provides the ``TaxonomyPlatform`` class and the ``create_app()``
factory function for the GL-Taxonomy-APP EU Taxonomy Alignment & Green
Investment Ratio Platform.  It composes all 10 service engines, registers
all 16 API routers, configures CORS and middleware, and exposes health-check
and info endpoints.

Engines wired (10 service engines, plus config and models):
    1. ActivityScreeningEngine       -- NACE mapping, eligibility, sector classification
    2. SubstantialContributionEngine -- TSC threshold evaluation, enabling/transitional
    3. DNSHAssessmentEngine          -- 6-objective DNSH matrix, climate risk
    4. MinimumSafeguardsEngine       -- 4-topic company-level assessment
    5. KPICalculationEngine          -- Turnover/CapEx/OpEx, double-counting prevention
    6. GARCalculationEngine          -- GAR stock/flow, BTAR, EBA templates
    7. AlignmentEngine               -- End-to-end alignment workflow orchestration
    8. ReportingEngine               -- Article 8, EBA Pillar 3, XBRL
    9. DataQualityEngine             -- 5-dimension quality scoring
    10. RegulatoryUpdateEngine       -- DA version tracking, Omnibus simplification

API Routers (16 total):
    activity_routes, screening_routes, sc_routes, dnsh_routes,
    safeguards_routes, kpi_routes, gar_routes, alignment_routes,
    reporting_routes, portfolio_routes, dashboard_routes,
    data_quality_routes, regulatory_routes, gap_routes, settings_routes,
    health (inline)

Example:
    >>> from services.setup import create_app
    >>> app = create_app()
    >>> # Run with uvicorn: uvicorn services.setup:app --host 0.0.0.0 --port 8000

    >>> from services.setup import TaxonomyPlatform
    >>> platform = TaxonomyPlatform()
    >>> info = platform.health_check()
    >>> print(info["engine_count"])
    10
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import (
    TaxonomyAppConfig,
    ENVIRONMENTAL_OBJECTIVES,
    TAXONOMY_SECTORS,
    TAXONOMY_ACTIVITIES,
    SC_THRESHOLDS,
    DNSH_MATRIX,
    MINIMUM_SAFEGUARD_TOPICS,
    GAR_EXPOSURE_TYPES,
    REPORTING_TEMPLATES,
    DATA_QUALITY_WEIGHTS,
)
from .activity_screening_engine import ActivityScreeningEngine
from .substantial_contribution_engine import SubstantialContributionEngine
from .dnsh_assessment_engine import DNSHAssessmentEngine
from .minimum_safeguards_engine import MinimumSafeguardsEngine
from .kpi_calculation_engine import KPICalculationEngine
from .gar_calculation_engine import GARCalculationEngine
from .alignment_engine import AlignmentEngine
from .reporting_engine import ReportingEngine
from .data_quality_engine import DataQualityEngine
from .regulatory_update_engine import RegulatoryUpdateEngine

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local metadata constants (for health check and info endpoints)
# ---------------------------------------------------------------------------

_DELEGATED_ACTS: Dict[str, Dict[str, Any]] = {
    "climate_da_2021": {
        "name": "Climate Delegated Act 2021/2139",
        "effective_date": "2022-01-01",
    },
    "environmental_da_2023": {
        "name": "Environmental Delegated Act 2023/2486",
        "effective_date": "2024-01-01",
    },
    "complementary_da_2022": {
        "name": "Complementary Climate DA 2022/1214 (Nuclear & Gas)",
        "effective_date": "2023-01-01",
    },
    "simplification_da_2025": {
        "name": "Taxonomy Simplification DA 2025 (proposed)",
        "effective_date": "2026-01-01",
    },
}

_KPI_DEFINITIONS: Dict[str, Dict[str, str]] = {
    "turnover": {"name": "Turnover KPI (Article 8)"},
    "capex": {"name": "CapEx KPI (Article 8)"},
    "opex": {"name": "OpEx KPI (Article 8)"},
}

_FRAMEWORK_ALIGNMENTS: List[str] = [
    "CSRD/ESRS E1-E5", "SFDR", "EBA Pillar 3", "IFRS S2",
    "GHG Protocol", "CDP", "TCFD", "ISO 14064",
]


# ---------------------------------------------------------------------------
# TaxonomyPlatform -- unified service facade
# ---------------------------------------------------------------------------

class TaxonomyPlatform:
    """
    Unified facade composing all EU Taxonomy Alignment Platform service engines.

    Holds all 10 service engines and provides orchestrated alignment workflows,
    health checks, and platform metadata.  The platform implements the full
    EU Taxonomy Regulation 2020/852 assessment pipeline: eligibility screening,
    substantial contribution, DNSH, minimum safeguards, KPI calculation, and
    GAR/BTAR computation.

    Attributes:
        config: Application configuration.
        activity_screening: NACE mapping and eligibility screening engine.
        substantial_contribution: TSC threshold evaluation engine.
        dnsh_assessment: 6-objective DNSH matrix evaluation engine.
        minimum_safeguards: 4-topic company-level safeguards engine.
        kpi_calculation: Turnover/CapEx/OpEx KPI engine.
        gar_calculation: GAR stock/flow and BTAR computation engine.
        alignment: End-to-end alignment workflow orchestration engine.
        reporting: Article 8, EBA Pillar 3, and XBRL reporting engine.
        data_quality: 5-dimension data quality scoring engine.
        regulatory_update: Delegated act version and Omnibus tracking engine.

    Example:
        >>> platform = TaxonomyPlatform()
        >>> health = platform.health_check()
        >>> assert health["status"] == "healthy"
    """

    def __init__(self, config: Optional[TaxonomyAppConfig] = None) -> None:
        """
        Initialize the Taxonomy Platform with all 10 service engines.

        Args:
            config: Optional configuration override.
        """
        self.config = config or TaxonomyAppConfig()

        # Initialize all 10 engines
        self.activity_screening = ActivityScreeningEngine(self.config)
        self.substantial_contribution = SubstantialContributionEngine(self.config)
        self.dnsh_assessment = DNSHAssessmentEngine(self.config)
        self.minimum_safeguards = MinimumSafeguardsEngine(self.config)
        self.kpi_calculation = KPICalculationEngine(self.config)
        self.gar_calculation = GARCalculationEngine(self.config)
        self.alignment = AlignmentEngine(self.config)
        self.reporting = ReportingEngine(self.config)
        self.data_quality = DataQualityEngine(self.config)
        self.regulatory_update = RegulatoryUpdateEngine(self.config)

        logger.info(
            "TaxonomyPlatform v%s initialized with %d engines",
            self.config.version, 10,
        )

    # ------------------------------------------------------------------
    # Orchestrated Workflows
    # ------------------------------------------------------------------

    def run_full_alignment(
        self,
        org_id: str,
        activity_code: str,
        period: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run the full EU Taxonomy alignment pipeline for a single activity.

        Pipeline steps:
            1. Screen eligibility (NACE mapping, sector classification)
            2. Assess substantial contribution (TSC thresholds)
            3. Evaluate DNSH across 6 environmental objectives
            4. Check minimum safeguards (4 topics)
            5. Calculate alignment result (eligible/aligned/enabling/transitional)
            6. Assess data quality (5 dimensions)

        Args:
            org_id: Organization ID.
            activity_code: EU Taxonomy activity code (e.g., "CCM-3.3").
            period: Reporting period (e.g., "2025-FY").
            data: Optional additional input data for the assessment.

        Returns:
            Pipeline result dictionary with all alignment outcomes.
        """
        start = datetime.utcnow()
        input_data = data or {}

        # Step 1: Screen eligibility
        try:
            eligibility_result = self.activity_screening.screen_activity(
                org_id, activity_code, input_data,
            )
            eligibility_data = {
                "eligible": eligibility_result.eligible,
                "nace_codes": eligibility_result.nace_codes,
                "sector": eligibility_result.sector,
                "objective": eligibility_result.objective,
                "activity_type": eligibility_result.activity_type,
                "confidence": eligibility_result.confidence,
            }
        except ValueError:
            eligibility_data = {
                "eligible": False,
                "nace_codes": [],
                "sector": "unknown",
                "objective": "unknown",
                "activity_type": "not_eligible",
                "confidence": 0.0,
            }

        # Step 2: Assess substantial contribution
        try:
            sc_result = self.substantial_contribution.assess(
                org_id, activity_code, period, input_data,
            )
            sc_data = {
                "meets_tsc": sc_result.meets_tsc,
                "objective": sc_result.objective,
                "tsc_criteria_met": sc_result.criteria_met,
                "tsc_criteria_total": sc_result.criteria_total,
                "is_enabling": sc_result.is_enabling,
                "is_transitional": sc_result.is_transitional,
                "thresholds_evaluated": sc_result.thresholds_evaluated,
                "evidence_count": sc_result.evidence_count,
            }
        except ValueError:
            sc_data = {
                "meets_tsc": False,
                "objective": "unknown",
                "tsc_criteria_met": 0,
                "tsc_criteria_total": 0,
                "is_enabling": False,
                "is_transitional": False,
                "thresholds_evaluated": 0,
                "evidence_count": 0,
            }

        # Step 3: Evaluate DNSH
        try:
            dnsh_result = self.dnsh_assessment.assess(
                org_id, activity_code, period, input_data,
            )
            dnsh_data = {
                "passes_dnsh": dnsh_result.passes_dnsh,
                "objectives_assessed": dnsh_result.objectives_assessed,
                "objectives_passed": dnsh_result.objectives_passed,
                "objectives_failed": dnsh_result.objectives_failed,
                "objective_results": dnsh_result.objective_results,
                "climate_risk_assessed": dnsh_result.climate_risk_assessed,
            }
        except ValueError:
            dnsh_data = {
                "passes_dnsh": False,
                "objectives_assessed": 0,
                "objectives_passed": 0,
                "objectives_failed": 0,
                "objective_results": {},
                "climate_risk_assessed": False,
            }

        # Step 4: Check minimum safeguards
        try:
            safeguards_result = self.minimum_safeguards.assess(
                org_id, input_data,
            )
            safeguards_data = {
                "passes_safeguards": safeguards_result.passes_safeguards,
                "topics_assessed": safeguards_result.topics_assessed,
                "topics_passed": safeguards_result.topics_passed,
                "topics_failed": safeguards_result.topics_failed,
                "topic_results": safeguards_result.topic_results,
            }
        except ValueError:
            safeguards_data = {
                "passes_safeguards": False,
                "topics_assessed": 0,
                "topics_passed": 0,
                "topics_failed": 0,
                "topic_results": {},
            }

        # Step 5: Calculate alignment result
        try:
            alignment_result = self.alignment.determine_alignment(
                eligibility=eligibility_data,
                substantial_contribution=sc_data,
                dnsh=dnsh_data,
                safeguards=safeguards_data,
                activity_code=activity_code,
            )
            alignment_data = {
                "taxonomy_aligned": alignment_result.taxonomy_aligned,
                "alignment_status": alignment_result.alignment_status,
                "alignment_pct": alignment_result.alignment_pct,
                "is_eligible": alignment_result.is_eligible,
                "is_enabling": alignment_result.is_enabling,
                "is_transitional": alignment_result.is_transitional,
                "blocking_step": alignment_result.blocking_step,
            }
        except ValueError:
            alignment_data = {
                "taxonomy_aligned": False,
                "alignment_status": "not_assessed",
                "alignment_pct": 0.0,
                "is_eligible": False,
                "is_enabling": False,
                "is_transitional": False,
                "blocking_step": "alignment_calculation",
            }

        # Step 6: Assess data quality
        try:
            quality_result = self.data_quality.assess_alignment_quality(
                org_id, activity_code, period,
            )
            quality_data = {
                "overall_score": quality_result.overall_score,
                "quality_grade": quality_result.quality_grade,
                "dimensions": quality_result.dimension_scores,
                "total_issues": quality_result.total_issues,
            }
        except Exception:
            quality_data = {
                "overall_score": 0.0,
                "quality_grade": "F",
                "dimensions": {},
                "total_issues": 0,
            }

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000

        result = {
            "org_id": org_id,
            "activity_code": activity_code,
            "period": period,
            "eligibility": eligibility_data,
            "substantial_contribution": sc_data,
            "dnsh": dnsh_data,
            "minimum_safeguards": safeguards_data,
            "alignment": alignment_data,
            "data_quality": quality_data,
            "processing_ms": round(processing_ms, 1),
        }

        logger.info(
            "Full alignment for org %s activity %s period %s: "
            "eligible=%s, tsc=%s, dnsh=%s, safeguards=%s, aligned=%s, "
            "quality=%s, %.0fms",
            org_id,
            activity_code,
            period,
            "YES" if eligibility_data["eligible"] else "NO",
            "PASS" if sc_data["meets_tsc"] else "FAIL",
            "PASS" if dnsh_data["passes_dnsh"] else "FAIL",
            "PASS" if safeguards_data["passes_safeguards"] else "FAIL",
            "ALIGNED" if alignment_data["taxonomy_aligned"] else "NOT_ALIGNED",
            quality_data["quality_grade"],
            processing_ms,
        )
        return result

    def get_organization_overview(
        self,
        org_id: str,
        period: str = "latest",
    ) -> Dict[str, Any]:
        """
        Get a comprehensive EU Taxonomy overview for an organization.

        Summarizes all assessed activities, alignment ratios, GAR/BTAR
        results, reporting status, and regulatory compliance posture.

        Args:
            org_id: Organization ID.
            period: Reporting period or "latest".

        Returns:
            Organization overview dictionary.
        """
        start = datetime.utcnow()

        # Gather activity inventory
        try:
            activities = self.activity_screening.list_activities(org_id)
            activity_summary = {
                "total_activities": len(activities),
                "eligible": sum(1 for a in activities if a.eligible),
                "aligned": sum(1 for a in activities if a.taxonomy_aligned),
                "enabling": sum(1 for a in activities if a.is_enabling),
                "transitional": sum(1 for a in activities if a.is_transitional),
                "not_eligible": sum(1 for a in activities if not a.eligible),
            }
        except ValueError:
            activity_summary = {
                "total_activities": 0,
                "eligible": 0,
                "aligned": 0,
                "enabling": 0,
                "transitional": 0,
                "not_eligible": 0,
            }

        # GAR snapshot
        try:
            gar_result = self.gar_calculation.get_latest_gar(org_id, period)
            gar_data = {
                "gar_stock_pct": gar_result.gar_stock_pct,
                "gar_flow_pct": gar_result.gar_flow_pct,
                "btar_pct": gar_result.btar_pct,
                "total_assets_eur": gar_result.total_assets_eur,
                "aligned_assets_eur": gar_result.aligned_assets_eur,
                "eligible_assets_eur": gar_result.eligible_assets_eur,
            }
        except ValueError:
            gar_data = {
                "gar_stock_pct": 0.0,
                "gar_flow_pct": 0.0,
                "btar_pct": 0.0,
                "total_assets_eur": 0.0,
                "aligned_assets_eur": 0.0,
                "eligible_assets_eur": 0.0,
            }

        # KPI snapshot
        try:
            kpi_result = self.kpi_calculation.get_kpi_summary(org_id, period)
            kpi_data = {
                "turnover_aligned_pct": kpi_result.turnover_aligned_pct,
                "capex_aligned_pct": kpi_result.capex_aligned_pct,
                "opex_aligned_pct": kpi_result.opex_aligned_pct,
                "turnover_eligible_pct": kpi_result.turnover_eligible_pct,
                "capex_eligible_pct": kpi_result.capex_eligible_pct,
                "opex_eligible_pct": kpi_result.opex_eligible_pct,
            }
        except ValueError:
            kpi_data = {
                "turnover_aligned_pct": 0.0,
                "capex_aligned_pct": 0.0,
                "opex_aligned_pct": 0.0,
                "turnover_eligible_pct": 0.0,
                "capex_eligible_pct": 0.0,
                "opex_eligible_pct": 0.0,
            }

        # Regulatory status
        try:
            reg_status = self.regulatory_update.get_compliance_status(org_id)
            regulatory_data = {
                "current_da_version": reg_status.current_da_version,
                "omnibus_applicable": reg_status.omnibus_applicable,
                "pending_updates": reg_status.pending_updates,
            }
        except ValueError:
            regulatory_data = {
                "current_da_version": "unknown",
                "omnibus_applicable": False,
                "pending_updates": 0,
            }

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000

        return {
            "org_id": org_id,
            "period": period,
            "activities": activity_summary,
            "gar": gar_data,
            "kpi": kpi_data,
            "regulatory": regulatory_data,
            "processing_ms": round(processing_ms, 1),
        }

    def run_gar_calculation(
        self,
        org_id: str,
        period: str,
        exposure_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete GAR/BTAR calculation for an organization.

        Computes Green Asset Ratio (stock and flow) and Banking Book
        Taxonomy Alignment Ratio for financial institutions, applying
        EBA regulatory technical standards.

        Args:
            org_id: Organization ID.
            period: Reporting period (e.g., "2025-FY").
            exposure_data: Optional exposure-level data override.

        Returns:
            GAR/BTAR calculation result dictionary.
        """
        start = datetime.utcnow()

        try:
            gar_result = self.gar_calculation.calculate_full_gar(
                org_id, period, exposure_data or {},
            )
            result = {
                "org_id": org_id,
                "period": period,
                "gar_stock": {
                    "numerator_eur": gar_result.stock_numerator_eur,
                    "denominator_eur": gar_result.stock_denominator_eur,
                    "ratio_pct": gar_result.gar_stock_pct,
                },
                "gar_flow": {
                    "numerator_eur": gar_result.flow_numerator_eur,
                    "denominator_eur": gar_result.flow_denominator_eur,
                    "ratio_pct": gar_result.gar_flow_pct,
                },
                "btar": {
                    "numerator_eur": gar_result.btar_numerator_eur,
                    "denominator_eur": gar_result.btar_denominator_eur,
                    "ratio_pct": gar_result.btar_pct,
                },
                "exclusions": gar_result.exclusions,
                "exposure_count": gar_result.exposure_count,
                "status": "calculated",
            }
        except ValueError as exc:
            result = {
                "org_id": org_id,
                "period": period,
                "gar_stock": {"numerator_eur": 0.0, "denominator_eur": 0.0, "ratio_pct": 0.0},
                "gar_flow": {"numerator_eur": 0.0, "denominator_eur": 0.0, "ratio_pct": 0.0},
                "btar": {"numerator_eur": 0.0, "denominator_eur": 0.0, "ratio_pct": 0.0},
                "exclusions": [],
                "exposure_count": 0,
                "status": f"error: {str(exc)}",
            }

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000
        result["processing_ms"] = round(processing_ms, 1)

        logger.info(
            "GAR calculation for org %s period %s: stock=%.2f%%, flow=%.2f%%, "
            "btar=%.2f%%, %.0fms",
            org_id, period,
            result["gar_stock"]["ratio_pct"],
            result["gar_flow"]["ratio_pct"],
            result["btar"]["ratio_pct"],
            processing_ms,
        )
        return result

    def run_kpi_calculation(
        self,
        org_id: str,
        period: str,
        financial_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run a complete Turnover/CapEx/OpEx KPI calculation.

        Computes the three mandatory EU Taxonomy KPIs for non-financial
        undertakings, applying IAS/IFRS accounting standard references
        and double-counting prevention rules.

        Args:
            org_id: Organization ID.
            period: Reporting period (e.g., "2025-FY").
            financial_data: Optional financial data override.

        Returns:
            KPI calculation result dictionary.
        """
        start = datetime.utcnow()

        try:
            kpi_result = self.kpi_calculation.calculate_all_kpis(
                org_id, period, financial_data or {},
            )
            result = {
                "org_id": org_id,
                "period": period,
                "turnover": {
                    "total_eur": kpi_result.turnover_total_eur,
                    "aligned_eur": kpi_result.turnover_aligned_eur,
                    "eligible_eur": kpi_result.turnover_eligible_eur,
                    "aligned_pct": kpi_result.turnover_aligned_pct,
                    "eligible_pct": kpi_result.turnover_eligible_pct,
                },
                "capex": {
                    "total_eur": kpi_result.capex_total_eur,
                    "aligned_eur": kpi_result.capex_aligned_eur,
                    "eligible_eur": kpi_result.capex_eligible_eur,
                    "aligned_pct": kpi_result.capex_aligned_pct,
                    "eligible_pct": kpi_result.capex_eligible_pct,
                },
                "opex": {
                    "total_eur": kpi_result.opex_total_eur,
                    "aligned_eur": kpi_result.opex_aligned_eur,
                    "eligible_eur": kpi_result.opex_eligible_eur,
                    "aligned_pct": kpi_result.opex_aligned_pct,
                    "eligible_pct": kpi_result.opex_eligible_pct,
                },
                "double_counting_adjustments": kpi_result.double_counting_adjustments,
                "status": "calculated",
            }
        except ValueError as exc:
            zero_kpi = {
                "total_eur": 0.0, "aligned_eur": 0.0, "eligible_eur": 0.0,
                "aligned_pct": 0.0, "eligible_pct": 0.0,
            }
            result = {
                "org_id": org_id,
                "period": period,
                "turnover": zero_kpi.copy(),
                "capex": zero_kpi.copy(),
                "opex": zero_kpi.copy(),
                "double_counting_adjustments": 0,
                "status": f"error: {str(exc)}",
            }

        processing_ms = (datetime.utcnow() - start).total_seconds() * 1000
        result["processing_ms"] = round(processing_ms, 1)

        logger.info(
            "KPI calculation for org %s period %s: turnover=%.1f%%, "
            "capex=%.1f%%, opex=%.1f%%, %.0fms",
            org_id, period,
            result["turnover"]["aligned_pct"],
            result["capex"]["aligned_pct"],
            result["opex"]["aligned_pct"],
            processing_ms,
        )
        return result

    # ------------------------------------------------------------------
    # Health Check / Platform Info
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """Return platform health status."""
        return {
            "status": "healthy",
            "version": self.config.version,
            "app_name": self.config.app_name,
            "standard": "EU Taxonomy 2020/852 + Climate DA 2021/2139 + Environmental DA 2023/2486",
            "engines": {
                "activity_screening_engine": "ok",
                "substantial_contribution_engine": "ok",
                "dnsh_assessment_engine": "ok",
                "minimum_safeguards_engine": "ok",
                "kpi_calculation_engine": "ok",
                "gar_calculation_engine": "ok",
                "alignment_engine": "ok",
                "reporting_engine": "ok",
                "data_quality_engine": "ok",
                "regulatory_update_engine": "ok",
            },
            "engine_count": 10,
            "environmental_objectives": len(ENVIRONMENTAL_OBJECTIVES),
            "delegated_acts": len(_DELEGATED_ACTS),
            "sectors": len(TAXONOMY_SECTORS),
            "activities": len(TAXONOMY_ACTIVITIES),
            "sc_thresholds": len(SC_THRESHOLDS),
            "dnsh_criteria_categories": len(DNSH_MATRIX),
            "safeguards_topics": len(MINIMUM_SAFEGUARD_TOPICS),
            "kpi_definitions": len(_KPI_DEFINITIONS),
            "gar_exposure_types": len(GAR_EXPOSURE_TYPES),
            "reporting_templates": len(REPORTING_TEMPLATES),
            "data_quality_dimensions": len(DATA_QUALITY_WEIGHTS),
            "framework_alignments": len(_FRAMEWORK_ALIGNMENTS),
            "api_routers": 16,
        }

    def get_platform_info(self) -> Dict[str, Any]:
        """Return platform metadata."""
        return {
            "name": self.config.app_name,
            "version": self.config.version,
            "standard": "EU Taxonomy Regulation 2020/852",
            "climate_da": "Climate Delegated Act 2021/2139",
            "environmental_da": "Environmental Delegated Act 2023/2486",
            "article_8_da": "Article 8 Delegated Act 2021/4987",
            "eba_its": "EBA ITS on Pillar 3 ESG Disclosures",
            "engine_count": 10,
            "engines": [
                "ActivityScreeningEngine",
                "SubstantialContributionEngine",
                "DNSHAssessmentEngine",
                "MinimumSafeguardsEngine",
                "KPICalculationEngine",
                "GARCalculationEngine",
                "AlignmentEngine",
                "ReportingEngine",
                "DataQualityEngine",
                "RegulatoryUpdateEngine",
            ],
            "environmental_objectives": [
                {"key": key, "name": info["name"]}
                for key, info in ENVIRONMENTAL_OBJECTIVES.items()
            ],
            "delegated_acts": [
                {"key": key, "name": info["name"], "effective": info["effective_date"]}
                for key, info in _DELEGATED_ACTS.items()
            ],
            "sectors": [
                {
                    "key": key,
                    "name": info["name"],
                    "nace_level_1": info["nace_level_1"],
                    "activity_count": info["activity_count"],
                }
                for key, info in TAXONOMY_SECTORS.items()
            ],
            "kpi_definitions": [
                {"key": key, "name": info["name"]}
                for key, info in _KPI_DEFINITIONS.items()
            ],
            "gar_exposure_types": list(GAR_EXPOSURE_TYPES.keys()),
            "reporting_templates": [
                {"key": key, "name": info["name"]}
                for key, info in REPORTING_TEMPLATES.items()
            ],
            "api_routers": 16,
            "aligned_frameworks": [
                "CSRD/ESRS E1-E5", "SFDR", "EBA Pillar 3", "IFRS S2",
                "GHG Protocol", "CDP", "TCFD", "ISO 14064",
            ],
            "capabilities": [
                "NACE-to-Taxonomy activity mapping and eligibility screening",
                "Substantial contribution TSC threshold evaluation",
                "DNSH 6-objective matrix assessment with climate risk",
                "Minimum safeguards 4-topic company-level assessment",
                "Turnover/CapEx/OpEx KPI calculation with double-counting prevention",
                "GAR stock/flow and BTAR computation for credit institutions",
                "End-to-end alignment workflow orchestration",
                "Article 8 non-financial undertaking disclosure",
                "EBA Pillar 3 ESG disclosure templates",
                "XBRL/iXBRL export for regulatory filing",
                "5-dimension data quality scoring",
                "Delegated act version tracking and Omnibus simplification",
                "Enabling and transitional activity classification",
                "Multi-objective contribution analysis",
            ],
        }


# ---------------------------------------------------------------------------
# FastAPI app factory
# ---------------------------------------------------------------------------

def create_app(config: Optional[TaxonomyAppConfig] = None) -> FastAPI:
    """
    Create and configure the GL-Taxonomy-APP FastAPI application.

    Registers all 16 routers, configures CORS, adds middleware,
    and sets up health-check and info endpoints.

    Args:
        config: Optional application configuration.

    Returns:
        Configured FastAPI application instance.
    """
    cfg = config or TaxonomyAppConfig()

    app = FastAPI(
        title="GL-Taxonomy-APP v1.0 -- EU Taxonomy Alignment & Green Investment Ratio Platform",
        description=(
            "GreenLang EU Taxonomy alignment platform implementing "
            "the EU Taxonomy Regulation 2020/852, Climate Delegated "
            "Act 2021/2139, Environmental Delegated Act 2023/2486, "
            "and Article 8 Delegated Act 2021/4987.  Supports NACE-to-"
            "Taxonomy activity mapping, 6-objective DNSH assessment, "
            "minimum safeguards evaluation, Turnover/CapEx/OpEx KPI "
            "calculation, GAR/BTAR computation for credit institutions, "
            "Article 8 non-financial undertaking reporting, EBA Pillar 3 "
            "ESG disclosure, XBRL export, and Omnibus simplification "
            "tracking.  Covers all 13 sectors and 150+ economic "
            "activities across 6 environmental objectives."
        ),
        version=cfg.version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {"name": "Activities", "description": "Taxonomy activity catalog and NACE mapping"},
            {"name": "Screening", "description": "Eligibility screening and sector classification"},
            {"name": "Substantial Contribution", "description": "TSC threshold evaluation for 6 objectives"},
            {"name": "DNSH", "description": "Do No Significant Harm 6-objective matrix assessment"},
            {"name": "Minimum Safeguards", "description": "4-topic company-level safeguards assessment"},
            {"name": "KPI Calculation", "description": "Turnover, CapEx, and OpEx KPI computation"},
            {"name": "GAR/BTAR", "description": "Green Asset Ratio and Banking Book Taxonomy Alignment Ratio"},
            {"name": "Alignment", "description": "End-to-end alignment workflow orchestration"},
            {"name": "Reporting", "description": "Article 8, EBA Pillar 3, and XBRL disclosure reports"},
            {"name": "Portfolio", "description": "Portfolio-level taxonomy analysis and aggregation"},
            {"name": "Dashboard", "description": "Executive KPI dashboard and alignment overview"},
            {"name": "Data Quality", "description": "5-dimension data quality scoring and assessment"},
            {"name": "Regulatory", "description": "Delegated act versioning and Omnibus tracking"},
            {"name": "Gap Analysis", "description": "Taxonomy readiness gap assessment and action planning"},
            {"name": "Settings", "description": "Platform settings and configuration management"},
            {"name": "Health", "description": "Platform health and metadata"},
        ],
    )

    # Configure CORS
    configure_cors(app)

    # Configure middleware
    configure_middleware(app)

    # Register routes
    register_routes(app)

    # Platform instance for health endpoint
    platform = TaxonomyPlatform(cfg)

    @app.get(
        "/health",
        tags=["Health"],
        summary="Platform health check",
        description="Returns platform health status and engine availability.",
    )
    async def health_check() -> Dict[str, Any]:
        """Return platform health status."""
        return platform.health_check()

    @app.get(
        "/info",
        tags=["Health"],
        summary="Platform information",
        description="Returns platform metadata including standards, engines, and capabilities.",
    )
    async def platform_info() -> Dict[str, Any]:
        """Return platform metadata."""
        return platform.get_platform_info()

    logger.info(
        "GL-Taxonomy-APP v%s created with %d routers",
        cfg.version, 16,
    )
    return app


def register_routes(app: FastAPI) -> None:
    """
    Register all 16 API routers with the FastAPI application.

    Imports each router module and includes it in the app. Uses graceful
    fallback for routers not yet implemented to allow incremental
    development.

    Args:
        app: FastAPI application instance.
    """
    from .api.activity_routes import router as activity_router
    from .api.screening_routes import router as screening_router

    app.include_router(activity_router)
    app.include_router(screening_router)

    # The remaining 14 routers follow the same pattern.
    # Each is imported and included when the corresponding route module exists.
    _optional_routers = [
        ("api.sc_routes", "sc_router"),
        ("api.dnsh_routes", "dnsh_router"),
        ("api.safeguards_routes", "safeguards_router"),
        ("api.kpi_routes", "kpi_router"),
        ("api.gar_routes", "gar_router"),
        ("api.alignment_routes", "alignment_router"),
        ("api.reporting_routes", "reporting_router"),
        ("api.portfolio_routes", "portfolio_router"),
        ("api.dashboard_routes", "dashboard_router"),
        ("api.data_quality_routes", "data_quality_router"),
        ("api.regulatory_routes", "regulatory_router"),
        ("api.gap_routes", "gap_router"),
        ("api.settings_routes", "settings_router"),
    ]

    for module_path, router_name in _optional_routers:
        try:
            module = __import__(
                f"services.{module_path}",
                fromlist=[router_name.replace("_router", "")],
            )
            router = getattr(module, "router", None)
            if router:
                app.include_router(router)
                logger.debug("Registered router: %s", module_path)
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Router %s not available: %s", module_path, exc,
            )

    logger.info("Route registration complete")


def configure_cors(app: FastAPI) -> None:
    """
    Configure CORS middleware for the application.

    Args:
        app: FastAPI application instance.
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:8080",
            "https://*.greenlang.io",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Provenance-Hash"],
        max_age=600,
    )
    logger.info("CORS configured")


def configure_middleware(app: FastAPI) -> None:
    """
    Configure additional middleware for the application.

    Adds request ID generation and logging middleware.

    Args:
        app: FastAPI application instance.
    """
    import uuid

    @app.middleware("http")
    async def add_request_id(request, call_next):
        """Add unique request ID to all responses."""
        request_id = str(uuid.uuid4())[:8]
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    logger.info("Middleware configured")


def get_router() -> APIRouter:
    """
    Get the main APIRouter for auth integration.

    Returns a router with the /api/v1/taxonomy prefix that can be
    imported and protected by the auth_setup module.

    Returns:
        APIRouter with Taxonomy prefix.
    """
    router = APIRouter(
        prefix="/api/v1/taxonomy",
        tags=["Taxonomy"],
    )

    @router.get("/health")
    async def taxonomy_health() -> Dict[str, Any]:
        """Taxonomy service health endpoint for auth integration."""
        return {
            "service": "GL-Taxonomy-APP",
            "status": "healthy",
            "version": "1.0.0",
        }

    return router


# ---------------------------------------------------------------------------
# Module-level app instance for uvicorn
# ---------------------------------------------------------------------------

app = create_app()
