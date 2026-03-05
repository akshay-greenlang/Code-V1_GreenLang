"""
CDP Data Connector -- MRV Agent Data Integration

This module integrates the 30 MRV agents with the CDP questionnaire for
auto-population of Scope 1/2/3 emissions data into M7 table formats.  It maps
MRV output to CDP question formats, performs unit conversion, data freshness
validation, manual override tracking, and reconciliation between auto-populated
and manually entered data.

Key capabilities:
  - Connect to 30 MRV agents for emissions data
  - Map MRV output to CDP M7 table formats (C6.1, C6.3, C6.5)
  - Auto-populate Scope 1/2/3 emissions data
  - Unit conversion per CDP requirements (metric tonnes CO2e)
  - Data freshness validation
  - Manual override tracking with justification
  - Reconciliation engine for discrepancy detection

Example:
    >>> connector = DataConnector(config, questionnaire_engine, response_manager)
    >>> result = connector.auto_populate("q-123", 2026)
    >>> print(f"Populated {result.questions_populated} questions")
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import CDPAppConfig, CDPModule, MRV_AGENT_TO_CDP_SCOPE
from .models import (
    AutoPopulationResult,
    MRVDataPoint,
    Response,
    _new_id,
    _now,
    _sha256,
)
from .questionnaire_engine import QuestionnaireEngine
from .response_manager import ResponseManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MRV Agent ID to internal prefix mapping
# ---------------------------------------------------------------------------

MRV_AGENT_PREFIXES: Dict[str, str] = {
    "MRV-001": "sc",   "MRV-002": "rf",   "MRV-003": "mc",
    "MRV-004": "pe",   "MRV-005": "fue",  "MRV-006": "lu",
    "MRV-007": "wt",   "MRV-008": "ag",   "MRV-009": "s2l",
    "MRV-010": "s2m",  "MRV-011": "shp",  "MRV-012": "cp",
    "MRV-013": "drr",  "MRV-014": "pgs",  "MRV-015": "cg",
    "MRV-016": "fea",  "MRV-017": "uto",  "MRV-018": "wg",
    "MRV-019": "bt",   "MRV-020": "ec",   "MRV-021": "ula",
    "MRV-022": "dto",  "MRV-023": "psp",  "MRV-024": "usp",
    "MRV-025": "eol",  "MRV-026": "dla",  "MRV-027": "frn",
    "MRV-028": "inv",  "MRV-029": "scm",  "MRV-030": "atl",
}


# CDP question to MRV scope mapping
CDP_QUESTION_MRV_MAP: Dict[str, Dict[str, Any]] = {
    "C7.1": {"scope": "scope_1", "agents": ["MRV-001", "MRV-002", "MRV-003", "MRV-004", "MRV-005", "MRV-006", "MRV-007", "MRV-008"], "format": "numeric"},
    "C7.1a": {"scope": "scope_1", "agents": ["MRV-001", "MRV-002", "MRV-003", "MRV-004", "MRV-005"], "format": "table"},
    "C7.1b": {"scope": "scope_1", "agents": ["MRV-001", "MRV-003"], "format": "table"},
    "C7.3": {"scope": "scope_2", "agents": ["MRV-009"], "format": "numeric"},
    "C7.3a": {"scope": "scope_2", "agents": ["MRV-010", "MRV-011", "MRV-012"], "format": "numeric"},
    "C7.6a": {"scope": "scope_3", "agents": ["MRV-014", "MRV-015", "MRV-016", "MRV-017", "MRV-018", "MRV-019", "MRV-020", "MRV-021", "MRV-022", "MRV-023", "MRV-024", "MRV-025", "MRV-026", "MRV-027", "MRV-028"], "format": "table"},
    "C7.13": {"scope": "energy", "agents": ["MRV-009", "MRV-010", "MRV-011", "MRV-012"], "format": "numeric"},
    "C7.14": {"scope": "energy", "agents": ["MRV-010"], "format": "percentage"},
}


class DataConnector:
    """
    CDP Data Connector -- integrates MRV agents with CDP questionnaire.

    Provides auto-population of emissions data from 30 MRV agents,
    unit conversion, data freshness validation, and reconciliation.

    Attributes:
        config: Application configuration.
        questionnaire_engine: Reference to questionnaire engine.
        response_manager: Reference to response manager.
        _mrv_data_cache: Cached MRV data points per organization.
        _override_log: Manual override tracking.

    Example:
        >>> connector = DataConnector(config, q_engine, r_manager)
        >>> result = connector.auto_populate("q-123", 2026)
    """

    def __init__(
        self,
        config: CDPAppConfig,
        questionnaire_engine: QuestionnaireEngine,
        response_manager: ResponseManager,
    ) -> None:
        """Initialize the Data Connector."""
        self.config = config
        self.questionnaire_engine = questionnaire_engine
        self.response_manager = response_manager
        self._mrv_data_cache: Dict[str, List[MRVDataPoint]] = {}
        self._override_log: Dict[str, Dict[str, Any]] = {}
        logger.info(
            "DataConnector initialized with %d MRV agent mappings",
            len(MRV_AGENT_TO_CDP_SCOPE),
        )

    # ------------------------------------------------------------------
    # Auto-Population
    # ------------------------------------------------------------------

    def auto_populate(
        self,
        questionnaire_id: str,
        reporting_year: int,
        org_id: Optional[str] = None,
        force_refresh: bool = False,
    ) -> AutoPopulationResult:
        """
        Auto-populate CDP emissions questions from MRV agent data.

        Fetches data from all 30 MRV agents, converts to CDP formats,
        and saves responses for applicable questions.

        Args:
            questionnaire_id: Target questionnaire ID.
            reporting_year: Reporting year for data filtering.
            org_id: Organization ID for MRV data lookup.
            force_refresh: Force re-fetch even if cached.

        Returns:
            AutoPopulationResult with summary statistics.
        """
        start_time = datetime.utcnow()

        # Fetch MRV data
        cache_key = f"{org_id}:{reporting_year}"
        if force_refresh or cache_key not in self._mrv_data_cache:
            mrv_data = self._fetch_all_mrv_data(org_id or "", reporting_year)
            self._mrv_data_cache[cache_key] = mrv_data
        else:
            mrv_data = self._mrv_data_cache[cache_key]

        # Aggregate by scope
        scope1_total = self._aggregate_scope(mrv_data, "scope_1")
        scope2_location = self._get_agent_total(mrv_data, "MRV-009")
        scope2_market = self._aggregate_scope(mrv_data, "scope_2")
        scope3_total = self._aggregate_scope(mrv_data, "scope_3")
        scope3_by_category = self._aggregate_scope3_by_category(mrv_data)

        # Populate questions
        populated = 0
        skipped = 0

        for question_number, mapping in CDP_QUESTION_MRV_MAP.items():
            try:
                success = self._populate_question(
                    questionnaire_id, question_number, mapping,
                    mrv_data, scope1_total, scope2_location,
                    scope2_market, scope3_total, scope3_by_category,
                )
                if success:
                    populated += 1
                else:
                    skipped += 1
            except Exception as e:
                logger.warning(
                    "Failed to auto-populate %s: %s", question_number, str(e),
                )
                skipped += 1

        # Validate data freshness
        freshness_valid = self._validate_freshness(mrv_data, reporting_year)

        elapsed = (datetime.utcnow() - start_time).total_seconds() * 1000

        result = AutoPopulationResult(
            questionnaire_id=questionnaire_id,
            data_points=mrv_data,
            questions_populated=populated,
            questions_skipped=skipped,
            scope1_total_tco2e=scope1_total,
            scope2_location_tco2e=scope2_location,
            scope2_market_tco2e=scope2_market,
            scope3_total_tco2e=scope3_total,
            scope3_by_category=scope3_by_category,
            data_freshness_valid=freshness_valid,
        )

        logger.info(
            "Auto-populated %d questions for questionnaire %s in %.1f ms "
            "(S1=%.1f, S2L=%.1f, S2M=%.1f, S3=%.1f tCO2e)",
            populated, questionnaire_id, elapsed,
            float(scope1_total), float(scope2_location),
            float(scope2_market), float(scope3_total),
        )
        return result

    # ------------------------------------------------------------------
    # Manual Override
    # ------------------------------------------------------------------

    def register_override(
        self,
        questionnaire_id: str,
        question_id: str,
        original_value: str,
        override_value: str,
        justification: str,
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Register a manual override for an auto-populated value.

        Args:
            questionnaire_id: Questionnaire ID.
            question_id: Question ID.
            original_value: Original auto-populated value.
            override_value: New manually entered value.
            justification: Reason for override.
            user_id: User making the override.

        Returns:
            Override record.
        """
        override_key = f"{questionnaire_id}:{question_id}"
        override_record = {
            "id": _new_id(),
            "questionnaire_id": questionnaire_id,
            "question_id": question_id,
            "original_value": original_value,
            "override_value": override_value,
            "justification": justification,
            "user_id": user_id,
            "timestamp": _now().isoformat(),
            "provenance_hash": _sha256(
                f"{original_value}:{override_value}:{justification}"
            ),
        }
        self._override_log[override_key] = override_record

        # Update the response
        response = self.response_manager.get_response(
            questionnaire_id, question_id,
        )
        if response:
            response.manual_override = True
            response.override_justification = justification
            response.content = override_value
            response.updated_at = _now()

        logger.info(
            "Registered override for %s: %s -> %s (reason: %s)",
            question_id, original_value, override_value, justification,
        )
        return override_record

    def get_overrides(self, questionnaire_id: str) -> List[Dict[str, Any]]:
        """Get all manual overrides for a questionnaire."""
        return [
            v for k, v in self._override_log.items()
            if k.startswith(questionnaire_id)
        ]

    # ------------------------------------------------------------------
    # Reconciliation
    # ------------------------------------------------------------------

    def reconcile(
        self,
        questionnaire_id: str,
        reporting_year: int,
        org_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Reconcile auto-populated data with manually entered data.

        Compares MRV agent values with current response values and
        flags discrepancies.

        Returns:
            Reconciliation result with discrepancy details.
        """
        cache_key = f"{org_id}:{reporting_year}"
        mrv_data = self._mrv_data_cache.get(cache_key, [])

        discrepancies = []
        responses = self.response_manager.get_all_responses(questionnaire_id)
        response_map = {r.question_id: r for r in responses}

        scope1_mrv = self._aggregate_scope(mrv_data, "scope_1")
        scope2_mrv = self._aggregate_scope(mrv_data, "scope_2")
        scope3_mrv = self._aggregate_scope(mrv_data, "scope_3")

        # Check key emission totals
        checks = [
            ("C7.1", scope1_mrv, "Scope 1 total"),
            ("C7.3", self._get_agent_total(mrv_data, "MRV-009"), "Scope 2 location-based"),
            ("C7.3a", scope2_mrv, "Scope 2 market-based"),
        ]

        for q_id, mrv_value, label in checks:
            response = response_map.get(q_id)
            if not response or not response.content:
                continue

            try:
                response_value = Decimal(response.content.strip().replace(",", ""))
                delta = abs(response_value - mrv_value)
                delta_pct = (delta / mrv_value * 100) if mrv_value > 0 else Decimal("0")

                if delta_pct > Decimal("5"):
                    discrepancies.append({
                        "question_id": q_id,
                        "label": label,
                        "mrv_value": float(mrv_value),
                        "response_value": float(response_value),
                        "delta_tco2e": float(delta),
                        "delta_pct": float(delta_pct),
                        "severity": "major" if delta_pct > Decimal("10") else "minor",
                    })
            except (ValueError, ArithmeticError):
                continue

        total_delta = sum(d["delta_tco2e"] for d in discrepancies)
        status = "clean"
        if any(d["severity"] == "major" for d in discrepancies):
            status = "major_discrepancy"
        elif discrepancies:
            status = "minor_differences"

        return {
            "questionnaire_id": questionnaire_id,
            "status": status,
            "discrepancy_count": len(discrepancies),
            "total_delta_tco2e": round(total_delta, 2),
            "discrepancies": discrepancies,
            "reconciled_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Data Freshness
    # ------------------------------------------------------------------

    def check_data_freshness(
        self,
        org_id: str,
        reporting_year: int,
    ) -> Dict[str, Any]:
        """
        Check freshness of MRV data for an organization.

        Data older than the configured threshold is flagged as stale.
        """
        cache_key = f"{org_id}:{reporting_year}"
        mrv_data = self._mrv_data_cache.get(cache_key, [])

        if not mrv_data:
            return {"status": "no_data", "agents_checked": 0, "stale_agents": []}

        max_age_days = 90  # Default freshness threshold
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)

        stale_agents = []
        for dp in mrv_data:
            if dp.data_timestamp < cutoff:
                stale_agents.append({
                    "agent_id": dp.agent_id,
                    "agent_name": dp.agent_name,
                    "last_update": dp.data_timestamp.isoformat(),
                    "days_old": (datetime.utcnow() - dp.data_timestamp).days,
                })

        return {
            "status": "fresh" if not stale_agents else "stale",
            "agents_checked": len(mrv_data),
            "stale_agents": stale_agents,
            "freshness_threshold_days": max_age_days,
            "checked_at": _now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Direct MRV Data Access
    # ------------------------------------------------------------------

    def get_scope_emissions(
        self,
        org_id: str,
        reporting_year: int,
        scope: str,
    ) -> Dict[str, Any]:
        """
        Get emissions data for a specific scope.

        Args:
            org_id: Organization ID.
            reporting_year: Reporting year.
            scope: "scope_1", "scope_2", or "scope_3".

        Returns:
            Emissions breakdown by agent.
        """
        cache_key = f"{org_id}:{reporting_year}"
        mrv_data = self._mrv_data_cache.get(cache_key, [])

        agents = []
        total = Decimal("0")

        for dp in mrv_data:
            if dp.scope == scope:
                agents.append({
                    "agent_id": dp.agent_id,
                    "agent_name": dp.agent_name,
                    "emissions_tco2e": float(dp.emissions_tco2e),
                    "methodology": dp.methodology,
                    "data_quality": dp.data_quality_score,
                })
                total += dp.emissions_tco2e

        return {
            "scope": scope,
            "total_tco2e": float(total),
            "agent_count": len(agents),
            "agents": agents,
        }

    def load_mrv_data(
        self,
        org_id: str,
        reporting_year: int,
        data_points: List[MRVDataPoint],
    ) -> None:
        """
        Load MRV data into the connector cache.

        Used for testing or when MRV agents provide data via API.
        """
        cache_key = f"{org_id}:{reporting_year}"
        self._mrv_data_cache[cache_key] = data_points
        logger.info(
            "Loaded %d MRV data points for org %s year %d",
            len(data_points), org_id, reporting_year,
        )

    # ------------------------------------------------------------------
    # Internal Helpers
    # ------------------------------------------------------------------

    def _fetch_all_mrv_data(
        self,
        org_id: str,
        reporting_year: int,
    ) -> List[MRVDataPoint]:
        """
        Fetch data from all 30 MRV agents.

        In production, this would make HTTP calls to MRV agent APIs.
        Returns cached or simulated data for the current implementation.
        """
        data_points = []

        for agent_id, agent_info in MRV_AGENT_TO_CDP_SCOPE.items():
            scope = agent_info.get("scope", "")
            if scope == "cross_cutting":
                continue

            dp = MRVDataPoint(
                agent_id=agent_id,
                agent_name=agent_info.get("name", ""),
                scope=scope,
                scope3_category=agent_info.get("category"),
                emissions_tco2e=Decimal("0"),
                methodology="calculation_based",
                data_quality_score=0.0,
                reporting_year=reporting_year,
                data_timestamp=_now(),
                is_fresh=True,
                confidence=0.0,
            )
            data_points.append(dp)

        logger.info(
            "Fetched %d MRV data points for org %s year %d",
            len(data_points), org_id, reporting_year,
        )
        return data_points

    def _populate_question(
        self,
        questionnaire_id: str,
        question_number: str,
        mapping: Dict[str, Any],
        mrv_data: List[MRVDataPoint],
        scope1_total: Decimal,
        scope2_location: Decimal,
        scope2_market: Decimal,
        scope3_total: Decimal,
        scope3_by_category: Dict[int, Decimal],
    ) -> bool:
        """Populate a single CDP question from MRV data."""
        fmt = mapping.get("format", "numeric")
        scope = mapping.get("scope", "")

        if fmt == "numeric":
            value = self._get_scope_value(
                scope, scope1_total, scope2_location,
                scope2_market, scope3_total,
            )
            content = str(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

            self.response_manager.save_response(
                questionnaire_id=questionnaire_id,
                question_id=question_number,
                content=content,
                numeric_value=value,
                user_id="data_connector",
            )

            # Mark as auto-populated
            resp = self.response_manager.get_response(
                questionnaire_id, question_number,
            )
            if resp:
                resp.is_auto_populated = True
                resp.auto_populate_source = scope
                resp.confidence = 0.85
            return True

        elif fmt == "table":
            table_data = self._build_table_data(
                scope, mapping.get("agents", []), mrv_data,
                scope3_by_category,
            )
            if not table_data:
                return False

            self.response_manager.save_response(
                questionnaire_id=questionnaire_id,
                question_id=question_number,
                table_data=table_data,
                user_id="data_connector",
            )

            resp = self.response_manager.get_response(
                questionnaire_id, question_number,
            )
            if resp:
                resp.is_auto_populated = True
                resp.auto_populate_source = scope
                resp.confidence = 0.80
            return True

        elif fmt == "percentage":
            # Placeholder for energy percentage calculations
            return False

        return False

    def _get_scope_value(
        self,
        scope: str,
        scope1: Decimal,
        scope2_loc: Decimal,
        scope2_mkt: Decimal,
        scope3: Decimal,
    ) -> Decimal:
        """Get the total value for a scope."""
        if scope == "scope_1":
            return scope1
        elif scope == "scope_2":
            return scope2_loc
        elif scope == "scope_3":
            return scope3
        elif scope == "energy":
            return scope2_loc + scope2_mkt
        return Decimal("0")

    def _build_table_data(
        self,
        scope: str,
        agent_ids: List[str],
        mrv_data: List[MRVDataPoint],
        scope3_by_category: Dict[int, Decimal],
    ) -> List[Dict[str, Any]]:
        """Build CDP table data from MRV agent data."""
        if scope == "scope_3":
            return self._build_scope3_table(scope3_by_category)
        elif scope == "scope_1":
            return self._build_scope1_table(agent_ids, mrv_data)
        elif scope == "scope_2":
            return self._build_scope2_table(agent_ids, mrv_data)
        return []

    def _build_scope1_table(
        self,
        agent_ids: List[str],
        mrv_data: List[MRVDataPoint],
    ) -> List[Dict[str, Any]]:
        """Build Scope 1 emissions breakdown table."""
        rows = []
        for dp in mrv_data:
            if dp.agent_id in agent_ids and dp.scope == "scope_1":
                rows.append({
                    "source": dp.agent_name,
                    "emissions_tco2e": float(dp.emissions_tco2e),
                    "methodology": dp.methodology or "calculation_based",
                    "data_quality": dp.data_quality_score,
                })
        return rows

    def _build_scope2_table(
        self,
        agent_ids: List[str],
        mrv_data: List[MRVDataPoint],
    ) -> List[Dict[str, Any]]:
        """Build Scope 2 emissions breakdown table."""
        rows = []
        for dp in mrv_data:
            if dp.agent_id in agent_ids and dp.scope == "scope_2":
                rows.append({
                    "source": dp.agent_name,
                    "emissions_tco2e": float(dp.emissions_tco2e),
                    "approach": "location_based" if "Location" in dp.agent_name else "market_based",
                })
        return rows

    def _build_scope3_table(
        self,
        scope3_by_category: Dict[int, Decimal],
    ) -> List[Dict[str, Any]]:
        """Build Scope 3 emissions by category table (C6.5 format)."""
        category_names = {
            1: "Purchased goods and services",
            2: "Capital goods",
            3: "Fuel- and energy-related activities",
            4: "Upstream transportation and distribution",
            5: "Waste generated in operations",
            6: "Business travel",
            7: "Employee commuting",
            8: "Upstream leased assets",
            9: "Downstream transportation and distribution",
            10: "Processing of sold products",
            11: "Use of sold products",
            12: "End-of-life treatment of sold products",
            13: "Downstream leased assets",
            14: "Franchises",
            15: "Investments",
        }

        rows = []
        for cat_num in range(1, 16):
            emissions = scope3_by_category.get(cat_num, Decimal("0"))
            rows.append({
                "category": cat_num,
                "category_name": category_names.get(cat_num, f"Category {cat_num}"),
                "emissions_tco2e": float(emissions),
                "evaluation_status": "relevant" if emissions > 0 else "not_evaluated",
                "methodology": "hybrid" if emissions > 0 else "",
            })
        return rows

    def _aggregate_scope(
        self,
        mrv_data: List[MRVDataPoint],
        scope: str,
    ) -> Decimal:
        """Aggregate total emissions for a scope."""
        return sum(
            (dp.emissions_tco2e for dp in mrv_data if dp.scope == scope),
            Decimal("0"),
        )

    def _get_agent_total(
        self,
        mrv_data: List[MRVDataPoint],
        agent_id: str,
    ) -> Decimal:
        """Get total emissions for a specific agent."""
        return sum(
            (dp.emissions_tco2e for dp in mrv_data if dp.agent_id == agent_id),
            Decimal("0"),
        )

    def _aggregate_scope3_by_category(
        self,
        mrv_data: List[MRVDataPoint],
    ) -> Dict[int, Decimal]:
        """Aggregate Scope 3 emissions by category number."""
        by_cat: Dict[int, Decimal] = {}
        for dp in mrv_data:
            if dp.scope == "scope_3" and dp.scope3_category:
                current = by_cat.get(dp.scope3_category, Decimal("0"))
                by_cat[dp.scope3_category] = current + dp.emissions_tco2e
        return by_cat

    def _validate_freshness(
        self,
        mrv_data: List[MRVDataPoint],
        reporting_year: int,
    ) -> bool:
        """Validate that all MRV data is within the reporting period."""
        if not mrv_data:
            return False

        cutoff = datetime(reporting_year, 1, 1) - timedelta(days=90)
        return all(dp.data_timestamp >= cutoff for dp in mrv_data)
