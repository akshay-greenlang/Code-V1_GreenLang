"""
Metrics & Targets Engine -- TCFD Pillar 4: Climate Metrics and Target Tracking

Implements the TCFD Metrics & Targets recommended disclosures:
  - MT (a): Climate-related metrics used by the organization
  - MT (b): GHG emissions (Scope 1, 2, 3)
  - MT (c): Climate targets and progress tracking

Also implements ISSB/IFRS S2 paragraph 29 cross-industry metrics (7 required)
and SASB/SICS industry-specific metrics for the 11 TCFD sectors.

Provides:
  - GHG emissions recording and retrieval (Scope 1/2/3)
  - Emissions intensity calculation (per revenue, employee, unit)
  - 7 ISSB cross-industry metric tracking
  - SASB/SICS industry-specific metric tracking
  - Custom metric registration and value recording
  - Climate target creation with base/target year
  - Target progress tracking with linear trajectory
  - SBTi alignment assessment
  - Peer benchmarking by sector
  - Implied temperature rise calculation
  - MT (a/b/c) disclosure generation

All calculations are deterministic (zero-hallucination).

Reference:
    - TCFD Final Report, Section G: Metrics and Targets (June 2017)
    - IFRS S2 Paragraphs 29-36 (Metrics and Targets)
    - GHG Protocol Corporate Standard
    - SBTi Criteria and Recommendations v5.1
    - SASB/SICS Standards

Example:
    >>> engine = MetricsTargetsEngine(config)
    >>> await engine.record_emissions("org-1", emissions_data)
    >>> target = await engine.create_target("org-1", target_data)
    >>> progress = await engine.track_target_progress("org-1", target.id, 2025)
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

from .config import (
    ISSB_CROSS_INDUSTRY_METRICS,
    ISSBMetricType,
    MetricCategory,
    MRV_AGENT_TO_TCFD_SCOPE,
    SBTiAlignment,
    SectorType,
    TargetType,
    TCFDAppConfig,
    TCFDPillar,
    TimeHorizon,
)
from .models import (
    ClimateMetric,
    ClimateTarget,
    CreateTargetRequest,
    CrossIndustryMetric,
    EmissionsMetric,
    IntensityMetric,
    MetricValue,
    RecordEmissionsRequest,
    RecordMetricRequest,
    SBTiAssessment,
    TargetProgress,
    _new_id,
    _now,
    _sha256,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# SASB/SICS Industry-Specific Metrics (subset for 11 TCFD sectors)
# ---------------------------------------------------------------------------

SASB_SICS_METRICS: Dict[str, List[Dict[str, str]]] = {
    "energy": [
        {"id": "SASB-EM-01", "name": "Gross Global Scope 1 Emissions", "unit": "tCO2e"},
        {"id": "SASB-EM-02", "name": "Flaring and Venting Emissions", "unit": "tCO2e"},
        {"id": "SASB-EM-03", "name": "Methane Emissions Intensity", "unit": "tCH4/boe"},
        {"id": "SASB-EM-04", "name": "Proved Reserves (Oil)", "unit": "MMbbl"},
        {"id": "SASB-EM-05", "name": "Proved Reserves (Gas)", "unit": "Bcf"},
    ],
    "transportation": [
        {"id": "SASB-TR-01", "name": "Fleet Fuel Efficiency", "unit": "gCO2/pkm"},
        {"id": "SASB-TR-02", "name": "Alternative Fuel Vehicle Share", "unit": "percent"},
        {"id": "SASB-TR-03", "name": "Revenue Ton-Miles", "unit": "million_RTM"},
    ],
    "materials_buildings": [
        {"id": "SASB-MB-01", "name": "Process Emissions Intensity", "unit": "tCO2e/tonne"},
        {"id": "SASB-MB-02", "name": "Energy Intensity of Production", "unit": "GJ/tonne"},
        {"id": "SASB-MB-03", "name": "Green Building Certified Area", "unit": "percent"},
    ],
    "agriculture_food_forest": [
        {"id": "SASB-AF-01", "name": "Scope 1 Agricultural Emissions", "unit": "tCO2e"},
        {"id": "SASB-AF-02", "name": "Water Withdrawal Intensity", "unit": "m3/tonne"},
        {"id": "SASB-AF-03", "name": "Deforestation-Free Sourcing", "unit": "percent"},
    ],
    "banking": [
        {"id": "SASB-BK-01", "name": "Financed Emissions (PCAF)", "unit": "tCO2e"},
        {"id": "SASB-BK-02", "name": "Green Bond Issuance", "unit": "million_USD"},
        {"id": "SASB-BK-03", "name": "Loan Book Alignment", "unit": "percent"},
    ],
    "insurance": [
        {"id": "SASB-IN-01", "name": "Climate-Adjusted Insured Losses", "unit": "million_USD"},
        {"id": "SASB-IN-02", "name": "Physical Risk Modeled Portfolio", "unit": "percent"},
    ],
    "asset_owners": [
        {"id": "SASB-AO-01", "name": "Portfolio Carbon Footprint", "unit": "tCO2e/M_USD"},
        {"id": "SASB-AO-02", "name": "WACI (Weighted Average Carbon Intensity)", "unit": "tCO2e/M_USD_rev"},
        {"id": "SASB-AO-03", "name": "Paris-Aligned AUM", "unit": "percent"},
    ],
    "asset_managers": [
        {"id": "SASB-AM-01", "name": "ESG-Integrated AUM", "unit": "percent"},
        {"id": "SASB-AM-02", "name": "Engagement Outcomes", "unit": "count"},
    ],
    "consumer_goods": [
        {"id": "SASB-CG-01", "name": "Product Lifecycle Emissions", "unit": "tCO2e"},
        {"id": "SASB-CG-02", "name": "Packaging Recyclability Rate", "unit": "percent"},
    ],
    "technology_media": [
        {"id": "SASB-TM-01", "name": "Data Center PUE", "unit": "ratio"},
        {"id": "SASB-TM-02", "name": "Renewable Energy Usage", "unit": "percent"},
        {"id": "SASB-TM-03", "name": "E-Waste Recycled", "unit": "percent"},
    ],
    "healthcare": [
        {"id": "SASB-HC-01", "name": "Facility Energy Intensity", "unit": "kWh/sqft"},
        {"id": "SASB-HC-02", "name": "Anesthetic Gas Emissions", "unit": "tCO2e"},
    ],
}


# ---------------------------------------------------------------------------
# SBTi Criteria Thresholds
# ---------------------------------------------------------------------------

SBTI_CRITERIA: Dict[str, Dict[str, Any]] = {
    "near_term": {
        "min_annual_reduction_1_5c": Decimal("4.2"),
        "min_annual_reduction_wb2c": Decimal("2.5"),
        "max_target_year_offset": 10,
        "scope_1_2_required": True,
        "scope_3_required_pct": Decimal("67"),
        "description": (
            "Near-term SBTi targets must cover at least 95% of Scope 1 & 2 "
            "emissions and 67% of Scope 3 if Scope 3 exceeds 40% of total."
        ),
    },
    "long_term": {
        "min_reduction_1_5c": Decimal("90"),
        "neutralization_required_pct": Decimal("10"),
        "target_year": 2050,
        "scope_coverage": "scope_1_2_3",
        "description": (
            "Long-term SBTi targets require at least 90% reduction in "
            "Scope 1, 2, and 3 by no later than 2050."
        ),
    },
}


class MetricsTargetsEngine:
    """
    TCFD Pillar 4: Metrics and targets engine covering MT (a/b/c).

    Manages GHG emissions data, climate metrics (cross-industry and
    industry-specific), targets, progress tracking, SBTi alignment,
    and peer benchmarking.

    Attributes:
        config: Application configuration.
        _emissions: In-memory emissions store keyed by org_id.
        _metrics: In-memory metric store keyed by org_id.
        _targets: In-memory target store keyed by org_id.
        _progress: In-memory progress store keyed by target_id.
        _cross_industry: Cross-industry metrics by org_id.
    """

    def __init__(self, config: Optional[TCFDAppConfig] = None) -> None:
        """Initialize MetricsTargetsEngine."""
        self.config = config or TCFDAppConfig()
        self._emissions: Dict[str, List[EmissionsMetric]] = {}
        self._metrics: Dict[str, List[ClimateMetric]] = {}
        self._targets: Dict[str, List[ClimateTarget]] = {}
        self._progress: Dict[str, List[TargetProgress]] = {}
        self._cross_industry: Dict[str, List[CrossIndustryMetric]] = {}
        logger.info("MetricsTargetsEngine initialized")

    # ------------------------------------------------------------------
    # GHG Emissions -- MT (b)
    # ------------------------------------------------------------------

    async def record_emissions(
        self,
        org_id: str,
        data: RecordEmissionsRequest,
    ) -> EmissionsMetric:
        """
        Record GHG emissions for a reporting year.

        Args:
            org_id: Organization ID.
            data: Emissions data (Scope 1/2/3).

        Returns:
            Created EmissionsMetric.
        """
        start = datetime.utcnow()

        emissions = EmissionsMetric(
            tenant_id="default",
            org_id=org_id,
            reporting_year=data.reporting_year,
            scope1_tco2e=data.scope1_tco2e,
            scope2_location_tco2e=data.scope2_location_tco2e,
            scope2_market_tco2e=data.scope2_market_tco2e,
            scope3_tco2e=data.scope3_tco2e,
            scope3_by_category=data.scope3_by_category or {},
            methodology=data.methodology,
        )

        if org_id not in self._emissions:
            self._emissions[org_id] = []
        self._emissions[org_id].append(emissions)

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            "Recorded emissions for org %s, year %d: total=%.0f tCO2e in %.1f ms",
            org_id, data.reporting_year, emissions.total_tco2e, elapsed_ms,
        )
        return emissions

    async def get_ghg_emissions(
        self,
        org_id: str,
        year: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Get GHG emissions data, optionally for a specific year.

        Args:
            org_id: Organization ID.
            year: Optional reporting year.

        Returns:
            Dict with emissions breakdown.
        """
        records = self._emissions.get(org_id, [])
        if year:
            records = [r for r in records if r.reporting_year == year]

        if not records:
            return {"org_id": org_id, "message": "No emissions data recorded"}

        latest = max(records, key=lambda r: r.reporting_year)

        all_years = sorted(set(r.reporting_year for r in self._emissions.get(org_id, [])))
        yoy_change = Decimal("0")
        if len(all_years) >= 2:
            prev = [r for r in self._emissions[org_id] if r.reporting_year == all_years[-2]]
            if prev and prev[0].total_tco2e > 0:
                yoy_change = (
                    (latest.total_tco2e - prev[0].total_tco2e)
                    / prev[0].total_tco2e * 100
                ).quantize(Decimal("0.1"))

        return {
            "org_id": org_id,
            "reporting_year": latest.reporting_year,
            "scope_1_tco2e": str(latest.scope1_tco2e),
            "scope_2_location_tco2e": str(latest.scope2_location_tco2e),
            "scope_2_market_tco2e": str(latest.scope2_market_tco2e),
            "scope_3_tco2e": str(latest.scope3_tco2e),
            "scope_3_by_category": {
                str(k): str(v) for k, v in latest.scope3_by_category.items()
            },
            "total_tco2e": str(latest.total_tco2e),
            "biogenic_co2_tco2e": str(latest.biogenic_co2_tco2e),
            "methodology": latest.methodology,
            "verification_status": latest.verification_status,
            "yoy_change_pct": str(yoy_change),
            "available_years": all_years,
        }

    async def get_emissions_time_series(
        self, org_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get emissions as a time series across all available years.

        Args:
            org_id: Organization ID.

        Returns:
            List of yearly emissions records.
        """
        records = sorted(
            self._emissions.get(org_id, []),
            key=lambda r: r.reporting_year,
        )
        return [
            {
                "year": r.reporting_year,
                "scope_1": str(r.scope1_tco2e),
                "scope_2_market": str(r.scope2_market_tco2e),
                "scope_3": str(r.scope3_tco2e),
                "total": str(r.total_tco2e),
            }
            for r in records
        ]

    # ------------------------------------------------------------------
    # Emissions Intensity
    # ------------------------------------------------------------------

    async def calculate_emissions_intensity(
        self,
        org_id: str,
        denominator_value: Decimal,
        denominator_unit: str = "million_usd_revenue",
        scope_coverage: str = "scope_1_2",
        year: Optional[int] = None,
    ) -> IntensityMetric:
        """
        Calculate emissions intensity metric.

        Args:
            org_id: Organization ID.
            denominator_value: Denominator value (e.g. revenue, employees).
            denominator_unit: Unit of denominator.
            scope_coverage: Which scopes to include.
            year: Reporting year (defaults to latest).

        Returns:
            IntensityMetric with computed intensity.
        """
        records = self._emissions.get(org_id, [])
        if year:
            records = [r for r in records if r.reporting_year == year]
        if not records:
            raise ValueError(f"No emissions data for org {org_id}")

        latest = max(records, key=lambda r: r.reporting_year)

        numerator = self._get_scope_total(latest, scope_coverage)

        intensity = IntensityMetric(
            tenant_id="default",
            org_id=org_id,
            metric_name=f"Emissions Intensity ({scope_coverage})",
            numerator_tco2e=numerator,
            denominator_value=denominator_value,
            denominator_unit=denominator_unit,
            reporting_year=latest.reporting_year,
            scope_coverage=scope_coverage,
        )

        logger.info(
            "Calculated intensity for org %s: %.2f %s",
            org_id, intensity.intensity_value,
            f"tCO2e/{denominator_unit}",
        )
        return intensity

    # ------------------------------------------------------------------
    # ISSB Cross-Industry Metrics -- MT (a)
    # ------------------------------------------------------------------

    async def get_cross_industry_metrics(
        self,
        org_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get the 7 ISSB cross-industry metrics with disclosure status.

        Args:
            org_id: Organization ID.

        Returns:
            List of the 7 required metrics with values and status.
        """
        stored = self._cross_industry.get(org_id, [])
        stored_by_type = {m.issb_metric_type: m for m in stored}

        result: List[Dict[str, Any]] = []
        for metric_def in ISSB_CROSS_INDUSTRY_METRICS:
            metric_type = ISSBMetricType(metric_def["id"].replace("ISSB-CI-0", "").lower()
                                         if len(metric_def["id"]) < 15 else "ghg_emissions")

            # Map by index in the list
            idx = ISSB_CROSS_INDUSTRY_METRICS.index(metric_def)
            type_map = [
                ISSBMetricType.GHG_EMISSIONS,
                ISSBMetricType.TRANSITION_RISK_ASSETS,
                ISSBMetricType.PHYSICAL_RISK_ASSETS,
                ISSBMetricType.OPPORTUNITY_REVENUE,
                ISSBMetricType.CAPITAL_DEPLOYMENT,
                ISSBMetricType.INTERNAL_CARBON_PRICE,
                ISSBMetricType.REMUNERATION_LINKED,
            ]
            metric_type = type_map[idx] if idx < len(type_map) else ISSBMetricType.GHG_EMISSIONS

            stored_metric = stored_by_type.get(metric_type)

            result.append({
                "issb_id": metric_def["id"],
                "metric_name": metric_def["name"],
                "ifrs_s2_paragraph": metric_def["ifrs_s2_paragraph"],
                "unit": metric_def["unit"],
                "description": metric_def["description"],
                "value": str(stored_metric.value) if stored_metric else "not_reported",
                "disclosed": stored_metric.disclosed if stored_metric else False,
            })

        disclosed_count = sum(1 for r in result if r["disclosed"])
        logger.info(
            "Cross-industry metrics for org %s: %d/7 disclosed",
            org_id, disclosed_count,
        )
        return result

    async def record_cross_industry_metric(
        self,
        org_id: str,
        metric_type: ISSBMetricType,
        value: Decimal,
        unit: str = "",
        year: Optional[int] = None,
    ) -> CrossIndustryMetric:
        """
        Record a value for one of the 7 ISSB cross-industry metrics.

        Args:
            org_id: Organization ID.
            metric_type: Which ISSB metric.
            value: Metric value.
            unit: Unit of measurement.
            year: Reporting year.

        Returns:
            Created CrossIndustryMetric.
        """
        paragraph_map = {
            ISSBMetricType.GHG_EMISSIONS: "29(a)",
            ISSBMetricType.TRANSITION_RISK_ASSETS: "29(b)",
            ISSBMetricType.PHYSICAL_RISK_ASSETS: "29(c)",
            ISSBMetricType.OPPORTUNITY_REVENUE: "29(d)",
            ISSBMetricType.CAPITAL_DEPLOYMENT: "29(e)",
            ISSBMetricType.INTERNAL_CARBON_PRICE: "29(f)",
            ISSBMetricType.REMUNERATION_LINKED: "29(g)",
        }

        metric = CrossIndustryMetric(
            tenant_id="default",
            org_id=org_id,
            issb_metric_type=metric_type,
            metric_name=metric_type.value,
            value=value,
            unit=unit,
            reporting_year=year or self.config.reporting_year,
            ifrs_s2_paragraph=paragraph_map.get(metric_type, "29"),
            disclosed=True,
        )

        if org_id not in self._cross_industry:
            self._cross_industry[org_id] = []
        existing = [
            i for i, m in enumerate(self._cross_industry[org_id])
            if m.issb_metric_type == metric_type
        ]
        if existing:
            self._cross_industry[org_id][existing[0]] = metric
        else:
            self._cross_industry[org_id].append(metric)

        logger.info(
            "Recorded cross-industry metric %s for org %s: %.2f",
            metric_type.value, org_id, value,
        )
        return metric

    # ------------------------------------------------------------------
    # Industry-Specific Metrics (SASB/SICS)
    # ------------------------------------------------------------------

    async def get_industry_metrics(
        self,
        sector: SectorType,
    ) -> List[Dict[str, str]]:
        """
        Get applicable SASB/SICS industry-specific metrics for a sector.

        Args:
            sector: TCFD sector type.

        Returns:
            List of applicable metrics for the sector.
        """
        return SASB_SICS_METRICS.get(sector.value, [])

    # ------------------------------------------------------------------
    # Custom Metrics
    # ------------------------------------------------------------------

    async def register_custom_metric(
        self,
        org_id: str,
        data: RecordMetricRequest,
    ) -> ClimateMetric:
        """
        Register and record a custom climate metric.

        Args:
            org_id: Organization ID.
            data: Metric recording request.

        Returns:
            Created ClimateMetric.
        """
        metric_value = MetricValue(
            metric_id="pending",
            reporting_year=data.reporting_year,
            value=data.value,
            unit=data.unit,
            data_quality=data.data_quality,
            source=data.source,
        )

        metric = ClimateMetric(
            tenant_id="default",
            org_id=org_id,
            metric_category=data.metric_category,
            issb_metric_type=data.issb_metric_type,
            metric_name=data.metric_name,
            description="",
            unit=data.unit,
            values=[metric_value],
            current_value=data.value,
            reporting_year=data.reporting_year,
            data_quality=data.data_quality,
        )

        if org_id not in self._metrics:
            self._metrics[org_id] = []
        self._metrics[org_id].append(metric)

        logger.info(
            "Registered custom metric '%s' for org %s: %.2f %s",
            data.metric_name, org_id, data.value, data.unit,
        )
        return metric

    async def record_metric_value(
        self,
        org_id: str,
        metric_id: str,
        value: Decimal,
        year: int,
        source: str = "",
    ) -> ClimateMetric:
        """
        Record a new value for an existing metric.

        Args:
            org_id: Organization ID.
            metric_id: Metric ID.
            value: New value.
            year: Reporting year.
            source: Data source.

        Returns:
            Updated ClimateMetric.

        Raises:
            ValueError: If metric not found.
        """
        metrics = self._metrics.get(org_id, [])
        for i, metric in enumerate(metrics):
            if metric.id == metric_id:
                new_val = MetricValue(
                    metric_id=metric_id,
                    reporting_year=year,
                    value=value,
                    unit=metric.unit,
                    source=source,
                )
                data = metric.model_dump()
                data["values"].append(new_val.model_dump())
                data["current_value"] = value
                data["reporting_year"] = year
                data["updated_at"] = _now()
                data["provenance_hash"] = ""
                updated = ClimateMetric(**data)
                self._metrics[org_id][i] = updated
                return updated

        raise ValueError(f"Metric {metric_id} not found")

    # ------------------------------------------------------------------
    # Target Management -- MT (c)
    # ------------------------------------------------------------------

    async def create_target(
        self,
        org_id: str,
        data: CreateTargetRequest,
    ) -> ClimateTarget:
        """
        Create a new climate target.

        Args:
            org_id: Organization ID.
            data: Target creation request.

        Returns:
            Created ClimateTarget.
        """
        reduction_pct = Decimal("0")
        if data.base_value > 0:
            reduction_pct = (
                (data.base_value - data.target_value) / data.base_value * 100
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        target = ClimateTarget(
            tenant_id="default",
            org_id=org_id,
            target_type=data.target_type,
            target_name=data.target_name,
            description=data.description,
            scope_coverage=data.scope_coverage,
            base_year=data.base_year,
            base_value=data.base_value,
            target_year=data.target_year,
            target_value=data.target_value,
            reduction_pct=reduction_pct,
            unit=data.unit,
            interim_milestones=data.interim_milestones or {},
            sbti_alignment=data.sbti_alignment,
        )

        if org_id not in self._targets:
            self._targets[org_id] = []
        self._targets[org_id].append(target)

        logger.info(
            "Created target '%s' for org %s: %.1f%% reduction by %d",
            data.target_name, org_id, reduction_pct, data.target_year,
        )
        return target

    async def list_targets(
        self,
        org_id: str,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[ClimateTarget]:
        """
        List climate targets with optional filters.

        Args:
            org_id: Organization ID.
            filters: Optional filters (target_type, status, sbti_alignment).

        Returns:
            List of ClimateTarget objects.
        """
        targets = list(self._targets.get(org_id, []))
        if filters:
            if "target_type" in filters:
                tt = TargetType(filters["target_type"])
                targets = [t for t in targets if t.target_type == tt]
            if "status" in filters:
                targets = [t for t in targets if t.status == filters["status"]]
            if "sbti_alignment" in filters:
                sbt = SBTiAlignment(filters["sbti_alignment"])
                targets = [t for t in targets if t.sbti_alignment == sbt]
        return targets

    async def track_target_progress(
        self,
        org_id: str,
        target_id: str,
        current_value: Decimal,
        year: Optional[int] = None,
    ) -> TargetProgress:
        """
        Track progress against a climate target.

        Calculates progress percentage, gap, on-track status, and
        required annual reduction rate to meet the target.

        Args:
            org_id: Organization ID.
            target_id: Target ID.
            current_value: Current measured value.
            year: Reporting year.

        Returns:
            TargetProgress record.
        """
        target = self._find_target(org_id, target_id)
        if target is None:
            raise ValueError(f"Target {target_id} not found")

        reporting_year = year or self.config.reporting_year

        total_reduction_needed = target.base_value - target.target_value
        actual_reduction = target.base_value - current_value
        progress_pct = Decimal("0")
        if total_reduction_needed > 0:
            progress_pct = (
                actual_reduction / total_reduction_needed * 100
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

        gap_to_target = current_value - target.target_value
        years_remaining = max(target.target_year - reporting_year, 1)

        expected_linear = self._linear_trajectory(
            target.base_value, target.target_value,
            target.base_year, target.target_year, reporting_year,
        )

        on_track = current_value <= expected_linear

        required_annual = Decimal("0")
        if years_remaining > 0 and current_value > 0:
            required_annual = (
                gap_to_target / current_value / Decimal(str(years_remaining)) * 100
            ).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
            required_annual = max(required_annual, Decimal("0"))

        yoy_change = Decimal("0")
        existing_progress = self._progress.get(target_id, [])
        if existing_progress:
            prev = max(existing_progress, key=lambda p: p.reporting_year)
            if prev.current_value > 0:
                yoy_change = (
                    (current_value - prev.current_value) / prev.current_value * 100
                ).quantize(Decimal("0.1"))

        progress = TargetProgress(
            tenant_id="default",
            target_id=target_id,
            reporting_year=reporting_year,
            current_value=current_value,
            progress_pct=min(progress_pct, Decimal("200")),
            gap_to_target=gap_to_target,
            annual_change_pct=yoy_change,
            on_track=on_track,
            required_annual_reduction_pct=required_annual,
        )

        if target_id not in self._progress:
            self._progress[target_id] = []
        self._progress[target_id].append(progress)

        logger.info(
            "Target '%s' progress: %.1f%%, on_track=%s, gap=%.0f",
            target.target_name, progress_pct, on_track, gap_to_target,
        )
        return progress

    # ------------------------------------------------------------------
    # SBTi Alignment Assessment
    # ------------------------------------------------------------------

    async def assess_sbti_alignment(
        self,
        org_id: str,
    ) -> SBTiAssessment:
        """
        Assess SBTi alignment across all active targets.

        Evaluates near-term and long-term target alignment against
        SBTi criteria for 1.5C and well-below 2C pathways.

        Args:
            org_id: Organization ID.

        Returns:
            SBTiAssessment record.
        """
        targets = self._targets.get(org_id, [])
        active = [t for t in targets if t.status == "active"]

        sbti_targets = [
            t for t in active
            if t.target_type in (TargetType.SCIENCE_BASED, TargetType.ABSOLUTE_REDUCTION)
        ]

        near_term_id = None
        long_term_id = None
        alignment = SBTiAlignment.NOT_ALIGNED
        actual_rate = Decimal("0")
        required_rate = SBTI_CRITERIA["near_term"]["min_annual_reduction_1_5c"]
        recommendations: List[str] = []

        for t in sbti_targets:
            years = max(t.target_year - t.base_year, 1)
            if t.base_value > 0:
                annual_rate = (
                    (t.base_value - t.target_value) / t.base_value / Decimal(str(years)) * 100
                ).quantize(Decimal("0.1"))
            else:
                annual_rate = Decimal("0")

            if t.target_year <= t.base_year + 10:
                near_term_id = t.id
                actual_rate = annual_rate
            else:
                long_term_id = t.id

        if near_term_id:
            if actual_rate >= SBTI_CRITERIA["near_term"]["min_annual_reduction_1_5c"]:
                alignment = SBTiAlignment.ONE_POINT_FIVE_C
            elif actual_rate >= SBTI_CRITERIA["near_term"]["min_annual_reduction_wb2c"]:
                alignment = SBTiAlignment.WELL_BELOW_2C
            else:
                alignment = SBTiAlignment.NOT_ALIGNED
                recommendations.append(
                    f"Increase annual reduction rate from {actual_rate}% to at least "
                    f"{required_rate}% to align with SBTi 1.5C pathway."
                )

        if not near_term_id:
            recommendations.append(
                "Set a near-term science-based target covering Scope 1 and 2 "
                "emissions with a target year within 5-10 years."
            )
        if not long_term_id:
            recommendations.append(
                "Set a long-term target for at least 90% reduction across "
                "Scope 1, 2, and 3 by 2050."
            )

        commitment = "not_committed"
        if near_term_id and long_term_id:
            commitment = "targets_set"
        elif near_term_id or long_term_id:
            commitment = "committed"

        gap = max(required_rate - actual_rate, Decimal("0"))

        assessment = SBTiAssessment(
            tenant_id="default",
            org_id=org_id,
            commitment_status=commitment,
            near_term_target_id=near_term_id,
            long_term_target_id=long_term_id,
            alignment=alignment,
            annual_reduction_rate_pct=actual_rate,
            required_reduction_rate_pct=required_rate,
            gap_pct=gap,
            recommendations=recommendations,
        )

        logger.info(
            "SBTi assessment for org %s: alignment=%s, rate=%.1f%%",
            org_id, alignment.value, actual_rate,
        )
        return assessment

    # ------------------------------------------------------------------
    # Peer Benchmarking
    # ------------------------------------------------------------------

    async def benchmark_against_peers(
        self,
        org_id: str,
        sector: SectorType,
        org_emissions_intensity: Decimal,
    ) -> Dict[str, Any]:
        """
        Benchmark emissions intensity against sector peers.

        Uses sector-level benchmark data to derive percentile ranking.

        Args:
            org_id: Organization ID.
            sector: TCFD sector.
            org_emissions_intensity: Organization intensity value.

        Returns:
            Dict with percentile ranking and peer comparison.
        """
        benchmarks = self._get_sector_benchmarks(sector)
        peer_median = benchmarks["median"]
        peer_p25 = benchmarks["p25"]
        peer_p75 = benchmarks["p75"]

        if org_emissions_intensity <= peer_p25:
            percentile = 25
            performance = "top_quartile"
        elif org_emissions_intensity <= peer_median:
            percentile = 50
            performance = "above_median"
        elif org_emissions_intensity <= peer_p75:
            percentile = 75
            performance = "below_median"
        else:
            percentile = 90
            performance = "bottom_quartile"

        gap_to_median = org_emissions_intensity - peer_median
        gap_pct = Decimal("0")
        if peer_median > 0:
            gap_pct = (gap_to_median / peer_median * 100).quantize(Decimal("0.1"))

        return {
            "org_id": org_id,
            "sector": sector.value,
            "org_intensity": str(org_emissions_intensity),
            "peer_median": str(peer_median),
            "peer_p25": str(peer_p25),
            "peer_p75": str(peer_p75),
            "percentile": percentile,
            "performance_band": performance,
            "gap_to_median": str(gap_to_median),
            "gap_pct": str(gap_pct),
            "recommendation": (
                "Continue best-in-class performance" if percentile <= 25
                else f"Reduce intensity by {abs(gap_pct)}% to reach sector median"
            ),
        }

    # ------------------------------------------------------------------
    # Implied Temperature Rise
    # ------------------------------------------------------------------

    async def calculate_implied_temperature_rise(
        self,
        org_id: str,
    ) -> Dict[str, Any]:
        """
        Calculate implied temperature rise based on emissions trajectory.

        Uses a simplified warming function: total emissions trajectory
        mapped to TCGA carbon budget allocations.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with implied temperature and alignment assessment.
        """
        records = sorted(
            self._emissions.get(org_id, []),
            key=lambda r: r.reporting_year,
        )
        if len(records) < 2:
            return {
                "org_id": org_id,
                "message": "At least 2 years of emissions data required",
            }

        first = records[0]
        last = records[-1]
        years_span = max(last.reporting_year - first.reporting_year, 1)

        if first.total_tco2e > 0:
            annual_change_rate = (
                (last.total_tco2e - first.total_tco2e)
                / first.total_tco2e
                / Decimal(str(years_span))
            )
        else:
            annual_change_rate = Decimal("0")

        projected_2050 = last.total_tco2e * (
            Decimal("1") + annual_change_rate
        ) ** Decimal(str(2050 - last.reporting_year))

        budget_1_5c = Decimal("500000000")
        budget_2_0c = Decimal("1150000000")

        cumulative = Decimal("0")
        current = last.total_tco2e
        for y in range(last.reporting_year, 2051):
            cumulative += current
            current = current * (Decimal("1") + annual_change_rate)

        if cumulative <= 0:
            implied_temp = Decimal("1.4")
        elif cumulative <= budget_1_5c:
            implied_temp = Decimal("1.5")
        elif cumulative <= budget_2_0c:
            ratio = (cumulative - budget_1_5c) / (budget_2_0c - budget_1_5c)
            implied_temp = Decimal("1.5") + ratio * Decimal("0.5")
        else:
            overshoot = cumulative / budget_2_0c
            implied_temp = Decimal("2.0") + (overshoot - Decimal("1")) * Decimal("1.5")

        implied_temp = min(
            implied_temp.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP),
            Decimal("4.0"),
        )

        if implied_temp <= Decimal("1.5"):
            alignment_str = "Paris-aligned (1.5C)"
        elif implied_temp <= Decimal("2.0"):
            alignment_str = "Well-below 2C aligned"
        elif implied_temp <= Decimal("2.5"):
            alignment_str = "Above 2C pathway"
        else:
            alignment_str = "Significantly above 2C"

        return {
            "org_id": org_id,
            "implied_temperature_rise_c": str(implied_temp),
            "alignment": alignment_str,
            "annual_emissions_change_rate": str(
                (annual_change_rate * 100).quantize(Decimal("0.1"))
            ),
            "projected_2050_tco2e": str(projected_2050.quantize(Decimal("0.01"))),
            "cumulative_to_2050_tco2e": str(cumulative.quantize(Decimal("0.01"))),
            "data_years": len(records),
            "methodology": "Simplified TCGA carbon budget approach",
        }

    # ------------------------------------------------------------------
    # Metrics Summary
    # ------------------------------------------------------------------

    async def get_metrics_summary(self, org_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of all metrics and targets.

        Args:
            org_id: Organization ID.

        Returns:
            Dict with emissions, metrics, targets, and compliance status.
        """
        emissions = self._emissions.get(org_id, [])
        metrics = self._metrics.get(org_id, [])
        targets = self._targets.get(org_id, [])
        cross_industry = self._cross_industry.get(org_id, [])

        latest_emissions = max(emissions, key=lambda e: e.reporting_year) if emissions else None

        active_targets = [t for t in targets if t.status == "active"]
        targets_on_track = 0
        for t in active_targets:
            progress_list = self._progress.get(t.id, [])
            if progress_list:
                latest_p = max(progress_list, key=lambda p: p.reporting_year)
                if latest_p.on_track:
                    targets_on_track += 1

        ci_disclosed = sum(1 for m in cross_industry if m.disclosed)

        return {
            "org_id": org_id,
            "emissions": {
                "latest_year": latest_emissions.reporting_year if latest_emissions else None,
                "total_tco2e": str(latest_emissions.total_tco2e) if latest_emissions else "0",
                "scope_1": str(latest_emissions.scope1_tco2e) if latest_emissions else "0",
                "scope_2_market": str(latest_emissions.scope2_market_tco2e) if latest_emissions else "0",
                "scope_3": str(latest_emissions.scope3_tco2e) if latest_emissions else "0",
                "data_years": len(emissions),
            },
            "custom_metrics": {
                "total_registered": len(metrics),
                "cross_industry_disclosed": f"{ci_disclosed}/7",
            },
            "targets": {
                "total": len(targets),
                "active": len(active_targets),
                "on_track": targets_on_track,
                "off_track": len(active_targets) - targets_on_track,
            },
        }

    # ------------------------------------------------------------------
    # Disclosure Generation -- MT (a/b/c)
    # ------------------------------------------------------------------

    async def generate_mt_disclosure(
        self,
        org_id: str,
        year: int,
    ) -> Dict[str, Any]:
        """
        Generate TCFD Metrics & Targets (a), (b), and (c) disclosure content.

        Args:
            org_id: Organization ID.
            year: Reporting year.

        Returns:
            Dict with mt_a, mt_b, mt_c disclosure sections.
        """
        emissions = self._emissions.get(org_id, [])
        metrics = self._metrics.get(org_id, [])
        targets = self._targets.get(org_id, [])
        cross_industry = self._cross_industry.get(org_id, [])

        latest_em = max(emissions, key=lambda e: e.reporting_year) if emissions else None
        ci_disclosed = sum(1 for m in cross_industry if m.disclosed)

        # MT (a): Climate Metrics
        mt_a_content = (
            f"The organization tracks {len(metrics)} climate-related metric(s) "
            f"across cross-industry and industry-specific categories. "
            f"Of the 7 ISSB cross-industry metrics (IFRS S2 para 29), "
            f"{ci_disclosed} have been disclosed."
        )

        # MT (b): GHG Emissions
        if latest_em:
            mt_b_content = (
                f"For the reporting year {latest_em.reporting_year}, the organization "
                f"reports total GHG emissions of {latest_em.total_tco2e:,.0f} tCO2e: "
                f"Scope 1 = {latest_em.scope1_tco2e:,.0f} tCO2e, "
                f"Scope 2 (market-based) = {latest_em.scope2_market_tco2e:,.0f} tCO2e, "
                f"Scope 3 = {latest_em.scope3_tco2e:,.0f} tCO2e. "
                f"Emissions were calculated using the {latest_em.methodology}."
            )
        else:
            mt_b_content = "GHG emissions data has not yet been recorded."

        # MT (c): Targets
        active_targets = [t for t in targets if t.status == "active"]
        sbti_aligned = [
            t for t in active_targets
            if t.sbti_alignment != SBTiAlignment.NOT_ALIGNED
        ]

        if active_targets:
            target_summaries = []
            for t in active_targets[:5]:
                target_summaries.append(
                    f"{t.target_name}: {t.reduction_pct}% reduction by {t.target_year}"
                )
            mt_c_content = (
                f"The organization has set {len(active_targets)} active climate "
                f"target(s), of which {len(sbti_aligned)} are SBTi-aligned. "
                f"Key targets include: {'; '.join(target_summaries)}."
            )
        else:
            mt_c_content = "No active climate targets have been set."

        compliance_a = self._score_mt_a(metrics, cross_industry)
        compliance_b = self._score_mt_b(latest_em)
        compliance_c = self._score_mt_c(active_targets, sbti_aligned)

        return {
            "org_id": org_id,
            "reporting_year": year,
            "mt_a": {
                "ref": "Metrics & Targets (a)",
                "title": "Climate Metrics",
                "content": mt_a_content,
                "compliance_score": compliance_a,
                "total_metrics": len(metrics),
                "cross_industry_disclosed": f"{ci_disclosed}/7",
            },
            "mt_b": {
                "ref": "Metrics & Targets (b)",
                "title": "GHG Emissions",
                "content": mt_b_content,
                "compliance_score": compliance_b,
                "emissions_year": latest_em.reporting_year if latest_em else None,
                "total_tco2e": str(latest_em.total_tco2e) if latest_em else "0",
            },
            "mt_c": {
                "ref": "Metrics & Targets (c)",
                "title": "Targets",
                "content": mt_c_content,
                "compliance_score": compliance_c,
                "active_targets": len(active_targets),
                "sbti_aligned": len(sbti_aligned),
            },
        }

    # ------------------------------------------------------------------
    # Private Helpers
    # ------------------------------------------------------------------

    def _find_target(
        self, org_id: str, target_id: str,
    ) -> Optional[ClimateTarget]:
        """Find target by ID."""
        for target in self._targets.get(org_id, []):
            if target.id == target_id:
                return target
        return None

    @staticmethod
    def _get_scope_total(
        emissions: EmissionsMetric,
        scope_coverage: str,
    ) -> Decimal:
        """Get emissions total for given scope coverage."""
        if scope_coverage == "scope_1":
            return emissions.scope1_tco2e
        if scope_coverage == "scope_2":
            return emissions.scope2_market_tco2e
        if scope_coverage == "scope_1_2":
            return emissions.scope1_tco2e + emissions.scope2_market_tco2e
        if scope_coverage in ("scope_1_2_3", "all"):
            return emissions.total_tco2e
        return emissions.scope1_tco2e + emissions.scope2_market_tco2e

    @staticmethod
    def _linear_trajectory(
        base_value: Decimal,
        target_value: Decimal,
        base_year: int,
        target_year: int,
        current_year: int,
    ) -> Decimal:
        """Calculate expected value on linear reduction trajectory."""
        total_years = max(target_year - base_year, 1)
        elapsed = current_year - base_year
        if elapsed <= 0:
            return base_value
        if elapsed >= total_years:
            return target_value
        annual_reduction = (base_value - target_value) / Decimal(str(total_years))
        return base_value - annual_reduction * Decimal(str(elapsed))

    @staticmethod
    def _get_sector_benchmarks(sector: SectorType) -> Dict[str, Decimal]:
        """Get sector benchmark intensity values (tCO2e/M USD revenue)."""
        benchmarks: Dict[str, Dict[str, Decimal]] = {
            "energy": {"p25": Decimal("120"), "median": Decimal("250"), "p75": Decimal("450")},
            "transportation": {"p25": Decimal("80"), "median": Decimal("160"), "p75": Decimal("280")},
            "materials_buildings": {"p25": Decimal("60"), "median": Decimal("130"), "p75": Decimal("220")},
            "agriculture_food_forest": {"p25": Decimal("40"), "median": Decimal("90"), "p75": Decimal("170")},
            "banking": {"p25": Decimal("3"), "median": Decimal("8"), "p75": Decimal("15")},
            "insurance": {"p25": Decimal("5"), "median": Decimal("12"), "p75": Decimal("22")},
            "asset_owners": {"p25": Decimal("4"), "median": Decimal("10"), "p75": Decimal("20")},
            "asset_managers": {"p25": Decimal("3"), "median": Decimal("7"), "p75": Decimal("14")},
            "consumer_goods": {"p25": Decimal("20"), "median": Decimal("50"), "p75": Decimal("100")},
            "technology_media": {"p25": Decimal("5"), "median": Decimal("15"), "p75": Decimal("30")},
            "healthcare": {"p25": Decimal("10"), "median": Decimal("25"), "p75": Decimal("50")},
        }
        return benchmarks.get(sector.value, {
            "p25": Decimal("20"),
            "median": Decimal("50"),
            "p75": Decimal("100"),
        })

    @staticmethod
    def _score_mt_a(
        metrics: List[ClimateMetric],
        cross_industry: List[CrossIndustryMetric],
    ) -> int:
        """Score MT (a) disclosure completeness (0-100)."""
        score = 0
        if metrics:
            score += 20
        ci_disclosed = sum(1 for m in cross_industry if m.disclosed)
        score += min(ci_disclosed * 10, 50)
        categories = set(m.metric_category for m in metrics)
        if MetricCategory.INDUSTRY_SPECIFIC in categories:
            score += 15
        if MetricCategory.CUSTOM in categories:
            score += 5
        if len(metrics) >= 5:
            score += 10
        return min(score, 100)

    @staticmethod
    def _score_mt_b(latest: Optional[EmissionsMetric]) -> int:
        """Score MT (b) disclosure completeness (0-100)."""
        if latest is None:
            return 0
        score = 10
        if latest.scope1_tco2e > 0:
            score += 20
        if latest.scope2_market_tco2e > 0:
            score += 15
        if latest.scope2_location_tco2e > 0:
            score += 5
        if latest.scope3_tco2e > 0:
            score += 20
        if latest.scope3_by_category:
            score += min(len(latest.scope3_by_category) * 2, 10)
        if latest.verification_status != "not_verified":
            score += 10
        if latest.methodology:
            score += 10
        return min(score, 100)

    @staticmethod
    def _score_mt_c(
        active_targets: List[ClimateTarget],
        sbti_aligned: List[ClimateTarget],
    ) -> int:
        """Score MT (c) disclosure completeness (0-100)."""
        if not active_targets:
            return 0
        score = 15
        if len(active_targets) >= 2:
            score += 10
        types_covered = set(t.target_type for t in active_targets)
        score += min(len(types_covered) * 5, 15)
        scopes = set(t.scope_coverage for t in active_targets)
        if "scope_1_2" in scopes or "all" in scopes:
            score += 10
        if any(s in scopes for s in ("scope_1_2_3", "all")):
            score += 10
        if sbti_aligned:
            score += 20
        with_milestones = [t for t in active_targets if t.interim_milestones]
        if with_milestones:
            score += 10
        with_reduction = [t for t in active_targets if t.reduction_pct > 0]
        if with_reduction:
            score += 10
        return min(score, 100)
