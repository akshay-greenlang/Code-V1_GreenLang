# -*- coding: utf-8 -*-
"""
Taxonomy Reporting Engine - PACK-008 EU Taxonomy Alignment

This module implements the disclosure generation engine for EU Taxonomy
Article 8 Delegated Regulation (EU) 2021/2178 and EBA Pillar 3 ESG
disclosures (CRR Article 449a, EBA ITS Templates 6-10).

The engine generates the mandatory three-table disclosure (Turnover, CapEx,
OpEx aligned/eligible per objective), nuclear/gas supplementary templates,
year-over-year comparison tables, XBRL/iXBRL tag mappings, and EBA
Templates 6 through 10 for credit institutions.

All report generation is deterministic and template-driven. No LLM calls
are used for any numeric outputs.

Example:
    >>> engine = TaxonomyReportingEngine()
    >>> report = engine.generate_article8_disclosure(kpi_data, alignment_data)
    >>> print(f"Tables generated: {len(report.tables)}")
"""

import hashlib
import json
import logging
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class EnvironmentalObjective(str, Enum):
    """EU Taxonomy six environmental objectives."""

    CCM = "CCM"
    CCA = "CCA"
    WTR = "WTR"
    CE = "CE"
    PPC = "PPC"
    BIO = "BIO"


class KPIType(str, Enum):
    """Mandatory KPI types for Article 8 disclosure."""

    TURNOVER = "TURNOVER"
    CAPEX = "CAPEX"
    OPEX = "OPEX"


class DisclosureType(str, Enum):
    """Disclosure document types."""

    ARTICLE_8 = "ARTICLE_8"
    EBA_PILLAR_3 = "EBA_PILLAR_3"
    SUPPLEMENTARY_NUCLEAR_GAS = "SUPPLEMENTARY_NUCLEAR_GAS"


class EBATemplateId(str, Enum):
    """EBA Pillar 3 ESG template identifiers."""

    TEMPLATE_6 = "TEMPLATE_6"   # GAR Summary
    TEMPLATE_7 = "TEMPLATE_7"   # GAR by Sector (NACE)
    TEMPLATE_8 = "TEMPLATE_8"   # BTAR
    TEMPLATE_9 = "TEMPLATE_9"   # GAR Flow
    TEMPLATE_10 = "TEMPLATE_10"  # Other Mitigating Actions


ALL_OBJECTIVES = [
    EnvironmentalObjective.CCM,
    EnvironmentalObjective.CCA,
    EnvironmentalObjective.WTR,
    EnvironmentalObjective.CE,
    EnvironmentalObjective.PPC,
    EnvironmentalObjective.BIO,
]


# ---------------------------------------------------------------------------
# Data Models -- Inputs
# ---------------------------------------------------------------------------


class ActivityKPIData(BaseModel):
    """KPI data for a single economic activity."""

    activity_id: str = Field(..., description="Economic activity identifier")
    activity_name: str = Field(..., description="Activity name")
    nace_code: Optional[str] = Field(None, description="NACE sector code")
    turnover: Decimal = Field(default=Decimal("0"), description="Turnover amount (EUR)")
    capex: Decimal = Field(default=Decimal("0"), description="CapEx amount (EUR)")
    opex: Decimal = Field(default=Decimal("0"), description="OpEx amount (EUR)")
    is_eligible: bool = Field(default=False, description="Taxonomy-eligible flag")
    is_aligned: bool = Field(default=False, description="Taxonomy-aligned flag")
    sc_objective: Optional[EnvironmentalObjective] = Field(
        None, description="SC objective if aligned"
    )
    is_transition: bool = Field(default=False, description="Transition activity")
    is_enabling: bool = Field(default=False, description="Enabling activity")
    is_nuclear: bool = Field(default=False, description="Nuclear activity (Complementary DA)")
    is_gas: bool = Field(default=False, description="Fossil gas activity (Complementary DA)")


class AlignmentData(BaseModel):
    """Portfolio-wide alignment summary data."""

    total_turnover: Decimal = Field(..., description="Total company turnover")
    total_capex: Decimal = Field(..., description="Total company CapEx")
    total_opex: Decimal = Field(..., description="Total company OpEx")
    reporting_period: str = Field(..., description="Reporting period (e.g. '2025')")
    entity_name: str = Field(..., description="Reporting entity name")
    entity_lei: Optional[str] = Field(None, description="LEI code")
    is_financial_institution: bool = Field(
        default=False, description="Whether entity is a financial institution"
    )


class GARInputData(BaseModel):
    """GAR data for EBA template generation."""

    gar_stock_ratio: Decimal = Field(..., description="GAR stock ratio")
    gar_flow_ratio: Decimal = Field(default=Decimal("0"), description="GAR flow ratio")
    btar_ratio: Decimal = Field(default=Decimal("0"), description="BTAR ratio")
    total_covered_assets: Decimal = Field(..., description="Total covered assets (EUR)")
    aligned_assets: Decimal = Field(..., description="Aligned assets (EUR)")
    eligible_assets: Decimal = Field(..., description="Eligible assets (EUR)")
    by_sector: Dict[str, Decimal] = Field(
        default_factory=dict, description="Aligned assets by NACE sector"
    )
    by_exposure_type: Dict[str, Decimal] = Field(
        default_factory=dict, description="Aligned assets by exposure type"
    )
    banking_book_total: Decimal = Field(
        default=Decimal("0"), description="Banking book total"
    )
    banking_book_aligned: Decimal = Field(
        default=Decimal("0"), description="Banking book aligned"
    )
    mitigating_actions: List[Dict[str, Any]] = Field(
        default_factory=list, description="Other mitigating actions not in GAR"
    )


class PriorPeriodData(BaseModel):
    """Prior-period KPI data for year-over-year comparison."""

    period: str = Field(..., description="Prior period label (e.g. '2024')")
    turnover_aligned_ratio: Decimal = Field(..., description="Prior turnover aligned ratio")
    capex_aligned_ratio: Decimal = Field(..., description="Prior CapEx aligned ratio")
    opex_aligned_ratio: Decimal = Field(..., description="Prior OpEx aligned ratio")
    turnover_eligible_ratio: Decimal = Field(..., description="Prior turnover eligible ratio")
    capex_eligible_ratio: Decimal = Field(..., description="Prior CapEx eligible ratio")
    opex_eligible_ratio: Decimal = Field(..., description="Prior OpEx eligible ratio")


# ---------------------------------------------------------------------------
# Data Models -- Outputs
# ---------------------------------------------------------------------------


class TableRow(BaseModel):
    """A single row in a disclosure table."""

    label: str = Field(..., description="Row label / activity name")
    values: Dict[str, Any] = Field(..., description="Column values keyed by column name")


class DisclosureTable(BaseModel):
    """A complete disclosure table."""

    table_id: str = Field(..., description="Table identifier (e.g. 'TABLE_1_TURNOVER')")
    title: str = Field(..., description="Table title")
    kpi_type: Optional[KPIType] = Field(None, description="KPI type if applicable")
    columns: List[str] = Field(..., description="Ordered column names")
    rows: List[TableRow] = Field(..., description="Table rows")
    totals: Dict[str, Any] = Field(default_factory=dict, description="Summary totals row")
    notes: List[str] = Field(default_factory=list, description="Footnotes")


class XBRLTag(BaseModel):
    """XBRL/iXBRL tag mapping."""

    element_name: str = Field(..., description="XBRL element name")
    namespace: str = Field(default="esrs", description="XBRL namespace prefix")
    value: str = Field(..., description="Tag value")
    unit: Optional[str] = Field(None, description="Unit (e.g. 'iso4217:EUR', 'xbrli:pure')")
    context_ref: str = Field(..., description="XBRL context reference")
    decimals: Optional[int] = Field(None, description="Decimal precision")


class XBRLOutput(BaseModel):
    """Collection of XBRL tags for a disclosure."""

    tags: List[XBRLTag] = Field(..., description="XBRL tag list")
    filing_date: str = Field(..., description="Filing date (ISO 8601)")
    entity_lei: Optional[str] = Field(None, description="Entity LEI")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class Article8Report(BaseModel):
    """Complete Article 8 disclosure output."""

    entity_name: str = Field(..., description="Reporting entity")
    reporting_period: str = Field(..., description="Reporting period")
    tables: List[DisclosureTable] = Field(..., description="Mandatory disclosure tables")
    supplementary_tables: List[DisclosureTable] = Field(
        default_factory=list, description="Nuclear/gas supplementary tables"
    )
    yoy_comparison: Optional[DisclosureTable] = Field(
        None, description="Year-over-year comparison table"
    )
    summary: Dict[str, Any] = Field(..., description="KPI summary ratios")
    generation_date: str = Field(..., description="Report generation timestamp")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


class EBATemplate(BaseModel):
    """A single EBA Pillar 3 template."""

    template_id: EBATemplateId = Field(..., description="Template identifier")
    title: str = Field(..., description="Template title")
    columns: List[str] = Field(..., description="Column headers")
    rows: List[TableRow] = Field(..., description="Template rows")
    totals: Dict[str, Any] = Field(default_factory=dict, description="Totals row")
    notes: List[str] = Field(default_factory=list, description="Template notes")


class EBATemplates(BaseModel):
    """Complete EBA Pillar 3 ESG disclosure package."""

    entity_name: str = Field(..., description="Credit institution name")
    reporting_period: str = Field(..., description="Reporting period")
    templates: List[EBATemplate] = Field(..., description="Templates 6 through 10")
    generation_date: str = Field(..., description="Generation timestamp")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")


# ---------------------------------------------------------------------------
# XBRL element mappings (representative subset)
# ---------------------------------------------------------------------------

_XBRL_ELEMENTS: Dict[str, str] = {
    "turnover_aligned_ratio": "esrs:TaxonomyAlignedTurnoverProportion",
    "turnover_eligible_ratio": "esrs:TaxonomyEligibleTurnoverProportion",
    "capex_aligned_ratio": "esrs:TaxonomyAlignedCapExProportion",
    "capex_eligible_ratio": "esrs:TaxonomyEligibleCapExProportion",
    "opex_aligned_ratio": "esrs:TaxonomyAlignedOpExProportion",
    "opex_eligible_ratio": "esrs:TaxonomyEligibleOpExProportion",
    "turnover_aligned_ccm": "esrs:TaxonomyAlignedTurnoverCCM",
    "turnover_aligned_cca": "esrs:TaxonomyAlignedTurnoverCCA",
    "capex_aligned_ccm": "esrs:TaxonomyAlignedCapExCCM",
    "capex_aligned_cca": "esrs:TaxonomyAlignedCapExCCA",
    "gar_stock": "eba:GreenAssetRatioStock",
    "gar_flow": "eba:GreenAssetRatioFlow",
    "btar": "eba:BankingBookTaxonomyAlignmentRatio",
    "nuclear_turnover": "esrs:NuclearActivityTurnoverProportion",
    "gas_turnover": "esrs:FossilGasActivityTurnoverProportion",
}


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class TaxonomyReportingEngine:
    """
    Taxonomy Reporting Engine for Article 8 and EBA Pillar 3 disclosures.

    Generates mandatory three-table disclosures (Turnover, CapEx, OpEx),
    nuclear/gas supplementary templates, year-over-year comparisons,
    XBRL tag mappings, and EBA Templates 6 through 10.

    Attributes:
        xbrl_elements: XBRL element namespace mappings

    Example:
        >>> engine = TaxonomyReportingEngine()
        >>> report = engine.generate_article8_disclosure(activities, alignment)
        >>> assert len(report.tables) == 3
    """

    def __init__(self) -> None:
        """Initialize the Taxonomy Reporting Engine."""
        self.xbrl_elements = _XBRL_ELEMENTS
        logger.info("TaxonomyReportingEngine initialized")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_article8_disclosure(
        self,
        kpi_data: List[ActivityKPIData],
        alignment_data: AlignmentData,
        prior_period: Optional[PriorPeriodData] = None,
    ) -> Article8Report:
        """
        Generate Article 8 Delegated Regulation (EU) 2021/2178 disclosure.

        Produces three mandatory tables (Turnover, CapEx, OpEx) with per-objective
        eligible/aligned breakdowns, optional nuclear/gas supplementary tables, and
        optional year-over-year comparison.

        Args:
            kpi_data: Per-activity KPI data
            alignment_data: Portfolio-wide alignment summary
            prior_period: Optional prior-period data for YoY comparison

        Returns:
            Article8Report with all disclosure tables

        Raises:
            ValueError: If kpi_data is empty
        """
        if not kpi_data:
            raise ValueError("KPI data cannot be empty")

        start = datetime.utcnow()
        logger.info(
            f"Generating Article 8 disclosure for {alignment_data.entity_name} "
            f"({len(kpi_data)} activities)"
        )

        # Mandatory tables
        table_turnover = self._build_kpi_table(
            kpi_data, alignment_data, KPIType.TURNOVER
        )
        table_capex = self._build_kpi_table(
            kpi_data, alignment_data, KPIType.CAPEX
        )
        table_opex = self._build_kpi_table(
            kpi_data, alignment_data, KPIType.OPEX
        )

        # Supplementary nuclear/gas
        supplementary = self._build_nuclear_gas_tables(kpi_data, alignment_data)

        # YoY comparison
        yoy = None
        if prior_period:
            yoy = self._build_yoy_table(kpi_data, alignment_data, prior_period)

        # Summary ratios
        summary = self._compute_summary(kpi_data, alignment_data)

        provenance = self._provenance({
            "type": "article8",
            "entity": alignment_data.entity_name,
            "period": alignment_data.reporting_period,
            "activities": len(kpi_data),
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(
            f"Article 8 disclosure generated in {elapsed_ms:.1f}ms: "
            f"3 mandatory + {len(supplementary)} supplementary tables"
        )

        return Article8Report(
            entity_name=alignment_data.entity_name,
            reporting_period=alignment_data.reporting_period,
            tables=[table_turnover, table_capex, table_opex],
            supplementary_tables=supplementary,
            yoy_comparison=yoy,
            summary=summary,
            generation_date=start.isoformat(),
            provenance_hash=provenance,
        )

    def generate_eba_templates(
        self,
        gar_data: GARInputData,
        entity_name: str,
        reporting_period: str,
    ) -> EBATemplates:
        """
        Generate EBA Pillar 3 ESG disclosure Templates 6-10.

        Args:
            gar_data: GAR/BTAR data and breakdowns
            entity_name: Credit institution name
            reporting_period: Reporting period label

        Returns:
            EBATemplates with Templates 6 through 10
        """
        start = datetime.utcnow()
        logger.info(f"Generating EBA Templates 6-10 for {entity_name}")

        t6 = self._build_template_6(gar_data, entity_name)
        t7 = self._build_template_7(gar_data, entity_name)
        t8 = self._build_template_8(gar_data, entity_name)
        t9 = self._build_template_9(gar_data, entity_name)
        t10 = self._build_template_10(gar_data, entity_name)

        provenance = self._provenance({
            "type": "eba_templates",
            "entity": entity_name,
            "period": reporting_period,
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(f"EBA Templates generated in {elapsed_ms:.1f}ms")

        return EBATemplates(
            entity_name=entity_name,
            reporting_period=reporting_period,
            templates=[t6, t7, t8, t9, t10],
            generation_date=start.isoformat(),
            provenance_hash=provenance,
        )

    def generate_xbrl_tags(
        self,
        report: Article8Report,
    ) -> XBRLOutput:
        """
        Generate XBRL/iXBRL tags for an Article 8 disclosure report.

        Maps KPI summary ratios and per-objective breakdowns to the ESRS/EBA
        XBRL taxonomy elements.

        Args:
            report: Completed Article 8 report

        Returns:
            XBRLOutput with tag list
        """
        start = datetime.utcnow()
        logger.info("Generating XBRL tags for Article 8 report")

        tags: List[XBRLTag] = []
        context_ref = f"ctx_{report.reporting_period}"

        # Map summary ratios
        for key, element in self.xbrl_elements.items():
            value = report.summary.get(key)
            if value is not None:
                is_ratio = "ratio" in key.lower() or "proportion" in element.lower()
                tags.append(XBRLTag(
                    element_name=element,
                    namespace="esrs" if element.startswith("esrs:") else "eba",
                    value=str(value),
                    unit="xbrli:pure" if is_ratio else "iso4217:EUR",
                    context_ref=context_ref,
                    decimals=6 if is_ratio else 2,
                ))

        provenance = self._provenance({
            "type": "xbrl",
            "period": report.reporting_period,
            "tag_count": len(tags),
            "ts": start.isoformat(),
        })

        elapsed_ms = (datetime.utcnow() - start).total_seconds() * 1000
        logger.info(f"Generated {len(tags)} XBRL tags in {elapsed_ms:.1f}ms")

        return XBRLOutput(
            tags=tags,
            filing_date=start.isoformat(),
            entity_lei=None,
            provenance_hash=provenance,
        )

    # ------------------------------------------------------------------
    # Article 8 table builders
    # ------------------------------------------------------------------

    def _build_kpi_table(
        self,
        kpi_data: List[ActivityKPIData],
        alignment_data: AlignmentData,
        kpi_type: KPIType,
    ) -> DisclosureTable:
        """Build one of the three mandatory KPI tables (Turnover / CapEx / OpEx)."""
        table_num = {"TURNOVER": "1", "CAPEX": "2", "OPEX": "3"}[kpi_type.value]
        table_id = f"TABLE_{table_num}_{kpi_type.value}"
        title = f"Table {table_num} - Proportion of {kpi_type.value} from taxonomy-aligned and eligible activities"

        columns = [
            "Economic Activity",
            "NACE Code",
            "Absolute Amount (EUR)",
            "Proportion of Total (%)",
        ]
        for obj in ALL_OBJECTIVES:
            columns.append(f"SC {obj.value} (%)")
        columns += [
            "Taxonomy-Aligned (%)",
            "Taxonomy-Eligible (%)",
            "Category (T/E)",
        ]

        total_kpi = self._get_total_kpi(alignment_data, kpi_type)
        rows: List[TableRow] = []

        # Aggregate per-objective totals
        obj_aligned: Dict[str, Decimal] = {o.value: Decimal("0") for o in ALL_OBJECTIVES}
        total_eligible = Decimal("0")
        total_aligned = Decimal("0")

        for activity in kpi_data:
            amount = self._get_activity_kpi(activity, kpi_type)
            proportion = self._pct(amount, total_kpi)

            # Per-objective SC contribution
            obj_values: Dict[str, Any] = {}
            for obj in ALL_OBJECTIVES:
                if activity.is_aligned and activity.sc_objective == obj:
                    obj_values[f"SC {obj.value} (%)"] = str(proportion)
                    obj_aligned[obj.value] += amount
                else:
                    obj_values[f"SC {obj.value} (%)"] = "0"

            aligned_pct = str(proportion) if activity.is_aligned else "0"
            eligible_pct = str(proportion) if activity.is_eligible else "0"

            if activity.is_aligned:
                total_aligned += amount
            if activity.is_eligible:
                total_eligible += amount

            category = ""
            if activity.is_transition:
                category = "T"
            elif activity.is_enabling:
                category = "E"

            row_values = {
                "Economic Activity": activity.activity_name,
                "NACE Code": activity.nace_code or "",
                "Absolute Amount (EUR)": str(amount),
                "Proportion of Total (%)": str(proportion),
                "Taxonomy-Aligned (%)": aligned_pct,
                "Taxonomy-Eligible (%)": eligible_pct,
                "Category (T/E)": category,
            }
            row_values.update(obj_values)
            rows.append(TableRow(label=activity.activity_name, values=row_values))

        # Totals row
        totals: Dict[str, Any] = {
            "Taxonomy-Aligned (%)": str(self._pct(total_aligned, total_kpi)),
            "Taxonomy-Eligible (%)": str(self._pct(total_eligible, total_kpi)),
            "Absolute Aligned (EUR)": str(total_aligned),
            "Absolute Eligible (EUR)": str(total_eligible),
        }
        for obj in ALL_OBJECTIVES:
            totals[f"SC {obj.value} (%)"] = str(
                self._pct(obj_aligned[obj.value], total_kpi)
            )

        return DisclosureTable(
            table_id=table_id,
            title=title,
            kpi_type=kpi_type,
            columns=columns,
            rows=rows,
            totals=totals,
        )

    def _build_nuclear_gas_tables(
        self,
        kpi_data: List[ActivityKPIData],
        alignment_data: AlignmentData,
    ) -> List[DisclosureTable]:
        """Build nuclear/gas supplementary tables per Complementary DA."""
        nuclear_activities = [a for a in kpi_data if a.is_nuclear]
        gas_activities = [a for a in kpi_data if a.is_gas]

        tables: List[DisclosureTable] = []

        if nuclear_activities:
            tables.append(self._build_supplementary(
                nuclear_activities, alignment_data, "NUCLEAR"
            ))
        if gas_activities:
            tables.append(self._build_supplementary(
                gas_activities, alignment_data, "GAS"
            ))

        return tables

    def _build_supplementary(
        self,
        activities: List[ActivityKPIData],
        alignment_data: AlignmentData,
        category: str,
    ) -> DisclosureTable:
        """Build a single nuclear or gas supplementary table."""
        table_id = f"SUPPLEMENTARY_{category}"
        title = f"Supplementary disclosure - {category.lower()} energy activities"
        columns = [
            "Economic Activity",
            "Turnover (EUR)",
            "Turnover (%)",
            "CapEx (EUR)",
            "CapEx (%)",
            "OpEx (EUR)",
            "OpEx (%)",
        ]

        rows: List[TableRow] = []
        for a in activities:
            rows.append(TableRow(
                label=a.activity_name,
                values={
                    "Economic Activity": a.activity_name,
                    "Turnover (EUR)": str(a.turnover),
                    "Turnover (%)": str(self._pct(a.turnover, alignment_data.total_turnover)),
                    "CapEx (EUR)": str(a.capex),
                    "CapEx (%)": str(self._pct(a.capex, alignment_data.total_capex)),
                    "OpEx (EUR)": str(a.opex),
                    "OpEx (%)": str(self._pct(a.opex, alignment_data.total_opex)),
                },
            ))

        return DisclosureTable(
            table_id=table_id,
            title=title,
            columns=columns,
            rows=rows,
            notes=[
                f"Activities disclosed under Complementary Climate DA (EU) 2022/1214 "
                f"({category.lower()} category)."
            ],
        )

    def _build_yoy_table(
        self,
        kpi_data: List[ActivityKPIData],
        alignment_data: AlignmentData,
        prior: PriorPeriodData,
    ) -> DisclosureTable:
        """Build year-over-year comparison table."""
        current_summary = self._compute_summary(kpi_data, alignment_data)

        columns = [
            "KPI",
            f"Current ({alignment_data.reporting_period})",
            f"Prior ({prior.period})",
            "Change (pp)",
        ]
        rows: List[TableRow] = []

        comparisons = [
            ("Turnover Aligned (%)", "turnover_aligned_ratio",
             prior.turnover_aligned_ratio),
            ("Turnover Eligible (%)", "turnover_eligible_ratio",
             prior.turnover_eligible_ratio),
            ("CapEx Aligned (%)", "capex_aligned_ratio",
             prior.capex_aligned_ratio),
            ("CapEx Eligible (%)", "capex_eligible_ratio",
             prior.capex_eligible_ratio),
            ("OpEx Aligned (%)", "opex_aligned_ratio",
             prior.opex_aligned_ratio),
            ("OpEx Eligible (%)", "opex_eligible_ratio",
             prior.opex_eligible_ratio),
        ]

        for label, key, prior_val in comparisons:
            current_val = Decimal(str(current_summary.get(key, 0)))
            change = current_val - prior_val
            rows.append(TableRow(
                label=label,
                values={
                    "KPI": label,
                    f"Current ({alignment_data.reporting_period})": str(current_val),
                    f"Prior ({prior.period})": str(prior_val),
                    "Change (pp)": str(change),
                },
            ))

        return DisclosureTable(
            table_id="YOY_COMPARISON",
            title="Year-over-Year KPI Comparison",
            columns=columns,
            rows=rows,
            notes=["Change expressed in percentage points (pp)."],
        )

    # ------------------------------------------------------------------
    # EBA Template builders
    # ------------------------------------------------------------------

    def _build_template_6(
        self, gar_data: GARInputData, entity: str
    ) -> EBATemplate:
        """Template 6 - GAR Summary."""
        columns = [
            "Metric",
            "Total Covered Assets (EUR)",
            "Taxonomy-Eligible (EUR)",
            "Taxonomy-Aligned (EUR)",
            "GAR (%)",
        ]
        rows = [
            TableRow(label="GAR Stock", values={
                "Metric": "Green Asset Ratio (Stock)",
                "Total Covered Assets (EUR)": str(gar_data.total_covered_assets),
                "Taxonomy-Eligible (EUR)": str(gar_data.eligible_assets),
                "Taxonomy-Aligned (EUR)": str(gar_data.aligned_assets),
                "GAR (%)": str(gar_data.gar_stock_ratio),
            }),
        ]
        return EBATemplate(
            template_id=EBATemplateId.TEMPLATE_6,
            title="Template 6 - Summary of GAR (Green Asset Ratio)",
            columns=columns,
            rows=rows,
            totals={
                "GAR (%)": str(gar_data.gar_stock_ratio),
            },
        )

    def _build_template_7(
        self, gar_data: GARInputData, entity: str
    ) -> EBATemplate:
        """Template 7 - GAR by Sector (NACE)."""
        columns = ["NACE Sector", "Aligned Assets (EUR)", "Proportion of GAR (%)"]
        rows: List[TableRow] = []

        total_aligned = gar_data.aligned_assets or Decimal("1")
        for sector, amount in sorted(gar_data.by_sector.items()):
            pct = self._pct(amount, total_aligned)
            rows.append(TableRow(
                label=sector,
                values={
                    "NACE Sector": sector,
                    "Aligned Assets (EUR)": str(amount),
                    "Proportion of GAR (%)": str(pct),
                },
            ))

        return EBATemplate(
            template_id=EBATemplateId.TEMPLATE_7,
            title="Template 7 - GAR by NACE Sector",
            columns=columns,
            rows=rows,
        )

    def _build_template_8(
        self, gar_data: GARInputData, entity: str
    ) -> EBATemplate:
        """Template 8 - BTAR."""
        columns = [
            "Metric",
            "Banking Book Total (EUR)",
            "Banking Book Aligned (EUR)",
            "BTAR (%)",
        ]
        rows = [
            TableRow(label="BTAR", values={
                "Metric": "Banking Book Taxonomy Alignment Ratio",
                "Banking Book Total (EUR)": str(gar_data.banking_book_total),
                "Banking Book Aligned (EUR)": str(gar_data.banking_book_aligned),
                "BTAR (%)": str(gar_data.btar_ratio),
            }),
        ]
        return EBATemplate(
            template_id=EBATemplateId.TEMPLATE_8,
            title="Template 8 - BTAR (Banking Book Taxonomy Alignment Ratio)",
            columns=columns,
            rows=rows,
            totals={"BTAR (%)": str(gar_data.btar_ratio)},
        )

    def _build_template_9(
        self, gar_data: GARInputData, entity: str
    ) -> EBATemplate:
        """Template 9 - GAR Flow."""
        columns = [
            "Exposure Type",
            "New Originations Aligned (EUR)",
            "Proportion (%)",
        ]
        rows: List[TableRow] = []

        for exp_type, amount in sorted(gar_data.by_exposure_type.items()):
            pct = self._pct(amount, gar_data.total_covered_assets)
            rows.append(TableRow(
                label=exp_type,
                values={
                    "Exposure Type": exp_type,
                    "New Originations Aligned (EUR)": str(amount),
                    "Proportion (%)": str(pct),
                },
            ))

        return EBATemplate(
            template_id=EBATemplateId.TEMPLATE_9,
            title="Template 9 - GAR (Flow)",
            columns=columns,
            rows=rows,
            totals={"GAR Flow (%)": str(gar_data.gar_flow_ratio)},
        )

    def _build_template_10(
        self, gar_data: GARInputData, entity: str
    ) -> EBATemplate:
        """Template 10 - Other Mitigating Actions."""
        columns = ["Action Description", "Amount (EUR)", "Category"]
        rows: List[TableRow] = []

        for action in gar_data.mitigating_actions:
            rows.append(TableRow(
                label=action.get("description", ""),
                values={
                    "Action Description": action.get("description", ""),
                    "Amount (EUR)": str(action.get("amount", 0)),
                    "Category": action.get("category", "Other"),
                },
            ))

        return EBATemplate(
            template_id=EBATemplateId.TEMPLATE_10,
            title="Template 10 - Other Mitigating Actions",
            columns=columns,
            rows=rows,
            notes=[
                "Exposures not included in GAR but subject to mitigating actions "
                "contributing to environmental objectives."
            ],
        )

    # ------------------------------------------------------------------
    # Summary computation
    # ------------------------------------------------------------------

    def _compute_summary(
        self,
        kpi_data: List[ActivityKPIData],
        alignment_data: AlignmentData,
    ) -> Dict[str, Any]:
        """Compute portfolio-level KPI summary ratios."""
        aligned_turnover = sum(
            (a.turnover for a in kpi_data if a.is_aligned), Decimal("0")
        )
        eligible_turnover = sum(
            (a.turnover for a in kpi_data if a.is_eligible), Decimal("0")
        )
        aligned_capex = sum(
            (a.capex for a in kpi_data if a.is_aligned), Decimal("0")
        )
        eligible_capex = sum(
            (a.capex for a in kpi_data if a.is_eligible), Decimal("0")
        )
        aligned_opex = sum(
            (a.opex for a in kpi_data if a.is_aligned), Decimal("0")
        )
        eligible_opex = sum(
            (a.opex for a in kpi_data if a.is_eligible), Decimal("0")
        )

        # Per-objective aligned turnover
        obj_turnover: Dict[str, Decimal] = {}
        obj_capex: Dict[str, Decimal] = {}
        for obj in ALL_OBJECTIVES:
            obj_turnover[obj.value] = sum(
                (a.turnover for a in kpi_data
                 if a.is_aligned and a.sc_objective == obj),
                Decimal("0"),
            )
            obj_capex[obj.value] = sum(
                (a.capex for a in kpi_data
                 if a.is_aligned and a.sc_objective == obj),
                Decimal("0"),
            )

        return {
            "turnover_aligned_ratio": self._pct(aligned_turnover, alignment_data.total_turnover),
            "turnover_eligible_ratio": self._pct(eligible_turnover, alignment_data.total_turnover),
            "capex_aligned_ratio": self._pct(aligned_capex, alignment_data.total_capex),
            "capex_eligible_ratio": self._pct(eligible_capex, alignment_data.total_capex),
            "opex_aligned_ratio": self._pct(aligned_opex, alignment_data.total_opex),
            "opex_eligible_ratio": self._pct(eligible_opex, alignment_data.total_opex),
            "aligned_turnover_eur": str(aligned_turnover),
            "eligible_turnover_eur": str(eligible_turnover),
            "aligned_capex_eur": str(aligned_capex),
            "eligible_capex_eur": str(eligible_capex),
            "aligned_opex_eur": str(aligned_opex),
            "eligible_opex_eur": str(eligible_opex),
            "turnover_aligned_ccm": str(
                self._pct(obj_turnover.get("CCM", Decimal("0")), alignment_data.total_turnover)
            ),
            "turnover_aligned_cca": str(
                self._pct(obj_turnover.get("CCA", Decimal("0")), alignment_data.total_turnover)
            ),
            "capex_aligned_ccm": str(
                self._pct(obj_capex.get("CCM", Decimal("0")), alignment_data.total_capex)
            ),
            "capex_aligned_cca": str(
                self._pct(obj_capex.get("CCA", Decimal("0")), alignment_data.total_capex)
            ),
        }

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_total_kpi(alignment_data: AlignmentData, kpi_type: KPIType) -> Decimal:
        """Extract the total KPI value from alignment data."""
        mapping = {
            KPIType.TURNOVER: alignment_data.total_turnover,
            KPIType.CAPEX: alignment_data.total_capex,
            KPIType.OPEX: alignment_data.total_opex,
        }
        return mapping[kpi_type]

    @staticmethod
    def _get_activity_kpi(activity: ActivityKPIData, kpi_type: KPIType) -> Decimal:
        """Extract KPI amount from an activity."""
        mapping = {
            KPIType.TURNOVER: activity.turnover,
            KPIType.CAPEX: activity.capex,
            KPIType.OPEX: activity.opex,
        }
        return mapping[kpi_type]

    @staticmethod
    def _pct(numerator: Decimal, denominator: Decimal) -> Decimal:
        """Calculate percentage as Decimal, returning 0 if denominator is zero."""
        if denominator == Decimal("0"):
            return Decimal("0")
        return (numerator * 100 / denominator).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )

    @staticmethod
    def _provenance(data: Dict[str, Any]) -> str:
        """Calculate SHA-256 hash for provenance tracking."""
        raw = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()
