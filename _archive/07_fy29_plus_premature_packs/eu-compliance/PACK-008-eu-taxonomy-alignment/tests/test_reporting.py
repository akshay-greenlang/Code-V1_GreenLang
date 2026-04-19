# -*- coding: utf-8 -*-
"""
Unit tests for PACK-008 EU Taxonomy Alignment Pack - Taxonomy Reporting Engine

Tests Article 8 disclosure table generation (Turnover, CapEx, OpEx), full disclosure
assembly, EBA Pillar 3 Templates 6-10, XBRL tag mapping, nuclear/gas supplementary
disclosures, year-over-year comparison, report metadata, and mandatory column validation.
"""

import pytest
import hashlib
import json
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Simulated Enumerations
# ---------------------------------------------------------------------------

ENVIRONMENTAL_OBJECTIVES = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]

KPI_TYPES = ["TURNOVER", "CAPEX", "OPEX"]

MANDATORY_TABLE_COLUMNS = [
    "economic_activity",
    "nace_code",
    "absolute_amount",
    "proportion_of_total",
    "ccm_sc",
    "cca_sc",
    "wtr_sc",
    "ce_sc",
    "ppc_sc",
    "bio_sc",
    "minimum_safeguards",
    "taxonomy_aligned_proportion",
    "taxonomy_eligible_proportion",
    "enabling_activity",
    "transitional_activity",
]

EBA_TEMPLATE_IDS = [
    "TEMPLATE_6",   # GAR Summary
    "TEMPLATE_7",   # GAR by Sector (NACE)
    "TEMPLATE_8",   # BTAR
    "TEMPLATE_9",   # GAR Flow
    "TEMPLATE_10",  # Other Mitigating Actions
]

XBRL_ELEMENTS = {
    "turnover_aligned_ratio": "esrs:TurnoverAlignedProportion",
    "capex_aligned_ratio": "esrs:CapExAlignedProportion",
    "opex_aligned_ratio": "esrs:OpExAlignedProportion",
    "turnover_eligible_ratio": "esrs:TurnoverEligibleProportion",
    "capex_eligible_ratio": "esrs:CapExEligibleProportion",
    "opex_eligible_ratio": "esrs:OpExEligibleProportion",
}


# ---------------------------------------------------------------------------
# Simulated Data Models and Helper Functions
# ---------------------------------------------------------------------------

def _compute_hash(data: Any) -> str:
    """Compute SHA-256 provenance hash."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _build_kpi_table(activities: List[Dict], total: float,
                     kpi_type: str) -> Dict[str, Any]:
    """Build a single KPI disclosure table."""
    kpi_key = kpi_type.lower()
    rows = []
    eligible_total = Decimal("0")
    aligned_total = Decimal("0")

    for act in activities:
        amount = Decimal(str(act.get(kpi_key, "0")))
        is_aligned = act.get("is_aligned", False)
        is_eligible = act.get("is_eligible", False)

        row = {
            "economic_activity": act.get("activity_name", ""),
            "nace_code": act.get("nace_code", ""),
            "absolute_amount": float(amount),
            "proportion_of_total": float(amount / Decimal(str(total))) if total > 0 else 0.0,
            "ccm_sc": "Y" if "CCM" in act.get("sc_objectives", []) else "N",
            "cca_sc": "Y" if "CCA" in act.get("sc_objectives", []) else "N",
            "wtr_sc": "N",
            "ce_sc": "N",
            "ppc_sc": "N",
            "bio_sc": "N",
            "minimum_safeguards": "Y" if is_aligned else "N",
            "taxonomy_aligned_proportion": float(amount / Decimal(str(total))) if is_aligned and total > 0 else 0.0,
            "taxonomy_eligible_proportion": float(amount / Decimal(str(total))) if is_eligible and total > 0 else 0.0,
            "enabling_activity": "N",
            "transitional_activity": "N",
        }
        rows.append(row)
        if is_eligible:
            eligible_total += amount
        if is_aligned:
            aligned_total += amount

    return {
        "kpi_type": kpi_type,
        "columns": MANDATORY_TABLE_COLUMNS,
        "rows": rows,
        "total_amount": float(total),
        "eligible_amount": float(eligible_total),
        "aligned_amount": float(aligned_total),
        "eligible_ratio": float(eligible_total / Decimal(str(total))) if total > 0 else 0.0,
        "aligned_ratio": float(aligned_total / Decimal(str(total))) if total > 0 else 0.0,
    }


class SimulatedReportingEngine:
    """Simulated Taxonomy Reporting Engine for testing."""

    def __init__(self):
        self.xbrl_elements = XBRL_ELEMENTS

    def generate_article8_turnover_table(self, activities: List[Dict],
                                         total_turnover: float) -> Dict[str, Any]:
        """Generate the Article 8 Turnover disclosure table."""
        return _build_kpi_table(activities, total_turnover, "TURNOVER")

    def generate_article8_capex_table(self, activities: List[Dict],
                                      total_capex: float) -> Dict[str, Any]:
        """Generate the Article 8 CapEx disclosure table."""
        return _build_kpi_table(activities, total_capex, "CAPEX")

    def generate_article8_opex_table(self, activities: List[Dict],
                                     total_opex: float) -> Dict[str, Any]:
        """Generate the Article 8 OpEx disclosure table."""
        return _build_kpi_table(activities, total_opex, "OPEX")

    def generate_full_article8_disclosure(
        self,
        activities: List[Dict],
        entity_name: str,
        reporting_period: str,
        total_turnover: float,
        total_capex: float,
        total_opex: float,
        prior_period: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Generate complete Article 8 disclosure with all three mandatory tables."""
        if not activities:
            raise ValueError("KPI data cannot be empty")

        start = datetime.utcnow()

        turnover_table = self.generate_article8_turnover_table(activities, total_turnover)
        capex_table = self.generate_article8_capex_table(activities, total_capex)
        opex_table = self.generate_article8_opex_table(activities, total_opex)

        yoy = None
        if prior_period:
            yoy = self._build_yoy(turnover_table, capex_table, opex_table, prior_period)

        summary = {
            "turnover_aligned_ratio": turnover_table["aligned_ratio"],
            "turnover_eligible_ratio": turnover_table["eligible_ratio"],
            "capex_aligned_ratio": capex_table["aligned_ratio"],
            "capex_eligible_ratio": capex_table["eligible_ratio"],
            "opex_aligned_ratio": opex_table["aligned_ratio"],
            "opex_eligible_ratio": opex_table["eligible_ratio"],
        }

        provenance = _compute_hash({
            "type": "article8",
            "entity": entity_name,
            "period": reporting_period,
            "activities": len(activities),
            "ts": start.isoformat(),
        })

        return {
            "entity_name": entity_name,
            "reporting_period": reporting_period,
            "tables": [turnover_table, capex_table, opex_table],
            "supplementary_tables": [],
            "yoy_comparison": yoy,
            "summary": summary,
            "generation_date": start.isoformat(),
            "provenance_hash": provenance,
        }

    def generate_eba_template(self, template_id: str, gar_data: Dict,
                              entity_name: str) -> Dict[str, Any]:
        """Generate a single EBA Pillar 3 template."""
        descriptions = {
            "TEMPLATE_6": "GAR Summary",
            "TEMPLATE_7": "GAR by Sector (NACE)",
            "TEMPLATE_8": "BTAR (Banking Book Taxonomy Alignment Ratio)",
            "TEMPLATE_9": "GAR Flow",
            "TEMPLATE_10": "Other Mitigating Actions",
        }
        return {
            "template_id": template_id,
            "template_name": descriptions.get(template_id, "Unknown"),
            "entity_name": entity_name,
            "data": gar_data,
            "generation_date": datetime.utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "type": "eba",
                "template": template_id,
                "entity": entity_name,
            }),
        }

    def generate_eba_templates(self, gar_data: Dict, entity_name: str,
                               reporting_period: str) -> Dict[str, Any]:
        """Generate EBA Templates 6-10."""
        templates = []
        for tid in EBA_TEMPLATE_IDS:
            templates.append(self.generate_eba_template(tid, gar_data, entity_name))

        return {
            "entity_name": entity_name,
            "reporting_period": reporting_period,
            "templates": templates,
            "generation_date": datetime.utcnow().isoformat(),
            "provenance_hash": _compute_hash({
                "type": "eba_templates",
                "entity": entity_name,
                "period": reporting_period,
            }),
        }

    def generate_xbrl_tags(self, report: Dict[str, Any]) -> Dict[str, Any]:
        """Generate XBRL/iXBRL tags for Article 8 disclosure."""
        tags = []
        context_ref = f"ctx_{report['reporting_period']}"
        summary = report.get("summary", {})

        for key, element in self.xbrl_elements.items():
            value = summary.get(key)
            if value is not None:
                is_ratio = "ratio" in key.lower()
                tags.append({
                    "element_name": element,
                    "namespace": "esrs" if element.startswith("esrs:") else "eba",
                    "value": str(value),
                    "unit": "xbrli:pure" if is_ratio else "iso4217:EUR",
                    "context_ref": context_ref,
                    "decimals": 6 if is_ratio else 2,
                })

        return {
            "tags": tags,
            "tag_count": len(tags),
            "provenance_hash": _compute_hash({
                "type": "xbrl",
                "period": report["reporting_period"],
                "tag_count": len(tags),
            }),
        }

    def generate_nuclear_gas_supplementary(
        self, activities: List[Dict], entity_name: str
    ) -> Dict[str, Any]:
        """Generate nuclear/gas supplementary disclosure per DA 2022/1214."""
        nuclear_activities = [
            a for a in activities
            if a.get("activity_type") in ("nuclear", "gas")
        ]

        return {
            "entity_name": entity_name,
            "disclosure_type": "SUPPLEMENTARY_NUCLEAR_GAS",
            "nuclear_gas_activities": nuclear_activities,
            "total_nuclear_gas_count": len(nuclear_activities),
            "templates_generated": [
                "Nuclear Template 1",
                "Nuclear Template 2",
                "Gas Template 3",
                "Gas Template 4",
                "Gas Template 5",
            ] if nuclear_activities else [],
            "provenance_hash": _compute_hash({
                "type": "nuclear_gas",
                "entity": entity_name,
                "count": len(nuclear_activities),
            }),
        }

    def generate_yoy_comparison(
        self, current_report: Dict[str, Any],
        prior_report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate year-over-year comparison tables."""
        comparisons = {}
        for key in ["turnover_aligned_ratio", "capex_aligned_ratio", "opex_aligned_ratio"]:
            current_val = current_report.get("summary", {}).get(key, 0.0)
            prior_val = prior_report.get("summary", {}).get(key, 0.0)
            comparisons[key] = {
                "current": current_val,
                "prior": prior_val,
                "change": current_val - prior_val,
                "change_pct": ((current_val - prior_val) / prior_val * 100.0
                               if prior_val != 0 else 0.0),
            }

        return {
            "comparisons": comparisons,
            "current_period": current_report.get("reporting_period", ""),
            "prior_period": prior_report.get("reporting_period", ""),
            "provenance_hash": _compute_hash(comparisons),
        }

    def _build_yoy(self, turnover: Dict, capex: Dict, opex: Dict,
                   prior: Dict) -> Dict[str, Any]:
        """Build YoY comparison from current tables and prior data."""
        return {
            "turnover_change": turnover["aligned_ratio"] - prior.get("turnover_aligned_ratio", 0.0),
            "capex_change": capex["aligned_ratio"] - prior.get("capex_aligned_ratio", 0.0),
            "opex_change": opex["aligned_ratio"] - prior.get("opex_aligned_ratio", 0.0),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def reporting_engine():
    """Create a simulated reporting engine."""
    return SimulatedReportingEngine()


@pytest.fixture
def sample_activities():
    """Sample activity financial data for disclosure."""
    return [
        {
            "activity_id": "CCM-4.1",
            "activity_name": "Solar PV electricity generation",
            "nace_code": "D35.11",
            "turnover": "500000",
            "capex": "120000",
            "opex": "30000",
            "is_eligible": True,
            "is_aligned": True,
            "sc_objectives": ["CCM"],
        },
        {
            "activity_id": "CCM-3.9",
            "activity_name": "Iron and steel manufacturing",
            "nace_code": "C24.10",
            "turnover": "1200000",
            "capex": "300000",
            "opex": "80000",
            "is_eligible": True,
            "is_aligned": False,
            "sc_objectives": ["CCM"],
        },
        {
            "activity_id": "CCM-7.1",
            "activity_name": "Construction of new buildings",
            "nace_code": "F41.10",
            "turnover": "800000",
            "capex": "200000",
            "opex": "50000",
            "is_eligible": True,
            "is_aligned": True,
            "sc_objectives": ["CCM", "CCA"],
        },
        {
            "activity_id": "NON-ELIGIBLE",
            "activity_name": "Consulting services",
            "nace_code": "M70.22",
            "turnover": "300000",
            "capex": "10000",
            "opex": "15000",
            "is_eligible": False,
            "is_aligned": False,
            "sc_objectives": [],
        },
    ]


@pytest.fixture
def sample_gar_data():
    """Sample GAR input data for EBA template generation."""
    return {
        "total_covered_assets": 18700000.0,
        "gar_numerator_aligned": 8550000.0,
        "gar_numerator_eligible": 11100000.0,
        "gar_ratio_aligned": pytest.approx(0.4572, abs=0.01),
        "gar_ratio_eligible": pytest.approx(0.5936, abs=0.01),
        "btar_ratio": pytest.approx(0.38, abs=0.01),
        "sectors": {
            "D35": {"aligned": 6000000.0, "eligible": 8000000.0},
            "C24": {"aligned": 50000.0, "eligible": 100000.0},
            "L68": {"aligned": 2000000.0, "eligible": 2500000.0},
            "F41": {"aligned": 500000.0, "eligible": 500000.0},
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestTaxonomyReporting:
    """Test suite for TaxonomyReportingEngine."""

    def test_generate_article8_turnover_table(self, reporting_engine, sample_activities):
        """Test generating the Article 8 Turnover disclosure table."""
        total_turnover = 2800000.0
        table = reporting_engine.generate_article8_turnover_table(
            sample_activities, total_turnover
        )

        assert table["kpi_type"] == "TURNOVER"
        assert len(table["rows"]) == len(sample_activities)
        assert table["total_amount"] == pytest.approx(total_turnover)
        assert table["eligible_amount"] > 0
        assert table["aligned_amount"] > 0
        assert 0 <= table["eligible_ratio"] <= 1.0
        assert 0 <= table["aligned_ratio"] <= 1.0
        # Aligned ratio should be <= eligible ratio
        assert table["aligned_ratio"] <= table["eligible_ratio"] + 1e-9
        assert len(table["columns"]) == len(MANDATORY_TABLE_COLUMNS)

    def test_generate_article8_capex_table(self, reporting_engine, sample_activities):
        """Test generating the Article 8 CapEx disclosure table."""
        total_capex = 630000.0
        table = reporting_engine.generate_article8_capex_table(
            sample_activities, total_capex
        )

        assert table["kpi_type"] == "CAPEX"
        assert len(table["rows"]) == len(sample_activities)
        assert table["total_amount"] == pytest.approx(total_capex)
        assert table["eligible_amount"] > 0

    def test_generate_article8_opex_table(self, reporting_engine, sample_activities):
        """Test generating the Article 8 OpEx disclosure table."""
        total_opex = 175000.0
        table = reporting_engine.generate_article8_opex_table(
            sample_activities, total_opex
        )

        assert table["kpi_type"] == "OPEX"
        assert len(table["rows"]) == len(sample_activities)
        assert table["total_amount"] == pytest.approx(total_opex)

    def test_generate_full_article8_disclosure(self, reporting_engine, sample_activities):
        """Test generating complete Article 8 disclosure with all three mandatory tables."""
        report = reporting_engine.generate_full_article8_disclosure(
            activities=sample_activities,
            entity_name="Acme Manufacturing GmbH",
            reporting_period="FY 2025",
            total_turnover=2800000.0,
            total_capex=630000.0,
            total_opex=175000.0,
        )

        assert report["entity_name"] == "Acme Manufacturing GmbH"
        assert report["reporting_period"] == "FY 2025"
        assert len(report["tables"]) == 3  # Turnover + CapEx + OpEx
        assert report["tables"][0]["kpi_type"] == "TURNOVER"
        assert report["tables"][1]["kpi_type"] == "CAPEX"
        assert report["tables"][2]["kpi_type"] == "OPEX"
        assert "summary" in report
        assert report["provenance_hash"] is not None
        assert len(report["provenance_hash"]) == 64
        assert report["generation_date"] is not None

    def test_eba_template_6_gar_summary(self, reporting_engine, sample_gar_data):
        """Test generating EBA Template 6 (GAR Summary)."""
        template = reporting_engine.generate_eba_template(
            "TEMPLATE_6", sample_gar_data, "TestBank AG"
        )

        assert template["template_id"] == "TEMPLATE_6"
        assert template["template_name"] == "GAR Summary"
        assert template["entity_name"] == "TestBank AG"
        assert template["data"] is not None
        assert template["provenance_hash"] is not None
        assert len(template["provenance_hash"]) == 64

    def test_eba_template_7_sector(self, reporting_engine, sample_gar_data):
        """Test generating EBA Template 7 (GAR by Sector/NACE)."""
        template = reporting_engine.generate_eba_template(
            "TEMPLATE_7", sample_gar_data, "TestBank AG"
        )

        assert template["template_id"] == "TEMPLATE_7"
        assert template["template_name"] == "GAR by Sector (NACE)"
        assert template["entity_name"] == "TestBank AG"

    def test_eba_template_8_btar(self, reporting_engine, sample_gar_data):
        """Test generating EBA Template 8 (BTAR)."""
        template = reporting_engine.generate_eba_template(
            "TEMPLATE_8", sample_gar_data, "TestBank AG"
        )

        assert template["template_id"] == "TEMPLATE_8"
        assert "BTAR" in template["template_name"]

    def test_eba_template_9_flow(self, reporting_engine, sample_gar_data):
        """Test generating EBA Template 9 (GAR Flow)."""
        template = reporting_engine.generate_eba_template(
            "TEMPLATE_9", sample_gar_data, "TestBank AG"
        )

        assert template["template_id"] == "TEMPLATE_9"
        assert "Flow" in template["template_name"]

    def test_eba_template_10_mitigating(self, reporting_engine, sample_gar_data):
        """Test generating EBA Template 10 (Other Mitigating Actions)."""
        template = reporting_engine.generate_eba_template(
            "TEMPLATE_10", sample_gar_data, "TestBank AG"
        )

        assert template["template_id"] == "TEMPLATE_10"
        assert "Mitigating" in template["template_name"]

    def test_xbrl_tagging(self, reporting_engine, sample_activities):
        """Test XBRL/iXBRL tag generation from an Article 8 report."""
        report = reporting_engine.generate_full_article8_disclosure(
            activities=sample_activities,
            entity_name="Acme GmbH",
            reporting_period="FY 2025",
            total_turnover=2800000.0,
            total_capex=630000.0,
            total_opex=175000.0,
        )

        xbrl_output = reporting_engine.generate_xbrl_tags(report)

        assert xbrl_output["tag_count"] >= 1
        assert len(xbrl_output["tags"]) == xbrl_output["tag_count"]
        assert xbrl_output["provenance_hash"] is not None
        assert len(xbrl_output["provenance_hash"]) == 64

        # Check tag structure
        for tag in xbrl_output["tags"]:
            assert "element_name" in tag
            assert "namespace" in tag
            assert tag["namespace"] in ("esrs", "eba")
            assert "value" in tag
            assert "context_ref" in tag
            assert "FY 2025" in tag["context_ref"]

    def test_nuclear_gas_supplementary(self, reporting_engine):
        """Test nuclear/gas supplementary disclosure generation per DA 2022/1214."""
        activities = [
            {
                "activity_id": "4.28",
                "activity_name": "Nuclear power generation",
                "activity_type": "nuclear",
                "turnover": "3000000",
            },
            {
                "activity_id": "4.29",
                "activity_name": "Natural gas electricity generation",
                "activity_type": "gas",
                "turnover": "1500000",
            },
            {
                "activity_id": "4.1",
                "activity_name": "Solar PV",
                "activity_type": "renewable",
                "turnover": "2000000",
            },
        ]

        result = reporting_engine.generate_nuclear_gas_supplementary(
            activities, "EnergyCorpGmbH"
        )

        assert result["disclosure_type"] == "SUPPLEMENTARY_NUCLEAR_GAS"
        assert result["entity_name"] == "EnergyCorpGmbH"
        assert result["total_nuclear_gas_count"] == 2
        assert len(result["nuclear_gas_activities"]) == 2
        assert len(result["templates_generated"]) >= 1
        assert result["provenance_hash"] is not None

    def test_yoy_comparison_tables(self, reporting_engine, sample_activities):
        """Test year-over-year comparison table generation."""
        current_report = reporting_engine.generate_full_article8_disclosure(
            activities=sample_activities,
            entity_name="Acme GmbH",
            reporting_period="FY 2025",
            total_turnover=2800000.0,
            total_capex=630000.0,
            total_opex=175000.0,
        )

        prior_report = {
            "reporting_period": "FY 2024",
            "summary": {
                "turnover_aligned_ratio": 0.30,
                "capex_aligned_ratio": 0.40,
                "opex_aligned_ratio": 0.35,
            },
        }

        yoy = reporting_engine.generate_yoy_comparison(current_report, prior_report)

        assert "comparisons" in yoy
        assert "current_period" in yoy
        assert "prior_period" in yoy
        assert yoy["current_period"] == "FY 2025"
        assert yoy["prior_period"] == "FY 2024"
        assert "turnover_aligned_ratio" in yoy["comparisons"]
        assert "capex_aligned_ratio" in yoy["comparisons"]
        assert "opex_aligned_ratio" in yoy["comparisons"]

        for key, comp in yoy["comparisons"].items():
            assert "current" in comp
            assert "prior" in comp
            assert "change" in comp

    def test_report_metadata(self, reporting_engine, sample_activities):
        """Test that generated reports include all required metadata fields."""
        report = reporting_engine.generate_full_article8_disclosure(
            activities=sample_activities,
            entity_name="MetaCorp AG",
            reporting_period="FY 2025",
            total_turnover=2800000.0,
            total_capex=630000.0,
            total_opex=175000.0,
        )

        # Required metadata fields
        assert "entity_name" in report
        assert "reporting_period" in report
        assert "generation_date" in report
        assert "provenance_hash" in report
        assert "tables" in report
        assert "summary" in report

        # Validate provenance hash format (SHA-256)
        h = report["provenance_hash"]
        assert isinstance(h, str)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

        # Validate generation date is valid ISO 8601
        gen_date = report["generation_date"]
        assert gen_date is not None
        parsed = datetime.fromisoformat(gen_date)
        assert parsed.year >= 2025

    def test_mandatory_table_columns(self, reporting_engine, sample_activities):
        """Test that all three mandatory tables contain the required column set."""
        report = reporting_engine.generate_full_article8_disclosure(
            activities=sample_activities,
            entity_name="ColCheck GmbH",
            reporting_period="FY 2025",
            total_turnover=2800000.0,
            total_capex=630000.0,
            total_opex=175000.0,
        )

        for table in report["tables"]:
            assert "columns" in table
            columns = table["columns"]
            for required_col in MANDATORY_TABLE_COLUMNS:
                assert required_col in columns, (
                    f"Missing mandatory column '{required_col}' in {table['kpi_type']} table"
                )

            # Each row should have the mandatory column keys
            for row in table["rows"]:
                for col in MANDATORY_TABLE_COLUMNS:
                    assert col in row, (
                        f"Row missing mandatory column '{col}' in {table['kpi_type']} table"
                    )
