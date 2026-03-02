# -*- coding: utf-8 -*-
"""
Unit tests for GL-CBAM-APP v1.1 Report Assembler

Tests report assembly:
- assemble_quarterly_report (full assembly)
- aggregate_by_cn_code (grouping, totals)
- aggregate_by_country (multi-country)
- aggregate_by_installation (multi-installation)
- apply_calculation_hierarchy (supplier > regional > default)
- apply_default_value_markup (2026: +10%, 2027: +20%, 2028+: +30%)
- calculate_complex_goods_rule (20% cap)
- generate_xml_output (valid XML structure)
- validate_report_completeness (required fields)
- Provenance hash consistency

Target: 60+ tests
"""

import pytest
import hashlib
import json
from datetime import datetime, date
from decimal import Decimal, ROUND_HALF_UP
from xml.etree import ElementTree as ET
from copy import deepcopy


# ---------------------------------------------------------------------------
# Inline report assembler for self-contained tests
# ---------------------------------------------------------------------------

class ReportAssembler:
    """Assembles CBAM quarterly reports from submission data."""

    DEFAULT_VALUE_MARKUP = {
        2026: Decimal("0.10"),
        2027: Decimal("0.20"),
    }
    DEFAULT_MARKUP_FALLBACK = Decimal("0.30")
    COMPLEX_GOODS_CAP_PCT = Decimal("20.0")

    def __init__(self):
        self._provenance_inputs = []

    def assemble_quarterly_report(self, *, quarter, importer_info,
                                  submissions, installations=None):
        year = int(quarter[:4])
        q = int(quarter[5])
        period_start, period_end = self._quarter_dates(year, q)

        by_cn = self.aggregate_by_cn_code(submissions)
        by_country = self.aggregate_by_country(submissions)
        by_installation = self.aggregate_by_installation(submissions)

        total_direct = sum(Decimal(str(s.get("direct_emissions_tco2", 0)))
                           for s in submissions)
        total_indirect = sum(Decimal(str(s.get("indirect_emissions_tco2", 0)))
                             for s in submissions)
        total_embedded = total_direct + total_indirect
        for s in submissions:
            for p in s.get("precursor_emissions", []):
                total_embedded += Decimal(str(p.get("emissions_tco2", 0)))

        report = {
            "report_metadata": {
                "report_id": f"CBAM-{quarter}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "quarter": quarter,
                "period_start": period_start.isoformat(),
                "period_end": period_end.isoformat(),
                "generated_at": datetime.utcnow().isoformat(),
                "version": "1.1.0",
            },
            "importer_declaration": importer_info,
            "goods_summary": {
                "total_submissions": len(submissions),
                "total_direct_tco2": str(total_direct.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP)),
                "total_indirect_tco2": str(total_indirect.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP)),
                "total_embedded_tco2": str(total_embedded.quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP)),
            },
            "aggregations": {
                "by_cn_code": by_cn,
                "by_country": by_country,
                "by_installation": by_installation,
            },
            "validation": self.validate_report_completeness(importer_info, submissions),
            "provenance_hash": self._generate_provenance_hash(
                quarter, importer_info, submissions
            ),
        }
        return report

    def _quarter_dates(self, year, q):
        starts = {1: date(year, 1, 1), 2: date(year, 4, 1),
                  3: date(year, 7, 1), 4: date(year, 10, 1)}
        ends = {1: date(year, 3, 31), 2: date(year, 6, 30),
                3: date(year, 9, 30), 4: date(year, 12, 31)}
        return starts[q], ends[q]

    def aggregate_by_cn_code(self, submissions):
        groups = {}
        for s in submissions:
            cn = s.get("cn_code", "unknown")
            if cn not in groups:
                groups[cn] = {"cn_code": cn, "count": 0,
                              "total_emissions_tco2": Decimal("0")}
            groups[cn]["count"] += 1
            groups[cn]["total_emissions_tco2"] += Decimal(
                str(s.get("direct_emissions_tco2", 0))
            ) + Decimal(str(s.get("indirect_emissions_tco2", 0)))
        result = []
        for cn, data in sorted(groups.items()):
            result.append({
                "cn_code": data["cn_code"],
                "count": data["count"],
                "total_emissions_tco2": str(data["total_emissions_tco2"].quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP)),
            })
        return result

    def aggregate_by_country(self, submissions):
        groups = {}
        for s in submissions:
            country = s.get("country_code", "unknown")
            if country not in groups:
                groups[country] = {"country_code": country, "count": 0,
                                   "total_emissions_tco2": Decimal("0")}
            groups[country]["count"] += 1
            groups[country]["total_emissions_tco2"] += Decimal(
                str(s.get("direct_emissions_tco2", 0))
            ) + Decimal(str(s.get("indirect_emissions_tco2", 0)))
        result = []
        for c, data in sorted(groups.items()):
            result.append({
                "country_code": data["country_code"],
                "count": data["count"],
                "total_emissions_tco2": str(data["total_emissions_tco2"].quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP)),
            })
        return result

    def aggregate_by_installation(self, submissions):
        groups = {}
        for s in submissions:
            inst = s.get("installation_id", "unknown")
            if inst not in groups:
                groups[inst] = {"installation_id": inst, "count": 0,
                                "total_emissions_tco2": Decimal("0")}
            groups[inst]["count"] += 1
            groups[inst]["total_emissions_tco2"] += Decimal(
                str(s.get("direct_emissions_tco2", 0))
            ) + Decimal(str(s.get("indirect_emissions_tco2", 0)))
        result = []
        for inst, data in sorted(groups.items()):
            result.append({
                "installation_id": data["installation_id"],
                "count": data["count"],
                "total_emissions_tco2": str(data["total_emissions_tco2"].quantize(
                    Decimal("0.001"), rounding=ROUND_HALF_UP)),
            })
        return result

    def apply_calculation_hierarchy(self, submission):
        method = submission.get("calculation_method", "eu_default")
        hierarchy = {"supplier_specific": 1, "regional_default": 2, "eu_default": 3}
        priority = hierarchy.get(method, 3)
        return {
            "method": method,
            "priority": priority,
            "is_supplier_specific": priority == 1,
        }

    def apply_default_value_markup(self, emissions_tco2, year):
        base = Decimal(str(emissions_tco2))
        markup_rate = self.DEFAULT_VALUE_MARKUP.get(
            year, self.DEFAULT_MARKUP_FALLBACK
        )
        markup = base * markup_rate
        return (base + markup).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def calculate_complex_goods_rule(self, submissions):
        total = len(submissions)
        complex_count = sum(
            1 for s in submissions
            if s.get("is_complex_good", False)
        )
        if total == 0:
            return {"complex_count": 0, "total": 0, "pct": Decimal("0"),
                    "exceeds_cap": False}
        pct = Decimal(str(complex_count)) / Decimal(str(total)) * 100
        pct = pct.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        return {
            "complex_count": complex_count,
            "total": total,
            "pct": pct,
            "exceeds_cap": pct > self.COMPLEX_GOODS_CAP_PCT,
        }

    def generate_xml_output(self, report):
        root = ET.Element("CBAMReport", version="1.1.0")
        meta = ET.SubElement(root, "ReportMetadata")
        for k, v in report["report_metadata"].items():
            child = ET.SubElement(meta, k.replace("_", ""))
            child.text = str(v)
        decl = ET.SubElement(root, "ImporterDeclaration")
        for k, v in report["importer_declaration"].items():
            child = ET.SubElement(decl, k.replace("_", ""))
            child.text = str(v)
        summary = ET.SubElement(root, "GoodsSummary")
        for k, v in report["goods_summary"].items():
            child = ET.SubElement(summary, k.replace("_", ""))
            child.text = str(v)
        agg = ET.SubElement(root, "Aggregations")
        for cn_entry in report["aggregations"]["by_cn_code"]:
            cn_elem = ET.SubElement(agg, "CNCode")
            for k, v in cn_entry.items():
                child = ET.SubElement(cn_elem, k)
                child.text = str(v)
        return ET.tostring(root, encoding="unicode", xml_declaration=True)

    def validate_report_completeness(self, importer_info, submissions):
        issues = []
        required_importer = ["importer_name", "importer_eori", "importer_country"]
        for field in required_importer:
            if not importer_info.get(field):
                issues.append(f"Missing importer field: {field}")
        if not submissions:
            issues.append("No submissions in report")
        for i, s in enumerate(submissions):
            if not s.get("cn_code"):
                issues.append(f"Submission {i}: missing cn_code")
            if not s.get("direct_emissions_tco2") and \
               not s.get("indirect_emissions_tco2"):
                issues.append(f"Submission {i}: no emissions data")
        return {"is_complete": len(issues) == 0, "issues": issues}

    def _generate_provenance_hash(self, quarter, importer_info, submissions):
        payload = json.dumps({
            "quarter": quarter,
            "importer": importer_info,
            "submission_count": len(submissions),
            "cn_codes": sorted(set(s.get("cn_code", "") for s in submissions)),
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def assembler():
    return ReportAssembler()


@pytest.fixture
def importer_info():
    return {
        "importer_name": "EuroImport BV",
        "importer_eori": "NL123456789012",
        "importer_country": "NL",
        "declarant_name": "Jan de Vries",
        "declarant_position": "Compliance Manager",
    }


@pytest.fixture
def sample_submissions():
    return [
        {
            "submission_id": "SUB-001",
            "supplier_id": "S1",
            "installation_id": "INST-001",
            "cn_code": "72031000",
            "country_code": "CN",
            "direct_emissions_tco2": 200,
            "indirect_emissions_tco2": 50,
            "calculation_method": "supplier_specific",
            "precursor_emissions": [],
        },
        {
            "submission_id": "SUB-002",
            "supplier_id": "S2",
            "installation_id": "INST-002",
            "cn_code": "25231000",
            "country_code": "TR",
            "direct_emissions_tco2": 80,
            "indirect_emissions_tco2": 20,
            "calculation_method": "eu_default",
            "precursor_emissions": [],
        },
        {
            "submission_id": "SUB-003",
            "supplier_id": "S1",
            "installation_id": "INST-001",
            "cn_code": "72031000",
            "country_code": "CN",
            "direct_emissions_tco2": 150,
            "indirect_emissions_tco2": 30,
            "calculation_method": "regional_default",
            "precursor_emissions": [],
        },
    ]


# ===========================================================================
# TEST CLASS -- assemble_quarterly_report
# ===========================================================================

class TestAssembleQuarterlyReport:
    """Tests for assemble_quarterly_report."""

    def test_full_assembly(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1",
            importer_info=importer_info,
            submissions=sample_submissions,
        )
        assert "report_metadata" in report
        assert "goods_summary" in report
        assert "aggregations" in report
        assert "validation" in report
        assert "provenance_hash" in report

    def test_report_metadata_fields(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        meta = report["report_metadata"]
        assert meta["quarter"] == "2026Q1"
        assert meta["version"] == "1.1.0"
        assert "report_id" in meta

    def test_goods_summary_totals(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        gs = report["goods_summary"]
        assert gs["total_submissions"] == 3
        assert Decimal(gs["total_direct_tco2"]) == Decimal("430.000")
        assert Decimal(gs["total_indirect_tco2"]) == Decimal("100.000")
        assert Decimal(gs["total_embedded_tco2"]) == Decimal("530.000")

    def test_empty_submissions(self, assembler, importer_info):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=[],
        )
        assert report["goods_summary"]["total_submissions"] == 0
        assert report["validation"]["is_complete"] is False


# ===========================================================================
# TEST CLASS -- aggregate_by_cn_code
# ===========================================================================

class TestAggregateByCNCode:
    """Tests for aggregate_by_cn_code."""

    def test_grouping(self, assembler, sample_submissions):
        result = assembler.aggregate_by_cn_code(sample_submissions)
        assert len(result) == 2  # 72031000 and 25231000

    def test_totals_correct(self, assembler, sample_submissions):
        result = assembler.aggregate_by_cn_code(sample_submissions)
        steel = next(r for r in result if r["cn_code"] == "72031000")
        assert steel["count"] == 2
        assert Decimal(steel["total_emissions_tco2"]) == Decimal("430.000")

    def test_single_cn_code(self, assembler):
        subs = [{"cn_code": "76011000", "direct_emissions_tco2": 50,
                 "indirect_emissions_tco2": 10}]
        result = assembler.aggregate_by_cn_code(subs)
        assert len(result) == 1
        assert result[0]["count"] == 1

    def test_sorted_by_cn_code(self, assembler, sample_submissions):
        result = assembler.aggregate_by_cn_code(sample_submissions)
        codes = [r["cn_code"] for r in result]
        assert codes == sorted(codes)


# ===========================================================================
# TEST CLASS -- aggregate_by_country
# ===========================================================================

class TestAggregateByCountry:
    """Tests for aggregate_by_country."""

    def test_multi_country(self, assembler, sample_submissions):
        result = assembler.aggregate_by_country(sample_submissions)
        assert len(result) == 2  # CN and TR

    def test_country_totals(self, assembler, sample_submissions):
        result = assembler.aggregate_by_country(sample_submissions)
        cn = next(r for r in result if r["country_code"] == "CN")
        assert cn["count"] == 2
        assert Decimal(cn["total_emissions_tco2"]) == Decimal("430.000")


# ===========================================================================
# TEST CLASS -- aggregate_by_installation
# ===========================================================================

class TestAggregateByInstallation:
    """Tests for aggregate_by_installation."""

    def test_multi_installation(self, assembler, sample_submissions):
        result = assembler.aggregate_by_installation(sample_submissions)
        assert len(result) == 2

    def test_installation_totals(self, assembler, sample_submissions):
        result = assembler.aggregate_by_installation(sample_submissions)
        inst1 = next(r for r in result if r["installation_id"] == "INST-001")
        assert inst1["count"] == 2


# ===========================================================================
# TEST CLASS -- apply_calculation_hierarchy
# ===========================================================================

class TestApplyCalculationHierarchy:
    """Tests for apply_calculation_hierarchy."""

    def test_supplier_specific_highest(self, assembler):
        result = assembler.apply_calculation_hierarchy(
            {"calculation_method": "supplier_specific"}
        )
        assert result["priority"] == 1
        assert result["is_supplier_specific"] is True

    def test_regional_default_middle(self, assembler):
        result = assembler.apply_calculation_hierarchy(
            {"calculation_method": "regional_default"}
        )
        assert result["priority"] == 2
        assert result["is_supplier_specific"] is False

    def test_eu_default_lowest(self, assembler):
        result = assembler.apply_calculation_hierarchy(
            {"calculation_method": "eu_default"}
        )
        assert result["priority"] == 3

    def test_unknown_method_defaults_to_lowest(self, assembler):
        result = assembler.apply_calculation_hierarchy(
            {"calculation_method": "unknown"}
        )
        assert result["priority"] == 3

    def test_missing_method_defaults(self, assembler):
        result = assembler.apply_calculation_hierarchy({})
        assert result["priority"] == 3


# ===========================================================================
# TEST CLASS -- apply_default_value_markup
# ===========================================================================

class TestApplyDefaultValueMarkup:
    """Tests for apply_default_value_markup phase-in."""

    @pytest.mark.parametrize("year,rate_pct", [
        (2026, 10), (2027, 20), (2028, 30), (2029, 30), (2035, 30),
    ])
    def test_markup_rates(self, assembler, year, rate_pct):
        base = Decimal("100")
        result = assembler.apply_default_value_markup(base, year)
        expected = base + base * Decimal(str(rate_pct)) / 100
        assert result == expected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)

    def test_markup_on_zero(self, assembler):
        result = assembler.apply_default_value_markup(0, 2026)
        assert result == Decimal("0.000")

    def test_markup_precision(self, assembler):
        result = assembler.apply_default_value_markup(Decimal("33.333"), 2026)
        expected = Decimal("33.333") * Decimal("1.10")
        expected = expected.quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
        assert result == expected

    def test_markup_large_value(self, assembler):
        result = assembler.apply_default_value_markup(Decimal("999999.999"), 2028)
        assert result > Decimal("999999.999")


# ===========================================================================
# TEST CLASS -- calculate_complex_goods_rule
# ===========================================================================

class TestCalculateComplexGoodsRule:
    """Tests for calculate_complex_goods_rule (20% cap)."""

    def test_no_complex_goods(self, assembler, sample_submissions):
        result = assembler.calculate_complex_goods_rule(sample_submissions)
        assert result["complex_count"] == 0
        assert result["exceeds_cap"] is False

    def test_below_cap(self, assembler):
        subs = [{"cn_code": "72031000"} for _ in range(10)]
        subs[0]["is_complex_good"] = True
        result = assembler.calculate_complex_goods_rule(subs)
        assert result["pct"] == Decimal("10.0")
        assert result["exceeds_cap"] is False

    def test_at_cap(self, assembler):
        subs = [{"cn_code": "72031000"} for _ in range(10)]
        subs[0]["is_complex_good"] = True
        subs[1]["is_complex_good"] = True
        result = assembler.calculate_complex_goods_rule(subs)
        assert result["pct"] == Decimal("20.0")
        assert result["exceeds_cap"] is False

    def test_exceeds_cap(self, assembler):
        subs = [{"cn_code": "72031000", "is_complex_good": True} for _ in range(3)]
        subs.extend([{"cn_code": "72031000"} for _ in range(7)])
        result = assembler.calculate_complex_goods_rule(subs)
        assert result["pct"] == Decimal("30.0")
        assert result["exceeds_cap"] is True

    def test_empty_submissions(self, assembler):
        result = assembler.calculate_complex_goods_rule([])
        assert result["exceeds_cap"] is False
        assert result["pct"] == Decimal("0")

    def test_all_complex(self, assembler):
        subs = [{"cn_code": "72031000", "is_complex_good": True} for _ in range(5)]
        result = assembler.calculate_complex_goods_rule(subs)
        assert result["pct"] == Decimal("100.0")
        assert result["exceeds_cap"] is True


# ===========================================================================
# TEST CLASS -- generate_xml_output
# ===========================================================================

class TestGenerateXMLOutput:
    """Tests for generate_xml_output."""

    def test_valid_xml(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        xml_str = assembler.generate_xml_output(report)
        root = ET.fromstring(xml_str)
        assert root.tag == "CBAMReport"

    def test_xml_has_metadata(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        xml_str = assembler.generate_xml_output(report)
        root = ET.fromstring(xml_str)
        meta = root.find("ReportMetadata")
        assert meta is not None

    def test_xml_has_declaration(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        xml_str = assembler.generate_xml_output(report)
        root = ET.fromstring(xml_str)
        decl = root.find("ImporterDeclaration")
        assert decl is not None

    def test_xml_version_attribute(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        xml_str = assembler.generate_xml_output(report)
        root = ET.fromstring(xml_str)
        assert root.get("version") == "1.1.0"

    def test_xml_cn_code_entries(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        xml_str = assembler.generate_xml_output(report)
        root = ET.fromstring(xml_str)
        cn_entries = root.findall(".//CNCode")
        assert len(cn_entries) == 2


# ===========================================================================
# TEST CLASS -- validate_report_completeness
# ===========================================================================

class TestValidateReportCompleteness:
    """Tests for validate_report_completeness."""

    def test_complete_report(self, assembler, importer_info, sample_submissions):
        result = assembler.validate_report_completeness(
            importer_info, sample_submissions
        )
        assert result["is_complete"] is True

    def test_missing_importer_name(self, assembler, sample_submissions):
        info = {"importer_eori": "NL123", "importer_country": "NL"}
        result = assembler.validate_report_completeness(info, sample_submissions)
        assert result["is_complete"] is False
        assert any("importer_name" in i for i in result["issues"])

    def test_missing_eori(self, assembler, sample_submissions):
        info = {"importer_name": "Test", "importer_country": "NL"}
        result = assembler.validate_report_completeness(info, sample_submissions)
        assert result["is_complete"] is False

    def test_empty_submissions(self, assembler, importer_info):
        result = assembler.validate_report_completeness(importer_info, [])
        assert result["is_complete"] is False

    def test_submission_missing_cn_code(self, assembler, importer_info):
        subs = [{"direct_emissions_tco2": 100, "indirect_emissions_tco2": 50}]
        result = assembler.validate_report_completeness(importer_info, subs)
        assert result["is_complete"] is False


# ===========================================================================
# TEST CLASS -- Provenance hash
# ===========================================================================

class TestProvenanceHash:
    """Tests for provenance hash consistency."""

    def test_hash_is_sha256(self, assembler, importer_info, sample_submissions):
        report = assembler.assemble_quarterly_report(
            quarter="2026Q1", importer_info=importer_info,
            submissions=sample_submissions,
        )
        h = report["provenance_hash"]
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_deterministic(self, assembler, importer_info, sample_submissions):
        h1 = assembler._generate_provenance_hash(
            "2026Q1", importer_info, sample_submissions
        )
        h2 = assembler._generate_provenance_hash(
            "2026Q1", importer_info, sample_submissions
        )
        assert h1 == h2

    def test_different_quarter_different_hash(self, assembler, importer_info,
                                              sample_submissions):
        h1 = assembler._generate_provenance_hash(
            "2026Q1", importer_info, sample_submissions
        )
        h2 = assembler._generate_provenance_hash(
            "2026Q2", importer_info, sample_submissions
        )
        assert h1 != h2

    def test_different_submissions_different_hash(self, assembler, importer_info):
        s1 = [{"cn_code": "72031000", "direct_emissions_tco2": 100,
               "indirect_emissions_tco2": 50}]
        s2 = [{"cn_code": "25231000", "direct_emissions_tco2": 80,
               "indirect_emissions_tco2": 20}]
        h1 = assembler._generate_provenance_hash("2026Q1", importer_info, s1)
        h2 = assembler._generate_provenance_hash("2026Q1", importer_info, s2)
        assert h1 != h2
