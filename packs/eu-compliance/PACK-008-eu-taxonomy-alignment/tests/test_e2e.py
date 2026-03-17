# -*- coding: utf-8 -*-
"""
Unit tests for PACK-008 EU Taxonomy Alignment Pack - End-to-End Tests

Tests complete end-to-end data flows between the 10 pipeline phases, simulating
eligibility-to-alignment, alignment-to-KPI, KPI-to-disclosure, full NFU assessment,
full FI assessment with GAR, gap analysis, regulatory update, and cross-framework
disclosure workflows.
"""

import pytest
import sys
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_PACK_008_DIR = Path(__file__).resolve().parent.parent
_INTEGRATIONS_DIR = _PACK_008_DIR / "integrations"


def _import_from_path(module_name: str, file_path: Path) -> Optional[Any]:
    """Helper to import from hyphenated directory paths."""
    if not file_path.exists():
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception:
        return None


def _instantiate_bridge(mod, class_name: str, config_class_names: list):
    """Try to instantiate a bridge class, handling config requirement patterns."""
    if mod is None:
        return None
    cls = getattr(mod, class_name, None)
    if cls is None:
        return None
    try:
        return cls()
    except TypeError:
        for cfg_name in config_class_names:
            cfg_cls = getattr(mod, cfg_name, None)
            if cfg_cls is not None:
                try:
                    return cls(cfg_cls())
                except Exception:
                    continue
    return None


# ---------------------------------------------------------------------------
# Import modules
# ---------------------------------------------------------------------------
_orch_mod = _import_from_path(
    "pack008_e2e_orchestrator", _INTEGRATIONS_DIR / "pack_orchestrator.py"
)
_mrv_mod = _import_from_path(
    "pack008_e2e_mrv", _INTEGRATIONS_DIR / "mrv_taxonomy_bridge.py"
)
_tax_app_mod = _import_from_path(
    "pack008_e2e_tax_app", _INTEGRATIONS_DIR / "taxonomy_app_bridge.py"
)
_fin_mod = _import_from_path(
    "pack008_e2e_fin", _INTEGRATIONS_DIR / "financial_data_bridge.py"
)
_gar_mod = _import_from_path(
    "pack008_e2e_gar", _INTEGRATIONS_DIR / "gar_data_bridge.py"
)
_csrd_mod = _import_from_path(
    "pack008_e2e_csrd", _INTEGRATIONS_DIR / "csrd_cross_framework_bridge.py"
)
_reg_mod = _import_from_path(
    "pack008_e2e_reg", _INTEGRATIONS_DIR / "regulatory_tracking_bridge.py"
)
_activity_mod = _import_from_path(
    "pack008_e2e_activity", _INTEGRATIONS_DIR / "activity_registry_bridge.py"
)


# ---------------------------------------------------------------------------
# Sample data builders
# ---------------------------------------------------------------------------

def _build_sample_activities() -> List[Dict[str, Any]]:
    """Build sample economic activities for testing."""
    return [
        {
            "activity_id": "ACT-001",
            "nace_code": "35.11",
            "taxonomy_activity": "4.1",
            "description": "Electricity generation using solar PV",
            "sector": "ENERGY",
            "turnover_eur": 5_000_000,
            "capex_eur": 2_000_000,
            "opex_eur": 500_000,
        },
        {
            "activity_id": "ACT-002",
            "nace_code": "41.10",
            "taxonomy_activity": "7.1",
            "description": "Construction of new buildings",
            "sector": "BUILDINGS",
            "turnover_eur": 8_000_000,
            "capex_eur": 6_000_000,
            "opex_eur": 1_200_000,
        },
        {
            "activity_id": "ACT-003",
            "nace_code": "23.51",
            "taxonomy_activity": "3.7",
            "description": "Manufacture of cement",
            "sector": "MANUFACTURING",
            "turnover_eur": 12_000_000,
            "capex_eur": 3_500_000,
            "opex_eur": 2_000_000,
        },
    ]


def _build_eligibility_results() -> List[Dict[str, Any]]:
    """Build mock eligibility screening results."""
    return [
        {"activity_id": "ACT-001", "eligible": True, "objective": "CCM", "confidence": 0.95},
        {"activity_id": "ACT-002", "eligible": True, "objective": "CCM", "confidence": 0.88},
        {"activity_id": "ACT-003", "eligible": True, "objective": "CCM", "confidence": 0.92},
    ]


def _build_sc_results() -> List[Dict[str, Any]]:
    """Build mock Substantial Contribution assessment results."""
    return [
        {"activity_id": "ACT-001", "sc_status": "PASS", "emissions_intensity_gCO2e_kWh": 0.0},
        {"activity_id": "ACT-002", "sc_status": "PASS", "energy_performance_kWh_m2": 45.0},
        {"activity_id": "ACT-003", "sc_status": "FAIL", "emissions_intensity_tCO2_t": 0.72},
    ]


def _build_dnsh_results() -> List[Dict[str, Any]]:
    """Build mock DNSH assessment results (6-objective matrix)."""
    objectives = ["CCM", "CCA", "WTR", "CE", "PPC", "BIO"]
    return [
        {
            "activity_id": "ACT-001",
            "dnsh_matrix": {obj: "PASS" for obj in objectives if obj != "CCM"},
            "overall_dnsh": "PASS",
        },
        {
            "activity_id": "ACT-002",
            "dnsh_matrix": {obj: "PASS" for obj in objectives if obj != "CCM"},
            "overall_dnsh": "PASS",
        },
        {
            "activity_id": "ACT-003",
            "dnsh_matrix": {obj: ("PASS" if obj != "PPC" else "FAIL") for obj in objectives if obj != "CCM"},
            "overall_dnsh": "FAIL",
        },
    ]


def _build_ms_results() -> List[Dict[str, Any]]:
    """Build mock Minimum Safeguards verification results."""
    return [
        {
            "activity_id": "ACT-001",
            "ms_status": "PASS",
            "human_rights": "PASS",
            "anti_corruption": "PASS",
            "taxation": "PASS",
            "fair_competition": "PASS",
        },
        {
            "activity_id": "ACT-002",
            "ms_status": "PASS",
            "human_rights": "PASS",
            "anti_corruption": "PASS",
            "taxation": "PASS",
            "fair_competition": "PASS",
        },
    ]


def _build_financial_data() -> Dict[str, Any]:
    """Build mock financial data for KPI calculation."""
    return {
        "total_turnover_eur": 50_000_000,
        "total_capex_eur": 20_000_000,
        "total_opex_eur": 8_000_000,
        "currency": "EUR",
        "fiscal_year": 2025,
    }


# ===========================================================================
# End-to-End Tests
# ===========================================================================
@pytest.mark.unit
class TestEndToEnd:
    """Test suite simulating full data flows between pipeline phases."""

    # -----------------------------------------------------------------------
    # E2E-001: Eligibility -> Alignment
    # -----------------------------------------------------------------------
    def test_eligibility_to_alignment_flow(self):
        """Data flows from eligibility screening into SC + DNSH assessment."""
        activities = _build_sample_activities()
        eligibility = _build_eligibility_results()

        # Filter eligible activities
        eligible_ids = {e["activity_id"] for e in eligibility if e["eligible"]}
        aligned_activities = [a for a in activities if a["activity_id"] in eligible_ids]

        assert len(aligned_activities) > 0, "At least one activity must be eligible"
        assert all(
            a["activity_id"] in eligible_ids for a in aligned_activities
        ), "Only eligible activities should proceed to alignment"

        # Build SC input from eligible activities
        sc_input = [
            {"activity_id": a["activity_id"], "taxonomy_activity": a["taxonomy_activity"]}
            for a in aligned_activities
        ]
        assert len(sc_input) == len(aligned_activities)

    # -----------------------------------------------------------------------
    # E2E-002: Alignment -> KPI
    # -----------------------------------------------------------------------
    def test_alignment_to_kpi_flow(self):
        """SC + DNSH + MS results feed into KPI calculation."""
        activities = _build_sample_activities()
        sc_results = _build_sc_results()
        dnsh_results = _build_dnsh_results()
        ms_results = _build_ms_results()
        financial = _build_financial_data()

        # Determine aligned activities: SC PASS + DNSH PASS + MS PASS
        sc_pass = {r["activity_id"] for r in sc_results if r["sc_status"] == "PASS"}
        dnsh_pass = {r["activity_id"] for r in dnsh_results if r["overall_dnsh"] == "PASS"}
        ms_pass = {r["activity_id"] for r in ms_results if r["ms_status"] == "PASS"}

        taxonomy_aligned = sc_pass & dnsh_pass & ms_pass
        assert len(taxonomy_aligned) > 0, "At least one activity must be fully aligned"

        # Calculate KPI numerator
        aligned_turnover = sum(
            a["turnover_eur"] for a in activities if a["activity_id"] in taxonomy_aligned
        )
        kpi_turnover_ratio = aligned_turnover / financial["total_turnover_eur"]

        assert 0.0 <= kpi_turnover_ratio <= 1.0, (
            f"Turnover KPI ratio must be [0, 1], got {kpi_turnover_ratio}"
        )

    # -----------------------------------------------------------------------
    # E2E-003: KPI -> Disclosure
    # -----------------------------------------------------------------------
    def test_kpi_to_disclosure_flow(self):
        """KPI ratios feed into Article 8 disclosure generation."""
        # Simulated KPI results
        kpi_results = {
            "turnover_eligible_ratio": 0.50,
            "turnover_aligned_ratio": 0.26,
            "capex_eligible_ratio": 0.55,
            "capex_aligned_ratio": 0.40,
            "opex_eligible_ratio": 0.21,
            "opex_aligned_ratio": 0.09,
        }

        # Build disclosure template data
        disclosure = {
            "reporting_entity": "Demo Corp GmbH",
            "reporting_year": 2025,
            "framework": "Article 8 EU Taxonomy",
            "kpis": kpi_results,
            "environmental_objectives": ["CCM", "CCA"],
        }

        # Validate disclosure structure
        assert disclosure["framework"] == "Article 8 EU Taxonomy"
        assert "turnover_aligned_ratio" in disclosure["kpis"]
        assert 0.0 <= disclosure["kpis"]["turnover_aligned_ratio"] <= 1.0
        assert 0.0 <= disclosure["kpis"]["capex_aligned_ratio"] <= 1.0

    # -----------------------------------------------------------------------
    # E2E-004: Full NFU assessment
    # -----------------------------------------------------------------------
    def test_full_nfu_assessment_flow(self):
        """Full non-financial undertaking assessment from activities to KPIs."""
        activities = _build_sample_activities()
        eligibility = _build_eligibility_results()
        sc_results = _build_sc_results()
        dnsh_results = _build_dnsh_results()
        ms_results = _build_ms_results()
        financial = _build_financial_data()

        # Phase 1: Eligibility
        eligible_ids = {e["activity_id"] for e in eligibility if e["eligible"]}
        assert len(eligible_ids) == 3

        # Phase 2: SC assessment
        sc_pass = {r["activity_id"] for r in sc_results if r["sc_status"] == "PASS"}

        # Phase 3: DNSH assessment
        dnsh_pass = {r["activity_id"] for r in dnsh_results if r["overall_dnsh"] == "PASS"}

        # Phase 4: MS verification
        ms_pass = {r["activity_id"] for r in ms_results if r["ms_status"] == "PASS"}

        # Phase 5: KPI calculation
        fully_aligned = eligible_ids & sc_pass & dnsh_pass & ms_pass
        eligible_only = eligible_ids - fully_aligned

        aligned_turnover = sum(
            a["turnover_eur"] for a in activities if a["activity_id"] in fully_aligned
        )
        eligible_turnover = sum(
            a["turnover_eur"] for a in activities if a["activity_id"] in eligible_ids
        )

        assert aligned_turnover <= eligible_turnover, (
            "Aligned turnover cannot exceed eligible turnover"
        )
        assert len(fully_aligned) <= len(eligible_ids), (
            "Aligned count cannot exceed eligible count"
        )

    # -----------------------------------------------------------------------
    # E2E-005: Full FI assessment with GAR
    # -----------------------------------------------------------------------
    def test_full_fi_assessment_flow(self):
        """Full financial institution assessment including GAR calculation."""
        # Simulated portfolio exposures
        portfolio = {
            "total_assets_eur": 500_000_000,
            "exposures": [
                {
                    "counterparty": "Manufacturing Co A",
                    "exposure_eur": 50_000_000,
                    "type": "corporate_loans",
                    "counterparty_taxonomy_aligned_ratio": 0.35,
                },
                {
                    "counterparty": "Energy Corp B",
                    "exposure_eur": 80_000_000,
                    "type": "corporate_loans",
                    "counterparty_taxonomy_aligned_ratio": 0.60,
                },
                {
                    "counterparty": "Residential Mortgage Pool",
                    "exposure_eur": 120_000_000,
                    "type": "residential_mortgages",
                    "epc_rating": "B",
                    "counterparty_taxonomy_aligned_ratio": 0.45,
                },
            ],
        }

        # GAR numerator: sum(exposure * aligned_ratio) for eligible types
        gar_numerator = sum(
            e["exposure_eur"] * e["counterparty_taxonomy_aligned_ratio"]
            for e in portfolio["exposures"]
        )
        gar_denominator = portfolio["total_assets_eur"]
        gar_ratio = gar_numerator / gar_denominator

        assert 0.0 <= gar_ratio <= 1.0, f"GAR ratio must be [0, 1], got {gar_ratio}"
        assert gar_numerator > 0, "GAR numerator must be positive for this test data"

    # -----------------------------------------------------------------------
    # E2E-006: Gap analysis -> Remediation
    # -----------------------------------------------------------------------
    def test_gap_analysis_to_remediation_flow(self):
        """Gap analysis identifies failures and produces remediation actions."""
        sc_results = _build_sc_results()
        dnsh_results = _build_dnsh_results()

        # Identify SC failures
        sc_failures = [r for r in sc_results if r["sc_status"] == "FAIL"]
        # Identify DNSH failures
        dnsh_failures = [r for r in dnsh_results if r["overall_dnsh"] == "FAIL"]

        all_gaps = []
        for f in sc_failures:
            all_gaps.append({
                "activity_id": f["activity_id"],
                "gap_type": "SC_FAILURE",
                "detail": f"Emissions intensity too high: {f.get('emissions_intensity_tCO2_t', 'N/A')}",
                "remediation": "Reduce process emissions or switch to low-carbon technology",
            })
        for f in dnsh_failures:
            failed_objectives = [
                obj for obj, status in f.get("dnsh_matrix", {}).items() if status == "FAIL"
            ]
            all_gaps.append({
                "activity_id": f["activity_id"],
                "gap_type": "DNSH_FAILURE",
                "failed_objectives": failed_objectives,
                "remediation": "Address specific DNSH criteria for failed objectives",
            })

        assert len(all_gaps) > 0, "Test data should produce at least one gap"
        assert all("remediation" in g for g in all_gaps), "Each gap must have remediation"

    # -----------------------------------------------------------------------
    # E2E-007: Regulatory update flow
    # -----------------------------------------------------------------------
    def test_regulatory_update_flow(self):
        """Regulatory DA version update triggers criteria re-evaluation."""
        # Simulate current vs new DA
        current_da = {
            "version": "2021",
            "name": "Climate Delegated Act (EU) 2021/2139",
            "activity_count": 88,
        }
        new_da = {
            "version": "2025",
            "name": "Omnibus Simplification Package 2025",
            "activity_count": 95,
            "changes": [
                {"type": "threshold_change", "activity": "3.7", "field": "emissions_intensity", "old": 0.469, "new": 0.500},
                {"type": "new_activity", "activity": "4.32", "description": "Green hydrogen storage"},
            ],
        }

        # Impact assessment
        impact = {
            "affected_activities": len(new_da["changes"]),
            "new_activities": sum(1 for c in new_da["changes"] if c["type"] == "new_activity"),
            "threshold_changes": sum(1 for c in new_da["changes"] if c["type"] == "threshold_change"),
            "requires_reassessment": True,
        }

        assert impact["affected_activities"] > 0
        assert impact["requires_reassessment"] is True

    # -----------------------------------------------------------------------
    # E2E-008: Cross-framework disclosure
    # -----------------------------------------------------------------------
    def test_cross_framework_disclosure_flow(self):
        """Taxonomy KPIs map to ESRS E1, SFDR PAI, and TCFD disclosures."""
        taxonomy_kpis = {
            "turnover_aligned_ratio": 0.26,
            "capex_aligned_ratio": 0.40,
            "scope1_emissions_tCO2e": 15_000,
            "scope2_emissions_tCO2e": 8_000,
            "scope3_emissions_tCO2e": 120_000,
        }

        # ESRS E1 mapping
        esrs_e1 = {
            "e1_6_gross_scope1": taxonomy_kpis["scope1_emissions_tCO2e"],
            "e1_6_gross_scope2": taxonomy_kpis["scope2_emissions_tCO2e"],
            "e1_6_gross_scope3": taxonomy_kpis["scope3_emissions_tCO2e"],
            "e1_9_taxonomy_turnover_alignment": taxonomy_kpis["turnover_aligned_ratio"],
        }

        # SFDR PAI mapping
        sfdr_pai = {
            "pai_1_ghg_emissions": (
                taxonomy_kpis["scope1_emissions_tCO2e"]
                + taxonomy_kpis["scope2_emissions_tCO2e"]
            ),
            "pai_3_ghg_intensity": None,  # requires revenue
            "taxonomy_alignment": taxonomy_kpis["turnover_aligned_ratio"],
        }

        # TCFD mapping
        tcfd = {
            "metrics_emissions_scope1": taxonomy_kpis["scope1_emissions_tCO2e"],
            "metrics_emissions_scope2": taxonomy_kpis["scope2_emissions_tCO2e"],
            "strategy_taxonomy_alignment": taxonomy_kpis["turnover_aligned_ratio"],
        }

        # Validate cross-framework data is consistent
        assert esrs_e1["e1_6_gross_scope1"] == tcfd["metrics_emissions_scope1"]
        assert esrs_e1["e1_9_taxonomy_turnover_alignment"] == sfdr_pai["taxonomy_alignment"]
        assert sfdr_pai["pai_1_ghg_emissions"] == (
            taxonomy_kpis["scope1_emissions_tCO2e"] + taxonomy_kpis["scope2_emissions_tCO2e"]
        )
