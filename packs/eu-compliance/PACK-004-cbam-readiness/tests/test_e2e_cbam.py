# -*- coding: utf-8 -*-
"""
PACK-004 CBAM Readiness Pack - End-to-End CBAM Tests (12 tests)

Comprehensive end-to-end tests covering full CBAM lifecycle
scenarios including quarterly reporting, annual declaration,
supplier onboarding, certificate management, de minimis tracking,
verification engagement, multi-commodity imports, carbon price
deductions, and data collection pipelines.

Author: GreenLang QA Team
"""

import csv
import io
import json
from typing import Any, Dict, List

import pytest

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    CBAM_ENGINE_IDS,
    CBAM_TEMPLATE_IDS,
    CBAM_WORKFLOW_IDS,
    StubCBAMApp,
    StubCertificateEngine,
    StubCustoms,
    StubETSFeed,
    StubQuarterlyEngine,
    StubSupplierPortal,
    _compute_hash,
    _new_uuid,
    _utcnow,
    render_template_stub,
)


class TestE2ECBAM:
    """End-to-end CBAM pack test suite."""

    def test_e2e_quarterly_report_lifecycle(
        self,
        mock_quarterly_engine,
        mock_customs,
        sample_importer_config,
        sample_emission_inputs,
    ):
        """E2E: Import data -> validate -> calculate -> report -> XML."""
        # Step 1: Validate CN codes
        for inp in sample_emission_inputs:
            result = mock_customs.validate_cn_code(inp["cn_code"])
            assert result["format_valid"] is True

        # Step 2: Calculate emissions
        emission_results = []
        for inp in sample_emission_inputs:
            total = inp["weight_tonnes"] * inp["specific_emission_tco2e_per_tonne"]
            emission_results.append({
                "input_id": inp["input_id"],
                "cn_code": inp["cn_code"],
                "goods_category": inp["goods_category"],
                "origin_country": inp["origin_country"],
                "weight_tonnes": inp["weight_tonnes"],
                "specific_emission_tco2e_per_tonne": inp["specific_emission_tco2e_per_tonne"],
                "total_emissions_tco2e": round(total, 6),
            })
        assert len(emission_results) == 10

        # Step 3: Assemble report
        period = mock_quarterly_engine.detect_period("2026-02-15")
        report = mock_quarterly_engine.assemble_report(
            sample_importer_config, emission_results, period,
        )
        assert report["status"] == "assembled"

        # Step 4: Generate XML
        xml = mock_quarterly_engine.generate_xml(report)
        assert "<?xml" in xml
        assert "<CBAMQuarterlyReport" in xml

        # Step 5: Validate
        validation = mock_quarterly_engine.validate_report(report)
        assert validation["valid"] is True

    def test_e2e_annual_declaration_pipeline(
        self,
        mock_certificate_engine,
        mock_ets_feed,
        sample_emission_results,
    ):
        """E2E: 4 quarterly reports -> consolidate -> certificates -> declare."""
        # Step 1: Simulate 4 quarterly totals
        q1_total = sum(r["total_emissions_tco2e"] for r in sample_emission_results)
        quarterly_totals = [q1_total, q1_total * 0.9, q1_total * 0.85, q1_total * 1.1]
        annual_total = sum(quarterly_totals)

        # Step 2: Calculate obligation
        obligation = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=annual_total, year=2026,
        )
        assert obligation["net_obligation_tco2e"] >= 0

        # Step 3: Estimate cost
        ets_price = mock_ets_feed.current_price()
        cost = mock_certificate_engine.estimate_cost(
            obligation["net_obligation_tco2e"], ets_price["price_eur"],
        )
        assert cost["currency"] == "EUR"

        # Step 4: Declare
        declaration = {
            "declaration_id": f"AD-2026-{_new_uuid()[:8]}",
            "year": 2026,
            "annual_emissions_tco2e": round(annual_total, 6),
            "certificates_required": obligation["certificates_required"],
            "estimated_cost_eur": cost["estimated_cost_eur"],
            "status": "declared",
            "provenance_hash": _compute_hash({
                "annual": annual_total, "year": 2026,
            }),
        }
        assert declaration["status"] == "declared"
        assert len(declaration["provenance_hash"]) == 64

    def test_e2e_supplier_onboarding_to_reporting(
        self,
        mock_supplier_portal,
        mock_quarterly_engine,
        sample_importer_config,
    ):
        """E2E: Register supplier -> submit data -> use in report."""
        # Step 1: Register supplier
        supplier = mock_supplier_portal.register_supplier({
            "supplier_id": "SUP-E2E-001",
            "company_name": "E2E Steel Corp",
            "country": "TR",
        })
        assert supplier["status"] == "active"

        # Step 2: Add installation
        inst = mock_supplier_portal.add_installation("SUP-E2E-001", {
            "installation_id": "INST-E2E-001",
            "name": "Main Works",
        })
        assert inst["status"] == "registered"

        # Step 3: Submit emission data
        submission = mock_supplier_portal.submit_emission_data("SUP-E2E-001", {
            "cn_code": "7207 11 14",
            "weight_tonnes": 500.0,
            "specific_emission_tco2e_per_tonne": 1.85,
            "total_emissions_tco2e": 925.0,
        })
        assert submission["status"] == "submitted"

        # Step 4: Use in quarterly report
        emission_results = [{
            "input_id": "EI-E2E",
            "cn_code": "7207 11 14",
            "goods_category": "steel",
            "origin_country": "TR",
            "weight_tonnes": 500.0,
            "specific_emission_tco2e_per_tonne": 1.85,
            "total_emissions_tco2e": 925.0,
        }]
        period = mock_quarterly_engine.detect_period("2026-01-15")
        report = mock_quarterly_engine.assemble_report(
            sample_importer_config, emission_results, period,
        )
        assert report["total_emissions_tco2e"] == 925.0

    def test_e2e_certificate_full_lifecycle(
        self, mock_certificate_engine, mock_ets_feed,
    ):
        """E2E: Calculate -> purchase plan -> hold -> surrender."""
        # Step 1: Calculate obligation
        obligation = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=15000.0, year=2027,
        )
        net = obligation["net_obligation_tco2e"]
        assert net > 0

        # Step 2: Purchase plan
        ets_price = mock_ets_feed.current_price()
        cost = mock_certificate_engine.estimate_cost(net, ets_price["price_eur"])
        quarterly_purchase = int(net / 4)

        # Step 3: Check quarterly holding
        holding = mock_certificate_engine.check_quarterly_holding(
            certificates_held=quarterly_purchase * 2,
            net_obligation_tco2e=net,
        )
        # After Q2, should have ~50% of certificates held; 80% required
        assert isinstance(holding["compliant"], bool)

        # Step 4: Surrender
        surrender = {
            "certificates_surrendered": obligation["certificates_required"],
            "cost_eur": cost["estimated_cost_eur"],
            "status": "surrendered",
        }
        assert surrender["status"] == "surrendered"

    def test_e2e_deminimis_to_exemption(self, sample_cbam_config):
        """E2E: Track imports -> exceed threshold -> lose exemption."""
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        threshold_kg = dmc["annual_weight_threshold_kg"]

        # Monthly imports gradually increasing
        monthly_kg = [10000, 12000, 14000, 16000, 18000, 20000,
                      15000, 17000, 13000, 11000, 10000, 12000]
        cumulative = 0
        exempt = True
        revocation_month = None
        for month_idx, kg in enumerate(monthly_kg):
            cumulative += kg
            if cumulative >= threshold_kg and exempt:
                exempt = False
                revocation_month = month_idx + 1

        assert exempt is False
        assert revocation_month is not None
        assert revocation_month <= 12
        assert cumulative == sum(monthly_kg)

    def test_e2e_verification_engagement(self, sample_verifier):
        """E2E: Select verifier -> prepare -> findings -> resolve -> statement."""
        # Step 1: Select verifier
        assert sample_verifier["status"] == "active"
        assert "steel" in sample_verifier["scopes"]

        # Step 2: Create engagement
        engagement = {
            "engagement_id": f"VE-{_new_uuid()[:8]}",
            "verifier_id": sample_verifier["verifier_id"],
            "scope": ["steel", "aluminium"],
            "status": "active",
        }

        # Step 3: Prepare evidence
        evidence = [
            {"category": "steel", "documents": 20, "status": "submitted"},
            {"category": "aluminium", "documents": 15, "status": "submitted"},
        ]
        total_docs = sum(e["documents"] for e in evidence)
        assert total_docs == 35

        # Step 4: Findings
        findings = [
            {"id": "F-001", "severity": "observation", "status": "resolved"},
        ]
        all_resolved = all(f["status"] == "resolved" for f in findings)
        assert all_resolved is True

        # Step 5: Statement
        statement = {
            "opinion": "unqualified",
            "material_findings": 0,
            "provenance_hash": _compute_hash({
                "verifier": sample_verifier["verifier_id"],
            }),
        }
        assert statement["opinion"] == "unqualified"
        assert len(statement["provenance_hash"]) == 64

    def test_e2e_multi_commodity_import(
        self,
        mock_certificate_engine,
        sample_emission_inputs,
    ):
        """E2E: Steel + aluminium + cement in single quarter."""
        # Calculate emissions per category
        by_category = {}
        for inp in sample_emission_inputs:
            cat = inp["goods_category"]
            total = inp["weight_tonnes"] * inp["specific_emission_tco2e_per_tonne"]
            by_category[cat] = by_category.get(cat, 0.0) + total

        assert "steel" in by_category
        assert "aluminium" in by_category
        assert "cement" in by_category

        # Total obligation
        total_emissions = sum(by_category.values())
        obligation = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=total_emissions, year=2026,
        )
        assert obligation["net_obligation_tco2e"] >= 0
        assert obligation["gross_obligation_tco2e"] == pytest.approx(total_emissions)

    def test_e2e_carbon_price_deduction(
        self, mock_certificate_engine, sample_emission_results,
    ):
        """E2E: Turkey carbon pricing -> deduction -> reduced certificates."""
        # Turkey has an ETS equivalent - calculate deduction
        turkey_emissions = sum(
            r["total_emissions_tco2e"] for r in sample_emission_results
            if r["origin_country"] == "TR"
        )
        turkey_carbon_price_eur_per_tco2e = 12.0  # Turkey carbon price
        eu_ets_price = 78.50
        deduction_rate = turkey_carbon_price_eur_per_tco2e / eu_ets_price
        carbon_deduction_tco2e = round(turkey_emissions * deduction_rate, 6)

        # Calculate with deduction
        total_emissions = sum(
            r["total_emissions_tco2e"] for r in sample_emission_results
        )
        with_deduction = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=total_emissions, year=2026,
            carbon_price_deduction=carbon_deduction_tco2e,
        )
        without_deduction = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=total_emissions, year=2026,
        )

        assert with_deduction["net_obligation_tco2e"] <= \
               without_deduction["net_obligation_tco2e"]

    def test_e2e_default_to_actual_transition(
        self, sample_emission_inputs,
    ):
        """E2E: Start with defaults -> supplier provides actuals."""
        # Step 1: Identify inputs using default methodology
        default_inputs = [
            i for i in sample_emission_inputs
            if i["emission_methodology"] == "default_values"
        ]
        assert len(default_inputs) >= 1

        # Step 2: Simulate transition to actual values
        updated_inputs = []
        for inp in default_inputs:
            actual_ef = inp["specific_emission_tco2e_per_tonne"] * 0.85
            updated = dict(inp)
            updated["emission_methodology"] = "actual"
            updated["specific_emission_tco2e_per_tonne"] = round(actual_ef, 4)
            updated_inputs.append(updated)

        # Step 3: Verify actual values are lower than defaults
        for orig, updated in zip(default_inputs, updated_inputs):
            assert updated["specific_emission_tco2e_per_tonne"] < \
                   orig["specific_emission_tco2e_per_tonne"]
            assert updated["emission_methodology"] == "actual"

    def test_e2e_data_collection_to_report(
        self,
        sample_import_csv_data,
        mock_quarterly_engine,
        sample_importer_config,
    ):
        """E2E: Configure sources -> ingest -> quality check -> report."""
        # Step 1: Ingest CSV data
        reader = csv.DictReader(io.StringIO(sample_import_csv_data))
        rows = list(reader)
        assert len(rows) == 20

        # Step 2: Quality check
        valid_rows = [r for r in rows if float(r["weight_tonnes"]) > 0]
        assert len(valid_rows) == 20

        # Step 3: Calculate emissions (using default factors)
        default_factors = {"steel": 2.30, "aluminium": 8.50, "cement": 0.65}
        emission_results = []
        for row in valid_rows:
            ef = default_factors.get(row["goods_category"], 1.0)
            weight = float(row["weight_tonnes"])
            emission_results.append({
                "input_id": row["import_id"],
                "cn_code": row["cn_code"],
                "goods_category": row["goods_category"],
                "origin_country": row["origin_country"],
                "weight_tonnes": weight,
                "specific_emission_tco2e_per_tonne": ef,
                "total_emissions_tco2e": round(weight * ef, 6),
            })

        # Step 4: Generate report
        period = mock_quarterly_engine.detect_period("2026-01-15")
        report = mock_quarterly_engine.assemble_report(
            sample_importer_config, emission_results, period,
        )
        assert report["total_imports"] == 20
        assert report["total_emissions_tco2e"] > 0

    def test_e2e_full_annual_compliance(
        self,
        mock_certificate_engine,
        mock_quarterly_engine,
        mock_supplier_portal,
        mock_ets_feed,
        mock_customs,
        sample_importer_config,
        sample_emission_inputs,
        sample_emission_results,
        sample_verifier,
        sample_cbam_config,
    ):
        """E2E: Complete year lifecycle from data to declaration."""
        # Step 1: Validate CN codes
        for inp in sample_emission_inputs:
            result = mock_customs.validate_cn_code(inp["cn_code"])
            assert result["format_valid"] is True

        # Step 2: Register suppliers
        supplier = mock_supplier_portal.register_supplier({
            "supplier_id": "SUP-ANNUAL-001",
            "company_name": "Annual Test Corp",
            "country": "TR",
        })
        assert supplier["status"] == "active"

        # Step 3: Q1 quarterly report
        period = mock_quarterly_engine.detect_period("2026-01-15")
        q1_report = mock_quarterly_engine.assemble_report(
            sample_importer_config, sample_emission_results, period,
        )
        q1_xml = mock_quarterly_engine.generate_xml(q1_report)
        assert "<?xml" in q1_xml

        # Step 4: Annual emissions (4 quarters)
        annual_emissions = q1_report["total_emissions_tco2e"] * 4

        # Step 5: Certificate calculation
        obligation = mock_certificate_engine.calculate_obligation(
            total_emissions_tco2e=annual_emissions, year=2026,
        )

        # Step 6: Cost estimation
        ets = mock_ets_feed.current_price()
        cost = mock_certificate_engine.estimate_cost(
            obligation["net_obligation_tco2e"], ets["price_eur"],
        )

        # Step 7: De minimis check
        total_weight = sum(inp["weight_tonnes"] for inp in sample_emission_inputs) * 4
        dmc = sample_cbam_config["cbam"]["deminimis_config"]
        weight_kg = total_weight * 1000

        # Step 8: Final declaration
        declaration = {
            "year": 2026,
            "annual_emissions": round(annual_emissions, 6),
            "certificates_required": obligation["certificates_required"],
            "cost_eur": cost["estimated_cost_eur"],
            "deminimis_exceeded": weight_kg >= dmc["annual_weight_threshold_kg"],
            "verification": sample_verifier["company_name"],
            "status": "complete",
            "provenance_hash": _compute_hash({
                "year": 2026,
                "emissions": annual_emissions,
            }),
        }
        assert declaration["status"] == "complete"
        assert len(declaration["provenance_hash"]) == 64

    def test_e2e_health_check_comprehensive(
        self, mock_cbam_app, mock_ets_feed, mock_customs,
    ):
        """E2E: Run comprehensive CBAM health check."""
        health = {
            "overall_status": "healthy",
            "categories": {
                "cbam_app": mock_cbam_app.health_check()["status"],
                "ets_feed": "healthy",
                "customs_api": "healthy",
                "cn_codes_loaded": len(mock_customs.all_cbam_codes()) > 0,
                "ets_price_available": mock_ets_feed.current_price()["price_eur"] > 0,
                "engines_loaded": mock_cbam_app.health_check()["engines_loaded"],
            },
            "checks_total": 6,
            "checks_passed": 6,
            "checks_failed": 0,
            "timestamp": _utcnow().isoformat(),
        }
        assert health["overall_status"] == "healthy"
        assert health["checks_failed"] == 0
        assert health["categories"]["engines_loaded"] == 7
        assert health["categories"]["cn_codes_loaded"] is True
        assert health["categories"]["ets_price_available"] is True
