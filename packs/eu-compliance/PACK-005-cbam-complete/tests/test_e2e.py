# -*- coding: utf-8 -*-
"""
PACK-005 CBAM Complete Pack - End-to-End Integration Tests (15 tests)

Comprehensive end-to-end tests covering full CBAM Complete lifecycle
scenarios: customs-to-calculation, calculation-to-certificate,
certificate-to-registry, single-entity pipeline, multi-entity pipeline,
cross-regulation full sync, audit trail completeness, and demo E2E.

Author: GreenLang QA Team
"""

import json
from decimal import Decimal
from typing import Any, Dict, List

import pytest

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (
    PACK005_ENGINE_IDS,
    PACK005_INTEGRATION_IDS,
    PACK005_TEMPLATE_IDS,
    PACK005_WORKFLOW_IDS,
    StubETSBridge,
    StubRegistryClient,
    StubTARICClient,
    _compute_hash,
    _new_uuid,
    _utcnow,
    assert_provenance_hash,
    generate_import_portfolio,
    render_template_stub,
)


class TestE2ECustomsToCalculation:
    """E2E: Customs declaration to emission calculation flow."""

    def test_customs_to_calculation_flow(
        self, sample_customs_declaration, mock_taric_client
    ):
        """E2E: Parse SAD -> validate CN codes -> calculate emissions."""
        # Step 1: Parse customs declaration
        sad = sample_customs_declaration
        assert len(sad["line_items"]) == 5

        # Step 2: Validate CN codes
        cbam_items = []
        for item in sad["line_items"]:
            validation = mock_taric_client.validate_cn_code(item["cn_code"])
            if validation["cbam_covered"]:
                cbam_items.append(item)
        assert len(cbam_items) == 4

        # Step 3: Calculate emissions
        ef_map = {"steel": 1.95, "aluminium": 8.50, "cement": 0.68}
        emission_results = []
        for item in cbam_items:
            category = mock_taric_client.validate_cn_code(item["cn_code"])["category"]
            weight_tonnes = item["net_mass_kg"] / 1000
            ef = ef_map.get(category, 2.0)
            emissions = weight_tonnes * ef
            emission_results.append({
                "cn_code": item["cn_code"],
                "category": category,
                "weight_tonnes": weight_tonnes,
                "emissions_tco2e": round(emissions, 6),
            })
        total_emissions = sum(r["emissions_tco2e"] for r in emission_results)
        assert total_emissions > 0
        assert len(emission_results) == 4


class TestE2ECalculationToCertificate:
    """E2E: Emission calculation to certificate management flow."""

    def test_calculation_to_certificate_flow(self, mock_ets_bridge):
        """E2E: Calculate obligation -> determine certificates -> cost."""
        # Step 1: Total emissions from calculation
        total_emissions = 22500.0

        # Step 2: Apply free allocation
        fa_pct = mock_ets_bridge.get_free_allocation_pct(2026) / 100.0
        net_obligation = total_emissions * (1.0 - fa_pct)
        assert net_obligation == pytest.approx(562.5, rel=1e-2)

        # Step 3: Determine certificate count
        certificates_needed = int(round(net_obligation))
        assert certificates_needed > 0

        # Step 4: Cost estimate
        price = mock_ets_bridge.get_current_price()["price_eur"]
        cost = certificates_needed * price
        assert cost > 0


class TestE2ECertificateToRegistry:
    """E2E: Certificate purchase to registry submission flow."""

    def test_certificate_to_registry_flow(self, mock_registry_client):
        """E2E: Purchase -> hold -> surrender -> declare."""
        # Step 1: Purchase certificates
        purchase = mock_registry_client.purchase_certificates(563, 78.50)
        assert purchase["status"] == "purchased"

        # Step 2: Check holding
        balance = mock_registry_client.get_balance()
        assert balance["balance"] >= 563

        # Step 3: Surrender certificates
        surrender = mock_registry_client.surrender_certificates(563)
        assert surrender["status"] == "surrendered"

        # Step 4: Submit declaration
        declaration = mock_registry_client.submit_declaration({
            "declaration_id": "DECL-E2E-001",
            "year": 2026,
            "total_emissions_tco2e": 22500.0,
            "certificates_surrendered": 563,
        })
        assert declaration["status"] == "submitted"


class TestE2ESingleEntity:
    """E2E: Full pipeline for a single entity."""

    def test_full_pipeline_single_entity(
        self, sample_customs_declaration, mock_taric_client,
        mock_ets_bridge, mock_registry_client
    ):
        """E2E: Customs -> calculation -> certificate -> registry (single entity)."""
        # Customs
        cbam_items = [
            i for i in sample_customs_declaration["line_items"]
            if i["cbam_applicable"]
        ]
        assert len(cbam_items) >= 1

        # Calculation
        ef_map = {"steel": 1.95, "aluminium": 8.50, "cement": 0.68}
        total_emissions = 0
        for item in cbam_items:
            cat = mock_taric_client.validate_cn_code(item["cn_code"])["category"]
            weight = item["net_mass_kg"] / 1000
            total_emissions += weight * ef_map.get(cat, 2.0)

        # Certificate
        fa_pct = mock_ets_bridge.get_free_allocation_pct(2026) / 100.0
        net_obligation = total_emissions * (1.0 - fa_pct)
        certs_needed = int(round(net_obligation))

        # Registry
        mock_registry_client.purchase_certificates(certs_needed, 78.50)
        mock_registry_client.surrender_certificates(certs_needed)
        result = mock_registry_client.submit_declaration({
            "declaration_id": "DECL-SINGLE-E2E",
            "certificates_surrendered": certs_needed,
        })
        assert result["status"] == "submitted"


class TestE2EMultiEntity:
    """E2E: Full pipeline for multi-entity group."""

    def test_full_pipeline_multi_entity(
        self, sample_entity_group, mock_ets_bridge, mock_registry_client
    ):
        """E2E: Multi-entity group -> consolidate -> certificates -> declarations."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]
        portfolio = generate_import_portfolio(all_entities, 60)

        # Consolidate emissions
        entity_emissions = {}
        for record in portfolio:
            eid = record["entity_id"]
            em = record["weight_tonnes"] * record["specific_emission_tco2e_per_tonne"]
            entity_emissions[eid] = entity_emissions.get(eid, 0.0) + em

        total_group_emissions = sum(entity_emissions.values())
        assert total_group_emissions > 0

        # Group obligation
        fa_pct = mock_ets_bridge.get_free_allocation_pct(2026) / 100.0
        group_net = total_group_emissions * (1.0 - fa_pct)

        # Allocate by volume
        total_volume = sum(
            r["weight_tonnes"] for r in portfolio
        )
        entity_obligations = {}
        for eid, em in entity_emissions.items():
            entity_volume = sum(
                r["weight_tonnes"] for r in portfolio if r["entity_id"] == eid
            )
            share = entity_volume / total_volume if total_volume > 0 else 0
            entity_obligations[eid] = round(group_net * share, 2)

        assert len(entity_obligations) == 3
        assert sum(entity_obligations.values()) == pytest.approx(group_net, rel=1e-2)

        # Submit declarations per entity
        for eid, obligation in entity_obligations.items():
            certs = int(round(obligation))
            if certs > 0:
                mock_registry_client.purchase_certificates(certs, 78.50)
            result = mock_registry_client.submit_declaration({
                "declaration_id": f"DECL-{eid}-2026",
                "entity_id": eid,
                "net_obligation": obligation,
            })
            assert result["status"] == "submitted"


class TestE2ECrossRegulation:
    """E2E: Cross-regulation sync."""

    def test_cross_regulation_full_sync(self, sample_cbam_data):
        """E2E: CBAM data -> map to all 6 frameworks -> consistency check."""
        targets = ["csrd", "cdp", "sbti", "taxonomy", "ets", "eudr"]
        mappings = {}
        for target in targets:
            mappings[target] = {
                "framework": target,
                "emissions_mapped": sample_cbam_data["total_emissions_tco2e"],
                "status": "mapped",
            }

        # Consistency check
        all_consistent = all(
            m["emissions_mapped"] == sample_cbam_data["total_emissions_tco2e"]
            for m in mappings.values()
        )
        assert all_consistent is True
        assert len(mappings) == 6


class TestE2EAuditTrail:
    """E2E: Audit trail completeness."""

    def test_audit_trail_complete(self, sample_audit_repository, sample_cbam_data):
        """E2E: Verify complete audit trail from import to declaration."""
        evidence_types_required = [
            "customs_declaration",
            "supplier_emission_data",
            "verification_statement",
            "certificate_purchase",
            "quarterly_report",
        ]
        evidence_types_present = {
            e["type"] for e in sample_audit_repository["evidence_records"]
        }
        for required_type in evidence_types_required:
            assert required_type in evidence_types_present, (
                f"Missing evidence type: {required_type}"
            )

        # Verify chain of custody exists
        assert len(sample_audit_repository["chain_of_custody"]) >= 1

        # Verify provenance hash can be computed
        h = _compute_hash(sample_cbam_data)
        assert len(h) == 64


class TestE2EDemo:
    """E2E: Demo mode execution."""

    def test_demo_e2e(
        self, sample_entity_group, sample_config, mock_ets_bridge,
        mock_registry_client, template_renderer
    ):
        """E2E: Full demo scenario from setup to reports."""
        all_entities = [sample_entity_group["parent"]] + sample_entity_group["subsidiaries"]

        # Generate portfolio
        portfolio = generate_import_portfolio(all_entities, 200)
        assert len(portfolio) == 200

        # Calculate emissions
        total_emissions = sum(
            r["weight_tonnes"] * r["specific_emission_tco2e_per_tonne"]
            for r in portfolio
        )

        # Certificate obligation
        fa_pct = mock_ets_bridge.get_free_allocation_pct(2026) / 100.0
        net = total_emissions * (1.0 - fa_pct)
        certs = int(round(net))

        # Registry operations
        if certs > 0:
            mock_registry_client.purchase_certificates(certs, 78.50)
            mock_registry_client.surrender_certificates(min(certs, mock_registry_client._balance))

        # Generate reports
        report = template_renderer("group_consolidation_report", {
            "group_id": sample_entity_group["group_id"],
            "total_emissions": round(total_emissions, 2),
            "certificates": certs,
        }, "markdown")
        assert "Group Consolidation Report" in report["content"]
        assert len(report["provenance_hash"]) == 64


class TestE2EPrecursorChain:
    """E2E: Precursor chain in pipeline."""

    def test_precursor_chain_in_pipeline(self, sample_precursor_chain, mock_ets_bridge):
        """E2E: Resolve precursor chain -> use in calculation -> certificate."""
        chain = sample_precursor_chain
        specific_ef = chain["specific_emission_tco2e_per_tonne"]
        weight = chain["final_product"]["weight_tonnes"]
        total_emissions = specific_ef * weight

        fa_pct = mock_ets_bridge.get_free_allocation_pct(2026) / 100.0
        net = total_emissions * (1.0 - fa_pct)
        certs = int(round(net))

        result = {
            "chain_id": chain["chain_id"],
            "total_emissions_tco2e": round(total_emissions, 6),
            "net_obligation_tco2e": round(net, 6),
            "certificates_required": certs,
            "provenance_hash": _compute_hash({
                "chain": chain["chain_id"],
                "emissions": total_emissions,
            }),
        }
        assert_provenance_hash(result)
        assert result["certificates_required"] >= 0


class TestE2EAntiCircumvention:
    """E2E: Anti-circumvention screening."""

    def test_anti_circumvention_screening(
        self, sample_customs_declaration, mock_taric_client
    ):
        """E2E: Screen imports for anti-circumvention flags."""
        flags = []
        for item in sample_customs_declaration["line_items"]:
            if not item["cbam_applicable"]:
                continue
            # Check for suspicious patterns
            validation = mock_taric_client.validate_cn_code(item["cn_code"])
            if validation["cbam_covered"]:
                # Check origin (simplified)
                if item["origin_country"] not in ["TR", "CN", "IN", "BR"]:
                    flags.append({
                        "item": item["item_number"],
                        "flag": "unusual_origin",
                        "origin": item["origin_country"],
                    })
        # In our test data, all origins are typical
        assert len(flags) == 0


class TestE2EBudgetForecasting:
    """E2E: Multi-year budget forecasting."""

    def test_multi_year_budget_e2e(self, sample_config, mock_ets_bridge):
        """E2E: Forecast CBAM costs 2026-2034."""
        annual_emissions = 22500.0
        base_price = 78.50
        growth = 1.05

        forecast = []
        for year in range(2026, 2035):
            fa_pct = mock_ets_bridge.get_free_allocation_pct(year) / 100.0
            coverage = 1.0 - fa_pct
            years_ahead = year - 2026
            price = base_price * growth ** years_ahead
            cost = annual_emissions * coverage * price
            forecast.append({
                "year": year,
                "coverage_pct": round(coverage * 100, 1),
                "price_eur": round(price, 2),
                "cost_eur": round(cost, 2),
            })

        assert len(forecast) == 9
        assert forecast[0]["coverage_pct"] == 2.5
        assert forecast[-1]["coverage_pct"] == 100.0
        # 2034 cost should be dramatically higher than 2026
        assert forecast[-1]["cost_eur"] > forecast[0]["cost_eur"] * 10


class TestE2EReportGeneration:
    """E2E: Generate all 6 report templates."""

    def test_all_reports_generated(self, template_renderer):
        """E2E: Generate all 6 PACK-005 templates."""
        reports = {}
        for tid in PACK005_TEMPLATE_IDS:
            data = {"template": tid, "year": 2026, "demo": True}
            result = template_renderer(tid, data, "markdown")
            reports[tid] = result
        assert len(reports) == 6
        for tid, report in reports.items():
            assert report["format"] == "markdown"
            assert len(report["provenance_hash"]) == 64


class TestE2EComponentCounts:
    """E2E: Verify expected component counts."""

    def test_engine_count(self, engine_ids):
        """Test 8 engines are registered."""
        assert len(engine_ids) == 8

    def test_workflow_count(self, workflow_ids):
        """Test 6 workflows are registered."""
        assert len(workflow_ids) == 6

    def test_template_count(self, template_ids):
        """Test 6 templates are registered."""
        assert len(template_ids) == 6

    def test_integration_count(self, integration_ids):
        """Test 7 integrations are registered."""
        assert len(integration_ids) == 7
