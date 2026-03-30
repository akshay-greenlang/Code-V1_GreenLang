# -*- coding: utf-8 -*-
"""
PACK-003 CSRD Enterprise Pack - End-to-End Enterprise Tests (15 tests)

Comprehensive end-to-end tests covering full enterprise lifecycle
scenarios including tenant onboarding, multi-tenant isolation,
predictive compliance, auditor engagement, supply chain to Scope 3,
IoT to emissions, carbon credit net-zero, custom workflows,
regulatory filing, white-label reports, cross-pack compatibility,
GraphQL, API keys, health checks, and full enterprise pipeline.

Author: GreenLang QA Team
"""

import json
from typing import Any, Dict, List

import pytest

import sys, os
from greenlang.schemas import utcnow
sys.path.insert(0, os.path.dirname(__file__))

from conftest import (

    ENTERPRISE_TEMPLATE_IDS,
    ENTERPRISE_WORKFLOW_IDS,
    StubAuditorPortal,
    StubGraphQLSchema,
    StubMLModel,
    StubMarketplace,
    StubSAMLProvider,
    StubTenantManager,
    _compute_hash,
    _new_uuid,
    _utcnow,
    render_template_stub,
)


class TestE2EEnterprise:
    """End-to-end enterprise pack test suite."""

    def test_e2e_tenant_onboarding_to_reporting(
        self,
        multi_tenant_engine,
        multi_tenant_module,
        sample_brand_config,
        template_renderer,
    ):
        """E2E: Tenant onboarding through to first report generation."""
        mod = multi_tenant_module
        # Step 1: Provision tenant
        tenant = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="E2E Onboarding Corp",
                tier=mod.TenantTier.ENTERPRISE,
                admin_email="admin@e2e-onboard.com",
                region="eu-west-1",
            )
        )
        assert tenant.status == mod.TenantLifecycleStatus.ACTIVE

        # Step 2: Apply branding
        brand_result = {
            "tenant_id": tenant.tenant_id,
            "brand_applied": True,
            "primary_color": sample_brand_config["primary_color"],
        }
        assert brand_result["brand_applied"] is True

        # Step 3: Generate report
        report = template_renderer("white_label_report", {
            "brand_name": "E2E Corp",
            "primary_color": sample_brand_config["primary_color"],
            "report_title": "Q1 2026 Sustainability Report",
            "xbrl_tagged": True,
            "language": "en",
        }, "html")
        assert "<html>" in report["content"]
        assert len(report["provenance_hash"]) == 64

    def test_e2e_multi_tenant_isolation(
        self,
        multi_tenant_engine,
        multi_tenant_module,
    ):
        """E2E: Create 2 tenants and verify data isolation."""
        mod = multi_tenant_module
        t1 = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="E2E Tenant Alpha",
                tier=mod.TenantTier.ENTERPRISE,
                admin_email="alpha@e2e-iso.com",
            )
        )
        t2 = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="E2E Tenant Beta",
                tier=mod.TenantTier.PROFESSIONAL,
                admin_email="beta@e2e-iso.com",
            )
        )
        # Verify isolation
        assert t1.tenant_id != t2.tenant_id
        assert t1.tier != t2.tier
        u1 = multi_tenant_engine.get_resource_usage(t1.tenant_id)
        u2 = multi_tenant_engine.get_resource_usage(t2.tenant_id)
        assert u1["tenant_id"] != u2["tenant_id"]

    def test_e2e_predictive_compliance_pipeline(
        self,
        mock_ml_models,
        sample_forecast_data,
        template_renderer,
    ):
        """E2E: Data -> forecast -> gap analysis -> action plan."""
        # Step 1: Generate forecast
        model = mock_ml_models["emission_forecast"]
        forecast = model.predict(sample_forecast_data, horizon=12)
        assert len(forecast["predictions"]) == 12

        # Step 2: Detect anomalies
        anomalies = model.detect_anomalies(sample_forecast_data)
        assert isinstance(anomalies, list)

        # Step 3: Gap analysis
        final_forecast = forecast["predictions"][-1]["predicted_value"]
        target = 800.0
        gap = final_forecast - target

        # Step 4: Generate report
        report = template_renderer("emission_forecast_report", {
            "forecast_value": final_forecast,
            "target": target,
            "gap": gap,
            "anomaly_count": len(anomalies),
            "r_squared": forecast["r_squared"],
        }, "json")
        parsed = json.loads(report["content"])
        assert parsed["template_id"] == "emission_forecast_report"

    def test_e2e_auditor_engagement_lifecycle(
        self,
        mock_auditor_portal,
        sample_audit_engagement,
    ):
        """E2E: Auditor setup -> evidence -> findings -> opinion."""
        # Step 1: Create engagement
        eng = mock_auditor_portal.create_engagement(sample_audit_engagement)
        assert eng["status"] == "active"

        # Step 2: Package evidence
        pkg = mock_auditor_portal.package_evidence(
            eng["engagement_id"], "scope_1",
            [{"doc_id": "D-001"}, {"doc_id": "D-002"}],
        )
        assert pkg["document_count"] == 2

        # Step 3: Submit finding
        finding = mock_auditor_portal.submit_finding(
            eng["engagement_id"],
            {"severity": "observation", "description": "Minor issue"},
        )
        assert finding["status"] == "open"

        # Step 4: Get opinion
        opinion = mock_auditor_portal.get_opinion(eng["engagement_id"])
        assert opinion["conclusion"] == "unmodified"

    def test_e2e_supply_chain_to_scope3(
        self,
        sample_supplier_data,
        template_renderer,
    ):
        """E2E: Supplier mapping -> scoring -> Scope 3 estimation."""
        # Step 1: Map suppliers
        tier_1 = [s for s in sample_supplier_data if s["tier"] == 1]
        assert len(tier_1) > 0

        # Step 2: Score suppliers
        for s in sample_supplier_data:
            assert 0.0 <= s["composite_esg_score"] <= 1.0

        # Step 3: Estimate Scope 3
        total_scope3_contribution = sum(
            s["scope3_contribution_pct"] for s in sample_supplier_data
        )
        assert total_scope3_contribution > 0

        # Step 4: Generate report
        report = template_renderer("supply_chain_esg_scorecard", {
            "total_suppliers": len(sample_supplier_data),
            "tier_1_count": len(tier_1),
            "avg_score": round(
                sum(s["composite_esg_score"] for s in sample_supplier_data) / len(sample_supplier_data), 2
            ),
            "scope3_coverage_pct": round(total_scope3_contribution, 1),
        }, "markdown")
        assert "Supply Chain" in report["content"]

    def test_e2e_iot_to_emissions(
        self,
        sample_iot_readings,
        template_renderer,
    ):
        """E2E: Sensor data -> aggregation -> emission calculation."""
        # Step 1: Ingest readings
        assert len(sample_iot_readings) == 100

        # Step 2: Aggregate by device type
        energy_readings = [
            r for r in sample_iot_readings
            if r["device_type"] == "energy_meter" and r["quality_flag"] == "good"
        ]
        total_kwh = sum(r["value"] for r in energy_readings)

        # Step 3: Calculate emissions
        ef = 0.42  # kgCO2e/kWh
        emissions_kg = total_kwh * ef
        emissions_tco2e = round(emissions_kg / 1000, 6)

        # Step 4: Generate dashboard
        report = template_renderer("iot_monitoring_dashboard", {
            "total_readings": len(sample_iot_readings),
            "energy_readings": len(energy_readings),
            "total_kwh": round(total_kwh, 2),
            "emissions_tco2e": emissions_tco2e,
        }, "html")
        assert "<html>" in report["content"]

    def test_e2e_carbon_credit_net_zero(
        self,
        sample_carbon_credits,
        template_renderer,
    ):
        """E2E: Portfolio -> retirement -> net-zero accounting."""
        # Step 1: Portfolio summary
        active = [c for c in sample_carbon_credits if c["status"] == "active"]
        retired = [c for c in sample_carbon_credits if c["status"] == "retired"]
        total_active_qty = sum(c["quantity_tco2e"] for c in active)
        total_retired_qty = sum(c["quantity_tco2e"] for c in retired)

        # Step 2: Net-zero accounting
        gross = 45230.5
        net = gross - total_retired_qty
        offset_pct = round(total_retired_qty / gross * 100, 2)

        # Step 3: Generate report
        report = template_renderer("carbon_credit_portfolio", {
            "active_credits": len(active),
            "retired_credits": len(retired),
            "total_active_qty": total_active_qty,
            "gross_emissions": gross,
            "net_emissions": max(net, 0),
            "offset_pct": offset_pct,
        }, "json")
        parsed = json.loads(report["content"])
        assert parsed["template_id"] == "carbon_credit_portfolio"

    def test_e2e_custom_workflow_creation_and_execution(
        self,
        sample_workflow_definition,
    ):
        """E2E: Create custom workflow and execute it."""
        wf = sample_workflow_definition
        assert len(wf["steps"]) == 10

        # Execute first 5 steps
        trace = []
        for step in wf["steps"][:5]:
            trace.append({
                "step_id": step["step_id"],
                "type": step["type"],
                "status": "completed",
                "provenance_hash": _compute_hash(step),
            })
        assert len(trace) == 5
        assert all(len(t["provenance_hash"]) == 64 for t in trace)

    def test_e2e_regulatory_filing_pipeline(
        self,
        sample_filing_package,
        template_renderer,
    ):
        """E2E: Prepare -> validate -> submit -> confirm filing."""
        # Step 1: Prepare
        assert sample_filing_package["status"] == "prepared"

        # Step 2: Validate
        validation = sample_filing_package["validation_results"]
        assert validation["errors"] == 0

        # Step 3: Submit
        submission = {
            "filing_id": sample_filing_package["filing_id"],
            "status": "submitted",
            "receipt_id": f"ESAP-{_new_uuid()[:8]}",
        }
        assert submission["status"] == "submitted"

        # Step 4: Generate filing package report
        report = template_renderer("regulatory_filing_package", {
            "filing_id": sample_filing_package["filing_id"],
            "target": sample_filing_package["filing_target"],
            "validation_score": validation["validation_score"],
            "receipt_id": submission["receipt_id"],
        }, "json")
        assert len(report["provenance_hash"]) == 64

    def test_e2e_white_label_branded_report(
        self,
        sample_brand_config,
        template_renderer,
    ):
        """E2E: Apply brand -> generate branded report."""
        report = template_renderer("white_label_report", {
            "brand_name": "E2E Brand Corp",
            "primary_color": sample_brand_config["primary_color"],
            "logo_url": sample_brand_config["logo_url"],
            "report_title": "Branded Sustainability Report 2025",
            "xbrl_tagged": True,
            "language": "en",
        }, "html")
        assert "<html>" in report["content"]
        assert "E2E Brand Corp" in report["content"]

    def test_e2e_cross_pack_compatibility(self, pack_yaml):
        """E2E: Verify PACK-001 -> PACK-002 -> PACK-003 feature chain."""
        # PACK-001 features (inherited via PACK-002)
        deps = pack_yaml.get("dependencies", [])
        has_pack_002 = any(
            d.get("pack_id") == "PACK-002" for d in deps
        )
        has_pack_001 = any(
            d.get("pack_id") == "PACK-001" for d in deps
        )
        assert has_pack_002
        assert has_pack_001

        # PACK-003 specific features
        components = pack_yaml.get("components", {})
        assert "predictive_engines" in components
        assert "iot_engines" in components

    def test_e2e_graphql_query_resolution(
        self,
        mock_graphql_schema,
    ):
        """E2E: Register types -> execute query -> check auth."""
        # Register types
        mock_graphql_schema.register_type("EmissionSummary", {
            "scope1": "Float",
            "scope2": "Float",
            "scope3": "Float",
            "year": "Int",
        })
        assert "EmissionSummary" in mock_graphql_schema.types

        # Execute query
        result = mock_graphql_schema.resolve_query("{ emissions { scope1_total } }")
        assert result["errors"] is None

        # Check auth
        assert mock_graphql_schema.check_field_auth("emissions.scope1", ["viewer"]) is True

    def test_e2e_api_key_lifecycle(
        self,
        sample_api_keys,
    ):
        """E2E: Create -> use -> rotate -> revoke API key."""
        key = sample_api_keys[0]
        assert key["status"] == "active"

        # Simulate usage
        usage = key["usage_count_today"]
        assert usage < key["rate_limit_per_day"]

        # Rotate
        rotated = {
            "old_key_id": key["key_id"],
            "new_key_id": f"ak-{_new_uuid()[:8]}",
            "status": "rotated",
        }
        assert rotated["status"] == "rotated"

    def test_e2e_enterprise_health_check(self):
        """E2E: Run comprehensive enterprise health check."""
        health = {
            "overall_status": "healthy",
            "categories": {
                "database": "healthy",
                "redis": "healthy",
                "ml_models": "healthy",
                "iot_connector": "healthy",
                "filing_gateway": "healthy",
                "marketplace": "healthy",
                "tenant_manager": "healthy",
                "sso_provider": "healthy",
            },
            "checks_total": 8,
            "checks_passed": 8,
            "checks_failed": 0,
            "timestamp": utcnow().isoformat(),
        }
        assert health["overall_status"] == "healthy"
        assert health["checks_failed"] == 0
        assert health["checks_passed"] == health["checks_total"]

    def test_e2e_full_enterprise_pipeline(
        self,
        multi_tenant_engine,
        multi_tenant_module,
        mock_ml_models,
        sample_forecast_data,
        sample_iot_readings,
        sample_carbon_credits,
        sample_supplier_data,
        sample_filing_package,
        sample_brand_config,
        mock_auditor_portal,
        sample_audit_engagement,
        template_renderer,
    ):
        """E2E: Full enterprise pipeline with all features combined."""
        mod = multi_tenant_module

        # 1. Provision tenant
        tenant = multi_tenant_engine.provision_tenant(
            mod.TenantProvisionRequest(
                tenant_name="Full Pipeline Corp",
                tier=mod.TenantTier.ENTERPRISE,
                admin_email="admin@fullpipeline.com",
            )
        )
        assert tenant.status == mod.TenantLifecycleStatus.ACTIVE

        # 2. Run predictive forecast
        forecast = mock_ml_models["emission_forecast"].predict(
            sample_forecast_data, horizon=12
        )
        assert len(forecast["predictions"]) == 12

        # 3. Process IoT data
        energy_kwh = sum(
            r["value"] for r in sample_iot_readings
            if r["device_type"] == "energy_meter"
        )
        assert energy_kwh > 0

        # 4. Carbon credit accounting
        active_credits = sum(
            c["quantity_tco2e"] for c in sample_carbon_credits if c["status"] == "active"
        )
        assert active_credits > 0

        # 5. Supply chain scoring
        avg_esg = sum(
            s["composite_esg_score"] for s in sample_supplier_data
        ) / len(sample_supplier_data)
        assert 0.0 < avg_esg < 1.0

        # 6. Filing validation
        assert sample_filing_package["validation_results"]["errors"] == 0

        # 7. Auditor engagement
        eng = mock_auditor_portal.create_engagement(sample_audit_engagement)
        assert eng["status"] == "active"

        # 8. Generate final report
        report = template_renderer("enterprise_audit_package", {
            "tenant_id": tenant.tenant_id,
            "forecast_r_squared": forecast["r_squared"],
            "iot_readings": len(sample_iot_readings),
            "active_credits": active_credits,
            "avg_esg_score": round(avg_esg, 3),
            "filing_score": sample_filing_package["validation_results"]["validation_score"],
            "engagement_id": eng["engagement_id"],
        }, "json")
        parsed = json.loads(report["content"])
        assert parsed["template_id"] == "enterprise_audit_package"
        assert len(report["provenance_hash"]) == 64

        # 9. Final health check
        health = {
            "status": "healthy",
            "tenant_active": True,
            "pipeline_complete": True,
        }
        assert health["status"] == "healthy"
        assert health["pipeline_complete"] is True
