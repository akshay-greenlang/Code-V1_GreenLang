# -*- coding: utf-8 -*-
"""
Unit tests for API endpoints - AGENT-EUDR-036

Tests the FastAPI router with 12 endpoints for DDS management,
operator registration, geolocation formatting, package assembly,
status tracking, audit trails, and health checks.

Author: GreenLang Platform Team
Date: March 2026
"""
from __future__ import annotations

import pytest

from greenlang.agents.eudr.eu_information_system_interface.api import (
    router,
    get_router,
    CreateDDSRequest,
    RegisterOperatorRequest,
    FormatGeolocationRequest,
    AssemblePackageRequest,
    ErrorResponse,
)

# Full prefix used by the router
_PREFIX = "/api/v1/eudr/eu-information-system-interface"


class TestRouter:
    """Test router configuration."""

    def test_router_prefix(self):
        assert router.prefix == "/api/v1/eudr/eu-information-system-interface"

    def test_router_tags(self):
        assert "EUDR EU Information System Interface" in router.tags

    def test_get_router_returns_same(self):
        r = get_router()
        assert r is router


class TestEndpointCount:
    """Test that all 12 endpoints are registered."""

    def test_endpoint_count(self):
        # Count the routes on the router
        route_count = len([r for r in router.routes if hasattr(r, "methods")])
        assert route_count == 12


class TestRequestSchemas:
    """Test request body schemas."""

    def test_create_dds_request(self):
        req = CreateDDSRequest(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="placing",
            commodity_lines=[
                {"commodity": "cocoa", "description": "Beans", "quantity": "1000",
                 "country_of_production": "GH",
                 "geolocation": {"format": "point", "point": {"latitude": "6.0", "longitude": "-1.0"},
                                 "country_code": "GH"}},
            ],
        )
        assert req.operator_id == "op-001"
        assert len(req.commodity_lines) == 1

    def test_create_dds_request_optional_fields(self):
        req = CreateDDSRequest(
            operator_id="op-001",
            eori_number="DE123456789012",
            dds_type="export",
            commodity_lines=[],
            risk_assessment_id="ra-001",
            mitigation_plan_id="mp-001",
            improvement_plan_id="ip-035-001",
        )
        assert req.risk_assessment_id == "ra-001"
        assert req.improvement_plan_id == "ip-035-001"

    def test_register_operator_request(self):
        req = RegisterOperatorRequest(
            operator_id="op-001",
            eori_number="DE123456789012",
            operator_type="operator",
            company_name="Green Trading GmbH",
            member_state="DE",
        )
        assert req.company_name == "Green Trading GmbH"

    def test_format_geolocation_request(self):
        req = FormatGeolocationRequest(
            coordinates=[{"lat": 6.688, "lng": -1.624}],
            country_code="GH",
        )
        assert req.country_code == "GH"
        assert len(req.coordinates) == 1

    def test_assemble_package_request(self):
        req = AssemblePackageRequest(
            dds_id="dds-001",
            documents=[{"type": "dds_form", "content": {}, "size": 1024}],
        )
        assert req.dds_id == "dds-001"

    def test_error_response(self):
        err = ErrorResponse(
            detail="Something went wrong",
            error_code="validation_error",
        )
        assert err.detail == "Something went wrong"
        assert err.error_code == "validation_error"


class TestEndpointPaths:
    """Test that expected endpoint paths are registered."""

    def _route_paths(self):
        return [r.path for r in router.routes if hasattr(r, "path")]

    def test_create_dds_path(self):
        assert f"{_PREFIX}/create-dds" in self._route_paths()

    def test_validate_dds_path(self):
        assert f"{_PREFIX}/validate-dds/{{dds_id}}" in self._route_paths()

    def test_submit_dds_path(self):
        assert f"{_PREFIX}/submit-dds/{{dds_id}}" in self._route_paths()

    def test_get_dds_path(self):
        assert f"{_PREFIX}/dds/{{dds_id}}" in self._route_paths()

    def test_list_dds_path(self):
        assert f"{_PREFIX}/dds" in self._route_paths()

    def test_register_operator_path(self):
        assert f"{_PREFIX}/register-operator" in self._route_paths()

    def test_format_geolocation_path(self):
        assert f"{_PREFIX}/format-geolocation" in self._route_paths()

    def test_assemble_package_path(self):
        assert f"{_PREFIX}/assemble-package" in self._route_paths()

    def test_status_path(self):
        assert f"{_PREFIX}/status/{{dds_id}}" in self._route_paths()

    def test_audit_trail_path(self):
        assert f"{_PREFIX}/audit/{{entity_type}}/{{entity_id}}" in self._route_paths()

    def test_audit_report_path(self):
        assert f"{_PREFIX}/audit-report/{{entity_type}}/{{entity_id}}" in self._route_paths()

    def test_health_path(self):
        assert f"{_PREFIX}/health" in self._route_paths()


class TestEndpointMethods:
    """Test HTTP methods for endpoints."""

    def _routes_by_path(self):
        routes = {}
        for r in router.routes:
            if hasattr(r, "path") and hasattr(r, "methods"):
                routes[r.path] = r.methods
        return routes

    def test_create_dds_is_post(self):
        routes = self._routes_by_path()
        assert "POST" in routes.get(f"{_PREFIX}/create-dds", set())

    def test_validate_dds_is_post(self):
        routes = self._routes_by_path()
        assert "POST" in routes.get(f"{_PREFIX}/validate-dds/{{dds_id}}", set())

    def test_submit_dds_is_post(self):
        routes = self._routes_by_path()
        assert "POST" in routes.get(f"{_PREFIX}/submit-dds/{{dds_id}}", set())

    def test_get_dds_is_get(self):
        routes = self._routes_by_path()
        assert "GET" in routes.get(f"{_PREFIX}/dds/{{dds_id}}", set())

    def test_list_dds_is_get(self):
        routes = self._routes_by_path()
        assert "GET" in routes.get(f"{_PREFIX}/dds", set())

    def test_register_operator_is_post(self):
        routes = self._routes_by_path()
        assert "POST" in routes.get(f"{_PREFIX}/register-operator", set())

    def test_health_is_get(self):
        routes = self._routes_by_path()
        assert "GET" in routes.get(f"{_PREFIX}/health", set())
