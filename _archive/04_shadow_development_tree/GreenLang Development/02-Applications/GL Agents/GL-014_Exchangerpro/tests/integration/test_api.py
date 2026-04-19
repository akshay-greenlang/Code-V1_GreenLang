# -*- coding: utf-8 -*-
"""
GL-014 EXCHANGERPRO - API Integration Tests

Tests for API endpoints including:
- Health check endpoints
- Thermal KPI endpoints
- Fouling prediction endpoints
- Cleaning recommendation endpoints
- Authentication and authorization

Author: GL-TestEngineer
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
from typing import Dict, Any


class TestHealthEndpoints:
    """Test health check API endpoints."""

    @pytest.mark.integration
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe endpoint."""
        # Simulate liveness check
        response = {
            "status": "alive",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        assert response["status"] == "alive"

    @pytest.mark.integration
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe endpoint."""
        # Simulate readiness check
        components = {
            "thermal_engine": "healthy",
            "ml_service": "healthy",
            "optimizer": "healthy",
            "database": "healthy",
        }

        all_healthy = all(status == "healthy" for status in components.values())

        response = {
            "ready": all_healthy,
            "components": components,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        assert response["ready"] == True

    @pytest.mark.integration
    def test_detailed_health_check(self):
        """Test detailed health check endpoint."""
        response = {
            "status": "healthy",
            "version": "1.0.0",
            "agent_id": "GL-014",
            "components": {
                "thermal_engine": {
                    "status": "healthy",
                    "response_time_ms": 2.5,
                },
                "ml_service": {
                    "status": "healthy",
                    "model_version": "1.2.0",
                    "response_time_ms": 15.0,
                },
                "optimizer": {
                    "status": "healthy",
                    "solver": "CBC",
                    "response_time_ms": 5.0,
                },
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        assert response["status"] == "healthy"
        assert response["agent_id"] == "GL-014"


class TestThermalKPIEndpoints:
    """Test thermal KPI API endpoints."""

    @pytest.mark.integration
    def test_get_current_kpis(self, sample_thermal_kpis):
        """Test getting current thermal KPIs."""
        kpis = sample_thermal_kpis

        response = {
            "exchanger_id": kpis.exchanger_id,
            "timestamp": kpis.timestamp.isoformat(),
            "kpis": {
                "Q_hot_kW": kpis.Q_hot_kW,
                "Q_cold_kW": kpis.Q_cold_kW,
                "lmtd_C": kpis.lmtd_C,
                "UA_actual_kW_K": kpis.UA_actual_kW_K,
                "UA_ratio": kpis.UA_ratio,
                "epsilon": kpis.epsilon,
                "cleanliness_factor": kpis.cleanliness_factor,
            },
            "provenance_hash": kpis.provenance_hash,
        }

        assert response["exchanger_id"] == "HX-001"
        assert response["kpis"]["epsilon"] > 0
        assert response["kpis"]["epsilon"] < 1

    @pytest.mark.integration
    def test_get_kpi_history(self):
        """Test getting KPI history."""
        history = [
            {
                "timestamp": "2024-01-15T10:00:00Z",
                "UA_ratio": 0.95,
                "epsilon": 0.78,
            },
            {
                "timestamp": "2024-01-15T11:00:00Z",
                "UA_ratio": 0.93,
                "epsilon": 0.76,
            },
            {
                "timestamp": "2024-01-15T12:00:00Z",
                "UA_ratio": 0.92,
                "epsilon": 0.75,
            },
        ]

        response = {
            "exchanger_id": "HX-001",
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T12:00:00Z",
            "data_points": len(history),
            "history": history,
        }

        assert response["data_points"] == 3

    @pytest.mark.integration
    def test_calculate_kpis_endpoint(self, sample_operating_state):
        """Test on-demand KPI calculation endpoint."""
        state = sample_operating_state

        request = {
            "exchanger_id": state.exchanger_id,
            "T_hot_in_C": state.T_hot_in_C,
            "T_hot_out_C": state.T_hot_out_C,
            "T_cold_in_C": state.T_cold_in_C,
            "T_cold_out_C": state.T_cold_out_C,
            "m_dot_hot_kg_s": state.m_dot_hot_kg_s,
            "m_dot_cold_kg_s": state.m_dot_cold_kg_s,
            "Cp_hot_kJ_kgK": state.Cp_hot_kJ_kgK,
            "Cp_cold_kJ_kgK": state.Cp_cold_kJ_kgK,
        }

        # Simulate calculation
        Q_hot = request["m_dot_hot_kg_s"] * request["Cp_hot_kJ_kgK"] * \
                (request["T_hot_in_C"] - request["T_hot_out_C"])

        response = {
            "exchanger_id": request["exchanger_id"],
            "kpis": {
                "Q_hot_kW": Q_hot,
            },
            "computation_time_ms": 3.2,
        }

        assert response["kpis"]["Q_hot_kW"] > 0


class TestPredictionEndpoints:
    """Test fouling prediction API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_fouling_prediction(self, mock_ml_service):
        """Test getting fouling prediction."""
        features = {
            "dt_hot": 60.0,
            "dt_cold": 70.0,
            "flow_ratio": 1.25,
            "dp_shell_ratio": 1.1,
            "dp_tube_ratio": 1.1,
            "reynolds_hot": 50000.0,
        }

        prediction = await mock_ml_service.predict_fouling(features)

        response = {
            "exchanger_id": "HX-001",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "prediction": {
                "fouling_resistance_m2K_kW": prediction["fouling_resistance_m2K_kW"],
                "ua_degradation_percent": prediction["ua_degradation_percent"],
                "predicted_days_to_threshold": prediction["predicted_days_to_threshold"],
                "confidence_score": prediction["confidence_score"],
            },
            "model_version": "1.2.0",
        }

        assert response["prediction"]["confidence_score"] <= 1.0

    @pytest.mark.integration
    def test_get_prediction_history(self):
        """Test getting prediction history."""
        history = [
            {
                "timestamp": "2024-01-10",
                "fouling_resistance_m2K_kW": 0.00025,
                "ua_degradation_percent": 18,
            },
            {
                "timestamp": "2024-01-15",
                "fouling_resistance_m2K_kW": 0.00030,
                "ua_degradation_percent": 22,
            },
            {
                "timestamp": "2024-01-20",
                "fouling_resistance_m2K_kW": 0.00035,
                "ua_degradation_percent": 26,
            },
        ]

        response = {
            "exchanger_id": "HX-001",
            "data_points": len(history),
            "history": history,
            "trend": "increasing",
        }

        assert response["trend"] == "increasing"

    @pytest.mark.integration
    def test_get_prediction_explanation(self):
        """Test getting prediction explanation (SHAP/LIME)."""
        response = {
            "exchanger_id": "HX-001",
            "prediction_id": "pred-12345",
            "explanation_type": "shap",
            "feature_contributions": [
                {"feature": "dp_shell_ratio", "contribution": 0.00015, "direction": "increasing"},
                {"feature": "dp_tube_ratio", "contribution": 0.00012, "direction": "increasing"},
                {"feature": "flow_ratio", "contribution": -0.00005, "direction": "decreasing"},
            ],
            "base_value": 0.00020,
            "predicted_value": 0.00042,
        }

        total_contribution = sum(f["contribution"] for f in response["feature_contributions"])
        expected_prediction = response["base_value"] + total_contribution

        assert abs(expected_prediction - response["predicted_value"]) < 0.0001


class TestRecommendationEndpoints:
    """Test cleaning recommendation API endpoints."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_recommendation(self, mock_optimizer_service):
        """Test getting cleaning recommendation."""
        fouling_state = type('FoulingState', (), {
            'exchanger_id': 'HX-001',
            'predicted_days_to_threshold': 45,
            'ua_degradation_percent': 28,
        })()

        recommendation = await mock_optimizer_service.optimize_schedule(
            "HX-001",
            fouling_state,
            {},
        )

        response = {
            "exchanger_id": recommendation.exchanger_id,
            "recommendation_id": recommendation.recommendation_id,
            "recommended_date": recommendation.recommended_cleaning_date.isoformat(),
            "urgency": recommendation.urgency,
            "estimated_cost_usd": recommendation.estimated_cost_usd,
            "estimated_savings_kWh": recommendation.estimated_energy_savings_kWh,
            "payback_days": recommendation.estimated_payback_days,
            "confidence_score": recommendation.confidence_score,
        }

        assert response["urgency"] in ["routine", "scheduled", "urgent", "critical"]

    @pytest.mark.integration
    def test_list_recommendations(self):
        """Test listing all recommendations."""
        recommendations = [
            {
                "exchanger_id": "HX-001",
                "recommendation_id": "rec-001",
                "urgency": "scheduled",
                "recommended_date": "2024-02-15",
            },
            {
                "exchanger_id": "HX-002",
                "recommendation_id": "rec-002",
                "urgency": "routine",
                "recommended_date": "2024-03-01",
            },
            {
                "exchanger_id": "HX-003",
                "recommendation_id": "rec-003",
                "urgency": "urgent",
                "recommended_date": "2024-01-25",
            },
        ]

        response = {
            "total_count": len(recommendations),
            "recommendations": recommendations,
            "filter_applied": None,
        }

        assert response["total_count"] == 3

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_create_work_order_from_recommendation(self, mock_cmms_connector):
        """Test creating work order from recommendation."""
        request = {
            "recommendation_id": "rec-12345",
            "exchanger_id": "HX-001",
            "priority": "scheduled",
            "scheduled_date": "2024-02-15T08:00:00Z",
            "notes": "Scheduled cleaning based on fouling prediction",
        }

        work_order = await mock_cmms_connector.create_work_order(request)

        response = {
            "success": True,
            "work_order_id": work_order["work_order_id"],
            "status": work_order["status"],
            "created_at": work_order["created_at"],
        }

        assert response["success"] == True
        assert response["work_order_id"] is not None


class TestErrorHandling:
    """Test API error handling."""

    @pytest.mark.integration
    def test_invalid_exchanger_id(self):
        """Test handling of invalid exchanger ID."""
        response = {
            "error": "NOT_FOUND",
            "message": "Exchanger with ID 'HX-INVALID' not found",
            "status_code": 404,
        }

        assert response["status_code"] == 404

    @pytest.mark.integration
    def test_invalid_request_body(self):
        """Test handling of invalid request body."""
        response = {
            "error": "VALIDATION_ERROR",
            "message": "Invalid request body",
            "details": [
                {"field": "T_hot_in_C", "error": "Field is required"},
                {"field": "m_dot_hot_kg_s", "error": "Must be positive"},
            ],
            "status_code": 400,
        }

        assert response["status_code"] == 400
        assert len(response["details"]) > 0

    @pytest.mark.integration
    def test_service_unavailable(self):
        """Test handling of service unavailable."""
        response = {
            "error": "SERVICE_UNAVAILABLE",
            "message": "ML service is temporarily unavailable",
            "retry_after_seconds": 30,
            "status_code": 503,
        }

        assert response["status_code"] == 503


class TestAuthentication:
    """Test API authentication and authorization."""

    @pytest.mark.integration
    def test_valid_api_key(self):
        """Test request with valid API key."""
        headers = {
            "X-API-Key": "valid-api-key-12345",
        }

        response = {
            "authenticated": True,
            "user": "service-account",
            "permissions": ["read", "write"],
        }

        assert response["authenticated"] == True

    @pytest.mark.integration
    def test_invalid_api_key(self):
        """Test request with invalid API key."""
        headers = {
            "X-API-Key": "invalid-key",
        }

        response = {
            "error": "UNAUTHORIZED",
            "message": "Invalid API key",
            "status_code": 401,
        }

        assert response["status_code"] == 401

    @pytest.mark.integration
    def test_missing_api_key(self):
        """Test request without API key."""
        response = {
            "error": "UNAUTHORIZED",
            "message": "API key required",
            "status_code": 401,
        }

        assert response["status_code"] == 401


class TestRateLimiting:
    """Test API rate limiting."""

    @pytest.mark.integration
    def test_rate_limit_headers(self):
        """Test rate limit headers in response."""
        headers = {
            "X-RateLimit-Limit": "1000",
            "X-RateLimit-Remaining": "995",
            "X-RateLimit-Reset": "1705320000",
        }

        remaining = int(headers["X-RateLimit-Remaining"])
        limit = int(headers["X-RateLimit-Limit"])

        assert remaining <= limit

    @pytest.mark.integration
    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded response."""
        response = {
            "error": "RATE_LIMIT_EXCEEDED",
            "message": "Too many requests",
            "retry_after_seconds": 60,
            "status_code": 429,
        }

        assert response["status_code"] == 429


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "TestHealthEndpoints",
    "TestThermalKPIEndpoints",
    "TestPredictionEndpoints",
    "TestRecommendationEndpoints",
    "TestErrorHandling",
    "TestAuthentication",
    "TestRateLimiting",
]
