"""
Unit Tests for Champion-Challenger Model Deployment.

Tests cover:
- Champion and challenger registration
- Deterministic request routing
- Outcome recording and aggregation
- Statistical significance testing
- Model promotion and rollback
- Thread-safe concurrent operations
"""

import hashlib
from threading import Thread

import pytest

from greenlang.ml.champion_challenger import (
    ChampionChallengerManager,
    ModelVersion,
    PromotionEvaluation,
    RequestOutcome,
    TrafficMode,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def manager():
    """Create manager instance for testing."""
    return ChampionChallengerManager(storage_path="/tmp/cc_test")


@pytest.fixture
def model_name():
    """Test model name."""
    return "heat_predictor"


@pytest.fixture
def champion_version():
    """Test champion version."""
    return "1.0.0"


@pytest.fixture
def challenger_version():
    """Test challenger version."""
    return "1.1.0"


# =============================================================================
# ModelVersion Tests
# =============================================================================

class TestModelVersion:
    """Test ModelVersion validation."""

    def test_valid_semantic_version(self):
        """Test valid semantic version."""
        model = ModelVersion(model_name="test_model", version="1.2.3")
        assert model.version == "1.2.3"

    def test_invalid_version_format(self):
        """Test invalid version format."""
        with pytest.raises(ValueError):
            ModelVersion(model_name="test_model", version="1.2")

    def test_invalid_version_non_numeric(self):
        """Test non-numeric version parts."""
        with pytest.raises(ValueError):
            ModelVersion(model_name="test_model", version="1.x.3")


# =============================================================================
# Champion Registration Tests
# =============================================================================

class TestChampionRegistration:
    """Test champion model registration."""

    def test_register_champion(self, manager, model_name, champion_version):
        """Test registering champion model."""
        manager.register_champion(model_name, champion_version)
        assert manager.champions[model_name] == champion_version

    def test_register_champion_invalid_version(self, manager, model_name):
        """Test registering champion with invalid version."""
        with pytest.raises(ValueError):
            manager.register_champion(model_name, "1.0")

    def test_register_multiple_champions(self, manager):
        """Test registering multiple champion models."""
        manager.register_champion("model_a", "1.0.0")
        manager.register_champion("model_b", "2.0.0")

        assert manager.champions["model_a"] == "1.0.0"
        assert manager.champions["model_b"] == "2.0.0"

    def test_update_champion(self, manager, model_name):
        """Test updating champion version."""
        manager.register_champion(model_name, "1.0.0")
        manager.register_champion(model_name, "1.1.0")

        assert manager.champions[model_name] == "1.1.0"


# =============================================================================
# Challenger Registration Tests
# =============================================================================

class TestChallengerRegistration:
    """Test challenger model registration."""

    def test_register_challenger(self, manager, model_name, champion_version, challenger_version):
        """Test registering challenger model."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version, traffic_percentage=10)

        assert model_name in manager.challengers
        assert manager.challengers[model_name]["version"] == challenger_version
        assert manager.challengers[model_name]["traffic_percentage"] == 10

    def test_register_challenger_without_champion(self, manager, model_name, challenger_version):
        """Test challenger registration fails without champion."""
        with pytest.raises(ValueError, match="Champion not registered"):
            manager.register_challenger(model_name, challenger_version)

    def test_register_challenger_invalid_traffic(self, manager, model_name, champion_version, challenger_version):
        """Test challenger registration with invalid traffic percentage."""
        manager.register_champion(model_name, champion_version)

        with pytest.raises(ValueError):
            manager.register_challenger(model_name, challenger_version, traffic_percentage=0)

        with pytest.raises(ValueError):
            manager.register_challenger(model_name, challenger_version, traffic_percentage=100)

    def test_challenger_traffic_modes(self, manager, model_name, champion_version):
        """Test different traffic modes."""
        manager.register_champion(model_name, champion_version)

        modes = [
            (TrafficMode.SHADOW, 1),
            (TrafficMode.CANARY_5, 5),
            (TrafficMode.CANARY_10, 10),
            (TrafficMode.AB_TEST, 50),
        ]

        for idx, (mode, traffic) in enumerate(modes):
            if model_name in manager.challengers:
                del manager.challengers[model_name]

            version = f"{idx + 1}.0.0"
            manager.register_challenger(
                model_name, version, traffic_percentage=traffic, mode=mode
            )
            assert manager.challengers[model_name]["mode"] == mode.value


# =============================================================================
# Request Routing Tests
# =============================================================================

class TestRequestRouting:
    """Test deterministic request routing."""

    def test_route_to_champion(self, manager, model_name, champion_version):
        """Test routing to champion when no challenger."""
        manager.register_champion(model_name, champion_version)
        version = manager.route_request("req_001", model_name)
        assert version == champion_version

    def test_deterministic_routing(self, manager, model_name, champion_version, challenger_version):
        """Test deterministic routing with same request_id."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version, traffic_percentage=50)

        version1 = manager.route_request("req_same", model_name)
        version2 = manager.route_request("req_same", model_name)
        assert version1 == version2

    def test_traffic_distribution(self, manager, model_name, champion_version, challenger_version):
        """Test traffic distribution across requests."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version, traffic_percentage=50)

        champion_count = 0
        challenger_count = 0

        for i in range(100):
            request_id = f"req_{i:03d}"
            version = manager.route_request(request_id, model_name)

            if version == champion_version:
                champion_count += 1
            else:
                challenger_count += 1

        assert 35 < champion_count < 65
        assert 35 < challenger_count < 65

    def test_route_unknown_model(self, manager):
        """Test routing to unknown model raises error."""
        with pytest.raises(ValueError):
            manager.route_request("req_001", "unknown_model")


# =============================================================================
# Outcome Recording Tests
# =============================================================================

class TestOutcomeRecording:
    """Test outcome recording and aggregation."""

    def test_record_outcome(self, manager, model_name, champion_version):
        """Test recording single outcome."""
        manager.register_champion(model_name, champion_version)

        manager.record_outcome(
            "req_001",
            champion_version,
            {"mae": 0.05, "rmse": 0.08},
            execution_time_ms=12.5,
        )

        outcomes = manager.outcomes[champion_version]
        assert len(outcomes) == 1
        assert outcomes[0].request_id == "req_001"
        assert outcomes[0].metrics["mae"] == 0.05

    def test_record_multiple_outcomes(self, manager, model_name, champion_version, challenger_version):
        """Test recording outcomes for multiple models."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)

        for i in range(10):
            manager.record_outcome(
                f"req_{i:03d}",
                champion_version if i % 2 == 0 else challenger_version,
                {"mae": 0.01 + i * 0.001, "rmse": 0.02 + i * 0.001},
            )

        assert len(manager.outcomes[champion_version]) == 5
        assert len(manager.outcomes[challenger_version]) == 5

    def test_record_outcome_with_features(self, manager, model_name, champion_version):
        """Test recording outcome with input features."""
        manager.record_outcome(
            "req_001",
            champion_version,
            {"mae": 0.05},
            features={"temp": 25.0, "humidity": 60},
        )

        outcome = manager.outcomes[champion_version][0]
        assert len(outcome.features_hash) == 64


# =============================================================================
# Evaluation Tests
# =============================================================================

class TestChallengerEvaluation:
    """Test challenger evaluation and statistical testing."""

    def test_evaluate_without_samples(self, manager, model_name, champion_version, challenger_version):
        """Test evaluation without sufficient samples."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)

        evaluation = manager.evaluate_challenger(model_name)

        assert not evaluation.should_promote
        assert evaluation.samples_collected == 0

    def test_evaluate_with_sufficient_samples(self, manager, model_name, champion_version, challenger_version):
        """Test evaluation with sufficient samples."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)

        for i in range(50):
            manager.record_outcome(f"champ_{i}", champion_version, {"mae": 0.05})
            manager.record_outcome(f"chal_{i}", challenger_version, {"mae": 0.03})

        evaluation = manager.evaluate_challenger(model_name)

        assert evaluation.samples_collected >= 50
        assert evaluation.champion_mean_metric > 0
        assert evaluation.challenger_mean_metric > 0

    def test_evaluate_unknown_model(self, manager):
        """Test evaluation of unknown model raises error."""
        with pytest.raises(ValueError):
            manager.evaluate_challenger("unknown_model")

    def test_evaluate_without_challenger(self, manager, model_name, champion_version):
        """Test evaluation without registered challenger."""
        manager.register_champion(model_name, champion_version)

        with pytest.raises(ValueError):
            manager.evaluate_challenger(model_name)


# =============================================================================
# Promotion Tests
# =============================================================================

class TestPromotion:
    """Test model promotion and rollback."""

    def test_promote_challenger(self, manager, model_name, champion_version, challenger_version):
        """Test promoting challenger to champion."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)

        success = manager.promote_challenger(model_name)

        assert success
        assert manager.champions[model_name] == challenger_version
        assert model_name not in manager.challengers

    def test_promote_without_challenger(self, manager, model_name, champion_version):
        """Test promotion fails without challenger."""
        manager.register_champion(model_name, champion_version)

        success = manager.promote_challenger(model_name)

        assert not success
        assert manager.champions[model_name] == champion_version

    def test_promotion_history(self, manager, model_name, champion_version, challenger_version):
        """Test promotion is recorded in history."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)
        manager.promote_challenger(model_name)

        promotion_events = [e for e in manager.promotion_history if e["event"] == "promotion"]
        assert len(promotion_events) == 1
        assert promotion_events[0]["new_champion"] == challenger_version

    def test_rollback(self, manager, model_name, champion_version, challenger_version):
        """Test rollback to previous version."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)
        manager.promote_challenger(model_name)

        success = manager.rollback(model_name, champion_version)

        assert success
        assert manager.champions[model_name] == champion_version

    def test_rollback_history(self, manager, model_name, champion_version, challenger_version):
        """Test rollback is recorded in history."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version)
        manager.promote_challenger(model_name)
        manager.rollback(model_name, champion_version)

        rollback_events = [e for e in manager.promotion_history if e["event"] == "rollback"]
        assert len(rollback_events) == 1


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Test concurrent access to manager."""

    def test_concurrent_routing(self, manager, model_name, champion_version, challenger_version):
        """Test routing under concurrent load."""
        manager.register_champion(model_name, champion_version)
        manager.register_challenger(model_name, challenger_version, traffic_percentage=50)

        routes = []

        def route_requests(request_prefix: str):
            for i in range(50):
                request_id = f"{request_prefix}_{i}"
                version = manager.route_request(request_id, model_name)
                routes.append(version)

        threads = [Thread(target=route_requests, args=(f"t_{i}",)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        assert len(routes) == 250
        champion_count = sum(1 for v in routes if v == champion_version)
        challenger_count = sum(1 for v in routes if v == challenger_version)

        assert champion_count > 0
        assert challenger_count > 0

    def test_concurrent_outcome_recording(self, manager, model_name, champion_version):
        """Test outcome recording under concurrent load."""
        manager.register_champion(model_name, champion_version)

        def record_outcomes(start_idx: int):
            for i in range(100):
                manager.record_outcome(
                    f"req_{start_idx}_{i}",
                    champion_version,
                    {"mae": 0.05 + i * 0.001},
                )

        threads = [Thread(target=record_outcomes, args=(i,)) for i in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        outcomes = manager.outcomes[champion_version]
        assert len(outcomes) == 500


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_complete_workflow(self, manager, model_name):
        """Test complete champion-challenger workflow."""
        manager.register_champion(model_name, "1.0.0")
        version1 = manager.route_request("req_001", model_name)
        assert version1 == "1.0.0"

        manager.register_challenger(model_name, "1.1.0", traffic_percentage=20)

        champion_routes = sum(
            1 for i in range(100)
            if manager.route_request(f"req_v2_{i}", model_name) == "1.0.0"
        )
        challenger_routes = 100 - champion_routes

        assert champion_routes > 0
        assert challenger_routes > 0

        for i in range(50):
            manager.record_outcome(f"champ_{i}", "1.0.0", {"mae": 0.05})
            manager.record_outcome(f"chal_{i}", "1.1.0", {"mae": 0.03})

        evaluation = manager.evaluate_challenger(model_name)
        assert evaluation.champion_version == "1.0.0"
        assert evaluation.challenger_version == "1.1.0"

    def test_multiple_models(self, manager):
        """Test managing multiple models."""
        models = ["model_a", "model_b", "model_c"]

        for model in models:
            manager.register_champion(model, "1.0.0")
            manager.register_challenger(model, "1.1.0", traffic_percentage=10)

        assert len(manager.champions) == 3
        assert len(manager.challengers) == 3

        for model in models:
            success = manager.promote_challenger(model)
            assert success

        for model in models:
            assert manager.champions[model] == "1.1.0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
