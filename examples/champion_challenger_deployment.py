"""
Champion-Challenger Model Deployment Example.

This example demonstrates the complete workflow for safe model promotion
using the champion-challenger pattern with the Process Heat agents.

Workflow:
    1. Register production champion model
    2. Register new challenger model with traffic split
    3. Route requests to both models
    4. Record prediction outcomes
    5. Evaluate challenger statistically
    6. Promote if performance is better
    7. Monitor for degradation and rollback if needed
"""

import logging
import time
from typing import Dict, List

import numpy as np

from greenlang.ml.champion_challenger import ChampionChallengerManager, TrafficMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Step 1: Initialize Manager
# =============================================================================

def initialize_manager() -> ChampionChallengerManager:
    """Initialize champion-challenger manager."""
    manager = ChampionChallengerManager(storage_path="./cc_deployment")
    logger.info("Initialized ChampionChallengerManager")
    return manager


# =============================================================================
# Step 2: Register Champion Model
# =============================================================================

def register_champion(manager: ChampionChallengerManager) -> None:
    """Register production champion model."""
    model_name = "process_heat_predictor"
    champion_version = "1.0.0"

    manager.register_champion(model_name, champion_version)
    logger.info(f"Registered champion: {model_name}@{champion_version}")


# =============================================================================
# Step 3: Deploy Challenger in Shadow Mode
# =============================================================================

def deploy_challenger_shadow_mode(manager: ChampionChallengerManager) -> None:
    """
    Deploy challenger in shadow mode (100% champion, log challenger).

    Shadow mode is the safest deployment option:
    - All traffic goes to champion (production impact: none)
    - Challenger responses logged but not used
    - Allows monitoring before traffic shift
    - Perfect for initial validation
    """
    model_name = "process_heat_predictor"
    challenger_version = "1.1.0"

    manager.register_challenger(
        model_name,
        challenger_version,
        traffic_percentage=1,
        mode=TrafficMode.SHADOW
    )
    logger.info(f"Deployed {challenger_version} in shadow mode")


# =============================================================================
# Step 4: Deploy Challenger in Canary Mode
# =============================================================================

def deploy_challenger_canary_mode(manager: ChampionChallengerManager) -> None:
    """
    Deploy challenger in canary mode (gradual traffic shift).

    Canary deployments gradually shift traffic:
    - Phase 1: 5% traffic to challenger (95% champion)
    - Phase 2: 10% traffic to challenger (90% champion)
    - Phase 3: 20% traffic to challenger (80% champion)
    - Full promotion when ready
    """
    model_name = "process_heat_predictor"
    challenger_version = "1.1.0"

    logger.info("Phase 1: Deploying in canary mode (5% traffic)")
    manager.register_challenger(
        model_name,
        challenger_version,
        traffic_percentage=5,
        mode=TrafficMode.CANARY_5
    )


# =============================================================================
# Step 5: Simulate Prediction Requests
# =============================================================================

def simulate_prediction_requests(
    manager: ChampionChallengerManager,
    num_requests: int = 100,
    champion_mae: float = 0.05,
    challenger_mae: float = 0.03
) -> None:
    """
    Simulate prediction requests and record outcomes.

    Args:
        manager: ChampionChallengerManager instance
        num_requests: Number of requests to simulate
        champion_mae: Champion model MAE (Mean Absolute Error)
        challenger_mae: Challenger model MAE
    """
    model_name = "process_heat_predictor"

    logger.info(f"Simulating {num_requests} prediction requests...")

    for i in range(num_requests):
        request_id = f"req_{i:06d}"
        model_version = manager.route_request(request_id, model_name)

        base_mae = champion_mae if model_version == "1.0.0" else challenger_mae
        actual_mae = base_mae + np.random.normal(0, 0.001)

        manager.record_outcome(
            request_id,
            model_version,
            {
                "mae": max(0.0, actual_mae),
                "rmse": actual_mae * 1.5,
                "r2_score": 0.95,
                "inference_time_ms": 15.2 if model_version == "1.0.0" else 12.8
            },
            execution_time_ms=15.2 if model_version == "1.0.0" else 12.8,
            features={"temperature": 45.0, "pressure": 100}
        )

        if (i + 1) % 25 == 0:
            logger.info(f"  Processed {i + 1}/{num_requests} requests")

    logger.info("Simulation complete")


# =============================================================================
# Step 6: Evaluate Challenger
# =============================================================================

def evaluate_challenger(manager: ChampionChallengerManager) -> None:
    """Evaluate challenger performance against champion."""
    model_name = "process_heat_predictor"

    logger.info("Evaluating challenger performance...")
    evaluation = manager.evaluate_challenger(model_name, confidence_level=0.95)

    logger.info(f"""
    Evaluation Results:
    -------------------
    Model: {evaluation.model_name}
    Champion: {evaluation.champion_version}
    Challenger: {evaluation.challenger_version}

    Champion Performance:
      - MAE: {evaluation.champion_mean_metric:.6f}

    Challenger Performance:
      - MAE: {evaluation.challenger_mean_metric:.6f}

    Improvements:
      - MAE improvement: {evaluation.metric_improvement_pct:.2f}%

    Statistical Decision:
      - Should Promote: {evaluation.should_promote}
      - Confidence Level: {evaluation.confidence_level}
      - P-value: {evaluation.p_value:.4f}
      - Samples: {evaluation.samples_collected}
    """)

    return evaluation


# =============================================================================
# Step 7: Promote Challenger
# =============================================================================

def promote_challenger(manager: ChampionChallengerManager) -> None:
    """Promote challenger to champion."""
    model_name = "process_heat_predictor"

    logger.info("Promoting challenger to champion...")
    success = manager.promote_challenger(model_name)

    if success:
        logger.info(f"Successfully promoted {model_name} to v{manager.champions[model_name]}")
    else:
        logger.error(f"Failed to promote {model_name}")


# =============================================================================
# Step 8: Monitor for Degradation and Rollback
# =============================================================================

def monitor_and_rollback(
    manager: ChampionChallengerManager,
    num_requests: int = 100,
    degraded_mae: float = 0.08
) -> None:
    """
    Monitor promoted challenger and rollback if degradation detected.

    In production, this would be a continuous monitoring process.
    This example simulates degradation and automatic rollback.
    """
    model_name = "process_heat_predictor"

    logger.info("Monitoring promoted model for degradation...")

    for i in range(num_requests):
        request_id = f"mon_{i:06d}"
        model_version = manager.route_request(request_id, model_name)

        actual_mae = degraded_mae + np.random.normal(0, 0.002)

        manager.record_outcome(
            request_id,
            model_version,
            {"mae": max(0.0, actual_mae)},
            execution_time_ms=15.2
        )

    evaluation = manager.evaluate_challenger(model_name)

    if not evaluation.should_promote:
        logger.warning("Degradation detected! Rolling back...")
        success = manager.rollback(model_name, "1.0.0")

        if success:
            logger.info("Successfully rolled back to 1.0.0")
        else:
            logger.error("Rollback failed!")


# =============================================================================
# Step 9: Get Deployment Status
# =============================================================================

def get_deployment_status(manager: ChampionChallengerManager) -> None:
    """Get current deployment status."""
    logger.info(f"""
    Current Deployment Status:
    --------------------------
    Champions: {manager.champions}
    Challengers: {manager.challengers}
    Promotion History ({len(manager.promotion_history)} events):
    """)

    for event in manager.promotion_history:
        logger.info(f"  - {event['timestamp']}: {event['event']}")


# =============================================================================
# Main Workflow
# =============================================================================

def main():
    """Execute complete champion-challenger workflow."""
    logger.info("Starting champion-challenger deployment workflow...")

    manager = initialize_manager()
    register_champion(manager)

    logger.info("\n=== Phase 1: Shadow Mode ===")
    deploy_challenger_shadow_mode(manager)
    simulate_prediction_requests(manager, num_requests=50, champion_mae=0.05, challenger_mae=0.03)
    evaluation = evaluate_challenger(manager)

    if evaluation.should_promote or evaluation.samples_collected > 0:
        logger.info("\n=== Phase 2: Canary Mode ===")

        if "process_heat_predictor" in manager.challengers:
            del manager.challengers["process_heat_predictor"]

        deploy_challenger_canary_mode(manager)
        simulate_prediction_requests(manager, num_requests=50, champion_mae=0.05, challenger_mae=0.03)

        evaluation = evaluate_challenger(manager)

        if evaluation.should_promote:
            logger.info("\n=== Phase 3: Promotion ===")
            promote_challenger(manager)
            get_deployment_status(manager)

            logger.info("\n=== Phase 4: Monitoring ===")
            time.sleep(0.5)
            simulate_prediction_requests(manager, num_requests=100, champion_mae=0.03, challenger_mae=0.03)
            logger.info("Monitoring complete - no degradation detected")
        else:
            logger.info("Challenger not statistically better in canary - keeping champion")
    else:
        logger.info("Challenger not ready in shadow mode - keeping champion")

    get_deployment_status(manager)
    logger.info("Champion-challenger workflow complete!")


if __name__ == "__main__":
    main()
