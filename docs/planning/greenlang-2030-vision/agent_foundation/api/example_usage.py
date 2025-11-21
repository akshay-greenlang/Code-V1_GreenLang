# -*- coding: utf-8 -*-
"""
Example Usage: GreenLang Health Check API

Demonstrates how to:
1. Initialize the FastAPI application with dependencies
2. Register dependencies with health manager
3. Run the API server
4. Test health check endpoints

Usage:
    python api/example_usage.py
"""

import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def initialize_dependencies():
    """
    Initialize all application dependencies.

    In a real application, this would:
    - Load configuration from environment/secrets
    - Initialize database connection pool
    - Initialize Redis connection pool
    - Configure LLM router with providers
    - Initialize vector store
    """
    logger.info("Initializing dependencies...")

    # Example: Initialize PostgreSQL
    # from ..database.postgres_manager import PostgresManager, PostgresConfig
    #
    # db_config = PostgresConfig(
    #     primary_host=os.getenv("POSTGRES_HOST", "localhost"),
    #     primary_port=int(os.getenv("POSTGRES_PORT", "5432")),
    #     database=os.getenv("POSTGRES_DATABASE", "greenlang"),
    #     user=os.getenv("POSTGRES_USER", "greenlang"),
    #     password=os.getenv("POSTGRES_PASSWORD"),
    #     pool_min_size=10,
    #     pool_max_size=20
    # )
    # db_manager = PostgresManager(db_config)
    # await db_manager.initialize()
    # logger.info("PostgreSQL initialized")

    # Example: Initialize Redis
    # from ..cache.redis_manager import RedisManager, RedisConfig
    #
    # redis_config = RedisConfig(
    #     host=os.getenv("REDIS_HOST", "localhost"),
    #     port=int(os.getenv("REDIS_PORT", "6379")),
    #     max_connections=50,
    #     cluster_mode="standalone"
    # )
    # redis_manager = RedisManager(redis_config)
    # await redis_manager.initialize()
    # logger.info("Redis initialized")

    # Example: Initialize LLM Router
    # from ..llm.llm_router import LLMRouter, LoadBalancingStrategy
    # from ..llm.anthropic_provider import AnthropicProvider
    #
    # llm_router = LLMRouter(strategy=LoadBalancingStrategy.LEAST_COST)
    #
    # # Register Anthropic provider
    # anthropic_provider = AnthropicProvider(
    #     api_key=os.getenv("ANTHROPIC_API_KEY")
    # )
    # llm_router.register_provider("anthropic", anthropic_provider, priority=1)
    # logger.info("LLM router initialized")

    # Example: Initialize Vector Store
    # from ..rag.vector_store import ChromaVectorStore
    #
    # vector_store = ChromaVectorStore(
    #     collection_name="greenlang_embeddings",
    #     persist_directory="/data/chroma"
    # )
    # logger.info("Vector store initialized")

    # For this example, we'll return None for all dependencies
    # In production, return actual initialized instances
    return None, None, None, None


async def register_health_dependencies(db_manager, redis_manager, llm_router, vector_store):
    """
    Register dependencies with health check manager.

    Args:
        db_manager: PostgresManager instance
        redis_manager: RedisManager instance
        llm_router: LLMRouter instance
        vector_store: VectorStore instance
    """
    logger.info("Registering dependencies with health manager...")

    from .health import health_manager

    health_manager.set_dependencies(
        db_manager=db_manager,
        redis_manager=redis_manager,
        llm_router=llm_router,
        vector_store=vector_store
    )

    # Mark startup as complete
    health_manager.mark_startup_complete()
    logger.info("Health manager initialized - startup complete")


async def test_health_endpoints():
    """
    Test all health check endpoints.

    Makes HTTP requests to verify endpoints are working.
    """
    import httpx

    base_url = "http://localhost:8000"

    logger.info("\n" + "="*60)
    logger.info("TESTING HEALTH CHECK ENDPOINTS")
    logger.info("="*60)

    async with httpx.AsyncClient() as client:
        # Test liveness probe
        logger.info("\n1. Testing Liveness Probe (/healthz)")
        logger.info("-" * 40)
        try:
            response = await client.get(f"{base_url}/healthz")
            logger.info(f"Status Code: {response.status_code}")
            logger.info(f"Response: {response.json()}")

            if response.status_code == 200:
                logger.info("✓ Liveness probe: PASSED")
            else:
                logger.error("✗ Liveness probe: FAILED")
        except Exception as e:
            logger.error(f"✗ Liveness probe error: {e}")

        # Test readiness probe
        logger.info("\n2. Testing Readiness Probe (/ready)")
        logger.info("-" * 40)
        try:
            response = await client.get(f"{base_url}/ready")
            logger.info(f"Status Code: {response.status_code}")
            data = response.json()
            logger.info(f"Overall Status: {data['status']}")

            logger.info("\nComponent Status:")
            for component in data.get("components", []):
                status_symbol = "✓" if component["status"] == "healthy" else "✗"
                logger.info(
                    f"  {status_symbol} {component['name']}: {component['status']} "
                    f"({component['response_time_ms']:.1f}ms)"
                )

            if response.status_code == 200:
                logger.info("\n✓ Readiness probe: PASSED")
            else:
                logger.warning("\n⚠ Readiness probe: DEGRADED (some components unhealthy)")
        except Exception as e:
            logger.error(f"✗ Readiness probe error: {e}")

        # Test startup probe
        logger.info("\n3. Testing Startup Probe (/startup)")
        logger.info("-" * 40)
        try:
            response = await client.get(f"{base_url}/startup")
            logger.info(f"Status Code: {response.status_code}")
            data = response.json()
            logger.info(f"Overall Status: {data['status']}")
            logger.info(f"Uptime: {data['uptime_seconds']:.1f} seconds")

            if response.status_code == 200:
                logger.info("✓ Startup probe: PASSED")
            else:
                logger.warning("⚠ Startup probe: FAILED (startup not complete)")
        except Exception as e:
            logger.error(f"✗ Startup probe error: {e}")

        # Test API info endpoint
        logger.info("\n4. Testing API Info (/api/v1/info)")
        logger.info("-" * 40)
        try:
            response = await client.get(f"{base_url}/api/v1/info")
            logger.info(f"Status Code: {response.status_code}")
            data = response.json()
            logger.info(f"API Name: {data['name']}")
            logger.info(f"Version: {data['version']}")
            logger.info(f"Uptime: {data['uptime_seconds']:.1f} seconds")
            logger.info("\nFeatures:")
            for feature in data.get("features", []):
                logger.info(f"  - {feature}")

            if response.status_code == 200:
                logger.info("\n✓ API info: PASSED")
        except Exception as e:
            logger.error(f"✗ API info error: {e}")

    logger.info("\n" + "="*60)
    logger.info("HEALTH CHECK TESTS COMPLETE")
    logger.info("="*60 + "\n")


async def run_api_server():
    """
    Run the FastAPI server with uvicorn.

    This starts the API and keeps it running.
    """
    import uvicorn

    logger.info("Starting GreenLang API server...")
    logger.info("API will be available at: http://localhost:8000")
    logger.info("API documentation: http://localhost:8000/api/docs")
    logger.info("Health checks:")
    logger.info("  - Liveness:  http://localhost:8000/healthz")
    logger.info("  - Readiness: http://localhost:8000/ready")
    logger.info("  - Startup:   http://localhost:8000/startup")
    logger.info("\nPress Ctrl+C to stop the server\n")

    config = uvicorn.Config(
        app="api.main:app",
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )

    server = uvicorn.Server(config)
    await server.serve()


async def main():
    """
    Main entry point.

    Demonstrates complete application lifecycle:
    1. Initialize dependencies
    2. Register with health manager
    3. Run API server
    """
    logger.info("="*60)
    logger.info("GreenLang Health Check API - Example Usage")
    logger.info("="*60 + "\n")

    try:
        # Initialize dependencies
        db_manager, redis_manager, llm_router, vector_store = await initialize_dependencies()

        # Register with health manager
        await register_health_dependencies(db_manager, redis_manager, llm_router, vector_store)

        # Start API server
        await run_api_server()

    except KeyboardInterrupt:
        logger.info("\n\nShutting down gracefully...")
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        raise


def example_kubernetes_health_checks():
    """
    Example: How to configure Kubernetes health checks.

    This is just documentation - shows the YAML configuration.
    """
    print("""
    ============================================================
    KUBERNETES HEALTH CHECK CONFIGURATION EXAMPLE
    ============================================================

    Add these probes to your Kubernetes Deployment:

    # Startup Probe (runs first, protects slow startup)
    startupProbe:
      httpGet:
        path: /startup
        port: 8000
      initialDelaySeconds: 0
      periodSeconds: 5
      timeoutSeconds: 3
      failureThreshold: 30  # Allow 150s for startup

    # Liveness Probe (restarts container if fails)
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8000
      initialDelaySeconds: 10
      periodSeconds: 5
      timeoutSeconds: 1
      failureThreshold: 3  # Restart after 15s

    # Readiness Probe (removes from service if fails)
    readinessProbe:
      httpGet:
        path: /ready
        port: 8000
      initialDelaySeconds: 30
      periodSeconds: 10
      timeoutSeconds: 5
      failureThreshold: 3  # Remove after 30s

    See kubernetes_health_checks.yaml for complete configuration.
    ============================================================
    """)


def example_testing_health_checks():
    """
    Example: How to test health checks locally.

    Shows various ways to test the endpoints.
    """
    print("""
    ============================================================
    TESTING HEALTH CHECKS LOCALLY
    ============================================================

    1. Start the API server:
       python api/example_usage.py

    2. Test with curl:
       curl http://localhost:8000/healthz
       curl http://localhost:8000/ready
       curl http://localhost:8000/startup

    3. Test with httpx (Python):
       import httpx
       response = httpx.get("http://localhost:8000/healthz")
       print(response.json())

    4. Run automated tests:
       pytest api/test_health_api.py -v

    5. Load test health endpoints:
       ab -n 1000 -c 10 http://localhost:8000/healthz

    6. Monitor with Kubernetes:
       kubectl get pods -w
       kubectl describe pod <pod-name>
       kubectl logs -f <pod-name>

    ============================================================
    """)


if __name__ == "__main__":
    """
    Run the example.

    Options:
    - Run API server: python api/example_usage.py
    - Show K8s config: python api/example_usage.py --k8s
    - Show test info: python api/example_usage.py --test
    """
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--k8s":
            example_kubernetes_health_checks()
        elif sys.argv[1] == "--test":
            example_testing_health_checks()
        else:
            print("Usage:")
            print("  python api/example_usage.py          # Run API server")
            print("  python api/example_usage.py --k8s    # Show Kubernetes config")
            print("  python api/example_usage.py --test   # Show testing examples")
    else:
        # Run the API server
        asyncio.run(main())
