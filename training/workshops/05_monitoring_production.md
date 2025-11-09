# Workshop 5: Monitoring & Production Deployment

**Duration:** 2 hours
**Level:** Advanced
**Prerequisites:** Workshops 1-4 completed

---

## Workshop Overview

Learn to deploy and monitor production-ready GreenLang applications. Master telemetry, logging, metrics, and operational best practices.

### Learning Objectives

- Implement comprehensive telemetry
- Set up structured logging
- Create custom metrics and dashboards
- Monitor agent health and performance
- Deploy agents to production
- Handle production incidents

---

## Part 1: Telemetry & Metrics (30 minutes)

### TelemetryManager

```python
from GL_COMMONS.infrastructure.telemetry import TelemetryManager

class EmissionAgent(Agent):
    def __init__(self):
        super().__init__()
        self.telemetry = TelemetryManager(
            service_name="emission_calculator",
            environment="production"
        )

    def execute(self):
        # Track execution
        with self.telemetry.track_execution("calculate_emission"):
            result = self._calculate()

            # Record metrics
            self.telemetry.record_metric(
                name="emissions_calculated",
                value=result["co2_kg"],
                tags={"company": result["company"]}
            )

            return result
```

### Custom Metrics

```python
# Counter - increases only
self.telemetry.increment("api_calls_total")
self.telemetry.increment("errors_total", tags={"error_type": "validation"})

# Gauge - can increase or decrease
self.telemetry.gauge("active_sessions", 45)
self.telemetry.gauge("cache_size_mb", 128.5)

# Histogram - distribution of values
self.telemetry.histogram("response_time_ms", 125.4)
self.telemetry.histogram("emission_value_tons", 1234.5)

# Timer - measure duration
with self.telemetry.timer("database_query"):
    results = db.query("SELECT ...")
```

### Performance Tracking

```python
class PerformanceMonitoredAgent(Agent):
    def execute(self):
        # Start performance tracking
        perf = self.telemetry.start_performance_tracking()

        # Execute logic
        result = self._process_data()

        # Record performance
        perf.record({
            "items_processed": len(result),
            "duration_ms": perf.elapsed_ms(),
            "memory_mb": perf.memory_usage_mb()
        })

        return result
```

---

## Part 2: Structured Logging (25 minutes)

### LoggingService

```python
from GL_COMMONS.infrastructure.logging import LoggingService

logger = LoggingService.get_logger(__name__)

# Structured logging with context
logger.info("Processing emission data", extra={
    "company_id": "TSLA",
    "year": 2023,
    "scope": "1,2,3",
    "record_count": 150
})

# Error logging with stack trace
try:
    result = calculate_emission()
except Exception as e:
    logger.error("Calculation failed", extra={
        "company_id": company_id,
        "error": str(e),
        "error_type": type(e).__name__
    }, exc_info=True)
```

### Log Levels

```python
# DEBUG - detailed diagnostic info
logger.debug("Cache lookup", extra={"key": cache_key})

# INFO - general informational messages
logger.info("Agent started", extra={"version": "1.0.0"})

# WARNING - warning messages
logger.warning("High memory usage", extra={"memory_mb": 950})

# ERROR - error messages
logger.error("Database connection failed", extra={"host": db_host})

# CRITICAL - critical errors
logger.critical("System shutdown", extra={"reason": "Out of memory"})
```

### Correlation IDs

```python
import uuid

class CorrelatedAgent(Agent):
    def execute(self):
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())

        # Add to all logs
        logger = LoggingService.get_logger(__name__, extra={
            "correlation_id": correlation_id,
            "agent": self.name
        })

        logger.info("Starting execution")

        try:
            result = self._process()
            logger.info("Execution complete", extra={"status": "success"})
            return result

        except Exception as e:
            logger.error("Execution failed", extra={
                "status": "failed",
                "error": str(e)
            })
            raise
```

---

## Part 3: Health Checks & Monitoring (20 minutes)

### Health Check Endpoint

```python
from GL_COMMONS.infrastructure.monitoring import HealthCheck

class AgentHealthCheck(HealthCheck):
    def __init__(self, agent):
        self.agent = agent

    def check_health(self):
        """Perform health check."""
        checks = {
            "database": self._check_database(),
            "cache": self._check_cache(),
            "llm": self._check_llm(),
            "memory": self._check_memory()
        }

        # Overall status
        all_healthy = all(check["status"] == "healthy" for check in checks.values())

        return {
            "status": "healthy" if all_healthy else "unhealthy",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }

    def _check_database(self):
        """Check database connection."""
        try:
            self.agent.db.query("SELECT 1")
            return {"status": "healthy", "latency_ms": 5}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_cache(self):
        """Check cache connection."""
        try:
            self.agent.cache.set("health_check", "ok", ttl=10)
            self.agent.cache.get("health_check")
            return {"status": "healthy"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_llm(self):
        """Check LLM service."""
        try:
            # Quick test call
            response = self.agent.llm_session.send_message("ping")
            return {"status": "healthy", "response_time_ms": 100}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def _check_memory(self):
        """Check memory usage."""
        import psutil
        memory = psutil.Process().memory_info()
        memory_mb = memory.rss / 1024 / 1024

        if memory_mb > 1000:  # Over 1GB
            return {"status": "warning", "memory_mb": memory_mb}
        return {"status": "healthy", "memory_mb": memory_mb}
```

### Readiness vs Liveness

```python
class ProductionAgent(Agent):
    def is_ready(self):
        """Readiness probe - can accept traffic?"""
        return (
            self.db is not None and
            self.cache is not None and
            self.setup_complete
        )

    def is_alive(self):
        """Liveness probe - is process alive?"""
        try:
            # Simple check
            return True
        except:
            return False
```

---

## Part 4: Alerting & Notifications (20 minutes)

### Alert Configuration

```python
from GL_COMMONS.infrastructure.monitoring import AlertManager

alerts = AlertManager()

# Error rate alert
alerts.add_alert(
    name="high_error_rate",
    condition="error_rate > 0.05",  # 5% error rate
    severity="critical",
    notification_channels=["slack:#incidents", "pagerduty"]
)

# Performance alert
alerts.add_alert(
    name="slow_response",
    condition="p95_response_time_ms > 2000",  # 2 second p95
    severity="warning",
    notification_channels=["slack:#performance"]
)

# Cost alert
alerts.add_alert(
    name="high_llm_cost",
    condition="daily_llm_cost > 100",  # $100/day
    severity="warning",
    notification_channels=["email:finance@company.com"]
)
```

### Custom Alerts

```python
class CostMonitoringAgent(Agent):
    def execute(self):
        result = self._process()

        # Check cost threshold
        daily_cost = self.telemetry.get_metric("daily_llm_cost")

        if daily_cost > 100:
            self._send_alert(
                severity="warning",
                message=f"Daily LLM cost exceeded $100: ${daily_cost:.2f}",
                metadata={
                    "cost": daily_cost,
                    "threshold": 100,
                    "agent": self.name
                }
            )

        return result

    def _send_alert(self, severity, message, metadata):
        """Send alert to notification channels."""
        AlertManager().send_alert(
            severity=severity,
            message=message,
            metadata=metadata
        )
```

---

## Part 5: Production Deployment (30 minutes)

### Deployment Checklist

```python
# deployment_checklist.py

class DeploymentValidator:
    """Validate agent before production deployment."""

    def validate(self, agent):
        """Run all validation checks."""
        checks = [
            self._check_infrastructure_usage(),
            self._check_error_handling(),
            self._check_logging(),
            self._check_metrics(),
            self._check_health_checks(),
            self._check_documentation(),
            self._check_tests()
        ]

        failed = [check for check in checks if not check["passed"]]

        if failed:
            raise DeploymentError(f"{len(failed)} checks failed", failed)

        return True

    def _check_infrastructure_usage(self):
        """Verify infrastructure is used correctly."""
        # Check no direct imports of openai, anthropic, redis, etc.
        violations = run_greenlang_first_check()

        return {
            "name": "Infrastructure Usage",
            "passed": len(violations) == 0,
            "violations": violations
        }

    def _check_error_handling(self):
        """Verify error handling is implemented."""
        # Check for try/except blocks
        # Check for retry logic
        # Check for graceful degradation
        pass

    def _check_logging(self):
        """Verify logging is comprehensive."""
        # Check for logger usage
        # Check for structured logging
        # Check for correlation IDs
        pass

    def _check_metrics(self):
        """Verify metrics are tracked."""
        # Check for telemetry usage
        # Check for custom metrics
        pass
```

### Environment Configuration

```python
# config/production.yaml
environment: production

infrastructure:
  database:
    host: prod-db.example.com
    pool_size: 20
    timeout: 30

  cache:
    host: prod-redis.example.com
    max_connections: 50

  llm:
    provider: openai
    model: gpt-4
    max_retries: 5
    timeout: 60

monitoring:
  telemetry_enabled: true
  log_level: INFO
  metrics_interval: 60

alerting:
  enabled: true
  channels:
    - slack:#production-alerts
    - pagerduty

limits:
  max_concurrent_executions: 100
  max_llm_cost_daily: 500
  max_memory_mb: 2048
```

### Deployment Script

```python
# deploy.py
import subprocess
import sys

class AgentDeployer:
    """Deploy agent to production."""

    def deploy(self, agent_name, version):
        """Execute deployment."""

        print(f"Deploying {agent_name} v{version}")

        # 1. Run tests
        print("Running tests...")
        self._run_tests()

        # 2. Run validation
        print("Running deployment validation...")
        self._validate_deployment()

        # 3. Build container
        print("Building container...")
        self._build_container(agent_name, version)

        # 4. Deploy to staging
        print("Deploying to staging...")
        self._deploy_to_staging(agent_name, version)

        # 5. Run smoke tests
        print("Running smoke tests...")
        self._run_smoke_tests()

        # 6. Deploy to production
        print("Deploying to production...")
        self._deploy_to_production(agent_name, version)

        # 7. Monitor deployment
        print("Monitoring deployment...")
        self._monitor_deployment(agent_name)

        print(f"✓ Deployment complete: {agent_name} v{version}")

    def _run_tests(self):
        """Run unit and integration tests."""
        result = subprocess.run(["pytest", "tests/"], capture_output=True)
        if result.returncode != 0:
            raise DeploymentError("Tests failed", result.stderr)

    def _validate_deployment(self):
        """Run deployment validation."""
        validator = DeploymentValidator()
        validator.validate(agent)

    def _build_container(self, agent_name, version):
        """Build Docker container."""
        subprocess.run([
            "docker", "build",
            "-t", f"{agent_name}:{version}",
            "."
        ])

    def _deploy_to_staging(self, agent_name, version):
        """Deploy to staging environment."""
        subprocess.run([
            "kubectl", "apply",
            "-f", f"k8s/staging/{agent_name}.yaml"
        ])

    def _run_smoke_tests(self):
        """Run smoke tests in staging."""
        # Test basic functionality
        # Test health endpoints
        # Test critical paths
        pass

    def _deploy_to_production(self, agent_name, version):
        """Deploy to production with rolling update."""
        subprocess.run([
            "kubectl", "rollout", "restart",
            f"deployment/{agent_name}",
            "-n", "production"
        ])

    def _monitor_deployment(self, agent_name):
        """Monitor deployment health."""
        # Check error rates
        # Check response times
        # Check resource usage
        pass
```

---

## Part 6: Hands-On Lab - Production-Ready Agent (35 minutes)

### Lab: Add Monitoring to Emission Agent

```python
# production_emission_agent.py
from GL_COMMONS.infrastructure.agents import Agent
from GL_COMMONS.infrastructure.llm import ChatSession
from GL_COMMONS.infrastructure.cache import CacheManager
from GL_COMMONS.infrastructure.telemetry import TelemetryManager
from GL_COMMONS.infrastructure.logging import LoggingService
from GL_COMMONS.infrastructure.monitoring import HealthCheck
import uuid

logger = LoggingService.get_logger(__name__)

class ProductionEmissionAgent(Agent):
    """Production-ready emission calculator with full monitoring."""

    def __init__(self):
        super().__init__(
            name="emission_calculator",
            version="2.0.0",
            description="Production emission calculator"
        )

        self.llm_session = None
        self.cache = None
        self.telemetry = None
        self.health_check = None

    def setup(self):
        """Initialize with monitoring."""
        logger.info("Setting up ProductionEmissionAgent", extra={
            "version": self.version,
            "environment": "production"
        })

        # Initialize infrastructure
        self.llm_session = ChatSession(
            provider="openai",
            model="gpt-4"
        )

        self.cache = CacheManager()

        # Initialize telemetry
        self.telemetry = TelemetryManager(
            service_name="emission_calculator",
            environment="production"
        )

        # Initialize health check
        self.health_check = HealthCheck(self)

        logger.info("Setup complete")

    def execute(self):
        """Execute with full monitoring."""

        # Generate correlation ID
        correlation_id = str(uuid.uuid4())

        # Create contextualized logger
        exec_logger = LoggingService.get_logger(__name__, extra={
            "correlation_id": correlation_id,
            "agent": self.name,
            "version": self.version
        })

        exec_logger.info("Execution started", extra={
            "input": self.input_data
        })

        # Track execution time
        with self.telemetry.timer("execution_duration_ms"):
            try:
                # Validate input
                self._validate_input()

                # Calculate emission
                result = self._calculate_emission()

                # Record metrics
                self.telemetry.increment("calculations_total")
                self.telemetry.histogram(
                    "emission_value_tons",
                    result["co2_kg"] / 1000
                )

                # Check cost threshold
                self._check_cost_threshold()

                exec_logger.info("Execution completed", extra={
                    "result": result,
                    "status": "success"
                })

                return result

            except Exception as e:
                # Track error
                self.telemetry.increment("errors_total", tags={
                    "error_type": type(e).__name__
                })

                exec_logger.error("Execution failed", extra={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "status": "failed"
                }, exc_info=True)

                # Send alert for critical errors
                if isinstance(e, CriticalError):
                    self._send_alert(
                        severity="critical",
                        message=f"Critical error in emission calculation: {e}",
                        metadata={"correlation_id": correlation_id}
                    )

                raise

    def _validate_input(self):
        """Validate input with metrics."""
        with self.telemetry.timer("validation_duration_ms"):
            # Validation logic
            pass

    def _calculate_emission(self):
        """Calculate emission with caching and metrics."""

        # Check cache
        cache_key = self._get_cache_key()

        with self.telemetry.timer("cache_lookup_ms"):
            cached = self.cache.get(cache_key)

        if cached:
            self.telemetry.increment("cache_hits")
            logger.debug("Cache hit", extra={"key": cache_key})
            return cached

        self.telemetry.increment("cache_misses")

        # Calculate with LLM
        with self.telemetry.timer("llm_call_ms"):
            result = self._call_llm()

        # Cache result
        self.cache.set(cache_key, result, ttl=3600)

        # Track cost
        llm_cost = self.llm_session.get_cost()
        self.telemetry.gauge("llm_cost_usd", llm_cost)

        return result

    def _check_cost_threshold(self):
        """Check daily cost threshold."""
        daily_cost = self.telemetry.get_metric("daily_llm_cost")

        if daily_cost > 100:
            logger.warning("Daily cost threshold exceeded", extra={
                "daily_cost": daily_cost,
                "threshold": 100
            })

            self._send_alert(
                severity="warning",
                message=f"Daily LLM cost exceeded threshold: ${daily_cost:.2f}",
                metadata={"cost": daily_cost}
            )

    def get_health(self):
        """Get health status."""
        return self.health_check.check_health()
```

### Test Monitoring

```python
# test_monitoring.py
from production_emission_agent import ProductionEmissionAgent

agent = ProductionEmissionAgent()
agent.setup()

# Execute multiple times to generate metrics
for i in range(100):
    result = agent.execute_with_input({
        "activity_type": "electricity_kwh",
        "amount": 100,
        "unit": "kWh"
    })

# Check metrics
metrics = agent.telemetry.get_all_metrics()
print("Metrics:")
print(f"  Calculations: {metrics['calculations_total']}")
print(f"  Cache hits: {metrics['cache_hits']}")
print(f"  Cache misses: {metrics['cache_misses']}")
print(f"  Hit rate: {metrics['cache_hits'] / 100 * 100:.1f}%")
print(f"  Avg duration: {metrics['execution_duration_ms']['avg']:.0f}ms")
print(f"  P95 duration: {metrics['execution_duration_ms']['p95']:.0f}ms")

# Check health
health = agent.get_health()
print(f"\nHealth: {health['status']}")
for check, status in health['checks'].items():
    print(f"  {check}: {status['status']}")
```

---

## Workshop Wrap-Up

### What You Learned

✓ Comprehensive telemetry and metrics
✓ Structured logging with correlation IDs
✓ Health checks and monitoring
✓ Alerting and notifications
✓ Production deployment process
✓ Built production-ready agent

### Key Takeaways

1. **Monitor everything** - You can't fix what you can't see
2. **Structured logging** - Makes debugging 10x easier
3. **Health checks** - Know when things break
4. **Alert wisely** - Too many alerts = ignored alerts
5. **Deploy safely** - Test, validate, monitor

### Homework

Production-ize an agent:
1. Add comprehensive telemetry
2. Implement structured logging
3. Create health checks
4. Set up alerts
5. Deploy to staging
6. Monitor for 24 hours

---

**Workshop Complete! Ready for Workshop 6: Advanced Topics**
