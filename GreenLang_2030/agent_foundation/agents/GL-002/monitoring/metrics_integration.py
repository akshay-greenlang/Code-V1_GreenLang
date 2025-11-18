"""
Metrics integration utilities for GL-002 BoilerEfficiencyOptimizer.

This module provides helper functions to integrate Prometheus metrics
into the orchestrator and tools seamlessly.
"""

import time
import functools
import logging
from typing import Callable, Any, Dict, Optional
from contextlib import contextmanager

from .metrics import (
    MetricsCollector,
    optimization_requests_total,
    optimization_duration_seconds,
    cache_hits_total,
    cache_misses_total,
    http_request_duration_seconds,
    http_requests_total,
    system_uptime_seconds,
    track_request_metrics,
    track_optimization_metrics
)

logger = logging.getLogger(__name__)


class MetricsIntegration:
    """
    Helper class to integrate metrics into GL-002 orchestrator.

    Usage:
        metrics = MetricsIntegration()

        # Track optimization
        with metrics.track_optimization('fuel_efficiency'):
            result = orchestrator.optimize()

        # Update boiler state
        metrics.update_boiler_state(boiler_id, state)
    """

    def __init__(self, start_time: Optional[float] = None):
        """
        Initialize metrics integration.

        Args:
            start_time: Application start time (for uptime tracking)
        """
        self.start_time = start_time or time.time()
        self._update_uptime()

    def _update_uptime(self):
        """Update system uptime metric."""
        uptime = time.time() - self.start_time
        system_uptime_seconds.set(uptime)

    @contextmanager
    def track_optimization(self, strategy: str):
        """
        Context manager to track optimization execution.

        Args:
            strategy: Optimization strategy name

        Example:
            with metrics.track_optimization('fuel_efficiency'):
                result = optimize_fuel()
        """
        start_time = time.time()

        try:
            yield
            # Success
            duration = time.time() - start_time
            optimization_duration_seconds.labels(strategy=strategy).observe(duration)
            optimization_requests_total.labels(
                strategy=strategy,
                status='success'
            ).inc()

        except TimeoutError:
            # Timeout
            duration = time.time() - start_time
            optimization_duration_seconds.labels(strategy=strategy).observe(duration)
            optimization_requests_total.labels(
                strategy=strategy,
                status='timeout'
            ).inc()
            raise

        except Exception as e:
            # Failure
            duration = time.time() - start_time
            optimization_duration_seconds.labels(strategy=strategy).observe(duration)
            optimization_requests_total.labels(
                strategy=strategy,
                status='failure'
            ).inc()
            logger.error(f"Optimization failed: {e}")
            raise

    def track_cache_operation(self, operation: str, hit: bool):
        """
        Track cache hit/miss.

        Args:
            operation: Cache operation name (e.g., 'state_analysis', 'combustion_opt')
            hit: True if cache hit, False if miss
        """
        MetricsCollector.record_cache_operation(operation, hit)

    def update_boiler_state(self, boiler_id: str, state: Dict[str, Any]):
        """
        Update boiler operational metrics.

        Args:
            boiler_id: Boiler identifier
            state: Boiler state dictionary with metrics
        """
        metrics = {
            "efficiency_percent": state.get("efficiency_percent", 0),
            "steam_flow_kg_hr": state.get("steam_flow_rate_kg_hr", 0),
            "fuel_flow_kg_hr": state.get("fuel_flow_rate_kg_hr", 0),
            "combustion_temperature_c": state.get("combustion_temperature_c", 0),
            "excess_air_percent": state.get("excess_air_percent", 0),
            "pressure_bar": state.get("pressure_bar", 0),
            "load_percent": state.get("load_percent", 0),
            "fuel_type": state.get("fuel_type", "unknown")
        }

        MetricsCollector.update_boiler_metrics(boiler_id, metrics)

    def update_emissions(self, boiler_id: str, emissions: Dict[str, Any]):
        """
        Update emissions metrics.

        Args:
            boiler_id: Boiler identifier
            emissions: Emissions data dictionary
        """
        emissions_data = {
            "co2_kg_hr": emissions.get("co2_emissions_kg_hr", 0),
            "nox_ppm": emissions.get("nox_emissions_ppm", 0),
            "co_ppm": emissions.get("co_ppm", 0),
            "so2_ppm": emissions.get("so2_ppm", 0),
            "compliance_status": emissions.get("compliance_status", "unknown")
        }

        MetricsCollector.update_emissions_metrics(boiler_id, emissions_data)

    def record_optimization_result(
        self,
        boiler_id: str,
        fuel_type: str,
        strategy: str,
        result: Dict[str, Any]
    ):
        """
        Record complete optimization result.

        Args:
            boiler_id: Boiler identifier
            fuel_type: Fuel type
            strategy: Optimization strategy
            result: Optimization result dictionary
        """
        MetricsCollector.record_optimization_result(
            boiler_id=boiler_id,
            fuel_type=fuel_type,
            strategy=strategy,
            result=result
        )

    def update_system_metrics(self):
        """Update system resource metrics."""
        try:
            import psutil

            process = psutil.Process()
            memory_info = process.memory_info()

            metrics = {
                "uptime_seconds": time.time() - self.start_time,
                "memory_rss_bytes": memory_info.rss,
                "memory_vms_bytes": memory_info.vms,
                "cpu_percent": process.cpu_percent(interval=0.1),
                "disk_usage": {}
            }

            # Get disk usage for all mount points
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    metrics["disk_usage"][partition.mountpoint] = usage.used
                except PermissionError:
                    pass

            MetricsCollector.update_system_metrics(metrics)

        except ImportError:
            logger.warning("psutil not installed, skipping system metrics")
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")


# Decorator for automatic optimization tracking
def track_optimization_execution(strategy: str):
    """
    Decorator to automatically track optimization execution.

    Args:
        strategy: Optimization strategy name

    Example:
        @track_optimization_execution('fuel_efficiency')
        async def optimize_fuel(data):
            return optimized_result
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                optimization_duration_seconds.labels(strategy=strategy).observe(duration)
                optimization_requests_total.labels(
                    strategy=strategy,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                duration = time.time() - start_time

                optimization_duration_seconds.labels(strategy=strategy).observe(duration)
                optimization_requests_total.labels(
                    strategy=strategy,
                    status='failure'
                ).inc()

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                optimization_duration_seconds.labels(strategy=strategy).observe(duration)
                optimization_requests_total.labels(
                    strategy=strategy,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                duration = time.time() - start_time

                optimization_duration_seconds.labels(strategy=strategy).observe(duration)
                optimization_requests_total.labels(
                    strategy=strategy,
                    status='failure'
                ).inc()

                raise

        # Return appropriate wrapper
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator


# Background task to update system metrics periodically
async def metrics_updater(metrics: MetricsIntegration, interval: int = 60):
    """
    Background task to update system metrics periodically.

    Args:
        metrics: MetricsIntegration instance
        interval: Update interval in seconds

    Example:
        import asyncio

        metrics = MetricsIntegration()
        asyncio.create_task(metrics_updater(metrics, interval=60))
    """
    import asyncio

    while True:
        try:
            metrics.update_system_metrics()
            await asyncio.sleep(interval)
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            await asyncio.sleep(interval)


# Example integration into orchestrator
def integrate_metrics_into_orchestrator(orchestrator):
    """
    Integrate metrics into existing orchestrator.

    Args:
        orchestrator: BoilerEfficiencyOptimizer instance

    Example:
        orchestrator = BoilerEfficiencyOptimizer(config)
        integrate_metrics_into_orchestrator(orchestrator)
    """
    # Initialize metrics
    metrics = MetricsIntegration()
    orchestrator._metrics_integration = metrics

    # Store original methods
    original_execute = orchestrator.execute
    original_optimize_combustion = orchestrator._optimize_combustion_async
    original_cache_get = orchestrator._results_cache.get

    # Wrap execute method
    async def execute_with_metrics(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await original_execute(input_data)

            # Update metrics from result
            if 'operational_state' in result:
                state = result['operational_state']
                boiler_id = input_data.get('boiler_data', {}).get('boiler_id', 'unknown')
                metrics.update_boiler_state(boiler_id, state)

            if 'emissions_optimization' in result:
                emissions = result['emissions_optimization']
                boiler_id = input_data.get('boiler_data', {}).get('boiler_id', 'unknown')
                metrics.update_emissions(boiler_id, emissions)

            return result

        except Exception as e:
            logger.error(f"Execute failed: {e}")
            raise

    # Wrap cache get method
    def cache_get_with_metrics(key: str):
        result = original_cache_get(key)
        hit = result is not None
        operation = key.split('_')[0] if '_' in key else 'unknown'
        metrics.track_cache_operation(operation, hit)
        return result

    # Replace methods
    orchestrator.execute = execute_with_metrics
    orchestrator._results_cache.get = cache_get_with_metrics

    logger.info("Metrics integration completed")

    return orchestrator
