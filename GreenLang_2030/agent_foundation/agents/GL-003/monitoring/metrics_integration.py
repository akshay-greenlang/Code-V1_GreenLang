"""
Metrics integration utilities for GL-003 SteamSystemAnalyzer.

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
    analysis_requests_total,
    analysis_duration_seconds,
    cache_hits_total,
    cache_misses_total,
    http_request_duration_seconds,
    http_requests_total,
    system_uptime_seconds,
    track_request_metrics,
    track_analysis_metrics
)

logger = logging.getLogger(__name__)


class MetricsIntegration:
    """
    Helper class to integrate metrics into GL-003 orchestrator.

    Usage:
        metrics = MetricsIntegration()

        # Track analysis
        with metrics.track_analysis('distribution_efficiency'):
            result = orchestrator.analyze()

        # Update steam system state
        metrics.update_steam_system_state(system_id, state)
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
    def track_analysis(self, analysis_type: str):
        """
        Context manager to track analysis execution.

        Args:
            analysis_type: Analysis type name

        Example:
            with metrics.track_analysis('leak_detection'):
                result = analyze_leaks()
        """
        start_time = time.time()

        try:
            yield
            # Success
            duration = time.time() - start_time
            analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
            analysis_requests_total.labels(
                analysis_type=analysis_type,
                status='success'
            ).inc()

        except TimeoutError:
            # Timeout
            duration = time.time() - start_time
            analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
            analysis_requests_total.labels(
                analysis_type=analysis_type,
                status='timeout'
            ).inc()
            raise

        except Exception as e:
            # Failure
            duration = time.time() - start_time
            analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
            analysis_requests_total.labels(
                analysis_type=analysis_type,
                status='failure'
            ).inc()
            logger.error(f"Analysis failed: {e}")
            raise

    def track_cache_operation(self, operation: str, hit: bool):
        """
        Track cache hit/miss.

        Args:
            operation: Cache operation name (e.g., 'steam_analysis', 'leak_detection')
            hit: True if cache hit, False if miss
        """
        MetricsCollector.record_cache_operation(operation, hit)

    def update_steam_system_state(self, system_id: str, state: Dict[str, Any]):
        """
        Update steam system operational metrics.

        Args:
            system_id: Steam system identifier
            state: System state dictionary with metrics
        """
        metrics = {
            "pressure_bar": state.get("pressure_bar", 0),
            "temperature_c": state.get("temperature_c", 0),
            "flow_rate_kg_hr": state.get("flow_rate_kg_hr", 0),
            "condensate_return_rate_kg_hr": state.get("condensate_return_rate_kg_hr", 0),
            "condensate_return_percent": state.get("condensate_return_percent", 0),
            "steam_quality_percent": state.get("steam_quality_percent", 0),
            "distribution_efficiency_percent": state.get("distribution_efficiency_percent", 0),
            "location": state.get("location", "unknown"),
            "steam_type": state.get("steam_type", "unknown")
        }

        MetricsCollector.update_steam_system_metrics(system_id, metrics)

    def update_leak_metrics(self, system_id: str, leaks: Dict[str, Any]):
        """
        Update leak detection metrics.

        Args:
            system_id: Steam system identifier
            leaks: Leak data dictionary
        """
        leak_data = {
            "active_leaks": leaks.get("active_leaks", {}),
            "total_leak_cost_usd_hr": leaks.get("total_leak_cost_usd_hr", 0)
        }

        MetricsCollector.update_leak_metrics(system_id, leak_data)

    def update_steam_trap_metrics(self, system_id: str, traps: Dict[str, Any]):
        """
        Update steam trap performance metrics.

        Args:
            system_id: Steam system identifier
            traps: Steam trap data dictionary
        """
        trap_data = {
            "operational_count": traps.get("operational_count", {}),
            "failed_count": traps.get("failed_count", {})
        }

        MetricsCollector.update_steam_trap_metrics(system_id, trap_data)

    def record_analysis_result(
        self,
        system_id: str,
        steam_type: str,
        analysis_type: str,
        result: Dict[str, Any]
    ):
        """
        Record complete analysis result.

        Args:
            system_id: Steam system identifier
            steam_type: Steam type
            analysis_type: Analysis type
            result: Analysis result dictionary
        """
        MetricsCollector.record_analysis_result(
            system_id=system_id,
            steam_type=steam_type,
            analysis_type=analysis_type,
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


# Decorator for automatic analysis tracking
def track_analysis_execution(analysis_type: str):
    """
    Decorator to automatically track analysis execution.

    Args:
        analysis_type: Analysis type name

    Example:
        @track_analysis_execution('distribution_efficiency')
        async def analyze_distribution(data):
            return analysis_result
    """
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time

                analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
                analysis_requests_total.labels(
                    analysis_type=analysis_type,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                duration = time.time() - start_time

                analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
                analysis_requests_total.labels(
                    analysis_type=analysis_type,
                    status='failure'
                ).inc()

                raise

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
                analysis_requests_total.labels(
                    analysis_type=analysis_type,
                    status='success'
                ).inc()

                return result

            except Exception as e:
                duration = time.time() - start_time

                analysis_duration_seconds.labels(analysis_type=analysis_type).observe(duration)
                analysis_requests_total.labels(
                    analysis_type=analysis_type,
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
        orchestrator: SteamSystemAnalyzer instance

    Example:
        orchestrator = SteamSystemAnalyzer(config)
        integrate_metrics_into_orchestrator(orchestrator)
    """
    # Initialize metrics
    metrics = MetricsIntegration()
    orchestrator._metrics_integration = metrics

    # Store original methods
    original_execute = orchestrator.execute
    original_analyze_distribution = orchestrator._analyze_distribution_async
    original_cache_get = orchestrator._results_cache.get

    # Wrap execute method
    async def execute_with_metrics(input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            result = await original_execute(input_data)

            # Update metrics from result
            if 'operational_state' in result:
                state = result['operational_state']
                system_id = input_data.get('steam_system_data', {}).get('system_id', 'unknown')
                metrics.update_steam_system_state(system_id, state)

            if 'leak_analysis' in result:
                leaks = result['leak_analysis']
                system_id = input_data.get('steam_system_data', {}).get('system_id', 'unknown')
                metrics.update_leak_metrics(system_id, leaks)

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
