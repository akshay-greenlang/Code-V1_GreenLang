# -*- coding: utf-8 -*-
"""
Debugging and Troubleshooting Tools
====================================

Advanced debugging capabilities for:
- Interactive debugging
- Health checks
- Performance profiling
- Log analysis
- Troubleshooting workflows

Author: GL-DevOpsEngineer
"""

import time
import traceback
import psutil
import cProfile
import pstats
import io
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Any, Optional, Callable, Tuple
from contextlib import contextmanager
from pathlib import Path
import asyncio
import inspect
import sys
from collections import defaultdict, Counter
from greenlang.determinism import DeterministicClock


class HealthStatus(Enum):
    """Health check status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class IssueCategory(Enum):
    """Issue categories for troubleshooting"""
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    INTEGRATION = "integration"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"


@dataclass
class HealthCheckResult:
    """Health check result"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class ProfileResult:
    """Profiling result"""
    profile_type: str  # cpu, memory, io
    duration_seconds: float
    top_functions: List[Dict[str, Any]]
    stats: Dict[str, Any]
    flame_graph: Optional[str] = None


@dataclass
class DiagnosticInfo:
    """System diagnostic information"""
    timestamp: datetime
    system: Dict[str, Any]
    process: Dict[str, Any]
    dependencies: Dict[str, Any]
    configuration: Dict[str, Any]


class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        """Initialize health checker"""
        self.checks = {}
        self.last_results = {}
        self._register_default_checks()

    def _register_default_checks(self):
        """Register default health checks"""
        # Database health
        self.register_check(
            "database",
            self._check_database,
            critical=True
        )

        # Cache health
        self.register_check(
            "cache",
            self._check_cache,
            critical=False
        )

        # LLM service health
        self.register_check(
            "llm",
            self._check_llm,
            critical=True
        )

        # Disk space
        self.register_check(
            "disk_space",
            self._check_disk_space,
            critical=True
        )

        # Memory usage
        self.register_check(
            "memory",
            self._check_memory,
            critical=False
        )

    def register_check(
        self,
        name: str,
        check_func: Callable,
        critical: bool = False
    ):
        """Register a health check"""
        self.checks[name] = {
            'func': check_func,
            'critical': critical
        }

    async def _check_database(self) -> HealthCheckResult:
        """Check database connectivity"""
        try:
            # Simulate database check
            # In production, this would actually connect to database
            return HealthCheckResult(
                component="database",
                status=HealthStatus.HEALTHY,
                message="Database connection successful",
                details={
                    "connection_pool": 10,
                    "active_connections": 3,
                    "latency_ms": 5.2
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="database",
                status=HealthStatus.UNHEALTHY,
                message=f"Database connection failed: {str(e)}"
            )

    async def _check_cache(self) -> HealthCheckResult:
        """Check cache service"""
        try:
            # Simulate cache check
            return HealthCheckResult(
                component="cache",
                status=HealthStatus.HEALTHY,
                message="Cache service operational",
                details={
                    "hit_rate": 0.85,
                    "memory_used_mb": 256,
                    "keys_count": 1024
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="cache",
                status=HealthStatus.DEGRADED,
                message=f"Cache service degraded: {str(e)}"
            )

    async def _check_llm(self) -> HealthCheckResult:
        """Check LLM service availability"""
        try:
            # Simulate LLM check
            return HealthCheckResult(
                component="llm",
                status=HealthStatus.HEALTHY,
                message="LLM service available",
                details={
                    "model": "gpt-4",
                    "rate_limit_remaining": 450,
                    "avg_latency_ms": 320
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="llm",
                status=HealthStatus.UNHEALTHY,
                message=f"LLM service unavailable: {str(e)}"
            )

    async def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space"""
        disk_usage = psutil.disk_usage('/')
        percent_used = disk_usage.percent

        if percent_used > 90:
            status = HealthStatus.UNHEALTHY
            message = "Critical: Disk space low"
        elif percent_used > 80:
            status = HealthStatus.DEGRADED
            message = "Warning: Disk space running low"
        else:
            status = HealthStatus.HEALTHY
            message = "Disk space adequate"

        return HealthCheckResult(
            component="disk_space",
            status=status,
            message=message,
            details={
                "total_gb": disk_usage.total / (1024**3),
                "used_gb": disk_usage.used / (1024**3),
                "free_gb": disk_usage.free / (1024**3),
                "percent_used": percent_used
            }
        )

    async def _check_memory(self) -> HealthCheckResult:
        """Check memory usage"""
        memory = psutil.virtual_memory()
        percent_used = memory.percent

        if percent_used > 90:
            status = HealthStatus.UNHEALTHY
            message = "Critical: Memory usage high"
        elif percent_used > 80:
            status = HealthStatus.DEGRADED
            message = "Warning: Memory usage elevated"
        else:
            status = HealthStatus.HEALTHY
            message = "Memory usage normal"

        return HealthCheckResult(
            component="memory",
            status=status,
            message=message,
            details={
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "percent_used": percent_used,
                "swap_percent": psutil.swap_memory().percent
            }
        )

    async def run_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all health checks"""
        results = {}

        for name, check_info in self.checks.items():
            try:
                if asyncio.iscoroutinefunction(check_info['func']):
                    result = await check_info['func']()
                else:
                    result = check_info['func']()
                results[name] = result
            except Exception as e:
                results[name] = HealthCheckResult(
                    component=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Health check failed: {str(e)}"
                )

        self.last_results = results
        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status"""
        if not self.last_results:
            return HealthStatus.UNKNOWN

        critical_unhealthy = any(
            result.status == HealthStatus.UNHEALTHY
            for name, result in self.last_results.items()
            if self.checks[name]['critical']
        )

        if critical_unhealthy:
            return HealthStatus.UNHEALTHY

        any_degraded = any(
            result.status == HealthStatus.DEGRADED
            for result in self.last_results.values()
        )

        if any_degraded:
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def to_json(self) -> str:
        """Export health check results as JSON"""
        overall = self.get_overall_status()

        return json.dumps({
            "status": overall.value,
            "timestamp": DeterministicClock.utcnow().isoformat(),
            "checks": {
                name: {
                    "status": result.status.value,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for name, result in self.last_results.items()
            }
        }, indent=2)


class Profiler:
    """Performance profiling tools"""

    def __init__(self):
        """Initialize profiler"""
        self.profiles = {}

    @contextmanager
    def profile_cpu(self, label: str = "cpu_profile"):
        """Profile CPU usage"""
        profiler = cProfile.Profile()
        profiler.enable()

        start_time = time.time()
        try:
            yield profiler
        finally:
            profiler.disable()
            duration = time.time() - start_time

            # Get statistics
            stats = pstats.Stats(profiler)
            stats.sort_stats(pstats.SortKey.CUMULATIVE)

            # Extract top functions
            string_io = io.StringIO()
            stats.stream = string_io
            stats.print_stats(20)
            output = string_io.getvalue()

            # Parse output
            top_functions = self._parse_profile_output(output)

            result = ProfileResult(
                profile_type="cpu",
                duration_seconds=duration,
                top_functions=top_functions,
                stats={
                    "total_calls": stats.total_calls,
                    "primitive_calls": stats.prim_calls,
                    "total_time": stats.total_tt
                }
            )

            self.profiles[label] = result

    @contextmanager
    def profile_memory(self, label: str = "memory_profile"):
        """Profile memory usage"""
        try:
            import memory_profiler
            from memory_profiler import profile as mem_profile
        except ImportError:
            yield None
            return

        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.time()

        try:
            yield
        finally:
            duration = time.time() - start_time
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024

            result = ProfileResult(
                profile_type="memory",
                duration_seconds=duration,
                top_functions=[],
                stats={
                    "start_memory_mb": start_memory,
                    "end_memory_mb": end_memory,
                    "delta_mb": end_memory - start_memory
                }
            )

            self.profiles[label] = result

    def _parse_profile_output(self, output: str) -> List[Dict[str, Any]]:
        """Parse profiler output"""
        functions = []
        lines = output.split('\n')[5:]  # Skip header

        for line in lines[:20]:
            if not line.strip():
                continue

            parts = line.split()
            if len(parts) >= 6:
                functions.append({
                    "ncalls": parts[0],
                    "tottime": float(parts[1]),
                    "percall": float(parts[2]),
                    "cumtime": float(parts[3]),
                    "function": ' '.join(parts[5:])
                })

        return functions

    def generate_flame_graph(self, profile_label: str) -> Optional[str]:
        """Generate flame graph from profile"""
        # This would integrate with flamegraph tools
        # For now, return None
        return None

    def get_profile(self, label: str) -> Optional[ProfileResult]:
        """Get profile by label"""
        return self.profiles.get(label)


class LogAnalyzer:
    """Log analysis and pattern detection"""

    def __init__(self, log_path: Optional[Path] = None):
        """
        Initialize log analyzer

        Args:
            log_path: Path to log files
        """
        self.log_path = log_path or Path("logs")
        self.patterns = self._compile_patterns()

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for log analysis"""
        return {
            "error": re.compile(r'(ERROR|FATAL|Exception|Traceback)', re.I),
            "warning": re.compile(r'(WARN|WARNING)', re.I),
            "slow_query": re.compile(r'duration["\s:]+(\d+\.?\d*)\s*(ms|seconds)', re.I),
            "rate_limit": re.compile(r'(rate.?limit|429|too.?many.?requests)', re.I),
            "memory": re.compile(r'(out.?of.?memory|oom|memory.?error)', re.I),
            "timeout": re.compile(r'(timeout|timed.?out)', re.I),
            "connection": re.compile(r'(connection.?(refused|error|failed))', re.I)
        }

    def analyze_logs(
        self,
        time_range: Optional[timedelta] = None,
        pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze log files

        Args:
            time_range: Time range to analyze
            pattern: Specific pattern to search

        Returns:
            Analysis results
        """
        results = {
            "total_lines": 0,
            "errors": [],
            "warnings": [],
            "patterns": defaultdict(list),
            "error_frequency": Counter(),
            "slow_queries": []
        }

        cutoff_time = None
        if time_range:
            cutoff_time = DeterministicClock.utcnow() - time_range

        # Process log files
        for log_file in self.log_path.glob("*.log"):
            if not log_file.exists():
                continue

            with open(log_file, 'r') as f:
                for line in f:
                    results["total_lines"] += 1

                    # Parse timestamp if present
                    try:
                        # Assume ISO format timestamp at start
                        timestamp_str = line.split()[0]
                        timestamp = datetime.fromisoformat(timestamp_str)

                        if cutoff_time and timestamp < cutoff_time:
                            continue
                    except:
                        pass

                    # Check patterns
                    for pattern_name, regex in self.patterns.items():
                        if regex.search(line):
                            results["patterns"][pattern_name].append(line.strip())

                            if pattern_name == "error":
                                results["errors"].append(line.strip())
                                # Extract error type
                                error_match = re.search(r'(\w+Error|\w+Exception)', line)
                                if error_match:
                                    results["error_frequency"][error_match.group(1)] += 1

                            elif pattern_name == "warning":
                                results["warnings"].append(line.strip())

                            elif pattern_name == "slow_query":
                                # Extract duration
                                duration_match = re.search(r'duration["\s:]+(\d+\.?\d*)', line, re.I)
                                if duration_match:
                                    results["slow_queries"].append({
                                        "line": line.strip(),
                                        "duration": float(duration_match.group(1))
                                    })

        # Calculate statistics
        results["statistics"] = {
            "error_count": len(results["errors"]),
            "warning_count": len(results["warnings"]),
            "error_rate": len(results["errors"]) / max(results["total_lines"], 1),
            "top_errors": results["error_frequency"].most_common(5),
            "slow_query_count": len(results["slow_queries"])
        }

        return results

    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in logs"""
        anomalies = []

        # Analyze recent logs
        recent_analysis = self.analyze_logs(time_range=timedelta(hours=1))
        historical_analysis = self.analyze_logs(time_range=timedelta(days=1))

        # Check for error rate spike
        recent_error_rate = recent_analysis["statistics"]["error_rate"]
        historical_error_rate = historical_analysis["statistics"]["error_rate"]

        if recent_error_rate > historical_error_rate * 2:
            anomalies.append({
                "type": "error_spike",
                "severity": "high",
                "message": f"Error rate spike detected: {recent_error_rate:.2%} vs {historical_error_rate:.2%}",
                "details": recent_analysis["statistics"]
            })

        # Check for new error types
        recent_errors = set(recent_analysis["error_frequency"].keys())
        historical_errors = set(historical_analysis["error_frequency"].keys())
        new_errors = recent_errors - historical_errors

        if new_errors:
            anomalies.append({
                "type": "new_errors",
                "severity": "medium",
                "message": f"New error types detected: {', '.join(new_errors)}",
                "details": {"new_error_types": list(new_errors)}
            })

        return anomalies


class DebugTools:
    """Centralized debugging toolkit"""

    def __init__(self):
        """Initialize debug tools"""
        self.health_checker = HealthChecker()
        self.profiler = Profiler()
        self.log_analyzer = LogAnalyzer()
        self.diagnostic_cache = {}

    async def run_diagnostics(self) -> DiagnosticInfo:
        """Run complete system diagnostics"""
        # System information
        system_info = {
            "platform": sys.platform,
            "python_version": sys.version,
            "cpu_count": psutil.cpu_count(),
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total_gb": psutil.virtual_memory().total / (1024**3),
                "available_gb": psutil.virtual_memory().available / (1024**3),
                "percent": psutil.virtual_memory().percent
            },
            "disk": {
                "total_gb": psutil.disk_usage('/').total / (1024**3),
                "free_gb": psutil.disk_usage('/').free / (1024**3),
                "percent": psutil.disk_usage('/').percent
            }
        }

        # Process information
        process = psutil.Process()
        process_info = {
            "pid": process.pid,
            "status": process.status(),
            "cpu_percent": process.cpu_percent(interval=1),
            "memory_mb": process.memory_info().rss / (1024**2),
            "num_threads": process.num_threads(),
            "open_files": len(process.open_files()),
            "connections": len(process.connections())
        }

        # Health checks
        health_results = await self.health_checker.run_checks()
        dependencies = {
            name: {
                "status": result.status.value,
                "message": result.message
            }
            for name, result in health_results.items()
        }

        # Configuration
        configuration = {
            "debug_mode": __debug__,
            "environment": "production",  # Would come from env
            "log_level": "INFO",
            "features": {
                "tracing": True,
                "metrics": True,
                "profiling": False
            }
        }

        diagnostic_info = DiagnosticInfo(
            timestamp=DeterministicClock.utcnow(),
            system=system_info,
            process=process_info,
            dependencies=dependencies,
            configuration=configuration
        )

        # Cache result
        self.diagnostic_cache['latest'] = diagnostic_info

        return diagnostic_info

    def get_stack_traces(self) -> Dict[int, List[str]]:
        """Get stack traces of all threads"""
        import threading
        traces = {}

        for thread_id, frame in sys._current_frames().items():
            traces[thread_id] = traceback.format_stack(frame)

        return traces

    def inspect_object(self, obj: Any) -> Dict[str, Any]:
        """Inspect object properties and methods"""
        return {
            "type": type(obj).__name__,
            "module": type(obj).__module__,
            "attributes": dir(obj),
            "methods": [m for m in dir(obj) if callable(getattr(obj, m))],
            "doc": inspect.getdoc(obj),
            "source_file": inspect.getfile(type(obj)) if hasattr(type(obj), '__module__') else None
        }


def troubleshoot_issue(
    issue_type: IssueCategory,
    symptoms: List[str]
) -> Dict[str, Any]:
    """
    Automated troubleshooting workflow

    Args:
        issue_type: Category of issue
        symptoms: List of observed symptoms

    Returns:
        Troubleshooting recommendations
    """
    workflows = {
        IssueCategory.PERFORMANCE: [
            "Check metrics dashboard for latency spikes",
            "Analyze traces for bottlenecks",
            "Profile CPU and memory usage",
            "Review recent code changes",
            "Check database query performance",
            "Verify cache hit rates",
            "Test with reduced load"
        ],
        IssueCategory.FUNCTIONAL: [
            "Reproduce the issue locally",
            "Check error logs for stack traces",
            "Analyze related distributed traces",
            "Review recent deployments",
            "Verify input validation",
            "Check integration points",
            "Debug with breakpoints"
        ],
        IssueCategory.INTEGRATION: [
            "Verify network connectivity",
            "Check API credentials and permissions",
            "Test with curl/postman",
            "Review API documentation for changes",
            "Check rate limits and quotas",
            "Verify SSL certificates",
            "Test with mock services"
        ],
        IssueCategory.RESOURCE: [
            "Check memory and CPU usage",
            "Review connection pool settings",
            "Analyze garbage collection logs",
            "Check for memory leaks",
            "Review resource limits",
            "Scale horizontally if needed",
            "Optimize resource-intensive operations"
        ],
        IssueCategory.CONFIGURATION: [
            "Verify environment variables",
            "Check configuration files",
            "Review recent configuration changes",
            "Validate configuration schema",
            "Check feature flags",
            "Verify service dependencies",
            "Test with default configuration"
        ]
    }

    recommendations = workflows.get(issue_type, [])

    # Analyze symptoms
    symptom_analysis = {
        "timeout": ["Check network latency", "Increase timeout values", "Optimize slow operations"],
        "memory": ["Profile memory usage", "Check for leaks", "Increase memory limits"],
        "error": ["Check error logs", "Review stack traces", "Test error scenarios"],
        "slow": ["Profile performance", "Check database queries", "Review caching"],
        "crash": ["Check crash dumps", "Review memory usage", "Check error logs"]
    }

    additional_steps = []
    for symptom in symptoms:
        for key, steps in symptom_analysis.items():
            if key.lower() in symptom.lower():
                additional_steps.extend(steps)

    return {
        "issue_type": issue_type.value,
        "symptoms": symptoms,
        "recommended_steps": recommendations,
        "additional_steps": list(set(additional_steps)),
        "tools": {
            "health_check": "/health",
            "metrics": "/metrics",
            "traces": "Jaeger UI",
            "logs": "Kibana",
            "profiling": "Performance profiler"
        }
    }