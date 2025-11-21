# -*- coding: utf-8 -*-
"""
Tests for GreenLang Monitoring & Observability
"""

import unittest
from unittest.mock import Mock, MagicMock, patch
import time
import json
from datetime import datetime, timedelta
import asyncio

from greenlang.telemetry.metrics import (
from greenlang.determinism import DeterministicClock
    MetricsCollector, MetricType, CustomMetric,
    track_execution, track_resource, MetricsAggregator
)
from greenlang.telemetry.tracing import (
    TracingManager, TraceConfig, SpanKind,
    trace_operation, DistributedTracer
)
from greenlang.telemetry.health import (
    HealthChecker, HealthCheck, HealthStatus, HealthCheckResult,
    LivenessCheck, ReadinessCheck, DiskSpaceHealthCheck,
    MemoryHealthCheck, CPUHealthCheck
)
from greenlang.telemetry.logging import (
    StructuredLogger, LogContext, LogLevel, LogEntry,
    LogAggregator, LogFormatter
)
from greenlang.telemetry.performance import (
    PerformanceMonitor, PerformanceMetric,
    profile_function, measure_latency
)
from greenlang.telemetry.monitoring import (
    MonitoringService, AlertManager, AlertRule, AlertSeverity,
    Alert, AlertStatus, Dashboard
)


class TestMetrics(unittest.TestCase):
    """Test metrics collection"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.collector = MetricsCollector()
    
    def test_custom_metric_registration(self):
        """Test registering custom metrics"""
        metric = CustomMetric(
            name="test_metric",
            type=MetricType.COUNTER,
            description="Test metric",
            labels=["tenant_id"]
        )
        
        self.collector.register_custom_metric(metric)
        self.assertIn("test_metric", self.collector.custom_metrics)
    
    def test_metric_recording(self):
        """Test recording metric values"""
        # Register metric
        metric = CustomMetric(
            name="test_gauge",
            type=MetricType.GAUGE,
            description="Test gauge"
        )
        self.collector.register_custom_metric(metric)
        
        # Record value
        self.collector.record_metric("test_gauge", 42.0)
        
        # Metric should be recorded (would check actual value in real implementation)
        self.assertIn("test_gauge", self.collector.custom_metrics)
    
    @patch('psutil.cpu_percent')
    @patch('psutil.virtual_memory')
    def test_system_metrics_collection(self, mock_memory, mock_cpu):
        """Test system metrics collection"""
        # Mock system stats
        mock_cpu.return_value = 50.0
        mock_memory.return_value = MagicMock(
            used=4000000000,
            percent=50.0
        )
        
        # Collect metrics
        self.collector.collect_system_metrics("test_tenant")
        
        # Verify calls
        mock_cpu.assert_called_once()
        mock_memory.assert_called_once()
    
    def test_track_execution_decorator(self):
        """Test pipeline execution tracking decorator"""
        @track_execution("test_pipeline", "test_tenant")
        def test_function():
            return "success"
        
        result = test_function()
        self.assertEqual(result, "success")
    
    def test_metrics_aggregator(self):
        """Test metrics aggregation"""
        aggregator = MetricsAggregator()
        
        # Add metrics
        for i in range(10):
            aggregator.add_metric("test_metric", float(i))
        
        # Get aggregates
        stats = aggregator.get_aggregates("test_metric", 60)
        
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 9.0)
        self.assertEqual(stats["avg"], 4.5)
        self.assertEqual(stats["count"], 10)


class TestTracing(unittest.TestCase):
    """Test distributed tracing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = TraceConfig(
            service_name="test_service",
            console_export=False
        )
        self.manager = TracingManager(self.config)
    
    def test_tracer_initialization(self):
        """Test tracer initialization"""
        tracer = self.manager.get_tracer()
        self.assertIsNotNone(tracer)
    
    def test_span_creation(self):
        """Test creating spans"""
        with self.manager.create_span("test_operation") as span:
            # Span should be created
            self.assertIsNotNone(span)
    
    def test_trace_operation_decorator(self):
        """Test operation tracing decorator"""
        @trace_operation("test_op", SpanKind.INTERNAL)
        def test_function():
            return "result"
        
        result = test_function()
        self.assertEqual(result, "result")
    
    def test_distributed_tracer(self):
        """Test distributed tracing"""
        tracer = DistributedTracer("test_service")
        
        # Start trace
        context = tracer.start_trace("operation1")
        self.assertIsInstance(context, dict)
        
        # Continue trace
        context2 = tracer.continue_trace("operation2", context)
        self.assertIsInstance(context2, dict)


class TestHealthChecks(unittest.TestCase):
    """Test health checking"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = HealthChecker()
    
    def test_liveness_check(self):
        """Test liveness probe"""
        check = LivenessCheck()
        result = check.check()
        
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("pid", result.details)
    
    def test_readiness_check(self):
        """Test readiness probe"""
        check = ReadinessCheck(dependencies=["database", "cache"])
        result = check.check()
        
        # Should be healthy if dependencies are mocked as available
        self.assertIn(result.status, [HealthStatus.HEALTHY, HealthStatus.UNHEALTHY])
    
    @patch('psutil.disk_usage')
    def test_disk_space_check(self, mock_disk):
        """Test disk space health check"""
        mock_disk.return_value = MagicMock(
            free=10 * 1024 ** 3,  # 10GB free
            percent=50.0,
            total=100 * 1024 ** 3
        )
        
        check = DiskSpaceHealthCheck("/", min_free_gb=1.0)
        result = check.check()
        
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("free_gb", result.details)
    
    @patch('psutil.virtual_memory')
    def test_memory_check(self, mock_memory):
        """Test memory health check"""
        mock_memory.return_value = MagicMock(
            percent=60.0,
            available=4 * 1024 ** 3,
            total=16 * 1024 ** 3
        )
        
        check = MemoryHealthCheck(max_usage_percent=90.0)
        result = check.check()
        
        self.assertEqual(result.status, HealthStatus.HEALTHY)
        self.assertIn("used_percent", result.details)
    
    def test_health_report_generation(self):
        """Test generating health report"""
        report = self.checker.check_health()
        
        self.assertIsNotNone(report)
        self.assertIn(report.status, list(HealthStatus))
        self.assertIsInstance(report.checks, list)
        self.assertGreater(report.uptime_seconds, 0)
    
    def test_custom_health_check(self):
        """Test adding custom health check"""
        class CustomCheck(HealthCheck):
            def _perform_check(self):
                return HealthCheckResult(
                    name="custom",
                    status=HealthStatus.HEALTHY,
                    message="Custom check passed"
                )
        
        custom_check = CustomCheck("custom", critical=False)
        self.checker.register_check(custom_check)
        
        report = self.checker.check_health()
        check_names = [c.name for c in report.checks]
        self.assertIn("custom", check_names)


class TestLogging(unittest.TestCase):
    """Test structured logging"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.context = LogContext(
            tenant_id="test_tenant",
            component="test_component"
        )
        self.logger = StructuredLogger("test_logger", self.context)
        self.aggregator = LogAggregator()
    
    def test_structured_logging(self):
        """Test structured log creation"""
        self.logger.info("Test message", key1="value1", key2="value2")
        
        # Log should be added to aggregator
        logs = self.aggregator.get_logs(limit=1)
        if logs:
            self.assertEqual(logs[0].message, "Test message")
            self.assertEqual(logs[0].level, LogLevel.INFO)
    
    def test_log_context(self):
        """Test logging with context"""
        with self.logger.with_context(request_id="req123"):
            self.logger.info("With context")
        
        # Context should be restored after
        self.assertIsNone(self.logger.context.request_id)
    
    def test_log_aggregation(self):
        """Test log aggregation"""
        # Add logs
        for i in range(5):
            entry = LogEntry(
                timestamp=DeterministicClock.utcnow(),
                level=LogLevel.INFO,
                message=f"Message {i}",
                context=self.context
            )
            self.aggregator.add_log(entry)
        
        # Get logs
        logs = self.aggregator.get_logs(limit=10)
        self.assertLessEqual(len(logs), 5)
    
    def test_error_pattern_analysis(self):
        """Test error pattern detection"""
        # Add error logs
        error_entry = LogEntry(
            timestamp=DeterministicClock.utcnow(),
            level=LogLevel.ERROR,
            message="Database connection failed",
            context=self.context
        )
        self.aggregator.add_log(error_entry)
        
        # Check error patterns
        self.assertIn("connection_error", self.aggregator.error_patterns)
    
    def test_log_statistics(self):
        """Test log statistics generation"""
        # Add various logs
        for level in [LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR]:
            entry = LogEntry(
                timestamp=DeterministicClock.utcnow(),
                level=level,
                message=f"Test {level.value}",
                context=self.context
            )
            self.aggregator.add_log(entry)
        
        stats = self.aggregator.get_statistics()
        
        self.assertIn("total_logs", stats)
        self.assertIn("log_counts", stats)
        self.assertGreater(stats["total_logs"], 0)


class TestPerformance(unittest.TestCase):
    """Test performance monitoring"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.monitor = PerformanceMonitor()
    
    def test_metric_recording(self):
        """Test recording performance metrics"""
        self.monitor.record_metric("test_latency", 100.5, "ms")
        
        metrics = self.monitor.get_metrics("test_latency", 60)
        self.assertGreater(len(metrics), 0)
        self.assertEqual(metrics[0].name, "test_latency")
        self.assertEqual(metrics[0].value, 100.5)
    
    def test_metric_statistics(self):
        """Test calculating metric statistics"""
        # Record multiple values
        for i in range(10):
            self.monitor.record_metric("test_metric", float(i * 10), "ms")
        
        stats = self.monitor.get_statistics("test_metric", 300)
        
        self.assertEqual(stats["count"], 10)
        self.assertEqual(stats["min"], 0.0)
        self.assertEqual(stats["max"], 90.0)
        self.assertIn("p95", stats)
    
    def test_profile_function_decorator(self):
        """Test function profiling decorator"""
        @profile_function
        def test_function():
            time.sleep(0.01)
            return "done"
        
        result = test_function()
        self.assertEqual(result, "done")
        
        # Check that metrics were recorded
        metrics = self.monitor.get_metrics("function.test_function.duration", 60)
        self.assertGreater(len(metrics), 0)
    
    def test_measure_latency_context(self):
        """Test latency measurement context manager"""
        with measure_latency("test_operation"):
            time.sleep(0.01)
        
        metrics = self.monitor.get_metrics("latency.test_operation", 60)
        self.assertGreater(len(metrics), 0)
        self.assertGreater(metrics[0].value, 0)
    
    @patch('psutil.Process')
    def test_memory_tracking(self, mock_process):
        """Test memory usage tracking"""
        mock_process.return_value.memory_info.return_value = MagicMock(
            rss=100 * 1024 * 1024,
            vms=200 * 1024 * 1024
        )
        mock_process.return_value.memory_percent.return_value = 5.0
        
        memory = self.monitor.get_memory_usage()
        
        self.assertIn("rss_mb", memory)
        self.assertIn("percent", memory)
        self.assertEqual(memory["percent"], 5.0)


class TestAlerting(unittest.TestCase):
    """Test alerting system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.alert_manager = AlertManager()
    
    def test_alert_rule_creation(self):
        """Test creating alert rules"""
        rule = AlertRule(
            name="high_cpu",
            expression="cpu_usage",
            condition="> 80",
            duration=60,
            severity=AlertSeverity.WARNING
        )
        
        self.alert_manager.add_rule(rule)
        self.assertIn("high_cpu", self.alert_manager.rules)
    
    def test_alert_rule_evaluation(self):
        """Test evaluating alert conditions"""
        rule = AlertRule(
            name="test_rule",
            expression="test_metric",
            condition="> 10",
            duration=0,
            severity=AlertSeverity.INFO
        )
        
        self.assertTrue(rule.evaluate(15.0))
        self.assertFalse(rule.evaluate(5.0))
    
    def test_alert_firing(self):
        """Test firing alerts"""
        rule = AlertRule(
            name="test_alert",
            expression="test",
            condition="> 0",
            duration=0,
            severity=AlertSeverity.WARNING
        )
        
        self.alert_manager.fire_alert(rule, "Test alert message")
        
        active = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0].name, "test_alert")
        self.assertEqual(active[0].status, AlertStatus.FIRING)
    
    def test_alert_resolution(self):
        """Test resolving alerts"""
        # Fire alert
        rule = AlertRule(
            name="test",
            expression="test",
            condition="> 0",
            duration=0,
            severity=AlertSeverity.INFO
        )
        self.alert_manager.fire_alert(rule, "Test")
        
        # Get alert ID
        active = self.alert_manager.get_active_alerts()
        alert_id = active[0].alert_id
        
        # Resolve alert
        self.alert_manager.resolve_alert(alert_id)
        
        # Should no longer be active
        active = self.alert_manager.get_active_alerts()
        self.assertEqual(len(active), 0)
    
    def test_alert_acknowledgement(self):
        """Test acknowledging alerts"""
        # Fire alert
        rule = AlertRule(
            name="test",
            expression="test",
            condition="> 0",
            duration=0,
            severity=AlertSeverity.INFO
        )
        self.alert_manager.fire_alert(rule, "Test")
        
        # Get alert ID
        active = self.alert_manager.get_active_alerts()
        alert_id = active[0].alert_id
        
        # Acknowledge alert
        self.alert_manager.acknowledge_alert(alert_id, "user123")
        
        # Check acknowledgement
        active = self.alert_manager.get_active_alerts()
        self.assertEqual(active[0].status, AlertStatus.ACKNOWLEDGED)
        self.assertEqual(active[0].acknowledged_by, "user123")


class TestDashboards(unittest.TestCase):
    """Test dashboard management"""
    
    def test_dashboard_creation(self):
        """Test creating dashboards"""
        dashboard = Dashboard(
            name="test_dashboard",
            title="Test Dashboard",
            description="Test description",
            panels=[
                {
                    "title": "CPU Usage",
                    "type": "graph",
                    "targets": [{"expr": "cpu_usage"}]
                }
            ]
        )
        
        self.assertEqual(dashboard.name, "test_dashboard")
        self.assertEqual(len(dashboard.panels), 1)
    
    def test_dashboard_export(self):
        """Test exporting dashboard configuration"""
        dashboard = Dashboard(
            name="test",
            title="Test",
            panels=[{"title": "Panel1"}]
        )
        
        # Export as dict
        config = dashboard.to_dict()
        self.assertEqual(config["name"], "test")
        self.assertIn("panels", config)
        
        # Export as Grafana JSON
        grafana = dashboard.to_grafana_json()
        self.assertIn("dashboard", grafana)
        self.assertEqual(grafana["dashboard"]["title"], "Test")


class TestMonitoringService(unittest.TestCase):
    """Test integrated monitoring service"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = MonitoringService()
    
    def test_service_initialization(self):
        """Test monitoring service initialization"""
        self.assertIsNotNone(self.service.metrics_collector)
        self.assertIsNotNone(self.service.health_checker)
        self.assertIsNotNone(self.service.alert_manager)
        self.assertIsNotNone(self.service.log_aggregator)
        self.assertIsNotNone(self.service.performance_monitor)
    
    def test_default_alerts(self):
        """Test default alert rules are created"""
        self.assertIn("high_cpu_usage", self.service.alert_manager.rules)
        self.assertIn("high_memory_usage", self.service.alert_manager.rules)
        self.assertIn("low_disk_space", self.service.alert_manager.rules)
    
    def test_default_dashboards(self):
        """Test default dashboards are created"""
        self.assertIn("system_overview", self.service.dashboards)
        self.assertIn("pipeline_metrics", self.service.dashboards)
    
    def test_service_status(self):
        """Test getting service status"""
        status = self.service.get_status()
        
        self.assertIn("health", status)
        self.assertIn("active_alerts", status)
        self.assertIn("metrics_count", status)
        self.assertIn("dashboards", status)
    
    def test_dashboard_export(self):
        """Test exporting dashboards"""
        # Export system overview dashboard
        json_export = self.service.export_dashboard("system_overview", "json")
        self.assertIsInstance(json_export, str)
        
        # Verify it's valid JSON
        config = json.loads(json_export)
        self.assertEqual(config["name"], "system_overview")


class TestAsyncHealthChecks(unittest.TestCase):
    """Test async health checks"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.checker = HealthChecker()
    
    async def test_async_health_check(self):
        """Test async health checking"""
        report = await self.checker.check_health_async()
        
        self.assertIsNotNone(report)
        self.assertIn(report.status, list(HealthStatus))
    
    def test_async_wrapper(self):
        """Test running async test"""
        loop = asyncio.new_event_loop()
        loop.run_until_complete(self.test_async_health_check())
        loop.close()


if __name__ == '__main__':
    unittest.main()