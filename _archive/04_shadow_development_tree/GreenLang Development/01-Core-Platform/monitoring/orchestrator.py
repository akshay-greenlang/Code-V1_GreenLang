# -*- coding: utf-8 -*-
"""
Monitoring System Orchestrator
===============================

Main orchestrator for the GreenLang monitoring system.
Coordinates all collectors, dashboards, and reporting.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
import yaml
import sys
import os
from greenlang.utilities.determinism import DeterministicClock

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from collectors.metrics_collector import MetricsCollector
from collectors.log_aggregator import LogAggregator
from collectors.violation_scanner import ViolationScanner
from collectors.health_checker import HealthChecker
from reports.report_generator import ReportGenerator
from alerts.alert_rules import AlertRuleEngine

# Configure logging with platform-agnostic path
LOG_DIR = Path(os.getenv("GREENLANG_LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = LOG_DIR / "monitoring.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class MonitoringOrchestrator:
    """
    Orchestrates all monitoring components.
    """

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.metrics_collector = None
        self.log_aggregator = None
        self.violation_scanner = None
        self.health_checker = None
        self.report_generator = None
        self.alert_engine = None

    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def initialize_components(self):
        """Initialize all monitoring components"""
        logger.info("Initializing monitoring components...")

        # Metrics Collector
        if self.config['collectors']['metrics_collector']['enabled']:
            pushgateway_url = self.config['prometheus']['pushgateway_url']
            self.metrics_collector = MetricsCollector(pushgateway_url)
            logger.info("Metrics collector initialized")

        # Log Aggregator
        if self.config['collectors']['log_aggregator']['enabled']:
            log_sources = self.config['collectors']['log_aggregator']['log_sources']
            self.log_aggregator = LogAggregator(log_sources)
            logger.info("Log aggregator initialized")

        # Violation Scanner
        if self.config['collectors']['violation_scanner']['enabled']:
            codebase_path = self.config['collectors']['violation_scanner']['codebase_path']
            self.violation_scanner = ViolationScanner(codebase_path)
            logger.info("Violation scanner initialized")

        # Health Checker
        if self.config['collectors']['health_checker']['enabled']:
            self.health_checker = HealthChecker()
            logger.info("Health checker initialized")

        # Report Generator
        output_dir = self.config['reporting']['output_dir']
        self.report_generator = ReportGenerator(output_dir)
        logger.info("Report generator initialized")

        # Alert Engine
        self.alert_engine = AlertRuleEngine()
        logger.info("Alert engine initialized")

        logger.info("All components initialized successfully")

    async def run_metrics_collection(self):
        """Run metrics collection cycle"""
        if not self.metrics_collector:
            return

        logger.info("Running metrics collection cycle...")
        try:
            codebase_path = self.config['collectors']['metrics_collector']['codebase_path']
            await self.metrics_collector.collect_all_metrics(codebase_path)
            self.metrics_collector.push_metrics()
            logger.info("Metrics collection completed successfully")
        except Exception as e:
            logger.error(f"Error in metrics collection: {e}")

    async def run_log_aggregation(self):
        """Run log aggregation cycle"""
        if not self.log_aggregator:
            return

        logger.info("Running log aggregation cycle...")
        try:
            await self.log_aggregator.aggregate_logs()

            # Export to Elasticsearch if configured
            es_config = self.config['collectors']['log_aggregator'].get('elasticsearch')
            if es_config:
                await self.log_aggregator.export_to_elasticsearch(
                    es_config['url'],
                    es_config['index']
                )

            logger.info("Log aggregation completed successfully")
        except Exception as e:
            logger.error(f"Error in log aggregation: {e}")

    async def run_violation_scan(self):
        """Run violation scanning cycle"""
        if not self.violation_scanner:
            return

        logger.info("Running violation scan cycle...")
        try:
            violations = await self.violation_scanner.scan_codebase()

            # Generate and save report
            report = self.violation_scanner.generate_report()
            report_path = Path(self.config['reporting']['output_dir']) / f"violations_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}.txt"
            report_path.write_text(report)

            logger.info(f"Violation scan completed. Found {len(violations)} violations")

            # Alert on critical violations
            critical_violations = [v for v in violations if v.severity == 'critical']
            if critical_violations:
                logger.warning(f"Found {len(critical_violations)} critical violations!")

        except Exception as e:
            logger.error(f"Error in violation scanning: {e}")

    async def run_health_checks(self):
        """Run health check cycle"""
        if not self.health_checker:
            return

        logger.info("Running health check cycle...")
        try:
            await self.health_checker.check_all_services()

            # Export to Prometheus
            pushgateway_url = self.config['prometheus']['pushgateway_url']
            await self.health_checker.export_to_prometheus(pushgateway_url)

            # Check for unhealthy services
            overall_status = self.health_checker.get_overall_status()
            if overall_status.value != "healthy":
                logger.warning(f"Infrastructure health status: {overall_status.value}")

            logger.info("Health checks completed successfully")
        except Exception as e:
            logger.error(f"Error in health checks: {e}")

    async def generate_weekly_report(self):
        """Generate weekly summary report"""
        logger.info("Generating weekly summary report...")
        try:
            # Gather data from various sources
            data = {
                'period_start': DeterministicClock.now() - asyncio.timedelta(days=7),
                'period_end': DeterministicClock.now(),
                'ium': 96.5,
                'ium_trend': 'Increasing',
                'cost_savings': 12500,
                'llm_savings': 8900,
                'dev_savings': 3600,
                'p95_latency': 245,
                'error_rate': 0.15,
                'highlights': [
                    "Achieved highest IUM score this quarter (96.5%)",
                    "Successfully onboarded 3 new developers",
                    "Deployed 15 new infrastructure features"
                ],
                'top_developers': [
                    {'name': 'akshay', 'team': 'platform', 'contributions': 156, 'ium': 99.5},
                    {'name': 'dev2', 'team': 'csrd', 'contributions': 98, 'ium': 97.2}
                ],
                'action_items': []
            }

            html = self.report_generator.generate_weekly_summary(data)

            # Send email (in production)
            # await self._send_email_report(html, self.config['reporting']['weekly_summary']['recipients'])

            logger.info("Weekly report generated successfully")
        except Exception as e:
            logger.error(f"Error generating weekly report: {e}")

    async def run_monitoring_cycle(self):
        """Run a complete monitoring cycle"""
        logger.info("=" * 80)
        logger.info("Starting monitoring cycle...")
        logger.info("=" * 80)

        # Run all collectors in parallel
        await asyncio.gather(
            self.run_metrics_collection(),
            self.run_log_aggregation(),
            self.run_health_checks(),
            return_exceptions=True
        )

        logger.info("Monitoring cycle completed")

    async def run_continuous(self):
        """Run monitoring continuously with configured intervals"""
        logger.info("Starting continuous monitoring...")

        while True:
            try:
                await self.run_monitoring_cycle()

                # Wait for next cycle (5 minutes by default)
                interval = self.config['collectors']['metrics_collector'].get('interval', '5m')
                wait_seconds = self._parse_interval(interval)
                logger.info(f"Waiting {wait_seconds} seconds until next cycle...")
                await asyncio.sleep(wait_seconds)

            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying

    def _parse_interval(self, interval: str) -> int:
        """Parse interval string (e.g., '5m', '1h') to seconds"""
        if interval.endswith('s'):
            return int(interval[:-1])
        elif interval.endswith('m'):
            return int(interval[:-1]) * 60
        elif interval.endswith('h'):
            return int(interval[:-1]) * 3600
        else:
            return 300  # Default 5 minutes

    def deploy_dashboards(self):
        """Deploy all Grafana dashboards"""
        logger.info("Deploying Grafana dashboards...")

        from dashboards.infrastructure_usage import InfrastructureUsageDashboard
        from dashboards.cost_savings import CostSavingsDashboard
        from dashboards.performance import PerformanceDashboard
        from dashboards.compliance import ComplianceDashboard
        from dashboards.productivity import ProductivityDashboard
        from dashboards.health import HealthDashboard

        grafana_url = self.config['grafana']['url']
        api_key = self.config['grafana']['api_key']

        dashboards = [
            InfrastructureUsageDashboard(),
            CostSavingsDashboard(),
            PerformanceDashboard(),
            ComplianceDashboard(),
            ProductivityDashboard(),
            HealthDashboard()
        ]

        for dashboard in dashboards:
            try:
                # Export to JSON (platform-agnostic path)
                dashboards_dir = Path(os.getenv("GREENLANG_MONITORING_DIR", Path(__file__).parent)) / "dashboards"
                dashboards_dir.mkdir(parents=True, exist_ok=True)
                output_path = dashboards_dir / f"{dashboard.dashboard_uid}.json"
                dashboard.export_to_file(str(output_path))
                logger.info(f"Dashboard exported: {dashboard.dashboard_title}")

                # Deploy to Grafana (if API key is configured)
                if api_key and api_key != "${GRAFANA_API_KEY}":
                    dashboard.deploy_to_grafana(grafana_url, api_key)
                    logger.info(f"Dashboard deployed: {dashboard.dashboard_title}")
            except Exception as e:
                logger.error(f"Error deploying dashboard {dashboard.dashboard_title}: {e}")

        logger.info("Dashboard deployment completed")

    def deploy_alert_rules(self):
        """Deploy alert rules to Prometheus"""
        logger.info("Deploying alert rules...")

        try:
            # Export Prometheus rules (platform-agnostic paths)
            alerts_dir = Path(os.getenv("GREENLANG_MONITORING_DIR", Path(__file__).parent)) / "alerts"
            alerts_dir.mkdir(parents=True, exist_ok=True)

            prometheus_path = alerts_dir / "prometheus_rules.json"
            self.alert_engine.export_prometheus_rules(str(prometheus_path))

            # Export Grafana alerts
            grafana_path = alerts_dir / "grafana_alerts.json"
            self.alert_engine.export_grafana_alerts(str(grafana_path))

            logger.info(f"Alert rules deployed: {len(self.alert_engine.rules)} rules")
        except Exception as e:
            logger.error(f"Error deploying alert rules: {e}")


async def main():
    """Main entry point"""
    # Platform-agnostic config path
    config_path = Path(os.getenv("GREENLANG_CONFIG_PATH",
                                  Path(__file__).parent / "config.yaml"))

    orchestrator = MonitoringOrchestrator(str(config_path))
    orchestrator.initialize_components()

    # Deploy dashboards and alerts
    orchestrator.deploy_dashboards()
    orchestrator.deploy_alert_rules()

    # Run monitoring cycle once for testing
    await orchestrator.run_monitoring_cycle()

    # Generate sample report
    await orchestrator.generate_weekly_report()

    # For continuous monitoring, uncomment:
    # await orchestrator.run_continuous()


if __name__ == "__main__":
    asyncio.run(main())
