# -*- coding: utf-8 -*-
"""
Metrics Collector Agent
========================

Background agent for collecting IUM, performance, and cost metrics.
Runs every 5 minutes via cron.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import re
from prometheus_client import CollectorRegistry, Gauge, Counter, Histogram, push_to_gateway
from prometheus_client.exposition import basic_auth_handler

logger = logging.getLogger(__name__)


class MetricsCollector:
    """
    Collects and pushes metrics to Prometheus Pushgateway.
    """

    def __init__(self, pushgateway_url: str = "localhost:9091"):
        self.pushgateway_url = pushgateway_url
        self.registry = CollectorRegistry()
        self._setup_metrics()

    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        # IUM Metrics
        self.ium_score = Gauge(
            'greenlang_ium_score',
            'Infrastructure Usage Metric score',
            ['application', 'team'],
            registry=self.registry
        )

        self.file_ium_score = Gauge(
            'greenlang_file_ium_score',
            'IUM score per file',
            ['application', 'team', 'file_path', 'infrastructure_lines', 'custom_lines'],
            registry=self.registry
        )

        self.component_usage = Gauge(
            'greenlang_component_usage',
            'Infrastructure component usage count',
            ['component', 'application'],
            registry=self.registry
        )

        self.custom_code_lines = Gauge(
            'greenlang_custom_code_lines',
            'Lines of custom code',
            ['application', 'module'],
            registry=self.registry
        )

        # Cost Metrics
        self.cost_savings_usd = Counter(
            'greenlang_cost_savings_usd',
            'Total cost savings in USD',
            ['application', 'optimization_type', 'service'],
            registry=self.registry
        )

        self.llm_cost_savings = Counter(
            'greenlang_llm_cost_savings_usd',
            'LLM cost savings from caching',
            ['application', 'provider'],
            registry=self.registry
        )

        self.cache_hit_rate = Gauge(
            'greenlang_cache_hit_rate',
            'Cache hit rate',
            ['service', 'cache_type'],
            registry=self.registry
        )

        self.developer_hours_saved = Counter(
            'greenlang_developer_hours_saved',
            'Developer hours saved',
            ['application', 'feature'],
            registry=self.registry
        )

        # Performance Metrics
        self.request_duration = Histogram(
            'greenlang_request_duration_seconds',
            'Request duration in seconds',
            ['service', 'endpoint'],
            registry=self.registry
        )

        self.agent_execution_seconds = Histogram(
            'greenlang_agent_execution_seconds',
            'Agent execution time',
            ['agent', 'application'],
            registry=self.registry,
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
        )

        self.cache_latency = Gauge(
            'greenlang_cache_latency_seconds',
            'Cache latency by layer',
            ['layer', 'service'],
            registry=self.registry
        )

        self.request_total = Counter(
            'greenlang_request_total',
            'Total requests',
            ['service'],
            registry=self.registry
        )

        self.request_errors_total = Counter(
            'greenlang_request_errors_total',
            'Total request errors',
            ['service', 'error_type'],
            registry=self.registry
        )

    async def collect_ium_metrics(self, codebase_path: str) -> None:
        """
        Collect IUM metrics from codebase analysis.

        Args:
            codebase_path: Path to codebase root
        """
        logger.info("Collecting IUM metrics...")

        try:
            # Scan codebase for infrastructure usage
            applications = self._discover_applications(codebase_path)

            for app in applications:
                ium_data = await self._calculate_ium(app)

                # Set application-level IUM
                self.ium_score.labels(
                    application=ium_data['name'],
                    team=ium_data['team']
                ).set(ium_data['ium_score'])

                # Set file-level IUM
                for file_info in ium_data['files']:
                    self.file_ium_score.labels(
                        application=ium_data['name'],
                        team=ium_data['team'],
                        file_path=file_info['path'],
                        infrastructure_lines=str(file_info['infra_lines']),
                        custom_lines=str(file_info['custom_lines'])
                    ).set(file_info['ium'])

                # Component usage
                for component, count in ium_data['components'].items():
                    self.component_usage.labels(
                        component=component,
                        application=ium_data['name']
                    ).set(count)

            logger.info(f"Collected IUM metrics for {len(applications)} applications")
        except Exception as e:
            logger.error(f"Error collecting IUM metrics: {e}")

    async def collect_cost_metrics(self) -> None:
        """Collect cost savings metrics"""
        logger.info("Collecting cost metrics...")

        try:
            # Query cache statistics
            cache_stats = await self._get_cache_stats()

            for service, stats in cache_stats.items():
                # Calculate LLM cost savings
                cache_savings = stats['cache_hits'] * stats['avg_cost_per_request']
                self.llm_cost_savings.labels(
                    application=service,
                    provider=stats['provider']
                ).inc(cache_savings)

                # Cache hit rate
                hit_rate = stats['cache_hits'] / max(stats['total_requests'], 1)
                self.cache_hit_rate.labels(
                    service=service,
                    cache_type='semantic'
                ).set(hit_rate)

            logger.info("Cost metrics collected successfully")
        except Exception as e:
            logger.error(f"Error collecting cost metrics: {e}")

    async def collect_performance_metrics(self) -> None:
        """Collect performance metrics from application logs"""
        logger.info("Collecting performance metrics...")

        try:
            # Query application logs for performance data
            perf_data = await self._query_performance_logs()

            for service, metrics in perf_data.items():
                # Request count
                self.request_total.labels(service=service).inc(metrics['request_count'])

                # Error count
                for error_type, count in metrics['errors'].items():
                    self.request_errors_total.labels(
                        service=service,
                        error_type=error_type
                    ).inc(count)

                # Cache latency by layer
                for layer, latency in metrics['cache_latency'].items():
                    self.cache_latency.labels(
                        layer=layer,
                        service=service
                    ).set(latency)

            logger.info("Performance metrics collected successfully")
        except Exception as e:
            logger.error(f"Error collecting performance metrics: {e}")

    def _discover_applications(self, codebase_path: str) -> List[Dict[str, Any]]:
        """Discover applications in codebase"""
        # In production, scan actual directory structure
        return [
            {
                'name': 'csrd-reporting',
                'team': 'csrd',
                'path': f'{codebase_path}/GL-CSRD-APP'
            },
            {
                'name': 'vcci-scope3',
                'team': 'carbon',
                'path': f'{codebase_path}/GL-VCCI-Carbon-APP'
            }
        ]

    async def _calculate_ium(self, app: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate IUM for an application.

        Args:
            app: Application info

        Returns:
            IUM calculation results
        """
        # Simulate IUM calculation
        # In production, parse actual code files
        return {
            'name': app['name'],
            'team': app['team'],
            'ium_score': 96.5,
            'files': [
                {
                    'path': 'src/agents/calculator.py',
                    'infra_lines': 180,
                    'custom_lines': 20,
                    'ium': 90.0
                },
                {
                    'path': 'src/services/broker.py',
                    'infra_lines': 250,
                    'custom_lines': 5,
                    'ium': 98.0
                }
            ],
            'components': {
                'LLMClient': 15,
                'SemanticCache': 8,
                'FactorBroker': 12,
                'EntityMDM': 10,
                'FormBuilder': 6
            }
        }

    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from Redis"""
        # In production, query actual Redis instance
        return {
            'csrd-reporting': {
                'cache_hits': 8520,
                'total_requests': 10000,
                'avg_cost_per_request': 0.002,
                'provider': 'openai'
            },
            'vcci-scope3': {
                'cache_hits': 6800,
                'total_requests': 8500,
                'avg_cost_per_request': 0.0018,
                'provider': 'anthropic'
            }
        }

    async def _query_performance_logs(self) -> Dict[str, Any]:
        """Query performance metrics from logs"""
        # In production, query actual log aggregation system
        return {
            'factor-broker': {
                'request_count': 15420,
                'errors': {
                    'timeout': 12,
                    'validation': 8,
                    'internal': 3
                },
                'cache_latency': {
                    'L1': 0.002,
                    'L2': 0.015,
                    'L3': 0.045
                }
            },
            'entity-mdm': {
                'request_count': 12850,
                'errors': {
                    'not_found': 45,
                    'validation': 12
                },
                'cache_latency': {
                    'L1': 0.001,
                    'L2': 0.012,
                    'L3': 0.038
                }
            }
        }

    async def collect_all_metrics(self, codebase_path: str) -> None:
        """Collect all metrics"""
        logger.info("Starting metrics collection cycle...")

        await asyncio.gather(
            self.collect_ium_metrics(codebase_path),
            self.collect_cost_metrics(),
            self.collect_performance_metrics()
        )

        logger.info("Metrics collection cycle completed")

    def push_metrics(self, job_name: str = "greenlang_metrics") -> None:
        """
        Push collected metrics to Prometheus Pushgateway.

        Args:
            job_name: Job name for metrics grouping
        """
        try:
            push_to_gateway(
                self.pushgateway_url,
                job=job_name,
                registry=self.registry
            )
            logger.info(f"Metrics pushed to Pushgateway: {self.pushgateway_url}")
        except Exception as e:
            logger.error(f"Failed to push metrics: {e}")


async def main():
    """Main entry point for metrics collector"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    collector = MetricsCollector(pushgateway_url="localhost:9091")

    # Collect metrics
    codebase_path = "C:\\Users\\aksha\\Code-V1_GreenLang"
    await collector.collect_all_metrics(codebase_path)

    # Push to Prometheus
    collector.push_metrics()

    logger.info("Metrics collection completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
