# -*- coding: utf-8 -*-
"""
Log Aggregator Agent
====================

Aggregate and process structured logs from all services.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict
import re
from greenlang.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class LogAggregator:
    """
    Aggregates logs from multiple sources and extracts metrics.
    """

    def __init__(self, log_sources: List[str]):
        self.log_sources = log_sources
        self.aggregated_data = defaultdict(list)

    async def aggregate_logs(self) -> None:
        """Aggregate logs from all configured sources"""
        logger.info("Starting log aggregation...")

        tasks = [self._process_log_source(source) for source in self.log_sources]
        await asyncio.gather(*tasks)

        logger.info(f"Aggregated logs from {len(self.log_sources)} sources")

    async def _process_log_source(self, source: str) -> None:
        """Process logs from a single source"""
        try:
            logs = await self._read_logs(source)
            parsed_logs = [self._parse_log_entry(log) for log in logs]

            for log_entry in parsed_logs:
                if log_entry:
                    category = log_entry.get('category', 'general')
                    self.aggregated_data[category].append(log_entry)

            logger.info(f"Processed {len(logs)} log entries from {source}")
        except Exception as e:
            logger.error(f"Error processing log source {source}: {e}")

    async def _read_logs(self, source: str) -> List[str]:
        """Read logs from source (file, API, stream)"""
        # In production: read from actual log files, Elasticsearch, CloudWatch, etc.
        sample_logs = [
            '{"timestamp": "2025-11-09T10:00:00Z", "level": "INFO", "service": "factor-broker", "message": "Request processed", "duration_ms": 45.2, "category": "performance"}',
            '{"timestamp": "2025-11-09T10:00:05Z", "level": "ERROR", "service": "entity-mdm", "message": "Database timeout", "error_type": "timeout", "category": "error"}',
            '{"timestamp": "2025-11-09T10:00:10Z", "level": "INFO", "service": "csrd-app", "message": "Cache hit", "cache_layer": "L2", "category": "cache"}',
        ]
        return sample_logs

    def _parse_log_entry(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Parse structured log entry"""
        try:
            return json.loads(log_line)
        except json.JSONDecodeError:
            # Try to parse unstructured logs
            return self._parse_unstructured_log(log_line)

    def _parse_unstructured_log(self, log_line: str) -> Optional[Dict[str, Any]]:
        """Parse unstructured log format"""
        # Example: "2025-11-09 10:00:00 [INFO] service-name: message"
        pattern = r'(\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2})\s\[(\w+)\]\s(\S+):\s(.+)'
        match = re.match(pattern, log_line)

        if match:
            timestamp, level, service, message = match.groups()
            return {
                'timestamp': timestamp,
                'level': level,
                'service': service,
                'message': message,
                'category': 'general'
            }
        return None

    def extract_performance_metrics(self) -> Dict[str, Any]:
        """Extract performance metrics from logs"""
        perf_logs = self.aggregated_data.get('performance', [])

        metrics = {
            'request_count': len(perf_logs),
            'avg_duration_ms': 0,
            'p95_duration_ms': 0,
            'by_service': defaultdict(list)
        }

        for log in perf_logs:
            service = log.get('service', 'unknown')
            duration = log.get('duration_ms', 0)
            metrics['by_service'][service].append(duration)

        # Calculate aggregates
        all_durations = [
            d for durations in metrics['by_service'].values()
            for d in durations
        ]

        if all_durations:
            metrics['avg_duration_ms'] = sum(all_durations) / len(all_durations)
            sorted_durations = sorted(all_durations)
            p95_index = int(len(sorted_durations) * 0.95)
            metrics['p95_duration_ms'] = sorted_durations[p95_index]

        return metrics

    def extract_error_metrics(self) -> Dict[str, Any]:
        """Extract error metrics from logs"""
        error_logs = self.aggregated_data.get('error', [])

        metrics = {
            'total_errors': len(error_logs),
            'by_type': defaultdict(int),
            'by_service': defaultdict(int),
            'critical_errors': []
        }

        for log in error_logs:
            error_type = log.get('error_type', 'unknown')
            service = log.get('service', 'unknown')

            metrics['by_type'][error_type] += 1
            metrics['by_service'][service] += 1

            if log.get('level') == 'CRITICAL':
                metrics['critical_errors'].append({
                    'timestamp': log.get('timestamp'),
                    'service': service,
                    'message': log.get('message')
                })

        return metrics

    def extract_cache_metrics(self) -> Dict[str, Any]:
        """Extract cache performance metrics"""
        cache_logs = self.aggregated_data.get('cache', [])

        metrics = {
            'total_requests': len(cache_logs),
            'hits': 0,
            'misses': 0,
            'hit_rate': 0,
            'by_layer': defaultdict(lambda: {'hits': 0, 'misses': 0})
        }

        for log in cache_logs:
            message = log.get('message', '')
            layer = log.get('cache_layer', 'unknown')

            if 'hit' in message.lower():
                metrics['hits'] += 1
                metrics['by_layer'][layer]['hits'] += 1
            elif 'miss' in message.lower():
                metrics['misses'] += 1
                metrics['by_layer'][layer]['misses'] += 1

        if metrics['total_requests'] > 0:
            metrics['hit_rate'] = metrics['hits'] / metrics['total_requests']

        return metrics

    async def export_to_elasticsearch(self, es_url: str, index: str) -> None:
        """Export aggregated logs to Elasticsearch"""
        logger.info(f"Exporting to Elasticsearch: {es_url}/{index}")

        # In production: use elasticsearch-py client
        import requests

        for category, logs in self.aggregated_data.items():
            for log_entry in logs:
                doc = {
                    **log_entry,
                    'aggregation_timestamp': DeterministicClock.now().isoformat()
                }

                # Bulk index would be more efficient
                try:
                    response = requests.post(
                        f"{es_url}/{index}/_doc",
                        json=doc,
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()
                except Exception as e:
                    logger.error(f"Failed to index log: {e}")

        logger.info("Export to Elasticsearch completed")

    def generate_summary_report(self) -> str:
        """Generate summary report of aggregated logs"""
        perf_metrics = self.extract_performance_metrics()
        error_metrics = self.extract_error_metrics()
        cache_metrics = self.extract_cache_metrics()

        report = f"""
Log Aggregation Summary Report
================================
Timestamp: {DeterministicClock.now().isoformat()}

Performance Metrics:
--------------------
Total Requests: {perf_metrics['request_count']}
Average Duration: {perf_metrics['avg_duration_ms']:.2f} ms
P95 Duration: {perf_metrics['p95_duration_ms']:.2f} ms

Error Metrics:
--------------
Total Errors: {error_metrics['total_errors']}
Critical Errors: {len(error_metrics['critical_errors'])}
Errors by Type:
"""

        for error_type, count in error_metrics['by_type'].items():
            report += f"  - {error_type}: {count}\n"

        report += f"""
Cache Metrics:
--------------
Total Cache Requests: {cache_metrics['total_requests']}
Cache Hits: {cache_metrics['hits']}
Cache Misses: {cache_metrics['misses']}
Hit Rate: {cache_metrics['hit_rate']:.2%}

Cache Performance by Layer:
"""

        for layer, stats in cache_metrics['by_layer'].items():
            total = stats['hits'] + stats['misses']
            hit_rate = stats['hits'] / total if total > 0 else 0
            report += f"  - {layer}: {hit_rate:.2%} ({stats['hits']}/{total})\n"

        return report


async def main():
    """Main entry point for log aggregator"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure log sources
    log_sources = [
        "/var/log/greenlang/factor-broker.log",
        "/var/log/greenlang/entity-mdm.log",
        "/var/log/greenlang/csrd-app.log",
        "/var/log/greenlang/vcci-app.log"
    ]

    aggregator = LogAggregator(log_sources)

    # Aggregate logs
    await aggregator.aggregate_logs()

    # Generate report
    report = aggregator.generate_summary_report()
    print(report)

    # Export to Elasticsearch (optional)
    # await aggregator.export_to_elasticsearch("http://localhost:9200", "greenlang-logs")


if __name__ == "__main__":
    asyncio.run(main())
