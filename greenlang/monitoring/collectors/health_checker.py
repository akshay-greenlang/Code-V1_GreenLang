"""
Health Checker Agent
====================

Check health and availability of all infrastructure services.

Author: Monitoring & Observability Team
Created: 2025-11-09
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import aiohttp
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Service health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class ServiceHealth:
    """Service health check result"""
    service_name: str
    status: HealthStatus
    response_time_ms: float
    last_check: datetime
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class HealthChecker:
    """
    Checks health of all infrastructure services.
    """

    def __init__(self):
        self.services = self._define_services()
        self.health_results: Dict[str, ServiceHealth] = {}

    def _define_services(self) -> Dict[str, Dict[str, Any]]:
        """Define services to monitor"""
        return {
            'factor-broker': {
                'url': 'http://localhost:8001/health',
                'timeout': 5,
                'critical': True
            },
            'entity-mdm': {
                'url': 'http://localhost:8002/health',
                'timeout': 5,
                'critical': True
            },
            'form-builder': {
                'url': 'http://localhost:8003/health',
                'timeout': 5,
                'critical': False
            },
            'api-gateway': {
                'url': 'http://localhost:8000/health',
                'timeout': 5,
                'critical': True
            },
            'redis-cache': {
                'url': 'redis://localhost:6379',
                'timeout': 3,
                'critical': True,
                'check_type': 'redis'
            },
            'postgresql': {
                'url': 'postgresql://localhost:5432/greenlang',
                'timeout': 5,
                'critical': True,
                'check_type': 'postgres'
            },
            'prometheus': {
                'url': 'http://localhost:9090/-/healthy',
                'timeout': 3,
                'critical': False
            },
            'grafana': {
                'url': 'http://localhost:3000/api/health',
                'timeout': 3,
                'critical': False
            }
        }

    async def check_all_services(self) -> Dict[str, ServiceHealth]:
        """Check health of all services"""
        logger.info("Starting health checks for all services...")

        tasks = [
            self._check_service(name, config)
            for name, config in self.services.items()
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, ServiceHealth):
                self.health_results[result.service_name] = result
            elif isinstance(result, Exception):
                logger.error(f"Health check exception: {result}")

        logger.info(f"Health checks completed for {len(self.health_results)} services")
        return self.health_results

    async def _check_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> ServiceHealth:
        """Check health of a single service"""
        check_type = config.get('check_type', 'http')

        if check_type == 'http':
            return await self._check_http_service(service_name, config)
        elif check_type == 'redis':
            return await self._check_redis_service(service_name, config)
        elif check_type == 'postgres':
            return await self._check_postgres_service(service_name, config)
        else:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNKNOWN,
                response_time_ms=0,
                last_check=datetime.now(),
                error_message="Unknown check type"
            )

    async def _check_http_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> ServiceHealth:
        """Check HTTP-based service"""
        start_time = datetime.now()

        try:
            timeout = aiohttp.ClientTimeout(total=config['timeout'])
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(config['url']) as response:
                    response_time = (datetime.now() - start_time).total_seconds() * 1000

                    if response.status == 200:
                        # Try to parse response for additional metadata
                        try:
                            data = await response.json()
                            metadata = {
                                'version': data.get('version'),
                                'uptime': data.get('uptime'),
                                'dependencies': data.get('dependencies')
                            }
                        except:
                            metadata = {}

                        return ServiceHealth(
                            service_name=service_name,
                            status=HealthStatus.HEALTHY,
                            response_time_ms=response_time,
                            last_check=datetime.now(),
                            metadata=metadata
                        )
                    else:
                        return ServiceHealth(
                            service_name=service_name,
                            status=HealthStatus.DEGRADED,
                            response_time_ms=response_time,
                            last_check=datetime.now(),
                            error_message=f"HTTP {response.status}"
                        )

        except asyncio.TimeoutError:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=config['timeout'] * 1000,
                last_check=datetime.now(),
                error_message="Request timeout"
            )
        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                last_check=datetime.now(),
                error_message=str(e)
            )

    async def _check_redis_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> ServiceHealth:
        """Check Redis service"""
        start_time = datetime.now()

        try:
            import redis.asyncio as redis

            client = redis.from_url(
                config['url'],
                socket_connect_timeout=config['timeout']
            )

            await client.ping()
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # Get Redis info
            info = await client.info()
            metadata = {
                'connected_clients': info.get('connected_clients'),
                'used_memory_human': info.get('used_memory_human'),
                'uptime_in_seconds': info.get('uptime_in_seconds')
            }

            await client.close()

            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata=metadata
            )

        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                last_check=datetime.now(),
                error_message=str(e)
            )

    async def _check_postgres_service(
        self,
        service_name: str,
        config: Dict[str, Any]
    ) -> ServiceHealth:
        """Check PostgreSQL service"""
        start_time = datetime.now()

        try:
            import asyncpg

            conn = await asyncpg.connect(
                config['url'],
                timeout=config['timeout']
            )

            # Simple health check query
            result = await conn.fetchval('SELECT 1')
            response_time = (datetime.now() - start_time).total_seconds() * 1000

            # Get database stats
            db_size = await conn.fetchval(
                "SELECT pg_database_size(current_database())"
            )
            active_connections = await conn.fetchval(
                "SELECT count(*) FROM pg_stat_activity"
            )

            metadata = {
                'database_size_bytes': db_size,
                'active_connections': active_connections
            }

            await conn.close()

            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.HEALTHY,
                response_time_ms=response_time,
                last_check=datetime.now(),
                metadata=metadata
            )

        except Exception as e:
            return ServiceHealth(
                service_name=service_name,
                status=HealthStatus.UNHEALTHY,
                response_time_ms=0,
                last_check=datetime.now(),
                error_message=str(e)
            )

    def get_overall_status(self) -> HealthStatus:
        """Get overall infrastructure health status"""
        if not self.health_results:
            return HealthStatus.UNKNOWN

        critical_services = [
            name for name, config in self.services.items()
            if config.get('critical', False)
        ]

        critical_health = [
            self.health_results[name].status
            for name in critical_services
            if name in self.health_results
        ]

        if any(status == HealthStatus.UNHEALTHY for status in critical_health):
            return HealthStatus.UNHEALTHY
        elif any(status == HealthStatus.DEGRADED for status in critical_health):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def generate_report(self) -> str:
        """Generate health check report"""
        overall_status = self.get_overall_status()

        report = f"""
Infrastructure Health Check Report
====================================
Timestamp: {datetime.now().isoformat()}
Overall Status: {overall_status.value.upper()}

Service Health Details:
-----------------------
"""

        for service_name, health in sorted(self.health_results.items()):
            critical = self.services[service_name].get('critical', False)
            critical_marker = " [CRITICAL]" if critical else ""

            report += f"""
{service_name}{critical_marker}:
  Status: {health.status.value.upper()}
  Response Time: {health.response_time_ms:.2f} ms
  Last Check: {health.last_check.isoformat()}
"""

            if health.error_message:
                report += f"  Error: {health.error_message}\n"

            if health.metadata:
                report += "  Metadata:\n"
                for key, value in health.metadata.items():
                    if value is not None:
                        report += f"    - {key}: {value}\n"

        # Summary statistics
        healthy = sum(1 for h in self.health_results.values() if h.status == HealthStatus.HEALTHY)
        degraded = sum(1 for h in self.health_results.values() if h.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for h in self.health_results.values() if h.status == HealthStatus.UNHEALTHY)

        report += f"""
Summary:
--------
Total Services: {len(self.health_results)}
Healthy: {healthy}
Degraded: {degraded}
Unhealthy: {unhealthy}
"""

        return report

    async def export_to_prometheus(self, pushgateway_url: str) -> None:
        """Export health metrics to Prometheus"""
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

        registry = CollectorRegistry()

        service_up = Gauge(
            'greenlang_service_up',
            'Service availability (1=up, 0=down)',
            ['service'],
            registry=registry
        )

        service_response_time = Gauge(
            'greenlang_service_response_time_ms',
            'Service response time in milliseconds',
            ['service'],
            registry=registry
        )

        for service_name, health in self.health_results.items():
            is_up = 1 if health.status == HealthStatus.HEALTHY else 0
            service_up.labels(service=service_name).set(is_up)
            service_response_time.labels(service=service_name).set(health.response_time_ms)

        push_to_gateway(pushgateway_url, job='health_check', registry=registry)
        logger.info(f"Health metrics pushed to {pushgateway_url}")


async def main():
    """Main entry point for health checker"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    checker = HealthChecker()

    # Run health checks
    await checker.check_all_services()

    # Generate report
    report = checker.generate_report()
    print(report)

    # Export to Prometheus (optional)
    # await checker.export_to_prometheus("localhost:9091")


if __name__ == "__main__":
    asyncio.run(main())
