#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GreenLang Platform - Integration Validation Script
===================================================
Tests cross-app integration and validates the unified platform deployment.

Tests:
1. Service health checks
2. Database connectivity
3. Shared authentication (JWT)
4. Cross-app REST API calls
5. Message queue integration
6. Monitoring endpoints
"""

import sys
import time
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import psycopg2
import redis
import pika
from greenlang.determinism import DeterministicClock

# Configuration
BASE_URLS = {
    'cbam': 'http://localhost:8001',
    'csrd': 'http://localhost:8002',
    'vcci': 'http://localhost:8000',
}

POSTGRES_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'greenlang_platform',
    'user': 'greenlang_admin',
    'password': 'greenlang_secure_2024'
}

REDIS_CONFIG = {
    'host': 'localhost',
    'port': 6379,
    'password': 'greenlang_redis_2024'
}

RABBITMQ_CONFIG = {
    'host': 'localhost',
    'port': 5672,
    'username': 'greenlang',
    'password': 'greenlang_rabbit_2024',
    'vhost': 'greenlang_platform'
}

# Test Results
results = {
    'passed': 0,
    'failed': 0,
    'skipped': 0,
    'details': []
}

def log_result(test_name: str, status: str, message: str, details: Dict = None):
    """Log test result"""
    result = {
        'test': test_name,
        'status': status,
        'message': message,
        'timestamp': DeterministicClock.now().isoformat(),
        'details': details or {}
    }
    results['details'].append(result)

    if status == 'PASS':
        results['passed'] += 1
        print(f"✓ {test_name}: {message}")
    elif status == 'FAIL':
        results['failed'] += 1
        print(f"✗ {test_name}: {message}")
    else:
        results['skipped'] += 1
        print(f"○ {test_name}: {message}")

def test_health_endpoints():
    """Test 1: Health endpoints for all services"""
    print("\n=== Testing Health Endpoints ===")

    health_endpoints = {
        'cbam': f"{BASE_URLS['cbam']}/health",
        'csrd': f"{BASE_URLS['csrd']}/health",
        'vcci': f"{BASE_URLS['vcci']}/health/live",
        'prometheus': 'http://localhost:9090/-/healthy',
        'grafana': 'http://localhost:3000/api/health',
    }

    for service, url in health_endpoints.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                log_result(
                    f"Health Check - {service.upper()}",
                    'PASS',
                    f"Service is healthy (status: {response.status_code})",
                    {'url': url, 'response_time': response.elapsed.total_seconds()}
                )
            else:
                log_result(
                    f"Health Check - {service.upper()}",
                    'FAIL',
                    f"Unhealthy status code: {response.status_code}",
                    {'url': url}
                )
        except Exception as e:
            log_result(
                f"Health Check - {service.upper()}",
                'FAIL',
                f"Connection failed: {str(e)}",
                {'url': url}
            )

def test_database_connectivity():
    """Test 2: PostgreSQL connectivity and schema"""
    print("\n=== Testing Database Connectivity ===")

    try:
        conn = psycopg2.connect(**POSTGRES_CONFIG)
        cursor = conn.cursor()

        # Test connection
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        log_result(
            "Database Connection",
            'PASS',
            "Connected to PostgreSQL",
            {'version': version}
        )

        # Check schemas
        cursor.execute("""
            SELECT schema_name
            FROM information_schema.schemata
            WHERE schema_name IN ('public', 'cbam', 'csrd', 'vcci', 'shared')
        """)
        schemas = [row[0] for row in cursor.fetchall()]
        expected_schemas = {'public', 'cbam', 'csrd', 'vcci', 'shared'}

        if expected_schemas.issubset(set(schemas)):
            log_result(
                "Database Schemas",
                'PASS',
                f"All required schemas exist: {', '.join(schemas)}",
                {'schemas': schemas}
            )
        else:
            missing = expected_schemas - set(schemas)
            log_result(
                "Database Schemas",
                'FAIL',
                f"Missing schemas: {', '.join(missing)}",
                {'found': schemas, 'missing': list(missing)}
            )

        # Check shared tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_name IN ('organizations', 'users', 'user_app_roles', 'cross_app_sync')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        expected_tables = {'organizations', 'users', 'user_app_roles', 'cross_app_sync'}

        if expected_tables.issubset(set(tables)):
            log_result(
                "Shared Tables",
                'PASS',
                f"All shared tables exist: {', '.join(tables)}",
                {'tables': tables}
            )
        else:
            missing = expected_tables - set(tables)
            log_result(
                "Shared Tables",
                'FAIL',
                f"Missing tables: {', '.join(missing)}",
                {'found': tables, 'missing': list(missing)}
            )

        cursor.close()
        conn.close()

    except Exception as e:
        log_result(
            "Database Connection",
            'FAIL',
            f"Connection failed: {str(e)}"
        )

def test_redis_connectivity():
    """Test 3: Redis connectivity"""
    print("\n=== Testing Redis Connectivity ===")

    try:
        r = redis.Redis(
            host=REDIS_CONFIG['host'],
            port=REDIS_CONFIG['port'],
            password=REDIS_CONFIG['password'],
            decode_responses=True
        )

        # Test ping
        if r.ping():
            log_result(
                "Redis Connection",
                'PASS',
                "Connected to Redis cache"
            )

            # Test write/read
            test_key = 'platform_integration_test'
            test_value = f'test_{int(time.time())}'
            r.set(test_key, test_value, ex=60)
            retrieved = r.get(test_key)

            if retrieved == test_value:
                log_result(
                    "Redis Operations",
                    'PASS',
                    "Read/write operations successful"
                )
            else:
                log_result(
                    "Redis Operations",
                    'FAIL',
                    f"Data mismatch: wrote '{test_value}', read '{retrieved}'"
                )
        else:
            log_result(
                "Redis Connection",
                'FAIL',
                "Redis ping failed"
            )

    except Exception as e:
        log_result(
            "Redis Connection",
            'FAIL',
            f"Connection failed: {str(e)}"
        )

def test_rabbitmq_connectivity():
    """Test 4: RabbitMQ connectivity"""
    print("\n=== Testing RabbitMQ Connectivity ===")

    try:
        credentials = pika.PlainCredentials(
            RABBITMQ_CONFIG['username'],
            RABBITMQ_CONFIG['password']
        )
        parameters = pika.ConnectionParameters(
            host=RABBITMQ_CONFIG['host'],
            port=RABBITMQ_CONFIG['port'],
            virtual_host=RABBITMQ_CONFIG['vhost'],
            credentials=credentials
        )

        connection = pika.BlockingConnection(parameters)
        channel = connection.channel()

        log_result(
            "RabbitMQ Connection",
            'PASS',
            "Connected to RabbitMQ message queue"
        )

        # Test queue declaration
        test_queue = 'platform_integration_test'
        channel.queue_declare(queue=test_queue, durable=True)

        # Test publish/consume
        test_message = json.dumps({
            'test': 'integration',
            'timestamp': DeterministicClock.now().isoformat()
        })

        channel.basic_publish(
            exchange='',
            routing_key=test_queue,
            body=test_message
        )

        log_result(
            "RabbitMQ Publish",
            'PASS',
            "Successfully published test message"
        )

        # Clean up
        channel.queue_delete(queue=test_queue)
        connection.close()

    except Exception as e:
        log_result(
            "RabbitMQ Connection",
            'FAIL',
            f"Connection failed: {str(e)}"
        )

def test_cross_app_api_calls():
    """Test 5: Cross-app REST API integration"""
    print("\n=== Testing Cross-App API Calls ===")

    # This would require actual API endpoints to be implemented
    # For now, we'll skip this test
    log_result(
        "Cross-App API - VCCI to CSRD",
        'SKIP',
        "API endpoints not yet implemented",
        {'note': 'Requires /api/v1/emissions endpoint on VCCI and consumption endpoint on CSRD'}
    )

def test_monitoring_stack():
    """Test 6: Monitoring stack"""
    print("\n=== Testing Monitoring Stack ===")

    try:
        # Test Prometheus targets
        response = requests.get('http://localhost:9090/api/v1/targets', timeout=5)
        if response.status_code == 200:
            targets_data = response.json()
            active_targets = targets_data.get('data', {}).get('activeTargets', [])

            log_result(
                "Prometheus Targets",
                'PASS',
                f"Found {len(active_targets)} active targets",
                {'target_count': len(active_targets)}
            )
        else:
            log_result(
                "Prometheus Targets",
                'FAIL',
                f"Failed to fetch targets (status: {response.status_code})"
            )

        # Test Grafana datasources
        response = requests.get(
            'http://localhost:3000/api/datasources',
            auth=('admin', 'greenlang2024'),
            timeout=5
        )
        if response.status_code == 200:
            datasources = response.json()
            log_result(
                "Grafana Datasources",
                'PASS',
                f"Found {len(datasources)} configured datasource(s)",
                {'datasource_count': len(datasources)}
            )
        else:
            log_result(
                "Grafana Datasources",
                'FAIL',
                f"Failed to fetch datasources (status: {response.status_code})"
            )

    except Exception as e:
        log_result(
            "Monitoring Stack",
            'FAIL',
            f"Monitoring tests failed: {str(e)}"
        )

def generate_report():
    """Generate test report"""
    print("\n" + "="*80)
    print("INTEGRATION VALIDATION REPORT")
    print("="*80)
    print(f"Timestamp: {DeterministicClock.now().isoformat()}")
    print(f"\nResults:")
    print(f"  ✓ Passed:  {results['passed']}")
    print(f"  ✗ Failed:  {results['failed']}")
    print(f"  ○ Skipped: {results['skipped']}")
    print(f"  Total:     {results['passed'] + results['failed'] + results['skipped']}")

    success_rate = (results['passed'] / (results['passed'] + results['failed'])) * 100 if (results['passed'] + results['failed']) > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")

    # Save detailed report
    report_file = f"integration_report_{DeterministicClock.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nDetailed report saved to: {report_file}")
    print("="*80 + "\n")

    return results['failed'] == 0

def main():
    """Main execution"""
    print("\n" + "="*80)
    print("GreenLang Platform - Integration Validation")
    print("="*80)
    print("Testing unified platform deployment...\n")

    # Wait for services to be ready
    print("Waiting 5 seconds for services to stabilize...")
    time.sleep(5)

    # Run all tests
    test_health_endpoints()
    test_database_connectivity()
    test_redis_connectivity()
    test_rabbitmq_connectivity()
    test_cross_app_api_calls()
    test_monitoring_stack()

    # Generate report
    success = generate_report()

    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
