#!/usr/bin/env python3
"""
CSRD Reporting Platform - Monitoring Setup Validation Script

This script validates that all monitoring and observability components
are properly installed and configured.

Author: GreenLang Operations Team (Team B3)
Date: 2025-11-08
"""

import sys
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓{Colors.END} {text}")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗{Colors.END} {text}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠{Colors.END} {text}")


def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ{Colors.END} {text}")


def check_python_dependencies() -> Tuple[bool, List[str]]:
    """Check if required Python packages are installed"""
    print_header("Python Dependencies Check")

    required_packages = [
        ('prometheus_client', 'prometheus-client'),
        ('sentry_sdk', 'sentry-sdk'),
        ('structlog', 'structlog'),
        ('pythonjsonlogger', 'python-json-logger'),
        ('psutil', 'psutil'),
        ('fastapi', 'fastapi'),
        ('pydantic', 'pydantic'),
    ]

    all_installed = True
    missing_packages = []

    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            print_success(f"{package_name} is installed")
        except ImportError:
            print_error(f"{package_name} is NOT installed")
            all_installed = False
            missing_packages.append(package_name)

    if all_installed:
        print_success("\nAll required Python dependencies are installed")
    else:
        print_error(f"\n{len(missing_packages)} package(s) missing")
        print_info(f"Install with: pip install {' '.join(missing_packages)}")

    return all_installed, missing_packages


def check_monitoring_files() -> Tuple[bool, List[str]]:
    """Check if monitoring files exist"""
    print_header("Monitoring Files Check")

    base_path = Path(__file__).parent.parent

    required_files = [
        'backend/health.py',
        'backend/logging_config.py',
        'backend/metrics.py',
        'backend/error_tracking.py',
        'monitoring/grafana-csrd-dashboard.json',
        'monitoring/prometheus.yml',
        'monitoring/alerts/alerts-csrd.yml',
        'MONITORING.md',
        'MONITORING_SETUP_GUIDE.md',
        'MONITORING_IMPLEMENTATION_SUMMARY.md',
    ]

    all_exist = True
    missing_files = []

    for file_path in required_files:
        full_path = base_path / file_path
        if full_path.exists():
            size = full_path.stat().st_size
            print_success(f"{file_path} ({size:,} bytes)")
        else:
            print_error(f"{file_path} NOT FOUND")
            all_exist = False
            missing_files.append(file_path)

    if all_exist:
        print_success(f"\nAll {len(required_files)} required files exist")
    else:
        print_error(f"\n{len(missing_files)} file(s) missing")

    return all_exist, missing_files


def check_module_imports() -> Tuple[bool, List[str]]:
    """Check if monitoring modules can be imported"""
    print_header("Module Import Check")

    modules = [
        'backend.health',
        'backend.logging_config',
        'backend.metrics',
        'backend.error_tracking',
    ]

    all_imported = True
    failed_imports = []

    # Add backend to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    for module in modules:
        try:
            __import__(module)
            print_success(f"{module} imported successfully")
        except Exception as e:
            print_error(f"{module} import failed: {str(e)}")
            all_imported = False
            failed_imports.append(module)

    if all_imported:
        print_success(f"\nAll {len(modules)} modules imported successfully")
    else:
        print_error(f"\n{len(failed_imports)} module(s) failed to import")

    return all_imported, failed_imports


def check_health_endpoints() -> Tuple[bool, Dict[str, Any]]:
    """Check if health check endpoints are properly configured"""
    print_header("Health Check Endpoints Validation")

    try:
        from backend.health import health_router

        endpoints = []
        for route in health_router.routes:
            endpoints.append({
                'path': route.path,
                'methods': route.methods,
                'name': route.name
            })

        expected_endpoints = [
            '/health',
            '/health/',
            '/health/live',
            '/health/ready',
            '/health/startup',
            '/health/esrs',
            '/health/esrs/{standard}'
        ]

        found_paths = set(ep['path'] for ep in endpoints)
        all_found = True

        for expected in expected_endpoints:
            # Handle parameterized routes
            expected_pattern = expected.replace('{standard}', '')
            found = any(expected_pattern in path for path in found_paths)

            if found or expected in found_paths:
                print_success(f"Endpoint: {expected}")
            else:
                print_error(f"Endpoint missing: {expected}")
                all_found = False

        if all_found:
            print_success(f"\nAll {len(expected_endpoints)} endpoints configured")
        else:
            print_error("\nSome endpoints are missing")

        return all_found, {'endpoints': endpoints}

    except Exception as e:
        print_error(f"Failed to validate health endpoints: {str(e)}")
        return False, {}


def check_metrics_defined() -> Tuple[bool, Dict[str, Any]]:
    """Check if Prometheus metrics are properly defined"""
    print_header("Prometheus Metrics Validation")

    try:
        from backend.metrics import (
            http_requests_total,
            esrs_data_point_coverage,
            validation_errors_total,
            agent_execution_duration_seconds,
            llm_api_cost_usd_total,
            compliance_deadline_days_remaining
        )

        metrics = {
            'http_requests_total': http_requests_total,
            'esrs_data_point_coverage': esrs_data_point_coverage,
            'validation_errors_total': validation_errors_total,
            'agent_execution_duration_seconds': agent_execution_duration_seconds,
            'llm_api_cost_usd_total': llm_api_cost_usd_total,
            'compliance_deadline_days_remaining': compliance_deadline_days_remaining,
        }

        all_defined = True
        for name, metric in metrics.items():
            if metric:
                print_success(f"Metric: {name}")
            else:
                print_error(f"Metric missing: {name}")
                all_defined = False

        if all_defined:
            print_success(f"\nAll {len(metrics)} key metrics defined")
        else:
            print_error("\nSome metrics are missing")

        return all_defined, {'metrics': list(metrics.keys())}

    except Exception as e:
        print_error(f"Failed to validate metrics: {str(e)}")
        return False, {}


def check_logging_config() -> Tuple[bool, Dict[str, Any]]:
    """Check if logging configuration is valid"""
    print_header("Logging Configuration Validation")

    try:
        from backend.logging_config import (
            setup_structured_logging,
            get_logger,
            LogContext,
            get_audit_logger
        )

        components = {
            'setup_structured_logging': setup_structured_logging,
            'get_logger': get_logger,
            'LogContext': LogContext,
            'get_audit_logger': get_audit_logger,
        }

        all_valid = True
        for name, component in components.items():
            if component:
                print_success(f"Component: {name}")
            else:
                print_error(f"Component missing: {name}")
                all_valid = False

        # Test basic functionality
        try:
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.log') as f:
                temp_log = f.name

            logger = setup_structured_logging(
                log_level='INFO',
                log_file=temp_log,
                enable_json=True,
                enable_console=False
            )

            logger.info("Test log message")

            if os.path.exists(temp_log) and os.path.getsize(temp_log) > 0:
                print_success("Logging test: Log file created and written")
                os.unlink(temp_log)
            else:
                print_error("Logging test: Log file not created")
                all_valid = False

        except Exception as e:
            print_error(f"Logging test failed: {str(e)}")
            all_valid = False

        if all_valid:
            print_success("\nLogging configuration is valid")
        else:
            print_error("\nLogging configuration has issues")

        return all_valid, {'components': list(components.keys())}

    except Exception as e:
        print_error(f"Failed to validate logging: {str(e)}")
        return False, {}


def check_sentry_config() -> Tuple[bool, Dict[str, Any]]:
    """Check if Sentry configuration is valid"""
    print_header("Sentry Integration Validation")

    try:
        from backend.error_tracking import (
            init_sentry,
            set_esrs_context,
            capture_exception,
            monitor_errors
        )

        components = {
            'init_sentry': init_sentry,
            'set_esrs_context': set_esrs_context,
            'capture_exception': capture_exception,
            'monitor_errors': monitor_errors,
        }

        all_valid = True
        for name, component in components.items():
            if component:
                print_success(f"Component: {name}")
            else:
                print_error(f"Component missing: {name}")
                all_valid = False

        # Check if Sentry DSN is configured (optional)
        sentry_dsn = os.getenv('SENTRY_DSN')
        if sentry_dsn:
            print_success("Sentry DSN is configured")
        else:
            print_warning("Sentry DSN not configured (optional)")

        if all_valid:
            print_success("\nSentry integration is valid")
        else:
            print_error("\nSentry integration has issues")

        return all_valid, {'components': list(components.keys())}

    except Exception as e:
        print_error(f"Failed to validate Sentry: {str(e)}")
        return False, {}


def print_summary(results: Dict[str, Tuple[bool, Any]]):
    """Print validation summary"""
    print_header("Validation Summary")

    total_checks = len(results)
    passed_checks = sum(1 for success, _ in results.values() if success)

    for check_name, (success, details) in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if success else f"{Colors.RED}FAIL{Colors.END}"
        print(f"{status} - {check_name}")

    print(f"\n{Colors.BOLD}Total: {passed_checks}/{total_checks} checks passed{Colors.END}")

    if passed_checks == total_checks:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ All monitoring components are properly configured!{Colors.END}\n")
        return True
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}✗ Some monitoring components need attention{Colors.END}\n")
        return False


def main():
    """Main validation function"""
    print(f"\n{Colors.BOLD}CSRD Reporting Platform - Monitoring Setup Validation{Colors.END}")
    print(f"{Colors.BOLD}Team B3: GL-CSRD Monitoring & Observability{Colors.END}")

    results = {}

    # Run all checks
    results['Python Dependencies'] = check_python_dependencies()
    results['Monitoring Files'] = check_monitoring_files()
    results['Module Imports'] = check_module_imports()
    results['Health Endpoints'] = check_health_endpoints()
    results['Prometheus Metrics'] = check_metrics_defined()
    results['Logging Configuration'] = check_logging_config()
    results['Sentry Integration'] = check_sentry_config()

    # Print summary
    all_passed = print_summary(results)

    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
