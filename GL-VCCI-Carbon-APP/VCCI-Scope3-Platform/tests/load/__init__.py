"""
GL-VCCI Scope 3 Carbon Intelligence Platform - Load Testing Suite

This package contains comprehensive load testing scenarios using Locust to validate
performance targets under realistic load conditions.

Performance Targets:
    - Ingestion: 100K transactions per hour sustained
    - Calculations: 10K calculations per second
    - API latency: p95 < 200ms on aggregates
    - Concurrent users: 1,000 users

Test Scenarios:
    1. Ramp-up: 0 → 1,000 users over 10 minutes
    2. Sustained load: 1,000 users for 1 hour
    3. Spike test: 1,000 → 5,000 users (sudden)
    4. Endurance: 500 users for 24 hours

Modules:
    - locustfile_rampup: Gradual user ramp-up scenario
    - locustfile_sustained: Sustained load scenario
    - locustfile_spike: Spike testing scenario
    - locustfile_endurance: Long-term endurance scenario
    - load_test_utils: Shared utilities for load testing
    - generate_performance_report: Performance report generation

Usage:
    # Run ramp-up test
    locust -f locustfile_rampup.py --host=http://localhost:8000 \\
           --users=1000 --spawn-rate=1.67 --run-time=10m

    # Run sustained load test
    locust -f locustfile_sustained.py --host=http://localhost:8000 \\
           --users=1000 --spawn-rate=50 --run-time=1h

    # Run spike test
    locust -f locustfile_spike.py --host=http://localhost:8000 \\
           --users=5000 --spawn-rate=4000 --run-time=20m

    # Run endurance test
    locust -f locustfile_endurance.py --host=http://localhost:8000 \\
           --users=500 --spawn-rate=10 --run-time=24h

Author: GL-VCCI Team
Version: 1.0.0
Phase: Phase 6 - Testing & Validation
"""

__version__ = "1.0.0"
__author__ = "GL-VCCI Team"

from .load_test_utils import (
    generate_realistic_procurement_data,
    generate_csv_data,
    monitor_system_resources,
    validate_performance_targets,
    create_test_user_auth,
)

__all__ = [
    "generate_realistic_procurement_data",
    "generate_csv_data",
    "monitor_system_resources",
    "validate_performance_targets",
    "create_test_user_auth",
]
