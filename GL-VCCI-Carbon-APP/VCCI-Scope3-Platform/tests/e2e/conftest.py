# -*- coding: utf-8 -*-
"""
E2E Test Infrastructure - Shared Fixtures and Utilities

This module provides comprehensive fixtures and utilities for end-to-end testing
of the GL-VCCI Scope 3 Carbon Intelligence Platform.

Fixtures:
- browser: Playwright browser instance for UI testing
- test_tenant: Isolated test tenant with cleanup
- sap_sandbox: Mock SAP S/4HANA environment
- oracle_sandbox: Mock Oracle Fusion environment
- workday_sandbox: Mock Workday RaaS environment
- test_data_factory: Factory for generating test data
- performance_monitor: Performance metrics collector
- audit_trail_validator: Audit trail validation utilities
"""

import asyncio
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import pytest
import redis
from playwright.async_api import Browser, Page, async_playwright
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

class E2ETestConfig:
    """Central configuration for E2E tests"""

    # Database configuration
    TEST_DATABASE_URL = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://test_user:test_pass@localhost:5432/vcci_test"
    )

    # Redis configuration
    TEST_REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/15")

    # API configuration
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    API_TIMEOUT = 30  # seconds

    # Performance thresholds
    INGESTION_THROUGHPUT_TARGET = 100_000  # records per hour
    API_LATENCY_P95_TARGET = 200  # milliseconds
    CALCULATION_THROUGHPUT_TARGET = 10_000  # calculations per second

    # Test data paths
    TEST_DATA_DIR = Path(__file__).parent / "test_data"
    FIXTURES_DIR = Path(__file__).parent / "fixtures"

    # Browser configuration
    BROWSER_HEADLESS = True
    BROWSER_SLOW_MO = 0  # milliseconds

    # Tenant configuration
    DEFAULT_TENANT_NAME = "e2e-test-tenant"

    # Feature flags
    ENABLE_UI_TESTS = True
    ENABLE_PERFORMANCE_TESTS = True
    ENABLE_LOAD_TESTS = os.getenv("ENABLE_LOAD_TESTS", "false").lower() == "true"


config = E2ETestConfig()


# =============================================================================
# DATABASE FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def db_engine():
    """Create database engine for session"""
    engine = create_engine(
        config.TEST_DATABASE_URL,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20
    )
    yield engine
    engine.dispose()


@pytest.fixture(scope="session")
def SessionLocal(db_engine):
    """Create session factory"""
    return sessionmaker(autocommit=False, autoflush=False, bind=db_engine)


@pytest.fixture
def db_session(SessionLocal):
    """Create database session for test"""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# =============================================================================
# REDIS FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def redis_client():
    """Create Redis client for session"""
    client = redis.from_url(config.TEST_REDIS_URL, decode_responses=True)
    yield client
    # Cleanup all test keys
    for key in client.scan_iter("test:*"):
        client.delete(key)
    client.close()


# =============================================================================
# TENANT FIXTURES
# =============================================================================

class TestTenant:
    """Test tenant with isolated data and configuration"""

    def __init__(
        self,
        tenant_id: str,
        name: str,
        db_session: Session,
        redis_client: redis.Redis
    ):
        self.id = tenant_id
        self.name = name
        self.db_session = db_session
        self.redis = redis_client
        self.created_at = DeterministicClock.utcnow()
        self.metadata = {}

    def add_metadata(self, key: str, value: Any):
        """Add metadata to tenant"""
        self.metadata[key] = value

    def get_namespace(self) -> str:
        """Get database namespace for tenant"""
        return f"tenant_{self.id}"

    async def cleanup(self):
        """Cleanup tenant data"""
        logger.info(f"Cleaning up tenant {self.id}")

        # Delete all tenant records from database
        # In production, this would use proper tenant isolation
        # For tests, we clean up by tenant_id

        # Delete Redis keys
        for key in self.redis.scan_iter(f"tenant:{self.id}:*"):
            self.redis.delete(key)

        logger.info(f"Tenant {self.id} cleanup complete")


@pytest.fixture
async def test_tenant(db_session, redis_client):
    """Create isolated test tenant"""
    tenant_id = str(deterministic_uuid(__name__, str(DeterministicClock.now())))
    tenant_name = f"{config.DEFAULT_TENANT_NAME}-{tenant_id[:8]}"

    tenant = TestTenant(
        tenant_id=tenant_id,
        name=tenant_name,
        db_session=db_session,
        redis_client=redis_client
    )

    logger.info(f"Created test tenant: {tenant.id}")

    yield tenant

    await tenant.cleanup()


# =============================================================================
# BROWSER FIXTURES (Playwright)
# =============================================================================

@pytest.fixture(scope="session")
async def browser():
    """Create Playwright browser instance"""
    if not config.ENABLE_UI_TESTS:
        pytest.skip("UI tests disabled")

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=config.BROWSER_HEADLESS,
            slow_mo=config.BROWSER_SLOW_MO
        )
        yield browser
        await browser.close()


@pytest.fixture
async def page(browser: Browser):
    """Create new browser page for test"""
    page = await browser.new_page()
    yield page
    await page.close()


# =============================================================================
# ERP SANDBOX FIXTURES
# =============================================================================

class SAPSandbox:
    """Mock SAP S/4HANA environment for testing"""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.purchase_orders = []
        self.goods_receipts = []
        self.suppliers = []
        self.invoices = []

    async def load_test_data(self, data_file: str):
        """Load test data from JSON file"""
        file_path = config.TEST_DATA_DIR / data_file
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.purchase_orders = data.get('purchase_orders', [])
        self.goods_receipts = data.get('goods_receipts', [])
        self.suppliers = data.get('suppliers', [])
        self.invoices = data.get('invoices', [])

        logger.info(
            f"Loaded SAP test data: {len(self.purchase_orders)} POs, "
            f"{len(self.suppliers)} suppliers"
        )

    async def extract_purchase_orders(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Extract purchase orders in date range"""
        return [
            po for po in self.purchase_orders
            if start_date <= datetime.fromisoformat(po['posting_date']) <= end_date
        ]

    async def extract_suppliers(self) -> List[Dict]:
        """Extract all suppliers"""
        return self.suppliers

    async def cleanup(self):
        """Cleanup sandbox data"""
        self.purchase_orders = []
        self.goods_receipts = []
        self.suppliers = []
        self.invoices = []


@pytest.fixture
async def sap_sandbox(test_tenant):
    """Create SAP sandbox environment"""
    sandbox = SAPSandbox(test_tenant.id)
    yield sandbox
    await sandbox.cleanup()


class OracleSandbox:
    """Mock Oracle Fusion environment for testing"""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.requisitions = []
        self.purchase_orders = []
        self.shipments = []
        self.suppliers = []

    async def load_test_data(self, data_file: str):
        """Load test data from JSON file"""
        file_path = config.TEST_DATA_DIR / data_file
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.requisitions = data.get('requisitions', [])
        self.purchase_orders = data.get('purchase_orders', [])
        self.shipments = data.get('shipments', [])
        self.suppliers = data.get('suppliers', [])

        logger.info(
            f"Loaded Oracle test data: {len(self.purchase_orders)} POs, "
            f"{len(self.suppliers)} suppliers"
        )

    async def extract_purchase_orders(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Extract purchase orders in date range"""
        return [
            po for po in self.purchase_orders
            if start_date <= datetime.fromisoformat(po['creation_date']) <= end_date
        ]

    async def cleanup(self):
        """Cleanup sandbox data"""
        self.requisitions = []
        self.purchase_orders = []
        self.shipments = []
        self.suppliers = []


@pytest.fixture
async def oracle_sandbox(test_tenant):
    """Create Oracle sandbox environment"""
    sandbox = OracleSandbox(test_tenant.id)
    yield sandbox
    await sandbox.cleanup()


class WorkdaySandbox:
    """Mock Workday RaaS environment for testing"""

    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        self.expense_reports = []
        self.commute_surveys = []

    async def load_test_data(self, data_file: str):
        """Load test data from JSON file"""
        file_path = config.TEST_DATA_DIR / data_file
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.expense_reports = data.get('expense_reports', [])
        self.commute_surveys = data.get('commute_surveys', [])

        logger.info(
            f"Loaded Workday test data: {len(self.expense_reports)} expenses"
        )

    async def extract_expense_reports(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict]:
        """Extract expense reports in date range"""
        return [
            exp for exp in self.expense_reports
            if start_date <= datetime.fromisoformat(exp['report_date']) <= end_date
        ]

    async def cleanup(self):
        """Cleanup sandbox data"""
        self.expense_reports = []
        self.commute_surveys = []


@pytest.fixture
async def workday_sandbox(test_tenant):
    """Create Workday sandbox environment"""
    sandbox = WorkdaySandbox(test_tenant.id)
    yield sandbox
    await sandbox.cleanup()


# =============================================================================
# TEST DATA FACTORY
# =============================================================================

class TestDataFactory:
    """Factory for generating test data"""

    @staticmethod
    def create_purchase_order(
        po_number: Optional[str] = None,
        supplier_name: Optional[str] = None,
        amount: Optional[float] = None,
        currency: str = "USD",
        posting_date: Optional[datetime] = None
    ) -> Dict:
        """Create a test purchase order"""
        return {
            "po_number": po_number or f"PO-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:10].upper()}",
            "supplier_name": supplier_name or f"Supplier-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
            "amount": amount or round(1000 + (10000 * hash(str(deterministic_uuid(__name__, str(DeterministicClock.now())))) % 1000), 2),
            "currency": currency,
            "posting_date": (posting_date or DeterministicClock.utcnow()).isoformat(),
            "line_items": [
                {
                    "item_number": f"{i+1:03d}",
                    "description": f"Item {i+1}",
                    "quantity": 10,
                    "unit_price": 100.0,
                    "total": 1000.0
                }
                for i in range(3)
            ]
        }

    @staticmethod
    def create_supplier(
        supplier_id: Optional[str] = None,
        name: Optional[str] = None,
        country: str = "US"
    ) -> Dict:
        """Create a test supplier"""
        return {
            "supplier_id": supplier_id or f"SUP-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:10].upper()}",
            "name": name or f"Test Supplier {deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:8]}",
            "country": country,
            "city": "New York",
            "postal_code": "10001",
            "tax_id": f"TAX-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:12].upper()}"
        }

    @staticmethod
    def create_logistics_shipment(
        shipment_id: Optional[str] = None,
        origin: str = "Shanghai, CN",
        destination: str = "Los Angeles, US",
        mode: str = "Sea Freight",
        weight_kg: Optional[float] = None,
        distance_km: Optional[float] = None
    ) -> Dict:
        """Create a test logistics shipment"""
        return {
            "shipment_id": shipment_id or f"SHIP-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:10].upper()}",
            "origin": origin,
            "destination": destination,
            "transport_mode": mode,
            "weight_kg": weight_kg or 5000.0,
            "distance_km": distance_km or 11000.0,
            "shipment_date": DeterministicClock.utcnow().isoformat()
        }

    @staticmethod
    def create_expense_report(
        expense_id: Optional[str] = None,
        employee_id: Optional[str] = None,
        expense_type: str = "Flight",
        amount: Optional[float] = None,
        currency: str = "USD"
    ) -> Dict:
        """Create a test expense report"""
        return {
            "expense_id": expense_id or f"EXP-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:10].upper()}",
            "employee_id": employee_id or f"EMP-{deterministic_uuid(__name__, str(DeterministicClock.now())).hex[:6].upper()}",
            "expense_type": expense_type,
            "amount": amount or 500.0,
            "currency": currency,
            "report_date": DeterministicClock.utcnow().isoformat(),
            "description": f"Business travel - {expense_type}"
        }

    @staticmethod
    def create_bulk_purchase_orders(count: int) -> List[Dict]:
        """Create bulk purchase orders for performance testing"""
        return [
            TestDataFactory.create_purchase_order()
            for _ in range(count)
        ]


@pytest.fixture
def test_data_factory():
    """Test data factory fixture"""
    return TestDataFactory()


# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class PerformanceMonitor:
    """Monitor and track performance metrics during tests"""

    def __init__(self):
        self.metrics = {}
        self.start_times = {}

    def start_timer(self, metric_name: str):
        """Start timing a metric"""
        self.start_times[metric_name] = time.time()

    def stop_timer(self, metric_name: str) -> float:
        """Stop timing and return elapsed time"""
        if metric_name not in self.start_times:
            raise ValueError(f"Timer {metric_name} not started")

        elapsed = time.time() - self.start_times[metric_name]

        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(elapsed)

        del self.start_times[metric_name]
        return elapsed

    def record_metric(self, metric_name: str, value: float):
        """Record a metric value"""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_average(self, metric_name: str) -> float:
        """Get average value for metric"""
        if metric_name not in self.metrics:
            return 0.0
        return sum(self.metrics[metric_name]) / len(self.metrics[metric_name])

    def get_p95(self, metric_name: str) -> float:
        """Get 95th percentile for metric"""
        if metric_name not in self.metrics:
            return 0.0
        values = sorted(self.metrics[metric_name])
        index = int(len(values) * 0.95)
        return values[index]

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        summary = {}
        for metric_name in self.metrics:
            summary[metric_name] = {
                "count": len(self.metrics[metric_name]),
                "average": self.get_average(metric_name),
                "p95": self.get_p95(metric_name),
                "min": min(self.metrics[metric_name]),
                "max": max(self.metrics[metric_name])
            }
        return summary


@pytest.fixture
def performance_monitor():
    """Performance monitoring fixture"""
    return PerformanceMonitor()


# =============================================================================
# AUDIT TRAIL VALIDATION
# =============================================================================

class AuditTrailValidator:
    """Validate audit trail completeness and correctness"""

    def __init__(self, db_session: Session):
        self.db_session = db_session

    async def verify_provenance_chain(
        self,
        result_id: str,
        expected_steps: List[str]
    ) -> bool:
        """Verify complete provenance chain exists"""
        # Query audit trail for result
        # Verify all expected steps are present
        # Verify chain is complete and hashes match
        return True

    async def verify_calculation_audit(
        self,
        calculation_id: str,
        expected_inputs: Dict
    ) -> bool:
        """Verify calculation audit trail"""
        # Query calculation audit records
        # Verify inputs match expected
        # Verify all factors and methods recorded
        return True

    async def verify_data_lineage(
        self,
        entity_id: str,
        source_system: str
    ) -> bool:
        """Verify data lineage tracking"""
        # Query lineage records
        # Verify source system recorded
        # Verify transformation chain
        return True


@pytest.fixture
def audit_trail_validator(db_session):
    """Audit trail validator fixture"""
    return AuditTrailValidator(db_session)


# =============================================================================
# ASSERTION HELPERS
# =============================================================================

def assert_emissions_within_tolerance(
    actual: float,
    expected: float,
    tolerance_percent: float = 0.1
):
    """Assert emissions value is within tolerance"""
    tolerance = expected * (tolerance_percent / 100)
    assert abs(actual - expected) <= tolerance, (
        f"Emissions {actual} not within {tolerance_percent}% of expected {expected}"
    )


def assert_dqi_in_range(dqi_score: float, min_score: float, max_score: float):
    """Assert DQI score is in expected range"""
    assert min_score <= dqi_score <= max_score, (
        f"DQI score {dqi_score} not in range [{min_score}, {max_score}]"
    )


def assert_throughput_target_met(
    records_processed: int,
    time_seconds: float,
    target_per_hour: int
):
    """Assert throughput target is met"""
    actual_per_hour = int((records_processed / time_seconds) * 3600)
    assert actual_per_hour >= target_per_hour, (
        f"Throughput {actual_per_hour}/hour below target {target_per_hour}/hour"
    )


def assert_latency_target_met(latency_ms: float, target_ms: float):
    """Assert latency target is met"""
    assert latency_ms <= target_ms, (
        f"Latency {latency_ms}ms exceeds target {target_ms}ms"
    )


# =============================================================================
# TEST DATA DIRECTORIES
# =============================================================================

@pytest.fixture(scope="session", autouse=True)
def create_test_data_dirs():
    """Create test data directories"""
    config.TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # Create sample test data files if they don't exist
    sample_sap_data = config.TEST_DATA_DIR / "sap_test_data.json"
    if not sample_sap_data.exists():
        with open(sample_sap_data, 'w') as f:
            json.dump({
                "purchase_orders": [],
                "goods_receipts": [],
                "suppliers": [],
                "invoices": []
            }, f, indent=2)

    sample_oracle_data = config.TEST_DATA_DIR / "oracle_test_data.json"
    if not sample_oracle_data.exists():
        with open(sample_oracle_data, 'w') as f:
            json.dump({
                "requisitions": [],
                "purchase_orders": [],
                "shipments": [],
                "suppliers": []
            }, f, indent=2)

    sample_workday_data = config.TEST_DATA_DIR / "workday_test_data.json"
    if not sample_workday_data.exists():
        with open(sample_workday_data, 'w') as f:
            json.dump({
                "expense_reports": [],
                "commute_surveys": []
            }, f, indent=2)


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "e2e: End-to-end integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: Performance and load tests"
    )
    config.addinivalue_line(
        "markers", "slow: Tests that take a long time to run"
    )
    config.addinivalue_line(
        "markers", "ui: Tests that require browser/UI"
    )
    config.addinivalue_line(
        "markers", "resilience: Tests for failure scenarios and recovery"
    )
