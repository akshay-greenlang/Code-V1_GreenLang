"""
GL-VCCI Load Testing Utilities

Shared utilities for generating test data, monitoring system resources,
and validating performance targets across all load test scenarios.

This module provides:
    - Realistic synthetic data generation
    - System resource monitoring
    - Performance target validation
    - Authentication helpers
    - CSV generation utilities
    - Common test data fixtures

Author: GL-VCCI Team
Version: 1.0.0
"""

import random
import string
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import psutil
import requests
from io import StringIO
import csv as csv_module


# ============================================================================
# DATA GENERATION UTILITIES
# ============================================================================

class RealisticDataGenerator:
    """
    Generates realistic synthetic procurement and emissions data for load testing.

    Data patterns match real-world distributions:
    - Supplier names (fortune 500 style)
    - Product descriptions (industrial materials)
    - Spend amounts (log-normal distribution)
    - Quantities (realistic ranges per product type)
    - Geographic locations (global distribution)
    """

    # Reference data for realistic generation
    SUPPLIER_PREFIXES = [
        "Global", "International", "United", "National", "Advanced",
        "Premier", "Standard", "Industrial", "Commercial", "Superior"
    ]

    SUPPLIER_TYPES = [
        "Steel", "Plastics", "Chemicals", "Electronics", "Textiles",
        "Logistics", "Manufacturing", "Materials", "Components", "Solutions"
    ]

    SUPPLIER_SUFFIXES = [
        "Corp", "Inc", "Ltd", "LLC", "GmbH", "SA", "AG", "PLC", "Co", "Group"
    ]

    PRODUCTS = {
        "steel": ["Steel Plate", "Steel Rod", "Steel Coil", "Steel Beam", "Stainless Steel"],
        "plastic": ["HDPE Resin", "PET Resin", "PP Resin", "PVC Compound", "Polycarbonate"],
        "chemical": ["Sulfuric Acid", "Sodium Hydroxide", "Ammonia", "Ethylene", "Benzene"],
        "electronic": ["PCB Board", "Semiconductors", "Resistors", "Capacitors", "IC Chips"],
        "textile": ["Cotton Fabric", "Polyester Fiber", "Nylon Thread", "Wool Fabric", "Synthetic Blend"]
    }

    COUNTRIES = [
        "USA", "China", "Germany", "Japan", "India", "UK", "France", "Italy",
        "Canada", "South Korea", "Mexico", "Brazil", "Spain", "Netherlands",
        "Poland", "Turkey", "Indonesia", "Thailand", "Vietnam", "Malaysia"
    ]

    CITIES = {
        "USA": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "China": ["Shanghai", "Beijing", "Shenzhen", "Guangzhou", "Chengdu"],
        "Germany": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Stuttgart"],
        "Japan": ["Tokyo", "Osaka", "Nagoya", "Yokohama", "Kyoto"],
        "India": ["Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata"]
    }

    UNITS = {
        "steel": ["kg", "ton", "lb"],
        "plastic": ["kg", "lb", "gallon"],
        "chemical": ["liter", "gallon", "kg"],
        "electronic": ["unit", "piece", "box"],
        "textile": ["meter", "yard", "kg"]
    }

    def __init__(self, seed: Optional[int] = None):
        """Initialize generator with optional seed for reproducibility."""
        if seed:
            random.seed(seed)

    def generate_supplier_name(self) -> str:
        """Generate realistic supplier company name."""
        prefix = random.choice(self.SUPPLIER_PREFIXES)
        type_ = random.choice(self.SUPPLIER_TYPES)
        suffix = random.choice(self.SUPPLIER_SUFFIXES)
        return f"{prefix} {type_} {suffix}"

    def generate_product_data(self) -> Tuple[str, str, str, float, str]:
        """
        Generate product with category, name, unit, quantity, and spend.

        Returns:
            Tuple of (category, product_name, unit, quantity, unit_price)
        """
        category = random.choice(list(self.PRODUCTS.keys()))
        product_name = random.choice(self.PRODUCTS[category])
        unit = random.choice(self.UNITS[category])

        # Log-normal distribution for realistic quantities
        if unit in ["kg", "lb", "ton"]:
            quantity = round(random.lognormvariate(6, 2), 2)  # Mean ~403 kg
        elif unit in ["liter", "gallon"]:
            quantity = round(random.lognormvariate(5, 1.5), 2)  # Mean ~148 liters
        else:
            quantity = round(random.lognormvariate(4, 1), 0)  # Mean ~54 units

        # Realistic unit prices
        if category == "steel":
            unit_price = round(random.uniform(0.5, 3.0), 2)  # $/kg
        elif category == "plastic":
            unit_price = round(random.uniform(1.0, 5.0), 2)
        elif category == "chemical":
            unit_price = round(random.uniform(0.8, 4.0), 2)
        elif category == "electronic":
            unit_price = round(random.uniform(5.0, 50.0), 2)
        else:
            unit_price = round(random.uniform(2.0, 15.0), 2)

        return category, product_name, unit, quantity, unit_price

    def generate_location(self) -> Tuple[str, str]:
        """Generate realistic country and city pair."""
        country = random.choice(self.COUNTRIES)
        city = random.choice(self.CITIES.get(country, ["N/A"]))
        return country, city

    def generate_transaction_id(self) -> str:
        """Generate unique transaction ID."""
        timestamp = int(time.time() * 1000)
        random_suffix = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        return f"TXN-{timestamp}-{random_suffix}"

    def generate_procurement_record(self) -> Dict[str, Any]:
        """
        Generate single realistic procurement record.

        Returns:
            Dictionary with all required fields for procurement ingestion.
        """
        supplier_name = self.generate_supplier_name()
        category, product_name, unit, quantity, unit_price = self.generate_product_data()
        country, city = self.generate_location()

        # Calculate spend
        spend_usd = round(quantity * unit_price, 2)

        # Random dates within last year
        days_ago = random.randint(1, 365)
        transaction_date = (datetime.now() - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        return {
            "transaction_id": self.generate_transaction_id(),
            "transaction_date": transaction_date,
            "supplier_name": supplier_name,
            "supplier_country": country,
            "supplier_city": city,
            "product_name": product_name,
            "product_category": category,
            "quantity": quantity,
            "unit": unit,
            "unit_price": unit_price,
            "spend_usd": spend_usd,
            "currency": "USD",
            "department": random.choice(["Procurement", "Manufacturing", "R&D", "Operations"]),
            "cost_center": f"CC-{random.randint(1000, 9999)}",
            "buyer_email": f"buyer{random.randint(1, 100)}@company.com"
        }


def generate_realistic_procurement_data(n: int, seed: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Generate n realistic procurement records.

    Args:
        n: Number of records to generate
        seed: Optional random seed for reproducibility

    Returns:
        List of procurement record dictionaries

    Example:
        >>> data = generate_realistic_procurement_data(1000)
        >>> len(data)
        1000
        >>> data[0].keys()
        dict_keys(['transaction_id', 'transaction_date', 'supplier_name', ...])
    """
    generator = RealisticDataGenerator(seed=seed)
    return [generator.generate_procurement_record() for _ in range(n)]


def generate_csv_data(n: int, seed: Optional[int] = None) -> str:
    """
    Generate CSV string with n procurement records.

    Args:
        n: Number of records to generate
        seed: Optional random seed for reproducibility

    Returns:
        CSV-formatted string ready for upload

    Example:
        >>> csv_data = generate_csv_data(100)
        >>> len(csv_data.split('\\n'))
        101  # 100 data rows + 1 header row
    """
    records = generate_realistic_procurement_data(n, seed=seed)

    if not records:
        return ""

    # Create CSV in memory
    output = StringIO()
    writer = csv_module.DictWriter(output, fieldnames=records[0].keys())
    writer.writeheader()
    writer.writerows(records)

    return output.getvalue()


# ============================================================================
# AUTHENTICATION UTILITIES
# ============================================================================

def create_test_user_auth(
    base_url: str,
    email: str,
    password: str,
    timeout: int = 10
) -> Optional[str]:
    """
    Authenticate test user and return access token.

    Args:
        base_url: API base URL (e.g., http://localhost:8000)
        email: User email
        password: User password
        timeout: Request timeout in seconds

    Returns:
        Access token string or None if authentication fails

    Example:
        >>> token = create_test_user_auth(
        ...     "http://localhost:8000",
        ...     "loadtest_1@example.com",
        ...     "LoadTest123!"
        ... )
        >>> token[:10]
        'eyJhbGciO'
    """
    try:
        response = requests.post(
            f"{base_url}/api/auth/login",
            json={"email": email, "password": password},
            timeout=timeout
        )
        response.raise_for_status()
        return response.json().get("access_token")
    except Exception as e:
        print(f"Authentication failed for {email}: {e}")
        return None


# ============================================================================
# SYSTEM MONITORING UTILITIES
# ============================================================================

class SystemMonitor:
    """
    Monitor system resources during load testing.

    Tracks:
        - CPU utilization (per core and overall)
        - Memory usage (RSS, VMS, available)
        - Disk I/O (read/write ops)
        - Network I/O (bytes sent/received)
        - Database connections (if monitoring enabled)
    """

    def __init__(self, process_name: Optional[str] = None):
        """
        Initialize system monitor.

        Args:
            process_name: Optional process name to monitor specifically
        """
        self.process_name = process_name
        self.start_time = time.time()
        self.initial_net_io = psutil.net_io_counters()
        self.initial_disk_io = psutil.disk_io_counters()

    def get_current_stats(self) -> Dict[str, Any]:
        """
        Get current system resource statistics.

        Returns:
            Dictionary with current resource usage metrics
        """
        # CPU stats
        cpu_percent = psutil.cpu_percent(interval=1, percpu=False)
        cpu_percent_per_core = psutil.cpu_percent(interval=1, percpu=True)

        # Memory stats
        memory = psutil.virtual_memory()

        # Disk I/O
        disk_io = psutil.disk_io_counters()
        disk_read_mb = (disk_io.read_bytes - self.initial_disk_io.read_bytes) / (1024 * 1024)
        disk_write_mb = (disk_io.write_bytes - self.initial_disk_io.write_bytes) / (1024 * 1024)

        # Network I/O
        net_io = psutil.net_io_counters()
        net_sent_mb = (net_io.bytes_sent - self.initial_net_io.bytes_sent) / (1024 * 1024)
        net_recv_mb = (net_io.bytes_recv - self.initial_net_io.bytes_recv) / (1024 * 1024)

        return {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": time.time() - self.start_time,
            "cpu": {
                "percent_overall": cpu_percent,
                "percent_per_core": cpu_percent_per_core,
                "count": psutil.cpu_count()
            },
            "memory": {
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3),
                "used_gb": memory.used / (1024**3),
                "percent": memory.percent
            },
            "disk_io": {
                "read_mb": round(disk_read_mb, 2),
                "write_mb": round(disk_write_mb, 2)
            },
            "network_io": {
                "sent_mb": round(net_sent_mb, 2),
                "recv_mb": round(net_recv_mb, 2)
            }
        }


def monitor_system_resources() -> Dict[str, Any]:
    """
    Get current system resource snapshot.

    Returns:
        Dictionary with CPU, memory, disk, and network stats

    Example:
        >>> stats = monitor_system_resources()
        >>> stats['cpu']['percent_overall'] < 100
        True
        >>> stats['memory']['percent'] < 100
        True
    """
    monitor = SystemMonitor()
    return monitor.get_current_stats()


# ============================================================================
# PERFORMANCE VALIDATION UTILITIES
# ============================================================================

class PerformanceValidator:
    """
    Validate load test results against performance targets.

    Targets:
        - p95 latency: < 200ms for aggregates
        - p99 latency: < 500ms for aggregates
        - Error rate: < 0.1%
        - Throughput: 1000+ RPS sustained
        - CPU utilization: < 70% (headroom)
        - Memory: stable (no leaks)
    """

    TARGETS = {
        "p95_latency_ms": 200,
        "p99_latency_ms": 500,
        "error_rate_pct": 0.1,
        "throughput_rps": 1000,
        "cpu_percent_max": 70,
        "memory_growth_mb_per_hour": 100  # Max acceptable memory growth
    }

    def __init__(self, custom_targets: Optional[Dict[str, float]] = None):
        """
        Initialize validator with optional custom targets.

        Args:
            custom_targets: Dictionary to override default targets
        """
        self.targets = self.TARGETS.copy()
        if custom_targets:
            self.targets.update(custom_targets)

    def validate_latency(self, p95: float, p99: float) -> Dict[str, Any]:
        """Validate latency metrics."""
        return {
            "p95": {
                "value": p95,
                "target": self.targets["p95_latency_ms"],
                "passed": p95 <= self.targets["p95_latency_ms"]
            },
            "p99": {
                "value": p99,
                "target": self.targets["p99_latency_ms"],
                "passed": p99 <= self.targets["p99_latency_ms"]
            }
        }

    def validate_error_rate(self, total_requests: int, failed_requests: int) -> Dict[str, Any]:
        """Validate error rate."""
        error_rate_pct = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        return {
            "error_rate_pct": error_rate_pct,
            "target": self.targets["error_rate_pct"],
            "passed": error_rate_pct <= self.targets["error_rate_pct"],
            "total_requests": total_requests,
            "failed_requests": failed_requests
        }

    def validate_throughput(self, rps: float) -> Dict[str, Any]:
        """Validate throughput (requests per second)."""
        return {
            "rps": rps,
            "target": self.targets["throughput_rps"],
            "passed": rps >= self.targets["throughput_rps"]
        }

    def validate_all(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate all performance metrics.

        Args:
            results: Dictionary with test results including:
                - p95_latency_ms
                - p99_latency_ms
                - total_requests
                - failed_requests
                - rps

        Returns:
            Validation report with pass/fail for each metric
        """
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "latency": self.validate_latency(
                results.get("p95_latency_ms", 0),
                results.get("p99_latency_ms", 0)
            ),
            "error_rate": self.validate_error_rate(
                results.get("total_requests", 0),
                results.get("failed_requests", 0)
            ),
            "throughput": self.validate_throughput(results.get("rps", 0))
        }

        # Overall pass/fail
        all_passed = all([
            validation_report["latency"]["p95"]["passed"],
            validation_report["latency"]["p99"]["passed"],
            validation_report["error_rate"]["passed"],
            validation_report["throughput"]["passed"]
        ])

        validation_report["overall_passed"] = all_passed

        return validation_report


def validate_performance_targets(results: Dict[str, Any]) -> bool:
    """
    Validate load test results against performance targets.

    Args:
        results: Dictionary with test results

    Returns:
        True if all targets met, False otherwise

    Example:
        >>> results = {
        ...     "p95_latency_ms": 180,
        ...     "p99_latency_ms": 450,
        ...     "total_requests": 10000,
        ...     "failed_requests": 5,
        ...     "rps": 1200
        ... }
        >>> validate_performance_targets(results)
        True
    """
    validator = PerformanceValidator()
    report = validator.validate_all(results)
    return report["overall_passed"]
