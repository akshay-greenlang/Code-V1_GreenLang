# -*- coding: utf-8 -*-
"""
GreenLang Mock Services

Comprehensive mock services for testing external dependencies:
- Mock Emission Factor Database
- Mock ERP Connector
- Mock Regulatory API
- Mock Cache Layer
- Mock Provenance Store

Author: GL-TestEngineer
Version: 1.0.0
"""

import asyncio
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock


# =============================================================================
# Mock Emission Factor Database
# =============================================================================

class MockEmissionFactorDB:
    """
    Mock emission factor database for testing.

    Provides deterministic emission factors from authoritative sources:
    - EPA (US)
    - DEFRA (UK)
    - IPCC (Global)
    """

    def __init__(self):
        self._factors = {
            # Natural Gas
            ("natural_gas", "US", 2023): {
                "ef_uri": "ef://EPA/natural_gas/US/2023",
                "ef_value": 0.0561,
                "ef_unit": "kgCO2e/MJ",
                "source": "EPA 40 CFR 98 Table C-1",
                "gwp_set": "AR6GWP100",
                "uncertainty": 0.05,
            },
            ("natural_gas", "GB", 2023): {
                "ef_uri": "ef://DEFRA/natural_gas/GB/2023",
                "ef_value": 0.0549,
                "ef_unit": "kgCO2e/MJ",
                "source": "UK DEFRA 2023",
                "gwp_set": "AR6GWP100",
                "uncertainty": 0.04,
            },
            # Diesel
            ("diesel", "US", 2023): {
                "ef_uri": "ef://EPA/diesel/US/2023",
                "ef_value": 0.0745,
                "ef_unit": "kgCO2e/MJ",
                "source": "EPA 40 CFR 98",
                "gwp_set": "AR6GWP100",
                "uncertainty": 0.03,
            },
            ("diesel", "GB", 2023): {
                "ef_uri": "ef://DEFRA/diesel/GB/2023",
                "ef_value": 0.0732,
                "ef_unit": "kgCO2e/MJ",
                "source": "UK DEFRA 2023",
                "gwp_set": "AR6GWP100",
                "uncertainty": 0.03,
            },
            # Electricity
            ("electricity", "US", 2023): {
                "ef_uri": "ef://eGRID/electricity/US/2023",
                "ef_value": 0.417,
                "ef_unit": "kgCO2e/kWh",
                "source": "EPA eGRID 2023",
                "gwp_set": "AR6GWP100",
                "uncertainty": 0.10,
            },
            ("electricity", "GB", 2023): {
                "ef_uri": "ef://DEFRA/electricity/GB/2023",
                "ef_value": 0.207,
                "ef_unit": "kgCO2e/kWh",
                "source": "UK DEFRA 2023",
                "gwp_set": "AR6GWP100",
                "uncertainty": 0.05,
            },
        }

    def lookup(self, fuel_type: str, region: str, year: int) -> Optional[Dict[str, Any]]:
        """Look up emission factor."""
        key = (fuel_type.lower(), region.upper(), year)
        return self._factors.get(key)

    def insert(self, fuel_type: str, region: str, year: int, record: Dict[str, Any]) -> bool:
        """Insert new emission factor."""
        key = (fuel_type.lower(), region.upper(), year)
        self._factors[key] = record
        return True

    def list_all(self) -> List[tuple]:
        """List all emission factor keys."""
        return list(self._factors.keys())


# =============================================================================
# Mock ERP Connector
# =============================================================================

class MockERPConnector:
    """
    Mock ERP connector for testing.

    Simulates SAP/Oracle-style ERP integration.
    """

    def __init__(self):
        self.connected = False
        self.session_id = None
        self._transactions = []
        self._fuel_purchases = [
            {
                "document_id": "PO-2024-001",
                "fuel_type": "diesel",
                "quantity": 5000,
                "unit": "L",
                "supplier": "Fuel Corp",
                "cost_center": "CC-001",
                "date": "2024-01-15",
            },
            {
                "document_id": "PO-2024-002",
                "fuel_type": "natural_gas",
                "quantity": 10000,
                "unit": "MJ",
                "supplier": "Gas Inc",
                "cost_center": "CC-001",
                "date": "2024-01-20",
            },
        ]
        self._energy_bills = [
            {
                "bill_id": "BILL-2024-001",
                "facility_id": "FAC-001",
                "energy_type": "electricity",
                "consumption_kwh": 50000,
                "billing_period": "2024-01",
                "provider": "Power Co",
            },
        ]

    async def connect(self, host: str, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Connect to ERP system."""
        if not credentials.get("api_key"):
            raise ConnectionError("Invalid credentials")

        self.connected = True
        self.session_id = f"ERP-SESSION-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        return {"status": "connected", "session_id": self.session_id}

    async def disconnect(self) -> Dict[str, Any]:
        """Disconnect from ERP system."""
        self.connected = False
        self.session_id = None
        return {"status": "disconnected"}

    async def fetch_fuel_purchases(
        self,
        date_range: tuple,
        cost_center: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch fuel purchase records."""
        if not self.connected:
            raise ConnectionError("Not connected to ERP")

        results = self._fuel_purchases.copy()
        if cost_center:
            results = [r for r in results if r.get("cost_center") == cost_center]

        return results

    async def fetch_energy_bills(
        self,
        date_range: tuple,
        facility_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Fetch energy bill records."""
        if not self.connected:
            raise ConnectionError("Not connected to ERP")

        results = self._energy_bills.copy()
        if facility_id:
            results = [r for r in results if r.get("facility_id") == facility_id]

        return results

    async def post_emissions_data(self, emissions_record: Dict[str, Any]) -> Dict[str, Any]:
        """Post emissions data to ERP."""
        if not self.connected:
            raise ConnectionError("Not connected to ERP")

        self._transactions.append(emissions_record)

        return {
            "status": "posted",
            "transaction_id": f"EMI-{len(self._transactions):04d}",
            "timestamp": datetime.now().isoformat(),
        }


# =============================================================================
# Mock Regulatory API
# =============================================================================

class MockRegulatoryAPI:
    """
    Mock regulatory data API for testing.

    Simulates EU CBAM and other regulatory APIs.
    """

    def __init__(self):
        self.authenticated = False
        self._cbam_benchmarks = {
            "steel_hot_rolled_coil": {
                "benchmark_value": 1.85,
                "unit": "tCO2e/tonne",
                "effective_date": "2026-01-01",
                "regulation": "EU 2023/1773",
            },
            "steel_rebar": {
                "benchmark_value": 1.35,
                "unit": "tCO2e/tonne",
                "effective_date": "2026-01-01",
                "regulation": "EU 2023/1773",
            },
            "cement_clinker": {
                "benchmark_value": 0.766,
                "unit": "tCO2e/tonne",
                "effective_date": "2026-01-01",
                "regulation": "EU 2023/1773",
            },
            "cement_portland": {
                "benchmark_value": 0.670,
                "unit": "tCO2e/tonne",
                "effective_date": "2026-01-01",
                "regulation": "EU 2023/1773",
            },
            "aluminum_unwrought": {
                "benchmark_value": 8.6,
                "unit": "tCO2e/tonne",
                "effective_date": "2026-01-01",
                "regulation": "EU 2023/1773",
            },
        }

    async def authenticate(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Authenticate with regulatory API."""
        if not credentials.get("cert_path"):
            raise ValueError("Certificate required")

        self.authenticated = True
        return {"status": "authenticated", "expires_at": "2024-12-31"}

    async def get_cbam_benchmark(self, product_type: str) -> Dict[str, Any]:
        """Get CBAM benchmark for product type."""
        if not self.authenticated:
            raise ValueError("Not authenticated")

        benchmark = self._cbam_benchmarks.get(product_type)
        if not benchmark:
            raise ValueError(f"Unknown product type: {product_type}")

        return {"product_type": product_type, **benchmark}

    async def submit_cbam_declaration(self, declaration: Dict[str, Any]) -> Dict[str, Any]:
        """Submit CBAM declaration."""
        if not self.authenticated:
            raise ValueError("Not authenticated")

        return {
            "declaration_id": f"CBAM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "status": "submitted",
            "submitted_at": datetime.now().isoformat(),
        }


# =============================================================================
# Mock Cache Layer
# =============================================================================

class MockCacheLayer:
    """
    Mock cache layer for testing.

    Simulates Redis-style caching with TTL support.
    """

    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._stats = {"hits": 0, "misses": 0}

    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        if key in self._cache:
            entry = self._cache[key]
            if entry["expires_at"] > datetime.now():
                self._stats["hits"] += 1
                return entry["value"]
            else:
                del self._cache[key]

        self._stats["misses"] += 1
        return None

    async def set(self, key: str, value: Any, ttl_seconds: int = 300) -> bool:
        """Set value in cache with TTL."""
        self._cache[key] = {
            "value": value,
            "expires_at": datetime.now() + timedelta(seconds=ttl_seconds),
        }
        return True

    async def delete(self, key: str) -> bool:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            return True
        return False

    async def clear(self) -> bool:
        """Clear all cache entries."""
        self._cache.clear()
        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total if total > 0 else 0
        return {**self._stats, "hit_rate": hit_rate}


# =============================================================================
# Mock Provenance Store
# =============================================================================

class MockProvenanceStore:
    """
    Mock provenance store for testing.

    Stores calculation provenance with SHA-256 hashes.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def save(self, provenance_hash: str, data: Dict[str, Any]) -> str:
        """Save provenance record."""
        self._store[provenance_hash] = {
            "data": data,
            "created_at": datetime.now().isoformat(),
        }
        return provenance_hash

    async def get(self, provenance_hash: str) -> Optional[Dict[str, Any]]:
        """Get provenance record by hash."""
        return self._store.get(provenance_hash)

    async def verify(self, provenance_hash: str, expected_data: Dict[str, Any]) -> bool:
        """Verify provenance integrity."""
        stored = self._store.get(provenance_hash)
        if not stored:
            return False

        actual_hash = hashlib.sha256(
            json.dumps(stored["data"], sort_keys=True).encode()
        ).hexdigest()
        expected_hash = hashlib.sha256(
            json.dumps(expected_data, sort_keys=True).encode()
        ).hexdigest()

        return actual_hash == expected_hash

    async def list_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """List provenance records by date range."""
        results = []
        for hash_val, entry in self._store.items():
            entry_date = datetime.fromisoformat(entry["created_at"])
            if start_date <= entry_date <= end_date:
                results.append({"hash": hash_val, **entry})
        return results


# =============================================================================
# Factory Functions
# =============================================================================

def create_mock_emission_factor_db() -> MockEmissionFactorDB:
    """Create a mock emission factor database."""
    return MockEmissionFactorDB()


def create_mock_erp_connector() -> MockERPConnector:
    """Create a mock ERP connector."""
    return MockERPConnector()


def create_mock_regulatory_api() -> MockRegulatoryAPI:
    """Create a mock regulatory API."""
    return MockRegulatoryAPI()


def create_mock_cache() -> MockCacheLayer:
    """Create a mock cache layer."""
    return MockCacheLayer()


def create_mock_provenance_store() -> MockProvenanceStore:
    """Create a mock provenance store."""
    return MockProvenanceStore()


# =============================================================================
# Pytest Fixtures (for import in conftest.py)
# =============================================================================

def get_mock_fixtures():
    """Return dict of fixture factories for use in conftest.py."""
    return {
        "mock_emission_factor_db": create_mock_emission_factor_db,
        "mock_erp_connector": create_mock_erp_connector,
        "mock_regulatory_api": create_mock_regulatory_api,
        "mock_cache": create_mock_cache,
        "mock_provenance_store": create_mock_provenance_store,
    }
