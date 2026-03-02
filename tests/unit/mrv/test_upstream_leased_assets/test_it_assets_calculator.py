# -*- coding: utf-8 -*-
"""
Unit tests for ITAssetsCalculatorEngine (AGENT-MRV-021, Engine 5)

35 tests covering server (PUE, utilization), network switch, storage array,
desktop, laptop, printer (active+standby), copier, data center allocation,
IT portfolio aggregation, batch, known values, and provenance.

Calculation:
    IT emissions = rated_power_W * utilization * PUE * hours / 1000 * grid_ef
    Printers:    = (active_power * active_hours + standby_power * standby_hours)
                   / 1000 * grid_ef

Author: GL-TestEngineer
Date: February 2026
"""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import MagicMock, patch
import pytest

try:
    from greenlang.upstream_leased_assets.it_assets_calculator import (
        ITAssetsCalculatorEngine,
    )
    from greenlang.upstream_leased_assets.models import (
        ITAssetType,
    )
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False

_SKIP = pytest.mark.skipif(
    not _AVAILABLE,
    reason="ITAssetsCalculatorEngine not available",
)

pytestmark = _SKIP


# ==============================================================================
# FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset singleton before and after every test."""
    if _AVAILABLE:
        ITAssetsCalculatorEngine.reset_instance()
    yield
    if _AVAILABLE:
        ITAssetsCalculatorEngine.reset_instance()


@pytest.fixture
def engine():
    """Create a fresh ITAssetsCalculatorEngine."""
    return ITAssetsCalculatorEngine()


# ==============================================================================
# SERVER CALCULATION TESTS
# ==============================================================================


class TestServerCalculation:
    """Test server emission calculations with PUE and utilization."""

    def test_server_basic(self, engine):
        """Test basic server calculation: 500W, 90% util, PUE 1.4."""
        result = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0
        assert isinstance(result["total_co2e_kg"], Decimal)

    def test_server_pue_increases_emissions(self, engine):
        """Test higher PUE produces more emissions."""
        low_pue = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.20"),
            "annual_hours": 8760,
            "region": "US",
        })
        high_pue = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.80"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert high_pue["total_co2e_kg"] > low_pue["total_co2e_kg"]

    def test_server_utilization_scales(self, engine):
        """Test higher utilization produces more emissions."""
        low_util = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.30"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        high_util = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert high_util["total_co2e_kg"] > low_util["total_co2e_kg"]

    def test_server_known_value(self, engine):
        """Test known value: 500W * 0.9 * 1.4 * 8760h / 1000 * 0.37170."""
        result = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        # 500 * 0.9 * 1.4 = 630W effective
        # 630 * 8760 / 1000 = 5518.8 kWh
        # 5518.8 * 0.37170 = ~2050 kg
        assert Decimal("1500") < result["total_co2e_kg"] < Decimal("3000")

    def test_provenance_hash_deterministic(self, engine):
        """Test provenance hash is deterministic."""
        inp = {
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        }
        r1 = engine.calculate(inp)
        r2 = engine.calculate(inp)
        assert r1["provenance_hash"] == r2["provenance_hash"]
        assert len(r1["provenance_hash"]) == 64


# ==============================================================================
# NETWORK AND STORAGE TESTS
# ==============================================================================


class TestNetworkStorageCalculation:
    """Test network switch and storage array calculations."""

    def test_network_switch(self, engine):
        """Test network switch: 350W, 80% util, PUE 1.4, 24/7."""
        result = engine.calculate({
            "it_type": "network",
            "rated_power_w": Decimal("350"),
            "utilization_pct": Decimal("0.80"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_storage_array(self, engine):
        """Test storage array: 800W, 70% util, PUE 1.3."""
        result = engine.calculate({
            "it_type": "storage",
            "rated_power_w": Decimal("800"),
            "utilization_pct": Decimal("0.70"),
            "pue": Decimal("1.30"),
            "annual_hours": 8760,
            "region": "DE",
        })
        assert result["total_co2e_kg"] > 0

    def test_storage_higher_than_network(self, engine):
        """Test storage array (800W) emits more than network (350W) at same util."""
        network = engine.calculate({
            "it_type": "network",
            "rated_power_w": Decimal("350"),
            "utilization_pct": Decimal("0.80"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        storage = engine.calculate({
            "it_type": "storage",
            "rated_power_w": Decimal("800"),
            "utilization_pct": Decimal("0.80"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert storage["total_co2e_kg"] > network["total_co2e_kg"]


# ==============================================================================
# DESKTOP AND LAPTOP TESTS
# ==============================================================================


class TestDesktopLaptopCalculation:
    """Test desktop and laptop calculations."""

    def test_desktop(self, engine):
        """Test desktop: 200W, 50% util, standard hours."""
        result = engine.calculate({
            "it_type": "desktop",
            "rated_power_w": Decimal("200"),
            "utilization_pct": Decimal("0.50"),
            "pue": Decimal("1.00"),
            "annual_hours": 2080,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_laptop(self, engine):
        """Test laptop: 65W, 60% util, standard hours."""
        result = engine.calculate({
            "it_type": "laptop",
            "rated_power_w": Decimal("65"),
            "utilization_pct": Decimal("0.60"),
            "pue": Decimal("1.00"),
            "annual_hours": 2080,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_laptop_lower_than_desktop(self, engine):
        """Test laptop emits less than desktop."""
        desktop = engine.calculate({
            "it_type": "desktop",
            "rated_power_w": Decimal("200"),
            "utilization_pct": Decimal("0.50"),
            "pue": Decimal("1.00"),
            "annual_hours": 2080,
            "region": "US",
        })
        laptop = engine.calculate({
            "it_type": "laptop",
            "rated_power_w": Decimal("65"),
            "utilization_pct": Decimal("0.60"),
            "pue": Decimal("1.00"),
            "annual_hours": 2080,
            "region": "US",
        })
        assert laptop["total_co2e_kg"] < desktop["total_co2e_kg"]


# ==============================================================================
# PRINTER AND COPIER TESTS
# ==============================================================================


class TestPrinterCopierCalculation:
    """Test printer and copier calculations with active+standby modes."""

    def test_printer_active_and_standby(self, engine):
        """Test printer with active and standby power modes."""
        result = engine.calculate({
            "it_type": "printer",
            "rated_power_w": Decimal("100"),
            "standby_power_w": Decimal("10"),
            "utilization_pct": Decimal("0.30"),
            "pue": Decimal("1.00"),
            "annual_hours": 2080,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    def test_copier(self, engine):
        """Test copier calculation."""
        result = engine.calculate({
            "it_type": "copier",
            "rated_power_w": Decimal("150"),
            "standby_power_w": Decimal("15"),
            "utilization_pct": Decimal("0.25"),
            "pue": Decimal("1.00"),
            "annual_hours": 2080,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# DATA CENTER ALLOCATION TESTS
# ==============================================================================


class TestDataCenterAllocation:
    """Test data center IT asset allocation."""

    def test_server_with_allocation(self, engine):
        """Test server with data center allocation share."""
        result = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "allocation_share": Decimal("0.25"),
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0


# ==============================================================================
# IT PORTFOLIO AGGREGATION TESTS
# ==============================================================================


class TestITPortfolioAggregation:
    """Test IT portfolio aggregation."""

    def test_portfolio_batch(self, engine):
        """Test batch processing of multiple IT asset types."""
        assets = [
            {
                "it_type": "server",
                "rated_power_w": Decimal("500"),
                "utilization_pct": Decimal("0.90"),
                "pue": Decimal("1.40"),
                "annual_hours": 8760,
                "region": "US",
            },
            {
                "it_type": "desktop",
                "rated_power_w": Decimal("200"),
                "utilization_pct": Decimal("0.50"),
                "pue": Decimal("1.00"),
                "annual_hours": 2080,
                "region": "US",
            },
            {
                "it_type": "laptop",
                "rated_power_w": Decimal("65"),
                "utilization_pct": Decimal("0.60"),
                "pue": Decimal("1.00"),
                "annual_hours": 2080,
                "region": "US",
            },
        ]
        results = engine.calculate_batch(assets)
        assert len(results) == 3
        assert all(r["total_co2e_kg"] > 0 for r in results)


# ==============================================================================
# EDGE CASES
# ==============================================================================


class TestITEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_power_raises_error(self, engine):
        """Test zero rated power raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "it_type": "server",
                "rated_power_w": Decimal("0"),
                "utilization_pct": Decimal("0.90"),
                "pue": Decimal("1.40"),
                "annual_hours": 8760,
                "region": "US",
            })

    def test_pue_below_one_raises_error(self, engine):
        """Test PUE below 1.0 raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "it_type": "server",
                "rated_power_w": Decimal("500"),
                "utilization_pct": Decimal("0.90"),
                "pue": Decimal("0.80"),
                "annual_hours": 8760,
                "region": "US",
            })

    def test_utilization_over_one_raises_error(self, engine):
        """Test utilization over 100% raises error."""
        with pytest.raises((ValueError, Exception)):
            engine.calculate({
                "it_type": "server",
                "rated_power_w": Decimal("500"),
                "utilization_pct": Decimal("1.50"),
                "pue": Decimal("1.40"),
                "annual_hours": 8760,
                "region": "US",
            })

    def test_different_regions_different_results(self, engine):
        """Test different grid regions produce different results."""
        us = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        fr = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "FR",
        })
        assert fr["total_co2e_kg"] < us["total_co2e_kg"]


# ==============================================================================
# PARAMETRIZED IT TESTS
# ==============================================================================


class TestITParametrized:
    """Parametrized tests for exhaustive IT asset coverage."""

    @pytest.mark.parametrize("pue", [
        Decimal("1.10"), Decimal("1.20"), Decimal("1.40"),
        Decimal("1.58"), Decimal("1.80"), Decimal("2.00"),
    ])
    def test_server_various_pue_values(self, engine, pue):
        """Test server with various PUE values."""
        result = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": pue,
            "annual_hours": 8760,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("utilization", [
        Decimal("0.10"), Decimal("0.30"), Decimal("0.50"),
        Decimal("0.70"), Decimal("0.90"), Decimal("1.00"),
    ])
    def test_server_various_utilization(self, engine, utilization):
        """Test server with various utilization percentages."""
        result = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": utilization,
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": "US",
        })
        assert result["total_co2e_kg"] > 0

    @pytest.mark.parametrize("region", [
        "US", "GB", "DE", "FR", "JP", "IN", "CN",
    ])
    def test_server_multiple_regions(self, engine, region):
        """Test server across multiple grid regions."""
        result = engine.calculate({
            "it_type": "server",
            "rated_power_w": Decimal("500"),
            "utilization_pct": Decimal("0.90"),
            "pue": Decimal("1.40"),
            "annual_hours": 8760,
            "region": region,
        })
        assert result["total_co2e_kg"] > 0
