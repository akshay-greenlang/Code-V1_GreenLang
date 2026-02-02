"""
Integration Tests: GL-003 UnifiedSteam Interface

Tests the integration between GL-012 SteamQual and GL-003 UnifiedSteam:
1. Property calculation interface
2. Saturation data exchange
3. Optimization recommendations
4. Data format compatibility
5. Provenance chain tracking

Author: GL-TestEngineer
Version: 1.0.0
Target Coverage: 85%+
"""

import pytest
import math
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch
from enum import Enum


# =============================================================================
# GL-003 Interface Simulation
# =============================================================================

@dataclass
class GL003PropertyRequest:
    """Request for steam properties from GL-003."""
    pressure_mpa: float
    temperature_k: float
    quality: Optional[float] = None
    request_id: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
        if not self.request_id:
            self.request_id = hashlib.sha256(
                f"{self.pressure_mpa}_{self.temperature_k}_{self.timestamp.isoformat()}".encode()
            ).hexdigest()[:16]


@dataclass
class GL003PropertyResponse:
    """Response with steam properties from GL-003."""
    request_id: str
    pressure_mpa: float
    temperature_k: float
    enthalpy_kj_kg: float
    entropy_kj_kg_k: float
    specific_volume_m3_kg: float
    internal_energy_kj_kg: float
    state: str  # "LIQUID", "TWO_PHASE", "VAPOR", "SUPERCRITICAL"
    quality: Optional[float]
    provenance_hash: str
    calculation_time_ms: float
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)


@dataclass
class GL003SaturationRequest:
    """Request for saturation properties from GL-003."""
    pressure_mpa: Optional[float] = None
    temperature_k: Optional[float] = None
    request_id: str = ""


@dataclass
class GL003SaturationResponse:
    """Response with saturation properties from GL-003."""
    request_id: str
    pressure_mpa: float
    temperature_k: float
    h_f_kj_kg: float
    h_g_kj_kg: float
    h_fg_kj_kg: float
    s_f_kj_kg_k: float
    s_g_kj_kg_k: float
    s_fg_kj_kg_k: float
    v_f_m3_kg: float
    v_g_m3_kg: float
    provenance_hash: str


@dataclass
class GL003Recommendation:
    """Optimization recommendation from GL-003."""
    recommendation_id: str
    recommendation_type: str
    asset_id: str
    description: str
    expected_benefit: float
    confidence_level: float
    priority: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"


# =============================================================================
# GL-003 Interface Adapter
# =============================================================================

class GL003InterfaceError(Exception):
    """Error in GL-003 interface communication."""
    pass


class GL003InterfaceAdapter:
    """
    Adapter for communication with GL-003 UnifiedSteam.

    This adapter handles:
    - Property calculations
    - Saturation data requests
    - Recommendation retrieval
    - Data format translation
    - Provenance chain tracking
    """

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize adapter with configuration."""
        self.config = config or {}
        self.base_url = self.config.get("base_url", "http://localhost:8003")
        self.timeout_s = self.config.get("timeout_s", 5.0)
        self._connected = False

    def connect(self) -> bool:
        """Connect to GL-003 service."""
        # In production, this would establish actual connection
        self._connected = True
        return True

    def disconnect(self) -> bool:
        """Disconnect from GL-003 service."""
        self._connected = False
        return True

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def get_properties(self, request: GL003PropertyRequest) -> GL003PropertyResponse:
        """
        Get steam properties from GL-003.

        Args:
            request: Property request

        Returns:
            Property response with calculated values
        """
        if not self._connected:
            raise GL003InterfaceError("Not connected to GL-003")

        import time
        start_time = time.perf_counter()

        # Simulate property calculation (in production, this calls GL-003)
        P = request.pressure_mpa
        T = request.temperature_k

        # Approximate saturation temperature
        T_sat = 453.0 + 50.0 * math.log(max(P, 0.001))

        # Determine state
        if T < T_sat - 0.5:
            state = "LIQUID"
            quality = None
            h = 4.186 * (T - 273.15)
            s = 4.186 * math.log(max(T / 273.15, 0.001))
            v = 0.001 * (1 + 0.0001 * (T - 273.15))
        elif T > T_sat + 0.5:
            state = "VAPOR"
            quality = None
            h = 2800 + 2.0 * (T - T_sat)
            s = 7.0 + 0.002 * (T - T_sat)
            v = 0.461526 * T / (P * 1000) if P > 0 else 1.0
        else:
            state = "TWO_PHASE"
            quality = request.quality if request.quality is not None else 0.9
            h_f = 4.186 * (T_sat - 273.15)
            h_fg = 2000 - 50 * math.log(max(P, 0.001))
            h = h_f + quality * h_fg
            s = 2.0 + quality * 4.5
            v = 0.001 + quality * 0.2

        u = h - P * 1000 * v

        calc_time = (time.perf_counter() - start_time) * 1000

        # Generate provenance hash
        hash_data = {
            "request_id": request.request_id,
            "P": round(P, 10),
            "T": round(T, 10),
            "h": round(h, 10),
        }
        provenance_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

        return GL003PropertyResponse(
            request_id=request.request_id,
            pressure_mpa=P,
            temperature_k=T,
            enthalpy_kj_kg=h,
            entropy_kj_kg_k=s,
            specific_volume_m3_kg=v,
            internal_energy_kj_kg=u,
            state=state,
            quality=quality if state == "TWO_PHASE" else None,
            provenance_hash=provenance_hash,
            calculation_time_ms=calc_time,
        )

    def get_saturation_properties(self, request: GL003SaturationRequest) -> GL003SaturationResponse:
        """
        Get saturation properties from GL-003.

        Args:
            request: Saturation request (by P or T)

        Returns:
            Saturation response with properties
        """
        if not self._connected:
            raise GL003InterfaceError("Not connected to GL-003")

        # Determine T and P
        if request.pressure_mpa is not None:
            P = request.pressure_mpa
            T = 453.0 + 50.0 * math.log(max(P, 0.001))
        elif request.temperature_k is not None:
            T = request.temperature_k
            P = math.exp((T - 453.0) / 50.0)
        else:
            raise GL003InterfaceError("Either pressure or temperature must be specified")

        # Calculate saturation properties
        h_f = 4.186 * (T - 273.15)
        h_fg = 2000 - 50 * math.log(max(P, 0.001))
        h_g = h_f + h_fg

        s_f = 4.186 * math.log(max(T / 273.15, 0.001))
        s_fg = 4.5 - 0.5 * math.log(max(P, 0.001))
        s_g = s_f + s_fg

        v_f = 0.001 * (1 + 0.0001 * (T - 273.15))
        v_g = 0.461526 * T / (P * 1000) if P > 0 else 1.0

        # Generate provenance hash
        hash_data = {"P": round(P, 10), "T": round(T, 10)}
        provenance_hash = hashlib.sha256(
            json.dumps(hash_data, sort_keys=True).encode()
        ).hexdigest()

        return GL003SaturationResponse(
            request_id=request.request_id,
            pressure_mpa=P,
            temperature_k=T,
            h_f_kj_kg=h_f,
            h_g_kj_kg=h_g,
            h_fg_kj_kg=h_fg,
            s_f_kj_kg_k=s_f,
            s_g_kj_kg_k=s_g,
            s_fg_kj_kg_k=s_fg,
            v_f_m3_kg=v_f,
            v_g_m3_kg=v_g,
            provenance_hash=provenance_hash,
        )

    def get_recommendations(self, asset_ids: List[str] = None) -> List[GL003Recommendation]:
        """
        Get optimization recommendations from GL-003.

        Args:
            asset_ids: Optional list of asset IDs to filter

        Returns:
            List of recommendations
        """
        if not self._connected:
            raise GL003InterfaceError("Not connected to GL-003")

        # Simulate recommendations (in production, this queries GL-003)
        return [
            GL003Recommendation(
                recommendation_id="REC-001",
                recommendation_type="SEPARATOR_EFFICIENCY",
                asset_id="SEP-001",
                description="Consider separator maintenance to improve efficiency",
                expected_benefit=5000.0,  # $/year
                confidence_level=0.85,
                priority="MEDIUM",
            ),
            GL003Recommendation(
                recommendation_id="REC-002",
                recommendation_type="BLOWDOWN_OPTIMIZATION",
                asset_id="DRUM-001",
                description="Optimize blowdown frequency to reduce steam loss",
                expected_benefit=3000.0,
                confidence_level=0.90,
                priority="LOW",
            ),
        ]

    def calculate_dryness_from_properties(
        self,
        pressure_mpa: float,
        enthalpy_kj_kg: float,
    ) -> Optional[float]:
        """
        Calculate dryness fraction using GL-003 saturation data.

        Args:
            pressure_mpa: Pressure in MPa
            enthalpy_kj_kg: Enthalpy in kJ/kg

        Returns:
            Dryness fraction (0-1) or None if not in two-phase
        """
        # Get saturation properties
        sat_request = GL003SaturationRequest(
            pressure_mpa=pressure_mpa,
            request_id=f"SAT-{datetime.now().timestamp()}",
        )
        sat = self.get_saturation_properties(sat_request)

        # Check if in two-phase region
        if enthalpy_kj_kg < sat.h_f_kj_kg:
            return 0.0  # Subcooled
        if enthalpy_kj_kg > sat.h_g_kj_kg:
            return 1.0  # Superheated

        # Calculate quality
        quality = (enthalpy_kj_kg - sat.h_f_kj_kg) / sat.h_fg_kj_kg

        return max(0.0, min(1.0, quality))


# =============================================================================
# GL-012 to GL-003 Integration Service
# =============================================================================

class GL012GL003IntegrationService:
    """
    Service for integrating GL-012 SteamQual with GL-003 UnifiedSteam.

    This service:
    - Coordinates quality calculations between agents
    - Manages data translation
    - Tracks provenance across systems
    - Handles error recovery
    """

    def __init__(self, gl003_adapter: GL003InterfaceAdapter = None):
        """Initialize service with GL-003 adapter."""
        self.gl003 = gl003_adapter or GL003InterfaceAdapter()
        self._provenance_chain = []

    def initialize(self) -> bool:
        """Initialize service and connect to GL-003."""
        return self.gl003.connect()

    def shutdown(self) -> bool:
        """Shutdown service and disconnect."""
        return self.gl003.disconnect()

    def estimate_quality_with_gl003(
        self,
        pressure_mpa: float,
        temperature_k: float,
        enthalpy_kj_kg: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Estimate steam quality using GL-003 property calculations.

        Args:
            pressure_mpa: Steam pressure
            temperature_k: Steam temperature
            enthalpy_kj_kg: Measured enthalpy (optional)

        Returns:
            Dictionary with quality estimate and metadata
        """
        # Request properties from GL-003
        prop_request = GL003PropertyRequest(
            pressure_mpa=pressure_mpa,
            temperature_k=temperature_k,
        )
        properties = self.gl003.get_properties(prop_request)

        # Use measured or calculated enthalpy
        h_for_quality = enthalpy_kj_kg if enthalpy_kj_kg is not None else properties.enthalpy_kj_kg

        # Calculate quality using saturation data
        quality = None
        if properties.state == "TWO_PHASE":
            quality = self.gl003.calculate_dryness_from_properties(
                pressure_mpa, h_for_quality
            )
        elif properties.state == "VAPOR":
            quality = 1.0
        elif properties.state == "LIQUID":
            quality = 0.0

        # Track provenance
        provenance_entry = {
            "step": "quality_estimation",
            "gl003_hash": properties.provenance_hash,
            "input_pressure": pressure_mpa,
            "input_temperature": temperature_k,
            "calculated_quality": quality,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._provenance_chain.append(provenance_entry)

        # Calculate combined provenance hash
        combined_hash = hashlib.sha256(
            json.dumps(provenance_entry, sort_keys=True).encode()
        ).hexdigest()

        return {
            "dryness_fraction": quality,
            "state": properties.state,
            "enthalpy_kj_kg": h_for_quality,
            "entropy_kj_kg_k": properties.entropy_kj_kg_k,
            "gl003_provenance": properties.provenance_hash,
            "combined_provenance": combined_hash,
            "calculation_time_ms": properties.calculation_time_ms,
        }

    def get_saturation_reference(self, pressure_mpa: float) -> Dict[str, Any]:
        """
        Get saturation reference data from GL-003.

        Args:
            pressure_mpa: Reference pressure

        Returns:
            Dictionary with saturation properties
        """
        sat_request = GL003SaturationRequest(
            pressure_mpa=pressure_mpa,
            request_id=f"REF-{datetime.now().timestamp()}",
        )
        sat = self.gl003.get_saturation_properties(sat_request)

        return {
            "pressure_mpa": sat.pressure_mpa,
            "temperature_k": sat.temperature_k,
            "h_f_kj_kg": sat.h_f_kj_kg,
            "h_g_kj_kg": sat.h_g_kj_kg,
            "h_fg_kj_kg": sat.h_fg_kj_kg,
            "s_f_kj_kg_k": sat.s_f_kj_kg_k,
            "s_g_kj_kg_k": sat.s_g_kj_kg_k,
            "provenance_hash": sat.provenance_hash,
        }

    def get_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Get optimization recommendations from GL-003.

        Returns:
            List of recommendations as dictionaries
        """
        recs = self.gl003.get_recommendations()

        return [
            {
                "id": r.recommendation_id,
                "type": r.recommendation_type,
                "asset": r.asset_id,
                "description": r.description,
                "benefit": r.expected_benefit,
                "confidence": r.confidence_level,
                "priority": r.priority,
            }
            for r in recs
        ]

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Get full provenance chain for audit."""
        return self._provenance_chain.copy()


# =============================================================================
# Test Classes
# =============================================================================

class TestGL003InterfaceAdapter:
    """Tests for GL-003 interface adapter."""

    @pytest.fixture
    def adapter(self) -> GL003InterfaceAdapter:
        """Create adapter instance."""
        return GL003InterfaceAdapter()

    def test_connect_succeeds(self, adapter):
        """Test connection succeeds."""
        result = adapter.connect()
        assert result is True
        assert adapter.is_connected()

    def test_disconnect_succeeds(self, adapter):
        """Test disconnection succeeds."""
        adapter.connect()
        result = adapter.disconnect()
        assert result is True
        assert not adapter.is_connected()

    def test_get_properties_requires_connection(self, adapter):
        """Test that get_properties requires connection."""
        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=453.0,
        )

        with pytest.raises(GL003InterfaceError):
            adapter.get_properties(request)

    def test_get_properties_returns_response(self, adapter):
        """Test that get_properties returns valid response."""
        adapter.connect()

        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=453.0,
        )

        response = adapter.get_properties(request)

        assert response.request_id == request.request_id
        assert response.pressure_mpa == request.pressure_mpa
        assert response.enthalpy_kj_kg > 0
        assert response.provenance_hash is not None

    def test_get_properties_identifies_vapor(self, adapter):
        """Test that vapor state is identified."""
        adapter.connect()

        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=550.0,  # Well above saturation
        )

        response = adapter.get_properties(request)

        assert response.state == "VAPOR"

    def test_get_properties_identifies_liquid(self, adapter):
        """Test that liquid state is identified."""
        adapter.connect()

        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=350.0,  # Below saturation
        )

        response = adapter.get_properties(request)

        assert response.state == "LIQUID"

    def test_get_properties_identifies_two_phase(self, adapter):
        """Test that two-phase state is identified."""
        adapter.connect()

        # At saturation temperature
        T_sat = 453.0 + 50.0 * math.log(1.0)  # ~453 K for 1 MPa

        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=T_sat,
            quality=0.9,
        )

        response = adapter.get_properties(request)

        assert response.state == "TWO_PHASE"

    def test_get_saturation_properties_by_pressure(self, adapter):
        """Test saturation properties by pressure."""
        adapter.connect()

        request = GL003SaturationRequest(
            pressure_mpa=1.0,
            request_id="SAT-001",
        )

        response = adapter.get_saturation_properties(request)

        assert response.h_f_kj_kg < response.h_g_kj_kg
        assert response.h_fg_kj_kg > 0
        assert response.provenance_hash is not None

    def test_get_saturation_properties_by_temperature(self, adapter):
        """Test saturation properties by temperature."""
        adapter.connect()

        request = GL003SaturationRequest(
            temperature_k=453.0,
            request_id="SAT-002",
        )

        response = adapter.get_saturation_properties(request)

        assert response.pressure_mpa > 0
        assert response.h_fg_kj_kg > 0

    def test_get_recommendations(self, adapter):
        """Test getting recommendations."""
        adapter.connect()

        recs = adapter.get_recommendations()

        assert len(recs) > 0
        assert all(r.recommendation_id for r in recs)

    def test_calculate_dryness(self, adapter):
        """Test dryness calculation from properties."""
        adapter.connect()

        # Get saturation data
        sat = adapter.get_saturation_properties(
            GL003SaturationRequest(pressure_mpa=1.0, request_id="SAT")
        )

        # Test at mid-quality enthalpy
        h_mid = sat.h_f_kj_kg + 0.5 * sat.h_fg_kj_kg

        quality = adapter.calculate_dryness_from_properties(1.0, h_mid)

        assert quality is not None
        assert 0.45 < quality < 0.55

    def test_provenance_hash_deterministic(self, adapter):
        """Test that provenance hash is deterministic."""
        adapter.connect()

        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=453.0,
            request_id="DET-001",
        )

        response1 = adapter.get_properties(request)
        response2 = adapter.get_properties(request)

        assert response1.provenance_hash == response2.provenance_hash


class TestGL012GL003IntegrationService:
    """Tests for integration service."""

    @pytest.fixture
    def service(self) -> GL012GL003IntegrationService:
        """Create integration service."""
        adapter = GL003InterfaceAdapter()
        service = GL012GL003IntegrationService(adapter)
        service.initialize()
        return service

    def test_initialize_connects_to_gl003(self, service):
        """Test that initialize connects to GL-003."""
        assert service.gl003.is_connected()

    def test_shutdown_disconnects(self, service):
        """Test that shutdown disconnects."""
        service.shutdown()
        assert not service.gl003.is_connected()

    def test_estimate_quality_returns_result(self, service):
        """Test that quality estimation returns result."""
        result = service.estimate_quality_with_gl003(
            pressure_mpa=1.0,
            temperature_k=453.0,
        )

        assert "dryness_fraction" in result
        assert "state" in result
        assert "gl003_provenance" in result
        assert "combined_provenance" in result

    def test_estimate_quality_vapor_state(self, service):
        """Test quality estimation for vapor."""
        result = service.estimate_quality_with_gl003(
            pressure_mpa=1.0,
            temperature_k=550.0,  # Superheated
        )

        assert result["state"] == "VAPOR"
        assert result["dryness_fraction"] == 1.0

    def test_estimate_quality_liquid_state(self, service):
        """Test quality estimation for liquid."""
        result = service.estimate_quality_with_gl003(
            pressure_mpa=1.0,
            temperature_k=350.0,  # Subcooled
        )

        assert result["state"] == "LIQUID"
        assert result["dryness_fraction"] == 0.0

    def test_estimate_quality_with_measured_enthalpy(self, service):
        """Test quality estimation with measured enthalpy."""
        # Get saturation data first
        sat = service.get_saturation_reference(1.0)

        # Provide measured enthalpy at 80% quality
        h_measured = sat["h_f_kj_kg"] + 0.8 * sat["h_fg_kj_kg"]

        result = service.estimate_quality_with_gl003(
            pressure_mpa=1.0,
            temperature_k=sat["temperature_k"],
            enthalpy_kj_kg=h_measured,
        )

        assert result["dryness_fraction"] is not None
        assert 0.75 < result["dryness_fraction"] < 0.85

    def test_get_saturation_reference(self, service):
        """Test getting saturation reference."""
        result = service.get_saturation_reference(1.0)

        assert result["pressure_mpa"] == 1.0
        assert result["h_fg_kj_kg"] > 0
        assert result["provenance_hash"] is not None

    def test_get_recommendations(self, service):
        """Test getting recommendations."""
        recs = service.get_optimization_recommendations()

        assert len(recs) > 0
        assert all("id" in r for r in recs)
        assert all("type" in r for r in recs)

    def test_provenance_chain_tracking(self, service):
        """Test that provenance chain is tracked."""
        # Make some calculations
        service.estimate_quality_with_gl003(1.0, 453.0)
        service.estimate_quality_with_gl003(2.0, 500.0)

        chain = service.get_provenance_chain()

        assert len(chain) >= 2
        assert all("gl003_hash" in entry for entry in chain)
        assert all("timestamp" in entry for entry in chain)


class TestDataFormatCompatibility:
    """Tests for data format compatibility between GL-012 and GL-003."""

    @pytest.fixture
    def adapter(self) -> GL003InterfaceAdapter:
        adapter = GL003InterfaceAdapter()
        adapter.connect()
        return adapter

    def test_pressure_units_consistent(self, adapter):
        """Test that pressure units are consistent (MPa)."""
        request = GL003PropertyRequest(
            pressure_mpa=1.0,  # 1 MPa = 10 bar
            temperature_k=453.0,
        )

        response = adapter.get_properties(request)

        assert response.pressure_mpa == 1.0

    def test_temperature_units_consistent(self, adapter):
        """Test that temperature units are consistent (K)."""
        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=453.0,  # Kelvin
        )

        response = adapter.get_properties(request)

        assert response.temperature_k == 453.0

    def test_enthalpy_units_consistent(self, adapter):
        """Test that enthalpy units are consistent (kJ/kg)."""
        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=453.0,
        )

        response = adapter.get_properties(request)

        # Enthalpy should be in reasonable range for kJ/kg
        assert 0 < response.enthalpy_kj_kg < 4000


class TestErrorHandling:
    """Tests for error handling in integration."""

    @pytest.fixture
    def adapter(self) -> GL003InterfaceAdapter:
        return GL003InterfaceAdapter()

    def test_not_connected_error(self, adapter):
        """Test error when not connected."""
        request = GL003PropertyRequest(
            pressure_mpa=1.0,
            temperature_k=453.0,
        )

        with pytest.raises(GL003InterfaceError):
            adapter.get_properties(request)

    def test_invalid_saturation_request(self, adapter):
        """Test error for invalid saturation request."""
        adapter.connect()

        request = GL003SaturationRequest()  # No P or T specified

        with pytest.raises(GL003InterfaceError):
            adapter.get_saturation_properties(request)


class TestDeterminism:
    """Tests for deterministic behavior across integration."""

    @pytest.fixture
    def service(self) -> GL012GL003IntegrationService:
        adapter = GL003InterfaceAdapter()
        service = GL012GL003IntegrationService(adapter)
        service.initialize()
        return service

    def test_repeated_estimation_deterministic(self, service):
        """Test that repeated estimation is deterministic."""
        results = [
            service.estimate_quality_with_gl003(1.0, 453.0)
            for _ in range(5)
        ]

        first = results[0]
        for r in results[1:]:
            assert r["dryness_fraction"] == first["dryness_fraction"]
            assert r["state"] == first["state"]

    def test_saturation_reference_deterministic(self, service):
        """Test that saturation reference is deterministic."""
        refs = [service.get_saturation_reference(1.0) for _ in range(5)]

        first = refs[0]
        for r in refs[1:]:
            assert r["h_fg_kj_kg"] == first["h_fg_kj_kg"]
            assert r["provenance_hash"] == first["provenance_hash"]


class TestPerformance:
    """Performance tests for GL-003 integration."""

    @pytest.fixture
    def service(self) -> GL012GL003IntegrationService:
        adapter = GL003InterfaceAdapter()
        service = GL012GL003IntegrationService(adapter)
        service.initialize()
        return service

    @pytest.mark.performance
    def test_property_calculation_time(self, service):
        """Test that property calculation is fast."""
        result = service.estimate_quality_with_gl003(1.0, 453.0)

        # Should complete in < 10ms
        assert result["calculation_time_ms"] < 10.0

    @pytest.mark.performance
    @pytest.mark.slow
    def test_throughput(self, service):
        """Test integration throughput."""
        import time

        n_iterations = 100
        start = time.perf_counter()

        for i in range(n_iterations):
            service.estimate_quality_with_gl003(1.0 + i * 0.01, 453.0)

        elapsed = time.perf_counter() - start
        throughput = n_iterations / elapsed

        # Target: > 50 requests per second
        assert throughput > 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
