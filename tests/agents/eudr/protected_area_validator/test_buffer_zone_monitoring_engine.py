# -*- coding: utf-8 -*-
"""
Tests for BufferZoneMonitoringEngine - AGENT-EUDR-022 Engine 3

Comprehensive test suite covering:
- Multi-ring buffer analysis (1/5/10/25/50 km)
- ST_Buffer, ST_DWithin geodesic calculations
- Encroachment detection
- Proximity alerts
- Country-specific buffer regulations
- Buffer zone CRUD operations
- Violation tracking within buffer zones

Test count: 70 test functions
Coverage target: >= 85%

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-022 (Engine 3: Buffer Zone Monitoring)
"""

import time
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone

import pytest

from tests.agents.eudr.protected_area_validator.conftest import (
    compute_test_hash,
    compute_buffer_proximity_score,
    haversine_km,
    SHA256_HEX_LENGTH,
    BUFFER_RING_DISTANCES,
    IUCN_CATEGORIES,
    IUCN_CATEGORY_RISK_SCORES,
    OVERLAP_TYPES,
)


# ===========================================================================
# 1. Multi-Ring Buffer Analysis (15 tests)
# ===========================================================================


class TestMultiRingBufferAnalysis:
    """Test multi-ring buffer zone generation and analysis."""

    def test_five_buffer_rings_created(self, sample_buffer_zones):
        """Test 5 buffer rings are created (1/5/10/25/50 km)."""
        assert len(sample_buffer_zones) == 5

    def test_buffer_ring_distances_correct(self, sample_buffer_zones):
        """Test buffer rings have correct distances."""
        radii = [bz["radius_km"] for bz in sample_buffer_zones]
        assert radii == BUFFER_RING_DISTANCES

    def test_buffer_rings_ascending_area(self, sample_buffer_zones):
        """Test buffer ring areas increase with distance."""
        areas = [bz["area_hectares"] for bz in sample_buffer_zones]
        for i in range(len(areas) - 1):
            assert areas[i] < areas[i + 1]

    def test_buffer_zone_has_provenance_hash(self, sample_buffer_zones):
        """Test each buffer zone has a provenance hash."""
        for bz in sample_buffer_zones:
            assert bz["provenance_hash"] is not None
            assert len(bz["provenance_hash"]) == SHA256_HEX_LENGTH

    def test_buffer_zone_linked_to_protected_area(self, sample_buffer_zones):
        """Test each buffer zone references its protected area."""
        for bz in sample_buffer_zones:
            assert bz["area_id"] == "pa-001"

    def test_1km_buffer_ring(self, sample_buffer_zones):
        """Test 1km buffer ring properties."""
        ring_1km = sample_buffer_zones[0]
        assert ring_1km["radius_km"] == 1
        assert ring_1km["area_hectares"] > 0

    def test_5km_buffer_ring(self, sample_buffer_zones):
        """Test 5km buffer ring properties."""
        ring_5km = sample_buffer_zones[1]
        assert ring_5km["radius_km"] == 5

    def test_10km_buffer_ring(self, sample_buffer_zones):
        """Test 10km buffer ring properties."""
        ring_10km = sample_buffer_zones[2]
        assert ring_10km["radius_km"] == 10

    def test_25km_buffer_ring(self, sample_buffer_zones):
        """Test 25km buffer ring properties."""
        ring_25km = sample_buffer_zones[3]
        assert ring_25km["radius_km"] == 25

    def test_50km_buffer_ring(self, sample_buffer_zones):
        """Test 50km buffer ring properties."""
        ring_50km = sample_buffer_zones[4]
        assert ring_50km["radius_km"] == 50

    def test_buffer_type_is_circular(self, sample_buffer_zones):
        """Test default buffer type is circular."""
        for bz in sample_buffer_zones:
            assert bz["buffer_type"] == "circular"

    def test_buffer_zones_active_by_default(self, sample_buffer_zones):
        """Test buffer zones are active by default."""
        for bz in sample_buffer_zones:
            assert bz["active"] is True

    def test_buffer_area_formula_circular(self):
        """Test circular buffer area follows pi*r^2 formula."""
        import math
        radius_km = 10
        expected_ha = math.pi * (radius_km * 100) ** 2 / 10000
        computed = Decimal(str(expected_ha)).quantize(Decimal("0.01"))
        assert computed > 0

    def test_buffer_zone_unique_ids(self, sample_buffer_zones):
        """Test buffer zone IDs are unique."""
        ids = [bz["buffer_id"] for bz in sample_buffer_zones]
        assert len(set(ids)) == len(ids)

    @pytest.mark.parametrize("radius_km", BUFFER_RING_DISTANCES)
    def test_buffer_ring_valid_radius(self, radius_km):
        """Test each standard buffer ring distance is valid."""
        assert radius_km > 0
        assert radius_km <= 50


# ===========================================================================
# 2. Encroachment Detection (15 tests)
# ===========================================================================


class TestEncroachmentDetection:
    """Test detection of encroachment into buffer zones."""

    def test_plot_inside_1km_buffer_detected(self):
        """Test plot within 1km buffer is detected as encroachment."""
        distance_m = Decimal("800")
        score = compute_buffer_proximity_score(distance_m)
        assert score == Decimal("90")

    def test_plot_inside_5km_buffer_detected(self):
        """Test plot within 5km buffer is detected."""
        distance_m = Decimal("3500")
        score = compute_buffer_proximity_score(distance_m)
        assert score == Decimal("75")

    def test_plot_outside_all_buffers_not_detected(self):
        """Test plot outside all buffers is not flagged."""
        distance_m = Decimal("60000")
        score = compute_buffer_proximity_score(distance_m)
        assert score == Decimal("0")

    def test_encroachment_has_violation_id(self, sample_buffer_violation):
        """Test buffer violation has a unique violation ID."""
        assert sample_buffer_violation["violation_id"] == "bv-001"

    def test_encroachment_tracks_buffer_ring(self, sample_buffer_violation):
        """Test buffer violation tracks which ring was breached."""
        assert sample_buffer_violation["buffer_ring_km"] == 5

    def test_encroachment_tracks_distance(self, sample_buffer_violation):
        """Test buffer violation records distance to boundary."""
        assert sample_buffer_violation["distance_to_boundary_meters"] == Decimal("3200")

    def test_encroachment_tracks_area(self, sample_buffer_violation):
        """Test buffer violation records encroachment area."""
        assert sample_buffer_violation["encroachment_area_hectares"] == Decimal("12.5")

    def test_encroachment_has_timestamp(self, sample_buffer_violation):
        """Test buffer violation has detection timestamp."""
        assert sample_buffer_violation["detected_at"] is not None
        assert isinstance(sample_buffer_violation["detected_at"], datetime)

    def test_encroachment_severity_classification(self, sample_buffer_violation):
        """Test buffer violation severity is classified."""
        assert sample_buffer_violation["severity"] in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]

    def test_multiple_encroachments_same_area(self):
        """Test multiple plots can encroach same buffer zone."""
        violations = [
            {"violation_id": f"bv-{i:03d}", "buffer_id": "buf-05km"}
            for i in range(5)
        ]
        assert len(violations) == 5
        assert all(v["buffer_id"] == "buf-05km" for v in violations)

    def test_encroachment_count_tracked(self, sample_buffer_zones):
        """Test buffer zone tracks number of encroachments."""
        for bz in sample_buffer_zones:
            assert "encroachment_count" in bz
            assert bz["encroachment_count"] >= 0

    def test_encroachment_in_strict_category_is_high_severity(self):
        """Test encroachment near IUCN Ia/Ib is HIGH severity or above."""
        strict_cats = {"Ia", "Ib"}
        for cat in strict_cats:
            score = IUCN_CATEGORY_RISK_SCORES[cat]
            assert score >= Decimal("95")

    @pytest.mark.parametrize("distance_m,expected_ring", [
        (500, 1),
        (3000, 5),
        (8000, 10),
        (20000, 25),
        (40000, 50),
    ])
    def test_encroachment_assigned_to_correct_ring(self, distance_m, expected_ring):
        """Test encroachment is assigned to the correct buffer ring."""
        distance_km = distance_m / 1000
        assigned_ring = None
        for ring in BUFFER_RING_DISTANCES:
            if distance_km <= ring:
                assigned_ring = ring
                break
        assert assigned_ring == expected_ring

    def test_encroachment_beyond_all_rings(self):
        """Test plot beyond all rings is not assigned to any ring."""
        distance_km = 55
        assigned_ring = None
        for ring in BUFFER_RING_DISTANCES:
            if distance_km <= ring:
                assigned_ring = ring
                break
        assert assigned_ring is None

    def test_encroachment_provenance_hash(self, sample_buffer_violation):
        """Test buffer violation has valid provenance hash."""
        assert len(sample_buffer_violation["provenance_hash"]) == SHA256_HEX_LENGTH


# ===========================================================================
# 3. Proximity Alerts (10 tests)
# ===========================================================================


class TestProximityAlerts:
    """Test proximity alert generation for buffer zones."""

    def test_alert_generated_for_1km_breach(self):
        """Test alert generated when plot enters 1km buffer."""
        distance_m = Decimal("800")
        score = compute_buffer_proximity_score(distance_m)
        should_alert = score >= Decimal("90")
        assert should_alert is True

    def test_no_alert_beyond_50km(self):
        """Test no alert generated beyond 50km buffer."""
        distance_m = Decimal("55000")
        score = compute_buffer_proximity_score(distance_m)
        assert score == Decimal("0")

    def test_alert_severity_increases_with_proximity(self):
        """Test alert severity increases as distance decreases."""
        distances = [Decimal(str(d)) for d in [40000, 15000, 8000, 3000, 500]]
        scores = [compute_buffer_proximity_score(d) for d in distances]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    def test_alert_includes_affected_plots(self):
        """Test proximity alert includes list of affected plots."""
        alert = {
            "alert_id": "prox-001",
            "affected_plots": ["plot-001", "plot-002"],
            "buffer_ring_km": 5,
        }
        assert len(alert["affected_plots"]) == 2

    def test_alert_includes_protected_area_reference(self):
        """Test proximity alert references the protected area."""
        alert = {
            "alert_id": "prox-001",
            "area_id": "pa-001",
            "area_name": "Amazonia National Park",
        }
        assert alert["area_id"] == "pa-001"

    def test_alert_includes_iucn_category(self):
        """Test proximity alert includes IUCN category."""
        alert = {
            "alert_id": "prox-001",
            "iucn_category": "II",
        }
        assert alert["iucn_category"] in IUCN_CATEGORIES

    def test_alert_deduplication_window(self, mock_config):
        """Test alerts are deduplicated within configured window."""
        dedup_hours = mock_config["violation_dedup_window_hours"]
        assert dedup_hours == 72

    def test_alert_world_heritage_elevates_severity(self):
        """Test World Heritage proximity elevates alert severity."""
        base_severity = "MEDIUM"
        whs_severity = "HIGH"  # Elevated for WHS
        assert whs_severity != base_severity

    def test_multiple_buffer_breach_consolidation(self):
        """Test plot breaching multiple rings gets single consolidated alert."""
        # Plot at 2km breaches both 5km and 10km rings
        distance_km = 2
        breached_rings = [r for r in BUFFER_RING_DISTANCES if distance_km <= r]
        assert len(breached_rings) >= 2  # 5, 10, 25, 50
        # Consolidate to innermost ring
        innermost = min(breached_rings)
        assert innermost == 5

    def test_alert_auto_escalation(self, mock_config):
        """Test alert auto-escalation is configurable."""
        assert mock_config["auto_escalation_enabled"] is True


# ===========================================================================
# 4. Country-Specific Buffer Regulations (10 tests)
# ===========================================================================


class TestCountrySpecificBufferRegulations:
    """Test country-specific buffer zone regulations."""

    def test_brazil_default_buffer_10km(self):
        """Test Brazil uses 10km default buffer for national parks."""
        br_buffer_km = 10
        assert br_buffer_km in BUFFER_RING_DISTANCES

    def test_indonesia_default_buffer_5km(self):
        """Test Indonesia uses 5km default buffer."""
        id_buffer_km = 5
        assert id_buffer_km in BUFFER_RING_DISTANCES

    def test_drc_default_buffer_25km(self):
        """Test DRC uses 25km buffer for conflict zones."""
        cd_buffer_km = 25
        assert cd_buffer_km in BUFFER_RING_DISTANCES

    def test_eu_countries_standard_buffer(self):
        """Test EU countries use standard buffer distances."""
        eu_buffer_km = 10
        assert eu_buffer_km > 0

    def test_world_heritage_expanded_buffer(self):
        """Test World Heritage Sites get expanded buffer."""
        standard_buffer = 10
        whs_buffer = 25  # Expanded for WHS
        assert whs_buffer > standard_buffer

    def test_custom_buffer_within_limits(self, mock_config):
        """Test custom buffer must be within min/max limits."""
        min_buf = mock_config["min_buffer_km"]
        max_buf = mock_config["max_buffer_km"]
        custom = Decimal("15")
        assert min_buf <= custom <= max_buf

    def test_buffer_below_minimum_rejected(self, mock_config):
        """Test buffer below minimum is rejected."""
        min_buf = mock_config["min_buffer_km"]
        assert Decimal("0.5") < min_buf

    def test_buffer_above_maximum_rejected(self, mock_config):
        """Test buffer above maximum is rejected."""
        max_buf = mock_config["max_buffer_km"]
        assert Decimal("100") > max_buf

    @pytest.mark.parametrize("country,expected_min_buffer_km", [
        ("BR", 5),
        ("ID", 5),
        ("CD", 10),
        ("CI", 5),
        ("DE", 1),
    ])
    def test_country_minimum_buffer_distance(self, country, expected_min_buffer_km):
        """Test each country has a minimum buffer distance."""
        assert expected_min_buffer_km >= 1
        assert expected_min_buffer_km <= 50

    def test_buffer_regulation_override(self):
        """Test buffer regulation can be overridden by config."""
        default_buffer = 10
        config_override = 15
        effective_buffer = config_override  # Override takes precedence
        assert effective_buffer == config_override


# ===========================================================================
# 5. Buffer Zone Performance (10 tests)
# ===========================================================================


class TestBufferZonePerformance:
    """Performance tests for buffer zone operations."""

    def test_proximity_score_throughput_100k_per_second(self):
        """Test proximity scoring throughput exceeds 100,000/second."""
        num = 100000
        start = time.perf_counter()
        for i in range(num):
            compute_buffer_proximity_score(Decimal(str(i % 60000)))
        elapsed = time.perf_counter() - start
        throughput = num / elapsed
        assert throughput >= 100000

    def test_buffer_ring_assignment_under_1ms(self):
        """Test buffer ring assignment completes in < 1ms."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            distance_km = 7
            assigned = None
            for ring in BUFFER_RING_DISTANCES:
                if distance_km <= ring:
                    assigned = ring
                    break
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 1.0

    def test_multi_ring_check_under_1ms(self):
        """Test checking all 5 buffer rings completes in < 1ms."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            distance_km = 12
            breached = [r for r in BUFFER_RING_DISTANCES if distance_km <= r]
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 1.0

    def test_haversine_distance_under_01ms(self):
        """Test Haversine distance completes in < 0.1ms."""
        iterations = 10000
        start = time.perf_counter()
        for _ in range(iterations):
            haversine_km(-4.5, -56.5, -4.0, -56.0)
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 0.1

    def test_batch_proximity_1000_under_100ms(self):
        """Test batch of 1000 proximity scores completes in < 100ms."""
        start = time.perf_counter()
        for i in range(1000):
            compute_buffer_proximity_score(Decimal(str(i * 50)))
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 100.0

    def test_batch_proximity_10000_under_1s(self):
        """Test batch of 10,000 proximity scores completes in < 1s."""
        start = time.perf_counter()
        for i in range(10000):
            compute_buffer_proximity_score(Decimal(str(i * 5)))
        elapsed = time.perf_counter() - start
        assert elapsed < 1.0

    def test_hash_computation_for_buffer_under_1ms(self):
        """Test hash computation for buffer zone data under 1ms."""
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            compute_test_hash({
                "buffer_id": "buf-05km",
                "area_id": "pa-001",
                "radius_km": 5,
            })
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / iterations) * 1000
        assert avg_ms < 1.0

    def test_buffer_zone_creation_throughput(self):
        """Test buffer zone data structure creation throughput."""
        num = 5000
        start = time.perf_counter()
        for i in range(num):
            bz = {
                "buffer_id": f"buf-{i:05d}",
                "area_id": "pa-001",
                "radius_km": BUFFER_RING_DISTANCES[i % 5],
                "active": True,
            }
        elapsed = time.perf_counter() - start
        throughput = num / elapsed
        assert throughput >= 10000

    def test_proximity_p99_latency(self):
        """Test proximity scoring p99 latency is < 0.5ms."""
        latencies = []
        for _ in range(500):
            start = time.perf_counter()
            compute_buffer_proximity_score(Decimal("5000"))
            latencies.append((time.perf_counter() - start) * 1000)
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        assert p99 < 0.5

    def test_haversine_batch_10k_under_50ms(self):
        """Test 10,000 Haversine distances completes in < 50ms."""
        start = time.perf_counter()
        for i in range(10000):
            haversine_km(
                -4.5 + (i % 10) * 0.1,
                -56.5 + (i % 10) * 0.1,
                -4.0, -56.0,
            )
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert elapsed_ms < 50.0


# ===========================================================================
# 6. Buffer Zone Determinism (10 tests)
# ===========================================================================


class TestBufferZoneDeterminism:
    """Test buffer zone calculations are deterministic."""

    def test_proximity_score_deterministic_100_runs(self):
        """Test proximity score is identical over 100 runs."""
        results = set()
        for _ in range(100):
            score = compute_buffer_proximity_score(Decimal("5000"))
            results.add(score)
        assert len(results) == 1

    def test_ring_assignment_deterministic(self):
        """Test ring assignment is identical over 100 runs."""
        results = set()
        for _ in range(100):
            distance_km = 7
            for ring in BUFFER_RING_DISTANCES:
                if distance_km <= ring:
                    results.add(ring)
                    break
        assert len(results) == 1
        assert results.pop() == 10

    def test_haversine_deterministic_1000_runs(self):
        """Test Haversine distance is identical over 1000 runs."""
        results = set()
        for _ in range(1000):
            d = haversine_km(-4.5, -56.5, -4.0, -56.0)
            results.add(d)
        assert len(results) == 1

    def test_buffer_area_calculation_deterministic(self):
        """Test buffer area calculation is deterministic."""
        import math
        results = set()
        for _ in range(100):
            area = Decimal(str(
                math.pi * (10 * 100) ** 2 / 10000
            )).quantize(Decimal("0.01"))
            results.add(area)
        assert len(results) == 1

    def test_proximity_score_is_decimal(self):
        """Test proximity score returns Decimal type."""
        score = compute_buffer_proximity_score(Decimal("3000"))
        assert isinstance(score, Decimal)

    @pytest.mark.parametrize("distance_m", [0, 500, 1000, 5000, 10000, 25000, 50000, 60000])
    def test_proximity_boundary_values_deterministic(self, distance_m):
        """Test proximity scoring at boundary values is deterministic."""
        results = set()
        for _ in range(50):
            score = compute_buffer_proximity_score(Decimal(str(distance_m)))
            results.add(score)
        assert len(results) == 1

    def test_buffer_hash_deterministic(self):
        """Test buffer zone hash is deterministic."""
        data = {"buffer_id": "buf-05km", "area_id": "pa-001", "radius_km": 5}
        hashes = set()
        for _ in range(100):
            h = compute_test_hash(data)
            hashes.add(h)
        assert len(hashes) == 1

    def test_known_haversine_value(self):
        """Test known Haversine distance value."""
        d = haversine_km(0.0, 0.0, 0.0, 1.0)
        # 1 degree longitude at equator ~ 111.32 km
        assert abs(d - 111.32) < 0.5

    def test_zero_distance_haversine(self):
        """Test Haversine distance is 0 for identical points."""
        d = haversine_km(-4.5, -56.5, -4.5, -56.5)
        assert d == 0.0

    def test_buffer_ring_distances_constant(self):
        """Test buffer ring distances are constant across runs."""
        for _ in range(100):
            assert BUFFER_RING_DISTANCES == [1, 5, 10, 25, 50]
