# -*- coding: utf-8 -*-
"""
Property-Based Tests for GridIntensityMockConnector
=====================================================

Comprehensive property-based testing using Hypothesis framework.

These tests verify universal invariants that must hold for ALL inputs,
not just specific examples. Each test generates arbitrary inputs and
verifies that the connector's guarantees hold.

Properties tested:
1. Series length matches requested hours
2. Timestamps are strictly increasing and aligned
3. All values within valid range [50, 900]
4. Deterministic output (same input → same output)
5. Regional variation (different regions → different data)
"""

import pytest
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from hypothesis import given, strategies as st, settings

from greenlang.connectors.grid.mock import GridIntensityMockConnector
from greenlang.connectors.models import GridIntensityQuery, TimeWindow, REGION_METADATA
from greenlang.connectors.context import ConnectorContext, CacheMode


# Mark all tests in this module as asyncio
pytestmark = pytest.mark.asyncio


# Helper strategies for generating test data

@st.composite
def time_window_strategy(draw, min_hours=1, max_hours=168):
    """
    Generate arbitrary time windows

    Args:
        draw: Hypothesis draw function
        min_hours: Minimum hours in window
        max_hours: Maximum hours in window (default 168 = 1 week)

    Returns:
        TimeWindow with random start and duration
    """
    # Generate random start time (any time in 2025)
    # NOTE: st.datetimes() requires naive datetimes for min/max bounds
    start_naive = draw(st.datetimes(
        min_value=datetime(2025, 1, 1),
        max_value=datetime(2025, 12, 31)
    ))

    # Normalize to hour boundary (zero out minutes, seconds, microseconds)
    # This ensures all generated times align with hourly resolution
    start = start_naive.replace(
        minute=0,
        second=0,
        microsecond=0,
        tzinfo=timezone.utc
    )

    # Generate random duration in hours
    hours = draw(st.integers(min_value=min_hours, max_value=max_hours))

    # Calculate end time
    end = start + timedelta(hours=hours)

    return TimeWindow(
        start=start,
        end=end,
        resolution="hour"
    )


@st.composite
def grid_query_strategy(draw, min_hours=1, max_hours=168):
    """
    Generate arbitrary GridIntensityQuery instances

    Args:
        draw: Hypothesis draw function
        min_hours: Minimum hours in window
        max_hours: Maximum hours in window

    Returns:
        GridIntensityQuery with random region and time window
    """
    region = draw(st.sampled_from(list(REGION_METADATA.keys())))
    window = draw(time_window_strategy(min_hours=min_hours, max_hours=max_hours))

    return GridIntensityQuery(region=region, window=window)


# Property 1: Series length matches hours
@pytest.mark.property
@given(
    hours=st.integers(min_value=1, max_value=168),  # 1 hour to 1 week
    region=st.sampled_from(list(REGION_METADATA.keys()))
)
@settings(max_examples=50)
async def test_property_series_length_matches_hours(hours, region):
    """
    Property: Series length MUST equal number of hours in window

    Invariant: len(payload.series) == (end - start).total_seconds() // 3600

    This property verifies that the connector generates exactly one data point
    per hour in the requested time window, with no missing or extra points.
    """
    # Create time window with exact number of hours
    start = datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=hours)

    query = GridIntensityQuery(
        region=region,
        window=TimeWindow(start=start, end=end, resolution="hour")
    )

    # Fetch data
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")
    payload, _ = await connector.fetch(query, ctx)

    # Verify series length
    assert len(payload.series) == hours, (
        f"Expected {hours} data points but got {len(payload.series)}"
    )

    # Verify no gaps in series
    calculated_hours = int((end - start).total_seconds() // 3600)
    assert len(payload.series) == calculated_hours, (
        f"Series length {len(payload.series)} does not match calculated hours {calculated_hours}"
    )


# Property 2: Timestamps strictly increasing
@pytest.mark.property
@given(query=grid_query_strategy(min_hours=2, max_hours=100))
@settings(max_examples=50)
async def test_property_timestamps_strictly_increasing(query):
    """
    Property: Timestamps MUST be strictly increasing with no duplicates

    Invariants:
    1. ts[i] < ts[i+1] for all i (strictly increasing)
    2. ts[i+1] - ts[i] == 1 hour (no gaps)
    3. ts[i].minute == 0 and ts[i].second == 0 (hourly boundaries)

    This property ensures that time series data is properly ordered and
    aligned to hourly boundaries with no duplicates or gaps.
    """
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")
    payload, _ = await connector.fetch(query, ctx)

    # Extract timestamps
    timestamps = [point.ts for point in payload.series]

    # Property 1: Strictly increasing (no duplicates)
    for i in range(len(timestamps) - 1):
        assert timestamps[i] < timestamps[i + 1], (
            f"Timestamps not strictly increasing at index {i}: "
            f"{timestamps[i]} >= {timestamps[i+1]}"
        )

    # Property 2: Exactly 1 hour gaps (no missing hours)
    for i in range(len(timestamps) - 1):
        delta = timestamps[i + 1] - timestamps[i]
        assert delta == timedelta(hours=1), (
            f"Gap between timestamps at index {i} is {delta}, expected 1 hour"
        )

    # Property 3: Timestamps aligned to hourly boundaries
    for i, ts in enumerate(timestamps):
        assert ts.minute == 0, f"Timestamp {i} not on hour boundary: minute={ts.minute}"
        assert ts.second == 0, f"Timestamp {i} not on hour boundary: second={ts.second}"
        assert ts.microsecond == 0, f"Timestamp {i} not on hour boundary: microsecond={ts.microsecond}"


# Property 3: Values within range [50, 900]
@pytest.mark.property
@given(query=grid_query_strategy(min_hours=1, max_hours=200))
@settings(max_examples=50)
async def test_property_values_within_range(query):
    """
    Property: All intensity values MUST be within [50, 900] gCO2/kWh

    Invariants:
    1. All values are Decimal type (exact precision)
    2. All values >= 50 (minimum grid intensity)
    3. All values <= 900 (maximum grid intensity)
    4. All values quantized to 0.1 g resolution

    This property ensures that generated carbon intensity values are
    physically plausible and meet the spec's range requirements.
    """
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")
    payload, _ = await connector.fetch(query, ctx)

    min_value = Decimal("50")
    max_value = Decimal("900")

    for i, point in enumerate(payload.series):
        # Property 1: Must be Decimal type
        assert isinstance(point.value, Decimal), (
            f"Point {i} value is not Decimal: {type(point.value)}"
        )

        # Property 2: Must be >= 50
        assert point.value >= min_value, (
            f"Point {i} value {point.value} is below minimum {min_value}"
        )

        # Property 3: Must be <= 900
        assert point.value <= max_value, (
            f"Point {i} value {point.value} is above maximum {max_value}"
        )

        # Property 4: Must be quantized to 0.1 g
        # Check that value has at most 1 decimal place
        value_str = str(point.value)
        if "." in value_str:
            decimal_places = len(value_str.split(".")[1])
            assert decimal_places <= 1, (
                f"Point {i} value {point.value} has more than 1 decimal place"
            )


# Property 4: Deterministic (same input → same output)
@pytest.mark.property
@given(query=grid_query_strategy(min_hours=1, max_hours=100))
@settings(max_examples=50)
async def test_property_deterministic_output(query):
    """
    Property: Same query MUST produce identical output

    Invariants:
    1. payload1.series == payload2.series (byte-exact match)
    2. provenance1.seed == provenance2.seed
    3. provenance1.query_hash == provenance2.query_hash
    4. All timestamps match exactly
    5. All values match exactly (Decimal precision)

    This is a critical property for snapshot-based testing and reproducibility.
    The connector must be completely deterministic with no randomness.
    """
    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")

    # Fetch twice with identical query
    payload1, prov1 = await connector.fetch(query, ctx)
    payload2, prov2 = await connector.fetch(query, ctx)

    # Property 1: Series must be identical
    assert len(payload1.series) == len(payload2.series), (
        f"Series lengths differ: {len(payload1.series)} vs {len(payload2.series)}"
    )

    for i, (point1, point2) in enumerate(zip(payload1.series, payload2.series)):
        # Timestamps must match exactly
        assert point1.ts == point2.ts, (
            f"Timestamps differ at index {i}: {point1.ts} vs {point2.ts}"
        )

        # Values must match exactly (Decimal precision)
        assert point1.value == point2.value, (
            f"Values differ at index {i}: {point1.value} vs {point2.value}"
        )

        # Units and quality must match
        assert point1.unit == point2.unit
        assert point1.quality == point2.quality

    # Property 2: Provenance seed must be identical
    assert prov1.seed == prov2.seed, (
        f"Seeds differ: {prov1.seed} vs {prov2.seed}"
    )

    # Property 3: Query hashes must be identical
    assert prov1.query_hash == prov2.query_hash, (
        f"Query hashes differ: {prov1.query_hash} vs {prov2.query_hash}"
    )

    # Property 4: Schema hashes must be identical
    assert prov1.schema_hash == prov2.schema_hash, (
        f"Schema hashes differ: {prov1.schema_hash} vs {prov2.schema_hash}"
    )

    # Property 5: Region and metadata must match
    assert payload1.region == payload2.region
    assert payload1.unit == payload2.unit
    assert payload1.resolution == payload2.resolution


# Property 5: Regional variation (different regions → different data)
@pytest.mark.property
@given(
    window=time_window_strategy(min_hours=1, max_hours=50),
    region1=st.sampled_from(list(REGION_METADATA.keys())),
    region2=st.sampled_from(list(REGION_METADATA.keys()))
)
@settings(max_examples=50)
async def test_property_regional_variation(window, region1, region2):
    """
    Property: Different regions MUST produce different data

    Invariants (when region1 != region2):
    1. At least one data point differs between regions
    2. Seeds differ between regions
    3. Data patterns are region-specific

    This property ensures that regional variation is actually implemented
    and that different grid regions have distinct carbon intensity patterns.
    """
    # Skip if regions are the same
    if region1 == region2:
        return

    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")

    # Create queries for both regions with same time window
    query1 = GridIntensityQuery(region=region1, window=window)
    query2 = GridIntensityQuery(region=region2, window=window)

    # Fetch data for both regions
    payload1, prov1 = await connector.fetch(query1, ctx)
    payload2, prov2 = await connector.fetch(query2, ctx)

    # Property 1: Seeds must differ (different regions)
    assert prov1.seed != prov2.seed, (
        f"Seeds are identical for different regions {region1} and {region2}: "
        f"{prov1.seed}"
    )

    # Property 2: At least one value must differ
    values1 = [point.value for point in payload1.series]
    values2 = [point.value for point in payload2.series]

    assert len(values1) == len(values2), (
        f"Different number of data points: {len(values1)} vs {len(values2)}"
    )

    differences = sum(1 for v1, v2 in zip(values1, values2) if v1 != v2)
    assert differences > 0, (
        f"No differences found between regions {region1} and {region2}. "
        f"All {len(values1)} values are identical."
    )

    # Property 3: Regions should be properly recorded in metadata
    assert payload1.region == region1
    assert payload2.region == region2

    # Property 4: Different query hashes (due to different regions)
    assert prov1.query_hash != prov2.query_hash, (
        f"Query hashes are identical for different regions: {prov1.query_hash}"
    )


# Additional edge case tests

@pytest.mark.property
@given(region=st.sampled_from(list(REGION_METADATA.keys())))
@settings(max_examples=20)
async def test_property_single_hour_window(region):
    """
    Property: Single-hour windows MUST work correctly

    Edge case: Minimum window size (1 hour)
    """
    start = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)

    query = GridIntensityQuery(
        region=region,
        window=TimeWindow(start=start, end=end, resolution="hour")
    )

    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")
    payload, _ = await connector.fetch(query, ctx)

    # Must have exactly 1 data point
    assert len(payload.series) == 1

    # Timestamp must match start
    assert payload.series[0].ts == start

    # Value must be in valid range
    assert Decimal("50") <= payload.series[0].value <= Decimal("900")


@pytest.mark.property
@given(
    region=st.sampled_from(list(REGION_METADATA.keys())),
    start_hour=st.integers(min_value=0, max_value=23)
)
@settings(max_examples=30)
async def test_property_different_start_times_affect_pattern(region, start_hour):
    """
    Property: Different start times produce different daily patterns

    Invariant: Starting at different hours of day should produce
    different sequences due to sinusoidal daily pattern.
    """
    # Create two queries starting at different hours but same day
    start1 = datetime(2025, 3, 15, start_hour, 0, tzinfo=timezone.utc)
    start2 = datetime(2025, 3, 15, (start_hour + 6) % 24, 0, tzinfo=timezone.utc)

    # Adjust dates if start2 wraps to next day
    if (start_hour + 6) >= 24:
        start2 = start2 + timedelta(days=1)

    query1 = GridIntensityQuery(
        region=region,
        window=TimeWindow(start=start1, end=start1 + timedelta(hours=12), resolution="hour")
    )

    query2 = GridIntensityQuery(
        region=region,
        window=TimeWindow(start=start2, end=start2 + timedelta(hours=12), resolution="hour")
    )

    connector = GridIntensityMockConnector()
    ctx = ConnectorContext.for_record("grid/intensity/mock")

    payload1, _ = await connector.fetch(query1, ctx)
    payload2, _ = await connector.fetch(query2, ctx)

    # First values should differ (different hours of day)
    assert payload1.series[0].value != payload2.series[0].value, (
        f"Values at different hours ({start_hour} vs {(start_hour + 6) % 24}) "
        f"should differ but both are {payload1.series[0].value}"
    )
