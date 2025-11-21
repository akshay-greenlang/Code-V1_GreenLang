# -*- coding: utf-8 -*-
"""
Mock Grid Intensity Connector
==============================

Deterministic mock connector for grid carbon intensity data.

Algorithm (from CTO's spec):
1. Seed = SHA-256(region + start + end + resolution)
2. Generate hourly series with seasonal daily pattern
3. Base curve: sinusoid(hour) + region-hash offset
4. Clamp to [50, 900] gCO2/kWh
5. Quantize to 0.1 g (use Decimal)
6. Quality = "simulated"

Key Features:
- Deterministic (same inputs → same outputs)
- No network calls (even in record mode)
- Snapshot-compatible (byte-exact replay)
- Provenance tracking
"""

import hashlib
import math
import json
from datetime import timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, getcontext
from typing import Tuple

from greenlang.connectors.base import Connector, ConnectorCapabilities, ConnectorProvenance
from greenlang.connectors.context import ConnectorContext, CacheMode
from greenlang.connectors.models import (
    GridIntensityQuery,
    GridIntensityPayload,
    TSPoint
)
from greenlang.connectors.errors import ConnectorReplayRequired, ConnectorError
from greenlang.connectors.snapshot import (
    write_canonical_snapshot,
    read_canonical_snapshot,
    compute_query_hash,
    compute_schema_hash
)

# Set decimal precision (match existing units.py)
getcontext().prec = 28


class GridIntensityMockConnector(Connector[GridIntensityQuery, GridIntensityPayload, None]):
    """
    Mock grid intensity connector with deterministic output

    Provides synthetic carbon intensity data for testing and development.

    Example:
        connector = GridIntensityMockConnector()
        ctx = ConnectorContext.for_record("grid/intensity/mock")

        query = GridIntensityQuery(
            region="CA-ON",
            window=TimeWindow(
                start=datetime(2025, 1, 1, 0, 0, tzinfo=timezone.utc),
                end=datetime(2025, 1, 2, 0, 0, tzinfo=timezone.utc),
                resolution="hour"
            )
        )

        payload, prov = await connector.fetch(query, ctx)
        # payload.series has 24 deterministic hourly points
    """

    connector_id = "grid/intensity/mock"
    connector_version = "0.1.0"

    @property
    def capabilities(self) -> ConnectorCapabilities:
        """Declare connector capabilities"""
        return ConnectorCapabilities(
            supports_streaming=False,
            supports_pagination=False,
            supports_push=False,
            requires_auth=False,
            rate_limit_per_hour=None,  # No rate limit (mock)
            max_batch_size=8760,  # 1 year of hourly data
            supports_time_series=True,
            min_resolution="hour"
        )

    def _compute_seed(self, query: GridIntensityQuery) -> str:
        """
        Compute deterministic seed from query

        Seed = SHA-256(region + start + end + resolution)

        Args:
            query: Grid intensity query

        Returns:
            Hex seed string
        """
        seed_string = (
            f"{query.region}|"
            f"{query.window.start.isoformat()}|"
            f"{query.window.end.isoformat()}|"
            f"{query.window.resolution}"
        )
        return hashlib.sha256(seed_string.encode("utf-8")).hexdigest()

    def _generate_series(self, query: GridIntensityQuery, seed: str) -> GridIntensityPayload:
        """
        Generate deterministic carbon intensity series

        Algorithm:
        1. Parse time window to get number of hours
        2. Compute region offset from seed hash
        3. For each hour:
           - Calculate hour of day (0-23)
           - Apply sinusoid: 300 + 200*sin(2π*hour/24) + region_offset
           - Clamp to [50, 900]
           - Quantize to 0.1 g (Decimal)
        4. Build payload with metadata

        Args:
            query: Grid intensity query
            seed: Hex seed string

        Returns:
            GridIntensityPayload with deterministic series
        """
        start = query.window.start
        end = query.window.end

        # Calculate number of hours
        delta = end - start
        n_hours = int(delta.total_seconds() // 3600)

        # Region offset from seed hash (0-199)
        region_offset = int(seed, 16) % 200

        # Generate series
        series = []
        current_time = start

        for i in range(n_hours):
            # Hour of day (0-23)
            hour_of_day = current_time.hour + (current_time.minute / 60.0)

            # Sinusoidal daily pattern
            # Peak at midday (hour 12), low at midnight (hour 0/24)
            # Base = 300, amplitude = 200, offset = region_offset
            value_float = (
                300 +
                200 * math.sin(2 * math.pi * (hour_of_day / 24.0)) +
                region_offset
            )

            # Clamp to [50, 900]
            value_float = max(50.0, min(900.0, value_float))

            # Quantize to 0.1 g using Decimal
            value_decimal = Decimal(value_float).quantize(
                Decimal("0.1"),
                rounding=ROUND_HALF_UP
            )

            # Create time series point
            point = TSPoint(
                ts=current_time,
                value=value_decimal,
                unit="gCO2/kWh",
                quality="simulated"
            )

            series.append(point)

            # Next hour
            current_time = current_time + timedelta(hours=1)

        # Build payload
        payload = GridIntensityPayload(
            series=series,
            region=query.region,
            unit="gCO2/kWh",
            resolution=query.window.resolution,
            metadata={
                "connector": self.connector_id,
                "version": self.connector_version,
                "algorithm": "seasonal_sinusoid",
                "seed": seed,
                "region_offset": region_offset,
                "data_points": len(series)
            }
        )

        return payload

    async def fetch(
        self,
        query: GridIntensityQuery,
        ctx: ConnectorContext
    ) -> Tuple[GridIntensityPayload, ConnectorProvenance]:
        """
        Fetch grid intensity data (deterministic mock)

        Mode handling:
        - REPLAY: Load from snapshot (error if missing)
        - RECORD: Generate and optionally save snapshot
        - GOLDEN: Load from golden snapshot

        Args:
            query: Grid intensity query
            ctx: Connector context

        Returns:
            Tuple of (payload, provenance)

        Raises:
            ConnectorReplayRequired: If in replay mode without snapshot
            ConnectorError: If snapshot invalid or generation fails
        """
        # Compute hashes for provenance
        query_hash = compute_query_hash(query)
        schema_hash = compute_schema_hash(GridIntensityPayload)
        seed = self._compute_seed(query)

        # REPLAY or GOLDEN mode: require snapshot
        if ctx.mode in (CacheMode.REPLAY, CacheMode.GOLDEN):
            # Check if snapshot provided
            if ctx.snapshot_path:
                from greenlang.connectors.snapshot import load_snapshot
                snapshot_bytes = load_snapshot(ctx.snapshot_path)
                data = read_canonical_snapshot(snapshot_bytes)

                # Validate connector ID matches
                if data["connector_id"] != self.connector_id:
                    raise ConnectorError(
                        f"Snapshot connector mismatch: expected {self.connector_id}, "
                        f"got {data['connector_id']}",
                        connector=self.connector_id,
                        context={"snapshot_connector": data["connector_id"]}
                    )

                # Reconstruct payload and provenance
                payload = GridIntensityPayload(**data["payload"])
                prov = ConnectorProvenance(**data["provenance"])

                return payload, prov

            else:
                # No snapshot in replay mode
                raise ConnectorReplayRequired(
                    f"Replay mode requires snapshot for query {query_hash[:8]}. "
                    f"Switch to record mode or provide snapshot path.",
                    connector=self.connector_id,
                    query_hash=query_hash,
                    context={
                        "query": query.dict(),
                        "hint": "Use ConnectorContext.for_record() or provide snapshot_path"
                    }
                )

        # RECORD mode: generate fresh data
        try:
            payload = self._generate_series(query, seed)

            # Build provenance
            prov = ConnectorProvenance(
                connector_id=self.connector_id,
                connector_version=self.connector_version,
                mode=ctx.mode.value,
                query_hash=query_hash,
                schema_hash=schema_hash,
                seed=seed,
                metadata={
                    "region": query.region,
                    "start": query.window.start.isoformat(),
                    "end": query.window.end.isoformat(),
                    "resolution": query.window.resolution,
                    "data_points": len(payload.series)
                }
            )

            # Optionally save snapshot in record mode
            if ctx.cache_dir:
                snapshot_bytes = self.snapshot(payload, prov)
                from greenlang.connectors.snapshot import save_snapshot
                saved_path = save_snapshot(
                    snapshot_bytes,
                    connector_id=self.connector_id,
                    query_hash=query_hash,
                    output_dir=ctx.cache_dir
                )
                self.logger.info(f"Saved snapshot to {saved_path}")

            return payload, prov

        except Exception as e:
            raise ConnectorError(
                f"Failed to generate mock data: {e}",
                connector=self.connector_id,
                original_error=e,
                context={"query": query.dict()}
            )

    # Override restore to properly reconstruct types
    def restore(self, raw: bytes) -> Tuple[GridIntensityPayload, ConnectorProvenance]:
        """
        Restore payload and provenance from snapshot

        Args:
            raw: Snapshot bytes

        Returns:
            Tuple of (GridIntensityPayload, ConnectorProvenance)
        """
        data = read_canonical_snapshot(raw)

        payload = GridIntensityPayload(**data["payload"])
        prov = ConnectorProvenance(**data["provenance"])

        return payload, prov
