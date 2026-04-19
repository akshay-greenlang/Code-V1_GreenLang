# -*- coding: utf-8 -*-
"""Usage metering — emit billing events for Stripe / internal analytics."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class UsageEvent:
    tenant_id: str
    tier: str
    metric: str  # api.requests | scope_engine.compute_minutes | storage.gb_hours
    quantity: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UsageEmitter:
    """Fan-out usage events to sinks (log, Kafka, Stripe Meter API, etc.)."""

    def __init__(self, sinks: list["UsageSink"] | None = None) -> None:
        self._sinks = sinks or [LogSink()]

    async def emit(self, event: UsageEvent) -> None:
        for sink in self._sinks:
            try:
                await sink.write(event)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Usage sink %s failed: %s", type(sink).__name__, exc)


class UsageSink:
    async def write(self, event: UsageEvent) -> None:  # pragma: no cover
        raise NotImplementedError


class LogSink(UsageSink):
    async def write(self, event: UsageEvent) -> None:
        logger.info("USAGE %s", event.to_dict())


class StripeMeterSink(UsageSink):
    """Stripe Meter API sink. Requires stripe-python + api key. Skeleton only."""

    def __init__(self, api_key: str) -> None:
        self._api_key = api_key

    async def write(self, event: UsageEvent) -> None:  # pragma: no cover
        # Pseudocode - real impl requires stripe package:
        # import stripe; stripe.api_key = self._api_key
        # stripe.billing.MeterEvent.create(
        #     event_name=event.metric,
        #     payload={"stripe_customer_id": event.tenant_id, "value": event.quantity},
        #     timestamp=int(event.timestamp),
        # )
        logger.debug("StripeMeterSink would send: %s", event.to_dict())
