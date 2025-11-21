# -*- coding: utf-8 -*-
"""
Catena-X PCF Exchange Client
Automotive Industry Data Exchange Network

Implements Catena-X PCF Exchange protocol for automotive supply chains.

Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
import uuid

from greenlang.services.pcf_exchange.models import (
from greenlang.determinism import DeterministicClock
from greenlang.determinism import deterministic_uuid, DeterministicClock
    PCFDataModel,
    PCFExchangeResponse,
)

logger = logging.getLogger(__name__)


class CatenaXClient:
    """
    Catena-X PCF Exchange Client.

    Implements Catena-X data exchange protocol for PCF sharing
    in automotive supply chains.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Catena-X client.

        Args:
            config: Client configuration
        """
        self.base_url = config.get(
            "base_url",
            "https://api.catena-x.net/pcf/v1"
        )
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30.0)

        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers()
        )

        logger.info(f"Initialized CatenaXClient (base_url={self.base_url})")

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["X-API-Key"] = self.api_key

        return headers

    async def get_pcf(self, pcf_id: str) -> PCFExchangeResponse:
        """
        Retrieve PCF by ID from Catena-X network.

        Args:
            pcf_id: PCF identifier

        Returns:
            PCF exchange response
        """
        try:
            response = await self._http_client.get(f"/pcf/{pcf_id}")
            response.raise_for_status()

            data = response.json()
            pcf_data = PCFDataModel(**data)

            return PCFExchangeResponse(
                success=True,
                pcf_data=pcf_data,
                exchange_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                timestamp=DeterministicClock.utcnow()
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching PCF {pcf_id}: {e}")
            return PCFExchangeResponse(
                success=False,
                validation_errors=[f"HTTP {e.response.status_code}: {e.response.text}"],
                timestamp=DeterministicClock.utcnow()
            )

        except Exception as e:
            logger.error(f"Error fetching PCF {pcf_id}: {e}", exc_info=True)
            return PCFExchangeResponse(
                success=False,
                validation_errors=[str(e)],
                timestamp=DeterministicClock.utcnow()
            )

    async def publish_pcf(self, pcf_data: PCFDataModel) -> PCFExchangeResponse:
        """
        Publish PCF to Catena-X network.

        Args:
            pcf_data: PCF data model

        Returns:
            PCF exchange response
        """
        try:
            # Convert to Catena-X format
            payload = self._convert_to_catenax_format(pcf_data)

            response = await self._http_client.post(
                "/pcf",
                json=payload
            )
            response.raise_for_status()

            return PCFExchangeResponse(
                success=True,
                pcf_data=pcf_data,
                exchange_id=str(deterministic_uuid(__name__, str(DeterministicClock.now()))),
                timestamp=DeterministicClock.utcnow()
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error publishing PCF: {e}")
            return PCFExchangeResponse(
                success=False,
                validation_errors=[f"HTTP {e.response.status_code}: {e.response.text}"],
                timestamp=DeterministicClock.utcnow()
            )

        except Exception as e:
            logger.error(f"Error publishing PCF: {e}", exc_info=True)
            return PCFExchangeResponse(
                success=False,
                validation_errors=[str(e)],
                timestamp=DeterministicClock.utcnow()
            )

    def _convert_to_catenax_format(self, pcf_data: PCFDataModel) -> Dict[str, Any]:
        """
        Convert PACT format to Catena-X format.

        Args:
            pcf_data: PCF data in PACT format

        Returns:
            PCF data in Catena-X format
        """
        # Catena-X uses PACT as base but may have specific extensions
        payload = pcf_data.dict(exclude_none=True, by_alias=True)

        # Add Catena-X specific fields if needed
        if not payload.get("extensions"):
            payload["extensions"] = {}

        payload["extensions"]["catena_x"] = {
            "network": "catena-x",
            "version": "1.0",
        }

        return payload

    async def close(self):
        """Close HTTP client."""
        await self._http_client.aclose()
