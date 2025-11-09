"""
PACT Pathfinder v2.0 Client
Partnership for Carbon Transparency - Pathfinder Framework

Implements PACT Pathfinder Technical Specifications v2.0 for PCF exchange.

Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import httpx
import uuid

from greenlang.services.pcf_exchange.models import (
    PCFDataModel,
    PCFExchangeResponse,
)

logger = logging.getLogger(__name__)


class PACTPathfinderClient:
    """
    PACT Pathfinder v2.0 API Client.

    Implements the Pathfinder Technical Specifications for PCF exchange:
    - GET /2/footprints/{id}
    - POST /2/footprints
    - GET /2/footprints (list action)
    - PUT /2/footprints/{id}
    - DELETE /2/footprints/{id}
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PACT client.

        Args:
            config: Client configuration with endpoint, auth, etc.
        """
        self.base_url = config.get(
            "base_url",
            "https://api.pathfinder.sine.dev/2"
        )
        self.api_key = config.get("api_key")
        self.timeout = config.get("timeout", 30.0)

        self._http_client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            headers=self._get_headers()
        )

        logger.info(f"Initialized PACTPathfinderClient (base_url={self.base_url})")

    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for API requests."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        return headers

    async def get_pcf(self, pcf_id: str) -> PCFExchangeResponse:
        """
        Retrieve PCF by ID.

        Args:
            pcf_id: PCF identifier

        Returns:
            PCF exchange response
        """
        try:
            response = await self._http_client.get(f"/footprints/{pcf_id}")
            response.raise_for_status()

            data = response.json()
            pcf_data = PCFDataModel(**data)

            return PCFExchangeResponse(
                success=True,
                pcf_data=pcf_data,
                exchange_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow()
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching PCF {pcf_id}: {e}")
            return PCFExchangeResponse(
                success=False,
                validation_errors=[f"HTTP {e.response.status_code}: {e.response.text}"],
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error fetching PCF {pcf_id}: {e}", exc_info=True)
            return PCFExchangeResponse(
                success=False,
                validation_errors=[str(e)],
                timestamp=datetime.utcnow()
            )

    async def publish_pcf(self, pcf_data: PCFDataModel) -> PCFExchangeResponse:
        """
        Publish PCF to PACT network.

        Args:
            pcf_data: PCF data model

        Returns:
            PCF exchange response
        """
        try:
            # Convert to dict for API
            payload = pcf_data.dict(exclude_none=True, by_alias=True)

            response = await self._http_client.post(
                "/footprints",
                json=payload
            )
            response.raise_for_status()

            return PCFExchangeResponse(
                success=True,
                pcf_data=pcf_data,
                exchange_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow()
            )

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error publishing PCF: {e}")
            return PCFExchangeResponse(
                success=False,
                validation_errors=[f"HTTP {e.response.status_code}: {e.response.text}"],
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error publishing PCF: {e}", exc_info=True)
            return PCFExchangeResponse(
                success=False,
                validation_errors=[str(e)],
                timestamp=datetime.utcnow()
            )

    async def list_pcfs(
        self,
        limit: int = 100,
        filter_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        List PCFs with filtering.

        Args:
            limit: Maximum number of results
            filter_params: Optional filter parameters

        Returns:
            Dictionary with PCF list
        """
        try:
            params = {"limit": limit}
            if filter_params:
                params.update(filter_params)

            response = await self._http_client.get(
                "/footprints",
                params=params
            )
            response.raise_for_status()

            return response.json()

        except Exception as e:
            logger.error(f"Error listing PCFs: {e}", exc_info=True)
            return {"data": [], "error": str(e)}

    async def update_pcf(
        self,
        pcf_id: str,
        pcf_data: PCFDataModel
    ) -> PCFExchangeResponse:
        """
        Update existing PCF.

        Args:
            pcf_id: PCF identifier
            pcf_data: Updated PCF data

        Returns:
            PCF exchange response
        """
        try:
            payload = pcf_data.dict(exclude_none=True, by_alias=True)

            response = await self._http_client.put(
                f"/footprints/{pcf_id}",
                json=payload
            )
            response.raise_for_status()

            return PCFExchangeResponse(
                success=True,
                pcf_data=pcf_data,
                exchange_id=str(uuid.uuid4()),
                timestamp=datetime.utcnow()
            )

        except Exception as e:
            logger.error(f"Error updating PCF {pcf_id}: {e}", exc_info=True)
            return PCFExchangeResponse(
                success=False,
                validation_errors=[str(e)],
                timestamp=datetime.utcnow()
            )

    async def delete_pcf(self, pcf_id: str) -> bool:
        """
        Delete PCF.

        Args:
            pcf_id: PCF identifier

        Returns:
            True if successful
        """
        try:
            response = await self._http_client.delete(f"/footprints/{pcf_id}")
            response.raise_for_status()
            return True

        except Exception as e:
            logger.error(f"Error deleting PCF {pcf_id}: {e}", exc_info=True)
            return False

    async def close(self):
        """Close HTTP client."""
        await self._http_client.aclose()
