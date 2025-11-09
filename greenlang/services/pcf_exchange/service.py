"""
PCF Exchange Service Implementation
Main orchestration service for PCF import/export

Coordinates PCF exchange across multiple protocols:
- PACT Pathfinder v2.0
- Catena-X
- SAP SDX

Version: 1.0.0
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

from greenlang.services.pcf_exchange.models import (
    PCFDataModel,
    PCFExchangeRequest,
    PCFExchangeResponse,
)
from greenlang.services.pcf_exchange.pact_client import PACTPathfinderClient
from greenlang.services.pcf_exchange.catenax_client import CatenaXClient

logger = logging.getLogger(__name__)


class PCFExchangeService:
    """
    Main PCF Exchange Service.

    Orchestrates PCF import/export across multiple exchange protocols
    with validation, versioning, and data quality checks.
    """

    def __init__(
        self,
        pact_config: Optional[Dict[str, Any]] = None,
        catenax_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize PCF Exchange Service.

        Args:
            pact_config: PACT Pathfinder client configuration
            catenax_config: Catena-X client configuration
        """
        self.pact_client = PACTPathfinderClient(pact_config or {})
        self.catenax_client = CatenaXClient(catenax_config or {})

        self._stats = {
            "total_exchanges": 0,
            "successful_exchanges": 0,
            "failed_exchanges": 0,
            "validations": 0,
        }

        logger.info("Initialized PCFExchangeService")

    async def exchange(
        self,
        request: PCFExchangeRequest
    ) -> PCFExchangeResponse:
        """
        Exchange PCF data.

        Args:
            request: PCF exchange request

        Returns:
            PCF exchange response
        """
        self._stats["total_exchanges"] += 1

        try:
            # Validate request
            validation_errors = self._validate_request(request)
            if validation_errors:
                return PCFExchangeResponse(
                    success=False,
                    validation_errors=validation_errors,
                    timestamp=datetime.utcnow()
                )

            # Route to appropriate client
            if request.target_system == "pact":
                response = await self._exchange_pact(request)
            elif request.target_system == "catenax":
                response = await self._exchange_catenax(request)
            elif request.target_system == "sap_sdx":
                response = await self._exchange_sap_sdx(request)
            else:
                return PCFExchangeResponse(
                    success=False,
                    validation_errors=[f"Unknown target system: {request.target_system}"],
                    timestamp=datetime.utcnow()
                )

            if response.success:
                self._stats["successful_exchanges"] += 1
            else:
                self._stats["failed_exchanges"] += 1

            return response

        except Exception as e:
            logger.error(f"PCF exchange failed: {e}", exc_info=True)
            self._stats["failed_exchanges"] += 1

            return PCFExchangeResponse(
                success=False,
                validation_errors=[str(e)],
                timestamp=datetime.utcnow()
            )

    def validate_pcf(
        self,
        pcf_data: PCFDataModel
    ) -> PCFExchangeResponse:
        """
        Validate PCF data without exchanging.

        Args:
            pcf_data: PCF data model

        Returns:
            Validation response
        """
        self._stats["validations"] += 1

        validation_errors = []
        warnings = []

        # Validate basic structure
        try:
            # Pydantic validation happens automatically
            pass
        except Exception as e:
            validation_errors.append(f"Structure validation failed: {e}")

        # Validate carbon footprint data
        pcf = pcf_data.pcf

        # Check that fossil emissions are present
        if pcf.fossil_ghg_emissions <= 0:
            validation_errors.append("Fossil GHG emissions must be > 0")

        # Check that PCF excluding biogenic is >= fossil emissions
        if pcf.p_cf_excluding_biogenic < pcf.fossil_ghg_emissions:
            validation_errors.append(
                "PCF excluding biogenic must be >= fossil GHG emissions"
            )

        # Check exempted emissions threshold
        if pcf.exempted_emissions_percent > 5.0:
            validation_errors.append(
                "Exempted emissions cannot exceed 5% per PACT guidelines"
            )

        # Check reference period
        if pcf.reference_period_end <= pcf.reference_period_start:
            validation_errors.append(
                "Reference period end must be after start"
            )

        # Warnings for data quality
        if pcf.primary_data_share is not None and pcf.primary_data_share < 50:
            warnings.append(
                f"Primary data share is low ({pcf.primary_data_share}%). "
                "Consider supplier engagement for better quality."
            )

        return PCFExchangeResponse(
            success=len(validation_errors) == 0,
            pcf_data=pcf_data if len(validation_errors) == 0 else None,
            validation_errors=validation_errors if validation_errors else None,
            warnings=warnings if warnings else None,
            timestamp=datetime.utcnow()
        )

    async def _exchange_pact(
        self,
        request: PCFExchangeRequest
    ) -> PCFExchangeResponse:
        """Exchange via PACT Pathfinder."""
        if request.operation == "import":
            return await self.pact_client.get_pcf(request.pcf_id)
        elif request.operation == "export":
            return await self.pact_client.publish_pcf(request.pcf_data)
        elif request.operation == "validate":
            return self.validate_pcf(request.pcf_data)
        else:
            return PCFExchangeResponse(
                success=False,
                validation_errors=[f"Unknown operation: {request.operation}"],
                timestamp=datetime.utcnow()
            )

    async def _exchange_catenax(
        self,
        request: PCFExchangeRequest
    ) -> PCFExchangeResponse:
        """Exchange via Catena-X."""
        if request.operation == "import":
            return await self.catenax_client.get_pcf(request.pcf_id)
        elif request.operation == "export":
            return await self.catenax_client.publish_pcf(request.pcf_data)
        elif request.operation == "validate":
            return self.validate_pcf(request.pcf_data)
        else:
            return PCFExchangeResponse(
                success=False,
                validation_errors=[f"Unknown operation: {request.operation}"],
                timestamp=datetime.utcnow()
            )

    async def _exchange_sap_sdx(
        self,
        request: PCFExchangeRequest
    ) -> PCFExchangeResponse:
        """Exchange via SAP SDX (stub implementation)."""
        # SAP SDX implementation would go here
        return PCFExchangeResponse(
            success=False,
            validation_errors=["SAP SDX integration not yet implemented"],
            timestamp=datetime.utcnow()
        )

    def _validate_request(
        self,
        request: PCFExchangeRequest
    ) -> Optional[list]:
        """Validate exchange request."""
        errors = []

        # Validate operation
        valid_operations = ["import", "export", "validate"]
        if request.operation not in valid_operations:
            errors.append(
                f"Invalid operation: {request.operation}. "
                f"Must be one of {valid_operations}"
            )

        # Validate target system
        valid_systems = ["pact", "catenax", "sap_sdx"]
        if request.target_system not in valid_systems:
            errors.append(
                f"Invalid target system: {request.target_system}. "
                f"Must be one of {valid_systems}"
            )

        # Validate operation-specific requirements
        if request.operation == "import" and not request.pcf_id:
            errors.append("PCF ID required for import operation")

        if request.operation in ["export", "validate"] and not request.pcf_data:
            errors.append("PCF data required for export/validate operation")

        return errors if errors else None

    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        return self._stats.copy()
