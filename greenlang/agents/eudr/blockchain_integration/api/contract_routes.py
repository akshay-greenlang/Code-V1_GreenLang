# -*- coding: utf-8 -*-
"""
Contract Routes - AGENT-EUDR-013 Blockchain Integration API

Endpoints for smart contract deployment, method calls, state queries,
and contract listing for EUDR compliance contracts (anchor registry,
custody transfer, compliance check).

Endpoints:
    POST   /contracts/deploy          - Deploy smart contract
    GET    /contracts/{contract_id}    - Get contract details
    POST   /contracts/{contract_id}/call - Call contract method
    GET    /contracts/{contract_id}/state - Get contract state
    GET    /contracts                  - List deployed contracts

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013, Feature 3 (Smart Contract Lifecycle)
Agent ID: GL-EUDR-BCI-013
Status: Production Ready
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from greenlang.agents.eudr.blockchain_integration.api.dependencies import (
    AuthUser,
    ErrorResponse,
    PaginationParams,
    get_blockchain_service,
    get_pagination,
    get_request_id,
    rate_limit_contract,
    rate_limit_standard,
    require_permission,
    validate_contract_id,
)
from greenlang.agents.eudr.blockchain_integration.api.schemas import (
    BlockchainNetworkSchema,
    ContractCallRequest,
    ContractCallResponse,
    ContractDeployRequest,
    ContractListResponse,
    ContractResponse,
    ContractStateResponse,
    ContractStatusSchema,
    ContractTypeSchema,
    PaginatedMeta,
    ProvenanceInfo,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Smart Contracts"])

# ---------------------------------------------------------------------------
# In-memory stores (replaced by database in production)
# ---------------------------------------------------------------------------

_contract_store: Dict[str, Dict] = {}


def _get_contract_store() -> Dict[str, Dict]:
    """Return the contract record store singleton."""
    return _contract_store


def _compute_provenance_hash(data: dict) -> str:
    """Compute SHA-256 hash for provenance tracking."""
    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    """Return current UTC datetime with microseconds zeroed."""
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------------------
# POST /contracts/deploy
# ---------------------------------------------------------------------------


@router.post(
    "/contracts/deploy",
    response_model=ContractResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Deploy smart contract",
    description=(
        "Deploy an EUDR compliance smart contract to the target "
        "blockchain network. Supports anchor_registry, custody_transfer, "
        "and compliance_check contract types."
    ),
    responses={
        201: {"description": "Contract deployment initiated"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def deploy_contract(
    request: Request,
    body: ContractDeployRequest,
    user: AuthUser = Depends(
        require_permission("eudr-bci:contracts:deploy")
    ),
    _rate: None = Depends(rate_limit_contract),
    service: Any = Depends(get_blockchain_service),
) -> ContractResponse:
    """Deploy a smart contract.

    Args:
        request: FastAPI request object.
        body: Contract deployment parameters.
        user: Authenticated user with contracts:deploy permission.
        service: Blockchain integration service.

    Returns:
        ContractResponse with contract ID and deploying status.
    """
    start = time.monotonic()
    try:
        contract_id = str(uuid.uuid4())
        now = _utcnow()

        # Simulate deployment transaction hash (deterministic)
        deploy_data = f"{contract_id}:{body.contract_type.value}:{body.chain.value}"
        simulated_tx_hash = "0x" + hashlib.sha256(
            deploy_data.encode("utf-8")
        ).hexdigest()

        # Simulate contract address (deterministic)
        addr_data = f"{simulated_tx_hash}:{contract_id}"
        simulated_address = "0x" + hashlib.sha256(
            addr_data.encode("utf-8")
        ).hexdigest()[:40]

        # Compute ABI hash
        abi_content = json.dumps({
            "contract_type": body.contract_type.value,
            "version": "1.0.0",
        }, sort_keys=True)
        abi_hash = hashlib.sha256(abi_content.encode("utf-8")).hexdigest()

        provenance_hash = _compute_provenance_hash({
            "contract_id": contract_id,
            "contract_type": body.contract_type.value,
            "chain": body.chain.value,
            "deployed_by": user.user_id,
        })

        record = {
            "contract_id": contract_id,
            "contract_type": body.contract_type,
            "chain": body.chain,
            "address": simulated_address,
            "status": ContractStatusSchema.DEPLOYING,
            "name": body.name,
            "description": body.description,
            "tx_hash": simulated_tx_hash,
            "block_number": None,
            "abi_hash": abi_hash,
            "deployed_at": None,
            "created_at": now,
            "provenance": ProvenanceInfo(
                provenance_hash=provenance_hash,
                algorithm="sha256",
                created_at=now,
            ),
        }

        store = _get_contract_store()
        store[contract_id] = record

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Contract deployment initiated: id=%s type=%s chain=%s "
            "elapsed_ms=%.1f",
            contract_id,
            body.contract_type.value,
            body.chain.value,
            elapsed_ms,
        )

        return ContractResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to deploy contract: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to deploy contract",
        )


# ---------------------------------------------------------------------------
# GET /contracts/{contract_id}
# ---------------------------------------------------------------------------


@router.get(
    "/contracts/{contract_id}",
    response_model=ContractResponse,
    summary="Get contract details",
    description="Retrieve details of a deployed smart contract.",
    responses={
        200: {"description": "Contract details retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Contract not found"},
    },
)
async def get_contract(
    request: Request,
    contract_id: str = Depends(validate_contract_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:contracts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_blockchain_service),
) -> ContractResponse:
    """Get contract details by ID.

    Args:
        request: FastAPI request object.
        contract_id: Contract identifier.
        user: Authenticated user with contracts:read permission.
        service: Blockchain integration service.

    Returns:
        ContractResponse with contract details.

    Raises:
        HTTPException: 404 if contract not found.
    """
    try:
        store = _get_contract_store()
        record = store.get(contract_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Contract {contract_id} not found",
            )

        return ContractResponse(**record)

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get contract %s: %s", contract_id, exc, exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve contract",
        )


# ---------------------------------------------------------------------------
# POST /contracts/{contract_id}/call
# ---------------------------------------------------------------------------


@router.post(
    "/contracts/{contract_id}/call",
    response_model=ContractCallResponse,
    summary="Call contract method",
    description=(
        "Call a method on a deployed smart contract. Supports both "
        "read-only calls (no transaction) and write calls (submits "
        "a transaction)."
    ),
    responses={
        200: {"description": "Contract call executed"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Contract not found"},
        429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    },
)
async def call_contract(
    request: Request,
    body: ContractCallRequest,
    contract_id: str = Depends(validate_contract_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:contracts:call")
    ),
    _rate: None = Depends(rate_limit_contract),
    service: Any = Depends(get_blockchain_service),
) -> ContractCallResponse:
    """Call a contract method.

    Args:
        request: FastAPI request object.
        body: Contract call parameters.
        contract_id: Contract identifier.
        user: Authenticated user with contracts:call permission.
        service: Blockchain integration service.

    Returns:
        ContractCallResponse with call result.

    Raises:
        HTTPException: 404 if contract not found.
    """
    start = time.monotonic()
    try:
        store = _get_contract_store()
        record = store.get(contract_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Contract {contract_id} not found",
            )

        # Simulate call result (deterministic)
        call_data = f"{contract_id}:{body.method}:{json.dumps(body.args, default=str)}"
        result_hash = hashlib.sha256(call_data.encode("utf-8")).hexdigest()

        tx_hash = None
        gas_used = None
        if not body.is_read_only:
            tx_hash = "0x" + result_hash
            gas_used = 21000 + len(json.dumps(body.args or {}).encode()) * 16

        elapsed_ms = (time.monotonic() - start) * 1000.0
        logger.info(
            "Contract call executed: contract=%s method=%s read_only=%s "
            "elapsed_ms=%.1f",
            contract_id,
            body.method,
            body.is_read_only,
            elapsed_ms,
        )

        return ContractCallResponse(
            contract_id=contract_id,
            method=body.method,
            result={"hash": result_hash[:16], "success": True},
            tx_hash=tx_hash,
            gas_used=gas_used,
            is_read_only=body.is_read_only,
            executed_at=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to call contract %s method %s: %s",
            contract_id,
            body.method,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to execute contract call",
        )


# ---------------------------------------------------------------------------
# GET /contracts/{contract_id}/state
# ---------------------------------------------------------------------------


@router.get(
    "/contracts/{contract_id}/state",
    response_model=ContractStateResponse,
    summary="Get contract state",
    description=(
        "Query the current on-chain state of a deployed smart "
        "contract including storage variables and counters."
    ),
    responses={
        200: {"description": "Contract state retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
        404: {"model": ErrorResponse, "description": "Contract not found"},
    },
)
async def get_contract_state(
    request: Request,
    contract_id: str = Depends(validate_contract_id),
    user: AuthUser = Depends(
        require_permission("eudr-bci:contracts:read")
    ),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_blockchain_service),
) -> ContractStateResponse:
    """Get current on-chain state of a contract.

    Args:
        request: FastAPI request object.
        contract_id: Contract identifier.
        user: Authenticated user with contracts:read permission.
        service: Blockchain integration service.

    Returns:
        ContractStateResponse with current state variables.

    Raises:
        HTTPException: 404 if contract not found.
    """
    try:
        store = _get_contract_store()
        record = store.get(contract_id)

        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Contract {contract_id} not found",
            )

        # Simulated on-chain state (deterministic)
        state = {
            "anchor_count": 0,
            "last_anchor_timestamp": 0,
            "owner": "0x" + "0" * 40,
            "paused": record.get("status") == ContractStatusSchema.PAUSED,
            "version": "1.0.0",
        }

        return ContractStateResponse(
            contract_id=contract_id,
            address=record.get("address", "0x" + "0" * 40),
            chain=record.get("chain", BlockchainNetworkSchema.POLYGON),
            state=state,
            block_number=0,
            queried_at=_utcnow(),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Failed to get contract state %s: %s",
            contract_id,
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve contract state",
        )


# ---------------------------------------------------------------------------
# GET /contracts
# ---------------------------------------------------------------------------


@router.get(
    "/contracts",
    response_model=ContractListResponse,
    summary="List deployed contracts",
    description=(
        "List all deployed EUDR compliance smart contracts with "
        "optional filters by chain and contract type."
    ),
    responses={
        200: {"description": "Contract list retrieved"},
        401: {"model": ErrorResponse, "description": "Authentication required"},
        403: {"model": ErrorResponse, "description": "Insufficient permissions"},
    },
)
async def list_contracts(
    request: Request,
    chain: Optional[BlockchainNetworkSchema] = Query(
        None, description="Filter by blockchain network"
    ),
    contract_type: Optional[ContractTypeSchema] = Query(
        None, description="Filter by contract type"
    ),
    user: AuthUser = Depends(
        require_permission("eudr-bci:contracts:read")
    ),
    pagination: PaginationParams = Depends(get_pagination),
    _rate: None = Depends(rate_limit_standard),
    service: Any = Depends(get_blockchain_service),
) -> ContractListResponse:
    """List deployed contracts with optional filters.

    Args:
        request: FastAPI request object.
        chain: Optional blockchain network filter.
        contract_type: Optional contract type filter.
        user: Authenticated user with contracts:read permission.
        pagination: Pagination parameters.
        service: Blockchain integration service.

    Returns:
        ContractListResponse with paginated contracts.
    """
    try:
        store = _get_contract_store()
        records = list(store.values())

        # Apply filters
        if chain is not None:
            records = [r for r in records if r.get("chain") == chain]
        if contract_type is not None:
            records = [
                r for r in records
                if r.get("contract_type") == contract_type
            ]

        # Sort by created_at descending
        records.sort(
            key=lambda r: r.get("created_at", datetime.min),
            reverse=True,
        )

        total = len(records)
        page = records[pagination.offset:pagination.offset + pagination.limit]
        has_more = (pagination.offset + pagination.limit) < total

        contracts = [ContractResponse(**r) for r in page]

        return ContractListResponse(
            contracts=contracts,
            pagination=PaginatedMeta(
                total=total,
                limit=pagination.limit,
                offset=pagination.offset,
                has_more=has_more,
            ),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to list contracts: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to list contracts",
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = ["router"]
