# -*- coding: utf-8 -*-
"""
Blockchain Integration REST API - AGENT-EUDR-013

FastAPI router package providing 37 REST endpoints for EUDR blockchain
integration operations including transaction anchoring, smart contract
management, multi-chain connections, on-chain verification, event
listening, Merkle proof generation, cross-party data sharing, and
compliance evidence packaging.

Route Modules:
    - anchor_routes: Transaction anchoring (single + batch, status, history)
    - contract_routes: Smart contract deploy, call, state, listing
    - chain_routes: Multi-chain connect, status, listing, gas estimation
    - verification_routes: On-chain verify (single, batch, Merkle proof)
    - event_routes: Event subscribe, unsubscribe, query, replay
    - merkle_routes: Merkle tree build, retrieve, proof, verify
    - sharing_routes: Cross-party grant, revoke, list, request, confirm
    - evidence_routes: Evidence package create, retrieve, download, verify
    - router: Main router with batch job and health endpoints

Auth Integration:
    - JWT authentication via SEC-001 dependency injection
    - RBAC via SEC-002 with eudr-bci:* permissions
    - Rate limiting via middleware decorator

Author: GreenLang Platform Team
Date: March 2026
PRD: AGENT-EUDR-013 (Blockchain Integration)
Agent ID: GL-EUDR-BCI-013
Status: Production Ready
"""

from greenlang.agents.eudr.blockchain_integration.api.router import (
    get_router,
    router,
)

__all__ = [
    "router",
    "get_router",
]
