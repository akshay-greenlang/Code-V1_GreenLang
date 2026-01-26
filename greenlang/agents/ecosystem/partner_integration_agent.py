# -*- coding: utf-8 -*-
"""
GL-ECO-X-004: Partner Integration Agent
========================================

Manages partner API integrations, data synchronization, and connectivity
with external systems and platforms.

Author: GreenLang Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import time
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from greenlang.agents.base import AgentConfig, AgentResult, BaseAgent
from greenlang.utilities.determinism import DeterministicClock

logger = logging.getLogger(__name__)


class IntegrationType(str, Enum):
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    SFTP = "sftp"
    DATABASE = "database"
    WEBHOOK = "webhook"
    OAUTH = "oauth"


class ConnectionStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"


class DataMapping(BaseModel):
    mapping_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    source_field: str = Field(..., description="Source field name")
    target_field: str = Field(..., description="Target field name")
    transformation: Optional[str] = Field(None, description="Transformation rule")


class SyncConfiguration(BaseModel):
    sync_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    connection_id: str = Field(..., description="Connection to sync")
    direction: str = Field(default="bidirectional", description="inbound/outbound/bidirectional")
    schedule: str = Field(default="0 * * * *", description="Cron schedule")
    mappings: List[DataMapping] = Field(default_factory=list)
    enabled: bool = Field(default=True)


class PartnerConnection(BaseModel):
    connection_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = Field(..., description="Connection name")
    partner_name: str = Field(..., description="Partner organization")
    integration_type: IntegrationType = Field(..., description="Type of integration")
    status: ConnectionStatus = Field(default=ConnectionStatus.PENDING)
    endpoint_url: Optional[str] = Field(None)
    credentials_ref: Optional[str] = Field(None, description="Reference to stored credentials")
    sync_config: Optional[SyncConfiguration] = Field(None)
    last_sync: Optional[datetime] = Field(None)
    created_at: datetime = Field(default_factory=DeterministicClock.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PartnerIntegrationInput(BaseModel):
    operation: str = Field(..., description="Operation to perform")
    connection: Optional[PartnerConnection] = Field(None)
    connection_id: Optional[str] = Field(None)
    sync_config: Optional[SyncConfiguration] = Field(None)
    data: Optional[Dict[str, Any]] = Field(None)

    @field_validator('operation')
    @classmethod
    def validate_operation(cls, v: str) -> str:
        valid_ops = {
            'create_connection', 'test_connection', 'update_connection',
            'delete_connection', 'list_connections', 'sync_data',
            'configure_sync', 'get_sync_status', 'get_statistics'
        }
        if v not in valid_ops:
            raise ValueError(f"Operation must be one of: {valid_ops}")
        return v


class PartnerIntegrationOutput(BaseModel):
    success: bool = Field(...)
    operation: str = Field(...)
    data: Dict[str, Any] = Field(default_factory=dict)
    provenance_hash: str = Field(default="")
    processing_time_ms: float = Field(default=0.0)
    timestamp: datetime = Field(default_factory=DeterministicClock.now)


class PartnerIntegrationAgent(BaseAgent):
    """GL-ECO-X-004: Partner Integration Agent"""

    AGENT_ID = "GL-ECO-X-004"
    AGENT_NAME = "Partner Integration Agent"
    VERSION = "1.0.0"

    def __init__(self, config: Optional[AgentConfig] = None):
        if config is None:
            config = AgentConfig(
                name=self.AGENT_NAME,
                description="Partner API integration management",
                version=self.VERSION,
            )
        super().__init__(config)
        self._connections: Dict[str, PartnerConnection] = {}
        self._sync_history: List[Dict[str, Any]] = []
        self.logger.info(f"Initialized {self.AGENT_ID}")

    def execute(self, input_data: Dict[str, Any]) -> AgentResult:
        start_time = time.time()
        try:
            pi_input = PartnerIntegrationInput(**input_data)
            result_data = self._route_operation(pi_input)
            provenance_hash = hashlib.sha256(
                json.dumps({"in": input_data, "out": result_data}, sort_keys=True, default=str).encode()
            ).hexdigest()[:16]

            output = PartnerIntegrationOutput(
                success=True,
                operation=pi_input.operation,
                data=result_data,
                provenance_hash=provenance_hash,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            return AgentResult(success=True, data=output.model_dump())
        except Exception as e:
            self.logger.error(f"Operation failed: {e}", exc_info=True)
            return AgentResult(success=False, error=str(e))

    def _route_operation(self, pi_input: PartnerIntegrationInput) -> Dict[str, Any]:
        op = pi_input.operation
        if op == "create_connection":
            return self._create_connection(pi_input.connection)
        elif op == "test_connection":
            return self._test_connection(pi_input.connection_id)
        elif op == "update_connection":
            return self._update_connection(pi_input.connection_id, pi_input.connection)
        elif op == "delete_connection":
            return self._delete_connection(pi_input.connection_id)
        elif op == "list_connections":
            return self._list_connections()
        elif op == "sync_data":
            return self._sync_data(pi_input.connection_id, pi_input.data)
        elif op == "configure_sync":
            return self._configure_sync(pi_input.connection_id, pi_input.sync_config)
        elif op == "get_sync_status":
            return self._get_sync_status(pi_input.connection_id)
        elif op == "get_statistics":
            return self._get_statistics()
        raise ValueError(f"Unknown operation: {op}")

    def _create_connection(self, conn: Optional[PartnerConnection]) -> Dict[str, Any]:
        if not conn:
            return {"error": "connection required"}
        self._connections[conn.connection_id] = conn
        return {"connection_id": conn.connection_id, "created": True}

    def _test_connection(self, conn_id: Optional[str]) -> Dict[str, Any]:
        if not conn_id or conn_id not in self._connections:
            return {"error": f"Connection not found: {conn_id}"}
        conn = self._connections[conn_id]
        conn.status = ConnectionStatus.ACTIVE  # Simulated test
        return {"connection_id": conn_id, "status": ConnectionStatus.ACTIVE.value, "test_passed": True}

    def _update_connection(self, conn_id: Optional[str], conn: Optional[PartnerConnection]) -> Dict[str, Any]:
        if not conn_id or conn_id not in self._connections:
            return {"error": f"Connection not found: {conn_id}"}
        if conn:
            self._connections[conn_id] = conn
        return {"connection_id": conn_id, "updated": True}

    def _delete_connection(self, conn_id: Optional[str]) -> Dict[str, Any]:
        if conn_id and conn_id in self._connections:
            del self._connections[conn_id]
            return {"connection_id": conn_id, "deleted": True}
        return {"error": f"Connection not found: {conn_id}"}

    def _list_connections(self) -> Dict[str, Any]:
        return {
            "connections": [c.model_dump() for c in self._connections.values()],
            "count": len(self._connections),
        }

    def _sync_data(self, conn_id: Optional[str], data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if not conn_id or conn_id not in self._connections:
            return {"error": f"Connection not found: {conn_id}"}
        conn = self._connections[conn_id]
        conn.last_sync = DeterministicClock.now()
        self._sync_history.append({"connection_id": conn_id, "timestamp": conn.last_sync, "records": len(data or {})})
        return {"connection_id": conn_id, "synced": True, "records": len(data or {})}

    def _configure_sync(self, conn_id: Optional[str], sync_config: Optional[SyncConfiguration]) -> Dict[str, Any]:
        if not conn_id or conn_id not in self._connections:
            return {"error": f"Connection not found: {conn_id}"}
        if sync_config:
            self._connections[conn_id].sync_config = sync_config
        return {"connection_id": conn_id, "sync_configured": True}

    def _get_sync_status(self, conn_id: Optional[str]) -> Dict[str, Any]:
        if not conn_id or conn_id not in self._connections:
            return {"error": f"Connection not found: {conn_id}"}
        conn = self._connections[conn_id]
        return {"connection_id": conn_id, "last_sync": conn.last_sync, "sync_enabled": bool(conn.sync_config and conn.sync_config.enabled)}

    def _get_statistics(self) -> Dict[str, Any]:
        return {
            "total_connections": len(self._connections),
            "active_connections": sum(1 for c in self._connections.values() if c.status == ConnectionStatus.ACTIVE),
            "total_syncs": len(self._sync_history),
        }
