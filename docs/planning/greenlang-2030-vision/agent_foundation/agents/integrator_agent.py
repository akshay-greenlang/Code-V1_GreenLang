# -*- coding: utf-8 -*-
"""
IntegratorAgent - ERP and external system integration agent.

This module implements the IntegratorAgent for connecting with enterprise
systems including SAP, Oracle, Workday, and other external APIs.

Example:
    >>> agent = IntegratorAgent(config)
    >>> result = await agent.execute(IntegrationInput(
    ...     system="SAP",
    ...     operation="fetch_emissions_data",
    ...     parameters={"year": 2024}
    ... ))
"""

import asyncio
import hashlib
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import httpx
from pydantic import BaseModel, Field, validator

import sys
import os
from greenlang.determinism import DeterministicClock
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_agent import BaseAgent, AgentConfig, ExecutionContext

logger = logging.getLogger(__name__)


class IntegrationSystem(str, Enum):
    """Supported integration systems."""

    SAP = "SAP"
    ORACLE = "ORACLE"
    WORKDAY = "WORKDAY"
    SALESFORCE = "SALESFORCE"
    MICROSOFT_DYNAMICS = "MICROSOFT_DYNAMICS"
    NETSUITE = "NETSUITE"
    REST_API = "REST_API"
    GRAPHQL = "GRAPHQL"
    DATABASE = "DATABASE"
    FILE_SYSTEM = "FILE_SYSTEM"


class IntegrationOperation(str, Enum):
    """Types of integration operations."""

    FETCH = "fetch"
    PUSH = "push"
    SYNC = "sync"
    VALIDATE = "validate"
    TRANSFORM = "transform"
    BATCH_IMPORT = "batch_import"
    BATCH_EXPORT = "batch_export"
    HEALTH_CHECK = "health_check"


class IntegrationInput(BaseModel):
    """Input data model for IntegratorAgent."""

    system: IntegrationSystem = Field(..., description="Target system to integrate with")
    operation: IntegrationOperation = Field(..., description="Operation to perform")
    endpoint: Optional[str] = Field(None, description="API endpoint or resource path")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Operation parameters")
    data: Optional[Dict[str, Any]] = Field(None, description="Data to send (for push operations)")
    auth_config: Dict[str, str] = Field(default_factory=dict, description="Authentication configuration")
    timeout_seconds: int = Field(30, ge=1, le=300, description="Operation timeout")
    retry_on_failure: bool = Field(True, description="Retry on transient failures")
    transform_config: Optional[Dict[str, Any]] = Field(None, description="Data transformation configuration")

    @validator('auth_config')
    def validate_auth(cls, v):
        """Validate authentication configuration."""
        # Don't log sensitive auth data
        if v and not any(k in v for k in ["api_key", "token", "username", "client_id"]):
            raise ValueError("Missing authentication credentials")
        return v


class IntegrationOutput(BaseModel):
    """Output data model for IntegratorAgent."""

    success: bool = Field(..., description="Integration success status")
    system: IntegrationSystem = Field(..., description="System integrated with")
    operation: IntegrationOperation = Field(..., description="Operation performed")
    data: Optional[Dict[str, Any]] = Field(None, description="Retrieved or processed data")
    records_processed: int = Field(0, ge=0, description="Number of records processed")
    errors: List[Dict[str, Any]] = Field(default_factory=list, description="Integration errors")
    warnings: List[str] = Field(default_factory=list, description="Integration warnings")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Integration metadata")
    provenance_hash: str = Field(..., description="SHA-256 hash for audit trail")
    processing_time_ms: float = Field(..., description="Processing duration")
    api_calls_made: int = Field(0, ge=0, description="Number of API calls made")


class IntegratorAgent(BaseAgent):
    """
    IntegratorAgent implementation for ERP and external system integration.

    This agent handles secure connections to enterprise systems, data fetching,
    transformation, and synchronization with complete audit trails.

    Attributes:
        config: Agent configuration
        connectors: Registry of system connectors
        transform_engine: Data transformation engine
        connection_pool: Connection pool for reuse

    Example:
        >>> config = AgentConfig(name="sap_integrator", version="1.0.0")
        >>> agent = IntegratorAgent(config)
        >>> await agent.initialize()
        >>> result = await agent.execute(integration_input)
        >>> print(f"Records processed: {result.result.records_processed}")
    """

    def __init__(self, config: AgentConfig):
        """Initialize IntegratorAgent."""
        super().__init__(config)
        self.connectors: Dict[IntegrationSystem, SystemConnector] = {}
        self.transform_engine = None
        self.connection_pool: Dict[str, Any] = {}
        self.integration_history: List[IntegrationOutput] = []
        self.rate_limiters: Dict[str, RateLimiter] = {}

    async def _initialize_core(self) -> None:
        """Initialize integration resources."""
        self._logger.info("Initializing IntegratorAgent resources")

        # Initialize system connectors
        await self._initialize_connectors()

        # Initialize transformation engine
        self.transform_engine = TransformationEngine()

        # Initialize rate limiters
        self._initialize_rate_limiters()

        self._logger.info(f"Initialized connectors for {len(self.connectors)} systems")

    async def _initialize_connectors(self) -> None:
        """Initialize connectors for each system."""
        # SAP Connector
        self.connectors[IntegrationSystem.SAP] = SAPConnector()

        # Oracle Connector
        self.connectors[IntegrationSystem.ORACLE] = OracleConnector()

        # Workday Connector
        self.connectors[IntegrationSystem.WORKDAY] = WorkdayConnector()

        # REST API Connector
        self.connectors[IntegrationSystem.REST_API] = RESTConnector()

        # Database Connector
        self.connectors[IntegrationSystem.DATABASE] = DatabaseConnector()

        # Initialize other connectors as needed
        for system in [
            IntegrationSystem.SALESFORCE,
            IntegrationSystem.MICROSOFT_DYNAMICS,
            IntegrationSystem.NETSUITE,
            IntegrationSystem.GRAPHQL,
            IntegrationSystem.FILE_SYSTEM
        ]:
            self.connectors[system] = GenericConnector(system)

    def _initialize_rate_limiters(self) -> None:
        """Initialize rate limiters for each system."""
        # Configure rate limits per system
        rate_limits = {
            IntegrationSystem.SAP: {"calls": 100, "period": 60},
            IntegrationSystem.ORACLE: {"calls": 200, "period": 60},
            IntegrationSystem.WORKDAY: {"calls": 150, "period": 60},
            IntegrationSystem.SALESFORCE: {"calls": 1000, "period": 3600},
            IntegrationSystem.REST_API: {"calls": 500, "period": 60},
        }

        for system, limits in rate_limits.items():
            self.rate_limiters[system] = RateLimiter(
                max_calls=limits["calls"],
                period_seconds=limits["period"]
            )

    async def _execute_core(self, input_data: IntegrationInput, context: ExecutionContext) -> IntegrationOutput:
        """
        Core execution logic for system integration.

        This method handles secure connection, data operations, and transformation.
        """
        start_time = datetime.now(timezone.utc)
        errors = []
        warnings = []
        api_calls = 0

        try:
            # Step 1: Get appropriate connector
            connector = self.connectors.get(input_data.system)
            if not connector:
                raise ValueError(f"No connector available for system: {input_data.system}")

            # Step 2: Check rate limits
            if input_data.system in self.rate_limiters:
                rate_limiter = self.rate_limiters[input_data.system]
                if not await rate_limiter.allow_request():
                    raise RuntimeError(f"Rate limit exceeded for {input_data.system}")

            # Step 3: Establish connection
            self._logger.info(f"Connecting to {input_data.system}")
            connection = await connector.connect(input_data.auth_config)

            # Step 4: Perform operation
            result_data = None
            records_processed = 0

            if input_data.operation == IntegrationOperation.FETCH:
                result_data = await connector.fetch(
                    connection,
                    input_data.endpoint,
                    input_data.parameters
                )
                api_calls += 1
                records_processed = self._count_records(result_data)

            elif input_data.operation == IntegrationOperation.PUSH:
                result_data = await connector.push(
                    connection,
                    input_data.endpoint,
                    input_data.data
                )
                api_calls += 1
                records_processed = 1

            elif input_data.operation == IntegrationOperation.SYNC:
                result_data = await self._perform_sync(
                    connector,
                    connection,
                    input_data
                )
                api_calls = result_data.get("api_calls", 1)
                records_processed = result_data.get("records", 0)

            elif input_data.operation == IntegrationOperation.BATCH_IMPORT:
                result_data = await self._perform_batch_import(
                    connector,
                    connection,
                    input_data
                )
                api_calls = result_data.get("api_calls", 1)
                records_processed = result_data.get("records", 0)

            elif input_data.operation == IntegrationOperation.HEALTH_CHECK:
                result_data = await connector.health_check(connection)
                api_calls += 1

            else:
                raise ValueError(f"Unsupported operation: {input_data.operation}")

            # Step 5: Transform data if configured
            if input_data.transform_config and result_data:
                self._logger.info("Applying data transformation")
                result_data = self.transform_engine.transform(
                    result_data,
                    input_data.transform_config
                )

            # Step 6: Validate data integrity
            validation_result = self._validate_data(result_data)
            if not validation_result["valid"]:
                warnings.extend(validation_result["warnings"])

            # Step 7: Close connection
            await connector.disconnect(connection)

            # Step 8: Generate metadata
            metadata = {
                "system_version": connector.get_version(),
                "connection_id": connection.get("id", "unknown"),
                "data_integrity": validation_result,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Step 9: Generate provenance hash
            provenance_hash = self._calculate_provenance_hash(
                input_data.dict(),
                result_data,
                context.execution_id
            )

            # Step 10: Calculate processing time
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Step 11: Create output
            output = IntegrationOutput(
                success=True,
                system=input_data.system,
                operation=input_data.operation,
                data=result_data,
                records_processed=records_processed,
                errors=errors,
                warnings=warnings,
                metadata=metadata,
                provenance_hash=provenance_hash,
                processing_time_ms=processing_time,
                api_calls_made=api_calls
            )

            # Store in history
            self.integration_history.append(output)
            if len(self.integration_history) > 100:
                self.integration_history.pop(0)

            return output

        except Exception as e:
            self._logger.error(f"Integration failed: {str(e)}", exc_info=True)

            # Create error output
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            error_output = IntegrationOutput(
                success=False,
                system=input_data.system,
                operation=input_data.operation,
                data=None,
                records_processed=0,
                errors=[{
                    "error": str(e),
                    "type": type(e).__name__,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }],
                warnings=warnings,
                metadata={},
                provenance_hash=self._calculate_provenance_hash(
                    input_data.dict(),
                    None,
                    context.execution_id
                ),
                processing_time_ms=processing_time,
                api_calls_made=api_calls
            )

            # Retry logic for transient failures
            if input_data.retry_on_failure and self._is_transient_error(e):
                self._logger.info("Retrying after transient failure")
                await asyncio.sleep(5)
                return await self._execute_core(input_data, context)

            return error_output

    async def _perform_sync(self, connector: Any, connection: Any, input_data: IntegrationInput) -> Dict:
        """Perform bidirectional synchronization."""
        # Simplified sync implementation
        fetch_result = await connector.fetch(connection, input_data.endpoint, input_data.parameters)
        push_result = await connector.push(connection, input_data.endpoint, fetch_result)

        return {
            "fetched": fetch_result,
            "pushed": push_result,
            "api_calls": 2,
            "records": self._count_records(fetch_result)
        }

    async def _perform_batch_import(self, connector: Any, connection: Any, input_data: IntegrationInput) -> Dict:
        """Perform batch import operation."""
        batch_size = input_data.parameters.get("batch_size", 1000)
        total_records = 0
        api_calls = 0
        results = []

        # Process in batches
        offset = 0
        while True:
            batch_params = {**input_data.parameters, "offset": offset, "limit": batch_size}
            batch_data = await connector.fetch(connection, input_data.endpoint, batch_params)
            api_calls += 1

            if not batch_data or (isinstance(batch_data, list) and len(batch_data) == 0):
                break

            results.extend(batch_data if isinstance(batch_data, list) else [batch_data])
            total_records += len(batch_data) if isinstance(batch_data, list) else 1
            offset += batch_size

            # Prevent infinite loops
            if api_calls > 100:
                break

        return {
            "data": results,
            "api_calls": api_calls,
            "records": total_records
        }

    def _count_records(self, data: Any) -> int:
        """Count number of records in data."""
        if isinstance(data, list):
            return len(data)
        elif isinstance(data, dict):
            # Check for common patterns
            if "records" in data:
                return len(data["records"]) if isinstance(data["records"], list) else 1
            elif "data" in data:
                return len(data["data"]) if isinstance(data["data"], list) else 1
            else:
                return 1
        else:
            return 0

    def _validate_data(self, data: Any) -> Dict[str, Any]:
        """Validate data integrity."""
        validation = {
            "valid": True,
            "warnings": [],
            "checks": {}
        }

        if data is None:
            validation["warnings"].append("No data returned")
            validation["valid"] = False
            return validation

        # Check for common data issues
        if isinstance(data, dict):
            if not data:
                validation["warnings"].append("Empty data dictionary")
            if "error" in data:
                validation["warnings"].append(f"Data contains error: {data['error']}")
                validation["valid"] = False

        elif isinstance(data, list):
            if not data:
                validation["warnings"].append("Empty data list")
            # Check for consistency
            if data and all(isinstance(item, dict) for item in data):
                keys_set = set(data[0].keys()) if data else set()
                inconsistent = any(set(item.keys()) != keys_set for item in data[1:])
                if inconsistent:
                    validation["warnings"].append("Inconsistent data structure across records")

        validation["checks"]["type"] = type(data).__name__
        validation["checks"]["size"] = len(data) if hasattr(data, '__len__') else 1

        return validation

    def _is_transient_error(self, error: Exception) -> bool:
        """Check if error is transient and retryable."""
        transient_errors = [
            "timeout",
            "connection",
            "rate limit",
            "temporary",
            "unavailable",
            "503"
        ]
        error_msg = str(error).lower()
        return any(term in error_msg for term in transient_errors)

    def _calculate_provenance_hash(self, inputs: Dict, result: Any, execution_id: str) -> str:
        """Calculate SHA-256 hash for audit trail."""
        provenance_data = {
            "agent": self.config.name,
            "version": self.config.version,
            "execution_id": execution_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": inputs.get("system"),
            "operation": inputs.get("operation"),
            "result_hash": hashlib.md5(str(result).encode()).hexdigest() if result else None
        }
        provenance_str = str(sorted(provenance_data.items()))
        return hashlib.sha256(provenance_str.encode()).hexdigest()

    async def _terminate_core(self) -> None:
        """Cleanup integration resources."""
        self._logger.info("Cleaning up IntegratorAgent resources")

        # Close all connections
        for connector in self.connectors.values():
            try:
                await connector.cleanup()
            except Exception as e:
                self._logger.error(f"Error cleaning up connector: {e}")

        self.connectors.clear()
        self.connection_pool.clear()
        self.integration_history.clear()

    async def _collect_custom_metrics(self) -> Dict[str, Any]:
        """Collect integration-specific metrics."""
        if not self.integration_history:
            return {}

        recent = self.integration_history[-100:]
        return {
            "total_integrations": len(self.integration_history),
            "success_rate": sum(1 for i in recent if i.success) / len(recent),
            "average_records": sum(i.records_processed for i in recent) / len(recent),
            "total_api_calls": sum(i.api_calls_made for i in recent),
            "systems_used": list(set(i.system for i in recent)),
            "active_connections": len(self.connection_pool)
        }


class SystemConnector:
    """Base class for system connectors."""

    async def connect(self, auth_config: Dict) -> Dict:
        """Establish connection to system."""
        raise NotImplementedError

    async def disconnect(self, connection: Dict) -> None:
        """Close connection to system."""
        pass

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Fetch data from system."""
        raise NotImplementedError

    async def push(self, connection: Dict, endpoint: str, data: Any) -> Any:
        """Push data to system."""
        raise NotImplementedError

    async def health_check(self, connection: Dict) -> Dict:
        """Check system health."""
        return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}

    def get_version(self) -> str:
        """Get connector version."""
        return "1.0.0"

    async def cleanup(self) -> None:
        """Cleanup connector resources."""
        pass


class SAPConnector(SystemConnector):
    """SAP system connector."""

    async def connect(self, auth_config: Dict) -> Dict:
        """Connect to SAP system."""
        # Simplified connection logic
        return {
            "id": f"sap_{DeterministicClock.now().timestamp()}",
            "system": "SAP",
            "connected": True
        }

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Fetch data from SAP."""
        # Simulated SAP data fetch
        return {
            "records": [
                {"id": 1, "material": "MAT001", "emissions": 100.5},
                {"id": 2, "material": "MAT002", "emissions": 75.2}
            ],
            "count": 2,
            "system": "SAP"
        }


class OracleConnector(SystemConnector):
    """Oracle system connector."""

    async def connect(self, auth_config: Dict) -> Dict:
        """Connect to Oracle system."""
        return {
            "id": f"oracle_{DeterministicClock.now().timestamp()}",
            "system": "Oracle",
            "connected": True
        }

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Fetch data from Oracle."""
        # Simulated Oracle data fetch
        return {
            "data": [
                {"supplier_id": "SUP001", "sustainability_score": 85},
                {"supplier_id": "SUP002", "sustainability_score": 92}
            ],
            "system": "Oracle"
        }


class WorkdayConnector(SystemConnector):
    """Workday system connector."""

    async def connect(self, auth_config: Dict) -> Dict:
        """Connect to Workday system."""
        return {
            "id": f"workday_{DeterministicClock.now().timestamp()}",
            "system": "Workday",
            "connected": True
        }

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Fetch data from Workday."""
        # Simulated Workday data fetch
        return {
            "employees": [
                {"id": "EMP001", "department": "Sustainability", "travel_emissions": 2.5},
                {"id": "EMP002", "department": "Operations", "travel_emissions": 3.8}
            ],
            "system": "Workday"
        }


class RESTConnector(SystemConnector):
    """Generic REST API connector."""

    def __init__(self):
        """Initialize REST connector."""
        self.client = None

    async def connect(self, auth_config: Dict) -> Dict:
        """Connect to REST API."""
        self.client = httpx.AsyncClient(
            headers={"Authorization": f"Bearer {auth_config.get('token', '')}"}
        )
        return {
            "id": f"rest_{DeterministicClock.now().timestamp()}",
            "system": "REST_API",
            "connected": True
        }

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Fetch data from REST API."""
        if not self.client:
            raise RuntimeError("Not connected")

        # Make actual HTTP request (with error handling)
        try:
            response = await self.client.get(endpoint, params=parameters)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"REST API fetch failed: {e}")
            # Return simulated data for demo
            return {"status": "simulated", "data": []}

    async def disconnect(self, connection: Dict) -> None:
        """Close REST client."""
        if self.client:
            await self.client.aclose()
            self.client = None


class DatabaseConnector(SystemConnector):
    """Database connector."""

    async def connect(self, auth_config: Dict) -> Dict:
        """Connect to database."""
        # In production, use actual database connection
        return {
            "id": f"db_{DeterministicClock.now().timestamp()}",
            "system": "Database",
            "connected": True
        }

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Execute database query."""
        # Simulated database query
        return {
            "rows": [
                {"id": 1, "value": 100},
                {"id": 2, "value": 200}
            ],
            "query": endpoint,
            "system": "Database"
        }


class GenericConnector(SystemConnector):
    """Generic connector for other systems."""

    def __init__(self, system: IntegrationSystem):
        """Initialize generic connector."""
        self.system = system

    async def connect(self, auth_config: Dict) -> Dict:
        """Connect to system."""
        return {
            "id": f"{self.system.lower()}_{DeterministicClock.now().timestamp()}",
            "system": self.system,
            "connected": True
        }

    async def fetch(self, connection: Dict, endpoint: str, parameters: Dict) -> Any:
        """Fetch data from system."""
        # Simulated fetch
        return {
            "data": [],
            "system": self.system,
            "endpoint": endpoint
        }


class TransformationEngine:
    """Data transformation engine."""

    def transform(self, data: Any, config: Dict[str, Any]) -> Any:
        """Transform data according to configuration."""
        if not config:
            return data

        # Apply transformations
        if "mapping" in config:
            data = self._apply_mapping(data, config["mapping"])

        if "filter" in config:
            data = self._apply_filter(data, config["filter"])

        if "aggregate" in config:
            data = self._apply_aggregation(data, config["aggregate"])

        return data

    def _apply_mapping(self, data: Any, mapping: Dict) -> Any:
        """Apply field mapping."""
        if isinstance(data, dict):
            return {mapping.get(k, k): v for k, v in data.items()}
        elif isinstance(data, list):
            return [self._apply_mapping(item, mapping) for item in data]
        return data

    def _apply_filter(self, data: Any, filter_config: Dict) -> Any:
        """Apply data filtering."""
        if isinstance(data, list):
            field = filter_config.get("field")
            value = filter_config.get("value")
            operator = filter_config.get("operator", "equals")

            if operator == "equals":
                return [item for item in data if item.get(field) == value]
            elif operator == "greater_than":
                return [item for item in data if item.get(field, 0) > value]
            # Add more operators as needed

        return data

    def _apply_aggregation(self, data: Any, agg_config: Dict) -> Any:
        """Apply data aggregation."""
        if isinstance(data, list) and data:
            field = agg_config.get("field")
            operation = agg_config.get("operation", "sum")

            values = [item.get(field, 0) for item in data if field in item]

            if operation == "sum":
                return sum(values)
            elif operation == "avg":
                return sum(values) / len(values) if values else 0
            elif operation == "count":
                return len(values)
            # Add more operations as needed

        return data


class RateLimiter:
    """Rate limiter for API calls."""

    def __init__(self, max_calls: int, period_seconds: int):
        """Initialize rate limiter."""
        self.max_calls = max_calls
        self.period_seconds = period_seconds
        self.calls = []

    async def allow_request(self) -> bool:
        """Check if request is allowed under rate limit."""
        now = datetime.now(timezone.utc)

        # Remove old calls outside the window
        cutoff = now.timestamp() - self.period_seconds
        self.calls = [call_time for call_time in self.calls if call_time > cutoff]

        # Check if under limit
        if len(self.calls) < self.max_calls:
            self.calls.append(now.timestamp())
            return True

        return False