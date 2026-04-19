# -*- coding: utf-8 -*-
"""
Pytest configuration for Contract Testing

Provides:
- Fixtures for contract test setup and teardown
- Pact provider/consumer configuration
- Event schema registry fixtures
- gRPC service stubs and mocks
- Contract verification utilities

Author: GreenLang Test Engineering
Date: December 2025
"""

import pytest
import json
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from unittest.mock import MagicMock, AsyncMock

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ==============================================================================
# Enums and Data Classes
# ==============================================================================

class ContractType(Enum):
    """Types of contracts."""
    REST_API = "rest_api"
    GRAPHQL = "graphql"
    EVENT = "event"
    GRPC = "grpc"


class ContractStatus(Enum):
    """Contract verification status."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ContractDefinition:
    """Represents a contract definition."""
    name: str
    version: str
    contract_type: ContractType
    provider: str
    consumer: str
    schema: Dict[str, Any]
    created_at: datetime = field(default_factory=datetime.now)
    last_verified: Optional[datetime] = None
    status: ContractStatus = ContractStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "contract_type": self.contract_type.value,
            "provider": self.provider,
            "consumer": self.consumer,
            "schema": self.schema,
            "created_at": self.created_at.isoformat(),
            "last_verified": self.last_verified.isoformat() if self.last_verified else None,
            "status": self.status.value,
        }


@dataclass
class ContractVerificationResult:
    """Result of contract verification."""
    contract_name: str
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    verified_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "contract_name": self.contract_name,
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
            "execution_time_ms": self.execution_time_ms,
            "verified_at": self.verified_at.isoformat(),
        }


# ==============================================================================
# Contract Registry
# ==============================================================================

class ContractRegistry:
    """
    Registry for managing contract definitions.

    Provides centralized storage and retrieval of contracts
    for verification testing.
    """

    def __init__(self, contracts_dir: Optional[Path] = None):
        """Initialize contract registry."""
        self.contracts_dir = contracts_dir or Path(__file__).parent / "contract_definitions"
        self.contracts: Dict[str, ContractDefinition] = {}
        self._load_contracts()

    def _load_contracts(self):
        """Load contracts from disk."""
        if not self.contracts_dir.exists():
            self.contracts_dir.mkdir(parents=True, exist_ok=True)
            return

        for file_path in self.contracts_dir.glob("*.json"):
            try:
                with open(file_path) as f:
                    data = json.load(f)
                contract = ContractDefinition(
                    name=data["name"],
                    version=data["version"],
                    contract_type=ContractType(data["contract_type"]),
                    provider=data["provider"],
                    consumer=data["consumer"],
                    schema=data["schema"],
                )
                self.contracts[contract.name] = contract
                logger.info(f"Loaded contract: {contract.name}")
            except Exception as e:
                logger.error(f"Failed to load contract {file_path}: {e}")

    def register(self, contract: ContractDefinition):
        """Register a contract."""
        self.contracts[contract.name] = contract
        logger.info(f"Registered contract: {contract.name}")

    def get(self, name: str) -> Optional[ContractDefinition]:
        """Get a contract by name."""
        return self.contracts.get(name)

    def list_by_type(self, contract_type: ContractType) -> List[ContractDefinition]:
        """List contracts by type."""
        return [c for c in self.contracts.values() if c.contract_type == contract_type]

    def list_by_provider(self, provider: str) -> List[ContractDefinition]:
        """List contracts by provider."""
        return [c for c in self.contracts.values() if c.provider == provider]

    def save_contract(self, contract: ContractDefinition):
        """Save contract to disk."""
        file_path = self.contracts_dir / f"{contract.name}.json"
        with open(file_path, "w") as f:
            json.dump(contract.to_dict(), f, indent=2)
        logger.info(f"Saved contract: {contract.name} to {file_path}")


# ==============================================================================
# Pact-like Mock Server
# ==============================================================================

class MockInteraction:
    """Represents a mock interaction for contract testing."""

    def __init__(
        self,
        description: str,
        request: Dict[str, Any],
        response: Dict[str, Any],
        provider_states: Optional[List[str]] = None,
    ):
        self.description = description
        self.request = request
        self.response = response
        self.provider_states = provider_states or []
        self.matched = False

    def matches(self, actual_request: Dict[str, Any]) -> bool:
        """Check if actual request matches expected request."""
        # Match method
        if self.request.get("method") != actual_request.get("method"):
            return False

        # Match path
        if self.request.get("path") != actual_request.get("path"):
            return False

        # Match headers (subset match)
        expected_headers = self.request.get("headers", {})
        actual_headers = actual_request.get("headers", {})
        for key, value in expected_headers.items():
            if actual_headers.get(key) != value:
                return False

        # Match query params
        expected_query = self.request.get("query", {})
        actual_query = actual_request.get("query", {})
        if expected_query and expected_query != actual_query:
            return False

        self.matched = True
        return True


class PactMockServer:
    """
    Pact-style mock server for consumer-driven contract testing.

    Simulates provider responses based on defined interactions.
    """

    def __init__(self, consumer: str, provider: str):
        self.consumer = consumer
        self.provider = provider
        self.interactions: List[MockInteraction] = []
        self.requests_received: List[Dict[str, Any]] = []

    def given(self, provider_state: str) -> "PactMockServer":
        """Set provider state for next interaction."""
        self._current_provider_states = [provider_state]
        return self

    def upon_receiving(self, description: str) -> "PactMockServer":
        """Set description for next interaction."""
        self._current_description = description
        return self

    def with_request(
        self,
        method: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
        query: Optional[Dict[str, str]] = None,
    ) -> "PactMockServer":
        """Define expected request."""
        self._current_request = {
            "method": method,
            "path": path,
            "headers": headers or {},
            "body": body,
            "query": query or {},
        }
        return self

    def will_respond_with(
        self,
        status: int,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Dict[str, Any]] = None,
    ) -> "PactMockServer":
        """Define expected response."""
        response = {
            "status": status,
            "headers": headers or {"Content-Type": "application/json"},
            "body": body or {},
        }

        interaction = MockInteraction(
            description=self._current_description,
            request=self._current_request,
            response=response,
            provider_states=getattr(self, "_current_provider_states", []),
        )
        self.interactions.append(interaction)

        # Reset state
        self._current_description = None
        self._current_request = None
        self._current_provider_states = []

        return self

    def handle_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle incoming request and return matching response."""
        self.requests_received.append(request)

        for interaction in self.interactions:
            if interaction.matches(request):
                return interaction.response

        return None

    def verify(self) -> ContractVerificationResult:
        """Verify all interactions were matched."""
        unmatched = [i for i in self.interactions if not i.matched]

        result = ContractVerificationResult(
            contract_name=f"{self.consumer}-{self.provider}",
            passed=len(unmatched) == 0,
            errors=[f"Unmatched interaction: {i.description}" for i in unmatched],
        )

        return result

    def write_pact(self, output_dir: Path):
        """Write pact file to disk."""
        pact = {
            "consumer": {"name": self.consumer},
            "provider": {"name": self.provider},
            "interactions": [
                {
                    "description": i.description,
                    "providerStates": [{"name": s} for s in i.provider_states],
                    "request": i.request,
                    "response": i.response,
                }
                for i in self.interactions
            ],
            "metadata": {
                "pactSpecification": {"version": "3.0.0"},
            },
        }

        output_path = output_dir / f"{self.consumer}-{self.provider}.json"
        with open(output_path, "w") as f:
            json.dump(pact, f, indent=2)

        logger.info(f"Wrote pact file: {output_path}")


# ==============================================================================
# Event Schema Validator
# ==============================================================================

class EventSchemaValidator:
    """
    Validates event payloads against defined schemas.

    Uses JSON Schema for validation.
    """

    def __init__(self):
        self.schemas: Dict[str, Dict[str, Any]] = {}

    def register_schema(self, event_type: str, schema: Dict[str, Any]):
        """Register an event schema."""
        self.schemas[event_type] = schema
        logger.info(f"Registered schema for event: {event_type}")

    def validate(self, event_type: str, payload: Dict[str, Any]) -> ContractVerificationResult:
        """Validate event payload against schema."""
        import time
        start_time = time.perf_counter()

        errors = []
        warnings = []

        if event_type not in self.schemas:
            errors.append(f"No schema registered for event type: {event_type}")
            return ContractVerificationResult(
                contract_name=event_type,
                passed=False,
                errors=errors,
                execution_time_ms=(time.perf_counter() - start_time) * 1000,
            )

        schema = self.schemas[event_type]

        # Validate required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in payload:
                errors.append(f"Missing required field: {field}")

        # Validate field types
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name in payload:
                expected_type = field_schema.get("type")
                actual_value = payload[field_name]

                if expected_type == "string" and not isinstance(actual_value, str):
                    errors.append(f"Field '{field_name}' should be string, got {type(actual_value).__name__}")
                elif expected_type == "number" and not isinstance(actual_value, (int, float)):
                    errors.append(f"Field '{field_name}' should be number, got {type(actual_value).__name__}")
                elif expected_type == "integer" and not isinstance(actual_value, int):
                    errors.append(f"Field '{field_name}' should be integer, got {type(actual_value).__name__}")
                elif expected_type == "boolean" and not isinstance(actual_value, bool):
                    errors.append(f"Field '{field_name}' should be boolean, got {type(actual_value).__name__}")
                elif expected_type == "array" and not isinstance(actual_value, list):
                    errors.append(f"Field '{field_name}' should be array, got {type(actual_value).__name__}")
                elif expected_type == "object" and not isinstance(actual_value, dict):
                    errors.append(f"Field '{field_name}' should be object, got {type(actual_value).__name__}")

        # Check for additional properties
        if not schema.get("additionalProperties", True):
            defined_props = set(properties.keys())
            actual_props = set(payload.keys())
            extra_props = actual_props - defined_props
            if extra_props:
                warnings.append(f"Additional properties found: {extra_props}")

        return ContractVerificationResult(
            contract_name=event_type,
            passed=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            execution_time_ms=(time.perf_counter() - start_time) * 1000,
        )


# ==============================================================================
# gRPC Contract Testing
# ==============================================================================

@dataclass
class GrpcMethodContract:
    """Contract for a gRPC method."""
    service_name: str
    method_name: str
    request_type: str
    response_type: str
    request_schema: Dict[str, Any]
    response_schema: Dict[str, Any]


class GrpcContractVerifier:
    """
    Verifies gRPC service contracts.

    Validates that services implement expected methods with correct schemas.
    """

    def __init__(self):
        self.methods: Dict[str, GrpcMethodContract] = {}

    def register_method(self, contract: GrpcMethodContract):
        """Register a gRPC method contract."""
        key = f"{contract.service_name}/{contract.method_name}"
        self.methods[key] = contract
        logger.info(f"Registered gRPC method: {key}")

    def verify_request(
        self,
        service_name: str,
        method_name: str,
        request: Dict[str, Any]
    ) -> ContractVerificationResult:
        """Verify gRPC request matches contract."""
        key = f"{service_name}/{method_name}"

        if key not in self.methods:
            return ContractVerificationResult(
                contract_name=key,
                passed=False,
                errors=[f"No contract for gRPC method: {key}"],
            )

        contract = self.methods[key]
        schema = contract.request_schema

        return self._validate_against_schema(key, request, schema)

    def verify_response(
        self,
        service_name: str,
        method_name: str,
        response: Dict[str, Any]
    ) -> ContractVerificationResult:
        """Verify gRPC response matches contract."""
        key = f"{service_name}/{method_name}"

        if key not in self.methods:
            return ContractVerificationResult(
                contract_name=key,
                passed=False,
                errors=[f"No contract for gRPC method: {key}"],
            )

        contract = self.methods[key]
        schema = contract.response_schema

        return self._validate_against_schema(key, response, schema)

    def _validate_against_schema(
        self,
        contract_name: str,
        data: Dict[str, Any],
        schema: Dict[str, Any]
    ) -> ContractVerificationResult:
        """Validate data against schema."""
        errors = []

        # Validate required fields
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")

        # Basic type validation
        properties = schema.get("properties", {})
        for field_name, field_schema in properties.items():
            if field_name in data:
                expected_type = field_schema.get("type")
                actual_value = data[field_name]

                type_valid = self._check_type(actual_value, expected_type)
                if not type_valid:
                    errors.append(
                        f"Field '{field_name}' type mismatch: expected {expected_type}, "
                        f"got {type(actual_value).__name__}"
                    )

        return ContractVerificationResult(
            contract_name=contract_name,
            passed=len(errors) == 0,
            errors=errors,
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        expected = type_map.get(expected_type)
        if expected is None:
            return True  # Unknown type, assume valid

        return isinstance(value, expected)


# ==============================================================================
# Pytest Fixtures
# ==============================================================================

@pytest.fixture(scope="session")
def contract_registry() -> ContractRegistry:
    """Provide contract registry."""
    return ContractRegistry()


@pytest.fixture
def pact_mock_server() -> Generator[PactMockServer, None, None]:
    """Provide Pact mock server for consumer tests."""
    server = PactMockServer(consumer="test_consumer", provider="test_provider")
    yield server


@pytest.fixture
def event_schema_validator() -> EventSchemaValidator:
    """Provide event schema validator."""
    validator = EventSchemaValidator()

    # Register common event schemas
    validator.register_schema("calculation.completed", {
        "type": "object",
        "required": ["calculation_id", "result", "timestamp"],
        "properties": {
            "calculation_id": {"type": "string"},
            "result": {"type": "number"},
            "timestamp": {"type": "string"},
            "metadata": {"type": "object"},
        },
    })

    validator.register_schema("calculation.failed", {
        "type": "object",
        "required": ["calculation_id", "error", "timestamp"],
        "properties": {
            "calculation_id": {"type": "string"},
            "error": {"type": "string"},
            "error_code": {"type": "string"},
            "timestamp": {"type": "string"},
        },
    })

    validator.register_schema("alarm.triggered", {
        "type": "object",
        "required": ["alarm_id", "severity", "message", "timestamp"],
        "properties": {
            "alarm_id": {"type": "string"},
            "severity": {"type": "string"},
            "message": {"type": "string"},
            "source": {"type": "string"},
            "timestamp": {"type": "string"},
            "acknowledged": {"type": "boolean"},
        },
    })

    return validator


@pytest.fixture
def grpc_contract_verifier() -> GrpcContractVerifier:
    """Provide gRPC contract verifier."""
    verifier = GrpcContractVerifier()

    # Register common gRPC method contracts
    verifier.register_method(GrpcMethodContract(
        service_name="CalculationService",
        method_name="Calculate",
        request_type="CalculationRequest",
        response_type="CalculationResponse",
        request_schema={
            "type": "object",
            "required": ["input_data", "calculation_type"],
            "properties": {
                "input_data": {"type": "object"},
                "calculation_type": {"type": "string"},
                "options": {"type": "object"},
            },
        },
        response_schema={
            "type": "object",
            "required": ["result", "provenance_hash"],
            "properties": {
                "result": {"type": "number"},
                "provenance_hash": {"type": "string"},
                "metadata": {"type": "object"},
            },
        },
    ))

    return verifier


@pytest.fixture
def pact_output_dir(tmp_path) -> Path:
    """Provide temporary directory for pact files."""
    pact_dir = tmp_path / "pacts"
    pact_dir.mkdir()
    return pact_dir


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "contract: mark test as a contract test",
    )
    config.addinivalue_line(
        "markers",
        "pact: mark test as a Pact consumer/provider test",
    )
    config.addinivalue_line(
        "markers",
        "event_contract: mark test as an event schema contract test",
    )
    config.addinivalue_line(
        "markers",
        "grpc_contract: mark test as a gRPC contract test",
    )


def pytest_collection_modifyitems(config, items):
    """Add contract marker to all tests in this directory."""
    for item in items:
        if "contract" in str(item.fspath):
            item.add_marker(pytest.mark.contract)
