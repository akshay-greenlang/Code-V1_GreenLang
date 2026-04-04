# GreenLang API Error Code Reference

## Overview

All GreenLang API errors follow a consistent structure.  Every exception in
the platform inherits from `GreenLangException` and carries rich context for
debugging, monitoring, and user feedback.

---

## Standard Error Response Format

All error responses use the following JSON envelope:

```json
{
  "error": {
    "code": "GL_AGENT_VALIDATION_ERROR",
    "message": "Missing required field: fuel_type",
    "details": {
      "agent_name": "FuelAgent",
      "invalid_fields": {
        "fuel_type": "required field missing"
      },
      "timestamp": "2026-04-04T10:30:00Z"
    }
  }
}
```

**Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `error.code` | string | Machine-readable error code (see tables below) |
| `error.message` | string | Human-readable error description |
| `error.details` | object | Context-specific details (varies by error type) |
| `error.details.agent_name` | string | Agent that raised the error (when applicable) |
| `error.details.timestamp` | string | ISO 8601 timestamp of when the error occurred |

---

## Error Code Prefix Convention

All error codes start with `GL_` followed by a domain prefix:

| Prefix | Domain | Module |
|--------|--------|--------|
| `GL_` | Base exception | `greenlang.exceptions.base` |
| `GL_AGENT_` | Agent execution | `greenlang.exceptions.agent` |
| `GL_WORKFLOW_` | Workflow orchestration | `greenlang.exceptions.workflow` |
| `GL_DATA_` | Data access and validation | `greenlang.exceptions.data` |
| `GL_CONNECTOR_` | External connectors | `greenlang.exceptions.connector` |
| `GL_SECURITY_` | Authentication and security | `greenlang.exceptions.security` |
| `GL_INTEGRATION_` | Service integrations | `greenlang.exceptions.integration` |
| `GL_INFRA_` | Infrastructure | `greenlang.exceptions.infrastructure` |
| `GL_COMPLIANCE_` | Regulatory compliance | `greenlang.exceptions.compliance` |
| `GL_CALC_` | Emission calculations | `greenlang.exceptions.calculation` |

---

## Agent Errors

Raised during agent execution, validation, or configuration.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_AGENT_VALIDATION_ERROR` | 400 | `ValidationError` | Input data fails validation. `details.invalid_fields` maps field names to reasons. | No |
| `GL_AGENT_EXECUTION_ERROR` | 500 | `ExecutionError` | Agent execution failed. `details.step` indicates the failing step; `details.cause` holds the original error. | No |
| `GL_AGENT_TIMEOUT_ERROR` | 504 | `TimeoutError` | Agent exceeded its execution time limit. `details.timeout_seconds` and `details.elapsed_seconds` are included. | Yes |
| `GL_AGENT_CONFIGURATION_ERROR` | 500 | `ConfigurationError` | Agent is misconfigured (e.g., missing API key, invalid config file). | No |

---

## Workflow Errors

Raised during DAG orchestration and workflow execution.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_WORKFLOW_D_A_G_ERROR` | 400 | `DAGError` | Workflow DAG is invalid (cycles, missing dependencies). `details.workflow_id` and `details.invalid_nodes` are included. | No |
| `GL_WORKFLOW_POLICY_VIOLATION` | 403 | `PolicyViolation` | Workflow violates a security, compliance, or resource policy. `details.policy_name` identifies the policy. | No |
| `GL_WORKFLOW_RESOURCE_ERROR` | 503 | `ResourceError` | Resource constraint exceeded (CPU, memory, disk). `details.resource_type`, `details.limit`, and `details.used` are provided. | Yes |
| `GL_WORKFLOW_ORCHESTRATION_ERROR` | 500 | `OrchestrationError` | Workflow orchestrator failed to coordinate agent execution. | No |

---

## Data Errors

Raised during data access, validation, and processing.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_DATA_INVALID_SCHEMA` | 400 | `InvalidSchema` | Data does not conform to expected schema. `details.expected_schema` and `details.schema_errors` are included. | No |
| `GL_DATA_MISSING_DATA` | 404 | `MissingData` | Required data fields or resources are not found. `details.data_type` and `details.missing_fields` are provided. | No |
| `GL_DATA_CORRUPTED_DATA` | 422 | `CorruptedData` | Data integrity check failed (checksum mismatch, malformed payload). `details.data_source` is included. | No |
| `GL_DATA_DATA_ACCESS_ERROR` | 502 | `DataAccessError` | Data source is unreachable or returned an error. `details.data_source` and `details.operation` are provided. | Yes |

---

## Connector Errors

Raised when external connector operations fail (ERP, grid APIs, satellite services).

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_CONNECTOR_CONNECTOR_CONFIG_ERROR` | 400 | `ConnectorConfigError` | Connector configuration is missing or invalid (e.g., missing API key). | No |
| `GL_CONNECTOR_CONNECTOR_AUTH_ERROR` | 401 | `ConnectorAuthError` | Connector authentication or authorization failed. `details.status_code` is included. | No |
| `GL_CONNECTOR_CONNECTOR_NETWORK_ERROR` | 502 | `ConnectorNetworkError` | Network failure (DNS, connection refused, TLS). `details.url` is included. | Yes |
| `GL_CONNECTOR_CONNECTOR_TIMEOUT_ERROR` | 504 | `ConnectorTimeoutError` | Request to external service timed out. `details.timeout_seconds` is provided. | Yes |
| `GL_CONNECTOR_CONNECTOR_RATE_LIMIT_ERROR` | 429 | `ConnectorRateLimitError` | External API rate limit exceeded. `details.retry_after_seconds` and `details.limit` are included. | Yes |
| `GL_CONNECTOR_CONNECTOR_NOT_FOUND_ERROR` | 404 | `ConnectorNotFoundError` | Requested resource not found in external service. | No |
| `GL_CONNECTOR_CONNECTOR_VALIDATION_ERROR` | 422 | `ConnectorValidationError` | Request or response data failed validation. `details.validation_errors` is a list of field-level errors. | No |
| `GL_CONNECTOR_CONNECTOR_SECURITY_ERROR` | 403 | `ConnectorSecurityError` | Egress blocked or security policy violated. `details.policy_violated` is included. | No |
| `GL_CONNECTOR_CONNECTOR_SERVER_ERROR` | 502 | `ConnectorServerError` | Upstream service returned a 5xx error. `details.status_code` is provided. | Yes |

---

## Security Errors

Raised for authentication, authorization, encryption, and security policy violations.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_SECURITY_AUTHENTICATION_ERROR` | 401 | `AuthenticationError` | Authentication failed (invalid credentials, expired token). `details.auth_method` indicates the method (`jwt`, `api_key`, `oauth2`, `mTLS`). | No |
| `GL_SECURITY_AUTHORIZATION_ERROR` | 403 | `AuthorizationError` | Insufficient permissions. `details.required_permission`, `details.user_role`, and `details.resource` are provided. | No |
| `GL_SECURITY_ENCRYPTION_ERROR` | 500 | `EncryptionError` | AES-256-GCM encryption or decryption failed. `details.operation` and `details.algorithm` are included. | No |
| `GL_SECURITY_SECRET_ACCESS_ERROR` | 503 | `SecretAccessError` | Secret retrieval from Vault or other backend failed. `details.secret_path` and `details.backend` are provided. | Yes |
| `GL_SECURITY_P_I_I_VIOLATION_ERROR` | 400 | `PIIViolationError` | PII detected in data that should be clean. `details.pii_types` lists the detected PII types; `details.field_name` identifies the field. | No |
| `GL_SECURITY_EGRESS_BLOCKED_ERROR` | 403 | `EgressBlockedError` | Outbound network request blocked by egress policy. `details.blocked_domain` and `details.policy_name` are provided. | No |
| `GL_SECURITY_CERTIFICATE_ERROR` | 502 | `CertificateError` | TLS certificate validation or rotation failed. `details.certificate_subject` and `details.expiry_date` are included. | Yes |

---

## Integration Errors

Raised when external service integrations fail (factor brokers, entity resolution, APIs).

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_INTEGRATION_EMISSION_FACTOR_ERROR` | 404 | `EmissionFactorError` | Emission factor lookup failed. `details.factor_source` (EPA, DEFRA, IPCC) and `details.query` are included. | No |
| `GL_INTEGRATION_ENTITY_RESOLUTION_ERROR` | 422 | `EntityResolutionError` | Cannot resolve entity against master data. `details.entity_type` and `details.entity_value` are provided. | No |
| `GL_INTEGRATION_EXTERNAL_SERVICE_ERROR` | 502 | `ExternalServiceError` | External service call failed. `details.service_name` and `details.operation` are included. | Yes |
| `GL_INTEGRATION_A_P_I_CLIENT_ERROR` | 400 | `APIClientError` | Outbound API call failed due to client-side issues. `details.endpoint`, `details.http_method`, and `details.status_code` are provided. | No |
| `GL_INTEGRATION_RATE_LIMIT_ERROR` | 429 | `RateLimitError` | Integrated service rejected request due to rate limiting. `details.service_name` and `details.retry_after_seconds` are included. | Yes |

---

## Infrastructure Errors

Raised for database, cache, storage, queue, and resilience pattern failures.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_INFRA_DATABASE_ERROR` | 503 | `DatabaseError` | Database connection, query, or transaction failed (PostgreSQL, TimescaleDB, pgvector). `details.database` and `details.operation` are included. | Yes |
| `GL_INFRA_CACHE_ERROR` | 503 | `CacheError` | Redis cache operation failed. `details.cache_key` and `details.operation` are provided. | Yes |
| `GL_INFRA_STORAGE_ERROR` | 503 | `StorageError` | S3/object storage operation failed. `details.bucket`, `details.object_key`, and `details.operation` are included. | Yes |
| `GL_INFRA_QUEUE_ERROR` | 503 | `QueueError` | Message queue operation failed. `details.queue_name` and `details.operation` are provided. | Yes |
| `GL_INFRA_CIRCUIT_BREAKER_OPEN_ERROR` | 503 | `CircuitBreakerOpenError` | Circuit breaker tripped due to consecutive failures. `details.circuit_name`, `details.failure_count`, and `details.reset_timeout_seconds` are provided. | Yes |
| `GL_INFRA_BULKHEAD_FULL_ERROR` | 503 | `BulkheadFullError` | Concurrency limit reached. `details.bulkhead_name` and `details.max_concurrent` are included. | Yes |
| `GL_INFRA_RETRY_EXHAUSTED_ERROR` | 503 | `RetryExhaustedError` | All retry attempts failed. `details.max_retries` and `details.last_error` are provided. | No |

---

## Compliance Errors

Raised when regulatory requirements are violated.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_COMPLIANCE_E_U_D_R_VIOLATION_ERROR` | 422 | `EUDRViolationError` | EUDR due diligence or traceability requirement not met. `details.commodity`, `details.requirement` (e.g., "Article 9(1)(d)"), and `details.risk_level` are provided. | No |
| `GL_COMPLIANCE_C_S_R_D_VIOLATION_ERROR` | 422 | `CSRDViolationError` | CSRD/ESRS disclosure requirement not met. `details.esrs_standard` (E1, S1, G1) and `details.disclosure_requirement` are included. | No |
| `GL_COMPLIANCE_C_B_A_M_VIOLATION_ERROR` | 422 | `CBAMViolationError` | CBAM reporting or embedded emissions requirement not met. `details.cn_code` and `details.reporting_period` are provided. | No |
| `GL_COMPLIANCE_REGULATORY_DEADLINE_ERROR` | 422 | `RegulatoryDeadlineError` | Regulatory reporting deadline missed or at risk. `details.regulation` and `details.deadline` (ISO 8601) are included. | No |
| `GL_COMPLIANCE_AUDIT_TRAIL_ERROR` | 500 | `AuditTrailError` | Audit trail integrity or completeness failure. `details.audit_event` and `details.record_id` are provided. | No |
| `GL_COMPLIANCE_PROVENANCE_ERROR` | 422 | `ProvenanceError` | SHA-256 provenance hash verification failed. `details.expected_hash`, `details.actual_hash`, and `details.data_source` are included. | No |

---

## Calculation Errors

Raised in the deterministic (zero-hallucination) calculation engine.

| Error Code | HTTP Status | Exception Class | Description | Retriable |
|------------|-------------|-----------------|-------------|-----------|
| `GL_CALC_EMISSION_CALCULATION_ERROR` | 422 | `EmissionCalculationError` | GHG emission calculation failed. `details.scope` (scope_1, scope_2, scope_3) and `details.calculation_step` are included. | No |
| `GL_CALC_UNIT_CONVERSION_ERROR` | 422 | `UnitConversionError` | Unit conversion failed (incompatible dimensions). `details.source_unit`, `details.target_unit`, and `details.value` are provided. | No |
| `GL_CALC_FACTOR_NOT_FOUND_ERROR` | 404 | `FactorNotFoundError` | Required emission factor not found. `details.factor_type` (gwp, emission_factor) and `details.lookup_key` are included. | No |
| `GL_CALC_METHODOLOGY_ERROR` | 422 | `MethodologyError` | Selected methodology cannot be applied. `details.methodology` (spend_based, activity_based) and `details.ghg_category` are provided. | No |
| `GL_CALC_BOUNDARY_ERROR` | 422 | `BoundaryError` | Organizational or operational boundary error. `details.boundary_type` (operational_control, equity_share) and `details.entity` are included. | No |

---

## Retriability

Each exception is classified as **retriable** or **non-retriable** by the
`is_retriable()` utility function in `greenlang.exceptions.utils`.

**Retriable exceptions** are transient failures where retrying may succeed:

- Timeouts, network errors, rate limits, server errors
- Database, cache, storage, queue failures
- Circuit breaker and bulkhead rejections
- Secret access failures (Vault may be temporarily down)
- Certificate errors (may be mid-rotation)

**Non-retriable exceptions** are deterministic failures:

- Validation, schema, policy violations
- Authentication and authorization failures
- Calculation errors, compliance violations
- Already-exhausted retries

### Best Practices for Error Handling

1. **Check `error.code`** to determine the error domain and type.
2. **Use `is_retriable()`** logic to decide whether to retry.
3. **Respect `Retry-After` headers** when receiving 429 responses.
4. **Implement exponential backoff** for retriable errors.
5. **Log `error.details`** for debugging -- it contains the full context including agent name, timestamps, and cause chains.

---

## Exception Utilities

### `format_exception_chain(exc)`

Formats the full exception chain (including `__cause__`) for logging:

```python
from greenlang.exceptions.utils import format_exception_chain

try:
    run_pipeline()
except GreenLangException as e:
    log.error(format_exception_chain(e))
```

### `is_retriable(exc)`

Determines whether an operation should be retried:

```python
from greenlang.exceptions.utils import is_retriable

try:
    result = call_external_service()
except Exception as e:
    if is_retriable(e):
        schedule_retry()
    else:
        raise
```

---

## Source Files

| File | Purpose |
|------|---------|
| `greenlang/utilities/exceptions/base.py` | `GreenLangException` base class with error code generation, JSON serialization |
| `greenlang/utilities/exceptions/agent.py` | `AgentException`, `ValidationError`, `ExecutionError`, `TimeoutError`, `ConfigurationError` |
| `greenlang/utilities/exceptions/workflow.py` | `WorkflowException`, `DAGError`, `PolicyViolation`, `ResourceError`, `OrchestrationError` |
| `greenlang/utilities/exceptions/data.py` | `DataException`, `InvalidSchema`, `MissingData`, `CorruptedData`, `DataAccessError` |
| `greenlang/utilities/exceptions/connector.py` | `ConnectorException` and 8 subclasses for external connector errors |
| `greenlang/utilities/exceptions/security.py` | `SecurityException` and 7 subclasses for auth, encryption, PII, egress, certificate errors |
| `greenlang/utilities/exceptions/integration.py` | `IntegrationException` and 5 subclasses for factor broker, entity resolution, API client errors |
| `greenlang/utilities/exceptions/infrastructure.py` | `InfrastructureException` and 7 subclasses for database, cache, storage, queue, resilience errors |
| `greenlang/utilities/exceptions/compliance.py` | `ComplianceException` and 6 subclasses for EUDR, CSRD, CBAM, audit trail, provenance errors |
| `greenlang/utilities/exceptions/calculation.py` | `CalculationException` and 5 subclasses for emission calculation, unit conversion, methodology errors |
| `greenlang/utilities/exceptions/utils.py` | `format_exception_chain()` and `is_retriable()` utilities |
| `greenlang/exceptions/__init__.py` | Backward-compatible re-export of all 60 exception classes |
