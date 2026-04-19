# GreenLang: Unify Error Handling Patterns PRD

## Objective

Migrate all scattered `class *Error(Exception)` and `class *Exception(Exception)` definitions across the `greenlang/` codebase to inherit from the centralized exception hierarchy rooted at `greenlang.utilities.exceptions.base.GreenLangException`.

## Current State

- **Centralized hierarchy** lives in `greenlang/utilities/exceptions/` with 8 modules:
  - `base.py` -- `GreenLangException(Exception)` (root)
  - `agent.py` -- `AgentException`, `ValidationError`, `ExecutionError`, `TimeoutError`, `ConfigurationError`
  - `workflow.py` -- `WorkflowException`, `DAGError`, `PolicyViolation`, `ResourceError`, `OrchestrationError`
  - `data.py` -- `DataException`, `InvalidSchema`, `MissingData`, `CorruptedData`, `DataAccessError`
  - `connector.py` -- `ConnectorException`, `ConnectorConfigError`, `ConnectorAuthError`, `ConnectorNetworkError`, `ConnectorTimeoutError`, `ConnectorRateLimitError`, `ConnectorNotFoundError`, `ConnectorValidationError`, `ConnectorSecurityError`, `ConnectorServerError`
  - `security.py` -- `SecurityException`, `AuthenticationError`, `AuthorizationError`, `EncryptionError`, `SecretAccessError`, `PIIViolationError`, `EgressBlockedError`, `CertificateError`
  - `utils.py` -- `format_exception_chain`, `is_retriable`
  - `__init__.py` -- Re-exports from all submodules
- **Shim layer** in `greenlang/exceptions/` re-exports from `greenlang/utilities/exceptions/` for backward compatibility
- **140 scattered** `class *(Exception):` definitions across 109 files in `greenlang/`
- Scattered exceptions do NOT participate in the centralized hierarchy (no `GreenLangException` base, no `error_code`, no `to_dict()`, no `timestamp`)

## Target State

- All domain exceptions inherit from `GreenLangException` (directly or via a domain branch)
- Zero `class *Error(Exception):` or `class *Exception(Exception):` for domain errors (only `GreenLangException(Exception)` at the root)
- All exceptions gain: `error_code`, `agent_name`, `context`, `timestamp`, `traceback_str`, `to_dict()`, `to_json()`
- Full backward compatibility: same class names, same constructor signatures, same import paths

## CRITICAL CONSTRAINTS

- **DO NOT MODIFY**: `2026_PRD_MVP/` directory
- **DO NOT MODIFY**: `cbam-pack-mvp/` directory
- **DO NOT MODIFY**: `applications/` directory
- **DO NOT MODIFY**: `packs/` directory
- **DO NOT MODIFY**: `greenlang/utilities/exceptions/` (these are the source-of-truth, read-only for this task)
- **BACKWARD COMPATIBILITY IS MANDATORY**: All existing class names, constructor parameters, and import paths must continue to work. Add parameters (with defaults), never remove them.

---

## Task 1: Verify Expanded Exception Hierarchy

Verify that all modules in the centralized exception hierarchy (`greenlang/utilities/exceptions/` and `greenlang/exceptions/`) import cleanly and all exception classes can be instantiated.

### Steps

1. Run `python -c "from greenlang.utilities.exceptions import *"` and confirm zero import errors.

2. Run `python -c "from greenlang.utilities.exceptions.base import GreenLangException; e = GreenLangException('test'); print(e.error_code, e.to_dict())"` and confirm it produces valid output.

3. Run `python -c "from greenlang.utilities.exceptions.connector import ConnectorException; e = ConnectorException('test', connector_name='test'); print(e.error_code)"` and confirm it produces `GL_CONNECTOR_CONNECTOR_EXCEPTION` or similar.

4. Run `python -c "from greenlang.utilities.exceptions.security import SecurityException; e = SecurityException('test'); print(e.error_code)"` and confirm it produces `GL_SECURITY_SECURITY_EXCEPTION` or similar.

5. Verify the shim layer works:
   ```
   python -c "import warnings; warnings.filterwarnings('ignore'); from greenlang.exceptions import GreenLangException, ValidationError; print('shim OK')"
   ```

6. Verify the `greenlang/exceptions/` shim directory has backward-compat files for the NEW modules too. If `greenlang/exceptions/connector.py` and `greenlang/exceptions/security.py` do not exist, create them as shims:
   ```python
   # greenlang/exceptions/connector.py
   """Backward compatibility shim. Use greenlang.utilities.exceptions.connector instead."""
   from greenlang.utilities.exceptions.connector import *
   ```
   ```python
   # greenlang/exceptions/security.py
   """Backward compatibility shim. Use greenlang.utilities.exceptions.security instead."""
   from greenlang.utilities.exceptions.security import *
   ```

7. Update `greenlang/exceptions/__init__.py` to also re-export from `connector` and `security` submodules (add them to the imports and `__all__` list). Keep the deprecation warning.

### Verification

- `python -c "from greenlang.exceptions.connector import ConnectorException; print('OK')"` succeeds
- `python -c "from greenlang.exceptions.security import SecurityException; print('OK')"` succeeds
- All 8 exception category branches instantiate with `.error_code`, `.to_dict()`, `.timestamp`

---

## Task 2: Migrate integration/connectors/errors.py

Make `ConnectorError` in `greenlang/integration/connectors/errors.py` inherit from `greenlang.utilities.exceptions.connector.ConnectorException` instead of plain `Exception`.

### File

`greenlang/integration/connectors/errors.py`

### Current State

- `ConnectorError(Exception)` -- base with `__init__(self, message, connector, status_code, request_id, url, context, original_error)`
- 11 subclasses: `ConnectorConfigError`, `ConnectorAuthError`, `ConnectorNetworkError`, `ConnectorTimeoutError`, `ConnectorRateLimit`, `ConnectorNotFound`, `ConnectorBadRequest`, `ConnectorServerError`, `ConnectorReplayRequired`, `ConnectorSnapshotNotFound`, `ConnectorSnapshotCorrupt`, `ConnectorSecurityError`, `ConnectorValidationError`
- Utility function: `classify_connector_error()`

### Migration Instructions

1. Change `ConnectorError` to inherit from `ConnectorException` (from `greenlang.utilities.exceptions.connector`):
   ```python
   from greenlang.utilities.exceptions.connector import ConnectorException as _BaseConnectorException

   class ConnectorError(_BaseConnectorException):
   ```

2. Update the `ConnectorError.__init__` to call `super().__init__()` with the right parameter mapping. The centralized `ConnectorException` expects `(message, connector_name, status_code, url, request_id, original_error, agent_name, context)`. Map the existing `connector` parameter to `connector_name`:
   ```python
   def __init__(self, message, connector, status_code=None, request_id=None, url=None, context=None, original_error=None):
       # Keep all existing attributes
       self.connector = connector  # backward compat alias
       super().__init__(
           message=message,
           connector_name=connector,
           status_code=status_code,
           url=url,
           request_id=request_id,
           original_error=original_error,
           context=context,
       )
   ```

3. All 11 subclasses that inherit from `ConnectorError` need NO changes (they already inherit from it). But verify each one still works by checking their `__init__` signatures pass through correctly.

4. The `ConnectorReplayRequired`, `ConnectorSnapshotNotFound`, and `ConnectorSnapshotCorrupt` classes are unique to this module (not in the centralized hierarchy). They should remain as-is, now inheriting through `ConnectorError -> ConnectorException -> GreenLangException`.

5. Similarly, `ConnectorBadRequest` is local. Keep it.

6. Keep the `classify_connector_error()` function unchanged.

7. Keep the `to_dict()` override on `ConnectorError` so the serialization format does not change.

### Verification

- `python -c "from greenlang.integration.connectors.errors import ConnectorError; e = ConnectorError('fail', 'test'); assert hasattr(e, 'error_code'); assert hasattr(e, 'timestamp'); print(e.to_dict()); print('OK')"`
- `python -c "from greenlang.integration.connectors.errors import ConnectorRateLimit; e = ConnectorRateLimit('rate', 'test', retry_after=60); assert isinstance(e, Exception); print('OK')"`
- All files that import from `greenlang.integration.connectors.errors` still import cleanly. Search with:
  ```
  grep -r "from greenlang.integration.connectors.errors import" greenlang/ --include="*.py"
  ```
  and verify each file imports without error.

---

## Task 3: Migrate intelligence/runtime/errors.py

Migrate the 5 GL-prefixed exception classes in the intelligence runtime to inherit from the centralized hierarchy.

### File

`greenlang/agents/intelligence/runtime/errors.py`

### Current State

5 classes, all inheriting from `Exception`:
- `GLValidationError(Exception)` -- JSON Schema validation failures
- `GLRuntimeError(Exception)` -- Runtime enforcement violations
- `GLSecurityError(Exception)` -- Security violations
- `GLDataError(Exception)` -- Data resolution failures
- `GLProvenanceError(Exception)` -- Provenance tracking failures

All share a pattern: `__init__(self, code, message, hint=None, path=None)` with `.code`, `.message`, `.hint`, `.path` attributes and a `to_dict()` method.

### Migration Instructions

1. Add imports at top:
   ```python
   from greenlang.utilities.exceptions.agent import ValidationError as _BaseValidationError
   from greenlang.utilities.exceptions.agent import ExecutionError as _BaseExecutionError
   from greenlang.utilities.exceptions.data import DataException as _BaseDataException
   from greenlang.utilities.exceptions.security import SecurityException as _BaseSecurityException
   from greenlang.utilities.exceptions.base import GreenLangException as _BaseGreenLangException
   ```

2. Map each class to the appropriate parent:
   - `GLValidationError` -> inherit from `_BaseValidationError` (agent validation)
   - `GLRuntimeError` -> inherit from `_BaseExecutionError` (agent execution)
   - `GLSecurityError` -> inherit from `_BaseSecurityException` (security)
   - `GLDataError` -> inherit from `_BaseDataException` (data)
   - `GLProvenanceError` -> inherit from `_BaseGreenLangException` (general, provenance is cross-cutting)

3. For each class, update `__init__` to call both the centralized parent AND preserve existing attributes:
   ```python
   class GLValidationError(_BaseValidationError):
       ARGS_SCHEMA = "ARGS_SCHEMA"
       RESULT_SCHEMA = "RESULT_SCHEMA"
       UNIT_UNKNOWN = "UNIT_UNKNOWN"

       def __init__(self, code, message, hint=None, path=None):
           self.code = code
           self.hint = hint or self._default_hint(code)
           self.path = path
           super().__init__(
               message=f"[{code}] {message}",
               context={"code": code, "hint": self.hint, "path": path},
           )
           # Override message to preserve original (super sets it to the full string)
           self.message = message
   ```

4. Keep all class-level constants (error code strings like `ARGS_SCHEMA`).

5. Keep the `_default_hint()` static methods.

6. Keep the `to_dict()` methods, but have them merge with the parent's `to_dict()`:
   ```python
   def to_dict(self):
       base = super().to_dict()
       base.update({
           "error_type": "GLValidationError",
           "code": self.code,
           "hint": self.hint,
           "path": self.path,
       })
       return base
   ```

7. Keep the `serialize_error()` function unchanged.

### Verification

- `python -c "from greenlang.agents.intelligence.runtime.errors import GLValidationError; e = GLValidationError('ARGS_SCHEMA', 'bad args'); assert hasattr(e, 'error_code'); assert hasattr(e, 'code'); assert e.code == 'ARGS_SCHEMA'; assert hasattr(e, 'hint'); print(e.to_dict()); print('OK')"`
- Repeat for `GLRuntimeError`, `GLSecurityError`, `GLDataError`, `GLProvenanceError`.
- Verify all 14 files that import from this module still work:
  ```
  grep -r "from greenlang.agents.intelligence.runtime.errors import\|from greenlang.agents.intelligence.runtime import errors" greenlang/ --include="*.py"
  ```
  Run `python -c "import <module>"` for each importing file.

---

## Task 4: Migrate Infrastructure Exception Classes

Find and migrate all `class *Error(Exception):` and `class *Exception(Exception):` in `greenlang/infrastructure/`.

### Scope

31 exception classes across 23 files. Key areas:

**RBAC Service** (10 classes across 3 files):
- `greenlang/infrastructure/rbac_service/role_service.py`: `RoleNotFoundError`, `SystemRoleProtectionError`, `RoleHierarchyCycleError`, `RoleHierarchyDepthError`, `DuplicateRoleError`
- `greenlang/infrastructure/rbac_service/permission_service.py`: `PermissionNotFoundError`, `DuplicatePermissionError`
- `greenlang/infrastructure/rbac_service/assignment_service.py`: `AssignmentNotFoundError`, `DuplicateAssignmentError`, `AssignmentAlreadyRevokedError`

**Secrets Service** (3 classes across 2 files):
- `greenlang/infrastructure/secrets_service/tenant_context.py`: `TenantAccessDeniedError`, `InvalidSecretPathError`
- `greenlang/infrastructure/secrets_service/secrets_service.py`: `SecretsServiceError`

**Encryption Service** (1 class):
- `greenlang/infrastructure/encryption_service/__init__.py`: `EncryptionError`

**PII Service** (3 classes across 3 files):
- `greenlang/infrastructure/pii_service/secure_vault.py`: `VaultError`
- `greenlang/infrastructure/pii_service/remediation/engine.py`: `RemediationError`
- `greenlang/infrastructure/pii_service/allowlist/manager.py`: (1 class)

**Security Scanning** (1 class):
- `greenlang/infrastructure/security_scanning/scanners/base.py`: `ScannerError`

**Agent Factory** (7 classes across 6 files):
- `greenlang/infrastructure/agent_factory/config/store.py`: `ConfigVersionConflictError`
- `greenlang/infrastructure/agent_factory/resilience/retry.py`: `RetryExhaustedError`
- `greenlang/infrastructure/agent_factory/resilience/circuit_breaker.py`: `CircuitOpenError`
- `greenlang/infrastructure/agent_factory/resilience/bulkhead.py`: `BulkheadFullError`
- `greenlang/infrastructure/agent_factory/metering/resource_quotas.py`: `QuotaExceededError`
- `greenlang/infrastructure/agent_factory/metering/budget.py`: `BudgetExceededError`
- `greenlang/infrastructure/agent_factory/lifecycle/states.py`: `InvalidTransitionError`

**Vulnerability Disclosure** (5 classes across 5 files):
- `triage_workflow.py`: `TriageError`
- `submission_handler.py`: `SubmissionError`
- `researcher_manager.py`: `ResearcherError`
- `disclosure_tracker.py`: `DisclosureError`
- `bounty_processor.py`: `BountyError`

**Feature Flags** (1 class):
- `greenlang/infrastructure/feature_flags/lifecycle/manager.py`: `InvalidTransitionError`

### Migration Instructions

1. For each file, add the appropriate import and change the base class:

   **Security-related** (RBAC, secrets, encryption, PII, security scanning) -> inherit from `SecurityException`:
   ```python
   from greenlang.utilities.exceptions.security import SecurityException

   class RoleNotFoundError(SecurityException):
       """..."""
       def __init__(self, message, *args, **kwargs):
           super().__init__(message, *args, **kwargs)
   ```

   **Agent Factory resilience** (retry, circuit breaker, bulkhead) -> inherit from `GreenLangException`:
   ```python
   from greenlang.utilities.exceptions.base import GreenLangException

   class RetryExhaustedError(GreenLangException):
   ```

   **Agent Factory metering** (quota, budget) -> inherit from `GreenLangException`:
   ```python
   from greenlang.utilities.exceptions.base import GreenLangException

   class QuotaExceededError(GreenLangException):
   ```

   **Vulnerability Disclosure** -> inherit from `SecurityException`:
   ```python
   from greenlang.utilities.exceptions.security import SecurityException

   class TriageError(SecurityException):
   ```

2. For each class: preserve ALL existing `__init__` parameters and attributes. Add `**kwargs` passthrough if needed to accept the `GreenLangException` params (`error_code`, `agent_name`, `context`).

3. Pattern for classes with no custom `__init__`:
   ```python
   class SomeError(SecurityException):
       """Docstring preserved."""
       pass  # inherits __init__ from SecurityException -> GreenLangException
   ```

4. Pattern for classes with custom `__init__` that takes `message` only:
   ```python
   class SomeError(SecurityException):
       """Docstring preserved."""
       def __init__(self, message: str, **kwargs):
           super().__init__(message, **kwargs)
   ```

5. Pattern for classes with custom `__init__` with extra params:
   ```python
   class RoleNotFoundError(SecurityException):
       """Docstring preserved."""
       def __init__(self, role_name: str, message: str = None, **kwargs):
           self.role_name = role_name
           message = message or f"Role not found: {role_name}"
           context = kwargs.pop("context", {})
           context["role_name"] = role_name
           super().__init__(message, context=context, **kwargs)
   ```

### Verification

For each migrated file:
```
python -c "from greenlang.infrastructure.rbac_service.role_service import RoleNotFoundError; e = RoleNotFoundError('test'); assert hasattr(e, 'error_code'); print('OK')"
```

Run this for all 23 files. Ensure zero import errors across the codebase:
```
grep -r "from greenlang.infrastructure" greenlang/ --include="*.py" -l | head -50
```
Spot-check 10 importing files with `python -c "import <module>"`.

---

## Task 5: Migrate EUDR Agent Exception Classes

Migrate all 13 custom exceptions in `greenlang/agents/eudr/` to inherit from the centralized hierarchy.

### Scope

13 exception classes across 13 files in two main agent directories:

**QR Code Generator** (8 classes):
- `qr_encoder.py`: `QREncoderError`
- `payload_composer.py`: `PayloadComposerError`
- `label_template_engine.py`: `LabelEngineError`
- `batch_code_generator.py`: `BatchCodeError`
- `bulk_generation_pipeline.py`: `BulkGenerationError`
- `code_lifecycle_manager.py`: `LifecycleError`
- `verification_url_builder.py`: `VerificationURLError`
- `anti_counterfeit_engine.py`: `AntiCounterfeitError`

**Mobile Data Collector** (4 classes):
- `sync_engine.py`: `SyncEngineError`
- `offline_form_engine.py`: `OfflineFormEngineError`
- `photo_evidence_collector.py`: `PhotoEvidenceError`
- `gps_capture_engine.py`: `GPSCaptureEngineError`

**Supply Chain Mapper** (1 class):
- `graph_engine.py`: `GraphEngineError`

### Migration Instructions

1. All EUDR exceptions should inherit from `GreenLangException` (they are agent-domain exceptions but not compliance-framework specific):
   ```python
   from greenlang.utilities.exceptions.base import GreenLangException

   class QREncoderError(GreenLangException):
   ```

2. For each class, preserve the existing `__init__` signature. Add passthrough to `super().__init__()`:
   ```python
   class QREncoderError(GreenLangException):
       """QR encoding error."""
       def __init__(self, message: str, **kwargs):
           super().__init__(message, **kwargs)
   ```

3. If any class has custom attributes in `__init__`, preserve them and pass relevant info into `context`:
   ```python
   class SyncEngineError(GreenLangException):
       def __init__(self, message: str, sync_id: str = None, **kwargs):
           self.sync_id = sync_id
           context = kwargs.pop("context", {})
           if sync_id:
               context["sync_id"] = sync_id
           super().__init__(message, context=context, **kwargs)
   ```

4. For each file, check the class `__init__` signature before migrating. Open the file, find the class definition, and read 20 lines to understand the constructor.

### Verification

For each of the 13 files:
```
python -c "from greenlang.agents.eudr.qr_code_generator.qr_encoder import QREncoderError; e = QREncoderError('test'); assert hasattr(e, 'error_code'); print('OK')"
```

Search for all imports of these classes and verify they still work:
```
grep -r "QREncoderError\|PayloadComposerError\|LabelEngineError\|BatchCodeError\|BulkGenerationError\|LifecycleError\|VerificationURLError\|AntiCounterfeitError\|SyncEngineError\|OfflineFormEngineError\|PhotoEvidenceError\|GPSCaptureEngineError\|GraphEngineError" greenlang/ --include="*.py" -l
```

---

## Task 6: Migrate Remaining Scattered Exceptions

Migrate all remaining `class *Error(Exception):` and `class *Exception(Exception):` across `greenlang/` that were not covered in Tasks 2-5.

### Discovery

Run a search for all remaining direct-Exception subclasses:
```
grep -rn "class \w\+Error(Exception):\|class \w\+Exception(Exception):" greenlang/ --include="*.py"
```

Exclude files already handled:
- `greenlang/utilities/exceptions/base.py` (the root, do not touch)
- `greenlang/integration/connectors/errors.py` (Task 2)
- `greenlang/agents/intelligence/runtime/errors.py` (Task 3)
- `greenlang/infrastructure/` (Task 4)
- `greenlang/agents/eudr/` (Task 5)

### Known Remaining Files (by area)

**Auth** (6 classes in 6 files):
- `greenlang/auth/jwt_handler.py`: `JWTError` -> `AuthenticationError`
- `greenlang/auth/api_key_manager.py`: `APIKeyError` -> `AuthenticationError`
- `greenlang/auth/mfa.py`: `MFAError` -> `AuthenticationError`
- `greenlang/auth/oauth_provider.py`: `OAuthError` -> `AuthenticationError`
- `greenlang/auth/saml_provider.py`: `SAMLError` -> `AuthenticationError`
- `greenlang/auth/ldap_provider.py`: `LDAPError` -> `AuthenticationError`
- `greenlang/auth/scim_provider.py`: `SCIMError` -> `SecurityException`

**Governance** (4 classes in 4 files):
- `greenlang/governance/validation/schema.py`: `SchemaValidationError` -> `DataException`
- `greenlang/governance/validation/hooks.py`: `ValidationError` -> `AgentException` (via `ValidationError`)
- `greenlang/governance/validation/decorators.py`: `ValidationException` -> `AgentException`
- `greenlang/governance/security/validators.py`: `ValidationError` -> `AgentException`
- `greenlang/governance/security/signatures.py`: `SignatureVerificationError` -> `SecurityException`
- `greenlang/governance/security/kms/base_kms.py`: `KMSProviderError` -> `SecurityException`

**Execution** (6+ classes across files):
- `greenlang/execution/resilience/timeout.py`: `TimeoutError` -> `GreenLangException`
- `greenlang/execution/resilience/retry.py`: `RetryableError` -> `GreenLangException`
- `greenlang/execution/resilience/circuit_breaker.py`: `CircuitBreakerError` -> `GreenLangException`
- `greenlang/execution/infrastructure/secrets/vault_client.py`: `VaultError` -> `SecurityException`
- `greenlang/execution/infrastructure/resilience/retry_policy.py`: `RetryExhaustedError` -> `GreenLangException`
- `greenlang/execution/infrastructure/resilience/circuit_breaker.py`: `CircuitBreakerOpenError` -> `GreenLangException`
- `greenlang/execution/infrastructure/resilience/bulkhead.py`: (if exists) -> `GreenLangException`
- `greenlang/execution/infrastructure/auth/rbac_manager.py`: (auth error) -> `SecurityException`
- `greenlang/execution/infrastructure/auth/oauth2_provider.py`: (auth error) -> `SecurityException`
- `greenlang/execution/infrastructure/api/services/*.py`: service errors -> `GreenLangException`
- `greenlang/execution/infrastructure/api/graphql_integration.py`: -> `GreenLangException`

**Extensions** (6+ classes):
- `greenlang/extensions/satellite/pipeline/analysis_pipeline.py`: `PipelineError` -> `GreenLangException`
- `greenlang/extensions/satellite/models/forest_classifier.py`: `ForestClassifierError` -> `GreenLangException`
- `greenlang/extensions/satellite/clients/sentinel2_client.py`: `Sentinel2ClientError` -> `ConnectorException`
- `greenlang/extensions/satellite/clients/landsat_client.py`: `LandsatClientError` -> `ConnectorException`
- `greenlang/extensions/satellite/analysis/vegetation_indices.py`: `VegetationIndicesError` -> `GreenLangException`
- `greenlang/extensions/satellite/analysis/change_detection.py`: `ChangeDetectionError` -> `GreenLangException`
- `greenlang/extensions/satellite/alerts/deforestation_alert.py`: `AlertSystemError` -> `GreenLangException`

**Monitoring** (3+ classes):
- `greenlang/monitoring/pushgateway.py`: `PushGatewayError` -> `GreenLangException`
- `greenlang/monitoring/grafana/client.py`: `GrafanaError` -> `GreenLangException`
- `greenlang/monitoring/sandbox/__init__.py`: `SandboxViolationError` -> `SecurityException`
- `greenlang/monitoring/sandbox/os_sandbox.py`: `OSSandboxError` -> `SecurityException`

**Agents (non-EUDR)** (misc):
- `greenlang/agents/calculation/emissions/validator.py`: `ValidationError` -> `AgentException`
- `greenlang/agents/process_heat/shared/base_agent.py`: `ProcessingError`, `SafetyError`, `ValidationError` -> `AgentException`
- `greenlang/agents/foundation/schema/suggestions/patches.py`: `PatchApplicationError` -> `GreenLangException`
- `greenlang/agents/foundation/schema/registry/git_backend.py`: 5 errors -> `DataException` or `GreenLangException`
- `greenlang/agents/foundation/schema/compiler/resolver.py`: 3 errors -> `DataException`
- `greenlang/agents/foundation/schema/compiler/parser.py`: `ParseError` -> `DataException`
- `greenlang/agents/foundation/schema/compiler/ir.py`: `CompilationError` -> `GreenLangException`
- `greenlang/agents/data/schema_migration/migration_executor.py`: `_TimeoutError` -> leave as internal (underscore prefix)
- `greenlang/agents/foundation/orchestrator/steps/validate_step.py`: `ValidationFailedError` -> `AgentException`
- `greenlang/agents/foundation/orchestrator/hooks/validation_hook.py`: `ValidationHookError` -> `AgentException`
- `greenlang/agents/foundation/orchestrator/audit/signing.py`: `SigningError` -> `SecurityException`
- `greenlang/agents/foundation/orchestrator/audit/event_store.py`: `EventStoreError` -> `DataException`
- `greenlang/agents/foundation/orchestrator/governance/policy_engine.py`: -> `WorkflowException`
- `greenlang/agents/foundation/orchestrator/governance/approvals.py`: -> `WorkflowException`
- `greenlang/agents/intelligence/runtime/json_validator.py`: 2 errors -> `DataException`
- `greenlang/agents/intelligence/budget.py`: -> `GreenLangException`
- `greenlang/agents/intelligence/runtime/jsonio.py`: -> `DataException`
- `greenlang/agents/intelligence/runtime/agent_tools.py`: 3 errors -> `AgentException`
- `greenlang/agents/intelligence/providers/resilience.py`: -> `GreenLangException`
- `greenlang/agents/intelligence/providers/errors.py`: -> `GreenLangException`
- `greenlang/agents/mrv/dual_reporting_reconciliation/dual_reporting_pipeline.py`: 2 errors -> `GreenLangException`

**Utilities** (3 classes):
- `greenlang/utilities/utils/unit_converter.py`: `UnitConversionError` -> `GreenLangException`
- `greenlang/utilities/serialization/canonical.py`: `CanonicalSerializationError` -> `GreenLangException`
- `greenlang/utilities/generator/spec_parser.py`: `ValidationError` -> `GreenLangException`
- `greenlang/utilities/factory/sdk/python/agent_factory.py`: `AgentFactoryError` -> `GreenLangException`

**Integration** (misc):
- `greenlang/integration/services/factor_broker/exceptions.py`: `FactorBrokerError` -> `GreenLangException`
- `greenlang/integration/sdk/emission_factor_client.py`: 3 errors -> `DataException`
- `greenlang/integration/integrations/base_connector.py`: `CircuitBreakerError` -> `GreenLangException`
- `greenlang/integration/services/entity_mdm/ml/exceptions.py`: `EntityResolutionMLException` -> `GreenLangException`

**Data** (1 class):
- `greenlang/data/validation.py`: `ValidationError` -> `DataException`

**Ecosystem** (3 classes):
- `greenlang/ecosystem/partners/webhook_security.py`: 3 errors -> `SecurityException`

**Other**:
- `greenlang/extensions/business/licensing.py`: 4 errors -> `GreenLangException`
- `greenlang/extensions/ml/explainability/api.py`: `HTTPException` -> leave as-is (internal to FastAPI shim)

### Migration Instructions

1. For each file, follow the same pattern as Tasks 2-5:
   - Add the appropriate import from `greenlang.utilities.exceptions.*`
   - Change base class from `Exception` to the mapped centralized exception
   - Preserve ALL existing constructor parameters and attributes
   - Add `**kwargs` passthrough for new GreenLangException params

2. **SKIP** these (leave as `Exception`):
   - `greenlang/agents/data/schema_migration/migration_executor.py`: `_TimeoutError` (internal, underscore-prefixed)
   - `greenlang/extensions/ml/explainability/api.py`: `HTTPException` (FastAPI/HTTP concept, not domain error)

3. Work through the files in alphabetical order by directory to ensure nothing is missed.

### Verification

After all migrations, run:
```
grep -rn "class \w\+Error(Exception):\|class \w\+Exception(Exception):" greenlang/ --include="*.py" | grep -v "utilities/exceptions/base.py" | grep -v "_TimeoutError" | grep -v "HTTPException"
```

The output should be EMPTY (zero remaining direct Exception subclasses for domain errors).

---

## Task 7: Final Verification

Run comprehensive verification to confirm the entire migration is complete and backward-compatible.

### Step 1: Hierarchy Import Test

```bash
python -c "from greenlang.utilities.exceptions import *; print('utilities OK')"
python -c "import warnings; warnings.filterwarnings('ignore'); from greenlang.exceptions import *; print('shim OK')"
python -c "from greenlang.utilities.exceptions.connector import *; print('connector OK')"
python -c "from greenlang.utilities.exceptions.security import *; print('security OK')"
```

All four must print OK.

### Step 2: Count Remaining Direct Exception Subclasses

```bash
grep -rn "class \w\+Error(Exception):\|class \w\+Exception(Exception):" greenlang/ --include="*.py"
```

**Goal**: Only these should remain:
- `greenlang/utilities/exceptions/base.py`: `GreenLangException(Exception)` (the root)
- `greenlang/agents/data/schema_migration/migration_executor.py`: `_TimeoutError(Exception)` (internal)
- `greenlang/extensions/ml/explainability/api.py`: `HTTPException(Exception)` (FastAPI shim)

Everything else should be ZERO.

### Step 3: Mass Import Test

For every file that was modified in Tasks 2-6, run an import test:

```python
import importlib
import sys

modules_to_test = [
    "greenlang.integration.connectors.errors",
    "greenlang.agents.intelligence.runtime.errors",
    "greenlang.infrastructure.rbac_service.role_service",
    "greenlang.infrastructure.rbac_service.permission_service",
    "greenlang.infrastructure.rbac_service.assignment_service",
    "greenlang.infrastructure.encryption_service",
    "greenlang.infrastructure.secrets_service.secrets_service",
    "greenlang.infrastructure.secrets_service.tenant_context",
    "greenlang.infrastructure.pii_service.secure_vault",
    "greenlang.infrastructure.security_scanning.scanners.base",
    "greenlang.infrastructure.agent_factory.resilience.retry",
    "greenlang.infrastructure.agent_factory.resilience.circuit_breaker",
    "greenlang.infrastructure.agent_factory.resilience.bulkhead",
    "greenlang.infrastructure.agent_factory.metering.resource_quotas",
    "greenlang.infrastructure.agent_factory.metering.budget",
    "greenlang.infrastructure.agent_factory.lifecycle.states",
    "greenlang.agents.eudr.qr_code_generator.qr_encoder",
    "greenlang.agents.eudr.qr_code_generator.payload_composer",
    "greenlang.agents.eudr.supply_chain_mapper.graph_engine",
    "greenlang.auth.jwt_handler",
    "greenlang.auth.api_key_manager",
    "greenlang.governance.validation.schema",
    "greenlang.execution.resilience.timeout",
    "greenlang.monitoring.pushgateway",
]

failed = []
for mod in modules_to_test:
    try:
        importlib.import_module(mod)
        print(f"  OK: {mod}")
    except Exception as e:
        print(f"  FAIL: {mod} -> {e}")
        failed.append(mod)

print(f"\n{'ALL PASSED' if not failed else f'{len(failed)} FAILED'}")
```

### Step 4: isinstance Check

Verify all migrated exceptions are now part of the GreenLangException hierarchy:

```python
from greenlang.utilities.exceptions.base import GreenLangException
from greenlang.integration.connectors.errors import ConnectorError
from greenlang.agents.intelligence.runtime.errors import GLValidationError

assert issubclass(ConnectorError, GreenLangException), "ConnectorError not in hierarchy"
assert issubclass(GLValidationError, GreenLangException), "GLValidationError not in hierarchy"
print("Hierarchy check PASSED")
```

### Step 5: Report

Print a summary:
- Total exception classes in `greenlang/`
- Total inheriting from `GreenLangException`
- Total still inheriting from `Exception` directly (should be 2-3 max)
- List any non-unified exceptions with file paths and justification for exclusion

### Success Criteria

- Zero import errors across all modified files
- Zero direct `Exception` subclasses for domain errors (excluding the 2-3 approved exclusions)
- All exceptions have `.error_code`, `.to_dict()`, `.timestamp` attributes
- All existing constructor signatures preserved (backward compatible)
- All existing import paths continue to work
