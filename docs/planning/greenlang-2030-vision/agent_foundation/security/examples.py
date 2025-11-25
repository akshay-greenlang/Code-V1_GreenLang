# -*- coding: utf-8 -*-
"""
Input Validation Examples - Common Usage Patterns.

This module demonstrates secure coding patterns using the input validation framework.
Each example shows both the INSECURE way (what NOT to do) and the SECURE way.

Run with: python -m security.examples
"""

import asyncio
import logging
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime

from greenlang.determinism import DeterministicClock
from security.input_validation import (
    InputValidator,
    TenantIdModel,
    UserIdModel,
    EmailModel,
    SafeQueryInput,
    SafePathInput,
    SafeUrlInput,
    SafeCommandInput,
    PaginationInput,
    FilterInput,
)

logger = logging.getLogger(__name__)


# ============================================================================
# EXAMPLE 1: Database Query
# ============================================================================

def example_database_query_insecure(user_input: str) -> str:
    """
    INSECURE: String concatenation in SQL queries.

    This is vulnerable to SQL injection.
    """
    # DANGER: Never do this!
    query = f"SELECT * FROM users WHERE tenant_id = '{user_input}'"

    # If user_input = "'; DROP TABLE users--"
    # Query becomes: SELECT * FROM users WHERE tenant_id = ''; DROP TABLE users--'
    # Result: All users table data deleted!

    return query


def example_database_query_secure(user_input: str) -> tuple:
    """
    SECURE: Parameterized queries with validation.

    This prevents SQL injection.
    """
    from database.postgres_manager_secure import SecureQueryBuilder

    # Step 1: Validate input
    validated_tenant = InputValidator.validate_alphanumeric(
        user_input, "tenant_id", min_length=3, max_length=255
    )

    # Step 2: Build safe query with parameterized values
    builder = SecureQueryBuilder("users")

    filters = [
        SafeQueryInput(
            field="tenant_id",  # Validated against whitelist
            value=validated_tenant,  # Validated for SQL injection
            operator="="  # Validated against operator whitelist
        )
    ]

    query, params = builder.build_select(filters=filters, limit=100, offset=0)

    # Result: query = "SELECT * FROM users WHERE tenant_id = $1 LIMIT 100 OFFSET 0"
    #         params = ["tenant-123"]
    # Safe: Values are bound as parameters, not concatenated

    logger.info(f"Safe query: {query}, params: {params}")
    return query, params


# ============================================================================
# EXAMPLE 2: Command Execution
# ============================================================================

def example_command_execution_insecure(user_input: str):
    """
    INSECURE: Shell command with user input.

    This is vulnerable to command injection.

    WARNING: This is an EXAMPLE of INSECURE code for educational purposes ONLY.
    DO NOT use this pattern in production code!
    """
    import subprocess
    import shlex

    # DANGER: Never do this!
    # This is commented out to prevent accidental execution
    # command = f"kubectl get pods {user_input}"

    # If user_input = "; rm -rf /"
    # Command becomes: kubectl get pods ; rm -rf /
    # Result: File system destroyed!

    # SECURITY FIX APPLIED: Use shell=False with shlex.split() instead
    # Original vulnerable code is commented out for safety
    logger.warning(
        "example_command_execution_insecure() called - "
        "this function demonstrates INSECURE patterns and should not be used!"
    )

    # SECURE ALTERNATIVE (replacing the vulnerable code):
    # Parse command safely without shell=True
    try:
        # Use shlex.split to properly parse arguments
        cmd_parts = shlex.split(f"kubectl get pods {user_input}")

        # Execute with shell=False (secure)
        subprocess.run(cmd_parts, shell=False, capture_output=True, timeout=30)

        logger.info("Command executed with shell=False (secure mode)")
    except (subprocess.TimeoutExpired, ValueError) as e:
        logger.error(f"Command execution failed: {e}")


def example_command_execution_secure(namespace: str, pod_name: str):
    """
    SECURE: Command execution with validation.

    This prevents command injection.
    """
    from factory.deployment_secure import SecureCommandExecutor

    executor = SecureCommandExecutor()

    # Execute with validation and shell=False
    result = executor.execute_kubectl(
        command="get",  # Validated against whitelist
        resource_type="pods",
        resource_name=pod_name,  # Validated for injection
        namespace=namespace,  # Validated for injection
        timeout=30
    )

    # Safe: Arguments passed as list, shell=False
    logger.info(f"Command result: {result}")


# ============================================================================
# EXAMPLE 3: File Path Handling
# ============================================================================

def example_file_path_insecure(user_input: str):
    """
    INSECURE: File access with user-controlled path.

    This is vulnerable to path traversal.
    """
    # DANGER: Never do this!
    file_path = f"/app/data/{user_input}"

    # If user_input = "../../etc/passwd"
    # Path becomes: /app/data/../../etc/passwd -> /etc/passwd
    # Result: Password file exposed!

    with open(file_path, 'r') as f:
        return f.read()


def example_file_path_secure(user_filename: str) -> Path:
    """
    SECURE: File path with validation.

    This prevents path traversal.
    """
    # Step 1: Validate filename (no path components)
    validated_filename = InputValidator.validate_alphanumeric(
        user_filename, "filename", max_length=255
    )

    # Step 2: Construct safe path
    base_dir = Path("/app/data")
    full_path = base_dir / f"{validated_filename}.json"

    # Step 3: Validate full path
    validated_path = InputValidator.validate_path(
        str(full_path),
        must_exist=False,
        allow_relative=False,
        allowed_extensions=['.json']
    )

    # Safe: Path cannot traverse outside base_dir
    logger.info(f"Safe path: {validated_path}")
    return validated_path


# ============================================================================
# EXAMPLE 4: URL Validation
# ============================================================================

def example_url_validation_insecure(user_url: str):
    """
    INSECURE: Fetching URL without validation.

    This is vulnerable to SSRF (Server-Side Request Forgery).
    """
    import requests

    # DANGER: Never do this!
    # If user_url = "http://169.254.169.254/latest/meta-data/"
    # Result: AWS metadata endpoint accessed, credentials leaked!

    response = requests.get(user_url)
    return response.text


def example_url_validation_secure(user_url: str) -> str:
    """
    SECURE: URL validation before fetching.

    This prevents SSRF attacks.
    """
    # Validate URL
    validated_url = InputValidator.validate_url(
        user_url,
        allowed_schemes=['https'],  # Only HTTPS
        allow_private_ips=False  # Block private IPs
    )

    # Safe to fetch
    import requests
    response = requests.get(validated_url, timeout=10)

    logger.info(f"Fetched URL: {validated_url}")
    return response.text


# ============================================================================
# EXAMPLE 5: API Request Validation
# ============================================================================

class CreateAgentRequestInsecure:
    """
    INSECURE: No input validation.
    """

    def __init__(self, data: dict):
        # DANGER: Accepting data without validation
        self.tenant_id = data.get("tenant_id")  # Could be SQL injection
        self.name = data.get("name")  # Could be XSS
        self.config = data.get("config")  # Could be malicious JSON


class CreateAgentRequestSecure:
    """
    SECURE: Comprehensive input validation.
    """

    def __init__(self, data: dict):
        # Validate tenant ID
        self.tenant_id = InputValidator.validate_alphanumeric(
            data.get("tenant_id", ""),
            "tenant_id",
            min_length=3,
            max_length=255
        )

        # Validate name
        self.name = InputValidator.validate_alphanumeric(
            data.get("name", ""),
            "name",
            min_length=1,
            max_length=255
        )

        # Validate config is valid JSON
        self.config = InputValidator.validate_json(data.get("config", {}))

        # Validate no XSS in description
        if "description" in data:
            self.description = InputValidator.validate_no_xss(
                data["description"],
                "description"
            )

        logger.info(f"Validated request: tenant={self.tenant_id}, name={self.name}")


# ============================================================================
# EXAMPLE 6: Pydantic Models
# ============================================================================

from pydantic import BaseModel, validator


class UserRegistrationSecure(BaseModel):
    """
    SECURE: Using Pydantic with custom validators.
    """

    tenant_id: str
    email: str
    name: str

    @validator('tenant_id')
    def validate_tenant_id(cls, v):
        """Validate tenant ID format."""
        return InputValidator.validate_alphanumeric(
            v, "tenant_id", min_length=3, max_length=255
        )

    @validator('email')
    def validate_email(cls, v):
        """Validate email format."""
        return InputValidator.validate_email(v)

    @validator('name')
    def validate_name(cls, v):
        """Validate name format."""
        return InputValidator.validate_alphanumeric(
            v, "name", min_length=1, max_length=255
        )


# ============================================================================
# EXAMPLE 7: Batch Validation
# ============================================================================

def example_batch_validation_secure(user_ids: List[str]) -> List[str]:
    """
    SECURE: Validating multiple inputs.

    This validates a batch of user IDs.
    """
    validated_ids = []
    errors = []

    for i, user_id in enumerate(user_ids):
        try:
            validated = InputValidator.validate_uuid(user_id, f"user_id[{i}]")
            validated_ids.append(validated)
        except ValueError as e:
            errors.append((i, user_id, str(e)))

    if errors:
        logger.warning(f"Batch validation failed for {len(errors)} items")
        raise ValueError(f"Invalid user IDs: {errors[:5]}")

    logger.info(f"Validated {len(validated_ids)} user IDs")
    return validated_ids


# ============================================================================
# EXAMPLE 8: Complex Database Query
# ============================================================================

async def example_complex_query_secure(
    tenant_id: str,
    status: Optional[str],
    limit: int,
    offset: int,
    sort_by: str
) -> tuple:
    """
    SECURE: Complex query with multiple filters and pagination.

    This demonstrates a complete secure query pattern.
    """
    from database.postgres_manager_secure import SecureQueryBuilder

    # Build filters
    filters = []

    # Tenant filter (required)
    validated_tenant = InputValidator.validate_alphanumeric(
        tenant_id, "tenant_id"
    )
    filters.append(SafeQueryInput(
        field="tenant_id",
        value=validated_tenant,
        operator="="
    ))

    # Status filter (optional)
    if status:
        validated_status = InputValidator.validate_alphanumeric(
            status, "status", max_length=50
        )
        filters.append(SafeQueryInput(
            field="status",
            value=validated_status,
            operator="="
        ))

    # Validate pagination
    validated_limit = InputValidator.validate_integer(
        limit, "limit", min_value=1, max_value=1000
    )
    validated_offset = InputValidator.validate_integer(
        offset, "offset", min_value=0
    )

    # Validate sort field
    validated_sort = InputValidator.validate_field_name(sort_by)

    # Build query
    builder = SecureQueryBuilder("agents")

    query, params = builder.build_select(
        filters=filters,
        columns=["tenant_id", "name", "status", "created_at"],
        limit=validated_limit,
        offset=validated_offset,
        sort_by=validated_sort,
        sort_direction="DESC"
    )

    logger.info(f"Complex query built: {query[:100]}...")
    return query, params


# ============================================================================
# EXAMPLE 9: Deployment with Validation
# ============================================================================

async def example_deployment_secure(
    agent_name: str,
    version: str,
    namespace: str,
    replicas: int
) -> Dict:
    """
    SECURE: Kubernetes deployment with validation.

    This demonstrates secure deployment patterns.
    """
    from factory.deployment_secure import SecureDeploymentManager

    # Validate all inputs
    validated_name = InputValidator.validate_alphanumeric(
        agent_name, "agent_name", max_length=253
    )

    validated_version = InputValidator.validate_alphanumeric(
        version, "version", max_length=128
    )

    validated_namespace = InputValidator.validate_alphanumeric(
        namespace, "namespace", max_length=63
    )

    validated_replicas = InputValidator.validate_integer(
        replicas, "replicas", min_value=1, max_value=100
    )

    # Deploy safely
    manager = SecureDeploymentManager()

    result = manager.deploy_agent(
        agent_name=validated_name,
        image_tag=validated_version,
        namespace=validated_namespace,
        replicas=validated_replicas
    )

    logger.info(f"Deployment successful: {validated_name}")
    return result


# ============================================================================
# EXAMPLE 10: Error Handling
# ============================================================================

async def example_error_handling_secure(user_input: str):
    """
    SECURE: Proper error handling for validation failures.

    This shows how to handle validation errors gracefully.
    """
    try:
        # Attempt validation
        tenant_id = InputValidator.validate_alphanumeric(
            user_input, "tenant_id"
        )

        # Process validated input
        logger.info(f"Valid tenant ID: {tenant_id}")
        return {"status": "success", "tenant_id": tenant_id}

    except ValueError as e:
        # Log security event
        logger.warning(
            f"Validation failed for tenant_id",
            extra={
                "input_preview": user_input[:50],
                "error": str(e),
                "timestamp": DeterministicClock.utcnow().isoformat()
            }
        )

        # Return user-friendly error
        return {
            "status": "error",
            "message": "Invalid tenant ID format",
            "details": "Tenant ID must contain only alphanumeric characters, hyphens, and underscores"
        }


# ============================================================================
# MAIN DEMO
# ============================================================================

def run_examples():
    """Run all examples to demonstrate usage."""

    print("=" * 80)
    print("INPUT VALIDATION EXAMPLES")
    print("=" * 80)

    # Example 1: Database Query
    print("\n1. Database Query (Secure)")
    query, params = example_database_query_secure("tenant-123")
    print(f"   Query: {query}")
    print(f"   Params: {params}")

    # Example 2: File Path
    print("\n2. File Path (Secure)")
    try:
        path = example_file_path_secure("config")
        print(f"   Path: {path}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 3: URL Validation
    print("\n3. URL Validation (Secure)")
    try:
        validated_url = InputValidator.validate_url(
            "https://api.example.com/data",
            allowed_schemes=['https']
        )
        print(f"   URL: {validated_url}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 4: Pydantic Model
    print("\n4. Pydantic Model (Secure)")
    try:
        user = UserRegistrationSecure(
            tenant_id="tenant-123",
            email="user@example.com",
            name="John-Doe"
        )
        print(f"   User: {user.dict()}")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 5: Batch Validation
    print("\n5. Batch Validation (Secure)")
    try:
        user_ids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "550e8400-e29b-41d4-a716-446655440000"
        ]
        validated = example_batch_validation_secure(user_ids)
        print(f"   Validated {len(validated)} UUIDs")
    except Exception as e:
        print(f"   Error: {e}")

    # Example 6: Error Handling
    print("\n6. Error Handling (Secure)")
    result = asyncio.run(example_error_handling_secure("tenant@invalid"))
    print(f"   Result: {result}")

    print("\n" + "=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run examples
    run_examples()
