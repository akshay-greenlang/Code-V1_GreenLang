"""
GreenLang Authentication and KMS Usage Examples
===============================================

This module demonstrates comprehensive usage patterns for the PostgreSQL backend
and KMS signing implementations in production scenarios.
"""

import os
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any

from greenlang.auth.backends.postgresql import (
    PostgreSQLBackend,
    DatabaseConfig,
    init_database,
    get_db_session
)
from greenlang.auth.permissions import (
    Permission,
    PermissionEffect,
    PermissionCondition
)
from greenlang.security.signing import (
    ExternalKMSSigner,
    SigningConfig,
    SigstoreSigner,
    DetachedSigner
)
from greenlang.security.kms.factory import (
    create_kms_provider,
    detect_kms_provider,
    KMSConfig
)


# ==============================================================================
# PostgreSQL Backend Examples
# ==============================================================================

def example_postgresql_setup():
    """Demonstrate PostgreSQL backend setup and initialization."""
    print("\n=== PostgreSQL Backend Setup ===\n")

    # Configuration from environment variables
    config = DatabaseConfig(
        host=os.environ.get("GL_DB_HOST", "localhost"),
        port=int(os.environ.get("GL_DB_PORT", 5432)),
        database=os.environ.get("GL_DB_NAME", "greenlang"),
        username=os.environ.get("GL_DB_USER", "gl_backend"),
        password=os.environ.get("GL_DB_PASSWORD", "secure_password"),
        pool_size=20,
        max_overflow=10,
        pool_timeout=30,
        pool_recycle=3600,  # Recycle connections after 1 hour
        echo=False  # Set to True for SQL debugging
    )

    # Initialize global database session
    db_session = init_database(config)
    print(f"Database initialized: {config.database} on {config.host}:{config.port}")

    # Create backend instance
    backend = PostgreSQLBackend(config)
    print(f"Backend created with pool size: {config.pool_size}")

    return backend


def example_permission_management(backend: PostgreSQLBackend):
    """Demonstrate permission CRUD operations."""
    print("\n=== Permission Management ===\n")

    # 1. Create a permission for emissions data access
    emission_permission = Permission(
        resource="emissions:scope3:*",
        action="read",
        effect=PermissionEffect.ALLOW,
        scope="organization:greenlang",
        conditions=[
            PermissionCondition(
                field="ip_range",
                operator="in",
                value=["10.0.0.0/8", "192.168.0.0/16"]
            ),
            PermissionCondition(
                field="time_range",
                operator="between",
                value=["08:00", "18:00"]
            )
        ],
        created_by="admin@greenlang.com"
    )

    created_perm = backend.create_permission(emission_permission)
    print(f"Created permission: {created_perm.permission_id}")
    print(f"  Resource: {created_perm.resource}")
    print(f"  Action: {created_perm.action}")
    print(f"  Effect: {created_perm.effect.value}")

    # 2. Create a deny permission for sensitive data
    sensitive_permission = Permission(
        resource="emissions:financial:*",
        action="delete",
        effect=PermissionEffect.DENY,
        scope="organization:greenlang",
        created_by="security@greenlang.com"
    )

    deny_perm = backend.create_permission(sensitive_permission)
    print(f"\nCreated deny permission: {deny_perm.permission_id}")

    # 3. List permissions with filtering
    print("\n--- Listing Permissions ---")
    permissions = backend.list_permissions(
        resource_pattern="emissions:*",
        effect=PermissionEffect.ALLOW,
        limit=10
    )

    for perm in permissions:
        print(f"  {perm.permission_id}: {perm.resource} -> {perm.action} ({perm.effect.value})")

    # 4. Update a permission
    created_perm.action = "write"
    created_perm.conditions.append(
        PermissionCondition(
            field="department",
            operator="eq",
            value="sustainability"
        )
    )

    updated_perm = backend.update_permission(created_perm)
    print(f"\nUpdated permission {updated_perm.permission_id}: action is now '{updated_perm.action}'")

    return created_perm.permission_id


def example_role_management(backend: PostgreSQLBackend, permission_ids: List[str]):
    """Demonstrate role creation and management."""
    print("\n=== Role Management ===\n")

    # 1. Create Sustainability Analyst role
    analyst_role = backend.create_role({
        "role_name": "sustainability_analyst",
        "description": "Can view and analyze emissions data",
        "permissions": permission_ids,
        "metadata": {
            "department": "sustainability",
            "level": "analyst",
            "created_date": datetime.utcnow().isoformat()
        }
    })

    print(f"Created role: {analyst_role['role_name']}")
    print(f"  ID: {analyst_role['role_id']}")
    print(f"  Permissions: {len(analyst_role['permissions'])} assigned")

    # 2. Create Compliance Officer role
    officer_role = backend.create_role({
        "role_name": "compliance_officer",
        "description": "Can manage compliance reports and audit trails",
        "permissions": [],  # Will add permissions later
        "metadata": {
            "department": "compliance",
            "level": "officer"
        },
        "is_system_role": False,
        "priority": 100
    })

    print(f"\nCreated role: {officer_role['role_name']}")

    # 3. Update role with additional permissions
    updated_role = backend.update_role(
        officer_role['role_id'],
        {
            "permissions": permission_ids,
            "metadata": {
                "department": "compliance",
                "level": "senior_officer",
                "last_updated": datetime.utcnow().isoformat()
            }
        }
    )

    print(f"\nUpdated role {updated_role['role_name']}: added {len(permission_ids)} permissions")

    # 4. Assign role to user
    user_assignment = backend.assign_role_to_user(
        "user123@greenlang.com",
        analyst_role['role_id'],
        expires_at=datetime.utcnow() + timedelta(days=90)  # 90-day assignment
    )

    print(f"\nAssigned role '{analyst_role['role_name']}' to user 'user123@greenlang.com'")
    print(f"  Expires: 90 days")

    return analyst_role['role_id']


def example_policy_management(backend: PostgreSQLBackend):
    """Demonstrate policy creation and management."""
    print("\n=== Policy Management ===\n")

    # 1. Create RBAC policy for emissions access
    rbac_policy = backend.create_policy({
        "policy_name": "emissions_data_access_policy",
        "policy_type": "rbac",
        "rules": {
            "roles": ["sustainability_analyst", "compliance_officer"],
            "resources": ["emissions:*", "reports:emissions:*"],
            "actions": ["read", "list", "export"],
            "effect": "allow"
        },
        "conditions": {
            "ip_restriction": {
                "type": "ip_range",
                "value": ["10.0.0.0/8"]
            },
            "time_restriction": {
                "type": "business_hours",
                "timezone": "UTC",
                "start": "08:00",
                "end": "18:00"
            }
        },
        "priority": 100,
        "enabled": True
    })

    print(f"Created RBAC policy: {rbac_policy['policy_name']}")
    print(f"  Type: {rbac_policy['policy_type']}")
    print(f"  Priority: {rbac_policy['priority']}")

    # 2. Create ABAC policy for data classification
    abac_policy = backend.create_policy({
        "policy_name": "data_classification_policy",
        "policy_type": "abac",
        "rules": {
            "attributes": {
                "data_classification": ["public", "internal"],
                "user_clearance": ["level1", "level2", "level3"]
            },
            "logic": "user_clearance >= data_classification"
        },
        "enabled": True,
        "expires_at": datetime.utcnow() + timedelta(days=365)
    })

    print(f"\nCreated ABAC policy: {abac_policy['policy_name']}")
    print(f"  Expires in 365 days")

    # 3. Create temporal access policy
    temporal_policy = backend.create_policy({
        "policy_name": "quarterly_report_access",
        "policy_type": "temporal",
        "rules": {
            "schedule": {
                "type": "recurring",
                "frequency": "quarterly",
                "duration_days": 15,  # Access for 15 days each quarter
                "start_day": 1
            },
            "resources": ["reports:quarterly:*"],
            "actions": ["read", "generate", "export"]
        },
        "enabled": True
    })

    print(f"\nCreated temporal policy: {temporal_policy['policy_name']}")

    return [rbac_policy['policy_id'], abac_policy['policy_id'], temporal_policy['policy_id']]


def example_audit_logging(backend: PostgreSQLBackend):
    """Demonstrate audit logging capabilities."""
    print("\n=== Audit Logging ===\n")

    # 1. Log successful authentication
    auth_log = backend.log_audit_event({
        "event_type": "auth.login",
        "user_id": "user123@greenlang.com",
        "result": "success",
        "ip_address": "192.168.1.100",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "session_id": "sess_abc123",
        "details": {
            "auth_method": "oauth2",
            "provider": "microsoft",
            "mfa_used": True
        }
    })

    print(f"Logged authentication event: {auth_log}")

    # 2. Log permission check
    perm_log = backend.log_audit_event({
        "event_type": "permission.check",
        "user_id": "user123@greenlang.com",
        "resource": "emissions:scope3:transport",
        "action": "read",
        "result": "success",
        "correlation_id": "corr_xyz789",
        "details": {
            "policy_evaluated": "emissions_data_access_policy",
            "decision_time_ms": 12
        }
    })

    print(f"Logged permission check: {perm_log}")

    # 3. Log data access
    data_log = backend.log_audit_event({
        "event_type": "data.access",
        "user_id": "analyst456@greenlang.com",
        "resource": "emissions:2024:q3",
        "action": "export",
        "result": "success",
        "details": {
            "format": "xlsx",
            "rows_exported": 15234,
            "file_size_mb": 2.3,
            "destination": "sharepoint"
        }
    })

    print(f"Logged data access: {data_log}")

    # 4. Retrieve audit logs
    print("\n--- Recent Audit Logs ---")
    recent_logs = backend.get_audit_logs(
        user_id="user123@greenlang.com",
        event_type="auth.login",
        limit=5
    )

    for log in recent_logs:
        print(f"  [{log['timestamp']}] {log['event_type']}: {log['result']}")

    return auth_log


def example_database_maintenance(backend: PostgreSQLBackend):
    """Demonstrate database maintenance operations."""
    print("\n=== Database Maintenance ===\n")

    # 1. Get database statistics
    stats = backend.get_statistics()
    print("Database Statistics:")
    print(f"  Permissions: {stats['permissions_count']}")
    print(f"  Roles: {stats['roles_count']}")
    print(f"  Policies: {stats['policies_count']}")
    print(f"  Audit Logs: {stats['audit_logs_count']}")
    print(f"  Active Policies: {stats['active_policies']}")
    print(f"  Recent Events (24h): {stats['recent_audit_events']}")
    print(f"  Cache Hit Rate: {stats['cache_hits'] / max(stats['backend_operations'], 1) * 100:.2f}%")

    # 2. Clean up expired data
    cleanup_results = backend.cleanup_expired_data(older_than_days=90)
    print("\nCleanup Results:")
    print(f"  Expired role assignments removed: {cleanup_results['expired_role_assignments']}")
    print(f"  Expired policies removed: {cleanup_results['expired_policies']}")
    print(f"  Old audit logs removed: {cleanup_results['old_audit_logs']}")

    # 3. Optimize indexes (would be done via SQL)
    print("\nMaintenance tasks completed successfully")


# ==============================================================================
# KMS Signing Examples
# ==============================================================================

def example_kms_setup():
    """Demonstrate KMS provider setup and configuration."""
    print("\n=== KMS Provider Setup ===\n")

    # Detect available provider
    provider = detect_kms_provider()
    print(f"Detected KMS provider: {provider or 'None (will use local signing)'}")

    # List available providers
    available = list_available_providers()
    print(f"Available providers: {', '.join(available)}")

    # Get provider requirements
    for provider in available:
        reqs = get_provider_requirements(provider)
        print(f"\n{provider.upper()} requirements:")
        for req in reqs:
            print(f"  - {req}")

    return provider


def example_aws_kms_signing():
    """Demonstrate AWS KMS signing."""
    print("\n=== AWS KMS Signing ===\n")

    # Set up AWS environment
    os.environ['GL_KMS_PROVIDER'] = 'aws'
    os.environ['GL_KMS_REGION'] = 'us-west-2'
    os.environ['GL_KMS_KEY_ID'] = 'arn:aws:kms:us-west-2:123456789012:key/12345678-1234-1234-1234-123456789012'
    os.environ['AWS_PROFILE'] = 'greenlang-prod'

    # Create signer
    config = SigningConfig(
        kms_key_id=os.environ['GL_KMS_KEY_ID'],
        algorithm="ecdsa"
    )

    try:
        signer = ExternalKMSSigner(config)

        # Sign emissions report
        report_data = json.dumps({
            "report_id": "2024-Q3-EMISSIONS",
            "total_emissions_tco2": 15234.56,
            "scope1": 3456.12,
            "scope2": 4567.89,
            "scope3": 7210.55,
            "timestamp": datetime.utcnow().isoformat()
        }).encode()

        result = signer.sign(report_data)

        print(f"Signed emissions report with AWS KMS")
        print(f"  Signature: {result.signature[:32].hex()}...")
        print(f"  Algorithm: {result.algorithm}")
        print(f"  Timestamp: {result.timestamp}")

        # Get signer info
        info = signer.get_signer_info()
        print(f"\nKMS Key Info:")
        print(f"  Provider: {info['provider']}")
        print(f"  Key ID: {info['key_id']}")
        print(f"  Algorithm: {info.get('algorithm', 'N/A')}")
        print(f"  Rotation Enabled: {info.get('rotation_enabled', False)}")

        return result

    except Exception as e:
        print(f"AWS KMS signing not available: {e}")
        return None


def example_azure_kms_signing():
    """Demonstrate Azure Key Vault signing."""
    print("\n=== Azure Key Vault Signing ===\n")

    # Set up Azure environment
    os.environ['GL_KMS_PROVIDER'] = 'azure'
    os.environ['AZURE_KEY_VAULT_URL'] = 'https://greenlang-vault.vault.azure.net'
    os.environ['AZURE_TENANT_ID'] = 'your-tenant-id'
    os.environ['AZURE_CLIENT_ID'] = 'your-client-id'
    os.environ['AZURE_CLIENT_SECRET'] = 'your-client-secret'
    os.environ['GL_KMS_KEY_ID'] = 'emissions-signing-key'

    config = SigningConfig(
        kms_key_id=os.environ['GL_KMS_KEY_ID'],
        algorithm="rsa"
    )

    try:
        signer = ExternalKMSSigner(config)

        # Sign compliance document
        compliance_data = json.dumps({
            "document_id": "CSRD-2024-REPORT",
            "framework": "CSRD",
            "status": "verified",
            "auditor": "external_auditor_123",
            "verification_date": datetime.utcnow().isoformat()
        }).encode()

        result = signer.sign(compliance_data)

        print(f"Signed compliance document with Azure Key Vault")
        print(f"  Signature: {result.signature[:32].hex()}...")
        print(f"  Algorithm: {result.algorithm}")

        return result

    except Exception as e:
        print(f"Azure Key Vault signing not available: {e}")
        return None


def example_gcp_kms_signing():
    """Demonstrate GCP Cloud KMS signing."""
    print("\n=== GCP Cloud KMS Signing ===\n")

    # Set up GCP environment
    os.environ['GL_KMS_PROVIDER'] = 'gcp'
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/path/to/service-account.json'
    os.environ['GL_KMS_KEY_ID'] = 'projects/greenlang/locations/global/keyRings/emissions/cryptoKeys/signing-key/cryptoKeyVersions/1'

    config = SigningConfig(
        kms_key_id=os.environ['GL_KMS_KEY_ID'],
        algorithm="ecdsa"
    )

    try:
        signer = ExternalKMSSigner(config)

        # Sign supply chain data
        supply_chain_data = json.dumps({
            "batch_id": "BATCH-2024-10-15",
            "supplier": "renewable_energy_corp",
            "carbon_intensity": 0.045,
            "renewable_percentage": 87.5,
            "certification": "RE100",
            "timestamp": datetime.utcnow().isoformat()
        }).encode()

        result = signer.sign(supply_chain_data)

        print(f"Signed supply chain data with GCP Cloud KMS")
        print(f"  Signature: {result.signature[:32].hex()}...")
        print(f"  Algorithm: {result.algorithm}")

        return result

    except Exception as e:
        print(f"GCP Cloud KMS signing not available: {e}")
        return None


def example_direct_kms_usage():
    """Demonstrate direct KMS provider usage."""
    print("\n=== Direct KMS Provider Usage ===\n")

    # Create KMS configuration
    kms_config = KMSConfig(
        provider="aws",  # or "azure", "gcp"
        key_id="test-key-id",
        region="us-west-2",
        cache_ttl_seconds=3600,
        max_retries=3,
        timeout_seconds=30
    )

    try:
        # Create KMS provider directly
        kms_provider = create_kms_provider(kms_config)

        # Sign data
        data = b"Direct KMS signing test data"
        sign_result = kms_provider.sign(data)

        print(f"Direct KMS signing successful")
        print(f"  Provider: {sign_result['provider']}")
        print(f"  Key ID: {sign_result['key_id']}")
        print(f"  Algorithm: {sign_result['algorithm']}")

        # Get public key for verification
        public_key = kms_provider.get_public_key()
        print(f"  Public key retrieved: {len(public_key)} bytes")

        # Verify signature
        is_valid = kms_provider.verify(data, sign_result['signature'])
        print(f"  Signature verification: {'VALID' if is_valid else 'INVALID'}")

        # Get key info
        key_info = kms_provider.get_cached_key_info()
        print(f"\nKey Information:")
        print(f"  Algorithm: {key_info.algorithm.value}")
        print(f"  Enabled: {key_info.enabled}")
        print(f"  Created: {key_info.created_at}")

        return sign_result

    except Exception as e:
        print(f"Direct KMS usage failed: {e}")
        return None


def example_fallback_signing():
    """Demonstrate fallback to local signing when KMS is unavailable."""
    print("\n=== Fallback Signing ===\n")

    # Clear KMS configuration to trigger fallback
    os.environ.pop('GL_KMS_PROVIDER', None)

    # Try to create KMS signer without proper config
    config = SigningConfig(
        algorithm="ed25519"  # No KMS key specified
    )

    # This will use local signing
    signer = DetachedSigner(config)

    # Sign data locally
    data = b"Local signing fallback test"
    result = signer.sign(data)

    print(f"Fallback to local signing successful")
    print(f"  Signature: {result.signature[:32].hex()}...")
    print(f"  Algorithm: {result.algorithm}")
    print(f"  Has public key: {result.public_key is not None}")

    return result


# ==============================================================================
# Combined Examples
# ==============================================================================

def example_signed_audit_logs(backend: PostgreSQLBackend):
    """Demonstrate signing audit logs with KMS."""
    print("\n=== Signed Audit Logs ===\n")

    # Create KMS signer
    config = SigningConfig(
        algorithm="ecdsa"
    )

    # Use local signing for demo (would use KMS in production)
    signer = DetachedSigner(config)

    # Create audit event
    event = {
        "event_type": "compliance.report.generated",
        "user_id": "compliance_officer@greenlang.com",
        "resource": "report:csrd:2024:q3",
        "action": "generate",
        "result": "success",
        "details": {
            "framework": "CSRD",
            "period": "2024-Q3",
            "metrics_included": 127,
            "data_quality_score": 0.95
        }
    }

    # Sign the event
    event_json = json.dumps(event, sort_keys=True).encode()
    sign_result = signer.sign(event_json)

    # Add signature to event
    event["signature"] = sign_result.signature.hex()
    event["signing_algorithm"] = sign_result.algorithm
    event["public_key"] = sign_result.public_key.hex() if sign_result.public_key else None

    # Log the signed event
    log_id = backend.log_audit_event(event)

    print(f"Created signed audit log: {log_id}")
    print(f"  Event type: {event['event_type']}")
    print(f"  Signature: {event['signature'][:32]}...")
    print(f"  Algorithm: {event['signing_algorithm']}")

    # Verify the signature
    from greenlang.security.signing import DetachedSigVerifier

    verifier = DetachedSigVerifier()
    try:
        verifier.verify(
            payload=event_json,
            signature=bytes.fromhex(event['signature']),
            public_key=bytes.fromhex(event['public_key']) if event['public_key'] else None,
            algorithm=event['signing_algorithm']
        )
        print(f"  Signature verification: VALID ✓")
    except Exception as e:
        print(f"  Signature verification: FAILED - {e}")

    return log_id


def example_complete_workflow():
    """Demonstrate complete authentication and signing workflow."""
    print("\n" + "="*60)
    print("COMPLETE GREENLANG AUTH & KMS WORKFLOW")
    print("="*60)

    # 1. Setup PostgreSQL backend
    backend = example_postgresql_setup()

    # 2. Create permissions
    perm_id = example_permission_management(backend)

    # 3. Create and manage roles
    role_id = example_role_management(backend, [perm_id])

    # 4. Create policies
    policy_ids = example_policy_management(backend)

    # 5. Log audit events
    audit_log = example_audit_logging(backend)

    # 6. Setup KMS
    provider = example_kms_setup()

    # 7. Sign with different providers
    aws_result = example_aws_kms_signing()
    azure_result = example_azure_kms_signing()
    gcp_result = example_gcp_kms_signing()

    # 8. Direct KMS usage
    direct_result = example_direct_kms_usage()

    # 9. Fallback signing
    fallback_result = example_fallback_signing()

    # 10. Signed audit logs
    signed_log = example_signed_audit_logs(backend)

    # 11. Database maintenance
    example_database_maintenance(backend)

    print("\n" + "="*60)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*60)
    print("\nSummary:")
    print(f"  ✓ PostgreSQL backend configured")
    print(f"  ✓ Permissions, roles, and policies created")
    print(f"  ✓ Audit logging enabled")
    print(f"  ✓ KMS signing tested ({provider or 'local'})")
    print(f"  ✓ Signed audit logs created")
    print(f"  ✓ Database maintenance performed")


def main():
    """Run all examples."""
    try:
        example_complete_workflow()
    except Exception as e:
        print(f"\nError in workflow: {e}")
        print("\nNote: Some examples require actual database and KMS setup.")
        print("See documentation for configuration details.")


if __name__ == "__main__":
    main()