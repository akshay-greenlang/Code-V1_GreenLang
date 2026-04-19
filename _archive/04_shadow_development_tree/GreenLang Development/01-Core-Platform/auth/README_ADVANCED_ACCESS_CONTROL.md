# Advanced Access Control (RBAC/ABAC) for GreenLang

**Phase 4 Implementation**
**Author:** GreenLang Framework Team
**Date:** November 2025
**Status:** Production Ready

## Overview

This document describes the Advanced Access Control system implemented in GreenLang Phase 4, providing fine-grained permission management, role hierarchies, attribute-based policies, delegation, temporal access controls, and comprehensive auditing.

## Table of Contents

1. [Architecture](#architecture)
2. [Core Components](#core-components)
3. [Permission Model](#permission-model)
4. [Role Hierarchy](#role-hierarchy)
5. [ABAC (Attribute-Based Access Control)](#abac-attribute-based-access-control)
6. [Permission Delegation](#permission-delegation)
7. [Temporal Access Control](#temporal-access-control)
8. [Audit Trail](#audit-trail)
9. [Usage Examples](#usage-examples)
10. [Integration Guide](#integration-guide)
11. [Security Considerations](#security-considerations)

---

## Architecture

The Advanced Access Control system consists of six integrated modules:

```
greenlang/auth/
├── permissions.py          # Fine-grained permission model (~800 lines)
├── roles.py               # Role hierarchy with inheritance (~650 lines)
├── abac.py                # Attribute-based access control (~900 lines)
├── delegation.py          # Permission delegation (~550 lines)
├── temporal_access.py     # Time-based access controls (~450 lines)
└── permission_audit.py    # Permission change audit trail (~400 lines)
```

### Key Design Principles

1. **Default Deny**: No access unless explicitly granted
2. **Explicit Deny Wins**: Deny permissions override allow permissions
3. **Least Privilege**: Grant minimum necessary permissions
4. **Audit Everything**: All permission changes are logged immutably
5. **Zero Trust**: Continuous verification of access rights

---

## Core Components

### 1. Fine-Grained Permissions (`permissions.py`)

Provides resource-level access control with pattern matching and conditions.

**Key Features:**
- Resource patterns with wildcards (`agent:*`, `workflow:carbon-audit`)
- Action patterns (`read`, `execute`, `*`)
- Permission conditions (attribute-based rules)
- Permission evaluation engine with caching
- PostgreSQL storage support

**Permission Format:**
```
resource:action[@scope]
```

**Examples:**
```python
from greenlang.auth import create_permission, PermissionEffect

# Allow reading all agents
perm1 = create_permission(
    resource="agent:*",
    action="read"
)

# Deny exporting confidential data
perm2 = create_permission(
    resource="data:confidential:*",
    action="export",
    effect=PermissionEffect.DENY
)

# Allow workflow execution with condition
perm3 = create_permission(
    resource="workflow:*",
    action="execute",
    conditions=[
        {
            "attribute": "user.department",
            "operator": "eq",
            "value": "operations"
        }
    ]
)
```

### 2. Role Hierarchy (`roles.py`)

Hierarchical role-based access control with permission inheritance.

**Built-in Roles:**
- `super_admin` - Full system access
- `admin` - Administrative access to tenant resources
- `manager` - Manage workflows and agents
- `analyst` - Read and execute workflows
- `viewer` - Read-only access

**Role Tree Example:**
```
Admin (all permissions)
  ├── Manager (manage workflows, agents)
  │   ├── Analyst (read/execute workflows)
  │   └── Operator (execute only)
  └── Viewer (read only)
```

**Usage:**
```python
from greenlang.auth import RoleManager, create_permission

manager = RoleManager()

# Create custom role
analytics_role = manager.create_role(
    name="analytics_specialist",
    display_name="Analytics Specialist",
    description="Analyze data and generate reports",
    permissions=[
        create_permission("data:*", "read"),
        create_permission("workflow:analytics:*", "execute")
    ],
    parent_role_ids=[viewer_role.role_id]
)

# Assign role to user
manager.assign_role(
    role_id=analytics_role.role_id,
    principal_id="user_123",
    assigned_by="admin_001"
)

# Check permission
has_access = manager.check_permission(
    principal_id="user_123",
    resource="data:customer_emissions",
    action="read"
)
```

### 3. ABAC - Attribute-Based Access Control (`abac.py`)

Policy-based access control using user, resource, and environment attributes.

**Attribute Providers:**
- **User Attributes**: department, role, clearance_level, location
- **Resource Attributes**: classification, owner, tags
- **Environment Attributes**: time_of_day, ip_address, is_business_hours

**Policy Example:**
```python
from greenlang.auth import ABACEvaluator, create_policy, PolicyEffect

evaluator = ABACEvaluator()

# Create policy: deny data export outside business hours
policy = create_policy(
    name="no-export-after-hours",
    effect=PolicyEffect.DENY,
    actions=["export"],
    resources=["data:*"],
    conditions=[
        {
            "attribute": "environment.is_business_hours",
            "operator": "eq",
            "value": False
        }
    ]
)

evaluator.add_policy(policy)

# Evaluate access
result = evaluator.evaluate(
    user_id="user_123",
    resource="data:emissions_2024",
    action="export",
    context={"ip_address": "192.168.1.100"}
)

if result.allowed:
    print("Access granted")
else:
    print(f"Access denied: {result.reason}")
```

**JSON Policy Format:**
```json
{
  "policy_id": "restrict-sensitive-data",
  "name": "Restrict Sensitive Data Access",
  "effect": "deny",
  "actions": ["read", "export"],
  "resources": ["data:sensitive:*"],
  "conditions": [
    {
      "attribute": "user.clearance_level",
      "operator": "lt",
      "value": 3
    }
  ],
  "priority": 100
}
```

### 4. Permission Delegation (`delegation.py`)

Temporary permission grants with usage limits and delegation chains.

**Features:**
- Time-limited delegations
- Usage count limits
- Delegation chains (with configurable depth)
- Constraint-based delegations
- Automatic expiration

**Usage:**
```python
from greenlang.auth import DelegationManager, create_temporary_delegation
from datetime import timedelta

manager = DelegationManager(max_delegation_chain=3)

# Create 7-day temporary delegation
delegation = create_temporary_delegation(
    manager=manager,
    delegator_id="manager_001",
    delegatee_id="analyst_123",
    permission=execute_workflow_perm,
    hours=24 * 7,  # 7 days
    reason="Project deadline assistance"
)

# Create limited-use delegation
delegation2 = manager.delegate(
    delegator_id="manager_001",
    delegatee_id="analyst_456",
    permission=export_data_perm,
    max_uses=10,
    duration=timedelta(days=30),
    reason="Monthly report generation"
)

# Get delegated permissions
perms = manager.get_delegated_permissions("analyst_123")

# Revoke delegation
manager.revoke(delegation.delegation_id, revoked_by="manager_001")
```

### 5. Temporal Access Control (`temporal_access.py`)

Time-based access with scheduled permissions and recurring access windows.

**Features:**
- Date range restrictions (valid_from, valid_until)
- Time windows (e.g., 9 AM - 5 PM)
- Recurring patterns (daily, weekly, monthly)
- Business hours enforcement
- Automatic expiration cleanup

**Usage:**
```python
from greenlang.auth import (
    TemporalAccessManager,
    create_business_hours_permission,
    create_weekend_permission
)

manager = TemporalAccessManager()

# Business hours only access
temp_perm = create_business_hours_permission(
    manager=manager,
    user_id="contractor_001",
    permission=execute_workflow_perm,
    start_hour=9,
    end_hour=17,
    weekdays_only=True
)

# Weekend maintenance access
weekend_perm = create_weekend_permission(
    manager=manager,
    user_id="maintenance_team",
    permission=admin_access_perm
)

# Custom schedule: Every Monday and Wednesday, 10 AM - 2 PM
from greenlang.auth import RecurrencePattern, RecurrenceType, TimeWindow
from datetime import time

custom_perm = manager.create_temporal_permission(
    user_id="part_time_analyst",
    permission=read_data_perm,
    time_windows=[
        TimeWindow(start_time=time(10, 0), end_time=time(14, 0))
    ],
    recurrence=RecurrencePattern(
        recurrence_type=RecurrenceType.WEEKLY,
        days_of_week=[0, 2]  # Monday, Wednesday
    )
)

# Get currently active permissions
active_perms = manager.get_active_permissions("user_123")
```

### 6. Permission Audit Trail (`permission_audit.py`)

Immutable audit logging for all permission changes with before/after snapshots.

**Logged Events:**
- Permission created/updated/deleted
- Role assigned/unassigned
- Role permissions modified
- Delegation created/revoked
- Policy changes
- Temporal permission changes

**Features:**
- Before/after snapshots
- Cryptographic integrity (hash chaining)
- Actor attribution
- Change calculation
- Compliance reporting

**Usage:**
```python
from greenlang.auth import get_permission_audit_logger, PermissionChangeType

logger = get_permission_audit_logger()

# Log permission creation
event = logger.log_permission_created(
    actor_id="admin_001",
    permission=new_perm.to_dict(),
    principal_id="user_123",
    reason="New project assignment",
    tenant_id="tenant_456",
    session_id="session_789"
)

# Log role assignment
event = logger.log_role_assigned(
    actor_id="manager_001",
    role_id="analyst_role",
    principal_id="user_123",
    role_snapshot=role.to_dict(),
    reason="Promotion to analyst"
)

# Query audit events
events = logger.query_events(
    start_time=datetime(2025, 11, 1),
    end_time=datetime(2025, 11, 30),
    principal_id="user_123"
)

# Verify integrity
is_valid, errors = logger.verify_integrity()

# Generate compliance report
report = logger.generate_compliance_report(
    start_time=datetime(2025, 11, 1),
    end_time=datetime(2025, 11, 30),
    tenant_id="tenant_456"
)
```

---

## Usage Examples

### Example 1: Complete Permission Setup

```python
from greenlang.auth import (
    RoleManager, PermissionEvaluator, ABACEvaluator,
    DelegationManager, TemporalAccessManager,
    get_permission_audit_logger,
    create_permission, PermissionEffect
)

# Initialize managers
role_mgr = RoleManager()
abac_eval = ABACEvaluator()
delegation_mgr = DelegationManager()
temporal_mgr = TemporalAccessManager()
audit_logger = get_permission_audit_logger()

# Create custom role
data_scientist_role = role_mgr.create_role(
    name="data_scientist",
    display_name="Data Scientist",
    permissions=[
        create_permission("data:*", "read"),
        create_permission("data:*", "analyze"),
        create_permission("model:*", "train"),
        create_permission("workflow:ml:*", "execute")
    ],
    created_by="admin_001"
)

# Assign role
role_mgr.assign_role(
    role_id=data_scientist_role.role_id,
    principal_id="ds_user_001",
    assigned_by="admin_001"
)

# Check permission
can_train = role_mgr.check_permission(
    principal_id="ds_user_001",
    resource="model:carbon_predictor",
    action="train"
)
```

### Example 2: ABAC Policy Enforcement

```python
from greenlang.auth import ABACEvaluator, create_policy, PolicyEffect

# Setup attribute providers
evaluator = ABACEvaluator()

# Set user attributes
evaluator.user_provider.set_user_attributes("user_123", {
    "department": "finance",
    "clearance_level": 2,
    "location": "US"
})

# Set resource attributes
evaluator.resource_provider.set_resource_attributes("data:financial_2024", {
    "classification": "confidential",
    "owner": "finance_dept",
    "region": "US"
})

# Create policies
policy1 = create_policy(
    name="finance-data-access",
    effect=PolicyEffect.ALLOW,
    actions=["read"],
    resources=["data:financial*"],
    conditions=[
        {
            "attribute": "user.department",
            "operator": "eq",
            "value": "finance"
        },
        {
            "attribute": "user.clearance_level",
            "operator": "gte",
            "value": 2
        }
    ]
)

evaluator.add_policy(policy1)

# Evaluate
result = evaluator.evaluate(
    user_id="user_123",
    resource="data:financial_2024",
    action="read"
)
```

### Example 3: Temporary Project Access

```python
from greenlang.auth import DelegationManager, TemporalAccessManager
from datetime import timedelta

delegation_mgr = DelegationManager()
temporal_mgr = TemporalAccessManager()

# Grant temporary project access for 30 days
project_access = temporal_mgr.create_temporal_permission(
    user_id="contractor_001",
    permission=create_permission("project:alpha:*", "*"),
    valid_from=datetime.utcnow(),
    valid_until=datetime.utcnow() + timedelta(days=30),
    description="Alpha project contractor access"
)

# Add business hours restriction
business_hours_access = create_business_hours_permission(
    manager=temporal_mgr,
    user_id="contractor_001",
    permission=create_permission("data:*", "read"),
    start_hour=9,
    end_hour=17,
    weekdays_only=True
)
```

---

## Integration Guide

### 1. Basic Setup

```python
from greenlang.auth import RoleManager

# Initialize role manager (includes built-in roles)
role_manager = RoleManager()

# Built-in roles are automatically created:
# - super_admin
# - admin
# - manager
# - analyst
# - viewer
```

### 2. Custom Authorization Middleware

```python
from greenlang.auth import RoleManager

role_manager = RoleManager()

def check_access(user_id: str, resource: str, action: str) -> bool:
    """Authorization middleware."""
    return role_manager.check_permission(user_id, resource, action)

# Use in API endpoint
@app.get("/api/workflows/{workflow_id}")
def get_workflow(workflow_id: str, current_user: str):
    if not check_access(current_user, f"workflow:{workflow_id}", "read"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Return workflow data
    ...
```

### 3. Multi-Layer Authorization

```python
from greenlang.auth import RoleManager, ABACEvaluator, TemporalAccessManager

def authorize_request(user_id, resource, action, context=None):
    """Multi-layer authorization check."""

    # Layer 1: Role-based permissions
    if not role_manager.check_permission(user_id, resource, action):
        return False, "Role-based access denied"

    # Layer 2: ABAC policies
    abac_result = abac_evaluator.evaluate(user_id, resource, action, context)
    if not abac_result.allowed:
        return False, f"Policy violation: {abac_result.reason}"

    # Layer 3: Temporal restrictions
    active_perms = temporal_manager.get_active_permissions(user_id)
    has_temporal = any(
        p.matches_request(resource, action)
        for p in active_perms
    )
    if not has_temporal:
        return False, "Outside allowed time window"

    return True, "Access granted"
```

---

## Security Considerations

### 1. Permission Evaluation Order

1. **DENY rules evaluated first** - Explicit denies always win
2. **Temporal constraints** - Check if current time is within allowed window
3. **ABAC policies** - Evaluate attribute-based conditions
4. **Role permissions** - Check inherited role permissions
5. **Delegated permissions** - Check if permission was delegated
6. **Default deny** - If no matching permissions, deny access

### 2. Audit Trail Integrity

- All permission changes are logged with cryptographic hashes
- Hash chaining prevents tampering with audit logs
- Before/after snapshots enable rollback and compliance reporting
- Actor attribution tracks who made each change

### 3. Delegation Security

- Maximum delegation chain length prevents unbounded chains
- Usage limits prevent abuse
- Time-based expiration ensures temporary nature
- Revocation cascades to child delegations

### 4. Best Practices

1. **Principle of Least Privilege**: Grant minimum necessary permissions
2. **Regular Access Reviews**: Audit user permissions quarterly
3. **Time-Bound Access**: Use temporal permissions for contractors
4. **Separation of Duties**: Use role hierarchies to enforce SoD
5. **Monitor Suspicious Activity**: Alert on high-risk permission changes
6. **Test Policies**: Verify ABAC policies before deployment

---

## File Locations

All implementation files are in `/c/Users/rshar/Desktop/Akshay Makar/Tools/GreenLang/Code V1_GreenLang/greenlang/auth/`:

- `permissions.py` (~800 lines) - Fine-grained permission model
- `roles.py` (~650 lines) - Role hierarchy with inheritance
- `abac.py` (~900 lines) - Attribute-based access control
- `delegation.py` (~550 lines) - Permission delegation
- `temporal_access.py` (~450 lines) - Time-based access controls
- `permission_audit.py` (~400 lines) - Permission change audit trail
- `__init__.py` - Module exports

---

## API Reference Summary

### Core Classes

- **PermissionEvaluator**: Evaluates permissions with caching
- **RoleManager**: Manages roles and assignments
- **ABACEvaluator**: Evaluates attribute-based policies
- **DelegationManager**: Manages permission delegations
- **TemporalAccessManager**: Manages time-based permissions
- **PermissionAuditLogger**: Logs permission changes immutably

### Helper Functions

- `create_permission()` - Create a permission
- `create_policy()` - Create an ABAC policy
- `create_temporary_delegation()` - Quick temporary delegation
- `create_business_hours_permission()` - Business hours access
- `get_permission_audit_logger()` - Get global audit logger

---

## Testing

Unit tests and integration tests are located in:
- `tests/auth/test_permissions.py`
- `tests/auth/test_roles.py`
- `tests/auth/test_abac.py`
- `tests/auth/test_delegation.py`
- `tests/auth/test_temporal_access.py`
- `tests/auth/test_permission_audit.py`
- `tests/auth/test_integration.py`

Run tests with:
```bash
pytest tests/auth/ -v
```

---

## Support

For questions or issues:
- GitHub Issues: https://github.com/akshay-greenlang/Code-V1_GreenLang/issues
- Documentation: See inline docstrings in each module
- Examples: `examples/auth/` directory

---

## License

Copyright (C) 2025 GreenLang Framework Team
Licensed under the same license as GreenLang.
