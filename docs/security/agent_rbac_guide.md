# Agent-Level RBAC Guide

## Overview

GreenLang's Agent-Level Role-Based Access Control (RBAC) provides fine-grained permission management for individual agents. This security layer enables organizations to control who can execute, configure, and access data from specific agents.

## Key Features

- **Fine-Grained Permissions**: 9 distinct permission types for granular access control
- **Predefined Roles**: 4 standard roles covering common use cases
- **Custom Roles**: Define organization-specific roles as needed
- **OPA Integration**: Policy-as-code using Open Policy Agent for advanced rules
- **Audit Trail**: Complete provenance tracking with SHA-256 hashes
- **CLI Management**: Easy-to-use command-line interface for RBAC operations
- **Persistence**: Automatic storage and retrieval of permissions

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      PolicyEnforcer                         │
│  ┌──────────────────┐           ┌─────────────────────┐    │
│  │  RBAC Manager    │◄─────────►│  OPA Policy Engine  │    │
│  │  (Python)        │           │  (Rego)             │    │
│  └──────────────────┘           └─────────────────────┘    │
│           │                                                  │
│           ▼                                                  │
│  ┌──────────────────┐                                       │
│  │ Agent Access     │                                       │
│  │ Control Lists    │                                       │
│  │ (JSON Storage)   │                                       │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │     Agent Implementation      │
        │  (GL-001, GL-002, etc.)       │
        └───────────────────────────────┘
```

## Permissions

### Permission Types

| Permission | Description | Example Use Case |
|-----------|-------------|------------------|
| `execute` | Execute agent | Run GL-001 to calculate emissions |
| `read_config` | Read agent configuration | View agent settings |
| `write_config` | Modify agent configuration | Update timeout settings |
| `read_data` | Read agent input/output data | View calculation results |
| `write_data` | Write agent data | Upload new input data |
| `manage_lifecycle` | Start/stop agent | Restart failed agent |
| `view_metrics` | View agent metrics | Check execution statistics |
| `export_provenance` | Export audit trail | Generate compliance report |
| `admin` | Full administrative access | All operations (grants all permissions) |

## Predefined Roles

### agent_viewer
**Read-only access for monitoring**

- Permissions: `read_config`, `view_metrics`
- Use Case: Auditors, stakeholders who need visibility without execution rights

```bash
greenlang rbac grant GL-001 auditor@company.com agent_viewer
```

### agent_operator
**Standard operational access**

- Permissions: `execute`, `read_config`, `read_data`, `view_metrics`, `export_provenance`
- Use Case: Data analysts, sustainability managers who execute agents regularly

```bash
greenlang rbac grant GL-001 analyst@company.com agent_operator
```

### agent_engineer
**Full agent management (except admin operations)**

- Permissions: All except `admin`
- Use Case: Engineers who configure and maintain agents

```bash
greenlang rbac grant GL-001 engineer@company.com agent_engineer
```

### agent_admin
**Full administrative access**

- Permissions: All permissions (including `admin`)
- Use Case: System administrators, platform owners

```bash
greenlang rbac grant GL-001 admin@company.com agent_admin
```

## Quick Start

### 1. Grant Permissions

Grant an operator role to a user for a specific agent:

```bash
greenlang rbac grant GL-001 user@example.com agent_operator
```

### 2. Check Permissions

Verify a user has specific permission:

```bash
greenlang rbac check GL-001 user@example.com execute
```

Output:
```
✓ GRANTED
User user@example.com has permission execute for agent GL-001

Granted by roles: agent_operator
```

### 3. List User Permissions

View all permissions for a user on an agent:

```bash
greenlang rbac list GL-001
```

Output:
```
┌────────────────────┬─────────────────┬─────────────────────────────────────────┐
│ User               │ Roles           │ Permissions                             │
├────────────────────┼─────────────────┼─────────────────────────────────────────┤
│ user@example.com   │ agent_operator  │ execute, read_config, read_data,        │
│                    │                 │ view_metrics, export_provenance         │
└────────────────────┴─────────────────┴─────────────────────────────────────────┘
```

### 4. Audit User Access

Audit all agent access for a user:

```bash
greenlang rbac audit user@example.com
```

Output:
```
┌──────────┬─────────────────┬──────────────────────────────────────┐
│ Agent ID │ Roles           │ Permissions                          │
├──────────┼─────────────────┼──────────────────────────────────────┤
│ GL-001   │ agent_operator  │ execute, read_config, read_data, ... │
│ GL-002   │ agent_viewer    │ read_config, view_metrics            │
│ GL-006   │ agent_engineer  │ execute, read_config, write_config...│
└──────────┴─────────────────┴──────────────────────────────────────┘
```

## Integration with Agents

### Python Implementation

Integrate RBAC into your agent implementation:

```python
from core.greenlang.policy.enforcer import PolicyEnforcer

class MyAgent:
    def __init__(self):
        self.enforcer = PolicyEnforcer()
        self.agent_id = "GL-001"

    async def execute(self, input_data, user: str):
        """Execute with RBAC check."""
        # Check execute permission
        result = self.enforcer.check_agent_execute(self.agent_id, user)

        if not result.allowed:
            raise PermissionError(result.reason)

        # Execute agent logic
        return self._process(input_data)

    def get_config(self, user: str):
        """Get config with RBAC check."""
        result = self.enforcer.check_agent_config_access(
            self.agent_id, user, "read"
        )

        if not result.allowed:
            raise PermissionError(result.reason)

        return self.config
```

See [examples/agent_rbac_example.py](../../examples/agent_rbac_example.py) for complete examples.

## Advanced Features

### Critical Agent Protection

Critical agents (GL-001, GL-002, GL-006, GL-010) require additional approval:

```python
# Execution with approval
context = {"has_approval": True}
result = enforcer.check_agent_execute("GL-001", user, context)
```

OPA policy automatically enforces approval requirement for critical agents.

### Custom Roles

Define organization-specific roles:

```python
from core.greenlang.policy.agent_rbac import AgentRole, AgentPermission

# Create custom role
custom_role = AgentRole(
    role_name="data_analyst",
    permissions={
        AgentPermission.EXECUTE,
        AgentPermission.READ_DATA,
        AgentPermission.VIEW_METRICS
    },
    description="Data analyst with execution and read access"
)

# Add to ACL
acl = enforcer.rbac_manager.get_acl("GL-001")
acl.add_custom_role(custom_role)

# Grant to user
enforcer.grant_agent_role("GL-001", "analyst@company.com", "data_analyst")
```

### OPA Policy Customization

Extend the default OPA policy in `core/greenlang/policy/bundles/agent_rbac.rego`:

```rego
# Add custom time-based restrictions
deny[msg] if {
    input.required_permission == "execute"
    input.context.environment == "production"
    is_weekend
    msg := "Production execution disabled on weekends"
}

is_weekend if {
    # Custom time logic
    true
}
```

### Data Classification

Restrict access based on data classification:

```python
# Check data access with classification
context = {
    "data_classification": "confidential"
}

result = enforcer.check_agent_data_access(
    "GL-001", user, "write", context
)
```

OPA policy enforces classification-based access control.

### Rate Limiting

Built-in rate limiting based on user roles:

- **agent_admin**: Unlimited
- **agent_engineer**: 100 executions/hour
- **agent_operator**: 50 executions/hour
- **agent_viewer**: 10 executions/hour

```python
context = {
    "executions_last_hour": 45
}

result = enforcer.check_agent_execute("GL-001", user, context)
```

## CLI Reference

### grant
Grant role to user for agent.

```bash
greenlang rbac grant <agent_id> <user> <role>

# Example
greenlang rbac grant GL-001 user@example.com agent_operator
```

### revoke
Revoke role from user for agent.

```bash
greenlang rbac revoke <agent_id> <user> <role>

# Example
greenlang rbac revoke GL-001 user@example.com agent_operator
```

### list
List all RBAC grants for agent.

```bash
greenlang rbac list <agent_id> [--format json]

# Example
greenlang rbac list GL-001
greenlang rbac list GL-001 --format json
```

### audit
Audit all agent permissions for user.

```bash
greenlang rbac audit <user> [--format json]

# Example
greenlang rbac audit user@example.com
greenlang rbac audit user@example.com --format json
```

### check
Check if user has specific permission.

```bash
greenlang rbac check <agent_id> <user> <permission>

# Example
greenlang rbac check GL-001 user@example.com execute
```

### roles
List all available predefined roles.

```bash
greenlang rbac roles [--detailed]

# Example
greenlang rbac roles --detailed
```

### export
Export complete RBAC audit log.

```bash
greenlang rbac export [--output <path>]

# Example
greenlang rbac export --output audit_2024.json
```

### create-acl
Create new ACL for agent.

```bash
greenlang rbac create-acl <agent_id>

# Example
greenlang rbac create-acl GL-001
```

### delete-acl
Delete ACL for agent (removes all grants).

```bash
greenlang rbac delete-acl <agent_id> [--confirm]

# Example
greenlang rbac delete-acl GL-001 --confirm
```

## Storage and Persistence

### Storage Location

RBAC data is stored in: `~/.greenlang/rbac/`

Each agent has a separate ACL file:
- `~/.greenlang/rbac/GL-001.json`
- `~/.greenlang/rbac/GL-002.json`
- etc.

### ACL File Format

```json
{
  "agent_id": "GL-001",
  "user_roles": {
    "user@example.com": ["agent_operator"],
    "admin@example.com": ["agent_admin"]
  },
  "custom_roles": {
    "data_analyst": {
      "role_name": "data_analyst",
      "permissions": ["execute", "read_data", "view_metrics"],
      "description": "Data analyst role"
    }
  }
}
```

### Backup and Recovery

Export audit log before major changes:

```bash
greenlang rbac export --output backup_$(date +%Y%m%d).json
```

Restore by recreating grants from audit log.

## Security Best Practices

### 1. Principle of Least Privilege

Grant only the minimum permissions required:

```bash
# ✅ Good: Grant viewer role for read-only access
greenlang rbac grant GL-001 auditor@company.com agent_viewer

# ❌ Bad: Grant admin role when not needed
greenlang rbac grant GL-001 auditor@company.com agent_admin
```

### 2. Regular Audits

Audit user access regularly:

```bash
# Audit all users with access to critical agent
greenlang rbac list GL-001

# Export audit log monthly
greenlang rbac export --output audit_$(date +%Y%m).json
```

### 3. Role Separation

Separate operational and administrative access:

```bash
# Operators execute agents
greenlang rbac grant GL-001 operator@company.com agent_operator

# Engineers configure agents
greenlang rbac grant GL-001 engineer@company.com agent_engineer

# Admins have full control
greenlang rbac grant GL-001 admin@company.com agent_admin
```

### 4. Critical Agent Protection

Use approval workflow for critical agents:

```python
# Production deployment with approval
if agent_id in CRITICAL_AGENTS:
    # Get approval from manager
    approval = get_manager_approval(user, agent_id)

    context = {"has_approval": approval}
    result = enforcer.check_agent_execute(agent_id, user, context)
```

### 5. Environment-Specific Policies

Apply stricter policies in production:

```python
context = {
    "environment": "production",
    "region": "us-east-1"
}

# Production config changes require engineer role
result = enforcer.check_agent_config_access(
    agent_id, user, "write", context
)
```

## Troubleshooting

### Permission Denied Errors

**Problem**: User receives permission denied error

**Solution**:
1. Check user has role: `greenlang rbac list GL-001`
2. Verify role has required permission: `greenlang rbac roles --detailed`
3. Check permission explicitly: `greenlang rbac check GL-001 user@example.com execute`

### ACL Not Found

**Problem**: `No ACL found for agent`

**Solution**:
```bash
# Create ACL for agent
greenlang rbac create-acl GL-001

# Grant role
greenlang rbac grant GL-001 user@example.com agent_operator
```

### OPA Policy Errors

**Problem**: OPA policy evaluation fails

**Solution**:
1. Check OPA is installed: `opa version`
2. Validate policy syntax: `opa test core/greenlang/policy/bundles/agent_rbac.rego`
3. Review policy logs in PolicyEnforcer

### Storage Issues

**Problem**: Permissions not persisting

**Solution**:
1. Check storage directory exists: `ls ~/.greenlang/rbac/`
2. Verify write permissions
3. Check disk space

## Migration Guide

### From Pack-Level to Agent-Level RBAC

If migrating from pack-level permissions:

```bash
# Old: Pack-level permission
# User had access to entire pack

# New: Agent-level permission
# Grant per-agent access
greenlang rbac grant GL-001 user@example.com agent_operator
greenlang rbac grant GL-002 user@example.com agent_operator
# ... for each agent
```

## FAQ

**Q: Can a user have multiple roles?**

A: Yes! Users can have multiple roles on the same agent. Permissions are aggregated.

```bash
greenlang rbac grant GL-001 user@example.com agent_viewer
greenlang rbac grant GL-001 user@example.com agent_operator
# User has permissions from both roles
```

**Q: What happens if no ACL is defined?**

A: Default policy applies: Allow `read_config` and `view_metrics`, deny everything else.

**Q: Can I override predefined roles?**

A: No, predefined roles cannot be overridden. Create custom roles instead.

**Q: How do I bulk grant permissions?**

A: Use a script to iterate over users/agents:

```bash
for agent in GL-001 GL-002 GL-003; do
    greenlang rbac grant $agent user@example.com agent_operator
done
```

**Q: Are permissions cached?**

A: No, permissions are evaluated on every check for security. ACLs are loaded from disk on PolicyEnforcer initialization.

**Q: How do I integrate with external IAM systems?**

A: Map external roles to GreenLang roles in your authentication layer, then grant appropriate GreenLang roles based on external group membership.

## Related Documentation

- [OPA Policy Guide](opa_policy_guide.md)
- [Security Best Practices](security_best_practices.md)
- [Agent Development Guide](../agents/development_guide.md)
- [API Reference](../api/rbac_reference.md)

## Support

For questions or issues:
- File issue: [GitHub Issues](https://github.com/greenlang/greenlang/issues)
- Documentation: [docs.greenlang.in](https://docs.greenlang.in)
- Email: support@greenlang.in
