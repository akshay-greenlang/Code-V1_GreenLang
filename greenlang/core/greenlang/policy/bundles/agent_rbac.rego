package greenlang.agent_rbac

# Agent-Level RBAC Policy
# ========================
#
# This policy enforces role-based access control for individual agents.
# It works in conjunction with the Python RBAC manager to provide
# fine-grained permission control.

import future.keywords.if
import future.keywords.in

# ============================================================================
# Default Deny
# ============================================================================

default allow := false

# ============================================================================
# Role-Based Permissions
# ============================================================================

# Allow if user has required permission via role
allow if {
    some role in input.user_roles
    role_permissions := data.roles[role].permissions
    input.required_permission in role_permissions
}

# Admin role has all permissions
allow if {
    some role in input.user_roles
    role == "agent_admin"
}

# ============================================================================
# Critical Agent Protection
# ============================================================================

# Critical agents that require additional approval for execution
critical_agents := ["GL-001", "GL-002", "GL-006", "GL-010"]

# Critical agents require approval for execute permission
deny[msg] if {
    input.agent_id in critical_agents
    input.required_permission == "execute"
    not input.has_approval
    msg := sprintf("Agent %s is critical and requires approval to execute", [input.agent_id])
}

# ============================================================================
# Time-Based Access Control
# ============================================================================

# Deny execution outside business hours for non-admin users
# (Business hours: Monday-Friday, 8 AM - 6 PM UTC)
deny[msg] if {
    input.required_permission == "execute"
    not is_admin(input.user_roles)
    not is_business_hours
    msg := "Agent execution outside business hours requires admin role"
}

is_business_hours if {
    # Note: In production, use time.now_ns() for actual time checks
    # This is a placeholder for demonstration
    true
}

# ============================================================================
# Data Classification Protection
# ============================================================================

# Deny data write if data is classified as confidential
deny[msg] if {
    input.required_permission == "data_write"
    input.context.data_classification == "confidential"
    not has_confidential_data_access(input.user_roles)
    msg := "Writing confidential data requires agent_engineer or agent_admin role"
}

has_confidential_data_access(roles) if {
    some role in roles
    role in ["agent_engineer", "agent_admin"]
}

# ============================================================================
# Configuration Change Protection
# ============================================================================

# Deny config write in production environment for non-engineers
deny[msg] if {
    input.required_permission == "write_config"
    input.context.environment == "production"
    not has_config_write_access(input.user_roles)
    msg := "Production configuration changes require agent_engineer or agent_admin role"
}

has_config_write_access(roles) if {
    some role in roles
    role in ["agent_engineer", "agent_admin"]
}

# ============================================================================
# Rate Limiting
# ============================================================================

# Deny execution if rate limit exceeded
deny[msg] if {
    input.required_permission == "execute"
    input.context.executions_last_hour > get_rate_limit(input.user_roles)
    msg := sprintf("Rate limit exceeded: %d executions in last hour", [input.context.executions_last_hour])
}

# Get rate limit based on user roles
get_rate_limit(roles) := limit if {
    # Admin: unlimited (9999)
    some role in roles
    role == "agent_admin"
    limit := 9999
} else := limit if {
    # Engineer: 100/hour
    some role in roles
    role == "agent_engineer"
    limit := 100
} else := limit if {
    # Operator: 50/hour
    some role in roles
    role == "agent_operator"
    limit := 50
} else := 10 {
    # Default: 10/hour
    true
}

# ============================================================================
# Audit Logging
# ============================================================================

# Log all denials for audit trail
audit[msg] if {
    not allow
    msg := sprintf("DENIED: User %s attempted %s on agent %s", [
        input.user,
        input.required_permission,
        input.agent_id
    ])
}

# Log all critical agent access attempts
audit[msg] if {
    input.agent_id in critical_agents
    input.required_permission == "execute"
    msg := sprintf("CRITICAL AGENT ACCESS: User %s executed agent %s (approved: %v)", [
        input.user,
        input.agent_id,
        input.has_approval
    ])
}

# Log all config write attempts in production
audit[msg] if {
    input.required_permission == "write_config"
    input.context.environment == "production"
    msg := sprintf("PRODUCTION CONFIG CHANGE: User %s modified config for agent %s", [
        input.user,
        input.agent_id
    ])
}

# ============================================================================
# Helper Functions
# ============================================================================

# Check if user has admin role
is_admin(roles) if {
    some role in roles
    role == "agent_admin"
}

# Check if user has any role
has_any_role(roles) if {
    count(roles) > 0
}

# Get all permissions for user (aggregated from all roles)
user_permissions[permission] if {
    some role in input.user_roles
    some permission in data.roles[role].permissions
}

# ============================================================================
# Decision Document
# ============================================================================

# Construct decision document for response
decision := {
    "allow": allow,
    "reason": reason,
    "deny": deny_messages,
    "audit": audit_messages,
    "user_permissions": user_permissions
}

# Construct reason message
reason := msg if {
    allow
    msg := "Permission granted"
} else := msg if {
    count(deny) > 0
    msg := concat("; ", deny)
} else := "Permission denied by default policy"

# Collect all deny messages
deny_messages := [msg | msg := deny[_]]

# Collect all audit messages
audit_messages := [msg | msg := audit[_]]
