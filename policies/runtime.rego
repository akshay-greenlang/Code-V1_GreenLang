package greenlang.runtime

# Default: deny execution unless explicitly allowed
default allow = false

# Allow execution if all conditions are met
allow {
    authenticated
    resource_limits_ok
    network_allowed
    data_residency_ok
    rate_limit_ok
}

# User must be authenticated
authenticated {
    input.user.authenticated == true
}

# Resource limits check
resource_limits_ok {
    not input.resources
}

resource_limits_ok {
    input.resources.memory_mb <= 4096
    input.resources.cpu_cores <= 4
    input.resources.timeout_seconds <= 3600
}

# Network access check - only allowed targets
network_allowed {
    not input.network_targets
}

network_allowed {
    input.network_targets
    all_targets_allowed
}

all_targets_allowed {
    targets_to_check := input.network_targets
    allowed_patterns := input.pack.policy.network
    
    # Check each target against allowed patterns
    all(targets_to_check, function(target) {
        some pattern in allowed_patterns
        match_pattern(target, pattern)
    })
}

# Helper to match network patterns (supports wildcards)
match_pattern(target, pattern) {
    pattern == "*"
}

match_pattern(target, pattern) {
    pattern == target
}

match_pattern(target, pattern) {
    endswith(pattern, ":*")
    prefix := substring(pattern, 0, length(pattern) - 2)
    startswith(target, prefix)
}

# Data residency check
data_residency_ok {
    not input.data_location
}

data_residency_ok {
    input.data_location
    input.data_location in input.pack.policy.data_residency
}

# Rate limiting
rate_limit_ok {
    not input.user.requests_per_minute
}

rate_limit_ok {
    input.user.requests_per_minute <= 100
}

rate_limit_ok {
    input.user.role == "premium"
    input.user.requests_per_minute <= 1000
}

# Denial reasons for better error messages
deny_reason["User not authenticated"] {
    not authenticated
}

deny_reason["Excessive memory requested"] {
    input.resources.memory_mb > 4096
}

deny_reason["Excessive CPU requested"] {
    input.resources.cpu_cores > 4
}

deny_reason["Timeout exceeds limit"] {
    input.resources.timeout_seconds > 3600
}

deny_reason["Network target not allowed"] {
    input.network_targets
    not all_targets_allowed
}

deny_reason["Data residency violation"] {
    input.data_location
    not (input.data_location in input.pack.policy.data_residency)
}

deny_reason["Rate limit exceeded"] {
    input.user.role != "premium"
    input.user.requests_per_minute > 100
}

deny_reason["Premium rate limit exceeded"] {
    input.user.role == "premium"
    input.user.requests_per_minute > 1000
}

# Additional runtime checks
deny_reason["Pipeline requires higher privilege"] {
    input.pipeline.requires_privilege
    input.user.role != "admin"
}

deny_reason["Sensitive data access requires clearance"] {
    input.data.classification == "confidential"
    input.user.clearance != "high"
}

# Audit logging requirements
audit_required {
    input.data.classification in ["confidential", "restricted"]
}

audit_log := msg {
    audit_required
    msg := sprintf("User %s accessed %s data at %s", [
        input.user.id,
        input.data.classification,
        input.timestamp
    ])
}