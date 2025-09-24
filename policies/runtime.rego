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

# Helper to match network patterns with strict validation
# Only allow specific patterns with explicit domain verification
match_pattern(target, pattern) {
    # Exact match only - most secure option
    pattern == target
}

match_pattern(target, pattern) {
    # Allow limited subdomain patterns with strict validation
    startswith(pattern, "*.")
    not pattern == "*.*"  # Deny top-level wildcards
    not pattern == "*"    # Deny catch-all wildcards

    # Extract parent domain and validate it's legitimate
    parent_domain := substring(pattern, 2, length(pattern))
    valid_parent_domain(parent_domain)

    # Ensure target ends with the parent domain
    endswith(target, parent_domain)

    # Ensure target is actually a subdomain (not the parent itself)
    target != parent_domain

    # Ensure subdomain part doesn't contain wildcards
    subdomain_part := substring(target, 0, length(target) - length(parent_domain) - 1)
    not contains(subdomain_part, "*")
}

# Validate parent domain format
valid_parent_domain(domain) {
    # Must contain at least one dot (e.g., example.com)
    contains(domain, ".")

    # Must not start or end with dot
    not startswith(domain, ".")
    not endswith(domain, ".")

    # Must not contain wildcards
    not contains(domain, "*")

    # Must not be too short (prevent abuse)
    count(split(domain, ".")) >= 2

    # Each part must be valid (no empty segments)
    all(split(domain, "."), function(part) {
        count(part) > 0
    })
}

# Data residency check - ALWAYS MANDATORY when data capabilities requested
data_residency_ok {
    # Skip check ONLY if pack declares no data capabilities at all
    not input.pack.declared_capabilities
}

data_residency_ok {
    # Skip check ONLY if no data-related capabilities are declared
    input.pack.declared_capabilities
    not any_data_capability_declared
}

data_residency_ok {
    # MANDATORY check: If any data capabilities are declared, residency must be enforced
    input.pack.declared_capabilities
    any_data_capability_declared
    input.data_location
    input.data_location in input.pack.policy.data_residency
}

# Helper to detect any data-related capabilities
any_data_capability_declared {
    some capability in input.pack.declared_capabilities
    data_related_capability(capability)
}

# Define what constitutes data-related capabilities
data_related_capability("data") { true }
data_related_capability("storage") { true }
data_related_capability("database") { true }
data_related_capability("cache") { true }
data_related_capability("queue") { true }
data_related_capability("pubsub") { true }

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
    input.pack.declared_capabilities
    any_data_capability_declared
    not input.data_location
}

deny_reason["Data location not in allowed residency zones"] {
    input.pack.declared_capabilities
    any_data_capability_declared
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