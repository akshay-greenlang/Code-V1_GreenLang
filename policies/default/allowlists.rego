package greenlang

# DEFAULT DENY for all operations
default allow_install    = false
default allow_execution  = false

# =============================================================================
# INSTALL-TIME CHECKS
# =============================================================================

# Allow installation only if:
# 1. Pack is signed and verified
# 2. Publisher is in the allowlist
allow_install {
    input.pack.signature_verified == true
    input.pack.publisher == allowed_publishers[_]
}

# Publisher allowlist (org-specific)
allowed_publishers[publisher] {
    publisher := input.org.allowed_publishers[_]
}

# Fallback publishers for default configuration
allowed_publishers[publisher] {
    not input.org.allowed_publishers
    publisher := default_allowed_publishers[_]
}

default_allowed_publishers := [
    "greenlang-official",
    "verified",
    "partner-1"
]

# =============================================================================
# EXECUTION-TIME CHECKS
# =============================================================================

# Allow execution only if:
# 1. Publisher is allowed
# 2. Region is allowed
# 3. User is authenticated
allow_execution {
    # Publisher check
    input.pack.publisher == allowed_publishers[_]

    # Region check
    input.env.region == allowed_regions[_]

    # Authentication check
    input.user.authenticated == true
}

# Region allowlist (org-specific)
allowed_regions[region] {
    region := input.org.allowed_regions[_]
}

# Fallback regions for default configuration
allowed_regions[region] {
    not input.org.allowed_regions
    region := default_allowed_regions[_]
}

default_allowed_regions := ["US", "EU", "APAC"]

# =============================================================================
# CAPABILITY GATES (default-deny)
# =============================================================================

# Capability is allowed only when:
# 1. Pack declares it in manifest
# 2. Organization allows it
# 3. Request explicitly needs it
capability_allowed[cap] {
    # Capability must be explicitly requested
    cap := input.request.requested_capabilities[_]

    # Pack must declare the capability in its manifest
    cap == input.pack.declared_capabilities[_]

    # Organization must allow the capability
    cap == allowed_capabilities[_]
}

# Organization's allowed capabilities
allowed_capabilities[cap] {
    cap := input.org.allowed_capabilities[_]
}

# Default allowed capabilities (minimal by default)
allowed_capabilities[cap] {
    not input.org.allowed_capabilities
    cap := default_allowed_capabilities[_]
}

# Very restrictive default - only filesystem access
default_allowed_capabilities := ["fs"]

# =============================================================================
# SPECIAL CASES & OVERRIDES
# =============================================================================

# Allow unsigned packs only with explicit override flag
allow_install {
    input.override.allow_unsigned == true
    input.pack.publisher == allowed_publishers[_]
    # Log warning - this should be tracked
    true
}

# Development mode override REMOVED for production security
# All policy checks are mandatory - no bypasses allowed

# =============================================================================
# DENY REASONS (for better error messages)
# =============================================================================

deny["POLICY.DENIED_INSTALL: Pack not signed or signature invalid"] {
    not input.pack.signature_verified
    not input.override.allow_unsigned
}

deny[msg] {
    not input.pack.publisher
    msg := "POLICY.DENIED_INSTALL: Publisher information missing"
}

deny[msg] {
    input.pack.publisher
    not (input.pack.publisher == allowed_publishers[_])
    msg := sprintf("POLICY.DENIED_INSTALL: Publisher '%s' not in allowed list", [input.pack.publisher])
}

deny[msg] {
    input.env.region
    not (input.env.region == allowed_regions[_])
    msg := sprintf("POLICY.DENIED_EXECUTION: Region '%s' not allowed", [input.env.region])
}

deny["POLICY.DENIED_EXECUTION: User not authenticated"] {
    not input.user.authenticated
}

deny["POLICY.DENIED_EXECUTION: User authentication invalid"] {
    input.user.authenticated == false
}

deny[msg] {
    cap := input.request.requested_capabilities[_]
    not (cap == input.pack.declared_capabilities[_])
    msg := sprintf("POLICY.DENIED_CAPABILITY: Capability '%s' not declared in pack manifest", [cap])
}

deny[msg] {
    cap := input.request.requested_capabilities[_]
    cap == input.pack.declared_capabilities[_]
    not (cap == allowed_capabilities[_])
    msg := sprintf("POLICY.DENIED_CAPABILITY: Capability '%s' not allowed by organization policy", [cap])
}

# =============================================================================
# DECISION OUTPUT
# =============================================================================

# Compose the final decision for install
decision := {
    "allow": allow_install,
    "reasons": deny_reasons,
    "publisher": input.pack.publisher,
    "verified": input.pack.signature_verified
} {
    input.stage == "install"
}

# Compose the final decision for execution
decision := {
    "allow": allow_execution,
    "reasons": deny_reasons,
    "capabilities": capability_allowed,
    "region": input.env.region
} {
    input.stage == "run"
}

# Collect all deny reasons
deny_reasons[reason] {
    deny[reason]
}

# Default decision structure
default decision = {
    "allow": false,
    "reasons": ["No matching policy rules"],
    "evaluated": true
}