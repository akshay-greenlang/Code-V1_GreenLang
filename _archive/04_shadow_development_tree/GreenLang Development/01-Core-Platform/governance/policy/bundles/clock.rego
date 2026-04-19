package greenlang.capabilities.clock

import rego.v1

# Default deny clock access
default allow := false

# Clock capability validation
allow if {
    # Check if clock capability is explicitly requested
    input.capabilities.clock.enabled == true

    # Verify the pack is signed and verified
    input.signature.verified == true

    # Check time manipulation limits
    validate_time_limits

    # Log clock access
    print("Clock capability granted for pack:", input.pack.name)
}

# Validate time manipulation limits
validate_time_limits if {
    # Check if time drift is within acceptable range (5 minutes)
    input.capabilities.clock.max_drift_seconds <= 300

    # Ensure no backward time travel
    input.capabilities.clock.allow_backward == false

    # Check rate limiting for time queries
    input.capabilities.clock.max_queries_per_minute <= 60
}

# Deny reasons
deny[reason] if {
    not input.capabilities.clock.enabled
    reason := "clock capability not requested in manifest"
}

deny[reason] if {
    not input.signature.verified
    reason := "unsigned packs cannot access clock"
}

deny[reason] if {
    input.capabilities.clock.max_drift_seconds > 300
    reason := sprintf("excessive time drift requested: %d seconds", [input.capabilities.clock.max_drift_seconds])
}

deny[reason] if {
    input.capabilities.clock.allow_backward == true
    reason := "backward time travel not permitted"
}

deny[reason] if {
    input.capabilities.clock.max_queries_per_minute > 60
    reason := sprintf("excessive clock queries: %d per minute", [input.capabilities.clock.max_queries_per_minute])
}

# Clock operation validation
validate_clock_operation[result] if {
    operation := input.operation

    # Check operation type
    operation.type in ["read", "sync", "adjust"]

    # Validate based on operation type
    operation.type == "read"
    result := validate_read_operation
}

validate_clock_operation[result] if {
    operation := input.operation
    operation.type == "sync"
    result := validate_sync_operation
}

validate_clock_operation[result] if {
    operation := input.operation
    operation.type == "adjust"
    result := validate_adjust_operation
}

# Read operation validation
validate_read_operation := {
    "allowed": true,
    "reason": "clock read permitted"
} if {
    # Check rate limiting
    input.operation.rate_count < input.capabilities.clock.max_queries_per_minute
}

validate_read_operation := {
    "allowed": false,
    "reason": "rate limit exceeded for clock reads"
} if {
    input.operation.rate_count >= input.capabilities.clock.max_queries_per_minute
}

# Sync operation validation
validate_sync_operation := {
    "allowed": true,
    "reason": "clock sync permitted with NTP server"
} if {
    # Only allow sync with approved NTP servers
    input.operation.ntp_server in data.approved_ntp_servers

    # Check sync frequency (max once per hour)
    input.operation.last_sync_elapsed_seconds >= 3600
}

validate_sync_operation := {
    "allowed": false,
    "reason": sprintf("unapproved NTP server: %s", [input.operation.ntp_server])
} if {
    not input.operation.ntp_server in data.approved_ntp_servers
}

# Adjust operation validation (for testing only)
validate_adjust_operation := {
    "allowed": true,
    "reason": "clock adjustment permitted in test environment"
} if {
    # Only allow in test/dev environments
    input.environment in ["test", "dev"]

    # Check adjustment is within limits
    abs(input.operation.adjustment_seconds) <= input.capabilities.clock.max_drift_seconds
}

validate_adjust_operation := {
    "allowed": false,
    "reason": "clock adjustment not permitted in production"
} if {
    input.environment == "production"
}

# Approved NTP servers
approved_ntp_servers := {
    "time.google.com",
    "time.cloudflare.com",
    "pool.ntp.org",
    "time.nist.gov",
    "time.windows.com"
}

# Security checks for replay attack prevention
prevent_replay_attack if {
    # Check timestamp is recent (within 5 minutes)
    current_time := time.now_ns() / 1000000000
    request_time := input.timestamp
    time_diff := abs(current_time - request_time)
    time_diff <= 300

    # Check nonce hasn't been used before
    not input.nonce in data.used_nonces
}

deny[reason] if {
    not prevent_replay_attack
    reason := "potential replay attack detected"
}

# Audit logging for clock operations
audit_log[entry] if {
    allow
    entry := {
        "timestamp": time.now_ns() / 1000000000,
        "pack": input.pack.name,
        "operation": input.operation.type,
        "environment": input.environment,
        "allowed": true
    }
}

audit_log[entry] if {
    not allow
    entry := {
        "timestamp": time.now_ns() / 1000000000,
        "pack": input.pack.name,
        "operation": input.operation.type,
        "environment": input.environment,
        "allowed": false,
        "deny_reasons": deny
    }
}