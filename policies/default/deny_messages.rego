package greenlang.messages

# =============================================================================
# STANDARDIZED DENIAL MESSAGES
# This policy provides human-readable denial messages with remediation hints
# =============================================================================

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# =============================================================================
# INSTALL DENIALS
# =============================================================================

install_denied[msg] {
    not input.pack.signature_verified
    not input.override.allow_unsigned
    msg := {
        "code": "POLICY.DENIED_INSTALL.UNSIGNED",
        "message": "Pack is not signed or signature verification failed",
        "remedy": "Sign the pack with 'gl pack sign' or use '--allow-unsigned' flag (dev only)",
        "severity": "error"
    }
}

install_denied[msg] {
    input.pack.signature_verified
    input.pack.signature_expired
    msg := {
        "code": "POLICY.DENIED_INSTALL.EXPIRED_SIGNATURE",
        "message": "Pack signature has expired",
        "remedy": "Re-sign the pack with a valid certificate",
        "severity": "error"
    }
}

install_denied[msg] {
    not input.pack.publisher
    msg := {
        "code": "POLICY.DENIED_INSTALL.NO_PUBLISHER",
        "message": "Pack does not specify a publisher",
        "remedy": "Add 'publisher' field to pack manifest",
        "severity": "error"
    }
}

install_denied[msg] {
    input.pack.publisher
    not publisher_allowed(input.pack.publisher)
    msg := {
        "code": "POLICY.DENIED_INSTALL.UNTRUSTED_PUBLISHER",
        "message": sprintf("Publisher '%s' is not in the allowed list", [input.pack.publisher]),
        "remedy": sprintf("Add '%s' to org.allowed_publishers or contact security team", [input.pack.publisher]),
        "severity": "error",
        "allowed_publishers": get_allowed_publishers
    }
}

install_denied[msg] {
    input.pack.size > 100000000
    msg := {
        "code": "POLICY.DENIED_INSTALL.SIZE_LIMIT",
        "message": sprintf("Pack size %dMB exceeds 100MB limit", [input.pack.size / 1000000]),
        "remedy": "Reduce pack size or request exception from security team",
        "severity": "error"
    }
}

# =============================================================================
# EXECUTION DENIALS
# =============================================================================

execution_denied[msg] {
    not input.user.authenticated
    msg := {
        "code": "POLICY.DENIED_EXECUTION.NOT_AUTHENTICATED",
        "message": "User is not authenticated",
        "remedy": "Authenticate with 'gl auth login'",
        "severity": "error"
    }
}

execution_denied[msg] {
    input.user.authenticated == false
    msg := {
        "code": "POLICY.DENIED_EXECUTION.AUTH_INVALID",
        "message": "User authentication is invalid or expired",
        "remedy": "Re-authenticate with 'gl auth login'",
        "severity": "error"
    }
}

execution_denied[msg] {
    input.env.region
    not region_allowed(input.env.region)
    msg := {
        "code": "POLICY.DENIED_EXECUTION.REGION_BLOCKED",
        "message": sprintf("Execution not allowed in region '%s'", [input.env.region]),
        "remedy": "Run from an allowed region or request exception",
        "severity": "error",
        "allowed_regions": get_allowed_regions
    }
}

execution_denied[msg] {
    input.resources.memory_mb > 4096
    msg := {
        "code": "POLICY.DENIED_EXECUTION.MEMORY_LIMIT",
        "message": sprintf("Requested memory %dMB exceeds 4GB limit", [input.resources.memory_mb]),
        "remedy": "Reduce memory requirements or use cloud execution",
        "severity": "error"
    }
}

execution_denied[msg] {
    input.resources.cpu_cores > 4
    msg := {
        "code": "POLICY.DENIED_EXECUTION.CPU_LIMIT",
        "message": sprintf("Requested %d CPU cores exceeds limit of 4", [input.resources.cpu_cores]),
        "remedy": "Reduce CPU requirements or use cloud execution",
        "severity": "error"
    }
}

# =============================================================================
# CAPABILITY DENIALS
# =============================================================================

capability_denied[msg] {
    cap := input.request.requested_capabilities[_]
    not capability_declared(cap)
    msg := {
        "code": "POLICY.DENIED_CAPABILITY.NOT_DECLARED",
        "message": sprintf("Capability '%s' is not declared in pack manifest", [cap]),
        "remedy": sprintf("Add '%s' to 'capabilities' in manifest.yaml", [cap]),
        "severity": "error",
        "requested": cap,
        "declared": input.pack.declared_capabilities
    }
}

capability_denied[msg] {
    cap := input.request.requested_capabilities[_]
    capability_declared(cap)
    not capability_org_allowed(cap)
    msg := {
        "code": "POLICY.DENIED_CAPABILITY.NOT_ALLOWED",
        "message": sprintf("Capability '%s' is not allowed by organization policy", [cap]),
        "remedy": sprintf("Request '%s' capability from security team", [cap]),
        "severity": "error",
        "requested": cap,
        "allowed": get_allowed_capabilities
    }
}

capability_denied[msg] {
    "net" in input.request.requested_capabilities
    not input.pack.policy.network
    msg := {
        "code": "POLICY.DENIED_CAPABILITY.NETWORK_TARGETS",
        "message": "Network capability requested but no allowed targets specified",
        "remedy": "Add 'policy.network' list to manifest with allowed domains",
        "severity": "error"
    }
}

capability_denied[msg] {
    "subprocess" in input.request.requested_capabilities
    not input.pack.policy.subprocess_allowlist
    msg := {
        "code": "POLICY.DENIED_CAPABILITY.SUBPROCESS_ALLOWLIST",
        "message": "Subprocess capability requested but no command allowlist specified",
        "remedy": "Add 'policy.subprocess_allowlist' to manifest",
        "severity": "warning"
    }
}

# =============================================================================
# DATA RESIDENCY DENIALS
# =============================================================================

data_denied[msg] {
    input.data.location
    input.pack.policy.data_residency
    not (input.data.location in input.pack.policy.data_residency)
    msg := {
        "code": "POLICY.DENIED_DATA.RESIDENCY",
        "message": sprintf("Data in '%s' violates residency requirements", [input.data.location]),
        "remedy": "Use data from allowed regions only",
        "severity": "error",
        "allowed_locations": input.pack.policy.data_residency
    }
}

data_denied[msg] {
    input.data.classification == "confidential"
    input.user.clearance != "high"
    msg := {
        "code": "POLICY.DENIED_DATA.CLEARANCE",
        "message": "Insufficient clearance for confidential data",
        "remedy": "Request appropriate clearance level",
        "severity": "error"
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

publisher_allowed(publisher) {
    publisher == input.org.allowed_publishers[_]
}

publisher_allowed(publisher) {
    not input.org.allowed_publishers
    publisher in ["greenlang-official", "verified", "partner-1"]
}

region_allowed(region) {
    region == input.org.allowed_regions[_]
}

region_allowed(region) {
    not input.org.allowed_regions
    region in ["US", "EU", "APAC"]
}

capability_declared(cap) {
    cap == input.pack.declared_capabilities[_]
}

capability_org_allowed(cap) {
    cap == input.org.allowed_capabilities[_]
}

capability_org_allowed(cap) {
    not input.org.allowed_capabilities
    cap == "fs"  # Only filesystem allowed by default
}

get_allowed_publishers[publisher] {
    publisher := input.org.allowed_publishers[_]
}

get_allowed_publishers[publisher] {
    not input.org.allowed_publishers
    publisher := ["greenlang-official", "verified", "partner-1"][_]
}

get_allowed_regions[region] {
    region := input.org.allowed_regions[_]
}

get_allowed_regions[region] {
    not input.org.allowed_regions
    region := ["US", "EU", "APAC"][_]
}

get_allowed_capabilities[cap] {
    cap := input.org.allowed_capabilities[_]
}

get_allowed_capabilities[cap] {
    not input.org.allowed_capabilities
    cap := "fs"
}

# =============================================================================
# AGGREGATED DENIAL MESSAGES
# =============================================================================

# Collect all denial messages with details
all_denials[denial] {
    install_denied[denial]
}

all_denials[denial] {
    execution_denied[denial]
}

all_denials[denial] {
    capability_denied[denial]
}

all_denials[denial] {
    data_denied[denial]
}

# Format denial for CLI output
formatted_denials[msg] {
    denial := all_denials[_]
    msg := sprintf("[%s] %s\n  → %s", [
        denial.code,
        denial.message,
        denial.remedy
    ])
}

# Summary for decision
denial_summary := {
    "denied": count(all_denials) > 0,
    "count": count(all_denials),
    "messages": formatted_denials,
    "details": all_denials
}