package container.admission

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Default deny for container operations
default allow := false

# Container registry allowlist
allowed_registries := {
    "ghcr.io/akshay-greenlang",
    "ghcr.io/greenlang",
    "docker.io/greenlang",
    "registry.greenlang.io"
}

# Required security annotations
required_annotations := {
    "security.capabilities.drop",
    "security.no-new-privileges",
    "org.opencontainers.image.version",
    "org.opencontainers.image.source"
}

# Vulnerability thresholds
max_critical_vulns := 0
max_high_vulns := 5
max_medium_vulns := 20

# Allow if all security requirements are met
allow if {
    # Image from allowed registry
    registry_allowed

    # Image is signed
    image_signed

    # SBOM is attached
    sbom_attached

    # Vulnerability scan passed
    vuln_scan_passed

    # Security context is properly configured
    security_context_valid

    # Required annotations present
    annotations_valid
}

# Check if registry is allowed
registry_allowed if {
    some registry in allowed_registries
    startswith(input.image.registry, registry)
}

# Verify image is signed with Cosign
image_signed if {
    input.image.signatures
    count(input.image.signatures) > 0
    some sig in input.image.signatures
    sig.verifier == "cosign"
    sig.verified == true
}

# SECURITY FIX: Verify SBOM is attached AND contains actual component data
# This prevents empty/placeholder SBOMs from bypassing security checks
sbom_attached if {
    input.image.sbom
    input.image.sbom.format in {"spdx", "cyclonedx"}
    input.image.sbom.attached == true
    sbom_has_components
}

# Validate SBOM contains actual components (not empty)
sbom_has_components if {
    input.image.sbom.components
    count(input.image.sbom.components) > 0
}

# Alternative field names depending on SBOM format
sbom_has_components if {
    input.image.sbom.packages
    count(input.image.sbom.packages) > 0
}

# Check vulnerability scan results
vuln_scan_passed if {
    input.image.vulns.critical <= max_critical_vulns
    input.image.vulns.high <= max_high_vulns
    input.image.vulns.medium <= max_medium_vulns
}

# Validate security context
security_context_valid if {
    # Must run as non-root
    input.securityContext.runAsNonRoot == true

    # User ID must be >= 10001
    input.securityContext.runAsUser >= 10001

    # No privilege escalation allowed
    input.securityContext.allowPrivilegeEscalation == false

    # Read-only root filesystem (with exceptions for volumes)
    input.securityContext.readOnlyRootFilesystem == true

    # All capabilities dropped
    input.securityContext.capabilities.drop[_] == "ALL"

    # No new privileges
    input.securityContext.seccompProfile.type in {"RuntimeDefault", "Localhost"}
}

# Validate required annotations
annotations_valid if {
    required := required_annotations
    provided := {key | input.metadata.annotations[key]}
    missing := required - provided
    count(missing) == 0
}

# Deny reasons with detailed messages
deny[msg] if {
    not registry_allowed
    msg := sprintf("Image registry '%s' not in allowlist. Allowed: %v",
        [input.image.registry, allowed_registries])
}

deny[msg] if {
    not image_signed
    msg := "Image must be signed with Cosign"
}

deny[msg] if {
    not sbom_attached
    msg := "Image must have SBOM attached (SPDX or CycloneDX format)"
}

deny[msg] if {
    input.image.sbom
    input.image.sbom.attached == true
    not sbom_has_components
    msg := "SBOM is attached but contains no components - empty SBOMs not allowed"
}

deny[msg] if {
    input.image.vulns.critical > max_critical_vulns
    msg := sprintf("Image has %d critical vulnerabilities (max allowed: %d)",
        [input.image.vulns.critical, max_critical_vulns])
}

deny[msg] if {
    input.image.vulns.high > max_high_vulns
    msg := sprintf("Image has %d high vulnerabilities (max allowed: %d)",
        [input.image.vulns.high, max_high_vulns])
}

deny[msg] if {
    not input.securityContext.runAsNonRoot
    msg := "Container must run as non-root user"
}

deny[msg] if {
    input.securityContext.runAsUser < 10001
    msg := sprintf("Container user ID must be >= 10001 (got: %d)",
        [input.securityContext.runAsUser])
}

deny[msg] if {
    input.securityContext.allowPrivilegeEscalation
    msg := "Container must not allow privilege escalation"
}

deny[msg] if {
    not input.securityContext.readOnlyRootFilesystem
    msg := "Container must use read-only root filesystem"
}

deny[msg] if {
    not annotations_valid
    missing := required_annotations - {key | input.metadata.annotations[key]}
    msg := sprintf("Missing required annotations: %v", [missing])
}

# Special rules for development environments
allow_dev if {
    input.environment == "development"
    input.override_key
    valid_override_key
}

valid_override_key if {
    # In dev, require explicit override with valid key
    input.override_key == data.dev_override_key
    time.now_ns() < input.override_expiry
}