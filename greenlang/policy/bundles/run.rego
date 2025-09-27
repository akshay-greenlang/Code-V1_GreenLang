package greenlang.decision

import rego.v1

# Default deny-by-default policy
default allow := false
default reason := "runtime policy denied"

# Allow pipeline execution if all conditions are met
allow if {
	signature_verified
	capabilities_validated
	egress_authorized
	resource_limits_ok
}

# Check signature verification (signed-only enforcement)
signature_verified if {
	input.signature.verified == true
}

# For development, allow unsigned with explicit flag
signature_verified if {
	input.allow_unsigned == true
	print("WARNING: Running unsigned pack in development mode")
}

# Validate capabilities follow default-deny principle
capabilities_validated if {
	# Check network capability
	network_capability_valid
	# Check filesystem capability
	filesystem_capability_valid
	# Check subprocess capability
	subprocess_capability_valid
}

# Network must be explicitly allowed and have egress allowlist
network_capability_valid if {
	input.capabilities.net.allow == false  # Network denied
}

network_capability_valid if {
	input.capabilities.net.allow == true
	count(input.capabilities.net.egress_allowlist) > 0  # Must have allowlist
}

# Filesystem must be explicitly allowed with path restrictions
filesystem_capability_valid if {
	input.capabilities.fs.allow == false  # FS denied
}

filesystem_capability_valid if {
	input.capabilities.fs.allow == true
	count(input.capabilities.fs.read_paths) > 0  # Must specify paths
	count(input.capabilities.fs.write_paths) > 0
	# Ensure write paths are restricted to /tmp
	all_paths_in_tmp(input.capabilities.fs.write_paths)
	# Ensure no dangerous paths
	not has_dangerous_path(input.capabilities.fs.write_paths)
	not has_dangerous_path(input.capabilities.fs.read_paths)
}

# Ensure all write paths are under /tmp
all_paths_in_tmp(paths) if {
	every path in paths {
		startswith(path, "/tmp/")
	}
}

# Alternative for Windows compatibility
all_paths_in_tmp(paths) if {
	every path in paths {
		regex.match("^(\/tmp\/|C:\\\\[Tt]emp\\\\|%TEMP%\\\\)", path)
	}
}

# Subprocess must be explicitly allowed with binary allowlist
subprocess_capability_valid if {
	input.capabilities.subprocess.allow == false  # Subprocess denied
}

subprocess_capability_valid if {
	input.capabilities.subprocess.allow == true
	count(input.capabilities.subprocess.allowlist) > 0  # Must have allowlist
	# Ensure no dangerous binaries
	not has_dangerous_binary(input.capabilities.subprocess.allowlist)
}

# Check for dangerous filesystem paths
has_dangerous_path(paths) if {
	path := paths[_]
	dangerous_paths := ["/", "/etc", "/root", "/sys", "/proc"]
	path in dangerous_paths
}

# Check for dangerous binaries
has_dangerous_binary(binaries) if {
	binary := binaries[_]
	dangerous_binaries := ["sh", "bash", "/bin/sh", "/bin/bash", "sudo", "su"]
	binary in dangerous_binaries
}

# Check that all egress targets are in allowlist
egress_authorized if {
	count(input.egress) == 0  # No egress needed
}

egress_authorized if {
	count(input.egress) > 0
	count(unauthorized_egress) == 0
}

# Collect unauthorized egress attempts
unauthorized_egress contains target if {
	target := input.egress[_]
	not target in input.pipeline.policy.network
}

# Check resource limits
resource_limits_ok if {
	input.pipeline.resources.memory <= input.pipeline.policy.max_memory
	input.pipeline.resources.cpu <= input.pipeline.policy.max_cpu
	input.pipeline.resources.disk <= input.pipeline.policy.max_disk
}

# Specific denial reasons
reason := "pack signature not verified (signed-only mode)" if {
	not signature_verified
}

reason := "capability validation failed - check network/fs/subprocess settings" if {
	signature_verified
	not capabilities_validated
}

reason := sprintf("network capability denied or missing egress allowlist") if {
	signature_verified
	not network_capability_valid
}

reason := sprintf("filesystem capability denied or has dangerous paths") if {
	signature_verified
	not filesystem_capability_valid
}

reason := sprintf("subprocess capability denied or has dangerous binaries") if {
	signature_verified
	not subprocess_capability_valid
}

reason := sprintf("egress to unauthorized domain(s): %s", [concat(", ", unauthorized_egress)]) if {
	signature_verified
	capabilities_validated
	count(unauthorized_egress) > 0
}

reason := "resource limits exceeded" if {
	signature_verified
	capabilities_validated
	egress_authorized
	not resource_limits_ok
}

# All stages must follow the same security policies
# Development should use explicit override flags if needed