package greenlang.decision

import rego.v1

# Default deny-by-default policy
default allow := false

# Default reason for denial
default reason := "policy denied"

# Allow installation if all conditions are met
allow if {
	license_allowed
	network_policy_present
	vintage_requirement_met
}

# License allowlist - deny GPL and restrictive licenses
license_allowed if {
	input.pack.license in ["Apache-2.0", "MIT", "BSD-3-Clause", "Commercial"]
}

# Network policy must be explicitly defined
network_policy_present if {
	count(input.pack.policy.network) > 0
}

# Emission factor vintage must be recent (2024+)
vintage_requirement_met if {
	input.pack.policy.ef_vintage_min >= 2024
}

# Specific denial reasons for better error messages
# Select most specific denial reason
reason := "GPL or restrictive license not allowed" if {
	not license_allowed
	input.pack.license in ["GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0"]
} else := "missing network allowlist - must specify allowed domains" if {
	not network_policy_present
	license_allowed
} else := "emission factor vintage too old - must be 2024 or newer" if {
	license_allowed
	network_policy_present
	not vintage_requirement_met
} else := sprintf("unsupported license: %s", [input.pack.license]) if {
	not license_allowed
	not input.pack.license in ["GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0"]
} else := "policy check passed" if {
	allow
}

# All stages must follow the same security policies
# Development should use explicit override flags if needed