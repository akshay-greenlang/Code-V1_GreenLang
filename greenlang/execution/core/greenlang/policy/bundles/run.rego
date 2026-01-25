package greenlang.decision

import rego.v1

# Default deny-by-default policy
default allow := false
default reason := "runtime policy denied"

# Allow pipeline execution if all conditions are met
allow if {
	egress_authorized
	resource_limits_ok
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
reason := sprintf("egress to unauthorized domain(s): %s", [concat(", ", unauthorized_egress)]) if {
	count(unauthorized_egress) > 0
}

reason := "resource limits exceeded" if {
	egress_authorized
	not resource_limits_ok
}

# All stages must follow the same security policies
# Development should use explicit override flags if needed