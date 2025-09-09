package greenlang.run

import rego.v1

# Default deny-by-default egress policy
default allow := false

# Default reason for denial
default reason := "egress denied by default policy"

# Allow pipeline execution if all network access is authorized
allow if {
	egress_authorized
	region_compliant
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
unauthorized_egress[target] {
	target := input.egress[_]
	not target in input.pipeline.policy.network
}

# Check data residency requirements
region_compliant if {
	not input.pipeline.policy.data_residency  # No requirements
}

region_compliant if {
	input.pipeline.policy.data_residency
	input.region in input.pipeline.policy.allowed_regions
}

# Check resource limits
resource_limits_ok if {
	input.pipeline.resources.memory <= input.pipeline.policy.max_memory
	input.pipeline.resources.cpu <= input.pipeline.policy.max_cpu
	input.pipeline.resources.disk <= input.pipeline.policy.max_disk
}

# Specific denial reasons for network violations
reason := sprintf("egress to unauthorized domain(s): %s", [concat(", ", unauthorized_egress)]) if {
	count(unauthorized_egress) > 0
}

reason := sprintf("execution not allowed in region: %s", [input.region]) if {
	not region_compliant
	input.pipeline.policy.data_residency
}

reason := "resource limits exceeded" if {
	not resource_limits_ok
}

# Allow specific known-good domains by default (infrastructure)
default_allowed_domains := [
	"api.openai.com",
	"api.anthropic.com", 
	"hub.greenlang.io",
	"github.com",
	"pypi.org"
]

# Enhanced egress check with default allowlist
egress_authorized if {
	count(input.egress) > 0
	all_egress_allowed
}

all_egress_allowed if {
	count([target | 
		target := input.egress[_]
		not target in input.pipeline.policy.network
		not target in default_allowed_domains
	]) == 0
}

# Time-based restrictions
reason := "execution not allowed outside business hours" if {
	input.pipeline.policy.business_hours_only
	not time.hour(time.now_ns()) in numbers.range(9, 17)
}

# Sensitive data protection
reason := "pipeline cannot access sensitive data without explicit approval" if {
	input.pipeline.accesses_pii
	not input.pipeline.policy.pii_approved
}

# Development vs production rules
allow if {
	input.stage == "dev"
	egress_authorized  # Relaxed for development
}

reason := "production execution requires stricter compliance" if {
	input.stage == "production"
	not (egress_authorized and region_compliant and resource_limits_ok)
}