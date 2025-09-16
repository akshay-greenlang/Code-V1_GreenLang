package greenlang.decision

import rego.v1

# Default deny-by-default policy
default allow := false
default reason := "runtime policy denied"

# Allow pipeline execution if all conditions are met
allow if {
	egress_authorized
	resource_limits_ok
	publisher_authorized
	region_compliant
	organization_approved
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
	not target in default_allowed_domains
}

# Default allowed domains (infrastructure)
default_allowed_domains := [
	"api.openai.com",
	"api.anthropic.com", 
	"hub.greenlang.io",
	"github.com",
	"pypi.org"
]

# Check resource limits
resource_limits_ok if {
	input.pipeline.resources.memory <= input.pipeline.policy.max_memory
	input.pipeline.resources.cpu <= input.pipeline.policy.max_cpu
	input.pipeline.resources.disk <= input.pipeline.policy.max_disk
}

# Publisher authorization for runtime
# Can be configured via data.greenlang.config.runtime_allowed_publishers
default publisher_authorized := false

publisher_authorized if {
	# Check against configured runtime allowlist if provided
	data.greenlang.config.runtime_allowed_publishers
	input.pipeline.publisher in data.greenlang.config.runtime_allowed_publishers
}

publisher_authorized if {
	# Fallback to default trusted publishers for runtime
	not data.greenlang.config.runtime_allowed_publishers
	input.pipeline.publisher in [
		"greenlang-official",
		"climatenza",
		"carbon-aware-foundation",
		"green-software-foundation",
		"internal"
	]
}

publisher_authorized if {
	# Allow if no publisher specified (local/dev pipeline)
	not input.pipeline.publisher
}

# Region compliance for data processing
# Can be configured via data.greenlang.config.runtime_allowed_regions
default region_compliant := false

region_compliant if {
	# Check against configured runtime regions if provided
	data.greenlang.config.runtime_allowed_regions
	input.pipeline.region in data.greenlang.config.runtime_allowed_regions
}

region_compliant if {
	# Fallback to default allowed regions for runtime (data residency)
	not data.greenlang.config.runtime_allowed_regions
	input.pipeline.region in [
		"us-west-2",
		"us-east-1",
		"eu-west-1",
		"eu-central-1",
		"ap-southeast-1",
		"local"  # Local execution
	]
}

region_compliant if {
	# Allow if no region specified (assume local execution)
	not input.pipeline.region
}

# Organization approval for runtime execution
# Can be configured via data.greenlang.config.runtime_allowed_orgs
default organization_approved := false

organization_approved if {
	# Check against configured runtime orgs if provided
	data.greenlang.config.runtime_allowed_orgs
	input.pipeline.organization in data.greenlang.config.runtime_allowed_orgs
}

organization_approved if {
	# Fallback to default allowed orgs for runtime
	not data.greenlang.config.runtime_allowed_orgs
	input.pipeline.organization in [
		"internal",
		"greenlang",
		"climatenza",
		"carbon-aware",
		"trusted-partner"
	]
}

organization_approved if {
	# Allow if no organization specified (personal/dev pipeline)
	not input.pipeline.organization
}

# Specific denial reasons
reason := sprintf("egress to unauthorized domain(s): %s", [concat(", ", unauthorized_egress)]) if {
	count(unauthorized_egress) > 0
}

reason := "resource limits exceeded" if {
	egress_authorized
	not resource_limits_ok
}

reason := sprintf("publisher not authorized for runtime: %s", [input.pipeline.publisher]) if {
	egress_authorized
	resource_limits_ok
	not publisher_authorized
	input.pipeline.publisher
}

reason := sprintf("region not compliant for data processing: %s", [input.pipeline.region]) if {
	egress_authorized
	resource_limits_ok
	publisher_authorized
	not region_compliant
	input.pipeline.region
}

reason := sprintf("organization not approved for runtime: %s", [input.pipeline.organization]) if {
	egress_authorized
	resource_limits_ok
	publisher_authorized
	region_compliant
	not organization_approved
	input.pipeline.organization
}

# Allow for development stage
allow if {
	input.stage == "dev"
	egress_authorized
}