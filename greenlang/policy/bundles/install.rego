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
	publisher_allowed
	region_allowed
	organization_allowed
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

# Publisher allowlist - trusted publishers only
# Can be configured via data.greenlang.config.allowed_publishers
default publisher_allowed := false

publisher_allowed if {
	# Check against configured allowlist if provided
	data.greenlang.config.allowed_publishers
	input.pack.publisher in data.greenlang.config.allowed_publishers
}

publisher_allowed if {
	# Fallback to default trusted publishers
	not data.greenlang.config.allowed_publishers
	input.pack.publisher in [
		"greenlang-official",
		"climatenza",
		"carbon-aware-foundation",
		"green-software-foundation"
	]
}

# Region allowlist - specify allowed data processing regions
# Can be configured via data.greenlang.config.allowed_regions
default region_allowed := false

region_allowed if {
	# Check against configured allowlist if provided
	data.greenlang.config.allowed_regions
	input.pack.region in data.greenlang.config.allowed_regions
}

region_allowed if {
	# Fallback to default allowed regions (data residency compliance)
	not data.greenlang.config.allowed_regions
	input.pack.region in [
		"us-west-2",
		"us-east-1",
		"eu-west-1",
		"eu-central-1",
		"ap-southeast-1"
	]
}

region_allowed if {
	# Allow if no region specified (local processing)
	not input.pack.region
}

# Organization allowlist - approved organizations only
# Can be configured via data.greenlang.config.allowed_orgs
default organization_allowed := false

organization_allowed if {
	# Check against configured allowlist if provided
	data.greenlang.config.allowed_orgs
	input.pack.organization in data.greenlang.config.allowed_orgs
}

organization_allowed if {
	# Fallback to default allowed orgs
	not data.greenlang.config.allowed_orgs
	input.pack.organization in [
		"internal",
		"greenlang",
		"climatenza",
		"carbon-aware"
	]
}

organization_allowed if {
	# Allow if no organization specified (personal/community pack)
	not input.pack.organization
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
} else := sprintf("publisher not in allowlist: %s", [input.pack.publisher]) if {
	license_allowed
	network_policy_present
	vintage_requirement_met
	not publisher_allowed
	input.pack.publisher
} else := sprintf("region not in allowlist: %s", [input.pack.region]) if {
	license_allowed
	network_policy_present
	vintage_requirement_met
	publisher_allowed
	not region_allowed
	input.pack.region
} else := sprintf("organization not in allowlist: %s", [input.pack.organization]) if {
	license_allowed
	network_policy_present
	vintage_requirement_met
	publisher_allowed
	region_allowed
	not organization_allowed
	input.pack.organization
} else := sprintf("unsupported license: %s", [input.pack.license]) if {
	not license_allowed
	not input.pack.license in ["GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0"]
} else := "policy check passed" if {
	allow
}

# Stage-specific rules (publish has stricter requirements)
allow if {
	input.stage == "dev"
	input.pack.license in ["Apache-2.0", "MIT", "BSD-3-Clause"]
	# More lenient for development
}