package greenlang

# Region Allowlist Policy
# ========================
#
# This policy allows execution only in approved regions
# SECURITY: Default-deny - explicit allow required

# Default deny - must explicitly allow
default decision = {"allow": false, "reason": "POLICY.DENIED: No matching allow policy"}

# Allow execution in approved regions
decision = {"allow": true, "reason": "Region approved"} {
    input.region in data.greenlang.config.runtime_allowed_regions
}

# Allow if region is in default safe list
decision = {"allow": true, "reason": "Region in safe list"} {
    input.region in [
        "us-west-2",
        "us-east-1",
        "eu-west-1",
        "eu-central-1",
        "ap-southeast-1"
    ]
}

# Deny execution in restricted regions
decision = {"allow": false, "reason": msg} {
    input.region in ["cn-north-1", "cn-northwest-1", "ru-central-1"]
    msg := sprintf("POLICY.DENIED: Region '%s' is restricted", [input.region])
}

# Deny if region not specified
decision = {"allow": false, "reason": "POLICY.DENIED: Region not specified"} {
    not input.region
}

# Deny if region unknown
decision = {"allow": false, "reason": msg} {
    input.region
    not (input.region in data.greenlang.config.runtime_allowed_regions)
    msg := sprintf("POLICY.DENIED: Region '%s' not in allowlist", [input.region])
}