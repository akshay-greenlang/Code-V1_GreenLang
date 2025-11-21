package greenlang.install

# Default: deny installation unless explicitly allowed
default allow = false

# Allow installation if all conditions are met
allow {
    license_ok
    network_policy_ok
    ef_vintage_ok
    security_ok
    no_vulnerabilities
}

# Check license is in allowlist
license_ok {
    input.pack.license == "Commercial"
}

license_ok {
    input.pack.license == "MIT"
}

license_ok {
    input.pack.license == "Apache-2.0"
}

# Network policy must be defined for packs that need network
network_policy_ok {
    input.pack.kind != "pack"
}

network_policy_ok {
    input.pack.kind == "pack"
    count(input.pack.policy.network) > 0
}

# SECURITY FIX: Emission factor vintage is MANDATORY for all packs
# No bypass allowed - all packs must declare EF vintage >= 2024
# This ensures climate data quality and prevents use of outdated emission factors
ef_vintage_ok {
    input.pack.policy.ef_vintage_min
    input.pack.policy.ef_vintage_min >= 2024
}

# Security requirements
security_ok {
    input.pack.security.sbom != null
}

# No known vulnerabilities
no_vulnerabilities {
    not input.vulnerabilities
}

no_vulnerabilities {
    input.vulnerabilities
    count(input.vulnerabilities.critical) == 0
    count(input.vulnerabilities.high) == 0
}

# Denial reasons for better error messages
deny_reason["License not in allowlist"] {
    not license_ok
}

deny_reason["Network policy allowlist is empty"] {
    input.pack.kind == "pack"
    count(input.pack.policy.network) == 0
}

deny_reason["Emission factor vintage missing"] {
    not input.pack.policy.ef_vintage_min
}

deny_reason["Emission factor vintage too old"] {
    input.pack.policy.ef_vintage_min
    input.pack.policy.ef_vintage_min < 2024
}

deny_reason["SBOM not provided"] {
    not input.pack.security.sbom
}

deny_reason["Critical vulnerabilities found"] {
    input.vulnerabilities
    count(input.vulnerabilities.critical) > 0
}

deny_reason["High severity vulnerabilities found"] {
    input.vulnerabilities
    count(input.vulnerabilities.high) > 0
}

# Additional checks for specific pack types
deny_reason["Dataset pack must specify data residency"] {
    input.pack.kind == "dataset"
    count(input.pack.policy.data_residency) == 0
}

deny_reason["Connector pack must specify network targets"] {
    input.pack.kind == "connector"
    count(input.pack.policy.network) == 0
}