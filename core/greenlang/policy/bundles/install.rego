package greenlang

# Install policy - Controls pack installation and publishing
# Evaluates at: pack add, pack publish

default decision = {"allow": false, "reason": "No matching rules"}

# Main decision point
decision = {
    "allow": allow,
    "reasons": reasons
}

# Collect all denial reasons
reasons[msg] {
    deny[msg]
}

# Allow if no denials and at least one allow rule matches
allow {
    count(reasons) == 0
    allow_rules[_]
}

# === DENY RULES ===

# Deny unsigned packs in production
deny[msg] {
    input.stage == "add"
    not input.pack.security.signatures
    msg := "Pack must be signed for installation"
}

# Deny packs without SBOM
deny[msg] {
    not input.pack.security.sbom
    msg := "Pack must include SBOM (Software Bill of Materials)"
}

# Deny non-approved licenses
deny[msg] {
    input.licenses[_] = license
    not license in ["MIT", "Apache-2.0", "BSD", "ISC", "CC0-1.0"]
    msg := sprintf("License '%s' not in approved list", [license])
}

# Deny packs with known vulnerabilities
deny[msg] {
    input.pack.security.vulnerabilities[_].severity in ["critical", "high"]
    msg := "Pack contains critical or high severity vulnerabilities"
}

# Deny packs that are too large
deny[msg] {
    input.pack.size > 104857600  # 100MB
    msg := sprintf("Pack size %d exceeds 100MB limit", [input.pack.size])
}

# Deny packs with suspicious files
deny[msg] {
    input.files[_] = file
    contains(file, "..")
    msg := sprintf("Suspicious file path: %s", [file])
}

deny[msg] {
    input.files[_] = file
    endswith(file, ".exe")
    msg := sprintf("Executable files not allowed: %s", [file])
}

deny[msg] {
    input.files[_] = file
    startswith(file, "/")
    msg := sprintf("Absolute paths not allowed: %s", [file])
}

# Deny old emission factor vintages
deny[msg] {
    input.pack.metadata.ef_vintage < 2023
    msg := sprintf("Emission factor vintage %d is too old (minimum: 2023)", [input.pack.metadata.ef_vintage])
}

# === ALLOW RULES ===

# Allow verified publishers
allow_rules[msg] {
    input.pack.publisher in ["greenlang", "greenlang-verified"]
    msg := "Verified publisher"
}

# Allow packs from official registry
allow_rules[msg] {
    input.pack.source == "hub.greenlang.io"
    msg := "From official registry"
}

# Allow signed and verified packs
allow_rules[msg] {
    input.pack.security.signatures
    input.pack.security.verified == true
    msg := "Signed and verified"
}

# Allow packs with clean security scan
allow_rules[msg] {
    input.pack.security.scan_status == "clean"
    count(input.pack.security.vulnerabilities) == 0
    msg := "Clean security scan"
}

# Allow local development packs (publish stage only)
allow_rules[msg] {
    input.stage == "publish"
    input.pack.version contains "-dev"
    msg := "Development version for publishing"
}