package greenlang

# Verified Publisher Policy
# =========================
#
# This policy allows installations only from verified publishers
# SECURITY: Default-deny - explicit allow required

# Default deny - must explicitly allow
default decision = {"allow": false, "reason": "POLICY.DENIED: No matching allow policy"}

# Allow verified publishers
decision = {"allow": true, "reason": "Publisher verified"} {
    input.pack.signature_verified == true
    input.pack.publisher in data.greenlang.config.allowed_publishers
}

# Allow if publisher is in the verified list (with signature)
decision = {"allow": true, "reason": "Verified publisher with valid signature"} {
    input.pack.signature_verified == true
    input.pack.publisher in [
        "greenlang-official",
        "climatenza",
        "carbon-aware-foundation",
        "green-software-foundation"
    ]
}

# Deny unsigned packs
decision = {"allow": false, "reason": "POLICY.DENIED: Pack must be signed"} {
    input.pack.signature_verified != true
}

# Deny unknown publishers
decision = {"allow": false, "reason": msg} {
    not (input.pack.publisher in data.greenlang.config.allowed_publishers)
    msg := sprintf("POLICY.DENIED: Publisher '%s' not in allowlist", [input.pack.publisher])
}