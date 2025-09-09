package greenlang

# Runtime policy - Controls pipeline execution
# Evaluates at: pipeline run

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

# Resource limits by profile
deny[msg] {
    input.profile == "dev"
    input.pipeline.resources.memory_mb > 2048
    msg := sprintf("Dev profile memory limit exceeded: %dMB (max: 2048MB)", [input.pipeline.resources.memory_mb])
}

deny[msg] {
    input.profile == "dev"
    input.pipeline.resources.cpu_cores > 2
    msg := sprintf("Dev profile CPU limit exceeded: %d cores (max: 2)", [input.pipeline.resources.cpu_cores])
}

deny[msg] {
    input.profile == "prod"
    input.pipeline.resources.memory_mb > 8192
    msg := sprintf("Prod profile memory limit exceeded: %dMB (max: 8192MB)", [input.pipeline.resources.memory_mb])
}

deny[msg] {
    input.profile == "prod"
    input.pipeline.resources.cpu_cores > 8
    msg := sprintf("Prod profile CPU limit exceeded: %d cores (max: 8)", [input.pipeline.resources.cpu_cores])
}

# Network egress control
deny[msg] {
    input.pipeline.network_access[_] = target
    not target in input.egress
    msg := sprintf("Network access to '%s' not in egress allowlist", [target])
}

# Deny access to sensitive data without proper clearance
deny[msg] {
    input.pipeline.data_access[_].classification == "confidential"
    not input.user.clearance in ["high", "admin"]
    msg := "Insufficient clearance for confidential data access"
}

deny[msg] {
    input.pipeline.data_access[_].classification == "restricted"
    not input.user.clearance in ["medium", "high", "admin"]
    msg := "Insufficient clearance for restricted data access"
}

# Region restrictions
deny[msg] {
    input.pipeline.data_access[_].region != input.region
    input.pipeline.data_access[_].residency_required == true
    msg := sprintf("Data residency violation: pipeline in %s accessing data in %s", [input.region, input.pipeline.data_access[_].region])
}

# Emission factor vintage requirements
deny[msg] {
    input.pipeline.requirements.ef_vintage_min > input.ef_vintage
    msg := sprintf("Emission factor vintage %d below minimum required %d", [input.ef_vintage, input.pipeline.requirements.ef_vintage_min])
}

# Rate limiting
deny[msg] {
    input.user.requests_per_minute > 100
    input.user.role != "premium"
    msg := sprintf("Rate limit exceeded: %d rpm (max: 100 for non-premium)", [input.user.requests_per_minute])
}

deny[msg] {
    input.user.requests_per_minute > 1000
    msg := sprintf("Rate limit exceeded: %d rpm (max: 1000)", [input.user.requests_per_minute])
}

# Time-based restrictions
deny[msg] {
    input.profile == "batch"
    hour := time.clock(time.now_ns())[0]
    hour >= 8
    hour <= 20
    msg := "Batch jobs only allowed outside business hours (8am-8pm)"
}

# === ALLOW RULES ===

# Allow authenticated users
allow_rules[msg] {
    input.user.authenticated == true
    msg := "User authenticated"
}

# Allow admin users (bypass most restrictions)
allow_rules[msg] {
    input.user.role == "admin"
    msg := "Admin user"
}

# Allow pipelines with valid signatures
allow_rules[msg] {
    input.pipeline.signature.verified == true
    msg := "Pipeline signature verified"
}

# Allow local development
allow_rules[msg] {
    input.profile == "dev"
    input.pipeline.source == "local"
    msg := "Local development pipeline"
}

# Allow whitelisted pipelines
allow_rules[msg] {
    input.pipeline.name in [
        "carbon-calculator",
        "energy-optimizer",
        "emissions-reporter"
    ]
    msg := "Whitelisted pipeline"
}

# Allow pipelines from verified packs
allow_rules[msg] {
    input.pipeline.pack.verified == true
    msg := "From verified pack"
}