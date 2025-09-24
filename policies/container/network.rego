package container.network

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Default deny all egress
default allow_egress := false

# Allowed egress destinations for containers
allowed_egress := {
    # Internal services - explicit service names for security
    {"namespace": "greenlang-system", "service": "api-server"},
    {"namespace": "greenlang-system", "service": "job-controller"},
    {"namespace": "greenlang-system", "service": "policy-engine"},
    {"namespace": "monitoring", "service": "prometheus"},
    {"namespace": "logging", "service": "loki"},

    # External registries (for pulling dependencies)
    {"host": "pypi.org", "port": 443},
    {"host": "files.pythonhosted.org", "port": 443},
    {"host": "registry.npmjs.org", "port": 443},

    # GreenLang services
    {"host": "api.greenlang.io", "port": 443},
    {"host": "hub.greenlang.io", "port": 443},
    {"host": "registry.greenlang.io", "port": 443},

    # Git repositories (for pack sources)
    {"host": "github.com", "port": 443},
    {"host": "gitlab.com", "port": 443},

    # Cloud provider metadata services (IMDSv2 only)
    {"host": "169.254.169.254", "port": 80, "require_token": true},  # AWS
    {"host": "metadata.google.internal", "port": 80},                 # GCP
    {"host": "169.254.169.254", "port": 80, "path": "/metadata"}     # Azure
}

# Allow egress if destination is in allowlist
allow_egress if {
    some dest in allowed_egress
    destination_matches(dest)
}

# Check if destination matches
destination_matches(allowed) if {
    # Match by namespace and service
    allowed.namespace
    input.destination.namespace == allowed.namespace
    service_matches(allowed.service, input.destination.service)
}

destination_matches(allowed) if {
    # Match by host and port
    allowed.host
    input.destination.host == allowed.host
    input.destination.port == allowed.port
    path_matches(allowed)
    token_matches(allowed)
}

# Service matching - only exact matches allowed for security
service_matches(pattern, service) if {
    pattern == service
}

# Path matching for specific endpoints
path_matches(allowed) if {
    not allowed.path
}

path_matches(allowed) if {
    allowed.path
    startswith(input.destination.path, allowed.path)
}

# Token requirement for cloud metadata services
token_matches(allowed) if {
    not allowed.require_token
}

token_matches(allowed) if {
    allowed.require_token
    input.headers["X-aws-ec2-metadata-token"]
}

token_matches(allowed) if {
    allowed.require_token
    input.headers["Metadata-Flavor"] == "Google"
}

# Deny messages
deny[msg] if {
    not allow_egress
    msg := sprintf("Egress to %s:%d not allowed",
        [input.destination.host, input.destination.port])
}

# Special rules for build containers
allow_egress if {
    input.labels["greenlang.io/build-container"] == "true"
    input.phase == "build"
    build_destination_allowed
}

build_destination_allowed if {
    # Additional destinations allowed during build
    input.destination.host in {
        "dl-cdn.alpinelinux.org",  # Alpine packages
        "deb.debian.org",           # Debian packages
        "security.debian.org",      # Security updates
        "archive.ubuntu.com",       # Ubuntu packages
        "security.ubuntu.com"       # Ubuntu security
    }
    input.destination.port == 443
}

# Ingress rules for containers
default allow_ingress := false

# Allowed ingress sources
allowed_ingress := {
    # From ingress controller
    {"namespace": "ingress-nginx", "service": "nginx-ingress-controller"},

    # From monitoring
    {"namespace": "monitoring", "service": "prometheus"},

    # From service mesh
    {"namespace": "istio-system", "service": "istio-ingressgateway"},

    # Internal services
    {"namespace": "greenlang-system", "label": "app.kubernetes.io/part-of=greenlang"}
}

allow_ingress if {
    some source in allowed_ingress
    source_matches(source)
}

source_matches(allowed) if {
    allowed.namespace
    input.source.namespace == allowed.namespace
    service_matches(allowed.service, input.source.service)
}

source_matches(allowed) if {
    allowed.namespace
    allowed.label
    input.source.namespace == allowed.namespace
    label_matches(allowed.label)
}

label_matches(label_requirement) if {
    [key, value] := split(label_requirement, "=")
    input.source.labels[key] == value
}

# Port restrictions
allowed_ports := {8080, 9090, 443}

deny[msg] if {
    not input.port in allowed_ports
    msg := sprintf("Port %d not allowed. Allowed ports: %v",
        [input.port, allowed_ports])
}

# Rate limiting rules
rate_limits := {
    "default": {"requests_per_second": 100, "burst": 200},
    "api": {"requests_per_second": 1000, "burst": 2000},
    "health": {"requests_per_second": 10, "burst": 20}
}

get_rate_limit[limit] if {
    endpoint_type := input.labels["greenlang.io/endpoint-type"]
    limit := rate_limits[endpoint_type]
}

get_rate_limit[limit] if {
    not input.labels["greenlang.io/endpoint-type"]
    limit := rate_limits.default
}