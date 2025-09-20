package container.resources

import future.keywords.contains
import future.keywords.if
import future.keywords.in

# Resource limits by container type
resource_limits := {
    "runner": {
        "cpu": {"request": "100m", "limit": "1000m"},
        "memory": {"request": "128Mi", "limit": "1Gi"},
        "ephemeral-storage": {"request": "1Gi", "limit": "5Gi"}
    },
    "full": {
        "cpu": {"request": "500m", "limit": "2000m"},
        "memory": {"request": "512Mi", "limit": "4Gi"},
        "ephemeral-storage": {"request": "2Gi", "limit": "10Gi"}
    },
    "build": {
        "cpu": {"request": "1000m", "limit": "4000m"},
        "memory": {"request": "2Gi", "limit": "8Gi"},
        "ephemeral-storage": {"request": "5Gi", "limit": "20Gi"}
    }
}

# Default deny for resource requests
default allow := false

# Allow if resources are within limits
allow if {
    container_type := get_container_type
    limits := resource_limits[container_type]
    resources_valid(limits)
}

# Get container type from labels or annotations
get_container_type := type if {
    type := input.metadata.labels["greenlang.io/container-type"]
}

get_container_type := "runner" if {
    not input.metadata.labels["greenlang.io/container-type"]
}

# Validate resource requests and limits
resources_valid(limits) if {
    cpu_valid(limits.cpu)
    memory_valid(limits.memory)
    storage_valid(limits["ephemeral-storage"])
}

# CPU validation
cpu_valid(limits) if {
    parse_cpu(input.resources.requests.cpu) >= parse_cpu(limits.request)
    parse_cpu(input.resources.limits.cpu) <= parse_cpu(limits.limit)
    parse_cpu(input.resources.requests.cpu) <= parse_cpu(input.resources.limits.cpu)
}

# Memory validation
memory_valid(limits) if {
    parse_memory(input.resources.requests.memory) >= parse_memory(limits.request)
    parse_memory(input.resources.limits.memory) <= parse_memory(limits.limit)
    parse_memory(input.resources.requests.memory) <= parse_memory(input.resources.limits.memory)
}

# Storage validation
storage_valid(limits) if {
    parse_storage(input.resources.requests["ephemeral-storage"]) >= parse_storage(limits.request)
    parse_storage(input.resources.limits["ephemeral-storage"]) <= parse_storage(limits.limit)
}

# Parse CPU values (convert to millicores)
parse_cpu(value) := result if {
    endswith(value, "m")
    result := to_number(trim_suffix(value, "m"))
}

parse_cpu(value) := result if {
    not endswith(value, "m")
    result := to_number(value) * 1000
}

# Parse memory values (convert to bytes)
parse_memory(value) := result if {
    endswith(value, "Gi")
    result := to_number(trim_suffix(value, "Gi")) * 1024 * 1024 * 1024
}

parse_memory(value) := result if {
    endswith(value, "Mi")
    result := to_number(trim_suffix(value, "Mi")) * 1024 * 1024
}

parse_memory(value) := result if {
    endswith(value, "Ki")
    result := to_number(trim_suffix(value, "Ki")) * 1024
}

parse_memory(value) := result if {
    is_number(value)
    result := value
}

# Parse storage values
parse_storage(value) := parse_memory(value)

# Deny messages for resource violations
deny[msg] if {
    not input.resources.requests
    msg := "Container must specify resource requests"
}

deny[msg] if {
    not input.resources.limits
    msg := "Container must specify resource limits"
}

deny[msg] if {
    input.resources.requests.cpu
    input.resources.limits.cpu
    parse_cpu(input.resources.requests.cpu) > parse_cpu(input.resources.limits.cpu)
    msg := "CPU request cannot exceed CPU limit"
}

deny[msg] if {
    input.resources.requests.memory
    input.resources.limits.memory
    parse_memory(input.resources.requests.memory) > parse_memory(input.resources.limits.memory)
    msg := "Memory request cannot exceed memory limit"
}

deny[msg] if {
    container_type := get_container_type
    limits := resource_limits[container_type]
    parse_cpu(input.resources.limits.cpu) > parse_cpu(limits.cpu.limit)
    msg := sprintf("CPU limit %s exceeds maximum %s for container type '%s'",
        [input.resources.limits.cpu, limits.cpu.limit, container_type])
}

deny[msg] if {
    container_type := get_container_type
    limits := resource_limits[container_type]
    parse_memory(input.resources.limits.memory) > parse_memory(limits.memory.limit)
    msg := sprintf("Memory limit %s exceeds maximum %s for container type '%s'",
        [input.resources.limits.memory, limits.memory.limit, container_type])
}

# Quota management at namespace level
namespace_quotas := {
    "greenlang-dev": {
        "pods": 50,
        "requests.cpu": "20",
        "requests.memory": "50Gi",
        "limits.cpu": "50",
        "limits.memory": "100Gi",
        "persistentvolumeclaims": 20
    },
    "greenlang-staging": {
        "pods": 100,
        "requests.cpu": "50",
        "requests.memory": "100Gi",
        "limits.cpu": "100",
        "limits.memory": "200Gi",
        "persistentvolumeclaims": 50
    },
    "greenlang-prod": {
        "pods": 500,
        "requests.cpu": "200",
        "requests.memory": "500Gi",
        "limits.cpu": "400",
        "limits.memory": "1Ti",
        "persistentvolumeclaims": 200
    }
}

# Priority classes for pod scheduling
priority_classes := {
    "critical": 1000,
    "high": 750,
    "normal": 500,
    "low": 250,
    "batch": 100
}

# Validate priority class
priority_valid if {
    input.priorityClassName
    priority_classes[input.priorityClassName]
}

priority_valid if {
    not input.priorityClassName
}

deny[msg] if {
    input.priorityClassName
    not priority_classes[input.priorityClassName]
    msg := sprintf("Invalid priority class '%s'. Allowed: %v",
        [input.priorityClassName, object.keys(priority_classes)])
}