# GreenLang OS-Level Sandbox

This directory contains the OS-level sandbox implementation for GreenLang, providing comprehensive security isolation that replaces the previous Python-level patching approach.

## Overview

The OS-level sandbox provides multiple layers of security:

1. **Linux Namespace Isolation** - Process, network, mount, UTS, IPC, and user namespaces
2. **Seccomp-BPF Syscall Filtering** - Restricts available system calls
3. **AppArmor/SELinux Profiles** - Filesystem and capability restrictions
4. **Container Integration** - Docker/gVisor support for stronger isolation
5. **Resource Limits** - CPU, memory, file descriptor, and process limits
6. **Network Filtering** - iptables-based network access control

## Files

- `os_sandbox.py` - Main OS-level sandbox implementation
- `seccomp_profiles.json` - Seccomp-BPF profiles for syscall filtering
- `apparmor_profile.txt` - AppArmor profiles for filesystem isolation
- `README.md` - This documentation file

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     GreenLang Runtime                          │
├─────────────────────────────────────────────────────────────────┤
│                    Runtime Guard                               │
│  ┌─────────────────┐    ┌─────────────────────────────────────┐ │
│  │  Python-level   │    │         OS-level Sandbox           │ │
│  │   Patching      │    │  ┌─────────────────────────────────┐ │ │
│  │   (Fallback)    │    │  │      Isolation Backend         │ │ │
│  └─────────────────┘    │  │  ┌─────────┬─────────┬─────────┐│ │ │
│                         │  │  │Namespace│Container│ gVisor  ││ │ │
│                         │  │  │Backend  │Backend  │Backend  ││ │ │
│                         │  │  └─────────┴─────────┴─────────┘│ │ │
│                         │  └─────────────────────────────────┘ │ │
│                         └─────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                    Linux Kernel                                │
│  ┌─────────────┬─────────────┬─────────────┬─────────────────┐  │
│  │ Namespaces  │ Seccomp-BPF │  AppArmor   │  Resource Limits │  │
│  │   (PID,NET, │  (Syscall   │ (Filesystem │   (CPU, Memory, │  │
│  │   MNT,UTS,  │  Filtering) │    Access)  │   File handles) │  │
│  │   IPC,USER) │             │             │                 │  │
│  └─────────────┴─────────────┴─────────────┴─────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Usage

### Basic Usage

```python
from greenlang.sandbox.os_sandbox import execute_sandboxed, create_default_config

# Execute function in OS sandbox
def my_function(x, y):
    return x + y

config = create_default_config()
result = execute_sandboxed(my_function, config, 10, 20)
```

### Advanced Configuration

```python
from greenlang.sandbox.os_sandbox import (
    OSSandbox,
    OSSandboxConfig,
    IsolationType,
    SandboxMode,
    ResourceLimits,
    NetworkConfig,
    FilesystemConfig
)

# Create custom configuration
config = OSSandboxConfig(
    isolation_type=IsolationType.CONTAINER,
    sandbox_mode=SandboxMode.ENFORCING,
    limits=ResourceLimits(
        memory_limit_bytes=512 * 1024 * 1024,  # 512MB
        cpu_time_limit_seconds=300,
        max_open_files=1024,
        max_processes=16
    ),
    network=NetworkConfig(
        allow_network=True,
        allowed_hosts=["api.openai.com", "httpbin.org"],
        blocked_hosts=["169.254.169.254"]  # Block metadata
    ),
    filesystem=FilesystemConfig(
        read_write_paths=["/tmp/sandbox"],
        blocked_paths=["/proc", "/sys", "/dev"]
    ),
    execution_timeout=600
)

# Use sandbox
with OSSandbox(config) as sandbox:
    result = sandbox.execute(my_function, 10, 20)
```

### Integration with Runtime Guard

The OS sandbox is automatically integrated with the runtime guard:

```python
from greenlang.runtime.guard import execute_with_guard

# This will use OS sandbox if available, fallback to Python patching
def risky_function():
    import subprocess
    return subprocess.run(["echo", "hello"], capture_output=True, text=True)

result = execute_with_guard(risky_function)
```

### Environment Configuration

Control sandbox behavior with environment variables:

```bash
# Force OS sandbox on/off
export GL_FORCE_OS_SANDBOX=1

# Set security level (standard, high, maximum)
export GL_SECURITY_LEVEL=high

# Enable specific isolation type
export GL_ISOLATION_TYPE=container

# Configure resource limits
export GL_MEMORY_LIMIT_MB=512
export GL_CPU_LIMIT_SECONDS=300
export GL_EXECUTION_TIMEOUT=600
```

## Isolation Types

### 1. Basic Isolation (`IsolationType.BASIC`)
- Process-level isolation only
- Resource limits via rlimit
- Minimal overhead
- Suitable for low-risk operations

### 2. Namespace Isolation (`IsolationType.NAMESPACE`)
- Linux namespace isolation
- Network, PID, mount namespace separation
- Seccomp-BPF syscall filtering
- AppArmor filesystem restrictions
- Good balance of security and performance

### 3. Container Isolation (`IsolationType.CONTAINER`)
- Docker container execution
- Complete filesystem isolation
- Network isolation
- Strong process separation
- Higher overhead but maximum compatibility

### 4. gVisor Isolation (`IsolationType.GVISOR`)
- gVisor userspace kernel
- Strongest isolation
- Intercepts all syscalls
- Maximum security with good performance

## Security Profiles

### Seccomp Profiles

Located in `seccomp_profiles.json`:

- **default** - Standard restrictive profile
- **compute_only** - Pure computation, no I/O
- **data_processing** - File I/O allowed
- **network_client** - Network operations allowed
- **blocked_dangerous** - Blocks specific dangerous syscalls

### AppArmor Profile

Located in `apparmor_profile.txt`:

- Restricts filesystem access
- Blocks dangerous capabilities
- Limits network operations
- Enforces resource limits
- Provides MAC (Mandatory Access Control)

## Resource Limits

The sandbox enforces various resource limits:

```python
ResourceLimits(
    memory_limit_bytes=512 * 1024 * 1024,    # Virtual memory limit
    virtual_memory_limit_bytes=1024 * 1024 * 1024,  # Total VM limit
    cpu_time_limit_seconds=300,              # CPU time limit
    cpu_percent_limit=50,                    # CPU percentage limit
    max_open_files=1024,                     # File descriptor limit
    max_file_size_bytes=100 * 1024 * 1024,   # Max file size (100MB)
    max_processes=16,                        # Process limit
    max_threads=32,                          # Thread limit
    disk_quota_bytes=1024 * 1024 * 1024      # Disk usage limit (1GB)
)
```

## Network Configuration

```python
NetworkConfig(
    allow_network=True,                      # Enable network access
    allow_loopback=True,                     # Allow localhost
    allowed_hosts=["api.openai.com"],        # Whitelist hosts
    allowed_ports=[80, 443],                 # Whitelist ports
    blocked_hosts=["169.254.169.254"],       # Block metadata services
    dns_servers=["8.8.8.8"],                # Custom DNS
    custom_iptables_rules=[                  # Custom firewall rules
        "-A OUTPUT -d 10.0.0.0/8 -j REJECT"
    ]
)
```

## Filesystem Configuration

```python
FilesystemConfig(
    create_temp_root=True,                   # Create temporary root
    read_only_paths=["/usr", "/lib"],        # Read-only mounts
    read_write_paths=["/tmp/sandbox"],       # Writable paths
    blocked_paths=["/proc", "/sys"],         # Blocked paths
    bind_mounts={"/host/data": "/data"},     # Bind mounts
    apparmor_profile="greenlang-sandbox"     # AppArmor profile name
)
```

## Error Handling and Fallbacks

The OS sandbox includes comprehensive error handling:

1. **Automatic Fallback** - Falls back to Python-level patching if OS sandbox fails
2. **Graceful Degradation** - Reduces security level if higher levels unavailable
3. **Detailed Logging** - Comprehensive audit trail of all operations
4. **Exception Translation** - Converts OS errors to meaningful exceptions

## Performance Considerations

| Isolation Type | Setup Time | Runtime Overhead | Security Level |
|----------------|------------|------------------|----------------|
| Basic          | ~1ms       | ~5%              | Low            |
| Namespace      | ~10ms      | ~10%             | Medium         |
| Container      | ~500ms     | ~15%             | High           |
| gVisor         | ~200ms     | ~20%             | Maximum        |

## Security Features

### Syscall Filtering
- Blocks dangerous syscalls (mount, ptrace, etc.)
- Allows only necessary operations
- Configurable profiles per use case

### Network Security
- Blocks cloud metadata endpoints
- Restricts private network access
- Configurable host/port whitelisting
- iptables-based filtering

### Filesystem Security
- Read-only system directories
- Temporary writable areas
- Path traversal protection
- AppArmor MAC enforcement

### Process Security
- PID namespace isolation
- Process limit enforcement
- No privilege escalation
- Signal isolation

## Deployment

### AppArmor Profile Installation

```bash
# Copy profile to AppArmor directory
sudo cp apparmor_profile.txt /etc/apparmor.d/greenlang-sandbox

# Load profile
sudo apparmor_parser -r /etc/apparmor.d/greenlang-sandbox

# Verify profile
sudo aa-status | grep greenlang
```

### Container Setup

```bash
# For gVisor support
# Install gVisor
curl -fsSL https://gvisor.dev/archive.key | sudo gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | sudo tee /etc/apt/sources.list.d/gvisor.list > /dev/null
sudo apt-get update && sudo apt-get install -y runsc

# Configure Docker to use gVisor
sudo dockerd --add-runtime runsc=/usr/bin/runsc
```

### Seccomp Profile Testing

```bash
# Test seccomp profile compilation
python3 -c "
import json
with open('seccomp_profiles.json') as f:
    profiles = json.load(f)
print(f'Loaded {len(profiles[\"profiles\"])} seccomp profiles')
"
```

## Troubleshooting

### Common Issues

1. **Namespace creation fails**
   ```
   Error: Failed to create namespace
   Solution: Check /proc/sys/user/max_user_namespaces
   ```

2. **Container runtime not found**
   ```
   Error: Docker/Podman not available
   Solution: Install container runtime or disable container isolation
   ```

3. **AppArmor profile loading fails**
   ```
   Error: Cannot load AppArmor profile
   Solution: Check profile syntax and AppArmor service status
   ```

### Debugging

Enable debug logging:

```python
import logging
logging.getLogger('greenlang.sandbox').setLevel(logging.DEBUG)
```

Check sandbox status:

```python
from greenlang.runtime.guard import get_sandbox_info
print(get_sandbox_info())
```

### Performance Tuning

1. Use appropriate isolation type for your use case
2. Configure resource limits based on workload
3. Use namespace isolation for best performance/security balance
4. Consider container warm-up for high-throughput scenarios

## Future Enhancements

- VM-based isolation support
- Windows sandbox integration
- Hardware security features (Intel MPX, etc.)
- Machine learning-based anomaly detection
- Dynamic security policy adaptation